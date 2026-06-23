import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Tuple


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])


def generalized_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    inter_x1 = torch.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    inter_y1 = torch.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    inter_x2 = torch.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    inter_y2 = torch.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    inter_area = inter_w * inter_h

    area_a = box_area(boxes_a)[:, None]
    area_b = box_area(boxes_b)[None, :]
    union_area = area_a + area_b - inter_area

    iou = inter_area / torch.clamp(union_area, min=1e-6)

    enc_x1 = torch.minimum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    enc_y1 = torch.minimum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    enc_x2 = torch.maximum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    enc_y2 = torch.maximum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1)

    giou = iou - (enc_area - union_area) / torch.clamp(enc_area, min=1e-6)
    return giou


def build_cost_matrix(
    pred_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    gt_boxes: torch.Tensor,
    cost_class: float = 1.0,
    cost_bbox: float = 5.0,
    cost_giou: float = 2.0,
) -> np.ndarray:
    probs = F.softmax(pred_logits, dim=-1).detach().cpu().numpy()
    gt_idx = gt_labels.detach().cpu().numpy().astype(np.int32)
    cost_cls = -probs[:, gt_idx]

    pb = pred_boxes.detach().cpu().numpy()
    gb = gt_boxes.detach().cpu().numpy()
    cost_l1 = np.sum(np.abs(pb[:, None, :] - gb[None, :, :]), axis=-1)

    pb_xyxy = box_cxcywh_to_xyxy(pred_boxes).detach().cpu().numpy()
    gb_xyxy = box_cxcywh_to_xyxy(gt_boxes).detach().cpu().numpy()
    cost_giou_mat = -generalized_iou(
        torch.tensor(pb_xyxy), torch.tensor(gb_xyxy)
    ).numpy()

    C = cost_class * cost_cls + cost_bbox * cost_l1 + cost_giou * cost_giou_mat
    return C.astype(np.float32)


def hungarian_match(
    pred_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    gt_boxes: torch.Tensor,
    cost_class: float = 1.0,
    cost_bbox: float = 5.0,
    cost_giou: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    if gt_labels.shape[0] == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    C = build_cost_matrix(
        pred_logits,
        pred_boxes,
        gt_labels,
        gt_boxes,
        cost_class,
        cost_bbox,
        cost_giou,
    )
    query_idx, gt_idx = linear_sum_assignment(C)
    return query_idx.astype(np.int64), gt_idx.astype(np.int64)


@dataclass
class LossStats:
    total: float
    cls: float
    bbox_l1: float
    bbox_giou: float
    num_matched: int


class HungarianLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        loss_class: float = 1.0,
        loss_bbox: float = 5.0,
        loss_giou: float = 2.0,
        no_obj_coef: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.loss_class = loss_class
        self.loss_bbox = loss_bbox
        self.loss_giou = loss_giou
        self.no_obj_coef = no_obj_coef

    def _cls_loss(
        self,
        pred_logits: torch.Tensor,
        gt_labels: torch.Tensor,
        query_idx: np.ndarray,
        gt_idx: np.ndarray,
    ) -> torch.Tensor:
        num_queries = pred_logits.shape[0]
        targets = np.full((num_queries,), self.num_classes, dtype=np.int32)
        if len(query_idx) > 0:
            targets[query_idx] = gt_labels.detach().cpu().numpy().astype(np.int32)[gt_idx]

        targets_t = torch.tensor(targets, device=pred_logits.device, dtype=torch.long)

        weights = torch.ones(self.num_classes + 1, device=pred_logits.device)
        weights[self.num_classes] = self.no_obj_coef

        log_probs = F.log_softmax(pred_logits, dim=-1)
        nll = -log_probs.gather(1, targets_t.unsqueeze(1)).squeeze(1)
        sample_weights = weights[targets_t]
        return (nll * sample_weights).mean()

    def _box_losses(
        self,
        pred_boxes: torch.Tensor,
        gt_boxes: torch.Tensor,
        query_idx: np.ndarray,
        gt_idx: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(query_idx) == 0:
            zero = torch.tensor(0.0, device=pred_boxes.device)
            return zero, zero

        matched_pred = pred_boxes[query_idx]
        matched_gt = gt_boxes[gt_idx]

        l1 = F.l1_loss(matched_pred, matched_gt)

        pb_xyxy = box_cxcywh_to_xyxy(matched_pred)
        gb_xyxy = box_cxcywh_to_xyxy(matched_gt)
        giou_diag = torch.diag(generalized_iou(pb_xyxy, gb_xyxy))
        giou = (1.0 - giou_diag).mean()

        return l1, giou

    def _single_pass(
        self,
        predictions: Dict[str, torch.Tensor],
        targets,
    ) -> Tuple[torch.Tensor, List[LossStats]]:
        pred_logits = predictions["logits"]
        pred_boxes = predictions["boxes"]
        batch_size = pred_logits.shape[0]

        total_cls = torch.tensor(0.0, device=pred_logits.device)
        total_l1 = torch.tensor(0.0, device=pred_logits.device)
        total_giou = torch.tensor(0.0, device=pred_logits.device)
        stats: List[LossStats] = []

        for i in range(batch_size):
            if isinstance(targets, dict):
                labels_np = targets["labels"][i].detach().cpu().numpy()
                valid_idx = [j for j, v in enumerate(labels_np.tolist()) if v > 0]
                if valid_idx:
                    idx = torch.tensor(valid_idx, device=pred_logits.device)
                    gt_labels = targets["labels"][i][idx]
                    gt_boxes = targets["boxes"][i][idx]
                else:
                    gt_labels = torch.tensor([], device=pred_logits.device, dtype=torch.long)
                    gt_boxes = torch.zeros((0, 4), device=pred_logits.device)
            else:
                gt_labels = targets[i]["labels"]
                gt_boxes = targets[i]["boxes"]

            q_idx, g_idx = hungarian_match(
                pred_logits[i],
                pred_boxes[i],
                gt_labels,
                gt_boxes,
                self.cost_class,
                self.cost_bbox,
                self.cost_giou,
            )

            cls_l = self._cls_loss(pred_logits[i], gt_labels, q_idx, g_idx)
            l1_l, giou_l = self._box_losses(pred_boxes[i], gt_boxes, q_idx, g_idx)

            total_cls = total_cls + cls_l
            total_l1 = total_l1 + l1_l
            total_giou = total_giou + giou_l

            stats.append(
                LossStats(
                    total=0.0,
                    cls=float(cls_l.item()),
                    bbox_l1=float(l1_l.item()),
                    bbox_giou=float(giou_l.item()),
                    num_matched=len(q_idx),
                )
            )

        loss = (
            self.loss_class * total_cls / batch_size
            + self.loss_bbox * total_l1 / batch_size
            + self.loss_giou * total_giou / batch_size
        )
        return loss, stats

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets,
        aux_weight: float = 0.5,
    ) -> Tuple[torch.Tensor, List[LossStats]]:
        final_loss, stats = self._single_pass(predictions, targets)

        aux_loss = torch.tensor(0.0, device=final_loss.device)
        for aux_pred in predictions.get("aux", []):
            layer_loss, _ = self._single_pass(aux_pred, targets)
            aux_loss = aux_loss + layer_loss

        total_loss = final_loss + aux_weight * aux_loss

        for s in stats:
            s.total = float(total_loss.item())

        return total_loss, stats


if __name__ == "__main__":
    from dinov3.utils.device import get_device

    device = get_device()
    loss_fn = HungarianLoss(num_classes=80)
    predictions = {
        "logits": torch.randn(2, 300, 81, device=device),
        "boxes": torch.rand(2, 300, 4, device=device),
    }
    targets = [
        {
            "labels": torch.tensor([1, 2, 3], device=device),
            "boxes": torch.rand(3, 4, device=device),
        },
        {
            "labels": torch.tensor([4, 5], device=device),
            "boxes": torch.rand(2, 4, device=device),
        },
    ]
    loss, stats = loss_fn(predictions, targets)
    print(loss)
    print(stats)
