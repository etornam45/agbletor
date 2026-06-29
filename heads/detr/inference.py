from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from dinov3.checkpoints.load import load_checkpoint
from dinov3.models import vit_small
from dinov3.utils.device import get_device
from heads.detr.dataset import (
    ID_TO_DISEASE_LABEL,
    ID_TO_SEVERITY_LABEL,
    NUM_DISEASE_CLASSES,
    NUM_SEVERITY_CLASSES,
    letterbox,
    load_ghana_agric_detections,
)
from heads.detr.transformer import build_detr

IMG_SIZE = 224
N_CLASSES = NUM_DISEASE_CLASSES + 1
N_SEVERITY_CLASSES = NUM_SEVERITY_CLASSES + 1
CHECKPOINT_PATH = "dinov3/checkpoints/model/detr_ghana_decoder.pt"


def cxcywh_norm_to_xyxy_orig(
    boxes: np.ndarray,
    scale: float,
    pad_x: int,
    pad_y: int,
    target_size: int,
) -> np.ndarray:
    cx = boxes[:, 0] * target_size
    cy = boxes[:, 1] * target_size
    w = boxes[:, 2] * target_size
    h = boxes[:, 3] * target_size

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    x1 = (x1 - pad_x) / scale
    y1 = (y1 - pad_y) / scale
    x2 = (x2 - pad_x) / scale
    y2 = (y2 - pad_y) / scale
    return np.stack([x1, y1, x2, y2], axis=-1)


def load_model(device: torch.device):
    dinov3_small = vit_small(
        patch_size=16,
        n_storage_tokens=4,
        layerscale_init=1e-5,
        mask_k_bias=True,
    )
    load_checkpoint(
        dinov3_small,
        "dinov3/checkpoints/model/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    )
    dinov3_small.to(device)
    dinov3_small.eval()

    detr_decoder = build_detr(
        d_model=384,
        num_layers=4,
        n_classes=N_CLASSES,
        n_severity_classes=N_SEVERITY_CLASSES,
        n_points=5,
    ).to(device)
    detr_decoder.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    )
    detr_decoder.eval()
    return dinov3_small, detr_decoder


def run_inference(
    image_path: Optional[str] = None,
    threshold: float = 0.5,
) -> Dict[str, List[dict]]:
    device = get_device()
    dinov3_small, detr_decoder = load_model(device)

    if image_path is None:
        hf_dataset, samples = load_ghana_agric_detections(split="test", img_size=IMG_SIZE)
        row = hf_dataset[samples[0]["hf_idx"]]
        img_pil = row["image"]
        if not isinstance(img_pil, Image.Image):
            img_pil = Image.open(img_pil).convert("RGB")
        else:
            img_pil = img_pil.convert("RGB")
    else:
        img_pil = Image.open(image_path).convert("RGB")

    img_pil_lb, scale, pad_x, pad_y = letterbox(img_pil, IMG_SIZE)
    img_arr = np.array(img_pil_lb, dtype=np.float32) / 255.0
    image = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        features = dinov3_small(image, masks=None, is_training=True)
        patch_tokens = features["x_norm_patchtokens"]
        output = detr_decoder(patch_tokens)

    logits = output["logits"][0]
    boxes = output["boxes"][0]
    severity_logits = output["severity_logits"][0]

    probs = F.softmax(logits, dim=-1)
    scores, label_idx = probs[:, 1:-1].max(dim=-1)
    labels = label_idx + 1

    sev_probs = F.softmax(severity_logits, dim=-1)
    sev_scores, sev_idx = sev_probs[:, 1:-1].max(dim=-1)
    severities = sev_idx + 1

    keep = scores > threshold
    scores_np = scores[keep].cpu().numpy()
    labels_np = labels[keep].cpu().numpy()
    boxes_np = boxes[keep].cpu().numpy()
    severities_np = severities[keep].cpu().numpy()
    sev_scores_np = sev_scores[keep].cpu().numpy()

    boxes_xyxy = cxcywh_norm_to_xyxy_orig(
        boxes_np, scale, pad_x, pad_y, IMG_SIZE
    )

    detections = []
    for score, label, sev_score, severity, bbox in zip(
        scores_np, labels_np, sev_scores_np, severities_np, boxes_xyxy
    ):
        detections.append(
            {
                "bbox": [float(b) for b in bbox],
                "disease_label": ID_TO_DISEASE_LABEL.get(int(label), "unknown"),
                "severity": ID_TO_SEVERITY_LABEL.get(int(severity), "none"),
                "score": float(score),
                "severity_score": float(sev_score),
            }
        )

    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(img_arr)

    for det in detections:
        x1_lb = det["bbox"][0] * scale + pad_x
        y1_lb = det["bbox"][1] * scale + pad_y
        x2_lb = det["bbox"][2] * scale + pad_x
        y2_lb = det["bbox"][3] * scale + pad_y
        rect = patches.Rectangle(
            (x1_lb, y1_lb),
            x2_lb - x1_lb,
            y2_lb - y1_lb,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
            alpha=0.9,
        )
        ax.add_patch(rect)
        ax.text(
            x1_lb,
            y1_lb,
            f"{det['disease_label']} ({det['severity']}): {det['score']:.2f}",
            bbox=dict(facecolor="red", alpha=0.5),
            fontsize=8,
            color="white",
        )

    plt.axis("off")
    plt.title(f"DETR Ghana Agric Predictions (threshold={threshold})")
    plt.savefig("detr_output.png")
    print("Results saved to detr_output.png")

    return {"detections": detections}


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else None
    result = run_inference(image_path=path)
    print(result)
