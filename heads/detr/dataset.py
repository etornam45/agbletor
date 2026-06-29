import ast
import json
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset

DATASET_NAME = "toufiqmusah/GhanaAgricVQA-Dataset"

DISEASE_LABELS = [
    "Corn_Cercospora_Leaf_Spot",
    "Corn_Common_Rust",
    "Corn_Healthy",
    "Corn_Northern_Leaf_Blight",
    "Corn_Streak",
    "Healthy",
    "Pepper_Anthracnose",
    "Pepper_Bacterial_Spot",
    "Pepper_Blossom_End_Rot",
    "Pepper_Cercospora",
    "Pepper_Fusarium",
    "Pepper_Healthy",
    "Pepper_Leaf_Blight",
    "Pepper_Leaf_Curl",
    "Pepper_Leaf_Mosaic",
    "Pepper_Septoria",
    "Tomato_Bacterial_Spot",
    "Tomato_Early_Blight",
    "Tomato_Fusarium",
    "Tomato_Healthy",
    "Tomato_Late_Blight",
    "Tomato_Leaf_Curl",
    "Tomato_Mosaic",
    "Tomato_Septoria",
]

SEVERITY_LABELS = ["mild", "moderate", "severe", "none"]

DISEASE_LABEL_TO_ID = {name: i + 1 for i, name in enumerate(DISEASE_LABELS)}
SEVERITY_LABEL_TO_ID = {name: i + 1 for i, name in enumerate(SEVERITY_LABELS)}
ID_TO_DISEASE_LABEL = {i + 1: name for i, name in enumerate(DISEASE_LABELS)}
ID_TO_SEVERITY_LABEL = {i + 1: name for i, name in enumerate(SEVERITY_LABELS)}

NUM_DISEASE_CLASSES = len(DISEASE_LABELS)
NUM_SEVERITY_CLASSES = len(SEVERITY_LABELS)


def letterbox(image, target_size):
    orig_w, orig_h = image.size
    scale = target_size / max(orig_w, orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    image = image.resize((new_w, new_h), Image.BILINEAR)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    canvas = Image.new("RGB", (target_size, target_size), (114, 114, 114))
    canvas.paste(image, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y


def parse_answer_dict(answer: Union[str, dict]) -> dict:
    if isinstance(answer, dict):
        return answer
    if isinstance(answer, str):
        stripped = answer.strip()
        if stripped.startswith("{"):
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(stripped)
                    if isinstance(parsed, dict):
                        return parsed
                except (json.JSONDecodeError, SyntaxError, ValueError):
                    continue
    return {}


def xyxy_to_xywh(boxes: List[List[float]]) -> List[List[float]]:
    if not boxes:
        return []
    result = []
    for x1, y1, x2, y2 in boxes:
        result.append([x1, y1, x2 - x1, y2 - y1])
    return result


def transform_boxes(boxes, scale, pad_x, pad_y, target_size):
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    boxes = np.array(boxes, dtype=np.float32)
    boxes[:, 0] = boxes[:, 0] * scale + pad_x
    boxes[:, 1] = boxes[:, 1] * scale + pad_y
    boxes[:, 2] = boxes[:, 2] * scale
    boxes[:, 3] = boxes[:, 3] * scale
    boxes[:, 0] = (boxes[:, 0] + boxes[:, 2] / 2) / target_size
    boxes[:, 1] = (boxes[:, 1] + boxes[:, 3] / 2) / target_size
    boxes[:, 2] /= target_size
    boxes[:, 3] /= target_size
    return np.clip(boxes, 0.0, 1.0)


def load_coco(img_dir, ann_file, img_size=640):
    coco = COCO(ann_file)
    samples = []

    for img_id in coco.getImgIds():
        img_info = coco.loadImgs(img_id)[0]
        path = f"{img_dir}/{img_info['file_name']}"

        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        orig_w, orig_h = img_info["width"], img_info["height"]

        scale = img_size / max(orig_w, orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        pad_x = (img_size - new_w) // 2
        pad_y = (img_size - new_h) // 2

        boxes = transform_boxes(
            [a["bbox"] for a in anns], scale, pad_x, pad_y, img_size
        )
        labels = np.array([a["category_id"] for a in anns], dtype=np.int32)

        samples.append(
            {
                "image_path": path,
                "boxes": boxes,
                "labels": labels,
                "num_objects": len(labels),
            }
        )

    return samples


class CocoDetectionDataset(Dataset):
    def __init__(self, samples, img_size=640):
        self.samples = samples
        self.img_size = img_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        image, _, _, _ = letterbox(image, self.img_size)
        image = np.array(image, dtype=np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # NCHW

        max_objs = 100
        boxes = np.zeros((max_objs, 4), dtype=np.float32)
        labels = np.zeros((max_objs,), dtype=np.int32)
        n = min(sample["num_objects"], max_objs)
        if n > 0:
            boxes[:n] = sample["boxes"][:n]
            labels[:n] = sample["labels"][:n]

        return {
            "image": image,
            "boxes": torch.from_numpy(boxes),
            "labels": torch.from_numpy(labels),
            "num_objects": n,
        }


def collate_fn(batch):
    images = torch.stack([b["image"] for b in batch])
    boxes = torch.stack([b["boxes"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    num_objects = torch.tensor([b["num_objects"] for b in batch])
    return {
        "image": images,
        "boxes": boxes,
        "labels": labels,
        "num_objects": num_objects,
    }


def make_dataloader(
    img_dir,
    ann_file,
    img_size=640,
    batch_size=16,
    shuffle=False,
    num_workers=4,
    pin_memory=None,
):
    samples = load_coco(img_dir, ann_file, img_size)
    dataset = CocoDetectionDataset(samples, img_size=img_size)

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=False,
    )
    num_batches = (len(dataset) + batch_size - 1) // batch_size
    return loader, num_batches


def load_ghana_agric_detections(
    split: str = "train",
    img_size: int = 224,
    language: str = "en",
    question_type: str = "identification",
):
    hf_dataset = load_dataset(DATASET_NAME, split=split)
    if language:
        langs = hf_dataset["language"]
        lang_indices = [
            i
            for i, lang in enumerate(langs)
            if (lang if lang is not None else "en") == language
        ]
        hf_dataset = hf_dataset.select(lang_indices)

    by_image: Dict[str, List[int]] = {}
    for idx in range(len(hf_dataset)):
        image_id = hf_dataset[idx].get("rail_image_id") or str(idx)
        by_image.setdefault(image_id, []).append(idx)

    selected_indices: List[int] = []
    for indices in by_image.values():
        chosen = indices[0]
        for idx in indices:
            if hf_dataset[idx].get("question_type") == question_type:
                chosen = idx
                break
        selected_indices.append(chosen)

    samples = []
    for idx in selected_indices:
        row = hf_dataset[idx]
        answer = parse_answer_dict(row["answer"])
        detections = answer.get("detections", [])
        if not detections:
            continue

        image = row["image"]
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")

        orig_w, orig_h = image.size
        scale = img_size / max(orig_w, orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        pad_x = (img_size - new_w) // 2
        pad_y = (img_size - new_h) // 2

        xywh_boxes = []
        labels = []
        severities = []
        for det in detections:
            disease = det.get("disease_label")
            if disease not in DISEASE_LABEL_TO_ID:
                continue
            bbox = det.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            xywh_boxes.append(xyxy_to_xywh([bbox])[0])
            labels.append(DISEASE_LABEL_TO_ID[disease])
            severity = det.get("severity", "none")
            severities.append(SEVERITY_LABEL_TO_ID.get(severity, SEVERITY_LABEL_TO_ID["none"]))

        if not labels:
            continue

        boxes = transform_boxes(xywh_boxes, scale, pad_x, pad_y, img_size)
        samples.append(
            {
                "hf_idx": idx,
                "boxes": boxes,
                "labels": np.array(labels, dtype=np.int32),
                "severities": np.array(severities, dtype=np.int32),
                "num_objects": len(labels),
            }
        )

    return hf_dataset, samples


class GhanaAgricDetectionDataset(Dataset):
    def __init__(self, hf_dataset, samples, img_size: int = 224):
        self.hf_dataset = hf_dataset
        self.samples = samples
        self.img_size = img_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        row = self.hf_dataset[sample["hf_idx"]]
        image = row["image"]
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")

        image, _, _, _ = letterbox(image, self.img_size)
        image = np.array(image, dtype=np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)

        max_objs = 100
        boxes = np.zeros((max_objs, 4), dtype=np.float32)
        labels = np.zeros((max_objs,), dtype=np.int32)
        severities = np.zeros((max_objs,), dtype=np.int32)
        n = min(sample["num_objects"], max_objs)
        if n > 0:
            boxes[:n] = sample["boxes"][:n]
            labels[:n] = sample["labels"][:n]
            severities[:n] = sample["severities"][:n]

        return {
            "image": image,
            "boxes": torch.from_numpy(boxes),
            "labels": torch.from_numpy(labels),
            "severities": torch.from_numpy(severities),
            "num_objects": n,
        }


def ghana_collate_fn(batch):
    images = torch.stack([b["image"] for b in batch])
    boxes = torch.stack([b["boxes"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    severities = torch.stack([b["severities"] for b in batch])
    num_objects = torch.tensor([b["num_objects"] for b in batch])
    return {
        "image": images,
        "boxes": boxes,
        "labels": labels,
        "severities": severities,
        "num_objects": num_objects,
    }


def make_ghana_dataloader(
    split: str = "train",
    img_size: int = 224,
    batch_size: int = 16,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    language: str = "en",
    question_type: str = "identification",
) -> Tuple[DataLoader, int]:
    hf_dataset, samples = load_ghana_agric_detections(
        split=split,
        img_size=img_size,
        language=language,
        question_type=question_type,
    )
    dataset = GhanaAgricDetectionDataset(hf_dataset, samples, img_size=img_size)

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=ghana_collate_fn,
        pin_memory=pin_memory,
        drop_last=split == "train",
    )
    num_batches = (len(dataset) + batch_size - 1) // batch_size
    return loader, num_batches


# Backwards-compatible alias
make_stream = make_dataloader


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    img_size = 224

    t0 = time.time()
    loader, _ = make_ghana_dataloader(
        split="train",
        img_size=img_size,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )
    print(f"Ghana Agric DataLoader ready in {time.time() - t0:.2f}s")

    batch = next(iter(loader))
    images = batch["image"]
    boxes = batch["boxes"].numpy()
    labels = batch["labels"].numpy()
    n_objs = batch["num_objects"].numpy()

    img = images[0].permute(1, 2, 0).numpy()

    print(img.shape, "objects:", n_objs[0])
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    for j, (cx, cy, w, h) in enumerate(boxes[0, : n_objs[0]]):
        x = (cx - w / 2) * img_size
        y = (cy - h / 2) * img_size
        label = ID_TO_DISEASE_LABEL.get(labels[0, j], "?")
        ax.add_patch(
            patches.Rectangle(
                (x, y),
                w * img_size,
                h * img_size,
                linewidth=1,
                edgecolor="pink",
                facecolor="none",
            )
        )
        ax.text(x, y, label, fontsize=6, color="white")
    plt.savefig("ghana_detr_sample.png")
    print("Saved ghana_detr_sample.png")
