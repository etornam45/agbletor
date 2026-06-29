# Ghana Agric DETR — Crop Disease Detection

Object detection head trained on [GhanaAgricVQA](https://huggingface.co/datasets/toufiqmusah/GhanaAgricVQA-Dataset) using DINOv3 ViT-S/16 (frozen) + deformable DETR decoder.

Detects crop disease regions with **bounding box**, **disease label**, and **severity** (mild / moderate / severe / none).

## Dataset

Annotations are parsed from the structured `answer` field in GhanaAgricVQA:

```python
{
  "text": "...",
  "detections": [
    {"bbox": [x1, y1, x2, y2], "disease_label": "Corn_Streak", "severity": "moderate"}
  ]
}
```

- One sample per image (deduplicated by `rail_image_id`, using `identification` QA rows)
- ~663 train images, ~117 test images
- 24 disease classes across maize, pepper, and tomato

No manual dataset download required — images are fetched from Hugging Face via `datasets`.

## Training

```bash
python -m heads.detr.train
```

Trains the DETR decoder on the Ghana Agric train split and validates on the test split each epoch. Checkpoint saved to:

```
dinov3/checkpoints/model/detr_ghana_decoder.pt
```

## Inference

```bash
# First test image from Ghana Agric test split
python -m heads.detr.inference

# Custom image
python -m heads.detr.inference path/to/image.jpg
```

Output format:

```python
{
  "detections": [
    {
      "bbox": [802.8, 637.4, 1155.8, 895.6],
      "disease_label": "Corn_Streak",
      "severity": "moderate",
      "score": 0.87,
      "severity_score": 0.72
    }
  ]
}
```

Visualization saved to `detr_output.png`.

## Visualize data loader

```bash
python -m heads.detr.dataset
```

Saves `ghana_detr_sample.png` with ground-truth boxes overlaid.

## COCO (legacy)

The original COCO loader remains in `dataset.py` (`make_dataloader`) for reference but is no longer used by default training.
