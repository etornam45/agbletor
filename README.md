# PyTorch DINOv3

A PyTorch implementation of **DINOv3** (Meta's self-supervised ViT models) with DETR object-detection and DINOv3+MiniCPM VQA heads. The backbone is vendored from Meta's [official DINOv3 repository](https://github.com/facebookresearch/dinov3).

Supports CUDA, Apple Silicon (MPS), and CPU with automatic device detection.

## Features

- **PyTorch-native**: Uses Meta's official DINOv3 ViT implementation.
- **Cross-platform**: Auto-detects CUDA > MPS > CPU.
- **Multiple Architectures**: `vit_small`, `vit_base`, `vit_large`, `vit_so400m`, `vit_huge2`, `vit_giant2`, `vit_7b`.
- **Advanced Components**: RoPE, SwiGLU FFN, LayerScale, storage/register tokens.
- **DETR Detection Head**: Trainable deformable-attention decoder on COCO using frozen DINOv3 patch tokens.
- **VQA Head**: DINOv3 + MiniCPM5-1B hybrid (LoRA + vision adapter) for GhanaAgricVQA crop-disease question answering.

## Installation

```bash
pip install -e .
```

Or install dependencies directly:

```bash
pip install torch transformers matplotlib numpy pillow scipy pycocotools tqdm pyyaml safetensors datasets nltk peft accelerate
```

## Usage

### 1. Load Pretrained Weights

Download an official DINOv3 `.pth` checkpoint and load it directly:

```bash
python dinov3/checkpoints/load.py \
    --pth-path dinov3/checkpoints/model/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
    --verify --hf-name facebook/dinov3-vits16-pretrain-lvd1689m
```

### 2. Running Inference

```python
import torch
from dinov3.models import vit_small
from dinov3.checkpoints.load import load_checkpoint
from dinov3.utils.device import get_device

device = get_device()
model = vit_small(patch_size=16, n_storage_tokens=4)
load_checkpoint(model, "path/to/checkpoint.pth")
model.to(device).eval()

image = torch.randn(1, 3, 224, 224, device=device)  # NCHW
outputs = model(image, is_training=True)
patch_tokens = outputs["x_norm_patchtokens"]  # (1, 196, 384)
```

### 3. DETR Training

```bash
python heads/detr/train.py
```

### 4. DETR Inference

```bash
python heads/detr/inference.py
```

### 5. VQA Training (DINOv3 + MiniCPM5-1B)

```bash
python -m heads.vqa.train --epochs 5 --batch-size 4
```

### 6. VQA Inference

```bash
python -m heads.vqa.inference
```

## Repository Structure

- `dinov3/models/`: Core ViT architecture (vendored from Meta).
- `dinov3/layers/`: RoPE, SwiGLU, Attention, etc. (vendored from Meta).
- `dinov3/checkpoints/`: Checkpoint loading utilities.
- `heads/detr/`: DETR detection head, training, and inference.
- `heads/vqa/`: DINOv3 + MiniCPM5-1B hybrid VQA head for GhanaAgricVQA.

## Acknowledgments

The DINOv3 backbone code is vendored from the official [DINOv3](https://github.com/facebookresearch/dinov3) repository by Meta AI, under the DINOv3 License Agreement.
