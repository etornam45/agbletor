# DINOv3 + MiniCPM5-1B VQA Head

Hybrid vision-language model for crop-disease VQA on [GhanaAgricVQA](https://huggingface.co/datasets/toufiqmusah/GhanaAgricVQA-Dataset):

- **Vision**: frozen DINOv3 ViT-S patch tokens (384-dim)
- **Vision adapter**: linear 384 → 1536 + LayerNorm + 2D sin-cos position encoding, pooled to 64 visual prefix tokens (LLaVA-style)
- **Language**: LoRA-tuned [MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B) (`LlamaForCausalLM`, ChatML template)

Old BLIP checkpoints (`vqa_hybrid.pt`) are incompatible with this architecture.

## Dataset

Loaded automatically from Hugging Face on first run (~3.8 GB).

```python
from datasets import load_dataset
dataset = load_dataset("toufiqmusah/GhanaAgricVQA-Dataset")
```

## Train

Requires DINOv3 ViT-S weights at `dinov3/checkpoints/model/dinov3_vits16_pretrain_lvd1689m-08c60483.pth` (auto-downloaded on first run if missing or corrupt).

First run also downloads MiniCPM5-1B (~2 GB). If the model is already cached locally, loading uses `local_files_only=True`.

```bash
python -m heads.vqa.train --epochs 5 --batch-size 4 --llm-lr 2e-4 --adapter-lr 1e-4
```

Resume from a checkpoint directory:

```bash
python -m heads.vqa.train --resume dinov3/checkpoints/model/vqa_minicpm --epochs 5
```

Saves to `dinov3/checkpoints/model/vqa_minicpm/`:

- `adapter/` — PEFT LoRA weights
- `vision_adapter.pt` — projection + LayerNorm

Best eval checkpoint: `vqa_minicpm_best/`.

## Inference

```bash
python -m heads.vqa.inference
```

```python
from heads.vqa.inference import run_inference

run_inference(
    image_path="path/to/image.jpg",
    question="What disease is affecting my maize plant?",
)
```

## Smoke Tests

```bash
python -m heads.vqa.model
python -m heads.vqa.dataset
```
