"""
Load official DINOv3 PyTorch checkpoints into dinov3 models.
"""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from PIL import Image

from dinov3.models import (
    vit_7b,
    vit_base,
    vit_giant2,
    vit_huge2,
    vit_large,
    vit_small,
    vit_so400m,
)

MODEL_MAP = {
    "vit_small": vit_small,
    "vit_base": vit_base,
    "vit_large": vit_large,
    "vit_so400m": vit_so400m,
    "vit_huge": vit_huge2,
    "vit_giant": vit_giant2,
    "vit_7b": vit_7b,
    "vits": vit_small,
    "vitb": vit_base,
    "vitl": vit_large,
    "vith": vit_huge2,
    "vitg": vit_giant2,
}


def guess_model_type(path: str) -> str:
    name = Path(path).name.lower()
    if "vits" in name or "vit-small" in name or "vits16" in name or "vits14" in name:
        return "vit_small"
    if "vitb" in name or "vit-base" in name:
        return "vit_base"
    if "vitl" in name or "vit-large" in name:
        return "vit_large"
    if "so400m" in name:
        return "vit_so400m"
    if "vith" in name or "vit-huge" in name:
        return "vit_huge"
    if "vitg" in name or "vit-giant" in name:
        return "vit_giant"
    if "7b" in name:
        return "vit_7b"
    return "vit_small"


def extract_state_dict(state: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    if "model" in state:
        state = state["model"]
    elif "teacher" in state:
        state = state["teacher"]
    return {k.replace("module.", ""): v for k, v in state.items() if isinstance(v, torch.Tensor)}


def infer_model_kwargs(state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "n_storage_tokens": 0,
        "layerscale_init": 1e-5,
        "mask_k_bias": True,
        "ffn_layer": "mlp",
        "ffn_ratio": 4.0,
    }

    if "storage_tokens" in state:
        kwargs["n_storage_tokens"] = int(state["storage_tokens"].shape[1])

    embed_dim = 0
    for k, v in state.items():
        if "blocks.0.norm1.weight" in k:
            embed_dim = v.shape[0]
        if "blocks.0.mlp.w1.weight" in k:
            kwargs["ffn_layer"] = "swiglu"
            swiglu_hidden = v.shape[0]
            if embed_dim > 0:
                kwargs["ffn_ratio"] = (swiglu_hidden * 1.5) / embed_dim
        elif "blocks.0.mlp.fc1.weight" in k:
            kwargs["ffn_layer"] = "mlp"
            hidden_dim = v.shape[0]
            if embed_dim > 0:
                kwargs["ffn_ratio"] = hidden_dim / embed_dim

    return kwargs


def build_model(
    model_type: str,
    patch_size: int,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> torch.nn.Module:
    if model_type not in MODEL_MAP:
        raise ValueError(
            f"Unknown model type '{model_type}'. Choose from: {list(MODEL_MAP.keys())}"
        )
    kwargs = model_kwargs or {}
    return MODEL_MAP[model_type](patch_size=patch_size, **kwargs)


def load_checkpoint(
    model: torch.nn.Module,
    pth_path: str,
    strict: bool = True,
) -> torch.nn.Module:
    state = torch.load(pth_path, map_location="cpu", weights_only=True)
    state_dict = extract_state_dict(state)
    model.load_state_dict(state_dict, strict=strict)
    return model


def load_pretrained(
    pth_path: str,
    model_type: Optional[str] = None,
    patch_size: int = 16,
    device: Optional[torch.device] = None,
) -> torch.nn.Module:
    state = torch.load(pth_path, map_location="cpu", weights_only=True)
    state_dict = extract_state_dict(state)

    if model_type is None:
        model_type = guess_model_type(pth_path)

    model_kwargs = infer_model_kwargs(state_dict)
    model = build_model(model_type, patch_size, model_kwargs)
    model.load_state_dict(state_dict, strict=True)

    if device is not None:
        model = model.to(device)
    return model


def verify_checkpoint(
    model: torch.nn.Module,
    hf_name: str,
    image_path: str,
    device: torch.device,
) -> None:
    try:
        from transformers import AutoModel, AutoProcessor
    except ImportError:
        print("Transformers not installed, skipping verification.")
        return

    print(f"\nVerifying checkpoint against HF model: {hf_name}")

    hf_model = AutoModel.from_pretrained(hf_name)
    processor = AutoProcessor.from_pretrained(hf_name)

    if not os.path.exists(image_path):
        print(f"Image {image_path} not found, skipping comparison.")
        return

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    model.eval()
    with torch.no_grad():
        out = model(pixel_values, is_training=True)
        local_cls = out["x_norm_clstoken"]
        hf_out = hf_model(**inputs)
        hf_cls = hf_out.last_hidden_state[:, 0, :].to(device)

    diff = torch.abs(local_cls - hf_cls)
    print(f"Max difference (CLS):  {diff.max().item():.6f}")
    print(f"Mean difference (CLS): {diff.mean().item():.6f}")

    sim = torch.nn.functional.cosine_similarity(local_cls, hf_cls, dim=-1).item()
    print(f"Cosine similarity:     {sim:.6f}")

    if sim > 0.999:
        print("Conversion verified!")
    else:
        print("Checkpoint might have issues. Check similarities.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load DINOv3 PyTorch checkpoint.")
    parser.add_argument("--pth-path", type=str, required=True, help="Path to .pth file")
    parser.add_argument(
        "--model-type",
        type=str,
        choices=list(MODEL_MAP.keys()),
        help="Model type (auto-detected if omitted)",
    )
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--hf-name", type=str, help="HuggingFace model for verification")
    parser.add_argument("--image-path", type=str, default="image.jpg")
    args = parser.parse_args()

    from dinov3.utils.device import get_device

    device = get_device()
    model = load_pretrained(args.pth_path, args.model_type, args.patch_size, device)
    print(f"Loaded {args.model_type or guess_model_type(args.pth_path)} on {device}")

    if args.verify and args.hf_name:
        verify_checkpoint(model, args.hf_name, args.image_path, device)
