"""
Generic converter for DINOv3 PyTorch checkpoints to MLX format.
Supports all model sizes from ViT-S to ViT-7B.
"""

import argparse
import os
from pathlib import Path
import torch
import mlx.core as mx
from PIL import Image

from dinov3.models import (
    vit_small,
    vit_base,
    vit_large,
    vit_so400m,
    vit_huge2,
    vit_giant2,
    vit_7b,
)

MODEL_MAP = {
    "vit_small": vit_small,
    "vit_base": vit_base,
    "vit_large": vit_large,
    "vit_so400m": vit_so400m,
    "vit_huge": vit_huge2,
    "vit_giant": vit_giant2,
    "vit_7b": vit_7b,
}

# Short aliases
MODEL_MAP.update(
    {
        "vits": vit_small,
        "vitb": vit_base,
        "vitl": vit_large,
        "vith": vit_huge2,
        "vitg": vit_giant2,
    }
)


def torch_to_mlx(v: torch.Tensor, key: str = "") -> mx.array:
    """Convert a PyTorch tensor to an MLX array, handling bfloat16 and Conv2d weights."""
    if v.dtype == torch.bfloat16:
        arr = mx.array(v.float().numpy(), dtype=mx.bfloat16)
    else:
        arr = mx.array(v.numpy())

    # MLX Conv2d expects OHWI; PyTorch stores as OIHW
    if "patch_embed.proj.weight" in key:
        arr = arr.transpose(0, 2, 3, 1)

    return arr


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
    return "vit_small"  # Default fallback


def convert(pth_path: str, out_path: str, model_type: str = None, patch_size: int = 16):
    print(f"Loading PyTorch checkpoint from {pth_path}...")
    state = torch.load(pth_path, map_location="cpu")

    # Handle cases where the state dict is nested (e.g., 'model' or 'teacher')
    if "model" in state:
        state = state["model"]
    elif "teacher" in state:
        state = state["teacher"]

    # Remove 'module.' prefix if present
    state = {k.replace("module.", ""): v for k, v in state.items()}

    # Detect storage tokens (registers)
    n_storage_tokens = 0
    if "storage_tokens" in state:
        n_storage_tokens = state["storage_tokens"].shape[1]

    # Auto-detect FFN type and ratio
    ffn_layer = "mlp"
    ffn_ratio = 4.0
    embed_dim = 0

    # Try to find embed_dim and hidden_dim
    for k, v in state.items():
        if "blocks.0.norm1.weight" in k:
            embed_dim = v.shape[0]
        if "blocks.0.mlp.w1.weight" in k:
            ffn_layer = "swiglu"
            swiglu_hidden = v.shape[0]
            # Since SwiGLUFFN does d = int(hidden_features * 2 / 3),
            # we need to pass a ffn_ratio that results in the correct swiglu_hidden.
            # hidden_features = swiglu_hidden * 3 / 2
            # ffn_ratio = hidden_features / embed_dim = (swiglu_hidden * 1.5) / embed_dim
            if embed_dim > 0:
                ffn_ratio = (swiglu_hidden * 1.5) / embed_dim
        elif "blocks.0.mlp.fc1.weight" in k:
            ffn_layer = "mlp"
            hidden_dim = v.shape[0]
            if embed_dim > 0:
                ffn_ratio = hidden_dim / embed_dim

    print(
        f"Detected FFN: type={ffn_layer}, ratio={ffn_ratio:.4f}, embed_dim={embed_dim}"
    )

    if model_type is None:
        model_type = guess_model_type(pth_path)
        print(f"Guessed model type: {model_type}")

    if model_type not in MODEL_MAP:
        print(
            f"Error: Unknown model type '{model_type}'. Available: {list(MODEL_MAP.keys())}"
        )
        return

    print(
        f"Initializing MLX model (type={model_type}, patch_size={patch_size}, storage_tokens={n_storage_tokens}, ffn={ffn_layer}, ratio={ffn_ratio:.4f})..."
    )
    model_fn = MODEL_MAP[model_type]
    model = model_fn(
        patch_size=patch_size,
        n_storage_tokens=n_storage_tokens,
        layerscale_init=1e-5,
        mask_k_bias=True,
        ffn_layer=ffn_layer,
        ffn_ratio=ffn_ratio,
    )

    weights = []
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            weights.append((k, torch_to_mlx(v, k)))

    print("Loading weights into MLX model...")
    model.load_weights(weights, strict=True)

    out_path_obj = Path(out_path)
    out_path_obj.parent.mkdir(parents=True, exist_ok=True)

    model.save_weights(str(out_path))
    print(f"Successfully saved MLX weights to {out_path}")


def verify_conversion(
    mlx_path: str, hf_name: str, image_path: str, model_type: str, patch_size: int
):
    """Optional verification against HuggingFace DINOv3 implementation if available."""
    try:
        from transformers import AutoModel, AutoProcessor
    except ImportError:
        print("Transformers not installed, skipping verification.")
        return

    print(f"\nVerifying conversion against HF model: {hf_name}")

    # Load MLX model
    state = mx.load(mlx_path)
    n_storage_tokens = 0
    if "storage_tokens" in state:
        n_storage_tokens = state["storage_tokens"].shape[1]

    model_fn = MODEL_MAP[model_type]
    mlx_model = model_fn(
        patch_size=patch_size,
        n_storage_tokens=n_storage_tokens,
        layerscale_init=1e-5,
        mask_k_bias=True,
    )
    mlx_model.load_weights(mlx_path)

    # Load HF model
    hf_model = AutoModel.from_pretrained(hf_name)
    processor = AutoProcessor.from_pretrained(hf_name)

    # Prepare image
    if not os.path.exists(image_path):
        print(f"Image {image_path} not found, skipping comparison.")
        return

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    pixel_values_pt = inputs["pixel_values"]
    pixel_values_mlx = mx.array(pixel_values_pt.permute(0, 2, 3, 1).numpy())

    # Forward passes
    mlx_out = mlx_model(pixel_values_mlx, is_training=True)
    mx.eval(mlx_out)
    mlx_cls = mlx_out["x_norm_clstoken"]

    with torch.no_grad():
        hf_out = hf_model(**inputs)
    pt_cls = mx.array(hf_out.last_hidden_state[:, 0, :].numpy())

    # Compare
    diff = mx.abs(mlx_cls - pt_cls)
    print(f"Max difference (CLS):  {mx.max(diff).item():.6f}")
    print(f"Mean difference (CLS): {mx.mean(diff).item():.6f}")

    dot = mx.sum(mlx_cls * pt_cls)
    norm_mlx = mx.sqrt(mx.sum(mlx_cls**2))
    norm_pt = mx.sqrt(mx.sum(pt_cls**2))
    sim = (dot / (norm_mlx * norm_pt)).item()
    print(f"Cosine similarity:     {sim:.6f}")

    if sim > 0.999:
        print("✅ Conversion verified!")
    else:
        print("⚠️ Conversion might have issues. Check similarities.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert DINOv3 PyTorch weights to MLX."
    )
    parser.add_argument(
        "--pth-path", type=str, help="Path to a single PyTorch .pth file"
    )
    parser.add_argument(
        "--dir", type=str, help="Directory to search for .pth files and convert all"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="dinov3/checkpoints/model",
        help="Output directory for MLX weights",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=list(MODEL_MAP.keys()),
        help="Specify model type",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=14,
        help="Patch size (default 14 for most DINOv3)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Attempt verification against HF (requires --hf-name)",
    )
    parser.add_argument(
        "--hf-name", type=str, help="HuggingFace model name for verification"
    )
    parser.add_argument(
        "--image-path", type=str, default="image.jpg", help="Image for verification"
    )

    args = parser.parse_args()

    if args.pth_path:
        out_name = Path(args.pth_path).stem + ".safetensors"
        out_path = os.path.join(args.out_dir, out_name)
        convert(args.pth_path, out_path, args.model_type, args.patch_size)

        if args.verify and args.hf_name:
            m_type = args.model_type or guess_model_type(args.pth_path)
            verify_conversion(
                out_path, args.hf_name, args.image_path, m_type, args.patch_size
            )

    elif args.dir:
        pth_files = list(Path(args.dir).glob("*.pth"))
        print(f"Found {len(pth_files)} .pth files in {args.dir}")
        for pth in pth_files:
            print(f"\n--- Converting {pth.name} ---")
            out_name = pth.stem + ".safetensors"
            out_path = os.path.join(args.out_dir, out_name)
            convert(str(pth), out_path, args.model_type, args.patch_size)
    else:
        # Default behavior (legacy) if no args provided
        DEFAULT_PTH = (
            "dinov3/checkpoints/model/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
        )
        if os.path.exists(DEFAULT_PTH):
            out_path = "dinov3/checkpoints/model/vit-small.safetensors"
            convert(DEFAULT_PTH, out_path, "vit_small", 16)
        else:
            parser.print_help()
