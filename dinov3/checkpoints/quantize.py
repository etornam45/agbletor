"""
Quantize DINOv3 MLX checkpoints using mlx.nn.quantize.

Example:
    python dinov3/checkpoints/quantize.py \\
        --weights dinov3/checkpoints/model/vit-small.safetensors \\
        --verify --image-path image.jpg
"""

import argparse
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from dinov3.checkpoints.convert import MODEL_MAP, guess_model_type
from dinov3.layers.attention import LinearKMaskedBias


def infer_model_kwargs(weights_path: str) -> Dict[str, Any]:
    """Infer constructor kwargs from a safetensors checkpoint."""
    state = mx.load(weights_path)
    keys = list(state.keys())

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
    for k in keys:
        if "blocks.0.norm1.weight" in k:
            embed_dim = int(state[k].shape[0])
        if "blocks.0.mlp.w1.weight" in k:
            kwargs["ffn_layer"] = "swiglu"
            swiglu_hidden = int(state[k].shape[0])
            if embed_dim > 0:
                kwargs["ffn_ratio"] = (swiglu_hidden * 1.5) / embed_dim
        elif "blocks.0.mlp.fc1.weight" in k:
            kwargs["ffn_layer"] = "mlp"
            hidden_dim = int(state[k].shape[0])
            if embed_dim > 0:
                kwargs["ffn_ratio"] = hidden_dim / embed_dim

    return kwargs


def build_model(
    model_type: str,
    patch_size: int,
    model_kwargs: Dict[str, Any],
) -> nn.Module:
    if model_type not in MODEL_MAP:
        raise ValueError(
            f"Unknown model type '{model_type}'. Choose from: {list(MODEL_MAP.keys())}"
        )
    return MODEL_MAP[model_type](patch_size=patch_size, **model_kwargs)


def make_class_predicate(
    skip_masked_qkv: bool,
) -> Callable[[str, nn.Module], bool]:
    def class_predicate(path: str, module: nn.Module) -> bool:
        if skip_masked_qkv and isinstance(module, LinearKMaskedBias):
            return False
        return hasattr(module, "to_quantized")

    return class_predicate


def quantize_model(
    model: nn.Module,
    *,
    bits: int,
    group_size: int,
    mode: str,
    quantize_input: bool,
    skip_masked_qkv: bool,
) -> None:
    predicate = make_class_predicate(skip_masked_qkv)
    nn.quantize(
        model,
        group_size=group_size,
        bits=bits,
        mode=mode,
        quantize_input=quantize_input,
        class_predicate=predicate,
    )


def load_image(image_path: str, img_size: int = 224) -> mx.array:
    from PIL import Image

    image = Image.open(image_path).convert("RGB").resize((img_size, img_size))
    arr = np.array(image, dtype=np.float32)
    return mx.array(arr)[None]


def forward_cls(model: nn.Module, image: mx.array) -> mx.array:
    out = model(image, is_training=True)
    mx.eval(out)
    return out["x_norm_clstoken"]


def verify_quantization(
    fp32_path: str,
    quant_path: str,
    model_type: str,
    patch_size: int,
    model_kwargs: Dict[str, Any],
    quant_config: Dict[str, Any],
    image_path: Optional[str],
) -> None:
    print("\nVerifying quantized weights against FP32 reference...")

    if image_path and os.path.exists(image_path):
        image = load_image(image_path)
    else:
        if image_path:
            print(f"Image {image_path} not found, using random input.")
        image = mx.random.uniform(shape=(1, 224, 224, 3))

    fp32_model = build_model(model_type, patch_size, model_kwargs)
    fp32_model.load_weights(fp32_path)
    fp32_model.eval()

    quant_model = build_model(model_type, patch_size, model_kwargs)
    quantize_model(quant_model, **quant_config)
    quant_model.load_weights(quant_path)
    quant_model.eval()

    fp32_cls = forward_cls(fp32_model, image)
    quant_cls = forward_cls(quant_model, image)

    diff = mx.abs(fp32_cls - quant_cls)
    dot = mx.sum(fp32_cls * quant_cls)
    norm_fp32 = mx.sqrt(mx.sum(fp32_cls**2))
    norm_quant = mx.sqrt(mx.sum(quant_cls**2))
    sim = (dot / (norm_fp32 * norm_quant)).item()

    print(f"Max difference (CLS):  {mx.max(diff).item():.6f}")
    print(f"Mean difference (CLS): {mx.mean(diff).item():.6f}")
    print(f"Cosine similarity:     {sim:.6f}")

    if sim > 0.99:
        print("Quantization looks good.")
    else:
        print("Large drift detected — try a higher bit width or group_size=64.")


def format_size(path: str) -> str:
    size_mb = os.path.getsize(path) / (1024 * 1024)
    return f"{size_mb:.2f} MB"


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize DINOv3 MLX checkpoints.")
    parser.add_argument(
        "--weights",
        type=str,
        default="dinov3/checkpoints/model/vit-small.safetensors",
        help="Input FP32 safetensors checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (default: <input>-q<bits>.safetensors)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=list(MODEL_MAP.keys()),
        default=None,
        help="Model architecture (auto-guessed from filename if omitted)",
    )
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 5, 6, 8])
    parser.add_argument(
        "--group-size",
        type=int,
        default=64,
        choices=[32, 64, 128],
        help="Affine quantization group size",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="affine",
        choices=["affine", "mxfp4", "mxfp8", "nvfp4"],
    )
    parser.add_argument(
        "--quantize-input",
        action="store_true",
        help="Quantize activations (nvfp4/mxfp8 only)",
    )
    parser.add_argument(
        "--quantize-masked-qkv",
        action="store_true",
        help="Also quantize LinearKMaskedBias QKV layers (may change K-bias masking)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Compare FP32 vs quantized CLS embeddings after quantization",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default="image.jpg",
        help="Image for --verify (random tensor if missing)",
    )
    args = parser.parse_args()

    weights_path = args.weights
    if not os.path.exists(weights_path):
        parser.error(f"Weights not found: {weights_path}")

    model_type = args.model_type or guess_model_type(weights_path)
    model_kwargs = infer_model_kwargs(weights_path)

    if args.output:
        out_path = args.output
    else:
        stem = Path(weights_path).stem
        out_path = str(Path(weights_path).with_name(f"{stem}-q{args.bits}.safetensors"))

    quant_config = {
        "bits": args.bits,
        "group_size": args.group_size,
        "mode": args.mode,
        "quantize_input": args.quantize_input,
        "skip_masked_qkv": not args.quantize_masked_qkv,
    }

    print(f"Loading {weights_path} ({format_size(weights_path)})...")
    print(f"Model type: {model_type}, patch_size: {args.patch_size}")
    print(f"Inferred kwargs: {model_kwargs}")
    print(
        f"Quantizing: bits={args.bits}, group_size={args.group_size}, "
        f"mode={args.mode}, skip_masked_qkv={quant_config['skip_masked_qkv']}"
    )

    model = build_model(model_type, args.patch_size, model_kwargs)
    model.load_weights(weights_path)
    quantize_model(model, **quant_config)

    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_weights(out_path)

    print(f"Saved {out_path} ({format_size(out_path)})")

    if args.verify:
        verify_quantization(
            weights_path,
            out_path,
            model_type,
            args.patch_size,
            model_kwargs,
            quant_config,
            args.image_path,
        )


if __name__ == "__main__":
    main()
