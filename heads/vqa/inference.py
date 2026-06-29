import argparse
from typing import Optional

import numpy as np
import torch
from PIL import Image

from dinov3.utils.device import get_device
from heads.detr.dataset import letterbox
from heads.vqa.dataset import encode_user_prompt, normalize_answer
from heads.vqa.model import (
    DEFAULT_CHECKPOINT_DIR,
    build_hybrid_model,
    decode_generated_answer,
    load_hybrid_checkpoint,
    resolve_checkpoint_dir,
)

IMG_SIZE = 224


def preprocess_image(image: Image.Image, img_size: int = IMG_SIZE) -> torch.Tensor:
    image, _, _, _ = letterbox(image.convert("RGB"), img_size)
    arr = np.array(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def run_inference(
    image_path: Optional[str] = None,
    question: Optional[str] = None,
    max_new_tokens: int = 128,
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
) -> str:
    device = get_device()
    checkpoint_path = resolve_checkpoint_dir(checkpoint_dir)
    print(f"Loading checkpoint: {checkpoint_path}")

    model, tokenizer = build_hybrid_model(device)
    load_hybrid_checkpoint(model, str(checkpoint_path), device, trainable_adapter=False)
    model.eval()

    if image_path is None:
        from heads.vqa.dataset import load_ghana_agric_split

        sample = load_ghana_agric_split("test")[0]
        image = sample["image"]
        if question is None:
            question = sample["question"]
        reference = normalize_answer(sample["answer"])
        print(f"Sample question: {question}")
        print(f"Reference answer: {reference[:200]}...")
    else:
        image = Image.open(image_path)

    if question is None:
        raise ValueError("question is required when image_path is provided")

    image_tensor = preprocess_image(image).to(device)
    prompt = encode_user_prompt(tokenizer, [question], device)
    prompt_len = prompt["input_ids"].shape[1]

    gen_ids = model.generate(
        image_tensor,
        prompt["input_ids"],
        attention_mask=prompt["attention_mask"],
        max_new_tokens=max_new_tokens,
        num_beams=1,
    )

    answer = decode_generated_answer(tokenizer, gen_ids, prompt_len)
    print(f"Generated answer: {answer}")
    return answer


def parse_args():
    parser = argparse.ArgumentParser(description="Run DINOv3 + MiniCPM VQA inference")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Checkpoint directory (uses *_best if present)",
    )
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    parser.add_argument("--question", type=str, default=None, help="Question about the image")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        resolve_checkpoint_dir(args.checkpoint)
    except FileNotFoundError as exc:
        print(exc)
    else:
        run_inference(
            image_path=args.image,
            question=args.question,
            max_new_tokens=args.max_new_tokens,
            checkpoint_dir=args.checkpoint,
        )
