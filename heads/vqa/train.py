import argparse
from pathlib import Path

import torch
import torch.optim as optim
from tqdm import tqdm

from dinov3.utils.device import get_device
from heads.vqa.dataset import encode_user_prompt, make_dataloader
from heads.vqa.model import (
    DEFAULT_CHECKPOINT_DIR,
    adapter_trainable_params,
    build_hybrid_model,
    decode_generated_answer,
    load_hybrid_checkpoint,
    save_hybrid_checkpoint,
    vision_adapter_params,
)

DEFAULT_CHECKPOINT = DEFAULT_CHECKPOINT_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Train DINOv3 + MiniCPM5-1B VQA hybrid")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--llm-lr", type=float, default=2e-4, help="LoRA learning rate")
    parser.add_argument(
        "--adapter-lr", type=float, default=1e-4, help="Vision adapter learning rate"
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def train_epoch(model, loader, optimizer, device):
    model.train()
    model.vision_model.eval()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for batch in pbar:
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(images, input_ids, attention_mask, labels=labels)
        loss = outputs.loss
        if not torch.isfinite(loss):
            print("Warning: non-finite loss, skipping batch")
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for group in optimizer.param_groups for p in group["params"]],
            1.0,
        )
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{total_loss / num_batches:.4f}")

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(model, loader, tokenizer, device, max_gen_samples: int = 8):
    model.eval()
    total_loss = 0.0
    references = []
    hypotheses = []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(images, input_ids, attention_mask, labels=labels)
        total_loss += outputs.loss.item()

        if len(references) < max_gen_samples:
            prompt = encode_user_prompt(tokenizer, [batch["question"][0]], device)
            prompt_len = prompt["input_ids"].shape[1]
            gen_ids = model.generate(
                images[:1],
                prompt["input_ids"][:1],
                attention_mask=prompt["attention_mask"][:1],
                max_new_tokens=128,
                num_beams=1,
            )
            hypotheses.append(
                decode_generated_answer(tokenizer, gen_ids, prompt_len)
            )
            references.append(batch["answer"][0])

    avg_loss = total_loss / max(len(loader), 1)
    return avg_loss, references, hypotheses


def main():
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    model, tokenizer = build_hybrid_model(
        device,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )

    if args.resume:
        load_hybrid_checkpoint(model, args.resume, device, trainable_adapter=True)

    if hasattr(model.llm, "gradient_checkpointing_enable"):
        model.llm.gradient_checkpointing_enable()

    train_loader, _ = make_dataloader(
        tokenizer,
        split="train",
        max_length=args.max_length,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    test_loader, _ = make_dataloader(
        tokenizer,
        split="test",
        max_length=args.max_length,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    optimizer = optim.AdamW(
        [
            {
                "params": vision_adapter_params(model),
                "lr": args.adapter_lr,
            },
            {
                "params": adapter_trainable_params(model),
                "lr": args.llm_lr,
            },
        ],
        weight_decay=args.weight_decay,
    )

    output_dir = Path(args.output)
    best_dir = output_dir.parent / f"{output_dir.name}_best"
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        eval_loss, refs, hyps = evaluate(model, test_loader, tokenizer, device)

        print(
            f"Epoch {epoch + 1}/{args.epochs}: "
            f"train_loss={train_loss:.4f}, eval_loss={eval_loss:.4f}"
        )
        if refs:
            print(f"  sample ref: {refs[0][:120]}...")
            print(f"  sample gen: {hyps[0][:120]}...")

        save_hybrid_checkpoint(model, str(output_dir))
        if eval_loss < best_loss:
            best_loss = eval_loss
            save_hybrid_checkpoint(model, str(best_dir))
            print(f"  saved best checkpoint (eval_loss={eval_loss:.4f})")

    print(f"Training complete. Checkpoints saved under {output_dir.parent}")


if __name__ == "__main__":
    main()
