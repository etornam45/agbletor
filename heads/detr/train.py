import argparse
import os

import torch
import torch.optim as optim
from tqdm import tqdm

from dinov3.checkpoints.load import load_checkpoint
from dinov3.models import vit_small
from dinov3.utils.device import get_device
from heads.detr.dataset import NUM_DISEASE_CLASSES, NUM_SEVERITY_CLASSES, make_ghana_dataloader
from heads.detr.matcher import HungarianLoss
from heads.detr.transformer import DETR, build_detr

DEFAULT_CHECKPOINT = "dinov3/checkpoints/model/detr_ghana_decoder.pt"
DEFAULT_BACKBONE = (
    "dinov3/checkpoints/model/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DINOv3 + DETR decoder on GhanaAgricVQA detections"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--n-points", type=int, default=5)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--val-split", type=str, default="test")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument(
        "--question-type",
        type=str,
        default="identification",
        help="QA type used when deduplicating images",
    )
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--backbone", type=str, default=DEFAULT_BACKBONE)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not load an existing checkpoint before training",
    )
    parser.add_argument(
        "--no-val",
        action="store_true",
        help="Skip validation after each epoch",
    )
    return parser.parse_args()


def train_step(model: DETR, loss_fn: HungarianLoss, img_embed, target):
    out = model(img_embed)
    loss, stats = loss_fn(out, target)
    return loss, stats


def eval_epoch(
    model: DETR,
    backbone,
    loss_fn: HungarianLoss,
    data_loader,
    num_batches,
    device,
):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in data_loader:
            image = batch["image"].to(device)
            target = {
                "boxes": batch["boxes"].to(device),
                "labels": batch["labels"].to(device),
                "severities": batch["severities"].to(device),
            }
            features = backbone(image, masks=None, is_training=True)
            patches = features["x_norm_patchtokens"]
            loss, _ = train_step(model, loss_fn, patches, target)
            total_loss += loss.item()
    model.train()
    return total_loss / max(num_batches, 1)


def main(args):
    device = get_device()
    print(f"Using device: {device}")

    n_classes = NUM_DISEASE_CLASSES + 1
    n_severity_classes = NUM_SEVERITY_CLASSES + 1

    loader, num_batches = make_ghana_dataloader(
        split=args.train_split,
        img_size=args.img_size,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        language=args.language,
        question_type=args.question_type,
    )
    val_loader, num_val_batches = None, 0
    if not args.no_val:
        val_loader, num_val_batches = make_ghana_dataloader(
            split=args.val_split,
            img_size=args.img_size,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            language=args.language,
            question_type=args.question_type,
        )

    print("Train batches per epoch:", num_batches)
    if val_loader is not None:
        print("Val batches per epoch:", num_val_batches)

    backbone = vit_small(
        patch_size=16,
        n_storage_tokens=4,
        layerscale_init=1e-5,
        mask_k_bias=True,
    )
    load_checkpoint(backbone, args.backbone)
    backbone.to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    total = sum(p.numel() for p in backbone.parameters())
    print(f"Total backbone parameters: {total / 1e6:.1f}M")

    detr_decoder = build_detr(
        d_model=384,
        num_layers=args.num_layers,
        n_classes=n_classes,
        n_severity_classes=n_severity_classes,
        n_points=args.n_points,
    ).to(device)

    if not args.no_resume and os.path.exists(args.checkpoint):
        state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
        detr_decoder.load_state_dict(state_dict)
        print(f"Resumed from {args.checkpoint}")

    total = sum(p.numel() for p in detr_decoder.parameters())
    print(f"Total Decoder parameters: {total / 1e6:.1f}M")

    optimizer = optim.AdamW(
        detr_decoder.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    loss_fn = HungarianLoss(
        num_classes=NUM_DISEASE_CLASSES,
        num_severity_classes=NUM_SEVERITY_CLASSES,
    )

    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)

    for epoch in range(args.epochs):
        total_loss = 0.0
        prog_bar = tqdm(loader, desc=f"Epoch {epoch + 1}", unit="batch", total=num_batches)
        for batch in prog_bar:
            image = batch["image"].to(device)
            target = {
                "boxes": batch["boxes"].to(device),
                "labels": batch["labels"].to(device),
                "severities": batch["severities"].to(device),
            }

            with torch.no_grad():
                features = backbone(image, masks=None, is_training=True)
                patches = features["x_norm_patchtokens"]

            optimizer.zero_grad()
            loss, _ = train_step(detr_decoder, loss_fn, patches, target)
            loss.backward()
            optimizer.step()

            prog_bar.set_postfix(loss=f"{loss.item():.4f}")
            total_loss += loss.item()
        prog_bar.close()

        msg = f"Epoch {epoch}, Train Loss: {total_loss / num_batches:.4f}"
        if val_loader is not None:
            val_loss = eval_epoch(
                detr_decoder, backbone, loss_fn, val_loader, num_val_batches, device
            )
            msg += f", Val Loss: {val_loss:.4f}"
        print(msg)

        torch.save(detr_decoder.state_dict(), args.checkpoint)
        print(f"Saved checkpoint to {args.checkpoint}")


if __name__ == "__main__":
    main(parse_args())
