import os
import torch
from tqdm import tqdm
import torch.optim as optim

from dinov3.checkpoints.load import load_checkpoint
from dinov3.models import vit_small
from dinov3.utils.device import get_device
from heads.detr.dataset import NUM_DISEASE_CLASSES, NUM_SEVERITY_CLASSES, make_ghana_dataloader
from heads.detr.matcher import HungarianLoss
from heads.detr.transformer import DETR, build_detr

device = get_device()
print(f"Using device: {device}")

IMG_SIZE = 224
N_CLASSES = NUM_DISEASE_CLASSES + 1
N_SEVERITY_CLASSES = NUM_SEVERITY_CLASSES + 1
CHECKPOINT_PATH = "dinov3/checkpoints/model/detr_ghana_decoder.pt"

loader, num_batches = make_ghana_dataloader(
    split="train",
    img_size=IMG_SIZE,
    batch_size=16,
    shuffle=True,
)
val_loader, num_val_batches = make_ghana_dataloader(
    split="test",
    img_size=IMG_SIZE,
    batch_size=16,
    shuffle=False,
)
print("Train batches per epoch:", num_batches)
print("Val batches per epoch:", num_val_batches)

BACKBONE_WEIGHTS = (
    "dinov3/checkpoints/model/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
)

dinov3_small = vit_small(
    patch_size=16,
    n_storage_tokens=4,
    layerscale_init=1e-5,
    mask_k_bias=True,
)
load_checkpoint(dinov3_small, BACKBONE_WEIGHTS)
dinov3_small.to(device)
dinov3_small.eval()
for p in dinov3_small.parameters():
    p.requires_grad = False

total = sum(p.numel() for p in dinov3_small.parameters())
print(f"Total backbone parameters: {total / 1e6:.1f}M")

detr_decoder = build_detr(
    d_model=384,
    num_layers=4,
    n_classes=N_CLASSES,
    n_severity_classes=N_SEVERITY_CLASSES,
    n_points=5,
).to(device)

if os.path.exists(CHECKPOINT_PATH):
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    detr_decoder.load_state_dict(state_dict)

total = sum(p.numel() for p in detr_decoder.parameters())
print(f"Total Decoder parameters: {total / 1e6:.1f}M")

optimizer = optim.AdamW(detr_decoder.parameters(), lr=1e-5, weight_decay=0.01)
lf = HungarianLoss(num_classes=NUM_DISEASE_CLASSES, num_severity_classes=NUM_SEVERITY_CLASSES)


def train_step(model: DETR, img_embed, target):
    out = model(img_embed)
    loss, stats = lf(out, target)
    return loss, stats


def eval_epoch(model: DETR, data_loader, num_batches):
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
            features = dinov3_small(image, masks=None, is_training=True)
            patches = features["x_norm_patchtokens"]
            loss, _ = train_step(model, patches, target)
            total_loss += loss.item()
    model.train()
    return total_loss / max(num_batches, 1)


for i in range(50):
    total_loss = 0.0
    prog_bar = tqdm(loader, desc="Training", unit="batch", total=num_batches)
    for batch in prog_bar:
        image = batch["image"].to(device)
        target = {
            "boxes": batch["boxes"].to(device),
            "labels": batch["labels"].to(device),
            "severities": batch["severities"].to(device),
        }

        with torch.no_grad():
            features = dinov3_small(image, masks=None, is_training=True)
            patches = features["x_norm_patchtokens"]

        optimizer.zero_grad()
        loss, _ = train_step(detr_decoder, patches, target)
        loss.backward()
        optimizer.step()

        prog_bar.set_postfix(loss=f"{loss.item():.4f}")
        total_loss += loss.item()
    prog_bar.close()

    val_loss = eval_epoch(detr_decoder, val_loader, num_val_batches)
    print(
        f"Epoch {i}, Train Loss: {total_loss / num_batches:.4f}, "
        f"Val Loss: {val_loss:.4f}"
    )
    torch.save(detr_decoder.state_dict(), CHECKPOINT_PATH)
