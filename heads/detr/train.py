import os
import torch
from tqdm import tqdm
import torch.optim as optim

from dinov3.checkpoints.load import load_checkpoint
from dinov3.models import vit_small
from dinov3.utils.device import get_device
from heads.detr.dataset import make_dataloader
from heads.detr.matcher import HungarianLoss
from heads.detr.transformer import DETR, build_detr

device = get_device()
print(f"Using device: {device}")

loader, num_batches = make_dataloader(
    "coco/images/val2017",
    "coco/annotations/instances_val2017.json",
    img_size=224,
    batch_size=16,
    shuffle=True,
)
print("Total batches per epoch:", num_batches)

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
    n_classes=92,
    n_points=5,
).to(device)

out_path = "dinov3/checkpoints/model/detr_decoder.pt"
if os.path.exists(out_path):
    state_dict = torch.load(out_path)
    detr_decoder.load_state_dict(state_dict)

total = sum(p.numel() for p in detr_decoder.parameters())
print(f"Total Decoder parameters: {total / 1e6:.1f}M")

optimizer = optim.AdamW(detr_decoder.parameters(), lr=1e-5, weight_decay=0.01)
lf = HungarianLoss(num_classes=91)


def train_step(model: DETR, img_embed, target):
    out = model(img_embed)
    loss, stats = lf(out, target)
    return loss, stats


for i in range(50):
    total_loss = 0.0
    prog_bar = tqdm(loader, desc="Training", unit="batch", total=num_batches)
    for batch in prog_bar:
        image = batch["image"].to(device)
        target = {
            "boxes": batch["boxes"].to(device),
            "labels": batch["labels"].to(device),
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
    print(f"Epoch {i}, Loss: {total_loss / num_batches}")
    torch.save(detr_decoder.state_dict(), out_path)
