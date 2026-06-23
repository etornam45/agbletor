from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch

from dinov3.checkpoints.load import load_checkpoint
from dinov3.models import vit_small
from dinov3.utils.device import get_device

device = get_device()

model = vit_small(
    patch_size=16,
    n_storage_tokens=4,
    layerscale_init=1e-5,
    mask_k_bias=True,
)
load_checkpoint(
    model,
    "dinov3/checkpoints/model/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
)
model.to(device)
model.eval()

image = Image.open("image.jpg")
image = image.resize((224, 224))
img_arr = np.array(image, dtype=np.float32) / 255.0
image_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(image_tensor, is_training=True)

x_norm_clstoken = output["x_norm_clstoken"]
x_storage_tokens = output["x_storage_tokens"]
x_norm_patchtokens = output["x_norm_patchtokens"]
x_prenorm = output["x_prenorm"]

image_np = img_arr
patch_tokens = x_norm_patchtokens[0]
heatmap = patch_tokens.mean(dim=-1).reshape(14, 14).cpu().numpy()

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(image_np)
axes[0].set_title("Original Image")
axes[0].axis("off")

im = axes[1].imshow(heatmap, cmap="viridis")
axes[1].set_title("Patch Token Heatmap (Mean)")
axes[1].axis("off")
plt.colorbar(im, ax=axes[1])

plt.tight_layout()
plt.show()
