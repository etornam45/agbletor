import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from dinov3.checkpoints.load import load_checkpoint
from dinov3.models import vit_small
from dinov3.utils.device import get_device
from heads.detr.dataset import letterbox
from heads.detr.transformer import build_detr

COCO_CLASSES = [
    "N/A",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def run_inference(image_path="test/test5.jpeg", threshold=0.62):
    device = get_device()

    dinov3_small = vit_small(
        patch_size=16,
        n_storage_tokens=4,
        layerscale_init=1e-5,
        mask_k_bias=True,
    )
    load_checkpoint(
        dinov3_small,
        "dinov3/checkpoints/model/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    )
    dinov3_small.to(device)
    dinov3_small.eval()

    detr_decoder = build_detr(
        d_model=384,
        num_layers=3,
        n_classes=92,
        n_points=4,
    ).to(device)
    detr_decoder.load_state_dict(
        torch.load(
            "dinov3/checkpoints/model/detr_decoder.pt",
            map_location=device,
            weights_only=True,
        )
    )
    detr_decoder.eval()

    img_pil = Image.open(image_path).convert("RGB")
    img_pil, _, _, _ = letterbox(img_pil, 224)
    img_arr = np.array(img_pil, dtype=np.float32) / 255.0
    image = torch.from_numpy(img_arr).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        features = dinov3_small(image, masks=None, is_training=True)
        patch_tokens = features["x_norm_patchtokens"]
        output = detr_decoder(patch_tokens)

    logits = output["logits"][0]
    boxes = output["boxes"][0]

    probs = F.softmax(logits, dim=-1)
    scores, labels = probs[:, :-1].max(dim=-1)

    keep = scores > threshold
    scores = scores[keep].cpu().numpy()
    labels = labels[keep].cpu().numpy()
    boxes = boxes[keep].cpu().numpy()

    img_size = 224
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(img_arr)

    for score, label, (cx, cy, w, h) in zip(scores, labels, boxes):
        x = (cx - w / 2) * img_size
        y = (cy - h / 2) * img_size
        pw = w * img_size
        ph = h * img_size

        rect = patches.Rectangle(
            (x, y), pw, ph, linewidth=2, edgecolor="red", facecolor="none", alpha=0.9
        )
        ax.add_patch(rect)

        class_name = (
            COCO_CLASSES[label] if label < len(COCO_CLASSES) else f"Class {label}"
        )
        ax.text(
            x,
            y,
            f"{class_name}: {score:.2f}",
            bbox=dict(facecolor="red", alpha=0.5),
            fontsize=8,
            color="white",
        )

    plt.axis("off")
    plt.title(f"DETR Predictions (threshold={threshold})")
    plt.savefig("detr_output.png")
    print("Results saved to detr_output.png")


if __name__ == "__main__":
    run_inference()
