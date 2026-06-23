import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset


def letterbox(image, target_size):
    orig_w, orig_h = image.size
    scale = target_size / max(orig_w, orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    image = image.resize((new_w, new_h), Image.BILINEAR)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    canvas = Image.new("RGB", (target_size, target_size), (114, 114, 114))
    canvas.paste(image, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y


def transform_boxes(boxes, scale, pad_x, pad_y, target_size):
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    boxes = np.array(boxes, dtype=np.float32)
    boxes[:, 0] = boxes[:, 0] * scale + pad_x
    boxes[:, 1] = boxes[:, 1] * scale + pad_y
    boxes[:, 2] = boxes[:, 2] * scale
    boxes[:, 3] = boxes[:, 3] * scale
    boxes[:, 0] = (boxes[:, 0] + boxes[:, 2] / 2) / target_size
    boxes[:, 1] = (boxes[:, 1] + boxes[:, 3] / 2) / target_size
    boxes[:, 2] /= target_size
    boxes[:, 3] /= target_size
    return np.clip(boxes, 0.0, 1.0)


def load_coco(img_dir, ann_file, img_size=640):
    coco = COCO(ann_file)
    samples = []

    for img_id in coco.getImgIds():
        img_info = coco.loadImgs(img_id)[0]
        path = f"{img_dir}/{img_info['file_name']}"

        ann_ids = coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        orig_w, orig_h = img_info["width"], img_info["height"]

        scale = img_size / max(orig_w, orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        pad_x = (img_size - new_w) // 2
        pad_y = (img_size - new_h) // 2

        boxes = transform_boxes(
            [a["bbox"] for a in anns], scale, pad_x, pad_y, img_size
        )
        labels = np.array([a["category_id"] for a in anns], dtype=np.int32)

        samples.append(
            {
                "image_path": path,
                "boxes": boxes,
                "labels": labels,
                "num_objects": len(labels),
            }
        )

    return samples


class CocoDetectionDataset(Dataset):
    def __init__(self, samples, img_size=640):
        self.samples = samples
        self.img_size = img_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        image, _, _, _ = letterbox(image, self.img_size)
        image = np.array(image, dtype=np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # NCHW

        max_objs = 100
        boxes = np.zeros((max_objs, 4), dtype=np.float32)
        labels = np.zeros((max_objs,), dtype=np.int32)
        n = min(sample["num_objects"], max_objs)
        if n > 0:
            boxes[:n] = sample["boxes"][:n]
            labels[:n] = sample["labels"][:n]

        return {
            "image": image,
            "boxes": torch.from_numpy(boxes),
            "labels": torch.from_numpy(labels),
            "num_objects": n,
        }


def collate_fn(batch):
    images = torch.stack([b["image"] for b in batch])
    boxes = torch.stack([b["boxes"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    num_objects = torch.tensor([b["num_objects"] for b in batch])
    return {
        "image": images,
        "boxes": boxes,
        "labels": labels,
        "num_objects": num_objects,
    }


def make_dataloader(
    img_dir,
    ann_file,
    img_size=640,
    batch_size=16,
    shuffle=False,
    num_workers=4,
    pin_memory=None,
):
    samples = load_coco(img_dir, ann_file, img_size)
    dataset = CocoDetectionDataset(samples, img_size=img_size)

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=False,
    )
    num_batches = (len(dataset) + batch_size - 1) // batch_size
    return loader, num_batches


# Backwards-compatible alias
make_stream = make_dataloader


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    img_size = 224

    t0 = time.time()
    loader, _ = make_dataloader(
        "coco/images/val2017",
        "coco/annotations/instances_val2017.json",
        img_size=img_size,
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )
    print(f"DataLoader ready in {time.time() - t0:.2f}s")

    batch = next(iter(loader))
    images = batch["image"]
    boxes = batch["boxes"].numpy()
    n_objs = batch["num_objects"].numpy()

    img = images[0].permute(1, 2, 0).numpy()

    print(img.shape)
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    for cx, cy, w, h in boxes[0, : n_objs[0]]:
        x = (cx - w / 2) * img_size
        y = (cy - h / 2) * img_size
        ax.add_patch(
            patches.Rectangle(
                (x, y),
                w * img_size,
                h * img_size,
                linewidth=1,
                edgecolor="pink",
                facecolor="none",
            )
        )
    plt.show()
