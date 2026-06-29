import ast
import json
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer

from heads.detr.dataset import letterbox
from heads.vqa.minicpm_loader import apply_chat, tokenize_chat_pair

DATASET_NAME = "toufiqmusah/GhanaAgricVQA-Dataset"


def normalize_answer(answer: Union[str, dict]) -> str:
    if isinstance(answer, dict):
        return answer.get("text", str(answer))
    if isinstance(answer, str):
        stripped = answer.strip()
        if stripped.startswith("{"):
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(stripped)
                    if isinstance(parsed, dict) and "text" in parsed:
                        return parsed["text"]
                except (json.JSONDecodeError, SyntaxError, ValueError):
                    continue
    return answer


def load_ghana_agric_split(split: str = "train", language: str = "en"):
    dataset = load_dataset(DATASET_NAME, split=split)
    if language:
        dataset = dataset.filter(lambda x: x.get("language", "en") == language)
    return dataset


class GhanaAgricVQADataset(Dataset):
    def __init__(
        self,
        hf_dataset,
        img_size: int = 224,
    ):
        self.data = hf_dataset
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        row = self.data[idx]
        image = row["image"]
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")

        image, _, _, _ = letterbox(image, self.img_size)
        image = np.array(image, dtype=np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)

        return {
            "image": image,
            "question": row["question"],
            "answer": normalize_answer(row["answer"]),
            "question_type": row.get("question_type", ""),
            "crop": row.get("crop", ""),
        }


def _pad_sequences(
    sequences: List[List[int]],
    pad_value: int,
) -> torch.Tensor:
    max_len = max(len(seq) for seq in sequences)
    batch = torch.full((len(sequences), max_len), pad_value, dtype=torch.long)
    for i, seq in enumerate(sequences):
        batch[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return batch


def _build_minicpm_batch(
    batch,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
) -> Dict[str, torch.Tensor]:
    input_ids_list = []
    labels_list = []

    for item in batch:
        ids, labels = tokenize_chat_pair(
            tokenizer,
            item["question"],
            item["answer"],
            max_length=max_length,
        )
        input_ids_list.append(ids)
        labels_list.append(labels)

    input_ids = _pad_sequences(input_ids_list, tokenizer.pad_token_id)
    labels = _pad_sequences(labels_list, -100)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    return {
        "image": torch.stack([item["image"] for item in batch]),
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "question": [item["question"] for item in batch],
        "answer": [item["answer"] for item in batch],
    }


def make_dataloader(
    tokenizer: PreTrainedTokenizer,
    split: str = "train",
    img_size: int = 224,
    max_length: int = 256,
    batch_size: int = 4,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    language: str = "en",
) -> Tuple[DataLoader, int]:
    hf_dataset = load_ghana_agric_split(split, language=language)
    dataset = GhanaAgricVQADataset(hf_dataset, img_size=img_size)

    def collate_fn(batch):
        return _build_minicpm_batch(batch, tokenizer, max_length)

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=split == "train",
    )
    return loader, len(loader)


def encode_user_prompt(
    tokenizer: PreTrainedTokenizer,
    questions: list[str],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    input_ids_list = []
    for question in questions:
        messages = [{"role": "user", "content": question}]
        ids = apply_chat(
            tokenizer,
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors=None,
        )
        input_ids_list.append(ids)

    input_ids = _pad_sequences(input_ids_list, tokenizer.pad_token_id).to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


if __name__ == "__main__":
    from heads.vqa.minicpm_loader import load_minicpm_tokenizer

    tokenizer = load_minicpm_tokenizer()
    loader, num_batches = make_dataloader(
        tokenizer,
        split="train",
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )
    print(f"Batches: {num_batches}")

    batch = next(iter(loader))
    print("image:", batch["image"].shape)
    print("input_ids:", batch["input_ids"].shape)
    print("labels:", batch["labels"].shape)
    print("question:", batch["question"][0])
    print("answer:", batch["answer"][0][:120], "...")
    supervised = (batch["labels"][0] != -100).sum().item()
    print("supervised tokens:", supervised)
