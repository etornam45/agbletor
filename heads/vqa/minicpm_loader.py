"""Load MiniCPM5-1B tokenizer and LoRA-wrapped Llama for the VQA head."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

MINICPM_MODEL_NAME = "openbmb/MiniCPM5-1B"
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def _hub_cache_dir() -> Path:
    return Path(
        os.environ.get(
            "HF_HOME",
            os.path.join(os.path.expanduser("~"), ".cache", "huggingface"),
        )
    )


def _minicpm_snapshot_dir(require_tokenizer: bool = False) -> Path | None:
    safe_name = MINICPM_MODEL_NAME.replace("/", "--")
    snapshots = _hub_cache_dir() / "hub" / f"models--{safe_name}" / "snapshots"
    if not snapshots.is_dir():
        return None
    candidates = []
    for snap in snapshots.iterdir():
        if not snap.is_dir() or not (snap / "config.json").exists():
            continue
        if require_tokenizer and not (
            (snap / "tokenizer.json").exists() or (snap / "tokenizer.model").exists()
        ):
            continue
        if not require_tokenizer and not (
            (snap / "model.safetensors.index.json").exists()
            or (snap / "pytorch_model.bin").exists()
        ):
            continue
        candidates.append(snap)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _pretrained_source(require_tokenizer: bool = False) -> str:
    snap = _minicpm_snapshot_dir(require_tokenizer=require_tokenizer)
    return str(snap) if snap is not None else MINICPM_MODEL_NAME


def _local_files_only(require_tokenizer: bool = False) -> bool:
    return _minicpm_snapshot_dir(require_tokenizer=require_tokenizer) is not None


def _model_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16
    if device.type == "mps":
        return torch.float32
    return torch.float16


def load_minicpm_tokenizer(
    model_name: str = MINICPM_MODEL_NAME,
) -> PreTrainedTokenizer:
    source = (
        _pretrained_source(require_tokenizer=True)
        if model_name == MINICPM_MODEL_NAME
        else model_name
    )
    local_only = _local_files_only(require_tokenizer=True) and model_name == MINICPM_MODEL_NAME
    if local_only:
        print(f"Loading MiniCPM tokenizer from local cache: {source}")
    elif model_name == MINICPM_MODEL_NAME:
        print(f"Downloading MiniCPM tokenizer ({model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(
        source,
        local_files_only=local_only,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_minicpm_llm(
    device: torch.device,
    model_name: str = MINICPM_MODEL_NAME,
    lora_r: int = 16,
    lora_alpha: int = 32,
    adapter_path: Optional[str] = None,
) -> torch.nn.Module:
    source = (
        _pretrained_source(require_tokenizer=False)
        if model_name == MINICPM_MODEL_NAME
        else model_name
    )
    local_only = (
        _local_files_only(require_tokenizer=False) and model_name == MINICPM_MODEL_NAME
    )
    dtype = _model_dtype(device)

    if local_only and adapter_path is None:
        print(f"Loading MiniCPM5-1B from local cache: {source}")
    elif adapter_path is None:
        print(f"Downloading MiniCPM5-1B ({model_name})...")

    load_kwargs = {
        "dtype": dtype,
        "local_files_only": local_only and adapter_path is None,
        "trust_remote_code": False,
    }
    if device.type == "mps":
        load_kwargs["attn_implementation"] = "eager"

    llm = AutoModelForCausalLM.from_pretrained(source, **load_kwargs)

    if adapter_path is not None:
        print(f"Loading LoRA adapter from {adapter_path}")
        llm = PeftModel.from_pretrained(llm, adapter_path, is_trainable=False)
    else:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=LORA_TARGET_MODULES,
            bias="none",
            task_type="CAUSAL_LM",
        )
        llm = get_peft_model(llm, lora_config)

    llm.to(device)
    return llm


def _to_token_ids(result) -> list[int]:
    """Normalize apply_chat_template output to plain token id lists."""
    if isinstance(result, list):
        if not result or isinstance(result[0], int):
            return result
        if isinstance(result[0], str):
            raise TypeError(
                "Expected token ids but got strings; check apply_chat_template output."
            )
    if hasattr(result, "input_ids"):
        ids = result["input_ids"]
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
        return list(ids)
    if hasattr(result, "ids"):
        return list(result.ids)
    if isinstance(result, str):
        raise TypeError("Expected token ids but got a string from apply_chat_template.")
    raise TypeError(f"Unsupported tokenization result type: {type(result)!r}")


def apply_chat(
    tokenizer: PreTrainedTokenizer,
    messages: List[dict],
    add_generation_prompt: bool = False,
    tokenize: bool = True,
    return_tensors: Optional[str] = "pt",
):
    result = tokenizer.apply_chat_template(
        messages,
        tokenize=tokenize,
        add_generation_prompt=add_generation_prompt,
        return_tensors=return_tensors,
    )
    if tokenize and return_tensors is None:
        return _to_token_ids(result)
    return result


def tokenize_chat_pair(
    tokenizer: PreTrainedTokenizer,
    question: str,
    answer: str,
    max_length: int,
) -> tuple[list[int], list[int]]:
    prompt_messages = [{"role": "user", "content": question}]
    full_messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]

    prompt_ids = apply_chat(
        tokenizer,
        prompt_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors=None,
    )
    full_ids = apply_chat(
        tokenizer,
        full_messages,
        add_generation_prompt=False,
        tokenize=True,
        return_tensors=None,
    )

    if len(full_ids) > max_length:
        full_ids = full_ids[:max_length]

    prompt_len = min(len(prompt_ids), len(full_ids))
    if prompt_len >= len(full_ids) and len(full_ids) > 1:
        prompt_len = len(full_ids) - 1

    labels = [-100] * prompt_len + full_ids[prompt_len:]
    labels = labels[: len(full_ids)]

    return full_ids, labels
