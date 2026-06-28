from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig

from .prompts import messages_for_record, shuffled_visible


TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_tokenizer(model_id: str, revision: str | None, cache_dir: str | None):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=cache_dir,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model(
    model_id: str,
    revision: str | None,
    *,
    cache_dir: str | None,
    load_in_4bit: bool = True,
    for_training: bool = False,
):
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=cache_dir,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
    )
    if for_training and load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    return model


def attach_lora(model, *, rank: int, alpha: int, dropout: float):
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )
    return get_peft_model(model, config)


def load_generation_model(
    model_id: str,
    revision: str | None,
    *,
    adapter: str | None,
    cache_dir: str | None,
):
    model = load_base_model(model_id, revision, cache_dir=cache_dir, load_in_4bit=True)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model


class RepairSftDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        tokenizer,
        *,
        prompt_mode: str,
        max_length: int,
        shuffle_traces: bool = False,
        seed: int = 13,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.prompt_mode = prompt_mode
        self.max_length = max_length
        self.trace_overrides = shuffled_visible(records, seed=seed) if shuffle_traces else [None] * len(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        record = self.records[index]
        prompt = self.tokenizer.apply_chat_template(
            messages_for_record(record, prompt_mode=self.prompt_mode, trace_override=self.trace_overrides[index]),
            tokenize=False,
            add_generation_prompt=True,
        )
        target = record.get("target_output", record.get("target_graph", record["target_program"])).strip()
        target = target + self.tokenizer.eos_token
        full = prompt + target
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=self.max_length)["input_ids"]
        full_ids = self.tokenizer(full, add_special_tokens=False, truncation=True, max_length=self.max_length)["input_ids"]
        labels = full_ids.copy()
        labels[: min(len(prompt_ids), len(labels))] = [-100] * min(len(prompt_ids), len(labels))
        return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


DslSftDataset = RepairSftDataset


class CausalCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(item["input_ids"]) for item in features)
        input_ids = []
        attention_mask = []
        labels = []
        for item in features:
            pad = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.tokenizer.pad_token_id] * pad)
            attention_mask.append(item["attention_mask"] + [0] * pad)
            labels.append(item["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
