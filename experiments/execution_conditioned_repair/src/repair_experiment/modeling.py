from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .prompts import messages_for_record, target_for_mode


QWEN_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_tokenizer(model_id: str, revision: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model(
    model_id: str,
    revision: str | None = None,
    *,
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
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
    )
    if for_training and load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    return model


def attach_lora(
    model,
    *,
    rank: int = 32,
    alpha: int = 64,
    dropout: float = 0.05,
):
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=QWEN_TARGET_MODULES,
    )
    return get_peft_model(model, config)


def load_model_for_generation(
    model_id: str,
    revision: str | None = None,
    adapter: str | None = None,
    *,
    load_in_4bit: bool = True,
):
    model = load_base_model(model_id, revision, load_in_4bit=load_in_4bit)
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
        mode: str,
        max_length: int,
        shuffle_traces: bool = False,
        seed: int = 13,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = max_length
        self.shuffle_traces = shuffle_traces
        rng = random.Random(seed)
        traces = [row.get("test_output_after_wrong_patch", "") for row in records]
        self.trace_overrides = traces[:]
        rng.shuffle(self.trace_overrides)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        record = self.records[index]
        trace_override = self.trace_overrides[index] if self.shuffle_traces else None
        messages = messages_for_record(record, self.mode, trace_override=trace_override)
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        target = target_for_mode(record, self.mode)
        full = prompt + target + self.tokenizer.eos_token
        prompt_ids = self.tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]
        full_ids = self.tokenizer(
            full,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]
        labels = full_ids.copy()
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len
        return {"input_ids": full_ids, "labels": labels, "attention_mask": [1] * len(full_ids)}


class CausalCollator:
    def __init__(self, tokenizer, pad_to_multiple_of: int = 8) -> None:
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(item["input_ids"]) for item in features)
        if self.pad_to_multiple_of:
            rem = max_len % self.pad_to_multiple_of
            if rem:
                max_len += self.pad_to_multiple_of - rem
        input_ids = []
        labels = []
        attention_mask = []
        pad_id = self.tokenizer.pad_token_id
        for item in features:
            pad = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad_id] * pad)
            labels.append(item["labels"] + [-100] * pad)
            attention_mask.append(item["attention_mask"] + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def render_generation_prompt(tokenizer, record: dict[str, Any], mode: str, trace_override: str | None = None) -> str:
    return tokenizer.apply_chat_template(
        messages_for_record(record, mode, trace_override=trace_override),
        tokenize=False,
        add_generation_prompt=True,
    )
