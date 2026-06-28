from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig


TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def prompt_text(tokenizer, prompt: str) -> str:
    messages = [
        {"role": "system", "content": "You compile deterministic transformation tasks into executable JSON ABI programs."},
        {"role": "user", "content": prompt},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_model(model_path: str, *, for_training: bool = False, load_in_4bit: bool = True):
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
    )
    if for_training and load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    return model


def attach_lora(model, rank: int = 8, alpha: int = 16, dropout: float = 0.05):
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )
    return get_peft_model(model, config)


def load_generation_model(model_path: str, adapter: str | None = None):
    model = load_model(model_path, for_training=False, load_in_4bit=True)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model


class CompilerDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], tokenizer, max_length: int = 1536) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        row = self.records[index]
        prompt = prompt_text(self.tokenizer, row["prompt"])
        target = row["target_text"] + self.tokenizer.eos_token
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=self.max_length)["input_ids"]
        full_ids = self.tokenizer(prompt + target, add_special_tokens=False, truncation=True, max_length=self.max_length)["input_ids"]
        labels = full_ids.copy()
        prefix = min(len(prompt_ids), len(labels))
        labels[:prefix] = [-100] * prefix
        return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


class CausalCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        input_ids = []
        attention_mask = []
        labels = []
        for feature in features:
            pad = max_len - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.tokenizer.pad_token_id] * pad)
            attention_mask.append(feature["attention_mask"] + [0] * pad)
            labels.append(feature["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def candidate_loss_batch(model, tokenizer, prompt: str, candidates: list[str], max_length: int = 1536) -> list[float]:
    rendered_prompt = prompt_text(tokenizer, prompt)
    prompt_ids = tokenizer(rendered_prompt, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
    encoded = []
    labels = []
    for cand in candidates:
        text = rendered_prompt + cand + tokenizer.eos_token
        ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
        lab = ids.copy()
        prefix = min(len(prompt_ids), len(lab))
        lab[:prefix] = [-100] * prefix
        encoded.append(ids)
        labels.append(lab)
    max_len = max(len(ids) for ids in encoded)
    input_ids = []
    attention_mask = []
    padded_labels = []
    for ids, lab in zip(encoded, labels):
        pad = max_len - len(ids)
        input_ids.append(ids + [tokenizer.pad_token_id] * pad)
        attention_mask.append([1] * len(ids) + [0] * pad)
        padded_labels.append(lab + [-100] * pad)
    device = next(model.parameters()).device
    batch = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device),
        "labels": torch.tensor(padded_labels, dtype=torch.long, device=device),
    }
    with torch.inference_mode():
        out = model(**batch)
        logits = out.logits[:, :-1, :].float()
        shift_labels = batch["labels"][:, 1:]
        token_losses = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape(shift_labels.shape)
        mask = shift_labels.ne(-100)
        losses = (token_losses * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    return [float(x) for x in losses.detach().cpu()]

