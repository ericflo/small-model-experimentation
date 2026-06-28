from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .humaneval_env import LETTERS


DEFAULT_MODEL_PATH = Path(
    "/workspace/.cache/huggingface/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
)


def load_tokenizer(model_path: Path = DEFAULT_MODEL_PATH) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def action_token_ids(tokenizer: Any) -> list[int]:
    ids: list[int] = []
    for letter in LETTERS:
        encoded = tokenizer.encode(letter, add_special_tokens=False)
        if len(encoded) != 1:
            raise RuntimeError(f"action {letter!r} is not a single token: {encoded}")
        ids.append(encoded[0])
    return ids


def load_base_model(model_path: Path = DEFAULT_MODEL_PATH, for_training: bool = False) -> Any:
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        quantization_config=quant_config,
        device_map="auto",
    )
    model.config.use_cache = False
    if for_training:
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)
    return model


def attach_new_lora(model: Any, r: int = 16, alpha: int = 32, dropout: float = 0.05) -> Any:
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    return get_peft_model(model, config)


def attach_existing_lora(model: Any, adapter_dir: Path, is_trainable: bool = False) -> Any:
    return PeftModel.from_pretrained(model, adapter_dir, is_trainable=is_trainable)


def last_token_action_logits(model: Any, tokenizer: Any, prompts: list[str], max_length: int = 2048) -> torch.Tensor:
    batch = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length, add_special_tokens=False)
    batch = {key: value.to(model.device) for key, value in batch.items()}
    out = model(**batch)
    lengths = batch["attention_mask"].sum(dim=1) - 1
    batch_idx = torch.arange(out.logits.shape[0], device=out.logits.device)
    logits = out.logits[batch_idx, lengths]
    ids = torch.tensor(action_token_ids(tokenizer), dtype=torch.long, device=logits.device)
    return logits.index_select(dim=-1, index=ids)
