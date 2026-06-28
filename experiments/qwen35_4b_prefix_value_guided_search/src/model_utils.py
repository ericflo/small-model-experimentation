from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


DEFAULT_MODEL_PATH = Path(
    "/workspace/.cache/huggingface/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
)


def load_tokenizer(model_path: Path = DEFAULT_MODEL_PATH, padding_side: str = "left") -> Any:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side
    return tokenizer


def code_chat_prompt(tokenizer: Any, task_text: str, entry_point: str, public_tests: list[str]) -> str:
    tests = "\n".join(public_tests)
    user = (
        f"Task: {task_text}\n"
        f"Define a function named `{entry_point}` and any helpers needed.\n"
        "Return only runnable Python code. Do not include markdown, explanations, tests, or reasoning.\n"
        "The code should satisfy these public tests:\n"
        f"{tests}\n"
    )
    system = "You are a Python coding assistant. Return only runnable Python code."
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{system}\n\n{user}\nPython code:\n"


def prefix_prompt(tokenizer: Any, task_text: str, entry_point: str, public_tests: list[str], max_lines: int) -> str:
    tests = "\n".join(public_tests)
    user = (
        f"Task: {task_text}\n"
        f"Define a function named `{entry_point}`.\n"
        "Write only the beginning of the Python solution, not the whole solution.\n"
        f"Return exactly the first {max_lines} non-empty lines of runnable Python code when possible.\n"
        "No markdown, no explanations, no tests.\n"
        "Public tests:\n"
        f"{tests}\n"
    )
    system = "You write Python code prefixes for later completion."
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{system}\n\n{user}\nPython prefix:\n"


def continue_prefix_prompt(tokenizer: Any, task_text: str, entry_point: str, public_tests: list[str], prefix_code: str) -> str:
    tests = "\n".join(public_tests)
    user = (
        f"Task: {task_text}\n"
        f"Complete the Python function named `{entry_point}`.\n"
        "You must preserve and continue this exact prefix. Return the complete runnable Python code, not just the suffix.\n\n"
        "Prefix:\n"
        f"{prefix_code.rstrip()}\n\n"
        "Public tests:\n"
        f"{tests}\n"
        "Return only code."
    )
    system = "You are a Python coding assistant. Return only runnable Python code."
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{system}\n\n{user}\nPython code:\n"


def load_generation_model(model_path: Path = DEFAULT_MODEL_PATH) -> Any:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model


def load_quant_model(model_path: Path = DEFAULT_MODEL_PATH, for_training: bool = False) -> Any:
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


@torch.no_grad()
def sample_prompt(
    model: Any,
    tokenizer: Any,
    prompt: str,
    count: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    batch_size: int,
    seed: int,
) -> list[str]:
    rows: list[str] = []
    device = model.device
    offset = 0
    while len(rows) < count:
        current = min(batch_size, count - len(rows))
        prompts = [prompt] * current
        batch = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        torch.manual_seed(seed + offset)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + offset)
        output = model.generate(
            **batch,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        prompt_len = batch["input_ids"].shape[1]
        rows.extend(tokenizer.batch_decode(output[:, prompt_len:], skip_special_tokens=True))
        offset += current
    return rows


def estimate_tokens(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))
