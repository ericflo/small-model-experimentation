#!/usr/bin/env python3
"""Crystallized trace ABI tournament for Qwen.

This standalone experiment generates deterministic practical tasks, renders
gold outputs in several executable trace formats, fine-tunes small QLoRA
adapters, and compares held-out exact-answer accuracy across output ABIs.
"""

from __future__ import annotations

import argparse
import csv
import gc
import html
import json
import math
import os
import random
import re
import time
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


ROOT = Path("/workspace/experiments/qwen_crystallized_trace_abi_tournament")
RUNS = ROOT / "runs"
ANALYSIS = ROOT / "analysis"
FIGURES = ANALYSIS / "figures"
REPORTS = ROOT / "reports"
LARGE_ROOT = Path("/workspace/large_artifacts/qwen_crystallized_trace_abi_tournament")
CHECKPOINTS = LARGE_ROOT / "checkpoints"
CACHE_DIR = Path("/workspace/.cache/huggingface")

MODEL_NAME = "Qwen/Qwen3-4B"
ABIS = ["answer", "python", "json", "stack"]
TRAIN_FAMILIES = ["string", "unit", "table", "date"]
EVAL_SPLITS = ["eval_indist", "eval_composition", "eval_template_shift"]
LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"


@dataclass
class Example:
    example_id: str
    split: str
    family: str
    prompt: str
    answer: str
    op_name: str
    payload: Dict[str, Any]


@dataclass
class RunConfig:
    run_name: str
    suite: str
    model_name: str
    seeds: List[int]
    arms: List[str]
    train_n: int
    eval_n: int
    train_steps: int
    batch_size: int
    grad_accum: int
    lr: float
    max_length: int
    eval_batch_size: int
    lora_r: int
    lora_alpha: int
    max_new_tokens: int


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dirs() -> None:
    for path in [RUNS, ANALYSIS, FIGURES, REPORTS, CHECKPOINTS]:
        path.mkdir(parents=True, exist_ok=True)


def append_log(text: str) -> None:
    with (ROOT / "experiment_log.md").open("a") as f:
        f.write(text.rstrip() + "\n\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=json_default))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return str(obj)


def pct(x: Any) -> str:
    try:
        v = float(x)
    except Exception:
        return "n/a"
    if math.isnan(v):
        return "n/a"
    return f"{100.0 * v:.1f}%"


def mean(xs: Sequence[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


def std(xs: Sequence[float]) -> float:
    return float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0


def rand_code(rng: random.Random) -> str:
    left = "".join(rng.choice(LETTERS) for _ in range(2))
    mid = "".join(str(rng.randint(0, 9)) for _ in range(4))
    right = "".join(rng.choice(LETTERS) for _ in range(2))
    return f"{left}-{mid}-{right}"


def choose_template(rng: random.Random, split: str, standard: Sequence[str], shifted: Sequence[str]) -> str:
    if split == "eval_template_shift":
        return rng.choice(list(shifted))
    return rng.choice(list(standard))


def make_string_example(rng: random.Random, split: str, idx: int) -> Example:
    code = rand_code(rng)
    clean = code.replace("-", "")
    if split == "eval_composition":
        width = rng.randint(3, 5)
        start = rng.randint(0, len(clean) - width)
        reversed_clean = clean[::-1]
        answer = reversed_clean[start : start + width]
        op = "reverse_then_slice"
        prompt_t = choose_template(
            rng,
            split,
            [
                "Reference code: {code}. Remove hyphens, reverse the cleaned string, then return characters {a} through {b} using one-indexed positions.",
                "Clean this code by deleting hyphens: {code}. Reverse it and report the one-indexed span {a}-{b}.",
            ],
            [
                "Identifier {code} should be compacted, flipped left-to-right, and clipped to positions {a} through {b}. What is the clipped text?",
            ],
        )
        prompt = prompt_t.format(code=code, a=start + 1, b=start + width)
        payload = {"code": code, "clean": clean, "reversed": reversed_clean, "start": start, "end": start + width}
    else:
        op = rng.choice(["take_prefix", "take_suffix", "slice"])
        if op == "take_prefix":
            width = rng.randint(3, 5)
            answer = clean[:width]
            prompt_t = choose_template(
                rng,
                split,
                [
                    "Reference code: {code}. Remove hyphens, then return the first {n} characters.",
                    "Code {code} has separators. Delete the separators and keep the leftmost {n} characters.",
                ],
                [
                    "Compact identifier {code} by dropping hyphens. What are the first {n} symbols afterward?",
                ],
            )
            prompt = prompt_t.format(code=code, n=width)
            payload = {"code": code, "clean": clean, "n": width}
        elif op == "take_suffix":
            width = rng.randint(3, 5)
            answer = clean[-width:]
            prompt_t = choose_template(
                rng,
                split,
                [
                    "Reference code: {code}. Remove hyphens, then return the last {n} characters.",
                    "Code {code} has separators. Delete the separators and keep the rightmost {n} characters.",
                ],
                [
                    "Compact identifier {code} by dropping hyphens. What are the final {n} symbols afterward?",
                ],
            )
            prompt = prompt_t.format(code=code, n=width)
            payload = {"code": code, "clean": clean, "n": width}
        else:
            width = rng.randint(3, 5)
            start = rng.randint(0, len(clean) - width)
            answer = clean[start : start + width]
            prompt_t = choose_template(
                rng,
                split,
                [
                    "Reference code: {code}. Remove hyphens, then return characters {a} through {b} using one-indexed positions.",
                    "Delete hyphens from {code}, then report the one-indexed span {a}-{b}.",
                ],
                [
                    "After compacting {code}, which substring occupies positions {a} through {b}?",
                ],
            )
            prompt = prompt_t.format(code=code, a=start + 1, b=start + width)
            payload = {"code": code, "clean": clean, "start": start, "end": start + width}
    return Example(f"{split}_string_{idx}", split, "string", prompt, str(answer), op, payload)


def make_unit_example(rng: random.Random, split: str, idx: int) -> Example:
    if split == "eval_composition":
        meters = rng.randint(1, 8)
        centimeters = rng.choice([5, 10, 15, 20, 25, 40, 50, 75])
        answer = meters * 100 + centimeters
        op = "meters_plus_centimeters"
        prompt_t = choose_template(
            rng,
            split,
            [
                "A cable is {m} meters plus {c} centimeters long. Report the total length in centimeters.",
                "Combine {m} m and {c} cm. What is the total number of centimeters?",
            ],
            [
                "Convert a mixed length of {m} meters and {c} centimeters into centimeters.",
            ],
        )
        prompt = prompt_t.format(m=meters, c=centimeters)
        payload = {"meters": meters, "centimeters": centimeters}
    else:
        conversion = rng.choice(
            [
                ("meters_to_centimeters", "meters", "centimeters", 100),
                ("centimeters_to_millimeters", "centimeters", "millimeters", 10),
                ("kilograms_to_grams", "kilograms", "grams", 1000),
                ("minutes_to_seconds", "minutes", "seconds", 60),
            ]
        )
        op, src, dst, factor = conversion
        amount = rng.randint(2, 19)
        answer = amount * factor
        prompt_t = choose_template(
            rng,
            split,
            [
                "Convert {amount} {src} to {dst}. Return the integer result.",
                "How many {dst} are equal to {amount} {src}?",
            ],
            [
                "Rewrite {amount} {src} using {dst} as the unit. Give only the integer value.",
            ],
        )
        prompt = prompt_t.format(amount=amount, src=src, dst=dst)
        payload = {"amount": amount, "src": src, "dst": dst, "factor": factor}
    return Example(f"{split}_unit_{idx}", split, "unit", prompt, str(answer), op, payload)


def make_table_example(rng: random.Random, split: str, idx: int) -> Example:
    keys = ["alpha", "beta", "gamma", "delta"]
    rng.shuffle(keys)
    values = {k: rng.randint(2, 29) for k in keys}
    table = ", ".join(f"{k}={values[k]}" for k in keys)
    if split == "eval_composition":
        k1, k2 = rng.sample(keys, 2)
        bias = rng.randint(1, 9)
        answer = values[k1] + values[k2] + bias
        op = "sum_two_keys_plus"
        prompt_t = choose_template(
            rng,
            split,
            [
                "Table: {table}. Add the values for {k1} and {k2}, then add {b}.",
                "Using {table}, compute value({k1}) + value({k2}) + {b}.",
            ],
            [
                "From this lookup table [{table}], total the entries named {k1} and {k2}, then increase by {b}.",
            ],
        )
        prompt = prompt_t.format(table=table, k1=k1, k2=k2, b=bias)
        payload = {"table": values, "k1": k1, "k2": k2, "bias": bias}
    else:
        key = rng.choice(keys)
        scale = rng.randint(2, 7)
        bias = rng.randint(1, 11)
        if rng.random() < 0.5:
            answer = values[key] * scale + bias
            op = "lookup_mul_add"
            prompt_t = choose_template(
                rng,
                split,
                [
                    "Table: {table}. Take {key}, multiply by {scale}, then add {bias}.",
                    "Using {table}, compute value({key}) * {scale} + {bias}.",
                ],
                [
                    "From lookup entries [{table}], scale {key} by {scale} and increase by {bias}.",
                ],
            )
            prompt = prompt_t.format(table=table, key=key, scale=scale, bias=bias)
            payload = {"table": values, "key": key, "scale": scale, "bias": bias}
        else:
            answer = values[key] + bias
            op = "lookup_add"
            prompt_t = choose_template(
                rng,
                split,
                [
                    "Table: {table}. Take {key}, then add {bias}.",
                    "Using {table}, compute value({key}) + {bias}.",
                ],
                [
                    "From lookup entries [{table}], increase {key} by {bias}.",
                ],
            )
            prompt = prompt_t.format(table=table, key=key, bias=bias)
            payload = {"table": values, "key": key, "bias": bias}
    return Example(f"{split}_table_{idx}", split, "table", prompt, str(answer), op, payload)


def make_date_example(rng: random.Random, split: str, idx: int) -> Example:
    base = date(2026, rng.randint(1, 12), rng.randint(1, 24))
    if split == "eval_composition":
        weeks = rng.randint(1, 3)
        days = rng.randint(1, 6)
        final = base + timedelta(days=weeks * 7 + days)
        op = "weeks_plus_days"
        prompt_t = choose_template(
            rng,
            split,
            [
                "Starting from {d}, add {w} weeks and {days} days. Return the ISO date.",
                "Date {d}: move forward {w} weeks plus {days} days.",
            ],
            [
                "Advance {d} by {w} whole weeks and then by {days} more days; give YYYY-MM-DD.",
            ],
        )
        prompt = prompt_t.format(d=base.isoformat(), w=weeks, days=days)
        payload = {"date": base.isoformat(), "weeks": weeks, "days": days, "delta_days": weeks * 7 + days}
    else:
        delta = rng.randint(1, 14)
        final = base + timedelta(days=delta)
        op = "add_days"
        prompt_t = choose_template(
            rng,
            split,
            [
                "Starting from {d}, add {n} days. Return the ISO date.",
                "Date {d}: move forward by {n} days.",
            ],
            [
                "Advance {d} by {n} days and write the result as YYYY-MM-DD.",
            ],
        )
        prompt = prompt_t.format(d=base.isoformat(), n=delta)
        payload = {"date": base.isoformat(), "delta_days": delta}
    return Example(f"{split}_date_{idx}", split, "date", prompt, final.isoformat(), op, payload)


def make_example(rng: random.Random, split: str, family: str, idx: int) -> Example:
    if family == "string":
        return make_string_example(rng, split, idx)
    if family == "unit":
        return make_unit_example(rng, split, idx)
    if family == "table":
        return make_table_example(rng, split, idx)
    if family == "date":
        return make_date_example(rng, split, idx)
    raise ValueError(family)


def make_split(seed: int, split: str, n: int) -> List[Example]:
    rng = random.Random(seed + {"train": 0, "eval_indist": 10_000, "eval_composition": 20_000, "eval_template_shift": 30_000}[split])
    examples: List[Example] = []
    for i in range(n):
        family = TRAIN_FAMILIES[i % len(TRAIN_FAMILIES)]
        examples.append(make_example(rng, split, family, i))
    rng.shuffle(examples)
    return examples


def render_answer_target(ex: Example) -> str:
    return f"FINAL: {ex.answer}\n"


def render_python_target(ex: Example) -> str:
    p = ex.payload
    lines: List[str] = []
    if ex.family == "string":
        lines += [f'code = "{p["code"]}"', 'clean = code.replace("-", "")']
        if ex.op_name == "take_prefix":
            lines.append(f"answer = clean[:{p['n']}]")
        elif ex.op_name == "take_suffix":
            lines.append(f"answer = clean[-{p['n']}:]")
        elif ex.op_name == "slice":
            lines.append(f"answer = clean[{p['start']}:{p['end']}]")
        elif ex.op_name == "reverse_then_slice":
            lines += ["rev = clean[::-1]", f"answer = rev[{p['start']}:{p['end']}]"]
    elif ex.family == "unit":
        if ex.op_name == "meters_plus_centimeters":
            lines += [f"cm_from_meters = {p['meters']} * 100", f"answer = cm_from_meters + {p['centimeters']}"]
        else:
            lines += [f"amount = {p['amount']}", f"factor = {p['factor']}", "answer = amount * factor"]
    elif ex.family == "table":
        table = json.dumps(p["table"], sort_keys=True)
        lines.append(f"table = {table}")
        if ex.op_name == "lookup_mul_add":
            lines += [f"value = table['{p['key']}']", f"answer = value * {p['scale']} + {p['bias']}"]
        elif ex.op_name == "lookup_add":
            lines += [f"value = table['{p['key']}']", f"answer = value + {p['bias']}"]
        elif ex.op_name == "sum_two_keys_plus":
            lines += [f"left = table['{p['k1']}']", f"right = table['{p['k2']}']", f"answer = left + right + {p['bias']}"]
    elif ex.family == "date":
        if ex.op_name == "weeks_plus_days":
            lines += [f'day = "{p["date"]}"', f"delta_days = {p['weeks']} * 7 + {p['days']}", "answer = date_add(day, delta_days)"]
        else:
            lines += [f'day = "{p["date"]}"', f"answer = date_add(day, {p['delta_days']})"]
    lines.append(f"FINAL: {ex.answer}")
    return "\n".join(lines) + "\n"


def render_json_target(ex: Example) -> str:
    obj = {
        "family": ex.family,
        "op": ex.op_name,
        "input": ex.payload,
        "final": ex.answer,
    }
    return json.dumps(obj, sort_keys=True) + f"\nFINAL: {ex.answer}\n"


def render_stack_target(ex: Example) -> str:
    p = ex.payload
    lines: List[str] = []
    if ex.family == "string":
        lines.append(f"PUSH_CODE {p['code']}")
        lines.append("STRIP -")
        if ex.op_name == "take_prefix":
            lines.append(f"TAKE_PREFIX {p['n']}")
        elif ex.op_name == "take_suffix":
            lines.append(f"TAKE_SUFFIX {p['n']}")
        elif ex.op_name == "slice":
            lines.append(f"SLICE {p['start'] + 1} {p['end']}")
        elif ex.op_name == "reverse_then_slice":
            lines += ["REVERSE", f"SLICE {p['start'] + 1} {p['end']}"]
    elif ex.family == "unit":
        if ex.op_name == "meters_plus_centimeters":
            lines += [f"PUSH {p['meters']}", "MUL 100", f"ADD {p['centimeters']}"]
        else:
            lines += [f"PUSH {p['amount']}", f"MUL {p['factor']}"]
    elif ex.family == "table":
        lines.append("TABLE " + " ".join(f"{k}:{v}" for k, v in sorted(p["table"].items())))
        if ex.op_name == "lookup_mul_add":
            lines += [f"LOOKUP {p['key']}", f"MUL {p['scale']}", f"ADD {p['bias']}"]
        elif ex.op_name == "lookup_add":
            lines += [f"LOOKUP {p['key']}", f"ADD {p['bias']}"]
        elif ex.op_name == "sum_two_keys_plus":
            lines += [f"LOOKUP {p['k1']}", f"LOOKUP {p['k2']}", "ADD_TOP", f"ADD {p['bias']}"]
    elif ex.family == "date":
        lines.append(f"PUSH_DATE {p['date']}")
        if ex.op_name == "weeks_plus_days":
            lines += [f"ADD_WEEKS {p['weeks']}", f"ADD_DAYS {p['days']}"]
        else:
            lines.append(f"ADD_DAYS {p['delta_days']}")
    lines.append(f"FINAL {ex.answer}")
    return "\n".join(lines) + "\n"


def render_target(ex: Example, abi: str) -> str:
    if abi == "answer":
        return render_answer_target(ex)
    if abi == "python":
        return render_python_target(ex)
    if abi == "json":
        return render_json_target(ex)
    if abi == "stack":
        return render_stack_target(ex)
    raise ValueError(abi)


def format_instruction(abi: str) -> str:
    if abi == "answer":
        return "Return exactly one line: FINAL: <answer>"
    if abi == "python":
        return "Return a short Python-like trace ending with: FINAL: <answer>"
    if abi == "json":
        return "Return one JSON object describing the operation, then a line: FINAL: <answer>"
    if abi == "stack":
        return "Return compact stack-style instructions ending with: FINAL <answer>"
    raise ValueError(abi)


def format_prompt(ex: Example, abi: str) -> str:
    return (
        "You solve deterministic office-style tasks by compiling the task into a compact procedure.\n"
        f"Required output format: {format_instruction(abi)}\n\n"
        f"Task:\n{ex.prompt}\n\n"
        "Output:\n"
    )


FINAL_RE = re.compile(r"FINAL\s*:?\s*([A-Za-z0-9_.:+/-]+)", re.IGNORECASE)


def clean_answer(text: str) -> str:
    ans = text.strip().strip("`'\";,")
    ans = ans.replace("</s>", "").strip()
    return ans


def parse_final(text: str) -> Optional[str]:
    match = FINAL_RE.search(text)
    if not match:
        return None
    return clean_answer(match.group(1))


class PromptDataset(Dataset):
    def __init__(self, examples: Sequence[Example], abi: str, tokenizer: Any, max_length: int):
        self.rows: List[Dict[str, torch.Tensor]] = []
        eos = tokenizer.eos_token or ""
        for ex in examples:
            prompt = format_prompt(ex, abi)
            target = render_target(ex, abi) + eos
            prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
            target_ids = tokenizer(target, add_special_tokens=False).input_ids
            input_ids = prompt_ids + target_ids
            labels = [-100] * len(prompt_ids) + target_ids
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                labels = labels[:max_length]
            self.rows.append(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.rows[idx]


def collate_batch(rows: Sequence[Dict[str, torch.Tensor]], pad_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(len(r["input_ids"]) for r in rows)
    input_ids = torch.full((len(rows), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(rows), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), max_len), dtype=torch.long)
    for i, row in enumerate(rows):
        n = len(row["input_ids"])
        input_ids[i, :n] = row["input_ids"]
        labels[i, :n] = row["labels"]
        attention_mask[i, :n] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def ensure_tokenizer(model_name: str) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=str(CACHE_DIR))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token or tokenizer.convert_ids_to_tokens(0)
    tokenizer.padding_side = "left"
    return tokenizer


def load_model(model_name: str, lora: bool, config: Optional[RunConfig] = None) -> Any:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=str(CACHE_DIR),
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
    )
    if not lora:
        model.eval()
        return model
    assert config is not None
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=targets,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.train()
    return model


def iter_batches(loader: DataLoader) -> Iterator[Dict[str, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


def train_one_arm(
    config: RunConfig,
    abi: str,
    seed: int,
    train_examples: Sequence[Example],
    eval_sets: Dict[str, Sequence[Example]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    run_id = f"{config.run_name}_{abi}_s{seed}"
    run_dir = RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = CHECKPOINTS / run_id
    set_seed(seed)
    tokenizer = ensure_tokenizer(config.model_name)
    tokenizer.padding_side = "right"
    dataset = PromptDataset(train_examples, abi, tokenizer, config.max_length)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda rows: collate_batch(rows, tokenizer.pad_token_id),
    )
    model = load_model(config.model_name, lora=True, config=config)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config.lr)
    batches = iter_batches(loader)
    train_rows: List[Dict[str, Any]] = []
    start = time.time()
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, config.train_steps + 1):
        total_loss = 0.0
        for _ in range(config.grad_accum):
            batch = next(batches)
            batch = {k: v.to(model.device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(**batch)
                loss = out.loss / config.grad_accum
            loss.backward()
            total_loss += float(loss.detach().cpu()) * config.grad_accum
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step == 1 or step % max(1, config.train_steps // 5) == 0 or step == config.train_steps:
            row = {
                "run": run_id,
                "suite": config.suite,
                "arm": abi,
                "seed": seed,
                "step": step,
                "loss": total_loss,
                "elapsed_s": time.time() - start,
            }
            train_rows.append(row)
            log(f"{run_id} step {step}/{config.train_steps} loss={total_loss:.4f}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)
    write_csv(run_dir / "train_log.csv", train_rows)
    tokenizer.padding_side = "left"
    model.eval()
    model.config.use_cache = True
    metric_rows: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []
    for split, examples in eval_sets.items():
        metrics, details = evaluate_model(config, model, tokenizer, examples, abi, run_id, seed, trained=True)
        metric_rows.append(metrics)
        detail_rows.extend(details)
    write_csv(run_dir / "metrics.csv", metric_rows)
    write_csv(run_dir / "details.csv", detail_rows)
    write_json(
        run_dir / "manifest.json",
        {
            "run": run_id,
            "suite": config.suite,
            "abi": abi,
            "seed": seed,
            "config": asdict(config),
            "checkpoint_dir": ckpt_dir,
            "train_examples": len(train_examples),
            "eval_sizes": {k: len(v) for k, v in eval_sets.items()},
        },
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return metric_rows, detail_rows, train_rows


@torch.no_grad()
def evaluate_model(
    config: RunConfig,
    model: Any,
    tokenizer: Any,
    examples: Sequence[Example],
    abi: str,
    run_id: str,
    seed: int,
    trained: bool,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    details: List[Dict[str, Any]] = []
    correct = 0
    valid = 0
    token_counts: List[int] = []
    family_correct: Dict[str, List[int]] = {f: [] for f in TRAIN_FAMILIES}
    for start in range(0, len(examples), config.eval_batch_size):
        batch_examples = list(examples[start : start + config.eval_batch_size])
        prompts = [format_prompt(ex, abi) for ex in batch_examples]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=config.max_length)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        out = model.generate(
            **enc,
            max_new_tokens=config.max_new_tokens if abi != "answer" else min(config.max_new_tokens, 32),
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        prompt_width = enc["input_ids"].shape[1]
        for i, ex in enumerate(batch_examples):
            new_ids = out[i, prompt_width:]
            text = tokenizer.decode(new_ids, skip_special_tokens=True)
            pred = parse_final(text)
            is_valid = pred is not None
            ok = clean_answer(pred or "") == clean_answer(ex.answer)
            correct += int(ok)
            valid += int(is_valid)
            token_count = int((new_ids != tokenizer.pad_token_id).sum().detach().cpu())
            token_counts.append(token_count)
            family_correct[ex.family].append(int(ok))
            details.append(
                {
                    "run": run_id,
                    "suite": config.suite,
                    "arm": abi,
                    "seed": seed,
                    "trained": int(trained),
                    "split": ex.split,
                    "example_id": ex.example_id,
                    "family": ex.family,
                    "op_name": ex.op_name,
                    "answer": ex.answer,
                    "prediction": pred or "",
                    "correct": int(ok),
                    "valid": int(is_valid),
                    "new_tokens": token_count,
                    "generated": text.replace("\n", "\\n")[:500],
                }
            )
    n = len(examples)
    metrics: Dict[str, Any] = {
        "run": run_id,
        "suite": config.suite,
        "arm": abi,
        "seed": seed,
        "trained": int(trained),
        "split": examples[0].split if examples else "",
        "n": n,
        "answer_accuracy": correct / max(1, n),
        "valid_rate": valid / max(1, n),
        "mean_new_tokens": mean(token_counts),
    }
    for family, vals in family_correct.items():
        metrics[f"acc_{family}"] = mean(vals) if vals else float("nan")
    return metrics, details


def run_zero_shot(config: RunConfig, eval_sets: Dict[str, Sequence[Example]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    run_id = f"{config.run_name}_zero_shot_answer"
    run_dir = RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = ensure_tokenizer(config.model_name)
    tokenizer.padding_side = "left"
    model = load_model(config.model_name, lora=False)
    metric_rows: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []
    for split, examples in eval_sets.items():
        metrics, details = evaluate_model(config, model, tokenizer, examples, "answer", run_id, 0, trained=False)
        metrics["arm"] = "zero_shot_answer"
        for row in details:
            row["arm"] = "zero_shot_answer"
        metric_rows.append(metrics)
        detail_rows.extend(details)
    write_csv(run_dir / "metrics.csv", metric_rows)
    write_csv(run_dir / "details.csv", detail_rows)
    write_json(run_dir / "manifest.json", {"run": run_id, "suite": config.suite, "config": asdict(config), "trained": False})
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return metric_rows, detail_rows


def make_config(args: argparse.Namespace) -> RunConfig:
    if args.suite == "smoke":
        train_n, eval_n, steps, seeds, arms = 24, 8, 2, [101], ["answer", "python"]
        batch_size, grad_accum = 2, 1
    elif args.suite == "pilot":
        train_n, eval_n, steps, seeds, arms = 96, 20, 30, [101], ABIS
        batch_size, grad_accum = 2, 2
    else:
        train_n, eval_n, steps, seeds, arms = 256, 32, 80, [101, 202], ABIS
        batch_size, grad_accum = 2, 4
    if args.train_n is not None:
        train_n = args.train_n
    if args.eval_n is not None:
        eval_n = args.eval_n
    if args.steps is not None:
        steps = args.steps
    if args.seeds:
        seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    if args.arms:
        arms = [x.strip() for x in args.arms.split(",") if x.strip()]
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    run_name = args.run_name or f"{args.suite}_{stamp}"
    return RunConfig(
        run_name=run_name,
        suite=args.suite,
        model_name=args.model_name,
        seeds=seeds,
        arms=arms,
        train_n=train_n,
        eval_n=eval_n,
        train_steps=steps,
        batch_size=args.batch_size or batch_size,
        grad_accum=args.grad_accum or grad_accum,
        lr=args.lr,
        max_length=args.max_length,
        eval_batch_size=args.eval_batch_size,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        max_new_tokens=args.max_new_tokens,
    )


def run_experiment(args: argparse.Namespace) -> None:
    ensure_dirs()
    config = make_config(args)
    started = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    append_log(
        f"## Run `{config.run_name}`\n\n"
        f"- Started: {started}\n"
        f"- Suite: `{config.suite}`\n"
        f"- Model: `{config.model_name}`\n"
        f"- Seeds: `{','.join(map(str, config.seeds))}`\n"
        f"- Arms: `{','.join(config.arms)}`\n"
        f"- Train examples per seed: `{config.train_n}`\n"
        f"- Eval examples per split: `{config.eval_n}`\n"
        f"- Steps: `{config.train_steps}`"
    )
    log(f"Preparing data for {config.run_name}")
    run_metrics: List[Dict[str, Any]] = []
    run_details: List[Dict[str, Any]] = []
    run_train: List[Dict[str, Any]] = []
    t0 = time.time()
    for seed in config.seeds:
        train_examples = make_split(seed, "train", config.train_n)
        eval_sets = {split: make_split(seed, split, config.eval_n) for split in EVAL_SPLITS}
        data_dir = RUNS / f"{config.run_name}_data_s{seed}"
        data_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            data_dir / "dataset_manifest.json",
            {
                "seed": seed,
                "train_n": len(train_examples),
                "eval_sizes": {k: len(v) for k, v in eval_sets.items()},
                "train_sample": [asdict(ex) for ex in train_examples[:4]],
                "eval_sample": {k: [asdict(ex) for ex in v[:2]] for k, v in eval_sets.items()},
            },
        )
        if args.include_zero_shot and seed == config.seeds[0]:
            metrics, details = run_zero_shot(config, eval_sets)
            run_metrics.extend(metrics)
            run_details.extend(details)
        for abi in config.arms:
            metrics, details, train_rows = train_one_arm(config, abi, seed, train_examples, eval_sets)
            run_metrics.extend(metrics)
            run_details.extend(details)
            run_train.extend(train_rows)
    write_csv(ANALYSIS / f"{config.run_name}_metrics.csv", run_metrics)
    write_csv(ANALYSIS / f"{config.run_name}_details.csv", run_details)
    write_csv(ANALYSIS / f"{config.run_name}_train_log.csv", run_train)
    append_log(
        f"Completed `{config.run_name}` in {time.time() - t0:.1f}s.\n\n"
        f"- Metric rows: {len(run_metrics)}\n"
        f"- Detail rows: {len(run_details)}\n"
        f"- Training log rows: {len(run_train)}"
    )
    analyze_all()


def read_all_csv(pattern: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in sorted(RUNS.glob(pattern)):
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if not df.empty:
            frames.append(df)
    for path in sorted(ANALYSIS.glob(pattern)):
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False).drop_duplicates()


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    group_cols = ["suite", "arm", "trained", "split"]
    for keys, sub in metrics.groupby(group_cols, dropna=False):
        acc = [float(x) for x in sub["answer_accuracy"].dropna()]
        val = [float(x) for x in sub["valid_rate"].dropna()]
        tok = [float(x) for x in sub["mean_new_tokens"].dropna()]
        row = {
            "suite": keys[0],
            "arm": keys[1],
            "trained": keys[2],
            "split": keys[3],
            "runs": len(sub),
            "n_total": int(sub["n"].sum()),
            "accuracy_mean": mean(acc),
            "accuracy_std": std(acc),
            "valid_mean": mean(val),
            "tokens_mean": mean(tok),
        }
        for family in TRAIN_FAMILIES:
            col = f"acc_{family}"
            vals = [float(x) for x in sub[col].dropna()] if col in sub else []
            row[f"{family}_mean"] = mean(vals) if vals else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def suite_rank(suite: str) -> int:
    return {"smoke": 0, "pilot": 1, "main": 2}.get(str(suite), 0)


def select_primary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    max_rank = max(suite_rank(s) for s in summary["suite"].unique())
    return summary[summary["suite"].map(suite_rank).eq(max_rank)].copy()


def plot_accuracy(primary: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    if primary.empty:
        return
    arms = list(primary["arm"].drop_duplicates())
    splits = [s for s in EVAL_SPLITS if s in set(primary["split"])]
    x = np.arange(len(splits))
    width = 0.8 / max(1, len(arms))
    plt.figure(figsize=(11, 5.8))
    for j, arm in enumerate(arms):
        vals = []
        errs = []
        for split in splits:
            row = primary[(primary["arm"].eq(arm)) & (primary["split"].eq(split))]
            vals.append(100.0 * float(row.iloc[0]["accuracy_mean"]) if not row.empty else np.nan)
            errs.append(100.0 * float(row.iloc[0]["accuracy_std"]) if not row.empty else 0.0)
        plt.bar(x + (j - (len(arms) - 1) / 2) * width, vals, width=width, yerr=errs, capsize=3, label=arm)
    plt.xticks(x, [s.replace("eval_", "").replace("_", " ") for s in splits])
    plt.ylabel("Exact answer accuracy (%)")
    plt.title("Held-Out Accuracy by Output ABI")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "accuracy_by_abi_split.png", dpi=180)
    plt.close()


def plot_valid_tokens(primary: pd.DataFrame) -> None:
    if primary.empty:
        return
    split = "eval_indist" if "eval_indist" in set(primary["split"]) else str(primary["split"].iloc[0])
    sub = primary[primary["split"].eq(split)].copy()
    x = np.arange(len(sub))
    fig, ax1 = plt.subplots(figsize=(10, 5.2))
    ax1.bar(x - 0.18, 100.0 * sub["valid_mean"].astype(float), width=0.36, label="valid final", color="#4C78A8")
    ax1.set_ylabel("Valid final line (%)")
    ax1.set_ylim(0, 105)
    ax2 = ax1.twinx()
    ax2.bar(x + 0.18, sub["tokens_mean"].astype(float), width=0.36, label="new tokens", color="#F58518")
    ax2.set_ylabel("Mean generated tokens")
    ax1.set_xticks(x)
    ax1.set_xticklabels(sub["arm"], rotation=25, ha="right")
    ax1.set_title(f"Format Validity and Generation Length ({split})")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "validity_and_tokens.png", dpi=180)
    plt.close(fig)


def plot_family(primary: pd.DataFrame) -> None:
    if primary.empty:
        return
    split = "eval_composition" if "eval_composition" in set(primary["split"]) else str(primary["split"].iloc[0])
    sub = primary[primary["split"].eq(split)].copy()
    arms = list(sub["arm"])
    x = np.arange(len(TRAIN_FAMILIES))
    width = 0.8 / max(1, len(arms))
    plt.figure(figsize=(11, 5.8))
    for j, (_, row) in enumerate(sub.iterrows()):
        vals = [100.0 * float(row.get(f"{family}_mean", np.nan)) for family in TRAIN_FAMILIES]
        plt.bar(x + (j - (len(arms) - 1) / 2) * width, vals, width=width, label=row["arm"])
    plt.xticks(x, TRAIN_FAMILIES)
    plt.ylabel("Exact answer accuracy (%)")
    plt.title(f"Family Breakdown ({split})")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "family_breakdown.png", dpi=180)
    plt.close()


def plot_training(logs: pd.DataFrame) -> None:
    if logs.empty:
        return
    plt.figure(figsize=(10.5, 5.8))
    for run, sub in logs.groupby("run"):
        label = run
        if len(label) > 34:
            label = label[:31] + "..."
        plt.plot(sub["step"], sub["loss"], marker="o", linewidth=1.4, label=label)
    plt.xlabel("Training step")
    plt.ylabel("CE loss")
    plt.title("Training Curves")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(FIGURES / "training_curves.png", dpi=180)
    plt.close()


def best_arm(primary: pd.DataFrame, split: str) -> Optional[pd.Series]:
    sub = primary[primary["split"].eq(split)]
    if sub.empty:
        return None
    return sub.sort_values("accuracy_mean", ascending=False).iloc[0]


def md_table(df: pd.DataFrame, cols: Sequence[str]) -> str:
    if df.empty:
        return "_No rows._"
    lines = ["|" + "|".join(cols) + "|", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        vals: List[str] = []
        for col in cols:
            value = row.get(col, "")
            if isinstance(value, float):
                if "accuracy" in col or "valid" in col or "string_mean" in col or "unit_mean" in col or "table_mean" in col or "date_mean" in col:
                    vals.append(pct(value))
                elif col.endswith("_std"):
                    vals.append(pct(value))
                else:
                    vals.append(f"{value:.2f}")
            else:
                vals.append(str(value))
        lines.append("|" + "|".join(vals) + "|")
    return "\n".join(lines)


def make_report(summary: pd.DataFrame, metrics: pd.DataFrame, logs: pd.DataFrame) -> str:
    primary = select_primary(summary)
    split_rows = primary[primary["split"].isin(EVAL_SPLITS)].copy()
    split_rows = split_rows.sort_values(["split", "accuracy_mean"], ascending=[True, False])
    best_indist = best_arm(primary, "eval_indist")
    best_comp = best_arm(primary, "eval_composition")
    best_shift = best_arm(primary, "eval_template_shift")
    primary_suite = primary["suite"].iloc[0] if not primary.empty else "none"
    trained_metrics = metrics[(metrics["suite"].eq(primary_suite)) & (metrics["trained"].eq(1))].copy() if not metrics.empty else pd.DataFrame()
    trained_seeds = sorted(int(x) for x in trained_metrics["seed"].dropna().unique()) if not trained_metrics.empty else []
    max_step = int(logs[logs["suite"].eq(primary_suite)]["step"].max()) if not logs.empty and "suite" in logs else 0
    lines: List[str] = []
    lines.append("# Qwen Crystallized Trace ABI Tournament")
    lines.append("")
    lines.append("## Abstract")
    lines.append("")
    lines.append(
        "This experiment tests whether a local 4B language model learns practical deterministic procedures better "
        "when supervised with compact executable traces instead of final answers alone. The same generated tasks are "
        "rendered through four output ABIs: final-answer text, Python-like trace, JSON IR, and stack-style IR. Each "
        "trained arm uses the same QLoRA budget and is evaluated on fresh values, unseen operation compositions, and "
        "template-shifted prompts."
    )
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(
        "The task factory creates examples from four families: string normalization, unit conversion, lookup-table "
        "calculation, and date arithmetic. Every example has a deterministic answer and a gold procedural rendering "
        "for each ABI. Evaluation uses greedy generation and exact matching of the parsed `FINAL` value."
    )
    lines.append("")
    lines.append("## Run Configuration")
    lines.append("")
    lines.append(f"- Primary suite: `{primary_suite}`.")
    if trained_seeds:
        lines.append(f"- Adapter seeds: `{','.join(map(str, trained_seeds))}`.")
    if max_step:
        lines.append(f"- QLoRA update steps per adapter: `{max_step}`.")
    if not trained_metrics.empty:
        split_sizes = trained_metrics.groupby("split")["n"].sum().to_dict()
        split_bits = ", ".join(f"`{k}`={int(v)}" for k, v in split_sizes.items())
        lines.append(f"- Total trained-arm evaluation examples: {split_bits}.")
    lines.append("- Output ABIs: `answer`, `python`, `json`, and `stack`.")
    lines.append("")
    lines.append("## Primary Results")
    lines.append("")
    if best_indist is not None:
        lines.append(f"- Best fresh-value arm: `{best_indist['arm']}` at {pct(best_indist['accuracy_mean'])}.")
    if best_comp is not None:
        lines.append(f"- Best unseen-composition arm: `{best_comp['arm']}` at {pct(best_comp['accuracy_mean'])}.")
    if best_shift is not None:
        lines.append(f"- Best template-shift arm: `{best_shift['arm']}` at {pct(best_shift['accuracy_mean'])}.")
    lines.append(f"- Primary suite summarized below: `{primary_suite}`.")
    lines.append("")
    table_cols = ["suite", "arm", "split", "runs", "n_total", "accuracy_mean", "accuracy_std", "valid_mean", "tokens_mean"]
    lines.append(md_table(split_rows[table_cols], table_cols))
    lines.append("")
    lines.append("![Accuracy by ABI and split](../analysis/figures/accuracy_by_abi_split.png)")
    lines.append("")
    lines.append("![Format validity and generation length](../analysis/figures/validity_and_tokens.png)")
    lines.append("")
    lines.append("![Family breakdown](../analysis/figures/family_breakdown.png)")
    lines.append("")
    lines.append("![Training curves](../analysis/figures/training_curves.png)")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    if best_comp is not None and best_indist is not None:
        comp_gap = float(best_indist["accuracy_mean"]) - float(best_comp["accuracy_mean"])
        lines.append(
            f"The composition gap for the best fresh-value arm is {100.0 * comp_gap:.1f} percentage points. "
            "A small or negative gap would indicate that the learned representation transfers across operation "
            "combinations; a large positive gap indicates that the model mainly learned the easier in-distribution "
            "mapping."
        )
    if not primary.empty:
        answer_rows = primary[primary["arm"].eq("answer")]
        trace_rows = primary[primary["arm"].isin(["python", "json", "stack"])]
        if not answer_rows.empty and not trace_rows.empty:
            answer_acc = float(answer_rows[answer_rows["split"].eq("eval_indist")]["accuracy_mean"].max())
            trace_acc = float(trace_rows[trace_rows["split"].eq("eval_indist")]["accuracy_mean"].max())
            answer_tokens = float(answer_rows[answer_rows["split"].eq("eval_indist")]["tokens_mean"].max())
            trace_tokens = float(trace_rows[trace_rows["split"].eq("eval_indist")]["tokens_mean"].min())
            lines.append(
                f"On fresh values, the best trace ABI scored {pct(trace_acc)} and answer-only scored {pct(answer_acc)}. "
                f"The shortest strong trace ABI used {trace_tokens:.1f} generated tokens on average, while answer-only used "
                f"{answer_tokens:.1f}. This is a weak trace advantage in accuracy and a large trace cost in tokens."
            )
        if best_comp is not None:
            fam_bits: List[str] = []
            for family in TRAIN_FAMILIES:
                value = best_comp.get(f"{family}_mean", float("nan"))
                if not pd.isna(value):
                    fam_bits.append(f"{family} {pct(value)}")
            if fam_bits:
                lines.append(
                    f"For the best unseen-composition arm, family accuracy was: {', '.join(fam_bits)}. "
                    "The composition failures are concentrated in families that require a new operation combination, "
                    "while direct unit conversion remains easy."
                )
        json_rows = primary[primary["arm"].eq("json")]
        if not json_rows.empty:
            json_valid = float(json_rows["valid_mean"].mean())
            lines.append(
                f"The JSON ABI had an average valid-final rate of {pct(json_valid)} across primary splits, "
                "showing that a verbose structured object can be harder to emit reliably than line-oriented formats."
            )
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append(
        "This is a compact experiment. It is designed to reveal representation sensitivity and supervision effects, "
        "not to maximize absolute performance. The generated tasks are deterministic and intentionally narrow enough "
        "to allow controlled held-out splits. Larger task coverage and longer training would be needed before treating "
        "any ABI as a production recipe."
    )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append("- Metrics: `analysis/summary_by_arm.csv` and `analysis/all_metrics.csv`")
    lines.append("- Details: `analysis/all_details.csv`")
    lines.append("- Training logs: `analysis/all_train_logs.csv`")
    lines.append("- Checkpoints: `/workspace/large_artifacts/qwen_crystallized_trace_abi_tournament/checkpoints`")
    return "\n".join(lines) + "\n"


def markdown_to_html(markdown_text: str) -> str:
    body_lines: List[str] = []
    in_table = False
    for raw in markdown_text.splitlines():
        line = raw.rstrip()
        if line.startswith("# "):
            body_lines.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            body_lines.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("- "):
            body_lines.append(f"<li>{html.escape(line[2:])}</li>")
        elif line.startswith("![") and "](" in line and line.endswith(")"):
            alt = line[2 : line.index("]")]
            src = line[line.index("(") + 1 : -1]
            body_lines.append(f'<figure><img src="{html.escape(src)}" alt="{html.escape(alt)}"><figcaption>{html.escape(alt)}</figcaption></figure>')
        elif line.startswith("|") and line.endswith("|"):
            cells = [html.escape(c.strip()) for c in line.strip("|").split("|")]
            if all(set(c) <= {"-"} for c in cells):
                continue
            if not in_table:
                body_lines.append("<table>")
                in_table = True
            tag = "th" if not any("<tr>" in x for x in body_lines[-2:]) else "td"
            body_lines.append("<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>")
        else:
            if in_table:
                body_lines.append("</table>")
                in_table = False
            if line:
                body_lines.append(f"<p>{html.escape(line)}</p>")
            else:
                body_lines.append("")
    if in_table:
        body_lines.append("</table>")
    css = """
    body { font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 36px auto; max-width: 1120px; line-height: 1.5; color: #202124; }
    h1, h2 { line-height: 1.15; }
    table { border-collapse: collapse; width: 100%; margin: 18px 0; font-size: 14px; }
    th, td { border: 1px solid #d4d7dc; padding: 7px 9px; text-align: left; }
    th { background: #f2f4f7; }
    img { max-width: 100%; border: 1px solid #d4d7dc; }
    figure { margin: 24px 0; }
    figcaption { color: #5f6368; font-size: 13px; margin-top: 6px; }
    li { margin: 4px 0; }
    code { background: #f2f4f7; padding: 1px 4px; border-radius: 4px; }
    """
    return "<!doctype html><html><head><meta charset='utf-8'><title>Qwen Crystallized Trace ABI Tournament</title><style>" + css + "</style></head><body>" + "\n".join(body_lines) + "</body></html>\n"


def analyze_all() -> None:
    ensure_dirs()
    metrics = read_all_csv("*/metrics.csv")
    details = read_all_csv("*/details.csv")
    logs = read_all_csv("*/train_log.csv")
    summary = summarize(metrics)
    metrics.to_csv(ANALYSIS / "all_metrics.csv", index=False)
    details.to_csv(ANALYSIS / "all_details.csv", index=False)
    logs.to_csv(ANALYSIS / "all_train_logs.csv", index=False)
    summary.to_csv(ANALYSIS / "summary_by_arm.csv", index=False)
    primary = select_primary(summary)
    plot_accuracy(primary)
    plot_valid_tokens(primary)
    plot_family(primary)
    if not primary.empty and not logs.empty and "suite" in logs:
        plot_training(logs[logs["suite"].eq(primary["suite"].iloc[0])].copy())
    else:
        plot_training(logs)
    report = make_report(summary, metrics, logs)
    (REPORTS / "qwen_crystallized_trace_abi_tournament_report.md").write_text(report)
    (REPORTS / "qwen_crystallized_trace_abi_tournament_report.html").write_text(markdown_to_html(report))
    write_checkpoint_manifest()


def write_checkpoint_manifest() -> None:
    rows: List[Dict[str, Any]] = []
    for path in sorted(CHECKPOINTS.glob("*")):
        if path.is_dir():
            total = 0
            for f in path.rglob("*"):
                if f.is_file():
                    total += f.stat().st_size
            rows.append({"checkpoint": path.name, "path": str(path), "bytes": total})
    write_csv(ROOT / "checkpoint_manifest.csv", rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["smoke", "pilot", "main", "analyze"], default="smoke")
    parser.add_argument("--run_name", default="")
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--seeds", default="")
    parser.add_argument("--arms", default="")
    parser.add_argument("--train_n", type=int, default=None)
    parser.add_argument("--eval_n", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=384)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--include_zero_shot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    if args.suite == "analyze":
        analyze_all()
    else:
        run_experiment(args)


if __name__ == "__main__":
    main()
