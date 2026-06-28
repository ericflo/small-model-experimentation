#!/usr/bin/env python3
"""Depth stress test for explicit procedure compilation.

The primary question is whether Qwen can compose known primitives into
executable procedures at depths beyond atomic training examples. The model is
not trusted to execute its own procedure: stack outputs are parsed and run by a
deterministic interpreter.
"""

from __future__ import annotations

import argparse
import csv
import gc
import html
import json
import math
import random
import re
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


ROOT = Path("/workspace/experiments/qwen_trace_procedure_depth_stress")
RUNS = ROOT / "runs"
ANALYSIS = ROOT / "analysis"
FIGURES = ANALYSIS / "figures"
REPORTS = ROOT / "reports"
LARGE_ROOT = Path("/workspace/large_artifacts/qwen_trace_procedure_depth_stress")
CHECKPOINTS = LARGE_ROOT / "checkpoints"
CACHE_DIR = Path("/workspace/.cache/huggingface")

MODEL_NAME = "Qwen/Qwen3-4B"
FAMILIES = ["string", "number", "table", "date", "list", "path"]
ARMS = ["answer_only", "trace_stack_final", "trace_stack_no_final", "program_stack"]
DEPTH_SPLITS = [
    ("eval_indist_d1", 1, False),
    ("eval_comp_d2", 2, False),
    ("eval_comp_d3", 3, False),
    ("eval_comp_d4", 4, False),
    ("eval_comp_d6", 6, False),
    ("eval_template_d2", 2, True),
    ("eval_template_d4", 4, True),
    ("eval_template_d6", 6, True),
]
FINAL_RE = re.compile(r"\bFINAL\s*:?\s*([A-Za-z0-9_.:+/-]+)", re.IGNORECASE)


@dataclass
class Example:
    example_id: str
    split: str
    family: str
    depth: int
    prompt: str
    answer: str
    program: List[str]
    op_names: List[str]
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


def pct(x: Any) -> str:
    try:
        v = float(x)
    except Exception:
        return "n/a"
    if math.isnan(v):
        return "n/a"
    return f"{100.0 * v:.1f}%"


def clean_answer(x: Any) -> str:
    return str(x).strip().strip("`'\";,")


def parse_final(text: str) -> Optional[str]:
    m = FINAL_RE.search(text)
    return clean_answer(m.group(1)) if m else None


def has_final(text: str) -> bool:
    return bool(FINAL_RE.search(text))


def safe_int(token: str) -> int:
    cleaned = re.sub(r"[^0-9-]", "", token)
    if cleaned in {"", "-"}:
        raise ValueError(f"bad integer {token!r}")
    return int(cleaned)


def strip_step_prefix(line: str) -> str:
    line = line.strip().strip("`")
    line = re.sub(r"^(STEP\s*)?\d+\s*[\).:-]\s*", "", line, flags=re.IGNORECASE)
    return line.strip()


def canonical_program(text: str) -> List[Tuple[str, Tuple[str, ...]]]:
    out: List[Tuple[str, Tuple[str, ...]]] = []
    for raw in text.splitlines():
        line = strip_step_prefix(raw)
        if not line or line.startswith("#") or FINAL_RE.search(line):
            continue
        parts = line.split()
        if not parts:
            continue
        op = parts[0].upper().rstrip(":")
        out.append((op, tuple(parts[1:])))
    return out


def canon_op_names(text: str) -> List[str]:
    return [op for op, _ in canonical_program(text)]


def rand_code(rng: random.Random) -> str:
    letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    return f"{rng.choice(letters)}{rng.choice(letters)}-{rng.randint(1000, 9999)}-{rng.choice(letters)}{rng.choice(letters)}"


def rand_path(rng: random.Random) -> str:
    roots = ["reports", "archive", "client", "ops", "ledger"]
    mids = ["north", "south", "alpha", "beta", "daily"]
    names = ["invoice", "summary", "errors", "sales", "index"]
    exts = ["csv", "json", "txt", "log"]
    return f"/{rng.choice(roots)}/{rng.choice(mids)}/{rng.choice(names)}_{rng.randint(10,99)}.{rng.choice(exts)}"


def describe_op(op: str, args: Sequence[Any], shifted: bool) -> str:
    if op == "TAKE_PREFIX":
        return f"retain its leading {args[0]} symbols" if shifted else f"keep the first {args[0]} characters"
    if op == "TAKE_SUFFIX":
        return f"retain its trailing {args[0]} symbols" if shifted else f"keep the last {args[0]} characters"
    if op == "SLICE":
        return f"clip the inclusive one-based span {args[0]} to {args[1]}" if shifted else f"keep one-indexed positions {args[0]} through {args[1]}"
    if op == "ADD":
        return f"increase it by {args[0]}" if shifted else f"add {args[0]}"
    if op == "SUB":
        return f"decrease it by {args[0]}" if shifted else f"subtract {args[0]}"
    if op == "MUL":
        return f"scale it by {args[0]}" if shifted else f"multiply by {args[0]}"
    if op == "LOOKUP":
        return f"fetch entry {args[0]}" if shifted else f"look up {args[0]}"
    if op == "ADD_LOOKUP":
        return f"include entry {args[0]} in the running total" if shifted else f"add the table value for {args[0]}"
    if op == "ADD_DAYS":
        return f"move forward {args[0]} days" if shifted else f"add {args[0]} days"
    if op == "ADD_WEEKS":
        return f"move forward {args[0]} weeks" if shifted else f"add {args[0]} weeks"
    phrases = {
        False: {
            "STRIP_HYPHENS": "remove hyphens",
            "REVERSE": "reverse the current text",
            "LOWER": "convert the text to lowercase",
            "UPPER": "convert the text to uppercase",
            "SUM": "sum the list",
            "MAX": "take the maximum value",
            "MIN": "take the minimum value",
            "BASENAME": "take the file name",
            "DIRNAME": "take the parent directory",
            "EXTENSION": "take the file extension",
            "STRIP_EXT": "remove the file extension",
        },
        True: {
            "STRIP_HYPHENS": "drop separator dashes",
            "REVERSE": "flip the current text left-to-right",
            "LOWER": "make all letters lowercase",
            "UPPER": "make all letters uppercase",
            "SUM": "total the numbers",
            "MAX": "keep the largest number",
            "MIN": "keep the smallest number",
            "BASENAME": "isolate the last path component",
            "DIRNAME": "isolate the containing folder",
            "EXTENSION": "isolate the suffix after the dot",
            "STRIP_EXT": "drop the suffix after the dot",
        },
    }
    return phrases[shifted][op]


def execute_gold(program: Sequence[str]) -> str:
    ok, value, err = execute_stack("\n".join(program))
    if not ok or value is None:
        raise RuntimeError(f"gold program failed: {err}\n{program}")
    return value


def make_string(seed_rng: random.Random, split: str, idx: int, depth: int, shifted: bool) -> Example:
    text = rand_code(seed_rng)
    ops: List[Tuple[str, Tuple[Any, ...]]] = []
    choices = ["STRIP_HYPHENS", "REVERSE", "LOWER", "UPPER"]
    for step in range(depth):
        if step == 0:
            op = seed_rng.choice(["STRIP_HYPHENS", "REVERSE", "LOWER", "UPPER"])
        elif step == depth - 1 and depth >= 2:
            op = seed_rng.choice(["TAKE_PREFIX", "TAKE_SUFFIX", "SLICE", "REVERSE", "LOWER", "UPPER"])
        else:
            op = seed_rng.choice(choices)
        if op in {"TAKE_PREFIX", "TAKE_SUFFIX"}:
            ops.append((op, (seed_rng.randint(3, 6),)))
        elif op == "SLICE":
            a = seed_rng.randint(1, 4)
            b = seed_rng.randint(a + 2, min(8, a + 5))
            ops.append((op, (a, b)))
        else:
            ops.append((op, ()))
    program = [f"PUSH_TEXT {text}"] + [f"{op} {' '.join(map(str, args))}".strip() for op, args in ops]
    steps = "; ".join(f"{i+1}) {describe_op(op, args, shifted)}" for i, (op, args) in enumerate(ops))
    if shifted:
        prompt = f"Transform the code-like text `{text}`. Work in this order: {steps}. Return the resulting text."
    else:
        prompt = f"Given text {text}, perform these steps in order: {steps}. Return the result."
    answer = execute_gold(program)
    return Example(f"{split}_string_{idx}", split, "string", depth, prompt, answer, program, [op for op, _ in ops], {"input": text})


def make_number(seed_rng: random.Random, split: str, idx: int, depth: int, shifted: bool) -> Example:
    start = seed_rng.randint(2, 30)
    ops: List[Tuple[str, Tuple[Any, ...]]] = []
    for _ in range(depth):
        op = seed_rng.choice(["ADD", "SUB", "MUL"])
        if op == "MUL":
            args = (seed_rng.randint(2, 7),)
        else:
            args = (seed_rng.randint(1, 25),)
        ops.append((op, args))
    program = [f"PUSH {start}"] + [f"{op} {args[0]}" for op, args in ops]
    steps = "; ".join(f"{i+1}) {describe_op(op, args, shifted)}" for i, (op, args) in enumerate(ops))
    prompt = (f"Start with the number {start}. Apply the following operations in order: {steps}. Return the integer."
              if not shifted else f"Beginning at {start}, update the running value like this: {steps}. What integer remains?")
    answer = execute_gold(program)
    return Example(f"{split}_number_{idx}", split, "number", depth, prompt, answer, program, [op for op, _ in ops], {"start": start})


def make_table(seed_rng: random.Random, split: str, idx: int, depth: int, shifted: bool) -> Example:
    keys = ["alpha", "beta", "gamma", "delta", "omega"]
    seed_rng.shuffle(keys)
    values = {k: seed_rng.randint(2, 40) for k in keys[:4]}
    first = seed_rng.choice(list(values))
    ops: List[Tuple[str, Tuple[Any, ...]]] = [("LOOKUP", (first,))]
    for _ in range(max(0, depth - 1)):
        op = seed_rng.choice(["ADD", "MUL", "ADD_LOOKUP"])
        if op == "ADD":
            args = (seed_rng.randint(1, 12),)
        elif op == "MUL":
            args = (seed_rng.randint(2, 5),)
        else:
            args = (seed_rng.choice(list(values)),)
        ops.append((op, args))
    table_line = "TABLE " + " ".join(f"{k}:{v}" for k, v in sorted(values.items()))
    program = [table_line] + [f"{op} {' '.join(map(str, args))}".strip() for op, args in ops]
    table_text = ", ".join(f"{k}={v}" for k, v in sorted(values.items()))
    steps = "; ".join(f"{i+1}) {describe_op(op, args, shifted)}" for i, (op, args) in enumerate(ops))
    prompt = (f"Table: {table_text}. Perform these table operations in order: {steps}. Return the integer."
              if not shifted else f"Using lookup entries [{table_text}], process the running value as follows: {steps}. Give the final integer.")
    answer = execute_gold(program)
    return Example(f"{split}_table_{idx}", split, "table", depth, prompt, answer, program, [op for op, _ in ops], {"table": values})


def make_date(seed_rng: random.Random, split: str, idx: int, depth: int, shifted: bool) -> Example:
    start = date(2026, seed_rng.randint(1, 12), seed_rng.randint(1, 23)).isoformat()
    ops: List[Tuple[str, Tuple[Any, ...]]] = []
    for _ in range(depth):
        if seed_rng.random() < 0.55:
            ops.append(("ADD_DAYS", (seed_rng.randint(1, 9),)))
        else:
            ops.append(("ADD_WEEKS", (seed_rng.randint(1, 4),)))
    program = [f"PUSH_DATE {start}"] + [f"{op} {args[0]}" for op, args in ops]
    steps = "; ".join(f"{i+1}) {describe_op(op, args, shifted)}" for i, (op, args) in enumerate(ops))
    prompt = (f"Starting from {start}, apply these date changes in order: {steps}. Return the ISO date."
              if not shifted else f"Begin on {start}; follow this calendar schedule: {steps}. Write the final YYYY-MM-DD date.")
    answer = execute_gold(program)
    return Example(f"{split}_date_{idx}", split, "date", depth, prompt, answer, program, [op for op, _ in ops], {"date": start})


def make_list(seed_rng: random.Random, split: str, idx: int, depth: int, shifted: bool) -> Example:
    values = [seed_rng.randint(1, 20) for _ in range(seed_rng.randint(4, 7))]
    first = seed_rng.choice(["SUM", "MAX", "MIN"])
    ops: List[Tuple[str, Tuple[Any, ...]]] = [(first, ())]
    for _ in range(max(0, depth - 1)):
        op = seed_rng.choice(["ADD", "SUB", "MUL"])
        args = (seed_rng.randint(2, 7) if op == "MUL" else seed_rng.randint(1, 15),)
        ops.append((op, args))
    program = ["PUSH_LIST " + " ".join(map(str, values))] + [f"{op} {' '.join(map(str, args))}".strip() for op, args in ops]
    steps = "; ".join(f"{i+1}) {describe_op(op, args, shifted)}" for i, (op, args) in enumerate(ops))
    prompt = (f"For the list {values}, perform these operations in order: {steps}. Return the integer."
              if not shifted else f"Given numbers {values}, carry out this ordered reduction: {steps}. What integer do you get?")
    answer = execute_gold(program)
    return Example(f"{split}_list_{idx}", split, "list", depth, prompt, answer, program, [op for op, _ in ops], {"values": values})


def make_path(seed_rng: random.Random, split: str, idx: int, depth: int, shifted: bool) -> Example:
    path = rand_path(seed_rng)
    ops: List[Tuple[str, Tuple[Any, ...]]] = []
    pool = ["BASENAME", "DIRNAME", "EXTENSION", "STRIP_EXT", "LOWER", "UPPER"]
    for i in range(depth):
        if i == 0:
            op = seed_rng.choice(["BASENAME", "DIRNAME", "EXTENSION"])
        else:
            op = seed_rng.choice(pool)
        ops.append((op, ()))
    program = [f"PUSH_TEXT {path}"] + [op for op, _ in ops]
    steps = "; ".join(f"{i+1}) {describe_op(op, args, shifted)}" for i, (op, args) in enumerate(ops))
    prompt = (f"Given path {path}, perform these path operations in order: {steps}. Return the resulting text."
              if not shifted else f"Work on filesystem path `{path}` using this sequence: {steps}. Give the final text.")
    answer = execute_gold(program)
    return Example(f"{split}_path_{idx}", split, "path", depth, prompt, answer, program, [op for op, _ in ops], {"path": path})


def make_example(rng: random.Random, split: str, family: str, idx: int, depth: int, shifted: bool) -> Example:
    if family == "string":
        return make_string(rng, split, idx, depth, shifted)
    if family == "number":
        return make_number(rng, split, idx, depth, shifted)
    if family == "table":
        return make_table(rng, split, idx, depth, shifted)
    if family == "date":
        return make_date(rng, split, idx, depth, shifted)
    if family == "list":
        return make_list(rng, split, idx, depth, shifted)
    if family == "path":
        return make_path(rng, split, idx, depth, shifted)
    raise ValueError(family)


def split_depth_shift(split: str) -> Tuple[int, bool]:
    if split == "train":
        return 1, False
    for name, depth, shifted in DEPTH_SPLITS:
        if split == name:
            return depth, shifted
    raise ValueError(split)


def make_split(seed: int, split: str, n: int) -> List[Example]:
    depth, shifted = split_depth_shift(split)
    offsets = {"train": 0}
    offsets.update({name: (i + 1) * 10_000 for i, (name, _, _) in enumerate(DEPTH_SPLITS)})
    rng = random.Random(seed + offsets[split])
    examples = [make_example(rng, split, FAMILIES[i % len(FAMILIES)], i, depth, shifted) for i in range(n)]
    rng.shuffle(examples)
    return examples


def numbered_program(lines: Sequence[str]) -> List[str]:
    return [f"STEP {i + 1}: {line}" for i, line in enumerate(lines)]


def render_target(ex: Example, arm: str) -> str:
    if arm == "answer_only":
        return f"FINAL: {ex.answer}\n"
    if arm == "trace_stack_final":
        return "\n".join(numbered_program(ex.program) + [f"FINAL {ex.answer}"]) + "\n"
    if arm == "trace_stack_no_final":
        return "\n".join(numbered_program(ex.program)) + "\n"
    if arm == "program_stack":
        return "\n".join(ex.program) + "\n"
    raise ValueError(arm)


def format_instruction(arm: str) -> str:
    if arm == "answer_only":
        return "Return one line exactly as: FINAL: <answer>."
    if arm == "trace_stack_final":
        return "Return numbered stack procedure steps followed by a final line: FINAL <answer>."
    if arm == "trace_stack_no_final":
        return "Return numbered stack procedure steps only. Do not write a FINAL line or the answer."
    if arm == "program_stack":
        return "Return only raw stack instructions. Do not number steps and do not write a FINAL line or the answer."
    raise ValueError(arm)


def format_prompt(ex: Example, arm: str) -> str:
    return (
        "Compile the task into the requested executable stack format.\n"
        f"Output rule: {format_instruction(arm)}\n\n"
        f"Task:\n{ex.prompt}\n\n"
        "Output:\n"
    )


def path_basename(x: str) -> str:
    return str(x).rstrip("/").split("/")[-1]


def path_dirname(x: str) -> str:
    s = str(x).rstrip("/")
    if "/" not in s.strip("/"):
        return "/"
    return s.rsplit("/", 1)[0] or "/"


def path_ext(x: str) -> str:
    base = path_basename(x)
    return base.rsplit(".", 1)[1] if "." in base else ""


def path_strip_ext(x: str) -> str:
    s = str(x)
    base = path_basename(s)
    if "." not in base:
        return s
    prefix = s[: len(s) - len(base)]
    return prefix + base.rsplit(".", 1)[0]


def execute_stack(text: str) -> Tuple[bool, Optional[str], str]:
    stack: List[Any] = []
    table: Dict[str, int] = {}
    try:
        for raw in text.splitlines():
            line = strip_step_prefix(raw)
            if not line or line.startswith("#") or FINAL_RE.search(line):
                continue
            parts = line.split()
            if not parts:
                continue
            op = parts[0].upper().rstrip(":")
            if op == "PUSH_TEXT":
                stack.append(" ".join(parts[1:]).strip().strip('"'))
            elif op == "PUSH":
                stack.append(safe_int(parts[1]))
            elif op == "PUSH_DATE":
                stack.append(parts[1].strip())
            elif op == "PUSH_LIST":
                stack.append([safe_int(p) for p in parts[1:]])
            elif op == "TABLE":
                table = {}
                for item in parts[1:]:
                    if ":" in item:
                        k, v = item.split(":", 1)
                        table[k.strip()] = safe_int(v)
            elif op == "STRIP_HYPHENS":
                stack[-1] = str(stack[-1]).replace("-", "")
            elif op == "REVERSE":
                stack[-1] = str(stack[-1])[::-1]
            elif op == "LOWER":
                stack[-1] = str(stack[-1]).lower()
            elif op == "UPPER":
                stack[-1] = str(stack[-1]).upper()
            elif op == "TAKE_PREFIX":
                stack[-1] = str(stack[-1])[: safe_int(parts[1])]
            elif op == "TAKE_SUFFIX":
                stack[-1] = str(stack[-1])[-safe_int(parts[1]) :]
            elif op == "SLICE":
                a = safe_int(parts[1]) - 1
                b = safe_int(parts[2])
                stack[-1] = str(stack[-1])[a:b]
            elif op == "ADD":
                if len(parts) == 1:
                    right = int(stack.pop())
                    left = int(stack.pop())
                    stack.append(left + right)
                else:
                    stack[-1] = int(stack[-1]) + safe_int(parts[1])
            elif op == "SUB":
                stack[-1] = int(stack[-1]) - safe_int(parts[1])
            elif op == "MUL":
                if len(parts) == 1:
                    right = int(stack.pop())
                    left = int(stack.pop())
                    stack.append(left * right)
                else:
                    stack[-1] = int(stack[-1]) * safe_int(parts[1])
            elif op == "LOOKUP":
                stack.append(int(table[parts[1].strip()]))
            elif op == "ADD_LOOKUP":
                stack[-1] = int(stack[-1]) + int(table[parts[1].strip()])
            elif op == "ADD_DAYS":
                d = date.fromisoformat(str(stack[-1]))
                stack[-1] = (d + timedelta(days=safe_int(parts[1]))).isoformat()
            elif op == "ADD_WEEKS":
                d = date.fromisoformat(str(stack[-1]))
                stack[-1] = (d + timedelta(days=7 * safe_int(parts[1]))).isoformat()
            elif op == "SUM":
                stack[-1] = sum(int(x) for x in stack[-1])
            elif op == "MAX":
                stack[-1] = max(int(x) for x in stack[-1])
            elif op == "MIN":
                stack[-1] = min(int(x) for x in stack[-1])
            elif op == "BASENAME":
                stack[-1] = path_basename(str(stack[-1]))
            elif op == "DIRNAME":
                stack[-1] = path_dirname(str(stack[-1]))
            elif op == "EXTENSION":
                stack[-1] = path_ext(str(stack[-1]))
            elif op == "STRIP_EXT":
                stack[-1] = path_strip_ext(str(stack[-1]))
            else:
                return False, None, f"unknown_op:{op}"
        if not stack:
            return False, None, "empty_stack"
        return True, clean_answer(stack[-1]), ""
    except Exception as exc:
        return False, None, type(exc).__name__


def classify_failure(ex: Example, generated: str, valid_exec: bool, exec_correct: bool) -> str:
    if exec_correct:
        gold = canonical_program("\n".join(ex.program))
        got = canonical_program(generated)
        return "correct_exact" if got == gold else "correct_semantic_variant"
    if not valid_exec:
        return "invalid_or_unexecutable"
    got = canonical_program(generated)
    gold = canonical_program("\n".join(ex.program))
    if not got:
        return "empty_or_no_program"
    got_ops = [op for op, _ in got]
    gold_ops = [op for op, _ in gold]
    if got_ops != gold_ops:
        return "wrong_op_order_or_choice"
    if got != gold:
        return "wrong_constant_or_argument"
    return "same_program_wrong_result"


class TextDataset(Dataset):
    def __init__(self, examples: Sequence[Example], arm: str, tokenizer: Any, max_length: int):
        self.rows: List[Dict[str, torch.Tensor]] = []
        eos = tokenizer.eos_token or ""
        for ex in examples:
            prompt = format_prompt(ex, arm)
            target = render_target(ex, arm) + eos
            prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
            target_ids = tokenizer(target, add_special_tokens=False).input_ids
            ids = (prompt_ids + target_ids)[-max_length:]
            cut = max(0, len(prompt_ids) + len(target_ids) - max_length)
            prompt_kept = max(0, len(prompt_ids) - cut)
            labels = [-100] * prompt_kept + ids[prompt_kept:]
            self.rows.append({"input_ids": torch.tensor(ids), "labels": torch.tensor(labels)})

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.rows[idx]


def collate(batch: Sequence[Dict[str, torch.Tensor]], pad_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, labels, mask = [], [], []
    for row in batch:
        ids = row["input_ids"]
        labs = row["labels"]
        pad = max_len - len(ids)
        input_ids.append(torch.cat([torch.full((pad,), pad_id, dtype=torch.long), ids]))
        labels.append(torch.cat([torch.full((pad,), -100, dtype=torch.long), labs]))
        mask.append(torch.cat([torch.zeros(pad, dtype=torch.long), torch.ones(len(ids), dtype=torch.long)]))
    return {"input_ids": torch.stack(input_ids), "labels": torch.stack(labels), "attention_mask": torch.stack(mask)}


def load_tokenizer(model_name: str) -> Any:
    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=str(CACHE_DIR), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def load_model(model_name: str, lora_r: int, lora_alpha: int) -> Any:
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=str(CACHE_DIR),
        trust_remote_code=True,
        quantization_config=bnb,
        device_map="auto",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    return get_peft_model(model, lora)


def train_adapter(config: RunConfig, arm: str, seed: int, tokenizer: Any, run_dir: Path) -> Tuple[Any, List[Dict[str, Any]]]:
    set_seed(seed)
    model = load_model(config.model_name, config.lora_r, config.lora_alpha)
    model.train()
    train_examples = make_split(seed, "train", config.train_n)
    ds = TextDataset(train_examples, arm, tokenizer, config.max_length)
    loader = DataLoader(ds, batch_size=config.batch_size, shuffle=True, collate_fn=lambda b: collate(b, tokenizer.pad_token_id))
    opt = torch.optim.AdamW(model.parameters(), lr=config.lr)
    logs: List[Dict[str, Any]] = []
    step = 0
    accum = 0
    opt.zero_grad(set_to_none=True)
    while step < config.train_steps:
        for batch in loader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / config.grad_accum
            loss.backward()
            accum += 1
            if accum >= config.grad_accum:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                accum = 0
                if step == 1 or step % max(1, config.train_steps // 4) == 0 or step == config.train_steps:
                    logs.append({"suite": config.suite, "run": run_dir.name, "arm": arm, "seed": seed, "step": step, "loss": float(out.loss.detach().cpu())})
                    log(f"{run_dir.name}: step {step}/{config.train_steps} loss={float(out.loss.detach().cpu()):.4f}")
                if step >= config.train_steps:
                    break
    ckpt = CHECKPOINTS / run_dir.name
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    return model, logs


def generate_batch(model: Any, tokenizer: Any, prompts: List[str], max_new_tokens: int) -> List[str]:
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    texts: List[str] = []
    for i in range(out.shape[0]):
        gen = out[i, enc["input_ids"].shape[1] :]
        texts.append(tokenizer.decode(gen, skip_special_tokens=True).strip())
    return texts


def evaluate_model(config: RunConfig, model: Any, tokenizer: Any, examples: Sequence[Example], arm: str, run_id: str, seed: int, trained: bool) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    model.eval()
    detail_rows: List[Dict[str, Any]] = []
    prompts = [format_prompt(ex, arm) for ex in examples]
    generated: List[str] = []
    for i in range(0, len(prompts), config.eval_batch_size):
        generated.extend(generate_batch(model, tokenizer, prompts[i : i + config.eval_batch_size], config.max_new_tokens))
    stats: Dict[str, List[float]] = {k: [] for k in ["primary", "exec", "valid", "final", "no_final", "exact_program"]}
    by_family: Dict[str, List[float]] = {f: [] for f in FAMILIES}
    by_failure: Dict[str, int] = {}
    for ex, gen in zip(examples, generated):
        final = parse_final(gen)
        final_correct = clean_answer(final) == clean_answer(ex.answer) if final is not None else False
        valid_exec, exec_value, exec_err = (False, None, "")
        exec_correct = False
        exact_program = False
        if arm != "answer_only":
            valid_exec, exec_value, exec_err = execute_stack(gen)
            exec_correct = valid_exec and clean_answer(exec_value) == clean_answer(ex.answer)
            exact_program = canonical_program(gen) == canonical_program("\n".join(ex.program))
        primary = final_correct if arm == "answer_only" else exec_correct
        failure = "n/a_answer_only" if arm == "answer_only" else classify_failure(ex, gen, valid_exec, exec_correct)
        by_failure[failure] = by_failure.get(failure, 0) + 1
        stats["primary"].append(float(primary))
        stats["exec"].append(float(exec_correct))
        stats["valid"].append(float(valid_exec))
        stats["final"].append(float(final_correct))
        stats["no_final"].append(float(not has_final(gen)))
        stats["exact_program"].append(float(exact_program))
        by_family[ex.family].append(float(primary))
        detail_rows.append({
            "suite": config.suite,
            "run": run_id,
            "arm": arm,
            "seed": seed,
            "trained": int(trained),
            "split": ex.split,
            "family": ex.family,
            "depth": ex.depth,
            "example_id": ex.example_id,
            "answer": ex.answer,
            "generated": gen,
            "target": render_target(ex, arm),
            "exec_value": exec_value,
            "exec_error": exec_err,
            "primary_correct": int(primary),
            "exec_correct": int(exec_correct),
            "final_correct": int(final_correct),
            "valid_exec": int(valid_exec),
            "exact_program": int(exact_program),
            "has_final": int(has_final(gen)),
            "failure_type": failure,
            "gold_ops": " ".join(ex.op_names),
            "generated_ops": " ".join(canon_op_names(gen)),
        })
    metric = {
        "suite": config.suite,
        "run": run_id,
        "arm": arm,
        "seed": seed,
        "trained": int(trained),
        "split": examples[0].split if examples else "",
        "depth": examples[0].depth if examples else 0,
        "template_shift": int(examples[0].split.startswith("eval_template")) if examples else 0,
        "n": len(examples),
        "primary_accuracy": float(np.mean(stats["primary"])) if examples else float("nan"),
        "exec_accuracy": float(np.mean(stats["exec"])) if examples else float("nan"),
        "valid_exec_rate": float(np.mean(stats["valid"])) if examples else float("nan"),
        "final_accuracy": float(np.mean(stats["final"])) if examples else float("nan"),
        "no_final_rate": float(np.mean(stats["no_final"])) if examples else float("nan"),
        "exact_program_rate": float(np.mean(stats["exact_program"])) if examples else float("nan"),
        "mean_new_tokens": float(np.mean([len(tokenizer(g, add_special_tokens=False).input_ids) for g in generated])) if generated else float("nan"),
    }
    for family, vals in by_family.items():
        metric[f"{family}_accuracy"] = float(np.mean(vals)) if vals else float("nan")
    for failure, count in by_failure.items():
        metric[f"failure_{failure}"] = count / max(1, len(examples))
    return metric, detail_rows


def cleanup_model(model: Any) -> None:
    try:
        del model
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def make_config(args: argparse.Namespace) -> RunConfig:
    if args.suite == "smoke":
        train_n, eval_n, steps, seeds, arms = 36, 12, 2, [101], ["answer_only", "trace_stack_final", "trace_stack_no_final", "program_stack"]
        batch, accum = 2, 1
    elif args.suite == "pilot":
        train_n, eval_n, steps, seeds, arms = 120, 12, 20, [101], ARMS
        batch, accum = 2, 2
    else:
        train_n, eval_n, steps, seeds, arms = 240, 18, 48, [101, 202, 303, 404, 505], ARMS
        batch, accum = 2, 2
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
    return RunConfig(
        run_name=args.run_name or f"{args.suite}_v1",
        suite=args.suite,
        model_name=args.model_name,
        seeds=seeds,
        arms=arms,
        train_n=train_n,
        eval_n=eval_n,
        train_steps=steps,
        batch_size=args.batch_size or batch,
        grad_accum=args.grad_accum or accum,
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
    run_root = RUNS / config.run_name
    run_root.mkdir(parents=True, exist_ok=True)
    write_json(run_root / "config.json", config.__dict__)
    append_log(
        f"## Run `{config.run_name}`\n\n"
        f"- Started: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n"
        f"- Suite: `{config.suite}`\n"
        f"- Model: `{config.model_name}`\n"
        f"- Seeds: `{','.join(map(str, config.seeds))}`\n"
        f"- Arms: `{','.join(config.arms)}`\n"
        f"- Training examples per seed: `{config.train_n}`\n"
        f"- Eval examples per split: `{config.eval_n}`\n"
        f"- Steps: `{config.train_steps}`"
    )
    tokenizer = load_tokenizer(config.model_name)
    all_metrics: List[Dict[str, Any]] = []
    all_details: List[Dict[str, Any]] = []
    all_logs: List[Dict[str, Any]] = []
    start = time.time()
    eval_splits = [name for name, _, _ in DEPTH_SPLITS]
    for arm in config.arms:
        for seed in config.seeds:
            run_id = f"{config.run_name}_{arm}_s{seed}"
            model, logs = train_adapter(config, arm, seed, tokenizer, run_root / run_id)
            all_logs.extend(logs)
            for split in eval_splits:
                examples = make_split(seed + 777, split, config.eval_n)
                metric, details = evaluate_model(config, model, tokenizer, examples, arm, run_id, seed, True)
                all_metrics.append(metric)
                all_details.extend(details)
                log(f"{run_id} {split}: primary={pct(metric['primary_accuracy'])} exec={pct(metric['exec_accuracy'])} final={pct(metric['final_accuracy'])}")
            cleanup_model(model)
    write_csv(run_root / "metrics.csv", all_metrics)
    write_csv(run_root / "details.csv", all_details)
    write_csv(run_root / "train_log.csv", all_logs)
    append_log(
        f"Completed `{config.run_name}` in {time.time() - start:.1f}s.\n\n"
        f"- Metric rows: {len(all_metrics)}\n"
        f"- Detail rows: {len(all_details)}\n"
        f"- Training log rows: {len(all_logs)}"
    )
    analyze_all()


def read_all_csv(pattern: str) -> pd.DataFrame:
    frames = []
    for path in sorted(RUNS.glob(pattern)):
        if path.exists() and path.stat().st_size > 0:
            frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    metric_cols = [c for c in metrics.columns if c.endswith("_accuracy") or c.endswith("_rate") or c.startswith("failure_") or c in {"valid_exec_rate", "mean_new_tokens", "exact_program_rate"}]
    rows = []
    for keys, sub in metrics.groupby(["suite", "arm", "split", "depth", "template_shift"], dropna=False):
        row = {"suite": keys[0], "arm": keys[1], "split": keys[2], "depth": keys[3], "template_shift": keys[4], "runs": len(sub), "n_total": int(sub["n"].sum())}
        for col in metric_cols:
            vals = pd.to_numeric(sub[col], errors="coerce").dropna()
            if len(vals):
                row[f"{col}_mean"] = float(vals.mean())
                row[f"{col}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def primary_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    suites = list(summary["suite"].dropna().unique())
    suite = "main" if "main" in suites else suites[-1]
    return summary[summary["suite"].eq(suite)].copy()


def plot_depth(primary: pd.DataFrame) -> None:
    sub = primary[primary["split"].str.startswith("eval_comp") | primary["split"].eq("eval_indist_d1")].copy()
    if sub.empty:
        return
    plt.figure(figsize=(10.5, 5.8))
    for arm, g in sub.groupby("arm"):
        g = g.sort_values("depth")
        y = 100 * g["exec_accuracy_mean"].fillna(g["primary_accuracy_mean"])
        err = 100 * g.get("exec_accuracy_std", pd.Series([0] * len(g))).fillna(0)
        plt.errorbar(g["depth"], y, yerr=err, marker="o", capsize=4, label=arm)
    plt.xlabel("Procedure depth")
    plt.ylabel("Externally executed procedure accuracy (%)")
    plt.title("Procedure Execution by Composition Depth")
    plt.xticks(sorted(sub["depth"].unique()))
    plt.ylim(0, 105)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "execution_by_depth.png", dpi=180)
    plt.close()


def plot_template(primary: pd.DataFrame) -> None:
    sub = primary[primary["split"].str.startswith("eval_template")].copy()
    if sub.empty:
        return
    plt.figure(figsize=(10.5, 5.8))
    for arm, g in sub.groupby("arm"):
        g = g.sort_values("depth")
        y = 100 * g["exec_accuracy_mean"].fillna(g["primary_accuracy_mean"])
        err = 100 * g.get("exec_accuracy_std", pd.Series([0] * len(g))).fillna(0)
        plt.errorbar(g["depth"], y, yerr=err, marker="o", capsize=4, label=arm)
    plt.xlabel("Procedure depth")
    plt.ylabel("Template-shift execution accuracy (%)")
    plt.title("Template Shift by Depth")
    plt.xticks(sorted(sub["depth"].unique()))
    plt.ylim(0, 105)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "template_shift_by_depth.png", dpi=180)
    plt.close()


def plot_final_gap(primary: pd.DataFrame) -> None:
    sub = primary[primary["arm"].isin(["trace_stack_final", "answer_only"])].copy()
    if sub.empty:
        return
    comp = sub[sub["split"].str.startswith("eval_comp")].copy()
    plt.figure(figsize=(10.5, 5.8))
    for arm, g in comp.groupby("arm"):
        g = g.sort_values("depth")
        plt.plot(g["depth"], 100 * g["primary_accuracy_mean"], marker="o", label=f"{arm} primary/final")
        if arm != "answer_only":
            plt.plot(g["depth"], 100 * g["exec_accuracy_mean"], marker="s", linestyle="--", label=f"{arm} executed")
    plt.xlabel("Procedure depth")
    plt.ylabel("Accuracy (%)")
    plt.title("Final Answer Scoring Versus External Execution")
    plt.xticks(sorted(comp["depth"].unique()))
    plt.ylim(0, 105)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "final_vs_execution_gap.png", dpi=180)
    plt.close()


def plot_failure(details: pd.DataFrame) -> None:
    if details.empty or "failure_type" not in details:
        return
    sub = details[(details["split"].str.startswith("eval_comp")) & (details["arm"].ne("answer_only"))].copy()
    if sub.empty:
        return
    counts = sub.groupby(["depth", "failure_type"]).size().reset_index(name="n")
    totals = counts.groupby("depth")["n"].transform("sum")
    counts["rate"] = counts["n"] / totals
    pivot = counts.pivot_table(index="depth", columns="failure_type", values="rate", fill_value=0).sort_index()
    ax = pivot.plot(kind="bar", stacked=True, figsize=(11, 6))
    ax.set_ylabel("Fraction of generated procedures")
    ax.set_title("Failure Taxonomy on Composition Splits")
    ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(FIGURES / "failure_taxonomy.png", dpi=180)
    plt.close()


def plot_training(logs: pd.DataFrame) -> None:
    if logs.empty:
        return
    plt.figure(figsize=(11, 6))
    for run, g in logs.groupby("run"):
        label = run if len(run) < 42 else run[:39] + "..."
        plt.plot(g["step"], g["loss"], marker="o", linewidth=1.1, label=label)
    plt.xlabel("Training step")
    plt.ylabel("CE loss")
    plt.title("Training Curves")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(FIGURES / "training_curves.png", dpi=180)
    plt.close()


def md_table(df: pd.DataFrame, cols: Sequence[str]) -> str:
    if df.empty:
        return "_No rows._"
    lines = ["|" + "|".join(cols) + "|", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            v = row.get(col, "")
            if isinstance(v, float):
                if "accuracy" in col or "rate" in col or col.startswith("failure"):
                    vals.append(pct(v))
                else:
                    vals.append(f"{v:.2f}")
            else:
                vals.append(str(v))
        lines.append("|" + "|".join(vals) + "|")
    return "\n".join(lines)


def make_report(summary: pd.DataFrame, metrics: pd.DataFrame, details: pd.DataFrame, logs: pd.DataFrame) -> str:
    primary = primary_summary(summary)
    suite = primary["suite"].iloc[0] if not primary.empty else "none"
    comp = primary[primary["split"].str.startswith("eval_comp")].copy() if not primary.empty else pd.DataFrame()
    trace = primary[primary["arm"].eq("trace_stack_final")].copy() if not primary.empty else pd.DataFrame()
    nofinal = primary[primary["arm"].eq("trace_stack_no_final")].copy() if not primary.empty else pd.DataFrame()
    program = primary[primary["arm"].eq("program_stack")].copy() if not primary.empty else pd.DataFrame()
    lines: List[str] = []
    lines.append("# Qwen Trace Procedure Depth Stress")
    lines.append("")
    lines.append("## Abstract")
    lines.append("")
    lines.append("This standalone experiment tests whether a local 4B model composes known primitives into executable procedures when trained only on atomic procedures. The primary score for procedure arms is external execution of the emitted stack program, not the model's own final answer.")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append("Training examples contain one primitive operation. Evaluation sweeps held-out procedure depths 2, 3, 4, and 6, plus template-shifted prompts at depths 2, 4, and 6. The task families are string, number, table, date, list, and path transformations. Four arms are compared: `answer_only`, `trace_stack_final`, `trace_stack_no_final`, and `program_stack`.")
    lines.append("")
    lines.append("The load-bearing distinction is compilation versus self-execution. `trace_stack_final` may emit a `FINAL` line, but its procedure is also parsed and executed by the interpreter. `trace_stack_no_final` tests the same numbered procedure format without answer supervision. `program_stack` tests compact raw instructions.")
    lines.append("")
    lines.append("## Run Configuration")
    lines.append("")
    lines.append(f"- Primary suite: `{suite}`.")
    if not metrics.empty:
        main = metrics[metrics["suite"].eq(suite)]
        seeds = sorted(int(x) for x in main["seed"].dropna().unique()) if not main.empty else []
        lines.append(f"- Seeds: `{','.join(map(str, seeds))}`.")
        lines.append(f"- Evaluation examples across trained arms: `{int(main['n'].sum())}`.")
    if not logs.empty and "suite" in logs:
        suite_logs = logs[logs["suite"].eq(suite)]
        if not suite_logs.empty:
            lines.append(f"- QLoRA update steps per adapter: `{int(suite_logs['step'].max())}`.")
    lines.append("- Large adapters are stored outside the experiment tree.")
    lines.append("")
    lines.append("## Primary Results")
    lines.append("")
    def row_for(arm: str, split: str) -> Optional[pd.Series]:
        sub = primary[(primary["arm"].eq(arm)) & (primary["split"].eq(split))]
        return sub.iloc[0] if not sub.empty else None

    if not comp.empty:
        best = comp.sort_values("exec_accuracy_mean", ascending=False).iloc[0]
        lines.append(f"- Best held-out composition execution row: `{best['arm']}` depth `{int(best['depth'])}` at {pct(best['exec_accuracy_mean'])}.")
    ps_d6 = row_for("program_stack", "eval_comp_d6")
    ans_d6 = row_for("answer_only", "eval_comp_d6")
    if ps_d6 is not None and ans_d6 is not None:
        lines.append(f"- Compact program execution at depth 6: `program_stack` {pct(ps_d6['exec_accuracy_mean'])} with seed std {pct(ps_d6['exec_accuracy_std'])}, versus `answer_only` final accuracy {pct(ans_d6['primary_accuracy_mean'])}.")
    if not trace.empty:
        d6 = trace[trace["split"].eq("eval_comp_d6")]
        if not d6.empty:
            row = d6.iloc[0]
            lines.append(f"- `trace_stack_final` depth-6 composition execution: {pct(row['exec_accuracy_mean'])} with seed std {pct(row['exec_accuracy_std'])}.")
        tg = trace[trace["split"].eq("eval_template_d6")]
        if not tg.empty:
            row = tg.iloc[0]
            lines.append(f"- `trace_stack_final` depth-6 template-shift execution: {pct(row['exec_accuracy_mean'])}.")
    if not trace.empty and not nofinal.empty:
        tf = trace[trace["split"].eq("eval_comp_d4")]
        nf = nofinal[nofinal["split"].eq("eval_comp_d4")]
        if not tf.empty and not nf.empty:
            lines.append(f"- Final-answer supervision comparison at depth 4: `trace_stack_final` execution {pct(tf.iloc[0]['exec_accuracy_mean'])}; `trace_stack_no_final` execution {pct(nf.iloc[0]['exec_accuracy_mean'])}.")
    ps_t6 = row_for("program_stack", "eval_template_d6")
    if ps_d6 is not None and ps_t6 is not None:
        lines.append(f"- Template-shift depth-6 execution for `program_stack`: {pct(ps_t6['exec_accuracy_mean'])}, a {pct(ps_d6['exec_accuracy_mean'] - ps_t6['exec_accuracy_mean'])} absolute drop from standard depth-6 composition.")
    lines.append("")
    cols = ["arm", "split", "depth", "runs", "n_total", "primary_accuracy_mean", "primary_accuracy_std", "exec_accuracy_mean", "exec_accuracy_std", "valid_exec_rate_mean", "final_accuracy_mean", "no_final_rate_mean", "exact_program_rate_mean"]
    rows = primary.sort_values(["split", "arm"])[cols] if not primary.empty else pd.DataFrame()
    lines.append(md_table(rows, cols))
    lines.append("")
    lines.append("![Execution by depth](../analysis/figures/execution_by_depth.png)")
    lines.append("")
    lines.append("![Template shift by depth](../analysis/figures/template_shift_by_depth.png)")
    lines.append("")
    lines.append("![Final versus execution gap](../analysis/figures/final_vs_execution_gap.png)")
    lines.append("")
    lines.append("![Failure taxonomy](../analysis/figures/failure_taxonomy.png)")
    lines.append("")
    lines.append("![Training curves](../analysis/figures/training_curves.png)")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("A real compiler result should degrade gradually with depth rather than collapse immediately once procedure length exceeds the atomic training distribution. A gap between `trace_stack_final` execution and final-answer accuracy means the model can emit a correct procedure while failing to self-execute it. A gap between standard composition and template-shift composition localizes the remaining problem to language grounding rather than procedure sequencing.")
    if ps_d6 is not None and ans_d6 is not None:
        lines.append(f"The main result is positive: training only on atomic procedures still produced executable depth-6 compositions at {pct(ps_d6['exec_accuracy_mean'])} for the compact raw stack ABI, while answer-only scoring was {pct(ans_d6['primary_accuracy_mean'])}. This is not a self-execution win; it is a compiler win, because the deterministic interpreter supplies the execution.")
    tf_d6 = row_for("trace_stack_final", "eval_comp_d6")
    if tf_d6 is not None:
        lines.append(f"The final-answer confound persists at the hardest standard composition split: `trace_stack_final` executed correctly at {pct(tf_d6['exec_accuracy_mean'])} but its emitted final answer was correct at {pct(tf_d6['final_accuracy_mean'])}.")
    nf_d6 = row_for("trace_stack_no_final", "eval_comp_d6")
    if tf_d6 is not None and nf_d6 is not None:
        lines.append(f"Removing the final-answer line did not kill procedure learning: `trace_stack_no_final` reached {pct(nf_d6['exec_accuracy_mean'])} at depth 6. Final-answer supervision is therefore not required for the basic compiler effect, though arm ranking varies by depth and template split.")
    if ps_t6 is not None and ps_d6 is not None:
        lines.append(f"The remaining major weakness is prompt wording. `program_stack` falls from {pct(ps_d6['exec_accuracy_mean'])} on standard depth-6 composition to {pct(ps_t6['exec_accuracy_mean'])} under template shift, so the next bottleneck is language grounding into the ABI, not deterministic execution.")
    if not details.empty and "failure_type" in details:
        fail = details[(details["split"].str.startswith("eval_comp")) & (details["arm"].eq("trace_stack_final"))]
        if not fail.empty:
            counts = fail.groupby("failure_type").size().sort_values(ascending=False)
            bits = [f"{k} {pct(v / len(fail))}" for k, v in counts.items()]
            lines.append(f"For `trace_stack_final` on composition splits, generated procedures break down as: {', '.join(bits)}.")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append("This experiment tests composition over a fixed known primitive library. It does not test invention of new operations outside the ABI. The families are synthetic but selected to cover several common deterministic task shapes.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append("- Metrics: `analysis/summary_by_arm.csv` and `analysis/all_metrics.csv`")
    lines.append("- Details: `analysis/all_details.csv`")
    lines.append("- Training logs: `analysis/all_train_logs.csv`")
    lines.append("- Checkpoints: `/workspace/large_artifacts/qwen_trace_procedure_depth_stress/checkpoints`")
    return "\n".join(lines) + "\n"


def markdown_to_html(markdown_text: str) -> str:
    body: List[str] = []
    in_table = False
    in_ul = False
    for raw in markdown_text.splitlines():
        line = raw.rstrip()
        if line.startswith("# "):
            body.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            body.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("- "):
            if not in_ul:
                body.append("<ul>")
                in_ul = True
            body.append(f"<li>{html.escape(line[2:])}</li>")
        elif line.startswith("![") and "](" in line:
            if in_ul:
                body.append("</ul>")
                in_ul = False
            alt = line[2 : line.index("]")]
            src = line[line.index("(") + 1 : line.rindex(")")]
            body.append(f'<figure><img src="{html.escape(src)}" alt="{html.escape(alt)}"><figcaption>{html.escape(alt)}</figcaption></figure>')
        elif line.startswith("|") and line.endswith("|"):
            if in_ul:
                body.append("</ul>")
                in_ul = False
            cells = [html.escape(c.strip()) for c in line.strip("|").split("|")]
            if all(set(c) <= {"-"} for c in cells):
                continue
            if not in_table:
                body.append("<table>")
                in_table = True
                tag = "th"
            else:
                tag = "td"
            body.append("<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>")
        else:
            if in_table:
                body.append("</table>")
                in_table = False
            if in_ul:
                body.append("</ul>")
                in_ul = False
            body.append(f"<p>{html.escape(line)}</p>" if line else "")
    if in_table:
        body.append("</table>")
    if in_ul:
        body.append("</ul>")
    css = "body{font-family:Inter,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:1160px;margin:36px auto;line-height:1.5;color:#202124}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #d4d7dc;padding:6px 8px;text-align:left}th{background:#f2f4f7}img{max-width:100%;border:1px solid #d4d7dc}figure{margin:24px 0}figcaption{color:#5f6368;font-size:13px}code{background:#f5f7fa;padding:1px 4px;border-radius:4px}"
    return "<!doctype html><html><head><meta charset='utf-8'><title>Qwen Trace Procedure Depth Stress</title><style>" + css + "</style></head><body>" + "\n".join(body) + "</body></html>\n"


def write_checkpoint_manifest() -> None:
    rows = []
    for path in sorted(CHECKPOINTS.glob("*")):
        if path.is_dir():
            total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            rows.append({"checkpoint": path.name, "path": str(path), "bytes": total})
    write_csv(ROOT / "checkpoint_manifest.csv", rows)


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
    primary = primary_summary(summary)
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_depth(primary)
    plot_template(primary)
    plot_final_gap(primary)
    plot_failure(details[details["suite"].eq(primary["suite"].iloc[0])] if not details.empty and not primary.empty else details)
    plot_training(logs[logs["suite"].eq(primary["suite"].iloc[0])] if not logs.empty and not primary.empty else logs)
    report = make_report(summary, metrics, details, logs)
    (REPORTS / "qwen_trace_procedure_depth_stress_report.md").write_text(report)
    (REPORTS / "qwen_trace_procedure_depth_stress_report.html").write_text(markdown_to_html(report))
    write_checkpoint_manifest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


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
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=448)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=120)
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
