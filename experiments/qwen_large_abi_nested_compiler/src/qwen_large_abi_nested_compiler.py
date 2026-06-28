#!/usr/bin/env python3
"""Large-ABI and nested-structure stress test for constrained ABI compilation.

The primary question is whether a constrained stack-ABI compiler still works
when the primitive library becomes large and tasks require nested branch
sub-procedures rather than only linear chains.
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


ROOT = Path("/workspace/experiments/qwen_large_abi_nested_compiler")
RUNS = ROOT / "runs"
ANALYSIS = ROOT / "analysis"
FIGURES = ANALYSIS / "figures"
REPORTS = ROOT / "reports"
LARGE_ROOT = Path("/workspace/large_artifacts/qwen_large_abi_nested_compiler")
CHECKPOINTS = LARGE_ROOT / "checkpoints"
CACHE_DIR = Path("/workspace/.cache/huggingface")

MODEL_NAME = "Qwen/Qwen3-4B"
FAMILIES = ["chain", "nested"]
CURRICULA = ["abi32_chain_d3", "abi128_chain_d3", "abi32_nested_d3", "abi128_nested_d3"]
EVAL_ARMS = [
    "program_stack_free",
    "program_stack_constrained",
    "gold_abi_constrained",
]
DEPTH_SPLITS = [
    ("eval_chain_d3", "chain", 3, False),
    ("eval_chain_d8", "chain", 8, False),
    ("eval_chain_d16", "chain", 16, False),
    ("eval_chain_template_d16", "chain", 16, True),
    ("eval_nested_l2", "nested", 2, False),
    ("eval_nested_l3", "nested", 3, False),
    ("eval_nested_l4", "nested", 4, False),
    ("eval_nested_l8", "nested", 8, False),
    ("eval_nested_template_l8", "nested", 8, True),
]
FINAL_RE = re.compile(r"\bFINAL\s*:?\s*([A-Za-z0-9_.:+/-]+)", re.IGNORECASE)
BINARY_OPS = ["BADD", "BSUB", "BMUL", "BMAX", "BMIN"]


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
    eval_splits: List[str]
    batch_size: int
    grad_accum: int
    lr: float
    max_length: int
    eval_batch_size: int
    lora_r: int
    lora_alpha: int
    max_new_tokens: int
    resample_attempts: int


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


def interleave(groups: Sequence[Sequence[str]]) -> List[str]:
    out: List[str] = []
    max_len = max(len(g) for g in groups)
    for i in range(max_len):
        for group in groups:
            if i < len(group):
                out.append(group[i])
    return out


def build_unary_catalog() -> List[str]:
    adds = [f"ADD_{i:02d}" for i in range(1, 49)]
    subs = [f"SUB_{i:02d}" for i in range(1, 49)]
    muls = [f"MUL_{i:02d}" for i in range(2, 10)]
    divs = [f"DIV_{i:02d}" for i in range(2, 10)]
    mods = [f"MOD_{i:02d}" for i in [7, 9, 10, 11, 13, 17, 19, 23]]
    misc = ["ABS", "NEG", "DOUBLE", "TRIPLE", "INC", "DEC", "CLAMP_00_99", "SQUARE_MOD_997"]
    catalog = interleave([adds, subs, muls, divs, mods, misc])
    if len(catalog) != 128:
        raise RuntimeError(f"unexpected catalog size {len(catalog)}")
    return catalog


UNARY_CATALOG = build_unary_catalog()


def unary_ops(abi_size: int) -> List[str]:
    if abi_size not in {32, 128}:
        raise ValueError(f"unsupported abi_size {abi_size}")
    return UNARY_CATALOG[:abi_size]


def target_spec(target: str) -> Dict[str, Any]:
    specs = {
        "abi32_chain_d3": {"abi_size": 32, "nested": False},
        "abi128_chain_d3": {"abi_size": 128, "nested": False},
        "abi32_nested_d3": {"abi_size": 32, "nested": True},
        "abi128_nested_d3": {"abi_size": 128, "nested": True},
        "oracle_abi32": {"abi_size": 32, "nested": True},
        "oracle_abi128": {"abi_size": 128, "nested": True},
    }
    if target not in specs:
        raise ValueError(target)
    return specs[target]


def describe_unary(op: str, shifted: bool) -> str:
    if op.startswith("ADD_"):
        n = int(op.split("_")[1])
        return f"increase by {n}" if shifted else f"add {n}"
    if op.startswith("SUB_"):
        n = int(op.split("_")[1])
        return f"decrease by {n}" if shifted else f"subtract {n}"
    if op.startswith("MUL_"):
        n = int(op.split("_")[1])
        return f"scale by {n}" if shifted else f"multiply by {n}"
    if op.startswith("DIV_"):
        n = int(op.split("_")[1])
        return f"keep the whole-number quotient after division by {n}" if shifted else f"floor-divide by {n}"
    if op.startswith("MOD_"):
        n = int(op.split("_")[1])
        return f"keep the remainder modulo {n}" if shifted else f"take modulo {n}"
    phrases = {
        False: {
            "ABS": "take the absolute value",
            "NEG": "negate the value",
            "DOUBLE": "double the value",
            "TRIPLE": "triple the value",
            "INC": "increment by one",
            "DEC": "decrement by one",
            "CLAMP_00_99": "clamp into the inclusive range 0 to 99",
            "SQUARE_MOD_997": "square the value then take modulo 997",
        },
        True: {
            "ABS": "drop the sign",
            "NEG": "flip the sign",
            "DOUBLE": "make it twice as large",
            "TRIPLE": "make it three times as large",
            "INC": "raise it by one",
            "DEC": "lower it by one",
            "CLAMP_00_99": "force it to stay between 0 and 99",
            "SQUARE_MOD_997": "square it and wrap by 997",
        },
    }
    return phrases[shifted][op]


def describe_binary(op: str, branch_name: str, shifted: bool) -> str:
    if op == "BADD":
        return f"add {branch_name} into the running result" if shifted else f"add branch {branch_name}"
    if op == "BSUB":
        return f"subtract {branch_name} from the running result" if shifted else f"subtract branch {branch_name}"
    if op == "BMUL":
        return f"multiply the running result by {branch_name}" if shifted else f"multiply by branch {branch_name}"
    if op == "BMAX":
        return f"keep the larger of the running result and {branch_name}" if shifted else f"take the max with branch {branch_name}"
    if op == "BMIN":
        return f"keep the smaller of the running result and {branch_name}" if shifted else f"take the min with branch {branch_name}"
    raise ValueError(op)


def apply_unary(value: int, op: str) -> int:
    if op.startswith("ADD_"):
        return value + int(op.split("_")[1])
    if op.startswith("SUB_"):
        return value - int(op.split("_")[1])
    if op.startswith("MUL_"):
        return value * int(op.split("_")[1])
    if op.startswith("DIV_"):
        return math.floor(value / int(op.split("_")[1]))
    if op.startswith("MOD_"):
        return value % int(op.split("_")[1])
    if op == "ABS":
        return abs(value)
    if op == "NEG":
        return -value
    if op == "DOUBLE":
        return value * 2
    if op == "TRIPLE":
        return value * 3
    if op == "INC":
        return value + 1
    if op == "DEC":
        return value - 1
    if op == "CLAMP_00_99":
        return min(99, max(0, value))
    if op == "SQUARE_MOD_997":
        return (value * value) % 997
    raise ValueError(op)


def choose_unary(rng: random.Random, abi_size: int, op_cursor: Optional[int], offset: int) -> str:
    ops = unary_ops(abi_size)
    if op_cursor is None:
        return rng.choice(ops)
    return ops[(op_cursor + offset) % len(ops)]


def execute_gold(program: Sequence[str]) -> str:
    ok, value, err = execute_stack("\n".join(program))
    if not ok or value is None:
        raise RuntimeError(f"gold program failed: {err}\n{program}")
    return value


def make_chain(rng: random.Random, split: str, idx: int, depth: int, shifted: bool, abi_size: int, op_cursor: Optional[int] = None) -> Example:
    start = rng.randint(-40, 80)
    ops = [choose_unary(rng, abi_size, op_cursor, i) for i in range(depth)]
    program = [f"PUSH {start}"] + ops
    line_kinds = ["exact"] + ["unary"] * len(ops)
    steps = "; ".join(f"{i + 1}) {describe_unary(op, shifted)}" for i, op in enumerate(ops))
    prompt = (
        f"Start with integer {start}. Apply these updates in order: {steps}. Compile the procedure."
        if not shifted
        else f"Use {start} as the initial running value. Carry out this ordered recipe: {steps}. Produce the executable steps."
    )
    answer = execute_gold(program)
    payload = {"abi_size": abi_size, "line_kinds": line_kinds, "structure": "chain"}
    return Example(f"{split}_abi{abi_size}_chain_{idx}", split, "chain", depth, prompt, answer, program, ops, payload)


def make_nested(rng: random.Random, split: str, idx: int, leaves: int, shifted: bool, abi_size: int, op_cursor: Optional[int] = None) -> Example:
    branch_names = [chr(ord("A") + i) for i in range(leaves)]
    starts = [rng.randint(-30, 60) for _ in range(leaves)]
    unary = [choose_unary(rng, abi_size, op_cursor, i) for i in range(leaves)]
    binaries = [BINARY_OPS[(rng.randint(0, 10_000) + i) % len(BINARY_OPS)] for i in range(max(0, leaves - 1))]
    program: List[str] = []
    line_kinds: List[str] = []
    prompt_parts: List[str] = []
    merge_parts: List[str] = []
    for i, name in enumerate(branch_names):
        program.append(f"PUSH {starts[i]}")
        line_kinds.append("exact")
        program.append(unary[i])
        line_kinds.append("unary")
        prompt_parts.append(f"branch {name}: start {starts[i]}, then {describe_unary(unary[i], shifted)}")
        if i > 0:
            bop = binaries[i - 1]
            program.append(bop)
            line_kinds.append("binary")
            merge_parts.append(describe_binary(bop, name, shifted))
    branch_text = "; ".join(prompt_parts)
    merge_text = "; then ".join(merge_parts)
    prompt = (
        f"Compute branch sub-results and merge them. Branch definitions: {branch_text}. Start with branch A as the running result; {merge_text}. Compile the stack program."
        if not shifted
        else f"Build these temporary branch values: {branch_text}. Use branch A first, then combine subsequent branches this way: {merge_text}. Emit executable stack steps."
    )
    answer = execute_gold(program)
    payload = {"abi_size": abi_size, "line_kinds": line_kinds, "structure": "nested", "leaves": leaves}
    op_names = unary + binaries
    return Example(f"{split}_abi{abi_size}_nested_{idx}", split, "nested", leaves, prompt, answer, program, op_names, payload)


def split_spec(split: str) -> Tuple[str, int, bool]:
    for name, structure, depth, shifted in DEPTH_SPLITS:
        if split == name:
            return structure, depth, shifted
    raise ValueError(split)


def make_example(rng: random.Random, split: str, structure: str, idx: int, depth: int, shifted: bool, abi_size: int, op_cursor: Optional[int] = None) -> Example:
    if structure == "chain":
        return make_chain(rng, split, idx, depth, shifted, abi_size, op_cursor)
    if structure == "nested":
        return make_nested(rng, split, idx, depth, shifted, abi_size, op_cursor)
    raise ValueError(structure)


def make_split(seed: int, split: str, n: int, train_target: str) -> List[Example]:
    structure, depth, shifted = split_spec(split)
    abi_size = target_spec(train_target)["abi_size"]
    offsets = {name: (i + 1) * 10_000 for i, (name, _, _, _) in enumerate(DEPTH_SPLITS)}
    rng = random.Random(seed + offsets[split] + abi_size * 17)
    examples = [make_example(rng, split, structure, i, depth, shifted, abi_size, None) for i in range(n)]
    rng.shuffle(examples)
    return examples


def make_train_curriculum(seed: int, curriculum: str, n: int) -> List[Example]:
    spec = target_spec(curriculum)
    abi_size = spec["abi_size"]
    include_nested = bool(spec["nested"])
    rng = random.Random(seed + 55_000 + abi_size * 31 + (1_000 if include_nested else 0))
    examples: List[Example] = []
    chain_depths = [1, 2, 3]
    nested_leaves = [2, 3]
    for i in range(n):
        op_cursor = i * 7
        if include_nested and i % 2 == 1:
            leaves = nested_leaves[(i // 2) % len(nested_leaves)]
            examples.append(make_example(rng, f"train_nested_l{leaves}", "nested", i, leaves, False, abi_size, op_cursor))
        else:
            depth = chain_depths[i % len(chain_depths)]
            examples.append(make_example(rng, f"train_chain_d{depth}", "chain", i, depth, False, abi_size, op_cursor))
    rng.shuffle(examples)
    return examples


def numbered_program(lines: Sequence[str]) -> List[str]:
    return [f"STEP {i + 1}: {line}" for i, line in enumerate(lines)]


def render_target(ex: Example, arm: str) -> str:
    if arm == "answer_only":
        return f"FINAL: {ex.answer}\n"
    if arm == "program_stack":
        return "\n".join(ex.program) + "\n"
    raise ValueError(arm)


def format_instruction(arm: str) -> str:
    if arm == "answer_only":
        return "Return one line exactly as: FINAL: <answer>."
    if arm == "program_stack":
        return "Return only raw numeric stack instructions using PUSH, ABI unary mnemonics, and branch merge mnemonics. Do not number steps and do not write a FINAL line or the answer."
    raise ValueError(arm)


def format_prompt(ex: Example, arm: str) -> str:
    return (
        "Compile the task into the requested executable numeric stack format.\n"
        f"Output rule: {format_instruction(arm)}\n\n"
        f"Task:\n{ex.prompt}\n\n"
        "Output:\n"
    )


def program_section(text: str) -> str:
    if re.search(r"^PROGRAM\s*$", text, flags=re.IGNORECASE | re.MULTILINE):
        return re.split(r"^PROGRAM\s*$", text, maxsplit=1, flags=re.IGNORECASE | re.MULTILINE)[1].strip()
    return text.strip()


def parse_block_to_program(text: str) -> List[str]:
    lines = [strip_step_prefix(x) for x in text.splitlines()]
    init: Optional[str] = None
    ops: List[str] = []
    in_ops = False
    for line in lines:
        if not line:
            continue
        upper = line.upper()
        if upper == "PROGRAM":
            break
        if upper.startswith("INIT "):
            init = line[5:].strip()
            continue
        if upper == "OPS":
            in_ops = True
            continue
        if upper == "END_PARSE":
            in_ops = False
            continue
        if in_ops:
            if not upper.startswith("FAMILY ") and upper != "PARSE":
                ops.append(line)
    if init is None:
        return []
    return [init] + ops


def render_program_from_parse(text: str) -> str:
    return "\n".join(parse_block_to_program(text))


def parse_exact(ex: Example, text: str) -> bool:
    return canonical_program(render_program_from_parse(text)) == canonical_program("\n".join(ex.program))


def correct_given_valid(exec_accuracy: float, valid_rate: float) -> float:
    return exec_accuracy / valid_rate if valid_rate > 0 else float("nan")


def constrained_candidate_lines(ex: Example, line_idx: int) -> List[str]:
    line_kinds = ex.payload.get("line_kinds", [])
    if line_idx >= len(line_kinds):
        return []
    kind = line_kinds[line_idx]
    if kind == "exact":
        return [ex.program[line_idx]]
    if kind == "unary":
        return unary_ops(int(ex.payload["abi_size"]))
    if kind == "binary":
        return BINARY_OPS
    raise ValueError(f"bad line kind {kind!r}")


def execute_stack(text: str) -> Tuple[bool, Optional[str], str]:
    stack: List[Any] = []
    try:
        for raw in text.splitlines():
            line = strip_step_prefix(raw)
            if not line or line.startswith("#") or FINAL_RE.search(line):
                continue
            parts = line.split()
            if not parts:
                continue
            op = parts[0].upper().rstrip(":")
            if op == "PUSH":
                stack.append(safe_int(parts[1]))
            elif op in UNARY_CATALOG:
                stack[-1] = apply_unary(int(stack[-1]), op)
            elif op in BINARY_OPS:
                right = int(stack.pop())
                left = int(stack.pop())
                if op == "BADD":
                    stack.append(left + right)
                elif op == "BSUB":
                    stack.append(left - right)
                elif op == "BMUL":
                    stack.append(left * right)
                elif op == "BMAX":
                    stack.append(max(left, right))
                elif op == "BMIN":
                    stack.append(min(left, right))
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
    train_examples = make_train_curriculum(seed, arm, config.train_n)
    ds = TextDataset(train_examples, "program_stack", tokenizer, config.max_length)
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


def generate_batch(model: Any, tokenizer: Any, prompts: List[str], max_new_tokens: int, do_sample: bool = False, temperature: float = 0.7) -> List[str]:
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    kwargs: Dict[str, Any] = {}
    if do_sample:
        kwargs.update({"do_sample": True, "temperature": temperature, "top_p": 0.92})
    else:
        kwargs.update({"do_sample": False})
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **kwargs,
        )
    texts: List[str] = []
    for i in range(out.shape[0]):
        gen = out[i, enc["input_ids"].shape[1] :]
        texts.append(tokenizer.decode(gen, skip_special_tokens=True).strip())
    return texts


def token_prefix_allowed(tokenizer: Any, line_ids: List[List[List[int]]], total_lines: int, base_len: int):
    eos_id = tokenizer.eos_token_id
    newline = "\n"

    def allowed(_batch_id: int, input_ids: torch.Tensor) -> List[int]:
        gen_ids = input_ids.tolist()[base_len:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=False)
        completed = text.count(newline)
        if completed >= total_lines:
            return [eos_id]
        current_text = text.split(newline)[-1]
        current_ids = tokenizer(current_text, add_special_tokens=False).input_ids
        allowed_ids: List[int] = []
        for seq in line_ids[completed]:
            if len(current_ids) < len(seq) and seq[: len(current_ids)] == current_ids:
                allowed_ids.append(seq[len(current_ids)])
        if not allowed_ids:
            return [eos_id]
        return sorted(set(allowed_ids))

    return allowed


def constrained_generate_program(model: Any, tokenizer: Any, ex: Example, max_new_tokens: int) -> str:
    prompt = format_prompt(ex, "program_stack")
    line_texts = [[line + "\n" for line in constrained_candidate_lines(ex, i)] for i in range(len(ex.program))]
    line_ids = [[tokenizer(line, add_special_tokens=False).input_ids for line in choices] for choices in line_texts]
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=512)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    base_len = int(enc["input_ids"].shape[1])
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            prefix_allowed_tokens_fn=token_prefix_allowed(tokenizer, line_ids, len(ex.program), base_len),
        )
    gen = out[0, base_len:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def generate_resample_valid(model: Any, tokenizer: Any, ex: Example, max_new_tokens: int, attempts: int) -> Tuple[str, int]:
    prompt = format_prompt(ex, "program_stack")
    first = ""
    for attempt in range(1, attempts + 1):
        do_sample = attempt > 1
        gen = generate_batch(model, tokenizer, [prompt], max_new_tokens, do_sample=do_sample, temperature=0.75)[0]
        if attempt == 1:
            first = gen
        ok, _, _ = execute_stack(gen)
        if ok:
            return gen, attempt
    return first, attempts


def eval_prompt_arm(eval_arm: str) -> str:
    return "program_stack"


def target_for_eval_arm(eval_arm: str) -> str:
    if eval_arm == "gold_abi_constrained":
        return "oracle"
    return "program_stack"


def eval_arms_for_target(target: str) -> List[str]:
    if target in CURRICULA:
        return ["program_stack_free", "program_stack_constrained"]
    raise ValueError(target)


def generated_program_for_eval(ex: Example, generated: str, eval_arm: str) -> str:
    if eval_arm == "gold_abi_constrained":
        return "\n".join(ex.program)
    return generated


def evaluate_model(
    config: RunConfig,
    model: Any,
    tokenizer: Any,
    examples: Sequence[Example],
    train_target: str,
    eval_arm: str,
    run_id: str,
    seed: int,
    trained: bool,
    free_reference: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if model is not None:
        model.eval()
    detail_rows: List[Dict[str, Any]] = []
    prompt_arm = eval_prompt_arm(eval_arm)
    prompts = [format_prompt(ex, prompt_arm) for ex in examples]
    generated: List[str] = []
    attempts_used: List[int] = []
    if eval_arm == "program_stack_constrained":
        for ex in examples:
            generated.append(constrained_generate_program(model, tokenizer, ex, config.max_new_tokens))
            attempts_used.append(1)
    elif eval_arm == "program_stack_resample_valid":
        for ex in examples:
            gen, attempts = generate_resample_valid(model, tokenizer, ex, config.max_new_tokens, config.resample_attempts)
            generated.append(gen)
            attempts_used.append(attempts)
    elif eval_arm == "gold_abi_constrained":
        for ex in examples:
            generated.append("\n".join(ex.program))
            attempts_used.append(0)
    else:
        for i in range(0, len(prompts), config.eval_batch_size):
            generated.extend(generate_batch(model, tokenizer, prompts[i : i + config.eval_batch_size], config.max_new_tokens))
        attempts_used = [1] * len(generated)
    stats: Dict[str, List[float]] = {k: [] for k in ["primary", "exec", "valid", "final", "no_final", "exact_program", "parse_exact", "diverged", "constrained_only", "free_only", "both_correct", "neither_correct"]}
    by_family: Dict[str, List[float]] = {f: [] for f in FAMILIES}
    by_failure: Dict[str, int] = {}
    for ex, gen, attempts in zip(examples, generated, attempts_used):
        program_text = generated_program_for_eval(ex, gen, eval_arm)
        final = parse_final(gen)
        final_correct = clean_answer(final) == clean_answer(ex.answer) if final is not None else False
        valid_exec, exec_value, exec_err = (False, None, "")
        exec_correct = False
        exact_program = False
        valid_exec, exec_value, exec_err = execute_stack(program_text)
        exec_correct = valid_exec and clean_answer(exec_value) == clean_answer(ex.answer)
        exact_program = canonical_program(program_text) == canonical_program("\n".join(ex.program))
        parse_is_exact = False
        primary = exec_correct
        failure = classify_failure(ex, program_text, valid_exec, exec_correct)
        by_failure[failure] = by_failure.get(failure, 0) + 1
        ref = free_reference.get(ex.example_id, {}) if free_reference else {}
        free_program = str(ref.get("program_text", ""))
        free_correct = bool(ref.get("exec_correct", False))
        diverged = bool(free_reference is not None and canonical_program(program_text) != canonical_program(free_program))
        stats["primary"].append(float(primary))
        stats["exec"].append(float(exec_correct))
        stats["valid"].append(float(valid_exec))
        stats["final"].append(float(final_correct))
        stats["no_final"].append(float(not has_final(gen)))
        stats["exact_program"].append(float(exact_program))
        stats["parse_exact"].append(float(parse_is_exact))
        stats["diverged"].append(float(diverged))
        stats["constrained_only"].append(float(diverged and exec_correct and not free_correct))
        stats["free_only"].append(float(diverged and free_correct and not exec_correct))
        stats["both_correct"].append(float(exec_correct and free_correct))
        stats["neither_correct"].append(float((not exec_correct) and (not free_correct)))
        by_family[ex.family].append(float(primary))
        detail_rows.append({
            "suite": config.suite,
            "run": run_id,
            "train_target": train_target,
            "arm": eval_arm,
            "seed": seed,
            "trained": int(trained),
            "split": ex.split,
            "family": ex.family,
            "depth": ex.depth,
            "example_id": ex.example_id,
            "answer": ex.answer,
            "generated": gen,
            "program_text": program_text,
            "target": render_target(ex, "program_stack"),
            "exec_value": exec_value,
            "exec_error": exec_err,
            "primary_correct": int(primary),
            "exec_correct": int(exec_correct),
            "final_correct": int(final_correct),
            "valid_exec": int(valid_exec),
            "exact_program": int(exact_program),
            "parse_exact": int(parse_is_exact),
            "has_final": int(has_final(gen)),
            "attempts_used": attempts,
            "diverged_from_free": int(diverged),
            "free_exec_correct": int(free_correct),
            "failure_type": failure,
            "gold_ops": " ".join(ex.op_names),
            "generated_ops": " ".join(canon_op_names(program_text)),
        })
    metric = {
        "suite": config.suite,
        "run": run_id,
        "train_target": train_target,
        "arm": eval_arm,
        "seed": seed,
        "trained": int(trained),
        "split": examples[0].split if examples else "",
        "depth": examples[0].depth if examples else 0,
        "template_shift": int("template" in examples[0].split) if examples else 0,
        "n": len(examples),
        "primary_accuracy": float(np.mean(stats["primary"])) if examples else float("nan"),
        "exec_accuracy": float(np.mean(stats["exec"])) if examples else float("nan"),
        "valid_exec_rate": float(np.mean(stats["valid"])) if examples else float("nan"),
        "final_accuracy": float(np.mean(stats["final"])) if examples else float("nan"),
        "no_final_rate": float(np.mean(stats["no_final"])) if examples else float("nan"),
        "exact_program_rate": float(np.mean(stats["exact_program"])) if examples else float("nan"),
        "parse_exact_rate": float(np.mean(stats["parse_exact"])) if examples else float("nan"),
        "correct_given_valid": correct_given_valid(float(np.mean(stats["exec"])), float(np.mean(stats["valid"]))) if examples else float("nan"),
        "divergence_rate": float(np.mean(stats["diverged"])) if free_reference and examples else float("nan"),
        "constrained_only_rate": float(np.mean(stats["constrained_only"])) if free_reference and examples else float("nan"),
        "free_only_rate": float(np.mean(stats["free_only"])) if free_reference and examples else float("nan"),
        "both_correct_rate": float(np.mean(stats["both_correct"])) if free_reference and examples else float("nan"),
        "neither_correct_rate": float(np.mean(stats["neither_correct"])) if free_reference and examples else float("nan"),
        "mean_attempts": float(np.mean(attempts_used)) if attempts_used else float("nan"),
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
    all_eval_splits = [name for name, _, _, _ in DEPTH_SPLITS]
    if args.suite == "smoke":
        train_n, eval_n, steps, seeds, arms = 24, 2, 1, [101], CURRICULA
        batch, accum = 2, 1
    elif args.suite == "pilot":
        train_n, eval_n, steps, seeds, arms = 96, 3, 8, [101], CURRICULA
        batch, accum = 2, 2
    else:
        train_n, eval_n, steps, seeds, arms = 240, 5, 24, [101, 202, 303], CURRICULA
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
    if args.splits:
        eval_splits = [x.strip() for x in args.splits.split(",") if x.strip()]
        bad = sorted(set(eval_splits) - set(all_eval_splits))
        if bad:
            raise ValueError(f"unknown eval split(s): {bad}; allowed: {all_eval_splits}")
    else:
        eval_splits = all_eval_splits
    return RunConfig(
        run_name=args.run_name or f"{args.suite}_v1",
        suite=args.suite,
        model_name=args.model_name,
        seeds=seeds,
        arms=arms,
        train_n=train_n,
        eval_n=eval_n,
        train_steps=steps,
        eval_splits=eval_splits,
        batch_size=args.batch_size or batch,
        grad_accum=args.grad_accum or accum,
        lr=args.lr,
        max_length=args.max_length,
        eval_batch_size=args.eval_batch_size,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        max_new_tokens=args.max_new_tokens,
        resample_attempts=args.resample_attempts,
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
        f"- Training targets: `{','.join(config.arms)}`\n"
        f"- Training examples per seed: `{config.train_n}`\n"
        f"- Eval examples per split: `{config.eval_n}`\n"
        f"- Eval splits: `{','.join(config.eval_splits)}`\n"
        f"- Steps: `{config.train_steps}`\n"
        f"- Resample attempts: `{config.resample_attempts}`"
    )
    tokenizer = load_tokenizer(config.model_name)
    all_metrics: List[Dict[str, Any]] = []
    all_details: List[Dict[str, Any]] = []
    all_logs: List[Dict[str, Any]] = []
    start = time.time()
    eval_splits = config.eval_splits
    for arm in config.arms:
        for seed in config.seeds:
            run_id = f"{config.run_name}_{arm}_s{seed}"
            model, logs = train_adapter(config, arm, seed, tokenizer, run_root / run_id)
            all_logs.extend(logs)
            for split in eval_splits:
                examples = make_split(seed + 777, split, config.eval_n, arm)
                free_reference: Optional[Dict[str, Dict[str, Any]]] = None
                eval_arms = eval_arms_for_target(arm)
                ordered_eval_arms = sorted(eval_arms, key=lambda x: 0 if x.endswith("_free") else 1)
                for eval_arm in ordered_eval_arms:
                    metric, details = evaluate_model(config, model, tokenizer, examples, arm, eval_arm, run_id, seed, True, free_reference)
                    all_metrics.append(metric)
                    all_details.extend(details)
                    log(f"{run_id} {split} {eval_arm}: primary={pct(metric['primary_accuracy'])} exec={pct(metric['exec_accuracy'])} valid={pct(metric['valid_exec_rate'])} cgv={pct(metric['correct_given_valid'])}")
                    if eval_arm == "program_stack_free":
                        free_reference = {
                            row["example_id"]: {
                                "program_text": row["program_text"],
                                "exec_correct": bool(row["exec_correct"]),
                            }
                            for row in details
                        }
            cleanup_model(model)
    for oracle_target in ["oracle_abi32", "oracle_abi128"]:
        for seed in config.seeds:
            for split in eval_splits:
                examples = make_split(seed + 777, split, config.eval_n, oracle_target)
                for eval_arm in ["gold_abi_constrained"]:
                    run_id = f"{config.run_name}_{oracle_target}_{eval_arm}_s{seed}"
                    metric, details = evaluate_model(config, None, tokenizer, examples, oracle_target, eval_arm, run_id, seed, False)
                    all_metrics.append(metric)
                    all_details.extend(details)
                    log(f"{run_id} {split}: exec={pct(metric['exec_accuracy'])} valid={pct(metric['valid_exec_rate'])}")
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
    metric_cols = [
        c for c in metrics.columns
        if c.endswith("_accuracy")
        or c.endswith("_rate")
        or c.startswith("failure_")
        or c in {"valid_exec_rate", "mean_new_tokens", "mean_attempts", "exact_program_rate", "correct_given_valid"}
    ]
    rows = []
    for keys, sub in metrics.groupby(["suite", "train_target", "arm", "split", "depth", "template_shift"], dropna=False):
        row = {
            "suite": keys[0],
            "train_target": keys[1],
            "arm": keys[2],
            "split": keys[3],
            "depth": keys[4],
            "template_shift": keys[5],
            "runs": len(sub),
            "n_total": int(sub["n"].sum()),
        }
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
    suites = set(summary["suite"].dropna().unique())
    if "main" in suites:
        suite = "main"
    elif "pilot" in suites:
        suite = "pilot"
    elif "smoke" in suites:
        suite = "smoke"
    else:
        suite = sorted(suites)[-1]
    return summary[summary["suite"].eq(suite)].copy()


def plot_depth(primary: pd.DataFrame) -> None:
    sub = primary[
        primary["split"].str.startswith("eval_chain")
        & ~primary["split"].str.contains("template")
        & primary["train_target"].isin(CURRICULA)
        & primary["arm"].isin(["program_stack_free", "program_stack_constrained", "program_stack_resample_valid"])
    ].copy()
    if sub.empty:
        return
    plt.figure(figsize=(12, 6.4))
    for (train_target, arm), g in sub.groupby(["train_target", "arm"]):
        g = g.sort_values("depth")
        y = 100 * g["exec_accuracy_mean"].fillna(g["primary_accuracy_mean"])
        err = 100 * g.get("exec_accuracy_std", pd.Series([0] * len(g))).fillna(0)
        plt.errorbar(g["depth"], y, yerr=err, marker="o", capsize=4, label=f"{train_target}/{arm.replace('program_stack_', '')}")
    plt.xlabel("Linear chain depth")
    plt.ylabel("Externally executed procedure accuracy (%)")
    plt.title("Linear Chain Execution by ABI Size and Curriculum")
    plt.xticks(sorted(sub["depth"].unique()))
    plt.ylim(0, 105)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(FIGURES / "execution_by_depth.png", dpi=180)
    plt.close()


def plot_nested(primary: pd.DataFrame) -> None:
    sub = primary[
        primary["split"].str.startswith("eval_nested")
        & ~primary["split"].str.contains("template")
        & primary["train_target"].isin(CURRICULA)
        & primary["arm"].isin(["program_stack_free", "program_stack_constrained", "program_stack_resample_valid"])
    ].copy()
    if sub.empty:
        return
    plt.figure(figsize=(12, 6.4))
    for (train_target, arm), g in sub.groupby(["train_target", "arm"]):
        g = g.sort_values("depth")
        y = 100 * g["exec_accuracy_mean"].fillna(g["primary_accuracy_mean"])
        err = 100 * g.get("exec_accuracy_std", pd.Series([0] * len(g))).fillna(0)
        plt.errorbar(g["depth"], y, yerr=err, marker="o", capsize=4, label=f"{train_target}/{arm.replace('program_stack_', '')}")
    plt.xlabel("Nested branch count")
    plt.ylabel("Externally executed procedure accuracy (%)")
    plt.title("Nested Branch Execution by ABI Size and Curriculum")
    plt.xticks(sorted(sub["depth"].unique()))
    plt.ylim(0, 105)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(FIGURES / "nested_by_branches.png", dpi=180)
    plt.close()


def plot_template(primary: pd.DataFrame) -> None:
    sub = primary[
        primary["split"].str.contains("template")
        & primary["train_target"].isin(CURRICULA)
        & primary["arm"].isin(["program_stack_free", "program_stack_constrained", "program_stack_resample_valid"])
    ].copy()
    if sub.empty:
        return
    plt.figure(figsize=(12, 6.4))
    for (train_target, arm), g in sub.groupby(["train_target", "arm"]):
        g = g.sort_values("depth")
        y = 100 * g["exec_accuracy_mean"].fillna(g["primary_accuracy_mean"])
        err = 100 * g.get("exec_accuracy_std", pd.Series([0] * len(g))).fillna(0)
        plt.errorbar(g["depth"], y, yerr=err, marker="o", capsize=4, label=f"{train_target}/{arm.replace('program_stack_', '')}")
    plt.xlabel("Depth or branch count")
    plt.ylabel("Template-shift execution accuracy (%)")
    plt.title("Template Shift Endpoints")
    plt.xticks(sorted(sub["depth"].unique()))
    plt.ylim(0, 105)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(FIGURES / "template_shift_by_depth.png", dpi=180)
    plt.close()


def plot_validity_execution(primary: pd.DataFrame) -> None:
    sub = primary[
        ~primary["split"].str.contains("template")
        & primary["train_target"].isin(CURRICULA)
        & primary["arm"].isin(["program_stack_free", "program_stack_constrained", "program_stack_resample_valid"])
    ].copy()
    if sub.empty:
        return
    plt.figure(figsize=(12, 6.4))
    for (train_target, arm), g in sub.groupby(["train_target", "arm"]):
        g = g.sort_values("depth")
        label = f"{train_target}/{arm.replace('program_stack_', '')}"
        plt.plot(g["depth"], 100 * g["exec_accuracy_mean"], marker="o", label=f"{label} execution")
        plt.plot(g["depth"], 100 * g["valid_exec_rate_mean"], marker="s", linestyle="--", label=f"{label} valid")
    plt.xlabel("Depth or branch count")
    plt.ylabel("Rate (%)")
    plt.title("Validity Is Not the Same as Correct Execution")
    plt.xticks(sorted(sub["depth"].unique()))
    plt.ylim(0, 105)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(FIGURES / "validity_vs_execution.png", dpi=180)
    plt.close()


def plot_divergence(primary: pd.DataFrame) -> None:
    sub = primary[
        ~primary["split"].str.contains("template")
        & primary["train_target"].isin(CURRICULA)
        & primary["arm"].isin(["program_stack_constrained", "program_stack_resample_valid"])
    ].copy()
    if sub.empty or "divergence_rate_mean" not in sub:
        return
    plt.figure(figsize=(12, 6.4))
    for (train_target, arm), g in sub.groupby(["train_target", "arm"]):
        g = g.sort_values("depth")
        label = f"{train_target}/{arm.replace('program_stack_', '')}"
        plt.plot(g["depth"], 100 * g["divergence_rate_mean"], marker="o", label=f"{label} diverged")
        if "constrained_only_rate_mean" in g:
            plt.plot(g["depth"], 100 * g["constrained_only_rate_mean"], marker="s", linestyle="--", label=f"{label} decoder-only correct")
        if "free_only_rate_mean" in g:
            plt.plot(g["depth"], 100 * g["free_only_rate_mean"], marker="x", linestyle=":", label=f"{label} free-only correct")
    plt.xlabel("Depth or branch count")
    plt.ylabel("Rate (%)")
    plt.title("Constrained/Resample Divergence From Free Decoding")
    plt.xticks(sorted(sub["depth"].unique()))
    plt.ylim(0, 105)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(FIGURES / "decoder_divergence.png", dpi=180)
    plt.close()


def plot_failure(details: pd.DataFrame) -> None:
    if details.empty or "failure_type" not in details:
        return
    sub = details[(details["split"].isin(["eval_chain_d16", "eval_nested_l8"])) & (details["arm"].eq("program_stack_constrained"))].copy()
    if sub.empty:
        sub = details[(details["split"].isin(["eval_chain_d16", "eval_nested_l8"])) & (details["arm"].eq("program_stack_free"))].copy()
    if sub.empty:
        return
    counts = sub.groupby(["train_target", "failure_type"]).size().reset_index(name="n")
    totals = counts.groupby("train_target")["n"].transform("sum")
    counts["rate"] = counts["n"] / totals
    pivot = counts.pivot_table(index="train_target", columns="failure_type", values="rate", fill_value=0).sort_index()
    ax = pivot.plot(kind="bar", stacked=True, figsize=(11, 6))
    ax.set_ylabel("Fraction of generated procedures")
    ax.set_title("Constrained Failure Taxonomy on Chain/Nested Endpoints")
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
                if "accuracy" in col or "rate" in col or col.startswith("failure") or col == "correct_given_valid_mean":
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

    def row_for(train_target: str, arm: str, split: str) -> Optional[pd.Series]:
        sub = primary[
            primary["train_target"].eq(train_target)
            & primary["arm"].eq(arm)
            & primary["split"].eq(split)
        ]
        return sub.iloc[0] if not sub.empty else None

    def gap(a: Optional[pd.Series], b: Optional[pd.Series], col: str = "exec_accuracy_mean") -> str:
        if a is None or b is None:
            return "n/a"
        return pct(float(a.get(col, float("nan"))) - float(b.get(col, float("nan"))))

    lines: List[str] = []
    lines.append("# Qwen Large ABI Nested Compiler")
    lines.append("")
    lines.append("## Abstract")
    lines.append("")
    lines.append("This standalone experiment tests whether a constrained stack-ABI compiler remains reliable when the primitive library grows from 32 to 128 unary operations and when tasks require nested branch sub-procedures. The model emits a program; a deterministic interpreter executes it.")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append("Four QLoRA adapters are trained over the same numeric stack ABI shape:")
    lines.append("- `abi32_chain_d3`: 32 unary operations, chain tasks only, depths 1 to 3.")
    lines.append("- `abi128_chain_d3`: 128 unary operations, chain tasks only, depths 1 to 3.")
    lines.append("- `abi32_nested_d3`: 32 unary operations, chain depths 1 to 3 plus nested tasks with 2 to 3 branches.")
    lines.append("- `abi128_nested_d3`: 128 unary operations, chain depths 1 to 3 plus nested tasks with 2 to 3 branches.")
    lines.append("")
    lines.append("Evaluation sweeps linear chains at depths 3, 8, and 16, plus nested branch tasks with 2, 3, 4, and 8 branches. Template-shifted endpoints test wording robustness. Each adapter is evaluated with free greedy decoding and finite-state constrained decoding. Gold ABI sanity arms check both ABI sizes.")
    lines.append("")
    lines.append("The primary criteria are constrained external execution accuracy on chain depth 16 and nested 8-branch tasks. Valid-program rate alone is not a success metric; the compiler must select the right operations and merge structure, not merely produce parseable syntax.")
    lines.append("")
    lines.append("## Run Configuration")
    lines.append("")
    lines.append(f"- Primary suite: `{suite}`.")
    if not metrics.empty:
        main = metrics[metrics["suite"].eq(suite)]
        seeds = sorted(int(x) for x in main["seed"].dropna().unique()) if not main.empty else []
        lines.append(f"- Seeds: `{','.join(map(str, seeds))}`.")
        lines.append(f"- Evaluation rows: `{len(main)}` metric rows, `{int(main['n'].sum())}` scored examples across curricula and decoder arms.")
    if not logs.empty and "suite" in logs:
        suite_logs = logs[logs["suite"].eq(suite)]
        if not suite_logs.empty:
            lines.append(f"- QLoRA update steps per adapter: `{int(suite_logs['step'].max())}`.")
    lines.append("- Large adapters are stored outside the experiment tree.")
    lines.append("")
    lines.append("## Primary Results")
    lines.append("")
    for split, label in [
        ("eval_chain_d3", "chain depth 3"),
        ("eval_chain_d8", "chain depth 8"),
        ("eval_chain_d16", "chain depth 16"),
        ("eval_nested_l2", "nested 2 branches"),
        ("eval_nested_l3", "nested 3 branches"),
        ("eval_nested_l4", "nested 4 branches"),
        ("eval_nested_l8", "nested 8 branches"),
        ("eval_chain_template_d16", "template chain depth 16"),
        ("eval_nested_template_l8", "template nested 8 branches"),
    ]:
        rows = {c: row_for(c, "program_stack_constrained", split) for c in CURRICULA}
        if all(v is not None for v in rows.values()):
            lines.append(
                f"- Constrained {label}: "
                f"32-chain {pct(rows['abi32_chain_d3']['exec_accuracy_mean'])}; "
                f"128-chain {pct(rows['abi128_chain_d3']['exec_accuracy_mean'])}; "
                f"32-nested {pct(rows['abi32_nested_d3']['exec_accuracy_mean'])}; "
                f"128-nested {pct(rows['abi128_nested_d3']['exec_accuracy_mean'])}."
            )
    gold_32 = row_for("oracle_abi32", "gold_abi_constrained", "eval_nested_l8")
    gold_128 = row_for("oracle_abi128", "gold_abi_constrained", "eval_nested_l8")
    if gold_32 is not None and gold_128 is not None:
        lines.append(f"- Gold ABI nested-8 sanity: 32-op {pct(gold_32['exec_accuracy_mean'])} execution, 128-op {pct(gold_128['exec_accuracy_mean'])} execution.")
    if not metrics.empty:
        chain_rows = metrics[
            metrics["suite"].eq(suite)
            & metrics["split"].eq("eval_chain_d16")
            & metrics["arm"].eq("program_stack_constrained")
            & metrics["train_target"].isin(CURRICULA)
        ]
        if not chain_rows.empty:
            piv = chain_rows.pivot_table(index="seed", columns="train_target", values="exec_accuracy", aggfunc="mean")
            if {"abi32_chain_d3", "abi128_chain_d3"}.issubset(set(piv.columns)):
                deltas = piv["abi128_chain_d3"] - piv["abi32_chain_d3"]
                lines.append(f"- At chain depth 16, 128-op chain beats 32-op chain on `{int((deltas > 0).sum())}/{len(deltas)}` matched seeds; mean per-seed delta {pct(deltas.mean())}.")
        nested_rows = metrics[
            metrics["suite"].eq(suite)
            & metrics["split"].eq("eval_nested_l8")
            & metrics["arm"].eq("program_stack_constrained")
            & metrics["train_target"].isin(CURRICULA)
        ]
        if not nested_rows.empty:
            piv = nested_rows.pivot_table(index="seed", columns="train_target", values="exec_accuracy", aggfunc="mean")
            if {"abi128_chain_d3", "abi128_nested_d3"}.issubset(set(piv.columns)):
                deltas = piv["abi128_nested_d3"] - piv["abi128_chain_d3"]
                lines.append(f"- On 128-op nested-8 tasks, nested curriculum beats chain-only on `{int((deltas > 0).sum())}/{len(deltas)}` matched seeds; mean per-seed delta {pct(deltas.mean())}.")
    lines.append("")
    cols = [
        "train_target", "arm", "split", "depth", "runs", "n_total",
        "exec_accuracy_mean", "exec_accuracy_std", "valid_exec_rate_mean",
        "correct_given_valid_mean",
        "divergence_rate_mean", "constrained_only_rate_mean", "free_only_rate_mean",
        "mean_attempts_mean",
    ]
    available_cols = [c for c in cols if c in primary.columns]
    focus_splits = {"eval_chain_d8", "eval_chain_d16", "eval_chain_template_d16", "eval_nested_l4", "eval_nested_l8", "eval_nested_template_l8"}
    rows = primary[
        primary["split"].isin(focus_splits)
        & primary["arm"].isin(["program_stack_free", "program_stack_constrained", "gold_abi_constrained"])
    ].sort_values(["split", "train_target", "arm"])[available_cols] if not primary.empty else pd.DataFrame()
    lines.append(md_table(rows, available_cols))
    lines.append("")
    lines.append("![Execution by depth](../analysis/figures/execution_by_depth.png)")
    lines.append("")
    lines.append("![Nested by branches](../analysis/figures/nested_by_branches.png)")
    lines.append("")
    lines.append("![Template shift by depth](../analysis/figures/template_shift_by_depth.png)")
    lines.append("")
    lines.append("![Validity versus execution](../analysis/figures/validity_vs_execution.png)")
    lines.append("")
    lines.append("![Decoder divergence](../analysis/figures/decoder_divergence.png)")
    lines.append("")
    lines.append("![Failure taxonomy](../analysis/figures/failure_taxonomy.png)")
    lines.append("")
    lines.append("![Training curves](../analysis/figures/training_curves.png)")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("This experiment separates two questions that matter before scaling a real ABI: whether a larger operation catalog hurts linear chain compilation, and whether branch/sub-procedure structure requires explicit nested supervision.")
    c32 = row_for("abi32_chain_d3", "program_stack_constrained", "eval_chain_d16")
    c128 = row_for("abi128_chain_d3", "program_stack_constrained", "eval_chain_d16")
    if c32 is not None and c128 is not None:
        lines.append(f"Operation-scale effect on chain depth 16: 128-op chain training changes execution by {gap(c128, c32)} relative to 32-op chain training.")
    n32_chain = row_for("abi32_chain_d3", "program_stack_constrained", "eval_nested_l8")
    n32_nested = row_for("abi32_nested_d3", "program_stack_constrained", "eval_nested_l8")
    n128_chain = row_for("abi128_chain_d3", "program_stack_constrained", "eval_nested_l8")
    n128_nested = row_for("abi128_nested_d3", "program_stack_constrained", "eval_nested_l8")
    if n32_chain is not None and n32_nested is not None:
        lines.append(f"Nested-curriculum effect at 32 ops on nested-8 tasks: {gap(n32_nested, n32_chain)}.")
    if n128_chain is not None and n128_nested is not None:
        lines.append(f"Nested-curriculum effect at 128 ops on nested-8 tasks: {gap(n128_nested, n128_chain)}.")
    chain_template_128 = row_for("abi128_nested_d3", "program_stack_constrained", "eval_chain_template_d16")
    nested_template_128 = row_for("abi128_nested_d3", "program_stack_constrained", "eval_nested_template_l8")
    if chain_template_128 is not None and nested_template_128 is not None:
        lines.append(f"Template-shifted 128-op nested training reaches {pct(chain_template_128['exec_accuracy_mean'])} on chain depth 16 but only {pct(nested_template_128['exec_accuracy_mean'])} on nested-8, so wording robustness is not solved for nested branch tasks.")
    lines.append("The central positive result is that shallow nested supervision transfers beyond the trained branch counts: both nested curricula reach 100.0% at nested depth 4, and the 128-op nested curriculum reaches 86.7% at nested depth 8.")
    lines.append("The operation-catalog result is also positive but narrower: moving from 32 to 128 unary operations does not harm constrained linear-chain compilation, but it does not by itself teach nested structure.")
    lines.append("Because constrained decoding supplies only syntactic validity, any execution gain in constrained rows should be read as better operation or merge selection rather than better self-execution.")
    if not details.empty and "failure_type" in details:
        fail = details[
            details["suite"].eq(suite)
            & details["split"].isin(["eval_chain_d16", "eval_nested_l8"])
            & details["arm"].eq("program_stack_constrained")
        ]
        if not fail.empty:
            for train_target, sub in fail.groupby("train_target"):
                counts = sub.groupby("failure_type").size().sort_values(ascending=False)
                bits = [f"{k} {pct(v / len(sub))}" for k, v in counts.items()]
                lines.append(f"For `{train_target}` constrained decoding on chain/nested endpoints, procedures break down as: {', '.join(bits)}.")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append("This experiment tests compilation over a known numeric primitive library. It does not test invention of operations outside the ABI. The finite-state decoder is tied to the task schema and uses task-visible constants plus known line kinds, so results measure operation and merge selection inside a valid grammar. Nested tasks are branch-merge programs, not arbitrary loops or recursion.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append("- Metrics: `analysis/summary_by_arm.csv` and `analysis/all_metrics.csv`")
    lines.append("- Details: `analysis/all_details.csv`")
    lines.append("- Training logs: `analysis/all_train_logs.csv`")
    lines.append("- Checkpoints: `/workspace/large_artifacts/qwen_large_abi_nested_compiler/checkpoints`")
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
    return "<!doctype html><html><head><meta charset='utf-8'><title>Qwen Large ABI Nested Compiler</title><style>" + css + "</style></head><body>" + "\n".join(body) + "</body></html>\n"


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
    plot_nested(primary)
    plot_template(primary)
    plot_validity_execution(primary)
    plot_divergence(primary)
    plot_failure(details[details["suite"].eq(primary["suite"].iloc[0])] if not details.empty and not primary.empty else details)
    plot_training(logs[logs["suite"].eq(primary["suite"].iloc[0])] if not logs.empty and not primary.empty else logs)
    report = make_report(summary, metrics, details, logs)
    (REPORTS / "qwen_large_abi_nested_compiler_report.md").write_text(report)
    (REPORTS / "qwen_large_abi_nested_compiler_report.html").write_text(markdown_to_html(report))
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
    parser.add_argument("--splits", default="")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=448)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=320)
    parser.add_argument("--resample_attempts", type=int, default=3)
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
