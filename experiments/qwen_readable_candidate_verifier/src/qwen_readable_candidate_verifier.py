#!/usr/bin/env python3
"""Readable Qwen candidate verifier.

This standalone experiment asks whether Qwen can select executable repair
candidates when the task and candidate are rendered as readable pseudocode,
claimed final values, and optional execution traces. Target answers and target
states are used only to label candidates during training and oracle evaluation.
"""

from __future__ import annotations

import argparse
import csv
import gc
import html
import importlib.util
import json
import math
import platform
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


ROOT = Path("/workspace/experiments/qwen_readable_candidate_verifier")
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
LARGE_ROOT = Path("/workspace/large_artifacts/qwen_readable_candidate_verifier")
EMBED_ROOT = LARGE_ROOT / "embeddings"
CHECKPOINT_ROOT = LARGE_ROOT / "checkpoints"

TAIL_MODULE_PATH = Path("/workspace/experiments/qwen_tail_repair_stability_critic/src/qwen_tail_repair_stability_critic.py")
TAIL_CACHE_ROOT = Path("/workspace/large_artifacts/qwen_tail_repair_stability_critic/candidate_groups")
SOURCE_MODULE_PATH = Path("/workspace/experiments/qwen_compiler_multiseed_reattribution/src/qwen_compiler_multiseed_reattribution.py")
OP_NAMES = ["ADD", "SUB", "MUL"]
OP_WORD = {0: "add", 1: "subtract", 2: "multiply"}


def log(message: str) -> None:
    print(message, flush=True)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=json_default))


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


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def ensure_pad_token(tokenizer: Any) -> None:
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token or tokenizer.convert_ids_to_tokens(0)


def dtype_from_string(name: str) -> torch.dtype:
    name = name.lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(name)


def parse_csv_list(raw: str) -> List[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def parse_int_list(raw: str) -> List[int]:
    return [int(x) for x in parse_csv_list(raw)]


def source_seed_from_run(run: str) -> int:
    match = re.search(r"_seed(\d+)$", run)
    return int(match.group(1)) if match else -1


def group_key(group: Any) -> str:
    return f"{group.source_seed}:{group.split}:{group.example_index}"


def candidate_key(group: Any, candidate_index: int) -> str:
    return f"{group_key(group)}:{candidate_index}"


def pct(value: float) -> str:
    if math.isnan(value):
        return "n/a"
    return f"{100.0 * value:.1f}%"


def safe_div(num: float, den: float) -> float:
    if abs(den) < 1e-12:
        return float("nan")
    return float(num) / float(den)


@dataclass
class SelectedGroup:
    group: Any
    prompt: str
    shortlist: List[int]
    role: str


class ValueHead(nn.Module):
    def __init__(self, input_dim: int, width: int, dropout: float, echo: bool, max_steps: int, modulus: int) -> None:
        super().__init__()
        self.echo = bool(echo)
        self.max_steps = int(max_steps)
        self.modulus = int(modulus)
        self.trunk = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.score = nn.Linear(width, 1)
        if self.echo:
            self.final_head = nn.Linear(width, self.modulus)
            self.state_head = nn.Linear(width, self.max_steps * self.modulus)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.trunk(x.float())
        out = {"score": self.score(h).squeeze(-1)}
        if self.echo:
            out["final_logits"] = self.final_head(h)
            out["state_logits"] = self.state_head(h).view(-1, self.max_steps, self.modulus)
        return out


class FeatureHead(nn.Module):
    def __init__(self, input_dim: int, width: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width, 1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"score": self.net(x.float()).squeeze(-1)}


class TraceTransformer(nn.Module):
    def __init__(self, feat_dim: int, width: int, layers: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Linear(feat_dim, width)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.score = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.encoder(self.proj(x.float()))
        return {"score": self.score(h[:, 0]).squeeze(-1)}


def load_tail_groups(tail: ModuleType, source_runs: Sequence[str], cache_run_name: str) -> List[Any]:
    groups: List[Any] = []
    for source_run in source_runs:
        path = TAIL_CACHE_ROOT / cache_run_name / f"{source_run}_groups.pt"
        if not path.exists():
            raise FileNotFoundError(path)
        log(f"[groups] load {path}")
        raw = torch.load(path, map_location="cpu")
        loaded = [tail.plain_to_group(row) for row in raw["groups"]]
        groups.extend(loaded)
        log(f"[groups] {source_run}: {len(loaded)}")
    return groups


def build_prompt_map(source: ModuleType, tokenizer: Any, args: argparse.Namespace) -> Dict[Tuple[str, int], str]:
    ds_args = argparse.Namespace(
        max_steps=args.max_steps,
        dataset_seed=args.dataset_seed,
        modulus=args.modulus,
        train_examples=args.cache_train_examples,
        val_examples=args.cache_val_examples,
        eval_examples=args.cache_eval_examples,
        paired_eval_pairs=args.cache_paired_eval_pairs,
    )
    datasets = build_datasets_like_tail(source, tokenizer, ds_args)
    out: Dict[Tuple[str, int], str] = {}
    for split, dataset in datasets.items():
        for idx, example in enumerate(dataset.examples):
            out[(split, idx)] = str(example.prompt)
    return out


def build_datasets_like_tail(source: ModuleType, tokenizer: Any, args: argparse.Namespace) -> Dict[str, Any]:
    max_steps = int(args.max_steps)
    seed = int(args.dataset_seed)
    train_gen = source.TextProgramGenerator(tokenizer, args.modulus, max_steps, seed + 10, "mixed")
    val_gen = source.TextProgramGenerator(tokenizer, args.modulus, max_steps, seed + 20, "mixed")
    standard_gen = source.TextProgramGenerator(tokenizer, args.modulus, max_steps, seed + 30, "standard")
    paraphrase_gen = source.TextProgramGenerator(tokenizer, args.modulus, max_steps, seed + 40, "paraphrase")
    heldout_gen = source.TextProgramGenerator(tokenizer, args.modulus, max_steps, seed + 50, "heldout")
    pair_gen = source.TextProgramGenerator(tokenizer, args.modulus, max_steps, seed + 60, "mixed")
    heldout_pair_gen = source.TextProgramGenerator(tokenizer, args.modulus, max_steps, seed + 70, "mixed")
    return {
        "train_mixed_L24": train_gen.dataset(args.train_examples, 24, 24),
        "val_mixed_L24": val_gen.dataset(args.val_examples, 24, 24),
        "standard_L24": standard_gen.dataset(args.eval_examples, 24, 24),
        "paraphrase_L24": paraphrase_gen.dataset(args.eval_examples, 24, 24),
        "heldout_L24": heldout_gen.dataset(args.eval_examples, 24, 24),
        "paired_L24": pair_gen.paired_dataset(args.paired_eval_pairs, 24, 24, ["standard", "paraphrase"]),
        "paired_heldout_L24": heldout_pair_gen.paired_dataset(args.paired_eval_pairs, 24, 24, ["standard", "heldout"]),
    }


def shortlist_candidates(group: Any, limit: int) -> List[int]:
    n = len(group.candidates)
    chosen: List[int] = [int(group.base_index)]
    priors = group.priors.float()

    def add_many(indices: Iterable[int]) -> None:
        for idx in indices:
            idx = int(idx)
            if 0 <= idx < n and idx not in chosen:
                chosen.append(idx)
                if len(chosen) >= limit:
                    return

    add_many(torch.argsort(priors, descending=True).tolist())
    if len(chosen) < limit:
        final_logp_col = 34
        if group.features.shape[1] > final_logp_col:
            add_many(torch.argsort(group.features[:, final_logp_col], descending=True).tolist())
    if len(chosen) < limit:
        state_logp_col = 35
        if group.features.shape[1] > state_logp_col:
            add_many(torch.argsort(group.features[:, state_logp_col], descending=True).tolist())
    if len(chosen) < limit:
        best_by_answer: Dict[int, int] = {}
        for idx, cand in sorted(enumerate(group.candidates), key=lambda row: row[1].prior, reverse=True):
            best_by_answer.setdefault(int(cand.answer), int(idx))
        add_many(best_by_answer.values())
    if len(chosen) < limit:
        add_many(range(n))
    return chosen[:limit]


def select_groups(groups: Sequence[Any], prompt_map: Dict[Tuple[str, int], str], args: argparse.Namespace) -> List[SelectedGroup]:
    rng = random.Random(args.selection_seed)
    source_runs = parse_csv_list(args.source_runs)
    source_seeds = [source_seed_from_run(run) for run in source_runs]
    heldout_seed = int(args.heldout_source_seed)
    by_split_seed: Dict[Tuple[str, int], List[Any]] = {}
    for group in groups:
        by_split_seed.setdefault((group.split, int(group.source_seed)), []).append(group)
    selected: List[SelectedGroup] = []

    def take(split: str, seed: int, count: int, role: str) -> None:
        rows = list(by_split_seed.get((split, seed), []))
        rows.sort(key=lambda g: int(g.example_index))
        if args.shuffle_group_selection:
            rng.shuffle(rows)
        for group in rows[:count]:
            prompt = prompt_map[(group.split, int(group.example_index))]
            selected.append(SelectedGroup(group=group, prompt=prompt, shortlist=shortlist_candidates(group, int(args.shortlist_size)), role=role))

    for seed in source_seeds:
        if seed != heldout_seed:
            take("train_mixed_L24", seed, int(args.train_per_source), "train")
            take("val_mixed_L24", seed, int(args.val_per_source), "val")
    for split in parse_csv_list(args.eval_splits):
        for seed in source_seeds:
            take(split, seed, int(args.eval_per_source), "eval")
    return selected


def op_phrase(op: int, arg: int) -> str:
    if int(op) == 0:
        return f"add {arg}"
    if int(op) == 1:
        return f"subtract {arg}"
    return f"multiply by {arg}"


def op_symbol(op: int, arg: int) -> str:
    if int(op) == 0:
        return f"+{arg}"
    if int(op) == 1:
        return f"-{arg}"
    return f"*{arg}"


def op_python_expr(op: int, arg: int) -> str:
    if int(op) == 0:
        return f"(x + {int(arg)})"
    if int(op) == 1:
        return f"(x - {int(arg)})"
    return f"(x * {int(arg)})"


def compact_task_prompt(prompt: str) -> str:
    natural = prompt.split("Latent executable program registers:", 1)[0]
    lines = [line.strip() for line in natural.splitlines() if line.strip()]
    compact: List[str] = []
    for line in lines:
        item = line.rstrip(".")
        item = item.replace("Compute a hidden value modulo ", "mod=")
        item = item.replace("Work in arithmetic modulo ", "mod=")
        item = item.replace("Track x using modulus ", "mod=")
        item = item.replace("Every update below is modulo ", "mod=")
        item = item.replace("Initial x = ", "init=")
        item = item.replace("Start with x equal to ", "init=")
        item = item.replace("Let x be ", "init=")
        item = item.replace("The starting value of x is ", "init=")
        item = item.replace("Step: ", "")
        item = item.replace("Next, ", "")
        item = item.replace("Now ", "")
        item = item.replace("Use an ", "")
        item = item.replace("Use a ", "")
        item = item.replace("increase x by ", "add ")
        item = item.replace("decrease x by ", "subtract ")
        item = item.replace("scale x by ", "multiply by ")
        item = item.replace(" update of ", " ")
        item = item.replace("Return x after the final step", "return final x")
        item = item.replace("Report x after all updates", "return final x")
        item = item.replace("Give the final value of x", "return final x")
        item = item.replace("What is x after the listed updates?", "return final x")
        if item.lower() in {"answer:", "final answer:", "result:", "value:"}:
            continue
        compact.append(item)
    return "; ".join(compact)


def program_lines(init_value: int, ops: Sequence[int], args: Sequence[int], length: int, modulus: int) -> List[str]:
    lines = [f"x = {int(init_value)}"]
    for step in range(int(length)):
        lines.append(f"x = {op_python_expr(int(ops[step]), int(args[step]))} % {int(modulus)}  # step {step + 1}")
    lines.append("return x")
    return lines


def render_task_program(group: Any, modulus: int) -> str:
    return "\n".join(program_lines(int(group.init_value), group.ops, group.args, int(group.length), int(modulus)))


def render_candidate_program(candidate: Any, length: int, modulus: int) -> str:
    return "\n".join(program_lines(int(candidate.init_value), candidate.ops, candidate.args, int(length), int(modulus)))


def candidate_trace_lines(candidate: Any, length: int, corrupt_trace: bool = False) -> List[str]:
    states = [int(x) for x in candidate.states[:length]]
    if corrupt_trace:
        states = list(reversed(states))
    lines = []
    for step in range(length):
        lines.append(f"step {step + 1}: after {op_phrase(int(candidate.ops[step]), int(candidate.args[step]))}, x = {states[step]}")
    return lines


def serialize_candidate(prompt: str, group: Any, candidate_index: int, mode: str, modulus: int = 97) -> str:
    cand = group.candidates[candidate_index]
    length = int(group.length)
    corrupt = mode in {"trace_corrupt", "readable_trace_corrupt"}
    task_program = render_task_program(group, modulus)
    candidate_program = render_candidate_program(cand, length, modulus)
    trace_text = "\n".join(candidate_trace_lines(cand, length, corrupt_trace=corrupt))
    claimed = int(cand.answer)
    mode = "program_trace" if mode == "full" else mode

    if mode == "task_value":
        return (
            "Compute the final value of this modular program. "
            "Use exact arithmetic and answer with one integer only.\n\n"
            "Program:\n"
            f"```python\n{task_program}\n```\n\n"
            "Final value:"
        )

    elif mode == "prompt_only":
        return (
            "Task program:\n"
            f"```python\n{task_program}\n```\n\n"
            "Question: Is a hidden candidate program equivalent to this task program? Answer yes or no.\n"
            "Answer:"
        )

    if mode == "candidate_only":
        return (
            "Candidate program:\n"
            f"```python\n{candidate_program}\n```\n\n"
            f"Claimed final value: {claimed}\n\n"
            "Question: Is the claimed final value consistent with the candidate program? Answer yes or no.\n"
            "Answer:"
        )

    if mode == "program_equiv":
        return (
            "Task program:\n"
            f"```python\n{task_program}\n```\n\n"
            "Candidate program:\n"
            f"```python\n{candidate_program}\n```\n\n"
            "Question: Do these two programs return the same final value? Answer yes or no.\n"
            "Answer:"
        )

    if mode == "program_final":
        return (
            "Task program:\n"
            f"```python\n{task_program}\n```\n\n"
            "Candidate program:\n"
            f"```python\n{candidate_program}\n```\n\n"
            f"Candidate claimed final value: {claimed}\n\n"
            "Question: Is this candidate a correct solution for the task program? Answer yes or no.\n"
            "Answer:"
        )

    if mode in {"program_trace", "trace_corrupt"}:
        label = "Candidate execution trace" if not corrupt else "Candidate execution trace with corrupted state order"
        return (
            "Task program:\n"
            f"```python\n{task_program}\n```\n\n"
            "Candidate program:\n"
            f"```python\n{candidate_program}\n```\n\n"
            f"Candidate claimed final value: {claimed}\n\n"
            f"{label}:\n{trace_text}\n\n"
            "Question: Is this candidate a correct solution for the task program? Answer yes or no.\n"
            "Answer:"
        )

    return (
        "Task program:\n"
        f"```python\n{task_program}\n```\n\n"
        "Candidate program:\n"
        f"```python\n{candidate_program}\n```\n\n"
        f"Candidate claimed final value: {claimed}\n\n"
        "Question: Is this candidate a correct solution for the task program? Answer yes or no.\n"
        "Answer:"
    )


@torch.no_grad()
def embed_texts(
    texts: Sequence[str],
    tokenizer: Any,
    model: Any,
    batch_size: int,
    max_length: int,
    device: torch.device,
    value_token_ids: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    embeddings: List[torch.Tensor] = []
    zero_scores: List[torch.Tensor] = []
    value_scores: List[torch.Tensor] = []
    yes_ids = tokenizer(" yes", add_special_tokens=False)["input_ids"]
    no_ids = tokenizer(" no", add_special_tokens=False)["input_ids"]
    yes_id = int(yes_ids[0])
    no_id = int(no_ids[0])
    value_ids_tensor: Optional[torch.Tensor] = None
    if value_token_ids is not None:
        value_ids_tensor = torch.tensor([int(x) for x in value_token_ids], dtype=torch.long, device=device)
    for start in range(0, len(texts), batch_size):
        batch_texts = list(texts[start : start + batch_size])
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model.model(**enc, output_hidden_states=True, use_cache=False)
        attention = enc["attention_mask"]
        last_pos = attention.long().sum(dim=1) - 1
        rows = torch.arange(last_pos.shape[0], device=device)
        hidden = out.hidden_states[-1][rows, last_pos].detach().float().cpu()
        logits = model.lm_head(out.hidden_states[-1][rows, last_pos]).detach().float()
        score = (logits[:, yes_id] - logits[:, no_id]).cpu()
        embeddings.append(hidden)
        zero_scores.append(score)
        if value_ids_tensor is not None:
            value_scores.append(logits.index_select(dim=1, index=value_ids_tensor).cpu())
        if torch.cuda.is_available() and (start // max(1, batch_size)) % 20 == 0:
            torch.cuda.empty_cache()
    value_out = torch.cat(value_scores, dim=0) if value_scores else None
    return torch.cat(embeddings, dim=0), torch.cat(zero_scores, dim=0), value_out


@torch.no_grad()
def score_yes_no_texts(
    texts: Sequence[str],
    tokenizer: Any,
    model: Any,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    scores: List[torch.Tensor] = []
    yes_id = int(tokenizer(" yes", add_special_tokens=False)["input_ids"][0])
    no_id = int(tokenizer(" no", add_special_tokens=False)["input_ids"][0])
    for start in range(0, len(texts), batch_size):
        batch_texts = list(texts[start : start + batch_size])
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model.model(**enc, use_cache=False)
        attention = enc["attention_mask"]
        last_pos = attention.long().sum(dim=1) - 1
        rows = torch.arange(last_pos.shape[0], device=device)
        hidden = out.last_hidden_state[rows, last_pos]
        logits = model.lm_head(hidden).detach().float()
        scores.append((logits[:, yes_id] - logits[:, no_id]).cpu())
        if torch.cuda.is_available() and (start // max(1, batch_size)) % 20 == 0:
            torch.cuda.empty_cache()
    return torch.cat(scores, dim=0)


def first_token_ids_for_values(tokenizer: Any, modulus: int) -> List[int]:
    ids: List[int] = []
    multi: List[int] = []
    for value in range(int(modulus)):
        toks = tokenizer(f" {value}", add_special_tokens=False)["input_ids"]
        if len(toks) != 1:
            multi.append(value)
        ids.append(int(toks[0]))
    if multi:
        log(f"[tokenizer] values with multi-token rendering use first-token approximation: {multi[:10]}{'...' if len(multi) > 10 else ''}")
    return ids


@torch.no_grad()
def score_value_continuations(
    prompts: Sequence[str],
    tokenizer: Any,
    model: Any,
    batch_size: int,
    max_length: int,
    device: torch.device,
    modulus: int,
) -> torch.Tensor:
    rows: List[List[float]] = []
    values = list(range(int(modulus)))
    for prompt in prompts:
        full_texts = [f"{prompt} {value}" for value in values]
        prompt_len = len(tokenizer(prompt, add_special_tokens=True)["input_ids"])
        scores = [0.0 for _ in values]
        for start in range(0, len(full_texts), batch_size):
            batch_texts = full_texts[start : start + batch_size]
            enc = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                add_special_tokens=True,
            )
            input_ids = enc["input_ids"].to(device)
            attention = enc["attention_mask"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention, use_cache=False)
            logp = F.log_softmax(out.logits.float(), dim=-1)
            for local in range(input_ids.shape[0]):
                absolute = start + local
                seq_len = int(attention[local].sum().item())
                total = 0.0
                for pos in range(prompt_len, seq_len):
                    total += float(logp[local, pos - 1, int(input_ids[local, pos].item())].detach().cpu())
                scores[absolute] = total
        rows.append(scores)
        if torch.cuda.is_available() and len(rows) % 16 == 0:
            torch.cuda.empty_cache()
    return torch.tensor(rows, dtype=torch.float32)


def load_qwen_reader(args: argparse.Namespace) -> Tuple[Any, Any, torch.device]:
    dtype = dtype_from_string(args.torch_dtype)
    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype if dtype != torch.float32 else torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True, use_fast=True)
    ensure_pad_token(tokenizer)
    tokenizer.padding_side = "right"
    kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if torch.cuda.is_available():
        kwargs["device_map"] = args.device_map
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    log(f"[qwen] load {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **kwargs)
    model.eval()
    model.config.use_cache = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return tokenizer, model, device


def build_or_load_embeddings(
    selected: Sequence[SelectedGroup],
    mode: str,
    tokenizer: Any,
    model: Any,
    device: torch.device,
    args: argparse.Namespace,
    run_dir: Path,
) -> Dict[str, Dict[str, torch.Tensor]]:
    cache_name = args.embedding_run_name if args.embedding_run_name else args.run_name
    cache_path = EMBED_ROOT / cache_name / f"{mode}_embeddings.pt"
    if cache_path.exists() and not args.rebuild_embeddings:
        log(f"[embed] load {cache_path}")
        return torch.load(cache_path, map_location="cpu")
    if tokenizer is None or model is None or device is None:
        raise RuntimeError(f"embedding cache missing and no Qwen reader loaded: {cache_path}")

    log(f"[embed] build mode={mode}")
    store: Dict[str, Dict[str, torch.Tensor]] = {}
    if mode == "task_value":
        texts = [serialize_candidate(row.prompt, row.group, row.shortlist[0], mode, int(args.modulus)) for row in selected]
        embs, zeros, _unused = embed_texts(texts, tokenizer, model, int(args.qwen_batch_size), int(args.max_length), device)
        value_scores = score_value_continuations(texts, tokenizer, model, max(1, int(args.qwen_batch_size) * 2), int(args.max_length), device, int(args.modulus))
        for i, row in enumerate(selected):
            key = group_key(row.group)
            answers = torch.tensor([int(row.group.candidates[int(cand_idx)].answer) for cand_idx in row.shortlist], dtype=torch.long)
            store[key] = {
                "embedding": embs[i].view(1, -1).repeat(len(row.shortlist), 1).half(),
                "zero_score": value_scores[i].index_select(0, answers).float(),
                "value_logits": value_scores[i].float(),
                "candidate_indices": torch.tensor(row.shortlist, dtype=torch.long),
            }
    train_modes = set(parse_csv_list(args.train_qwen_modes))
    if mode != "task_value" and mode not in train_modes:
        if mode == "prompt_only":
            texts = [serialize_candidate(row.prompt, row.group, row.shortlist[0], mode, int(args.modulus)) for row in selected]
            zeros = score_yes_no_texts(texts, tokenizer, model, int(args.qwen_batch_size), int(args.max_length), device)
            for i, row in enumerate(selected):
                key = group_key(row.group)
                n = len(row.shortlist)
                store[key] = {
                    "embedding": torch.zeros(n, 1, dtype=torch.float16),
                    "zero_score": zeros[i].view(1).repeat(n).float(),
                    "candidate_indices": torch.tensor(row.shortlist, dtype=torch.long),
                }
        else:
            texts: List[str] = []
            owner: List[Tuple[str, int]] = []
            for row in selected:
                key = group_key(row.group)
                for local, cand_idx in enumerate(row.shortlist):
                    texts.append(serialize_candidate(row.prompt, row.group, cand_idx, mode, int(args.modulus)))
                    owner.append((key, local))
            zeros = score_yes_no_texts(texts, tokenizer, model, int(args.qwen_batch_size), int(args.max_length), device)
            offsets: Dict[str, List[int]] = {}
            for i, (key, _local) in enumerate(owner):
                offsets.setdefault(key, []).append(i)
            selected_by_key = {group_key(row.group): row for row in selected}
            for key, idxs in offsets.items():
                row = selected_by_key[key]
                store[key] = {
                    "embedding": torch.zeros(len(idxs), 1, dtype=torch.float16),
                    "zero_score": zeros[idxs].float(),
                    "candidate_indices": torch.tensor(row.shortlist, dtype=torch.long),
                }
    elif mode == "prompt_only":
        texts = [serialize_candidate(row.prompt, row.group, row.shortlist[0], mode, int(args.modulus)) for row in selected]
        embs, zeros, _value_scores = embed_texts(texts, tokenizer, model, int(args.qwen_batch_size), int(args.max_length), device)
        for i, row in enumerate(selected):
            key = group_key(row.group)
            n = len(row.shortlist)
            store[key] = {
                "embedding": embs[i].view(1, -1).repeat(n, 1).half(),
                "zero_score": zeros[i].view(1).repeat(n).float(),
                "candidate_indices": torch.tensor(row.shortlist, dtype=torch.long),
            }
    elif mode != "task_value":
        texts: List[str] = []
        owner: List[Tuple[str, int]] = []
        for row in selected:
            key = group_key(row.group)
            for local, cand_idx in enumerate(row.shortlist):
                texts.append(serialize_candidate(row.prompt, row.group, cand_idx, mode, int(args.modulus)))
                owner.append((key, local))
        embs, zeros, _value_scores = embed_texts(texts, tokenizer, model, int(args.qwen_batch_size), int(args.max_length), device)
        offsets: Dict[str, List[int]] = {}
        for i, (key, _local) in enumerate(owner):
            offsets.setdefault(key, []).append(i)
        selected_by_key = {group_key(row.group): row for row in selected}
        for key, idxs in offsets.items():
            row = selected_by_key[key]
            store[key] = {
                "embedding": embs[idxs].half(),
                "zero_score": zeros[idxs].float(),
                "candidate_indices": torch.tensor(row.shortlist, dtype=torch.long),
            }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(store, cache_path)
    write_json(run_dir / f"{mode}_embedding_manifest.json", {"path": str(cache_path), "groups": len(store), "mode": mode})
    return store


def trace_tensor_for_group(group: Any, shortlist: Sequence[int], modulus: int, max_steps: int, corrupt_trace: bool = False) -> torch.Tensor:
    rows: List[List[List[float]]] = []
    base = group.candidates[int(group.base_index)]
    for cand_idx in shortlist:
        cand = group.candidates[int(cand_idx)]
        prior = float(cand.prior)
        edit_count = float(cand.edit_count)
        answer = float(cand.answer) / max(1.0, float(modulus - 1))
        init = float(cand.init_value) / max(1.0, float(modulus - 1))
        tokens: List[List[float]] = []
        tokens.append([1.0, 0.0, 0.0, init, answer, prior / 64.0, edit_count / 16.0, float(cand.changed), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        states = [int(x) for x in cand.states[: int(group.length)]]
        if corrupt_trace:
            states = list(reversed(states))
        for step in range(max_steps):
            if step < int(group.length):
                op = int(cand.ops[step])
                bop = int(base.ops[step])
                arg = float(cand.args[step]) / max(1.0, float(modulus - 1))
                barg = float(base.args[step]) / max(1.0, float(modulus - 1))
                st = float(states[step]) / max(1.0, float(modulus - 1))
                bst = float(base.states[step]) / max(1.0, float(modulus - 1))
                tokens.append([
                    0.0,
                    1.0,
                    float(step + 1) / max(1.0, float(max_steps)),
                    1.0 if op == 0 else 0.0,
                    1.0 if op == 1 else 0.0,
                    1.0 if op == 2 else 0.0,
                    arg,
                    st,
                    1.0 if bop == 0 else 0.0,
                    1.0 if bop == 1 else 0.0,
                    1.0 if bop == 2 else 0.0,
                    barg,
                    bst,
                    1.0 if op != bop else 0.0,
                    abs(arg - barg),
                    abs(st - bst),
                ])
            else:
                tokens.append([0.0] * 16)
        rows.append(tokens)
    return torch.tensor(rows, dtype=torch.float32)


def feature_tensor_for_group(tail: ModuleType, group: Any, shortlist: Sequence[int], args: argparse.Namespace) -> torch.Tensor:
    detail = tail.detail_features_for_group(group, int(args.tail_detail_window), int(args.modulus))
    full = torch.cat([group.features.float(), detail.float()], dim=1)
    return full[torch.tensor(shortlist, dtype=torch.long)]


def labels_for_group(group: Any, shortlist: Sequence[int], shuffled: bool = False, rng: Optional[random.Random] = None) -> torch.Tensor:
    labels = group.answer_labels[torch.tensor(shortlist, dtype=torch.long)].bool().clone()
    if shuffled:
        if rng is None:
            rng = random.Random(0)
        values = labels.tolist()
        rng.shuffle(values)
        labels = torch.tensor(values, dtype=torch.bool)
    return labels


def group_rank_loss(scores: torch.Tensor, labels: torch.Tensor, base_local_index: int = 0, no_positive_base_weight: float = 0.1) -> torch.Tensor:
    if bool(labels.any()):
        return torch.logsumexp(scores, dim=0) - torch.logsumexp(scores[labels], dim=0)
    target = torch.tensor([base_local_index], dtype=torch.long, device=scores.device)
    return F.cross_entropy(scores.view(1, -1), target) * float(no_positive_base_weight)


def echo_loss(outputs: Dict[str, torch.Tensor], group: Any, shortlist: Sequence[int], device: torch.device, max_steps: int) -> torch.Tensor:
    if "final_logits" not in outputs:
        return torch.tensor(0.0, device=device)
    final_targets = torch.tensor([int(group.candidates[int(i)].answer) for i in shortlist], dtype=torch.long, device=device)
    final_loss = F.cross_entropy(outputs["final_logits"], final_targets)
    state_targets = torch.tensor(
        [[max(0, int(x)) for x in group.candidates[int(i)].states[:max_steps]] for i in shortlist],
        dtype=torch.long,
        device=device,
    )
    state_logits = outputs["state_logits"]
    state_loss = F.cross_entropy(state_logits.view(-1, state_logits.shape[-1]), state_targets.view(-1))
    return final_loss + state_loss


def train_qwen_head(
    selected: Sequence[SelectedGroup],
    store: Dict[str, Dict[str, torch.Tensor]],
    args: argparse.Namespace,
    run_dir: Path,
    arm: str,
    seed: int,
    echo: bool = False,
    shuffled: bool = False,
) -> Tuple[ValueHead, Dict[str, Any], List[Dict[str, Any]]]:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    train_rows = [row for row in selected if row.role == "train"]
    val_rows = [row for row in selected if row.role == "val"]
    input_dim = int(next(iter(store.values()))["embedding"].shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ValueHead(input_dim, int(args.head_width), float(args.dropout), echo, int(args.max_steps), int(args.modulus)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_score = -1e9
    logs: List[Dict[str, Any]] = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        total = 0.0
        rng.shuffle(train_rows)
        for row in train_rows:
            key = group_key(row.group)
            x = store[key]["embedding"].to(device).float()
            labels = labels_for_group(row.group, row.shortlist, shuffled=shuffled, rng=rng).to(device)
            out = model(x)
            loss = group_rank_loss(out["score"], labels, 0, float(args.no_positive_base_weight))
            if echo:
                loss = loss + float(args.echo_weight) * echo_loss(out, row.group, row.shortlist, device, int(args.max_steps))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
            opt.step()
            total += float(loss.detach().cpu())
        selected_idx = select_with_model(selected, store, model, "qwen", None, args)
        val_metrics = aggregate_metrics([row for row in selected if row.role == "val"], selected_idx)
        val_acc = float(val_metrics.get("accuracy", 0.0))
        val_damage = float(val_metrics.get("damage_rate", 0.0))
        val_recovery = float(val_metrics.get("recovery_rate", 0.0))
        utility = val_acc + 0.25 * val_recovery - 0.75 * val_damage
        logs.append({
            "arm": arm,
            "seed": seed,
            "epoch": epoch,
            "train_loss": total / max(1, len(train_rows)),
            "val_accuracy": val_acc,
            "val_damage_rate": val_damage,
            "val_recovery_rate": val_recovery,
            "val_utility": utility,
        })
        if utility > best_score:
            best_score = utility
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    ckpt = CHECKPOINT_ROOT / args.run_name / f"{arm}_seed{seed}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "input_dim": input_dim, "args": vars(args), "seed": seed, "echo": echo}, ckpt)
    return model, {"arm": arm, "seed": seed, "checkpoint": str(ckpt), "best_utility": best_score}, logs


def fit_feature_stats(selected: Sequence[SelectedGroup], tail: ModuleType, args: argparse.Namespace) -> Tuple[torch.Tensor, torch.Tensor]:
    xs = []
    for row in selected:
        if row.role == "train":
            xs.append(feature_tensor_for_group(tail, row.group, row.shortlist, args))
    x = torch.cat(xs, dim=0)
    mean = x.mean(dim=0)
    std = x.std(dim=0).clamp_min(1e-4)
    return mean, std


def train_feature_head(
    selected: Sequence[SelectedGroup],
    tail: ModuleType,
    args: argparse.Namespace,
    arm: str,
    seed: int,
) -> Tuple[FeatureHead, Dict[str, Any], List[Dict[str, Any]], Tuple[torch.Tensor, torch.Tensor]]:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    train_rows = [row for row in selected if row.role == "train"]
    mean, std = fit_feature_stats(selected, tail, args)
    input_dim = int(mean.numel())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FeatureHead(input_dim, int(args.head_width), float(args.dropout)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    best_state = None
    best_score = -1e9
    logs: List[Dict[str, Any]] = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        total = 0.0
        rng.shuffle(train_rows)
        for row in train_rows:
            x = (feature_tensor_for_group(tail, row.group, row.shortlist, args) - mean) / std
            x = x.to(device)
            labels = labels_for_group(row.group, row.shortlist).to(device)
            out = model(x)
            loss = group_rank_loss(out["score"], labels, 0, float(args.no_positive_base_weight))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
            opt.step()
            total += float(loss.detach().cpu())
        selected_idx = select_with_model(selected, {}, model, "feature", (tail, mean, std), args)
        val_metrics = aggregate_metrics([row for row in selected if row.role == "val"], selected_idx)
        utility = float(val_metrics.get("accuracy", 0.0)) + 0.25 * float(val_metrics.get("recovery_rate", 0.0)) - 0.75 * float(val_metrics.get("damage_rate", 0.0))
        logs.append({"arm": arm, "seed": seed, "epoch": epoch, "train_loss": total / max(1, len(train_rows)), "val_utility": utility, **{f"val_{k}": v for k, v in val_metrics.items()}})
        if utility > best_score:
            best_score = utility
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"arm": arm, "seed": seed, "best_utility": best_score}, logs, (mean, std)


def train_trace_head(
    selected: Sequence[SelectedGroup],
    args: argparse.Namespace,
    arm: str,
    seed: int,
    corrupt_trace: bool = False,
) -> Tuple[TraceTransformer, Dict[str, Any], List[Dict[str, Any]]]:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    train_rows = [row for row in selected if row.role == "train"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TraceTransformer(16, int(args.trace_width), int(args.trace_layers), int(args.trace_heads), float(args.dropout)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    best_state = None
    best_score = -1e9
    logs: List[Dict[str, Any]] = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        total = 0.0
        rng.shuffle(train_rows)
        for row in train_rows:
            x = trace_tensor_for_group(row.group, row.shortlist, int(args.modulus), int(args.max_steps), corrupt_trace=corrupt_trace).to(device)
            labels = labels_for_group(row.group, row.shortlist).to(device)
            out = model(x)
            loss = group_rank_loss(out["score"], labels, 0, float(args.no_positive_base_weight))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
            opt.step()
            total += float(loss.detach().cpu())
        selected_idx = select_with_model(selected, {}, model, "trace_corrupt" if corrupt_trace else "trace", None, args)
        val_metrics = aggregate_metrics([row for row in selected if row.role == "val"], selected_idx)
        utility = float(val_metrics.get("accuracy", 0.0)) + 0.25 * float(val_metrics.get("recovery_rate", 0.0)) - 0.75 * float(val_metrics.get("damage_rate", 0.0))
        logs.append({"arm": arm, "seed": seed, "epoch": epoch, "train_loss": total / max(1, len(train_rows)), "val_utility": utility, **{f"val_{k}": v for k, v in val_metrics.items()}})
        if utility > best_score:
            best_score = utility
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"arm": arm, "seed": seed, "best_utility": best_score}, logs


@torch.no_grad()
def select_with_model(
    rows: Sequence[SelectedGroup],
    store: Dict[str, Dict[str, torch.Tensor]],
    model: nn.Module,
    model_kind: str,
    aux: Any,
    args: argparse.Namespace,
) -> Dict[str, int]:
    device = next(model.parameters()).device
    model.eval()
    out: Dict[str, int] = {}
    for row in rows:
        key = group_key(row.group)
        if model_kind == "qwen":
            x = store[key]["embedding"].to(device).float()
            scores = model(x)["score"].detach().cpu()
        elif model_kind == "feature":
            tail, mean, std = aux
            x = (feature_tensor_for_group(tail, row.group, row.shortlist, args) - mean) / std
            scores = model(x.to(device))["score"].detach().cpu()
        elif model_kind == "trace":
            x = trace_tensor_for_group(row.group, row.shortlist, int(args.modulus), int(args.max_steps), False).to(device)
            scores = model(x)["score"].detach().cpu()
        elif model_kind == "trace_corrupt":
            x = trace_tensor_for_group(row.group, row.shortlist, int(args.modulus), int(args.max_steps), True).to(device)
            scores = model(x)["score"].detach().cpu()
        else:
            raise ValueError(model_kind)
        out[key] = int(row.shortlist[int(torch.argmax(scores).item())])
    return out


def select_zero(rows: Sequence[SelectedGroup], store: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, int]:
    out = {}
    for row in rows:
        key = group_key(row.group)
        scores = store[key]["zero_score"]
        out[key] = int(row.shortlist[int(torch.argmax(scores).item())])
    return out


@torch.no_grad()
def scores_with_model(
    rows: Sequence[SelectedGroup],
    store: Dict[str, Dict[str, torch.Tensor]],
    model: nn.Module,
    model_kind: str,
    aux: Any,
    args: argparse.Namespace,
) -> Dict[str, torch.Tensor]:
    device = next(model.parameters()).device
    model.eval()
    out: Dict[str, torch.Tensor] = {}
    for row in rows:
        key = group_key(row.group)
        if model_kind == "qwen":
            x = store[key]["embedding"].to(device).float()
            scores = model(x)["score"].detach().cpu()
        elif model_kind == "feature":
            tail, mean, std = aux
            x = (feature_tensor_for_group(tail, row.group, row.shortlist, args) - mean) / std
            scores = model(x.to(device))["score"].detach().cpu()
        elif model_kind == "trace":
            x = trace_tensor_for_group(row.group, row.shortlist, int(args.modulus), int(args.max_steps), False).to(device)
            scores = model(x)["score"].detach().cpu()
        elif model_kind == "trace_corrupt":
            x = trace_tensor_for_group(row.group, row.shortlist, int(args.modulus), int(args.max_steps), True).to(device)
            scores = model(x)["score"].detach().cpu()
        else:
            raise ValueError(model_kind)
        out[key] = scores
    return out


def select_from_scores(rows: Sequence[SelectedGroup], scores_by_key: Dict[str, torch.Tensor]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for row in rows:
        key = group_key(row.group)
        scores = scores_by_key[key]
        out[key] = int(row.shortlist[int(torch.argmax(scores).item())])
    return out


def select_with_base_margin(rows: Sequence[SelectedGroup], scores_by_key: Dict[str, torch.Tensor], margin: float) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for row in rows:
        key = group_key(row.group)
        scores = scores_by_key[key]
        best_local = int(torch.argmax(scores).item())
        base_local = row.shortlist.index(int(row.group.base_index)) if int(row.group.base_index) in row.shortlist else 0
        if best_local != base_local and float(scores[best_local] - scores[base_local]) > float(margin):
            out[key] = int(row.shortlist[best_local])
        else:
            out[key] = int(row.group.base_index)
    return out


def tune_base_margin(rows: Sequence[SelectedGroup], scores_by_key: Dict[str, torch.Tensor]) -> Tuple[float, Dict[str, float]]:
    val_rows = [row for row in rows if row.role == "val"]
    candidates = [0.0]
    for row in val_rows:
        key = group_key(row.group)
        scores = scores_by_key[key]
        base_local = row.shortlist.index(int(row.group.base_index)) if int(row.group.base_index) in row.shortlist else 0
        for local in range(len(row.shortlist)):
            if local != base_local:
                candidates.append(float(scores[local] - scores[base_local]))
    candidates.extend([min(candidates) - 1.0, max(candidates) + 1.0])
    best_margin = max(candidates) + 1.0
    best_utility = -1e9
    best_metrics: Dict[str, float] = {}
    for margin in sorted(set(candidates)):
        sel = select_with_base_margin(rows, scores_by_key, margin)
        metrics = aggregate_metrics(val_rows, sel)
        utility = float(metrics.get("accuracy", 0.0)) + 0.25 * float(metrics.get("recovery_rate", 0.0)) - 0.75 * float(metrics.get("damage_rate", 0.0))
        if utility > best_utility:
            best_utility = utility
            best_margin = float(margin)
            best_metrics = metrics
            best_metrics["val_utility"] = utility
    return best_margin, best_metrics


def select_base(rows: Sequence[SelectedGroup]) -> Dict[str, int]:
    return {group_key(row.group): int(row.group.base_index) for row in rows}


def select_oracle(rows: Sequence[SelectedGroup], shortlist_only: bool) -> Dict[str, int]:
    out = {}
    for row in rows:
        candidates = row.shortlist if shortlist_only else list(range(len(row.group.candidates)))
        positives = [idx for idx in candidates if bool(row.group.candidates[int(idx)].answer_exact)]
        if positives:
            positives.sort(key=lambda idx: float(row.group.candidates[int(idx)].prior), reverse=True)
            out[group_key(row.group)] = int(positives[0])
        else:
            out[group_key(row.group)] = int(row.group.base_index)
    return out


def select_majority_answer(rows: Sequence[SelectedGroup]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for row in rows:
        counts: Dict[int, int] = {}
        best_prior: Dict[int, Tuple[float, int]] = {}
        for cand_idx in row.shortlist:
            cand = row.group.candidates[int(cand_idx)]
            answer = int(cand.answer)
            counts[answer] = counts.get(answer, 0) + 1
            prior = float(cand.prior)
            if answer not in best_prior or prior > best_prior[answer][0]:
                best_prior[answer] = (prior, int(cand_idx))
        best_answer = sorted(counts, key=lambda ans: (counts[ans], best_prior[ans][0]), reverse=True)[0]
        out[group_key(row.group)] = int(best_prior[best_answer][1])
    return out


def group_difficulty(row: SelectedGroup) -> str:
    counts: Dict[int, int] = {}
    for cand_idx in row.shortlist:
        answer = int(row.group.candidates[int(cand_idx)].answer)
        counts[answer] = counts.get(answer, 0) + 1
    target = int(row.group.answer)
    target_count = counts.get(target, 0)
    max_wrong = max([count for answer, count in counts.items() if answer != target] or [0])
    return "easy" if target_count > max_wrong else "hard"


def binary_auc(labels: Sequence[bool], scores: Sequence[float]) -> float:
    pos = [float(s) for y, s in zip(labels, scores) if bool(y)]
    neg = [float(s) for y, s in zip(labels, scores) if not bool(y)]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    total = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
            total += 1.0
    return safe_div(wins, total)


def mean_group_auc(rows: Sequence[SelectedGroup], scores_by_key: Optional[Dict[str, torch.Tensor]]) -> float:
    if scores_by_key is None:
        return float("nan")
    aucs: List[float] = []
    for row in rows:
        key = group_key(row.group)
        if key not in scores_by_key:
            continue
        scores = [float(x) for x in scores_by_key[key].detach().cpu().tolist()]
        labels = [bool(row.group.candidates[int(idx)].answer_exact) for idx in row.shortlist]
        auc = binary_auc(labels, scores)
        if not math.isnan(auc):
            aucs.append(auc)
    return float(np.mean(aucs)) if aucs else float("nan")


def aggregate_metrics(rows: Sequence[SelectedGroup], selected: Dict[str, int], scores_by_key: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, float]:
    if not rows:
        return {}
    n = len(rows)
    correct = 0
    state_correct = 0
    base_correct = 0
    oracle_correct = 0
    shortlist_oracle = 0
    changed = 0
    damage = 0
    damage_den = 0
    recovery = 0
    recovery_den = 0
    no_op = 0
    for row in rows:
        group = row.group
        key = group_key(group)
        sel_idx = int(selected.get(key, int(group.base_index)))
        cand = group.candidates[sel_idx]
        base = group.candidates[int(group.base_index)]
        sel_correct = bool(cand.answer_exact)
        base_ok = bool(base.answer_exact)
        has_oracle = bool(group.answer_labels.any().item())
        has_short = any(bool(group.candidates[int(i)].answer_exact) for i in row.shortlist)
        correct += int(sel_correct)
        state_correct += int(bool(cand.state_exact))
        base_correct += int(base_ok)
        oracle_correct += int(has_oracle)
        shortlist_oracle += int(has_short)
        changed += int(sel_idx != int(group.base_index))
        no_op += int(sel_idx == int(group.base_index))
        if base_ok:
            damage_den += 1
            damage += int(not sel_correct)
        else:
            recovery_den += 1
            recovery += int(sel_correct)
    accuracy = correct / n
    base_acc = base_correct / n
    oracle_acc = oracle_correct / n
    short_oracle_acc = shortlist_oracle / n
    return {
        "n": float(n),
        "accuracy": accuracy,
        "state_accuracy": state_correct / n,
        "base_accuracy": base_acc,
        "oracle_accuracy": oracle_acc,
        "shortlist_oracle_accuracy": short_oracle_acc,
        "gap_capture": safe_div(accuracy - base_acc, oracle_acc - base_acc),
        "shortlist_gap_capture": safe_div(accuracy - base_acc, short_oracle_acc - base_acc),
        "coverage_capture": safe_div(accuracy, oracle_acc),
        "shortlist_coverage_capture": safe_div(accuracy, short_oracle_acc),
        "changed_fraction": changed / n,
        "no_op_fraction": no_op / n,
        "damage_rate": safe_div(damage, damage_den),
        "recovery_rate": safe_div(recovery, recovery_den),
        "ranking_auc": mean_group_auc(rows, scores_by_key),
    }


def metric_rows_for_arm(
    selected_rows: Sequence[SelectedGroup],
    selected: Dict[str, int],
    arm: str,
    seed: int,
    train_source: str,
    scores_by_key: Optional[Dict[str, torch.Tensor]] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    eval_rows = [row for row in selected_rows if row.role == "eval"]
    for split in sorted(set(row.group.split for row in eval_rows)):
        split_rows = [row for row in eval_rows if row.group.split == split]
        metrics = aggregate_metrics(split_rows, selected, scores_by_key)
        rows.append({"arm": arm, "seed": seed, "split": split, "difficulty": "all", "source_seed": "all", "source_holdout": "all", "train_source": train_source, **metrics})
        for difficulty in ["easy", "hard"]:
            diff_rows = [row for row in split_rows if group_difficulty(row) == difficulty]
            if diff_rows:
                rows.append({"arm": arm, "seed": seed, "split": split, "difficulty": difficulty, "source_seed": "all", "source_holdout": "all", "train_source": train_source, **aggregate_metrics(diff_rows, selected, scores_by_key)})
        for source_seed in sorted(set(int(row.group.source_seed) for row in split_rows)):
            sub = [row for row in split_rows if int(row.group.source_seed) == source_seed]
            rows.append({"arm": arm, "seed": seed, "split": split, "difficulty": "all", "source_seed": source_seed, "source_holdout": int(source_seed), "train_source": train_source, **aggregate_metrics(sub, selected, scores_by_key)})
    val_rows = [row for row in selected_rows if row.role == "val"]
    if val_rows:
        rows.append({"arm": arm, "seed": seed, "split": "val_mixed_L24", "difficulty": "all", "source_seed": "train_sources", "source_holdout": "train_sources", "train_source": train_source, **aggregate_metrics(val_rows, selected, scores_by_key)})
    return rows


def candidate_summary_rows(selected: Sequence[SelectedGroup]) -> List[Dict[str, Any]]:
    rows = []
    eval_and_val = selected
    for split in sorted(set(row.group.split for row in eval_and_val)):
        split_rows = [row for row in eval_and_val if row.group.split == split]
        for difficulty in ["all", "easy", "hard"]:
            diff_rows = split_rows if difficulty == "all" else [row for row in split_rows if group_difficulty(row) == difficulty]
            for source in ["all"] + sorted(set(int(row.group.source_seed) for row in diff_rows)):
                sub = diff_rows if source == "all" else [row for row in diff_rows if int(row.group.source_seed) == source]
                if not sub:
                    continue
                target_counts = []
                target_margins = []
                for row in sub:
                    counts: Dict[int, int] = {}
                    for idx in row.shortlist:
                        ans = int(row.group.candidates[int(idx)].answer)
                        counts[ans] = counts.get(ans, 0) + 1
                    target = int(row.group.answer)
                    wrong = [count for ans, count in counts.items() if ans != target]
                    target_counts.append(counts.get(target, 0))
                    target_margins.append(counts.get(target, 0) - (max(wrong) if wrong else 0))
                rows.append({
                    "split": split,
                    "difficulty": difficulty,
                    "source_seed": source,
                    "groups": len(sub),
                    "avg_full_candidates": float(np.mean([len(row.group.candidates) for row in sub])),
                    "avg_shortlist": float(np.mean([len(row.shortlist) for row in sub])),
                    "avg_target_answer_count": float(np.mean(target_counts)),
                    "avg_target_answer_margin": float(np.mean(target_margins)),
                    "base_accuracy": float(np.mean([bool(row.group.candidates[int(row.group.base_index)].answer_exact) for row in sub])),
                    "full_oracle_accuracy": float(np.mean([bool(row.group.answer_labels.any().item()) for row in sub])),
                    "shortlist_oracle_accuracy": float(np.mean([any(bool(row.group.candidates[int(i)].answer_exact) for i in row.shortlist) for row in sub])),
                    "shortlist_oracle_capture": safe_div(
                        float(np.mean([any(bool(row.group.candidates[int(i)].answer_exact) for i in row.shortlist) for row in sub])),
                        float(np.mean([bool(row.group.answer_labels.any().item()) for row in sub])),
                    ),
                })
    return rows


def make_figures(metrics: pd.DataFrame, candidate_summary: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    if metrics.empty:
        return
    std = metrics[(metrics["split"] == "standard_L24") & (metrics["source_seed"] == "all") & (metrics["difficulty"] == "all")].copy()
    if not std.empty:
        order = [
            "base", "oracle_full", "oracle_shortlist", "majority_answer",
            "zero_task_value", "zero_program_equiv", "zero_program_final", "zero_program_trace", "zero_trace_corrupt", "zero_candidate_only", "zero_prompt_only",
            "zero_task_value_gated", "zero_program_final_gated", "zero_program_trace_gated",
            "feature_mlp", "trace_transformer",
            "qwen_program_final", "qwen_program_final_gated", "qwen_program_trace", "qwen_program_trace_gated", "qwen_program_final_shuffled",
        ]
        std["arm"] = pd.Categorical(std["arm"], categories=order, ordered=True)
        std = std.sort_values(["arm", "seed"])
        agg = std.groupby("arm", observed=False)["accuracy"].agg(["mean", "std"]).reset_index().dropna(subset=["mean"])
        plt.figure(figsize=(11, 5))
        plt.bar(agg["arm"].astype(str), agg["mean"], yerr=agg["std"].fillna(0.0), color="#4c78a8")
        plt.xticks(rotation=35, ha="right")
        plt.ylabel("Exact answer accuracy")
        plt.title("Standard L24 Accuracy by Arm")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(FIGURES / "standard_accuracy_by_arm.png", dpi=160)
        plt.close()

        cap = std.groupby("arm", observed=False)["gap_capture"].agg(["mean", "std"]).reset_index().dropna(subset=["mean"])
        plt.figure(figsize=(11, 5))
        plt.axhline(0, color="#777", linewidth=0.8)
        plt.bar(cap["arm"].astype(str), cap["mean"], yerr=cap["std"].fillna(0.0), color="#f58518")
        plt.xticks(rotation=35, ha="right")
        plt.ylabel("Oracle gap capture")
        plt.title("Standard L24 Oracle-Gap Capture")
        plt.tight_layout()
        plt.savefig(FIGURES / "standard_gap_capture_by_arm.png", dpi=160)
        plt.close()

    eval_all = metrics[(metrics["source_seed"] == "all") & (metrics["difficulty"] == "all") & metrics["split"].str.contains("L24")].copy()
    focus = eval_all[eval_all["arm"].isin(["base", "oracle_full", "oracle_shortlist", "majority_answer", "zero_task_value", "zero_program_final", "zero_program_trace", "zero_trace_corrupt", "qwen_program_final", "qwen_program_final_gated", "qwen_program_trace", "qwen_program_trace_gated"])]
    if not focus.empty:
        pivot = focus.groupby(["split", "arm"], observed=False)["accuracy"].mean().reset_index()
        splits = list(sorted(pivot["split"].unique()))
        arms = list(pivot["arm"].unique())
        x = np.arange(len(splits))
        width = 0.8 / max(1, len(arms))
        plt.figure(figsize=(12, 5))
        for i, arm in enumerate(arms):
            vals = [float(pivot[(pivot["split"] == split) & (pivot["arm"] == arm)]["accuracy"].mean()) if not pivot[(pivot["split"] == split) & (pivot["arm"] == arm)].empty else 0.0 for split in splits]
            plt.bar(x + (i - len(arms) / 2) * width + width / 2, vals, width=width, label=arm)
        plt.xticks(x, splits, rotation=25, ha="right")
        plt.ylabel("Exact answer accuracy")
        plt.title("Accuracy Across Splits")
        plt.legend(fontsize=8, ncols=2)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(FIGURES / "accuracy_by_split.png", dpi=160)
        plt.close()

    if not candidate_summary.empty:
        std_cov = candidate_summary[(candidate_summary["split"] == "standard_L24") & (candidate_summary["difficulty"] == "all") & (candidate_summary["source_seed"] != "all")]
        if not std_cov.empty:
            plt.figure(figsize=(7, 4))
            x = np.arange(len(std_cov))
            plt.bar(x - 0.2, std_cov["base_accuracy"], width=0.2, label="base")
            plt.bar(x, std_cov["shortlist_oracle_accuracy"], width=0.2, label="shortlist oracle")
            plt.bar(x + 0.2, std_cov["full_oracle_accuracy"], width=0.2, label="full oracle")
            plt.xticks(x, [str(v) for v in std_cov["source_seed"]])
            plt.ylabel("Accuracy / coverage")
            plt.xlabel("Source seed")
            plt.title("Standard L24 Candidate Coverage by Source")
            plt.legend()
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig(FIGURES / "coverage_by_source_seed.png", dpi=160)
            plt.close()

        diff_cov = candidate_summary[(candidate_summary["split"] == "standard_L24") & (candidate_summary["source_seed"] == "all") & (candidate_summary["difficulty"].isin(["easy", "hard"]))]
        if not diff_cov.empty:
            plt.figure(figsize=(6, 4))
            x = np.arange(len(diff_cov))
            plt.bar(x - 0.2, diff_cov["base_accuracy"], width=0.2, label="base")
            plt.bar(x, diff_cov["shortlist_oracle_accuracy"], width=0.2, label="shortlist oracle")
            plt.bar(x + 0.2, diff_cov["full_oracle_accuracy"], width=0.2, label="full oracle")
            plt.xticks(x, [str(v) for v in diff_cov["difficulty"]])
            plt.ylabel("Accuracy / coverage")
            plt.title("Standard L24 Coverage by Difficulty")
            plt.legend()
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig(FIGURES / "coverage_by_difficulty.png", dpi=160)
            plt.close()


def markdown_table(df: pd.DataFrame, columns: Sequence[str], max_rows: int = 40) -> str:
    if df.empty:
        return "_No rows._\n"
    sub = df.loc[:, list(columns)].head(max_rows).copy()
    percent_markers = ("accuracy", "capture", "fraction", "rate")
    for col in sub.columns:
        if pd.api.types.is_float_dtype(sub[col]):
            if any(marker in col for marker in percent_markers):
                sub[col] = sub[col].map(lambda x: pct(float(x)) if not pd.isna(x) else "n/a")
            else:
                sub[col] = sub[col].map(lambda x: f"{float(x):.3f}" if not pd.isna(x) else "n/a")
    return sub.to_markdown(index=False) + "\n"


def make_report(run_dir: Path, args: argparse.Namespace, metrics_rows: Sequence[Dict[str, Any]], candidate_rows: Sequence[Dict[str, Any]], train_logs: Sequence[Dict[str, Any]]) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(metrics_rows)
    cand = pd.DataFrame(candidate_rows)
    logs = pd.DataFrame(train_logs)
    write_csv(REPORTS / "metrics.csv", metrics_rows)
    write_csv(REPORTS / "candidate_summary.csv", candidate_rows)
    write_csv(REPORTS / "training_log.csv", train_logs)
    make_figures(metrics, cand)

    std = metrics[(metrics["split"] == "standard_L24") & (metrics["source_seed"] == "all") & (metrics["difficulty"] == "all")].copy()
    if not std.empty:
        headline = std.groupby("arm", observed=False).agg(
            mean_accuracy=("accuracy", "mean"),
            mean_gap_capture=("gap_capture", "mean"),
            mean_auc=("ranking_auc", "mean"),
            mean_changed=("changed_fraction", "mean"),
            mean_damage=("damage_rate", "mean"),
            mean_recovery=("recovery_rate", "mean"),
        ).reset_index()
    else:
        headline = pd.DataFrame()

    md: List[str] = []
    md.append("# Qwen Readable Candidate Verifier Report\n")
    md.append("## Summary\n")
    md.append(
        "This standalone experiment tests whether a Qwen reader can select executable repair candidates when the task and "
        "candidate are rendered as readable pseudocode, claimed final values, and optional execution traces. Learned selectors "
        "are trained only from offline labels; at inference they do not receive the target answer or target state.\n"
    )
    if not headline.empty:
        base_rows = headline[headline["arm"] == "base"]
        short_rows = headline[headline["arm"] == "oracle_shortlist"]
        full_rows = headline[headline["arm"] == "oracle_full"]
        scored = headline[~headline["arm"].isin(["base", "oracle_full", "oracle_shortlist"])]
        if not base_rows.empty and not scored.empty:
            base_acc = float(base_rows.iloc[0]["mean_accuracy"])
            best_acc = float(scored["mean_accuracy"].max())
            tied = sorted(scored[scored["mean_accuracy"] >= best_acc - 1e-12]["arm"].astype(str).tolist())
            short_acc = float(short_rows.iloc[0]["mean_accuracy"]) if not short_rows.empty else float("nan")
            full_acc = float(full_rows.iloc[0]["mean_accuracy"]) if not full_rows.empty else float("nan")
            if best_acc <= base_acc + 1e-12:
                md.append(
                    f"On `standard_L24`, no non-oracle selector improved over the no-repair base policy: "
                    f"base was {pct(base_acc)}, the best tied selector accuracy was {pct(best_acc)} "
                    f"({', '.join(tied[:5])}), deployable-shortlist oracle accuracy was {pct(short_acc)}, "
                    f"and full-pool oracle accuracy was {pct(full_acc)}.\n"
                )
            else:
                best_rows = scored[scored["mean_accuracy"] >= best_acc - 1e-12]
                best_gap = float(best_rows["mean_gap_capture"].max())
                best_arms = ", ".join(sorted(best_rows["arm"].astype(str).tolist())[:5])
                md.append(
                    f"On `standard_L24`, the best non-oracle selector was `{best_arms}` at {pct(best_acc)} accuracy "
                    f"and {pct(best_gap)} oracle-gap capture versus {pct(base_acc)} for no repair.\n"
                )
            best_auc = scored.sort_values("mean_auc", ascending=False).head(1)
            if not best_auc.empty and not pd.isna(best_auc.iloc[0]["mean_auc"]):
                md.append(f"The best mean ranking AUC on `standard_L24` was `{best_auc.iloc[0]['arm']}` at {float(best_auc.iloc[0]['mean_auc']):.3f}.\n")
            frozen_equiv = headline[headline["arm"] == "zero_program_equiv"]
            frozen_trace = headline[headline["arm"] == "zero_program_trace"]
            corrupt_trace = headline[headline["arm"] == "zero_trace_corrupt"]
            if not frozen_equiv.empty and not frozen_trace.empty and not corrupt_trace.empty:
                md.append(
                    f"Frozen program-equivalence reading reached {pct(float(frozen_equiv.iloc[0]['mean_accuracy']))} accuracy "
                    f"with AUC {float(frozen_equiv.iloc[0]['mean_auc']):.3f}; adding full trace text dropped to "
                    f"{pct(float(frozen_trace.iloc[0]['mean_accuracy']))} with AUC {float(frozen_trace.iloc[0]['mean_auc']):.3f}, "
                    f"while corrupted trace dropped to {pct(float(corrupt_trace.iloc[0]['mean_accuracy']))} with AUC {float(corrupt_trace.iloc[0]['mean_auc']):.3f}.\n"
                )
    md.append("## Setup\n")
    md.append(f"- Base reader: `{args.model_id}`.\n")
    md.append(f"- Candidate shortlist size: `{args.shortlist_size}`.\n")
    md.append(f"- Held-out source seed: `{args.heldout_source_seed}`.\n")
    md.append(f"- Train groups per non-held-out source: `{args.train_per_source}`; validation groups per non-held-out source: `{args.val_per_source}`; eval groups per source/split: `{args.eval_per_source}`.\n")
    md.append(f"- Frozen readable modes: `{args.embedding_modes}`.\n")
    md.append(f"- Trained Qwen head modes: `{args.train_qwen_modes}`.\n")
    md.append("\nSelectors:\n")
    md.append("- `base`: no repair.\n")
    md.append("- `oracle_full`: best answer-correct candidate in the full candidate pool.\n")
    md.append("- `oracle_shortlist`: best answer-correct candidate in the deployable shortlist.\n")
    md.append("- `majority_answer`: chooses the most common claimed final value in the shortlist.\n")
    md.append("- `zero_task_value`: frozen Qwen reads the task program and scores candidates by the logit of each candidate's claimed final value.\n")
    md.append("- `zero_program_equiv`: frozen Qwen reads task program and candidate program, then answers whether they return the same value.\n")
    md.append("- `zero_program_final`: frozen Qwen reads task program, candidate program, and candidate claimed final value.\n")
    md.append("- `zero_program_trace`: frozen Qwen also receives a readable execution trace.\n")
    md.append("- `zero_trace_corrupt`: same as trace mode but with corrupted state order.\n")
    md.append("- `qwen_<mode>`: learned groupwise ranking head over frozen Qwen embeddings for that readable mode.\n")
    md.append("- `_gated`: validation-tuned base fallback for the corresponding score.\n")
    md.append("- Additional controls: prompt-only, candidate-only, feature-only, trace-only, and shuffled-label arms.\n")
    md.append("\n## Candidate Coverage\n")
    md.append(markdown_table(cand[(cand["source_seed"].astype(str) == "all") & (cand["difficulty"] == "all")], ["split", "groups", "avg_full_candidates", "avg_shortlist", "avg_target_answer_count", "avg_target_answer_margin", "base_accuracy", "shortlist_oracle_accuracy", "full_oracle_accuracy", "shortlist_oracle_capture"]))
    md.append("![Coverage by source seed](figures/coverage_by_source_seed.png)\n")
    md.append("![Coverage by difficulty](figures/coverage_by_difficulty.png)\n")
    md.append("\n## Standard L24 Gate\n")
    md.append(markdown_table(headline, ["arm", "mean_accuracy", "mean_gap_capture", "mean_auc", "mean_changed", "mean_damage", "mean_recovery"], max_rows=80))
    md.append("![Standard accuracy](figures/standard_accuracy_by_arm.png)\n")
    md.append("![Standard gap capture](figures/standard_gap_capture_by_arm.png)\n")
    md.append("\n## Easy/Hard Standard L24 Readout\n")
    diff_rows = metrics[(metrics["source_seed"] == "all") & (metrics["split"] == "standard_L24") & (metrics["difficulty"].isin(["easy", "hard"])) & (metrics["arm"].isin(["base", "oracle_shortlist", "majority_answer", "zero_task_value", "zero_program_final", "zero_program_trace", "zero_trace_corrupt", "qwen_program_final", "qwen_program_final_gated", "qwen_program_trace", "qwen_program_trace_gated"]))]
    md.append(markdown_table(diff_rows, ["difficulty", "arm", "seed", "n", "accuracy", "gap_capture", "ranking_auc", "changed_fraction", "damage_rate", "recovery_rate"], max_rows=100))
    md.append("\n## Split Results\n")
    split_rows = metrics[(metrics["source_seed"] == "all") & (metrics["difficulty"] == "all") & metrics["arm"].isin(["base", "oracle_full", "oracle_shortlist", "majority_answer", "zero_task_value", "zero_program_equiv", "zero_program_final", "zero_program_trace", "zero_trace_corrupt", "zero_task_value_gated", "zero_program_final_gated", "zero_program_trace_gated", "qwen_program_final", "qwen_program_final_gated", "qwen_program_trace", "qwen_program_trace_gated"])]
    md.append(markdown_table(split_rows, ["split", "arm", "seed", "accuracy", "gap_capture", "ranking_auc", "changed_fraction", "damage_rate", "recovery_rate"], max_rows=120))
    md.append("![Accuracy by split](figures/accuracy_by_split.png)\n")
    md.append("\n## Held-Out Source Readout\n")
    held = metrics[(metrics["source_seed"].astype(str) == str(args.heldout_source_seed)) & (metrics["split"] == "standard_L24") & (metrics["difficulty"] == "all")]
    md.append(markdown_table(held, ["arm", "seed", "accuracy", "base_accuracy", "oracle_accuracy", "gap_capture", "ranking_auc", "changed_fraction", "damage_rate", "recovery_rate"], max_rows=100))
    md.append("\n## Training Dynamics\n")
    if not logs.empty:
        final_logs = logs.sort_values("epoch").groupby(["arm", "seed"], as_index=False).tail(1)
        md.append(markdown_table(final_logs, [c for c in ["arm", "seed", "epoch", "train_loss", "val_accuracy", "val_damage_rate", "val_recovery_rate", "val_utility"] if c in final_logs.columns], max_rows=80))
    else:
        md.append("_No learned-arm training logs._\n")
    md.append("\n## Interpretation\n")
    md.append(
        "The decisive measurement is not raw accuracy alone. The report separates candidate coverage from selector capture, "
        "because a selector cannot choose a correct program that is absent from its candidate shortlist. A useful selector should "
        "capture a stable fraction of the available oracle gap, rank correct candidates above wrong candidates, make nontrivial repairs, "
        "and avoid damaging already-correct base programs. Easy and hard rows test whether failures are caused by representation readability "
        "or by candidate sets where the target answer is not distinguishable from simple shortlist statistics. In this run the easy/hard split "
        "is itself diagnostic: only one `standard_L24` group was easy under the claimed-output frequency criterion, so most of the evaluated "
        "repair decisions sit in the hard regime.\n"
    )
    md.append("\n## Artifacts\n")
    md.append(f"- Run directory: `{run_dir}`\n")
    md.append(f"- Large embeddings: `{EMBED_ROOT / (args.embedding_run_name if args.embedding_run_name else args.run_name)}`\n")
    md.append(f"- Large checkpoints: `{CHECKPOINT_ROOT / args.run_name}`\n")
    md.append(f"- Metrics CSV: `{REPORTS / 'metrics.csv'}`\n")
    md.append(f"- Candidate summary CSV: `{REPORTS / 'candidate_summary.csv'}`\n")

    md_text = "\n".join(md)
    report_md = REPORTS / "qwen_readable_candidate_verifier_report.md"
    report_md.write_text(md_text)

    html_text = markdown_to_html(md_text, "Qwen Readable Candidate Verifier")
    (REPORTS / "qwen_readable_candidate_verifier_report.html").write_text(html_text)


def markdown_to_html(md: str, title: str) -> str:
    lines = md.splitlines()
    out = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title>",
        "<style>body{font-family:Inter,Arial,sans-serif;max-width:1120px;margin:32px auto;line-height:1.45;color:#1f2933}table{border-collapse:collapse;font-size:12px;margin:12px 0;display:block;overflow-x:auto}td,th{border:1px solid #cfd8dc;padding:5px 7px}th{background:#eef3f7}code{background:#f4f6f8;padding:1px 4px;border-radius:4px}img{max-width:100%;border:1px solid #d8dee6;margin:14px 0}li{margin:4px 0}</style>",
        "</head><body>",
    ]
    in_table = False
    table_lines: List[str] = []

    def flush_table() -> None:
        nonlocal table_lines, in_table
        if not table_lines:
            return
        header = [c.strip() for c in table_lines[0].strip("|").split("|")]
        out.append("<table><thead><tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in header) + "</tr></thead><tbody>")
        for row in table_lines[2:]:
            cells = [c.strip() for c in row.strip("|").split("|")]
            out.append("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in cells) + "</tr>")
        out.append("</tbody></table>")
        table_lines = []
        in_table = False

    for line in lines:
        if line.startswith("|"):
            in_table = True
            table_lines.append(line)
            continue
        if in_table:
            flush_table()
        if line.startswith("# "):
            out.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            out.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("!["):
            m = re.match(r"!\[(.*?)\]\((.*?)\)", line)
            if m:
                out.append(f"<img alt='{html.escape(m.group(1))}' src='{html.escape(m.group(2))}'>")
        elif line.startswith("- "):
            out.append(f"<p>&bull; {html.escape(line[2:])}</p>")
        elif not line.strip():
            out.append("")
        else:
            escaped = html.escape(line)
            escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
            out.append(f"<p>{escaped}</p>")
    if in_table:
        flush_table()
    out.append("</body></html>")
    return "\n".join(out)


def run(args: argparse.Namespace) -> None:
    start = time.time()
    for path in [RUNS, REPORTS, FIGURES, EMBED_ROOT, CHECKPOINT_ROOT]:
        path.mkdir(parents=True, exist_ok=True)
    run_dir = RUNS / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "run_config.json", {"args": vars(args), "platform": platform.platform(), "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})

    tail = load_module("tail_repair_source_for_qwen_reader", TAIL_MODULE_PATH)
    source = load_module("source_compiler_for_prompt_rebuild", SOURCE_MODULE_PATH)
    source_runs = parse_csv_list(args.source_runs)
    groups = load_tail_groups(tail, source_runs, args.cache_run_name)

    tokenizer_for_prompts = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True, use_fast=True)
    ensure_pad_token(tokenizer_for_prompts)
    prompt_map = build_prompt_map(source, tokenizer_for_prompts, args)
    selected = select_groups(groups, prompt_map, args)
    write_json(run_dir / "selection_manifest.json", {
        "selected_groups": len(selected),
        "roles": {role: sum(1 for row in selected if row.role == role) for role in sorted(set(row.role for row in selected))},
        "source_runs": source_runs,
        "heldout_source_seed": args.heldout_source_seed,
    })
    candidate_rows = candidate_summary_rows(selected)
    write_csv(run_dir / "candidate_summary.csv", candidate_rows)

    embedding_modes = parse_csv_list(args.embedding_modes)
    cache_name = args.embedding_run_name if args.embedding_run_name else args.run_name
    missing_embedding = any(not (EMBED_ROOT / cache_name / f"{mode}_embeddings.pt").exists() for mode in embedding_modes)
    tokenizer = qwen = device = None
    if missing_embedding or args.rebuild_embeddings:
        tokenizer, qwen, device = load_qwen_reader(args)
    stores = {
        mode: build_or_load_embeddings(selected, mode, tokenizer, qwen, device, args, run_dir)
        for mode in embedding_modes
    }
    if qwen is not None:
        del qwen
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics_rows: List[Dict[str, Any]] = []
    train_logs: List[Dict[str, Any]] = []
    ckpt_rows: List[Dict[str, Any]] = []

    base_sel = select_base(selected)
    oracle_full = select_oracle(selected, shortlist_only=False)
    oracle_short = select_oracle(selected, shortlist_only=True)
    majority = select_majority_answer(selected)
    for arm, sel in [("base", base_sel), ("oracle_full", oracle_full), ("oracle_shortlist", oracle_short), ("majority_answer", majority)]:
        metrics_rows.extend(metric_rows_for_arm(selected, sel, arm, -1, "fixed"))

    zero_scores_by_mode: Dict[str, Dict[str, torch.Tensor]] = {}
    for mode in embedding_modes:
        scores = {group_key(row.group): stores[mode][group_key(row.group)]["zero_score"] for row in selected}
        zero_scores_by_mode[mode] = scores
        arm = f"zero_{mode}"
        sel = select_from_scores(selected, scores)
        metrics_rows.extend(metric_rows_for_arm(selected, sel, arm, -1, "fixed", scores))
        margin, margin_metrics = tune_base_margin(selected, scores)
        gated_arm = f"{arm}_gated"
        gated_sel = select_with_base_margin(selected, scores, margin)
        metrics_rows.extend(metric_rows_for_arm(selected, gated_sel, gated_arm, -1, "fixed", scores))
        ckpt_rows.append({"arm": gated_arm, "seed": -1, "margin": margin, **{f"val_{k}": v for k, v in margin_metrics.items()}})

    for seed in parse_int_list(args.critic_seeds):
        if int(args.run_feature_controls):
            feature, ckpt, logs, aux = train_feature_head(selected, tail, args, "feature_mlp", seed)
            ckpt_rows.append(ckpt)
            train_logs.extend(logs)
            feature_scores = scores_with_model(selected, {}, feature, "feature", (tail, aux[0], aux[1]), args)
            sel = select_from_scores(selected, feature_scores)
            metrics_rows.extend(metric_rows_for_arm(selected, sel, "feature_mlp", seed, "train_sources", feature_scores))

            trace, ckpt, logs = train_trace_head(selected, args, "trace_transformer", seed, corrupt_trace=False)
            ckpt_rows.append(ckpt)
            train_logs.extend(logs)
            trace_scores = scores_with_model(selected, {}, trace, "trace", None, args)
            sel = select_from_scores(selected, trace_scores)
            metrics_rows.extend(metric_rows_for_arm(selected, sel, "trace_transformer", seed, "train_sources", trace_scores))

        for mode in parse_csv_list(args.train_qwen_modes):
            if mode not in stores:
                raise ValueError(f"train_qwen_modes includes mode without embeddings: {mode}")
            arm = f"qwen_{mode}"
            qrank, ckpt, logs = train_qwen_head(selected, stores[mode], args, run_dir, arm, seed, echo=False, shuffled=False)
            ckpt_rows.append(ckpt)
            train_logs.extend(logs)
            qrank_scores = scores_with_model(selected, stores[mode], qrank, "qwen", None, args)
            sel = select_from_scores(selected, qrank_scores)
            metrics_rows.extend(metric_rows_for_arm(selected, sel, arm, seed, "train_sources", qrank_scores))
            margin, margin_metrics = tune_base_margin(selected, qrank_scores)
            gated_arm = f"{arm}_gated"
            ckpt_rows.append({"arm": gated_arm, "seed": seed, "margin": margin, **{f"val_{k}": v for k, v in margin_metrics.items()}})
            metrics_rows.extend(metric_rows_for_arm(selected, select_with_base_margin(selected, qrank_scores, margin), gated_arm, seed, "train_sources", qrank_scores))

        if int(args.run_shuffled_control):
            mode = parse_csv_list(args.train_qwen_modes)[0]
            arm = f"qwen_{mode}_shuffled"
            qshuf, ckpt, logs = train_qwen_head(selected, stores[mode], args, run_dir, arm, seed, echo=False, shuffled=True)
            ckpt_rows.append(ckpt)
            train_logs.extend(logs)
            shuf_scores = scores_with_model(selected, stores[mode], qshuf, "qwen", None, args)
            sel = select_from_scores(selected, shuf_scores)
            metrics_rows.extend(metric_rows_for_arm(selected, sel, arm, seed, "train_sources", shuf_scores))

    write_csv(run_dir / "metrics.csv", metrics_rows)
    write_csv(run_dir / "training_log.csv", train_logs)
    write_json(run_dir / "checkpoint_manifest.json", ckpt_rows)
    make_report(run_dir, args, metrics_rows, candidate_rows, train_logs)
    write_json(run_dir / "run_summary.json", {"elapsed_sec": round(time.time() - start, 3), "metrics_rows": len(metrics_rows), "train_log_rows": len(train_logs), "selected_groups": len(selected)})


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", default="main_qwen_readable_candidate_verifier")
    p.add_argument("--suite", choices=["smoke", "main"], default="smoke")
    p.add_argument("--model_id", default="Qwen/Qwen3-4B")
    p.add_argument("--source_runs", default="main_expand_copy_seed123,main_expand_copy_seed456,main_expand_copy_seed789")
    p.add_argument("--cache_run_name", default="main_tail_repair_critic_v1")
    p.add_argument("--dataset_seed", type=int, default=4242)
    p.add_argument("--cache_train_examples", type=int, default=160)
    p.add_argument("--cache_val_examples", type=int, default=48)
    p.add_argument("--cache_eval_examples", type=int, default=64)
    p.add_argument("--cache_paired_eval_pairs", type=int, default=32)
    p.add_argument("--modulus", type=int, default=97)
    p.add_argument("--max_steps", type=int, default=24)
    p.add_argument("--shortlist_size", type=int, default=64)
    p.add_argument("--train_per_source", type=int, default=32)
    p.add_argument("--val_per_source", type=int, default=12)
    p.add_argument("--eval_per_source", type=int, default=24)
    p.add_argument("--eval_splits", default="standard_L24,paraphrase_L24,heldout_L24,paired_L24,paired_heldout_L24")
    p.add_argument("--heldout_source_seed", type=int, default=789)
    p.add_argument("--selection_seed", type=int, default=17)
    p.add_argument("--shuffle_group_selection", action="store_true")
    p.add_argument("--critic_seeds", default="101")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--head_width", type=int, default=256)
    p.add_argument("--trace_width", type=int, default=128)
    p.add_argument("--trace_layers", type=int, default=2)
    p.add_argument("--trace_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--echo_weight", type=float, default=0.2)
    p.add_argument("--no_positive_base_weight", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--tail_detail_window", type=int, default=8)
    p.add_argument("--qwen_batch_size", type=int, default=8)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--torch_dtype", default="bf16")
    p.add_argument("--load_in_4bit", type=int, default=1)
    p.add_argument("--device_map", default="auto")
    p.add_argument("--rebuild_embeddings", action="store_true")
    p.add_argument("--embedding_run_name", default="")
    p.add_argument("--embedding_modes", default="task_value,program_equiv,program_final,program_trace,trace_corrupt,candidate_only,prompt_only")
    p.add_argument("--train_qwen_modes", default="program_final,program_trace")
    p.add_argument("--run_feature_controls", type=int, default=1)
    p.add_argument("--run_shuffled_control", type=int, default=1)
    return p


def apply_suite_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.suite == "smoke":
        if args.run_name == "main_qwen_readable_candidate_verifier":
            args.run_name = "smoke_qwen_readable_candidate_verifier"
        args.shortlist_size = min(args.shortlist_size, 8)
        args.train_per_source = min(args.train_per_source, 4)
        args.val_per_source = min(args.val_per_source, 2)
        args.eval_per_source = min(args.eval_per_source, 4)
        args.eval_splits = "standard_L24,paired_L24"
        args.epochs = min(args.epochs, 2)
        args.critic_seeds = parse_csv_list(args.critic_seeds)[0]
        args.qwen_batch_size = min(args.qwen_batch_size, 4)
        args.max_length = min(args.max_length, 768)
        args.embedding_modes = "task_value,program_final,program_trace,trace_corrupt,candidate_only,prompt_only"
        args.train_qwen_modes = "program_final"
    return args


def main() -> None:
    args = build_arg_parser().parse_args()
    args.load_in_4bit = bool(args.load_in_4bit)
    args.run_feature_controls = bool(args.run_feature_controls)
    args.run_shuffled_control = bool(args.run_shuffled_control)
    args = apply_suite_defaults(args)
    run(args)


if __name__ == "__main__":
    main()
