#!/usr/bin/env python3
"""Small frozen-Qwen direct-answer baseline for public PROSE transformations."""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path("/workspace/experiments/qwen_public_prose_abi_gate")
ANALYSIS = ROOT / "analysis"
RUNS = ROOT / "runs"
CACHE_DIR = Path("/workspace/.cache/huggingface")
MODEL_NAME = "Qwen/Qwen3-4B"

sys.path.insert(0, str(ROOT / "src"))
from qwen_public_prose_abi_gate import load_tasks, split_examples  # noqa: E402


@dataclass
class PromptCase:
    task_id: str
    family: str
    features: str
    abi_covered: bool
    train_pairs: List[Tuple[Tuple[str, ...], str]]
    query: Tuple[str, ...]
    target: str


def render_inputs(vals: Sequence[str]) -> str:
    if len(vals) == 1:
        return vals[0]
    return " | ".join(f"col{i}={v}" for i, v in enumerate(vals))


def make_prompt(case: PromptCase) -> str:
    lines = [
        "Infer the text transformation from the examples.",
        "Return only the transformed output for the query. Do not explain.",
        "",
        "Examples:",
    ]
    for inp, out in case.train_pairs:
        lines.append(f"Input: {render_inputs(inp)}")
        lines.append(f"Output: {out}")
    lines.append("")
    lines.append("Query:")
    lines.append(f"Input: {render_inputs(case.query)}")
    lines.append("Output:")
    return "\n".join(lines)


def clean_prediction(text: str) -> str:
    text = text.strip()
    text = re.sub(r"(?is)<think>.*?</think>", "", text).strip()
    if text.lower().startswith("output:"):
        text = text.split(":", 1)[1].strip()
    first = text.splitlines()[0].strip() if text else ""
    if (first.startswith('"') and first.endswith('"')) or (first.startswith("'") and first.endswith("'")):
        first = first[1:-1].strip()
    return first


def load_model() -> Tuple[Any, Any]:
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=str(CACHE_DIR), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=str(CACHE_DIR),
        trust_remote_code=True,
        quantization_config=bnb,
        device_map="auto",
    )
    model.eval()
    return tok, model


def build_cases(sample_n: int, seed: int, train_n: int) -> List[PromptCase]:
    details = pd.read_csv(ANALYSIS / "details.csv")
    best = details[details["tier"] == "concat"].copy()
    best = best.sample(frac=1.0, random_state=seed)
    selected = best.head(sample_n)
    task_map = {t.task_id: t for t in load_tasks(limit=None, min_examples=5)}
    cases: List[PromptCase] = []
    for row in selected.itertuples(index=False):
        task = task_map[row.task_id]
        train, heldout = split_examples(task, train_n=train_n, heldout_cap=1)
        if not heldout:
            continue
        cases.append(
            PromptCase(
                task_id=row.task_id,
                family=row.family,
                features=row.features,
                abi_covered=bool(row.covered),
                train_pairs=[(e.inputs, e.output) for e in train],
                query=heldout[0].inputs,
                target=heldout[0].output,
            )
        )
    return cases


@torch.inference_mode()
def run_cases(cases: List[PromptCase], tok: Any, model: Any, max_new_tokens: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i, case in enumerate(cases, start=1):
        prompt = make_prompt(case)
        messages = [
            {"role": "system", "content": "You are a precise text transformation function."},
            {"role": "user", "content": prompt},
        ]
        try:
            rendered = tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = tok(rendered, return_tensors="pt").to(model.device)
        out = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
        gen = tok.decode(out[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True)
        pred = clean_prediction(gen)
        exact = pred == case.target
        rows.append(
            {
                "task_id": case.task_id,
                "family": case.family,
                "features": case.features,
                "abi_covered": case.abi_covered,
                "target": case.target,
                "prediction": pred,
                "raw_generation": gen,
                "exact": exact,
                "query": render_inputs(case.query),
            }
        )
        if i == 1 or i % 10 == 0 or i == len(cases):
            print(f"{i}/{len(cases)} exact_so_far={sum(r['exact'] for r in rows)}/{len(rows)}", flush=True)
    return rows


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    parts = [
        {"slice": "overall", "tasks": len(rows), "exact": float(rows["exact"].mean()) if len(rows) else 0.0},
    ]
    for covered, sub in rows.groupby("abi_covered"):
        parts.append(
            {
                "slice": f"abi_covered={bool(covered)}",
                "tasks": len(sub),
                "exact": float(sub["exact"].mean()) if len(sub) else 0.0,
            }
        )
    for fam, sub in rows.groupby("family"):
        if len(sub) >= 3:
            parts.append({"slice": f"family={fam}", "tasks": len(sub), "exact": float(sub["exact"].mean())})
    return pd.DataFrame(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20260627)
    ap.add_argument("--train-n", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    args = ap.parse_args()

    RUNS.mkdir(parents=True, exist_ok=True)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    started = time.time()
    cases = build_cases(args.sample_n, args.seed, args.train_n)
    print(f"Loaded {len(cases)} cases from sample_n={args.sample_n}", flush=True)
    tok, model = load_model()
    rows = pd.DataFrame(run_cases(cases, tok, model, args.max_new_tokens))
    rows.to_csv(ANALYSIS / "qwen_direct_sample.csv", index=False)
    summary = summarize(rows)
    summary.to_csv(ANALYSIS / "qwen_direct_summary.csv", index=False)
    meta = {
        "model": MODEL_NAME,
        "sample_n": args.sample_n,
        "actual_n": len(rows),
        "seed": args.seed,
        "train_n": args.train_n,
        "max_new_tokens": args.max_new_tokens,
        "elapsed_sec": round(time.time() - started, 2),
    }
    (RUNS / "qwen_direct_sample_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(summary.to_string(index=False))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
