#!/usr/bin/env python3
"""Evaluate native Qwen direct-answer behavior on the VM task splits."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from qwen_vm_agent_echo_qlora_experiment import make_splits
from typed_bytecode_core import RUNS, TaskExample, set_seed


INT_RE = re.compile(r"-?\d+")


def parse_answer(text: str) -> Optional[int]:
    for match in INT_RE.finditer(text):
        value = int(match.group(0))
        if 0 <= value <= 96:
            return value
    return None


def prompt_for(example: TaskExample) -> str:
    return (
        "Answer the task with only one integer from 0 to 96. Do not explain.\n"
        f"Task: {example.prompt}\n"
        "Answer:"
    )


@torch.no_grad()
def evaluate_split(model: Any, tokenizer: Any, examples: List[TaskExample], max_new_tokens: int) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    correct = parsed = 0
    for idx, ex in enumerate(examples):
        prompt = prompt_for(ex)
        enc = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(
            **enc,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(out[0, enc["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        pred = parse_answer(text)
        parsed += int(pred is not None)
        ok = pred == int(ex.answer)
        correct += int(ok)
        rows.append({"idx": idx, "prompt": ex.prompt, "answer": ex.answer, "raw": text, "pred": pred if pred is not None else "", "correct": int(ok)})
    n = max(1, len(examples))
    return {"n": len(examples), "accuracy": correct / n, "parse_rate": parsed / n}, rows


def write_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
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


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_name", required=True)
    p.add_argument("--model_name", default="Qwen/Qwen3-4B")
    p.add_argument("--max_new_tokens", type=int, default=16)
    return p


def main() -> None:
    args = parser().parse_args()
    run_dir = RUNS / args.run_name
    manifest = json.loads((run_dir / "dataset_manifest.json").read_text())
    split_args = argparse.Namespace(**manifest["args"])
    set_seed(int(split_args.seed))
    splits = make_splits(split_args)
    eval_splits = ["val_mixed", "fresh_standard", "fresh_paraphrase", "fresh_paired", "hard_composition"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True, quantization_config=quant, device_map="auto")
    model.eval()

    metric_rows: List[Dict[str, Any]] = []
    sample_rows: List[Dict[str, Any]] = []
    for split in eval_splits:
        metrics, rows = evaluate_split(model, tokenizer, splits[split], args.max_new_tokens)
        metrics.update({"run": args.run_name, "split": split, "variant": "native_qwen_direct"})
        metric_rows.append(metrics)
        for row in rows[:5]:
            row = dict(row)
            row.update({"run": args.run_name, "split": split, "variant": "native_qwen_direct"})
            sample_rows.append(row)
        print(f"[native] split={split} accuracy={metrics['accuracy']:.4f} parse={metrics['parse_rate']:.4f}", flush=True)
    write_rows(run_dir / "native_qwen_metrics.csv", metric_rows)
    write_rows(run_dir / "native_qwen_samples.csv", sample_rows)


if __name__ == "__main__":
    main()
