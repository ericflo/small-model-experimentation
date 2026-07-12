#!/usr/bin/env python3
"""Exploratory entropy/varentropy audit at recovery plan and action seams."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def distribution_metrics(logits: torch.Tensor, target: int) -> dict:
    values = logits.float()
    log_probs = torch.log_softmax(values, dim=-1)
    probs = log_probs.exp()
    surprisal = -log_probs
    entropy = (probs * surprisal).sum()
    varentropy = (probs * (surprisal - entropy).square()).sum()
    target_logprob = log_probs[target]
    return {
        "entropy_nats": float(entropy.item()),
        "varentropy_nats2": float(varentropy.item()),
        "target_logprob": float(target_logprob.item()),
        "target_rank": int((values > values[target]).sum().item()) + 1,
        "target_token_id": int(target),
    }


def seam_inputs(row: dict, tokenizer) -> dict[str, tuple[list[int], int]]:
    prompt = tokenizer.apply_chat_template(
        row["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    think = row["think"].strip() + "\n</think>\n\n"
    answer = row["answer"].strip() + tokenizer.eos_token
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    plan_ids = tokenizer(prompt + think, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt + think + answer, add_special_tokens=False)["input_ids"]
    if plan_ids[:len(prompt_ids)] != prompt_ids or full_ids[:len(plan_ids)] != plan_ids:
        raise AssertionError(f"seam token boundary merged for {row['id']}")
    return {
        "plan": (prompt_ids, plan_ids[len(prompt_ids)]),
        "action": (plan_ids, full_ids[len(plan_ids)]),
    }


def evaluate_model(model_path: Path, rows: list[dict]) -> list[dict]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True, use_fast=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True,
        device_map="cuda", dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    model.eval()
    results = []
    with torch.inference_mode():
        for index, row in enumerate(rows):
            item = {
                "id": row["id"],
                "task_id": row["task_id"],
                "family": row["family"],
                "transition": row["transition"],
                "operator": row["operator"],
                "seams": {},
            }
            for seam, (ids, target) in seam_inputs(row, tokenizer).items():
                tensor = torch.tensor([ids], dtype=torch.long, device=model.device)
                logits = model(
                    input_ids=tensor,
                    attention_mask=torch.ones_like(tensor),
                    logits_to_keep=1,
                    use_cache=False,
                ).logits[0, -1]
                item["seams"][seam] = distribution_metrics(logits, target)
            results.append(item)
            if (index + 1) % 14 == 0:
                print(f"[uncertainty] {model_path.name}: {index + 1}/{len(rows)}", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results


def summarize(rows: list[dict]) -> dict:
    result = {}
    by_transition = defaultdict(list)
    for row in rows:
        by_transition[row["transition"]].append(row)
    for transition, members in sorted(by_transition.items()):
        result[transition] = {}
        for seam in ("plan", "action"):
            result[transition][seam] = {
                key: statistics.mean(row["seams"][seam][key] for row in members)
                for key in ("entropy_nats", "varentropy_nats2", "target_logprob", "target_rank")
            }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-model", type=Path, required=True)
    parser.add_argument("--after-model", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--rows-per-transition", type=int, default=6)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    all_rows = [json.loads(line) for line in args.bank.read_text().splitlines() if line.strip()]
    by_transition = defaultdict(list)
    for row in sorted(all_rows, key=lambda item: item["id"]):
        by_transition[row["transition"]].append(row)
    selected = [
        row
        for transition in sorted(by_transition)
        for row in by_transition[transition][:args.rows_per_transition]
    ]
    if any(len(rows) < args.rows_per_transition for rows in by_transition.values()):
        raise SystemExit("bank has too few rows for the registered uncertainty sample")
    before_rows = evaluate_model(args.before_model, selected)
    after_rows = evaluate_model(args.after_model, selected)
    before = summarize(before_rows)
    after = summarize(after_rows)
    deltas = {}
    for transition in before:
        deltas[transition] = {}
        for seam in before[transition]:
            deltas[transition][seam] = {
                key: after[transition][seam][key] - before[transition][seam][key]
                for key in before[transition][seam]
            }
    result = {
        "schema_version": 1,
        "status": "exploratory_non_gating",
        "before_model": str(args.before_model.resolve()),
        "after_model": str(args.after_model.resolve()),
        "bank": str(args.bank.resolve()),
        "bank_sha256": sha256_file(args.bank),
        "rows_per_transition": args.rows_per_transition,
        "rows": len(selected),
        "before": before,
        "after": after,
        "after_minus_before": deltas,
        "before_rows": before_rows,
        "after_rows": after_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("before_rows", "after_rows")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
