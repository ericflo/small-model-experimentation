#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.abi import Example, candidate_programs, canonical_json, equal, examples_pass, safe_execute  # noqa: E402
from src.data_gen import make_tasks  # noqa: E402
from src.modeling import candidate_loss_batch, load_generation_model, load_jsonl, load_tokenizer  # noqa: E402


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def examples_from_record(row: dict[str, Any], key: str) -> list[Example]:
    return [Example(item["input"], item["output"], key) for item in row[key]]


def select_first_visible(row: dict[str, Any]) -> tuple[str, float]:
    return row["candidate_texts"][0], 0.0


def select_oracle(row: dict[str, Any]) -> tuple[str, float]:
    return row["target_text"], 0.0


def select_random_visible(row: dict[str, Any], rng: random.Random) -> tuple[str, float]:
    return rng.choice(row["candidate_texts"]), 0.0


def select_model(row: dict[str, Any], model, tokenizer, batch_size: int) -> tuple[str, float]:
    candidates = row["candidate_texts"]
    losses: list[float] = []
    for start in range(0, len(candidates), batch_size):
        chunk = candidates[start : start + batch_size]
        losses.extend(candidate_loss_batch(model, tokenizer, row["prompt"], chunk))
    best_idx = min(range(len(candidates)), key=lambda i: losses[i])
    return candidates[best_idx], losses[best_idx]


def eval_choice(row: dict[str, Any], selected_text: str, score: float) -> dict[str, Any]:
    selected = json.loads(selected_text)
    visible = examples_from_record(row, "visible")
    hidden = examples_from_record(row, "hidden")
    adversarial = examples_from_record(row, "adversarial")
    raw_examples = visible + hidden
    all_examples = raw_examples + adversarial
    visible_pass = examples_pass(selected, visible)
    raw_pass = examples_pass(selected, raw_examples)
    filtered_pass = examples_pass(selected, all_examples)
    target_exact = selected_text == row["target_text"]
    return {
        "task_id": row["task_id"],
        "domain": row["domain"],
        "depth": row["depth"],
        "candidate_count": row["candidate_count"],
        "selected_text": selected_text,
        "target_text": row["target_text"],
        "score": score,
        "target_exact": target_exact,
        "visible_pass": visible_pass,
        "raw_pass": raw_pass,
        "filtered_pass": filtered_pass,
    }


def summarize(rows: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    def metrics(group: list[dict[str, Any]]) -> dict[str, Any]:
        if not group:
            return {"n": 0}
        return {
            "n": len(group),
            "target_exact": sum(row["target_exact"] for row in group),
            "target_exact_rate": sum(row["target_exact"] for row in group) / len(group),
            "raw_pass": sum(row["raw_pass"] for row in group),
            "raw_accuracy": sum(row["raw_pass"] for row in group) / len(group),
            "filtered_pass": sum(row["filtered_pass"] for row in group),
            "filtered_accuracy": sum(row["filtered_pass"] for row in group) / len(group),
            "visible_pass": sum(row["visible_pass"] for row in group),
            "candidate_count_mean": sum(row["candidate_count"] for row in group) / len(group),
        }

    domains = sorted({row["domain"] for row in rows})
    depth1 = [row for row in rows if row["depth"] == 1]
    depth2p = [row for row in rows if row["depth"] >= 2]
    return {
        "arm": arm,
        "overall": metrics(rows),
        "depth_1": metrics(depth1),
        "depth_2_plus": metrics(depth2p),
        "by_domain": {domain: metrics([row for row in rows if row["domain"] == domain]) for domain in domains},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["first_visible", "random_visible", "oracle", "base", "lora"], required=True)
    parser.add_argument("--model-path", default="/workspace/.cache/huggingface/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--eval", type=Path, default=ROOT / "data" / "eval.jsonl")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    records = load_jsonl(args.eval)
    model = None
    tokenizer = None
    if args.arm in {"base", "lora"}:
        tokenizer = load_tokenizer(args.model_path)
        model = load_generation_model(args.model_path, args.adapter if args.arm == "lora" else None)

    out_rows = []
    rng = random.Random(args.seed)
    for idx, row in enumerate(records):
        if args.arm == "first_visible":
            selected, score = select_first_visible(row)
        elif args.arm == "random_visible":
            selected, score = select_random_visible(row, rng)
        elif args.arm == "oracle":
            selected, score = select_oracle(row)
        else:
            assert model is not None and tokenizer is not None
            selected, score = select_model(row, model, tokenizer, args.batch_size)
        out_rows.append(eval_choice(row, selected, score))
        if (idx + 1) % 8 == 0:
            print(f"evaluated {idx + 1}/{len(records)}", flush=True)

    summary = summarize(out_rows, args.arm)
    out_dir = ROOT / "reports" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / f"{args.arm}_summary.json", summary)
    with (out_dir / f"{args.arm}_records.jsonl").open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if model is not None:
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
