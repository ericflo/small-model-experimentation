#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.jsonl import load_jsonl  # noqa: E402
from src.model_utils import DEFAULT_MODEL_PATH, load_tokenizer  # noqa: E402
from src.strategy_utils import (  # noqa: E402
    STRATEGY_KEYS,
    classify_strategy,
    normalized_code_hash,
    shuffled_key,
    strategy_prompt,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument("--mode", choices=["semantic", "shuffled"], required=True)
    parser.add_argument("--max-per-task", type=int, default=8)
    parser.add_argument("--max-per-task-strategy", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    tokenizer = load_tokenizer(args.model_path)
    records = load_jsonl(args.records)
    rows = []
    task_counts = Counter()
    semantic_counts = Counter()
    train_counts = Counter()
    tasks_with_correct = 0

    for record in records:
        selected = []
        seen_hashes = set()
        per_strategy = Counter()
        correct = [c for c in record.get("candidates", []) if c.get("full_pass") and c.get("parse_status") == "parsed" and c.get("code")]
        correct = sorted(correct, key=lambda c: c.get("order", 0))
        if correct:
            tasks_with_correct += 1
        for candidate in correct:
            code = candidate["code"].strip() + "\n"
            code_hash = normalized_code_hash(code)
            if code_hash in seen_hashes:
                continue
            semantic_key = classify_strategy(code, record["entry_point"])
            if per_strategy[semantic_key] >= args.max_per_task_strategy:
                continue
            seen_hashes.add(code_hash)
            per_strategy[semantic_key] += 1
            train_key = semantic_key if args.mode == "semantic" else shuffled_key(semantic_key, rng)
            selected.append(
                {
                    "prompt": strategy_prompt(record, train_key, tokenizer),
                    "target": code,
                    "task_id": record["task_id"],
                    "record_id": record["record_id"],
                    "candidate_id": candidate["candidate_id"],
                    "source": candidate.get("source", ""),
                    "semantic_strategy": semantic_key,
                    "train_strategy": train_key,
                    "code_hash": code_hash,
                }
            )
            if len(selected) >= args.max_per_task:
                break
        for row in selected:
            rows.append(row)
            task_counts[row["task_id"]] += 1
            semantic_counts[row["semantic_strategy"]] += 1
            train_counts[row["train_strategy"]] += 1

    rng.shuffle(rows)
    write_jsonl(args.out, rows)
    manifest = {
        "mode": args.mode,
        "records_in": str(args.records),
        "rows": len(rows),
        "source_records": len(records),
        "tasks_with_correct": tasks_with_correct,
        "tasks_used": len(task_counts),
        "max_per_task": args.max_per_task,
        "max_per_task_strategy": args.max_per_task_strategy,
        "strategy_keys": STRATEGY_KEYS,
        "semantic_strategy_counts": dict(sorted(semantic_counts.items())),
        "train_strategy_counts": dict(sorted(train_counts.items())),
        "examples_per_task_mean": (sum(task_counts.values()) / len(task_counts)) if task_counts else 0.0,
        "out": str(args.out),
    }
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
