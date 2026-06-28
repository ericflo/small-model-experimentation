#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.operator_library import build_operator_library  # noqa: E402
from src.tasks import build_records, slot_examples  # noqa: E402


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library-size", type=int, default=512)
    parser.add_argument("--train-records", type=int, default=768)
    parser.add_argument("--eval-records", type=int, default=96)
    parser.add_argument("--visible-count", type=int, default=6)
    parser.add_argument("--hidden-count", type=int, default=18)
    parser.add_argument("--query-count", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260624)
    args = parser.parse_args()

    train_records = build_records(
        library_size=args.library_size,
        count=args.train_records,
        visible_count=args.visible_count,
        hidden_count=args.hidden_count,
        query_count=args.query_count,
        seed=args.seed,
        split="train",
    )
    eval_records = build_records(
        library_size=args.library_size,
        count=args.eval_records,
        visible_count=args.visible_count,
        hidden_count=args.hidden_count,
        query_count=args.query_count,
        seed=args.seed + 1009,
        split="eval",
    )
    train_slots = slot_examples(train_records)
    eval_slots = slot_examples(eval_records)
    operators = build_operator_library(args.library_size)

    write_jsonl(ROOT / "data" / "train_records.jsonl", train_records)
    write_jsonl(ROOT / "data" / "eval_records.jsonl", eval_records)
    write_jsonl(ROOT / "data" / "train_slots.jsonl", train_slots)
    write_jsonl(ROOT / "data" / "eval_slots.jsonl", eval_slots)

    manifest = {
        "experiment": "qwen35_4b_inventory_shortlister_training",
        "library_size": args.library_size,
        "operator_signature": "list[int] -> int",
        "operator_count": len(operators),
        "operator_families": dict(sorted(Counter(operator.family for operator in operators).items())),
        "train_records": len(train_records),
        "eval_records": len(eval_records),
        "train_slots": len(train_slots),
        "eval_slots": len(eval_slots),
        "visible_count": args.visible_count,
        "hidden_count": args.hidden_count,
        "query_count": args.query_count,
        "seed": args.seed,
        "templates_train": dict(sorted(Counter(row["template"] for row in train_records).items())),
        "templates_eval": dict(sorted(Counter(row["template"] for row in eval_records).items())),
        "candidate_budgets": [1024, 4096, 16384],
        "slot_topk_for_budgets": {"1024": 32, "4096": 64, "16384": 128},
        "base_model_path": "/workspace/.cache/huggingface/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    }
    (ROOT / "data" / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

