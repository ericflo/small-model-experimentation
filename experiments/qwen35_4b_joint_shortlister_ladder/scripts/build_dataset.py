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
from src.tasks import build_ladder_records, pair_examples  # noqa: E402


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def parse_sizes(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library-sizes", type=parse_sizes, default=[64, 128, 256, 512])
    parser.add_argument("--train-per-cell", type=int, default=96)
    parser.add_argument("--eval-per-cell", type=int, default=8)
    parser.add_argument("--visible-count", type=int, default=6)
    parser.add_argument("--hidden-count", type=int, default=18)
    parser.add_argument("--query-count", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260624)
    args = parser.parse_args()

    train_records = build_ladder_records(
        library_sizes=args.library_sizes,
        count_per_cell=args.train_per_cell,
        visible_count=args.visible_count,
        hidden_count=args.hidden_count,
        query_count=args.query_count,
        seed=args.seed,
        split="train",
    )
    eval_records = build_ladder_records(
        library_sizes=args.library_sizes,
        count_per_cell=args.eval_per_cell,
        visible_count=args.visible_count,
        hidden_count=args.hidden_count,
        query_count=args.query_count,
        seed=args.seed + 1009,
        split="eval",
    )
    train_pairs = pair_examples(train_records)
    eval_pairs = pair_examples(eval_records)
    operators = build_operator_library(max(args.library_sizes))

    write_jsonl(ROOT / "data" / "train_records.jsonl", train_records)
    write_jsonl(ROOT / "data" / "eval_records.jsonl", eval_records)
    write_jsonl(ROOT / "data" / "train_pairs.jsonl", train_pairs)
    write_jsonl(ROOT / "data" / "eval_pairs.jsonl", eval_pairs)

    manifest = {
        "experiment": "qwen35_4b_joint_shortlister_ladder",
        "library_sizes": args.library_sizes,
        "operator_signature": "list[int] -> int",
        "operator_count_max": len(operators),
        "operator_families": dict(sorted(Counter(operator.family for operator in operators).items())),
        "train_records": len(train_records),
        "eval_records": len(eval_records),
        "train_pairs": len(train_pairs),
        "eval_pairs": len(eval_pairs),
        "visible_count": args.visible_count,
        "hidden_count": args.hidden_count,
        "query_count": args.query_count,
        "seed": args.seed,
        "train_by_library": dict(sorted(Counter(row["library_size"] for row in train_records).items())),
        "eval_by_library": dict(sorted(Counter(row["library_size"] for row in eval_records).items())),
        "train_by_template": dict(sorted(Counter(row["template"] for row in train_records).items())),
        "eval_by_template": dict(sorted(Counter(row["template"] for row in eval_records).items())),
        "record_local_aliases": True,
        "base_model_path": "/workspace/.cache/huggingface/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    }
    (ROOT / "data" / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

