#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.code_env import verifier_prompt  # noqa: E402
from src.jsonl import load_jsonl, write_jsonl  # noqa: E402


def build_examples(records: list[dict[str, Any]], split: str, balance: bool, seed: int, include_public_fail: bool) -> list[dict[str, Any]]:
    positives: list[dict[str, Any]] = []
    negatives: list[dict[str, Any]] = []
    for record in records:
        for candidate in record["candidates"]:
            if candidate.get("parse_status") != "parsed" or not candidate.get("safe"):
                continue
            if not include_public_fail and not candidate.get("visible_all_pass"):
                continue
            row = {
                "split": split,
                "dataset": record["dataset"],
                "record_id": record["record_id"],
                "task_id": record["task_id"],
                "candidate_id": candidate["candidate_id"],
                "label": "A" if candidate.get("full_pass") else "B",
                "label_index": 0 if candidate.get("full_pass") else 1,
                "prompt": verifier_prompt(record, candidate),
            }
            if candidate.get("full_pass"):
                positives.append(row)
            else:
                negatives.append(row)
    rng = random.Random(seed)
    if balance and positives:
        rng.shuffle(negatives)
        negatives = negatives[: max(len(positives) * 2, len(positives))]
    rows = positives + negatives
    rng.shuffle(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-records", type=Path, default=ROOT / "data" / "mbpp_train_records.jsonl")
    parser.add_argument("--mbpp-eval-records", type=Path, default=ROOT / "data" / "mbpp_eval_records.jsonl")
    parser.add_argument("--humaneval-records", type=Path, default=ROOT / "data" / "humaneval_eval_records.jsonl")
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--visible-only", action="store_true")
    args = parser.parse_args()

    include_public_fail = not args.visible_only
    train = build_examples(load_jsonl(args.train_records), "train", balance=True, seed=args.seed, include_public_fail=include_public_fail)
    mbpp_eval = build_examples(load_jsonl(args.mbpp_eval_records), "mbpp_eval", balance=False, seed=args.seed + 1, include_public_fail=include_public_fail)
    humaneval = build_examples(load_jsonl(args.humaneval_records), "humaneval_eval", balance=False, seed=args.seed + 2, include_public_fail=include_public_fail)
    write_jsonl(ROOT / "data" / "train_verifier_examples.jsonl", train)
    write_jsonl(ROOT / "data" / "mbpp_eval_verifier_examples.jsonl", mbpp_eval)
    write_jsonl(ROOT / "data" / "humaneval_eval_verifier_examples.jsonl", humaneval)
    print(
        {
            "train": len(train),
            "mbpp_eval": len(mbpp_eval),
            "humaneval_eval": len(humaneval),
            "train_positive": sum(row["label"] == "A" for row in train),
            "train_negative": sum(row["label"] == "B" for row in train),
            "include_public_fail": include_public_fail,
        }
    )


if __name__ == "__main__":
    main()
