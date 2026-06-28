#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.code_env import (  # noqa: E402
    build_candidate_examples,
    build_humaneval_record,
    build_mbpp_record,
    record_coverage,
    visible_candidates,
    write_jsonl,
)


def collect_records(rows: list[dict[str, Any]], builder: Any, split: str, target: int, args: argparse.Namespace, seed_offset: int) -> tuple[list[dict[str, Any]], Counter[str], int]:
    records: list[dict[str, Any]] = []
    skips: Counter[str] = Counter()
    seen = 0
    for index, raw in enumerate(tqdm(rows, desc=f"build-{split}")):
        seen += 1
        record, reason = builder(
            raw,
            split=split,
            index=index,
            visible_tests=args.visible_tests,
            max_candidates=args.candidate_count,
            seed=args.seed + seed_offset,
        )
        if record is None:
            skips[reason or "unknown"] += 1
            continue
        records.append(record)
        if len(records) >= target:
            break
    return records, skips, seen


def record_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    visible_counts = [len(visible_candidates(record)) for record in records]
    positive_counts = [sum(1 for candidate in visible_candidates(record) if candidate["hidden_all_pass"]) for record in records]
    return {
        "records": len(records),
        "coverage": sum(1 for record in records if record_coverage(record)) / len(records) if records else 0.0,
        "visible_candidates_mean": sum(visible_counts) / len(visible_counts) if visible_counts else 0.0,
        "hidden_pass_visible_candidates_mean": sum(positive_counts) / len(positive_counts) if positive_counts else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mbpp-train", type=int, default=160)
    parser.add_argument("--mbpp-valid", type=int, default=40)
    parser.add_argument("--humaneval-eval", type=int, default=40)
    parser.add_argument("--visible-tests", type=int, default=1)
    parser.add_argument("--candidate-count", type=int, default=18)
    parser.add_argument("--seed", type=int, default=20260625)
    args = parser.parse_args()

    data_dir = ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    mbpp = load_dataset("google-research-datasets/mbpp")
    humaneval = load_dataset("openai/openai_humaneval", split="test")
    mbpp_train_records, train_skips, train_seen = collect_records(
        list(mbpp["train"]), build_mbpp_record, "mbpp_train", args.mbpp_train, args, 0
    )
    mbpp_valid_records, valid_skips, valid_seen = collect_records(
        list(mbpp["validation"]) + list(mbpp["test"]), build_mbpp_record, "mbpp_valid", args.mbpp_valid, args, 10000
    )
    humaneval_records, humaneval_skips, humaneval_seen = collect_records(
        list(humaneval), build_humaneval_record, "humaneval_eval", args.humaneval_eval, args, 20000
    )

    if len(mbpp_train_records) < args.mbpp_train:
        raise RuntimeError(f"not enough MBPP train records: {len(mbpp_train_records)} skips={dict(train_skips)}")
    if len(mbpp_valid_records) < args.mbpp_valid:
        raise RuntimeError(f"not enough MBPP valid records: {len(mbpp_valid_records)} skips={dict(valid_skips)}")
    if len(humaneval_records) < args.humaneval_eval:
        raise RuntimeError(f"not enough HumanEval eval records: {len(humaneval_records)} skips={dict(humaneval_skips)}")

    train_examples = build_candidate_examples(mbpp_train_records, "train", balance=True, seed=args.seed)
    valid_examples = build_candidate_examples(mbpp_valid_records, "valid", balance=False, seed=args.seed + 1)
    human_examples = build_candidate_examples(humaneval_records, "eval", balance=False, seed=args.seed + 2)

    write_jsonl(data_dir / "mbpp_train_records.jsonl", mbpp_train_records)
    write_jsonl(data_dir / "mbpp_valid_records.jsonl", mbpp_valid_records)
    write_jsonl(data_dir / "humaneval_eval_records.jsonl", humaneval_records)
    write_jsonl(data_dir / "train_verifier_examples.jsonl", train_examples)
    write_jsonl(data_dir / "valid_verifier_examples.jsonl", valid_examples)
    write_jsonl(data_dir / "humaneval_verifier_examples.jsonl", human_examples)

    manifest = {
        "experiment": "qwen35_4b_oracle_distilled_semantic_verifier",
        "seed": args.seed,
        "visible_tests": args.visible_tests,
        "candidate_count": args.candidate_count,
        "datasets": {
            "train": "google-research-datasets/mbpp:train",
            "valid": "google-research-datasets/mbpp:validation+test prefix",
            "eval": "openai/openai_humaneval:test",
        },
        "records": {
            "mbpp_train": record_summary(mbpp_train_records),
            "mbpp_valid": record_summary(mbpp_valid_records),
            "humaneval_eval": record_summary(humaneval_records),
        },
        "examples": {
            "train": len(train_examples),
            "valid": len(valid_examples),
            "humaneval": len(human_examples),
        },
        "raw_seen": {
            "mbpp_train": train_seen,
            "mbpp_valid": valid_seen,
            "humaneval_eval": humaneval_seen,
        },
        "skips": {
            "mbpp_train": dict(train_skips),
            "mbpp_valid": dict(valid_skips),
            "humaneval_eval": dict(humaneval_skips),
        },
        "paths": {
            "mbpp_train_records": "data/mbpp_train_records.jsonl",
            "mbpp_valid_records": "data/mbpp_valid_records.jsonl",
            "humaneval_eval_records": "data/humaneval_eval_records.jsonl",
            "train_examples": "data/train_verifier_examples.jsonl",
            "valid_examples": "data/valid_verifier_examples.jsonl",
            "humaneval_examples": "data/humaneval_verifier_examples.jsonl",
        },
    }
    (data_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

