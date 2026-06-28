#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import abi_oracle  # noqa: E402


def raw_to_record(raw: dict[str, Any], split: str, visible_tests: int) -> dict[str, Any]:
    tests = list(raw.get("test_list") or [])
    entry = abi_oracle.parse_entry(tests[0]) if tests else "unknown"
    parsed = [abi_oracle.parse_assert(test) for test in tests + list(raw.get("challenge_test_list") or [])]
    examples = [ex for ex in parsed if ex is not None]
    visible_examples = [ex for ex in [abi_oracle.parse_assert(test) for test in tests[:visible_tests]] if ex is not None]
    return {
        "record_id": f"mbpp_{split}_{raw['task_id']}",
        "task_id": raw["task_id"],
        "task_text": raw["text"],
        "entry_point": entry or "unknown",
        "tests": tests,
        "challenge_tests": list(raw.get("challenge_test_list") or []),
        "examples": examples,
        "visible_examples": visible_examples,
        "parseable": len(examples) == len(tests) + len(raw.get("challenge_test_list") or []),
    }


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exclude-first", type=int, default=160)
    parser.add_argument("--sample-count", type=int, default=160)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 17, 23])
    parser.add_argument("--visible-tests", type=int, default=1)
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "gate1_split_sweep.json")
    args = parser.parse_args()

    test_rows = list(load_dataset("google-research-datasets/mbpp")["test"])
    pool = test_rows[args.exclude_first :]
    results = []
    for seed in args.seeds:
        rng = random.Random(seed)
        sample = rng.sample(pool, min(args.sample_count, len(pool)))
        records = [raw_to_record(raw, "test", args.visible_tests) for raw in sample]
        rows = [abi_oracle.run_task(record) for record in records]
        summary = abi_oracle.summarize(rows)
        summary.update(
            {
                "seed": seed,
                "sample_count": len(sample),
                "exclude_first": args.exclude_first,
                "visible_tests": args.visible_tests,
                "task_ids": [row["task_id"] for row in rows],
            }
        )
        results.append(summary)
    aggregate = {
        "results": results,
        "coverage_values": [item["overall"]["oracle_coverage"] for item in results],
        "covered_counts": [item["overall"]["oracle_covered"] for item in results],
        "mean_coverage": sum(item["overall"]["oracle_coverage"] for item in results) / len(results),
        "min_coverage": min(item["overall"]["oracle_coverage"] for item in results),
        "max_coverage": max(item["overall"]["oracle_coverage"] for item in results),
    }
    write_json(args.out, aggregate)
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
