#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_prefix_gate import summarize_arm
from src.jsonl import load_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--full-samples", type=int, required=True)
    parser.add_argument("--prefix-count", type=int, required=True)
    parser.add_argument("--completions-per-prefix", type=int, required=True)
    parser.add_argument("--prefix-lines", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=20260626)
    args = parser.parse_args()

    rows = load_jsonl(args.records)
    summary = {
        "experiment": "qwen35_4b_prefix_value_guided_search",
        "split": args.split,
        "count": args.count,
        "offset": args.offset,
        "full_samples": args.full_samples,
        "prefix_count": args.prefix_count,
        "completions_per_prefix": args.completions_per_prefix,
        "completion_budget_matched": args.full_samples == args.prefix_count * args.completions_per_prefix,
        "prefix_lines": args.prefix_lines,
        "temperature": args.temperature,
        "seed": args.seed,
        "arms": {
            "full_sample_base": summarize_arm(rows, key="full_samples"),
            "prefix_union": summarize_arm(rows, key="prefix_union"),
            "prefix_oracle_selected": summarize_arm(rows, key="prefix_oracle_selected"),
            "prefix_lexical_selected": summarize_arm(rows, key="prefix_lexical_selected"),
            "prefix_random_selected": summarize_arm(rows, key="prefix_random_selected"),
        },
        "mean_valid_prefixes": sum(row["prefix_count_valid"] for row in rows) / max(1, len(rows)),
    }
    write_json(args.summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
