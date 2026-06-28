#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.coverage_utils import EXPERIMENT, summarize_records, write_manifest  # noqa: E402
from src.jsonl import load_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-label", type=str, default="local_base_direct_k4")
    args = parser.parse_args()

    rows = load_jsonl(args.records)
    usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "forward_tokens": 0}
    for row in rows:
        row_usage = row.get("token_usage", {})
        for key in usage:
            usage[key] += int(row_usage.get(key, 0))
        if not row.get("token_usage"):
            for candidate in row.get("candidates", []):
                usage["calls"] += 1
                usage["prompt_tokens"] += int(candidate.get("prompt_tokens", 0))
                usage["completion_tokens"] += int(candidate.get("completion_tokens", 0))
                usage["forward_tokens"] += int(candidate.get("forward_tokens", 0))
    misses = [
        int(row["task_id"])
        for row in rows
        if not any(candidate.get("full_pass") for candidate in row.get("candidates", []))
    ]
    manifest = {
        "experiment": EXPERIMENT,
        "arm_name": "base_direct_k4",
        "source_label": args.source_label,
        "dataset": "mbpp",
        "split": "heldout",
        "visible_tests": 1,
        "samples_per_task": 4,
        "records": summarize_records(rows),
        "miss_count": len(misses),
        "miss_tasks": misses,
        "token_usage": usage,
        "path": str(args.records),
    }
    write_manifest(args.out, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
