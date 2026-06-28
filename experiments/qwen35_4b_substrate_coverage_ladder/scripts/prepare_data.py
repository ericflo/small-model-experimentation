#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.jsonl import load_jsonl, write_json, write_jsonl
from src.mbpp_env import infer_entry_point, parse_assert_case


def slim_residual_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": "mbpp",
        "split": "heldout_residual",
        "task_id": record["task_id"],
        "record_id": f"mbpp_residual_{record['task_id']}",
        "entry_point": record["entry_point"],
        "task_text": record["task_text"],
        "public_cases": record["public_cases"],
        "hidden_asserts": record["hidden_asserts"],
        "all_asserts": record.get("all_asserts", []),
        "setup_code": record.get("setup_code", ""),
        "baseline_candidate_count": record.get("candidate_count"),
        "baseline_visible_coverage": record.get("visible_coverage"),
        "baseline_hidden_coverage": record.get("coverage"),
    }


def load_mbpp_train_library() -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/mbpp", "full", split="train")
    rows: list[dict[str, Any]] = []
    for item in ds:
        tests = list(item.get("test_list", []))
        entry = infer_entry_point(tests)
        if not entry:
            continue
        public_cases = []
        for assertion in tests[:1]:
            case = parse_assert_case(assertion)
            if case:
                public_cases.append(case)
        rows.append(
            {
                "dataset": "mbpp_train",
                "task_id": int(item["task_id"]),
                "entry_point": entry,
                "task_text": item.get("text", ""),
                "reference_code": item.get("code", ""),
                "public_cases": public_cases,
                "setup_code": item.get("test_setup_code", ""),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=None, help="Optional raw base-pool records for rebuilding the residual file.")
    parser.add_argument("--k128", type=Path, default=None, help="Optional raw K128-pool records for rebuilding the residual file.")
    parser.add_argument("--coverage-index", type=Path, default=ROOT / "data/baseline_coverage_index.jsonl")
    parser.add_argument("--out-residual", type=Path, default=ROOT / "data/residual_tasks.jsonl")
    parser.add_argument("--out-train-library", type=Path, default=ROOT / "data/mbpp_train_library.jsonl")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/dataset_manifest.json")
    args = parser.parse_args()

    if args.base and args.k128:
        base_rows = load_jsonl(args.base)
        k128_rows = load_jsonl(args.k128)
        base_misses = sorted(row["task_id"] for row in base_rows if not row.get("coverage", False))
        k128_residual = [row for row in k128_rows if not row.get("coverage", False)]
        residual_rows = [slim_residual_record(row) for row in k128_residual]
        residual_ids = sorted(row["task_id"] for row in residual_rows)
        write_jsonl(args.out_residual, residual_rows)
    else:
        residual_rows = load_jsonl(args.out_residual)
        coverage_index = load_jsonl(args.coverage_index)
        base_misses = sorted(row["task_id"] for row in coverage_index if not row.get("base_k4_covered", False))
        residual_ids = sorted(row["task_id"] for row in residual_rows)

    train_library = load_mbpp_train_library()

    write_jsonl(args.out_train_library, train_library)
    write_json(
        args.manifest,
        {
            "experiment": "qwen35_4b_substrate_coverage_ladder",
            "base_miss_count": len(base_misses),
            "base_miss_task_ids": base_misses,
            "k128_residual_count": len(residual_ids),
            "k128_residual_task_ids": residual_ids,
            "train_library_count": len(train_library),
            "residual_file": str(args.out_residual.relative_to(ROOT)),
            "train_library_file": str(args.out_train_library.relative_to(ROOT)),
        },
    )
    print(f"wrote {len(residual_rows)} residual tasks: {residual_ids}")
    print(f"wrote {len(train_library)} train-library entries")


if __name__ == "__main__":
    main()
