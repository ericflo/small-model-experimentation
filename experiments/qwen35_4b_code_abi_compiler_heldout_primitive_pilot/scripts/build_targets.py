#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import abi_oracle  # noqa: E402


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def target_text(program: dict[str, Any]) -> str:
    return canonical_json(program)


def make_prompt_record(row: dict[str, Any], target: dict[str, Any] | None, split_name: str) -> dict[str, Any]:
    return {
        "record_id": row["record_id"],
        "task_id": row["task_id"],
        "task_text": row["task_text"],
        "entry_point": row["entry_point"],
        "slice": row["slice"],
        "example_count": row["example_count"],
        "visible_count": row["visible_count"],
        "candidate_count": row["candidate_count"],
        "visible_consistent_count": row["visible_consistent_count"],
        "full_winner_count": row["full_winner_count"],
        "oracle_covered": row["oracle_covered"],
        "winning_depth": row["winning_depth"],
        "winning_category": row["winning_category"],
        "target_program": target,
        "target_text": target_text(target) if target else "",
        "source_split": split_name,
    }


def run_slice(split: str, offset: int, count: int, visible_tests: int, split_name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    records = abi_oracle.load_records(split, count, offset, visible_tests)
    rows = [abi_oracle.run_task(record) for record in records]
    summary = abi_oracle.summarize(rows)
    summary.update({"split": split, "offset": offset, "count": count, "visible_tests": visible_tests, "split_name": split_name})
    targets = [make_prompt_record(row, row["winning_program"], split_name) for row in rows if row["oracle_covered"]]
    all_records = [make_prompt_record(row, row["winning_program"], split_name) for row in rows]
    return all_records, targets, summary


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def depth_summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {str(k): v for k, v in sorted(Counter(row["winning_depth"] for row in rows if row["oracle_covered"]).items())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visible-tests", type=int, default=1)
    parser.add_argument("--calibration-count", type=int, default=160)
    parser.add_argument("--calibration-offset", type=int, default=0)
    parser.add_argument("--heldout-count", type=int, default=160)
    parser.add_argument("--heldout-offset", type=int, default=160)
    parser.add_argument("--train-count", type=int, default=374)
    parser.add_argument("--train-offset", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--summary", type=Path, default=ROOT / "reports" / "gate1_summary.json")
    args = parser.parse_args()

    calibration_all, calibration_targets, calibration_summary = run_slice(
        "test", args.calibration_offset, args.calibration_count, args.visible_tests, "calibration"
    )
    heldout_all, heldout_targets, heldout_summary = run_slice(
        "test", args.heldout_offset, args.heldout_count, args.visible_tests, "heldout"
    )
    train_all, train_targets, train_summary = run_slice(
        "train", args.train_offset, args.train_count, args.visible_tests, "train"
    )

    # Small validation split from train targets, deterministic by task_id.
    train_targets_sorted = sorted(train_targets, key=lambda row: row["task_id"])
    validation_targets = train_targets_sorted[::5]
    training_targets = [row for row in train_targets_sorted if row not in validation_targets]

    write_jsonl(args.out_dir / "calibration_all.jsonl", calibration_all)
    write_jsonl(args.out_dir / "calibration_targets.jsonl", calibration_targets)
    write_jsonl(args.out_dir / "heldout_all.jsonl", heldout_all)
    write_jsonl(args.out_dir / "heldout_targets.jsonl", heldout_targets)
    write_jsonl(args.out_dir / "compiler_train.jsonl", training_targets)
    write_jsonl(args.out_dir / "compiler_validation.jsonl", validation_targets)

    gate = {
        "calibration": calibration_summary,
        "heldout": heldout_summary,
        "train": train_summary,
        "coverage_drop": calibration_summary["overall"]["oracle_coverage"] - heldout_summary["overall"]["oracle_coverage"],
        "target_counts": {
            "calibration": len(calibration_targets),
            "heldout": len(heldout_targets),
            "train": len(training_targets),
            "validation": len(validation_targets),
        },
        "depth_counts": {
            "calibration": depth_summary(calibration_all),
            "heldout": depth_summary(heldout_all),
            "train": depth_summary(train_all),
        },
        "heldout_by_depth": dict(
            sorted(
                {
                    str(depth): {
                        "covered": sum(1 for row in heldout_all if row["oracle_covered"] and row["winning_depth"] == depth),
                        "n": sum(1 for row in heldout_all if row["winning_depth"] == depth or (row["oracle_covered"] and row["winning_depth"] == depth)),
                    }
                    for depth in sorted(set(row["winning_depth"] for row in heldout_all if row["oracle_covered"]))
                }.items()
            )
        ),
    }
    write_json(args.summary, gate)
    print(json.dumps(gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
