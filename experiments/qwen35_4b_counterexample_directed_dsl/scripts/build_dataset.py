#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_gen import HOLDOUT_FAMILIES, TRAIN_FAMILIES, build_dataset, write_jsonl  # noqa: E402
from src.dsl import program_case_passes  # noqa: E402


def validate(rows):
    for record in rows:
        for case in record["visible"] + record["hidden"]:
            if not program_case_passes(record["target_program"], case):
                raise RuntimeError(f"oracle failed on {record['id']}")


def write_split(out_dir: Path, strategy: str, args) -> dict:
    data = build_dataset(
        seed=args.seed,
        train_records=args.train_records,
        iid_eval_records=args.iid_eval_records,
        holdout_records_per_family=args.holdout_records_per_family,
        visible_cases=args.visible_cases,
        hidden_cases=args.hidden_cases,
        pool_size=args.pool_size,
        trace_strategy=strategy,
    )
    split_dir = out_dir / strategy
    split_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(split_dir / "dsl_train.jsonl", data["train"])
    write_jsonl(split_dir / "dsl_eval_iid.jsonl", data["iid"])
    write_jsonl(split_dir / "dsl_eval_holdout.jsonl", data["holdout"])
    validate(data["train"] + data["iid"] + data["holdout"])
    manifest = {
        "trace_strategy": strategy,
        "seed": args.seed,
        "train_records": len(data["train"]),
        "iid_eval_records": len(data["iid"]),
        "holdout_records": len(data["holdout"]),
        "visible_cases_per_record": args.visible_cases,
        "hidden_cases_per_record": args.hidden_cases,
        "candidate_pool_size": args.pool_size,
        "train_families": TRAIN_FAMILIES,
        "holdout_families": HOLDOUT_FAMILIES,
    }
    (split_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--train-records", type=int, default=240)
    parser.add_argument("--iid-eval-records", type=int, default=60)
    parser.add_argument("--holdout-records-per-family", type=int, default=24)
    parser.add_argument("--visible-cases", type=int, default=6)
    parser.add_argument("--hidden-cases", type=int, default=18)
    parser.add_argument("--pool-size", type=int, default=160)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    manifests = {
        "random": write_split(args.out_dir, "random", args),
        "counterexample": write_split(args.out_dir, "counterexample", args),
    }
    top_manifest = {"datasets": manifests}
    (args.out_dir / "dataset_manifest.json").write_text(json.dumps(top_manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(top_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

