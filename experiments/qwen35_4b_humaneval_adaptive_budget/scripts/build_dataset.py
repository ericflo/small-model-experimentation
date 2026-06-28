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

from src.budget_policy import build_rollout_states  # noqa: E402
from src.humaneval_env import build_record, write_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-tasks", type=int, default=40)
    parser.add_argument("--eval-tasks", type=int, default=20)
    parser.add_argument("--visible-tests", type=int, default=2)
    parser.add_argument("--probe-tests", type=int, default=32)
    parser.add_argument("--hidden-tests", type=int, default=32)
    parser.add_argument("--candidate-count", type=int, default=16)
    parser.add_argument("--max-budget", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260625)
    args = parser.parse_args()

    out_dir = ROOT / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    skips: Counter[str] = Counter()
    raw_rows = list(load_dataset("openai/openai_humaneval", split="test"))
    need = args.train_tasks + args.eval_tasks
    for raw_index, raw in enumerate(tqdm(raw_rows, desc="build-humaneval-records")):
        split = "train" if len(records) < args.train_tasks else "eval"
        record, reason = build_record(
            raw,
            split=split,
            index=raw_index,
            visible_tests=args.visible_tests,
            probe_tests=args.probe_tests,
            hidden_tests=args.hidden_tests,
            max_candidates=args.candidate_count,
            seed=args.seed,
        )
        if record is None:
            skips[reason or "unknown"] += 1
            continue
        records.append(record)
        if len(records) >= need:
            break
    train_records = records[: args.train_tasks]
    eval_records = records[args.train_tasks : args.train_tasks + args.eval_tasks]
    if len(train_records) < args.train_tasks or len(eval_records) < args.eval_tasks:
        raise RuntimeError(
            f"not enough usable tasks: train={len(train_records)} eval={len(eval_records)} skips={dict(skips)}"
        )

    train_states = [state for record in train_records for state in build_rollout_states(record, args.max_budget)]
    eval_states = [state for record in eval_records for state in build_rollout_states(record, args.max_budget)]
    write_jsonl(out_dir / "train_records.jsonl", train_records)
    write_jsonl(out_dir / "eval_records.jsonl", eval_records)
    write_jsonl(out_dir / "train_budget_states.jsonl", train_states)
    write_jsonl(out_dir / "eval_budget_states.jsonl", eval_states)

    manifest = {
        "experiment": "qwen35_4b_humaneval_adaptive_budget",
        "dataset": "openai/openai_humaneval",
        "seed": args.seed,
        "visible_tests": args.visible_tests,
        "probe_tests": args.probe_tests,
        "hidden_tests": args.hidden_tests,
        "candidate_count": args.candidate_count,
        "max_budget": args.max_budget,
        "raw_rows_seen": raw_index + 1,
        "skips": dict(skips),
        "train_records": {"records": len(train_records), "path": "data/train_records.jsonl"},
        "eval_records": {"records": len(eval_records), "path": "data/eval_records.jsonl"},
        "train_states": {"states": len(train_states), "path": "data/train_budget_states.jsonl"},
        "eval_states": {"states": len(eval_states), "path": "data/eval_budget_states.jsonl"},
    }
    (out_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

