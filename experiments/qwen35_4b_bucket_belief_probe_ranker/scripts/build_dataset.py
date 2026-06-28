#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.operator_env import LETTERS, action_diagnostics, build_operator_library, generate_record, select_action_queries, write_jsonl  # noqa: E402
from src.prompts import process_prompt  # noqa: E402


def build_process_states(records: list[dict[str, Any]], states_per_record: int, action_source: str) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    operator_cache: dict[int, Any] = {}
    for record in records:
        n = record["library_size"]
        operators = operator_cache.setdefault(n, build_operator_library(n))
        used: list[int] = []
        for step in range(states_per_record):
            queries = select_action_queries(record, operators, used, action_source, action_count=len(LETTERS))
            if len(queries) != len(LETTERS):
                break
            diag = action_diagnostics(record, operators, queries, used)
            rewards = {row["letter"]: row["reward"] for row in diag["actions"]}
            if diag["candidate_count"] <= 1 or max(rewards.values()) <= min(rewards.values()) + 1e-9:
                break
            state = {
                "state_id": f"{record['record_id']}_s{step}",
                "record_id": record["record_id"],
                "split": record["split"],
                "step": step,
                "library_size": record["library_size"],
                "template": record["template"],
                "used_query_indices": list(used),
                "action_query_indices": queries,
                "action_source": action_source,
                "candidate_count": diag["candidate_count"],
                "oracle_action": diag["oracle_action"],
                "oracle_query_index": diag["oracle_query_index"],
                "reward_by_action": rewards,
                "prompt": process_prompt(record, diag),
            }
            states.append(state)
            used.append(diag["oracle_query_index"])
    return states


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_cell = Counter((row["split"], row["library_size"], row["template"]) for row in records)
    return {
        "records": len(records),
        "by_cell": {f"{split}/n{n}/{template}": count for (split, n, template), count in sorted(by_cell.items())},
    }


def summarize_states(states: list[dict[str, Any]]) -> dict[str, Any]:
    by_step: dict[int, int] = defaultdict(int)
    by_action: Counter[str] = Counter()
    candidate_counts: list[int] = []
    for row in states:
        by_step[int(row["step"])] += 1
        by_action[row["oracle_action"]] += 1
        candidate_counts.append(int(row["candidate_count"]))
    return {
        "states": len(states),
        "by_step": dict(sorted(by_step.items())),
        "oracle_action_histogram": dict(sorted(by_action.items())),
        "candidate_count_mean": sum(candidate_counts) / max(len(candidate_counts), 1),
        "candidate_count_max": max(candidate_counts) if candidate_counts else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--train-per-cell", type=int, default=80)
    parser.add_argument("--eval-per-cell", type=int, default=16)
    parser.add_argument("--states-per-record", type=int, default=3)
    parser.add_argument("--query-pool-cases", type=int, default=96)
    parser.add_argument("--action-source", choices=["random8", "mined8"], default="mined8")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_records: list[dict[str, Any]] = []
    eval_records: list[dict[str, Any]] = []

    train_sizes = [64, 128, 256]
    eval_sizes = [64, 128, 256, 512]
    templates = ["pair_affine_mod", "pair_compare_gate"]
    min_candidates = {
        "pair_affine_mod": 24,
        "pair_compare_gate": 96,
    }

    for split, sizes, per_cell, target in [
        ("train", train_sizes, args.train_per_cell, train_records),
        ("eval", eval_sizes, args.eval_per_cell, eval_records),
    ]:
        for n in sizes:
            for template in templates:
                for i in range(per_cell):
                    record_id = f"{split}_n{n}_{template}_{i:04d}"
                    target.append(
                        generate_record(
                            rng,
                            record_id=record_id,
                            split=split,
                            library_size=n,
                            template=template,
                            min_initial_candidates=min_candidates[template],
                            query_pool_cases=args.query_pool_cases,
                        )
                    )

    train_states = build_process_states(train_records, args.states_per_record, args.action_source)
    eval_states = build_process_states(eval_records, args.states_per_record, args.action_source)

    write_jsonl(args.out_dir / "train_records.jsonl", train_records)
    write_jsonl(args.out_dir / "eval_records.jsonl", eval_records)
    write_jsonl(args.out_dir / "process_train_states.jsonl", train_states)
    write_jsonl(args.out_dir / "process_eval_states.jsonl", eval_states)

    manifest = {
        "seed": args.seed,
        "operator_inventory_max": 512,
        "visible_cases": 4,
        "query_pool_cases": args.query_pool_cases,
        "action_source": args.action_source,
        "hidden_cases": 16,
        "train_sizes": train_sizes,
        "eval_sizes": eval_sizes,
        "templates": templates,
        "states_per_record": args.states_per_record,
        "train_records": summarize_records(train_records),
        "eval_records": summarize_records(eval_records),
        "train_states": summarize_states(train_states),
        "eval_states": summarize_states(eval_states),
    }
    (args.out_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
