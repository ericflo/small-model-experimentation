#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.budget_policy import ACTION_LETTERS, build_rollout_states  # noqa: E402
from src.operator_env import build_operator_library, generate_record, write_jsonl  # noqa: E402


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_cell = Counter((row["split"], row["library_size"], row["template"]) for row in records)
    return {
        "records": len(records),
        "by_cell": {f"{split}/n{n}/{template}": count for (split, n, template), count in sorted(by_cell.items())},
    }


def summarize_states(states: list[dict[str, Any]]) -> dict[str, Any]:
    by_cell = Counter((row["split"], row["library_size"], row["template"]) for row in states)
    by_budget = Counter(int(row["budget"]) for row in states)
    by_label = Counter(row["label"] for row in states)
    stop_budget = defaultdict(list)
    for row in states:
        if row["label"] == "A":
            stop_budget[(row["split"], row["template"])].append(int(row["budget"]))
    return {
        "states": len(states),
        "by_cell": {f"{split}/n{n}/{template}": count for (split, n, template), count in sorted(by_cell.items())},
        "by_budget": dict(sorted(by_budget.items())),
        "labels": {letter: by_label.get(letter, 0) for letter in ACTION_LETTERS},
        "candidate_count_mean": sum(int(row["candidate_count"]) for row in states) / max(len(states), 1),
        "hidden_equivalent_candidates_mean": sum(int(row["hidden_equivalent_candidates"]) for row in states)
        / max(len(states), 1),
        "first_stop_budget_mean_by_template": {
            f"{split}/{template}": sum(values) / max(len(values), 1)
            for (split, template), values in sorted(stop_budget.items())
        },
    }


def build_records(
    rng: random.Random,
    split: str,
    sizes: list[int],
    per_cell: int,
    query_pool_cases: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    templates = ["pair_affine_mod", "pair_compare_gate"]
    min_candidates = {
        "pair_affine_mod": 24,
        "pair_compare_gate": 96,
    }
    for n in sizes:
        for template in templates:
            for i in range(per_cell):
                record_id = f"{split}_n{n}_{template}_{i:04d}"
                records.append(
                    generate_record(
                        rng,
                        record_id=record_id,
                        split=split,
                        library_size=n,
                        template=template,
                        min_initial_candidates=min_candidates[template],
                        query_pool_cases=query_pool_cases,
                    )
                )
    return records


def build_states(records: list[dict[str, Any]], max_budget: int) -> list[dict[str, Any]]:
    operator_cache: dict[int, Any] = {}
    states: list[dict[str, Any]] = []
    for record in tqdm(records, desc="rollout-states"):
        operators = operator_cache.setdefault(record["library_size"], build_operator_library(record["library_size"]))
        for state in build_rollout_states(record, operators, max_budget=max_budget):
            state["state_id"] = f"{record['record_id']}_b{state['budget']:02d}"
            states.append(state)
    return states


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--train-per-cell", type=int, default=40)
    parser.add_argument("--eval-per-cell", type=int, default=20)
    parser.add_argument("--query-pool-cases", type=int, default=96)
    parser.add_argument("--max-budget", type=int, default=10)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    train_sizes = [64, 128, 256]
    eval_sizes = [64, 128, 256, 512]
    train_records = build_records(rng, "train", train_sizes, args.train_per_cell, args.query_pool_cases)
    eval_records = build_records(rng, "eval", eval_sizes, args.eval_per_cell, args.query_pool_cases)
    train_states = build_states(train_records, args.max_budget)
    eval_states = build_states(eval_records, args.max_budget)

    write_jsonl(args.out_dir / "train_records.jsonl", train_records)
    write_jsonl(args.out_dir / "eval_records.jsonl", eval_records)
    write_jsonl(args.out_dir / "train_budget_states.jsonl", train_states)
    write_jsonl(args.out_dir / "eval_budget_states.jsonl", eval_states)

    manifest = {
        "seed": args.seed,
        "query_pool_cases": args.query_pool_cases,
        "visible_cases": 4,
        "hidden_cases": 16,
        "max_budget": args.max_budget,
        "train_sizes": train_sizes,
        "eval_sizes": eval_sizes,
        "templates": ["pair_affine_mod", "pair_compare_gate"],
        "train_records": summarize_records(train_records),
        "eval_records": summarize_records(eval_records),
        "train_states": summarize_states(train_states),
        "eval_states": summarize_states(eval_states),
    }
    (args.out_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
