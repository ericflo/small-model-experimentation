#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.operator_library import build_operator_library  # noqa: E402
from src.search import active_trace, evaluate_candidates  # noqa: E402


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def metric(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    successes = sum(1 for row in rows if row.get(key))
    return {"successes": successes, "records": len(rows), "rate": successes / len(rows) if rows else 0.0}


def avg(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row.get(key, 0)) for row in rows) / len(rows) if rows else 0.0


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    weight = pos - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    out = []
    for key_values, subset in sorted(groups.items()):
        target_ranks = [float(row["target_rank"]) for row in subset]
        data = dict(zip(keys, key_values))
        data.update(
            {
                "records": len(subset),
                "target_in_raw_candidates": metric(subset, "target_in_raw_candidates"),
                "target_in_visible_candidates": metric(subset, "target_in_visible_candidates"),
                "candidate_oracle_hidden_all": metric(subset, "candidate_oracle_hidden_all"),
                "selected_hidden_all": metric(subset, "selected_hidden_all"),
                "avg_raw_candidate_count": round(avg(subset, "raw_candidate_count"), 3),
                "avg_visible_consistent_count": round(avg(subset, "visible_consistent_count"), 3),
                "avg_visible_consistent_operator_count": round(avg(subset, "visible_consistent_operator_count"), 3),
                "target_rank_p50": round(quantile(target_ranks, 0.5), 3),
                "target_rank_p90": round(quantile(target_ranks, 0.9), 3),
            }
        )
        out.append(data)
    return out


def summarize_prefix(rows: list[dict[str, Any]], keys: list[str], budgets: list[int]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    out = []
    for key_values, subset in sorted(groups.items()):
        base = dict(zip(keys, key_values))
        for budget in budgets:
            successes = sum(1 for row in subset if row["prefix_budget_hits"].get(str(budget), False))
            out.append(
                {
                    **base,
                    "budget": budget,
                    "records": len(subset),
                    "target_in_prefix": {
                        "successes": successes,
                        "records": len(subset),
                        "rate": successes / len(subset) if subset else 0.0,
                    },
                }
            )
    return out


def summarize_active(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    out = []
    for key_values, subset in sorted(groups.items()):
        data = dict(zip(keys, key_values))
        data.update(
            {
                "records": len(subset),
                "selected_hidden_all": metric(subset, "selected_hidden_all"),
                "avg_candidate_count": round(avg(subset, "candidate_count"), 3),
                "avg_queries_used": round(avg(subset, "queries_used"), 3),
            }
        )
        out.append(data)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budgets", default="0,1,2,3")
    parser.add_argument("--prefix-budgets", default="1024,4096,16384")
    parser.add_argument("--max-records", type=int)
    args = parser.parse_args()

    records = load_jsonl(args.data)
    if args.max_records:
        records = records[: args.max_records]
    max_library_size = max(record["library_size"] for record in records)
    library = build_operator_library(max_library_size)
    budgets = [int(item) for item in args.budgets.split(",") if item.strip()]
    prefix_budgets = [int(item) for item in args.prefix_budgets.split(",") if item.strip()]

    candidate_rows: list[dict[str, Any]] = []
    active_rows: list[dict[str, Any]] = []

    for record in tqdm(records, desc="scaling-search"):
        summary = evaluate_candidates(record, library)
        candidate_rows.append(
            {
                "id": record["id"],
                "library_size": record["library_size"],
                "template": record["template"],
                "hole_count": record["hole_count"],
                "output_kind": record["output_kind"],
                "target_bucket": record["target_bucket"],
                "target_operators": record["target_operators"],
                "target_program": record["target_program"],
                **summary,
            }
        )
        for policy in ["active_max_split", "oracle_elimination"]:
            for row in active_trace(record, library, budgets=budgets, policy=policy):
                active_rows.append(
                    {
                        "id": record["id"],
                        "library_size": record["library_size"],
                        "template": record["template"],
                        "hole_count": record["hole_count"],
                        "output_kind": record["output_kind"],
                        "target_bucket": record["target_bucket"],
                        "policy": policy,
                        **row,
                    }
                )

    result = {
        "data": str(args.data),
        "records": len(records),
        "budgets": budgets,
        "prefix_budgets": prefix_budgets,
        "summary_by_library_and_depth": summarize(candidate_rows, ["library_size", "hole_count"]),
        "summary_by_library_template": summarize(candidate_rows, ["library_size", "hole_count", "template"]),
        "summary_by_target_bucket": summarize(candidate_rows, ["library_size", "hole_count", "target_bucket"]),
        "prefix_summary_by_library_and_depth": summarize_prefix(candidate_rows, ["library_size", "hole_count"], prefix_budgets),
        "active_summary_by_library_and_depth": summarize_active(
            active_rows,
            ["library_size", "hole_count", "policy", "budget"],
        ),
        "candidate_rows": candidate_rows,
        "active_rows": active_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"candidate_rows": len(candidate_rows), "active_rows": len(active_rows)}, indent=2))


if __name__ == "__main__":
    main()

