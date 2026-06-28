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

from src.operator_search import active_trace, candidate_summary, complete_operator_hole  # noqa: E402
from src.operator_tasks import CLOSED_OPERATOR_NAMES, FULL_OPERATOR_NAMES  # noqa: E402


ARMS = {
    "arm1_closed_vocab": CLOSED_OPERATOR_NAMES,
    "arm0_full_inventory": FULL_OPERATOR_NAMES,
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def metric(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    successes = sum(1 for row in rows if row.get(key))
    return {"successes": successes, "records": len(rows), "rate": successes / len(rows) if rows else 0.0}


def avg(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row.get(key, 0)) for row in rows) / len(rows) if rows else 0.0


def summarize(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    out = []
    for key_values, subset in sorted(groups.items()):
        data = dict(zip(keys, key_values))
        data.update(
            {
                "records": len(subset),
                "target_in_raw_candidates": metric(subset, "target_in_raw_candidates"),
                "target_in_visible_candidates": metric(subset, "target_in_visible_candidates"),
                "candidate_oracle_hidden_all": metric(subset, "candidate_oracle_hidden_all"),
                "selected_hidden_all": metric(subset, "selected_hidden_all"),
                "selected_visible_all": metric(subset, "selected_visible_all"),
                "avg_raw_candidate_count": round(avg(subset, "raw_candidate_count"), 3),
                "avg_visible_consistent_count": round(avg(subset, "visible_consistent_count"), 3),
                "avg_visible_consistent_operator_count": round(avg(subset, "visible_consistent_operator_count"), 3),
            }
        )
        out.append(data)
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
                "avg_operator_candidate_count": round(avg(subset, "operator_candidate_count"), 3),
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
    parser.add_argument("--max-candidates", type=int, default=10000)
    parser.add_argument("--max-records", type=int)
    args = parser.parse_args()

    records = load_jsonl(args.data)
    if args.max_records:
        records = records[: args.max_records]
    budgets = [int(item) for item in args.budgets.split(",") if item.strip()]
    candidate_rows: list[dict[str, Any]] = []
    active_rows: list[dict[str, Any]] = []

    for record in tqdm(records, desc="operator-search"):
        for arm, operators in ARMS.items():
            programs = complete_operator_hole(record, operators, max_candidates=args.max_candidates)
            summary = candidate_summary(record, programs)
            candidate_rows.append(
                {
                    "id": record["id"],
                    "family": record["family"],
                    "operator": record["operator"],
                    "operator_status": record["operator_status"],
                    "operator_signature": record["operator_signature"],
                    "template": record["template"],
                    "composition_depth": record["composition_depth"],
                    "arm": arm,
                    "operator_inventory": operators,
                    "target_program": record["target_program"],
                    **summary,
                }
            )
            for policy in ["active_max_split", "oracle_elimination"]:
                for row in active_trace(record, programs, budgets=budgets, policy=policy):
                    active_rows.append(
                        {
                            "id": record["id"],
                            "family": record["family"],
                            "operator": record["operator"],
                            "operator_status": record["operator_status"],
                            "template": record["template"],
                            "composition_depth": record["composition_depth"],
                            "arm": arm,
                            **row,
                        }
                    )

    result = {
        "data": str(args.data),
        "records": len(records),
        "arms": ARMS,
        "budgets": budgets,
        "max_candidates": args.max_candidates,
        "summary_by_status": summarize(candidate_rows, ["arm", "operator_status"]),
        "summary_by_operator": summarize(candidate_rows, ["arm", "operator_status", "operator"]),
        "summary_by_template": summarize(candidate_rows, ["arm", "operator_status", "template"]),
        "active_summary_by_status": summarize_active(active_rows, ["arm", "operator_status", "policy", "budget"]),
        "candidate_rows": candidate_rows,
        "active_rows": active_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"candidate_rows": len(candidate_rows), "active_rows": len(active_rows)}, indent=2))


if __name__ == "__main__":
    main()

