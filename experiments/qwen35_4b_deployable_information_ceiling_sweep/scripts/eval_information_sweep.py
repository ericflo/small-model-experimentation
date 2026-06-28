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

from src.operator_env import (  # noqa: E402
    build_operator_library,
    candidate_mask,
    first_prior_pair,
    full_pool_choice,
    hidden_equivalent_count,
    load_jsonl,
    pair_hidden_matches,
)


def evaluate_state(record: dict[str, Any], operators: Any, used: list[int]) -> dict[str, Any]:
    mask = candidate_mask(record, operators, used)
    selected = first_prior_pair(mask)
    target_pair = tuple(record["target_pair"])
    return {
        "candidate_count": int(mask.sum()),
        "target_reachable": bool(mask[target_pair]),
        "selected_pair": list(selected) if selected is not None else None,
        "selected_exact_pair": bool(selected == target_pair),
        "selected_hidden_all": pair_hidden_matches(record, operators, selected),
        "hidden_equivalent_candidates": hidden_equivalent_count(record, operators, mask),
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["policy"], row["visible_total"], row["library_size"], row["template"], row["budget"])].append(row)
    summary: list[dict[str, Any]] = []
    for (policy, visible_total, library_size, template, budget), items in sorted(groups.items()):
        n = len(items)
        summary.append(
            {
                "policy": policy,
                "visible_total": visible_total,
                "library_size": library_size,
                "template": template,
                "budget": budget,
                "records": n,
                "selected_hidden_all": sum(1 for row in items if row["selected_hidden_all"]) / n,
                "selected_exact_pair": sum(1 for row in items if row["selected_exact_pair"]) / n,
                "target_reachable": sum(1 for row in items if row["target_reachable"]) / n,
                "candidate_count_mean": sum(int(row["candidate_count"]) for row in items) / n,
                "hidden_equivalent_candidates_mean": sum(int(row["hidden_equivalent_candidates"]) for row in items) / n,
            }
        )
    return summary


def run_policy(
    records: list[dict[str, Any]],
    policy: str,
    visible_extra: int,
    max_budget: int,
) -> dict[str, Any]:
    operator_cache: dict[int, Any] = {}
    rows: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    for record in tqdm(records, desc=f"{policy}-visible+{visible_extra}"):
        operators = operator_cache.setdefault(record["library_size"], build_operator_library(record["library_size"]))
        used = list(range(visible_extra))
        base_metrics = evaluate_state(record, operators, used)
        rows.append(
            {
                "policy": policy,
                "record_id": record["record_id"],
                "library_size": record["library_size"],
                "template": record["template"],
                "visible_base": len(record["visible_cases"]),
                "visible_extra": visible_extra,
                "visible_total": len(record["visible_cases"]) + visible_extra,
                "budget": 0,
                "used_query_indices": list(used),
                **base_metrics,
            }
        )
        for step in range(max_budget):
            if len(used) >= len(record["query_pool"]):
                break
            if policy == "greedy_uniform_split":
                choice = full_pool_choice(record, operators, used, "fullpool_max_split")
            elif policy == "target_aware_oracle":
                choice = full_pool_choice(record, operators, used, "fullpool_oracle")
            else:
                raise ValueError(policy)
            used.append(int(choice["query_index"]))
            metrics = evaluate_state(record, operators, used)
            rows.append(
                {
                    "policy": policy,
                    "record_id": record["record_id"],
                    "library_size": record["library_size"],
                    "template": record["template"],
                    "visible_base": len(record["visible_cases"]),
                    "visible_extra": visible_extra,
                    "visible_total": len(record["visible_cases"]) + visible_extra,
                    "budget": step + 1,
                    "used_query_indices": list(used),
                    "chosen_query_index": int(choice["query_index"]),
                    "chosen_reward": choice.get("reward"),
                    **metrics,
                }
            )
            actions.append(
                {
                    "policy": policy,
                    "record_id": record["record_id"],
                    "library_size": record["library_size"],
                    "template": record["template"],
                    "visible_total": len(record["visible_cases"]) + visible_extra,
                    "step": step,
                    "chosen_query_index": int(choice["query_index"]),
                    "chosen_reward": choice.get("reward"),
                }
            )
    return {
        "policy": policy,
        "visible_extra": visible_extra,
        "visible_total": len(records[0]["visible_cases"]) + visible_extra if records else visible_extra,
        "max_budget": max_budget,
        "records": rows,
        "actions": actions,
        "summary": summarize(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, default=ROOT / "data" / "eval_records.jsonl")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports" / "eval")
    parser.add_argument("--max-budget", type=int, default=10)
    parser.add_argument("--visible-extra", type=int, nargs="+", default=[0, 4, 8, 12])
    parser.add_argument("--policies", nargs="+", default=["greedy_uniform_split", "target_aware_oracle"])
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    records = load_jsonl(args.records)
    if args.limit:
        records = records[: args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for visible_extra in args.visible_extra:
        if visible_extra + args.max_budget >= len(records[0]["query_pool"]):
            raise ValueError("visible_extra + max_budget must leave enough query-pool cases")
        for policy in args.policies:
            payload = run_policy(records, policy, visible_extra, args.max_budget)
            out = args.out_dir / f"{policy}_visible{payload['visible_total']}.json"
            out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            print(json.dumps({"out": str(out), "rows": len(payload["records"])}, indent=2))


if __name__ == "__main__":
    main()
