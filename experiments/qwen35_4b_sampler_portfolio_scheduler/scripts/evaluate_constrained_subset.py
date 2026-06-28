#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.jsonl import load_jsonl, write_json, write_jsonl  # noqa: E402


def hidden(candidate: dict[str, Any]) -> bool:
    return bool(candidate.get("full_pass"))


def visible(candidate: dict[str, Any]) -> bool:
    return bool(candidate.get("visible_all_pass"))


def parsed(candidate: dict[str, Any]) -> bool:
    return candidate.get("parse_status") == "parsed"


def tokens(candidate: dict[str, Any]) -> int:
    return int(candidate.get("forward_tokens") or 0)


def by_task(path: Path) -> dict[int, dict[str, Any]]:
    return {int(row["task_id"]): row for row in load_jsonl(path)}


class Pools:
    def __init__(self, hot4: dict[int, dict[str, Any]], hot8: dict[int, dict[str, Any]], constrained: dict[int, dict[str, Any]]) -> None:
        self.hot4 = hot4
        self.hot8 = hot8
        self.constrained = constrained
        self.task_ids = sorted(set(hot4) & set(hot8) & set(constrained))

    def record(self, task_id: int) -> dict[str, Any]:
        return self.hot8[task_id]

    def hot(self, task_id: int, k: int) -> list[dict[str, Any]]:
        source = self.hot8[task_id] if k > 4 else self.hot4[task_id]
        return source.get("candidates", [])[:k]

    def constrained4(self, task_id: int) -> list[dict[str, Any]]:
        return self.constrained[task_id].get("candidates", [])[:4]


def record_metrics(record: dict[str, Any], candidates: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    denom = max(len(candidates), 1)
    return {
        "record_id": record["record_id"],
        "task_id": int(record["task_id"]),
        "arm_name": arm,
        "candidate_count": len(candidates),
        "coverage": any(hidden(c) for c in candidates),
        "pass1_proxy": bool(candidates and hidden(candidates[0])),
        "visible_coverage": any(visible(c) for c in candidates),
        "parse_success_count": sum(parsed(c) for c in candidates),
        "visible_candidate_count": sum(visible(c) for c in candidates),
        "hidden_pass_candidate_count": sum(hidden(c) for c in candidates),
        "distinct_functional_rate": len({str(c.get("functional_signature")) for c in candidates}) / denom,
        "distinct_behavior_rate": len({str(c.get("behavior_signature")) for c in candidates}) / denom,
        "forward_tokens": sum(tokens(c) for c in candidates),
        "sources": [c.get("source") for c in candidates],
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    return {
        "records": n,
        "coverage": sum(r["coverage"] for r in rows) / n,
        "pass1_proxy": sum(r["pass1_proxy"] for r in rows) / n,
        "visible_coverage": sum(r["visible_coverage"] for r in rows) / n,
        "candidate_count_mean": sum(r["candidate_count"] for r in rows) / n,
        "parse_success_mean": sum(r["parse_success_count"] for r in rows) / n,
        "visible_candidates_mean": sum(r["visible_candidate_count"] for r in rows) / n,
        "hidden_pass_candidates_mean": sum(r["hidden_pass_candidate_count"] for r in rows) / n,
        "distinct_functional_rate_mean": sum(r["distinct_functional_rate"] for r in rows) / n,
        "distinct_behavior_rate_mean": sum(r["distinct_behavior_rate"] for r in rows) / n,
        "forward_tokens": sum(r["forward_tokens"] for r in rows),
        "covered_tasks": sorted(r["task_id"] for r in rows if r["coverage"]),
        "pass1_tasks": sorted(r["task_id"] for r in rows if r["pass1_proxy"]),
    }


def evaluate(pools: Pools, name: str, chooser: Callable[[Pools, int], list[dict[str, Any]]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = [record_metrics(pools.record(task_id), chooser(pools, task_id), name) for task_id in pools.task_ids]
    return {"arm_name": name, "records": summarize(rows)}, rows


def oracle_choose_arm(pools: Pools, task_id: int) -> list[dict[str, Any]]:
    options = [
        pools.hot(task_id, 4),
        pools.hot(task_id, 8),
        pools.constrained4(task_id),
        pools.hot(task_id, 4) + pools.constrained4(task_id),
    ]
    options.sort(key=lambda rows: (any(hidden(c) for c in rows), -sum(tokens(c) for c in rows), -len(rows)), reverse=True)
    return options[0]


def visible_gate_hot4_then_constrained(pools: Pools, task_id: int) -> list[dict[str, Any]]:
    hot = pools.hot(task_id, 4)
    if any(visible(c) for c in hot):
        return hot
    return hot + pools.constrained4(task_id)


def visible_gate_hot2_then_constrained(pools: Pools, task_id: int) -> list[dict[str, Any]]:
    hot = pools.hot(task_id, 2)
    if any(visible(c) for c in hot):
        return hot
    return hot + pools.constrained4(task_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hot4", type=Path, required=True)
    parser.add_argument("--hot8", type=Path, required=True)
    parser.add_argument("--constrained", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-task-out", type=Path, required=True)
    args = parser.parse_args()

    pools = Pools(by_task(args.hot4), by_task(args.hot8), by_task(args.constrained))
    arms: list[tuple[str, Callable[[Pools, int], list[dict[str, Any]]]]] = [
        ("subset_base_hot_k4", lambda p, t: p.hot(t, 4)),
        ("subset_base_hot_k8", lambda p, t: p.hot(t, 8)),
        ("subset_constrained_k4", lambda p, t: p.constrained4(t)),
        ("subset_hot2_plus_constrained4", lambda p, t: p.hot(t, 2) + p.constrained4(t)),
        ("subset_hot4_plus_constrained4", lambda p, t: p.hot(t, 4) + p.constrained4(t)),
        ("subset_constrained4_plus_hot4", lambda p, t: p.constrained4(t) + p.hot(t, 4)),
        ("subset_visible_gate_hot2_then_constrained", visible_gate_hot2_then_constrained),
        ("subset_visible_gate_hot4_then_constrained", visible_gate_hot4_then_constrained),
        ("subset_oracle_choose_arm", oracle_choose_arm),
    ]
    summaries = []
    per_task = []
    for name, chooser in arms:
        summary, rows = evaluate(pools, name, chooser)
        summaries.append(summary)
        per_task.extend(rows)
    payload = {"task_count": len(pools.task_ids), "results": summaries}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, payload)
    write_jsonl(args.per_task_out, per_task)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
