#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.jsonl import load_jsonl, write_json, write_jsonl
from src.mbpp_env import execute_public_and_asserts
from src.templates import generate_candidates


RUNG_ARMS = {
    "manual_core": {"manual_core"},
    "manual_expanded": {"manual_core", "manual_expanded"},
    "retrieved_transplant": {"retrieved_transplant"},
    "combined": {"manual_core", "manual_expanded", "retrieved_transplant"},
}


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "task_count": len(results),
        "rungs": {},
        "per_task": {},
    }
    for rung, allowed_arms in RUNG_ARMS.items():
        solved = 0
        visible_pass_candidates = 0
        hidden_pass_candidates = 0
        visible_hidden_fail_candidates = 0
        total_candidates = 0
        solved_task_ids: list[int] = []
        for rec in results:
            cands = [cand for cand in rec["candidates"] if cand["arm"] in allowed_arms]
            total_candidates += len(cands)
            visible = [cand for cand in cands if cand["visible_all_pass"]]
            hidden = [cand for cand in cands if cand["full_pass"]]
            visible_pass_candidates += len(visible)
            hidden_pass_candidates += len(hidden)
            visible_hidden_fail_candidates += len([cand for cand in visible if not cand["full_pass"]])
            if hidden:
                solved += 1
                solved_task_ids.append(rec["task_id"])
        false_pass_rate = (
            visible_hidden_fail_candidates / visible_pass_candidates if visible_pass_candidates else 0.0
        )
        summary["rungs"][rung] = {
            "coverage": solved / len(results) if results else 0.0,
            "solved_count": solved,
            "solved_task_ids": solved_task_ids,
            "total_candidates": total_candidates,
            "mean_candidates_per_task": total_candidates / len(results) if results else 0.0,
            "visible_pass_candidates": visible_pass_candidates,
            "hidden_pass_candidates": hidden_pass_candidates,
            "visible_hidden_fail_candidates": visible_hidden_fail_candidates,
            "visible_pass_hidden_fail_rate": false_pass_rate,
        }
    for rec in results:
        task_summary: dict[str, Any] = {}
        for rung, allowed_arms in RUNG_ARMS.items():
            cands = [cand for cand in rec["candidates"] if cand["arm"] in allowed_arms]
            task_summary[rung] = {
                "candidate_count": len(cands),
                "visible_pass_count": sum(1 for cand in cands if cand["visible_all_pass"]),
                "hidden_pass_count": sum(1 for cand in cands if cand["full_pass"]),
                "solved": any(cand["full_pass"] for cand in cands),
                "winning_templates": [
                    cand["template_id"] for cand in cands if cand["full_pass"]
                ][:10],
            }
        summary["per_task"][str(rec["task_id"])] = task_summary
    return summary


def run_task(
    record: dict[str, Any],
    train_library: list[dict[str, Any]],
    arms: list[str],
    retrieval_top_k: int,
    timeout_s: float,
) -> dict[str, Any]:
    start = time.time()
    raw_candidates = generate_candidates(record, train_library, arms=arms, retrieval_top_k=retrieval_top_k)
    evaluated: list[dict[str, Any]] = []
    for cand in raw_candidates:
        result = execute_public_and_asserts(
            cand["code"],
            public_cases=record["public_cases"],
            hidden_asserts=record["hidden_asserts"],
            setup_code=record.get("setup_code", ""),
            timeout_s=timeout_s,
        )
        evaluated.append(
            {
                **cand,
                "safe": result.get("safe", False),
                "safety_reason": result.get("safety_reason", ""),
                "public_passed": result.get("public_passed", []),
                "public_outputs": result.get("public_outputs", []),
                "visible_all_pass": bool(result.get("visible_all_pass", False)),
                "full_pass": bool(result.get("full_pass", False)),
                "runtime_error": result.get("runtime_error", ""),
                "timeout": bool(result.get("timeout", False)),
            }
        )
    return {
        "task_id": record["task_id"],
        "record_id": record["record_id"],
        "entry_point": record["entry_point"],
        "task_text": record["task_text"],
        "public_cases": record["public_cases"],
        "candidate_count": len(evaluated),
        "elapsed_s": round(time.time() - start, 3),
        "candidates": evaluated,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, default=ROOT / "data/residual_tasks.jsonl")
    parser.add_argument("--train-library", type=Path, default=ROOT / "data/mbpp_train_library.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--arms",
        nargs="+",
        default=["manual_core", "manual_expanded", "retrieved_transplant"],
        choices=["manual_core", "manual_expanded", "retrieved_transplant"],
    )
    parser.add_argument("--retrieval-top-k", type=int, default=80)
    parser.add_argument("--timeout-s", type=float, default=5.0)
    args = parser.parse_args()

    tasks = load_jsonl(args.tasks)
    if args.limit:
        tasks = tasks[: args.limit]
    train_library = load_jsonl(args.train_library)

    results = []
    for idx, record in enumerate(tasks, start=1):
        print(f"[{idx}/{len(tasks)}] task_id={record['task_id']} entry={record['entry_point']}", flush=True)
        result = run_task(
            record,
            train_library=train_library,
            arms=args.arms,
            retrieval_top_k=args.retrieval_top_k,
            timeout_s=args.timeout_s,
        )
        solved = any(cand["full_pass"] for cand in result["candidates"])
        visible = sum(1 for cand in result["candidates"] if cand["visible_all_pass"])
        print(
            f"  candidates={result['candidate_count']} visible_pass={visible} hidden_solved={solved} elapsed={result['elapsed_s']}s",
            flush=True,
        )
        results.append(result)

    write_jsonl(args.output, results)
    summary = summarize(results)
    write_json(args.summary, summary)
    print(f"wrote {args.output}")
    print(f"wrote {args.summary}")


if __name__ == "__main__":
    main()
