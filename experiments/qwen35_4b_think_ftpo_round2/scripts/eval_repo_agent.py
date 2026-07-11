#!/usr/bin/env python3
"""Evaluate a merged model in the held-out iterative repository-repair harness."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

import harness  # noqa: E402
import repo_agent  # noqa: E402
import repo_tasks  # noqa: E402
from vllm_runner import SamplingConfig  # noqa: E402


def aggregate(trajectories: list[dict], tasks, mode: str) -> dict:
    by_task = defaultdict(list)
    for row in trajectories:
        by_task[row["task_id"]].append(row)
    task_rows = []
    for task in tasks:
        rows = by_task[task.task_id]
        if not rows:
            raise RuntimeError(f"missing trajectory for {task.task_id}")
        task_rows.append({
            "task_id": task.task_id, "family": task.family,
            "success": any(r["success"] for r in rows),
            "patch_correct": any(r["patch_correct"] for r in rows),
            "submitted": any(r["submitted"] for r in rows),
            "sampled_tokens": sum(r["sampled_tokens"] for r in rows),
            "turns": sum(r["turns"] for r in rows),
            "invalid_actions": sum(r["invalid_actions"] for r in rows),
            "patch_calls": sum(r["patch_calls"] for r in rows),
        })
    per_family = {}
    for family in sorted({r["family"] for r in task_rows}):
        rows = [r for r in task_rows if r["family"] == family]
        per_family[family] = {
            "n": len(rows), "success": sum(r["success"] for r in rows) / len(rows),
            "patch_correct": sum(r["patch_correct"] for r in rows) / len(rows),
            "submit_rate": sum(r["submitted"] for r in rows) / len(rows),
        }
    n = len(task_rows)
    return {
        "mode": mode, "n_tasks": n,
        "success": sum(r["success"] for r in task_rows) / n,
        "patch_correct": sum(r["patch_correct"] for r in task_rows) / n,
        "submit_rate": sum(r["submitted"] for r in task_rows) / n,
        "mean_sampled_tokens": statistics.mean(r["sampled_tokens"] for r in task_rows),
        "max_sampled_tokens": max(r["sampled_tokens"] for r in task_rows),
        "mean_turns": statistics.mean(r["turns"] for r in task_rows),
        "invalid_action_rate_per_turn": (
            sum(r["invalid_actions"] for r in task_rows)
            / max(sum(r["turns"] for r in task_rows), 1)),
        "mean_patch_calls": statistics.mean(r["patch_calls"] for r in task_rows),
        "per_family": per_family, "tasks": task_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--model", default=None, help="merged checkpoint; omitted for base")
    parser.add_argument("--mode", choices=["deep", "sample_more"], default="deep")
    parser.add_argument("--tasks-per-family", type=int, default=None, help="smoke override")
    args = parser.parse_args()

    cfg = yaml.safe_load((EXP / "configs" / "default.yaml").read_text())
    acfg = cfg["repo_agent"]
    n_per = args.tasks_per_family or int(acfg["tasks_per_family"])
    tasks = repo_tasks.make_tasks(list(acfg["families"]), n_per, int(acfg["seed"]))
    max_per_call = int(acfg["think_budget"]) + int(acfg["answer_max_tokens"])
    if args.mode == "deep":
        trajectories = 1; max_turns = int(acfg["deep_turns"]); greedy = True
    else:
        if args.arm != "base":
            raise SystemExit("the preregistered sample-more comparator is base-only")
        trajectories = int(acfg["sample_more_trajectories"])
        max_turns = int(acfg["sample_more_turns_each"]); greedy = False
    reserved = trajectories * max_turns * max_per_call
    if reserved != int(acfg["max_sampled_tokens_per_task"]):
        raise SystemExit(f"compute mismatch: reserved {reserved}, frozen cap "
                         f"{acfg['max_sampled_tokens_per_task']}")

    runner = harness.make_runner(cfg["engine"], model_override=args.model)
    if greedy:
        sampling = SamplingConfig(
            thinking="budget", thinking_budget=int(acfg["think_budget"]), n=1,
            answer_max_tokens=int(acfg["answer_max_tokens"]), greedy=True,
            run_seed=8810)
    else:
        sampling = SamplingConfig(
            thinking="budget", thinking_budget=int(acfg["think_budget"]), n=1,
            answer_max_tokens=int(acfg["answer_max_tokens"]), greedy=False,
            temperature=float(acfg["sample_temperature"]),
            top_p=float(acfg["sample_top_p"]), top_k=int(acfg["sample_top_k"]),
            run_seed=8811)
    specs = [(task, trajectory) for task in tasks for trajectory in range(trajectories)]
    started = time.time()
    rows = repo_agent.run_episodes(runner, specs, sampling, max_turns=max_turns)
    out = {
        "arm": args.arm, "model": args.model or cfg["model"]["id"],
        "reserved_sampled_tokens_per_task": reserved,
        "trajectory_rows": rows,
        "aggregate": aggregate(rows, tasks, args.mode),
        "wall_s": time.time() - started,
    }
    suffix = "_smoke" if args.tasks_per_family else ""
    dest = EXP / "runs" / f"repo_agent_{args.arm}_{args.mode}{suffix}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps({"arm": args.arm, "wall_s": out["wall_s"],
                      **{k: v for k, v in out["aggregate"].items()
                         if k not in ("tasks", "per_family")}}, indent=2))
    print(json.dumps(out["aggregate"]["per_family"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
