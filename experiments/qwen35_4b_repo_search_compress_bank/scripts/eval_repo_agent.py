#!/usr/bin/env python3
"""Evaluate a merged checkpoint in deep or matched-sampling repository mode."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import yaml

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "src"))

import bank  # noqa: E402
import harness  # noqa: E402
import repo_agent  # noqa: E402
import repo_tasks  # noqa: E402
from vllm_runner import SamplingConfig  # noqa: E402


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def aggregate(rows: list[dict], tasks: list[repo_tasks.RepoTask], mode: str) -> dict:
    by_task = defaultdict(list)
    for row in rows:
        by_task[row["task_id"]].append(row)
    task_rows = []
    for task in tasks:
        trajectories = by_task[task.task_id]
        successful = [row for row in trajectories if row["workspace_success"]]
        verified = [row for row in trajectories if row["verified_after_final_patch"]]
        task_rows.append({
            "task_id": task.task_id,
            "family": task.family,
            "success": bool(successful),
            "verified_after_final_patch": any(row["verified_after_final_patch"] for row in successful),
            "commit_after_pass": any(row["commit_after_pass"] for row in verified),
            "submitted": any(row["submitted"] for row in trajectories),
            "sampled_tokens": sum(row["sampled_tokens"] for row in trajectories),
            "turns": sum(row["turns"] for row in trajectories),
            "invalid_actions": sum(row["invalid_actions"] for row in trajectories),
            "trajectory_successes": [row["workspace_success"] for row in trajectories],
        })
    per_family = {}
    for family in sorted({row["family"] for row in task_rows}):
        subset = [row for row in task_rows if row["family"] == family]
        per_family[family] = {
            "n": len(subset),
            "success": sum(row["success"] for row in subset) / len(subset),
            "submit_rate": sum(row["submitted"] for row in subset) / len(subset),
        }
    n = len(task_rows)
    successes = [row for row in task_rows if row["success"]]
    verified = [row for row in task_rows if row["verified_after_final_patch"]]
    total_turns = sum(row["turns"] for row in task_rows)
    return {
        "mode": mode,
        "n_tasks": n,
        "success": sum(row["success"] for row in task_rows) / n,
        "submit_rate": sum(row["submitted"] for row in task_rows) / n,
        "verified_given_success": (
            sum(row["verified_after_final_patch"] for row in successes) / len(successes)
            if successes else 0.0
        ),
        "commit_given_verified": (
            sum(row["commit_after_pass"] for row in verified) / len(verified)
            if verified else 0.0
        ),
        "invalid_action_rate_per_turn": sum(row["invalid_actions"] for row in task_rows) / max(total_turns, 1),
        "mean_sampled_tokens": statistics.mean(row["sampled_tokens"] for row in task_rows),
        "max_sampled_tokens": max(row["sampled_tokens"] for row in task_rows),
        "mean_turns": statistics.mean(row["turns"] for row in task_rows),
        "per_family": per_family,
        "tasks": task_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXP / "configs" / "default.yaml")
    parser.add_argument("--arm", required=True)
    parser.add_argument("--model", default=None, help="merged checkpoint; omit for pinned base")
    parser.add_argument("--block", choices=["trained_dev", "transfer_dev", "transfer_confirm"], required=True)
    parser.add_argument("--mode", choices=["deep", "sample_more"], default="deep")
    parser.add_argument("--tasks-per-family", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    ecfg = cfg["evaluation"]
    block = ecfg["blocks"][args.block]
    families = cfg["families"][block["families"]]
    n_tasks = args.tasks_per_family or int(block["tasks_per_family"])
    tasks = repo_tasks.make_tasks(tuple(families), n_tasks, int(block["seed"]), args.block)
    per_call = int(ecfg["think_budget"]) + int(ecfg["answer_max_tokens"])
    if args.mode == "deep":
        trajectories = 1
        max_turns = int(ecfg["deep_turns"])
        greedy = True
        run_seed = int(ecfg["deep_run_seed"])
    else:
        trajectories = int(ecfg["sample_more_trajectories"])
        max_turns = int(ecfg["sample_more_turns_each"])
        greedy = False
        run_seed = int(ecfg["sample_run_seed"])
    reserved = trajectories * max_turns * per_call
    expected = int(ecfg["deep_turns"]) * per_call
    if reserved != expected:
        raise SystemExit(f"compute mismatch: mode reserves {reserved}, deep reserves {expected}")

    model_override = None
    model_label = cfg["model"]["id"]
    if args.model and args.model != cfg["model"]["id"]:
        model_path = resolve(args.model)
        if not (model_path / "config.json").is_file():
            raise SystemExit(f"not a merged checkpoint: {model_path}")
        model_override = str(model_path)
        model_label = str(model_path.resolve())
    runner = harness.make_runner(cfg["engine"], model_override=model_override)
    sampling = SamplingConfig(
        thinking="budget",
        thinking_budget=int(ecfg["think_budget"]),
        n=1,
        answer_max_tokens=int(ecfg["answer_max_tokens"]),
        greedy=greedy,
        temperature=None if greedy else float(ecfg["sample_temperature"]),
        top_p=None if greedy else float(ecfg["sample_top_p"]),
        top_k=None if greedy else int(ecfg["sample_top_k"]),
        run_seed=run_seed,
    )
    specs = [(task, trajectory) for task in tasks for trajectory in range(trajectories)]
    rows = repo_agent.run_episodes(runner, specs, sampling, max_turns=max_turns)
    payload = {
        "schema_version": 1,
        "arm": args.arm,
        "model": model_label,
        "block": args.block,
        "mode": args.mode,
        "reserved_sampled_tokens_per_task": reserved,
        "task_manifest_sha256": repo_tasks.manifest_digest(tasks),
        "aggregate": aggregate(rows, tasks, args.mode),
        "trajectories": rows,
    }
    bank.assert_firewall_clean(payload, tasks)
    if args.output:
        output = args.output
    else:
        root = resolve(cfg["artifacts"]["root"])
        output = root / "eval" / f"{args.block}_{args.arm}_{args.mode}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    slim = {key: value for key, value in payload["aggregate"].items() if key not in ("tasks", "per_family")}
    print(json.dumps({"output": str(output), **slim, "per_family": payload["aggregate"]["per_family"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
