#!/usr/bin/env python3
"""Evaluate normal or controlled-recovery coding episodes at matched compute."""

from __future__ import annotations

import argparse
import hashlib
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


def aggregate(rows: list[dict], mode: str, answer_max_tokens: int) -> dict:
    by_case = defaultdict(list)
    for row in rows:
        by_case[row["case_id"]].append(row)
    cases = []
    for case_id in sorted(by_case):
        trajectories = by_case[case_id]
        first = trajectories[0]
        successful = [row for row in trajectories if row["workspace_success"]]
        verified = [row for row in trajectories if row["verified_after_final_patch"]]
        cases.append({
            "case_id": case_id,
            "task_id": first["task_id"],
            "family": first["family"],
            "scenario": first["scenario"],
            "success": bool(successful),
            "verified_after_final_patch": any(
                row["verified_after_final_patch"] for row in successful
            ),
            "commit_after_pass": any(row["commit_after_pass"] for row in verified),
            "submitted": any(row["submitted"] for row in trajectories),
            "rejected_patch_changed_immediately": any(
                row["rejected_patch_changed_immediately"] for row in trajectories
            ),
            "rejected_patch_changed_within_two": any(
                row["rejected_patch_changed_within_two"] for row in trajectories
            ),
            "rejected_patch_valid_changed_within_two": any(
                row["rejected_patch_valid_changed_within_two"] for row in trajectories
            ),
            "failed_test_diagnose_or_revise_immediately": any(
                row["failed_test_diagnose_or_revise_immediately"] for row in trajectories
            ),
            "failed_test_changed_patch_within_two": any(
                row["failed_test_changed_patch_within_two"] for row in trajectories
            ),
            "sampled_tokens": sum(row["sampled_tokens"] for row in trajectories),
            "turns": sum(row["turns"] for row in trajectories),
            "invalid_actions": sum(row["invalid_actions"] for row in trajectories),
            "trajectory_successes": [row["workspace_success"] for row in trajectories],
        })
    per_scenario = {}
    for scenario in sorted({row["scenario"] for row in cases}):
        subset = [row for row in cases if row["scenario"] == scenario]
        per_scenario[scenario] = {
            "n": len(subset),
            "success": sum(row["success"] for row in subset) / len(subset),
            "submit_rate": sum(row["submitted"] for row in subset) / len(subset),
            "immediate_transition_rate": (
                sum(row["rejected_patch_changed_immediately"] for row in subset) / len(subset)
                if scenario == "rejected_patch"
                else sum(
                    row["failed_test_diagnose_or_revise_immediately"] for row in subset
                ) / len(subset)
                if scenario == "failed_test" else None
            ),
            "changed_patch_within_two": (
                sum(row["rejected_patch_changed_within_two"] for row in subset) / len(subset)
                if scenario == "rejected_patch"
                else sum(row["failed_test_changed_patch_within_two"] for row in subset) / len(subset)
                if scenario == "failed_test" else None
            ),
            "valid_changed_patch_within_two": (
                sum(row["rejected_patch_valid_changed_within_two"] for row in subset) / len(subset)
                if scenario == "rejected_patch" else None
            ),
        }
    per_family = {}
    for family in sorted({row["family"] for row in cases}):
        subset = [row for row in cases if row["family"] == family]
        per_family[family] = {
            "n": len(subset),
            "success": sum(row["success"] for row in subset) / len(subset),
            "submit_rate": sum(row["submitted"] for row in subset) / len(subset),
        }
    successful = [row for row in cases if row["success"]]
    verified = [row for row in cases if row["verified_after_final_patch"]]
    total_turns = sum(row["turns"] for row in cases)
    all_steps = [step for row in rows for step in row["steps"]]
    invalid_steps = [step for step in all_steps if step["operator"] == "INVALID"]
    answer_cap_hits = [
        step for step in all_steps if step["n_answer_tokens"] >= answer_max_tokens
    ]
    return {
        "mode": mode,
        "n_cases": len(cases),
        "n_tasks": len({row["task_id"] for row in cases}),
        "success": sum(row["success"] for row in cases) / len(cases),
        "submit_rate": sum(row["submitted"] for row in cases) / len(cases),
        "verified_given_success": (
            sum(row["verified_after_final_patch"] for row in successful) / len(successful)
            if successful else 0.0
        ),
        "commit_given_verified": (
            sum(row["commit_after_pass"] for row in verified) / len(verified)
            if verified else 0.0
        ),
        "invalid_action_rate_per_turn": (
            sum(row["invalid_actions"] for row in cases) / max(total_turns, 1)
        ),
        "answer_cap_hit_rate_per_turn": len(answer_cap_hits) / max(total_turns, 1),
        "invalid_answer_cap_hit_fraction": (
            sum(step["n_answer_tokens"] >= answer_max_tokens for step in invalid_steps)
            / len(invalid_steps)
            if invalid_steps else 0.0
        ),
        "mean_answer_tokens_per_turn": (
            statistics.mean(step["n_answer_tokens"] for step in all_steps)
            if all_steps else 0.0
        ),
        "max_answer_tokens": max((step["n_answer_tokens"] for step in all_steps), default=0),
        "mean_sampled_tokens": statistics.mean(row["sampled_tokens"] for row in cases),
        "max_sampled_tokens": max(row["sampled_tokens"] for row in cases),
        "mean_turns": statistics.mean(row["turns"] for row in cases),
        "per_scenario": per_scenario,
        "per_family": per_family,
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXP / "configs" / "default.yaml")
    parser.add_argument("--arm", required=True)
    parser.add_argument("--model", required=True, help="merged checkpoint")
    parser.add_argument(
        "--block", choices=["headroom_a", "headroom_b"],
        required=True,
    )
    parser.add_argument("--scenario-set", choices=["normal", "recovery"], required=True)
    parser.add_argument("--mode", choices=["deep", "sample_more"], default="deep")
    parser.add_argument("--scaffold", action="store_true")
    parser.add_argument("--tasks-per-family", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    ecfg = cfg["evaluation"]
    block = ecfg["blocks"][args.block]
    families = cfg["families"][block["families"]]
    n_tasks = args.tasks_per_family or int(block["tasks_per_family"])
    tasks = repo_tasks.make_tasks(tuple(families), n_tasks, int(block["seed"]), args.block)
    content_digests = [repo_tasks.content_digest(task) for task in tasks]
    if len(set(content_digests)) != len(content_digests):
        raise SystemExit(f"{args.block} contains duplicate repository content")
    if args.block == "headroom_b":
        prior = ecfg["blocks"]["headroom_a"]
        prior_tasks = repo_tasks.make_tasks(
            tuple(cfg["families"][prior["families"]]),
            int(prior["tasks_per_family"]),
            int(prior["seed"]),
            "headroom_a",
        )
        if set(content_digests) & {
            repo_tasks.content_digest(task) for task in prior_tasks
        }:
            raise SystemExit("headroom B overlaps headroom A repository content")
    scenarios = (
        ("normal",) if args.scenario_set == "normal"
        else ("rejected_patch", "failed_test")
    )
    budget = ecfg[args.scenario_set]
    per_call = int(ecfg["think_budget"]) + int(ecfg["answer_max_tokens"])
    if args.mode == "deep":
        trajectories = 1
        max_turns = int(budget["deep_turns"])
        greedy = True
        run_seed = int(ecfg["deep_run_seed"])
    else:
        trajectories = int(budget["sample_more_trajectories"])
        max_turns = int(budget["sample_more_turns_each"])
        greedy = False
        run_seed = int(ecfg["sample_run_seed"])
    reserved = trajectories * max_turns * per_call
    expected = int(budget["deep_turns"]) * per_call
    if reserved != expected:
        raise SystemExit(f"compute mismatch: mode reserves {reserved}, deep reserves {expected}")
    if args.scaffold and (args.scenario_set != "recovery" or args.mode != "deep"):
        raise SystemExit("the registered external scaffold applies only to deep recovery")

    model_path = resolve(args.model)
    if not (model_path / "config.json").is_file():
        raise SystemExit(f"not a merged checkpoint: {model_path}")
    runner = harness.make_runner(cfg["engine"], model_override=str(model_path))
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
    specs = [
        (task, trajectory, scenario)
        for task in tasks
        for scenario in scenarios
        for trajectory in range(trajectories)
    ]
    rows = repo_agent.run_episodes(
        runner, specs, sampling, max_turns=max_turns, scaffold=args.scaffold
    )
    payload = {
        "schema_version": 1,
        "arm": args.arm,
        "model": str(model_path.resolve()),
        "block": args.block,
        "scenario_set": args.scenario_set,
        "mode": args.mode,
        "scaffold": args.scaffold,
        "reserved_sampled_tokens_per_case": reserved,
        "task_manifest_sha256": repo_tasks.manifest_digest(tasks),
        "task_content_manifest_sha256": hashlib.sha256(
            json.dumps(sorted(content_digests), separators=(",", ":")).encode()
        ).hexdigest(),
        "aggregate": aggregate(rows, args.mode, int(ecfg["answer_max_tokens"])),
        "trajectories": rows,
    }
    bank.assert_firewall_clean(payload, tasks)
    if args.output:
        output = args.output
    else:
        root = resolve(cfg["artifacts"]["root"])
        suffix = "_scaffold" if args.scaffold else ""
        output = root / "eval" / (
            f"{args.block}_{args.scenario_set}_{args.arm}_{args.mode}{suffix}.json"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    slim = {
        key: value for key, value in payload["aggregate"].items()
        if key not in ("cases", "per_family")
    }
    print(json.dumps({"output": str(output), **slim,
                      "per_family": payload["aggregate"]["per_family"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
