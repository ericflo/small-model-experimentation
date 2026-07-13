#!/usr/bin/env python3
"""Evaluate old-family recovery/loop retention with the current pinned backend."""

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
import retention_tasks_legacy  # noqa: E402
from vllm_runner import SamplingConfig  # noqa: E402


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def adapt(task: retention_tasks_legacy.RepoTask) -> repo_tasks.RepoTask:
    source_path = task.oracle_patches[0].path
    return repo_tasks.RepoTask(
        task_id=task.task_id,
        family=task.family,
        split=task.split,
        issue=task.issue,
        files=task.files,
        hidden_test=task.hidden_test,
        oracle_patches=tuple(
            repo_tasks.Patch(row.path, row.old, row.new) for row in task.oracle_patches
        ),
        partial_patches=tuple(
            repo_tasks.Patch(row.path, row.old, row.new) for row in task.partial_patches
        ),
        difficulty=task.difficulty,
        pair_id=task.task_id,
        branch=0,
        evidence_channel="legacy_retention",
        evidence_path=source_path,
        evidence_path_regime="legacy_retention",
        acquisition_query_skin="symbol",
        acquisition_query=source_path.removeprefix("src/").removesuffix(".py"),
        evidence_marker="",
        explicit_contract=True,
    )


def aggregate(rows: list[dict], answer_max_tokens: int) -> dict:
    by_case: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_case[row["case_id"]].append(row)
    cases = []
    for case_id, trajectories in sorted(by_case.items()):
        first = trajectories[0]
        cases.append({
            "case_id": case_id,
            "task_id": first["task_id"],
            "family": first["family"],
            "scenario": first["scenario"],
            "success": any(row["workspace_success"] for row in trajectories),
            "verified": any(row["verified_after_final_patch"] for row in trajectories),
            "commit": any(row["commit_after_pass"] for row in trajectories),
            "rejected_transition": any(
                row["rejected_patch_valid_changed_within_two"] for row in trajectories
            ),
            "failed_transition": any(
                row["failed_test_changed_patch_within_two"] for row in trajectories
            ),
            "sampled_tokens": sum(row["sampled_tokens"] for row in trajectories),
            "logical_model_tokens": sum(
                row["logical_model_tokens"] for row in trajectories
            ),
            "turns": sum(row["turns"] for row in trajectories),
            "trajectory_costs": [
                {
                    "trajectory": row["trajectory"],
                    "sampled_tokens": row["sampled_tokens"],
                    "logical_model_tokens": row["logical_model_tokens"],
                    "workspace_success": row["workspace_success"],
                }
                for row in sorted(trajectories, key=lambda item: item["trajectory"])
            ],
        })
    all_steps = [step for row in rows for step in row["steps"]]
    contacts = [
        step for step in all_steps
        if step["n_answer_tokens"] >= answer_max_tokens
        or step.get("finish_reason") == "length"
    ]
    total_turns = sum(row["turns"] for row in rows)
    def rate(field: str, subset: list[dict] = cases) -> float:
        return sum(bool(row[field]) for row in subset) / len(subset) if subset else 0.0
    per_family = {}
    for family in sorted({row["family"] for row in cases}):
        subset = [row for row in cases if row["family"] == family]
        per_family[family] = {
            "n": len(subset),
            "success": rate("success", subset),
            "verified": rate("verified", subset),
            "commit": rate("commit", subset),
            "rejected_transition": rate("rejected_transition", subset),
            "failed_transition": rate("failed_transition", subset),
        }
    return {
        "n_cases": len(cases),
        "success": rate("success"),
        "verified": rate("verified"),
        "commit": rate("commit"),
        "rejected_transition": rate("rejected_transition"),
        "failed_transition": rate("failed_transition"),
        "invalid_action_rate_per_turn": sum(row["invalid_actions"] for row in rows)
        / max(total_turns, 1),
        "answer_cap_hit_rate_per_turn": len(contacts) / max(total_turns, 1),
        "mean_sampled_tokens": statistics.mean(row["sampled_tokens"] for row in cases),
        "mean_logical_model_tokens": statistics.mean(
            row["logical_model_tokens"] for row in cases
        ),
        "per_family": per_family,
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXP / "configs" / "default.yaml")
    parser.add_argument("--design-lock", type=Path, required=True)
    parser.add_argument("--arm", choices=["start", "candidate"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--expected-weight-sha256", required=True)
    parser.add_argument("--block", required=True)
    parser.add_argument("--scenario-set", choices=["normal", "recovery"], required=True)
    parser.add_argument("--mode", choices=["deep", "sample_more", "sample_pool"], default="deep")
    parser.add_argument("--answer-max-tokens", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        harness.validate_model_execution_lock(
            EXP, args.design_lock, "scripts/eval_retention.py"
        )
    except ValueError as exc:
        raise SystemExit(f"model execution is not design-locked: {exc}") from exc
    try:
        harness.validate_canonical_config_path(EXP, args.config)
    except ValueError as exc:
        raise SystemExit(f"retention config is not frozen: {exc}") from exc
    cfg = yaml.safe_load(args.config.read_text())
    if args.mode != "deep":
        raise SystemExit("legacy retention model execution is frozen to deep mode")
    if args.answer_max_tokens not in cfg["evaluation"]["interface_answer_rungs"]:
        raise SystemExit("retention answer-max-tokens is not a frozen interface rung")
    block = cfg["evaluation"]["blocks"].get(args.block)
    if not block or not block.get("legacy_retention"):
        raise SystemExit(f"not a registered legacy retention block: {args.block}")
    families = tuple(cfg["families"][block["families"]])
    legacy = retention_tasks_legacy.make_tasks(
        families, int(block["tasks_per_family"]), int(block["seed"]), args.block
    )
    tasks = [adapt(task) for task in legacy]
    scenarios = ("normal",) if args.scenario_set == "normal" else (
        "rejected_patch", "failed_test"
    )
    budget = cfg["evaluation"][args.scenario_set]
    if args.mode == "deep":
        trajectories = 1
        turns = int(budget["deep_turns"])
        greedy = True
        seed = int(cfg["evaluation"]["deep_run_seed"])
    elif args.mode == "sample_more":
        trajectories = int(budget["sample_more_trajectories"])
        turns = int(budget["sample_more_turns_each"])
        greedy = False
        seed = int(cfg["evaluation"]["sample_run_seed"])
    else:
        trajectories = int(cfg["evaluation"]["sample_more_pool_trajectories"])
        turns = int(budget["sample_more_turns_each"])
        greedy = False
        seed = int(cfg["evaluation"]["sample_run_seed"])
    per_call = int(cfg["evaluation"]["think_budget"]) + args.answer_max_tokens
    reserved = trajectories * turns * per_call
    deep_reserved = int(budget["deep_turns"]) * per_call
    if args.mode != "sample_pool" and reserved != deep_reserved:
        raise SystemExit("retention compute reservation mismatch")
    model = resolve(args.model).resolve()
    checkpoint_role = "start" if args.arm == "start" else "evidence_binding"
    try:
        checkpoint = harness.validate_registered_checkpoint(
            EXP, model, cfg, args.design_lock, checkpoint_role
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"retention checkpoint is not registered: {exc}") from exc
    observed = checkpoint["model_weight_sha256"]
    if observed != args.expected_weight_sha256:
        raise SystemExit("retention model hash differs from the registered invocation")
    tokenizer_provenance = {
        key: checkpoint[key] for key in harness.TOKENIZER_PROVENANCE_KEYS
    }
    runner = harness.make_runner(cfg["engine"], model_override=str(model))
    sampling = SamplingConfig(
        thinking="budget",
        thinking_budget=int(cfg["evaluation"]["think_budget"]),
        n=1,
        answer_max_tokens=args.answer_max_tokens,
        greedy=greedy,
        temperature=None if greedy else float(cfg["evaluation"]["sample_temperature"]),
        top_p=None if greedy else float(cfg["evaluation"]["sample_top_p"]),
        top_k=None if greedy else int(cfg["evaluation"]["sample_top_k"]),
        run_seed=seed,
    )
    specs = [
        (task, trajectory, scenario)
        for task in tasks for scenario in scenarios for trajectory in range(trajectories)
    ]
    rows, summaries = repo_agent.run_episodes(
        runner, specs, sampling, max_turns=turns
    )
    result = {
        "schema_version": 1,
        "arm": args.arm,
        "model": str(model.resolve()),
        "model_weight_sha256": observed,
        "model_config_sha256": checkpoint["model_config_sha256"],
        "model_generation_config_sha256": checkpoint[
            "generation_config_sha256"
        ],
        "merge_receipt_sha256": checkpoint["merge_receipt_sha256"],
        **tokenizer_provenance,
        "config_sha256": sha256_file(args.config),
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
        "repo_agent_sha256": sha256_file(EXP / "src" / "repo_agent.py"),
        "task_generator_sha256": sha256_file(
            EXP / "src" / "retention_tasks_legacy.py"
        ),
        "block": args.block,
        "scenario_set": args.scenario_set,
        "mode": args.mode,
        "answer_max_tokens": args.answer_max_tokens,
        "think_budget": int(cfg["evaluation"]["think_budget"]),
        "tasks_per_family": int(block["tasks_per_family"]),
        "reserved_sampled_tokens_per_case": reserved,
        "deep_reserved_sampled_tokens_per_case": deep_reserved,
        "task_manifest_sha256": repo_tasks.manifest_digest(tasks),
        "history_policy": "canonical_first_valid_tool_call",
        "runner_summaries": summaries,
        "aggregate": aggregate(rows, args.answer_max_tokens),
        "trajectories": rows,
    }
    bank.assert_firewall_clean(result, tasks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        **{key: value for key, value in result["aggregate"].items() if key not in ("cases", "per_family")},
        "per_family": result["aggregate"]["per_family"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
