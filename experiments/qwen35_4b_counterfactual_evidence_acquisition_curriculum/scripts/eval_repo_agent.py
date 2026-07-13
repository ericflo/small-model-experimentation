#!/usr/bin/env python3
"""Evaluate counterfactual evidence acquisition in the real coding loop."""

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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cross_patch_fails(task: repo_tasks.RepoTask, counterpart: repo_tasks.RepoTask, action: dict | None) -> bool:
    if not action or action.get("tool") != "patch":
        return False
    env = repo_tasks.RepoEnv(counterpart)
    try:
        result = env.patch(
            str(action.get("path", "")),
            str(action.get("old", "")),
            str(action.get("new", "")),
        )
        if not result.startswith("PATCH_OK"):
            return False
        visible, hidden = env.score_workspace()
        return not (visible and hidden)
    finally:
        env.close()


def annotate_cross_branch(rows: list[dict], tasks: list[repo_tasks.RepoTask]) -> None:
    task_by_id = {task.task_id: task for task in tasks}
    counterpart = {
        task.task_id: other
        for task in tasks
        for other in tasks
        if other.pair_id == task.pair_id and other.branch != task.branch
    }
    for row in rows:
        task = task_by_id[row["task_id"]]
        row["first_changed_patch_cross_fails_counterpart"] = _cross_patch_fails(
            task,
            counterpart[task.task_id],
            row.get("first_changed_patch_action"),
        )


def answer_limit_contact(step: dict, answer_max_tokens: int) -> bool:
    """Treat every backend length stop as a cap contact, even below nominal A."""
    return bool(
        step["n_answer_tokens"] >= answer_max_tokens
        or step.get("finish_reason") == "length"
        or (
            step.get("forced_close")
            and step.get("stage2_finish_reason") == "length"
        )
    )


def aggregate(rows: list[dict], mode: str, answer_max_tokens: int) -> dict:
    by_case: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_case[row["case_id"]].append(row)
    cases = []
    for case_id in sorted(by_case):
        trajectories = by_case[case_id]
        first = trajectories[0]
        successful = [row for row in trajectories if row["workspace_success"]]
        verified = [row for row in trajectories if row["verified_after_final_patch"]]
        aligned_rows = [
            row for row in trajectories
            if row["evidence_acquired_before_first_patch"]
            and row["first_changed_patch_full_correct"]
            and row["first_changed_patch_cross_fails_counterpart"]
            and row["first_changed_patch_before_generated_test"]
        ]
        first_correct_rows = [
            row for row in trajectories if row["first_changed_patch_full_correct"]
        ]
        cases.append({
            "case_id": case_id,
            "task_id": first["task_id"],
            "family": first["family"],
            "pair_id": first["pair_id"],
            "branch": first["branch"],
            "evidence_channel": first["evidence_channel"],
            "evidence_path_regime": first["task"]["evidence_path_regime"],
            "acquisition_query_skin": first["task"]["acquisition_query_skin"],
            "explicit_contract": first["explicit_contract"],
            "scenario": first["scenario"],
            "success": bool(successful),
            "preverifier_member_success": bool(aligned_rows),
            "first_changed_patch_full_correct": bool(first_correct_rows),
            "first_changed_patch_before_generated_test": any(
                row["first_changed_patch_before_generated_test"]
                for row in first_correct_rows
            ),
            "evidence_acquired_before_first_patch": any(
                row["evidence_acquired_before_first_patch"] for row in trajectories
            ),
            "cross_branch_failure": any(
                row["first_changed_patch_cross_fails_counterpart"]
                for row in first_correct_rows
            ),
            "unnecessary_evidence_before_first_patch": any(
                row["unnecessary_evidence_before_first_patch"] for row in trajectories
            ),
            "any_search_before_first_patch": any(
                row["any_search_before_first_patch"] for row in trajectories
            ),
            "non_source_inspects_before_first_patch": sum(
                row["non_source_inspects_before_first_patch"] for row in trajectories
            ),
            "verified_after_final_patch": any(
                row["verified_after_final_patch"] for row in trajectories
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
            "logical_model_input_tokens": sum(
                row["logical_model_input_tokens"] for row in trajectories
            ),
            "injected_tokens": sum(row["injected_tokens"] for row in trajectories),
            "logical_model_tokens": sum(
                row["logical_model_tokens"] for row in trajectories
            ),
            "turns": sum(row["turns"] for row in trajectories),
            "invalid_actions": sum(row["invalid_actions"] for row in trajectories),
            "trajectory_successes": [row["workspace_success"] for row in trajectories],
            "trajectory_costs": [
                {
                    "trajectory": row["trajectory"],
                    "sampled_tokens": row["sampled_tokens"],
                    "logical_model_input_tokens": row["logical_model_input_tokens"],
                    "injected_tokens": row["injected_tokens"],
                    "logical_model_tokens": row["logical_model_tokens"],
                    "workspace_success": row["workspace_success"],
                    "preverifier_member_success": bool(
                        row["evidence_acquired_before_first_patch"]
                        and row["first_changed_patch_full_correct"]
                        and row["first_changed_patch_cross_fails_counterpart"]
                        and row["first_changed_patch_before_generated_test"]
                    ),
                }
                for row in sorted(trajectories, key=lambda item: item["trajectory"])
            ],
        })

    dyads = []
    by_pair: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for case in cases:
        by_pair[(case["pair_id"], case["scenario"])].append(case)
    for (pair_id, scenario), members in sorted(by_pair.items()):
        if len(members) != 2 or {member["branch"] for member in members} != {0, 1}:
            raise AssertionError(f"incomplete counterfactual dyad: {pair_id} {scenario}")
        dyads.append({
            "pair_id": pair_id,
            "scenario": scenario,
            "family": members[0]["family"],
            "evidence_channel": members[0]["evidence_channel"],
            "evidence_path_regime": members[0]["evidence_path_regime"],
            "acquisition_query_skin": members[0]["acquisition_query_skin"],
            "explicit_contract": members[0]["explicit_contract"],
            "paired_preverifier_success": all(
                member["preverifier_member_success"] for member in members
            ),
            "paired_first_patch_full_correct": all(
                member["first_changed_patch_full_correct"] for member in members
            ),
            "paired_evidence_acquired": all(
                member["evidence_acquired_before_first_patch"] for member in members
            ),
            "paired_cross_branch_failure": all(
                member["cross_branch_failure"] for member in members
            ),
            "paired_terminal_success": all(member["success"] for member in members),
        })

    def case_rate(field: str, subset: list[dict] = cases) -> float:
        return sum(bool(row[field]) for row in subset) / len(subset) if subset else 0.0

    def dyad_rate(field: str, subset: list[dict] = dyads) -> float:
        return sum(bool(row[field]) for row in subset) / len(subset) if subset else 0.0

    per_scenario = {}
    for scenario in sorted({row["scenario"] for row in cases}):
        subset = [row for row in cases if row["scenario"] == scenario]
        per_scenario[scenario] = {
            "n": len(subset),
            "success": case_rate("success", subset),
            "first_patch_full_correct": case_rate(
                "first_changed_patch_full_correct", subset
            ),
            "evidence_acquired_before_first_patch": case_rate(
                "evidence_acquired_before_first_patch", subset
            ),
            "changed_patch_within_two": (
                case_rate("rejected_patch_valid_changed_within_two", subset)
                if scenario == "rejected_patch"
                else case_rate("failed_test_changed_patch_within_two", subset)
                if scenario == "failed_test"
                else None
            ),
            "valid_changed_patch_within_two": (
                case_rate("rejected_patch_valid_changed_within_two", subset)
                if scenario == "rejected_patch"
                else None
            ),
        }
    per_family = {}
    for family in sorted({row["family"] for row in cases}):
        subset = [row for row in cases if row["family"] == family]
        family_dyads = [row for row in dyads if row["family"] == family]
        per_family[family] = {
            "n": len(subset),
            "success": case_rate("success", subset),
            "first_patch_full_correct": case_rate(
                "first_changed_patch_full_correct", subset
            ),
            "evidence_acquired_before_first_patch": case_rate(
                "evidence_acquired_before_first_patch", subset
            ),
            "paired_preverifier_success": dyad_rate(
                "paired_preverifier_success", family_dyads
            ),
        }
    per_channel = {}
    for channel in sorted({row["evidence_channel"] for row in cases}):
        subset = [row for row in cases if row["evidence_channel"] == channel]
        channel_dyads = [row for row in dyads if row["evidence_channel"] == channel]
        per_channel[channel] = {
            "n": len(subset),
            "success": case_rate("success", subset),
            "evidence_acquired_before_first_patch": case_rate(
                "evidence_acquired_before_first_patch", subset
            ),
            "paired_preverifier_success": dyad_rate(
                "paired_preverifier_success", channel_dyads
            ),
        }
    per_query_skin = {}
    for skin in sorted({row["acquisition_query_skin"] for row in cases}):
        subset = [row for row in cases if row["acquisition_query_skin"] == skin]
        skin_dyads = [row for row in dyads if row["acquisition_query_skin"] == skin]
        per_query_skin[skin] = {
            "n": len(subset),
            "success": case_rate("success", subset),
            "evidence_acquired_before_first_patch": case_rate(
                "evidence_acquired_before_first_patch", subset
            ),
            "paired_preverifier_success": dyad_rate(
                "paired_preverifier_success", skin_dyads
            ),
        }

    successful = [row for row in cases if row["success"]]
    verified = [row for row in cases if row["verified_after_final_patch"]]
    total_turns = sum(row["turns"] for row in cases)
    all_steps = [step for row in rows for step in row["steps"]]
    invalid_steps = [step for step in all_steps if step["operator"] == "INVALID"]
    answer_cap_hits = [
        step for step in all_steps
        if answer_limit_contact(step, answer_max_tokens)
    ]
    unusable_cap_hits = [step for step in answer_cap_hits if step["operator"] == "INVALID"]
    return {
        "mode": mode,
        "n_cases": len(cases),
        "n_dyads": len(dyads),
        "success": case_rate("success"),
        "paired_terminal_success": dyad_rate("paired_terminal_success"),
        "first_patch_full_correct": case_rate("first_changed_patch_full_correct"),
        "evidence_acquired_before_first_patch": case_rate(
            "evidence_acquired_before_first_patch"
        ),
        "paired_preverifier_success": dyad_rate("paired_preverifier_success"),
        "paired_first_patch_full_correct": dyad_rate(
            "paired_first_patch_full_correct"
        ),
        "paired_evidence_acquired": dyad_rate("paired_evidence_acquired"),
        "paired_cross_branch_failure": dyad_rate("paired_cross_branch_failure"),
        "unnecessary_evidence_before_first_patch": case_rate(
            "unnecessary_evidence_before_first_patch"
        ),
        "any_search_before_first_patch": case_rate("any_search_before_first_patch"),
        "mean_non_source_inspects_before_first_patch": statistics.mean(
            row["non_source_inspects_before_first_patch"] for row in cases
        ),
        "submit_rate": case_rate("submitted"),
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
        "raw_answer_cap_hit_rate_per_turn": len(answer_cap_hits) / max(total_turns, 1),
        "unusable_answer_cap_hit_rate_per_turn": len(unusable_cap_hits) / max(total_turns, 1),
        "invalid_steps_at_answer_cap_fraction": (
            sum(step in answer_cap_hits for step in invalid_steps)
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
        "mean_logical_model_tokens": statistics.mean(
            row["logical_model_tokens"] for row in cases
        ),
        "max_logical_model_tokens": max(
            row["logical_model_tokens"] for row in cases
        ),
        "mean_turns": statistics.mean(row["turns"] for row in cases),
        "per_scenario": per_scenario,
        "per_family": per_family,
        "per_channel": per_channel,
        "per_query_skin": per_query_skin,
        "cases": cases,
        "dyads": dyads,
    }


def _assert_pair_hygiene(tasks: list[repo_tasks.RepoTask], contract: str) -> None:
    by_pair: dict[str, list[repo_tasks.RepoTask]] = defaultdict(list)
    for task in tasks:
        by_pair[task.pair_id].append(task)
    for pair_id, members in by_pair.items():
        if len(members) != 2 or {task.branch for task in members} != {0, 1}:
            raise AssertionError(f"incomplete pair: {pair_id}")
        a, b = sorted(members, key=lambda task: task.branch)
        if a.evidence_path != b.evidence_path or a.evidence_channel != b.evidence_channel:
            raise AssertionError(f"evidence topology differs inside pair: {pair_id}")
        if contract == "inferred":
            if repo_tasks.pair_static_digest(a) != repo_tasks.pair_static_digest(b):
                raise AssertionError(f"non-evidence bytes differ inside pair: {pair_id}")
            differing = {
                path for path in a.files if a.files[path] != b.files[path]
            }
            if differing != {a.evidence_path}:
                raise AssertionError((pair_id, differing, a.evidence_path))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXP / "configs" / "default.yaml")
    parser.add_argument("--design-lock", type=Path, required=True)
    parser.add_argument(
        "--arm",
        choices=[
            "start", "incumbent", "candidate",
            "explicit_redundant", "shuffled_binding",
        ],
        required=True,
    )
    parser.add_argument("--model", required=True, help="merged checkpoint")
    parser.add_argument("--expected-weight-sha256", required=True)
    parser.add_argument("--block", required=True)
    parser.add_argument("--contract", choices=["inferred", "explicit"], default="inferred")
    parser.add_argument(
        "--scenario-set",
        choices=["acquisition", "injected", "random", "normal", "recovery"],
        required=True,
    )
    parser.add_argument(
        "--mode", choices=["deep", "sample_more", "sample_pool"], default="deep"
    )
    parser.add_argument("--scaffold", action="store_true")
    parser.add_argument("--tasks-per-family", type=int, default=None)
    parser.add_argument("--answer-max-tokens", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    try:
        harness.validate_model_execution_lock(
            EXP, args.design_lock, "scripts/eval_repo_agent.py"
        )
    except ValueError as exc:
        raise SystemExit(f"model execution is not design-locked: {exc}") from exc
    try:
        harness.validate_canonical_config_path(EXP, args.config)
    except ValueError as exc:
        raise SystemExit(f"evaluation config is not frozen: {exc}") from exc

    cfg = yaml.safe_load(args.config.read_text())
    ecfg = cfg["evaluation"]
    answer_max_tokens = (
        int(args.answer_max_tokens)
        if args.answer_max_tokens is not None
        else int(ecfg["answer_max_tokens"])
    )
    if answer_max_tokens < 1:
        raise SystemExit("answer-max-tokens must be positive")
    if answer_max_tokens not in cfg["evaluation"]["interface_answer_rungs"]:
        raise SystemExit("answer-max-tokens is not a frozen interface rung")
    if args.block not in ecfg["blocks"]:
        raise SystemExit(f"unknown block: {args.block}")
    block = ecfg["blocks"][args.block]
    families = tuple(cfg["families"][block["families"]])
    registered_tasks_per_family = int(block["tasks_per_family"])
    if (
        args.tasks_per_family is not None
        and args.tasks_per_family != registered_tasks_per_family
    ):
        raise SystemExit("tasks-per-family differs from the frozen block")
    n_tasks = registered_tasks_per_family
    if n_tasks < 2 or n_tasks % 2:
        raise SystemExit("counterfactual tasks-per-family must be a positive even number")
    explicit = args.contract == "explicit"
    tasks = repo_tasks.make_pairs(
        families,
        n_tasks // 2,
        int(block["seed"]),
        args.block,
        explicit_contract=explicit,
    )
    _assert_pair_hygiene(tasks, args.contract)
    content_digests = [repo_tasks.content_digest(task) for task in tasks]
    if len(set(content_digests)) != len(content_digests):
        raise SystemExit(f"{args.block} contains duplicate repository content")
    for prior_name in block.get("disjoint_from", []):
        prior = ecfg["blocks"][prior_name]
        prior_tasks = repo_tasks.make_pairs(
            tuple(cfg["families"][prior["families"]]),
            int(prior["tasks_per_family"]) // 2,
            int(prior["seed"]),
            prior_name,
            explicit_contract=explicit,
        )
        if set(content_digests) & {
            repo_tasks.content_digest(task) for task in prior_tasks
        }:
            raise SystemExit(f"{args.block} overlaps {prior_name}")

    scenarios = {
        "acquisition": ("ambiguous_source",),
        "injected": ("evidence_injected",),
        "random": ("nondiscriminating_search_injected",),
        "normal": ("normal",),
        "recovery": ("rejected_patch", "failed_test"),
    }[args.scenario_set]
    budget_key = (
        "controlled"
        if args.scenario_set in ("acquisition", "injected", "random")
        else args.scenario_set
    )
    budget = ecfg[budget_key]
    per_call = int(ecfg["think_budget"]) + answer_max_tokens
    if args.mode == "deep":
        trajectories = 1
        max_turns = int(budget["deep_turns"])
        greedy = True
        run_seed = int(ecfg["deep_run_seed"])
    elif args.mode == "sample_more":
        trajectories = int(budget["sample_more_trajectories"])
        max_turns = int(budget["sample_more_turns_each"])
        greedy = False
        run_seed = int(ecfg["sample_run_seed"])
    else:
        trajectories = int(ecfg["sample_more_pool_trajectories"])
        max_turns = int(budget["sample_more_turns_each"])
        greedy = False
        run_seed = int(ecfg["sample_run_seed"])
    reserved = trajectories * max_turns * per_call
    expected = int(budget["deep_turns"]) * per_call
    if args.mode != "sample_pool" and reserved != expected:
        raise SystemExit(f"compute mismatch: mode reserves {reserved}, deep reserves {expected}")
    if args.scaffold and (args.scenario_set != "recovery" or args.mode != "deep"):
        raise SystemExit("the registered external scaffold applies only to deep recovery")

    model_path = resolve(args.model).resolve()
    checkpoint_role = {
        "start": "start",
        "incumbent": "anchor",
        "candidate": "evidence_binding",
        "explicit_redundant": "explicit_redundant",
        "shuffled_binding": "shuffled_binding",
    }[args.arm]
    try:
        checkpoint = harness.validate_registered_checkpoint(
            EXP, model_path, cfg, args.design_lock, checkpoint_role
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"evaluation checkpoint is not registered: {exc}") from exc
    observed_weight_sha256 = checkpoint["model_weight_sha256"]
    if observed_weight_sha256 != args.expected_weight_sha256:
        raise SystemExit("model weight hash differs from the registered invocation")
    tokenizer_provenance = {
        key: checkpoint[key] for key in harness.TOKENIZER_PROVENANCE_KEYS
    }
    runner = harness.make_runner(cfg["engine"], model_override=str(model_path))
    sampling = SamplingConfig(
        thinking="budget",
        thinking_budget=int(ecfg["think_budget"]),
        n=1,
        answer_max_tokens=answer_max_tokens,
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
    rows, runner_summaries = repo_agent.run_episodes(
        runner, specs, sampling, max_turns=max_turns, scaffold=args.scaffold
    )
    annotate_cross_branch(rows, tasks)
    aggregate_result = aggregate(rows, args.mode, answer_max_tokens)
    payload = {
        "schema_version": 1,
        "arm": args.arm,
        "model": str(model_path.resolve()),
        "model_weight_sha256": observed_weight_sha256,
        "model_config_sha256": checkpoint["model_config_sha256"],
        "model_generation_config_sha256": checkpoint[
            "generation_config_sha256"
        ],
        "merge_receipt_sha256": checkpoint["merge_receipt_sha256"],
        **tokenizer_provenance,
        "block": args.block,
        "config_sha256": sha256_file(args.config),
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
        "repo_agent_sha256": sha256_file(EXP / "src" / "repo_agent.py"),
        "task_generator_sha256": sha256_file(EXP / "src" / "repo_tasks.py"),
        "contract": args.contract,
        "scenario_set": args.scenario_set,
        "mode": args.mode,
        "scaffold": args.scaffold,
        "history_policy": "canonical_first_valid_tool_call",
        "think_budget": int(ecfg["think_budget"]),
        "answer_max_tokens": answer_max_tokens,
        "tasks_per_family": n_tasks,
        "reserved_sampled_tokens_per_case": reserved,
        "deep_reserved_sampled_tokens_per_case": expected,
        "sample_pool_capacity_multiplier": reserved / expected,
        "task_manifest_sha256": repo_tasks.manifest_digest(tasks),
        "task_content_manifest_sha256": hashlib.sha256(
            json.dumps(sorted(content_digests), separators=(",", ":")).encode()
        ).hexdigest(),
        "pair_static_manifest_sha256": hashlib.sha256(
            json.dumps(
                sorted(repo_tasks.pair_static_digest(task) for task in tasks),
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "composed_mapping_manifest": [
            {
                "task_id": task.task_id,
                "pair_id": task.pair_id,
                "branch": task.branch,
                "family": task.family,
                "evidence_channel": task.evidence_channel,
                "evidence_path": task.evidence_path,
                "evidence_path_regime": task.evidence_path_regime,
                "acquisition_query_skin": task.acquisition_query_skin,
                "evidence_sha256": hashlib.sha256(
                    task.files[task.evidence_path].encode()
                ).hexdigest(),
                "acquisition_query_sha256": hashlib.sha256(
                    task.acquisition_query.encode()
                ).hexdigest(),
                "oracle_first_patch_sha256": hashlib.sha256(
                    json.dumps(
                        task.oracle_patches[0].__dict__, sort_keys=True,
                        separators=(",", ":")
                    ).encode()
                ).hexdigest(),
            }
            for task in tasks
        ],
        "runner_summaries": runner_summaries,
        "aggregate": aggregate_result,
        "trajectories": rows,
    }
    bank.assert_firewall_clean(payload, tasks)
    if args.output:
        output = args.output
    else:
        root = resolve(cfg["artifacts"]["root"])
        suffix = "_scaffold" if args.scaffold else ""
        output = root / "eval" / (
            f"{args.block}_{args.contract}_{args.scenario_set}_{args.arm}_{args.mode}{suffix}.json"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    slim = {
        key: value for key, value in aggregate_result.items()
        if key not in ("cases", "dyads", "per_family", "per_channel")
    }
    print(json.dumps({
        "output": str(output),
        **slim,
        "per_family": aggregate_result["per_family"],
        "per_channel": aggregate_result["per_channel"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
