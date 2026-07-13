#!/usr/bin/env python3
"""Gate legacy broad and transaction-loop retention before Menagerie."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml

from downstream_common import (
    EXP,
    HISTORY_POLICY,
    _validate_runner_summaries,
    fail,
    finite_rate,
    is_sha256,
    read_json,
    sha256_file,
)

import repo_tasks  # noqa: E402
import retention_tasks_legacy  # noqa: E402
import harness  # noqa: E402


def _raw_step_hits_cap(step: dict, answer_max_tokens: int) -> bool:
    return bool(
        int(step.get("n_answer_tokens", 0)) >= answer_max_tokens
        or step.get("finish_reason") == "length"
    )


def _quality_step_hits_cap(step: dict, answer_max_tokens: int) -> bool:
    return bool(
        _raw_step_hits_cap(step, answer_max_tokens)
        or (
            step.get("forced_close")
            and step.get("stage2_finish_reason") == "length"
        )
    )


def _recompute_turn_rates(payload: dict) -> dict:
    trajectories = payload.get("trajectories")
    if not isinstance(trajectories, list) or not trajectories:
        fail("legacy retention receipt has no trajectories")
    steps = []
    total_turns = 0
    invalid_actions = 0
    for row in trajectories:
        row_steps = row.get("steps")
        if not isinstance(row_steps, list):
            fail("legacy retention trajectory has no steps")
        steps.extend(row_steps)
        total_turns += int(row.get("turns", -1))
        invalid_actions += int(row.get("invalid_actions", -1))
    if total_turns < 1 or invalid_actions < 0:
        fail("legacy retention turn accounting is malformed")
    answer_max_tokens = int(payload["answer_max_tokens"])
    raw_cap_hits = [
        step for step in steps if _raw_step_hits_cap(step, answer_max_tokens)
    ]
    quality_cap_hits = [
        step for step in steps if _quality_step_hits_cap(step, answer_max_tokens)
    ]
    unusable = [
        step for step in quality_cap_hits if step.get("operator") == "INVALID"
    ]
    return {
        "invalid_action_rate_per_turn": invalid_actions / total_turns,
        "raw_answer_cap_hit_rate_per_turn": len(raw_cap_hits) / total_turns,
        "unusable_answer_cap_hit_rate_per_turn": len(unusable) / total_turns,
    }


def _adapt_registered_task(task: retention_tasks_legacy.RepoTask) -> repo_tasks.RepoTask:
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


def _registered_retention_tasks(cfg: dict, block: str) -> list[repo_tasks.RepoTask]:
    block_cfg = cfg["evaluation"]["blocks"][block]
    families = tuple(cfg["families"][block_cfg["families"]])
    legacy = retention_tasks_legacy.make_tasks(
        families,
        int(block_cfg["tasks_per_family"]),
        int(block_cfg["seed"]),
        block,
    )
    return [_adapt_registered_task(task) for task in legacy]


def validate_retention_receipt(
    path: Path,
    cfg: dict,
    *,
    block: str,
    scenario_set: str,
    arm: str,
    expected_weight_sha256: str | None = None,
) -> dict:
    payload = read_json(path)
    observed = (
        payload.get("block"),
        payload.get("scenario_set"),
        payload.get("mode"),
        payload.get("arm"),
    )
    expected = (block, scenario_set, "deep", arm)
    if observed != expected:
        fail(f"wrong legacy retention receipt {path}: {observed} != {expected}")
    if payload.get("schema_version") != 1:
        fail(f"unsupported legacy retention schema: {path}")
    registered_block = cfg["evaluation"]["blocks"].get(block)
    if not registered_block or registered_block.get("legacy_retention") is not True:
        fail(f"unregistered legacy retention block: {block}")
    weight_hash = payload.get("model_weight_sha256")
    if not is_sha256(weight_hash):
        fail(f"invalid legacy retention model hash: {path}")
    if expected_weight_sha256 is not None and weight_hash != expected_weight_sha256:
        fail(f"wrong legacy retention model weights: {path}")
    if payload.get("history_policy") != HISTORY_POLICY:
        fail(f"legacy retention history policy drifted: {path}")
    if payload.get("think_budget") != int(cfg["evaluation"]["think_budget"]):
        fail(f"legacy retention thinking allowance drifted: {path}")
    answer_max_tokens = payload.get("answer_max_tokens")
    if answer_max_tokens not in cfg["evaluation"]["interface_answer_rungs"]:
        fail(f"legacy retention answer allowance drifted: {path}")
    if not is_sha256(payload.get("task_manifest_sha256")):
        fail(f"legacy retention task manifest is missing: {path}")
    config_path_value = cfg.get("__analysis_config_path__")
    if not isinstance(config_path_value, str):
        fail("analyzer config path was not registered before retention validation")
    expected_provenance = {
        "config_sha256": sha256_file(Path(config_path_value)),
        "evaluator_sha256": sha256_file(EXP / "scripts" / "eval_retention.py"),
        "repo_agent_sha256": sha256_file(EXP / "src" / "repo_agent.py"),
        "task_generator_sha256": sha256_file(EXP / "src" / "retention_tasks_legacy.py"),
    }
    for key, expected_hash in expected_provenance.items():
        if payload.get(key) != expected_hash:
            fail(f"legacy retention provenance drift at {key}: {path}")
    try:
        model_path = Path(payload["model"]).resolve()
    except (KeyError, TypeError) as exc:
        fail(f"legacy retention receipt has no model path: {path}: {exc}")
    model_files = {
        "model_weight_sha256": model_path / "model.safetensors",
        "model_config_sha256": model_path / "config.json",
        "model_generation_config_sha256": (
            model_path / "generation_config.json"
        ),
        "merge_receipt_sha256": model_path / "merge_receipt.json",
    }
    for key, model_file in model_files.items():
        if not model_file.is_file() or payload.get(key) != sha256_file(model_file):
            fail(f"legacy retention model provenance drift at {key}: {path}")
    try:
        tokenizer = harness.validate_registered_tokenizer_provenance(
            model_path, payload
        )
        merge_receipt = read_json(model_path / "merge_receipt.json")
        registered = harness.validate_registered_tokenizer_provenance(
            model_path, merge_receipt, allow_absent=True
        )
    except (OSError, ValueError) as exc:
        fail(f"legacy retention tokenizer provenance drift: {path}: {exc}")
    if tokenizer != registered:
        fail(f"legacy retention/merge tokenizer provenance mismatch: {path}")
    if (
        tokenizer["tokenizer_manifest_sha256"]
        != cfg["model"]["start_tokenizer_manifest_sha256"]
        or tokenizer["tokenizer_compatibility_sha256"]
        != cfg["model"]["tokenizer_compatibility_sha256"]
    ):
        fail(f"legacy retention tokenizer is not the frozen start identity: {path}")
    registered_tasks = _registered_retention_tasks(cfg, block)
    registered_task_count = int(registered_block["tasks_per_family"])
    if payload.get("tasks_per_family") != registered_task_count:
        fail(f"legacy retention receipt used a task-count override: {path}")
    if payload.get("task_manifest_sha256") != repo_tasks.manifest_digest(registered_tasks):
        fail(f"legacy retention task manifest differs from the registry: {path}")
    budget = cfg["evaluation"][scenario_set]
    per_call = int(cfg["evaluation"]["think_budget"]) + int(answer_max_tokens)
    expected_reserved = int(budget["deep_turns"]) * per_call
    if (
        payload.get("reserved_sampled_tokens_per_case") != expected_reserved
        or payload.get("deep_reserved_sampled_tokens_per_case") != expected_reserved
    ):
        fail(f"legacy retention compute reservation drifted: {path}")
    _validate_runner_summaries(
        payload,
        cfg,
        path,
        mode="deep",
        answer_max_tokens=int(answer_max_tokens),
    )
    aggregate = payload.get("aggregate")
    if not isinstance(aggregate, dict):
        fail(f"legacy retention aggregate is missing: {path}")
    cases = aggregate.get("cases")
    if not isinstance(cases, list) or not cases or aggregate.get("n_cases") != len(cases):
        fail(f"legacy retention case rows/count are malformed: {path}")
    case_ids = [row.get("case_id") for row in cases]
    if len(set(case_ids)) != len(case_ids) or any(not value for value in case_ids):
        fail(f"legacy retention case IDs are malformed: {path}")
    expected_scenarios = {"normal"} if scenario_set == "normal" else {
        "rejected_patch",
        "failed_test",
    }
    if {row.get("scenario") for row in cases} != expected_scenarios:
        fail(f"legacy retention scenarios are incomplete: {path}")
    expected_case_count = len(registered_tasks) * len(expected_scenarios)
    if len(cases) != expected_case_count:
        fail(f"legacy retention case count differs from the registry: {path}")
    expected_trajectory_ids = {
        (task.task_id, scenario, 0)
        for task in registered_tasks
        for scenario in expected_scenarios
    }
    try:
        observed_trajectory_ids = {
            (row["task_id"], row["scenario"], int(row["trajectory"]))
            for row in payload["trajectories"]
        }
    except (KeyError, TypeError, ValueError) as exc:
        fail(f"legacy retention trajectory identity is malformed: {path}: {exc}")
    if (
        observed_trajectory_ids != expected_trajectory_ids
        or len(payload["trajectories"]) != len(expected_trajectory_ids)
    ):
        fail(f"legacy retention trajectories differ from the registered grid: {path}")
    summary_requests = sum(
        int(row["counts"].get("requests", -1)) for row in payload["runner_summaries"]
    )
    trajectory_turns = sum(int(row.get("turns", -1)) for row in payload["trajectories"])
    if summary_requests != trajectory_turns:
        fail(f"legacy runner request count differs from trajectory turns: {path}")
    expected_families = set(cfg["families"][registered_block["families"]])
    if (
        {row.get("family") for row in cases} != expected_families
        or set(aggregate.get("per_family") or {}) != expected_families
    ):
        fail(f"legacy retention families drifted: {path}")
    observed_success = sum(bool(row.get("success")) for row in cases) / len(cases)
    if not math.isclose(
        float(aggregate.get("success")), observed_success, rel_tol=0.0, abs_tol=1e-12
    ):
        fail(f"legacy retention aggregate success disagrees with cases: {path}")
    for metric in ("verified", "commit"):
        observed_metric = sum(bool(row.get(metric)) for row in cases) / len(cases)
        if not math.isclose(
            float(aggregate.get(metric)),
            observed_metric,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            fail(f"legacy retention aggregate {metric} disagrees with cases: {path}")
    for family in expected_families:
        members = [row for row in cases if row["family"] == family]
        observed_family = sum(bool(row.get("success")) for row in members) / len(members)
        if not math.isclose(
            float(aggregate["per_family"][family]["success"]),
            observed_family,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            fail(f"legacy retention family aggregate disagrees with cases: {path}")
    turn_rates = _recompute_turn_rates(payload)
    if not math.isclose(
        float(aggregate.get("invalid_action_rate_per_turn")),
        turn_rates["invalid_action_rate_per_turn"],
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        fail(f"legacy retention invalid-action rate disagrees with trajectories: {path}")
    # Older evaluator receipts expose only raw contact. Recompute the stricter
    # unusable-cap metric here and carry it forward explicitly.
    raw_aggregate = aggregate.get("answer_cap_hit_rate_per_turn")
    if not math.isclose(
        float(raw_aggregate),
        turn_rates["raw_answer_cap_hit_rate_per_turn"],
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        fail(f"legacy retention raw cap rate disagrees with trajectories: {path}")
    payload = {**payload, "analyzer_turn_rates": turn_rates}
    return payload


def _scenario_rate(payload: dict, scenario: str, field: str) -> float:
    rows = [
        row for row in payload["aggregate"]["cases"] if row.get("scenario") == scenario
    ]
    if not rows:
        fail(f"legacy retention receipt has no {scenario} cases")
    return sum(bool(row.get(field)) for row in rows) / len(rows)


def _family_success(payload: dict, family: str) -> float:
    return finite_rate(
        payload["aggregate"]["per_family"][family]["success"],
        f"per_family.{family}.success",
    )


def _conditional_case_rate(
    payload: dict, *, numerator: str, denominator: str
) -> float:
    denominator_rows = [
        row for row in payload["aggregate"]["cases"] if bool(row.get(denominator))
    ]
    if not denominator_rows:
        return 0.0
    return sum(bool(row.get(numerator)) for row in denominator_rows) / len(
        denominator_rows
    )


def analyze_substrate(
    cfg: dict,
    *,
    name: str,
    candidate_normal: dict,
    start_normal: dict,
    candidate_recovery: dict,
    start_recovery: dict,
) -> dict:
    gates = cfg["retention_gates"]
    payloads = (candidate_normal, start_normal, candidate_recovery, start_recovery)
    if len({row["task_manifest_sha256"] for row in payloads}) != 1:
        fail(f"{name} retention task manifests differ")
    if len({int(row["answer_max_tokens"]) for row in payloads}) != 1:
        fail(f"{name} retention answer allowances differ")
    answer_max_tokens = int(candidate_normal["answer_max_tokens"])
    if candidate_normal["model_weight_sha256"] != candidate_recovery["model_weight_sha256"]:
        fail(f"{name} candidate retention receipts use different model weights")
    normal_delta = float(candidate_normal["aggregate"]["success"]) - float(
        start_normal["aggregate"]["success"]
    )
    recovery_delta = float(candidate_recovery["aggregate"]["success"]) - float(
        start_recovery["aggregate"]["success"]
    )
    rejected = _scenario_rate(
        candidate_recovery, "rejected_patch", "rejected_transition"
    )
    failed = _scenario_rate(candidate_recovery, "failed_test", "failed_transition")
    verified_given_success = _conditional_case_rate(
        candidate_normal, numerator="verified", denominator="success"
    )
    commit_given_verified = _conditional_case_rate(
        candidate_normal, numerator="commit", denominator="verified"
    )
    invalid_deltas = {
        "normal": candidate_normal["analyzer_turn_rates"]["invalid_action_rate_per_turn"]
        - start_normal["analyzer_turn_rates"]["invalid_action_rate_per_turn"],
        "recovery": candidate_recovery["analyzer_turn_rates"]["invalid_action_rate_per_turn"]
        - start_recovery["analyzer_turn_rates"]["invalid_action_rate_per_turn"],
    }
    unusable_cap_deltas = {
        "normal": candidate_normal["analyzer_turn_rates"][
            "unusable_answer_cap_hit_rate_per_turn"
        ] - start_normal["analyzer_turn_rates"]["unusable_answer_cap_hit_rate_per_turn"],
        "recovery": candidate_recovery["analyzer_turn_rates"][
            "unusable_answer_cap_hit_rate_per_turn"
        ] - start_recovery["analyzer_turn_rates"]["unusable_answer_cap_hit_rate_per_turn"],
    }
    families = sorted(candidate_normal["aggregate"]["per_family"])
    family_deltas = {
        scenario: {
            family: _family_success(candidate, family) - _family_success(start, family)
            for family in families
        }
        for scenario, candidate, start in (
            ("normal", candidate_normal, start_normal),
            ("recovery", candidate_recovery, start_recovery),
        )
    }
    worst_family_delta = min(
        delta for rows in family_deltas.values() for delta in rows.values()
    )
    checks = {
        "normal_terminal": normal_delta
        >= float(gates["normal_terminal_delta_vs_start_min"]),
        "recovery_terminal": recovery_delta
        >= float(gates["recovery_terminal_delta_vs_start_min"]),
        "rejected_transition": rejected
        >= float(gates["rejected_transition_absolute_min"]),
        "failed_transition": failed >= float(gates["failed_transition_absolute_min"]),
        "verified": verified_given_success >= float(gates["verified_absolute_min"]),
        "commit": commit_given_verified >= float(gates["commit_absolute_min"]),
        "invalid_action": max(invalid_deltas.values())
        <= float(gates["invalid_action_delta_vs_start_max"]),
        "unusable_cap": max(unusable_cap_deltas.values())
        <= float(gates["unusable_cap_delta_vs_start_max"]),
        "maximum_single_family_terminal_regression": worst_family_delta
        >= float(gates["maximum_single_family_terminal_regression"]),
    }
    return {
        "task_manifest_sha256": candidate_normal["task_manifest_sha256"],
        "answer_max_tokens": answer_max_tokens,
        "candidate_model_weight_sha256": candidate_normal["model_weight_sha256"],
        "normal_terminal": {
            "candidate": candidate_normal["aggregate"]["success"],
            "start": start_normal["aggregate"]["success"],
            "delta": normal_delta,
        },
        "recovery_terminal": {
            "candidate": candidate_recovery["aggregate"]["success"],
            "start": start_recovery["aggregate"]["success"],
            "delta": recovery_delta,
        },
        "transition_metrics": {"rejected": rejected, "failed": failed},
        "normal_verified_given_success": verified_given_success,
        "normal_commit_given_verified": commit_given_verified,
        "invalid_action_deltas_vs_start": invalid_deltas,
        "unusable_cap_deltas_vs_start": unusable_cap_deltas,
        "family_terminal_deltas_vs_start": family_deltas,
        "worst_family_terminal_delta": worst_family_delta,
        "checks": checks,
        "passed": all(checks.values()),
    }


def analyze(cfg: dict, *, broad: dict[str, dict], transaction: dict[str, dict]) -> dict:
    broad_result = analyze_substrate(cfg, name="broad", **broad)
    transaction_result = analyze_substrate(cfg, name="transaction", **transaction)
    candidate_hashes = {
        broad_result["candidate_model_weight_sha256"],
        transaction_result["candidate_model_weight_sha256"],
    }
    if len(candidate_hashes) != 1:
        fail("broad and transaction retention receipts use different candidate weights")
    answer_allowances = {
        broad_result["answer_max_tokens"],
        transaction_result["answer_max_tokens"],
    }
    if len(answer_allowances) != 1:
        fail("broad and transaction retention use different answer allowances")
    checks = {
        "broad_retention": broad_result["passed"],
        "transaction_retention": transaction_result["passed"],
    }
    passed = all(checks.values())
    return {
        "schema_version": 1,
        "stage": "legacy_retention",
        "answer_max_tokens": answer_allowances.pop(),
        "candidate_model_weight_sha256": candidate_hashes.pop(),
        "substrates": {"broad": broad_result, "transaction": transaction_result},
        "checks": checks,
        "gate": {
            "passed": passed,
            "verdict": "RETENTION_PASS" if passed else "RETENTION_FAIL",
        },
        "menagerie_authorized": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXP / "configs" / "default.yaml")
    for substrate in ("broad", "transaction"):
        for arm in ("candidate", "start"):
            for scenario in ("normal", "recovery"):
                parser.add_argument(
                    f"--{substrate}-{arm}-{scenario}", type=Path, required=True
                )
    parser.add_argument("--expected-candidate-weight-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cfg["__analysis_config_path__"] = str(args.config.resolve())
    blocks = {
        "broad": "old_broad_retention",
        "transaction": "old_transaction_retention",
    }
    inputs: dict[str, dict[str, dict]] = {}
    candidate_hash: str | None = args.expected_candidate_weight_sha256
    receipt_metadata = {}
    for substrate, block in blocks.items():
        inputs[substrate] = {}
        for arm in ("candidate", "start"):
            for scenario in ("normal", "recovery"):
                key = f"{arm}_{scenario}"
                path = getattr(args, f"{substrate}_{arm}_{scenario}")
                expected_hash = (
                    cfg["model"]["start_weight_sha256"] if arm == "start" else candidate_hash
                )
                payload = validate_retention_receipt(
                    path,
                    cfg,
                    block=block,
                    scenario_set=scenario,
                    arm=arm,
                    expected_weight_sha256=expected_hash,
                )
                inputs[substrate][key] = payload
                receipt_metadata[f"{substrate}_{key}"] = {
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                }
    result = analyze(cfg, broad=inputs["broad"], transaction=inputs["transaction"])
    result["analyzer_sha256"] = sha256_file(Path(__file__).resolve())
    result["config_sha256"] = sha256_file(args.config)
    result["receipts"] = receipt_metadata
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["gate"]["passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
