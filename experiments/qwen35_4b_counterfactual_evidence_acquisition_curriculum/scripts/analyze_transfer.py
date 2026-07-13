#!/usr/bin/env python3
"""Gate one transfer block against paired model, sampling, and scaffold controls."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import yaml

from downstream_common import (
    EXP,
    assert_behavior_peers,
    fail,
    finite_rate,
    paired_differences,
    read_json,
    sha256_file,
    validate_behavior_receipt,
    validate_selected_answer_allowance,
)


BOOTSTRAP_ITERATIONS = 10_000


def paired_bootstrap(
    differences: list[float], *, seed: int, iterations: int = BOOTSTRAP_ITERATIONS
) -> dict:
    """Percentile bootstrap over counterfactual dyads, never over branches."""
    if not differences:
        fail("cannot bootstrap an empty dyad set")
    rng = random.Random(seed)
    n = len(differences)
    means = sorted(
        sum(differences[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(iterations)
    )
    lower_index = max(0, math.floor(0.025 * (iterations - 1)))
    upper_index = min(iterations - 1, math.ceil(0.975 * (iterations - 1)))
    return {
        "unit": "counterfactual_dyad",
        "n": n,
        "iterations": iterations,
        "seed": seed,
        "mean_delta": sum(differences) / n,
        "lower_95": means[lower_index],
        "upper_95": means[upper_index],
    }


def _scenario_metric(payload: dict, scenario: str) -> float:
    table = payload["aggregate"].get("per_scenario") or {}
    row = table.get(scenario)
    if not isinstance(row, dict):
        fail(f"missing recovery scenario aggregate: {scenario}")
    metric = (
        "valid_changed_patch_within_two"
        if scenario == "rejected_patch"
        else "changed_patch_within_two"
    )
    return finite_rate(row.get(metric), f"per_scenario.{scenario}.{metric}")


def _validate_match_summary(matcher: dict, label: str) -> float:
    if (
        matcher.get("schema_version") != 1
        or matcher.get("status") != "PASS"
        or matcher.get("primary_control") != "dual_overmatch"
        or matcher.get("full_pool_is_oracle_only") is not True
    ):
        fail(f"{label} is not a registered sample-prefix matcher receipt")
    summaries = matcher.get("summaries")
    if not isinstance(summaries, dict) or "dual_overmatch" not in summaries:
        fail(f"{label} has no dual-overmatch summary")
    summary = summaries["dual_overmatch"]
    members = summary.get("members")
    dyads = summary.get("dyads")
    if not isinstance(members, list) or not members or not isinstance(dyads, list) or not dyads:
        fail(f"{label} has no matched members/dyads")
    if summary.get("n_cases") != len(members) or summary.get("n_dyads") != len(dyads):
        fail(f"{label} matched counts do not agree with its rows")
    by_pair: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for member in members:
        by_pair[(member.get("pair_id"), member.get("scenario"))].append(member)
        if int(member.get("trajectories", 0)) < 1:
            fail(f"{label} dual overmatch contains an empty prefix")
        if int(member.get("sampled_margin", -1)) < 0 or int(member.get("logical_margin", -1)) < 0:
            fail(f"{label} dual overmatch does not overmatch both actual budgets")
    dyad_map = {(row.get("pair_id"), row.get("scenario")): row for row in dyads}
    if len(dyad_map) != len(dyads) or set(dyad_map) != set(by_pair):
        fail(f"{label} matched dyad identities are malformed")
    for key, rows in by_pair.items():
        if len(rows) != 2 or {row.get("branch") for row in rows} != {0, 1}:
            fail(f"{label} has an incomplete matched dyad: {key}")
        expected = all(bool(row.get("preverifier_member_success")) for row in rows)
        if bool(dyad_map[key].get("paired_preverifier_success")) != expected:
            fail(f"{label} matched dyad outcome does not agree with members: {key}")
    observed_rate = sum(
        bool(row.get("paired_preverifier_success")) for row in dyads
    ) / len(dyads)
    if not math.isclose(
        float(summary.get("paired_preverifier_success")),
        observed_rate,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        fail(f"{label} paired-preverifier rate does not agree with dyads")
    return observed_rate


def _recompute_dual_prefix(
    costs: list[dict], *, target_sampled: int, target_logical: int
) -> dict:
    sampled = 0
    logical = 0
    prefix = []
    for row in sorted(costs, key=lambda item: int(item["trajectory"])):
        sampled += int(row["sampled_tokens"])
        logical += int(row["logical_model_tokens"])
        prefix.append(row)
        if sampled >= target_sampled and logical >= target_logical:
            return {
                "trajectories": len(prefix),
                "sampled_tokens": sampled,
                "logical_model_tokens": logical,
                "sampled_margin": sampled - target_sampled,
                "logical_margin": logical - target_logical,
                "workspace_success": any(row["workspace_success"] for row in prefix),
                "preverifier_member_success": any(
                    row["preverifier_member_success"] for row in prefix
                ),
            }
    fail("sample pool cannot reach the candidate's actual dual token budget")


def validate_matcher_receipt(
    matcher_path: Path,
    cfg: dict,
    *,
    block: str,
    candidate_path: Path,
    candidate: dict,
    pool_arm: str,
    pool_weight_sha256: str,
) -> dict:
    matcher = read_json(matcher_path)
    rate = _validate_match_summary(matcher, str(matcher_path))
    if matcher.get("target_model_weight_sha256") != candidate["model_weight_sha256"]:
        fail(f"sample matcher target is not the candidate: {matcher_path}")
    if matcher.get("pool_model_weight_sha256") != pool_weight_sha256:
        fail(f"sample matcher pool has the wrong {pool_arm} weights: {matcher_path}")
    try:
        target_path = Path(matcher["target"])
        pool_path = Path(matcher["sample_pool"])
    except (KeyError, TypeError) as exc:
        fail(f"sample matcher omits raw receipt provenance: {matcher_path}: {exc}")
    if not target_path.is_file() or not pool_path.is_file():
        fail(f"sample matcher raw receipt is unavailable: {matcher_path}")
    target_sha = sha256_file(target_path)
    pool_sha = sha256_file(pool_path)
    if target_sha != sha256_file(candidate_path):
        fail(f"sample matcher target receipt differs from candidate input: {matcher_path}")
    if (
        matcher.get("target_receipt_sha256") != target_sha
        or matcher.get("sample_pool_receipt_sha256") != pool_sha
        or matcher.get("analyzer_sha256")
        != sha256_file(EXP / "scripts" / "analyze_sample_pool.py")
    ):
        fail(f"sample matcher provenance is stale: {matcher_path}")
    pool = validate_behavior_receipt(
        pool_path,
        cfg,
        block=block,
        contract="inferred",
        scenario_set="acquisition",
        mode="sample_pool",
        arm=pool_arm,
        expected_weight_sha256=pool_weight_sha256,
        scaffold=False,
    )
    for key in (
        "block",
        "contract",
        "scenario_set",
        "history_policy",
        "think_budget",
        "answer_max_tokens",
        "task_manifest_sha256",
        "task_content_manifest_sha256",
        "pair_static_manifest_sha256",
        "composed_mapping_manifest",
    ):
        if pool.get(key) != candidate.get(key):
            fail(f"sample matcher pool/candidate mismatch at {key}: {matcher_path}")
    expected_cases = {row["case_id"] for row in candidate["aggregate"]["cases"]}
    matcher_cases = matcher.get("cases")
    if not isinstance(matcher_cases, list) or {
        row.get("case_id") for row in matcher_cases
    } != expected_cases:
        fail(f"sample matcher cases differ from candidate cases: {matcher_path}")
    if any(row.get("match", {}).get("pool_exhausted") for row in matcher_cases):
        fail(f"sample matcher exhausted its preregistered pool: {matcher_path}")
    candidate_cases = {
        row["case_id"]: row for row in candidate["aggregate"]["cases"]
    }
    pool_cases = {row["case_id"]: row for row in pool["aggregate"]["cases"]}
    matcher_by_case = {row["case_id"]: row for row in matcher_cases}
    if set(pool_cases) != expected_cases or len(matcher_by_case) != len(matcher_cases):
        fail(f"sample matcher pool/case identities are malformed: {matcher_path}")
    recomputed_members = []
    for case_id in sorted(expected_cases):
        target_case = candidate_cases[case_id]
        expected_prefix = _recompute_dual_prefix(
            pool_cases[case_id]["trajectory_costs"],
            target_sampled=int(target_case["sampled_tokens"]),
            target_logical=int(target_case["logical_model_tokens"]),
        )
        matcher_case = matcher_by_case[case_id]
        if matcher_case.get("match", {}).get("dual_overmatch") != expected_prefix:
            fail(f"sample matcher dual prefix does not recompute: {matcher_path} {case_id}")
        expected_identity = {
            key: target_case[key]
            for key in ("case_id", "pair_id", "branch", "family", "scenario")
        }
        if {key: matcher_case.get(key) for key in expected_identity} != expected_identity:
            fail(f"sample matcher case metadata differs from target: {matcher_path} {case_id}")
        recomputed_members.append({**expected_identity, **expected_prefix})
    observed_members = matcher["summaries"]["dual_overmatch"]["members"]
    if sorted(observed_members, key=lambda row: row["case_id"]) != recomputed_members:
        fail(f"sample matcher member summary does not recompute: {matcher_path}")
    return {
        "path": str(matcher_path.resolve()),
        "sha256": sha256_file(matcher_path),
        "pool_receipt": str(pool_path.resolve()),
        "pool_receipt_sha256": sha256_file(pool_path),
        "pool_arm": pool_arm,
        "pool_model_weight_sha256": pool_weight_sha256,
        "dual_overmatch_paired_preverifier_success": rate,
    }


def analyze(
    cfg: dict,
    *,
    block: str,
    candidate: dict,
    start: dict,
    incumbent: dict,
    explicit_control: dict,
    shuffled_control: dict,
    nondiscriminating_search: dict,
    candidate_injected: dict,
    candidate_normal: dict,
    start_normal: dict,
    candidate_recovery: dict,
    start_recovery: dict,
    candidate_recovery_scaffold: dict,
    candidate_explicit: dict,
    sample_match_start: dict,
    sample_match_incumbent: dict,
) -> dict:
    gates = cfg["transfer_gates"]
    acquisition_peers = {
        "candidate": candidate,
        "start": start,
        "incumbent": incumbent,
        "explicit_control": explicit_control,
        "shuffled_control": shuffled_control,
    }
    assert_behavior_peers(acquisition_peers)
    assert_behavior_peers(
        {
            "candidate": candidate,
            "nondiscriminating_search": nondiscriminating_search,
            "injected": candidate_injected,
        },
        include_scenario=False,
    )
    assert_behavior_peers({"candidate_normal": candidate_normal, "start_normal": start_normal})
    assert_behavior_peers(
        {"candidate_recovery": candidate_recovery, "start_recovery": start_recovery}
    )
    assert_behavior_peers(
        {
            "candidate_recovery": candidate_recovery,
            "candidate_recovery_scaffold": candidate_recovery_scaffold,
        },
        include_scaffold=False,
    )
    assert_behavior_peers(
        {
            "candidate_normal": candidate_normal,
            "candidate_recovery": candidate_recovery,
        },
        include_scenario=False,
    )
    behavior_payloads = [
        *acquisition_peers.values(),
        nondiscriminating_search,
        candidate_injected,
        candidate_normal,
        start_normal,
        candidate_recovery,
        start_recovery,
        candidate_recovery_scaffold,
        candidate_explicit,
    ]
    answer_max_tokens = validate_selected_answer_allowance(behavior_payloads)
    candidate_hash = candidate["model_weight_sha256"]
    for name, payload in {
        "nondiscriminating_search": nondiscriminating_search,
        "candidate_injected": candidate_injected,
        "candidate_normal": candidate_normal,
        "candidate_recovery": candidate_recovery,
        "candidate_recovery_scaffold": candidate_recovery_scaffold,
        "candidate_explicit": candidate_explicit,
    }.items():
        if payload.get("model_weight_sha256") != candidate_hash:
            fail(f"{name} is not from the candidate checkpoint")

    aggregates = {name: row["aggregate"] for name, row in acquisition_peers.items()}
    candidate_paired = float(aggregates["candidate"]["paired_preverifier_success"])
    deltas = {
        name: candidate_paired - float(aggregates[name]["paired_preverifier_success"])
        for name in ("start", "incumbent", "explicit_control", "shuffled_control")
    }
    deltas["nondiscriminating_search"] = candidate_paired - float(
        nondiscriminating_search["aggregate"]["paired_preverifier_success"]
    )
    sample_rates = {
        "start": finite_rate(
            sample_match_start["dual_overmatch_paired_preverifier_success"],
            "sample_match_start.dual_overmatch_paired_preverifier_success",
        ),
        "incumbent": finite_rate(
            sample_match_incumbent["dual_overmatch_paired_preverifier_success"],
            "sample_match_incumbent.dual_overmatch_paired_preverifier_success",
        ),
    }
    stronger_sample_arm = max(sample_rates, key=lambda name: (sample_rates[name], name))
    stronger_sample_rate = sample_rates[stronger_sample_arm]
    deltas["stronger_sample_more"] = candidate_paired - stronger_sample_rate

    bootstrap = paired_bootstrap(
        paired_differences(candidate, start),
        seed=int(cfg["evaluation"]["blocks"][block]["seed"]),
    )
    if not math.isclose(
        bootstrap["mean_delta"], deltas["start"], rel_tol=0.0, abs_tol=1e-12
    ):
        fail("dyad-level paired delta disagrees with aggregate transfer delta")

    expected_families = set(cfg["families"]["acquisition_transfer"])
    if set(aggregates["candidate"]["per_family"]) != expected_families:
        fail("candidate transfer receipt does not contain exactly the registered families")
    family_deltas = {
        family: float(aggregates["candidate"]["per_family"][family]["paired_preverifier_success"])
        - float(aggregates["start"]["per_family"][family]["paired_preverifier_success"])
        for family in sorted(expected_families)
    }
    nonnegative_families = [name for name, delta in family_deltas.items() if delta >= 0.0]

    required_channels = set(gates["required_transfer_channels"])
    candidate_channels = aggregates["candidate"]["per_channel"]
    channel_rates = {
        name: float(row["paired_preverifier_success"])
        for name, row in candidate_channels.items()
    }
    required_skins = set(gates["required_transfer_query_skins"])
    candidate_skins = aggregates["candidate"]["per_query_skin"]
    query_skin_rates = {
        name: float(row["paired_preverifier_success"])
        for name, row in candidate_skins.items()
    }

    normal_delta = float(candidate_normal["aggregate"]["success"]) - float(
        start_normal["aggregate"]["success"]
    )
    rejected = _scenario_metric(candidate_recovery, "rejected_patch")
    failed = _scenario_metric(candidate_recovery, "failed_test")
    scaffold_rejected = _scenario_metric(candidate_recovery_scaffold, "rejected_patch")
    scaffold_failed = _scenario_metric(candidate_recovery_scaffold, "failed_test")
    invalid_deltas = {
        "acquisition": float(candidate["aggregate"]["invalid_action_rate_per_turn"])
        - float(start["aggregate"]["invalid_action_rate_per_turn"]),
        "normal": float(candidate_normal["aggregate"]["invalid_action_rate_per_turn"])
        - float(start_normal["aggregate"]["invalid_action_rate_per_turn"]),
        "recovery": float(candidate_recovery["aggregate"]["invalid_action_rate_per_turn"])
        - float(start_recovery["aggregate"]["invalid_action_rate_per_turn"]),
    }
    unusable_cap_deltas = {
        "acquisition": float(candidate["aggregate"]["unusable_answer_cap_hit_rate_per_turn"])
        - float(start["aggregate"]["unusable_answer_cap_hit_rate_per_turn"]),
        "normal": float(candidate_normal["aggregate"]["unusable_answer_cap_hit_rate_per_turn"])
        - float(start_normal["aggregate"]["unusable_answer_cap_hit_rate_per_turn"]),
        "recovery": float(candidate_recovery["aggregate"]["unusable_answer_cap_hit_rate_per_turn"])
        - float(start_recovery["aggregate"]["unusable_answer_cap_hit_rate_per_turn"]),
    }
    explicit_first_patch = float(candidate_explicit["aggregate"]["first_patch_full_correct"])
    explicit_unnecessary = float(
        candidate_explicit["aggregate"]["unnecessary_evidence_before_first_patch"]
    )
    injected_paired = float(
        candidate_injected["aggregate"]["paired_preverifier_success"]
    )
    checks = {
        "paired_delta_vs_start": deltas["start"]
        >= float(gates["paired_preverifier_delta_vs_start_min"]),
        "paired_delta_vs_incumbent": deltas["incumbent"]
        >= float(gates["paired_preverifier_delta_vs_incumbent_min"]),
        "paired_delta_vs_explicit_control": deltas["explicit_control"]
        >= float(gates["paired_preverifier_delta_vs_controls_min"]),
        "paired_delta_vs_shuffled_control": deltas["shuffled_control"]
        >= float(gates["paired_preverifier_delta_vs_controls_min"]),
        "paired_delta_vs_nondiscriminating_search": deltas["nondiscriminating_search"]
        >= float(gates["paired_preverifier_delta_vs_nondiscriminating_search_min"]),
        "paired_delta_vs_stronger_sample_more": deltas["stronger_sample_more"]
        >= float(gates["paired_preverifier_delta_vs_sample_more_min"]),
        "paired_bootstrap_lower_bound": bootstrap["lower_95"]
        >= float(gates["paired_bootstrap_lower_bound_min"]),
        "injected_ceiling": injected_paired
        >= float(gates["injected_paired_preverifier_success_min"]),
        "normal_terminal_delta": normal_delta
        >= float(gates["normal_terminal_delta_vs_start_min"]),
        "minimum_nonnegative_families": len(nonnegative_families)
        >= int(gates["minimum_nonnegative_families"]),
        "maximum_single_family_regression": min(family_deltas.values())
        >= float(gates["maximum_single_family_regression"]),
        "rejected_transition": rejected
        >= float(gates["rejected_transition_absolute_min"]),
        "failed_transition": failed >= float(gates["failed_transition_absolute_min"]),
        "verified": float(candidate_normal["aggregate"]["verified_given_success"])
        >= float(gates["verified_absolute_min"]),
        "commit": float(candidate_normal["aggregate"]["commit_given_verified"])
        >= float(gates["commit_absolute_min"]),
        "invalid_action": max(invalid_deltas.values())
        <= float(gates["invalid_action_delta_vs_start_max"]),
        "unusable_cap": max(unusable_cap_deltas.values())
        <= float(gates["unusable_cap_delta_vs_start_max"]),
        "explicit_first_patch_retention": explicit_first_patch
        >= float(gates["explicit_first_patch_retention_min"]),
        "explicit_unnecessary_evidence": explicit_unnecessary
        <= float(gates["explicit_unnecessary_evidence_max"]),
        "all_transfer_channels": set(channel_rates) == required_channels
        and all(
            channel_rates[name] >= float(gates["per_channel_paired_preverifier_min"])
            for name in required_channels
        ),
        "all_transfer_query_skins": set(query_skin_rates) == required_skins
        and all(
            query_skin_rates[name]
            >= float(gates["per_query_skin_paired_preverifier_min"])
            for name in required_skins
        ),
        "evidence_search_efficiency": float(
            candidate["aggregate"]["mean_non_source_inspects_before_first_patch"]
        )
        <= float(gates["mean_non_source_inspects_before_first_patch_max"]),
    }
    gate_passed = all(checks.values())
    return {
        "schema_version": 1,
        "stage": "transfer",
        "block": block,
        "answer_max_tokens": answer_max_tokens,
        "task_manifest_sha256": candidate["task_manifest_sha256"],
        "candidate_model_weight_sha256": candidate_hash,
        "paired_preverifier_success": {
            **{name: aggregates[name]["paired_preverifier_success"] for name in aggregates},
            "nondiscriminating_search": nondiscriminating_search["aggregate"][
                "paired_preverifier_success"
            ],
            "candidate_injected": injected_paired,
            "sample_match_start": sample_rates["start"],
            "sample_match_incumbent": sample_rates["incumbent"],
        },
        "candidate_deltas": deltas,
        "sample_more_comparator": {
            "selected_stronger_arm": stronger_sample_arm,
            "rates": sample_rates,
            "full_pool_used_for_gate": False,
        },
        "paired_bootstrap_vs_start": bootstrap,
        "family_deltas_vs_start": family_deltas,
        "nonnegative_families": nonnegative_families,
        "channel_rates": channel_rates,
        "query_skin_rates": query_skin_rates,
        "normal_terminal": {
            "candidate": candidate_normal["aggregate"]["success"],
            "start": start_normal["aggregate"]["success"],
            "delta": normal_delta,
        },
        "recovery": {
            "candidate_terminal": candidate_recovery["aggregate"]["success"],
            "start_terminal": start_recovery["aggregate"]["success"],
            "scaffold_terminal": candidate_recovery_scaffold["aggregate"]["success"],
            "rejected_transition": rejected,
            "failed_transition": failed,
            "scaffold_rejected_transition": scaffold_rejected,
            "scaffold_failed_transition": scaffold_failed,
            "candidate_delta_vs_scaffold_terminal": float(
                candidate_recovery["aggregate"]["success"]
            ) - float(candidate_recovery_scaffold["aggregate"]["success"]),
        },
        "evidence_tool_cost": {
            "acquisition_mean_non_source_inspects_before_first_patch": candidate[
                "aggregate"
            ]["mean_non_source_inspects_before_first_patch"],
            "acquisition_mean_sampled_tokens": candidate["aggregate"]["mean_sampled_tokens"],
            "injected_mean_sampled_tokens": candidate_injected["aggregate"]["mean_sampled_tokens"],
            "acquisition_minus_injected_mean_sampled_tokens": float(
                candidate["aggregate"]["mean_sampled_tokens"]
            ) - float(candidate_injected["aggregate"]["mean_sampled_tokens"]),
            "acquisition_mean_logical_model_tokens": candidate["aggregate"][
                "mean_logical_model_tokens"
            ],
            "injected_mean_logical_model_tokens": candidate_injected["aggregate"][
                "mean_logical_model_tokens"
            ],
            "acquisition_minus_injected_mean_logical_model_tokens": float(
                candidate["aggregate"]["mean_logical_model_tokens"]
            ) - float(candidate_injected["aggregate"]["mean_logical_model_tokens"]),
        },
        "invalid_action_deltas_vs_start": invalid_deltas,
        "unusable_cap_deltas_vs_start": unusable_cap_deltas,
        "explicit_retention": {
            "first_patch_full_correct": explicit_first_patch,
            "unnecessary_evidence_before_first_patch": explicit_unnecessary,
        },
        "scaffold_is_diagnostic_only": True,
        "checks": checks,
        "gate": {
            "passed": gate_passed,
            "verdict": "TRANSFER_PASS" if gate_passed else "TRANSFER_FAIL",
        },
        "menagerie_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXP / "configs" / "default.yaml")
    parser.add_argument("--block", choices=["transfer_dev", "transfer_confirm"], required=True)
    for name in (
        "candidate",
        "start",
        "incumbent",
        "explicit-control",
        "shuffled-control",
        "control-search",
        "candidate-injected",
        "candidate-normal",
        "start-normal",
        "candidate-recovery",
        "start-recovery",
        "candidate-recovery-scaffold",
        "candidate-explicit",
        "sample-match-start",
        "sample-match-incumbent",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--expected-candidate-weight-sha256", required=True)
    parser.add_argument("--expected-explicit-control-weight-sha256", required=True)
    parser.add_argument("--expected-shuffled-control-weight-sha256", required=True)
    parser.add_argument("--feasibility", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cfg["__analysis_config_path__"] = str(args.config.resolve())
    acquisition = {
        "block": args.block,
        "contract": "inferred",
        "scenario_set": "acquisition",
        "mode": "deep",
        "scaffold": False,
    }
    candidate = validate_behavior_receipt(
        args.candidate, cfg, arm="candidate",
        expected_weight_sha256=args.expected_candidate_weight_sha256,
        **acquisition,
    )
    expected_feasibility = EXP / "analysis" / f"{args.block}_feasibility.json"
    if args.feasibility.resolve() != expected_feasibility.resolve():
        fail("transfer gate requires its registered comparator-feasibility receipt")
    feasibility = read_json(args.feasibility)
    feasibility_inputs = {
        "candidate": args.candidate,
        "control_search": args.control_search,
        "sample_match_start": args.sample_match_start,
        "sample_match_incumbent": args.sample_match_incumbent,
        "baseline_feasibility": (
            EXP / "analysis" / f"{args.block}_baseline_feasibility.json"
        ),
    }
    expected_feasibility_receipts = {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in feasibility_inputs.items()
    }
    if (
        feasibility.get("schema_version") != 1
        or feasibility.get("stage") != "transfer_feasibility"
        or feasibility.get("phase") != "comparators"
        or feasibility.get("block") != args.block
        or feasibility.get("gate", {}).get("passed") is not True
        or feasibility.get("analyzer_sha256")
        != sha256_file(EXP / "scripts" / "analyze_transfer_feasibility.py")
        or feasibility.get("config_sha256") != sha256_file(args.config)
        or feasibility.get("receipts") != expected_feasibility_receipts
    ):
        fail("transfer gate has a stale, failed, or mismatched feasibility ancestor")
    start = validate_behavior_receipt(
        args.start,
        cfg,
        arm="start",
        expected_weight_sha256=cfg["model"]["start_weight_sha256"],
        **acquisition,
    )
    incumbent = validate_behavior_receipt(
        args.incumbent,
        cfg,
        arm="incumbent",
        expected_weight_sha256=cfg["model"]["anchor_weight_sha256"],
        **acquisition,
    )
    explicit_control = validate_behavior_receipt(
        args.explicit_control, cfg, arm="explicit_redundant",
        expected_weight_sha256=args.expected_explicit_control_weight_sha256,
        **acquisition,
    )
    shuffled_control = validate_behavior_receipt(
        args.shuffled_control, cfg, arm="shuffled_binding",
        expected_weight_sha256=args.expected_shuffled_control_weight_sha256,
        **acquisition,
    )
    candidate_hash = candidate["model_weight_sha256"]

    def candidate_receipt(path: Path, scenario_set: str, *, scaffold: bool = False) -> dict:
        return validate_behavior_receipt(
            path,
            cfg,
            block=args.block,
            contract="inferred",
            scenario_set=scenario_set,
            mode="deep",
            arm="candidate",
            expected_weight_sha256=candidate_hash,
            scaffold=scaffold,
        )

    nondiscriminating_search = candidate_receipt(args.control_search, "random")
    candidate_injected = candidate_receipt(args.candidate_injected, "injected")
    candidate_normal = candidate_receipt(args.candidate_normal, "normal")
    candidate_recovery = candidate_receipt(args.candidate_recovery, "recovery")
    candidate_recovery_scaffold = candidate_receipt(
        args.candidate_recovery_scaffold, "recovery", scaffold=True
    )
    start_normal = validate_behavior_receipt(
        args.start_normal,
        cfg,
        block=args.block,
        contract="inferred",
        scenario_set="normal",
        mode="deep",
        arm="start",
        expected_weight_sha256=cfg["model"]["start_weight_sha256"],
        scaffold=False,
    )
    start_recovery = validate_behavior_receipt(
        args.start_recovery,
        cfg,
        block=args.block,
        contract="inferred",
        scenario_set="recovery",
        mode="deep",
        arm="start",
        expected_weight_sha256=cfg["model"]["start_weight_sha256"],
        scaffold=False,
    )
    candidate_explicit = validate_behavior_receipt(
        args.candidate_explicit,
        cfg,
        block="explicit_retention",
        contract="explicit",
        scenario_set="acquisition",
        mode="deep",
        arm="candidate",
        expected_weight_sha256=candidate_hash,
        scaffold=False,
    )
    sample_match_start = validate_matcher_receipt(
        args.sample_match_start,
        cfg,
        block=args.block,
        candidate_path=args.candidate,
        candidate=candidate,
        pool_arm="start",
        pool_weight_sha256=cfg["model"]["start_weight_sha256"],
    )
    sample_match_incumbent = validate_matcher_receipt(
        args.sample_match_incumbent,
        cfg,
        block=args.block,
        candidate_path=args.candidate,
        candidate=candidate,
        pool_arm="incumbent",
        pool_weight_sha256=cfg["model"]["anchor_weight_sha256"],
    )
    result = analyze(
        cfg,
        block=args.block,
        candidate=candidate,
        start=start,
        incumbent=incumbent,
        explicit_control=explicit_control,
        shuffled_control=shuffled_control,
        nondiscriminating_search=nondiscriminating_search,
        candidate_injected=candidate_injected,
        candidate_normal=candidate_normal,
        start_normal=start_normal,
        candidate_recovery=candidate_recovery,
        start_recovery=start_recovery,
        candidate_recovery_scaffold=candidate_recovery_scaffold,
        candidate_explicit=candidate_explicit,
        sample_match_start=sample_match_start,
        sample_match_incumbent=sample_match_incumbent,
    )
    result["analyzer_sha256"] = sha256_file(Path(__file__).resolve())
    result["config_sha256"] = sha256_file(args.config)
    receipt_paths = {
        key: value
        for key, value in vars(args).items()
        if isinstance(value, Path) and key not in {"config", "out"}
    }
    result["receipts"] = {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in receipt_paths.items()
    }
    result["sample_matcher_validation"] = {
        "start": sample_match_start,
        "incumbent": sample_match_incumbent,
    }
    result["feasibility_ancestor"] = {
        "path": str(args.feasibility.resolve()),
        "sha256": sha256_file(args.feasibility),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["gate"]["passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
