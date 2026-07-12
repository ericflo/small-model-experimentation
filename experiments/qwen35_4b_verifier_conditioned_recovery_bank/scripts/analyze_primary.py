#!/usr/bin/env python3
"""Apply preregistered recovery, control, retention, and locality gates."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import yaml

EXP = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def case_map(payload: dict) -> dict[str, dict]:
    return {row["case_id"]: row for row in payload["aggregate"]["cases"]}


def paired_delta(left: dict[str, dict], right: dict[str, dict], seed: int) -> dict:
    if set(left) != set(right):
        raise SystemExit(
            f"paired case mismatch: left-only={len(set(left)-set(right))} "
            f"right-only={len(set(right)-set(left))}"
        )
    keys = sorted(left)
    differences = [float(left[key]["success"]) - float(right[key]["success"]) for key in keys]
    rng = random.Random(seed)
    samples = []
    for _ in range(10000):
        samples.append(sum(differences[rng.randrange(len(keys))] for _ in keys) / len(keys))
    samples.sort()
    return {
        "n": len(keys),
        "delta": sum(differences) / len(differences),
        "left_only": sum(value == 1 for value in differences),
        "right_only": sum(value == -1 for value in differences),
        "ties": sum(value == 0 for value in differences),
        "ci95": [samples[249], samples[9749]],
    }


def family_deltas(left: dict[str, dict], right: dict[str, dict]) -> dict[str, float]:
    result = {}
    families = sorted({row["family"] for row in left.values()})
    for family in families:
        keys = [key for key, row in left.items() if row["family"] == family]
        result[family] = sum(
            float(left[key]["success"]) - float(right[key]["success"]) for key in keys
        ) / len(keys)
    return result


def scenario_metric(payload: dict, scenario: str, key: str) -> float:
    return float(payload["aggregate"]["per_scenario"][scenario][key])


def validate_inputs(recovery: dict[str, dict], normal: dict[str, dict], block: str) -> None:
    for name, payload in recovery.items():
        if (
            payload["block"] != block
            or payload["scenario_set"] != "recovery"
            or payload["task_manifest_sha256"] != next(iter(recovery.values()))["task_manifest_sha256"]
        ):
            raise SystemExit(f"mismatched recovery input {name}")
    for name, payload in normal.items():
        if payload["block"] != block or payload["scenario_set"] != "normal":
            raise SystemExit(f"mismatched normal input {name}")
    manifests = {
        payload["task_manifest_sha256"] for payload in (*recovery.values(), *normal.values())
    }
    if len(manifests) != 1:
        raise SystemExit("recovery and normal manifests differ")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXP / "configs" / "default.yaml")
    parser.add_argument("--block", choices=["transfer_dev", "transfer_confirm"], required=True)
    parser.add_argument("--candidate-recovery", type=Path, required=True)
    parser.add_argument("--base-recovery", type=Path, required=True)
    parser.add_argument("--happy-recovery", type=Path, required=True)
    parser.add_argument("--sample-more-recovery", type=Path, required=True)
    parser.add_argument("--scaffold-recovery", type=Path, required=True)
    parser.add_argument("--candidate-normal", type=Path, required=True)
    parser.add_argument("--base-normal", type=Path, required=True)
    parser.add_argument("--locality", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    gates = cfg["gates"]
    recovery = {
        "candidate": load(args.candidate_recovery),
        "base": load(args.base_recovery),
        "happy": load(args.happy_recovery),
        "sample_more": load(args.sample_more_recovery),
        "scaffold": load(args.scaffold_recovery),
    }
    normal = {
        "candidate": load(args.candidate_normal),
        "base": load(args.base_normal),
    }
    validate_inputs(recovery, normal, args.block)
    locality = load(args.locality)
    candidate_cases = case_map(recovery["candidate"])
    base_cases = case_map(recovery["base"])
    contrasts = {
        name: paired_delta(candidate_cases, case_map(recovery[name]), 85100 + index)
        for index, name in enumerate(("base", "happy", "sample_more", "scaffold"))
    }
    normal_contrast = paired_delta(
        case_map(normal["candidate"]), case_map(normal["base"]), seed=85110
    )
    family = family_deltas(candidate_cases, base_cases)
    candidate = recovery["candidate"]["aggregate"]
    incumbent = recovery["base"]["aggregate"]
    candidate_normal = normal["candidate"]["aggregate"]
    incumbent_normal = normal["base"]["aggregate"]
    rejected_transition = scenario_metric(
        recovery["candidate"], "rejected_patch", "immediate_transition_rate"
    )
    rejected_base = scenario_metric(
        recovery["base"], "rejected_patch", "immediate_transition_rate"
    )
    failed_transition = scenario_metric(
        recovery["candidate"], "failed_test", "changed_patch_within_two"
    )
    failed_base = scenario_metric(
        recovery["base"], "failed_test", "changed_patch_within_two"
    )
    checks = {
        "recovery_vs_base": contrasts["base"]["delta"] >= float(gates["recovery_delta_vs_base_min"]),
        "recovery_vs_happy": contrasts["happy"]["delta"] >= float(gates["recovery_delta_vs_happy_min"]),
        "recovery_vs_sample_more": contrasts["sample_more"]["delta"] >= float(gates["recovery_delta_vs_sample_more_min"]),
        "recovery_vs_scaffold": contrasts["scaffold"]["delta"] >= float(gates["recovery_delta_vs_scaffold_min"]),
        "bootstrap_vs_base": contrasts["base"]["ci95"][0] >= float(gates["paired_bootstrap_lower_bound_min"]),
        "bootstrap_vs_sample_more": contrasts["sample_more"]["ci95"][0] >= float(gates["paired_bootstrap_lower_bound_min"]),
        "rejected_patch_transition_absolute": rejected_transition >= float(gates["rejected_patch_transition_absolute_min"]),
        "rejected_patch_transition_delta": rejected_transition - rejected_base >= float(gates["rejected_patch_transition_delta_min"]),
        "failed_test_transition_absolute": failed_transition >= float(gates["failed_test_transition_absolute_min"]),
        "failed_test_transition_delta": failed_transition - failed_base >= float(gates["failed_test_transition_delta_min"]),
        "normal_success_retention": normal_contrast["delta"] >= float(gates["normal_success_delta_min"]),
        "normal_verification_absolute": candidate_normal["verified_given_success"] >= float(gates["normal_verified_absolute_min"]),
        "normal_verification_retention": candidate_normal["verified_given_success"] - incumbent_normal["verified_given_success"] >= float(gates["normal_verified_delta_min"]),
        "normal_commit_absolute": candidate_normal["commit_given_verified"] >= float(gates["normal_commit_absolute_min"]),
        "normal_commit_retention": candidate_normal["commit_given_verified"] - incumbent_normal["commit_given_verified"] >= float(gates["normal_commit_delta_min"]),
        "invalid_action_retention": candidate["invalid_action_rate_per_turn"] - incumbent["invalid_action_rate_per_turn"] <= float(gates["invalid_action_rate_delta_max"]),
        "family_nonnegative_count": sum(value >= 0 for value in family.values()) >= int(gates["minimum_nonnegative_transfer_families"]),
        "family_max_regression": min(family.values()) >= float(gates["maximum_single_family_regression"]),
        "locality": bool(locality["gate"]["passed"]),
    }
    result = {
        "schema_version": 1,
        "stage": args.block,
        "candidate_arm": recovery["candidate"]["arm"],
        "contrasts": contrasts,
        "normal_contrast": normal_contrast,
        "family_deltas_vs_base": family,
        "transition_metrics": {
            "rejected_patch_candidate": rejected_transition,
            "rejected_patch_base": rejected_base,
            "failed_test_changed_patch_within_two_candidate": failed_transition,
            "failed_test_changed_patch_within_two_base": failed_base,
        },
        "candidate_recovery_metrics": {
            key: candidate[key] for key in (
                "success", "verified_given_success", "commit_given_verified",
                "invalid_action_rate_per_turn", "submit_rate",
            )
        },
        "candidate_normal_metrics": {
            key: candidate_normal[key] for key in (
                "success", "verified_given_success", "commit_given_verified",
                "invalid_action_rate_per_turn", "submit_rate",
            )
        },
        "locality": {
            "median_non_target_centered_logit_drift": locality["median_non_target_centered_logit_drift"],
            "ceiling": locality["ceiling"],
            "passed": locality["gate"]["passed"],
        },
        "checks": checks,
        "gate": {"passed": all(checks.values())},
        "downstream_authorization": (
            "run_next_stage" if all(checks.values()) else "stop_before_next_stage_and_menagerie"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["gate"]["passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
