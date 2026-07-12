#!/usr/bin/env python3
"""Evaluate the preregistered public branch selector against frozen controls."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import yaml


EXP = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def eval_cases(payload: dict) -> dict[str, bool]:
    return {row["case_id"]: bool(row["success"]) for row in payload["aggregate"]["cases"]}


def tournament_cases(payload: dict) -> dict[str, bool]:
    result = {}
    for row in payload["cases"]:
        result[row["case_id"]] = bool(row[f"{row['public_arm']}_success"])
    return result


def paired_ci(left: dict[str, bool], right: dict[str, bool], seed: int) -> list[float]:
    if left.keys() != right.keys():
        raise ValueError("paired case IDs differ")
    keys = sorted(left)
    rng = random.Random(seed)
    draws = []
    for _ in range(10_000):
        sampled = [keys[rng.randrange(len(keys))] for _ in keys]
        draws.append(sum(int(left[key]) - int(right[key]) for key in sampled) / len(keys))
    draws.sort()
    return [draws[249], draws[9749]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXP / "configs" / "default.yaml")
    parser.add_argument("--tournament", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--action", type=Path, required=True)
    parser.add_argument("--candidate-sample-more", type=Path, required=True)
    parser.add_argument("--action-sample-more", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    gates = cfg["gates"]
    tournament = load(args.tournament)
    controls = {
        "candidate": load(args.candidate),
        "action": load(args.action),
        "candidate_sample_more": load(args.candidate_sample_more),
        "action_sample_more": load(args.action_sample_more),
    }
    success = {name: payload["aggregate"]["success"] for name, payload in controls.items()}
    public_success = float(tournament["public"]["success"])
    best_single_name = max(("candidate", "action"), key=lambda name: success[name])
    best_single = success[best_single_name]
    public_cases = tournament_cases(tournament)
    ci_best_single = paired_ci(public_cases, eval_cases(controls[best_single_name]), 85611)
    best_sample_name = max(
        ("candidate_sample_more", "action_sample_more"), key=lambda name: success[name]
    )
    ci_best_sample = paired_ci(public_cases, eval_cases(controls[best_sample_name]), 85612)

    family_deltas = {}
    public_families = tournament["public"]["per_family"]
    control_families = controls[best_single_name]["aggregate"]["per_family"]
    for family in sorted(public_families):
        family_deltas[family] = (
            public_families[family]["success"] - control_families[family]["success"]
        )

    candidate_aggregate = controls["candidate"]["aggregate"]
    contrasts = {
        "best_single": public_success - best_single,
        "candidate_sample_more": public_success - success["candidate_sample_more"],
        "action_sample_more": public_success - success["action_sample_more"],
        "expected_random": public_success - tournament["expected_random"]["success"],
    }
    checks = {
        "success_vs_best_single": contrasts["best_single"]
        >= float(gates["success_delta_vs_best_single_min"]),
        "success_vs_candidate_sample_more": contrasts["candidate_sample_more"]
        >= float(gates["success_delta_vs_candidate_sample_more_min"]),
        "success_vs_action_sample_more": contrasts["action_sample_more"]
        >= float(gates["success_delta_vs_action_sample_more_min"]),
        "success_vs_expected_random": contrasts["expected_random"]
        >= float(gates["success_delta_vs_random_selector_min"]),
        "bootstrap_vs_best_single": ci_best_single[0]
        >= float(gates["paired_bootstrap_lower_bound_min"]),
        "bootstrap_vs_best_sample_more": ci_best_sample[0]
        >= float(gates["paired_bootstrap_lower_bound_min"]),
        "oracle_union_capture": tournament["oracle_union"]["public_capture"]
        >= float(gates["oracle_union_capture_min"]),
        "rejected_patch_transition": tournament["public"]["per_scenario"][
            "rejected_patch"
        ]["changed_patch_within_two"]
        >= float(gates["rejected_patch_transition_absolute_min"]),
        "failed_test_transition": tournament["public"]["per_scenario"]["failed_test"][
            "changed_patch_within_two"
        ]
        >= float(gates["failed_test_transition_absolute_min"]),
        "invalid_action_retention": tournament["public"]["invalid_action_rate_per_turn"]
        <= candidate_aggregate["invalid_action_rate_per_turn"]
        + float(gates["invalid_action_delta_vs_candidate_max"]),
        "answer_cap_retention": tournament["public"]["answer_cap_hit_rate_per_turn"]
        <= candidate_aggregate["answer_cap_hit_rate_per_turn"]
        + float(gates["answer_cap_hit_delta_vs_candidate_max"]),
        "family_nonnegative_count": sum(value >= 0 for value in family_deltas.values())
        >= int(gates["minimum_nonnegative_families_vs_best_single"]),
        "family_max_regression": min(family_deltas.values())
        >= float(gates["maximum_single_family_regression"]),
    }
    result = {
        "schema_version": 1,
        "block": tournament["block"],
        "public_success": public_success,
        "control_success": success,
        "best_single": best_single_name,
        "best_sample_more": best_sample_name,
        "contrasts": contrasts,
        "paired_ci95": {
            "best_single": ci_best_single,
            "best_sample_more": ci_best_sample,
        },
        "family_deltas_vs_best_single": family_deltas,
        "public_metrics": tournament["public"],
        "oracle_union": tournament["oracle_union"],
        "checks": checks,
        "gate": {"passed": all(checks.values())},
        "downstream_authorization": (
            "run_next_stage" if all(checks.values()) else "stop_before_next_stage"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["gate"]["passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
