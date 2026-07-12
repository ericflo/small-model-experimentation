#!/usr/bin/env python3
"""Apply the frozen payload-harness calibration and locality gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

EXP = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if payload["block"] != "calibration" or payload["scenario_set"] != "recovery":
        raise SystemExit(f"not a calibration recovery receipt: {path}")
    return payload


def metric(payload: dict, scenario: str, key: str) -> float:
    return float(payload["aggregate"]["per_scenario"][scenario][key])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXP / "configs" / "default.yaml")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--happy", type=Path, required=True)
    parser.add_argument("--action", type=Path, required=True)
    parser.add_argument("--locality", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    gates = cfg["calibration_gates"]
    payloads = {
        "candidate": load(args.candidate),
        "base": load(args.base),
        "happy": load(args.happy),
        "action": load(args.action),
    }
    manifests = {payload["task_manifest_sha256"] for payload in payloads.values()}
    if len(manifests) != 1:
        raise SystemExit("calibration manifests differ")
    candidate = payloads["candidate"]["aggregate"]
    base = payloads["base"]["aggregate"]
    deltas = {
        name: float(candidate["success"]) - float(payload["aggregate"]["success"])
        for name, payload in payloads.items()
        if name != "candidate"
    }
    rejected = metric(
        payloads["candidate"], "rejected_patch", "valid_changed_patch_within_two"
    )
    rejected_base = metric(
        payloads["base"], "rejected_patch", "valid_changed_patch_within_two"
    )
    failed = metric(payloads["candidate"], "failed_test", "changed_patch_within_two")
    failed_base = metric(payloads["base"], "failed_test", "changed_patch_within_two")
    invalid_delta = (
        float(candidate["invalid_action_rate_per_turn"])
        - float(base["invalid_action_rate_per_turn"])
    )
    answer_cap_delta = (
        float(candidate["answer_cap_hit_rate_per_turn"])
        - float(base["answer_cap_hit_rate_per_turn"])
    )
    locality = json.loads(args.locality.read_text())
    locality_candidate = locality["candidates"].get("candidate")
    if locality_candidate is None:
        raise SystemExit("fresh locality receipt has no candidate")
    checks = {
        "locality": bool(locality["gate"]["passed"]),
        "recovery_vs_base": deltas["base"] >= float(gates["recovery_delta_vs_base_min"]),
        "recovery_vs_happy": deltas["happy"] >= float(gates["recovery_delta_vs_happy_min"]),
        "recovery_vs_action": deltas["action"] >= float(gates["recovery_delta_vs_action_min"]),
        "invalid_action_retention": invalid_delta <= float(
            gates["invalid_action_delta_vs_base_max"]
        ),
        "answer_cap_retention": answer_cap_delta <= float(
            gates["answer_cap_hit_delta_vs_base_max"]
        ),
        "rejected_transition_absolute": rejected >= float(
            gates["rejected_patch_valid_changed_within_two_absolute_min"]
        ),
        "rejected_transition_delta": rejected - rejected_base >= float(
            gates["rejected_patch_valid_changed_within_two_delta_min"]
        ),
        "failed_transition_absolute": failed >= float(
            gates["failed_test_changed_patch_within_two_absolute_min"]
        ),
        "failed_transition_delta": failed - failed_base >= float(
            gates["failed_test_changed_patch_within_two_delta_min"]
        ),
        "verification": float(candidate["verified_given_success"]) >= float(
            gates["verified_given_success_absolute_min"]
        ),
        "commit": float(candidate["commit_given_verified"]) >= float(
            gates["commit_given_verified_absolute_min"]
        ),
    }
    result = {
        "schema_version": 1,
        "stage": "calibration",
        "candidate_arm": payloads["candidate"]["arm"],
        "task_manifest_sha256": next(iter(manifests)),
        "success": {name: payload["aggregate"]["success"] for name, payload in payloads.items()},
        "candidate_deltas": deltas,
        "candidate_metrics": {
            key: candidate[key]
            for key in (
                "success",
                "invalid_action_rate_per_turn",
                "answer_cap_hit_rate_per_turn",
                "invalid_answer_cap_hit_fraction",
                "verified_given_success",
                "commit_given_verified",
                "mean_sampled_tokens",
            )
        },
        "transition_metrics": {
            "rejected_candidate": rejected,
            "rejected_base": rejected_base,
            "failed_candidate": failed,
            "failed_base": failed_base,
        },
        "locality": {
            key: locality_candidate[key]
            for key in (
                "median_non_target_centered_logit_drift",
                "mean_entropy_delta",
                "mean_varentropy_delta",
                "gate",
            )
        },
        "invalid_action_delta_vs_base": invalid_delta,
        "answer_cap_hit_delta_vs_base": answer_cap_delta,
        "checks": checks,
        "gate": {"passed": all(checks.values())},
        "downstream_authorization": (
            "run_transfer_dev" if all(checks.values()) else "stop_before_transfer"
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["gate"]["passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
