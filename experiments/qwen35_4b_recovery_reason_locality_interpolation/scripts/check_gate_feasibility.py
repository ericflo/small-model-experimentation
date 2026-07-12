#!/usr/bin/env python3
"""Prove every registered transfer gate remains attainable before candidate exposure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

EXP = Path(__file__).resolve().parents[1]


def load(path: Path, block: str, scenario_set: str) -> dict:
    payload = json.loads(path.read_text())
    if payload["block"] != block or payload["scenario_set"] != scenario_set:
        raise SystemExit(f"wrong block/scenario in {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXP / "configs" / "default.yaml")
    parser.add_argument("--block", choices=["transfer_dev", "transfer_confirm"], required=True)
    parser.add_argument("--base-recovery", type=Path, required=True)
    parser.add_argument("--happy-recovery", type=Path, required=True)
    parser.add_argument("--action-recovery", type=Path, required=True)
    parser.add_argument("--sample-more-recovery", type=Path, required=True)
    parser.add_argument("--scaffold-recovery", type=Path, required=True)
    parser.add_argument("--base-normal", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    gates = cfg["transfer_gates"]
    recovery = {
        "base": load(args.base_recovery, args.block, "recovery"),
        "happy": load(args.happy_recovery, args.block, "recovery"),
        "action": load(args.action_recovery, args.block, "recovery"),
        "sample_more": load(args.sample_more_recovery, args.block, "recovery"),
        "scaffold": load(args.scaffold_recovery, args.block, "recovery"),
    }
    normal = load(args.base_normal, args.block, "normal")
    manifests = {
        payload["task_manifest_sha256"] for payload in (*recovery.values(), normal)
    }
    if len(manifests) != 1:
        raise SystemExit("control manifests differ")
    checks = {}
    thresholds = {
        "base": float(gates["recovery_delta_vs_base_min"]),
        "happy": float(gates["recovery_delta_vs_happy_min"]),
        "action": float(gates["recovery_delta_vs_action_min"]),
        "sample_more": float(gates["recovery_delta_vs_sample_more_min"]),
        "scaffold": float(gates["recovery_delta_vs_scaffold_min"]),
    }
    for name, threshold in thresholds.items():
        control_success = float(recovery[name]["aggregate"]["success"])
        checks[f"recovery_vs_{name}"] = 1.0 - control_success >= threshold
    base_scenarios = recovery["base"]["aggregate"]["per_scenario"]
    action_scenarios = recovery["action"]["aggregate"]["per_scenario"]
    checks.update({
        "rejected_absolute": float(gates["rejected_patch_transition_absolute_min"]) <= 1.0,
        "rejected_delta": (
            1.0 - float(base_scenarios["rejected_patch"]["immediate_transition_rate"])
            >= float(gates["rejected_patch_transition_delta_min"])
        ),
        "failed_absolute": float(gates["failed_test_transition_absolute_min"]) <= 1.0,
        "failed_delta": (
            1.0 - float(base_scenarios["failed_test"]["changed_patch_within_two"])
            >= float(gates["failed_test_transition_delta_min"])
        ),
        "normal_success": (
            1.0 - float(normal["aggregate"]["success"])
            >= float(gates["normal_success_delta_min"])
        ),
        "normal_verified_absolute": float(gates["normal_verified_absolute_min"]) <= 1.0,
        "normal_verified_delta": (
            1.0 - float(normal["aggregate"]["verified_given_success"])
            >= float(gates["normal_verified_delta_min"])
        ),
        "normal_commit_absolute": float(gates["normal_commit_absolute_min"]) <= 1.0,
        "normal_commit_delta": (
            1.0 - float(normal["aggregate"]["commit_given_verified"])
            >= float(gates["normal_commit_delta_min"])
        ),
        "invalid_action": (
            -float(recovery["base"]["aggregate"]["invalid_action_rate_per_turn"])
            <= float(gates["invalid_action_rate_delta_max"])
        ),
        "family_count": int(gates["minimum_nonnegative_transfer_families"]) <= len(
            recovery["base"]["aggregate"]["per_family"]
        ),
        "family_regression": all(
            1.0 - float(row["success"]) >= float(gates["maximum_single_family_regression"])
            for row in recovery["base"]["aggregate"]["per_family"].values()
        ),
    })
    invalid_headroom = float(recovery["action"]["aggregate"]["invalid_action_rate_per_turn"])
    rejected_headroom = 1.0 - float(
        action_scenarios["rejected_patch"]["immediate_transition_rate"]
    )
    checks["plan_contrast_mechanism"] = (
        invalid_headroom >= float(gates["action_invalid_rate_improvement_min"])
        or rejected_headroom >= float(gates["action_rejected_transition_improvement_min"])
    )
    result = {
        "schema_version": 1,
        "block": args.block,
        "task_manifest_sha256": next(iter(manifests)),
        "checks": checks,
        "gate": {"passed": all(checks.values())},
        "control_success": {
            name: payload["aggregate"]["success"] for name, payload in recovery.items()
        },
        "action_mechanism_headroom": {
            "invalid_rate": invalid_headroom,
            "rejected_transition": rejected_headroom,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["gate"]["passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
