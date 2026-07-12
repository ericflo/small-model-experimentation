#!/usr/bin/env python3
"""Prove frozen calibration gates are attainable before candidate evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

EXP = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if payload["block"] != "calibration" or payload["scenario_set"] != "recovery":
        raise SystemExit(f"wrong calibration input: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXP / "configs" / "default.yaml")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--happy", type=Path, required=True)
    parser.add_argument("--action", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    gates = cfg["calibration_gates"]
    controls = {"base": load(args.base), "happy": load(args.happy), "action": load(args.action)}
    manifests = {payload["task_manifest_sha256"] for payload in controls.values()}
    if len(manifests) != 1:
        raise SystemExit("calibration control manifests differ")
    base = controls["base"]["aggregate"]
    checks = {
        "recovery_vs_base": 1.0 - float(base["success"]) >= float(
            gates["recovery_delta_vs_base_min"]
        ),
        "recovery_vs_happy": 1.0 - float(controls["happy"]["aggregate"]["success"]) >= float(
            gates["recovery_delta_vs_happy_min"]
        ),
        "recovery_vs_action": 1.0 - float(controls["action"]["aggregate"]["success"]) >= float(
            gates["recovery_delta_vs_action_min"]
        ),
        "invalid_action": -float(base["invalid_action_rate_per_turn"]) <= float(
            gates["invalid_action_delta_vs_base_max"]
        ),
        "answer_cap": -float(base["answer_cap_hit_rate_per_turn"]) <= float(
            gates["answer_cap_hit_delta_vs_base_max"]
        ),
        "rejected_absolute": float(
            gates["rejected_patch_valid_changed_within_two_absolute_min"]
        ) <= 1.0,
        "rejected_delta": 1.0 - float(
            base["per_scenario"]["rejected_patch"]["valid_changed_patch_within_two"]
        ) >= float(gates["rejected_patch_valid_changed_within_two_delta_min"]),
        "failed_absolute": float(
            gates["failed_test_changed_patch_within_two_absolute_min"]
        ) <= 1.0,
        "failed_delta": 1.0 - float(
            base["per_scenario"]["failed_test"]["changed_patch_within_two"]
        ) >= float(gates["failed_test_changed_patch_within_two_delta_min"]),
        "verification": float(gates["verified_given_success_absolute_min"]) <= 1.0,
        "commit": float(gates["commit_given_verified_absolute_min"]) <= 1.0,
    }
    result = {
        "schema_version": 1,
        "stage": "calibration",
        "task_manifest_sha256": next(iter(manifests)),
        "control_success": {
            name: payload["aggregate"]["success"] for name, payload in controls.items()
        },
        "checks": checks,
        "gate": {"passed": all(checks.values())},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["gate"]["passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
