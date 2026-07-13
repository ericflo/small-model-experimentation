#!/usr/bin/env python3
"""Gate broad recovery and normal-loop retention before Menagerie."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

EXP = Path(__file__).resolve().parents[1]


def load(path: Path, scenario_set: str) -> dict:
    payload = json.loads(path.read_text())
    if payload["block"] != "broad_recovery" or payload["scenario_set"] != scenario_set:
        raise SystemExit(f"wrong retention input: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXP / "configs" / "default.yaml")
    parser.add_argument("--candidate-recovery", type=Path, required=True)
    parser.add_argument("--start-recovery", type=Path, required=True)
    parser.add_argument("--candidate-normal", type=Path, required=True)
    parser.add_argument("--start-normal", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    gates = yaml.safe_load(args.config.read_text())["retention_gates"]
    candidate_recovery = load(args.candidate_recovery, "recovery")
    start_recovery = load(args.start_recovery, "recovery")
    candidate_normal = load(args.candidate_normal, "normal")
    start_normal = load(args.start_normal, "normal")
    manifests = {
        (payload["task_manifest_sha256"], payload["task_content_manifest_sha256"])
        for payload in (
            candidate_recovery, start_recovery, candidate_normal, start_normal
        )
    }
    if len(manifests) != 1:
        raise SystemExit("retention manifests differ")
    cr = candidate_recovery["aggregate"]
    sr = start_recovery["aggregate"]
    cn = candidate_normal["aggregate"]
    sn = start_normal["aggregate"]
    rejected = float(cr["per_scenario"]["rejected_patch"]["valid_changed_patch_within_two"])
    failed = float(cr["per_scenario"]["failed_test"]["changed_patch_within_two"])
    recovery_delta = float(cr["success"]) - float(sr["success"])
    normal_delta = float(cn["success"]) - float(sn["success"])
    invalid_delta = float(cr["invalid_action_rate_per_turn"]) - float(sr["invalid_action_rate_per_turn"])
    cap_delta = float(cr["answer_cap_hit_rate_per_turn"]) - float(sr["answer_cap_hit_rate_per_turn"])
    checks = {
        "recovery_success": recovery_delta >= float(gates["recovery_success_delta_vs_start_min"]),
        "normal_success": normal_delta >= float(gates["normal_success_delta_vs_start_min"]),
        "verified": float(cn["verified_given_success"]) >= float(gates["verified_absolute_min"]),
        "commit": float(cn["commit_given_verified"]) >= float(gates["commit_absolute_min"]),
        "rejected_transition": rejected >= float(gates["rejected_transition_absolute_min"]),
        "failed_transition": failed >= float(gates["failed_transition_absolute_min"]),
        "invalid_action": invalid_delta <= float(gates["invalid_action_delta_vs_start_max"]),
        "answer_cap": cap_delta <= float(gates["answer_cap_hit_delta_vs_start_max"]),
    }
    result = {
        "schema_version": 1, "stage": "broad_retention",
        "recovery_success": {"candidate": cr["success"], "start": sr["success"], "delta": recovery_delta},
        "normal_success": {"candidate": cn["success"], "start": sn["success"], "delta": normal_delta},
        "normal_verified_given_success": cn["verified_given_success"],
        "normal_commit_given_verified": cn["commit_given_verified"],
        "transition_metrics": {"rejected": rejected, "failed": failed},
        "invalid_action_delta_vs_start": invalid_delta,
        "answer_cap_hit_delta_vs_start": cap_delta,
        "checks": checks, "gate": {"passed": all(checks.values())},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["gate"]["passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
