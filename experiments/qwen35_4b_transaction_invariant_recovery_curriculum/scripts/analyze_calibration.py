#!/usr/bin/env python3
"""Apply frozen transaction-family calibration feasibility or candidate gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

EXP = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if payload["block"] != "transaction_calibration" or payload["scenario_set"] != "recovery":
        raise SystemExit(f"not a transaction calibration receipt: {path}")
    return payload


def scenario(payload: dict, name: str, key: str) -> float:
    return float(payload["aggregate"]["per_scenario"][name][key])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXP / "configs" / "default.yaml")
    parser.add_argument("--start", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--locality", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    gates = cfg["calibration_gates"]
    start = load(args.start)
    replay = load(args.replay)
    if start["task_manifest_sha256"] != replay["task_manifest_sha256"]:
        raise SystemExit("calibration control manifests differ")

    if args.candidate is None:
        checks = {
            "success_absolute": 1.0 >= float(gates["success_absolute_min"]),
            "success_vs_start": 1.0 - float(start["aggregate"]["success"])
            >= float(gates["success_delta_vs_start_min"]),
            "success_vs_replay": 1.0 - float(replay["aggregate"]["success"])
            >= float(gates["success_delta_vs_replay_min"]),
            "invalid_action": -float(start["aggregate"]["invalid_action_rate_per_turn"])
            <= float(gates["invalid_action_delta_vs_start_max"]),
            "answer_cap": -float(start["aggregate"]["answer_cap_hit_rate_per_turn"])
            <= float(gates["answer_cap_hit_delta_vs_start_max"]),
            "rejected_transition": float(gates["rejected_transition_absolute_min"]) <= 1.0,
            "failed_transition": float(gates["failed_transition_absolute_min"]) <= 1.0,
        }
        result = {
            "schema_version": 1,
            "stage": "transaction_calibration_feasibility",
            "task_manifest_sha256": start["task_manifest_sha256"],
            "control_success": {
                "start": start["aggregate"]["success"],
                "replay": replay["aggregate"]["success"],
            },
            "checks": checks,
            "gate": {"passed": all(checks.values())},
        }
    else:
        if args.locality is None:
            raise SystemExit("candidate calibration requires --locality")
        candidate = load(args.candidate)
        if candidate["task_manifest_sha256"] != start["task_manifest_sha256"]:
            raise SystemExit("candidate calibration manifest differs")
        cagg = candidate["aggregate"]
        sagg = start["aggregate"]
        locality = json.loads(args.locality.read_text())
        rejected = scenario(candidate, "rejected_patch", "valid_changed_patch_within_two")
        failed = scenario(candidate, "failed_test", "changed_patch_within_two")
        deltas = {
            "start": float(cagg["success"]) - float(sagg["success"]),
            "replay": float(cagg["success"]) - float(replay["aggregate"]["success"]),
        }
        invalid_delta = float(cagg["invalid_action_rate_per_turn"]) - float(
            sagg["invalid_action_rate_per_turn"]
        )
        cap_delta = float(cagg["answer_cap_hit_rate_per_turn"]) - float(
            sagg["answer_cap_hit_rate_per_turn"]
        )
        checks = {
            "locality": bool(locality["gate"]["passed"]),
            "success_absolute": float(cagg["success"]) >= float(gates["success_absolute_min"]),
            "success_vs_start": deltas["start"] >= float(gates["success_delta_vs_start_min"]),
            "success_vs_replay": deltas["replay"] >= float(gates["success_delta_vs_replay_min"]),
            "invalid_action": invalid_delta <= float(gates["invalid_action_delta_vs_start_max"]),
            "answer_cap": cap_delta <= float(gates["answer_cap_hit_delta_vs_start_max"]),
            "rejected_transition": rejected >= float(gates["rejected_transition_absolute_min"]),
            "failed_transition": failed >= float(gates["failed_transition_absolute_min"]),
        }
        result = {
            "schema_version": 1,
            "stage": "transaction_calibration",
            "task_manifest_sha256": start["task_manifest_sha256"],
            "success": {
                "candidate": cagg["success"], "start": sagg["success"],
                "replay": replay["aggregate"]["success"],
            },
            "candidate_deltas": deltas,
            "transition_metrics": {"rejected": rejected, "failed": failed},
            "invalid_action_delta_vs_start": invalid_delta,
            "answer_cap_hit_delta_vs_start": cap_delta,
            "locality": {
                key: locality[key] for key in (
                    "median_non_target_centered_logit_drift", "mean_entropy_delta",
                    "mean_varentropy_delta",
                )
            },
            "checks": checks,
            "gate": {"passed": all(checks.values())},
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["gate"]["passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
