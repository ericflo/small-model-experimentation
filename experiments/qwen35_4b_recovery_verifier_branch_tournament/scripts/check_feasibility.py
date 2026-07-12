#!/usr/bin/env python3
"""Prove the frozen tournament thresholds are attainable before selector scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


EXP = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXP / "configs" / "default.yaml")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--action", type=Path, required=True)
    parser.add_argument("--candidate-sample-more", type=Path, required=True)
    parser.add_argument("--action-sample-more", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    gates = cfg["gates"]
    candidate = load(args.candidate)
    action = load(args.action)
    sample_candidate = load(args.candidate_sample_more)
    sample_action = load(args.action_sample_more)
    candidate_cases = {
        row["case_id"]: bool(row["success"])
        for row in candidate["aggregate"]["cases"]
    }
    action_cases = {
        row["case_id"]: bool(row["success"])
        for row in action["aggregate"]["cases"]
    }
    if candidate_cases.keys() != action_cases.keys():
        raise SystemExit("candidate/action case IDs differ")
    union = sum(candidate_cases[key] or action_cases[key] for key in candidate_cases) / len(
        candidate_cases
    )
    success = {
        "candidate": candidate["aggregate"]["success"],
        "action": action["aggregate"]["success"],
        "candidate_sample_more": sample_candidate["aggregate"]["success"],
        "action_sample_more": sample_action["aggregate"]["success"],
    }
    checks = {
        "union_vs_best_single": union
        >= max(success["candidate"], success["action"])
        + float(gates["success_delta_vs_best_single_min"]),
        "union_vs_candidate_sample_more": union
        >= success["candidate_sample_more"]
        + float(gates["success_delta_vs_candidate_sample_more_min"]),
        "union_vs_action_sample_more": union
        >= success["action_sample_more"]
        + float(gates["success_delta_vs_action_sample_more_min"]),
    }
    result = {
        "schema_version": 1,
        "block": candidate["block"],
        "oracle_union_ceiling": union,
        "control_success": success,
        "checks": checks,
        "gate": {"passed": all(checks.values())},
        "downstream_authorization": (
            "score_frozen_selector" if all(checks.values()) else "stop_before_selector"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["gate"]["passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
