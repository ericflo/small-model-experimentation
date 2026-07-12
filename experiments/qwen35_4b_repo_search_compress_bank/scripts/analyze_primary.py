#!/usr/bin/env python3
"""Stop on necessary compact gates before funding the action-only control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from analyze_repo import load, paired_delta, task_map

EXP = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXP / "configs" / "default.yaml")
    parser.add_argument("--candidate-trained", type=Path, required=True)
    parser.add_argument("--apex-trained", type=Path, required=True)
    parser.add_argument("--candidate-transfer", type=Path, required=True)
    parser.add_argument("--apex-transfer", type=Path, required=True)
    parser.add_argument("--sample-more-transfer", type=Path, required=True)
    parser.add_argument("--locality", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    gates = config["gates"]
    compact_trained = load(args.candidate_trained)
    apex_trained = load(args.apex_trained)
    compact = load(args.candidate_transfer)
    apex = load(args.apex_transfer)
    sample = load(args.sample_more_transfer)
    locality = load(args.locality)
    contrasts = {
        "compact_vs_apex": paired_delta(task_map(compact), task_map(apex), seed=73510),
        "compact_vs_apex_sample_more": paired_delta(
            task_map(compact), task_map(sample), seed=73512
        ),
        "trained_compact_vs_apex": paired_delta(
            task_map(compact_trained), task_map(apex_trained), seed=73513
        ),
    }
    candidate = compact["aggregate"]
    incumbent = apex["aggregate"]
    checks = {
        "trained_family_retention": contrasts["trained_compact_vs_apex"]["delta"] >= float(gates["trained_family_delta_min"]),
        "transfer_vs_apex": contrasts["compact_vs_apex"]["delta"] >= float(gates["transfer_delta_vs_apex_min"]),
        "transfer_vs_sample_more": contrasts["compact_vs_apex_sample_more"]["delta"] >= float(gates["transfer_delta_vs_apex_sample_more_min"]),
        "bootstrap_vs_apex": contrasts["compact_vs_apex"]["ci95"][0] >= float(gates["paired_bootstrap_lower_bound_min"]),
        "bootstrap_vs_sample_more": contrasts["compact_vs_apex_sample_more"]["ci95"][0] >= float(gates["paired_bootstrap_lower_bound_min"]),
        "invalid_actions": candidate["invalid_action_rate_per_turn"] <= incumbent["invalid_action_rate_per_turn"] + float(gates["invalid_action_rate_delta_max"]),
        "verification_absolute": candidate["verified_given_success"] >= float(gates["verified_given_success_absolute_min"]),
        "verification_retention": candidate["verified_given_success"] >= incumbent["verified_given_success"] + float(gates["verified_given_success_delta_min"]),
        "commit_absolute": candidate["commit_given_verified"] >= float(gates["commit_given_verified_absolute_min"]),
        "commit_retention": candidate["commit_given_verified"] >= incumbent["commit_given_verified"] + float(gates["commit_given_verified_delta_min"]),
        "locality": bool(locality["gate"]["passed"]),
    }
    result = {
        "schema_version": 1,
        "stage": "necessary_gate_before_action_only",
        "contrasts": contrasts,
        "candidate_metrics": {
            key: candidate[key] for key in (
                "success", "verified_given_success", "commit_given_verified",
                "invalid_action_rate_per_turn", "submit_rate",
            )
        },
        "apex_metrics": {
            key: incumbent[key] for key in (
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
        "downstream_authorization": "train_action_only" if all(checks.values()) else "stop_before_action_only_confirmation_and_menagerie",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["gate"]["passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
