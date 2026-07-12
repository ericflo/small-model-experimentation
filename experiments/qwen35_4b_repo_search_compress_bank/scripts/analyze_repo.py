#!/usr/bin/env python3
"""Paired bootstrap and preregistered repository gate analysis."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

import yaml

EXP = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def task_map(payload: dict) -> dict[str, dict]:
    return {row["task_id"]: row for row in payload["aggregate"]["tasks"]}


def paired_delta(left: dict[str, dict], right: dict[str, dict], *, seed: int) -> dict:
    if set(left) != set(right):
        raise ValueError("paired task ids differ")
    ids = sorted(left)
    differences = [int(left[item]["success"]) - int(right[item]["success"]) for item in ids]
    rng = random.Random(seed)
    samples = []
    for _ in range(20000):
        samples.append(sum(differences[rng.randrange(len(ids))] for _ in ids) / len(ids))
    samples.sort()
    return {
        "n": len(ids),
        "left_success": sum(left[item]["success"] for item in ids) / len(ids),
        "right_success": sum(right[item]["success"] for item in ids) / len(ids),
        "delta": statistics.mean(differences),
        "ci95": [samples[int(0.025 * len(samples))], samples[int(0.975 * len(samples))]],
        "left_only": sum(left[item]["success"] and not right[item]["success"] for item in ids),
        "right_only": sum(right[item]["success"] and not left[item]["success"] for item in ids),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXP / "configs" / "default.yaml")
    parser.add_argument("--candidate-trained", type=Path, required=True)
    parser.add_argument("--apex-trained", type=Path, required=True)
    parser.add_argument("--candidate-transfer", type=Path, required=True)
    parser.add_argument("--apex-transfer", type=Path, required=True)
    parser.add_argument("--action-transfer", type=Path, required=True)
    parser.add_argument("--sample-more-transfer", type=Path, required=True)
    parser.add_argument("--locality", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    gates = config["gates"]
    candidate_trained = load(args.candidate_trained)
    apex_trained = load(args.apex_trained)
    candidate = load(args.candidate_transfer)
    apex = load(args.apex_transfer)
    action = load(args.action_transfer)
    sample = load(args.sample_more_transfer)
    locality = load(args.locality)
    maps = {
        "candidate": task_map(candidate),
        "apex": task_map(apex),
        "action_only": task_map(action),
        "apex_sample_more": task_map(sample),
    }
    contrasts = {
        "compact_vs_apex": paired_delta(maps["candidate"], maps["apex"], seed=73510),
        "compact_vs_action_only": paired_delta(maps["candidate"], maps["action_only"], seed=73511),
        "compact_vs_apex_sample_more": paired_delta(maps["candidate"], maps["apex_sample_more"], seed=73512),
        "trained_compact_vs_apex": paired_delta(
            task_map(candidate_trained), task_map(apex_trained), seed=73513
        ),
    }
    candidate_agg = candidate["aggregate"]
    apex_agg = apex["aggregate"]
    checks = {
        "trained_family_retention": contrasts["trained_compact_vs_apex"]["delta"] >= float(gates["trained_family_delta_min"]),
        "transfer_vs_apex": contrasts["compact_vs_apex"]["delta"] >= float(gates["transfer_delta_vs_apex_min"]),
        "transfer_vs_action_only": contrasts["compact_vs_action_only"]["delta"] >= float(gates["transfer_delta_vs_action_only_min"]),
        "transfer_vs_sample_more": contrasts["compact_vs_apex_sample_more"]["delta"] >= float(gates["transfer_delta_vs_apex_sample_more_min"]),
        "bootstrap_vs_apex": contrasts["compact_vs_apex"]["ci95"][0] >= float(gates["paired_bootstrap_lower_bound_min"]),
        "bootstrap_vs_action_only": contrasts["compact_vs_action_only"]["ci95"][0] >= float(gates["paired_bootstrap_lower_bound_min"]),
        "bootstrap_vs_sample_more": contrasts["compact_vs_apex_sample_more"]["ci95"][0] >= float(gates["paired_bootstrap_lower_bound_min"]),
        "invalid_actions": candidate_agg["invalid_action_rate_per_turn"] <= apex_agg["invalid_action_rate_per_turn"] + float(gates["invalid_action_rate_delta_max"]),
        "verification_absolute": candidate_agg["verified_given_success"] >= float(gates["verified_given_success_absolute_min"]),
        "verification_retention": candidate_agg["verified_given_success"] >= apex_agg["verified_given_success"] + float(gates["verified_given_success_delta_min"]),
        "commit_absolute": candidate_agg["commit_given_verified"] >= float(gates["commit_given_verified_absolute_min"]),
        "commit_retention": candidate_agg["commit_given_verified"] >= apex_agg["commit_given_verified"] + float(gates["commit_given_verified_delta_min"]),
        "locality": bool(locality["gate"]["passed"]),
    }
    result = {
        "schema_version": 1,
        "contrasts": contrasts,
        "candidate_metrics": {
            key: candidate_agg[key]
            for key in (
                "success", "verified_given_success", "commit_given_verified",
                "invalid_action_rate_per_turn", "submit_rate",
            )
        },
        "apex_metrics": {
            key: apex_agg[key]
            for key in (
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
        "downstream_authorization": "confirm_transfer" if all(checks.values()) else "stop_before_confirmation_and_menagerie",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["gate"]["passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
