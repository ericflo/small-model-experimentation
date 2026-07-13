#!/usr/bin/env python3
"""Apply frozen validation-policy feasibility or candidate transfer gates."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import yaml

EXP = Path(__file__).resolve().parents[1]
SENTINEL = "atomic_reservations"


def load(path: Path, block: str) -> dict:
    payload = json.loads(path.read_text())
    if payload["block"] != block or payload["scenario_set"] != "recovery":
        raise SystemExit(f"wrong policy-transfer input: {path}")
    return payload


def cases(payload: dict) -> dict[str, dict]:
    return {row["case_id"]: row for row in payload["aggregate"]["cases"]}


def paired(left: dict[str, dict], right: dict[str, dict], seed: int) -> dict:
    if set(left) != set(right):
        raise SystemExit("paired policy case IDs differ")
    keys = sorted(left)
    differences = [float(left[key]["success"]) - float(right[key]["success"]) for key in keys]
    rng = random.Random(seed)
    samples = sorted(
        sum(differences[rng.randrange(len(keys))] for _ in keys) / len(keys)
        for _ in range(10000)
    )
    return {
        "n": len(keys),
        "delta": sum(differences) / len(keys),
        "left_only": sum(value == 1 for value in differences),
        "right_only": sum(value == -1 for value in differences),
        "ties": sum(value == 0 for value in differences),
        "ci95": [samples[249], samples[9749]],
    }


def family_deltas(left: dict[str, dict], right: dict[str, dict]) -> dict[str, float]:
    result = {}
    for family in sorted({row["family"] for row in left.values()}):
        ids = [key for key, row in left.items() if row["family"] == family]
        result[family] = sum(
            float(left[key]["success"]) - float(right[key]["success"]) for key in ids
        ) / len(ids)
    return result


def transition(payload: dict, scenario: str, key: str) -> float:
    return float(payload["aggregate"]["per_scenario"][scenario][key])


def same_manifests(payloads: list[dict]) -> bool:
    return len({
        (row["task_manifest_sha256"], row["task_content_manifest_sha256"])
        for row in payloads
    }) == 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXP / "configs" / "default.yaml")
    parser.add_argument("--block", choices=["policy_dev", "policy_confirm"], required=True)
    parser.add_argument("--start", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--sample-more", type=Path, required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    gates = yaml.safe_load(args.config.read_text())["policy_gates"]
    controls = {
        "start": load(args.start, args.block),
        "control": load(args.control, args.block),
        "sample_more": load(args.sample_more, args.block),
    }
    if not same_manifests(list(controls.values())):
        raise SystemExit("policy control manifests differ")
    families = set(controls["start"]["aggregate"]["per_family"])
    fresh_families = sorted(families - {SENTINEL})
    if SENTINEL not in families or len(fresh_families) != 3:
        raise SystemExit("policy block must contain one sentinel and three fresh families")

    if args.candidate is None:
        thresholds = {
            "start": float(gates["success_delta_vs_start_min"]),
            "control": float(gates["success_delta_vs_control_min"]),
            "sample_more": float(gates["success_delta_vs_sample_more_min"]),
        }
        checks = {
            f"success_vs_{name}": 1.0 - float(payload["aggregate"]["success"])
            >= thresholds[name]
            for name, payload in controls.items()
        }
        checks.update({
            "sentinel_absolute": float(gates["atomic_reservations_success_absolute_min"]) <= 1.0,
            "rejected_transition": float(gates["rejected_transition_absolute_min"]) <= 1.0,
            "failed_transition": float(gates["failed_transition_absolute_min"]) <= 1.0,
            "verified": float(gates["verified_absolute_min"]) <= 1.0,
            "commit": float(gates["commit_absolute_min"]) <= 1.0,
            "invalid_action": -float(controls["start"]["aggregate"]["invalid_action_rate_per_turn"])
            <= float(gates["invalid_action_delta_vs_start_max"]),
            "answer_cap": -float(controls["start"]["aggregate"]["answer_cap_hit_rate_per_turn"])
            <= float(gates["answer_cap_hit_delta_vs_start_max"]),
            "fresh_family_count": int(gates["minimum_nonnegative_families"])
            <= len(fresh_families),
            "fresh_family_regression": all(
                1.0 - float(controls["start"]["aggregate"]["per_family"][family]["success"])
                >= float(gates["maximum_single_family_regression"])
                for family in fresh_families
            ),
        })
        result = {
            "schema_version": 1,
            "stage": f"{args.block}_feasibility",
            "task_manifest_sha256": controls["start"]["task_manifest_sha256"],
            "task_content_manifest_sha256": controls["start"]["task_content_manifest_sha256"],
            "fresh_families": fresh_families,
            "sentinel": SENTINEL,
            "control_success": {
                name: row["aggregate"]["success"] for name, row in controls.items()
            },
            "checks": checks,
            "gate": {"passed": all(checks.values())},
        }
    else:
        candidate = load(args.candidate, args.block)
        if not same_manifests([*controls.values(), candidate]):
            raise SystemExit("candidate policy manifest differs")
        candidate_cases = cases(candidate)
        contrasts = {
            name: paired(candidate_cases, cases(payload), 87390 + index)
            for index, (name, payload) in enumerate(controls.items())
        }
        family = family_deltas(candidate_cases, cases(controls["start"]))
        fresh_family = {name: family[name] for name in fresh_families}
        cagg = candidate["aggregate"]
        sagg = controls["start"]["aggregate"]
        rejected = transition(candidate, "rejected_patch", "valid_changed_patch_within_two")
        failed = transition(candidate, "failed_test", "changed_patch_within_two")
        invalid_delta = float(cagg["invalid_action_rate_per_turn"]) - float(
            sagg["invalid_action_rate_per_turn"]
        )
        cap_delta = float(cagg["answer_cap_hit_rate_per_turn"]) - float(
            sagg["answer_cap_hit_rate_per_turn"]
        )
        sentinel_success = float(cagg["per_family"][SENTINEL]["success"])
        checks = {
            "success_vs_start": contrasts["start"]["delta"] >= float(gates["success_delta_vs_start_min"]),
            "success_vs_control": contrasts["control"]["delta"] >= float(gates["success_delta_vs_control_min"]),
            "success_vs_sample_more": contrasts["sample_more"]["delta"] >= float(gates["success_delta_vs_sample_more_min"]),
            "bootstrap_vs_start": contrasts["start"]["ci95"][0]
            >= float(gates["paired_bootstrap_lower_bound_min"]),
            "sentinel_absolute": sentinel_success
            >= float(gates["atomic_reservations_success_absolute_min"]),
            "rejected_transition": rejected >= float(gates["rejected_transition_absolute_min"]),
            "failed_transition": failed >= float(gates["failed_transition_absolute_min"]),
            "verified": float(cagg["verified_given_success"])
            >= float(gates["verified_absolute_min"]),
            "commit": float(cagg["commit_given_verified"])
            >= float(gates["commit_absolute_min"]),
            "invalid_action": invalid_delta <= float(gates["invalid_action_delta_vs_start_max"]),
            "answer_cap": cap_delta <= float(gates["answer_cap_hit_delta_vs_start_max"]),
            "fresh_family_nonnegative_count": sum(value >= 0 for value in fresh_family.values())
            >= int(gates["minimum_nonnegative_families"]),
            "fresh_family_max_regression": min(fresh_family.values())
            >= float(gates["maximum_single_family_regression"]),
        }
        result = {
            "schema_version": 1,
            "stage": args.block,
            "task_manifest_sha256": candidate["task_manifest_sha256"],
            "task_content_manifest_sha256": candidate["task_content_manifest_sha256"],
            "success": {
                "candidate": cagg["success"],
                **{name: payload["aggregate"]["success"] for name, payload in controls.items()},
            },
            "contrasts": contrasts,
            "family_deltas_vs_start": family,
            "fresh_family_deltas_vs_start": fresh_family,
            "sentinel": {"family": SENTINEL, "candidate_success": sentinel_success},
            "transition_metrics": {"rejected": rejected, "failed": failed},
            "verified_given_success": cagg["verified_given_success"],
            "commit_given_verified": cagg["commit_given_verified"],
            "invalid_action_delta_vs_start": invalid_delta,
            "answer_cap_hit_delta_vs_start": cap_delta,
            "checks": checks,
            "gate": {"passed": all(checks.values())},
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["gate"]["passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
