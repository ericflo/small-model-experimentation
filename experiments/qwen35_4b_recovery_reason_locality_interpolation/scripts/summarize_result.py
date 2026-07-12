#!/usr/bin/env python3
"""Build the compact committed receipt for the stopped interpolation run."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
ARTIFACTS = ROOT / "large_artifacts" / "qwen35_4b_recovery_reason_locality_interpolation"
PARENT = ROOT / "large_artifacts" / "qwen35_4b_verifier_conditioned_recovery_bank"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compact_eval(payload: dict) -> dict:
    aggregate = payload["aggregate"]
    return {
        "success": aggregate["success"],
        "failed_test_success": aggregate["per_scenario"]["failed_test"]["success"],
        "rejected_patch_success": aggregate["per_scenario"]["rejected_patch"]["success"],
        "invalid_action_rate_per_turn": aggregate["invalid_action_rate_per_turn"],
        "rejected_patch_immediate_transition": aggregate["per_scenario"]["rejected_patch"][
            "immediate_transition_rate"
        ],
        "failed_test_changed_patch_within_two": aggregate["per_scenario"]["failed_test"][
            "changed_patch_within_two"
        ],
        "submit_rate": aggregate["submit_rate"],
        "verified_given_success": aggregate["verified_given_success"],
        "commit_given_verified": aggregate["commit_given_verified"],
        "mean_sampled_tokens": aggregate["mean_sampled_tokens"],
        "mean_turns": aggregate["mean_turns"],
    }


def truncation_diagnostic(payload: dict) -> dict:
    steps = [step for trajectory in payload["trajectories"] for step in trajectory["steps"]]
    invalid = [step for step in steps if step["operator"] == "INVALID"]
    invalid_trajectories = [
        trajectory for trajectory in payload["trajectories"] if trajectory["invalid_actions"] > 0
    ]
    rejected = [
        trajectory for trajectory in payload["trajectories"]
        if trajectory["scenario"] == "rejected_patch"
    ]
    changed_within_two = []
    first_two_patterns = Counter()
    for trajectory in rejected:
        first_two = trajectory["steps"][:2]
        first_two_patterns["->".join(step["operator"] for step in first_two)] += 1
        changed_within_two.append(any(
            step["operator"] == "PATCH" and step["before_digest"] != step["after_digest"]
            for step in first_two
        ))
    return {
        "invalid_actions": len(invalid),
        "invalid_trajectories": len(invalid_trajectories),
        "invalid_trajectory_success": (
            sum(row["workspace_success"] for row in invalid_trajectories)
            / len(invalid_trajectories)
            if invalid_trajectories else None
        ),
        "all_invalid_answer_tokens_equal_cap": all(
            step["n_answer_tokens"] == 256 for step in invalid
        ),
        "all_invalid_thinking_closed": all(step["thinking_closed"] for step in invalid),
        "forced_close_fraction_invalid": (
            sum(step["forced_close"] for step in invalid) / len(invalid) if invalid else None
        ),
        "invalid_parse_statuses": dict(Counter(step["parse_status"] for step in invalid)),
        "rejected_patch_changed_within_two": sum(changed_within_two) / len(changed_within_two),
        "rejected_first_two_operator_patterns": dict(sorted(first_two_patterns.items())),
    }


def main() -> int:
    names = ("reason_mix_010", "reason_mix_018", "reason_mix_024", "reason_mix_030")
    locality_path = ARTIFACTS / "eval" / "locality_screen.json"
    selection_path = EXP / "analysis" / "candidate_selection.json"
    locality = load(locality_path)
    selection = load(selection_path)
    calibration_paths = {
        "base": PARENT / "eval" / "calibration_recovery_base_deep.json",
        "happy_action": PARENT / "eval" / "calibration_recovery_happy_action_deep.json",
        "action_anchor": PARENT / "eval" / "calibration_recovery_recovery_action_deep.json",
        **{
            name: ARTIFACTS / "eval" / f"calibration_recovery_{name}_deep.json"
            for name in names
        },
    }
    calibrations = {name: load(path) for name, path in calibration_paths.items()}
    merge_paths = {
        name: ARTIFACTS / "merged" / name / "interpolation_receipt.json" for name in names
    }
    merges = {name: load(path) for name, path in merge_paths.items()}
    focus = calibrations["reason_mix_018"]
    result = {
        "schema_version": 1,
        "experiment": EXP.name,
        "verdict": "LOCAL_BUT_NO_BEHAVIOR",
        "stopped_at": "frozen_calibration_policy_gate",
        "transfer_exposed": False,
        "menagerie_exposed": False,
        "headline": {
            "best_success_arm": "reason_mix_018",
            "best_success": focus["aggregate"]["success"],
            "best_success_locality_drift": locality["candidates"]["reason_mix_018"][
                "median_non_target_centered_logit_drift"
            ],
            "eligible_candidates": selection["eligible_candidates"],
        },
        "locality": {
            name: {
                key: locality["candidates"][name][key]
                for key in (
                    "median_non_target_centered_logit_drift",
                    "mean_entropy_delta",
                    "mean_varentropy_delta",
                    "gate",
                )
            }
            for name in ("action_anchor", *names, "reason_endpoint")
        },
        "calibration": {
            name: compact_eval(payload) for name, payload in calibrations.items()
        },
        "merge": {
            name: {
                "reason_lambda": receipt["reason_lambda"],
                "mixed_delta_frobenius_norm_sum": receipt[
                    "delta_frobenius_norm_sums"
                ]["mixed"],
                "weight_sha256": receipt["weight_files"][0]["sha256"],
            }
            for name, receipt in merges.items()
        },
        "selection": selection,
        "exploratory_failure_forensics": truncation_diagnostic(focus),
        "source_sha256": {
            "locality_screen": sha256(locality_path),
            "candidate_selection": sha256(selection_path),
            "calibration": {
                name: sha256(path) for name, path in calibration_paths.items()
            },
            "merge_receipts": {name: sha256(path) for name, path in merge_paths.items()},
        },
    }
    output = EXP / "reports" / "result_receipt.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "verdict": result["verdict"],
        "headline": result["headline"],
        "forensics": result["exploratory_failure_forensics"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
