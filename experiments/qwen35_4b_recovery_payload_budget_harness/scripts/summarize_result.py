#!/usr/bin/env python3
"""Build the compact committed receipt for the payload-harness run."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
ARTIFACTS = ROOT / "large_artifacts" / EXP.name


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compact_eval(payload: dict) -> dict:
    aggregate = payload["aggregate"]
    scenarios = aggregate["per_scenario"]
    return {
        "success": aggregate["success"],
        "submit_rate": aggregate["submit_rate"],
        "verified_given_success": aggregate["verified_given_success"],
        "commit_given_verified": aggregate["commit_given_verified"],
        "invalid_action_rate_per_turn": aggregate["invalid_action_rate_per_turn"],
        "answer_cap_hit_rate_per_turn": aggregate["answer_cap_hit_rate_per_turn"],
        "mean_sampled_tokens": aggregate["mean_sampled_tokens"],
        "mean_turns": aggregate["mean_turns"],
        "failed_test_success": scenarios.get("failed_test", {}).get("success"),
        "failed_test_changed_patch_within_two": scenarios.get("failed_test", {}).get(
            "changed_patch_within_two"
        ),
        "rejected_patch_success": scenarios.get("rejected_patch", {}).get("success"),
        "rejected_patch_valid_changed_within_two": scenarios.get(
            "rejected_patch", {}
        ).get("valid_changed_patch_within_two"),
    }


def paired_forensics(candidate: dict, action: dict) -> dict:
    candidate_by_id = {row["case_id"]: row for row in candidate["trajectories"]}
    action_by_id = {row["case_id"]: row for row in action["trajectories"]}
    if candidate_by_id.keys() != action_by_id.keys():
        raise ValueError("candidate/action case IDs differ")

    wins: Counter[str] = Counter()
    union_success = 0
    both_success = 0
    for case_id, candidate_row in candidate_by_id.items():
        action_row = action_by_id[case_id]
        candidate_success = bool(candidate_row["success"])
        action_success = bool(action_row["success"])
        union_success += candidate_success or action_success
        both_success += candidate_success and action_success
        if candidate_success == action_success:
            continue
        winner = "candidate" if candidate_success else "action"
        wins[f"{winner}|{candidate_row['family']}|{candidate_row['scenario']}"] += 1

    n = len(candidate_by_id)
    return {
        "n": n,
        "candidate_only": sum(value for key, value in wins.items() if key.startswith("candidate|")),
        "action_only": sum(value for key, value in wins.items() if key.startswith("action|")),
        "both_success": both_success,
        "both_fail": n - union_success,
        "union_success": union_success / n,
        "winner_cells": dict(sorted(wins.items())),
    }


def main() -> int:
    eval_dir = ARTIFACTS / "eval"
    stages = ("calibration", "transfer_dev", "transfer_confirm")
    recovery_arms = {
        "calibration": ("base_deep", "happy_deep", "action_deep", "candidate_deep"),
        "transfer_dev": (
            "base_deep",
            "happy_deep",
            "action_deep",
            "base_sample_more",
            "base_deep_scaffold",
            "candidate_deep",
        ),
        "transfer_confirm": (
            "base_deep",
            "happy_deep",
            "action_deep",
            "base_sample_more",
            "base_deep_scaffold",
            "candidate_deep",
        ),
    }
    eval_paths: dict[str, Path] = {}
    evaluations: dict[str, dict] = {}
    for stage in stages:
        for arm in recovery_arms[stage]:
            path = eval_dir / f"{stage}_recovery_{arm}.json"
            key = f"{stage}/recovery/{arm}"
            eval_paths[key] = path
            evaluations[key] = load(path)
    for stage in ("transfer_dev", "transfer_confirm"):
        for arm in ("base_deep", "candidate_deep"):
            path = eval_dir / f"{stage}_normal_{arm}.json"
            key = f"{stage}/normal/{arm}"
            eval_paths[key] = path
            evaluations[key] = load(path)

    analysis_paths = {
        path.stem: path for path in sorted((EXP / "analysis").glob("*.json"))
    }
    analyses = {name: load(path) for name, path in analysis_paths.items()}
    locality_path = eval_dir / "locality_candidate.json"
    locality = load(locality_path)

    paired = {}
    for stage in ("transfer_dev", "transfer_confirm"):
        paired[stage] = paired_forensics(
            evaluations[f"{stage}/recovery/candidate_deep"],
            evaluations[f"{stage}/recovery/action_deep"],
        )

    result = {
        "schema_version": 1,
        "experiment": EXP.name,
        "verdict": "TRANSFER_CONFIRM_FAIL",
        "stopped_at": "transfer_confirm",
        "transfer_dev_passed": analyses["transfer_dev_gate"]["gate"]["passed"],
        "transfer_confirm_passed": analyses["transfer_confirm_gate"]["gate"]["passed"],
        "menagerie_exposed": False,
        "headline": {
            "failed_check": "recovery_vs_action",
            "candidate_success": evaluations[
                "transfer_confirm/recovery/candidate_deep"
            ]["aggregate"]["success"],
            "action_success": evaluations[
                "transfer_confirm/recovery/action_deep"
            ]["aggregate"]["success"],
            "candidate_delta_vs_base": analyses["transfer_confirm_gate"]["contrasts"][
                "base"
            ]["delta"],
            "candidate_delta_vs_sample_more": analyses["transfer_confirm_gate"][
                "contrasts"
            ]["sample_more"]["delta"],
            "candidate_action_union_success": paired["transfer_confirm"]["union_success"],
        },
        "locality": locality,
        "evaluations": {
            key: compact_eval(payload) for key, payload in evaluations.items()
        },
        "gates": analyses,
        "candidate_action_complementarity": paired,
        "source_sha256": {
            "locality": sha256(locality_path),
            "evaluations": {key: sha256(path) for key, path in eval_paths.items()},
            "analysis": {name: sha256(path) for name, path in analysis_paths.items()},
        },
    }
    output = EXP / "reports" / "result_receipt.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "verdict": result["verdict"],
                "headline": result["headline"],
                "paired": paired,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
