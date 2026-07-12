#!/usr/bin/env python3
"""Create the compact committed receipt for the stopped recovery-bank run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def source(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def calibration(payload: dict) -> dict:
    aggregate = payload["aggregate"]
    return {
        "success": aggregate["success"],
        "submit_rate": aggregate["submit_rate"],
        "verified_given_success": aggregate["verified_given_success"],
        "commit_given_verified": aggregate["commit_given_verified"],
        "invalid_action_rate_per_turn": aggregate["invalid_action_rate_per_turn"],
        "mean_sampled_tokens": aggregate["mean_sampled_tokens"],
        "mean_turns": aggregate["mean_turns"],
        "per_scenario": aggregate["per_scenario"],
        "per_family": aggregate["per_family"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifact_root
    harvest_path = root / "harvest" / "trajectories.json"
    bank_path = root / "bank" / "receipt.json"
    selection_path = args.experiment / "analysis" / "candidate_selection.json"
    harvest = load(harvest_path)
    bank = load(bank_path)
    arms = ("happy_action", "recovery_action", "recovery_reason")
    training = {}
    merge = {}
    evaluations = {}
    sources = {
        "harvest": source(harvest_path),
        "bank": source(bank_path),
        "selection": source(selection_path),
    }
    for arm in arms:
        train_path = root / "adapters" / arm / "training_receipt.json"
        merge_path = root / "merged" / arm / "merge_receipt.json"
        eval_path = root / "eval" / f"calibration_recovery_{arm}_deep.json"
        train = load(train_path)
        merged = load(merge_path)
        training[arm] = {
            key: train[key] for key in (
                "optimizer_steps", "training_loss", "wall_seconds", "peak_cuda_bytes",
                "weighted_action_mass", "weighted_plan_mass", "max_encoded_tokens",
            )
        }
        merge[arm] = {
            key: merged[key] for key in (
                "applied_lora_modules", "nonzero_lora_modules",
                "delta_frobenius_norm_sum", "delta_frobenius_norm_max",
            )
        }
        evaluations[arm] = calibration(load(eval_path))
        sources[f"training_{arm}"] = source(train_path)
        sources[f"merge_{arm}"] = source(merge_path)
        sources[f"calibration_{arm}"] = source(eval_path)
    base_eval_path = root / "eval" / "calibration_recovery_base_deep.json"
    evaluations["base"] = calibration(load(base_eval_path))
    sources["calibration_base"] = source(base_eval_path)
    selection = load(selection_path)

    locality_files = {
        "happy_action": root / "eval" / "locality_happy_action_exploratory.json",
        "recovery_action": root / "eval" / "locality_recovery_action_exploratory.json",
        "recovery_reason": root / "eval" / "locality_recovery_reason.json",
    }
    locality = {}
    for arm, path in locality_files.items():
        item = load(path)
        locality[arm] = {
            "median_non_target_centered_logit_drift": item[
                "median_non_target_centered_logit_drift"
            ],
            "mean_entropy_delta": item["mean_entropy_delta"],
            "ceiling": item["ceiling"],
            "passed": item["gate"]["passed"],
            "status": "registered" if arm == "recovery_reason" else "exploratory_control",
        }
        sources[f"locality_{arm}"] = source(path)

    uncertainty = {}
    for arm in ("recovery_action", "recovery_reason"):
        path = root / "eval" / f"transition_uncertainty_{arm}{'_exploratory' if arm == 'recovery_action' else ''}.json"
        item = load(path)
        uncertainty[arm] = {
            transition: {
                seam: {
                    "entropy_before": item["before"][transition][seam]["entropy_nats"],
                    "entropy_after": item["after"][transition][seam]["entropy_nats"],
                    "varentropy_before": item["before"][transition][seam]["varentropy_nats2"],
                    "varentropy_after": item["after"][transition][seam]["varentropy_nats2"],
                    "target_rank_before": item["before"][transition][seam]["target_rank"],
                    "target_rank_after": item["after"][transition][seam]["target_rank"],
                }
                for seam in ("plan", "action")
            }
            for transition in item["before"]
        }
        sources[f"uncertainty_{arm}"] = source(path)

    receipt = {
        "schema_version": 1,
        "experiment": "qwen35_4b_verifier_conditioned_recovery_bank",
        "verdict": "stopped_at_registered_locality_gate",
        "menagerie_exposed": False,
        "transfer_blocks_exposed": False,
        "harvest": {
            "tasks": harvest["summary"]["tasks"],
            "covered_tasks": harvest["summary"]["covered_tasks"],
            "task_coverage": harvest["summary"]["task_coverage"],
            "successful_trajectories": harvest["summary"]["successful_trajectories"],
            "per_family": harvest["summary"]["per_family"],
        },
        "bank": {
            "covered_tasks": bank["covered_tasks"],
            "rows_per_arm": {arm: bank["files"][arm]["rows"] for arm in arms},
            "replay_pass_rate": bank["replay_pass_rate"],
            "gates": bank["gates"],
            "target_operator_action_mass": bank["target_operator_action_mass"],
            "reason_plan_mass_fraction": bank["balance"]["recovery_reason"][
                "plan_mass_fraction"
            ],
        },
        "training": training,
        "merge": merge,
        "calibration": evaluations,
        "selection": {
            "selected_arm": selection["selected_arm"],
            "scores": selection["scores"],
            "success_delta_vs_base": selection["success_delta_vs_base"],
            "success_delta_vs_happy": selection["success_delta_vs_happy"],
            "transition_delta_vs_happy": selection["transition_delta_vs_happy"],
            "gate": selection["gate"],
        },
        "locality": locality,
        "uncertainty": uncertainty,
        "downstream_authorization": "none; create a new experiment for any dose/interpolation follow-up",
        "sources": sources,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(args.out), "verdict": receipt["verdict"],
        "selected_arm": selection["selected_arm"],
        "selected_success": evaluations[selection["selected_arm"]]["success"],
        "selected_locality": locality[selection["selected_arm"]],
        "menagerie_exposed": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
