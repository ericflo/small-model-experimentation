#!/usr/bin/env python3
"""Outcome-blind power receipt from the terminal parent task-level effect."""

from __future__ import annotations

import json
import hashlib
import math
import statistics
from pathlib import Path
from statistics import NormalDist

import yaml


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
PARENT = ROOT / "experiments" / "qwen35_4b_commit_slot_jacobian_value_transport"


def main() -> int:
    config = yaml.safe_load((EXP / "configs" / "default.yaml").read_text(encoding="utf-8"))
    parent = json.loads(
        (PARENT / "analysis" / "selection_diagnostics.json").read_text(encoding="utf-8")
    )
    parent_path = PARENT / "analysis" / "selection_diagnostics.json"
    if parent["terminal_decision"] != "COMMIT_SLOT_SEAM_FAIL":
        raise RuntimeError("parent decision changed")
    differences = [float(row["real_minus_shuffled"]) for row in parent["task_rows"]]
    effect = statistics.mean(differences)
    standard_deviation = statistics.stdev(differences)
    alpha = 0.05
    target_power = 0.80
    z_alpha = NormalDist().inv_cdf(1.0 - alpha)
    z_power = NormalDist().inv_cdf(target_power)
    required_tasks = math.ceil(
        ((z_alpha + z_power) * standard_deviation / effect) ** 2
    )
    planned_selection = int(config["data"]["seam_selection_tasks"])
    planned_confirmation = int(config["data"]["seam_confirmation_tasks"])
    planned_power = NormalDist().cdf(
        effect * math.sqrt(planned_selection) / standard_deviation - z_alpha
    )
    expected_lower = effect - z_alpha * standard_deviation / math.sqrt(planned_selection)
    result = {
        "schema_version": 1,
        "scientific_result": False,
        "method": "one-sided normal approximation using parent task-level differences",
        "actual_gate_uses_nonparametric_task_bootstrap": True,
        "parent_experiment": "qwen35_4b_commit_slot_jacobian_value_transport",
        "parent_analysis_sha256": hashlib.sha256(parent_path.read_bytes()).hexdigest(),
        "parent_terminal_decision": parent["terminal_decision"],
        "parent_task_units": len(differences),
        "parent_real_minus_shuffled_mean": effect,
        "parent_task_standard_deviation": standard_deviation,
        "one_sided_alpha": alpha,
        "target_power": target_power,
        "required_tasks_per_seam_stage": required_tasks,
        "planned_selection_tasks": planned_selection,
        "planned_confirmation_tasks": planned_confirmation,
        "planned_traces_per_seam_stage": (
            planned_selection * int(config["generation"]["traces_per_task"])
        ),
        "planned_approximate_power": planned_power,
        "expected_normal_approximation_lower_bound": expected_lower,
        "passed": bool(
            planned_selection >= required_tasks
            and planned_confirmation >= required_tasks
            and planned_power >= target_power
        ),
        "scope": (
            "powered for the parent's observed real-minus-shuffled task effect; "
            "not a promise that the effect replicates and not power for no-thought"
        ),
    }
    if not result["passed"]:
        raise RuntimeError(f"planned seam stages are underpowered: {result}")
    output = EXP / "runs" / "smoke" / "power_receipt.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
