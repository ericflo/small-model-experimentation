#!/usr/bin/env python3
"""Build the compact, chart-ready result summary from sealed run receipts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


EXP = Path(__file__).resolve().parents[1]


def _read(relative: str) -> dict[str, Any]:
    return json.loads((EXP / relative).read_text(encoding="utf-8"))


def _pct(value: float) -> float:
    return round(100.0 * float(value), 4)


def main() -> int:
    data_audit = _read("runs/data_audit.json")
    oracle = _read("runs/oracle_gate.json")
    calibration = _read("runs/calibration_verdict.json")
    model = _read("runs/calibration_model_receipt.json")
    brute = _read("runs/full_brute.json")
    compaction = _read("analysis/calibration_compaction.json")
    method_order = ["thinking", "nothink", "nextop", "surface", "random"]
    chart = {
        "categories": ["Thinking", "No-think", "Next-op", "Surface", "Random"],
        "macro_within_task_auroc_percent": [
            _pct(calibration["metrics"][method]["macro_auroc"]["estimate"])
            for method in method_order
        ],
        "live_child_recall_at_4_percent": [
            _pct(calibration["metrics"][method]["live_recall_at_beam"]["estimate"])
            for method in method_order
        ],
    }
    result = {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "verdict": {
            "classification": calibration["classification"],
            "recognition_gate_passed": calibration["passed"],
            "primary_search_run": False,
            "banking_run": False,
            "reason": "thinking viability failed every preregistered recognition/actionability check",
        },
        "data_integrity": {
            "passed": data_audit["passed"],
            "task_oracle_pairs_checked": data_audit["task_oracle_pairs_checked"],
            "exact_receipts_recomputed": data_audit["exact_depth_receipts"]["recomputed"],
            "common_behavior_signatures": data_audit["behavioral_disjointness"][
                "unique_signatures"
            ],
            "cross_split_collisions": len(
                data_audit["behavioral_disjointness"]["cross_split_collision_groups"]
            ),
        },
        "oracle_development_gate": oracle["splits"]["development"],
        "full_brute_primary": {
            "task_count": brute["task_count"],
            "path_coverage_rate": brute["path_coverage_rate"],
            "selected_hidden_success_rate": brute["selected_hidden_success_rate"],
            "parallel_wall_seconds": brute["parallel_wall_seconds"],
            "logical_type_skeleton_leaves": brute["logical_accounting"][
                "type_skeleton_leaves"
            ],
            "aggregate_accounting": brute["aggregate_accounting"],
        },
        "recognition_calibration": {
            "n_tasks": calibration["n_tasks"],
            "n_candidates": calibration["n_candidates"],
            "gate_checks": calibration["gate_checks"],
            "methods": {method: calibration["metrics"][method] for method in method_order},
            "thinking_vs_strongest_auc": calibration["thinking_auc_difference"],
            "thinking_vs_strongest_recall": calibration["thinking_recall_difference"],
            "strongest_auc_baseline": calibration["strongest_auc_baseline"],
            "strongest_recall_baseline": calibration["strongest_recall_baseline"],
            "thinking_forced_close_rate": calibration["thinking_forced_close_rate"],
            "task_shuffle_canary": calibration["task_shuffle_canary"],
        },
        "model_resources": {
            "wall_seconds": model["wall_seconds"],
            "scoring_summaries": model["scoring_summaries"],
        },
        "raw_trace_compaction": compaction,
        "chart": chart,
    }
    output = EXP / "analysis" / "summary.json"
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"output": str(output), "classification": calibration["classification"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
