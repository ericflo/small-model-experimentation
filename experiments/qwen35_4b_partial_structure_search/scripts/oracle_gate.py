#!/usr/bin/env python3
"""CPU-only gate: is semantic type-prefix viability a useful compressed search state?"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))
sys.path.insert(0, str(EXP / "scripts"))
import families as F  # noqa: E402
import experiment_common as C  # noqa: E402


EVALUATED_SPLITS = ("calibration", "development")


def _oracle_beam(
    task: dict[str, Any], oracle: dict[str, Any], beam_width: int
) -> dict[str, Any]:
    depth = int(task["depth"])
    successes = C.success_rows(oracle)
    live, fills = C.live_prefix_maps(successes, depth)
    frontier: list[tuple[str, ...]] = [()]
    expanded = 0
    per_layer = []
    for layer in range(1, depth + 1):
        candidates = [prefix + (operation,) for prefix in frontier for operation in F.TYPES]
        expanded += len(candidates)
        candidates.sort(key=lambda prefix: (-live.get(prefix, 0), -fills.get(prefix, 0), prefix))
        frontier = candidates[:beam_width]
        per_layer.append(
            {
                "layer": layer,
                "expanded": len(candidates),
                "retained": len(frontier),
                "retained_live": sum(live.get(prefix, 0) > 0 for prefix in frontier),
            }
        )
    candidate_pipelines = []
    fill_receipts = []
    for skeleton in frontier:
        pipelines, receipt = C.fills_passing(
            skeleton, task, splits=("visible", "label_probe")
        )
        candidate_pipelines.extend(pipelines)
        fill_receipts.append(receipt)
    chosen, selector = C.consensus_select(candidate_pipelines, task)
    hidden_each = [C.hidden_grade(pipeline, task) for pipeline in candidate_pipelines]
    return {
        "task_id": task["task_id"],
        "depth": depth,
        "beam_width": beam_width,
        "expanded_prefix_nodes": expanded,
        "completed_skeletons": len(frontier),
        "full_completed_skeletons": len(F.TYPES) ** depth,
        "completed_skeleton_compression": (len(F.TYPES) ** depth) / max(1, len(frontier)),
        "path_coverage_hidden": any(hidden_each),
        "selected_hidden_success": C.hidden_grade(chosen, task),
        "visible_label_candidate_count": len(candidate_pipelines),
        "selector": selector,
        "fill_accounting": C.aggregate_counts(fill_receipts),
        "per_layer": per_layer,
    }


def _evaluate_split(
    split: str,
    tasks: list[dict[str, Any]],
    oracles: list[dict[str, Any]],
    beam_width: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    task_ids = [str(task["task_id"]) for task in tasks]
    oracle_ids = [str(row["task_id"]) for row in oracles]
    if task_ids != oracle_ids:
        raise RuntimeError(f"{split} task/oracle IDs are missing, duplicated, or out of order")
    oracle_by_id = {str(row["task_id"]): row for row in oracles}
    rows = []
    density: dict[int, list[float]] = defaultdict(list)
    for task in tasks:
        task_id = str(task["task_id"])
        oracle = oracle_by_id[task_id]
        row = _oracle_beam(task, oracle, beam_width)
        row["split"] = split
        rows.append(row)
        live, _ = C.live_prefix_maps(C.success_rows(oracle), int(task["depth"]))
        for length in range(1, int(task["depth"]) + 1):
            live_count = sum(len(prefix) == length for prefix in live)
            density[length].append(live_count / (len(F.TYPES) ** length))
    if not rows:
        raise RuntimeError(f"{split} oracle gate input is empty")
    summary = {
        "n": len(rows),
        "path_rate": sum(row["path_coverage_hidden"] for row in rows) / len(rows),
        "selected_rate": sum(row["selected_hidden_success"] for row in rows) / len(rows),
        "mean_compression": sum(row["completed_skeleton_compression"] for row in rows)
        / len(rows),
        "mean_oracle_wall_seconds": sum(float(row["wall_seconds"]) for row in oracles)
        / len(oracles),
        "live_density_by_length": {
            str(length): sum(values) / len(values) for length, values in density.items()
        },
    }
    return rows, summary


def build_result(
    calibration_tasks: list[dict[str, Any]],
    calibration_oracles: list[dict[str, Any]],
    development_tasks: list[dict[str, Any]],
    development_oracles: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Build the gate without accepting or loading any primary artifacts."""

    beam = int(config["judge"]["beam_width"])
    split_inputs = {
        "calibration": (calibration_tasks, calibration_oracles),
        "development": (development_tasks, development_oracles),
    }
    all_rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    for split in EVALUATED_SPLITS:
        rows, summary = _evaluate_split(split, *split_inputs[split], beam)
        all_rows.extend(rows)
        summaries[split] = summary
    gate_basis = summaries["development"]
    gate_cfg = config["oracle_gate"]
    passed = bool(
        gate_basis["path_rate"] >= float(gate_cfg["min_hidden_path_rate"])
        and gate_basis["mean_compression"]
        >= float(gate_cfg["min_completed_skeleton_compression"])
    )
    return {
        "schema_version": 2,
        "gate": "oracle_state_usefulness",
        "gate_basis_split": "development",
        "evaluated_splits": list(EVALUATED_SPLITS),
        "primary_artifacts_loaded": False,
        "primary_hidden_used_for_gate": False,
        "development_hidden_used_for_gate": True,
        "passed": passed,
        "thresholds": gate_cfg,
        "splits": summaries,
        "rows": all_rows,
        "hidden_used_for_labels": False,
    }


def gate_source_paths(suffix: str) -> dict[str, Path]:
    return {
        f"{split}_{kind}": EXP / "data" / f"{split}_{kind}{suffix}.jsonl"
        for split in EVALUATED_SPLITS
        for kind in ("tasks", "oracle")
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_gate_data(suffix: str) -> dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    """Load only calibration and dedicated oracle-development artifacts."""

    paths = gate_source_paths(suffix)
    return {
        split: (
            C.load_jsonl(paths[f"{split}_tasks"]),
            C.load_jsonl(paths[f"{split}_oracle"]),
        )
        for split in EVALUATED_SPLITS
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = C.load_config()
    suffix = "_smoke" if args.smoke else ""
    loaded = load_gate_data(suffix)
    result = build_result(
        *loaded["calibration"],
        *loaded["development"],
        cfg,
    )
    result["source_artifacts"] = {
        str(path.relative_to(EXP)): {"sha256": _sha256_file(path)}
        for path in gate_source_paths(suffix).values()
    }
    out = EXP / "runs" / f"oracle_gate{suffix}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "gate_basis_split": result["gate_basis_split"],
                "splits": result["splits"],
                "primary_artifacts_loaded": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if result["passed"] or args.smoke else 1


if __name__ == "__main__":
    raise SystemExit(main())
