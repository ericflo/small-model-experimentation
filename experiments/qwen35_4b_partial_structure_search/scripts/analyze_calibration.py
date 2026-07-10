#!/usr/bin/env python3
"""Analyze actionable within-task/sibling calibration and apply the launch gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))
import experiment_common as C  # noqa: E402
import stats as S  # noqa: E402


def _surface_scores(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Leave-one-task-out smoothed structural prior; no task outcomes leak into its own score."""
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task_id"])].append(row)
    result = {}
    for heldout, held_rows in by_task.items():
        counts: dict[tuple[Any, ...], list[int]] = defaultdict(lambda: [0, 0])
        fallback: dict[tuple[Any, ...], list[int]] = defaultdict(lambda: [0, 0])
        for task_id, train_rows in by_task.items():
            if task_id == heldout:
                continue
            for row in train_rows:
                prefix = tuple(row["candidate_prefix"])
                parent_last = prefix[-2] if len(prefix) >= 2 else "ROOT"
                key = (int(row["prefix_len"]), parent_last, row["child_operation"])
                counts[key][0] += int(bool(row["live"]))
                counts[key][1] += 1
                back = (int(row["prefix_len"]), row["child_operation"])
                fallback[back][0] += int(bool(row["live"]))
                fallback[back][1] += 1
        for row in held_rows:
            prefix = tuple(row["candidate_prefix"])
            key = (
                int(row["prefix_len"]),
                prefix[-2] if len(prefix) >= 2 else "ROOT",
                row["child_operation"],
            )
            pos, total = counts.get(key, fallback.get((int(row["prefix_len"]), row["child_operation"]), [0, 0]))
            result[str(row["id"])] = (pos + 1.0) / (total + 2.0)
    return result


def _random_score(record_id: str) -> float:
    return int.from_bytes(hashlib.sha256(record_id.encode()).digest()[:8], "big") / 2**64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _method_rows(suffix: str) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    candidates = C.load_jsonl(EXP / "data" / f"calibration_candidates{suffix}.jsonl")
    scores: dict[str, dict[str, float]] = {}
    for tag in ("thinking", "nothink"):
        rows = C.load_jsonl(EXP / "runs" / f"calibration_{tag}{suffix}.jsonl")
        scores[tag] = {str(row["id"]): float(row["p_viable"]) for row in rows}
    next_rows = C.load_jsonl(EXP / "runs" / f"calibration_nextop{suffix}.jsonl")
    next_by_group = {str(row["id"]): row["choice_probabilities"] for row in next_rows}
    scores["nextop"] = {
        str(row["id"]): float(next_by_group[str(row["parent_group"])][row["child_operation"]])
        for row in candidates
    }
    scores["surface"] = _surface_scores(candidates)
    scores["random"] = {str(row["id"]): _random_score(str(row["id"])) for row in candidates}
    return candidates, scores


def _per_task_metrics(
    rows: list[dict[str, Any]], method_scores: dict[str, float], beam: int
) -> dict[str, dict[str, float]]:
    by_task_depth: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task_depth[(str(row["task_id"]), int(row["prefix_len"]))].append(row)
        groups[str(row["parent_group"])].append(row)
    task_aucs: dict[str, list[float]] = defaultdict(list)
    for (task_id, _depth), cell in by_task_depth.items():
        value = S.auroc(
            [method_scores[str(row["id"])] for row in cell],
            [int(bool(row["live"])) for row in cell],
        )
        if value is not None:
            task_aucs[task_id].append(value)
    task_recall: dict[str, list[float]] = defaultdict(list)
    task_best_rank: dict[str, list[float]] = defaultdict(list)
    for group in groups.values():
        live_n = sum(bool(row["live"]) for row in group)
        if live_n == 0:
            continue
        ranked = sorted(group, key=lambda row: (-method_scores[str(row["id"])], str(row["id"])))
        task_id = str(group[0]["task_id"])
        kept_live = sum(bool(row["live"]) for row in ranked[:beam])
        task_recall[task_id].append(kept_live / live_n)
        task_best_rank[task_id].append(
            float(min(index + 1 for index, row in enumerate(ranked) if row["live"]))
        )
    tasks = sorted({str(row["task_id"]) for row in rows})
    return {
        task: {
            "macro_auroc": sum(task_aucs[task]) / len(task_aucs[task]) if task_aucs[task] else math.nan,
            "live_recall_at_beam": sum(task_recall[task]) / len(task_recall[task]) if task_recall[task] else math.nan,
            "best_live_rank": sum(task_best_rank[task]) / len(task_best_rank[task]) if task_best_rank[task] else math.nan,
        }
        for task in tasks
    }


def _bootstrap_metric(per_task: dict[str, dict[str, float]], key: str, reps: int, seed: int) -> dict[str, Any]:
    values = {task: metrics[key] for task, metrics in per_task.items() if math.isfinite(metrics[key])}
    return S.cluster_bootstrap(values, lambda xs: sum(xs) / len(xs), reps=reps, seed=seed)


def _bootstrap_difference(
    left: dict[str, dict[str, float]],
    right: dict[str, dict[str, float]],
    key: str,
    reps: int,
    seed: int,
) -> dict[str, Any]:
    values = {
        task: left[task][key] - right[task][key]
        for task in sorted(set(left) & set(right))
        if math.isfinite(left[task][key]) and math.isfinite(right[task][key])
    }
    return S.cluster_bootstrap(values, lambda xs: sum(xs) / len(xs), reps=reps, seed=seed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    suffix = "_smoke" if args.smoke else ""
    cfg = C.load_config()
    reps = 500 if args.smoke else int(cfg["calibration"]["bootstrap_reps"])
    beam = int(cfg["judge"]["beam_width"])
    rows, scores = _method_rows(suffix)
    per_task = {method: _per_task_metrics(rows, values, beam) for method, values in scores.items()}
    metrics = {}
    for index, method in enumerate(scores):
        metrics[method] = {
            "macro_auroc": _bootstrap_metric(per_task[method], "macro_auroc", reps, 10 + index),
            "live_recall_at_beam": _bootstrap_metric(
                per_task[method], "live_recall_at_beam", reps, 30 + index
            ),
            "best_live_rank": _bootstrap_metric(per_task[method], "best_live_rank", reps, 50 + index),
            "pooled_auroc_diagnostic": S.auroc(
                [scores[method][str(row["id"])] for row in rows],
                [int(bool(row["live"])) for row in rows],
            ),
        }
    baselines = [method for method in scores if method != "thinking"]
    strongest_auc = max(baselines, key=lambda method: metrics[method]["macro_auroc"]["estimate"])
    strongest_recall = max(
        baselines, key=lambda method: metrics[method]["live_recall_at_beam"]["estimate"]
    )
    auc_diff = _bootstrap_difference(
        per_task["thinking"], per_task[strongest_auc], "macro_auroc", reps, 71
    )
    recall_diff = _bootstrap_difference(
        per_task["thinking"], per_task[strongest_recall], "live_recall_at_beam", reps, 72
    )
    gates = cfg["calibration"]
    think_auc = metrics["thinking"]["macro_auroc"]
    gate_checks = {
        "macro_auroc_point": think_auc["estimate"] >= float(gates["min_macro_auroc"]),
        "macro_auroc_lcb": think_auc["ci_low"] > float(gates["min_macro_auroc_lcb"]),
        "auroc_lift_point": auc_diff["estimate"] >= float(gates["min_auroc_lift"]),
        "auroc_lift_lcb": auc_diff["ci_low"] > 0,
        "recall_lift_point": recall_diff["estimate"]
        >= float(gates["min_recall_at_beam_lift"]),
        "recall_lift_lcb": recall_diff["ci_low"] > 0,
    }
    passed = all(gate_checks.values())
    readable = all(
        gate_checks[key]
        for key in (
            "macro_auroc_point",
            "macro_auroc_lcb",
            "auroc_lift_point",
            "auroc_lift_lcb",
        )
    )
    classification = (
        "recognition_gate_passed"
        if passed
        else "G2_readable_but_nonactionable"
        if readable
        else "G1_unreadable_partial_state"
    )
    think_rows = C.load_jsonl(EXP / "runs" / f"calibration_thinking{suffix}.jsonl")
    shuffled_rows = C.load_jsonl(EXP / "runs" / f"calibration_task_shuffled{suffix}.jsonl")
    canary = None
    if shuffled_rows:
        shuffled_scores = {
            str(row["meta"]["original_id"]): float(row["p_viable"]) for row in shuffled_rows
        }
        subset = [row for row in rows if str(row["id"]) in shuffled_scores]
        original_subset_scores = {
            str(row["id"]): scores["thinking"][str(row["id"])] for row in subset
        }
        original_per_task = _per_task_metrics(subset, original_subset_scores, beam)
        shuffled_per_task = _per_task_metrics(subset, shuffled_scores, beam)
        canary = {
            "n_tasks": len({row["task_id"] for row in subset}),
            "n_candidates": len(subset),
            "original_macro_auroc": _bootstrap_metric(
                original_per_task, "macro_auroc", reps, 81
            ),
            "task_shuffled_macro_auroc": _bootstrap_metric(
                shuffled_per_task, "macro_auroc", reps, 82
            ),
            "original_minus_shuffled": _bootstrap_difference(
                original_per_task, shuffled_per_task, "macro_auroc", reps, 83
            ),
        }
    verdict = {
        "schema_version": 1,
        "gate": "actionable_partial_structure_recognition",
        "passed": passed,
        "classification": classification,
        "gate_checks": gate_checks,
        "smoke": args.smoke,
        "n_candidates": len(rows),
        "n_tasks": len({row["task_id"] for row in rows}),
        "beam_width": beam,
        "metrics": metrics,
        "strongest_auc_baseline": strongest_auc,
        "strongest_recall_baseline": strongest_recall,
        "thinking_auc_difference": auc_diff,
        "thinking_recall_difference": recall_diff,
        "thinking_forced_close_rate": sum(bool(row["forced_close"]) for row in think_rows) / len(think_rows),
        "task_shuffle_canary": canary,
        "thresholds": {
            key: gates[key]
            for key in (
                "min_macro_auroc",
                "min_macro_auroc_lcb",
                "min_auroc_lift",
                "min_recall_at_beam_lift",
            )
        },
        "per_task": per_task,
        "source_artifacts": {
            str(path.relative_to(EXP)): {"sha256": _sha256(path)}
            for path in (
                EXP / "data" / f"calibration_candidates{suffix}.jsonl",
                EXP / "runs" / f"calibration_thinking{suffix}.jsonl",
                EXP / "runs" / f"calibration_nothink{suffix}.jsonl",
                EXP / "runs" / f"calibration_nextop{suffix}.jsonl",
                EXP / "runs" / f"calibration_task_shuffled{suffix}.jsonl",
                EXP / "runs" / f"calibration_model_receipt{suffix}.json",
            )
        },
    }
    out = EXP / "runs" / f"calibration_verdict{suffix}.json"
    out.write_text(json.dumps(verdict, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": passed,
                "thinking_auroc": think_auc,
                "strongest_auc_baseline": strongest_auc,
                "auc_difference": auc_diff,
                "strongest_recall_baseline": strongest_recall,
                "recall_difference": recall_diff,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if passed or args.smoke else 1


if __name__ == "__main__":
    raise SystemExit(main())
