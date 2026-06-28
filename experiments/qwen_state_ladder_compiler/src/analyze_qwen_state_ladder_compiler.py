#!/usr/bin/env python3
"""Analyze Qwen state-ladder compiler runs."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt


ROOT = Path("experiments/qwen_state_ladder_compiler")
RUNS = ROOT / "runs"
ANALYSIS = ROOT / "analysis"
FIGURES = ANALYSIS / "figures"
CHECKPOINT_ROOT = Path("large_artifacts/qwen_state_ladder_compiler/checkpoints")
MAIN_RUNS = {
    "main_qwen3_4b_qlora_state_ladder_curriculum_s900",
    "main_qwen3_4b_qlora_state_ladder_w025_curriculum_s900",
    "control_qwen3_4b_qlora_curriculum_no_state_ladder_s900",
    "control_qwen3_4b_qlora_answer_only_curriculum_s900",
}


def pct(x: float) -> str:
    if x != x:
        return "n/a"
    return f"{100.0 * x:.1f}%"


def load_runs() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(RUNS.glob("*/results.json")):
        data = json.loads(path.read_text())
        for variant, result in data.get("variants", {}).items():
            for split, metrics in result.get("final_metrics", {}).items():
                rows.append({
                    "run": path.parent.name,
                    "variant": variant,
                    "split": split,
                    **{k: v for k, v in metrics.items() if isinstance(v, (int, float, str))},
                })
    return rows


def load_best_logged_checkpoints() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    metric = "paired_len24_executor_accuracy"
    for path in sorted(RUNS.glob("*/results.json")):
        if path.parent.name not in MAIN_RUNS:
            continue
        data = json.loads(path.read_text())
        for variant, result in data.get("variants", {}).items():
            log_rows = result.get("train_log", [])
            candidates = [row for row in log_rows if isinstance(row.get(metric), (int, float)) and row.get(metric) == row.get(metric)]
            if not candidates:
                continue
            best = max(candidates, key=lambda row: row.get(metric, float("-inf")))
            rows.append({
                "run": path.parent.name,
                "variant": variant,
                "selection_metric": metric,
                "step": best.get("step"),
                "stage": best.get("stage"),
                "paired_len24_executor_accuracy": best.get("paired_len24_executor_accuracy"),
                "paired_len24_state_all_exact": best.get("paired_len24_state_all_exact"),
                "paired_len24_state_prefix_fraction": best.get("paired_len24_state_prefix_fraction"),
                "paired_len24_compiler_pair_state_consistency": best.get("paired_len24_compiler_pair_state_consistency"),
                "standard_len24_executor_accuracy": best.get("standard_len24_executor_accuracy"),
                "paraphrase_len24_executor_accuracy": best.get("paraphrase_len24_executor_accuracy"),
            })
    return rows


def split_length(split: str) -> int:
    match = re.search(r"len(\d+)", split)
    if match is None:
        return 0
    return int(match.group(1))


def split_mode(split: str) -> str:
    if "_len" in split:
        return split.rsplit("_len", 1)[0]
    return "default"


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_checkpoint_manifest() -> None:
    rows: List[Dict[str, Any]] = []
    for path in sorted(CHECKPOINT_ROOT.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(CHECKPOINT_ROOT)
        run = rel.parts[0] if len(rel.parts) > 0 else ""
        variant = rel.parts[1] if len(rel.parts) > 1 else ""
        rows.append({
            "run": run,
            "variant": variant,
            "checkpoint": str(path),
            "bytes": path.stat().st_size,
        })
    write_csv(ROOT / "checkpoint_manifest.csv", rows)


def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = load_runs()
    write_csv(ANALYSIS / "all_final_metrics.csv", rows)
    main_rows = [row for row in rows if str(row.get("run")) in MAIN_RUNS]
    summary_rows = main_rows or rows
    write_csv(ANALYSIS / "final_metrics.csv", summary_rows)
    best_rows = load_best_logged_checkpoints()
    write_csv(ANALYSIS / "best_logged_checkpoints.csv", best_rows)
    write_checkpoint_manifest()
    lines = ["# Qwen State-Ladder Compiler Analysis Summary", "", "## Final Metrics", ""]
    if summary_rows:
        cols = [
            "run",
            "variant",
            "split",
            "direct_accuracy",
            "executor_accuracy",
            "executor_target_mass",
            "init_accuracy",
            "init_pos_accuracy",
            "op_accuracy",
            "arg_accuracy",
            "op_pos_accuracy",
            "arg_pos_accuracy",
            "program_exact",
            "state_accuracy",
            "state_all_exact",
            "state_prefix_fraction",
            "executor_pair_answer_consistency",
            "executor_pair_both_correct",
            "compiler_pair_program_consistency",
            "compiler_pair_state_consistency",
            "direct_pair_answer_consistency",
            "direct_pair_both_correct",
        ]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["---"] * len(cols)) + "|")
        for row in summary_rows:
            vals = []
            for col in cols:
                val = row.get(col, "")
                vals.append(pct(val) if isinstance(val, float) else str(val))
            lines.append("| " + " | ".join(vals) + " |")

        by_variant: Dict[str, List[Dict[str, Any]]] = {}
        for row in summary_rows:
            by_variant.setdefault(str(row["variant"]), []).append(row)
        for metric in ["direct_accuracy", "executor_accuracy", "program_exact", "state_all_exact", "state_prefix_fraction"]:
            plt.figure(figsize=(8, 4.5))
            by_line: Dict[str, List[Dict[str, Any]]] = {}
            for variant, vrows in by_variant.items():
                for row in vrows:
                    by_line.setdefault(f"{variant}:{split_mode(str(row['split']))}", []).append(row)
            plotted = False
            for label, vrows in sorted(by_line.items()):
                xs: List[int] = []
                ys: List[float] = []
                for row in sorted(vrows, key=lambda r: split_length(str(r["split"]))):
                    val = row.get(metric)
                    if isinstance(val, float) and val == val:
                        xs.append(split_length(str(row["split"])))
                        ys.append(100.0 * val)
                if xs:
                    plt.plot(xs, ys, marker="o", label=label)
                    plotted = True
            plt.xlabel("Evaluation length")
            plt.ylabel(metric.replace("_", " ") + " (%)")
            plt.ylim(0, 105)
            plt.grid(True, alpha=0.25)
            if plotted:
                plt.legend()
            plt.tight_layout()
            plt.savefig(FIGURES / f"{metric}.png", dpi=160)
            plt.close()
        if best_rows:
            lines.extend(["", "## Best Logged Paired L24 Checkpoints", ""])
            best_cols = [
                "run",
                "variant",
                "step",
                "stage",
                "paired_len24_executor_accuracy",
                "paired_len24_state_prefix_fraction",
                "paired_len24_compiler_pair_state_consistency",
                "standard_len24_executor_accuracy",
                "paraphrase_len24_executor_accuracy",
            ]
            lines.append("| " + " | ".join(best_cols) + " |")
            lines.append("|" + "|".join(["---"] * len(best_cols)) + "|")
            for row in best_rows:
                vals = []
                for col in best_cols:
                    val = row.get(col, "")
                    vals.append(pct(val) if isinstance(val, float) else str(val))
                lines.append("| " + " | ".join(vals) + " |")
    else:
        lines.append("No runs found.")
    (ANALYSIS / "summary.md").write_text("\n".join(lines) + "\n")
    print(ANALYSIS / "summary.md")


if __name__ == "__main__":
    main()
