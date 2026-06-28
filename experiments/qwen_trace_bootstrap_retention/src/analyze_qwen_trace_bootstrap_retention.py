#!/usr/bin/env python3
"""Analyze Qwen trace-bootstrap retention runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt


ROOT = Path("experiments/qwen_trace_bootstrap_retention")
RUNS = ROOT / "runs"
ANALYSIS = ROOT / "analysis"
FIGURES = ANALYSIS / "figures"


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


def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = load_runs()
    write_csv(ANALYSIS / "final_metrics.csv", rows)
    lines = ["# Qwen Trace Bootstrap Retention Analysis Summary", "", "## Final Metrics", ""]
    if rows:
        cols = [
            "run",
            "variant",
            "split",
            "direct_accuracy",
            "executor_accuracy",
            "executor_target_mass",
            "init_accuracy",
            "op_accuracy",
            "arg_accuracy",
            "program_exact",
        ]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["---"] * len(cols)) + "|")
        for row in rows:
            vals = []
            for col in cols:
                val = row.get(col, "")
                vals.append(pct(val) if isinstance(val, float) else str(val))
            lines.append("| " + " | ".join(vals) + " |")

        by_variant: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            by_variant.setdefault(str(row["variant"]), []).append(row)
        for metric in ["direct_accuracy", "executor_accuracy", "program_exact"]:
            plt.figure(figsize=(8, 4.5))
            for variant, vrows in sorted(by_variant.items()):
                xs = []
                ys = []
                for row in sorted(vrows, key=lambda r: int(str(r["split"]).replace("len", ""))):
                    val = row.get(metric)
                    if isinstance(val, float) and val == val:
                        xs.append(int(str(row["split"]).replace("len", "")))
                        ys.append(100.0 * val)
                if xs:
                    plt.plot(xs, ys, marker="o", label=variant)
            plt.xlabel("Evaluation length")
            plt.ylabel(metric.replace("_", " ") + " (%)")
            plt.ylim(0, 105)
            plt.grid(True, alpha=0.25)
            plt.legend()
            plt.tight_layout()
            plt.savefig(FIGURES / f"{metric}.png", dpi=160)
            plt.close()
    else:
        lines.append("No runs found.")
    (ANALYSIS / "summary.md").write_text("\n".join(lines) + "\n")
    print(ANALYSIS / "summary.md")


if __name__ == "__main__":
    main()
