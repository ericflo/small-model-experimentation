#!/usr/bin/env python3
"""Analyze Qwen register-token latent compiler runs."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path("experiments/qwen_register_token_latent_compiler")
RUNS = ROOT / "runs"
ANALYSIS = ROOT / "analysis"
FIGURES = ANALYSIS / "figures"


def as_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return math.nan


def pct(value: Any) -> str:
    x = as_float(value)
    if math.isnan(x):
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
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def choose_primary(rows: List[Dict[str, Any]]) -> str:
    preferred = "main_register_trace_s600"
    if any(row.get("run") == preferred for row in rows):
        return preferred
    mains = sorted({str(row.get("run", "")) for row in rows if str(row.get("run", "")).startswith("main")})
    if mains:
        return mains[-1]
    runs = sorted({str(row.get("run", "")) for row in rows if row.get("run")})
    return runs[-1] if runs else ""


def sort_key(row: Dict[str, Any]) -> tuple:
    split = str(row.get("split", ""))
    length = 0
    if "len" in split:
        try:
            length = int(split.rsplit("len", 1)[1])
        except Exception:
            length = 0
    return (str(row.get("variant", "")), split.startswith("paired"), split.startswith("paraphrase"), length)


def write_summary(rows: List[Dict[str, Any]], primary_run: str) -> None:
    primary = [row for row in rows if row.get("run") == primary_run]
    lines = [
        "# Qwen Register-Token Latent Compiler Analysis Summary",
        "",
        f"Primary run: `{primary_run}`" if primary_run else "Primary run: n/a",
        "",
        "## Final Metrics",
        "",
        "| variant | split | direct | executor | mass | init | op | arg | program | prefix | pair both | pair state consistency |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(primary, key=sort_key):
        lines.append(
            "| {variant} | {split} | {direct} | {exec} | {mass} | {init} | {op} | {arg} | {program} | {prefix} | {both} | {state} |".format(
                variant=row.get("variant", ""),
                split=row.get("split", ""),
                direct=pct(row.get("direct_accuracy")),
                exec=pct(row.get("executor_accuracy")),
                mass=pct(row.get("executor_target_mass")),
                init=pct(row.get("init_accuracy")),
                op=pct(row.get("op_accuracy")),
                arg=pct(row.get("arg_accuracy")),
                program=pct(row.get("program_exact")),
                prefix=pct(row.get("state_prefix_fraction")),
                both=pct(row.get("executor_pair_both_correct")),
                state=pct(row.get("compiler_pair_state_consistency")),
            )
        )
    if not primary:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
    lines.extend(["", "## All Runs", ""])
    lines.append("| run | variant | split | direct | executor | program | pair both |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for row in sorted(rows, key=lambda r: (str(r.get("run", "")), sort_key(r))):
        if "len24" not in str(row.get("split", "")) and not str(row.get("split", "")).startswith("fresh"):
            continue
        lines.append(
            "| {run} | {variant} | {split} | {direct} | {exec} | {program} | {both} |".format(
                run=row.get("run", ""),
                variant=row.get("variant", ""),
                split=row.get("split", ""),
                direct=pct(row.get("direct_accuracy")),
                exec=pct(row.get("executor_accuracy")),
                program=pct(row.get("program_exact")),
                both=pct(row.get("executor_pair_both_correct")),
            )
        )
    (ANALYSIS / "summary.md").write_text("\n".join(lines) + "\n")


def split_length(split: str) -> int:
    if "len" not in split:
        return 0
    try:
        return int(split.rsplit("len", 1)[1])
    except Exception:
        return 0


def write_figures(rows: List[Dict[str, Any]], primary_run: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[figures] matplotlib unavailable: {exc}")
        return
    FIGURES.mkdir(parents=True, exist_ok=True)
    primary = [row for row in rows if row.get("run") == primary_run]
    if not primary:
        return
    for metric in ["direct_accuracy", "executor_accuracy", "program_exact", "state_prefix_fraction"]:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for variant in sorted({str(row.get("variant", "")) for row in primary}):
            for prefix, marker in [("standard", "o"), ("paraphrase", "s"), ("paired", "^")]:
                subset = [
                    row for row in primary
                    if row.get("variant") == variant and str(row.get("split", "")).startswith(prefix)
                ]
                xs: List[int] = []
                ys: List[float] = []
                for row in sorted(subset, key=lambda r: split_length(str(r.get("split", "")))):
                    val = as_float(row.get(metric))
                    if not math.isnan(val):
                        xs.append(split_length(str(row.get("split", ""))))
                        ys.append(100.0 * val)
                if xs:
                    ax.plot(xs, ys, marker=marker, label=f"{variant} {prefix}")
        ax.set_xlabel("Evaluation length")
        ax.set_ylabel(metric.replace("_", " ") + " (%)")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, fontsize=7)
        fig.tight_layout()
        fig.savefig(FIGURES / f"{metric}.png", dpi=160)
        plt.close(fig)


def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = load_runs()
    primary = choose_primary(rows)
    write_csv(ANALYSIS / "all_final_metrics.csv", rows)
    write_csv(ANALYSIS / "final_metrics.csv", [row for row in rows if row.get("run") == primary])
    write_summary(rows, primary)
    write_figures(rows, primary)
    print(ANALYSIS / "summary.md")


if __name__ == "__main__":
    main()
