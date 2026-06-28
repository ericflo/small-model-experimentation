#!/usr/bin/env python3
"""Analyze the controlled latent executor experiment."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
RUN_ROOT = EXPERIMENT_DIR / "runs"
OUT = EXPERIMENT_DIR / "analysis"
FIG = OUT / "figures"


RUNS = {
    "categorical_mod31": RUN_ROOT / "pilot_categorical_mod31/metrics_step00400.csv",
    "categorical_mod97": RUN_ROOT / "categorical_mod97/metrics_step00250.csv",
    "static_mod31": RUN_ROOT / "static_mod31/metrics_step00400.csv",
    "static_mod97": RUN_ROOT / "static_mod97/metrics_step00150.csv",
    "unstructured_mod31": RUN_ROOT / "pilot_executor_mod31/metrics_step00500.csv",
}


def wilson(acc: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    denom = 1 + z * z / n
    center = (acc + z * z / (2 * n)) / denom
    half = z * math.sqrt(acc * (1 - acc) / n + z * z / (4 * n * n)) / denom
    return max(0, center - half), min(1, center + half)


def load_metrics() -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for run, path in RUNS.items():
        df = pd.read_csv(path)
        df["run"] = run
        if "mod31" in run:
            df["modulus"] = 31
        elif "mod97" in run:
            df["modulus"] = 97
        else:
            df["modulus"] = None
        rows.append(df)
    all_df = pd.concat(rows, ignore_index=True)
    intervals = all_df.apply(lambda r: wilson(float(r.pair_accuracy), int(r.n)), axis=1)
    all_df["pair_ci_low"] = [x[0] for x in intervals]
    all_df["pair_ci_high"] = [x[1] for x in intervals]
    return all_df


def plot_threshold_heatmap(df: pd.DataFrame) -> None:
    sub = df[df["run"] == "categorical_mod97"].copy()
    pivot = sub.pivot_table(index="length", columns="k", values="pair_accuracy").sort_index()
    fig, ax = plt.subplots(figsize=(10, 5.2))
    im = ax.imshow(pivot.values * 100, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=100)
    ax.set_title("Modulo-97 categorical latent executor: exact pair accuracy")
    ax.set_xlabel("internal recurrent steps K")
    ax.set_ylabel("program length L")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(i) for i in pivot.index])
    for y, length in enumerate(pivot.index):
        for x, k in enumerate(pivot.columns):
            val = pivot.loc[length, k] * 100
            color = "white" if val < 70 else "black"
            ax.text(x, y, f"{val:.0f}", ha="center", va="center", color=color, fontsize=9)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("exact pair accuracy (%)")
    fig.tight_layout()
    fig.savefig(FIG / "mod97_k_threshold_heatmap.png", dpi=180)
    plt.close(fig)


def plot_k_curves(df: pd.DataFrame) -> None:
    sub = df[df["run"] == "categorical_mod97"].copy()
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for length, grp in sub.groupby("length"):
        grp = grp.sort_values("k")
        ax.plot(grp["k"], grp["pair_accuracy"] * 100, marker="o", linewidth=2, label=f"L={length}")
    ax.set_title("Accuracy rises when K reaches program length")
    ax.set_xlabel("internal recurrent steps K")
    ax.set_ylabel("exact pair accuracy (%)")
    ax.set_ylim(-3, 103)
    ax.set_xticks(sorted(sub["k"].unique()))
    ax.grid(True, alpha=0.25)
    ax.legend(title="program length", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "mod97_k_curves.png", dpi=180)
    plt.close(fig)


def plot_baselines(df: pd.DataFrame) -> None:
    rows = []
    for run in ["categorical_mod97", "static_mod97", "unstructured_mod31", "static_mod31"]:
        sub = df[df["run"] == run]
        if run in {"categorical_mod97", "unstructured_mod31"}:
            sub = sub.loc[sub.groupby("length")["pair_accuracy"].idxmax()]
        rows.append(sub.assign(display_run=run))
    plot_df = pd.concat(rows, ignore_index=True)
    labels = {
        "categorical_mod97": "Categorical recurrent p=97 (best K)",
        "static_mod97": "Static p=97",
        "unstructured_mod31": "Unstructured recurrent p=31 (best K)",
        "static_mod31": "Static p=31",
    }
    fig, ax = plt.subplots(figsize=(9, 5.2))
    palette = {
        "categorical_mod97": "#2ca02c",
        "static_mod97": "#d62728",
        "unstructured_mod31": "#ff7f0e",
        "static_mod31": "#9467bd",
    }
    for run in labels:
        sub = plot_df[plot_df["display_run"] == run].sort_values("length")
        ax.plot(
            sub["length"],
            sub["pair_accuracy"] * 100,
            marker="o",
            linewidth=2,
            color=palette[run],
            label=labels[run],
        )
    ax.set_ylabel("exact pair accuracy (%)")
    ax.set_xlabel("program length")
    ax.set_title("Recurrent categorical executor versus controls")
    ax.set_ylim(0, 105)
    ax.set_xticks([4, 8, 12, 16, 24])
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="center right")
    fig.tight_layout()
    fig.savefig(FIG / "executor_vs_controls.png", dpi=180)
    plt.close(fig)


def write_summary(df: pd.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    cat97 = df[df["run"] == "categorical_mod97"]
    for length, grp in cat97.groupby("length"):
        pre = grp[grp["k"] < length]["pair_accuracy"].max()
        at = grp[grp["k"] >= length]["pair_accuracy"].iloc[0]
        summary_rows.append(
            {
                "length": int(length),
                "best_before_k_reaches_length": pre,
                "first_accuracy_at_k_ge_length": at,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "threshold_summary_mod97.csv", index=False)
    with open(OUT / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Latent Executor Analysis Summary\n\n")
        f.write("## Modulo-97 K Threshold Summary\n\n")
        f.write(summary.to_markdown(index=False))
        f.write("\n\n## All Metrics\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    df = load_metrics()
    df.to_csv(OUT / "all_metrics_long.csv", index=False)
    plot_threshold_heatmap(df)
    plot_k_curves(df)
    plot_baselines(df)
    write_summary(df)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
