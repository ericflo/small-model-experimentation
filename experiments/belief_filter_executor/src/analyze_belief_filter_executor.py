#!/usr/bin/env python3
"""Analyze belief filter executor experiment runs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
RUN_DIR = EXPERIMENT_DIR / "runs"
OUT = EXPERIMENT_DIR / "analysis"
FIG = OUT / "figures"


RUN_LABELS = {
    "main_joint_mod31": "Joint recurrent p=31",
    "control_marginal_mod31": "Marginal recurrent p=31",
    "control_static_mod31": "Static p=31",
    "pilot_joint_mod11": "Joint recurrent p=11",
    "control_marginal_mod11": "Marginal recurrent p=11",
    "control_static_mod11": "Static p=11",
}


def metric_step(path: Path) -> int:
    match = re.search(r"metrics_step(\d+)\.csv$", path.name)
    if not match:
        raise ValueError(path)
    return int(match.group(1))


def load_metrics() -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for metrics_path in sorted(RUN_DIR.glob("*/metrics_step*.csv")):
        run = metrics_path.parent.name
        df = pd.read_csv(metrics_path)
        df["run"] = run
        df["run_label"] = RUN_LABELS.get(run, run)
        df["step"] = metric_step(metrics_path)
        frames.append(df)
    if not frames:
        raise SystemExit(f"No metrics found in {RUN_DIR}")
    return pd.concat(frames, ignore_index=True)


def final_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for run, sub in df.groupby("run"):
        rows.append(sub[sub["step"] == sub["step"].max()])
    return pd.concat(rows, ignore_index=True)


def main_joint(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["run"] == "main_joint_mod31"].copy()
    if sub.empty:
        raise SystemExit("missing main_joint_mod31")
    return sub[sub["step"] == sub["step"].max()].copy()


def threshold_summary(main: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    for length, grp in main.groupby("length"):
        before = grp[(grp["k"] >= 0) & (grp["k"] < length)]
        at = grp[grp["k"] >= length].sort_values("k").head(1)
        rows.append(
            {
                "length": int(length),
                "mean_support_size": float(at["mean_support_size"].iloc[0]),
                "best_target_mass_before_k_ge_l": float(before["target_mass"].max()) if not before.empty else float("nan"),
                "first_k_ge_l": int(at["k"].iloc[0]),
                "target_mass_at_first_k_ge_l": float(at["target_mass"].iloc[0]),
                "top1_at_first_k_ge_l": float(at["top1_on_support"].iloc[0]),
                "target_nll_at_first_k_ge_l": float(at["target_nll"].iloc[0]),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "mod31_threshold_summary.csv", index=False)
    return out


def plot_heatmap(main: pd.DataFrame, value: str, filename: str, title: str) -> None:
    pivot = main.pivot_table(index="length", columns="k", values=value).sort_index()
    fig, ax = plt.subplots(figsize=(10, 5.4))
    im = ax.imshow(pivot.values * 100, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=100)
    ax.set_title(title)
    ax.set_xlabel("internal recurrent steps K")
    ax.set_ylabel("program length L")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(i) for i in pivot.index])
    for y, length in enumerate(pivot.index):
        for x, k in enumerate(pivot.columns):
            val = pivot.loc[length, k] * 100
            color = "white" if val < 55 else "black"
            ax.text(x, y, f"{val:.0f}", ha="center", va="center", color=color, fontsize=9)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(value.replace("_", " ") + " (%)")
    fig.tight_layout()
    fig.savefig(FIG / filename, dpi=180)
    plt.close(fig)


def plot_curves(main: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    for length, grp in main.groupby("length"):
        grp = grp[grp["k"] >= 0].sort_values("k")
        ax.plot(grp["k"], grp["target_mass"] * 100, marker="o", linewidth=2, label=f"L={length}")
    ax.set_title("Target support mass rises when K reaches program length")
    ax.set_xlabel("internal recurrent steps K")
    ax.set_ylabel("target support mass (%)")
    ax.set_xticks(sorted(main["k"].unique()))
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "mod31_target_mass_k_curves.png", dpi=180)
    plt.close(fig)


def control_summary(final: pd.DataFrame) -> pd.DataFrame:
    wanted = ["main_joint_mod31", "control_marginal_mod31", "control_static_mod31"]
    rows: List[Dict] = []
    for run, grp in final[final["run"].isin(wanted)].groupby("run"):
        for length, g in grp.groupby("length"):
            if run == "main_joint_mod31":
                pick = g[g["k"] >= length].sort_values("k").head(1)
            elif run == "control_marginal_mod31":
                pick = g.sort_values("target_mass", ascending=False).head(1)
            else:
                pick = g.head(1)
            rows.append(
                {
                    "run": RUN_LABELS.get(run, run),
                    "length": int(length),
                    "target_mass": float(pick["target_mass"].iloc[0]),
                    "top1_on_support": float(pick["top1_on_support"].iloc[0]),
                    "target_nll": float(pick["target_nll"].iloc[0]),
                    "mean_support_size": float(pick["mean_support_size"].iloc[0]),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "mod31_control_summary.csv", index=False)
    return out


def plot_controls(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    colors = {
        "Joint recurrent p=31": "#2ca02c",
        "Marginal recurrent p=31": "#d62728",
        "Static p=31": "#9467bd",
    }
    for run, grp in summary.groupby("run"):
        grp = grp.sort_values("length")
        ax.plot(grp["length"], grp["target_mass"] * 100, marker="o", linewidth=2.5, label=run, color=colors.get(run))
    ax.set_title("Joint filter executor versus controls")
    ax.set_xlabel("program length L")
    ax.set_ylabel("target support mass (%)")
    ax.set_xticks([4, 8, 12, 16, 24])
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "mod31_controls_target_mass.png", dpi=180)
    plt.close(fig)


def write_summary(threshold: pd.DataFrame, controls: pd.DataFrame, final: pd.DataFrame) -> None:
    summary_runs = [
        "main_joint_mod31",
        "control_marginal_mod31",
        "control_static_mod31",
        "pilot_joint_mod11",
        "control_marginal_mod11",
        "control_static_mod11",
    ]
    final_summary = final[final["run"].isin(summary_runs)].copy()
    with (OUT / "summary.md").open("w", encoding="utf-8") as f:
        f.write("# Belief Filter Executor Analysis Summary\n\n")
        f.write("## Main Modulus-31 Threshold Summary\n\n")
        f.write(threshold.to_markdown(index=False))
        f.write("\n\n## Modulus-31 Control Summary\n\n")
        f.write(controls.to_markdown(index=False))
        f.write("\n\n## Final Metrics By Run\n\n")
        cols = ["run_label", "step", "length", "k", "target_mass", "top1_on_support", "target_nll", "mean_support_size"]
        f.write(final_summary[cols].sort_values(["run_label", "length", "k"]).to_markdown(index=False))
        f.write("\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    df = load_metrics()
    df.to_csv(OUT / "all_metrics_long.csv", index=False)
    final = final_metrics(df)
    final.to_csv(OUT / "final_metrics_long.csv", index=False)
    main_df = main_joint(df)
    thresh = threshold_summary(main_df)
    controls = control_summary(final)
    plot_heatmap(main_df, "target_mass", "mod31_target_mass_heatmap.png", "Modulo-31 joint filter executor: target support mass")
    plot_heatmap(main_df, "top1_on_support", "mod31_top1_support_heatmap.png", "Modulo-31 joint filter executor: top-1 in target support")
    plot_curves(main_df)
    plot_controls(controls)
    write_summary(thresh, controls, final)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
