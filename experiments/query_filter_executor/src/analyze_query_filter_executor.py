#!/usr/bin/env python3
"""Analyze query-supervised filter executor experiment runs."""

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
    for _, sub in df.groupby("run"):
        rows.append(sub[sub["step"] == sub["step"].max()])
    return pd.concat(rows, ignore_index=True)


def main_joint(final: pd.DataFrame) -> pd.DataFrame:
    sub = final[final["run"] == "main_joint_mod31"].copy()
    if sub.empty:
        raise SystemExit("missing main_joint_mod31")
    return sub


def aggregate_query_types(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["run", "run_label", "step", "length", "k"]
    metric_cols = [
        "query_target_mass",
        "query_target_nll",
        "query_top1_on_support",
        "mean_query_support_size",
        "belief_target_mass",
        "belief_target_nll",
        "belief_top1_on_support",
        "mean_belief_support_size",
    ]
    return df.groupby(group_cols, as_index=False)[metric_cols].mean()


def threshold_summary(main_agg: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    for length, grp in main_agg.groupby("length"):
        before = grp[(grp["k"] >= 0) & (grp["k"] < length)]
        at = grp[grp["k"] >= length].sort_values("k").head(1)
        if at.empty:
            continue
        rows.append(
            {
                "length": int(length),
                "mean_query_support_size": float(at["mean_query_support_size"].iloc[0]),
                "mean_belief_support_size": float(at["mean_belief_support_size"].iloc[0]),
                "best_query_mass_before_k_ge_l": float(before["query_target_mass"].max()) if not before.empty else float("nan"),
                "best_belief_mass_before_k_ge_l": float(before["belief_target_mass"].max()) if not before.empty else float("nan"),
                "first_k_ge_l": int(at["k"].iloc[0]),
                "query_mass_at_first_k_ge_l": float(at["query_target_mass"].iloc[0]),
                "belief_mass_at_first_k_ge_l": float(at["belief_target_mass"].iloc[0]),
                "query_top1_at_first_k_ge_l": float(at["query_top1_on_support"].iloc[0]),
                "belief_top1_at_first_k_ge_l": float(at["belief_top1_on_support"].iloc[0]),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "mod31_threshold_summary.csv", index=False)
    return out


def per_query_threshold(main: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    for (length, query_type), grp in main.groupby(["length", "query_type"]):
        before = grp[(grp["k"] >= 0) & (grp["k"] < length)]
        at = grp[grp["k"] >= length].sort_values("k").head(1)
        if at.empty:
            continue
        rows.append(
            {
                "length": int(length),
                "query_type": str(query_type),
                "best_query_mass_before_k_ge_l": float(before["query_target_mass"].max()) if not before.empty else float("nan"),
                "first_k_ge_l": int(at["k"].iloc[0]),
                "query_mass_at_first_k_ge_l": float(at["query_target_mass"].iloc[0]),
                "belief_mass_at_first_k_ge_l": float(at["belief_target_mass"].iloc[0]),
                "query_top1_at_first_k_ge_l": float(at["query_top1_on_support"].iloc[0]),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "mod31_per_query_threshold_summary.csv", index=False)
    return out


def control_summary(final_agg: pd.DataFrame) -> pd.DataFrame:
    wanted = ["main_joint_mod31", "control_marginal_mod31", "control_static_mod31"]
    rows: List[Dict] = []
    for run, grp in final_agg[final_agg["run"].isin(wanted)].groupby("run"):
        for length, g in grp.groupby("length"):
            if run == "main_joint_mod31":
                pick = g[g["k"] >= length].sort_values("k").head(1)
            elif run == "control_marginal_mod31":
                pick = g.sort_values("query_target_mass", ascending=False).head(1)
            else:
                pick = g.head(1)
            rows.append(
                {
                    "run": RUN_LABELS.get(run, run),
                    "length": int(length),
                    "query_target_mass": float(pick["query_target_mass"].iloc[0]),
                    "query_top1_on_support": float(pick["query_top1_on_support"].iloc[0]),
                    "query_target_nll": float(pick["query_target_nll"].iloc[0]),
                    "belief_target_mass": float(pick["belief_target_mass"].iloc[0]),
                    "belief_top1_on_support": float(pick["belief_top1_on_support"].iloc[0]),
                    "belief_target_nll": float(pick["belief_target_nll"].iloc[0]),
                    "mean_query_support_size": float(pick["mean_query_support_size"].iloc[0]),
                    "mean_belief_support_size": float(pick["mean_belief_support_size"].iloc[0]),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "mod31_control_summary.csv", index=False)
    return out


def plot_heatmap(data: pd.DataFrame, value: str, filename: str, title: str) -> None:
    pivot = data.pivot_table(index="length", columns="k", values=value).sort_index()
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


def plot_curves(data: pd.DataFrame, value: str, filename: str, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    for length, grp in data.groupby("length"):
        grp = grp[grp["k"] >= 0].sort_values("k")
        ax.plot(grp["k"], grp[value] * 100, marker="o", linewidth=2, label=f"L={length}")
    ax.set_title(title)
    ax.set_xlabel("internal recurrent steps K")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted(data["k"].unique()))
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / filename, dpi=180)
    plt.close(fig)


def plot_controls(summary: pd.DataFrame, value: str, filename: str, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    colors = {
        "Joint recurrent p=31": "#2ca02c",
        "Marginal recurrent p=31": "#d62728",
        "Static p=31": "#9467bd",
    }
    for run, grp in summary.groupby("run"):
        grp = grp.sort_values("length")
        ax.plot(grp["length"], grp[value] * 100, marker="o", linewidth=2.5, label=run, color=colors.get(run))
    ax.set_title(title)
    ax.set_xlabel("program length L")
    ax.set_ylabel(ylabel)
    ax.set_xticks([4, 8, 12, 16, 24])
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / filename, dpi=180)
    plt.close(fig)


def write_summary(threshold: pd.DataFrame, per_query: pd.DataFrame, controls: pd.DataFrame, final: pd.DataFrame) -> None:
    summary_runs = [
        "main_joint_mod31",
        "control_marginal_mod31",
        "control_static_mod31",
        "pilot_joint_mod11",
        "control_marginal_mod11",
        "control_static_mod11",
    ]
    final_summary = final[final["run"].isin(summary_runs)].copy()
    cols = [
        "run_label",
        "step",
        "length",
        "query_type",
        "k",
        "query_target_mass",
        "query_top1_on_support",
        "belief_target_mass",
        "belief_top1_on_support",
    ]
    with (OUT / "summary.md").open("w", encoding="utf-8") as f:
        f.write("# Query Filter Executor Analysis Summary\n\n")
        f.write("## Main Modulus-31 Threshold Summary\n\n")
        f.write(threshold.to_markdown(index=False))
        f.write("\n\n## Main Modulus-31 Per-Query Threshold Summary\n\n")
        f.write(per_query.to_markdown(index=False))
        f.write("\n\n## Modulus-31 Control Summary\n\n")
        f.write(controls.to_markdown(index=False))
        f.write("\n\n## Final Metrics By Run\n\n")
        f.write(final_summary[cols].sort_values(["run_label", "length", "query_type", "k"]).to_markdown(index=False))
        f.write("\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    df = load_metrics()
    df.to_csv(OUT / "all_metrics_long.csv", index=False)
    final = final_metrics(df)
    final.to_csv(OUT / "final_metrics_long.csv", index=False)
    final_agg = aggregate_query_types(final)
    final_agg.to_csv(OUT / "final_metrics_query_mean.csv", index=False)
    main_df = main_joint(final)
    main_agg = aggregate_query_types(main_df)
    threshold = threshold_summary(main_agg)
    per_query = per_query_threshold(main_df)
    controls = control_summary(final_agg)
    plot_heatmap(main_agg, "query_target_mass", "mod31_query_mass_heatmap.png", "Modulo-31 joint query filter: query target mass")
    plot_heatmap(main_agg, "belief_target_mass", "mod31_belief_mass_heatmap.png", "Modulo-31 joint query filter: hidden belief target mass")
    plot_heatmap(main_agg, "query_top1_on_support", "mod31_query_top1_heatmap.png", "Modulo-31 joint query filter: query top-1 in support")
    plot_curves(
        main_agg,
        "query_target_mass",
        "mod31_query_mass_k_curves.png",
        "Query target mass rises when K reaches program length",
        "query target mass (%)",
    )
    plot_curves(
        main_agg,
        "belief_target_mass",
        "mod31_belief_mass_k_curves.png",
        "Hidden belief mass rises when K reaches program length",
        "hidden belief target mass (%)",
    )
    plot_controls(
        controls,
        "query_target_mass",
        "mod31_controls_query_mass.png",
        "Query-supervised executor versus controls",
        "query target mass (%)",
    )
    plot_controls(
        controls,
        "belief_target_mass",
        "mod31_controls_belief_mass.png",
        "Hidden belief audit versus controls",
        "hidden belief target mass (%)",
    )
    write_summary(threshold, per_query, controls, final)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
