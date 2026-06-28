#!/usr/bin/env python3
"""Analyze dense latent query executor experiment runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
RUN_DIR = EXPERIMENT_DIR / "runs"
OUT = EXPERIMENT_DIR / "analysis"
FIG = OUT / "figures"


RUN_LABELS = {
    "main_dense_mod31": "Dense recurrent p=31",
    "control_static_mod31": "Static compiler p=31",
    "control_shuffled_mod31": "Shuffled recurrent p=31",
    "pilot_dense_mod11": "Dense recurrent p=11",
    "control_static_mod11": "Static compiler p=11",
    "control_shuffled_mod11": "Shuffled recurrent p=11",
    "smoke_dense_mod7": "Smoke dense p=7",
}

METRIC_COLS = [
    "query_target_mass",
    "query_target_nll",
    "query_top1_on_support",
    "mean_query_support_size",
    "probe_belief_target_mass",
    "probe_belief_target_nll",
    "probe_belief_top1_on_support",
    "mean_belief_support_size",
]


def train_step_for_run(run_dir: Path) -> int:
    results = run_dir / "results.json"
    if not results.exists():
        return -1
    with results.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return int(data.get("args", {}).get("train_steps", -1))


def load_metrics() -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for metrics_path in sorted(RUN_DIR.glob("*/metrics_final.csv")):
        run = metrics_path.parent.name
        df = pd.read_csv(metrics_path)
        df["run"] = run
        df["run_label"] = RUN_LABELS.get(run, run)
        df["step"] = train_step_for_run(metrics_path.parent)
        frames.append(df)
    if not frames:
        raise SystemExit(f"No metrics found in {RUN_DIR}")
    return pd.concat(frames, ignore_index=True)


def aggregate_query_types(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["run", "run_label", "step", "model", "length", "k"]
    return df.groupby(group_cols, as_index=False)[METRIC_COLS].mean()


def main_dense(final: pd.DataFrame) -> pd.DataFrame:
    sub = final[final["run"] == "main_dense_mod31"].copy()
    if sub.empty:
        raise SystemExit("missing main_dense_mod31")
    return sub


def main_dense_agg(final_agg: pd.DataFrame) -> pd.DataFrame:
    sub = final_agg[final_agg["run"] == "main_dense_mod31"].copy()
    if sub.empty:
        raise SystemExit("missing main_dense_mod31")
    return sub


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
                "best_probe_belief_mass_before_k_ge_l": float(before["probe_belief_target_mass"].max()) if not before.empty else float("nan"),
                "first_k_ge_l": int(at["k"].iloc[0]),
                "query_mass_at_first_k_ge_l": float(at["query_target_mass"].iloc[0]),
                "probe_belief_mass_at_first_k_ge_l": float(at["probe_belief_target_mass"].iloc[0]),
                "query_top1_at_first_k_ge_l": float(at["query_top1_on_support"].iloc[0]),
                "probe_belief_top1_at_first_k_ge_l": float(at["probe_belief_top1_on_support"].iloc[0]),
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
                "best_probe_belief_mass_before_k_ge_l": float(before["probe_belief_target_mass"].max()) if not before.empty else float("nan"),
                "first_k_ge_l": int(at["k"].iloc[0]),
                "query_mass_at_first_k_ge_l": float(at["query_target_mass"].iloc[0]),
                "probe_belief_mass_at_first_k_ge_l": float(at["probe_belief_target_mass"].iloc[0]),
                "query_top1_at_first_k_ge_l": float(at["query_top1_on_support"].iloc[0]),
                "probe_belief_top1_at_first_k_ge_l": float(at["probe_belief_top1_on_support"].iloc[0]),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "mod31_per_query_threshold_summary.csv", index=False)
    return out


def control_summary(final_agg: pd.DataFrame) -> pd.DataFrame:
    wanted = ["main_dense_mod31", "control_static_mod31", "control_shuffled_mod31"]
    rows: List[Dict] = []
    for run, grp in final_agg[final_agg["run"].isin(wanted)].groupby("run"):
        for length, g in grp.groupby("length"):
            if run == "main_dense_mod31":
                pick = g[g["k"] >= length].sort_values("k").head(1)
            elif run == "control_shuffled_mod31":
                pick = g[g["k"] >= length].sort_values("k").head(1)
                if pick.empty:
                    pick = g.sort_values("query_target_mass", ascending=False).head(1)
            else:
                pick = g.head(1)
            if pick.empty:
                continue
            rows.append(
                {
                    "run": RUN_LABELS.get(run, run),
                    "length": int(length),
                    "k": int(pick["k"].iloc[0]),
                    "query_target_mass": float(pick["query_target_mass"].iloc[0]),
                    "query_top1_on_support": float(pick["query_top1_on_support"].iloc[0]),
                    "query_target_nll": float(pick["query_target_nll"].iloc[0]),
                    "probe_belief_target_mass": float(pick["probe_belief_target_mass"].iloc[0]),
                    "probe_belief_top1_on_support": float(pick["probe_belief_top1_on_support"].iloc[0]),
                    "probe_belief_target_nll": float(pick["probe_belief_target_nll"].iloc[0]),
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
    ax.set_xticks(sorted(data[data["k"] >= 0]["k"].unique()))
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / filename, dpi=180)
    plt.close(fig)


def plot_controls(summary: pd.DataFrame, value: str, filename: str, title: str, ylabel: str) -> None:
    if summary.empty:
        return
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    colors = {
        "Dense recurrent p=31": "#2ca02c",
        "Static compiler p=31": "#9467bd",
        "Shuffled recurrent p=31": "#d62728",
    }
    for run, grp in summary.groupby("run"):
        grp = grp.sort_values("length")
        ax.plot(grp["length"], grp[value] * 100, marker="o", linewidth=2.5, label=run, color=colors.get(run))
    ax.set_title(title)
    ax.set_xlabel("program length L")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted(summary["length"].unique()))
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / filename, dpi=180)
    plt.close(fig)


def percent(x: float) -> str:
    return f"{100 * x:.1f}%"


def md_table(df: pd.DataFrame, columns: List[str], rename: Dict[str, str] | None = None, pct_cols: List[str] | None = None) -> str:
    rename = rename or {}
    pct_cols = pct_cols or []
    lines = []
    headers = [rename.get(c, c) for c in columns]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---" for _ in columns]) + "|")
    for _, row in df[columns].iterrows():
        vals = []
        for col in columns:
            val = row[col]
            if col in pct_cols:
                vals.append(percent(float(val)))
            elif col in {"length", "first_k_ge_l", "k", "step"}:
                vals.append(str(int(val)))
            elif isinstance(val, float):
                vals.append(f"{val:.3f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_summary(threshold: pd.DataFrame, per_query: pd.DataFrame, controls: pd.DataFrame, final_agg: pd.DataFrame) -> None:
    summary_runs = [
        "main_dense_mod31",
        "control_static_mod31",
        "control_shuffled_mod31",
        "pilot_dense_mod11",
        "control_static_mod11",
        "control_shuffled_mod11",
    ]
    final_summary = final_agg[final_agg["run"].isin(summary_runs)].copy()
    cols = [
        "run_label",
        "step",
        "model",
        "length",
        "k",
        "query_target_mass",
        "query_top1_on_support",
        "probe_belief_target_mass",
        "probe_belief_top1_on_support",
    ]
    final_summary[cols].to_csv(OUT / "final_metrics_query_mean.csv", index=False)

    lines = [
        "# Dense Latent Query Executor Analysis Summary",
        "",
        "This summary is generated from `runs/*/metrics_final.csv`.",
        "",
        "## Main Modulus-31 Threshold",
        "",
        md_table(
            threshold,
            [
                "length",
                "best_query_mass_before_k_ge_l",
                "best_probe_belief_mass_before_k_ge_l",
                "first_k_ge_l",
                "query_mass_at_first_k_ge_l",
                "probe_belief_mass_at_first_k_ge_l",
                "query_top1_at_first_k_ge_l",
            ],
            rename={
                "length": "L",
                "best_query_mass_before_k_ge_l": "best query mass K<L",
                "best_probe_belief_mass_before_k_ge_l": "best probe belief mass K<L",
                "first_k_ge_l": "first K>=L",
                "query_mass_at_first_k_ge_l": "query mass",
                "probe_belief_mass_at_first_k_ge_l": "probe belief mass",
                "query_top1_at_first_k_ge_l": "query top1",
            },
            pct_cols=[
                "best_query_mass_before_k_ge_l",
                "best_probe_belief_mass_before_k_ge_l",
                "query_mass_at_first_k_ge_l",
                "probe_belief_mass_at_first_k_ge_l",
                "query_top1_at_first_k_ge_l",
            ],
        ),
        "",
        "## Main Modulus-31 Per-Query Threshold",
        "",
        md_table(
            per_query,
            [
                "length",
                "query_type",
                "first_k_ge_l",
                "query_mass_at_first_k_ge_l",
                "probe_belief_mass_at_first_k_ge_l",
                "query_top1_at_first_k_ge_l",
            ],
            rename={
                "length": "L",
                "query_type": "query",
                "first_k_ge_l": "first K>=L",
                "query_mass_at_first_k_ge_l": "query mass",
                "probe_belief_mass_at_first_k_ge_l": "probe belief mass",
                "query_top1_at_first_k_ge_l": "query top1",
            },
            pct_cols=[
                "query_mass_at_first_k_ge_l",
                "probe_belief_mass_at_first_k_ge_l",
                "query_top1_at_first_k_ge_l",
            ],
        ),
        "",
        "## Modulus-31 Controls",
        "",
        md_table(
            controls,
            [
                "run",
                "length",
                "k",
                "query_target_mass",
                "query_top1_on_support",
                "probe_belief_target_mass",
                "probe_belief_top1_on_support",
            ],
            rename={
                "run": "model",
                "length": "L",
                "query_target_mass": "query mass",
                "query_top1_on_support": "query top1",
                "probe_belief_target_mass": "probe belief mass",
                "probe_belief_top1_on_support": "probe belief top1",
            },
            pct_cols=[
                "query_target_mass",
                "query_top1_on_support",
                "probe_belief_target_mass",
                "probe_belief_top1_on_support",
            ],
        )
        if not controls.empty
        else "_Modulus-31 control rows are not present yet._",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    df = load_metrics()
    df.to_csv(OUT / "all_metrics_long.csv", index=False)
    df.to_csv(OUT / "final_metrics_long.csv", index=False)

    final_agg = aggregate_query_types(df)
    final_agg.to_csv(OUT / "final_metrics_query_mean.csv", index=False)

    main = main_dense(df)
    main_agg = main_dense_agg(final_agg)
    threshold = threshold_summary(main_agg)
    per_query = per_query_threshold(main)
    controls = control_summary(final_agg)

    plot_heatmap(main_agg, "query_target_mass", "mod31_query_mass_heatmap.png", "Dense recurrent query target mass")
    plot_heatmap(main_agg, "query_top1_on_support", "mod31_query_top1_heatmap.png", "Dense recurrent query top-1 on support")
    plot_heatmap(main_agg, "probe_belief_target_mass", "mod31_probe_belief_mass_heatmap.png", "Dense recurrent probe belief target mass")
    plot_curves(main_agg, "query_target_mass", "mod31_query_mass_k_curves.png", "Dense recurrent query mass by K", "query target mass (%)")
    plot_curves(
        main_agg,
        "probe_belief_target_mass",
        "mod31_probe_belief_mass_k_curves.png",
        "Dense recurrent probe belief mass by K",
        "probe belief target mass (%)",
    )
    plot_controls(controls, "query_target_mass", "mod31_controls_query_mass.png", "Modulus-31 query mass controls", "query target mass (%)")
    plot_controls(
        controls,
        "probe_belief_target_mass",
        "mod31_controls_probe_belief_mass.png",
        "Modulus-31 probe belief mass controls",
        "probe belief target mass (%)",
    )
    write_summary(threshold, per_query, controls, final_agg)
    print(f"wrote analysis to {OUT}")


if __name__ == "__main__":
    main()
