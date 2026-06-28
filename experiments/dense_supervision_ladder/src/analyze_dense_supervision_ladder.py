#!/usr/bin/env python3
"""Analyze dense supervision ladder runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
RUN_DIR = EXPERIMENT_DIR / "runs"
OUT = EXPERIMENT_DIR / "analysis"
FIG = OUT / "figures"

SUPERVISION_ORDER = [
    "sampled_final",
    "soft_final_query",
    "prefix_query",
    "sparse_belief",
    "full_belief",
]

SUPERVISION_LABELS = {
    "sampled_final": "Sampled final",
    "soft_final_query": "Soft final query",
    "prefix_query": "Prefix query",
    "sparse_belief": "Sparse belief",
    "full_belief": "Full belief",
}

COLORS = {
    "sampled_final": "#6f6f6f",
    "soft_final_query": "#1f77b4",
    "prefix_query": "#2ca02c",
    "sparse_belief": "#ff7f0e",
    "full_belief": "#d62728",
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
    "decoder_belief_target_mass",
    "decoder_belief_target_nll",
    "decoder_belief_top1_on_support",
    "mean_decoder_belief_support_size",
]


def load_results(run_dir: Path) -> Dict:
    path = run_dir / "results.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def phase_from_run(run: str) -> str:
    if run.startswith("main_"):
        return "main"
    if run.startswith("pilot_"):
        return "pilot"
    if run.startswith("smoke_"):
        return "smoke"
    return "other"


def load_metrics() -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for metrics_path in sorted(RUN_DIR.glob("*/metrics_final.csv")):
        run_dir = metrics_path.parent
        run = run_dir.name
        results = load_results(run_dir)
        args = results.get("args", {})
        supervision = args.get("supervision", run.split("_mod")[0].split("_", 1)[-1])
        modulus = int(args.get("modulus", -1))
        step = int(args.get("train_steps", -1))
        df = pd.read_csv(metrics_path)
        for col in METRIC_COLS:
            if col not in df:
                df[col] = float("nan")
        df["run"] = run
        df["phase"] = phase_from_run(run)
        df["supervision"] = supervision
        df["supervision_label"] = SUPERVISION_LABELS.get(supervision, supervision)
        df["modulus"] = modulus
        df["step"] = step
        frames.append(df)
    if not frames:
        raise SystemExit(f"No metrics found in {RUN_DIR}")
    out = pd.concat(frames, ignore_index=True)
    out["supervision"] = pd.Categorical(out["supervision"], SUPERVISION_ORDER, ordered=True)
    return out.sort_values(["phase", "modulus", "supervision", "length", "k", "query_type"])


def aggregate_query_types(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "run",
        "phase",
        "supervision",
        "supervision_label",
        "modulus",
        "step",
        "model",
        "length",
        "k",
    ]
    return df.groupby(group_cols, observed=True, as_index=False)[METRIC_COLS].mean()


def first_k_ge_l_summary(agg: pd.DataFrame, phase: str, modulus: int) -> pd.DataFrame:
    rows: List[Dict] = []
    sub = agg[(agg["phase"] == phase) & (agg["modulus"] == modulus)]
    for (supervision, length), grp in sub.groupby(["supervision", "length"], observed=True):
        before = grp[(grp["k"] >= 0) & (grp["k"] < length)]
        at = grp[grp["k"] >= length].sort_values("k").head(1)
        if at.empty:
            continue
        row = at.iloc[0]
        rows.append(
            {
                "phase": phase,
                "modulus": modulus,
                "supervision": str(supervision),
                "supervision_label": SUPERVISION_LABELS.get(str(supervision), str(supervision)),
                "length": int(length),
                "first_k_ge_l": int(row["k"]),
                "best_query_mass_before_k_ge_l": float(before["query_target_mass"].max()) if not before.empty else float("nan"),
                "best_probe_belief_mass_before_k_ge_l": float(before["probe_belief_target_mass"].max()) if not before.empty else float("nan"),
                "best_decoder_belief_mass_before_k_ge_l": float(before["decoder_belief_target_mass"].max()) if not before.empty else float("nan"),
                "query_target_mass": float(row["query_target_mass"]),
                "query_top1_on_support": float(row["query_top1_on_support"]),
                "probe_belief_target_mass": float(row["probe_belief_target_mass"]),
                "probe_belief_top1_on_support": float(row["probe_belief_top1_on_support"]),
                "decoder_belief_target_mass": float(row["decoder_belief_target_mass"]),
                "decoder_belief_top1_on_support": float(row["decoder_belief_top1_on_support"]),
                "mean_query_support_size": float(row["mean_query_support_size"]),
                "mean_belief_support_size": float(row["mean_belief_support_size"]),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["supervision"] = pd.Categorical(out["supervision"], SUPERVISION_ORDER, ordered=True)
        out = out.sort_values(["supervision", "length"])
    return out


def per_query_first_k_ge_l(df: pd.DataFrame, phase: str, modulus: int) -> pd.DataFrame:
    rows: List[Dict] = []
    sub = df[(df["phase"] == phase) & (df["modulus"] == modulus)]
    for (supervision, length, query_type), grp in sub.groupby(["supervision", "length", "query_type"], observed=True):
        before = grp[(grp["k"] >= 0) & (grp["k"] < length)]
        at = grp[grp["k"] >= length].sort_values("k").head(1)
        if at.empty:
            continue
        row = at.iloc[0]
        rows.append(
            {
                "phase": phase,
                "modulus": modulus,
                "supervision": str(supervision),
                "supervision_label": SUPERVISION_LABELS.get(str(supervision), str(supervision)),
                "length": int(length),
                "query_type": str(query_type),
                "first_k_ge_l": int(row["k"]),
                "best_query_mass_before_k_ge_l": float(before["query_target_mass"].max()) if not before.empty else float("nan"),
                "query_target_mass": float(row["query_target_mass"]),
                "query_top1_on_support": float(row["query_top1_on_support"]),
                "probe_belief_target_mass": float(row["probe_belief_target_mass"]),
                "decoder_belief_target_mass": float(row["decoder_belief_target_mass"]),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["supervision"] = pd.Categorical(out["supervision"], SUPERVISION_ORDER, ordered=True)
        out = out.sort_values(["supervision", "length", "query_type"])
    return out


def percent(x: float) -> str:
    return f"{100 * x:.1f}%"


def md_table(df: pd.DataFrame, columns: List[str], rename: Dict[str, str] | None = None, pct_cols: Iterable[str] = ()) -> str:
    rename = rename or {}
    pct_cols = set(pct_cols)
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
            elif col in {"length", "first_k_ge_l", "modulus", "step"}:
                vals.append(str(int(val)))
            elif isinstance(val, float):
                vals.append(f"{val:.3f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def plot_ladder(summary: pd.DataFrame, value: str, filename: str, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    for supervision in SUPERVISION_ORDER:
        grp = summary[summary["supervision"].astype(str) == supervision].sort_values("length")
        if grp.empty:
            continue
        ax.plot(
            grp["length"],
            grp[value] * 100,
            marker="o",
            linewidth=2.4,
            label=SUPERVISION_LABELS[supervision],
            color=COLORS[supervision],
        )
    ax.set_title(title)
    ax.set_xlabel("program length L")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted(summary["length"].unique()))
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / filename, dpi=180)
    plt.close(fig)


def plot_k_curves(agg: pd.DataFrame, supervision: str, value: str, filename: str, title: str, ylabel: str) -> None:
    data = agg[(agg["phase"] == "main") & (agg["modulus"] == 31) & (agg["supervision"].astype(str) == supervision)]
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    for length, grp in data.groupby("length"):
        grp = grp.sort_values("k")
        ax.plot(grp["k"], grp[value] * 100, marker="o", linewidth=2, label=f"L={length}")
    ax.set_title(title)
    ax.set_xlabel("internal recurrent steps K")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted(data["k"].unique()))
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / filename, dpi=180)
    plt.close(fig)


def plot_heatmap(agg: pd.DataFrame, supervision: str, value: str, filename: str, title: str) -> None:
    data = agg[(agg["phase"] == "main") & (agg["modulus"] == 31) & (agg["supervision"].astype(str) == supervision)]
    if data.empty:
        return
    pivot = data.pivot_table(index="length", columns="k", values=value, observed=True).sort_index()
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
            ax.text(x, y, f"{val:.0f}", ha="center", va="center", color=color, fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(value.replace("_", " ") + " (%)")
    fig.tight_layout()
    fig.savefig(FIG / filename, dpi=180)
    plt.close(fig)


def write_summary(main_summary: pd.DataFrame, pilot_summary: pd.DataFrame) -> None:
    key_cols = [
        "supervision_label",
        "length",
        "first_k_ge_l",
        "query_target_mass",
        "query_top1_on_support",
        "probe_belief_target_mass",
        "decoder_belief_target_mass",
    ]
    pct_cols = [
        "query_target_mass",
        "query_top1_on_support",
        "probe_belief_target_mass",
        "decoder_belief_target_mass",
    ]
    lines = [
        "# Dense Supervision Ladder Analysis Summary",
        "",
        "This summary is generated from `runs/*/metrics_final.csv`.",
        "",
        "## Modulus 31 Ladder at K=L",
        "",
        md_table(
            main_summary,
            key_cols,
            rename={
                "supervision_label": "supervision",
                "length": "L",
                "first_k_ge_l": "K",
                "query_target_mass": "query mass",
                "query_top1_on_support": "query top1",
                "probe_belief_target_mass": "probe belief",
                "decoder_belief_target_mass": "decoder belief",
            },
            pct_cols=pct_cols,
        ),
        "",
        "## Modulus 11 Pilot at K=L",
        "",
        md_table(
            pilot_summary,
            key_cols,
            rename={
                "supervision_label": "supervision",
                "length": "L",
                "first_k_ge_l": "K",
                "query_target_mass": "query mass",
                "query_top1_on_support": "query top1",
                "probe_belief_target_mass": "probe belief",
                "decoder_belief_target_mass": "decoder belief",
            },
            pct_cols=pct_cols,
        ),
        "",
        "## Generated Figures",
        "",
        "- `figures/mod31_ladder_query_mass_at_k_ge_l.png`",
        "- `figures/mod31_ladder_probe_belief_mass_at_k_ge_l.png`",
        "- `figures/mod31_ladder_decoder_belief_mass_at_k_ge_l.png`",
        "- `figures/mod31_full_belief_query_mass_heatmap.png`",
        "- `figures/mod31_full_belief_decoder_mass_heatmap.png`",
        "- `figures/mod31_query_mass_by_k_<supervision>.png`",
    ]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    df = load_metrics()
    df.to_csv(OUT / "all_metrics_long.csv", index=False)

    agg = aggregate_query_types(df)
    agg.to_csv(OUT / "final_metrics_query_mean.csv", index=False)

    main_summary = first_k_ge_l_summary(agg, "main", 31)
    pilot_summary = first_k_ge_l_summary(agg, "pilot", 11)
    main_summary.to_csv(OUT / "mod31_ladder_threshold_summary.csv", index=False)
    pilot_summary.to_csv(OUT / "mod11_ladder_threshold_summary.csv", index=False)

    per_query = per_query_first_k_ge_l(df, "main", 31)
    per_query.to_csv(OUT / "mod31_ladder_per_query_threshold_summary.csv", index=False)

    plot_ladder(
        main_summary,
        "query_target_mass",
        "mod31_ladder_query_mass_at_k_ge_l.png",
        "Modulus 31 query mass at K=L",
        "query target mass (%)",
    )
    plot_ladder(
        main_summary,
        "probe_belief_target_mass",
        "mod31_ladder_probe_belief_mass_at_k_ge_l.png",
        "Modulus 31 post-hoc probe belief mass at K=L",
        "probe belief target mass (%)",
    )
    plot_ladder(
        main_summary,
        "decoder_belief_target_mass",
        "mod31_ladder_decoder_belief_mass_at_k_ge_l.png",
        "Modulus 31 trained decoder belief mass at K=L",
        "decoder belief target mass (%)",
    )
    plot_ladder(
        pilot_summary,
        "query_target_mass",
        "mod11_ladder_query_mass_at_k_ge_l.png",
        "Modulus 11 query mass at K=L",
        "query target mass (%)",
    )

    for supervision in SUPERVISION_ORDER:
        plot_k_curves(
            agg,
            supervision,
            "query_target_mass",
            f"mod31_query_mass_by_k_{supervision}.png",
            f"Modulus 31 query mass by K: {SUPERVISION_LABELS[supervision]}",
            "query target mass (%)",
        )

    plot_heatmap(
        agg,
        "full_belief",
        "query_target_mass",
        "mod31_full_belief_query_mass_heatmap.png",
        "Full belief query mass by length and K",
    )
    plot_heatmap(
        agg,
        "full_belief",
        "decoder_belief_target_mass",
        "mod31_full_belief_decoder_mass_heatmap.png",
        "Full belief decoder mass by length and K",
    )

    write_summary(main_summary, pilot_summary)
    print(f"wrote analysis to {OUT}")


if __name__ == "__main__":
    main()
