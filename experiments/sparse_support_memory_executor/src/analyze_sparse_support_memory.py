#!/usr/bin/env python3
"""Analyze sparse support-memory executor runs."""

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

METRIC_COLS = [
    "decoder_query_target_mass",
    "decoder_query_target_nll",
    "decoder_query_top1_on_support",
    "mean_decoder_query_support_size",
    "decoder_belief_target_mass",
    "decoder_belief_target_nll",
    "decoder_belief_top1_on_support",
    "mean_decoder_belief_support_size",
    "empty_slot_rate",
    "mean_active_slots",
    "mean_total_slot_weight",
    "mean_target_support_size",
]


def load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def phase_from_run(run: str) -> str:
    for phase in ("scale", "main", "pilot", "smoke"):
        if run.startswith(f"{phase}_"):
            return phase
    return "other"


def load_metrics() -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for metrics_path in sorted(RUN_DIR.glob("*/metrics_final.csv")):
        run_dir = metrics_path.parent
        run = run_dir.name
        results = load_json(run_dir / "results.json")
        args = results.get("args", {})
        df = pd.read_csv(metrics_path)
        for col in METRIC_COLS:
            if col not in df:
                df[col] = float("nan")
        df["run"] = run
        df["phase"] = phase_from_run(run)
        df["variant"] = df.get("variant", args.get("variant_name", run))
        df["modulus"] = int(args.get("modulus", df.get("modulus", -1)))
        df["slot_capacity"] = int(args.get("slot_capacity", df.get("slot_capacity", -1)))
        frames.append(df)
    if not frames:
        raise SystemExit(f"No metrics found in {RUN_DIR}")
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["phase", "modulus", "slot_capacity", "variant", "length", "query_type", "k"])


def aggregate_query_types(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "run",
        "phase",
        "variant",
        "model",
        "modulus",
        "observe_mod",
        "observe_prob",
        "op_family",
        "query_family",
        "slot_capacity",
        "init_strategy",
        "length",
        "k",
    ]
    return df.groupby(group_cols, observed=True, as_index=False)[METRIC_COLS].mean()


def first_k_ge_l_summary(agg: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    for (phase, modulus, slot_capacity, variant, length), grp in agg.groupby(
        ["phase", "modulus", "slot_capacity", "variant", "length"], observed=True
    ):
        at = grp[grp["k"] >= length].sort_values("k").head(1)
        before = grp[(grp["k"] >= 0) & (grp["k"] < length)]
        if at.empty:
            continue
        row = at.iloc[0]
        rows.append(
            {
                "phase": phase,
                "modulus": int(modulus),
                "slot_capacity": int(slot_capacity),
                "variant": variant,
                "length": int(length),
                "first_k_ge_l": int(row["k"]),
                "best_decoder_query_mass_before_k_ge_l": float(before["decoder_query_target_mass"].max())
                if not before.empty
                else float("nan"),
                "decoder_query_target_mass": float(row["decoder_query_target_mass"]),
                "decoder_belief_target_mass": float(row["decoder_belief_target_mass"]),
                "empty_slot_rate": float(row["empty_slot_rate"]),
                "mean_active_slots": float(row["mean_active_slots"]),
                "mean_target_support_size": float(row["mean_target_support_size"]),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["phase", "modulus", "slot_capacity", "length"])
    return out


def percent(x: float) -> str:
    return f"{100 * x:.1f}%"


def md_table(df: pd.DataFrame, columns: List[str], pct_cols: Iterable[str] = ()) -> str:
    pct_cols = set(pct_cols)
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join(["---" for _ in columns]) + "|")
    for _, row in df[columns].iterrows():
        vals = []
        for col in columns:
            val = row[col]
            if col in pct_cols:
                vals.append(percent(float(val)))
            elif col in {"modulus", "slot_capacity", "length", "first_k_ge_l"}:
                vals.append(str(int(val)))
            elif isinstance(val, float):
                vals.append(f"{val:.3f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def variant_order(df: pd.DataFrame) -> List[str]:
    return [
        str(v)
        for v in df.sort_values(["modulus", "slot_capacity", "variant"])["variant"].drop_duplicates().tolist()
    ]


def plot_at_k_ge_l(summary: pd.DataFrame, phase: str, modulus: int, value: str, filename: str, ylabel: str) -> None:
    data = summary[(summary["phase"] == phase) & (summary["modulus"] == modulus)]
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(9.8, 5.6))
    for variant in variant_order(data):
        grp = data[data["variant"] == variant].sort_values("length")
        ax.plot(grp["length"], grp[value] * 100, marker="o", linewidth=2.2, label=variant)
    ax.set_title(f"{phase} modulus {modulus}: {ylabel} at first K >= L")
    ax.set_xlabel("program length L")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted(data["length"].unique()))
    if value != "mean_active_slots":
        ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / filename, dpi=180)
    plt.close(fig)


def plot_capacity_at_longest(summary: pd.DataFrame, phase: str, modulus: int, value: str, filename: str, ylabel: str) -> None:
    data = summary[(summary["phase"] == phase) & (summary["modulus"] == modulus)]
    if data.empty:
        return
    rows = []
    for slot_capacity, grp in data.groupby("slot_capacity"):
        length = int(grp["length"].max())
        row = grp[grp["length"] == length].iloc[0]
        rows.append({"slot_capacity": int(slot_capacity), "length": length, value: float(row[value])})
    cap = pd.DataFrame(rows).sort_values("slot_capacity")
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    y = cap[value] * 100 if value != "mean_active_slots" else cap[value]
    ax.plot(cap["slot_capacity"], y, marker="o", linewidth=2.2)
    ax.axvline(modulus, color="black", linewidth=1.2, alpha=0.4, linestyle="--", label="S=p")
    ax.set_title(f"{phase} modulus {modulus}: longest-length {ylabel}")
    ax.set_xlabel("slot capacity S")
    ax.set_ylabel(ylabel)
    if value != "mean_active_slots":
        ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / filename, dpi=180)
    plt.close(fig)


def write_summary(df: pd.DataFrame, agg: pd.DataFrame, threshold: pd.DataFrame) -> None:
    lines: List[str] = []
    lines.append("# Sparse Support Memory Analysis Summary\n")
    lines.append("## Runs\n")
    run_cols = [
        "phase",
        "modulus",
        "variant",
        "slot_capacity",
        "observe_prob",
        "length",
        "k",
        "n",
    ]
    run_table = (
        df.groupby(["phase", "modulus", "variant", "slot_capacity", "observe_prob"], as_index=False)
        .agg(length=("length", "max"), k=("k", "max"), n=("n", "max"))
        .sort_values(["phase", "modulus", "slot_capacity"])
    )
    lines.append(md_table(run_table, run_cols))
    lines.append("\n## First K >= L Summary\n")
    if threshold.empty:
        lines.append("No threshold rows were available.\n")
    else:
        for (phase, modulus), grp in threshold.groupby(["phase", "modulus"], observed=True):
            lines.append(f"### {phase} modulus {int(modulus)}\n")
            lines.append(
                md_table(
                    grp[
                        [
                            "slot_capacity",
                            "length",
                            "first_k_ge_l",
                            "decoder_query_target_mass",
                            "decoder_belief_target_mass",
                            "empty_slot_rate",
                            "mean_active_slots",
                            "mean_target_support_size",
                        ]
                    ],
                    [
                        "slot_capacity",
                        "length",
                        "first_k_ge_l",
                        "decoder_query_target_mass",
                        "decoder_belief_target_mass",
                        "empty_slot_rate",
                        "mean_active_slots",
                        "mean_target_support_size",
                    ],
                    pct_cols=[
                        "decoder_query_target_mass",
                        "decoder_belief_target_mass",
                        "empty_slot_rate",
                    ],
                )
            )
            lines.append("")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    df = load_metrics()
    agg = aggregate_query_types(df)
    threshold = first_k_ge_l_summary(agg)
    df.to_csv(OUT / "all_metrics_long.csv", index=False)
    agg.to_csv(OUT / "all_metrics_query_mean.csv", index=False)
    threshold.to_csv(OUT / "first_k_ge_l_summary.csv", index=False)
    write_summary(df, agg, threshold)
    for (phase, modulus), _ in threshold.groupby(["phase", "modulus"], observed=True):
        prefix = f"{phase}_mod{int(modulus)}"
        plot_at_k_ge_l(
            threshold,
            phase,
            int(modulus),
            "decoder_query_target_mass",
            f"{prefix}_decoder_query_mass_at_k_ge_l.png",
            "decoder query target mass",
        )
        plot_at_k_ge_l(
            threshold,
            phase,
            int(modulus),
            "decoder_belief_target_mass",
            f"{prefix}_decoder_belief_mass_at_k_ge_l.png",
            "decoder belief target mass",
        )
        plot_at_k_ge_l(
            threshold,
            phase,
            int(modulus),
            "empty_slot_rate",
            f"{prefix}_empty_slot_rate_at_k_ge_l.png",
            "empty slot rate",
        )
        plot_capacity_at_longest(
            threshold,
            phase,
            int(modulus),
            "decoder_query_target_mass",
            f"{prefix}_capacity_longest_decoder_query_mass.png",
            "decoder query target mass",
        )
    print(f"Wrote analysis to {OUT}")


if __name__ == "__main__":
    main()
