#!/usr/bin/env python3
"""Analyze dense teacher-distillation bottleneck runs."""

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
    "decoder_query_target_mass",
    "decoder_query_target_nll",
    "decoder_query_top1_on_support",
    "mean_decoder_query_support_size",
]


def load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def phase_from_run(run: str) -> str:
    for phase in ("main", "pilot", "smoke"):
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
        df["train_steps"] = int(args.get("train_steps", -1))
        df["train_max_len"] = int(args.get("train_max_len", -1))
        frames.append(df)
    if not frames:
        raise SystemExit(f"No metrics found in {RUN_DIR}")
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["phase", "modulus", "variant", "length", "query_type", "k"])


def aggregate_query_types(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "run",
        "phase",
        "variant",
        "modulus",
        "model",
        "supervision",
        "transition",
        "decoder_type",
        "state_dim",
        "instr_dim",
        "decoder_rank",
        "train_steps",
        "train_max_len",
        "length",
        "k",
    ]
    return df.groupby(group_cols, observed=True, as_index=False)[METRIC_COLS].mean()


def first_k_ge_l_summary(agg: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    for (phase, modulus, variant, length), grp in agg.groupby(["phase", "modulus", "variant", "length"], observed=True):
        at = grp[grp["k"] >= length].sort_values("k").head(1)
        before = grp[(grp["k"] >= 0) & (grp["k"] < length)]
        if at.empty:
            continue
        row = at.iloc[0]
        rows.append(
            {
                "phase": phase,
                "modulus": int(modulus),
                "variant": variant,
                "transition": row["transition"],
                "decoder_type": row["decoder_type"],
                "state_dim": int(row["state_dim"]),
                "instr_dim": int(row["instr_dim"]),
                "decoder_rank": int(row["decoder_rank"]),
                "train_steps": int(row["train_steps"]),
                "length": int(length),
                "first_k_ge_l": int(row["k"]),
                "best_decoder_query_mass_before_k_ge_l": float(before["decoder_query_target_mass"].max())
                if not before.empty
                else float("nan"),
                "decoder_query_target_mass": float(row["decoder_query_target_mass"]),
                "decoder_belief_target_mass": float(row["decoder_belief_target_mass"]),
                "probe_belief_target_mass": float(row["probe_belief_target_mass"]),
                "query_target_mass": float(row["query_target_mass"]),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["phase", "modulus", "variant", "length"])
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
            elif col in {"length", "first_k_ge_l", "modulus", "state_dim", "train_steps"}:
                vals.append(str(int(val)))
            elif isinstance(val, float):
                vals.append(f"{val:.3f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def variant_order(df: pd.DataFrame) -> List[str]:
    def key(name: str) -> tuple:
        phase_rank = 0 if name.startswith("main_") else 1 if name.startswith("pilot_") else 2
        lowrank_rank = 1 if "lowrank" in name else 0
        residual_rank = 1 if "residual" in name else 0
        return (phase_rank, lowrank_rank, residual_rank, name)

    return sorted(df["variant"].dropna().astype(str).unique(), key=key)


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
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / filename, dpi=180)
    plt.close(fig)


def plot_k_curves(agg: pd.DataFrame, phase: str, modulus: int, variant: str, value: str, filename: str, ylabel: str) -> None:
    data = agg[(agg["phase"] == phase) & (agg["modulus"] == modulus) & (agg["variant"] == variant)]
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(9.8, 5.6))
    for length, grp in data.groupby("length"):
        grp = grp.sort_values("k")
        ax.plot(grp["k"], grp[value] * 100, marker="o", linewidth=2.0, label=f"L={int(length)}")
    ax.set_title(f"{variant}: {ylabel} by recurrent steps")
    ax.set_xlabel("internal recurrent steps K")
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted(data["k"].unique()))
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / filename, dpi=180)
    plt.close(fig)


def write_summary(df: pd.DataFrame, agg: pd.DataFrame, threshold: pd.DataFrame) -> None:
    lines: List[str] = []
    lines.append("# Dense Teacher Distillation Analysis Summary\n")
    lines.append("## Runs\n")
    run_cols = [
        "phase",
        "modulus",
        "variant",
        "transition",
        "decoder_type",
        "state_dim",
        "train_steps",
        "train_max_len",
    ]
    runs = (
        df[run_cols]
        .drop_duplicates()
        .sort_values(["phase", "modulus", "variant"])
        .reset_index(drop=True)
    )
    lines.append(md_table(runs, run_cols))
    lines.append("\n## First K >= L Summary\n")
    if threshold.empty:
        lines.append("No recurrent threshold rows were available.\n")
    else:
        cols = [
            "phase",
            "modulus",
            "variant",
            "length",
            "decoder_query_target_mass",
            "decoder_belief_target_mass",
            "probe_belief_target_mass",
            "query_target_mass",
        ]
        lines.append(
            md_table(
                threshold,
                cols,
                pct_cols=[
                    "decoder_query_target_mass",
                    "decoder_belief_target_mass",
                    "probe_belief_target_mass",
                    "query_target_mass",
                ],
            )
        )
    lines.append("\n## Files\n")
    lines.append("- `all_metrics_long.csv`: per-query metrics loaded from each run.")
    lines.append("- `final_metrics_query_mean.csv`: metrics averaged across query types.")
    lines.append("- `threshold_summary.csv`: first available K satisfying K >= L.")
    lines.append("- `figures/`: generated metric plots.")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    df = load_metrics()
    agg = aggregate_query_types(df)
    threshold = first_k_ge_l_summary(agg)

    df.to_csv(OUT / "all_metrics_long.csv", index=False)
    agg.to_csv(OUT / "final_metrics_query_mean.csv", index=False)
    threshold.to_csv(OUT / "threshold_summary.csv", index=False)

    for phase in sorted(threshold["phase"].unique()) if not threshold.empty else []:
        for modulus in sorted(threshold[threshold["phase"] == phase]["modulus"].unique()):
            plot_at_k_ge_l(
                threshold,
                phase,
                int(modulus),
                "decoder_query_target_mass",
                f"{phase}_mod{int(modulus)}_decoder_query_mass_at_k_ge_l.png",
                "decoder-projected query target mass",
            )
            plot_at_k_ge_l(
                threshold,
                phase,
                int(modulus),
                "decoder_belief_target_mass",
                f"{phase}_mod{int(modulus)}_decoder_belief_mass_at_k_ge_l.png",
                "decoded belief target mass",
            )
            phase_mod = agg[(agg["phase"] == phase) & (agg["modulus"] == modulus)]
            for variant in variant_order(phase_mod):
                plot_k_curves(
                    agg,
                    phase,
                    int(modulus),
                    variant,
                    "decoder_query_target_mass",
                    f"{phase}_mod{int(modulus)}_{variant}_decoder_query_by_k.png",
                    "decoder-projected query target mass",
                )

    write_summary(df, agg, threshold)
    print(f"[analysis] wrote {OUT}")


if __name__ == "__main__":
    main()
