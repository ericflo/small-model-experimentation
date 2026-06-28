#!/usr/bin/env python3
"""Analyze latent Qwen fast-weight experiment runs and generate figures."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
RUNS = [
    ("Full recurrent", EXPERIMENT_DIR / "runs/main_qwen35_hook_full_seed7/results.json"),
    ("K=0-only control", EXPERIMENT_DIR / "runs/control_qwen35_hook_traink0_seed7/results.json"),
    ("Aux value loss", EXPERIMENT_DIR / "runs/main_qwen35_hook_aux02_seed7/results.json"),
]
RETESTS = [
    ("Full step 200 retest", EXPERIMENT_DIR / "runs/eval_main_step200_n250/eval_only_results.json"),
    ("Aux final retest", EXPERIMENT_DIR / "runs/eval_aux_final_n250/eval_only_results.json"),
]
OUT = EXPERIMENT_DIR / "analysis"
FIG = OUT / "figures"


def wilson_interval(acc: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    denom = 1 + z * z / n
    center = (acc + z * z / (2 * n)) / denom
    half = z * math.sqrt((acc * (1 - acc) / n) + (z * z / (4 * n * n))) / denom
    return max(0.0, center - half), min(1.0, center + half)


def read_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_eval_n(data: Dict) -> int:
    args = data.get("metadata", {}).get("args", {})
    return int(args.get("eval_batches", 0)) * int(args.get("eval_batch_size", 0))


def load_training_runs() -> pd.DataFrame:
    rows: List[Dict] = []
    for run_name, path in RUNS:
        data = read_json(path)
        n = run_eval_n(data) or 100
        for rec in data.get("eval", []):
            step = int(rec["step"])
            if "base_val" in rec:
                rows.append(
                    {
                        "run": "Frozen base",
                        "source": run_name,
                        "step": step,
                        "split": "val",
                        "k": "base",
                        "k_num": -1,
                        "accuracy": float(rec["base_val"]),
                        "n": n,
                    }
                )
                rows.append(
                    {
                        "run": "Frozen base",
                        "source": run_name,
                        "step": step,
                        "split": "hard",
                        "k": "base",
                        "k_num": -1,
                        "accuracy": float(rec["base_hard"]),
                        "n": n,
                    }
                )
            for split in ["val", "hard"]:
                if split not in rec:
                    continue
                for k, acc in rec[split].items():
                    rows.append(
                        {
                            "run": run_name,
                            "source": run_name,
                            "step": step,
                            "split": split,
                            "k": str(k),
                            "k_num": int(k),
                            "accuracy": float(acc),
                            "n": n,
                        }
                    )
    df = pd.DataFrame(rows)
    if not df.empty:
        intervals = df.apply(lambda r: wilson_interval(float(r.accuracy), int(r.n)), axis=1)
        df["ci_low"] = [x[0] for x in intervals]
        df["ci_high"] = [x[1] for x in intervals]
    return df


def load_retests() -> pd.DataFrame:
    rows: List[Dict] = []
    for name, path in RETESTS:
        data = read_json(path)
        ev = data["eval"]
        n = int(ev["num_examples_per_split"])
        for split in ["val", "hard"]:
            for k, acc in ev[split].items():
                rows.append(
                    {
                        "run": name,
                        "checkpoint": data["checkpoint"],
                        "split": split,
                        "k": str(k),
                        "k_num": int(k),
                        "accuracy": float(acc),
                        "n": n,
                    }
                )
    df = pd.DataFrame(rows)
    if not df.empty:
        intervals = df.apply(lambda r: wilson_interval(float(r.accuracy), int(r.n)), axis=1)
        df["ci_low"] = [x[0] for x in intervals]
        df["ci_high"] = [x[1] for x in intervals]
    return df


def plot_k_advantage(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, split in zip(axes, ["val", "hard"]):
        sub = df[(df["split"] == split) & (df["k_num"] >= 0)]
        for run_name in ["Full recurrent", "K=0-only control", "Aux value loss"]:
            r = sub[sub["run"] == run_name]
            if r.empty:
                continue
            points = []
            for step, grp in r.groupby("step"):
                k0 = grp.loc[grp["k_num"] == 0, "accuracy"]
                rest = grp.loc[grp["k_num"] > 0, "accuracy"]
                if k0.empty or rest.empty:
                    continue
                points.append((step, float(rest.max() - k0.iloc[0])))
            if points:
                xs, ys = zip(*sorted(points))
                ax.plot(xs, [100 * y for y in ys], marker="o", linewidth=2, label=run_name)
        ax.axhline(0, color="#333333", linewidth=1)
        ax.set_title(f"Best recurrent gain over K=0 ({split})")
        ax.set_xlabel("optimizer step")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("accuracy gain, percentage points")
    axes[1].legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "best_k_gain_over_time.png", dpi=180)
    plt.close(fig)


def plot_full_heatmap(df: pd.DataFrame) -> None:
    full = df[(df["run"] == "Full recurrent") & (df["k_num"] >= 0)]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    for ax, split in zip(axes, ["val", "hard"]):
        pivot = (
            full[full["split"] == split]
            .pivot_table(index="k_num", columns="step", values="accuracy")
            .sort_index()
        )
        im = ax.imshow(pivot.values * 100, aspect="auto", origin="lower", cmap="viridis", vmin=14, vmax=32)
        ax.set_title(f"Full recurrent accuracy ({split})")
        ax.set_xlabel("optimizer step")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(c) for c in pivot.columns], rotation=45)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([str(i) for i in pivot.index])
        ax.set_ylabel("K")
        for y in range(pivot.shape[0]):
            for x in range(pivot.shape[1]):
                ax.text(x, y, f"{pivot.values[y, x]*100:.0f}", ha="center", va="center", color="white", fontsize=8)
    fig.subplots_adjust(right=0.88, wspace=0.18, bottom=0.18)
    cax = fig.add_axes([0.90, 0.18, 0.025, 0.70])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("accuracy (%)")
    fig.savefig(FIG / "full_recurrent_heatmap.png", dpi=180)
    plt.close(fig)


def plot_retests(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for ax, ((run_name, split), grp) in zip(axes.ravel(), df.groupby(["run", "split"], sort=False)):
        grp = grp.sort_values("k_num")
        xs = range(len(grp))
        y = grp["accuracy"].to_numpy() * 100
        err_low = (grp["accuracy"] - grp["ci_low"]).to_numpy() * 100
        err_high = (grp["ci_high"] - grp["accuracy"]).to_numpy() * 100
        ax.bar(xs, y, yerr=[err_low, err_high], capsize=4, color="#4c78a8")
        ax.axhline(20, color="#555555", linestyle="--", linewidth=1, label="chance")
        ax.set_xticks(list(xs))
        ax.set_xticklabels([f"K={k}" for k in grp["k"]])
        ax.set_ylim(0, 36)
        ax.set_title(f"{run_name}: {split}")
        ax.grid(axis="y", alpha=0.25)
    axes[0, 0].set_ylabel("accuracy (%)")
    axes[1, 0].set_ylabel("accuracy (%)")
    fig.tight_layout()
    fig.savefig(FIG / "large_retest_bars.png", dpi=180)
    plt.close(fig)


def write_summary(train_df: pd.DataFrame, retest_df: pd.DataFrame) -> None:
    rows = []
    for run_name in ["Full recurrent", "K=0-only control", "Aux value loss"]:
        sub = train_df[(train_df["run"] == run_name) & (train_df["k_num"] >= 0)]
        for split in ["val", "hard"]:
            s = sub[sub["split"] == split]
            if s.empty:
                continue
            best = s.loc[s["accuracy"].idxmax()]
            for step, grp in s.groupby("step"):
                k0 = grp.loc[grp["k_num"] == 0, "accuracy"]
                rest = grp.loc[grp["k_num"] > 0, "accuracy"]
                if not k0.empty and not rest.empty:
                    rows.append(
                        {
                            "run": run_name,
                            "split": split,
                            "step": int(step),
                            "best_k_gt0_gain_pp": 100 * float(rest.max() - k0.iloc[0]),
                        }
                    )
            rows.append(
                {
                    "run": run_name,
                    "split": split,
                    "step": "best observed",
                    "best_k_gt0_gain_pp": f"best accuracy {best.accuracy*100:.1f}% at step {int(best.step)}, K={int(best.k_num)}",
                }
            )
    gain_df = pd.DataFrame(rows)
    gain_df.to_csv(OUT / "k_gain_summary.csv", index=False)

    with open(OUT / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Latent Fast-Weight Experiment Summary\n\n")
        f.write("## Large Retests\n\n")
        f.write(retest_df.to_markdown(index=False))
        f.write("\n\n## K Gain Summary\n\n")
        f.write(gain_df.to_markdown(index=False))
        f.write("\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    train_df = load_training_runs()
    retest_df = load_retests()
    train_df.to_csv(OUT / "training_accuracy_long.csv", index=False)
    retest_df.to_csv(OUT / "large_retests_long.csv", index=False)
    plot_k_advantage(train_df)
    plot_full_heatmap(train_df)
    plot_retests(retest_df)
    write_summary(train_df, retest_df)
    print(f"Wrote analysis to {OUT}")


if __name__ == "__main__":
    main()
