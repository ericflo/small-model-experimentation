#!/usr/bin/env python3
"""Aggregate and plot Qwen latent beam program compiler runs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("experiments/qwen_latent_beam_program_compiler")
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
CHECKPOINT_ROOT = Path("large_artifacts/qwen_latent_beam_program_compiler/checkpoints")

RUN_ORDER = [
    "pilot_beam4_s120",
    "pilot_latent_beam4_s160",
    "pilot_single_compiler_len8_s300",
    "pilot_single_compiler_len24_unpaired_s600",
    "main_single_compiler_len24_paired_s750",
]

RUN_LABELS = {
    "pilot_beam4_s120": "prompt beams",
    "pilot_latent_beam4_s160": "latent beams",
    "pilot_single_compiler_len8_s300": "single compiler max8",
    "pilot_single_compiler_len24_unpaired_s600": "single compiler max24 mixed",
    "main_single_compiler_len24_paired_s750": "single compiler paired",
}


def parse_split(split: str) -> Dict[str, Any]:
    if "_len" not in split:
        return {"template": split, "length": math.nan}
    template, raw_len = split.rsplit("_len", 1)
    return {"template": template, "length": int(raw_len)}


def load_results() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(RUNS.glob("*/results.json")):
        with path.open() as f:
            data = json.load(f)
        rows.append(data | {"run": path.parent.name})
    return rows


def final_metrics(results: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for data in results:
        run = data["run"]
        for split, metrics in data.get("final_metrics", {}).items():
            parsed = parse_split(split)
            rows.append(
                {
                    "run": run,
                    "run_label": RUN_LABELS.get(run, run),
                    "split": split,
                    **parsed,
                    **metrics,
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        order = {run: idx for idx, run in enumerate(RUN_ORDER)}
        df["run_order"] = df["run"].map(lambda x: order.get(x, len(order)))
        df = df.sort_values(["run_order", "template", "length"])
    return df


def train_log(results: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for data in results:
        run = data["run"]
        for row in data.get("train_log", []):
            rows.append({"run": run, "run_label": RUN_LABELS.get(run, run), **row})
    df = pd.DataFrame(rows)
    if not df.empty:
        order = {run: idx for idx, run in enumerate(RUN_ORDER)}
        df["run_order"] = df["run"].map(lambda x: order.get(x, len(order)))
        df = df.sort_values(["run_order", "step"])
    return df


def run_summary(results: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for data in results:
        args = data.get("args", {})
        meta = data.get("metadata", {})
        rows.append(
            {
                "run": data["run"],
                "run_label": RUN_LABELS.get(data["run"], data["run"]),
                "train_seconds": data.get("train_seconds"),
                "beam_count": data.get("beam_count", args.get("beam_count")),
                "beam_register_mode": data.get("beam_register_mode", args.get("beam_register_mode", "prompt")),
                "prompt_bank_count": data.get("prompt_bank_count", args.get("beam_count")),
                "max_steps": args.get("max_steps"),
                "train_size": args.get("train_size"),
                "eval_size": args.get("eval_size"),
                "train_batch_size": args.get("train_batch_size"),
                "paired_train": args.get("paired_train"),
                "paired_eval": args.get("paired_eval"),
                "curriculum_stages": args.get("curriculum_stages"),
                "gpu_name": meta.get("gpu_name"),
                "gpu_vram_gb": meta.get("gpu_vram_gb"),
            }
        )
    df = pd.DataFrame(rows)
    order = {run: idx for idx, run in enumerate(RUN_ORDER)}
    if not df.empty:
        df["run_order"] = df["run"].map(lambda x: order.get(x, len(order)))
        df = df.sort_values(["run_order", "run"])
    return df


def checkpoint_manifest(results: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for data in results:
        run = data["run"]
        for idx, ckpt in enumerate(data.get("checkpoints", [])):
            rows.append({"run": run, "artifact": f"checkpoint_{idx}", "path": ckpt})
        root = CHECKPOINT_ROOT / run
        if root.exists():
            rows.append({"run": run, "artifact": "checkpoint_dir", "path": str(root)})
    return pd.DataFrame(rows)


def save_bar_accuracy(final_df: pd.DataFrame) -> None:
    focus = final_df[
        final_df["run"].isin(
            [
                "pilot_latent_beam4_s160",
                "pilot_single_compiler_len8_s300",
                "pilot_single_compiler_len24_unpaired_s600",
                "main_single_compiler_len24_paired_s750",
            ]
        )
        & final_df["template"].isin(["standard", "paraphrase", "paired"])
    ].copy()
    if focus.empty:
        return
    templates = ["standard", "paraphrase", "paired"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
    for ax, template in zip(axes, templates):
        cur = focus[focus["template"] == template]
        if cur.empty:
            ax.axis("off")
            continue
        pivot = cur.pivot_table(index="length", columns="run_label", values="selected_accuracy", aggfunc="mean")
        pivot = pivot.sort_index()
        pivot.plot(kind="bar", ax=ax, width=0.82)
        ax.set_title(f"{template} prompts")
        ax.set_xlabel("program length")
        ax.set_ylabel("selected accuracy" if ax is axes[0] else "")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Final answer accuracy by run and prompt family")
    fig.tight_layout()
    fig.savefig(FIGURES / "final_accuracy_by_run_length.png", dpi=180)
    plt.close(fig)


def save_program_vs_answer(final_df: pd.DataFrame) -> None:
    focus = final_df[
        final_df["run"].isin(["pilot_single_compiler_len24_unpaired_s600", "main_single_compiler_len24_paired_s750"])
        & final_df["template"].isin(["standard", "paraphrase", "paired"])
    ].copy()
    if focus.empty:
        return
    focus["label"] = focus["run_label"] + " / " + focus["split"]
    focus = focus.sort_values(["run_order", "template", "length"])
    x = range(len(focus))
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.bar([i - 0.18 for i in x], focus["selected_accuracy"], width=0.36, label="answer accuracy")
    ax.bar([i + 0.18 for i in x], focus["selected_program_exact"], width=0.36, label="exact program")
    ax.set_xticks(list(x))
    ax.set_xticklabels(focus["label"], rotation=60, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("fraction")
    ax.set_title("Answer accuracy tracks exact executable program recovery")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "program_exact_vs_answer_accuracy.png", dpi=180)
    plt.close(fig)


def metric_at(split: str, metric: str) -> str:
    return f"{split}_{metric}"


def save_training_curves(log_df: pd.DataFrame) -> None:
    focus_runs = ["pilot_single_compiler_len24_unpaired_s600", "main_single_compiler_len24_paired_s750"]
    focus = log_df[log_df["run"].isin(focus_runs)].copy()
    if focus.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    for ax, split in zip(axes, ["standard_len24", "paraphrase_len24"]):
        col = metric_at(split, "selected_accuracy")
        for run in focus_runs:
            cur = focus[focus["run"] == run]
            if col in cur:
                ax.plot(cur["step"], cur[col], marker="o", label=RUN_LABELS.get(run, run))
        ax.set_title(split.replace("_", " "))
        ax.set_xlabel("training step")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("selected accuracy")
    axes[1].legend()
    fig.suptitle("Length-24 learning is curriculum- and template-sensitive")
    fig.tight_layout()
    fig.savefig(FIGURES / "length24_training_curves.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for run in focus_runs:
        cur = focus[focus["run"] == run]
        ax.plot(cur["step"], cur["loss"], marker="o", label=RUN_LABELS.get(run, run))
    ax.set_title("Training loss")
    ax.set_xlabel("training step")
    ax.set_ylabel("loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "training_loss_curves.png", dpi=180)
    plt.close(fig)


def save_prefix_heatmap(final_df: pd.DataFrame) -> None:
    focus = final_df[
        final_df["run"].isin(["pilot_single_compiler_len24_unpaired_s600", "main_single_compiler_len24_paired_s750"])
        & final_df["template"].isin(["standard", "paraphrase", "paired"])
    ].copy()
    if focus.empty:
        return
    focus["row"] = focus["run_label"] + " / " + focus["template"]
    pivot = focus.pivot_table(index="row", columns="length", values="selected_state_prefix_fraction", aggfunc="mean")
    pivot = pivot.sort_index()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    image = ax.imshow(pivot.fillna(0).values, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_xlabel("program length")
    ax.set_title("State-prefix correctness")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if val < 0.65 else "black", fontsize=8)
    fig.colorbar(image, ax=ax, label="prefix fraction")
    fig.tight_layout()
    fig.savefig(FIGURES / "state_prefix_heatmap.png", dpi=180)
    plt.close(fig)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    results = load_results()
    final_df = final_metrics(results)
    log_df = train_log(results)
    summary_df = run_summary(results)
    manifest_df = checkpoint_manifest(results)
    final_df.to_csv(REPORTS / "aggregate_final_metrics.csv", index=False)
    log_df.to_csv(REPORTS / "aggregate_train_log.csv", index=False)
    summary_df.to_csv(REPORTS / "run_summary.csv", index=False)
    manifest_df.to_csv(REPORTS / "checkpoint_manifest_all.csv", index=False)
    save_bar_accuracy(final_df)
    save_program_vs_answer(final_df)
    save_training_curves(log_df)
    save_prefix_heatmap(final_df)
    print(f"wrote {REPORTS}")


if __name__ == "__main__":
    main()
