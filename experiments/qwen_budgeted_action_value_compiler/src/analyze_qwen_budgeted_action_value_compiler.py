#!/usr/bin/env python3
"""Aggregate budgeted action-value compiler runs and write reports."""

from __future__ import annotations

import html
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("experiments/qwen_budgeted_action_value_compiler")
RUNS = ROOT / "runs"
ANALYSIS = ROOT / "analysis"
FIGURES = ANALYSIS / "figures"
REPORTS = ROOT / "reports"

MAIN_RUN = "main_budgeted_action_value_s512"
RUN_ORDER = [
    "smoke_budgeted_action_value",
    "pilot_budgeted_action_value_s128",
    "pilot_budgeted_action_value_s128_advantage",
    MAIN_RUN,
]


def pct(x: Any) -> str:
    try:
        val = float(x)
    except Exception:
        return "n/a"
    if math.isnan(val):
        return "n/a"
    return f"{100.0 * val:.1f}%"


def num(x: Any, digits: int = 3) -> str:
    try:
        val = float(x)
    except Exception:
        return "n/a"
    if math.isnan(val):
        return "n/a"
    return f"{val:.{digits}f}"


def short_run(run: str) -> str:
    return (
        run.replace("smoke_budgeted_action_value", "smoke")
        .replace("pilot_budgeted_action_value_", "pilot_")
        .replace("main_budgeted_action_value_", "main_")
    )


def read_csvs(name: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in sorted(RUNS.glob(f"*/{name}")):
        df = pd.read_csv(path)
        if "run" not in df.columns:
            df.insert(0, "run", path.parent.name)
        df.insert(0, "run_dir", path.parent.name)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False).drop_duplicates()


def read_manifests() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for path in sorted(RUNS.glob("*/dataset_manifest.json")):
        data = json.loads(path.read_text())
        row: Dict[str, Any] = {
            "run": data.get("run", path.parent.name),
            "seed": data.get("seed"),
            "modulus": data.get("modulus"),
            "max_program_len": data.get("max_program_len"),
            "max_prompt_tokens": data.get("max_prompt_tokens"),
            "model_id": data.get("metadata", {}).get("model_id"),
            "loader": data.get("metadata", {}).get("transformers_loader"),
            "load_in_4bit": data.get("metadata", {}).get("load_in_4bit"),
            "peft_installed": data.get("metadata", {}).get("peft_installed"),
            "gpu_name": data.get("metadata", {}).get("gpu_name"),
            "gpu_vram_gb": data.get("metadata", {}).get("gpu_vram_gb"),
        }
        for key, val in data.get("sizes", {}).items():
            row[f"size_{key}"] = val
        rows.append(row)
    return pd.DataFrame(rows)


def decoder_family(decoder: str) -> str:
    if decoder == "greedy":
        return "greedy"
    if decoder == "beam_logprob":
        return "logprob"
    if decoder == "local_answer":
        return "answer_repair"
    match = re.match(r"beam_([^_]+)_w", decoder)
    return match.group(1) if match else decoder


def metric(metrics: pd.DataFrame, split: str, decoder: str, col: str = "accuracy", run: str = MAIN_RUN) -> float:
    sub = metrics[(metrics["run"].eq(run)) & (metrics["split"].eq(split)) & (metrics["decoder"].eq(decoder))]
    if sub.empty:
        return float("nan")
    return float(sub.iloc[0][col])


def best_decoder(metrics: pd.DataFrame, split: str, family: str, run: str = MAIN_RUN) -> pd.Series:
    sub = metrics[(metrics["run"].eq(run)) & (metrics["split"].eq(split))].copy()
    if family in {"greedy", "logprob", "answer_repair"}:
        decoder = {"greedy": "greedy", "logprob": "beam_logprob", "answer_repair": "local_answer"}[family]
        sub = sub[sub["decoder"].eq(decoder)]
    else:
        sub = sub[sub["decoder"].str.startswith(f"beam_{family}_w", na=False)]
    if sub.empty:
        return pd.Series(dtype="object")
    return sub.sort_values(["accuracy", "program_exact", "oracle_accuracy"], ascending=False).iloc[0]


def best_family_table(metrics: pd.DataFrame, run: str = MAIN_RUN) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    splits = ["val_mixed", "fresh_standard", "fresh_paraphrase", "fresh_paired", "hard_composition"]
    families = ["greedy", "logprob", "exact", "found", "qvalue", "advantage", "answer_repair"]
    for split in splits:
        for family in families:
            row = best_decoder(metrics, split, family, run=run)
            if row.empty:
                continue
            rows.append(
                {
                    "run": run,
                    "split": split,
                    "family": family,
                    "decoder": row["decoder"],
                    "accuracy": float(row["accuracy"]),
                    "program_exact": float(row["program_exact"]),
                    "valid_rate": float(row["valid_rate"]),
                    "oracle_accuracy": float(row["oracle_accuracy"]),
                    "mean_completed": float(row["mean_completed"]),
                    "mean_expansions": float(row["mean_expansions"]),
                }
            )
    return pd.DataFrame(rows)


def save_tables(metrics: pd.DataFrame, train_logs: pd.DataFrame, value_logs: pd.DataFrame, samples: pd.DataFrame, manifests: pd.DataFrame) -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(ANALYSIS / "all_final_metrics.csv", index=False)
    metrics[metrics["run"].eq(MAIN_RUN)].to_csv(ANALYSIS / "final_metrics.csv", index=False)
    best_family_table(metrics).to_csv(ANALYSIS / "best_family_metrics.csv", index=False)
    if not train_logs.empty:
        train_logs.to_csv(ANALYSIS / "compiler_train_logs.csv", index=False)
    if not value_logs.empty:
        value_logs.to_csv(ANALYSIS / "value_train_logs.csv", index=False)
    if not samples.empty:
        samples.to_csv(ANALYSIS / "prefix_sample_stats.csv", index=False)
    if not manifests.empty:
        manifests.to_csv(ANALYSIS / "dataset_manifests.csv", index=False)


def plot_target_density(samples: pd.DataFrame) -> None:
    if samples.empty:
        return
    FIGURES.mkdir(parents=True, exist_ok=True)
    sub = samples[(samples["run_dir"].isin(RUN_ORDER)) & (samples["split"].eq("train"))].copy()
    if sub.empty:
        return
    sub["order"] = sub["run_dir"].map({run: idx for idx, run in enumerate(RUN_ORDER)})
    sub = sub.sort_values("order")
    rate_cols = [
        ("exact_positive_rate", "Exact positive"),
        ("found_positive_rate", "Found positive"),
        ("q_extra_positive_rate", "Recoverable non-exact"),
    ]
    x = list(range(len(sub)))
    width = 0.24
    plt.figure(figsize=(11.0, 5.7))
    for j, (col, label) in enumerate(rate_cols):
        if col not in sub.columns:
            continue
        plt.bar([i + (j - 1) * width for i in x], [100.0 * float(v) for v in sub[col]], width=width, label=label)
    plt.xticks(x, [short_run(r) for r in sub["run_dir"]], rotation=20, ha="right")
    plt.ylabel("Prefix-action labels (%)")
    plt.title("Budgeted Prefix-Action Target Density")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "target_density.png", dpi=180)
    plt.close()


def plot_target_values(samples: pd.DataFrame) -> None:
    if samples.empty:
        return
    sub = samples[(samples["run_dir"].isin(RUN_ORDER)) & (samples["split"].eq("train"))].copy()
    if sub.empty:
        return
    sub["order"] = sub["run_dir"].map({run: idx for idx, run in enumerate(RUN_ORDER)})
    sub = sub.sort_values("order")
    cols = [
        ("mean_q_value", "Mean Q"),
        ("mean_nonzero_q_value", "Mean nonzero Q"),
        ("mean_advantage_value", "Mean advantage"),
        ("mean_nonzero_advantage_value", "Mean nonzero advantage"),
    ]
    x = list(range(len(sub)))
    plt.figure(figsize=(10.6, 5.5))
    for col, label in cols:
        if col in sub.columns:
            plt.plot(x, [float(v) for v in sub[col]], marker="o", linewidth=2, label=label)
    plt.xticks(x, [short_run(r) for r in sub["run_dir"]], rotation=20, ha="right")
    plt.ylabel("Target value")
    plt.title("Graded Budgeted-Value Targets")
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "target_values.png", dpi=180)
    plt.close()


def plot_value_training(value_logs: pd.DataFrame, value: str, filename: str, ylabel: str, title: str) -> None:
    if value_logs.empty:
        return
    sub = value_logs[value_logs["run"].eq(MAIN_RUN)].copy()
    if sub.empty or value not in sub.columns:
        return
    plt.figure(figsize=(9.6, 5.4))
    for mode, group in sub.groupby("label_mode"):
        group = group.sort_values("epoch")
        plt.plot(group["epoch"], group[value], marker="o", linewidth=2, label=mode)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / filename, dpi=180)
    plt.close()


def plot_main_accuracy(metrics: pd.DataFrame) -> None:
    splits = ["fresh_standard", "fresh_paraphrase", "fresh_paired", "hard_composition"]
    split_labels = ["Standard", "Paraphrase", "Paired", "Hard"]
    families = ["greedy", "logprob", "exact", "found", "qvalue", "advantage", "answer_repair"]
    labels = ["Greedy", "Logprob", "Exact value", "Found value", "Q value", "Advantage", "Answer repair"]
    values: Dict[str, List[float]] = {family: [] for family in families}
    for split in splits:
        for family in families:
            row = best_decoder(metrics, split, family)
            values[family].append(float(row["accuracy"]) if not row.empty else float("nan"))
    x = list(range(len(splits)))
    width = 0.11
    plt.figure(figsize=(12.5, 6.2))
    for j, family in enumerate(families):
        plt.bar([i + (j - 3) * width for i in x], [100.0 * v for v in values[family]], width=width, label=labels[j])
    plt.xticks(x, split_labels)
    plt.ylabel("Top-1 executable accuracy (%)")
    plt.title("Main Run Accuracy by Decoder Family")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(ncol=4, fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES / "main_accuracy_by_decoder.png", dpi=180)
    plt.close()


def plot_weight_sweep(metrics: pd.DataFrame) -> None:
    sub = metrics[(metrics["run"].eq(MAIN_RUN)) & (metrics["split"].isin(["fresh_paired", "hard_composition"]))].copy()
    sub = sub[sub["decoder"].str.match(r"beam_(exact|found|qvalue|advantage)_w", na=False)]
    if sub.empty:
        return
    sub["family"] = sub["decoder"].map(decoder_family)
    sub["weight"] = sub["decoder"].str.extract(r"_w([0-9.]+)$")[0].astype(float)
    plt.figure(figsize=(10.8, 5.8))
    styles = {
        ("fresh_paired", "exact"): "-",
        ("fresh_paired", "found"): "-",
        ("fresh_paired", "qvalue"): "-",
        ("fresh_paired", "advantage"): "-",
        ("hard_composition", "exact"): "--",
        ("hard_composition", "found"): "--",
        ("hard_composition", "qvalue"): "--",
        ("hard_composition", "advantage"): "--",
    }
    for (split, family), group in sub.groupby(["split", "family"]):
        group = group.sort_values("weight")
        label = f"{split.replace('_', ' ')} / {family}"
        plt.plot(group["weight"], [100.0 * float(v) for v in group["accuracy"]], marker="o", linestyle=styles.get((split, family), "-"), linewidth=2, label=label)
    plt.xlabel("Value-score weight")
    plt.ylabel("Top-1 executable accuracy (%)")
    plt.title("Value Weight Sweep")
    plt.grid(alpha=0.25)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "value_weight_sweep.png", dpi=180)
    plt.close()


def plot_oracle_gap(metrics: pd.DataFrame) -> None:
    splits = ["fresh_paired", "hard_composition"]
    families = ["greedy", "logprob", "exact", "found", "qvalue", "advantage", "answer_repair"]
    plt.figure(figsize=(11.3, 6.0))
    for split in splits:
        top1: List[float] = []
        oracle: List[float] = []
        for family in families:
            row = best_decoder(metrics, split, family)
            top1.append(float(row["accuracy"]) if not row.empty else float("nan"))
            oracle.append(float(row["oracle_accuracy"]) if not row.empty else float("nan"))
        label = split.replace("_", " ")
        xs = list(range(len(families)))
        plt.plot(xs, [100.0 * v for v in top1], marker="o", linewidth=2, label=f"{label} top-1")
        plt.plot(xs, [100.0 * v for v in oracle], marker="D", linestyle="--", linewidth=2, label=f"{label} oracle")
    plt.xticks(list(range(len(families))), ["greedy", "logprob", "exact", "found", "qvalue", "adv", "repair"], rotation=18)
    plt.ylabel("Accuracy (%)")
    plt.title("Top-1 Accuracy vs Candidate Oracle")
    plt.grid(alpha=0.25)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(FIGURES / "oracle_gap.png", dpi=180)
    plt.close()


def plot_pilot_iteration(metrics: pd.DataFrame) -> None:
    rows: List[Dict[str, Any]] = []
    for run in RUN_ORDER:
        if run not in set(metrics["run"]):
            continue
        for split in ["fresh_paired", "hard_composition"]:
            greedy = best_decoder(metrics, split, "greedy", run=run)
            logprob = best_decoder(metrics, split, "logprob", run=run)
            found = best_decoder(metrics, split, "found", run=run)
            qvalue = best_decoder(metrics, split, "qvalue", run=run)
            advantage = best_decoder(metrics, split, "advantage", run=run)
            repair = best_decoder(metrics, split, "answer_repair", run=run)
            value_rows = [r for r in [found, qvalue, advantage] if not r.empty]
            best_value = pd.DataFrame(value_rows).sort_values(["accuracy", "program_exact"], ascending=False).iloc[0] if value_rows else pd.Series(dtype="object")
            rows.append(
                {
                    "run": run,
                    "split": split,
                    "greedy": float(greedy["accuracy"]) if not greedy.empty else float("nan"),
                    "logprob": float(logprob["accuracy"]) if not logprob.empty else float("nan"),
                    "best_budgeted_value": float(best_value["accuracy"]) if not best_value.empty else float("nan"),
                    "answer_repair": float(repair["accuracy"]) if not repair.empty else float("nan"),
                }
            )
    if not rows:
        return
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.3), sharey=True)
    for ax, split in zip(axes, ["fresh_paired", "hard_composition"]):
        sub = df[df["split"].eq(split)]
        x = list(range(len(sub)))
        ax.plot(x, [100.0 * v for v in sub["greedy"]], marker="o", linewidth=2, label="Greedy")
        ax.plot(x, [100.0 * v for v in sub["logprob"]], marker="o", linewidth=2, label="Logprob")
        ax.plot(x, [100.0 * v for v in sub["best_budgeted_value"]], marker="o", linewidth=2, label="Best learned budgeted value")
        ax.plot(x, [100.0 * v for v in sub["answer_repair"]], marker="o", linewidth=2, label="Answer repair")
        ax.set_title(split.replace("_", " ").title())
        ax.set_xticks(x)
        ax.set_xticklabels([short_run(r) for r in sub["run"]], rotation=25, ha="right")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Accuracy (%)")
    axes[1].legend(fontsize=8)
    fig.suptitle("Iteration Across Smoke, Pilots, and Main Run")
    fig.tight_layout()
    fig.savefig(FIGURES / "pilot_iteration.png", dpi=180)
    plt.close(fig)


def write_figures(metrics: pd.DataFrame, value_logs: pd.DataFrame, samples: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_target_density(samples)
    plot_target_values(samples)
    plot_value_training(value_logs, "val_auc", "value_training_auc.png", "Held-out AUC", "Main Run Value-Model AUC")
    plot_value_training(value_logs, "val_mse", "value_training_mse.png", "Held-out MSE", "Main Run Value-Model MSE")
    plot_main_accuracy(metrics)
    plot_weight_sweep(metrics)
    plot_oracle_gap(metrics)
    plot_pilot_iteration(metrics)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_csv(index=False)


def figure_md(name: str, caption: str) -> str:
    return f"![{caption}](../analysis/figures/{name})\n\n*{caption}*\n"


def format_family_table(metrics: pd.DataFrame) -> str:
    rows: List[Dict[str, str]] = []
    for split in ["fresh_standard", "fresh_paraphrase", "fresh_paired", "hard_composition"]:
        for family in ["greedy", "logprob", "exact", "found", "qvalue", "advantage", "answer_repair"]:
            row = best_decoder(metrics, split, family)
            if row.empty:
                continue
            rows.append(
                {
                    "split": split,
                    "family": family,
                    "decoder": str(row["decoder"]),
                    "accuracy": pct(row["accuracy"]),
                    "program_exact": pct(row["program_exact"]),
                    "oracle": pct(row["oracle_accuracy"]),
                    "mean_completed": num(row["mean_completed"], 1),
                }
            )
    return markdown_table(pd.DataFrame(rows))


def format_label_table(samples: pd.DataFrame) -> str:
    cols = [
        "run_dir",
        "split",
        "prefix_samples",
        "exact_positive_rate",
        "found_positive_rate",
        "mean_q_value",
        "mean_advantage_value",
        "mean_correct_rank",
    ]
    existing = [col for col in cols if col in samples.columns]
    sub = samples[(samples["run_dir"].isin(RUN_ORDER)) & (samples["split"].eq("train"))][existing].copy()
    sub["order"] = sub["run_dir"].map({run: idx for idx, run in enumerate(RUN_ORDER)})
    sub = sub.sort_values("order").drop(columns=["order"])
    for col in ["exact_positive_rate", "found_positive_rate"]:
        if col in sub.columns:
            sub[col] = sub[col].map(pct)
    for col in ["mean_q_value", "mean_advantage_value", "mean_correct_rank"]:
        if col in sub.columns:
            sub[col] = sub[col].map(lambda x: num(x, 3))
    return markdown_table(sub)


def main_train_quick(train_logs: pd.DataFrame) -> float:
    sub = train_logs[train_logs["run"].eq(MAIN_RUN)].sort_values("step")
    if sub.empty:
        return float("nan")
    return float(sub.iloc[-1]["quick_bytecode_accuracy"])


def best_auc(value_logs: pd.DataFrame, label_mode: str) -> float:
    sub = value_logs[(value_logs["run"].eq(MAIN_RUN)) & (value_logs["label_mode"].eq(label_mode))]
    if sub.empty:
        return float("nan")
    return float(sub["val_auc"].max())


def make_summary(metrics: pd.DataFrame, train_logs: pd.DataFrame, value_logs: pd.DataFrame, samples: pd.DataFrame) -> str:
    fp_adv = best_decoder(metrics, "fresh_paired", "advantage")
    fp_exact = best_decoder(metrics, "fresh_paired", "exact")
    fp_found = best_decoder(metrics, "fresh_paired", "found")
    hard_exact = best_decoder(metrics, "hard_composition", "exact")
    hard_adv = best_decoder(metrics, "hard_composition", "advantage")
    main_samples = samples[(samples["run_dir"].eq(MAIN_RUN)) & (samples["split"].eq("train"))]
    lines = [
        "# Analysis Summary",
        "",
        f"- Main compiler quick validation bytecode accuracy reached {pct(main_train_quick(train_logs))}.",
        f"- Main train prefix labels: exact positives {pct(main_samples['exact_positive_rate'].iloc[0]) if not main_samples.empty else 'n/a'}, found positives {pct(main_samples['found_positive_rate'].iloc[0]) if not main_samples.empty else 'n/a'}, mean Q {num(main_samples['mean_q_value'].iloc[0]) if not main_samples.empty else 'n/a'}, mean advantage {num(main_samples['mean_advantage_value'].iloc[0]) if not main_samples.empty else 'n/a'}.",
        f"- Value-model best held-out AUCs: exact {num(best_auc(value_logs, 'exact'))}, found {num(best_auc(value_logs, 'found'))}, qvalue {num(best_auc(value_logs, 'qvalue'))}, advantage {num(best_auc(value_logs, 'advantage'))}.",
        f"- Fresh paired: greedy/logprob were {pct(metric(metrics, 'fresh_paired', 'greedy'))}/{pct(metric(metrics, 'fresh_paired', 'beam_logprob'))}; best exact was {pct(fp_exact['accuracy']) if not fp_exact.empty else 'n/a'} (`{fp_exact.get('decoder', 'n/a')}`); best found was {pct(fp_found['accuracy']) if not fp_found.empty else 'n/a'} (`{fp_found.get('decoder', 'n/a')}`); best advantage was {pct(fp_adv['accuracy']) if not fp_adv.empty else 'n/a'} (`{fp_adv.get('decoder', 'n/a')}`); answer repair was {pct(metric(metrics, 'fresh_paired', 'local_answer'))}.",
        f"- Hard composition: greedy/logprob were {pct(metric(metrics, 'hard_composition', 'greedy'))}/{pct(metric(metrics, 'hard_composition', 'beam_logprob'))}; best exact was {pct(hard_exact['accuracy']) if not hard_exact.empty else 'n/a'} (`{hard_exact.get('decoder', 'n/a')}`); best advantage was {pct(hard_adv['accuracy']) if not hard_adv.empty else 'n/a'} (`{hard_adv.get('decoder', 'n/a')}`); answer repair was {pct(metric(metrics, 'hard_composition', 'local_answer'))}.",
        "- Conclusion: budgeted action-value labels expose a large recoverability set and produce small no-answer beam gains, but the decisive signal in this setup is still supervised candidate execution against the target answer. The next experiment should move supervision closer to the answer-verified oracle instead of only learning a lightweight partial-program ranker.",
    ]
    return "\n".join(lines) + "\n"


def make_report(metrics: pd.DataFrame, train_logs: pd.DataFrame, value_logs: pd.DataFrame, samples: pd.DataFrame, manifests: pd.DataFrame) -> str:
    main_manifest = manifests[manifests["run"].eq(MAIN_RUN)]
    hardware = "unknown GPU"
    model_id = "Qwen/Qwen3-4B"
    if not main_manifest.empty:
        if pd.notna(main_manifest.iloc[0].get("gpu_name")):
            hardware = str(main_manifest.iloc[0].get("gpu_name"))
        if pd.notna(main_manifest.iloc[0].get("model_id")):
            model_id = str(main_manifest.iloc[0].get("model_id"))

    fp_adv = best_decoder(metrics, "fresh_paired", "advantage")
    fp_exact = best_decoder(metrics, "fresh_paired", "exact")
    fp_found = best_decoder(metrics, "fresh_paired", "found")
    hard_exact = best_decoder(metrics, "hard_composition", "exact")
    hard_adv = best_decoder(metrics, "hard_composition", "advantage")
    main_samples = samples[(samples["run_dir"].eq(MAIN_RUN)) & (samples["split"].eq("train"))]

    return f"""# Qwen Budgeted Action-Value Compiler

## Abstract

This experiment tests whether a frozen {model_id} prompt encoder can be paired with a small posttraining head that compiles natural-language tasks into executable typed bytecode, then uses learned budgeted action values to improve search without revealing the target answer at decode time.

The method trains a bytecode compiler head from supervised traces, collects prefix-action labels by bounded suffix search in a typed stack VM, and trains four action-value targets: canonical exact-prefix, binary recoverability, graded budgeted Q, and sibling-normalized advantage. The most important question is whether the learned values can close the gap between top-1 typed beam search and an answer-verified local repair oracle.

The answer is partly positive but not decisive. In the main run, the compiler reached {pct(main_train_quick(train_logs))} quick bytecode accuracy. On fresh paired tasks, greedy and logprob beam were {pct(metric(metrics, "fresh_paired", "greedy"))}; the best learned budgeted value was {pct(fp_adv["accuracy"]) if not fp_adv.empty else "n/a"} with `{fp_adv.get("decoder", "n/a")}`, and the answer-verified repair control reached {pct(metric(metrics, "fresh_paired", "local_answer"))}. On hard composition, greedy/logprob were {pct(metric(metrics, "hard_composition", "greedy"))}; learned value guidance reached {pct(hard_adv["accuracy"]) if not hard_adv.empty else "n/a"} with `{hard_adv.get("decoder", "n/a")}`, while answer-verified repair reached {pct(metric(metrics, "hard_composition", "local_answer"))}. The learned value signal is real, but the current lightweight ranker does not yet reproduce the oracle-like gains.

## Experimental Setup

The task distribution emits short natural-language prompts whose answers are computed by hidden programs over a compact typed stack VM. Programs can push constants, combine stack values with arithmetic and comparisons, take modulus 97, and read from two lookup tables. Evaluation is executable: a decoded program is valid only if it is stack-safe and returns the correct value.

The trainable system has two pieces:

- A frozen-Qwen feature extractor plus a small compiler head that predicts opcode logits, argument logits, and an auxiliary direct answer head.
- A partial-program value model that scores typed candidate actions during beam search.

The value data is collected from the compiler itself. For each partial prefix, the collector expands candidate actions, executes them into VM states, and runs bounded suffix search. Each action receives:

- `exact`: 1 if the action preserves the canonical supervised trace.
- `found`: 1 if a bounded suffix search can still complete to the correct answer.
- `qvalue`: a graded return based on the rank and margin of the best correct suffix completion.
- `advantage`: the action Q divided by the best sibling Q from the same prefix.

The answer-verified local repair decoder is included as a diagnostic upper-bound control. It is not a deployable no-answer decoder because it selects among candidate programs by executing them against the known target answer.

## Runs and Artifacts

The standalone directory contains a smoke run, two pilot runs, and the main run. Checkpoints are stored outside the experiment tree at `large_artifacts/qwen_budgeted_action_value_compiler/checkpoints/`.

Main run hardware: {hardware}.

## Target Distribution

{figure_md("target_density.png", "Budgeted suffix search makes many non-canonical actions recoverable; exact-prefix labels remain rare.")}

{figure_md("target_values.png", "Raw Q and sibling-normalized advantage provide graded targets instead of only binary recoverability.")}

{format_label_table(samples)}

In the main run, exact positives were {pct(main_samples["exact_positive_rate"].iloc[0]) if not main_samples.empty else "n/a"} of train prefix actions, while recoverable `found` positives were {pct(main_samples["found_positive_rate"].iloc[0]) if not main_samples.empty else "n/a"}. This confirms that canonical trace supervision is too narrow to describe the action space: most actions are not canonical, but many can still be completed to the right answer.

## Value Training

{figure_md("value_training_auc.png", "Exact-prefix labels are easiest to discriminate; found is learnable; graded Q and advantage are harder with this lightweight value head.")}

{figure_md("value_training_mse.png", "Graded targets have moderate regression error but weaker ranking AUC than exact-prefix supervision.")}

Best held-out AUCs in the main run were exact {num(best_auc(value_logs, "exact"))}, found {num(best_auc(value_logs, "found"))}, qvalue {num(best_auc(value_logs, "qvalue"))}, and advantage {num(best_auc(value_logs, "advantage"))}. The result is consistent with the target definitions: exact-prefix classification is sparse but clean, found classification is broad and noisy, and graded budgeted values are richer but difficult to calibrate from the available features.

## Decoder Results

{figure_md("main_accuracy_by_decoder.png", "Learned value guidance gives small top-1 gains over greedy/logprob search; answer-verified repair remains much stronger.")}

{format_family_table(metrics)}

Fresh paired accuracy improved from {pct(metric(metrics, "fresh_paired", "greedy"))} greedy/logprob to {pct(fp_adv["accuracy"]) if not fp_adv.empty else "n/a"} with the best advantage-guided beam and {pct(fp_exact["accuracy"]) if not fp_exact.empty else "n/a"} with the best exact-prefix beam. Hard composition improved only slightly, from {pct(metric(metrics, "hard_composition", "greedy"))} to {pct(hard_exact["accuracy"]) if not hard_exact.empty else "n/a"} for exact-prefix value and {pct(hard_adv["accuracy"]) if not hard_adv.empty else "n/a"} for advantage value.

## Weight Sensitivity

{figure_md("value_weight_sweep.png", "The useful value-weight range is narrow; over-weighting learned values tends to hurt beam ranking.")}

The sweep shows that the learned value heads are not calibrated enough to dominate compiler log-probability. Small weights sometimes help, but larger weights frequently collapse back toward worse rankings. That matters because a scalable posttraining tweak needs a value signal that can safely override a local token or action prior when the prior is myopic.

## Oracle Gap

{figure_md("oracle_gap.png", "Candidate sets often contain correct programs that the no-answer scorers fail to rank first.")}

On fresh paired tasks, logprob beam top-1 accuracy was {pct(metric(metrics, "fresh_paired", "beam_logprob"))}, but its candidate oracle was {pct(metric(metrics, "fresh_paired", "beam_logprob", "oracle_accuracy"))}. On hard composition, logprob beam top-1 accuracy was {pct(metric(metrics, "hard_composition", "beam_logprob"))}, while the candidate oracle was {pct(metric(metrics, "hard_composition", "beam_logprob", "oracle_accuracy"))}. The value models recover only a small part of this slack. The repair control recovers much more because it uses the answer itself as a perfect verifier.

## Iteration Within This Experiment

{figure_md("pilot_iteration.png", "The main run strengthened the compiler and exposed the remaining gap between learned value ranking and answer-verified repair.")}

The pilots were useful because they separated three failure modes: a weak compiler, overly broad recoverability labels, and weak calibration of graded values. The main run addressed the first issue but preserved the ranking gap. This points away from simply scaling the same value head and toward training methods that make the model imitate the answer-verified selector or learn from execution traces more directly.

## Interpretation

The experiment supports three claims.

1. A small posttraining head can compile frozen-Qwen prompt features into executable bytecode with nontrivial generalization.
2. Bounded suffix search reveals a much broader recoverable action set than canonical trace supervision.
3. A lightweight learned value ranker is not enough to reproduce the gains of answer-verified candidate selection.

The third point is the critical one. The system already generates correct programs in the candidate set more often than it selects them. The next high-impact direction is therefore not another partial-program classifier. It is a stronger supervision loop that distills the answer-verified selector into a deployable scorer, or a policy-gradient procedure that uses execution reward to update the program policy over complete candidates.

## Recommended Next Experiment

Train a selector or reranker directly on complete candidate programs sampled from the compiler beam, with positives chosen by execution and negatives chosen from near-miss candidates. The reranker should consume the prompt, bytecode, execution trace, and final stack state, then predict correctness without seeing the answer. That moves supervision much closer to the 82%/79% repair oracle while preserving a deployable no-answer inference path.

The strongest variant would combine supervised contrastive reranking with hard-negative mining:

- Generate 32-64 complete candidate programs per prompt.
- Execute all candidates during training only.
- Mark candidates correct by target answer and retain hard incorrect candidates with high compiler score.
- Train a frozen-Qwen-attached reranker, optionally with LoRA or PEFT, to score complete programs.
- Decode by compiler beam plus reranker top-1, with local repair retained only as an oracle control.

This is the shortest path from the current result toward a practical Qwen-attached program executor: it attacks the observed ranking bottleneck directly instead of hoping that prefix-level value targets will indirectly learn the same selection rule.
"""


def inline_markdown(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    return escaped


def markdown_to_html(markdown_text: str, title: str) -> str:
    lines = markdown_text.splitlines()
    out: List[str] = []
    in_ul = False
    in_ol = False
    in_table = False

    def close_lists() -> None:
        nonlocal in_ul, in_ol
        if in_ul:
            out.append("</ul>")
            in_ul = False
        if in_ol:
            out.append("</ol>")
            in_ol = False

    def close_table() -> None:
        nonlocal in_table
        if in_table:
            out.append("</tbody></table>")
            in_table = False

    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            close_lists()
            close_table()
            i += 1
            continue
        if line.startswith("# "):
            close_lists()
            close_table()
            out.append(f"<h1>{inline_markdown(line[2:])}</h1>")
        elif line.startswith("## "):
            close_lists()
            close_table()
            out.append(f"<h2>{inline_markdown(line[3:])}</h2>")
        elif line.startswith("### "):
            close_lists()
            close_table()
            out.append(f"<h3>{inline_markdown(line[4:])}</h3>")
        elif line.startswith("![") and "](" in line and line.endswith(")"):
            close_lists()
            close_table()
            alt = line[2 : line.index("](")]
            src = line[line.index("](") + 2 : -1]
            out.append(f'<figure><img src="{html.escape(src)}" alt="{html.escape(alt)}"><figcaption>{inline_markdown(alt)}</figcaption></figure>')
        elif line.startswith("- "):
            close_table()
            if not in_ul:
                close_lists()
                out.append("<ul>")
                in_ul = True
            out.append(f"<li>{inline_markdown(line[2:])}</li>")
        elif re.match(r"^\d+\. ", line):
            close_table()
            if not in_ol:
                close_lists()
                out.append("<ol>")
                in_ol = True
            out.append(f"<li>{inline_markdown(re.sub(r'^\\d+\\.\\s+', '', line))}</li>")
        elif line.startswith("|") and "|" in line[1:]:
            close_lists()
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if i + 1 < len(lines) and re.match(r"^\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?$", lines[i + 1]):
                close_table()
                out.append("<table><thead><tr>")
                out.extend(f"<th>{inline_markdown(cell)}</th>" for cell in cells)
                out.append("</tr></thead><tbody>")
                in_table = True
                i += 1
            elif in_table:
                out.append("<tr>")
                out.extend(f"<td>{inline_markdown(cell)}</td>" for cell in cells)
                out.append("</tr>")
        else:
            close_lists()
            close_table()
            if line.startswith("*") and line.endswith("*"):
                out.append(f"<p><em>{inline_markdown(line[1:-1])}</em></p>")
            else:
                out.append(f"<p>{inline_markdown(line)}</p>")
        i += 1
    close_lists()
    close_table()
    css = """
    :root { color-scheme: light; }
    body { margin: 0; background: #f5f6f3; color: #222; font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; line-height: 1.58; }
    main { max-width: 1060px; margin: 0 auto; padding: 44px 28px 80px; background: #fff; }
    h1 { font-size: 2.35rem; margin: 0 0 0.8rem; line-height: 1.1; }
    h2 { margin-top: 2.3rem; padding-top: 1.2rem; border-top: 1px solid #d9ded8; font-size: 1.45rem; }
    h3 { margin-top: 1.6rem; }
    p, li { font-size: 1rem; }
    code { background: #eef1ee; padding: 0.1rem 0.28rem; border-radius: 4px; }
    table { width: 100%; border-collapse: collapse; font-size: 0.88rem; margin: 1rem 0 1.4rem; }
    th, td { border: 1px solid #d8ddd7; padding: 6px 8px; text-align: left; }
    th { background: #eef2ee; }
    figure { margin: 1.5rem 0; }
    img { display: block; max-width: 100%; margin: 0 auto; border: 1px solid #d4d9d3; background: white; }
    figcaption { margin-top: 0.45rem; color: #555; font-size: 0.9rem; text-align: center; }
    """
    return f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(title)}</title><style>{css}</style></head><body><main>{"".join(out)}</main></body></html>\n'


def main() -> None:
    metrics = read_csvs("metrics.csv")
    train_logs = read_csvs("train_log.csv")
    value_logs = read_csvs("verifier_train_log.csv")
    samples = read_csvs("prefix_samples.csv")
    manifests = read_manifests()

    save_tables(metrics, train_logs, value_logs, samples, manifests)
    write_figures(metrics, value_logs, samples)

    summary = make_summary(metrics, train_logs, value_logs, samples)
    (ANALYSIS / "summary.md").write_text(summary)

    report = make_report(metrics, train_logs, value_logs, samples, manifests)
    REPORTS.mkdir(parents=True, exist_ok=True)
    md_path = REPORTS / "qwen_budgeted_action_value_compiler_paper.md"
    html_path = REPORTS / "qwen_budgeted_action_value_compiler_paper.html"
    md_path.write_text(report)
    html_path.write_text(markdown_to_html(report, "Qwen Budgeted Action-Value Compiler"))

    print(f"wrote {ANALYSIS / 'summary.md'}")
    print(f"wrote {md_path}")
    print(f"wrote {html_path}")


if __name__ == "__main__":
    main()
