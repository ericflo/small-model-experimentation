#!/usr/bin/env python3
"""Build standalone Markdown/HTML reports for the Fuyu VM experiment."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("/workspace/experiments/qwen_fuyu_vm_grpo_echo")
RUNS = ROOT / "runs"
ANALYSIS = ROOT / "analysis"
REPORTS = ROOT / "reports"

PILOT_RUNS = {
    "Shaped GRPO + ECHO": "pilot_shaped_echo_s32_20260624",
    "Process Preference": "pilot_process_dpo_s32_20260624",
}


def pct(x: float) -> str:
    return f"{100.0 * float(x):.1f}%"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def read_csv(run: str, name: str) -> pd.DataFrame:
    path = RUNS / run / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def summarize_accuracy() -> pd.DataFrame:
    rows: List[dict] = []
    for label, run in PILOT_RUNS.items():
        metrics = read_csv(run, "metrics.csv")
        if metrics.empty:
            continue
        deployed = metrics[metrics["mode"].isin(["learned", "value_gated", "forced"])]
        for (phase, mode), group in deployed.groupby(["phase", "mode"]):
            rows.append(
                {
                    "arm": label,
                    "phase": phase,
                    "mode": mode,
                    "mean_accuracy": group["accuracy"].mean(),
                    "k8_accuracy": group[group["k"] == 8]["accuracy"].mean(),
                    "false_stop_rate": group["false_stop_rate"].mean(),
                    "mean_steps": group["mean_steps"].mean(),
                }
            )
        oracle = metrics[(metrics["mode"] == "oracle_teacher") & (metrics["k"] == metrics["k"].max())]
        if not oracle.empty:
            rows.append(
                {
                    "arm": label,
                    "phase": "oracle_teacher",
                    "mode": f"k={int(metrics['k'].max())}",
                    "mean_accuracy": oracle["accuracy"].mean(),
                    "k8_accuracy": oracle["accuracy"].mean(),
                    "false_stop_rate": oracle["false_stop_rate"].mean(),
                    "mean_steps": oracle["mean_steps"].mean(),
                }
            )
    return pd.DataFrame(rows)


def save_accuracy_chart(summary: pd.DataFrame) -> Path:
    out = ANALYSIS / "accuracy_by_arm.png"
    plot_df = summary[
        summary["phase"].isin(["bc_policy", "grpo_echo_r1_policy", "process_dpo_r1_policy", "oracle_teacher"])
        & summary["mode"].isin(["learned", "forced", "k=8"])
    ].copy()
    plot_df["label"] = plot_df["arm"] + "\n" + plot_df["phase"].str.replace("_", " ") + "\n" + plot_df["mode"]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = ["#4c78a8" if "bc policy" in x else "#f58518" if "grpo" in x else "#54a24b" if "process" in x else "#b279a2" for x in plot_df["label"]]
    ax.bar(range(len(plot_df)), plot_df["mean_accuracy"], color=colors)
    ax.set_ylabel("Mean accuracy")
    ax.set_ylim(0, max(0.5, float(plot_df["mean_accuracy"].max()) + 0.08))
    ax.set_title("Deployable Accuracy Fell After Both Decision-Optimization Updates")
    ax.set_xticks(range(len(plot_df)))
    ax.set_xticklabels(plot_df["label"], rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(plot_df["mean_accuracy"]):
        ax.text(idx, value + 0.01, pct(value), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def save_k8_chart() -> Path:
    out = ANALYSIS / "k8_split_accuracy.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, (label, run) in zip(axes, PILOT_RUNS.items()):
        metrics = read_csv(run, "metrics.csv")
        data = metrics[(metrics["k"] == 8) & (metrics["mode"].isin(["learned", "forced"]))]
        pivot = data.pivot_table(index="split", columns=["phase", "mode"], values="accuracy").fillna(0.0)
        pivot.plot(kind="bar", ax=ax)
        ax.set_title(label)
        ax.set_ylabel("K=8 accuracy")
        ax.set_ylim(0, 0.36)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle("K=8 Split Accuracy: Updates Did Not Improve Generalization")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def save_train_chart() -> Path:
    out = ANALYSIS / "train_diagnostics.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    for ax, (label, run) in zip(axes, PILOT_RUNS.items()):
        train = read_csv(run, "train_log.csv")
        train["x"] = range(len(train))
        ax.plot(train["x"], train["action_accuracy"], marker="o", label="action")
        ax.plot(train["x"], train["stop_accuracy"], marker="o", label="STOP")
        ax.plot(train["x"], train["next_final_accuracy"], marker="o", label="next-final ECHO")
        ax.set_xticks(train["x"])
        ax.set_xticklabels(train["phase"], rotation=20, ha="right")
        ax.set_ylim(0, 1.0)
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Training Diagnostics: Decision Updates Damaged STOP/Action Calibration")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def save_process_chart() -> Path:
    out = ANALYSIS / "process_signals.png"
    grpo = read_csv(PILOT_RUNS["Shaped GRPO + ECHO"], "rollout_stats.csv")
    proc = read_csv(PILOT_RUNS["Process Preference"], "trajectory_stats.csv")
    rows = []
    if not grpo.empty:
        r = grpo.iloc[-1]
        rows.extend(
            [
                {"signal": "GRPO rollout success", "value": r["success_rate"]},
                {"signal": "GRPO false STOP", "value": r["false_stop_rate"]},
                {"signal": "reachable-after edits", "value": r["reachable_after_rate"]},
                {"signal": "destroyed reachability", "value": r["destroyed_reachability_rate"]},
            ]
        )
    if not proc.empty:
        r = proc[proc["phase"].str.contains("process_dpo")].iloc[-1]
        rows.extend(
            [
                {"signal": "process rollout success", "value": r["rollout_success_rate"]},
                {"signal": "repair found/state", "value": r["repair_found_states"] / max(1, r["search_states"])},
            ]
        )
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(df["signal"], df["value"], color=["#f58518", "#e45756", "#54a24b", "#eeca3b", "#72b7b2", "#b279a2"][: len(df)])
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.0)
    ax.set_title("Process Signals Existed, But Did Not Transfer To Better Deployment")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["signal"], rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(df["value"]):
        ax.text(idx, value + 0.02, pct(value), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def report_tables(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    acc = summary[
        summary["phase"].isin(["bc_policy", "grpo_echo_r1_policy", "process_dpo_r1_policy", "oracle_teacher"])
        & summary["mode"].isin(["learned", "forced", "value_gated", "k=8"])
    ].copy()
    acc["mean_accuracy"] = acc["mean_accuracy"].map(pct)
    acc["k8_accuracy"] = acc["k8_accuracy"].map(pct)
    acc["false_stop_rate"] = acc["false_stop_rate"].map(pct)
    acc["mean_steps"] = acc["mean_steps"].map(lambda x: f"{x:.2f}")
    acc = acc[["arm", "phase", "mode", "mean_accuracy", "k8_accuracy", "false_stop_rate", "mean_steps"]]

    stats_rows = []
    grpo = read_csv(PILOT_RUNS["Shaped GRPO + ECHO"], "rollout_stats.csv")
    if not grpo.empty:
        r = grpo.iloc[-1]
        stats_rows.append(
            {
                "arm": "Shaped GRPO + ECHO",
                "states": int(r["states"]),
                "success/reachability": f"{pct(r['success_rate'])} rollout success; {pct(r['reachable_after_rate'])} reachable-after",
                "failure signal": f"{pct(r['false_stop_rate'])} false STOP; {pct(r['destroyed_reachability_rate'])} destroyed reachability",
            }
        )
    proc = read_csv(PILOT_RUNS["Process Preference"], "trajectory_stats.csv")
    if not proc.empty:
        r = proc[proc["phase"].str.contains("process_dpo")].iloc[-1]
        stats_rows.append(
            {
                "arm": "Process Preference",
                "states": int(r["states"]),
                "success/reachability": f"{pct(r['rollout_success_rate'])} rollout success; {int(r['repair_found_states'])}/{int(r['search_states'])} repair labels",
                "failure signal": f"{pct(r['false_stop_states'] / max(1, r['states']))} false-STOP states; {r['mean_search_candidates']:.0f} candidates/state",
            }
        )
    stats = pd.DataFrame(stats_rows)

    train_rows = []
    for label, run in PILOT_RUNS.items():
        train = read_csv(run, "train_log.csv")
        for _, r in train.iterrows():
            train_rows.append(
                {
                    "arm": label,
                    "phase": r["phase"],
                    "action_acc": pct(r["action_accuracy"]),
                    "stop_acc": pct(r["stop_accuracy"]),
                    "rank_acc": pct(r["rank_pair_accuracy"]),
                    "next_final": pct(r["next_final_accuracy"]),
                    "train_states": int(r["train_states"]),
                }
            )
    train_table = pd.DataFrame(train_rows)
    return acc, stats, train_table


def write_reports() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    summary = summarize_accuracy()
    charts = [
        save_accuracy_chart(summary),
        save_k8_chart(),
        save_train_chart(),
        save_process_chart(),
    ]
    acc_table, stats_table, train_table = report_tables(summary)
    manifest = {}
    for label, run in PILOT_RUNS.items():
        path = RUNS / run / "dataset_manifest.json"
        manifest[label] = json.loads(path.read_text()) if path.exists() else {}

    md = f"""# Qwen Fuyu VM GRPO-ECHO Report

## Summary

This standalone experiment tested whether a Qwen 4B controller can improve a
typed bytecode VM policy when every VM step is one full-model pass over prompt
tokens plus dense VM-state embeddings. The policy never emits natural-language
action tokens. It emits direct structured action/value/ECHO heads only.

The experiment did not pass the scale-up gate. Both decision-optimization arms
created useful-looking process signals, but both reduced deployable accuracy
relative to the behavior-cloned policy. The main failure was not lack of reward
signal; it was that decision updates disturbed action/STOP calibration and
produced worse VM edit trajectories.

## What Ran

- Base model: `Qwen/Qwen3-4B` loaded with 4-bit QLoRA adapters.
- Interface: prompt token embeddings concatenated with dense VM-state tokens via
  `inputs_embeds`; structured heads predicted `STOP`, opcode edits, argument
  edits, solved value, distance, and next-observation ECHO targets.
- Training data per pilot: 32 synthetic train tasks, 16 examples per evaluation
  split, 2 BC epochs, then one decision-optimization round.
- Large checkpoints: `/workspace/large_artifacts/qwen_fuyu_vm_grpo_echo/checkpoints/`.

## Headline Metrics

{md_table(acc_table)}

![Accuracy by arm](../analysis/{charts[0].name})

![K=8 split accuracy](../analysis/{charts[1].name})

## Process Signals

The shaped-GRPO arm produced nonzero rollout reward variance and measurable
reachability signal. The process-preference arm produced many on-policy repair
labels and an 80% pairwise ranking accuracy during training. Neither translated
into better deployment.

{md_table(stats_table)}

![Process signals](../analysis/{charts[3].name})

## Training Diagnostics

{md_table(train_table)}

![Training diagnostics](../analysis/{charts[2].name})

## Interpretation

The dense-state whole-network loop is mechanically viable: Qwen can consume the
VM state as embeddings, predict structured actions, predict next VM observations,
and be trained end to end with QLoRA. The failed gate is the decision update.

Shaped GRPO separated sampled trajectories: first-round sampled rollout success
was 9.4%, reward standard deviation was 0.50, and 56.3% of non-STOP edits
preserved reachability. But after one update, mean learned accuracy fell from
10.0% to 7.1%, and K=8 accuracy collapsed to 0% across the five evaluation
splits in the measured modes. The update reduced false STOP but shifted the
policy toward repeated invalid slot edits.

Process preference had a stronger low-variance teacher: 226 on-policy states,
97 repair-labeled states, and 28.1% rollout success before the update. The
pairwise ranking term reached 80.2% training accuracy. Deployment still fell:
learned mean accuracy went from 9.2% to 7.9%, forced mean accuracy went from
9.2% to 5.8%, and learned false STOP doubled from 7.1% to 14.2%.

## Gate Decision

Do not scale this actor-update recipe. The next serious version should separate
the model's roles:

1. Use Qwen as a value/prior model inside verified beam/search rather than as
   the sole actor.
2. Train process preferences on verifier-labeled states, but evaluate them as
   search heuristics before allowing them to directly mutate the policy.
3. Keep ECHO as an ablation, not a core claim; it learns next-state prediction
   but did not protect decision quality here.
4. Add a shuffled-reward/preference control only after the unshuffled signal
   improves deployment on this small gate.

## Reproduction

Main pilot commands are recoverable from each run's `dataset_manifest.json`.
Primary result directories:

- `runs/pilot_shaped_echo_s32_20260624`
- `runs/pilot_process_dpo_s32_20260624`

"""
    (REPORTS / "qwen_fuyu_vm_grpo_echo_report.md").write_text(md)

    body = "\n".join(
        [
            "<h1>Qwen Fuyu VM GRPO-ECHO Report</h1>",
            "<h2>Summary</h2>",
            "<p>This standalone experiment tested a token-free dense-state VM controller: prompt embeddings plus dense VM-state embeddings in, structured action/value/ECHO heads out.</p>",
            "<p><strong>Result:</strong> the experiment did not pass the scale-up gate. Both decision-optimization arms produced process signal but reduced deployable accuracy after one update.</p>",
            "<h2>Headline Metrics</h2>",
            acc_table.to_html(index=False, escape=True),
            f'<img src="../analysis/{charts[0].name}" alt="Accuracy by arm">',
            f'<img src="../analysis/{charts[1].name}" alt="K=8 split accuracy">',
            "<h2>Process Signals</h2>",
            stats_table.to_html(index=False, escape=True),
            f'<img src="../analysis/{charts[3].name}" alt="Process signals">',
            "<h2>Training Diagnostics</h2>",
            train_table.to_html(index=False, escape=True),
            f'<img src="../analysis/{charts[2].name}" alt="Training diagnostics">',
            "<h2>Interpretation</h2>",
            "<p>The dense-state whole-network loop is mechanically viable, but direct actor updates damaged action and STOP calibration. Shaped GRPO reduced false STOP while producing repeated invalid edits; process preference produced repair labels and strong ranking accuracy but still regressed deployment.</p>",
            "<h2>Gate Decision</h2>",
            "<p>Do not scale this actor-update recipe. The next version should use Qwen as a value/prior model inside verified search, train process preferences as search heuristics, and keep ECHO as an ablation rather than a bundled claim.</p>",
        ]
    )
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Qwen Fuyu VM GRPO-ECHO Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 36px auto; max-width: 1120px; color: #1f2933; line-height: 1.5; }}
    h1, h2 {{ color: #111827; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; font-size: 14px; }}
    th, td {{ border: 1px solid #d0d7de; padding: 7px 9px; text-align: left; }}
    th {{ background: #f3f4f6; }}
    img {{ display: block; width: 100%; max-width: 1050px; margin: 18px 0 32px; border: 1px solid #d0d7de; }}
    strong {{ color: #7c2d12; }}
  </style>
</head>
<body>
{body}
</body>
</html>
"""
    (REPORTS / "qwen_fuyu_vm_grpo_echo_report.html").write_text(html_doc)


if __name__ == "__main__":
    write_reports()
