#!/usr/bin/env python3
"""Generate charts and standalone reports for the experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("experiments/qwen_search_augmented_rollout_distillation")
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"
ANALYSIS = ROOT / "analysis"

MAIN_RUN = "main_search_r1_rank00_e2_20260624"
PILOT_RANK = "pilot_search_r2_rank05_20260624"
PILOT_NORANK = "pilot_search_r1_rank00_e2_20260624"

SPLITS = ["val_mixed", "fresh_standard", "fresh_paraphrase", "fresh_paired", "hard_composition"]
SPLIT_LABELS = {
    "val_mixed": "Val mixed",
    "fresh_standard": "Fresh standard",
    "fresh_paraphrase": "Fresh paraphrase",
    "fresh_paired": "Fresh paired",
    "hard_composition": "Hard composition",
}
DEPLOYABLE_MODES = ["learned", "value_gated", "forced"]


def pct(x: float) -> str:
    return f"{100.0 * float(x):.1f}%"


def load_run(run: str) -> Dict[str, pd.DataFrame]:
    base = RUNS / run
    data = {
        "metrics": pd.read_csv(base / "metrics.csv"),
        "train": pd.read_csv(base / "train_log.csv"),
        "traj": pd.read_csv(base / "trajectory_stats.csv"),
    }
    native = base / "native_qwen_metrics.csv"
    if native.exists():
        data["native"] = pd.read_csv(native)
    return data


def active_best(metrics: pd.DataFrame) -> pd.DataFrame:
    active = metrics[metrics["mode"].isin(DEPLOYABLE_MODES) & metrics["k"].gt(0)]
    out = active.groupby(["phase", "split"], as_index=False)["accuracy"].max()
    return out


def main_summary(main: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    metrics = main["metrics"]
    active = active_best(metrics).pivot(index="split", columns="phase", values="accuracy")
    native = main["native"].set_index("split")["accuracy"]
    blank = metrics[metrics["mode"].eq("blank")].drop_duplicates("split").set_index("split")["accuracy"]
    oracle = (
        metrics[metrics["mode"].eq("oracle_teacher")]
        .groupby("split")["accuracy"]
        .max()
    )
    rows = []
    for split in SPLITS:
        rows.append(
            {
                "split": split,
                "native_qwen": float(native.get(split, 0.0)),
                "blank_k0": float(blank.get(split, 0.0)),
                "bc_active_best": float(active.get("bc_policy", pd.Series()).get(split, 0.0)),
                "search_active_best": float(active.get("search_r1_policy", pd.Series()).get(split, 0.0)),
                "oracle_teacher": float(oracle.get(split, 0.0)),
            }
        )
    return pd.DataFrame(rows)


def save_main_summary_chart(summary: pd.DataFrame) -> Path:
    chart_path = ANALYSIS / "main_active_accuracy.png"
    fig, ax = plt.subplots(figsize=(12, 5.8))
    x = range(len(summary))
    width = 0.16
    series = [
        ("native_qwen", "Native Qwen"),
        ("blank_k0", "Blank K=0"),
        ("bc_active_best", "BC VM active"),
        ("search_active_best", "Search VM active"),
        ("oracle_teacher", "Oracle teacher"),
    ]
    offsets = [-2, -1, 0, 1, 2]
    colors = ["#5f6b7a", "#c7a44a", "#2f7d6d", "#5a67d8", "#a23b72"]
    for (col, label), offset, color in zip(series, offsets, colors):
        ax.bar([i + offset * width for i in x], summary[col], width=width, label=label, color=color)
    ax.set_xticks(list(x))
    ax.set_xticklabels([SPLIT_LABELS[s] for s in summary["split"]], rotation=18, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Main Run Accuracy: Active VM Editing vs Baselines")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=160)
    plt.close(fig)
    return chart_path


def save_k_curve_chart(metrics: pd.DataFrame) -> Path:
    chart_path = ANALYSIS / "main_k_curves_learned.png"
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), sharey=True)
    axes = axes.ravel()
    for ax, split in zip(axes, SPLITS):
        for phase, color, label in [
            ("bc_policy", "#2f7d6d", "BC VM"),
            ("search_r1_policy", "#5a67d8", "Search VM"),
        ]:
            sub = metrics[(metrics["split"].eq(split)) & (metrics["phase"].eq(phase)) & (metrics["mode"].eq("learned"))]
            ax.plot(sub["k"], sub["accuracy"], marker="o", label=label, color=color)
        oracle = metrics[(metrics["split"].eq(split)) & (metrics["mode"].eq("oracle_teacher"))]
        ax.plot(oracle["k"], oracle["accuracy"], linestyle="--", color="#a23b72", alpha=0.8, label="Oracle")
        ax.set_title(SPLIT_LABELS[split])
        ax.set_xlabel("K recurrent edits")
        ax.grid(axis="y", alpha=0.25)
    axes[-1].axis("off")
    axes[0].set_ylabel("Accuracy")
    axes[3].set_ylabel("Accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", frameon=False)
    fig.suptitle("Learned Greedy K Curves")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(chart_path, dpi=160)
    plt.close(fig)
    return chart_path


def save_training_chart(train: pd.DataFrame) -> Path:
    chart_path = ANALYSIS / "main_training_metrics.png"
    plot = train.copy()
    plot["step"] = range(1, len(plot) + 1)
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    for col, label, color in [
        ("action_accuracy", "Action", "#2f7d6d"),
        ("arg_accuracy", "Argument", "#c75d2c"),
        ("stop_accuracy", "STOP", "#5a67d8"),
        ("rank_pair_accuracy", "Pair rank", "#a23b72"),
    ]:
        ax.plot(plot["step"], plot[col], marker="o", label=label, color=color)
    ax.set_xticks(plot["step"])
    ax.set_xticklabels([f"{row.phase}\ne{int(row.epoch)}" for row in plot.itertuples()], rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Training Diagnostics")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=4)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=160)
    plt.close(fig)
    return chart_path


def save_repair_chart(traj: pd.DataFrame) -> Path:
    chart_path = ANALYSIS / "main_repair_diagnostics.png"
    search = traj[traj["phase"].eq("search_r1_states")].iloc[0]
    values = {
        "Repair found": search["repair_found_states"] / max(1, search["search_states"]),
        "False stop states": search["false_stop_states"] / max(1, search["states"]),
        "Rollout success": search["rollout_success_rate"],
    }
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(values.keys(), values.values(), color=["#2f7d6d", "#c75d2c", "#5a67d8"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title("Search-State Label Diagnostics")
    ax.grid(axis="y", alpha=0.25)
    for i, v in enumerate(values.values()):
        ax.text(i, v + 0.025, pct(v), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(chart_path, dpi=160)
    plt.close(fig)
    return chart_path


def save_pilot_chart() -> Path:
    chart_path = ANALYSIS / "pilot_comparison.png"
    rows: List[Dict[str, object]] = []
    for run, label in [
        (PILOT_RANK, "Rank 0.5"),
        (PILOT_NORANK, "No rank"),
    ]:
        data = load_run(run)
        best = active_best(data["metrics"])
        for split in SPLITS:
            for phase in ["bc_policy", "search_r1_policy"]:
                match = best[(best["split"].eq(split)) & (best["phase"].eq(phase))]
                rows.append(
                    {
                        "pilot": label,
                        "split": split,
                        "phase": phase,
                        "accuracy": float(match["accuracy"].iloc[0]) if len(match) else 0.0,
                    }
                )
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, pilot in zip(axes, ["Rank 0.5", "No rank"]):
        sub = df[df["pilot"].eq(pilot)]
        x = range(len(SPLITS))
        width = 0.34
        for offset, phase, label, color in [
            (-0.5, "bc_policy", "BC", "#2f7d6d"),
            (0.5, "search_r1_policy", "Search", "#5a67d8"),
        ]:
            vals = [
                float(sub[(sub["split"].eq(split)) & (sub["phase"].eq(phase))]["accuracy"].iloc[0])
                for split in SPLITS
            ]
            ax.bar([i + offset * width for i in x], vals, width=width, label=label, color=color)
        ax.set_title(pilot)
        ax.set_xticks(list(x))
        ax.set_xticklabels([SPLIT_LABELS[s].replace(" ", "\n") for s in SPLITS])
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Active best accuracy")
    axes[0].legend(frameon=False)
    fig.suptitle("Pilot Choice: Pairwise Ranking Was Not Helpful")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(chart_path, dpi=160)
    plt.close(fig)
    return chart_path


def format_summary_table(summary: pd.DataFrame) -> pd.DataFrame:
    table = summary.copy()
    table["split"] = table["split"].map(SPLIT_LABELS)
    for col in ["native_qwen", "blank_k0", "bc_active_best", "search_active_best", "oracle_teacher"]:
        table[col] = table[col].map(pct)
    table = table.rename(
        columns={
            "split": "Split",
            "native_qwen": "Native Qwen",
            "blank_k0": "Blank K=0",
            "bc_active_best": "BC VM active",
            "search_active_best": "Search VM active",
            "oracle_teacher": "Oracle teacher",
        }
    )
    return table


def markdown_table(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def write_reports(paths: Dict[str, Path], summary: pd.DataFrame, main: Dict[str, pd.DataFrame]) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    table = format_summary_table(summary)
    train = main["train"].copy()
    traj = main["traj"].copy()
    search_row = traj[traj["phase"].eq("search_r1_states")].iloc[0]
    train_small = train[
        [
            "phase",
            "epoch",
            "action_accuracy",
            "arg_accuracy",
            "stop_accuracy",
            "rank_pair_accuracy",
            "train_states",
        ]
    ].copy()
    for col in ["action_accuracy", "arg_accuracy", "stop_accuracy", "rank_pair_accuracy"]:
        train_small[col] = train_small[col].map(pct)
    train_small = train_small.rename(
        columns={
            "phase": "Phase",
            "epoch": "Epoch",
            "action_accuracy": "Action acc",
            "arg_accuracy": "Arg acc",
            "stop_accuracy": "STOP acc",
            "rank_pair_accuracy": "Pair-rank acc",
            "train_states": "States",
        }
    )

    verdict = (
        "Search-augmented rollout distillation did not improve the main active VM controller. "
        "The behavior-cloned VM controller already produced the best active score on three "
        "of five splits, tied search on one, and search improved only val mixed from 6.2% "
        "to 12.5%, still far below native Qwen. The oracle teacher remained much higher "
        "at 68.8% to 100.0%, so the task and VM action space still have large headroom."
    )

    md = f"""# Search-Augmented Rollout Distillation

## Verdict

{verdict}

The main positive result is narrow: the behavior-cloned VM controller beat native Qwen on fresh paraphrase tasks when using active K>0 edits ({pct(summary.loc[summary.split.eq('fresh_paraphrase'), 'bc_active_best'].iloc[0])} vs {pct(summary.loc[summary.split.eq('fresh_paraphrase'), 'native_qwen'].iloc[0])}). The search-distilled controller did not preserve that gain and did not close the oracle gap.

## What Was Tested

This experiment trains `Qwen/Qwen3-4B` as a recurrent controller for a typed bytecode VM. Each recurrent step is one Qwen forward pass over the task prompt plus dense VM-state tokens. The model predicts one structured edit action or `STOP`; the VM executes the current program; then the updated VM state is fed back into the model for the next step.

The intervention is search-augmented rollout distillation. After behavior cloning on gold traces, the learned policy is rolled out on training tasks. For each policy-visited state, bounded repair search finds a verified completion when possible. The model is then trained on the first action of that repaired trajectory.

## Main Accuracy

These are active K>0 VM-editing scores except for the explicit blank column. This avoids counting passive K=0 blank-program hits as real recurrent computation.

{markdown_table(table)}

![Main active accuracy](../analysis/{paths['main'].name})

## K Curves

The greedy learned policy shows no clean monotonic K-scaling. Some splits improve with more edits, but others stay flat or degrade. The oracle curve confirms that high accuracy is reachable in the same VM environment when the trajectory is chosen correctly.

![K curves](../analysis/{paths['kcurves'].name})

## Training Diagnostics

{markdown_table(train_small)}

![Training metrics](../analysis/{paths['training'].name})

Behavior cloning reached 60.0% local action accuracy, but active solve accuracy remained much lower. Search-distillation collected many verified repair labels, but training on those labels reduced local action accuracy to 50.4% and did not improve the active controller.

## Repair Diagnostics

Search-state collection found verified completions for {int(search_row.repair_found_states)} of {int(search_row.search_states)} policy-visited states ({pct(search_row.repair_found_states / max(1, search_row.search_states))}). It also saw {int(search_row.false_stop_states)} false-stop states and {pct(search_row.rollout_success_rate)} rollout success before retraining.

![Repair diagnostics](../analysis/{paths['repair'].name})

## Pilot Result

The first pilot used pairwise positive-vs-negative ranking with weight 0.5. It did not improve rollout accuracy and made the second on-policy collection worse. The selected main recipe disabled the ranking loss and trained two epochs on search-labeled states.

![Pilot comparison](../analysis/{paths['pilot'].name})

## Interpretation

The experiment gives a clear negative result for this specific recipe. The bottleneck is not simply obtaining verified repair labels: the main run found verified completions for 98.1% of policy-visited states. The problem is turning those labels into a policy that chooses useful global trajectories at inference time.

The strongest evidence is the gap between local and global metrics. Behavior cloning reached 60.0% action accuracy, and the oracle teacher reached 68.8% to 100.0% by split, but the best active deployable VM score was only 37.5%. Search-distillation increased neither the best active score nor the K-scaling pattern.

## Next Experiment

The next high-impact experiment should stop treating repaired trajectories as ordinary one-step imitation labels. The more direct attack is rollout-level optimization: sample complete VM rollouts, score them with exact VM reward and false-stop penalties, then train preference or policy-gradient updates over full trajectories. The assets from this experiment are enough to do that: a recurrent dense-state controller, a value/distance head, exact executable rewards, and a repair procedure that can produce successful contrastive rollouts.
"""

    report_md = REPORTS / "search_augmented_rollout_distillation_report.md"
    report_md.write_text(md)

    html_table = table.to_html(index=False, classes="data")
    train_html = train_small.to_html(index=False, classes="data")
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Search-Augmented Rollout Distillation</title>
  <style>
    body {{ font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; color: #1f2933; background: #f6f7f9; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 32px 20px 56px; }}
    section {{ background: #ffffff; border: 1px solid #d8dee6; border-radius: 8px; padding: 22px 24px; margin: 18px 0; }}
    h1 {{ font-size: 32px; margin: 0 0 10px; letter-spacing: 0; }}
    h2 {{ font-size: 21px; margin: 0 0 12px; letter-spacing: 0; }}
    p {{ line-height: 1.58; }}
    .verdict {{ border-left: 5px solid #5a67d8; }}
    img {{ max-width: 100%; height: auto; display: block; margin: 14px 0 4px; border: 1px solid #d8dee6; border-radius: 6px; }}
    table.data {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
    table.data th, table.data td {{ border: 1px solid #d8dee6; padding: 8px 10px; text-align: right; }}
    table.data th:first-child, table.data td:first-child {{ text-align: left; }}
    table.data th {{ background: #eef2f6; }}
    code {{ background: #eef2f6; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
<main>
  <h1>Search-Augmented Rollout Distillation</h1>
  <section class="verdict">
    <h2>Verdict</h2>
    <p>{verdict}</p>
    <p>The behavior-cloned VM controller beat native Qwen on fresh paraphrase tasks when using active K&gt;0 edits ({pct(summary.loc[summary.split.eq('fresh_paraphrase'), 'bc_active_best'].iloc[0])} vs {pct(summary.loc[summary.split.eq('fresh_paraphrase'), 'native_qwen'].iloc[0])}). The search-distilled controller did not preserve that gain.</p>
  </section>
  <section>
    <h2>What Was Tested</h2>
    <p>This experiment trains <code>Qwen/Qwen3-4B</code> as a recurrent controller for a typed bytecode VM. Each recurrent step is one Qwen forward pass over the task prompt plus dense VM-state tokens. The model predicts one structured edit action or <code>STOP</code>; the VM executes the current program; then the updated VM state is fed back into the model.</p>
    <p>The intervention is search-augmented rollout distillation: after behavior cloning, policy-visited states are repaired by bounded verified search, and the first action of the repaired trajectory becomes new supervision.</p>
  </section>
  <section>
    <h2>Main Accuracy</h2>
    <p>Scores are active K&gt;0 VM-editing scores except for the explicit blank column.</p>
    {html_table}
    <img src="../analysis/{paths['main'].name}" alt="Main active accuracy chart">
  </section>
  <section>
    <h2>K Curves</h2>
    <p>The greedy learned policy shows no clean monotonic K-scaling. The oracle curve confirms that the VM environment has substantial reachable headroom.</p>
    <img src="../analysis/{paths['kcurves'].name}" alt="K curves">
  </section>
  <section>
    <h2>Training Diagnostics</h2>
    {train_html}
    <img src="../analysis/{paths['training'].name}" alt="Training metrics">
  </section>
  <section>
    <h2>Repair Diagnostics</h2>
    <p>Search-state collection found verified completions for {int(search_row.repair_found_states)} of {int(search_row.search_states)} policy-visited states ({pct(search_row.repair_found_states / max(1, search_row.search_states))}). It also saw {int(search_row.false_stop_states)} false-stop states and {pct(search_row.rollout_success_rate)} rollout success before retraining.</p>
    <img src="../analysis/{paths['repair'].name}" alt="Repair diagnostics">
  </section>
  <section>
    <h2>Pilot Result</h2>
    <p>Pairwise ranking with weight 0.5 did not improve rollout accuracy and made the next on-policy collection worse. The main recipe disabled ranking and trained two search epochs.</p>
    <img src="../analysis/{paths['pilot'].name}" alt="Pilot comparison">
  </section>
  <section>
    <h2>Interpretation</h2>
    <p>The bottleneck is not simply obtaining verified repair labels: the main run found verified completions for 98.1% of policy-visited states. The problem is turning those labels into a policy that chooses useful global trajectories at inference time.</p>
    <p>The next high-impact experiment should use rollout-level optimization: sample complete VM rollouts, score them with exact VM reward and false-stop penalties, then train preference or policy-gradient updates over full trajectories.</p>
  </section>
</main>
</body>
</html>
"""
    (REPORTS / "search_augmented_rollout_distillation_report.html").write_text(html)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    main_data = load_run(MAIN_RUN)
    summary = main_summary(main_data)
    paths = {
        "main": save_main_summary_chart(summary),
        "kcurves": save_k_curve_chart(main_data["metrics"]),
        "training": save_training_chart(main_data["train"]),
        "repair": save_repair_chart(main_data["traj"]),
        "pilot": save_pilot_chart(),
    }
    write_reports(paths, summary, main_data)
    summary_out = summary.copy()
    summary_out.to_csv(ANALYSIS / "main_summary.csv", index=False)


if __name__ == "__main__":
    main()
