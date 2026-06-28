#!/usr/bin/env python3
"""Aggregate, plot, and report action-conditioned VM-ECHO policy iteration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import markdown as markdown_lib
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("experiments/qwen_action_conditioned_vm_echo_policy_iteration")
RUNS = ROOT / "runs"
ANALYSIS = ROOT / "analysis"
FIGURES = ANALYSIS / "figures"
REPORTS = ROOT / "reports"
MAIN_RUN = "main_action_vm_echo_s192_thr070"
REPORT_STEM = "qwen_action_conditioned_vm_echo_policy_iteration_report"


def ensure_dirs() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)


def read_csvs(name: str) -> pd.DataFrame:
    frames = []
    for path in sorted(RUNS.glob(f"*/{name}")):
        df = pd.read_csv(path)
        df["run"] = path.parent.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_metadata() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(RUNS.glob("*/dataset_manifest.json")):
        with path.open() as f:
            out[path.parent.name] = json.load(f)
    return out


def pct(x: float) -> str:
    return f"{100.0 * float(x):.1f}%"


def md_table(df: pd.DataFrame, columns: Iterable[str]) -> str:
    cols = list(columns)
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df[cols].iterrows():
        rows.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(rows)


def phase_order(phase: str) -> int:
    order = {
        "seed_supervised": 0,
        "learned_rerank_seed": 1,
        "learned_policy_distill": 2,
        "answer_verified_distill": 3,
        "full_supervised": 4,
    }
    return order.get(phase, 99)


def phase_label(phase: str) -> str:
    return {
        "seed_supervised": "Seed",
        "learned_rerank_seed": "Learned rerank",
        "learned_policy_distill": "Learned distill",
        "answer_verified_distill": "Answer distill",
        "full_supervised": "Full sup.",
    }.get(phase, phase)


def main_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    out = metrics[metrics["run"].eq(MAIN_RUN)].copy()
    out["phase_order"] = out["phase"].map(phase_order)
    return out.sort_values(["phase_order", "split"])


def write_aggregates(metrics: pd.DataFrame, train: pd.DataFrame, ctrain: pd.DataFrame, targets: pd.DataFrame, cstats: pd.DataFrame, ceval: pd.DataFrame) -> None:
    metrics.to_csv(ANALYSIS / "all_metrics.csv", index=False)
    train.to_csv(ANALYSIS / "all_compiler_train_logs.csv", index=False)
    ctrain.to_csv(ANALYSIS / "all_consequence_train_logs.csv", index=False)
    targets.to_csv(ANALYSIS / "all_target_selection.csv", index=False)
    cstats.to_csv(ANALYSIS / "all_candidate_group_stats.csv", index=False)
    ceval.to_csv(ANALYSIS / "all_consequence_selection_eval.csv", index=False)
    main_metrics(metrics).to_csv(ANALYSIS / "main_metrics.csv", index=False)
    for name, df in [
        ("main_compiler_train_log.csv", train),
        ("main_consequence_train_log.csv", ctrain),
        ("main_target_selection.csv", targets),
        ("main_candidate_group_stats.csv", cstats),
        ("main_consequence_selection_eval.csv", ceval),
    ]:
        if not df.empty:
            df[df["run"].eq(MAIN_RUN)].to_csv(ANALYSIS / name, index=False)


def plot_main_accuracy(metrics: pd.DataFrame) -> None:
    df = main_metrics(metrics)
    phases = ["seed_supervised", "learned_policy_distill", "answer_verified_distill", "full_supervised"]
    splits = ["fresh_paired", "hard_composition"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    colors = {"direct_accuracy": "#4C78A8", "answer_search_accuracy": "#F58518"}
    for ax, split in zip(axes, splits):
        sub = df[df["split"].eq(split) & df["phase"].isin(phases)].sort_values("phase_order")
        x = range(len(sub))
        ax.bar([i - 0.18 for i in x], 100 * sub["direct_accuracy"], width=0.36, color=colors["direct_accuracy"], label="Direct")
        ax.bar([i + 0.18 for i in x], 100 * sub["answer_search_accuracy"], width=0.36, color=colors["answer_search_accuracy"], label="Answer search")
        ax.plot(list(x), 100 * sub["oracle_accuracy"], color="#333333", marker="o", linestyle="--", label="Oracle")
        ax.set_xticks(list(x), [phase_label(p) for p in sub["phase"]], rotation=18)
        ax.set_title(split.replace("_", " "))
        ax.set_ylabel("Accuracy (%)")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("Main run compiler accuracy")
    fig.tight_layout()
    fig.savefig(FIGURES / "main_accuracy_by_phase.png", dpi=180)
    plt.close(fig)


def plot_learned_rerank(metrics: pd.DataFrame) -> None:
    df = main_metrics(metrics)
    seed = df[df["phase"].eq("learned_rerank_seed")].copy()
    splits = ["val_mixed", "fresh_standard", "fresh_paraphrase", "fresh_paired", "hard_composition"]
    seed = seed[seed["split"].isin(splits)]
    x = range(len(seed))
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.bar([i - 0.25 for i in x], 100 * seed["direct_accuracy"], width=0.25, label="Compiler top-1", color="#4C78A8")
    ax.bar([i for i in x], 100 * seed["learned_accuracy"], width=0.25, label="Learned selector", color="#F58518")
    ax.bar([i + 0.25 for i in x], 100 * seed["oracle_accuracy"], width=0.25, label="Oracle in candidate set", color="#72B7B2")
    ax.set_xticks(list(x), [s.replace("_", "\n") for s in seed["split"]])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Action-conditioned learned reranking on seed candidates")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "learned_rerank_gap.png", dpi=180)
    plt.close(fig)


def plot_targets(targets: pd.DataFrame) -> None:
    df = targets[targets["run"].eq(MAIN_RUN)].copy()
    if df.empty:
        return
    x = range(len(df))
    fig, ax1 = plt.subplots(figsize=(9, 5.2))
    ax2 = ax1.twinx()
    labels = [str(p).replace("_targets", "") for p in df["phase"]]
    ax1.bar([i - 0.15 for i in x], df["targets"], width=0.3, color="#4C78A8", label="Targets")
    ax2.bar([i + 0.15 for i in x], 100 * df["selected_correct_rate"], width=0.3, color="#F58518", label="Target precision")
    ax1.set_xticks(list(x), labels, rotation=12)
    ax1.set_ylabel("Targets selected")
    ax2.set_ylabel("Known target correctness (%)")
    ax1.set_title("Policy-distillation targets")
    ax1.grid(axis="y", alpha=0.25)
    lines, names = ax1.get_legend_handles_labels()
    lines2, names2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, names + names2, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES / "target_selection.png", dpi=180)
    plt.close(fig)


def plot_consequence_training(ctrain: pd.DataFrame) -> None:
    df = ctrain[ctrain["run"].eq(MAIN_RUN)].copy()
    if df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(df["epoch"], df["quick_base_accuracy"] * 100, marker="o", label="Base top-1")
    axes[0].plot(df["epoch"], df["quick_learned_accuracy"] * 100, marker="o", label="Learned selector")
    axes[0].plot(df["epoch"], df["quick_oracle_accuracy"] * 100, marker="o", linestyle="--", label="Oracle")
    axes[0].set_title("Validation candidate selection")
    axes[0].set_xlabel("Consequence epoch")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(df["epoch"], df["correct_loss"], marker="o", label="Correct CE")
    axes[1].plot(df["epoch"], df["group_loss"], marker="o", label="Group loss")
    axes[1].plot(df["epoch"], df["trace_top_loss"], marker="o", label="Trace top")
    axes[1].set_title("Consequence losses")
    axes[1].set_xlabel("Consequence epoch")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "consequence_training.png", dpi=180)
    plt.close(fig)


def plot_pilot_threshold(metrics: pd.DataFrame, targets: pd.DataFrame) -> None:
    runs = ["pilot_action_vm_echo_s96_v2", "pilot_action_vm_echo_s96_thr070"]
    rows = []
    for run in runs:
        m = metrics[metrics["run"].eq(run) & metrics["phase"].eq("learned_policy_distill") & metrics["split"].isin(["fresh_paired", "hard_composition"])]
        t = targets[targets["run"].eq(run) & targets["phase"].eq("learned_policy_targets")]
        if m.empty or t.empty:
            continue
        threshold = float(t["min_score"].iloc[0])
        for _, row in m.iterrows():
            rows.append({"threshold": threshold, "split": row["split"], "direct": row["direct_accuracy"], "search": row["answer_search_accuracy"], "targets": int(t["targets"].iloc[0]), "precision": float(t["selected_correct_rate"].iloc[0])})
    df = pd.DataFrame(rows)
    if df.empty:
        return
    df.to_csv(ANALYSIS / "pilot_threshold_comparison.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for split in ["fresh_paired", "hard_composition"]:
        sub = df[df["split"].eq(split)]
        axes[0].plot(sub["threshold"], 100 * sub["direct"], marker="o", label=split.replace("_", " "))
    axes[0].set_title("Pilot direct accuracy after learned distill")
    axes[0].set_xlabel("Distillation score threshold")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    by_thr = df.drop_duplicates("threshold")
    axes[1].bar([str(x) for x in by_thr["threshold"]], by_thr["targets"], color="#4C78A8", label="Targets")
    ax2 = axes[1].twinx()
    ax2.plot([str(x) for x in by_thr["threshold"]], 100 * by_thr["precision"], marker="o", color="#F58518", label="Precision")
    axes[1].set_title("Pilot target volume vs precision")
    axes[1].set_xlabel("Distillation score threshold")
    axes[1].set_ylabel("Targets")
    ax2.set_ylabel("Known correctness (%)")
    fig.tight_layout()
    fig.savefig(FIGURES / "pilot_threshold_comparison.png", dpi=180)
    plt.close(fig)


def generate_figures(metrics: pd.DataFrame, targets: pd.DataFrame, ctrain: pd.DataFrame) -> None:
    plot_main_accuracy(metrics)
    plot_learned_rerank(metrics)
    plot_targets(targets)
    plot_consequence_training(ctrain)
    plot_pilot_threshold(metrics, targets)


def make_main_table(metrics: pd.DataFrame) -> pd.DataFrame:
    df = main_metrics(metrics)
    keep = df[df["split"].isin(["fresh_standard", "fresh_paraphrase", "fresh_paired", "hard_composition"])]
    keep = keep[keep["phase"].isin(["seed_supervised", "learned_policy_distill", "answer_verified_distill", "full_supervised"])]
    out = keep[["phase", "split", "direct_accuracy", "answer_search_accuracy", "oracle_accuracy", "learned_accuracy", "program_exact"]].copy()
    out["phase"] = out["phase"].map(phase_label)
    for col in ["direct_accuracy", "answer_search_accuracy", "oracle_accuracy", "learned_accuracy", "program_exact"]:
        out[col] = out[col].map(lambda x: "" if pd.isna(x) else pct(x))
    out.columns = ["Phase", "Split", "Direct", "Answer search", "Oracle", "Learned selector", "Program exact"]
    return out


def report_text(metrics: pd.DataFrame, targets: pd.DataFrame, cstats: pd.DataFrame, ceval: pd.DataFrame, metadata: dict[str, dict[str, Any]]) -> str:
    table = make_main_table(metrics)
    main = main_metrics(metrics)
    manifest = metadata.get(MAIN_RUN, {})
    sizes = manifest.get("sizes", {})
    def row(phase: str, split: str) -> pd.Series:
        return main[main["phase"].eq(phase) & main["split"].eq(split)].iloc[0]
    seed_pair = row("seed_supervised", "fresh_paired")
    learned_pair = row("learned_policy_distill", "fresh_paired")
    answer_pair = row("answer_verified_distill", "fresh_paired")
    full_pair = row("full_supervised", "fresh_paired")
    seed_hard = row("seed_supervised", "hard_composition")
    learned_hard = row("learned_policy_distill", "hard_composition")
    target_main = targets[targets["run"].eq(MAIN_RUN)].copy()
    target_table = target_main[["phase", "targets", "oracle_found_rate", "selected_correct_rate", "selected_valid_rate", "changed_rate", "mean_selected_score"]].copy()
    target_table["phase"] = target_table["phase"].map(lambda x: str(x).replace("_targets", ""))
    for col in ["oracle_found_rate", "selected_correct_rate", "selected_valid_rate", "changed_rate"]:
        target_table[col] = target_table[col].map(lambda x: "" if pd.isna(x) else pct(x))
    target_table["mean_selected_score"] = target_table["mean_selected_score"].map(lambda x: "" if pd.isna(x) else f"{float(x):.3f}")
    cstats_main = cstats[cstats["run"].eq(MAIN_RUN)]
    ceval_main = ceval[ceval["run"].eq(MAIN_RUN)]
    train_candidates = cstats_main[cstats_main["phase"].eq("train_candidates")].iloc[0]
    val_eval = ceval_main[ceval_main["phase"].eq("val_candidates")].iloc[0]
    return f"""# Action-Conditioned VM-ECHO Policy Iteration

## Abstract

This experiment tests a candidate-conditioned route from program search to better direct program emission. A frozen-Qwen compiler first proposes typed bytecode candidates. A consequence model then receives the prompt representation and a candidate program, and learns from VM execution labels: validity, final value, stack trace, and whether that candidate solves the prompt. The learned selector is then used to choose policy-distillation targets for the compiler.

The main result is mixed. The learned consequence selector did not close the oracle gap in the main run: on validation candidates it moved from {pct(val_eval["base_accuracy"])} base top-1 accuracy to {pct(val_eval["learned_accuracy"])}, with an oracle of {pct(val_eval["oracle_accuracy"])}. However, filtered learned-policy distillation still improved the compiler on several generalization splits. Fresh-paired direct accuracy increased from {pct(seed_pair["direct_accuracy"])} to {pct(learned_pair["direct_accuracy"])}, and hard-composition answer-search accuracy increased from {pct(seed_hard["answer_search_accuracy"])} to {pct(learned_hard["answer_search_accuracy"])}. The fully supervised ceiling remained much higher: fresh-paired direct accuracy reached {pct(full_pair["direct_accuracy"])}.

## Setup

- Base model: `{manifest.get("model_name", "Qwen/Qwen3-4B")}`, used only as a frozen hidden-state feature extractor.
- Seed examples: `{sizes.get("seed_train", "?")}`.
- Candidate-training prompts: `{sizes.get("unlabeled_train", "?")}`.
- Full-supervised examples: `{sizes.get("full_supervised_train", "?")}`.
- Fresh split size: `{sizes.get("fresh_standard", "?")}`.
- Candidate search: top-k `{manifest.get("candidate_topk", "?")}`, second-order argument pairs `{manifest.get("max_two_arg_pairs", "?")}`, max candidates `{manifest.get("max_candidates", "?")}`.
- Learned-target threshold: `{manifest.get("distill_min_score", "?")}`.
- Checkpoints: `large_artifacts/qwen_action_conditioned_vm_echo_policy_iteration/checkpoints/{MAIN_RUN}/`.

## Main Results

{md_table(table, table.columns)}

![Main accuracy](../analysis/figures/main_accuracy_by_phase.png)

## Learned Candidate Selection

The action-conditioned selector learned a real but weak signal. It selected mostly valid programs, but it did not reliably select answer-correct programs at main scale.

Training candidate set:

- Groups: `{int(train_candidates["groups"])}` prompts.
- Candidates: `{int(train_candidates["candidates"])}` programs.
- Positive candidate rate: {pct(train_candidates["positive_rate"])}.
- Prompts with at least one positive candidate: {pct(train_candidates["oracle_found_rate"])}.

![Learned rerank gap](../analysis/figures/learned_rerank_gap.png)

## Policy-Distillation Targets

{md_table(target_table, target_table.columns)}

The learned selector chose more targets than answer verification, but with much lower known precision. The useful signal is that even imperfect learned targets improved some direct and search metrics, suggesting that consequence-conditioned filtering is not useless. The limiting factor is selector precision.

![Target selection](../analysis/figures/target_selection.png)

## Training Dynamics

![Consequence training](../analysis/figures/consequence_training.png)

## Threshold Pilot

A small threshold sweep was used to avoid using every learned-selected target. The higher threshold traded volume for precision and produced better pilot direct accuracy.

![Pilot threshold comparison](../analysis/figures/pilot_threshold_comparison.png)

## Interpretation

The candidate-conditioned objective is closer to the desired mechanism than target-trace-only supervision: it asks what a proposed program will do, not only what the correct program should look like. But this implementation still underfits the hardest part: comparing candidate consequences to the prompt-implied answer. The next improvement should strengthen the selector, not the compiler head. Good next changes are pairwise preference training over candidates from the same prompt, harder negative mining, and using the compiler's answer representation directly inside the consequence model.

## Artifacts

- `experiments/qwen_action_conditioned_vm_echo_policy_iteration/runs/{MAIN_RUN}/metrics.csv`
- `experiments/qwen_action_conditioned_vm_echo_policy_iteration/runs/{MAIN_RUN}/target_selection.csv`
- `experiments/qwen_action_conditioned_vm_echo_policy_iteration/runs/{MAIN_RUN}/consequence_train_log.csv`
- `experiments/qwen_action_conditioned_vm_echo_policy_iteration/analysis/main_metrics.csv`
- `experiments/qwen_action_conditioned_vm_echo_policy_iteration/reports/{REPORT_STEM}.md`
- `experiments/qwen_action_conditioned_vm_echo_policy_iteration/reports/{REPORT_STEM}.html`
"""


def write_report(markdown: str) -> None:
    md_path = REPORTS / f"{REPORT_STEM}.md"
    html_path = REPORTS / f"{REPORT_STEM}.html"
    md_path.write_text(markdown)
    rendered = markdown_lib.markdown(markdown, extensions=["tables", "fenced_code"])
    css = """
body { font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; color: #172033; background: #f6f7fb; }
main { max-width: 1120px; margin: 0 auto; padding: 40px 24px 64px; background: white; min-height: 100vh; }
h1 { font-size: 34px; line-height: 1.1; margin-bottom: 12px; color: #111827; }
h2 { color: #111827; margin-top: 34px; border-top: 1px solid #e5e7eb; padding-top: 24px; }
p, li { line-height: 1.55; }
code { background: #eef2f7; padding: 2px 5px; border-radius: 4px; }
table { border-collapse: collapse; width: 100%; margin: 16px 0 24px; font-size: 14px; }
th, td { border: 1px solid #d9e0ea; padding: 8px 10px; text-align: left; }
th { background: #edf2f7; }
img { max-width: 100%; display: block; margin: 18px 0 30px; border: 1px solid #d9e0ea; }
.note { color: #4b5563; font-size: 14px; }
"""
    body = f"<main><p class='note'>Standalone report generated from <code>{MAIN_RUN}</code>.</p>{rendered}</main>"
    html_path.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>Action-Conditioned VM-ECHO Policy Iteration</title><style>{css}</style></head><body>{body}</body></html>")


def write_summary(metrics: pd.DataFrame, targets: pd.DataFrame, ceval: pd.DataFrame) -> None:
    main = main_metrics(metrics)
    def row(phase: str, split: str) -> pd.Series:
        return main[main["phase"].eq(phase) & main["split"].eq(split)].iloc[0]
    seed_pair = row("seed_supervised", "fresh_paired")
    learned_pair = row("learned_policy_distill", "fresh_paired")
    hard_seed = row("seed_supervised", "hard_composition")
    hard_learned = row("learned_policy_distill", "hard_composition")
    val_eval = ceval[ceval["run"].eq(MAIN_RUN) & ceval["phase"].eq("val_candidates")].iloc[0]
    lines = [
        "# Analysis Summary",
        "",
        f"Main run: `{MAIN_RUN}`",
        f"Validation candidate selector: base {pct(val_eval['base_accuracy'])}, learned {pct(val_eval['learned_accuracy'])}, oracle {pct(val_eval['oracle_accuracy'])}.",
        f"Fresh paired direct: seed {pct(seed_pair['direct_accuracy'])}, learned distill {pct(learned_pair['direct_accuracy'])}.",
        f"Fresh paired search: seed {pct(seed_pair['answer_search_accuracy'])}, learned distill {pct(learned_pair['answer_search_accuracy'])}.",
        f"Hard composition search: seed {pct(hard_seed['answer_search_accuracy'])}, learned distill {pct(hard_learned['answer_search_accuracy'])}.",
        "",
        "Figures:",
        "- `analysis/figures/main_accuracy_by_phase.png`",
        "- `analysis/figures/learned_rerank_gap.png`",
        "- `analysis/figures/target_selection.png`",
        "- `analysis/figures/consequence_training.png`",
        "- `analysis/figures/pilot_threshold_comparison.png`",
    ]
    if not targets.empty:
        lines += ["", "Main target selection:", targets[targets["run"].eq(MAIN_RUN)].to_string(index=False)]
    (ANALYSIS / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ensure_dirs()
    metrics = read_csvs("metrics.csv")
    compiler_train = read_csvs("compiler_train_log.csv")
    consequence_train = read_csvs("consequence_train_log.csv")
    targets = read_csvs("target_selection.csv")
    candidate_stats = read_csvs("candidate_group_stats.csv")
    consequence_eval = read_csvs("consequence_selection_eval.csv")
    metadata = load_metadata()
    if metrics.empty:
        raise SystemExit("No metrics.csv files found")
    write_aggregates(metrics, compiler_train, consequence_train, targets, candidate_stats, consequence_eval)
    generate_figures(metrics, targets, consequence_train)
    markdown = report_text(metrics, targets, candidate_stats, consequence_eval, metadata)
    write_report(markdown)
    write_summary(metrics, targets, consequence_eval)
    print(f"[analysis] wrote {REPORTS / (REPORT_STEM + '.md')}")
    print(f"[analysis] wrote {REPORTS / (REPORT_STEM + '.html')}")


if __name__ == "__main__":
    main()
