#!/usr/bin/env python3
"""Aggregate, plot, and report the Qwen complete-program trace reranker experiment."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any

import markdown
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("experiments/qwen_complete_program_trace_reranker")
RUNS = ROOT / "runs"
ANALYSIS = ROOT / "analysis"
FIGURES = ANALYSIS / "figures"
REPORTS = ROOT / "reports"
CHECKPOINT_ROOT = Path("large_artifacts/qwen_complete_program_trace_reranker/checkpoints")
MAIN_RUN = "main_complete_program_trace_reranker_s384_answer"

RUN_LABELS = {
    "smoke_complete_program_trace_reranker": "smoke",
    "pilot_complete_program_trace_reranker_s96": "answer labels",
    "pilot_complete_program_trace_reranker_s96_state": "state labels",
    "pilot_complete_program_trace_reranker_s96_oracle": "single-oracle labels",
    "pilot_complete_program_trace_reranker_s96_repairfocus": "repair-focused oracle",
    MAIN_RUN: "main answer-label run",
}

MAIN_SPLITS = [
    "val_mixed_len6",
    "fresh_standard_len6",
    "fresh_paraphrase_len6",
    "fresh_paired_len6",
    "hard_standard_len8",
    "hard_paraphrase_len8",
    "harder_standard_len10",
    "harder_paraphrase_len10",
]

DOMAIN_SPLITS = [
    "domain_arithmetic_len6",
    "domain_calendar_len6",
    "domain_unit_len6",
    "domain_list_len6",
    "domain_boolean_len6",
    "domain_lookup_len6",
]

ACC_COLUMNS = [
    ("base_executor_accuracy", "base"),
    ("soft_trace_executor_accuracy", "soft trace"),
    ("learned_executor_accuracy", "learned"),
    ("pair_rerank_executor_accuracy", "pair rerank"),
    ("oracle_executor_accuracy", "oracle"),
]


def pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{100.0 * float(value):.1f}%"


def fmt_float(value: Any, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"


def split_label(split: str) -> str:
    return (
        split.replace("_len", " L")
        .replace("fresh_", "fresh ")
        .replace("harder_", "harder ")
        .replace("hard_", "hard ")
        .replace("val_", "validation ")
        .replace("domain_", "")
        .replace("_", " ")
    )


def read_metrics() -> pd.DataFrame:
    frames = []
    for run_dir in sorted(RUNS.iterdir()):
        path = run_dir / "metrics.csv"
        if path.exists():
            frame = pd.read_csv(path)
            if "run" not in frame.columns:
                frame = pd.concat([pd.Series(run_dir.name, index=frame.index, name="run"), frame], axis=1)
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No metrics.csv files under {RUNS}")
    return pd.concat(frames, ignore_index=True)


def read_train_logs() -> pd.DataFrame:
    frames = []
    for run_dir in sorted(RUNS.iterdir()):
        path = run_dir / "verifier_train_log.csv"
        if path.exists():
            frame = pd.read_csv(path)
            frame = pd.concat([pd.Series(run_dir.name, index=frame.index, name="run"), frame], axis=1)
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def read_metadata() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for run_dir in sorted(RUNS.iterdir()):
        path = run_dir / "results.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        out[run_dir.name] = data.get("metadata", {})
    return out


def metric(metrics: pd.DataFrame, run: str, split: str, column: str) -> float | None:
    row = metrics[(metrics["run"] == run) & (metrics["split"] == split)]
    if row.empty or column not in row.columns:
        return None
    value = row.iloc[0][column]
    if pd.isna(value):
        return None
    return float(value)


def main_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = metrics[(metrics["run"] == MAIN_RUN) & (metrics["split"].isin(MAIN_SPLITS + DOMAIN_SPLITS))].copy()
    order = {split: i for i, split in enumerate(MAIN_SPLITS + DOMAIN_SPLITS)}
    rows["_order"] = rows["split"].map(order)
    return rows.sort_values("_order").drop(columns=["_order"])


def write_csvs(metrics: pd.DataFrame, train_logs: pd.DataFrame) -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(ANALYSIS / "all_final_metrics.csv", index=False)
    main_rows(metrics).to_csv(ANALYSIS / "final_metrics.csv", index=False)
    train_logs.to_csv(ANALYSIS / "verifier_train_logs.csv", index=False)


def plot_accuracy_by_split(metrics: pd.DataFrame) -> None:
    rows = main_rows(metrics)
    rows = rows[rows["split"].isin(MAIN_SPLITS)]
    labels = [split_label(s) for s in rows["split"]]
    fig, ax = plt.subplots(figsize=(13, 6.8))
    x = range(len(rows))
    width = 0.16
    offsets = [-2 * width, -width, 0, width, 2 * width]
    for (column, label), offset in zip(ACC_COLUMNS, offsets):
        vals = [100.0 * float(v) if column in rows.columns and not pd.isna(v) else float("nan") for v in rows[column]]
        ax.bar([i + offset for i in x], vals, width=width, label=label)
    ax.set_ylabel("answer accuracy (%)")
    ax.set_title("Main run: candidate selection accuracy by split")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0, 105)
    ax.legend(ncols=5, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "accuracy_by_split.png", dpi=180)
    plt.close(fig)


def plot_oracle_gap(metrics: pd.DataFrame) -> None:
    rows = main_rows(metrics)
    rows = rows[rows["split"].isin(MAIN_SPLITS)]
    labels = [split_label(s) for s in rows["split"]]
    learned = [100.0 * float(v) for v in rows["learned_oracle_gap_recovered"]]
    pair = []
    for _, row in rows.iterrows():
        base = row.get("base_executor_accuracy")
        oracle = row.get("oracle_executor_accuracy")
        pair_acc = row.get("pair_rerank_executor_accuracy")
        if pd.isna(pair_acc) or pd.isna(base) or pd.isna(oracle) or float(oracle) <= float(base):
            pair.append(float("nan"))
        else:
            pair.append(100.0 * (float(pair_acc) - float(base)) / (float(oracle) - float(base)))
    fig, ax = plt.subplots(figsize=(12, 5.8))
    x = range(len(rows))
    ax.axhline(0, color="#333333", linewidth=1)
    ax.bar([i - 0.18 for i in x], learned, width=0.36, label="learned")
    ax.bar([i + 0.18 for i in x], pair, width=0.36, label="pair rerank")
    ax.set_ylabel("oracle gap recovered (%)")
    ax.set_title("How much reachable candidate headroom was captured?")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "oracle_gap_recovered.png", dpi=180)
    plt.close(fig)


def plot_candidate_density(metrics: pd.DataFrame) -> None:
    rows = main_rows(metrics)
    rows = rows[rows["split"].isin(MAIN_SPLITS)]
    labels = [split_label(s) for s in rows["split"]]
    cols = [
        ("avg_candidates", "candidates"),
        ("avg_positive_candidates", "answer-correct"),
        ("avg_state_exact_candidates", "state-exact"),
        ("avg_program_exact_candidates", "program-exact"),
    ]
    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(rows))
    width = 0.2
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    for (column, label), offset in zip(cols, offsets):
        vals = [float(v) if not pd.isna(v) else float("nan") for v in rows[column]]
        ax.bar([i + offset for i in x], vals, width=width, label=label)
    ax.set_ylabel("mean candidates per prompt")
    ax.set_title("Candidate set density")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.legend(ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "candidate_density.png", dpi=180)
    plt.close(fig)


def plot_validation_curves(train_logs: pd.DataFrame) -> None:
    if train_logs.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 6))
    for run, label in RUN_LABELS.items():
        if run == "smoke_complete_program_trace_reranker":
            continue
        rows = train_logs[train_logs["run"] == run]
        if rows.empty:
            continue
        ax.plot(rows["epoch"], 100.0 * rows["val_learned_executor_accuracy"], marker="o", label=label)
    main = train_logs[train_logs["run"] == MAIN_RUN]
    if not main.empty:
        ax.axhline(100.0 * float(main.iloc[0]["val_base_executor_accuracy"]), color="#444444", linestyle="--", label="main base")
        ax.axhline(100.0 * float(main.iloc[0]["val_oracle_executor_accuracy"]), color="#777777", linestyle=":", label="main oracle")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation answer accuracy (%)")
    ax.set_title("Validation accuracy during verifier training")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "validation_curves.png", dpi=180)
    plt.close(fig)


def plot_pilot_comparison(metrics: pd.DataFrame) -> None:
    runs = [
        "pilot_complete_program_trace_reranker_s96",
        "pilot_complete_program_trace_reranker_s96_state",
        "pilot_complete_program_trace_reranker_s96_oracle",
        "pilot_complete_program_trace_reranker_s96_repairfocus",
    ]
    split = "fresh_paired_len6"
    rows = metrics[(metrics["run"].isin(runs)) & (metrics["split"] == split)].copy()
    if rows.empty:
        return
    labels = [RUN_LABELS.get(run, run) for run in rows["run"]]
    cols = [
        ("base_executor_accuracy", "base"),
        ("learned_executor_accuracy", "learned"),
        ("pair_rerank_executor_accuracy", "pair rerank"),
        ("oracle_executor_accuracy", "oracle"),
    ]
    fig, ax = plt.subplots(figsize=(11, 5.8))
    x = range(len(rows))
    width = 0.2
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    for (column, label), offset in zip(cols, offsets):
        vals = [100.0 * float(v) if not pd.isna(v) else float("nan") for v in rows[column]]
        ax.bar([i + offset for i in x], vals, width=width, label=label)
    ax.set_ylabel("fresh paired L6 accuracy (%)")
    ax.set_title("Pilot label objective comparison")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 105)
    ax.legend(ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "pilot_comparison.png", dpi=180)
    plt.close(fig)


def plot_domain_breakdown(metrics: pd.DataFrame) -> None:
    rows = main_rows(metrics)
    rows = rows[rows["split"].isin(DOMAIN_SPLITS)]
    if rows.empty:
        return
    labels = [split_label(s).replace(" L6", "") for s in rows["split"]]
    cols = [
        ("base_executor_accuracy", "base"),
        ("learned_executor_accuracy", "learned"),
        ("oracle_executor_accuracy", "oracle"),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    x = range(len(rows))
    width = 0.25
    offsets = [-width, 0, width]
    for (column, label), offset in zip(cols, offsets):
        vals = [100.0 * float(v) if not pd.isna(v) else float("nan") for v in rows[column]]
        ax.bar([i + offset for i in x], vals, width=width, label=label)
    ax.set_ylabel("answer accuracy (%)")
    ax.set_title("Domain breakdown on length-6 prompts")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 105)
    ax.legend(ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "domain_breakdown.png", dpi=180)
    plt.close(fig)


def generate_figures(metrics: pd.DataFrame, train_logs: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_accuracy_by_split(metrics)
    plot_oracle_gap(metrics)
    plot_candidate_density(metrics)
    plot_validation_curves(train_logs)
    plot_pilot_comparison(metrics)
    plot_domain_breakdown(metrics)


def write_manifest(metadata: dict[str, dict[str, Any]]) -> None:
    rows: list[dict[str, str]] = []
    compiler = CHECKPOINT_ROOT / "fixed_mixed_vm_trace_compiler_s512"
    if compiler.exists():
        rows.append(
            {
                "run": "fixed_mixed_vm_trace_compiler_s512",
                "artifact": "compiler_adapter",
                "path": str(compiler),
                "description": "Frozen Qwen-attached hidden-VM compiler used for all reranker runs.",
            }
        )
    for run_dir in sorted(CHECKPOINT_ROOT.iterdir()) if CHECKPOINT_ROOT.exists() else []:
        ckpt = run_dir / "candidate_trace_verifier.pt"
        if not ckpt.exists():
            continue
        meta = metadata.get(run_dir.name, {})
        args = meta.get("args", {})
        desc = (
            f"Candidate trace verifier checkpoint; positive_label={args.get('positive_label', 'unknown')}, "
            f"train_examples={args.get('train_examples', 'unknown')}, epochs={args.get('verifier_epochs', 'unknown')}."
        )
        rows.append({"run": run_dir.name, "artifact": "candidate_trace_verifier", "path": str(ckpt), "description": desc})
    with (ROOT / "checkpoint_manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "artifact", "path", "description"])
        writer.writeheader()
        writer.writerows(rows)


def make_metrics_table(metrics: pd.DataFrame, splits: list[str]) -> str:
    header = "| split | base | soft trace | learned | pair rerank | oracle | learned gap recovered | avg candidates | answer positives |\n"
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    body = []
    for split in splits:
        row = metrics[(metrics["run"] == MAIN_RUN) & (metrics["split"] == split)]
        if row.empty:
            continue
        r = row.iloc[0]
        body.append(
            "| {split} | {base} | {soft} | {learned} | {pair} | {oracle} | {gap} | {cands} | {pos} |".format(
                split=split_label(split),
                base=pct(r.get("base_executor_accuracy")),
                soft=pct(r.get("soft_trace_executor_accuracy")),
                learned=pct(r.get("learned_executor_accuracy")),
                pair=pct(r.get("pair_rerank_executor_accuracy")),
                oracle=pct(r.get("oracle_executor_accuracy")),
                gap=pct(r.get("learned_oracle_gap_recovered")),
                cands=fmt_float(r.get("avg_candidates"), 1),
                pos=fmt_float(r.get("avg_positive_candidates"), 1),
            )
        )
    return header + sep + "\n".join(body)


def make_domain_table(metrics: pd.DataFrame) -> str:
    header = "| domain | base | learned | oracle | learned delta |\n"
    sep = "|---|---:|---:|---:|---:|\n"
    body = []
    for split in DOMAIN_SPLITS:
        row = metrics[(metrics["run"] == MAIN_RUN) & (metrics["split"] == split)]
        if row.empty:
            continue
        r = row.iloc[0]
        base = r.get("base_executor_accuracy")
        learned = r.get("learned_executor_accuracy")
        delta = None if pd.isna(base) or pd.isna(learned) else float(learned) - float(base)
        body.append(
            "| {domain} | {base} | {learned} | {oracle} | {delta} |".format(
                domain=split_label(split).replace(" L6", ""),
                base=pct(base),
                learned=pct(learned),
                oracle=pct(r.get("oracle_executor_accuracy")),
                delta=pct(delta),
            )
        )
    return header + sep + "\n".join(body)


def make_report(metrics: pd.DataFrame, train_logs: pd.DataFrame, metadata: dict[str, dict[str, Any]]) -> str:
    main_meta = metadata.get(MAIN_RUN, {})
    args = main_meta.get("args", {})
    compiler_args = main_meta.get("compiler_args", {})
    best_epoch = None
    main_train = train_logs[train_logs["run"] == MAIN_RUN] if not train_logs.empty else pd.DataFrame()
    if not main_train.empty:
        best_idx = main_train["val_learned_executor_accuracy"].idxmax()
        best_epoch = int(main_train.loc[best_idx, "epoch"])
    fresh_std_delta = (
        metric(metrics, MAIN_RUN, "fresh_standard_len6", "learned_executor_accuracy")
        - metric(metrics, MAIN_RUN, "fresh_standard_len6", "base_executor_accuracy")
    )
    hard10_delta = (
        metric(metrics, MAIN_RUN, "harder_standard_len10", "learned_executor_accuracy")
        - metric(metrics, MAIN_RUN, "harder_standard_len10", "base_executor_accuracy")
    )
    paired_delta = (
        metric(metrics, MAIN_RUN, "fresh_paired_len6", "learned_executor_accuracy")
        - metric(metrics, MAIN_RUN, "fresh_paired_len6", "base_executor_accuracy")
    )
    oracle_low = metrics[(metrics["run"] == MAIN_RUN) & (metrics["split"].isin(MAIN_SPLITS))]["oracle_executor_accuracy"].min()
    oracle_high = metrics[(metrics["run"] == MAIN_RUN) & (metrics["split"].isin(MAIN_SPLITS))]["oracle_executor_accuracy"].max()
    return f"""# Qwen Complete-Program Trace Reranker

## Abstract

This experiment tests a complete-program posttraining route for improving a frozen local Qwen compiler. A Qwen3-4B hidden-state adapter emits a compact virtual-machine program. Around that program, the system enumerates local executable edits and trains a small context-conditioned verifier to select the best candidate using only prompt hidden-state context, candidate features, and execution traces. The verifier never sees the target answer at inference time.

The result is diagnostic rather than successful. The candidate search contains answer-correct programs for {pct(oracle_low)} to {pct(oracle_high)} of the main evaluation prompts, so the executable candidate set has substantial reachable headroom. The learned verifier captures only small in-distribution gains and loses accuracy on paired and longer-chain splits. The bottleneck is therefore candidate selection and credit assignment, not candidate availability.

## Question

Can a small posttraining module make a frozen Qwen-attached compiler select better complete programs by inspecting executable traces, without forcing the model to generate every intermediate reasoning step as text?

## Method

- Base model: `{compiler_args.get("model_id", "Qwen/Qwen3-4B")}`.
- Value space: arithmetic over modulus `{compiler_args.get("value_modulus", 97)}` with up to `{compiler_args.get("max_steps", 10)}` VM steps.
- Frozen compiler: Qwen hidden states feed a trained hidden-VM compiler checkpoint.
- Candidate generator: local edits around the compiler argmax program, with top-k alternatives and up to `{args.get("repair_max_edits", 2)}` edits.
- Verifier input: candidate execution trace, scalar candidate features, and compact prompt hidden-state summaries.
- Main verifier: `{args.get("trace_layers", 3)}` trace-transformer layers, width `{args.get("trace_d_model", 128)}`, `{args.get("trace_heads", 4)}` heads.
- Main training set: `{args.get("train_examples", 384)}` prompts, length range `{args.get("train_min_len", 1)}` to `{args.get("train_max_len", 6)}`, positive label `{args.get("positive_label", "answer")}`.
- Main checkpoint selection: best validation learned accuracy; selected epoch `{best_epoch}`.

The key baselines are:

- `base`: the frozen compiler argmax program.
- `soft trace`: a differentiable executor scoring heuristic.
- `learned`: the trained verifier top-1 selection.
- `pair rerank`: pair-level consistency reranking for paraphrase pairs.
- `oracle`: any answer-correct candidate in the generated local neighborhood.

## Main Results

{make_metrics_table(metrics, MAIN_SPLITS)}

![Main accuracy by split](../analysis/figures/accuracy_by_split.png)

![Oracle gap recovered](../analysis/figures/oracle_gap_recovered.png)

Fresh length-6 standard prompts improved from {pct(metric(metrics, MAIN_RUN, "fresh_standard_len6", "base_executor_accuracy"))} to {pct(metric(metrics, MAIN_RUN, "fresh_standard_len6", "learned_executor_accuracy"))}, a {pct(fresh_std_delta)} absolute gain. Fresh length-6 paraphrase prompts improved from {pct(metric(metrics, MAIN_RUN, "fresh_paraphrase_len6", "base_executor_accuracy"))} to {pct(metric(metrics, MAIN_RUN, "fresh_paraphrase_len6", "learned_executor_accuracy"))}. These gains are real but small relative to the oracle.

The same selector did not extrapolate. Fresh paired length-6 accuracy moved from {pct(metric(metrics, MAIN_RUN, "fresh_paired_len6", "base_executor_accuracy"))} to {pct(metric(metrics, MAIN_RUN, "fresh_paired_len6", "learned_executor_accuracy"))}, a {pct(paired_delta)} absolute change. Harder standard length-10 moved from {pct(metric(metrics, MAIN_RUN, "harder_standard_len10", "base_executor_accuracy"))} to {pct(metric(metrics, MAIN_RUN, "harder_standard_len10", "learned_executor_accuracy"))}, a {pct(hard10_delta)} absolute change. The oracle stayed high on these splits, so the selector failed to locate available correct candidates.

## Candidate Geometry

![Candidate density](../analysis/figures/candidate_density.png)

The candidate generator is broad. On length-6 fresh splits it produces about 111 candidates per prompt, with roughly 28 to 30 answer-correct candidates. On length-10 splits it produces 263 candidates per prompt, with many answer-correct candidates but far fewer state-exact or program-exact candidates. This explains why answer-only labels are easy to satisfy but weakly identify the best computational trace.

## Objective Pilots

![Pilot objective comparison](../analysis/figures/pilot_comparison.png)

Several label objectives were tested before the main run:

- Answer-correct labels learned a nontrivial selector on validation but were noisy because many candidates share the final answer.
- State-exact labels were too conservative and mostly preserved the base program.
- Single-oracle labels were sharper but still mostly preserved the base program.
- Repair-focused weighting did not overcome the base-preservation tendency.

![Validation curves](../analysis/figures/validation_curves.png)

The main run eventually recovered the best validation learned accuracy at epoch {best_epoch}, but validation gains were small while train accuracy rose strongly. That is the signature of a selector that can fit candidate artifacts without learning a robust preference rule for unseen prompts.

## Domain Breakdown

{make_domain_table(metrics)}

![Domain breakdown](../analysis/figures/domain_breakdown.png)

The verifier helped arithmetic, calendar, unit, and list prompts modestly, left lookup unchanged, and hurt boolean prompts. Boolean has many answer-correct candidates but relatively little trace-identifying signal because final answers collapse to few values.

## Interpretation

This experiment answers one useful subquestion: local executable candidate search is not the limiting factor. The oracle remains high even when the base compiler is weak. The limiting factor is how to train a verifier that identifies the right complete program from a dense equivalence class of answer-correct candidates.

The current learned verifier is too passive. It sees traces and context, but its supervision says many different programs are equally good whenever they hit the final answer. Sharper labels alone did not fix this because the base program dominates many training groups and the correct non-base candidates are sparse. The result argues against scaling this exact reranker unchanged.

## Most Impactful Next Options

1. Train the selector from executable teacher traces, not just final answers.
   Use the oracle candidate to produce dense supervision over every intermediate state and operation, then train the verifier or compiler with a margin that explicitly ranks state-consistent candidates above answer-only candidates. This directly targets the observed dense-label failure.

2. Convert reranking into preference learning on repairable failures.
   Filter training to groups where the base program is wrong and at least one non-base candidate is right, then train pairwise preferences with hard negatives that share the answer but diverge in state. This removes the base-preservation shortcut.

3. Let Qwen read serialized candidate traces.
   Instead of only a small verifier, serialize a small shortlist of candidate programs and traces back into Qwen hidden states or text tokens, then train a LoRA selector head. This tests whether the frozen model already has the semantic machinery needed to choose among executable candidates when the candidates are made legible.

4. Train a differentiable interpreter objective upstream.
   Backpropagate through the soft executor into the adapter with auxiliary losses on intermediate states, then use the discrete candidate oracle only for evaluation. This attacks the compiler's crystallized program quality rather than relying on post-hoc selection.

The first option is the cleanest next experiment because it directly matches the failure mode found here: answer-correct candidate availability is high, but answer-only selection is underdetermined.

## Artifacts

- Aggregate metrics: `experiments/qwen_complete_program_trace_reranker/analysis/all_final_metrics.csv`
- Main metrics: `experiments/qwen_complete_program_trace_reranker/analysis/final_metrics.csv`
- Training logs: `experiments/qwen_complete_program_trace_reranker/analysis/verifier_train_logs.csv`
- Checkpoint manifest: `experiments/qwen_complete_program_trace_reranker/checkpoint_manifest.csv`
- Large checkpoints: `large_artifacts/qwen_complete_program_trace_reranker/checkpoints`
"""


def write_report(markdown_text: str) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    md_path = REPORTS / "qwen_complete_program_trace_reranker_report.md"
    html_path = REPORTS / "qwen_complete_program_trace_reranker_report.html"
    md_path.write_text(markdown_text)
    body = markdown.markdown(markdown_text, extensions=["tables", "fenced_code"])
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Qwen Complete-Program Trace Reranker</title>
<style>
:root {{
  color-scheme: light;
  --ink: #1f2933;
  --muted: #5f6c7b;
  --line: #d9e2ec;
  --panel: #f7f9fb;
  --accent: #0f766e;
}}
body {{
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: var(--ink);
  background: #ffffff;
  line-height: 1.55;
}}
main {{
  max-width: 1120px;
  margin: 0 auto;
  padding: 48px 28px 72px;
}}
h1 {{
  font-size: 2.35rem;
  margin: 0 0 1rem;
  letter-spacing: 0;
}}
h2 {{
  border-top: 1px solid var(--line);
  padding-top: 1.3rem;
  margin-top: 2.2rem;
  font-size: 1.35rem;
}}
p, li {{
  font-size: 1rem;
}}
code {{
  background: var(--panel);
  padding: 0.12rem 0.28rem;
  border-radius: 4px;
}}
table {{
  border-collapse: collapse;
  width: 100%;
  margin: 1rem 0 1.6rem;
  font-size: 0.9rem;
}}
th, td {{
  border-bottom: 1px solid var(--line);
  padding: 0.5rem 0.55rem;
  text-align: right;
  vertical-align: top;
}}
th:first-child, td:first-child {{
  text-align: left;
}}
th {{
  background: var(--panel);
}}
img {{
  width: 100%;
  max-width: 1050px;
  display: block;
  margin: 1.2rem auto 1.8rem;
  border: 1px solid var(--line);
}}
a {{
  color: var(--accent);
}}
</style>
</head>
<body>
<main>
{body}
</main>
</body>
</html>
"""
    html_path.write_text(html_doc)


def write_summary(metrics: pd.DataFrame, train_logs: pd.DataFrame) -> None:
    summary = "# Complete-Program Trace Reranker Analysis Summary\n\n"
    main_train = train_logs[train_logs["run"] == MAIN_RUN] if not train_logs.empty else pd.DataFrame()
    if not main_train.empty:
        best_idx = main_train["val_learned_executor_accuracy"].idxmax()
        summary += f"Best verifier epoch: {int(main_train.loc[best_idx, 'epoch'])}\n\n"
    summary += make_metrics_table(metrics, MAIN_SPLITS + DOMAIN_SPLITS)
    summary += "\n"
    (ANALYSIS / "summary.md").write_text(summary)


def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    metrics = read_metrics()
    train_logs = read_train_logs()
    metadata = read_metadata()
    write_csvs(metrics, train_logs)
    generate_figures(metrics, train_logs)
    write_manifest(metadata)
    write_summary(metrics, train_logs)
    report = make_report(metrics, train_logs, metadata)
    write_report(report)
    print(f"[analysis] wrote {ANALYSIS / 'final_metrics.csv'}")
    print(f"[analysis] wrote {REPORTS / 'qwen_complete_program_trace_reranker_report.md'}")
    print(f"[analysis] wrote {REPORTS / 'qwen_complete_program_trace_reranker_report.html'}")


if __name__ == "__main__":
    main()
