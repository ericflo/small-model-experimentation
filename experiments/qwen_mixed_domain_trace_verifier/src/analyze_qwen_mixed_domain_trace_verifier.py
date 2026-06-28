#!/usr/bin/env python3
"""Analyze and report the mixed-domain Qwen trace-verifier experiment."""

from __future__ import annotations

import csv
import html
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("experiments/qwen_mixed_domain_trace_verifier")
RUNS = ROOT / "runs"
ANALYSIS = ROOT / "analysis"
REPORTS = ROOT / "reports"
CHECKPOINT_ROOT = Path("large_artifacts/qwen_mixed_domain_trace_verifier/checkpoints")

MAIN_RUNS = [
    "main_len6_weighted_trace_verifier_s512",
    "main_len6_8_val8_weighted_trace_verifier_s512",
]

RUN_LABELS = {
    "main_len6_weighted_trace_verifier_s512": "Len6 verifier",
    "main_len6_8_val8_weighted_trace_verifier_s512": "Len6-8 verifier",
}

KEY_SPLITS = [
    "fresh_standard_len6",
    "fresh_paraphrase_len6",
    "fresh_paired_len6",
    "hard_standard_len8",
    "hard_paraphrase_len8",
    "harder_standard_len10",
    "harder_paraphrase_len10",
]


def pct(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if math.isnan(v):
        return "n/a"
    return f"{100.0 * v:.1f}%"


def pp(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if math.isnan(v):
        return "n/a"
    return f"{100.0 * v:+.1f} pp"


def read_metrics(run: str) -> pd.DataFrame:
    path = RUNS / run / "metrics.csv"
    df = pd.read_csv(path)
    df = df.drop(columns=[col for col in ["run", "run_label"] if col in df.columns])
    prefix = pd.DataFrame({"run": [run] * len(df), "run_label": [RUN_LABELS.get(run, run)] * len(df)})
    return pd.concat([prefix, df.reset_index(drop=True)], axis=1)


def read_train_log(run: str) -> pd.DataFrame:
    path = RUNS / run / "verifier_train_log.csv"
    df = pd.read_csv(path)
    df = df.drop(columns=[col for col in ["run", "run_label"] if col in df.columns])
    prefix = pd.DataFrame({"run": [run] * len(df), "run_label": [RUN_LABELS.get(run, run)] * len(df)})
    return pd.concat([prefix, df.reset_index(drop=True)], axis=1)


def read_any_metrics(run: str) -> pd.DataFrame:
    if run in MAIN_RUNS:
        return read_metrics(run)
    df = pd.read_csv(RUNS / run / "metrics.csv")
    df = df.drop(columns=[col for col in ["run", "run_label"] if col in df.columns])
    prefix = pd.DataFrame({"run": [run] * len(df), "run_label": [run] * len(df)})
    return pd.concat([prefix, df.reset_index(drop=True)], axis=1)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def save_accuracy_chart(df: pd.DataFrame, path: Path) -> None:
    rows = df[df["split"].isin(KEY_SPLITS)].copy()
    selectors = ["base_executor_accuracy", "learned_executor_accuracy", "pair_rerank_executor_accuracy", "oracle_executor_accuracy"]
    selector_labels = ["Base", "Learned", "Pair rerank", "Oracle"]
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    for ax, (run, group) in zip(axes, rows.groupby("run", sort=False)):
        x = range(len(KEY_SPLITS))
        width = 0.19
        for i, (col, label) in enumerate(zip(selectors, selector_labels)):
            vals = [float(group[group["split"] == split][col].iloc[0]) if col in group and not group[group["split"] == split].empty else math.nan for split in KEY_SPLITS]
            offsets = [v + (i - 1.5) * width for v in x]
            ax.bar(offsets, vals, width=width, label=label)
        ax.set_title(RUN_LABELS[run])
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("exact state accuracy")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(ncol=4, fontsize=9)
    axes[-1].set_xticks(list(range(len(KEY_SPLITS))))
    axes[-1].set_xticklabels(KEY_SPLITS, rotation=25, ha="right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_gap_chart(df: pd.DataFrame, path: Path) -> None:
    rows = df[df["split"].isin(KEY_SPLITS)].copy()
    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.36
    x = range(len(KEY_SPLITS))
    for i, (run, group) in enumerate(rows.groupby("run", sort=False)):
        vals = []
        for split in KEY_SPLITS:
            row = group[group["split"] == split].iloc[0]
            vals.append(float(row.get("learned_oracle_gap_recovered", math.nan)))
        ax.bar([v + (i - 0.5) * width for v in x], vals, width=width, label=RUN_LABELS[run])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylim(-0.1, 0.35)
    ax.set_ylabel("learned fraction of oracle gap recovered")
    ax.set_xticks(list(x))
    ax.set_xticklabels(KEY_SPLITS, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_pair_chart(df: pd.DataFrame, path: Path) -> None:
    paired = df[df["split"] == "fresh_paired_len6"].copy()
    selectors = [
        ("base_pair_both_correct", "Base both correct"),
        ("learned_pair_both_correct", "Learned both correct"),
        ("pair_rerank_pair_both_correct", "Pair rerank both correct"),
        ("oracle_pair_both_correct", "Oracle both correct"),
    ]
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.18
    x = range(len(paired))
    for i, (col, label) in enumerate(selectors):
        vals = [float(row.get(col, math.nan)) for _, row in paired.iterrows()]
        ax.bar([v + (i - 1.5) * width for v in x], vals, width=width, label=label)
    ax.set_xticks(list(x))
    ax.set_xticklabels([RUN_LABELS[row.run] for _, row in paired.iterrows()])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("paired prompts both correct")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_domain_chart(df: pd.DataFrame, path: Path) -> None:
    run = "main_len6_8_val8_weighted_trace_verifier_s512"
    rows = df[(df["run"] == run) & df["split"].str.startswith("domain_")].copy()
    rows["domain"] = rows["split"].str.replace("domain_", "", regex=False).str.replace("_len6", "", regex=False)
    fig, ax = plt.subplots(figsize=(11, 5))
    selectors = [("base_executor_accuracy", "Base"), ("learned_executor_accuracy", "Learned"), ("oracle_executor_accuracy", "Oracle")]
    width = 0.24
    x = range(len(rows))
    for i, (col, label) in enumerate(selectors):
        vals = [float(row[col]) for _, row in rows.iterrows()]
        ax.bar([v + (i - 1) * width for v in x], vals, width=width, label=label)
    ax.set_xticks(list(x))
    ax.set_xticklabels(rows["domain"], rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("exact state accuracy")
    ax.set_title("Len6-8 verifier domain breakdown at length 6")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_validation_chart(train_logs: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for run, group in train_logs.groupby("run", sort=False):
        ax.plot(group["epoch"], group["val_base_executor_accuracy"], linestyle="--", label=f"{RUN_LABELS[run]} base")
        ax.plot(group["epoch"], group["val_learned_executor_accuracy"], marker="o", markersize=3, label=f"{RUN_LABELS[run]} learned")
        ax.plot(group["epoch"], group["val_oracle_executor_accuracy"], linestyle=":", label=f"{RUN_LABELS[run]} oracle")
    ax.set_xlabel("verifier epoch")
    ax.set_ylabel("validation exact state accuracy")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def metric_row(df: pd.DataFrame, run: str, split: str) -> pd.Series:
    rows = df[(df["run"] == run) & (df["split"] == split)]
    if rows.empty:
        raise KeyError((run, split))
    return rows.iloc[0]


def improvement_table(df: pd.DataFrame, run: str) -> str:
    lines = [
        "| split | base | learned | pair-rerank | oracle | learned delta | oracle gap recovered |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in KEY_SPLITS:
        row = metric_row(df, run, split)
        base = row["base_executor_accuracy"]
        learned = row["learned_executor_accuracy"]
        pair = row.get("pair_rerank_executor_accuracy", math.nan)
        oracle = row["oracle_executor_accuracy"]
        lines.append(
            f"| `{split}` | {pct(base)} | {pct(learned)} | {pct(pair)} | {pct(oracle)} | {pp(learned - base)} | {pct(row.get('learned_oracle_gap_recovered'))} |"
        )
    return "\n".join(lines)


def build_markdown(df: pd.DataFrame, train_logs: pd.DataFrame) -> str:
    len6 = "main_len6_weighted_trace_verifier_s512"
    len68 = "main_len6_8_val8_weighted_trace_verifier_s512"
    paired6 = metric_row(df, len68, "fresh_paired_len6")
    hard_para = metric_row(df, len68, "hard_paraphrase_len8")
    hard_std10 = metric_row(df, len68, "harder_standard_len10")
    pair_delta = paired6["pair_rerank_executor_accuracy"] - paired6["base_executor_accuracy"]
    learned_delta = paired6["learned_executor_accuracy"] - paired6["base_executor_accuracy"]
    hard_para_delta = hard_para["learned_executor_accuracy"] - hard_para["base_executor_accuracy"]
    hard_std10_delta = hard_std10["learned_executor_accuracy"] - hard_std10["base_executor_accuracy"]
    best_epochs = train_logs.loc[train_logs.groupby("run")["val_learned_executor_accuracy"].idxmax()][
        ["run", "epoch", "val_base_executor_accuracy", "val_learned_executor_accuracy", "val_oracle_executor_accuracy"]
    ]
    best_epoch_lines = [
        "| verifier | best epoch | base val | learned val | oracle val |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in best_epochs.iterrows():
        best_epoch_lines.append(
            f"| {RUN_LABELS[row['run']]} | {int(row['epoch'])} | {pct(row['val_base_executor_accuracy'])} | {pct(row['val_learned_executor_accuracy'])} | {pct(row['val_oracle_executor_accuracy'])} |"
        )

    md = f"""# Qwen Mixed-Domain Candidate-Trace Verifier

## Abstract

This experiment tests whether a small learned verifier can improve a frozen
Qwen-attached hidden-VM compiler by selecting among local executable candidate
traces. The model under test emits a hidden program consisting of an initial
value plus a sequence of VM operations and arguments. The verifier never sees
gold answers or gold states at test time; it scores candidate traces using
compiler priors, local edit metadata, soft-executor support, and the candidate's
own executed trajectory.

The strongest arm trained the verifier on length-6-to-8 traces and selected the
checkpoint on length-8 validation. On fresh paired length-6 prompts, the base
compiler achieved {pct(paired6['base_executor_accuracy'])}, the learned verifier
achieved {pct(paired6['learned_executor_accuracy'])}, and paired reranking
achieved {pct(paired6['pair_rerank_executor_accuracy'])}. That is a learned gain
of {pp(learned_delta)} and a paired-rerank gain of {pp(pair_delta)}. The same arm
also improved hard paraphrase length-8 from {pct(hard_para['base_executor_accuracy'])}
to {pct(hard_para['learned_executor_accuracy'])} ({pp(hard_para_delta)}) and harder
standard length-10 from {pct(hard_std10['base_executor_accuracy'])} to
{pct(hard_std10['learned_executor_accuracy'])} ({pp(hard_std10_delta)}).

The oracle selector remains far higher than the learned verifier. This means the
candidate set often contains a correct executable trace, but the learned scoring
function recovers only a minority of the available headroom.

## Setup

- Backbone/compiler input: a frozen Qwen-attached hidden-VM compiler checkpoint
  localized under this experiment's large-artifact directory.
- VM value modulus: 97.
- VM operations: ADD, SUB, MUL, ADD7, SUB7, SET, MAX, MIN, XOR, GT.
- Domains: arithmetic, calendar, unit conversion, list aggregation, boolean
  thresholding, and table lookup.
- Candidate neighborhood: top-3 compiler values for init/op/arg slots, one-edit
  candidates, same-step op+arg two-edit candidates, and two-argument edits over
  up to the first 10 slots.
- Candidate labels for verifier training: exact answer and exact state trajectory
  match. These labels are offline supervision only.
- Test-time selectors:
  - `base`: unedited compiler argmax trace.
  - `prior`: highest compiler-prior candidate.
  - `soft_trace`: highest soft-executor state-support candidate.
  - `learned`: learned trace-verifier argmax.
  - `pair_rerank`: learned verifier plus agreement bonus across paired standard
    and paraphrased prompts.
  - `oracle`: highest-prior candidate with exact gold state trajectory.

## Main Results

![Accuracy by split](../analysis/accuracy_by_split.png)

### Length-6 Verifier

{improvement_table(df, len6)}

### Length-6-to-8 Verifier

{improvement_table(df, len68)}

## Oracle Gap

![Oracle gap recovered](../analysis/oracle_gap_recovered.png)

The learned verifier is real but not yet close to the oracle. The length-6-to-8
arm recovers about {pct(paired6['learned_oracle_gap_recovered'])} of the fresh
paired length-6 oracle gap and {pct(hard_para['learned_oracle_gap_recovered'])}
of the hard paraphrase length-8 oracle gap. This is enough to be useful, but it
also shows that most of the selection problem remains unsolved.

## Paired Prompt Behavior

![Paired consistency](../analysis/paired_consistency.png)

Paired reranking is the cleanest inference-time gain. For the length-6-to-8 arm,
paired prompts both correct improved from {pct(paired6['base_pair_both_correct'])}
under the base compiler to {pct(paired6['pair_rerank_pair_both_correct'])} with
pair reranking, while the oracle pair ceiling was {pct(paired6['oracle_pair_both_correct'])}.

## Domain Breakdown

![Domain breakdown](../analysis/domain_breakdown_len68.png)

The strongest domain gains in the length-6-to-8 arm were list aggregation,
boolean thresholding, lookup, and unit conversion. Arithmetic and calendar
remain harder for the verifier despite large oracle headroom, suggesting that
the trace features are not yet identifying the right local arithmetic repairs.

## Training Dynamics

![Validation curves](../analysis/validation_curves.png)

{chr(10).join(best_epoch_lines)}

The verifier has a narrow useful checkpoint window. Later epochs can over-edit,
so validation-based checkpoint selection is required.

## Interpretation

The experiment supports the narrow claim that hidden executable traces are
repairable by a small learned selection layer. The best arm gives a meaningful
inference-time lift without changing the frozen Qwen compiler. It does not
support a broad claim of large universal intelligence gain: the learned verifier
recovers only part of a much larger oracle candidate-selection ceiling, and the
hardest paraphrase length-10 split remains very weak.

The next technical bottleneck is verifier quality, not candidate availability.
Useful next steps are stronger trace encoders, explicit contrastive training on
base-wrong/repairable groups, and candidate generation that proposes structured
multi-slot fixes without relying on a small local edit budget.
"""
    return md


def markdown_to_html(md: str) -> str:
    lines = md.splitlines()
    out: List[str] = []
    in_ul = False
    in_table = False
    table_rows: List[str] = []

    def flush_ul() -> None:
        nonlocal in_ul
        if in_ul:
            out.append("</ul>")
            in_ul = False

    def flush_table() -> None:
        nonlocal in_table, table_rows
        if not in_table:
            return
        out.append("<table>")
        for i, row in enumerate(table_rows):
            cells = [c.strip() for c in row.strip("|").split("|")]
            if i == 1 and all(set(c) <= {"-", ":"} for c in cells):
                continue
            tag = "th" if i == 0 else "td"
            out.append("<tr>" + "".join(f"<{tag}>{html.escape(c)}</{tag}>" for c in cells) + "</tr>")
        out.append("</table>")
        table_rows = []
        in_table = False

    for raw in lines:
        line = raw.rstrip()
        if line.startswith("|") and line.endswith("|"):
            flush_ul()
            in_table = True
            table_rows.append(line)
            continue
        flush_table()
        if not line:
            flush_ul()
            continue
        if line.startswith("# "):
            flush_ul()
            out.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            flush_ul()
            out.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("### "):
            flush_ul()
            out.append(f"<h3>{html.escape(line[4:])}</h3>")
        elif line.startswith("- "):
            if not in_ul:
                out.append("<ul>")
                in_ul = True
            out.append(f"<li>{html.escape(line[2:])}</li>")
        elif line.startswith("![") and "](" in line and line.endswith(")"):
            flush_ul()
            alt = line[2 : line.index("]")]
            src = line[line.index("(") + 1 : -1]
            out.append(f'<figure><img src="{html.escape(src)}" alt="{html.escape(alt)}"><figcaption>{html.escape(alt)}</figcaption></figure>')
        else:
            flush_ul()
            escaped = html.escape(line)
            escaped = escaped.replace("`", "")
            out.append(f"<p>{escaped}</p>")
    flush_ul()
    flush_table()
    css = """
body { font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 42px auto; max-width: 1120px; line-height: 1.55; color: #172026; }
h1, h2, h3 { line-height: 1.2; color: #101820; }
h1 { font-size: 34px; margin-bottom: 8px; }
h2 { font-size: 24px; margin-top: 36px; border-bottom: 1px solid #d8dee4; padding-bottom: 6px; }
h3 { font-size: 18px; margin-top: 26px; }
table { border-collapse: collapse; width: 100%; margin: 16px 0 24px; font-size: 14px; }
th, td { border: 1px solid #d8dee4; padding: 7px 9px; text-align: right; }
th:first-child, td:first-child { text-align: left; }
th { background: #f6f8fa; }
img { max-width: 100%; border: 1px solid #d8dee4; border-radius: 6px; }
figure { margin: 20px 0 28px; }
figcaption { color: #667085; font-size: 13px; margin-top: 6px; }
code { background: #f6f8fa; padding: 1px 4px; border-radius: 4px; }
"""
    return "<!doctype html><html><head><meta charset='utf-8'><title>Qwen Mixed-Domain Candidate-Trace Verifier</title><style>" + css + "</style></head><body>" + "\n".join(out) + "</body></html>\n"


def checkpoint_manifest() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    fixed = CHECKPOINT_ROOT / "fixed_mixed_vm_trace_compiler_s512"
    for path in sorted(fixed.rglob("*")):
        if path.is_file():
            rows.append({"run": "fixed_mixed_vm_trace_compiler_s512", "artifact": "fixed_compiler", "path": str(path), "bytes": path.stat().st_size})
    for run_dir in sorted(CHECKPOINT_ROOT.iterdir()):
        if not run_dir.is_dir() or run_dir.name == "fixed_mixed_vm_trace_compiler_s512":
            continue
        path = run_dir / "candidate_trace_verifier.pt"
        if path.exists():
            rows.append({"run": run_dir.name, "artifact": "candidate_trace_verifier", "path": str(path), "bytes": path.stat().st_size})
    return rows


def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    metrics = pd.concat([read_metrics(run) for run in MAIN_RUNS], ignore_index=True)
    train_logs = pd.concat([read_train_log(run) for run in MAIN_RUNS], ignore_index=True)
    metrics.to_csv(ANALYSIS / "main_final_metrics.csv", index=False)
    metrics.to_csv(ANALYSIS / "final_metrics.csv", index=False)

    all_runs = sorted([path.name for path in RUNS.iterdir() if (path / "metrics.csv").exists()])
    all_metrics = pd.concat([read_any_metrics(run) for run in all_runs], ignore_index=True)
    all_metrics.to_csv(ANALYSIS / "all_final_metrics.csv", index=False)
    train_logs.to_csv(ANALYSIS / "main_verifier_train_logs.csv", index=False)

    save_accuracy_chart(metrics, ANALYSIS / "accuracy_by_split.png")
    save_gap_chart(metrics, ANALYSIS / "oracle_gap_recovered.png")
    save_pair_chart(metrics, ANALYSIS / "paired_consistency.png")
    save_domain_chart(metrics, ANALYSIS / "domain_breakdown_len68.png")
    save_validation_chart(train_logs, ANALYSIS / "validation_curves.png")

    md = build_markdown(metrics, train_logs)
    (REPORTS / "qwen_mixed_domain_trace_verifier_paper.md").write_text(md)
    (REPORTS / "qwen_mixed_domain_trace_verifier_paper.html").write_text(markdown_to_html(md))

    summary_lines = [
        "# Analysis Summary",
        "",
        "Generated artifacts:",
        "",
        "- `analysis/main_final_metrics.csv`",
        "- `analysis/all_final_metrics.csv`",
        "- `analysis/main_verifier_train_logs.csv`",
        "- `analysis/accuracy_by_split.png`",
        "- `analysis/oracle_gap_recovered.png`",
        "- `analysis/paired_consistency.png`",
        "- `analysis/domain_breakdown_len68.png`",
        "- `analysis/validation_curves.png`",
        "- `reports/qwen_mixed_domain_trace_verifier_paper.md`",
        "- `reports/qwen_mixed_domain_trace_verifier_paper.html`",
        "",
        "Key result: the length-6-to-8 verifier plus pair reranking reached "
        f"{pct(metric_row(metrics, 'main_len6_8_val8_weighted_trace_verifier_s512', 'fresh_paired_len6')['pair_rerank_executor_accuracy'])} "
        "fresh paired length-6 accuracy versus "
        f"{pct(metric_row(metrics, 'main_len6_8_val8_weighted_trace_verifier_s512', 'fresh_paired_len6')['base_executor_accuracy'])} base.",
    ]
    (ANALYSIS / "summary.md").write_text("\n".join(summary_lines) + "\n")
    write_csv(ROOT / "checkpoint_manifest.csv", checkpoint_manifest())
    print(f"[done] wrote analysis and reports under {ROOT}")


if __name__ == "__main__":
    main()
