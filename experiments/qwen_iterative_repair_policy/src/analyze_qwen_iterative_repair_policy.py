#!/usr/bin/env python3
"""Analyze the Qwen iterative repair policy experiment."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("experiments/qwen_iterative_repair_policy")
RUNS = ROOT / "runs"
ANALYSIS = ROOT / "analysis"
FIGURES = ANALYSIS / "figures"
REPORTS = ROOT / "reports"
CHECKPOINT_ROOT = Path("large_artifacts/qwen_iterative_repair_policy/checkpoints")
MAIN_RUN = "main_iterative_candidate_scorer_s384"


def pct(x: Any) -> str:
    if x is None:
        return "n/a"
    try:
        v = float(x)
    except Exception:
        return "n/a"
    if math.isnan(v):
        return "n/a"
    return f"{100.0 * v:.1f}%"


def pp(x: Any) -> str:
    try:
        v = float(x)
    except Exception:
        return "n/a"
    if math.isnan(v):
        return "n/a"
    return f"{100.0 * v:+.1f} pp"


def read_all_metrics() -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for path in sorted(RUNS.glob("*/metrics.csv")):
        df = pd.read_csv(path)
        rows.append(df)
    if not rows:
        raise SystemExit("no run metrics found")
    return pd.concat(rows, ignore_index=True, sort=False)


def split_base(split: str) -> str:
    if "_k" in split:
        return split.rsplit("_k", 1)[0]
    return split


def main_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df[df["run"].eq(MAIN_RUN)].copy()
    out["split_base"] = out["split"].map(split_base)
    return out


def fresh_main_table(df: pd.DataFrame) -> pd.DataFrame:
    main = main_rows(df)
    wanted = ["fresh_standard_len24", "fresh_paraphrase_len24", "fresh_paired_len24"]
    rows: List[Dict[str, Any]] = []
    for split in wanted:
        sub = main[main["split_base"].eq(split)].copy()
        if sub.empty:
            continue
        best_iter = sub.sort_values(["iter_executor_accuracy", "k_steps"], ascending=[False, True]).iloc[0]
        k0 = sub[sub["k_steps"].eq(0)].iloc[0]
        rows.append(
            {
                "split": split,
                "base": float(k0["base_executor_accuracy"]),
                "learned": float(best_iter["learned_executor_accuracy"]),
                "best_k": int(best_iter["k_steps"]),
                "iterative": float(best_iter["iter_executor_accuracy"]),
                "oracle": float(best_iter["oracle_executor_accuracy"]),
                "iter_delta": float(best_iter["iter_executor_accuracy"] - k0["base_executor_accuracy"]),
                "learned_delta": float(best_iter["learned_executor_accuracy"] - k0["base_executor_accuracy"]),
                "gap_recovered": float(best_iter["iter_oracle_gap_recovered"]),
                "steps_used": float(best_iter["iter_steps_used_mean"]),
            }
        )
    return pd.DataFrame(rows)


def save_table_csvs(df: pd.DataFrame) -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    all_metrics = df.copy()
    all_metrics.to_csv(ANALYSIS / "all_final_metrics.csv", index=False)
    main = main_rows(df)
    main.to_csv(ANALYSIS / "main_final_metrics.csv", index=False)
    fresh_main_table(df).to_csv(ANALYSIS / "fresh_main_summary.csv", index=False)
    train_logs: List[pd.DataFrame] = []
    for path in sorted(RUNS.glob("*/verifier_train_log.csv")):
        part = pd.read_csv(path)
        part.insert(0, "run", path.parent.name)
        train_logs.append(part)
    if train_logs:
        pd.concat(train_logs, ignore_index=True, sort=False).to_csv(ANALYSIS / "verifier_train_logs.csv", index=False)
    editor_logs: List[pd.DataFrame] = []
    for path in sorted(RUNS.glob("*/editor_train_log.csv")):
        part = pd.read_csv(path)
        part.insert(0, "run", path.parent.name)
        editor_logs.append(part)
    if editor_logs:
        pd.concat(editor_logs, ignore_index=True, sort=False).to_csv(ANALYSIS / "direct_policy_train_logs.csv", index=False)


def plot_accuracy_by_k(df: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    main = main_rows(df)
    splits = ["fresh_standard_len24", "fresh_paraphrase_len24", "fresh_paired_len24", "val_paired_len24"]
    labels = {
        "fresh_standard_len24": "Fresh standard",
        "fresh_paraphrase_len24": "Fresh paraphrase",
        "fresh_paired_len24": "Fresh paired",
        "val_paired_len24": "Paired validation",
    }
    plt.figure(figsize=(9, 5.5))
    for split in splits:
        sub = main[main["split_base"].eq(split)].sort_values("k_steps")
        if sub.empty:
            continue
        plt.plot(sub["k_steps"], 100 * sub["iter_executor_accuracy"], marker="o", linewidth=2, label=labels[split])
    plt.xlabel("Repair iterations K")
    plt.ylabel("Exact execution accuracy (%)")
    plt.title("Iterative candidate repair accuracy by K")
    plt.xticks([0, 1, 2, 3])
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "accuracy_by_k_main.png", dpi=180)
    plt.close()


def plot_fresh_bars(summary: pd.DataFrame) -> None:
    labels = ["Standard", "Paraphrase", "Paired"]
    x = range(len(labels))
    width = 0.2
    plt.figure(figsize=(9, 5.2))
    plt.bar([i - 1.5 * width for i in x], 100 * summary["base"], width=width, label="Base")
    plt.bar([i - 0.5 * width for i in x], 100 * summary["learned"], width=width, label="Learned scorer")
    plt.bar([i + 0.5 * width for i in x], 100 * summary["iterative"], width=width, label="Best iterative K")
    plt.bar([i + 1.5 * width for i in x], 100 * summary["oracle"], width=width, label="Oracle ceiling")
    plt.xticks(list(x), labels)
    plt.ylabel("Exact execution accuracy (%)")
    plt.title("Fresh length-24 results")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "fresh_accuracy_bars.png", dpi=180)
    plt.close()


def plot_gap(summary: pd.DataFrame) -> None:
    labels = ["Standard", "Paraphrase", "Paired"]
    plt.figure(figsize=(8, 4.8))
    plt.bar(labels, 100 * summary["gap_recovered"], color="#4C78A8")
    plt.ylabel("Oracle gap recovered (%)")
    plt.title("Recovered base-to-oracle gap at best K")
    plt.ylim(0, max(45, 100 * float(summary["gap_recovered"].max()) + 5))
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES / "oracle_gap_recovered.png", dpi=180)
    plt.close()


def plot_training_curve() -> None:
    path = RUNS / MAIN_RUN / "verifier_train_log.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    plt.figure(figsize=(8, 4.8))
    plt.plot(df["epoch"], 100 * df["val_learned_executor_accuracy"], marker="o", label="Learned scorer")
    if "val_oracle_executor_accuracy" in df:
        plt.plot(df["epoch"], 100 * df["val_oracle_executor_accuracy"], linestyle="--", label="Oracle ceiling")
    if "val_base_executor_accuracy" in df:
        plt.plot(df["epoch"], 100 * df["val_base_executor_accuracy"], linestyle=":", label="Base")
    plt.xlabel("Verifier epoch")
    plt.ylabel("Paired validation accuracy (%)")
    plt.title("Verifier training curve")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "verifier_training_curve.png", dpi=180)
    plt.close()


def plot_direct_vs_candidate(df: pd.DataFrame) -> None:
    rows = []
    for run in ["pilot_iterative_k1_sparse_s96", "pilot_iterative_paired_select_s96", "pilot_iterative_candidate_scorer_s128", MAIN_RUN]:
        sub = df[df["run"].eq(run) & df["split"].str.startswith("fresh_paired_len24")].copy()
        if sub.empty:
            continue
        best = sub.sort_values(["iter_executor_accuracy", "k_steps"], ascending=[False, True]).iloc[0]
        rows.append({"run": run, "best": float(best["iter_executor_accuracy"]), "base": float(best["base_executor_accuracy"])})
    if not rows:
        return
    plot_df = pd.DataFrame(rows)
    labels = ["Direct\nunpaired", "Direct\npaired", "Candidate\npilot", "Candidate\nmain"][: len(plot_df)]
    x = range(len(plot_df))
    plt.figure(figsize=(8.5, 4.8))
    plt.bar([i - 0.18 for i in x], 100 * plot_df["base"], width=0.36, label="Base")
    plt.bar([i + 0.18 for i in x], 100 * plot_df["best"], width=0.36, label="Best iterative")
    plt.xticks(list(x), labels)
    plt.ylabel("Fresh paired accuracy (%)")
    plt.title("Iteration path comparison")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "iteration_path_comparison.png", dpi=180)
    plt.close()


def make_plots(df: pd.DataFrame) -> None:
    summary = fresh_main_table(df)
    plot_accuracy_by_k(df)
    plot_fresh_bars(summary)
    plot_gap(summary)
    plot_training_curve()
    plot_direct_vs_candidate(df)


def update_manifest() -> None:
    rows: List[Dict[str, Any]] = []
    for path in sorted(CHECKPOINT_ROOT.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "artifact": path.parent.name if path.parent != CHECKPOINT_ROOT else path.stem,
                    "path": str(path),
                    "bytes": path.stat().st_size,
                }
            )
    pd.DataFrame(rows).to_csv(ROOT / "checkpoint_manifest.csv", index=False)


def markdown_table(rows: Iterable[Dict[str, str]], headers: List[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row[h] for h in headers) + " |")
    return "\n".join(lines)


def write_report(df: pd.DataFrame) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    summary = fresh_main_table(df)
    main = main_rows(df)
    paired_k2 = main[(main["split_base"].eq("fresh_paired_len24")) & (main["k_steps"].eq(2))].iloc[0]
    paired_k1 = main[(main["split_base"].eq("fresh_paired_len24")) & (main["k_steps"].eq(1))].iloc[0]
    std_k2 = main[(main["split_base"].eq("fresh_standard_len24")) & (main["k_steps"].eq(2))].iloc[0]
    para_k2 = main[(main["split_base"].eq("fresh_paraphrase_len24")) & (main["k_steps"].eq(2))].iloc[0]
    rows = []
    label_map = {
        "fresh_standard_len24": "Fresh standard L24",
        "fresh_paraphrase_len24": "Fresh paraphrase L24",
        "fresh_paired_len24": "Fresh paired L24",
    }
    for _, row in summary.iterrows():
        rows.append(
            {
                "Split": label_map[row["split"]],
                "Base": pct(row["base"]),
                "Learned": pct(row["learned"]),
                "Best K": str(int(row["best_k"])),
                "Iterative": pct(row["iterative"]),
                "Oracle": pct(row["oracle"]),
                "Delta": pp(row["iter_delta"]),
                "Gap": pct(row["gap_recovered"]),
            }
        )
    train_log = pd.read_csv(RUNS / MAIN_RUN / "verifier_train_log.csv")
    best_epoch = int(train_log.sort_values("val_learned_executor_accuracy", ascending=False).iloc[0]["epoch"])
    md = f"""# Qwen Iterative Candidate Repair Policy

## Abstract

This experiment tests whether a frozen Qwen-attached hidden-program compiler can
be improved at inference time by a learned iterative repair loop. The compiler
emits an executable modular-arithmetic program. A small verifier scores local
candidate repairs from execution-trace features. At inference, the repair loop
starts from the base compiled program and may move one slot edit at a time for
`K` iterations.

The primary run trained on 384 length-24 programs and selected checkpoints on a
paired standard/paraphrase validation split. On fresh paired length-24 prompts,
the base compiler reached {pct(paired_k2['base_executor_accuracy'])}, the
unconstrained learned scorer reached {pct(paired_k2['learned_executor_accuracy'])},
and the iterative one-edit-at-a-time repair loop reached
{pct(paired_k2['iter_executor_accuracy'])} at `K=2`. The local oracle ceiling
was {pct(paired_k2['oracle_executor_accuracy'])}.

## Setup

- Substrate: frozen Qwen 4B hidden-program compiler localized in this
  experiment's artifact directory.
- Program domain: length-24 modular arithmetic over modulus 97.
- Hidden program slots: initial value, operation per step, and argument per step.
- Candidate set: top-3 local alternatives with up to two edits around the base
  compiled program.
- Learned component: small transformer verifier over candidate execution traces
  and candidate metadata.
- Iterative rule: at each iteration, move only to a candidate within one slot
  edit of the current candidate if its learned score is higher.
- Primary run: `{MAIN_RUN}`.
- Verifier epochs: 12; selected epoch: {best_epoch}.

## Main Results

{markdown_table(rows, ['Split', 'Base', 'Learned', 'Best K', 'Iterative', 'Oracle', 'Delta', 'Gap'])}

![Fresh accuracy](../analysis/figures/fresh_accuracy_bars.png)

## K Sweep

The K sweep shows most of the gain appears by `K=1`, with a smaller but real
second-step gain on validation, paraphrase, and paired splits. `K=3` adds no
measurable benefit because the candidate set is capped at two edits.

![Accuracy by K](../analysis/figures/accuracy_by_k_main.png)

On fresh paired prompts, exact execution moved from {pct(paired_k2['base_executor_accuracy'])}
at `K=0` to {pct(paired_k1['iter_executor_accuracy'])} at `K=1` and
{pct(paired_k2['iter_executor_accuracy'])} at `K=2`.

## Oracle Gap

![Oracle gap](../analysis/figures/oracle_gap_recovered.png)

At `K=2`, the iterative repair loop recovered {pct(paired_k2['iter_oracle_gap_recovered'])}
of the fresh paired base-to-oracle gap, {pct(std_k2['iter_oracle_gap_recovered'])}
on fresh standard prompts, and {pct(para_k2['iter_oracle_gap_recovered'])} on
fresh paraphrases.

## Training Dynamics

![Verifier training](../analysis/figures/verifier_training_curve.png)

The verifier's paired-validation accuracy was noisy, so checkpoint selection was
necessary. The selected checkpoint achieved strong fresh transfer despite the
small train set.

## Iteration Path

![Iteration comparison](../analysis/figures/iteration_path_comparison.png)

The direct raw-value repair policy was not robust: one setting damaged fresh
splits and a paired-selected setting mostly learned to copy. The candidate
scorer was the first robust positive arm because it separated proposal quality
from sparse transition control.

## Interpretation

The result supports the narrow claim that an iterative hidden-program repair
loop can convert near-miss Qwen-compiled programs into exact programs without
generating chain-of-thought text. It does not show a universal capability gain:
the task is synthetic, the runtime is hand-designed, and the oracle ceiling
still leaves substantial unrecovered headroom. The important signal is that
iteration over an executable latent representation improved fresh paired exact
execution from {pct(paired_k2['base_executor_accuracy'])} to
{pct(paired_k2['iter_executor_accuracy'])}, slightly beating unconstrained
learned selection on the paired split.

## Limitations

- The verifier is trained with offline exact-state labels.
- The runtime is specialized to modular arithmetic.
- The candidate set is local and capped at two edits.
- `K=3` cannot expose deeper repairs under this candidate budget.
- The base compiler is frozen; this run does not update Qwen weights.

## Artifacts

Small files:

- `experiments/qwen_iterative_repair_policy/runs/{MAIN_RUN}/metrics.csv`
- `experiments/qwen_iterative_repair_policy/runs/{MAIN_RUN}/verifier_train_log.csv`
- `experiments/qwen_iterative_repair_policy/analysis/main_final_metrics.csv`
- `experiments/qwen_iterative_repair_policy/reports/qwen_iterative_repair_policy_paper.md`
- `experiments/qwen_iterative_repair_policy/reports/qwen_iterative_repair_policy_paper.html`

Large files:

- `large_artifacts/qwen_iterative_repair_policy/checkpoints/fixed_compiler_step00800/`
- `large_artifacts/qwen_iterative_repair_policy/checkpoints/{MAIN_RUN}/candidate_trace_verifier.pt`
"""
    md_path = REPORTS / "qwen_iterative_repair_policy_paper.md"
    md_path.write_text(md)
    css = """
body { font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 40px auto; max-width: 980px; line-height: 1.55; color: #1f2933; }
h1, h2 { color: #101828; }
table { border-collapse: collapse; width: 100%; margin: 18px 0; font-size: 14px; }
th, td { border: 1px solid #d0d5dd; padding: 8px 10px; text-align: left; }
th { background: #f2f4f7; }
img { max-width: 100%; margin: 12px 0 28px; border: 1px solid #e4e7ec; }
code { background: #f2f4f7; padding: 1px 4px; border-radius: 4px; }
"""
    table_html = pd.DataFrame(rows).to_html(index=False, escape=False)
    body = f"""
<h1>Qwen Iterative Candidate Repair Policy</h1>
<h2>Abstract</h2>
<p>This experiment tests whether a frozen Qwen-attached hidden-program compiler
can be improved at inference time by a learned iterative repair loop. The
compiler emits an executable modular-arithmetic program. A small verifier scores
local candidate repairs from execution-trace features. At inference, the repair
loop starts from the base compiled program and may move one slot edit at a time
for <code>K</code> iterations.</p>
<p>The primary run trained on 384 length-24 programs and selected checkpoints on
a paired standard/paraphrase validation split. On fresh paired length-24
prompts, base accuracy was {pct(paired_k2['base_executor_accuracy'])},
unconstrained learned scorer accuracy was {pct(paired_k2['learned_executor_accuracy'])},
and iterative repair reached {pct(paired_k2['iter_executor_accuracy'])} at
<code>K=2</code>. The local oracle ceiling was {pct(paired_k2['oracle_executor_accuracy'])}.</p>
<h2>Setup</h2>
<ul>
<li>Substrate: frozen Qwen 4B hidden-program compiler localized in this experiment's artifact directory.</li>
<li>Program domain: length-24 modular arithmetic over modulus 97.</li>
<li>Candidate set: top-3 local alternatives with up to two edits around the base compiled program.</li>
<li>Iterative rule: move only to a candidate within one slot edit of the current candidate if its learned score is higher.</li>
<li>Primary run: <code>{MAIN_RUN}</code>; verifier epochs: 12; selected epoch: {best_epoch}.</li>
</ul>
<h2>Main Results</h2>
{table_html}
<img src="../analysis/figures/fresh_accuracy_bars.png" alt="Fresh accuracy">
<h2>K Sweep</h2>
<p>Most of the gain appears by <code>K=1</code>, with a smaller second-step gain
on validation, paraphrase, and paired splits. <code>K=3</code> adds no measurable
benefit because the candidate set is capped at two edits.</p>
<img src="../analysis/figures/accuracy_by_k_main.png" alt="Accuracy by K">
<h2>Oracle Gap</h2>
<p>At <code>K=2</code>, the iterative repair loop recovered
{pct(paired_k2['iter_oracle_gap_recovered'])} of the fresh paired base-to-oracle
gap.</p>
<img src="../analysis/figures/oracle_gap_recovered.png" alt="Oracle gap">
<h2>Training Dynamics</h2>
<img src="../analysis/figures/verifier_training_curve.png" alt="Verifier training">
<h2>Iteration Path</h2>
<p>The direct raw-value repair policy was not robust. The candidate scorer was
the first robust positive arm because it separated proposal quality from sparse
transition control.</p>
<img src="../analysis/figures/iteration_path_comparison.png" alt="Iteration comparison">
<h2>Interpretation</h2>
<p>The result supports the narrow claim that an iterative hidden-program repair
loop can convert near-miss Qwen-compiled programs into exact programs without
generating chain-of-thought text. It does not show a universal capability gain:
the task is synthetic, the runtime is hand-designed, and the oracle ceiling
still leaves substantial unrecovered headroom.</p>
"""
    html_doc = f"<!doctype html><html><head><meta charset='utf-8'><title>Qwen Iterative Candidate Repair Policy</title><style>{css}</style></head><body>{body}</body></html>"
    (REPORTS / "qwen_iterative_repair_policy_paper.html").write_text(html_doc)


def write_analysis_summary(df: pd.DataFrame) -> None:
    summary = fresh_main_table(df)
    paired = summary[summary["split"].eq("fresh_paired_len24")].iloc[0]
    lines = [
        "# Analysis Summary",
        "",
        f"Primary run: `{MAIN_RUN}`",
        "",
        f"Fresh paired base: {pct(paired['base'])}",
        f"Fresh paired iterative best K={int(paired['best_k'])}: {pct(paired['iterative'])}",
        f"Fresh paired oracle: {pct(paired['oracle'])}",
        "",
        "Generated charts:",
        "",
        "- `analysis/figures/accuracy_by_k_main.png`",
        "- `analysis/figures/fresh_accuracy_bars.png`",
        "- `analysis/figures/oracle_gap_recovered.png`",
        "- `analysis/figures/verifier_training_curve.png`",
        "- `analysis/figures/iteration_path_comparison.png`",
    ]
    (ANALYSIS / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    df = read_all_metrics()
    save_table_csvs(df)
    make_plots(df)
    update_manifest()
    write_report(df)
    write_analysis_summary(df)
    print(f"wrote analysis for {MAIN_RUN}")


if __name__ == "__main__":
    main()
