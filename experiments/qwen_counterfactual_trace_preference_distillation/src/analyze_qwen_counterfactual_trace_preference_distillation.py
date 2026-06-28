#!/usr/bin/env python3
"""Analyze counterfactual trace preference distillation runs."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("experiments/qwen_counterfactual_trace_preference_distillation")
RUNS = ROOT / "runs"
ANALYSIS = ROOT / "analysis"
FIGURES = ANALYSIS / "figures"
REPORTS = ROOT / "reports"
MAIN_RUN = "main_counterfactual_trace_preference_s192_c1024"
REPORT_STEM = "qwen_counterfactual_trace_preference_distillation_report"

PHASE_LABELS = {
    "seed_supervised": "Seed",
    "counterfactual_preference_selector": "Preference selector",
    "answer_verified_distill": "Answer-verified",
    "preference_selected_distill": "Preference-selected",
    "best_quality_distill": "Best-quality",
    "full_supervised": "Full supervised",
}

SPLIT_LABELS = {
    "val_mixed": "validation",
    "fresh_standard": "fresh standard",
    "fresh_paraphrase": "fresh paraphrase",
    "fresh_paired": "fresh paired",
    "hard_composition": "hard composition",
}


def pct(x: Optional[float]) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{100.0 * float(x):.1f}%"


def num(x: Optional[float]) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{float(x):.2f}"


def read_all(name: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in sorted(RUNS.glob(f"*/{name}")):
        df = pd.read_csv(path, engine="python", on_bad_lines="skip")
        df["run"] = path.parent.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def write_csvs() -> Dict[str, pd.DataFrame]:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    data = {
        "metrics": read_all("metrics.csv"),
        "train": read_all("train_log.csv"),
        "candidates": read_all("candidate_group_stats.csv"),
        "targets": read_all("target_selection.csv"),
    }
    conventional = {
        "metrics": ("all_metrics.csv", "main_metrics.csv"),
        "train": ("all_train_logs.csv", "main_train_log.csv"),
        "candidates": ("all_candidate_group_stats.csv", "main_candidate_group_stats.csv"),
        "targets": ("all_target_selection.csv", "main_target_selection.csv"),
    }
    for name, df in data.items():
        if df.empty:
            continue
        df.to_csv(ANALYSIS / f"all_{name}.csv", index=False)
        df[df["run"].eq(MAIN_RUN)].to_csv(ANALYSIS / f"main_{name}.csv", index=False)
        all_name, main_name = conventional[name]
        df.to_csv(ANALYSIS / all_name, index=False)
        df[df["run"].eq(MAIN_RUN)].to_csv(ANALYSIS / main_name, index=False)
    return data


def main_df(data: Dict[str, pd.DataFrame], key: str) -> pd.DataFrame:
    df = data[key]
    return df[df["run"].eq(MAIN_RUN)].copy() if not df.empty else df


def value(main: pd.DataFrame, phase: str, split: str, col: str) -> float:
    row = main[(main["phase"].eq(phase)) & (main["split"].eq(split))]
    return float(row.iloc[0][col]) if not row.empty and col in row else float("nan")


def main_metrics_table(main: pd.DataFrame) -> str:
    rows: List[List[str]] = []
    phases = ["seed_supervised", "counterfactual_preference_selector", "answer_verified_distill", "preference_selected_distill", "best_quality_distill", "full_supervised"]
    splits = ["fresh_standard", "fresh_paraphrase", "fresh_paired", "hard_composition"]
    for phase in phases:
        for split in splits:
            row = main[(main["phase"].eq(phase)) & (main["split"].eq(split))]
            if row.empty:
                continue
            r = row.iloc[0]
            rows.append(
                [
                    PHASE_LABELS.get(phase, phase),
                    SPLIT_LABELS.get(split, split),
                    pct(r.get("direct_accuracy")),
                    pct(r.get("answer_search_accuracy")),
                    pct(r.get("oracle_accuracy")),
                    pct(r.get("preference_rerank_accuracy")),
                    pct(r.get("program_exact")),
                ]
            )
    header = "| Phase | Split | Direct | Answer search | Oracle | Preference rerank | Program exact |\n| --- | --- | ---: | ---: | ---: | ---: | ---: |"
    body = "\n".join("| " + " | ".join(row) + " |" for row in rows)
    return header + "\n" + body


def target_table(targets: pd.DataFrame) -> str:
    rows: List[List[str]] = []
    for _, r in targets.iterrows():
        rows.append(
            [
                str(r["phase"]),
                str(int(r["targets"])),
                pct(r["selected_correct_rate"]),
                pct(r["selected_program_exact_rate"]),
                pct(r["selected_trace_consistent_rate"]),
                num(r["selected_quality_mean"]),
                pct(r["changed_rate"]),
            ]
        )
    header = "| Target source | Targets | Correct | Canonical | Trace-consistent | Mean quality | Changed |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: |"
    return header + "\n" + "\n".join("| " + " | ".join(row) + " |" for row in rows)


def plot_main_accuracy(metrics: pd.DataFrame) -> None:
    main = metrics[metrics["run"].eq(MAIN_RUN)].copy()
    phases = ["seed_supervised", "answer_verified_distill", "preference_selected_distill", "best_quality_distill", "full_supervised"]
    splits = ["fresh_standard", "fresh_paraphrase", "fresh_paired", "hard_composition"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, metric, title in zip(axes, ["direct_accuracy", "answer_search_accuracy"], ["Direct Executable Accuracy", "Answer-Verified Search Accuracy"]):
        pivot = main[main["phase"].isin(phases) & main["split"].isin(splits)].pivot(index="split", columns="phase", values=metric).reindex(splits)
        pivot = pivot.rename(index=SPLIT_LABELS, columns=PHASE_LABELS)
        pivot.plot(kind="bar", ax=ax, width=0.82)
        ax.set_title(title)
        ax.set_ylabel("accuracy")
        ax.set_ylim(0, 1.0)
        ax.tick_params(axis="x", rotation=25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "main_accuracy_by_phase.png", dpi=180)
    plt.close(fig)


def plot_preference_selector(metrics: pd.DataFrame) -> None:
    main = metrics[(metrics["run"].eq(MAIN_RUN)) & (metrics["phase"].eq("counterfactual_preference_selector"))].copy()
    if main.empty:
        return
    splits = ["val_mixed", "fresh_standard", "fresh_paraphrase", "fresh_paired", "hard_composition"]
    plot = main.set_index("split")[["direct_accuracy", "preference_rerank_accuracy", "oracle_accuracy"]].reindex(splits)
    plot = plot.rename(index=SPLIT_LABELS, columns={"direct_accuracy": "base direct", "preference_rerank_accuracy": "preference selected", "oracle_accuracy": "candidate oracle"})
    fig, ax = plt.subplots(figsize=(10, 5))
    plot.plot(kind="bar", ax=ax, width=0.82)
    ax.set_ylim(0, 0.65)
    ax.set_ylabel("accuracy")
    ax.set_title("No-Answer Preference Selection")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(FIGURES / "preference_selector_vs_oracle.png", dpi=180)
    plt.close(fig)


def plot_candidate_surface(candidates: pd.DataFrame) -> None:
    main = candidates[(candidates["run"].eq(MAIN_RUN)) & (candidates["phase"].eq("seed_unlabeled_candidates"))].copy()
    if main.empty:
        return
    r = main.iloc[0]
    labels = ["valid", "answer-correct", "trace-consistent", "canonical", "oracle prompt", "base correct", "counterfactual"]
    values = [
        r["valid_rate"],
        r["correct_rate"],
        r["trace_consistent_rate"],
        r["canonical_rate"],
        r["oracle_found_rate"],
        r["base_correct_rate"],
        r["counterfactual_group_rate"],
    ]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(labels, values, color=["#4c78a8", "#f58518", "#54a24b", "#b279a2", "#e45756", "#72b7b2", "#ff9da6"])
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("rate")
    ax.set_title("Main Candidate Surface")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIGURES / "candidate_surface.png", dpi=180)
    plt.close(fig)


def plot_target_quality(targets: pd.DataFrame) -> None:
    main = targets[targets["run"].eq(MAIN_RUN)].copy()
    if main.empty:
        return
    labels = {
        "answer_verified_targets": "answer-verified",
        "best_quality_targets": "best-quality",
        "preference_selected_targets": "preference-selected",
    }
    main["label"] = main["phase"].map(labels).fillna(main["phase"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    main.set_index("label")[["selected_correct_rate", "selected_program_exact_rate", "selected_trace_consistent_rate"]].plot(kind="bar", ax=axes[0])
    axes[0].set_title("Selected Target Precision")
    axes[0].set_ylim(0, 1.05)
    axes[0].tick_params(axis="x", rotation=20)
    main.set_index("label")[["targets", "selected_quality_mean"]].plot(kind="bar", secondary_y=["selected_quality_mean"], ax=axes[1])
    axes[1].set_title("Target Count and Mean Quality")
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIGURES / "target_quality.png", dpi=180)
    plt.close(fig)


def plot_training(train: pd.DataFrame) -> None:
    main = train[train["run"].eq(MAIN_RUN)].copy()
    if main.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for phase, df in main.groupby("phase"):
        if "loss" in df:
            axes[0].plot(df["epoch"], df["loss"], marker="o", label=phase)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend(fontsize=7)
    pref = main[main["phase"].eq("counterfactual_preference")]
    if not pref.empty:
        axes[1].plot(pref["epoch"], pref["val_selected_correct_rate"], marker="o", label="selected")
        axes[1].plot(pref["epoch"], pref["val_oracle_found_rate"], marker="o", label="oracle")
        axes[1].plot(pref["epoch"], pref["val_base_correct_rate"], marker="o", label="base")
    axes[1].set_title("Preference Validation Curve")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(0, 0.5)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "training_and_preference_curves.png", dpi=180)
    plt.close(fig)


def plot_pilots(metrics: pd.DataFrame) -> None:
    runs = ["pilot_pref_s96_c256_m192", "pilot_pref_s96_c256_features", MAIN_RUN]
    rows: List[Dict[str, object]] = []
    for run in runs:
        df = metrics[(metrics["run"].eq(run)) & (metrics["phase"].eq("counterfactual_preference_selector"))]
        for split in ["val_mixed", "fresh_paired", "hard_composition"]:
            item = df[df["split"].eq(split)]
            if not item.empty:
                r = item.iloc[0]
                rows.append({"run": run.replace("pilot_pref_s96_c256_", "").replace("main_counterfactual_trace_preference_s192_c1024", "main"), "split": SPLIT_LABELS[split], "preference": r["preference_rerank_accuracy"], "oracle": r["oracle_accuracy"], "direct": r["direct_accuracy"]})
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(ANALYSIS / "pilot_selector_comparison.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    for ax, split in zip(axes, ["validation", "fresh paired", "hard composition"]):
        pivot = df[df["split"].eq(split)].set_index("run")[["direct", "preference", "oracle"]]
        pivot.plot(kind="bar", ax=ax)
        ax.set_title(split)
        ax.set_ylim(0, 0.6)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Selector Iteration Comparison")
    fig.tight_layout()
    fig.savefig(FIGURES / "selector_iteration_comparison.png", dpi=180)
    plt.close(fig)


def write_summary(data: Dict[str, pd.DataFrame]) -> None:
    metrics = main_df(data, "metrics")
    candidates = main_df(data, "candidates")
    targets = main_df(data, "targets")
    cand = candidates[candidates["phase"].eq("seed_unlabeled_candidates")].iloc[0]
    lines = [
        "# Analysis Summary",
        "",
        f"Main run: `{MAIN_RUN}`",
        f"Candidate surface: `{int(cand['candidates'])}` candidates, {pct(cand['oracle_found_rate'])} prompt-level oracle, {pct(cand['counterfactual_group_rate'])} counterfactual groups.",
        f"Preference selector fresh paired: direct {pct(value(metrics, 'counterfactual_preference_selector', 'fresh_paired', 'direct_accuracy'))}, selected {pct(value(metrics, 'counterfactual_preference_selector', 'fresh_paired', 'preference_rerank_accuracy'))}, oracle {pct(value(metrics, 'counterfactual_preference_selector', 'fresh_paired', 'oracle_accuracy'))}.",
        f"Preference-selected distill fresh paired direct/search: {pct(value(metrics, 'preference_selected_distill', 'fresh_paired', 'direct_accuracy'))} / {pct(value(metrics, 'preference_selected_distill', 'fresh_paired', 'answer_search_accuracy'))}.",
        f"Answer-verified distill fresh paired direct/search: {pct(value(metrics, 'answer_verified_distill', 'fresh_paired', 'direct_accuracy'))} / {pct(value(metrics, 'answer_verified_distill', 'fresh_paired', 'answer_search_accuracy'))}.",
        f"Full-supervised ceiling fresh paired direct/search: {pct(value(metrics, 'full_supervised', 'fresh_paired', 'direct_accuracy'))} / {pct(value(metrics, 'full_supervised', 'fresh_paired', 'answer_search_accuracy'))}.",
        "",
        "Figures:",
        "- `analysis/figures/main_accuracy_by_phase.png`",
        "- `analysis/figures/preference_selector_vs_oracle.png`",
        "- `analysis/figures/candidate_surface.png`",
        "- `analysis/figures/target_quality.png`",
        "- `analysis/figures/training_and_preference_curves.png`",
        "- `analysis/figures/selector_iteration_comparison.png`",
    ]
    (ANALYSIS / "summary.md").write_text("\n".join(lines) + "\n")


def markdown_report(data: Dict[str, pd.DataFrame]) -> str:
    metrics = main_df(data, "metrics")
    candidates = main_df(data, "candidates")
    targets = main_df(data, "targets")
    cand = candidates[candidates["phase"].eq("seed_unlabeled_candidates")].iloc[0]
    pref_targets = targets[targets["phase"].eq("preference_selected_targets")].iloc[0]
    answer_targets = targets[targets["phase"].eq("answer_verified_targets")].iloc[0]
    best_targets = targets[targets["phase"].eq("best_quality_targets")].iloc[0]
    report = f"""# Counterfactual Trace Preference Distillation

## Abstract

This standalone experiment tests whether a Qwen-attached typed-bytecode compiler can learn a no-answer repair selector from hard counterfactual execution traces. The compiler emits executable VM programs from frozen `Qwen/Qwen3-4B` hidden states. Each prompt gets a local candidate set; every candidate is executed; and candidates are ranked by this quality order:

`invalid < valid_wrong < answer_correct < trace_consistent < canonical`

The main run generated `{int(cand['candidates'])}` candidates from `1024` training prompts. The candidate surface was real: `{pct(cand['oracle_found_rate'])}` of prompts had an answer-correct candidate, while only `{pct(cand['base_correct_rate'])}` were already correct at the base decode. That left `{pct(cand['counterfactual_group_rate'])}` true counterfactual repair groups.

The result is mixed. The feature-bridged preference selector learned a weak but real signal on validation, reaching `{pct(value(metrics, 'counterfactual_preference_selector', 'val_mixed', 'preference_rerank_accuracy'))}` selection accuracy against a `{pct(value(metrics, 'counterfactual_preference_selector', 'val_mixed', 'oracle_accuracy'))}` oracle. It did not generalize robustly across all held-out splits: fresh-paired selection was `{pct(value(metrics, 'counterfactual_preference_selector', 'fresh_paired', 'preference_rerank_accuracy'))}` against a `{pct(value(metrics, 'counterfactual_preference_selector', 'fresh_paired', 'oracle_accuracy'))}` oracle.

Distilling the learned selector produced small direct gains on the hardest held-out cells but degraded search relative to answer-verified targets. Fresh-paired direct accuracy was `{pct(value(metrics, 'preference_selected_distill', 'fresh_paired', 'direct_accuracy'))}` for preference-selected distillation versus `{pct(value(metrics, 'answer_verified_distill', 'fresh_paired', 'direct_accuracy'))}` for answer-verified distillation. Hard-composition direct accuracy was `{pct(value(metrics, 'preference_selected_distill', 'hard_composition', 'direct_accuracy'))}` versus `{pct(value(metrics, 'answer_verified_distill', 'hard_composition', 'direct_accuracy'))}`. But fresh-paired search was lower: `{pct(value(metrics, 'preference_selected_distill', 'fresh_paired', 'answer_search_accuracy'))}` versus `{pct(value(metrics, 'answer_verified_distill', 'fresh_paired', 'answer_search_accuracy'))}`.

The full-supervised ceiling stayed high: `{pct(value(metrics, 'full_supervised', 'fresh_paired', 'direct_accuracy'))}` direct and `{pct(value(metrics, 'full_supervised', 'fresh_paired', 'answer_search_accuracy'))}` search on fresh paired, with `{pct(value(metrics, 'full_supervised', 'hard_composition', 'direct_accuracy'))}` direct on hard composition. The substrate can learn the executable compiler; the preference selector is still the bottleneck.

## Setup

- Base model: frozen `Qwen/Qwen3-4B` hidden-state extractor.
- Compiler: transformer slot decoder that emits typed stack-machine bytecode.
- Candidate generation: local edit/search around the base decode, capped at `256` candidates per prompt.
- Preference training set: only counterfactual groups where a better candidate exists than the base decode.
- Preference inputs: prompt feature, candidate bytecode, normalized program prior, prompt answer-head logprob of the candidate's VM final value, VM validity, and VM final value.
- Main run: `192` seed examples, `1024` candidate prompts, `128` examples per eval split.
- Large checkpoints: `large_artifacts/qwen_counterfactual_trace_preference_distillation/checkpoints/{MAIN_RUN}/`.

## Main Results

{main_metrics_table(metrics)}

![Main accuracy](../analysis/figures/main_accuracy_by_phase.png)

## Candidate Surface

![Candidate surface](../analysis/figures/candidate_surface.png)

The surface had enough headroom to test selection. Candidate-level answer correctness was only `{pct(cand['correct_rate'])}`, but prompt-level oracle accuracy was `{pct(cand['oracle_found_rate'])}`. Trace-consistent and canonical candidates were much rarer: `{pct(cand['trace_consistent_rate'])}` and `{pct(cand['canonical_rate'])}` at candidate level. That means the intended quality order was active, but most supervision was still effectively answer-correct versus valid-wrong.

## Target Quality

{target_table(targets)}

![Target quality](../analysis/figures/target_quality.png)

Preference-selected targets were broad but noisy: `{int(pref_targets['targets'])}` targets with only `{pct(pref_targets['selected_correct_rate'])}` answer-correct precision. Answer-verified and best-quality targets were perfectly answer-correct by construction, but they covered only `{int(answer_targets['targets'])}` prompts. Best-quality targets slightly increased canonical/trace-consistent selection from `{pct(answer_targets['selected_trace_consistent_rate'])}` to `{pct(best_targets['selected_trace_consistent_rate'])}`, but this was too small to change deployable accuracy.

## Preference Selection

![Preference selector](../analysis/figures/preference_selector_vs_oracle.png)

The preference selector did not solve no-answer credit assignment. Its best validation checkpoint improved over the base decode on validation, but on fresh standard, fresh paraphrase, and fresh paired it selected worse than the base direct program. The model learned to stay valid, but not reliably to identify the prompt-correct final value.

## Iterations

![Selector iterations](../analysis/figures/selector_iteration_comparison.png)

The first pilot had true counterfactual groups but a weak selector. Adding candidate features, especially the prompt answer-head logprob of the candidate's final value, improved the pilot selector substantially. The scaled main run retained a validation signal but lost much of the held-out gain, which points to calibration/generalization rather than candidate availability as the remaining issue.

## Training

![Training](../analysis/figures/training_and_preference_curves.png)

The full-supervised branch separated sharply after about eight epochs and reached the high ceiling. The repair-distillation branches trained smoothly but plateaued far below that ceiling, consistent with target quality and selector precision being the limiting factors.

## Interpretation

The experiment answers the core question narrowly. Counterfactual trace preference learning can find some no-answer repair signal, and it can produce small direct gains on hard held-out cells. It does not yet provide a reliable path to folding the repair oracle into the compiler. The main failure is that selected preference targets are too noisy: the selector chooses valid programs almost always, but answer-correct programs only `{pct(pref_targets['selected_correct_rate'])}` of the time.

The most useful next step is not simply scaling this objective. The next design should either filter preference-selected targets by confidence to raise precision, or move candidate comparison into a richer Qwen-readable representation where the model can compare prompt semantics, VM final values, and execution traces more directly.

## Artifacts

- `experiments/qwen_counterfactual_trace_preference_distillation/runs/{MAIN_RUN}/metrics.csv`
- `experiments/qwen_counterfactual_trace_preference_distillation/runs/{MAIN_RUN}/train_log.csv`
- `experiments/qwen_counterfactual_trace_preference_distillation/runs/{MAIN_RUN}/target_selection.csv`
- `experiments/qwen_counterfactual_trace_preference_distillation/analysis/main_metrics.csv`
- `experiments/qwen_counterfactual_trace_preference_distillation/reports/{REPORT_STEM}.md`
- `experiments/qwen_counterfactual_trace_preference_distillation/reports/{REPORT_STEM}.html`
"""
    return report


def write_html(markdown: str) -> None:
    try:
        import markdown as markdown_lib

        body = markdown_lib.markdown(markdown, extensions=["tables"])
    except Exception:
        body = "<pre>" + html.escape(markdown) + "</pre>"
    page = f"""<!doctype html><html><head><meta charset='utf-8'><title>Counterfactual Trace Preference Distillation</title><style>
body {{ font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; color: #172033; background: #f5f7fb; }}
main {{ max-width: 1120px; margin: 0 auto; padding: 40px 24px 64px; background: white; min-height: 100vh; }}
h1 {{ font-size: 34px; line-height: 1.1; margin-bottom: 12px; color: #111827; }}
h2 {{ color: #111827; margin-top: 34px; border-top: 1px solid #e5e7eb; padding-top: 24px; }}
p, li {{ line-height: 1.56; }}
code {{ background: #eef2f7; padding: 2px 5px; border-radius: 4px; }}
table {{ border-collapse: collapse; width: 100%; margin: 16px 0 24px; font-size: 13px; }}
th, td {{ border: 1px solid #d9e0ea; padding: 8px 9px; text-align: left; }}
th {{ background: #edf2f7; }}
img {{ max-width: 100%; display: block; margin: 18px 0 30px; border: 1px solid #d9e0ea; }}
.note {{ color: #4b5563; font-size: 14px; }}
</style></head><body><main><p class='note'>Standalone report generated from <code>{MAIN_RUN}</code>.</p>{body}</main></body></html>"""
    (REPORTS / f"{REPORT_STEM}.html").write_text(page)


def main() -> None:
    data = write_csvs()
    if data["metrics"].empty:
        raise SystemExit("no metrics found")
    plot_main_accuracy(data["metrics"])
    plot_preference_selector(data["metrics"])
    plot_candidate_surface(data["candidates"])
    plot_target_quality(data["targets"])
    plot_training(data["train"])
    plot_pilots(data["metrics"])
    write_summary(data)
    md = markdown_report(data)
    (REPORTS / f"{REPORT_STEM}.md").write_text(md)
    write_html(md)
    print(f"[analysis] wrote {REPORTS / f'{REPORT_STEM}.md'}")
    print(f"[analysis] wrote {REPORTS / f'{REPORT_STEM}.html'}")


if __name__ == "__main__":
    main()
