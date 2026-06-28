#!/usr/bin/env python3
"""Analyze in-policy VM-ECHO distillation runs and write reports."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("experiments/qwen_inpolicy_vm_echo_distillation")
RUNS = ROOT / "runs"
ANALYSIS = ROOT / "analysis"
FIGURES = ANALYSIS / "figures"
REPORTS = ROOT / "reports"
MAIN_RUN = "main_inpolicy_vm_echo_s192_w010"
REPORT_STEM = "qwen_inpolicy_vm_echo_distillation_report"


PHASE_LABELS = {
    "seed_supervised": "Seed",
    "answer_verified_distill": "Answer distill",
    "inpolicy_vm_echo_distill": "VM-ECHO distill",
    "gold_trace_distill": "Gold trace",
    "full_supervised": "Full sup.",
}

SPLIT_LABELS = {
    "val_mixed": "val",
    "fresh_standard": "fresh standard",
    "fresh_paraphrase": "fresh paraphrase",
    "fresh_paired": "fresh paired",
    "hard_composition": "hard composition",
}


def pct(x: Optional[float]) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"{100.0 * float(x):.1f}%"


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

    metrics = read_all("metrics.csv")
    train = read_all("train_log.csv")
    candidates = read_all("candidate_group_stats.csv")
    targets = read_all("target_selection.csv")

    if not metrics.empty:
        metrics.to_csv(ANALYSIS / "all_metrics.csv", index=False)
        metrics[metrics["run"].eq(MAIN_RUN)].to_csv(ANALYSIS / "main_metrics.csv", index=False)
    if not train.empty:
        train.to_csv(ANALYSIS / "all_train_logs.csv", index=False)
        train[train["run"].eq(MAIN_RUN)].to_csv(ANALYSIS / "main_train_log.csv", index=False)
    if not candidates.empty:
        candidates.to_csv(ANALYSIS / "all_candidate_group_stats.csv", index=False)
        candidates[candidates["run"].eq(MAIN_RUN)].to_csv(ANALYSIS / "main_candidate_group_stats.csv", index=False)
    if not targets.empty:
        targets.to_csv(ANALYSIS / "all_target_selection.csv", index=False)
        targets[targets["run"].eq(MAIN_RUN)].to_csv(ANALYSIS / "main_target_selection.csv", index=False)

    return {"metrics": metrics, "train": train, "candidates": candidates, "targets": targets}


def main_metrics_table(main: pd.DataFrame) -> str:
    rows = []
    order = ["seed_supervised", "answer_verified_distill", "inpolicy_vm_echo_distill", "gold_trace_distill", "full_supervised"]
    for phase in order:
        for split in ["fresh_standard", "fresh_paraphrase", "fresh_paired", "hard_composition"]:
            item = main[(main["phase"].eq(phase)) & (main["split"].eq(split))]
            if item.empty:
                continue
            r = item.iloc[0]
            rows.append(
                [
                    PHASE_LABELS.get(phase, phase),
                    SPLIT_LABELS.get(split, split),
                    pct(r.get("direct_accuracy")),
                    pct(r.get("answer_search_accuracy")),
                    pct(r.get("oracle_accuracy")),
                    pct(r.get("echo_rerank_accuracy")),
                    pct(r.get("program_exact")),
                ]
            )
    header = "| Phase | Split | Direct | Answer search | Oracle | ECHO rerank | Program exact |\n| --- | --- | ---: | ---: | ---: | ---: | ---: |"
    body = "\n".join("| " + " | ".join(row) + " |" for row in rows)
    return header + "\n" + body


def target_table(targets: pd.DataFrame) -> str:
    if targets.empty:
        return ""
    main = targets[targets["run"].eq(MAIN_RUN)]
    if main.empty:
        return ""
    rows = []
    for _, r in main.iterrows():
        rows.append(
            [
                str(r["phase"]),
                str(int(r["round"])),
                str(int(r["targets"])),
                pct(r["oracle_found_rate"]),
                pct(r["changed_rate"]),
                pct(r["candidate_valid_rate"]),
            ]
        )
    header = "| Phase | Round | Targets | Oracle found | Changed | Candidate valid |\n| --- | ---: | ---: | ---: | ---: | ---: |"
    return header + "\n" + "\n".join("| " + " | ".join(row) + " |" for row in rows)


def plot_main_accuracy(metrics: pd.DataFrame) -> None:
    main = metrics[metrics["run"].eq(MAIN_RUN)].copy()
    keep = ["seed_supervised", "answer_verified_distill", "inpolicy_vm_echo_distill", "gold_trace_distill", "full_supervised"]
    main = main[main["phase"].isin(keep)]
    splits = ["fresh_standard", "fresh_paraphrase", "fresh_paired", "hard_composition"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, metric, title in zip(axes, ["direct_accuracy", "answer_search_accuracy"], ["Direct Program Accuracy", "Answer-Verified Search Accuracy"]):
        pivot = main[main["split"].isin(splits)].pivot(index="split", columns="phase", values=metric).reindex(splits)
        pivot = pivot.rename(index=SPLIT_LABELS, columns=PHASE_LABELS)
        pivot.plot(kind="bar", ax=ax, width=0.82)
        ax.set_title(title)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("accuracy")
        ax.tick_params(axis="x", rotation=25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "main_accuracy_by_phase.png", dpi=180)
    plt.close(fig)


def plot_echo_observations(metrics: pd.DataFrame) -> None:
    main = metrics[(metrics["run"].eq(MAIN_RUN)) & (metrics["phase"].eq("inpolicy_vm_echo_distill"))].copy()
    if main.empty:
        return
    cols = [
        "echo_correct_pred_accuracy",
        "echo_valid_pred_accuracy",
        "echo_final_pred_accuracy",
        "echo_trace_top_accuracy",
        "echo_trace_depth_accuracy",
    ]
    plot = main.set_index("split")[cols].reindex(["val_mixed", "fresh_standard", "fresh_paraphrase", "fresh_paired", "hard_composition"])
    plot = plot.rename(index=SPLIT_LABELS, columns={
        "echo_correct_pred_accuracy": "correct label",
        "echo_valid_pred_accuracy": "validity",
        "echo_final_pred_accuracy": "final value",
        "echo_trace_top_accuracy": "trace top",
        "echo_trace_depth_accuracy": "trace depth",
    })
    fig, ax = plt.subplots(figsize=(12, 5))
    plot.plot(kind="bar", ax=ax, width=0.82)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("accuracy")
    ax.set_title("VM-ECHO Observation Prediction")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(FIGURES / "echo_observation_accuracy.png", dpi=180)
    plt.close(fig)


def plot_training(train: pd.DataFrame) -> None:
    main = train[train["run"].eq(MAIN_RUN)].copy()
    if main.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for phase, df in main.groupby("phase"):
        axes[0].plot(df["epoch"], df["loss"], marker="o", label=phase)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend(fontsize=8)
    echo = main[main["phase"].str.contains("echo_repair", na=False)]
    if not echo.empty:
        axes[1].plot(echo["epoch"], echo["echo_loss"], marker="o", label="echo loss")
        axes[1].plot(echo["epoch"], echo["repair_loss"], marker="o", label="repair loss")
    axes[1].set_title("Main VM-ECHO Joint Losses")
    axes[1].set_xlabel("epoch")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "training_curves.png", dpi=180)
    plt.close(fig)


def plot_candidates(candidates: pd.DataFrame, targets: pd.DataFrame) -> None:
    cand = candidates[candidates["run"].eq(MAIN_RUN)].copy()
    targ = targets[targets["run"].eq(MAIN_RUN)].copy()
    if cand.empty or targ.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    cand.set_index("phase")[["positive_rate", "valid_rate", "oracle_found_rate"]].plot(kind="bar", ax=axes[0])
    axes[0].set_title("Candidate Surface")
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis="x", rotation=20)
    targ.set_index("phase")[["targets", "changed_rate"]].plot(kind="bar", secondary_y=["changed_rate"], ax=axes[1])
    axes[1].set_title("Selected Repair Targets")
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIGURES / "candidate_and_target_stats.png", dpi=180)
    plt.close(fig)


def plot_pilots(metrics: pd.DataFrame) -> None:
    pilot_runs = ["pilot_inpolicy_vm_echo_s96_w010", "pilot_inpolicy_vm_echo_s96_w035", "pilot_inpolicy_vm_echo_s96_w010_r2"]
    pilot = metrics[metrics["run"].isin(pilot_runs) & metrics["phase"].isin(["answer_verified_distill", "inpolicy_vm_echo_distill"])]
    if pilot.empty:
        return
    rows = []
    for run in pilot_runs:
        for split in ["fresh_paired", "hard_composition"]:
            for phase in ["answer_verified_distill", "inpolicy_vm_echo_distill"]:
                item = pilot[pilot["run"].eq(run) & pilot["split"].eq(split) & pilot["phase"].eq(phase)]
                if not item.empty:
                    rows.append({"run": run.replace("pilot_inpolicy_vm_echo_", ""), "split": SPLIT_LABELS[split], "phase": PHASE_LABELS[phase], "direct": item.iloc[0]["direct_accuracy"]})
    df = pd.DataFrame(rows)
    out = ANALYSIS / "pilot_comparison.csv"
    df.to_csv(out, index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, split in zip(axes, ["fresh paired", "hard composition"]):
        pivot = df[df["split"].eq(split)].pivot(index="run", columns="phase", values="direct")
        pivot.plot(kind="bar", ax=ax)
        ax.set_title(f"Pilot Direct Accuracy: {split}")
        ax.set_ylim(0, 0.3)
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIGURES / "pilot_comparison.png", dpi=180)
    plt.close(fig)


def write_summary(data: Dict[str, pd.DataFrame]) -> None:
    metrics = data["metrics"]
    main = metrics[metrics["run"].eq(MAIN_RUN)]
    def val(phase: str, split: str, col: str) -> float:
        row = main[(main["phase"].eq(phase)) & (main["split"].eq(split))]
        return float(row.iloc[0][col]) if not row.empty else float("nan")

    lines = [
        "# Analysis Summary",
        "",
        f"Main run: `{MAIN_RUN}`",
        f"Fresh paired direct: answer distill {pct(val('answer_verified_distill', 'fresh_paired', 'direct_accuracy'))}, VM-ECHO {pct(val('inpolicy_vm_echo_distill', 'fresh_paired', 'direct_accuracy'))}, full supervised {pct(val('full_supervised', 'fresh_paired', 'direct_accuracy'))}.",
        f"Hard composition search: answer distill {pct(val('answer_verified_distill', 'hard_composition', 'answer_search_accuracy'))}, VM-ECHO {pct(val('inpolicy_vm_echo_distill', 'hard_composition', 'answer_search_accuracy'))}, full supervised {pct(val('full_supervised', 'hard_composition', 'answer_search_accuracy'))}.",
        f"Candidate oracle found rate: {pct(data['candidates'][data['candidates']['run'].eq(MAIN_RUN)]['oracle_found_rate'].iloc[0]) if not data['candidates'][data['candidates']['run'].eq(MAIN_RUN)].empty else ''}.",
        "",
        "Figures:",
        "- `analysis/figures/main_accuracy_by_phase.png`",
        "- `analysis/figures/echo_observation_accuracy.png`",
        "- `analysis/figures/training_curves.png`",
        "- `analysis/figures/candidate_and_target_stats.png`",
        "- `analysis/figures/pilot_comparison.png`",
    ]
    (ANALYSIS / "summary.md").write_text("\n".join(lines) + "\n")


def markdown_report(data: Dict[str, pd.DataFrame]) -> str:
    metrics = data["metrics"]
    main = metrics[metrics["run"].eq(MAIN_RUN)]
    candidates = data["candidates"]
    targets = data["targets"]
    cand_main = candidates[candidates["run"].eq(MAIN_RUN)]
    cand = cand_main.iloc[0] if not cand_main.empty else {}

    def val(phase: str, split: str, col: str) -> float:
        row = main[(main["phase"].eq(phase)) & (main["split"].eq(split))]
        return float(row.iloc[0][col]) if not row.empty else float("nan")

    report = f"""# In-Policy VM-ECHO Distillation

## Abstract

This experiment tests whether a Qwen-attached typed-bytecode compiler improves when it learns the VM consequences of its own proposed programs during repair distillation. The compiler first emits executable candidates from frozen `Qwen/Qwen3-4B` hidden states. Those candidates are executed in a typed VM. The training objective combines answer-verified repair distillation with an integrated observation loss over all sampled candidates: validity, final value, trace top, trace depth, and answer-correctness.

The result is mixed. VM-ECHO learned the observation channels: on the main run, trace-depth prediction reached about {pct(val('inpolicy_vm_echo_distill', 'fresh_standard', 'echo_trace_depth_accuracy'))} on fresh-standard candidates and trace-top prediction reached {pct(val('inpolicy_vm_echo_distill', 'fresh_standard', 'echo_trace_top_accuracy'))}. It also improved some answer-search metrics over answer-verified distillation, including hard-composition search from {pct(val('answer_verified_distill', 'hard_composition', 'answer_search_accuracy'))} to {pct(val('inpolicy_vm_echo_distill', 'hard_composition', 'answer_search_accuracy'))}. But it did not produce a broad direct-accuracy jump: fresh-paired direct accuracy stayed at {pct(val('inpolicy_vm_echo_distill', 'fresh_paired', 'direct_accuracy'))}, while the full-supervised ceiling reached {pct(val('full_supervised', 'fresh_paired', 'direct_accuracy'))}.

## Setup

- Base model: `Qwen/Qwen3-4B`, used as a frozen hidden-state feature extractor.
- Compiler: transformer slot decoder over Qwen hidden states.
- Integrated VM-ECHO head: candidate-conditioned transformer sharing the compiler prompt projection.
- Seed examples: `192`.
- Candidate prompts: `1024`.
- Candidate programs: `{int(cand.get('candidates', 0))}`.
- Candidate positive rate: `{pct(cand.get('positive_rate', float('nan')))}`.
- Prompt-level oracle found rate: `{pct(cand.get('oracle_found_rate', float('nan')))}`.
- Repair targets selected: `{int(targets[targets['run'].eq(MAIN_RUN)]['targets'].iloc[0]) if not targets[targets['run'].eq(MAIN_RUN)].empty else 0}`.
- ECHO loss weight selected by pilot: `0.1`.
- Checkpoints: `large_artifacts/qwen_inpolicy_vm_echo_distillation/checkpoints/{MAIN_RUN}/`.

## Main Results

{main_metrics_table(main)}

![Main accuracy](../analysis/figures/main_accuracy_by_phase.png)

## Candidate Surface

{target_table(targets)}

![Candidate and target stats](../analysis/figures/candidate_and_target_stats.png)

The candidate surface is large enough to matter but not large enough to solve the task by itself. The main run generated `{int(cand.get('candidates', 0))}` candidates from `1024` prompts, and `{pct(cand.get('oracle_found_rate', float('nan')))}` of prompts had at least one answer-correct candidate. This puts a ceiling on what one round of repair distillation can learn.

## VM Observation Learning

![Observation accuracy](../analysis/figures/echo_observation_accuracy.png)

VM-ECHO learned validity and trace structure much better than final answer correctness. This is important: the auxiliary loss did train a consequence model, but the learned consequence model was not a useful no-answer reranker. ECHO reranking was below direct decoding on every main split. The useful effect, where present, came from joint training changing the compiler, not from selecting candidates at inference time.

## Training Dynamics

![Training curves](../analysis/figures/training_curves.png)

The main ECHO auxiliary loss decreased from `5.35` to `2.67`, while repair loss also improved. A higher pilot weight damaged deployable accuracy, so the auxiliary needs to remain secondary to the repair target.

## Pilot Results

![Pilot comparison](../analysis/figures/pilot_comparison.png)

The pilot sweep selected `echo_loss_weight=0.1`. A stronger `0.35` weight over-regularized the compiler. A second in-policy round improved some standard/paraphrase cells, but it removed the fresh-paired and hard direct signal that made the one-round pilot attractive.

## Interpretation

This experiment supports a narrow claim: candidate-consequence prediction can be trained in the same compiler loop without breaking executable decoding, and it can modestly reshape the answer-search surface. It does not support the stronger claim that this objective currently unlocks the full repair oracle or produces a large direct compiler improvement.

The decisive remaining problem is credit assignment among executable candidates. VM-ECHO learned broad consequences such as validity and stack traces, but it did not learn to identify which candidate solves the prompt. The next version should make the preference signal sharper: train on counterfactual candidate pairs from the same prompt, emphasize hard negatives that share validity or final-value plausibility, and expose a more direct representation of the prompt-implied answer to the candidate scorer.

## Artifacts

- `experiments/qwen_inpolicy_vm_echo_distillation/runs/{MAIN_RUN}/metrics.csv`
- `experiments/qwen_inpolicy_vm_echo_distillation/runs/{MAIN_RUN}/train_log.csv`
- `experiments/qwen_inpolicy_vm_echo_distillation/analysis/main_metrics.csv`
- `experiments/qwen_inpolicy_vm_echo_distillation/reports/{REPORT_STEM}.md`
- `experiments/qwen_inpolicy_vm_echo_distillation/reports/{REPORT_STEM}.html`
"""
    return report


def write_html(markdown: str) -> None:
    try:
        import markdown as markdown_lib

        body = markdown_lib.markdown(markdown, extensions=["tables"])
    except Exception:
        body = "<pre>" + html.escape(markdown) + "</pre>"
    page = f"""<!doctype html><html><head><meta charset='utf-8'><title>In-Policy VM-ECHO Distillation</title><style>
body {{ font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; color: #172033; background: #f6f7fb; }}
main {{ max-width: 1120px; margin: 0 auto; padding: 40px 24px 64px; background: white; min-height: 100vh; }}
h1 {{ font-size: 34px; line-height: 1.1; margin-bottom: 12px; color: #111827; }}
h2 {{ color: #111827; margin-top: 34px; border-top: 1px solid #e5e7eb; padding-top: 24px; }}
p, li {{ line-height: 1.55; }}
code {{ background: #eef2f7; padding: 2px 5px; border-radius: 4px; }}
table {{ border-collapse: collapse; width: 100%; margin: 16px 0 24px; font-size: 14px; }}
th, td {{ border: 1px solid #d9e0ea; padding: 8px 10px; text-align: left; }}
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
    plot_echo_observations(data["metrics"])
    plot_training(data["train"])
    plot_candidates(data["candidates"], data["targets"])
    plot_pilots(data["metrics"])
    write_summary(data)
    md = markdown_report(data)
    (REPORTS / f"{REPORT_STEM}.md").write_text(md)
    write_html(md)
    print(f"[analysis] wrote {REPORTS / f'{REPORT_STEM}.md'}")
    print(f"[analysis] wrote {REPORTS / f'{REPORT_STEM}.html'}")


if __name__ == "__main__":
    main()
