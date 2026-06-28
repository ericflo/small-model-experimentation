#!/usr/bin/env python3
"""Aggregate, plot, and report the VM-ECHO trace-distillation experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import markdown as markdown_lib
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("experiments/qwen_vm_echo_trace_distillation")
RUNS = ROOT / "runs"
ANALYSIS = ROOT / "analysis"
FIGURES = ANALYSIS / "figures"
REPORTS = ROOT / "reports"
MAIN_RUN = "main_vm_echo_s192_w003"
REPORT_STEM = "qwen_vm_echo_trace_distillation_report"


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


def phase_order(phase: str) -> int:
    order = {
        "seed_supervised": 0,
        "expert_round_1": 1,
        "expert_round_2": 2,
        "full_supervised": 3,
    }
    for key, idx in order.items():
        if phase.endswith(key):
            return idx
    return 99


def phase_label(phase: str) -> str:
    for prefix in ["baseline_", "vm_echo_"]:
        if phase.startswith(prefix):
            phase = phase[len(prefix) :]
    return {
        "seed_supervised": "Seed",
        "expert_round_1": "Expert R1",
        "expert_round_2": "Expert R2",
        "full_supervised": "Full sup.",
    }.get(phase, phase)


def pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def md_table(df: pd.DataFrame, columns: Iterable[str]) -> str:
    cols = list(columns)
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df[cols].iterrows():
        rows.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(rows)


def main_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    df = metrics[metrics["run"].eq(MAIN_RUN)].copy()
    df["phase_order"] = df["phase"].map(phase_order)
    return df.sort_values(["arm", "phase_order", "split"])


def write_aggregates(metrics: pd.DataFrame, train_logs: pd.DataFrame, target_logs: pd.DataFrame) -> None:
    metrics.to_csv(ANALYSIS / "all_metrics.csv", index=False)
    train_logs.to_csv(ANALYSIS / "all_train_logs.csv", index=False)
    target_logs.to_csv(ANALYSIS / "all_expert_targets.csv", index=False)
    main = main_metrics(metrics)
    main.to_csv(ANALYSIS / "main_metrics.csv", index=False)
    if not target_logs.empty:
        target_logs[target_logs["run"].eq(MAIN_RUN)].to_csv(ANALYSIS / "main_expert_targets.csv", index=False)


def plot_main_phase_curves(metrics: pd.DataFrame) -> None:
    main = main_metrics(metrics)
    splits = ["fresh_paired", "hard_composition"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    colors = {"baseline": "#4C78A8", "vm_echo": "#F58518"}
    for ax, split in zip(axes, splits):
        sub = main[main["split"].eq(split)]
        for arm in ["baseline", "vm_echo"]:
            arm_df = sub[sub["arm"].eq(arm)].sort_values("phase_order")
            ax.plot(
                [phase_label(p) for p in arm_df["phase"]],
                100 * arm_df["direct_accuracy"],
                marker="o",
                linewidth=2,
                color=colors[arm],
                linestyle="-",
                label=f"{arm} direct",
            )
            ax.plot(
                [phase_label(p) for p in arm_df["phase"]],
                100 * arm_df["search_accuracy"],
                marker="s",
                linewidth=2,
                color=colors[arm],
                linestyle="--",
                label=f"{arm} search",
            )
        ax.set_title(split.replace("_", " "))
        ax.set_ylabel("Accuracy (%)")
        ax.grid(alpha=0.25)
        ax.tick_params(axis="x", rotation=20)
    axes[0].legend(fontsize=8)
    fig.suptitle("Main run accuracy by training phase")
    fig.tight_layout()
    fig.savefig(FIGURES / "main_phase_curves.png", dpi=180)
    plt.close(fig)


def plot_full_supervised_split_bars(metrics: pd.DataFrame) -> None:
    main = main_metrics(metrics)
    phases = ["baseline_full_supervised", "vm_echo_full_supervised"]
    splits = ["fresh_standard", "fresh_paraphrase", "fresh_paired", "hard_composition"]
    sub = main[main["phase"].isin(phases) & main["split"].isin(splits)].copy()
    labels = [s.replace("_", "\n") for s in splits]
    x = range(len(splits))
    width = 0.2
    fig, ax = plt.subplots(figsize=(11, 5.6))
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    series = [
        ("baseline", "direct_accuracy", "Baseline direct", "#4C78A8"),
        ("vm_echo", "direct_accuracy", "VM-ECHO direct", "#F58518"),
        ("baseline", "search_accuracy", "Baseline search", "#72B7B2"),
        ("vm_echo", "search_accuracy", "VM-ECHO search", "#E45756"),
    ]
    for off, (arm, metric, label, color) in zip(offsets, series):
        values = []
        for split in splits:
            row = sub[sub["arm"].eq(arm) & sub["split"].eq(split)].iloc[0]
            values.append(100 * row[metric])
        ax.bar([i + off for i in x], values, width=width, label=label, color=color)
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Full-supervised generalization by split")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "full_supervised_split_bars.png", dpi=180)
    plt.close(fig)


def plot_echo_observation(metrics: pd.DataFrame) -> None:
    main = main_metrics(metrics)
    sub = main[main["arm"].eq("vm_echo") & main["split"].isin(["val_mixed", "fresh_paired", "hard_composition"])].copy()
    sub = sub.sort_values(["split", "phase_order"])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    observed = [
        ("echo_trace_top_accuracy", "Trace top"),
        ("echo_trace_depth_accuracy", "Trace depth"),
        ("echo_final_accuracy", "Final value"),
    ]
    for ax, (metric, title) in zip(axes, observed):
        for split in ["val_mixed", "fresh_paired", "hard_composition"]:
            d = sub[sub["split"].eq(split)]
            ax.plot([phase_label(p) for p in d["phase"]], 100 * d[metric], marker="o", linewidth=2, label=split.replace("_", " "))
        ax.set_title(title)
        ax.set_ylabel("Observation accuracy (%)")
        ax.grid(alpha=0.25)
        ax.tick_params(axis="x", rotation=20)
    axes[0].legend(fontsize=8)
    fig.suptitle("VM-ECHO observation prediction in the main run")
    fig.tight_layout()
    fig.savefig(FIGURES / "echo_observation_accuracy.png", dpi=180)
    plt.close(fig)


def plot_expert_targets(target_logs: pd.DataFrame) -> None:
    if target_logs.empty:
        return
    sub = target_logs[target_logs["run"].eq(MAIN_RUN)].copy()
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    labels = [f"{a} R{r}" for a, r in zip(sub["arm"], sub["round"])]
    ax.bar(labels, 100 * sub["found_rate"], color=["#4C78A8" if a == "baseline" else "#F58518" for a in sub["arm"]])
    ax.set_ylabel("Answer-verified target rate (%)")
    ax.set_title("Expert target collection")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "expert_target_rates.png", dpi=180)
    plt.close(fig)


def plot_weight_sweep(metrics: pd.DataFrame) -> None:
    runs = {
        "baseline": ("pilot_vm_echo_s96", "baseline"),
        "0.35": ("pilot_vm_echo_s96", "vm_echo"),
        "0.10": ("pilot_vm_echo_s96_w010", "vm_echo"),
        "0.03": ("pilot_vm_echo_s96_w003", "vm_echo"),
    }
    rows = []
    for label, (run, arm) in runs.items():
        sub = metrics[metrics["run"].eq(run) & metrics["arm"].eq(arm) & metrics["split"].isin(["fresh_paired", "hard_composition"])].copy()
        for phase_key in ["seed_supervised", "expert_round_1", "full_supervised"]:
            d = sub[sub["phase"].str.endswith(phase_key)]
            if d.empty:
                continue
            rows.append(
                {
                    "weight": label,
                    "phase": phase_key,
                    "fresh_paired_search": float(d[d["split"].eq("fresh_paired")]["search_accuracy"].iloc[0]),
                    "hard_search": float(d[d["split"].eq("hard_composition")]["search_accuracy"].iloc[0]),
                }
            )
    sweep = pd.DataFrame(rows)
    sweep.to_csv(ANALYSIS / "pilot_weight_sweep.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, metric, title in [
        (axes[0], "fresh_paired_search", "Fresh paired search"),
        (axes[1], "hard_search", "Hard composition search"),
    ]:
        for phase in ["seed_supervised", "expert_round_1", "full_supervised"]:
            d = sweep[sweep["phase"].eq(phase)]
            ax.plot(d["weight"], 100 * d[metric], marker="o", linewidth=2, label=phase_label(phase))
        ax.set_title(title)
        ax.set_xlabel("ECHO weight")
        ax.set_ylabel("Accuracy (%)")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("Pilot ECHO-weight sweep")
    fig.tight_layout()
    fig.savefig(FIGURES / "pilot_weight_sweep.png", dpi=180)
    plt.close(fig)


def generate_figures(metrics: pd.DataFrame, target_logs: pd.DataFrame) -> None:
    plot_main_phase_curves(metrics)
    plot_full_supervised_split_bars(metrics)
    plot_echo_observation(metrics)
    plot_expert_targets(target_logs)
    plot_weight_sweep(metrics)


def make_main_table(metrics: pd.DataFrame) -> pd.DataFrame:
    main = main_metrics(metrics)
    keep = main[main["split"].isin(["fresh_standard", "fresh_paraphrase", "fresh_paired", "hard_composition"])].copy()
    keep = keep[keep["phase"].isin(["baseline_expert_round_2", "vm_echo_expert_round_2", "baseline_full_supervised", "vm_echo_full_supervised"])]
    out = keep[["arm", "phase", "split", "direct_accuracy", "search_accuracy", "program_exact", "echo_trace_top_accuracy"]].copy()
    out["phase"] = out["phase"].map(phase_label)
    for col in ["direct_accuracy", "search_accuracy", "program_exact", "echo_trace_top_accuracy"]:
        out[col] = out[col].map(pct)
    out.columns = ["Arm", "Phase", "Split", "Direct", "Search", "Program exact", "Trace-top acc."]
    return out


def report_text(metrics: pd.DataFrame, target_logs: pd.DataFrame, metadata: dict[str, dict[str, Any]]) -> str:
    main = main_metrics(metrics)
    table = make_main_table(metrics)
    full = main[main["phase"].str.endswith("full_supervised")]
    expert2 = main[main["phase"].str.endswith("expert_round_2")]
    fp_full_base = full[full["arm"].eq("baseline") & full["split"].eq("fresh_paired")].iloc[0]
    fp_full_echo = full[full["arm"].eq("vm_echo") & full["split"].eq("fresh_paired")].iloc[0]
    hard_e2_base = expert2[expert2["arm"].eq("baseline") & expert2["split"].eq("hard_composition")].iloc[0]
    hard_e2_echo = expert2[expert2["arm"].eq("vm_echo") & expert2["split"].eq("hard_composition")].iloc[0]
    target_main = target_logs[target_logs["run"].eq(MAIN_RUN)] if not target_logs.empty else pd.DataFrame()
    target_lines = ""
    if not target_main.empty:
        target_lines = md_table(
            target_main[["arm", "round", "targets", "found_rate", "candidate_valid_rate"]].assign(
                found_rate=lambda d: d["found_rate"].map(pct),
                candidate_valid_rate=lambda d: d["candidate_valid_rate"].map(pct),
            ),
            ["arm", "round", "targets", "found_rate", "candidate_valid_rate"],
        )
    manifest = metadata.get(MAIN_RUN, {})
    sizes = manifest.get("sizes", {})
    return f"""# VM-ECHO Trace Distillation for a Frozen-Qwen Bytecode Compiler

## Abstract

This standalone experiment tests whether a frozen-Qwen typed-bytecode compiler benefits from an auxiliary VM-observation objective. The baseline learns to emit bytecode and a final answer. The VM-ECHO arm gets the same program loss plus a low-weight loss for predicting execution observations: VM validity, final value, stack top after each active bytecode slot, and stack depth after each active bytecode slot.

The result is mixed. VM-ECHO clearly learns the VM observation channels: in the main full-supervised arm, fresh-paired trace-top prediction rises from {pct(float(fp_full_base["echo_trace_top_accuracy"]))} to {pct(float(fp_full_echo["echo_trace_top_accuracy"]))}. That extra semantic signal does not translate into a broad direct-accuracy jump. It gives modest local gains in some search/oracle settings, for example hard-composition expert-round-2 search rises from {pct(float(hard_e2_base["search_accuracy"]))} to {pct(float(hard_e2_echo["search_accuracy"]))}, and full-supervised fresh-paired search rises from {pct(float(fp_full_base["search_accuracy"]))} to {pct(float(fp_full_echo["search_accuracy"]))}. But full-supervised fresh-standard direct accuracy falls from {pct(float(full[full["arm"].eq("baseline") & full["split"].eq("fresh_standard")]["direct_accuracy"].iloc[0]))} to {pct(float(full[full["arm"].eq("vm_echo") & full["split"].eq("fresh_standard")]["direct_accuracy"].iloc[0]))}. This is not a universal improvement.

## Setup

- Base model: `{manifest.get("model_name", "Qwen/Qwen3-4B")}`, used only as a frozen hidden-state feature extractor.
- Compiler: transformer-decoder slot head over Qwen hidden states.
- VM: typed stack bytecode with `{sizes.get("seed_train", "?")}` seed examples, `{sizes.get("unlabeled_train", "?")}` unlabeled expert-iteration prompts, `{sizes.get("full_supervised_train", "?")}` full-supervised examples, and `{sizes.get("fresh_standard", "?")}` examples per fresh split.
- Main ECHO weight: `0.03`. Pilot weights `0.35`, `0.10`, and `0.03` were used only to choose a non-destructive auxiliary-loss scale.
- Checkpoints: `large_artifacts/qwen_vm_echo_trace_distillation/checkpoints/{MAIN_RUN}/`.

## Main Results

{md_table(table, table.columns)}

![Main phase curves](../analysis/figures/main_phase_curves.png)

![Full-supervised split bars](../analysis/figures/full_supervised_split_bars.png)

## VM Observation Learning

The auxiliary heads learned the execution-observation task, especially stack depth. Trace-top accuracy also rose substantially in the full-supervised VM-ECHO arm, but final-value prediction stayed modest because it is a 97-way target and the main answer head already carries a separate final-answer signal.

![ECHO observation accuracy](../analysis/figures/echo_observation_accuracy.png)

## Expert Target Collection

VM-ECHO at weight `0.03` did not collapse the candidate set. It collected slightly more round-1 expert targets than the baseline and slightly fewer round-2 targets.

{target_lines}

![Expert target rates](../analysis/figures/expert_target_rates.png)

## Weight Sweep

The pilot sweep showed why the main run used a low weight. At `0.35`, VM-ECHO learned observations but damaged candidate search. At `0.03`, it preserved the search surface better.

![Pilot weight sweep](../analysis/figures/pilot_weight_sweep.png)

## Interpretation

The useful finding is not that VM-ECHO is a breakthrough by itself. The useful finding is sharper: a consequence-prediction loss can be attached to a frozen-Qwen bytecode compiler without breaking typed decoding, and it can make the model learn nontrivial VM-state predictions. However, teacher-forced trace prediction is only weakly coupled to choosing better programs. The next version should condition the observation predictor on candidate programs sampled from the compiler, so the model learns consequences of its own actions rather than consequences of the gold target alone.

## Artifacts

- `experiments/qwen_vm_echo_trace_distillation/runs/{MAIN_RUN}/metrics.csv`
- `experiments/qwen_vm_echo_trace_distillation/runs/{MAIN_RUN}/train_log.csv`
- `experiments/qwen_vm_echo_trace_distillation/analysis/main_metrics.csv`
- `experiments/qwen_vm_echo_trace_distillation/reports/{REPORT_STEM}.md`
- `experiments/qwen_vm_echo_trace_distillation/reports/{REPORT_STEM}.html`
"""


def write_report(markdown: str, main_table: pd.DataFrame) -> None:
    md_path = REPORTS / f"{REPORT_STEM}.md"
    html_path = REPORTS / f"{REPORT_STEM}.html"
    md_path.write_text(markdown)
    rendered = markdown_lib.markdown(markdown, extensions=["tables", "fenced_code"])
    css = """
body { font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; color: #18212f; background: #f7f8fb; }
main { max-width: 1100px; margin: 0 auto; padding: 40px 24px 64px; background: white; min-height: 100vh; }
h1, h2 { color: #111827; }
h1 { font-size: 34px; line-height: 1.1; margin-bottom: 12px; }
h2 { margin-top: 34px; border-top: 1px solid #e5e7eb; padding-top: 24px; }
p, li { line-height: 1.55; }
code { background: #eef2f7; padding: 2px 5px; border-radius: 4px; }
table { border-collapse: collapse; width: 100%; margin: 16px 0 24px; font-size: 14px; }
th, td { border: 1px solid #d9e0ea; padding: 8px 10px; text-align: left; }
th { background: #edf2f7; }
img { max-width: 100%; display: block; margin: 18px 0 30px; border: 1px solid #d9e0ea; }
.note { color: #4b5563; font-size: 14px; }
"""
    body = f"""
<main>
<p class="note">Standalone report generated from <code>{MAIN_RUN}</code>.</p>
{rendered}
</main>
"""
    html_path.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>VM-ECHO Trace Distillation</title><style>{css}</style></head><body>{body}</body></html>")


def write_summary(metrics: pd.DataFrame, target_logs: pd.DataFrame) -> None:
    main = main_metrics(metrics)
    full = main[main["phase"].str.endswith("full_supervised")]
    paired_base = full[full["arm"].eq("baseline") & full["split"].eq("fresh_paired")].iloc[0]
    paired_echo = full[full["arm"].eq("vm_echo") & full["split"].eq("fresh_paired")].iloc[0]
    lines = [
        "# Analysis Summary",
        "",
        f"Main run: `{MAIN_RUN}`",
        f"Fresh paired full-supervised direct: baseline {pct(float(paired_base['direct_accuracy']))}, VM-ECHO {pct(float(paired_echo['direct_accuracy']))}.",
        f"Fresh paired full-supervised search: baseline {pct(float(paired_base['search_accuracy']))}, VM-ECHO {pct(float(paired_echo['search_accuracy']))}.",
        f"Fresh paired trace-top observation: baseline {pct(float(paired_base['echo_trace_top_accuracy']))}, VM-ECHO {pct(float(paired_echo['echo_trace_top_accuracy']))}.",
        "",
        "Figures:",
        "- `analysis/figures/main_phase_curves.png`",
        "- `analysis/figures/full_supervised_split_bars.png`",
        "- `analysis/figures/echo_observation_accuracy.png`",
        "- `analysis/figures/expert_target_rates.png`",
        "- `analysis/figures/pilot_weight_sweep.png`",
    ]
    if not target_logs.empty:
        main_targets = target_logs[target_logs["run"].eq(MAIN_RUN)]
        lines += ["", "Main expert targets:", main_targets.to_string(index=False)]
    (ANALYSIS / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    ensure_dirs()
    metrics = read_csvs("metrics.csv")
    train_logs = read_csvs("train_log.csv")
    target_logs = read_csvs("expert_targets.csv")
    metadata = load_metadata()
    if metrics.empty:
        raise SystemExit("No metrics.csv files found")
    write_aggregates(metrics, train_logs, target_logs)
    generate_figures(metrics, target_logs)
    main_table = make_main_table(metrics)
    markdown = report_text(metrics, target_logs, metadata)
    write_report(markdown, main_table)
    write_summary(metrics, target_logs)
    print(f"[analysis] wrote {REPORTS / (REPORT_STEM + '.md')}")
    print(f"[analysis] wrote {REPORTS / (REPORT_STEM + '.html')}")


if __name__ == "__main__":
    main()
