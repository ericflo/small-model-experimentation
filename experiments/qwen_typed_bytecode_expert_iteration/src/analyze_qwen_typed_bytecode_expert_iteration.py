#!/usr/bin/env python3
"""Analyze typed-bytecode expert-iteration runs."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import markdown


ROOT = Path("experiments/qwen_typed_bytecode_expert_iteration")
RUNS = ROOT / "runs"
ANALYSIS = ROOT / "analysis"
FIGURES = ANALYSIS / "figures"
REPORTS = ROOT / "reports"
CHECKPOINT_ROOT = Path("large_artifacts/qwen_typed_bytecode_expert_iteration/checkpoints")
MAIN_RUN = "main_typed_bytecode_ei_s384_u4096"
QWEN_RUN = "qwen_head_pilot_s384_u2048"


def pct(x: Any) -> str:
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


def read_metrics() -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in sorted(RUNS.glob("*/metrics.csv")):
        frames.append(pd.read_csv(path))
    if not frames:
        raise SystemExit("no metrics.csv files found")
    return pd.concat(frames, ignore_index=True, sort=False)


def read_targets() -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in sorted(RUNS.glob("*/expert_targets.csv")):
        df = pd.read_csv(path)
        df.insert(0, "run", path.parent.name)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def read_train_logs() -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in sorted(RUNS.glob("*/train_log.csv")):
        df = pd.read_csv(path)
        df.insert(0, "run", path.parent.name)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def main_metrics(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["run"].eq(MAIN_RUN)].copy()


def phase_order(phase: str) -> int:
    if phase == "seed_supervised":
        return 0
    if phase.startswith("expert_round_"):
        return int(phase.rsplit("_", 1)[-1])
    if phase == "full_supervised":
        return 99
    return 50


def save_tables(metrics: pd.DataFrame, targets: pd.DataFrame, logs: pd.DataFrame) -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(ANALYSIS / "all_final_metrics.csv", index=False)
    main = main_metrics(metrics)
    main.to_csv(ANALYSIS / "final_metrics.csv", index=False)
    if not targets.empty:
        targets.to_csv(ANALYSIS / "expert_target_quality.csv", index=False)
    if not logs.empty:
        logs.to_csv(ANALYSIS / "train_logs.csv", index=False)


def plot_phase_lines(main: pd.DataFrame, metric: str, filename: str, title: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    splits = ["val_mixed", "fresh_standard", "fresh_paraphrase", "fresh_paired", "hard_composition"]
    labels = {
        "val_mixed": "Validation",
        "fresh_standard": "Fresh standard",
        "fresh_paraphrase": "Fresh paraphrase",
        "fresh_paired": "Fresh paired",
        "hard_composition": "Hard composition",
    }
    phases = sorted(main["phase"].unique(), key=phase_order)
    x = list(range(len(phases)))
    xlabels = [p.replace("seed_supervised", "seed").replace("expert_round_", "EI ").replace("full_supervised", "full") for p in phases]
    plt.figure(figsize=(10, 5.6))
    for split in splits:
        sub = main[main["split"].eq(split)].copy()
        if sub.empty:
            continue
        values = []
        for phase in phases:
            row = sub[sub["phase"].eq(phase)]
            values.append(float(row.iloc[0][metric]) if not row.empty else float("nan"))
        plt.plot(x, [100 * v for v in values], marker="o", linewidth=2, label=labels[split])
    plt.xticks(x, xlabels, rotation=20)
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / filename, dpi=180)
    plt.close()


def plot_fresh_paired_progress(main: pd.DataFrame) -> None:
    sub = main[main["split"].eq("fresh_paired")].sort_values("phase", key=lambda s: s.map(phase_order))
    labels = [p.replace("seed_supervised", "seed").replace("expert_round_", "EI ").replace("full_supervised", "full") for p in sub["phase"]]
    x = range(len(sub))
    plt.figure(figsize=(9, 5.4))
    plt.plot(x, 100 * sub["direct_accuracy"], marker="o", linewidth=2, label="Direct compiler")
    plt.plot(x, 100 * sub["search_accuracy"], marker="o", linewidth=2, label="Answer-verified local search")
    plt.plot(x, 100 * sub["program_exact"], marker="o", linewidth=2, label="Program exact")
    plt.xticks(list(x), labels, rotation=20)
    plt.ylabel("Fresh paired accuracy (%)")
    plt.title("Fresh paired progress")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "fresh_paired_progress.png", dpi=180)
    plt.close()


def plot_target_quality(targets: pd.DataFrame) -> None:
    sub = targets[targets["run"].eq(MAIN_RUN)].copy()
    if sub.empty:
        return
    plt.figure(figsize=(8.5, 5.0))
    plt.plot(sub["round"], 100 * sub["found_rate"], marker="o", linewidth=2, label="Answer-verified target found")
    plt.plot(sub["round"], 100 * sub["changed_rate"], marker="o", linewidth=2, label="Targets changed from base")
    plt.plot(sub["round"], 100 * sub["candidate_valid_rate"], marker="o", linewidth=2, label="Candidate valid rate")
    plt.xlabel("Expert-iteration round")
    plt.ylabel("Rate (%)")
    plt.title("Expert target quality")
    plt.xticks(list(sub["round"]))
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "expert_target_quality.png", dpi=180)
    plt.close()


def plot_ceiling_bars(main: pd.DataFrame) -> None:
    splits = ["fresh_standard", "fresh_paraphrase", "fresh_paired", "hard_composition"]
    labels = ["Standard", "Paraphrase", "Paired", "Hard"]
    seed = [float(main[(main.split.eq(s)) & (main.phase.eq("seed_supervised"))]["direct_accuracy"].iloc[0]) for s in splits]
    ei = [float(main[(main.split.eq(s)) & (main.phase.eq("expert_round_4"))]["direct_accuracy"].iloc[0]) for s in splits]
    full = [float(main[(main.split.eq(s)) & (main.phase.eq("full_supervised"))]["direct_accuracy"].iloc[0]) for s in splits]
    width = 0.24
    x = range(len(splits))
    plt.figure(figsize=(9.5, 5.4))
    plt.bar([i - width for i in x], [100 * v for v in seed], width=width, label="Seed supervised")
    plt.bar(list(x), [100 * v for v in ei], width=width, label="Expert iteration")
    plt.bar([i + width for i in x], [100 * v for v in full], width=width, label="Full supervised")
    plt.xticks(list(x), labels)
    plt.ylabel("Direct compiler accuracy (%)")
    plt.title("Direct compiler accuracy by training regime")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "direct_regime_bars.png", dpi=180)
    plt.close()


def plot_qwen_progress(metrics: pd.DataFrame) -> None:
    sub = metrics[(metrics["run"].eq(QWEN_RUN)) & (metrics["split"].eq("fresh_paired"))].copy()
    if sub.empty:
        return
    sub = sub.sort_values("phase", key=lambda s: s.map(phase_order))
    labels = [p.replace("seed_supervised", "seed").replace("expert_round_", "EI ").replace("full_supervised", "full") for p in sub["phase"]]
    x = range(len(sub))
    plt.figure(figsize=(9, 5.2))
    plt.plot(x, 100 * sub["direct_accuracy"], marker="o", linewidth=2, label="Direct compiler")
    plt.plot(x, 100 * sub["search_accuracy"], marker="o", linewidth=2, label="Answer-verified search")
    plt.plot(x, 100 * sub["program_exact"], marker="o", linewidth=2, label="Program exact")
    plt.xticks(list(x), labels, rotation=20)
    plt.ylabel("Fresh paired accuracy (%)")
    plt.title("Frozen-Qwen head pilot progress")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "qwen_head_fresh_paired_progress.png", dpi=180)
    plt.close()


def make_summary(main: pd.DataFrame, targets: pd.DataFrame) -> str:
    all_metrics = read_metrics()
    fp = main[main["split"].eq("fresh_paired")].copy()
    seed = fp[fp["phase"].eq("seed_supervised")].iloc[0]
    ei = fp[fp["phase"].eq("expert_round_4")].iloc[0]
    full = fp[fp["phase"].eq("full_supervised")].iloc[0]
    hard = main[main["split"].eq("hard_composition")].copy()
    hard_seed = hard[hard["phase"].eq("seed_supervised")].iloc[0]
    hard_ei = hard[hard["phase"].eq("expert_round_4")].iloc[0]
    hard_full = hard[hard["phase"].eq("full_supervised")].iloc[0]
    target_main = targets[targets["run"].eq(MAIN_RUN)].copy() if not targets.empty else pd.DataFrame()
    final_targets = target_main[target_main["round"].eq(target_main["round"].max())].iloc[0] if not target_main.empty else None
    lines = [
        "# Qwen Typed Bytecode Expert Iteration Summary",
        "",
        f"Primary run: `{MAIN_RUN}`",
        "",
        "## Headline",
        "",
        f"- Fresh paired seed direct accuracy: {pct(seed['direct_accuracy'])}",
        f"- Fresh paired expert-iteration direct accuracy: {pct(ei['direct_accuracy'])} ({pp(ei['direct_accuracy'] - seed['direct_accuracy'])})",
        f"- Fresh paired full-supervised direct accuracy: {pct(full['direct_accuracy'])}",
        f"- Fresh paired expert-iteration search accuracy: {pct(ei['search_accuracy'])}",
        f"- Fresh paired full-supervised search accuracy: {pct(full['search_accuracy'])}",
        f"- Hard-composition seed/expert/full direct: {pct(hard_seed['direct_accuracy'])} / {pct(hard_ei['direct_accuracy'])} / {pct(hard_full['direct_accuracy'])}",
    ]
    if final_targets is not None:
        lines.extend(
            [
                "",
                "## Final Expert Round",
                "",
                f"- Answer-verified targets found: {pct(final_targets['found_rate'])}",
                f"- Changed-target rate among found targets: {pct(final_targets['changed_rate'])}",
                f"- Mean local candidates per prompt: {float(final_targets['mean_candidates']):.1f}",
                f"- Candidate valid rate: {pct(final_targets['candidate_valid_rate'])}",
            ]
        )
    qwen = all_metrics[all_metrics["run"].eq(QWEN_RUN)].copy()
    if not qwen.empty:
        qfp = qwen[qwen["split"].eq("fresh_paired")]
        q_seed = qfp[qfp["phase"].eq("seed_supervised")].iloc[0]
        q_ei = qfp[qfp["phase"].eq("expert_round_3")].iloc[0]
        q_full = qfp[qfp["phase"].eq("full_supervised")].iloc[0]
        lines.extend(
            [
                "",
                "## Frozen-Qwen Attached Pilot",
                "",
                f"- Fresh paired seed direct accuracy: {pct(q_seed['direct_accuracy'])}",
                f"- Fresh paired expert-iteration direct accuracy: {pct(q_ei['direct_accuracy'])} ({pp(q_ei['direct_accuracy'] - q_seed['direct_accuracy'])})",
                f"- Fresh paired full-supervised direct accuracy: {pct(q_full['direct_accuracy'])}",
                f"- Fresh paired expert-iteration search accuracy: {pct(q_ei['search_accuracy'])}",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The typed stack-machine ABI made dense bytecode supervision highly learnable. Full supervised training reached near-ceiling fresh accuracy and preserved strong hard-composition transfer. Answer-verified expert iteration also improved the deployable compiler, but it did not close the full supervised gap. The remaining bottleneck is target quality: final-answer verification produces many useful targets, but it is weaker than direct bytecode traces and can admit semantically accidental programs.",
        ]
    )
    text = "\n".join(lines) + "\n"
    (ANALYSIS / "summary.md").write_text(text)
    return text


def table_md(rows: pd.DataFrame, columns: List[str], label_map: Dict[str, str]) -> str:
    out = ["| " + " | ".join(label_map.get(c, c) for c in columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in rows.iterrows():
        vals = []
        for c in columns:
            if c in {
                "direct_accuracy",
                "search_accuracy",
                "oracle_accuracy",
                "program_exact",
                "found_rate",
                "gap_recovered",
                "changed_rate",
                "candidate_valid_rate",
            }:
                vals.append(pct(row[c]))
            elif c in {"round", "targets"}:
                vals.append(str(int(row[c])))
            elif c == "mean_candidates":
                vals.append(f"{float(row[c]):.1f}")
            else:
                vals.append(str(row[c]))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def make_report(metrics: pd.DataFrame, targets: pd.DataFrame) -> None:
    main = main_metrics(metrics)
    fp = main[main["split"].eq("fresh_paired")].sort_values("phase", key=lambda s: s.map(phase_order))
    final_phases = main[main["phase"].isin(["seed_supervised", "expert_round_4", "full_supervised"])].copy()
    final_fresh = final_phases[final_phases["split"].isin(["fresh_standard", "fresh_paraphrase", "fresh_paired", "hard_composition"])].copy()
    final_fresh["phase"] = final_fresh["phase"].replace(
        {"seed_supervised": "Seed supervised", "expert_round_4": "Expert iteration R4", "full_supervised": "Full supervised"}
    )
    final_fresh["split"] = final_fresh["split"].replace(
        {
            "fresh_standard": "Fresh standard",
            "fresh_paraphrase": "Fresh paraphrase",
            "fresh_paired": "Fresh paired",
            "hard_composition": "Hard composition",
        }
    )
    target_main = targets[targets["run"].eq(MAIN_RUN)].copy() if not targets.empty else pd.DataFrame()
    qwen = metrics[metrics["run"].eq(QWEN_RUN)].copy()
    qwen_target = targets[targets["run"].eq(QWEN_RUN)].copy() if not targets.empty else pd.DataFrame()

    lines = [
        "# Qwen Typed Bytecode Expert Iteration",
        "",
        "## Abstract",
        "",
        "This experiment tests a typed-bytecode posttraining recipe in a controlled text-to-program compiler. A compact transformer reads natural-language prompts and emits a fixed-length typed stack-machine program. The bytecode is validated and executed by an exact interpreter. The main question is whether answer-verified local search can create useful expert-iteration targets, and how that compares with dense supervised bytecode traces.",
        "",
        "The primary run trained a seed compiler on 384 gold bytecode traces, then ran four rounds of answer-verified expert iteration over 4,096 generated training prompts. A separate full-supervised ceiling trained on 4,096 gold traces. On fresh paired prompts, the seed compiler reached "
        + f"{pct(fp[fp.phase.eq('seed_supervised')]['direct_accuracy'].iloc[0])}; expert iteration reached "
        + f"{pct(fp[fp.phase.eq('expert_round_4')]['direct_accuracy'].iloc[0])}; dense full supervision reached "
        + f"{pct(fp[fp.phase.eq('full_supervised')]['direct_accuracy'].iloc[0])}.",
        "",
        "## Setup",
        "",
        "- Runtime: exact typed stack-machine bytecode over bounded i32 values modulo 97.",
        "- Opcodes: `PUSH`, arithmetic, comparisons, min/max, two lookup host calls, `END`, and `PAD`.",
        "- Domains: modular arithmetic, calendar offsets, unit scaling, list aggregation, boolean thresholds, and table lookup.",
        "- Compiler: compact transformer encoder/decoder over tokenized prompts and fixed program slots.",
        "- Qwen-attached pilot: frozen `Qwen/Qwen3-4B` hidden states with a trainable bytecode compiler head.",
        "- Expert iteration: local candidates are generated from compiler logits, executed, and accepted as training targets when their final answer matches the task answer.",
        "- Primary run: `main_typed_bytecode_ei_s384_u4096`.",
        "",
        "## Main Results",
        "",
        table_md(
            final_fresh[["phase", "split", "direct_accuracy", "search_accuracy", "program_exact", "found_rate"]],
            ["phase", "split", "direct_accuracy", "search_accuracy", "program_exact", "found_rate"],
            {
                "phase": "Training regime",
                "split": "Split",
                "direct_accuracy": "Direct",
                "search_accuracy": "Search",
                "program_exact": "Program exact",
                "found_rate": "Target found",
            },
        ),
        "",
        "![Direct regime bars](../analysis/figures/direct_regime_bars.png)",
        "",
        "## Expert-Iteration Curve",
        "",
        "Answer-verified expert iteration produced a real deployable improvement, not just a search-time improvement. The same trained compiler is evaluated directly after each round.",
        "",
        "![Fresh paired progress](../analysis/figures/fresh_paired_progress.png)",
        "",
        "![Direct phase lines](../analysis/figures/direct_accuracy_by_phase.png)",
        "",
        "## Search Headroom",
        "",
        "Local answer-verified search remained substantially stronger than direct decoding through the expert-iteration rounds, which means the compiler still leaves repairable mistakes on the table.",
        "",
        "![Search phase lines](../analysis/figures/search_accuracy_by_phase.png)",
        "",
        "## Target Quality",
        "",
    ]
    if not target_main.empty:
        lines.extend(
            [
                table_md(
                    target_main[["round", "targets", "found_rate", "changed_rate", "mean_candidates", "candidate_valid_rate"]],
                    ["round", "targets", "found_rate", "changed_rate", "mean_candidates", "candidate_valid_rate"],
                    {
                        "round": "Round",
                        "targets": "Targets",
                        "found_rate": "Found",
                        "changed_rate": "Changed",
                        "mean_candidates": "Candidates",
                        "candidate_valid_rate": "Valid candidates",
                    },
                ),
                "",
                "![Expert target quality](../analysis/figures/expert_target_quality.png)",
                "",
            ]
        )
    if not qwen.empty:
        qwen_final = qwen[qwen["phase"].isin(["seed_supervised", "expert_round_3", "full_supervised"])].copy()
        qwen_final = qwen_final[qwen_final["split"].isin(["fresh_standard", "fresh_paraphrase", "fresh_paired", "hard_composition"])]
        qwen_final["phase"] = qwen_final["phase"].replace(
            {"seed_supervised": "Qwen seed", "expert_round_3": "Qwen expert iteration R3", "full_supervised": "Qwen full supervised"}
        )
        qwen_final["split"] = qwen_final["split"].replace(
            {
                "fresh_standard": "Fresh standard",
                "fresh_paraphrase": "Fresh paraphrase",
                "fresh_paired": "Fresh paired",
                "hard_composition": "Hard composition",
            }
        )
        lines.extend(
            [
                "## Frozen-Qwen Attached Pilot",
                "",
                "A companion pilot attached the same bytecode head to frozen `Qwen/Qwen3-4B` token hidden states. Only the bytecode head was trained; Qwen itself was not LoRA-tuned in this run. This checks whether the method still has signal when the text front end is a real 4B model representation.",
                "",
                table_md(
                    qwen_final[["phase", "split", "direct_accuracy", "search_accuracy", "program_exact", "found_rate"]],
                    ["phase", "split", "direct_accuracy", "search_accuracy", "program_exact", "found_rate"],
                    {
                        "phase": "Training regime",
                        "split": "Split",
                        "direct_accuracy": "Direct",
                        "search_accuracy": "Search",
                        "program_exact": "Program exact",
                        "found_rate": "Target found",
                    },
                ),
                "",
                "![Frozen-Qwen head progress](../analysis/figures/qwen_head_fresh_paired_progress.png)",
                "",
            ]
        )
        if not qwen_target.empty:
            lines.extend(
                [
                    "Qwen-head target quality:",
                    "",
                    table_md(
                        qwen_target[["round", "targets", "found_rate", "changed_rate", "mean_candidates", "candidate_valid_rate"]],
                        ["round", "targets", "found_rate", "changed_rate", "mean_candidates", "candidate_valid_rate"],
                        {
                            "round": "Round",
                            "targets": "Targets",
                            "found_rate": "Found",
                            "changed_rate": "Changed",
                            "mean_candidates": "Candidates",
                            "candidate_valid_rate": "Valid candidates",
                        },
                    ),
                    "",
                ]
            )
    lines.extend(
        [
            "## Interpretation",
            "",
            "The result is positive for the typed-bytecode substrate and mixed for answer-only expert iteration. Dense bytecode traces are extremely effective: full supervision nearly saturates fresh standard, paraphrase, and paired splits. Expert iteration also helps, moving the seed compiler upward on every fresh split, but it does not approach the dense-trace ceiling. The frozen-Qwen pilot shows the same qualitative pattern: expert iteration improves the trainable Qwen-attached head, while dense bytecode supervision remains much stronger. This suggests that the next method improvement should focus on stronger process verification, multi-input consistency, or prefix-level search targets rather than merely increasing the number of final-answer-verified candidates.",
            "",
            "The hard-composition split is the useful warning. Full supervision reached high but not saturated hard accuracy, while expert iteration improved less. This means the bytecode ABI is learnable, but longer or more compositional programs still need either more trace coverage or a better search/value loop.",
            "",
            "## Limitations",
            "",
            "- The primary controlled run uses a compact transformer compiler; the separate Qwen-attached pilot trains only a head on frozen Qwen hidden states, not Qwen LoRA weights.",
            "- The tasks are generated and bounded; they are not open-ended natural language reasoning tasks.",
            "- Answer verification uses known task answers during training-target construction.",
            "- Local search is slot-neighborhood search, not full program synthesis.",
            "- Final-answer verification can accept accidental programs that compute the right scalar answer without matching the intended program.",
            "",
            "## Artifacts",
            "",
            "Small files:",
            "",
            "- `experiments/qwen_typed_bytecode_expert_iteration/runs/main_typed_bytecode_ei_s384_u4096/metrics.csv`",
            "- `experiments/qwen_typed_bytecode_expert_iteration/runs/main_typed_bytecode_ei_s384_u4096/train_log.csv`",
            "- `experiments/qwen_typed_bytecode_expert_iteration/runs/main_typed_bytecode_ei_s384_u4096/expert_targets.csv`",
            "- `experiments/qwen_typed_bytecode_expert_iteration/runs/qwen_head_pilot_s384_u2048/metrics.csv`",
            "- `experiments/qwen_typed_bytecode_expert_iteration/runs/qwen_head_pilot_s384_u2048/expert_targets.csv`",
            "- `experiments/qwen_typed_bytecode_expert_iteration/analysis/final_metrics.csv`",
            "- `experiments/qwen_typed_bytecode_expert_iteration/analysis/summary.md`",
            "- `experiments/qwen_typed_bytecode_expert_iteration/reports/qwen_typed_bytecode_expert_iteration_paper.md`",
            "- `experiments/qwen_typed_bytecode_expert_iteration/reports/qwen_typed_bytecode_expert_iteration_paper.html`",
            "",
            "Large files:",
            "",
            "- `large_artifacts/qwen_typed_bytecode_expert_iteration/checkpoints/main_typed_bytecode_ei_s384_u4096/`",
            "- `large_artifacts/qwen_typed_bytecode_expert_iteration/checkpoints/qwen_head_pilot_s384_u2048/`",
            "",
        ]
    )
    report = "\n".join(lines)
    REPORTS.mkdir(parents=True, exist_ok=True)
    md_path = REPORTS / "qwen_typed_bytecode_expert_iteration_paper.md"
    html_path = REPORTS / "qwen_typed_bytecode_expert_iteration_paper.html"
    md_path.write_text(report)
    body = markdown.markdown(report, extensions=["tables", "fenced_code"])
    style = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; max-width: 980px; margin: 40px auto; line-height: 1.55; color: #1f2933; }
table { border-collapse: collapse; width: 100%; margin: 18px 0; font-size: 14px; }
th, td { border: 1px solid #d7dde5; padding: 7px 9px; text-align: left; }
th { background: #f3f5f7; }
img { max-width: 100%; border: 1px solid #e3e7ed; border-radius: 6px; margin: 12px 0 24px; }
code { background: #f3f5f7; padding: 1px 4px; border-radius: 4px; }
"""
    html_doc = f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape('Qwen Typed Bytecode Expert Iteration')}</title><style>{style}</style></head><body>{body}</body></html>"
    html_path.write_text(html_doc)


def main() -> None:
    metrics = read_metrics()
    targets = read_targets()
    logs = read_train_logs()
    save_tables(metrics, targets, logs)
    main = main_metrics(metrics)
    plot_phase_lines(main, "direct_accuracy", "direct_accuracy_by_phase.png", "Direct compiler accuracy by phase")
    plot_phase_lines(main, "search_accuracy", "search_accuracy_by_phase.png", "Answer-verified search accuracy by phase")
    plot_fresh_paired_progress(main)
    plot_target_quality(targets)
    plot_ceiling_bars(main)
    plot_qwen_progress(metrics)
    make_summary(main, targets)
    make_report(metrics, targets)


if __name__ == "__main__":
    main()
