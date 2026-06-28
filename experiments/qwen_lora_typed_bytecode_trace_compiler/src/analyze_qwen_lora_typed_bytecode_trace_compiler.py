#!/usr/bin/env python3
"""Aggregate Qwen LoRA typed-bytecode trace compiler runs."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import markdown
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("experiments/qwen_lora_typed_bytecode_trace_compiler")
RUNS = ROOT / "runs"
ANALYSIS = ROOT / "analysis"
FIGURES = ANALYSIS / "figures"
REPORTS = ROOT / "reports"
CHECKPOINT_ROOT = Path("large_artifacts/qwen_lora_typed_bytecode_trace_compiler/checkpoints")

MAIN_RUN = "main_qwen3_4b_qlora_trace_s512"
FROZEN_RUN = "control_qwen3_4b_frozen_trace_s512"
EI_RUN = "main_qwen3_4b_qlora_trace_ei_s256_u1024"
ANSWER_RUN = "control_qwen3_4b_qlora_answer_s512"
PILOT_RUN = "pilot_qwen3_4b_qlora_trace_s128"


def pct(x: Any) -> str:
    try:
        val = float(x)
    except Exception:
        return "n/a"
    if math.isnan(val):
        return "n/a"
    return f"{100.0 * val:.1f}%"


def pp(x: Any) -> str:
    try:
        val = float(x)
    except Exception:
        return "n/a"
    if math.isnan(val):
        return "n/a"
    return f"{100.0 * val:+.1f} pp"


def phase_order(phase: str) -> int:
    if phase in {"trace_supervised", "answer_only", "seed_trace"}:
        return 0
    if phase.startswith("expert_round_"):
        return int(phase.rsplit("_", 1)[-1])
    return 99


def read_metrics() -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in sorted(RUNS.glob("*/metrics.csv")):
        df = pd.read_csv(path)
        frames.append(df)
    if not frames:
        raise SystemExit("no metrics.csv files found")
    return pd.concat(frames, ignore_index=True, sort=False)


def read_logs() -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in sorted(RUNS.glob("*/train_log.csv")):
        df = pd.read_csv(path)
        df.insert(0, "run_dir", path.parent.name)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def read_targets() -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in sorted(RUNS.glob("*/target_log.csv")):
        df = pd.read_csv(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def read_manifests() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for path in sorted(RUNS.glob("*/dataset_manifest.json")):
        data = json.loads(path.read_text())
        rows.append(
            {
                "run": data["run"],
                "variant": data["variant"],
                "max_prompt_tokens": data.get("max_prompt_tokens"),
                "gpu_name": data.get("metadata", {}).get("gpu_name"),
                "gpu_vram_gb": data.get("metadata", {}).get("gpu_vram_gb"),
                "lora_r": data.get("metadata", {}).get("lora_r"),
                **{f"size_{k}": v for k, v in data.get("sizes", {}).items()},
            }
        )
    return pd.DataFrame(rows)


def final_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: List[pd.Series] = []
    for run, sub in metrics.groupby("run"):
        max_order = max(phase_order(str(p)) for p in sub["phase"].unique())
        finals = sub[sub["phase"].map(lambda p: phase_order(str(p)) == max_order)]
        rows.extend([row for _, row in finals.iterrows()])
    return pd.DataFrame(rows)


def metric(metrics: pd.DataFrame, run: str, split: str, col: str, phase: str | None = None) -> float:
    sub = metrics[(metrics["run"].eq(run)) & (metrics["split"].eq(split))]
    if phase is not None:
        sub = sub[sub["phase"].eq(phase)]
    if sub.empty:
        return float("nan")
    if phase is None and len(sub["phase"].unique()) > 1:
        max_order = max(phase_order(str(p)) for p in sub["phase"].unique())
        sub = sub[sub["phase"].map(lambda p: phase_order(str(p)) == max_order)]
    return float(sub.iloc[0][col])


def save_tables(metrics: pd.DataFrame, logs: pd.DataFrame, targets: pd.DataFrame, manifests: pd.DataFrame) -> pd.DataFrame:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(ANALYSIS / "all_final_metrics.csv", index=False)
    finals = final_rows(metrics)
    finals.to_csv(ANALYSIS / "final_metrics.csv", index=False)
    if not logs.empty:
        logs.to_csv(ANALYSIS / "train_logs.csv", index=False)
    if not targets.empty:
        targets.to_csv(ANALYSIS / "expert_target_quality.csv", index=False)
    if not manifests.empty:
        manifests.to_csv(ANALYSIS / "dataset_manifests.csv", index=False)
    return finals


def plot_main_comparison(finals: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    runs = [FROZEN_RUN, MAIN_RUN, EI_RUN, ANSWER_RUN]
    labels = ["Frozen trace", "QLoRA trace", "QLoRA trace EI", "QLoRA answer"]
    splits = ["fresh_standard", "fresh_paraphrase", "fresh_paired", "hard_composition"]
    split_labels = ["Standard", "Paraphrase", "Paired", "Hard"]
    x = range(len(splits))
    width = 0.19
    plt.figure(figsize=(10.5, 5.8))
    for j, (run, label) in enumerate(zip(runs, labels)):
        vals: List[float] = []
        for split in splits:
            row = finals[(finals["run"].eq(run)) & (finals["split"].eq(split))]
            if row.empty:
                vals.append(float("nan"))
            elif run == ANSWER_RUN:
                vals.append(float(row.iloc[0]["answer_head_accuracy"]))
            else:
                vals.append(float(row.iloc[0]["bytecode_accuracy"]))
        plt.bar([i + (j - 1.5) * width for i in x], [100 * v for v in vals], width=width, label=label)
    plt.xticks(list(x), split_labels)
    plt.ylabel("Final accuracy (%)")
    plt.title("Final Direct Accuracy by Condition")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "final_direct_accuracy_by_condition.png", dpi=180)
    plt.close()


def plot_search_gap(finals: pd.DataFrame) -> None:
    splits = ["fresh_paired", "hard_composition"]
    labels = ["Fresh paired", "Hard composition"]
    runs = [FROZEN_RUN, MAIN_RUN, EI_RUN]
    run_labels = ["Frozen trace", "QLoRA trace", "QLoRA trace EI"]
    x = range(len(runs))
    width = 0.28
    plt.figure(figsize=(9.5, 5.5))
    for i, split in enumerate(splits):
        direct = []
        search = []
        for run in runs:
            row = finals[(finals["run"].eq(run)) & (finals["split"].eq(split))].iloc[0]
            direct.append(float(row["bytecode_accuracy"]))
            search.append(float(row["search_accuracy"]))
        offsets = [j + (i - 0.5) * width for j in x]
        plt.bar(offsets, [100 * v for v in direct], width=width, alpha=0.78, label=f"{labels[i]} direct")
        plt.scatter(offsets, [100 * v for v in search], marker="D", s=60, label=f"{labels[i]} search")
    plt.xticks(list(x), run_labels)
    plt.ylabel("Accuracy (%)")
    plt.title("Direct Decoding vs Answer-Verified Local Search")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(FIGURES / "direct_vs_search_gap.png", dpi=180)
    plt.close()


def plot_learning_curves(logs: pd.DataFrame) -> None:
    if logs.empty:
        return
    selected = logs[logs["run"].isin([PILOT_RUN, MAIN_RUN, FROZEN_RUN, EI_RUN, ANSWER_RUN])].copy()
    if selected.empty:
        return
    plt.figure(figsize=(10.5, 5.8))
    labels = {
        PILOT_RUN: "Pilot QLoRA trace",
        MAIN_RUN: "Main QLoRA trace",
        FROZEN_RUN: "Frozen trace",
        EI_RUN: "QLoRA trace EI",
        ANSWER_RUN: "QLoRA answer",
    }
    for run, sub in selected.groupby("run"):
        sub = sub.sort_values("step")
        if run == ANSWER_RUN:
            col = "quick_answer_head_accuracy"
        else:
            col = "quick_bytecode_accuracy"
        if col not in sub:
            continue
        plt.plot(sub["step"], 100 * sub[col], marker="o", linewidth=2, label=labels.get(run, run))
    plt.xlabel("Training step")
    plt.ylabel("Quick validation accuracy (%)")
    plt.title("Training Curves")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "training_curves.png", dpi=180)
    plt.close()


def plot_ei_progress(metrics: pd.DataFrame) -> None:
    sub = metrics[(metrics["run"].eq(EI_RUN)) & (metrics["split"].isin(["fresh_paired", "hard_composition"]))].copy()
    if sub.empty:
        return
    phases = sorted(sub["phase"].unique(), key=phase_order)
    x = range(len(phases))
    labels = [p.replace("seed_trace", "seed").replace("expert_round_", "EI ") for p in phases]
    plt.figure(figsize=(9.5, 5.4))
    for split, split_label in [("fresh_paired", "Fresh paired"), ("hard_composition", "Hard composition")]:
        cur = sub[sub["split"].eq(split)]
        direct = [float(cur[cur["phase"].eq(phase)]["bytecode_accuracy"].iloc[0]) for phase in phases]
        search = [float(cur[cur["phase"].eq(phase)]["search_accuracy"].iloc[0]) for phase in phases]
        plt.plot(x, [100 * v for v in direct], marker="o", linewidth=2, label=f"{split_label} direct")
        plt.plot(x, [100 * v for v in search], marker="D", linestyle="--", linewidth=2, label=f"{split_label} search")
    plt.xticks(list(x), labels)
    plt.ylabel("Accuracy (%)")
    plt.title("Expert-Iteration Progress")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "expert_iteration_progress.png", dpi=180)
    plt.close()


def plot_target_quality(targets: pd.DataFrame) -> None:
    if targets.empty:
        return
    sub = targets[targets["run"].eq(EI_RUN)].copy()
    if sub.empty:
        return
    plt.figure(figsize=(8.5, 5.0))
    plt.plot(sub["round"], 100 * sub["found_rate"], marker="o", linewidth=2, label="Targets found")
    plt.plot(sub["round"], 100 * sub["changed_rate"], marker="o", linewidth=2, label="Targets changed")
    plt.plot(sub["round"], 100 * sub["candidate_valid_rate"], marker="o", linewidth=2, label="Candidate valid")
    plt.xticks(list(sub["round"]))
    plt.xlabel("Expert-iteration round")
    plt.ylabel("Rate (%)")
    plt.title("Answer-Verified Target Quality")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "expert_target_quality.png", dpi=180)
    plt.close()


def make_summary(metrics: pd.DataFrame, finals: pd.DataFrame, targets: pd.DataFrame) -> str:
    q_fp = metric(finals, MAIN_RUN, "fresh_paired", "bytecode_accuracy")
    q_fp_search = metric(finals, MAIN_RUN, "fresh_paired", "search_accuracy")
    q_hard = metric(finals, MAIN_RUN, "hard_composition", "bytecode_accuracy")
    f_fp = metric(finals, FROZEN_RUN, "fresh_paired", "bytecode_accuracy")
    ans_fp = metric(finals, ANSWER_RUN, "fresh_paired", "answer_head_accuracy")
    ei_seed = metric(metrics, EI_RUN, "fresh_paired", "bytecode_accuracy", "seed_trace")
    ei_final = metric(metrics, EI_RUN, "fresh_paired", "bytecode_accuracy", "expert_round_2")
    lines = [
        "# Analysis Summary",
        "",
        f"- Main QLoRA trace run reached {pct(q_fp)} direct executable bytecode accuracy on fresh paired prompts and {pct(q_fp_search)} with answer-verified local search.",
        f"- Hard-composition direct bytecode accuracy for the main run was {pct(q_hard)}.",
        f"- The frozen-Qwen trace-head control reached {pct(f_fp)} fresh paired direct bytecode accuracy, only {pp(q_fp - f_fp)} behind the live QLoRA trace run.",
        f"- The answer-only QLoRA control reached {pct(ans_fp)} fresh paired answer accuracy, far below executable trace supervision.",
        f"- Expert iteration improved fresh paired direct bytecode from {pct(ei_seed)} after seed training to {pct(ei_final)} after two rounds, but did not beat dense 512-trace supervision.",
    ]
    if not targets.empty:
        rows = targets[targets["run"].eq(EI_RUN)]
        if not rows.empty:
            r1 = rows[rows["round"].eq(1)].iloc[0]
            r2 = rows[rows["round"].eq(2)].iloc[0]
            lines.extend(
                [
                    f"- Expert target found rate rose from {pct(r1['found_rate'])} to {pct(r2['found_rate'])}; changed-target rate fell from {pct(r1['changed_rate'])} to {pct(r2['changed_rate'])}.",
                ]
            )
    return "\n".join(lines) + "\n"


def figure_md(name: str, caption: str) -> str:
    return f"![{caption}](../analysis/figures/{name})\n\n*{caption}*\n"


def make_report(metrics: pd.DataFrame, finals: pd.DataFrame, targets: pd.DataFrame, manifests: pd.DataFrame) -> str:
    q_fp = metric(finals, MAIN_RUN, "fresh_paired", "bytecode_accuracy")
    q_fp_search = metric(finals, MAIN_RUN, "fresh_paired", "search_accuracy")
    q_hard = metric(finals, MAIN_RUN, "hard_composition", "bytecode_accuracy")
    q_hard_search = metric(finals, MAIN_RUN, "hard_composition", "search_accuracy")
    frozen_fp = metric(finals, FROZEN_RUN, "fresh_paired", "bytecode_accuracy")
    frozen_hard = metric(finals, FROZEN_RUN, "hard_composition", "bytecode_accuracy")
    answer_fp = metric(finals, ANSWER_RUN, "fresh_paired", "answer_head_accuracy")
    ei_seed_fp = metric(metrics, EI_RUN, "fresh_paired", "bytecode_accuracy", "seed_trace")
    ei_final_fp = metric(metrics, EI_RUN, "fresh_paired", "bytecode_accuracy", "expert_round_2")
    hard_exact = metric(finals, MAIN_RUN, "hard_composition", "program_exact")
    fp_exact = metric(finals, MAIN_RUN, "fresh_paired", "program_exact")

    hardware = "unknown GPU"
    if not manifests.empty and "gpu_name" in manifests:
        vals = [x for x in manifests["gpu_name"].dropna().unique()]
        if vals:
            hardware = str(vals[0])

    target_text = "No expert-iteration targets were recorded."
    if not targets.empty:
        rows = targets[targets["run"].eq(EI_RUN)]
        if not rows.empty:
            r1 = rows[rows["round"].eq(1)].iloc[0]
            r2 = rows[rows["round"].eq(2)].iloc[0]
            target_text = (
                f"Round 1 found targets for {pct(r1['found_rate'])} of unlabeled prompts and changed "
                f"{pct(r1['changed_rate'])} of found targets. Round 2 found {pct(r2['found_rate'])} "
                f"and changed {pct(r2['changed_rate'])}."
            )

    table = finals[finals["run"].isin([FROZEN_RUN, MAIN_RUN, EI_RUN, ANSWER_RUN])].copy()
    table = table[table["split"].isin(["fresh_standard", "fresh_paraphrase", "fresh_paired", "hard_composition"])]
    table["direct_metric"] = table.apply(
        lambda r: r["answer_head_accuracy"] if r["run"] == ANSWER_RUN else r["bytecode_accuracy"], axis=1
    )
    compact = table[["run", "variant", "phase", "split", "direct_metric", "search_accuracy", "program_exact"]].copy()
    compact["direct_metric"] = compact["direct_metric"].map(pct)
    compact["search_accuracy"] = compact["search_accuracy"].map(pct)
    compact["program_exact"] = compact["program_exact"].map(pct)
    md_table = compact.to_markdown(index=False)

    return f"""# Qwen LoRA Typed-Bytecode Trace Compiler

## Abstract

This experiment tests whether a local Qwen 4B model can be posttrained to compile short natural-language reasoning tasks into executable typed bytecode. The runtime is a bounded stack machine with arithmetic, comparisons, min/max, modulo, and two table-lookup host calls. The main condition trains QLoRA adapters and a bytecode compiler head from dense gold execution traces. Controls train the same compiler head on frozen Qwen hidden states and train QLoRA only from final answers.

The main QLoRA trace run reached {pct(q_fp)} direct executable bytecode accuracy on fresh paired prompts and {pct(q_fp_search)} when a small answer-verified local search repaired nearby candidates. On hard-composition prompts it reached {pct(q_hard)} direct and {pct(q_hard_search)} with local search. This is a large gain over answer-only training ({pct(answer_fp)} fresh paired), but only a modest gain over a frozen-Qwen trace-head control ({pct(frozen_fp)} fresh paired, {pct(frozen_hard)} hard).

## Experimental Question

The question is whether a small posttraining modification can make a 4B language model use an internal executable format: the prompt is read by Qwen, a lightweight compiler emits bytecode, and a fixed VM executes that bytecode to produce the answer. The result should separate three mechanisms:

- whether dense executable trace supervision is much stronger than final-answer labels;
- whether live LoRA adaptation improves over a frozen Qwen feature extractor;
- whether answer-verified expert iteration can turn final-answer feedback into better bytecode targets.

## Runtime

The bytecode VM is a typed stack machine over bounded integer values modulo 97. Programs have at most 16 slots and use these opcodes:

`PAD`, `PUSH`, `ADD`, `SUB`, `MUL`, `MOD`, `MAX`, `MIN`, `GT`, `EQ`, `LOOKUP_A`, `LOOKUP_B`, `END`.

The decoder is constrained to produce stack-valid programs. Direct accuracy means the greedy constrained program executed to the correct answer. Search accuracy means a small local candidate set contained a correct executable program and the answer verifier selected it.

## Data

The task generator creates natural-language prompts with executable gold bytecode across six domains:

- modular arithmetic chains;
- weekday offsets;
- unit scaling;
- list sum/max/min;
- boolean threshold checks;
- table lookup.

Evaluation uses fresh standard prompts, fresh paraphrases, paired standard/paraphrase prompts sharing the same latent program, and a hard-composition split with longer arithmetic, longer lists, and larger offsets/factors.

## Conditions

- `frozen_trace`: Qwen is frozen; hidden states are cached; only the bytecode head trains on gold traces.
- `qlora_trace`: Qwen receives QLoRA adapters; the adapters and bytecode head train jointly on gold traces.
- `qlora_trace_ei`: QLoRA starts from 256 gold traces, then answer-verified local search collects bytecode targets from unlabeled prompts for two training rounds.
- `qlora_answer`: QLoRA and a direct answer head train only on final answer labels.

All QLoRA conditions trained 16.5M adapter parameters, about 0.41% of the model, on {hardware}.

## Results

{md_table}

{figure_md("final_direct_accuracy_by_condition.png", "Final direct accuracy. For bytecode conditions this is executable bytecode accuracy; for answer-only it is direct answer-head accuracy.")}

{figure_md("direct_vs_search_gap.png", "Direct bytecode decoding compared with answer-verified local search.")}

{figure_md("training_curves.png", "Quick validation curves during training.")}

{figure_md("expert_iteration_progress.png", "Expert-iteration phase progress on fresh paired and hard-composition splits.")}

{figure_md("expert_target_quality.png", "Answer-verified target quality during expert iteration.")}

## Interpretation

Dense executable trace supervision is the decisive ingredient in this setup. The answer-only control remained low ({pct(answer_fp)} fresh paired), while trace-supervised bytecode reached {pct(q_fp)} fresh paired direct execution and {pct(q_hard)} hard-composition direct execution. The trained system is not merely producing valid syntax: program exactness reached {pct(fp_exact)} on fresh paired prompts and {pct(hard_exact)} on hard-composition prompts.

Live QLoRA helped, but the effect was smaller than the headline trace-supervision effect. On fresh paired prompts, QLoRA trace was {pct(q_fp)} versus {pct(frozen_fp)} for the frozen trace head, a {pp(q_fp - frozen_fp)} difference. On hard composition, QLoRA trace was {pct(q_hard)} versus {pct(frozen_hard)}, a {pp(q_hard - frozen_hard)} difference. The strongest conclusion is therefore not that LoRA alone installed a new executor; it is that Qwen's existing hidden states already support a strong executable compiler head when dense trace supervision is available, and light adaptation gives a modest additional lift.

The local-search gap remains important. The main QLoRA run reached {pct(q_fp_search)} fresh paired search accuracy while greedy direct execution was {pct(q_fp)}. That means many failures are near misses: the correct program often appears after small opcode/argument edits. This points toward process-level decoding or verifier-guided prefix search as a more promising next step than simply training a larger final answer head.

Expert iteration partially worked but did not beat dense trace supervision. Fresh paired direct bytecode improved from {pct(ei_seed_fp)} after seed training to {pct(ei_final_fp)} after two rounds. {target_text} The generated targets were useful, but not as useful as more dense gold traces.

## Limitations

The task distribution is synthetic and still narrow. The VM is deliberately small, and the answer verifier is exact because the generator supplies known answers. Search accuracy should not be read as deployable accuracy unless an external verifier is available. Also, the compiler head is separate from the base LM output head; this is a posttraining-attached executor, not yet a model that emits bytecode through its native token channel.

## Next Experiment

The next high-value experiment should attack the remaining direct/search gap. The best candidate is a prefix-level process verifier for partial bytecode: train a value model over `(prompt, partial program, VM state)` to score whether a prefix can still complete to a correct program, then use beam/A* search over typed bytecode prefixes. That directly targets the observed failure mode: correct programs are often nearby, but greedy slot decoding picks the wrong argument or early opcode.

The second priority is a trace factory with a broader set of crystallized tasks: string normalization, JSON/path extraction, date arithmetic, unit conversion, spreadsheet formulas, regex-like matching, and multi-hop table lookup. This experiment shows executable traces are high-bandwidth supervision; the scaling question is whether a much broader trace corpus preserves the same effect.
"""


def write_html(markdown_text: str) -> None:
    body = markdown.markdown(markdown_text, extensions=["tables", "fenced_code"])
    css = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; line-height: 1.55; color: #20242a; max-width: 980px; margin: 36px auto; padding: 0 24px; }
h1, h2, h3 { line-height: 1.2; color: #111827; }
table { border-collapse: collapse; width: 100%; font-size: 14px; margin: 18px 0; }
th, td { border: 1px solid #d8dee9; padding: 7px 9px; text-align: left; }
th { background: #f3f6fa; }
img { max-width: 100%; border: 1px solid #e5e7eb; border-radius: 6px; margin-top: 8px; }
code { background: #f5f7fb; padding: 2px 4px; border-radius: 4px; }
pre code { display: block; padding: 12px; overflow-x: auto; }
"""
    html_text = f"<!doctype html><html><head><meta charset='utf-8'><title>Qwen LoRA Typed-Bytecode Trace Compiler</title><style>{css}</style></head><body>{body}</body></html>"
    (REPORTS / "qwen_lora_typed_bytecode_trace_compiler_paper.html").write_text(html_text)


def main() -> None:
    metrics = read_metrics()
    logs = read_logs()
    targets = read_targets()
    manifests = read_manifests()
    finals = save_tables(metrics, logs, targets, manifests)

    plot_main_comparison(finals)
    plot_search_gap(finals)
    plot_learning_curves(logs)
    plot_ei_progress(metrics)
    plot_target_quality(targets)

    summary = make_summary(metrics, finals, targets)
    (ANALYSIS / "summary.md").write_text(summary)

    REPORTS.mkdir(parents=True, exist_ok=True)
    report = make_report(metrics, finals, targets, manifests)
    (REPORTS / "qwen_lora_typed_bytecode_trace_compiler_paper.md").write_text(report)
    write_html(report)
    print(f"wrote {ANALYSIS / 'summary.md'}")
    print(f"wrote {REPORTS / 'qwen_lora_typed_bytecode_trace_compiler_paper.md'}")
    print(f"wrote {REPORTS / 'qwen_lora_typed_bytecode_trace_compiler_paper.html'}")


if __name__ == "__main__":
    main()
