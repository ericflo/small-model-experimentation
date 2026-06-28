#!/usr/bin/env python3
"""Analyze Qwen prefix-state process verifier runs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import markdown
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("experiments/qwen_prefix_state_process_verifier")
RUNS = ROOT / "runs"
ANALYSIS = ROOT / "analysis"
FIGURES = ANALYSIS / "figures"
REPORTS = ROOT / "reports"
CHECKPOINT_ROOT = Path("large_artifacts/qwen_prefix_state_process_verifier/checkpoints")
MAIN_RUN = "main_prefix_state_verifier_s512"
PILOT_RUN = "pilot_prefix_state_verifier_s128"


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


def read_csvs(name: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in sorted(RUNS.glob(f"*/{name}")):
        df = pd.read_csv(path)
        df.insert(0, "run_dir", path.parent.name)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    return out.drop_duplicates()


def read_manifests() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for path in sorted(RUNS.glob("*/dataset_manifest.json")):
        data = json.loads(path.read_text())
        row = {
            "run": data["run"],
            "seed": data["seed"],
            "max_prompt_tokens": data.get("max_prompt_tokens"),
            "gpu_name": data.get("metadata", {}).get("gpu_name"),
            "gpu_vram_gb": data.get("metadata", {}).get("gpu_vram_gb"),
        }
        for key, val in data.get("sizes", {}).items():
            row[f"size_{key}"] = val
        rows.append(row)
    return pd.DataFrame(rows)


def metric(df: pd.DataFrame, split: str, decoder: str, col: str = "accuracy", run: str = MAIN_RUN) -> float:
    sub = df[(df["run"].eq(run)) & (df["split"].eq(split)) & (df["decoder"].eq(decoder))]
    if sub.empty:
        return float("nan")
    return float(sub.iloc[0][col])


def best_verifier_rows(df: pd.DataFrame, run: str = MAIN_RUN) -> pd.DataFrame:
    rows = []
    for split, sub in df[(df["run"].eq(run)) & (df["decoder"].str.startswith("beam_verifier"))].groupby("split"):
        rows.append(sub.sort_values(["accuracy", "program_exact", "oracle_accuracy"], ascending=False).iloc[0])
    return pd.DataFrame(rows)


def save_tables(metrics: pd.DataFrame, logs: pd.DataFrame, verifier_logs: pd.DataFrame, prefix_samples: pd.DataFrame, manifests: pd.DataFrame) -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(ANALYSIS / "final_metrics.csv", index=False)
    if not logs.empty:
        logs.to_csv(ANALYSIS / "compiler_train_logs.csv", index=False)
    if not verifier_logs.empty:
        verifier_logs.to_csv(ANALYSIS / "verifier_train_logs.csv", index=False)
    if not prefix_samples.empty:
        prefix_samples.to_csv(ANALYSIS / "prefix_sample_stats.csv", index=False)
    if not manifests.empty:
        manifests.to_csv(ANALYSIS / "dataset_manifests.csv", index=False)


def plot_main_accuracy(metrics: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    splits = ["fresh_standard", "fresh_paraphrase", "fresh_paired", "hard_composition"]
    split_labels = ["Standard", "Paraphrase", "Paired", "Hard"]
    decoders = ["greedy", "local_answer", "beam_logprob", "best_verifier"]
    labels = ["Greedy", "Answer repair", "Logprob beam", "Best verifier beam"]
    best = best_verifier_rows(metrics)
    values: Dict[str, List[float]] = {d: [] for d in decoders}
    for split in splits:
        values["greedy"].append(metric(metrics, split, "greedy"))
        values["local_answer"].append(metric(metrics, split, "local_answer"))
        values["beam_logprob"].append(metric(metrics, split, "beam_logprob"))
        row = best[best["split"].eq(split)]
        values["best_verifier"].append(float(row.iloc[0]["accuracy"]) if not row.empty else float("nan"))
    x = range(len(splits))
    width = 0.2
    plt.figure(figsize=(10.5, 5.8))
    for j, dec in enumerate(decoders):
        plt.bar([i + (j - 1.5) * width for i in x], [100 * v for v in values[dec]], width=width, label=labels[j])
    plt.xticks(list(x), split_labels)
    plt.ylabel("Top-1 executable accuracy (%)")
    plt.title("Main Run Accuracy by Decoder")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "main_accuracy_by_decoder.png", dpi=180)
    plt.close()


def plot_oracle_gap(metrics: pd.DataFrame) -> None:
    splits = ["fresh_paired", "hard_composition"]
    labels = ["Fresh paired", "Hard composition"]
    plt.figure(figsize=(9.5, 5.5))
    for split, label in zip(splits, labels):
        sub = metrics[(metrics["run"].eq(MAIN_RUN)) & (metrics["split"].eq(split))]
        dec = ["greedy", "beam_logprob", "beam_verifier_w2", "beam_verifier_w4", "beam_verifier_w8", "beam_verifier_w16"]
        xs = list(range(len(dec)))
        acc = [float(sub[sub["decoder"].eq(d)]["accuracy"].iloc[0]) for d in dec]
        oracle = [float(sub[sub["decoder"].eq(d)]["oracle_accuracy"].iloc[0]) for d in dec]
        plt.plot(xs, [100 * v for v in acc], marker="o", linewidth=2, label=f"{label} top-1")
        plt.plot(xs, [100 * v for v in oracle], marker="D", linestyle="--", linewidth=2, label=f"{label} oracle")
    plt.xticks(list(range(6)), ["greedy", "logprob", "w2", "w4", "w8", "w16"], rotation=15)
    plt.ylabel("Accuracy (%)")
    plt.title("Beam Top-1 vs Beam Oracle")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "beam_oracle_gap.png", dpi=180)
    plt.close()


def plot_verifier_training(verifier_logs: pd.DataFrame) -> None:
    if verifier_logs.empty:
        return
    plt.figure(figsize=(9.5, 5.4))
    for run, sub in verifier_logs.groupby("run"):
        if run not in {PILOT_RUN, MAIN_RUN}:
            continue
        sub = sub.sort_values("epoch")
        plt.plot(sub["epoch"], sub["val_auc"], marker="o", linewidth=2, label=f"{run} AUC")
    plt.xlabel("Verifier epoch")
    plt.ylabel("Held-out prefix AUC")
    plt.title("Verifier Training Curves")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "verifier_training_auc.png", dpi=180)
    plt.close()


def plot_prefix_samples(prefix_samples: pd.DataFrame) -> None:
    if prefix_samples.empty:
        return
    sub = prefix_samples[prefix_samples["run_dir"].isin([PILOT_RUN, MAIN_RUN]) & prefix_samples["split"].eq("train")].copy()
    if sub.empty:
        return
    plt.figure(figsize=(8.5, 5.0))
    plt.bar(sub["run_dir"], sub["prefix_samples"], label="Total prefix samples")
    plt.bar(sub["run_dir"], sub["positive_samples"], label="Positive samples")
    plt.ylabel("Samples")
    plt.title("Prefix Training Sample Counts")
    plt.xticks(rotation=15, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "prefix_sample_counts.png", dpi=180)
    plt.close()


def make_summary(metrics: pd.DataFrame, verifier_logs: pd.DataFrame, prefix_samples: pd.DataFrame) -> str:
    best = best_verifier_rows(metrics)
    fp_best = float(best[best["split"].eq("fresh_paired")]["accuracy"].iloc[0])
    fp_best_decoder = str(best[best["split"].eq("fresh_paired")]["decoder"].iloc[0])
    hard_best = float(best[best["split"].eq("hard_composition")]["accuracy"].iloc[0])
    hard_best_decoder = str(best[best["split"].eq("hard_composition")]["decoder"].iloc[0])
    fp_greedy = metric(metrics, "fresh_paired", "greedy")
    fp_local = metric(metrics, "fresh_paired", "local_answer")
    fp_oracle = metric(metrics, "fresh_paired", "beam_logprob", "oracle_accuracy")
    hard_greedy = metric(metrics, "hard_composition", "greedy")
    hard_local = metric(metrics, "hard_composition", "local_answer")
    hard_oracle = metric(metrics, "hard_composition", "beam_logprob", "oracle_accuracy")
    main_v = verifier_logs[verifier_logs["run"].eq(MAIN_RUN)].drop_duplicates()
    best_auc = float(main_v["val_auc"].max()) if not main_v.empty else float("nan")
    return "\n".join(
        [
            "# Analysis Summary",
            "",
            f"- Main verifier held-out prefix AUC reached {best_auc:.3f}.",
            f"- Fresh paired greedy accuracy was {pct(fp_greedy)}; best verifier beam was {pct(fp_best)} (`{fp_best_decoder}`); local answer repair was {pct(fp_local)}.",
            f"- Fresh paired logprob beam oracle was {pct(fp_oracle)}, so correct programs were often in the beam even when top-1 selection did not improve much.",
            f"- Hard-composition greedy accuracy was {pct(hard_greedy)}; best verifier beam was {pct(hard_best)} (`{hard_best_decoder}`); local answer repair was {pct(hard_local)}.",
            f"- Hard-composition logprob beam oracle was {pct(hard_oracle)}, exposing a large remaining reranking gap.",
        ]
    ) + "\n"


def figure_md(name: str, caption: str) -> str:
    return f"![{caption}](../analysis/figures/{name})\n\n*{caption}*\n"


def make_report(metrics: pd.DataFrame, verifier_logs: pd.DataFrame, prefix_samples: pd.DataFrame, manifests: pd.DataFrame) -> str:
    best = best_verifier_rows(metrics)
    fp_best = float(best[best["split"].eq("fresh_paired")]["accuracy"].iloc[0])
    fp_best_decoder = str(best[best["split"].eq("fresh_paired")]["decoder"].iloc[0])
    hard_best = float(best[best["split"].eq("hard_composition")]["accuracy"].iloc[0])
    hard_best_decoder = str(best[best["split"].eq("hard_composition")]["decoder"].iloc[0])
    fp_greedy = metric(metrics, "fresh_paired", "greedy")
    fp_local = metric(metrics, "fresh_paired", "local_answer")
    fp_logprob = metric(metrics, "fresh_paired", "beam_logprob")
    fp_oracle = metric(metrics, "fresh_paired", "beam_logprob", "oracle_accuracy")
    hard_greedy = metric(metrics, "hard_composition", "greedy")
    hard_local = metric(metrics, "hard_composition", "local_answer")
    hard_logprob = metric(metrics, "hard_composition", "beam_logprob")
    hard_oracle = metric(metrics, "hard_composition", "beam_logprob", "oracle_accuracy")
    main_v = verifier_logs[verifier_logs["run"].eq(MAIN_RUN)].drop_duplicates()
    best_auc = float(main_v["val_auc"].max()) if not main_v.empty else float("nan")
    main_prefix = prefix_samples[(prefix_samples["run_dir"].eq(MAIN_RUN)) & (prefix_samples["split"].eq("train"))]
    prefix_count = int(main_prefix["prefix_samples"].iloc[0]) if not main_prefix.empty else 0
    positive_rate = float(main_prefix["positive_rate"].iloc[0]) if not main_prefix.empty else float("nan")
    hardware = "unknown GPU"
    if not manifests.empty and "gpu_name" in manifests and not manifests["gpu_name"].dropna().empty:
        hardware = str(manifests["gpu_name"].dropna().iloc[0])

    table = metrics[(metrics["run"].eq(MAIN_RUN)) & (metrics["split"].isin(["fresh_standard", "fresh_paraphrase", "fresh_paired", "hard_composition"]))].copy()
    table = table[["split", "decoder", "accuracy", "program_exact", "oracle_accuracy", "mean_expansions"]]
    table["accuracy"] = table["accuracy"].map(pct)
    table["program_exact"] = table["program_exact"].map(pct)
    table["oracle_accuracy"] = table["oracle_accuracy"].map(pct)
    table["mean_expansions"] = table["mean_expansions"].map(lambda x: f"{float(x):.1f}")
    md_table = table.to_markdown(index=False)

    return f"""# Qwen Prefix-State Process Verifier

## Abstract

This experiment tests a learned prefix-state verifier for executable bytecode search. A frozen Qwen 4B model encodes the prompt, a trained compiler head emits opcode and argument distributions, and a prefix verifier scores partial bytecode prefixes together with the current VM stack. The verifier is then used to guide typed beam search without using the final answer at decode time.

The verifier learned the prefix classification task well, reaching held-out prefix AUC {best_auc:.3f} on the main run. It produced modest top-1 decoding gains, especially on hard-composition prompts: hard accuracy improved from {pct(hard_greedy)} greedy to {pct(hard_best)} with `{hard_best_decoder}`. Fresh paired accuracy improved from {pct(fp_greedy)} greedy to {pct(fp_best)} with `{fp_best_decoder}`. The much larger gap remained between no-answer top-1 decoding and answer-verified repair: fresh paired local answer repair reached {pct(fp_local)}, and hard-composition local answer repair reached {pct(hard_local)}.

## Experimental Question

The question is whether a process verifier over partial programs can convert a high-oracle beam into better deployable top-1 bytecode. The verifier sees a prompt representation, the current bytecode prefix, the VM stack after the proposed next action, and the proposed action. It predicts whether the prefix remains consistent with a known correct executable trace.

## Runtime And Decoder

The runtime is a bounded typed stack machine over integers modulo 97. Programs have at most 16 slots and use arithmetic, comparison, min/max, modulo, and lookup opcodes. Search expands only type-valid actions, so invalid stack programs are pruned before scoring.

The evaluated decoders are:

- `greedy`: constrained greedy compiler decoding;
- `local_answer`: complete-program local repair selected by final-answer verification;
- `beam_logprob`: typed beam search using compiler log probability only;
- `beam_verifier_w*`: typed beam search using compiler log probability plus a weighted sum of prefix-verifier log-scores.

## Data And Training

The task generator creates prompts across modular arithmetic chains, weekday offsets, unit scaling, list aggregation, boolean threshold checks, and table lookup. The main run used 512 compiler-trace prompts, 512 verifier prompts, 128 examples per evaluation split, and ran on {hardware}.

The verifier training set contained {prefix_count:,} prefix samples with {pct(positive_rate)} positives. Positives are gold-consistent executable prefixes; negatives are off-path prefixes generated by the compiler's own typed beam distribution.

## Results

{md_table}

{figure_md("main_accuracy_by_decoder.png", "Main run top-1 executable accuracy by decoder.")}

{figure_md("beam_oracle_gap.png", "Top-1 beam accuracy compared with beam oracle accuracy.")}

{figure_md("verifier_training_auc.png", "Held-out prefix verifier AUC during training.")}

{figure_md("prefix_sample_counts.png", "Prefix sample counts for pilot and main runs.")}

## Interpretation

The prefix verifier does learn a meaningful process signal: AUC above 0.93 is not a weak classifier. It also raises some top-1 no-answer accuracy: hard-composition accuracy improved by {pp(hard_best - hard_greedy)}, and fresh paired improved by {pp(fp_best - fp_greedy)}. However, these gains are much smaller than the available beam/search headroom.

The central result is the distinction between containment and selection. On fresh paired prompts, logprob beam top-1 was {pct(fp_logprob)}, but the same beam contained a correct executable program {pct(fp_oracle)} of the time. On hard composition, logprob beam top-1 was {pct(hard_logprob)}, while the beam oracle was {pct(hard_oracle)}. The correct programs are frequently in the beam; this verifier formulation does not yet rank them aggressively enough.

The answer-verified local repair remains a strong upper comparison, reaching {pct(fp_local)} fresh paired and {pct(hard_local)} hard. That condition uses final-answer feedback at decode time, so it is not the deployable no-answer setting, but it shows that the compiler's nearby candidate space is much better than greedy decoding.

## Failure Analysis

This verifier is trained mostly as a gold-prefix classifier. That is a useful process signal, but it is not the same as semantic reachability. A prefix can deviate from the canonical trace and still complete to a correct program, while a gold-looking prefix can still lose because of later argument choices. The scorer also accumulates prefix log-sigmoid penalties, which can over-penalize longer correct programs and does not directly optimize final top-1 answer accuracy.

## Next Step

The next version should train a semantic value model rather than an exact-prefix classifier. Labels should come from suffix completion search: for a partial prefix, ask whether any bounded continuation can still execute to the correct answer. That would turn the verifier from "does this match the teacher trace?" into "is this prefix still live?" The search policy should then optimize expected completion success, not prefix exactness.

A second improvement is to distill successful verifier-beam or answer-verified beam programs back into the compiler head, so the direct compiler learns from the high-oracle beam instead of relying on expensive search at inference time.
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
"""
    html_text = f"<!doctype html><html><head><meta charset='utf-8'><title>Qwen Prefix-State Process Verifier</title><style>{css}</style></head><body>{body}</body></html>"
    (REPORTS / "qwen_prefix_state_process_verifier_paper.html").write_text(html_text)


def main() -> None:
    metrics = read_csvs("metrics.csv")
    logs = read_csvs("train_log.csv")
    verifier_logs = read_csvs("verifier_train_log.csv")
    prefix_samples = read_csvs("prefix_samples.csv")
    manifests = read_manifests()

    save_tables(metrics, logs, verifier_logs, prefix_samples, manifests)
    plot_main_accuracy(metrics)
    plot_oracle_gap(metrics)
    plot_verifier_training(verifier_logs)
    plot_prefix_samples(prefix_samples)

    summary = make_summary(metrics, verifier_logs, prefix_samples)
    (ANALYSIS / "summary.md").write_text(summary)

    REPORTS.mkdir(parents=True, exist_ok=True)
    report = make_report(metrics, verifier_logs, prefix_samples, manifests)
    (REPORTS / "qwen_prefix_state_process_verifier_paper.md").write_text(report)
    write_html(report)
    print(f"wrote {ANALYSIS / 'summary.md'}")
    print(f"wrote {REPORTS / 'qwen_prefix_state_process_verifier_paper.md'}")
    print(f"wrote {REPORTS / 'qwen_prefix_state_process_verifier_paper.html'}")


if __name__ == "__main__":
    main()
