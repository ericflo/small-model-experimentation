#!/usr/bin/env python3
"""Analyze Qwen semantic prefix value model runs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List

import markdown
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("experiments/qwen_semantic_prefix_value_model")
RUNS = ROOT / "runs"
ANALYSIS = ROOT / "analysis"
FIGURES = ANALYSIS / "figures"
REPORTS = ROOT / "reports"
MAIN_RUN = "main_semantic_prefix_value_s512_top1"
PILOT_RUNS = [
    "smoke_semantic_prefix_value",
    "pilot_semantic_prefix_value_s128_cached",
    "pilot_semantic_prefix_value_s128_ranked",
    "pilot_semantic_prefix_value_s128_top1",
]


def pct(x: Any) -> str:
    try:
        val = float(x)
    except Exception:
        return "n/a"
    if math.isnan(val):
        return "n/a"
    return f"{100.0 * val:.1f}%"


def read_csvs(name: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in sorted(RUNS.glob(f"*/{name}")):
        df = pd.read_csv(path)
        df.insert(0, "run_dir", path.parent.name)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False).drop_duplicates()


def read_manifests() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for path in sorted(RUNS.glob("*/dataset_manifest.json")):
        data = json.loads(path.read_text())
        row = {
            "run": data.get("run", path.parent.name),
            "seed": data.get("seed"),
            "max_prompt_tokens": data.get("max_prompt_tokens"),
            "model_id": data.get("metadata", {}).get("model_id"),
            "gpu_name": data.get("metadata", {}).get("gpu_name"),
            "gpu_vram_gb": data.get("metadata", {}).get("gpu_vram_gb"),
        }
        for key, val in data.get("sizes", {}).items():
            row[f"size_{key}"] = val
        rows.append(row)
    return pd.DataFrame(rows)


def metric(metrics: pd.DataFrame, split: str, decoder: str, col: str = "accuracy", run: str = MAIN_RUN) -> float:
    sub = metrics[(metrics["run"].eq(run)) & (metrics["split"].eq(split)) & (metrics["decoder"].eq(decoder))]
    if sub.empty:
        return float("nan")
    return float(sub.iloc[0][col])


def best_decoder(metrics: pd.DataFrame, split: str, prefix: str, run: str = MAIN_RUN) -> pd.Series:
    sub = metrics[(metrics["run"].eq(run)) & (metrics["split"].eq(split)) & (metrics["decoder"].str.startswith(prefix))]
    if sub.empty:
        return pd.Series(dtype="object")
    return sub.sort_values(["accuracy", "program_exact", "oracle_accuracy"], ascending=False).iloc[0]


def save_tables(metrics: pd.DataFrame, train_logs: pd.DataFrame, verifier_logs: pd.DataFrame, samples: pd.DataFrame, manifests: pd.DataFrame) -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(ANALYSIS / "final_metrics.csv", index=False)
    metrics.to_csv(ANALYSIS / "all_final_metrics.csv", index=False)
    if not train_logs.empty:
        train_logs.to_csv(ANALYSIS / "compiler_train_logs.csv", index=False)
    if not verifier_logs.empty:
        verifier_logs.to_csv(ANALYSIS / "verifier_train_logs.csv", index=False)
    if not samples.empty:
        samples.to_csv(ANALYSIS / "prefix_sample_stats.csv", index=False)
    if not manifests.empty:
        manifests.to_csv(ANALYSIS / "dataset_manifests.csv", index=False)


def plot_label_density(samples: pd.DataFrame) -> None:
    if samples.empty:
        return
    FIGURES.mkdir(parents=True, exist_ok=True)
    runs = [r for r in PILOT_RUNS + [MAIN_RUN] if r in set(samples["run_dir"])]
    sub = samples[(samples["run_dir"].isin(runs)) & (samples["split"].eq("train"))].copy()
    if sub.empty:
        return
    labels = {
        "exact_positive_rate": "Exact",
        "raw_semantic_positive_rate": "Raw semantic",
        "semantic_positive_rate": "Filtered semantic",
    }
    x = range(len(sub))
    width = 0.24
    plt.figure(figsize=(11.0, 5.8))
    for j, (col, label) in enumerate(labels.items()):
        vals = [100.0 * float(v) for v in sub[col]]
        plt.bar([i + (j - 1) * width for i in x], vals, width=width, label=label)
    plt.xticks(list(x), [r.replace("pilot_semantic_prefix_value_", "pilot_").replace("main_semantic_prefix_value_", "main_") for r in sub["run_dir"]], rotation=25, ha="right")
    plt.ylabel("Positive labels (%)")
    plt.title("Prefix-Action Label Density")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "label_density.png", dpi=180)
    plt.close()


def plot_verifier_auc(verifier_logs: pd.DataFrame) -> None:
    if verifier_logs.empty:
        return
    sub = verifier_logs[verifier_logs["run"].eq(MAIN_RUN)].copy()
    if sub.empty:
        return
    plt.figure(figsize=(9.5, 5.3))
    for mode, group in sub.groupby("label_mode"):
        group = group.sort_values("epoch")
        plt.plot(group["epoch"], group["val_auc"], marker="o", linewidth=2, label=f"{mode} value")
    plt.xlabel("Epoch")
    plt.ylabel("Held-out AUC")
    plt.title("Main Run Value-Model AUC")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "verifier_auc.png", dpi=180)
    plt.close()


def plot_main_accuracy(metrics: pd.DataFrame) -> None:
    splits = ["fresh_standard", "fresh_paraphrase", "fresh_paired", "hard_composition"]
    split_labels = ["Standard", "Paraphrase", "Paired", "Hard"]
    series = ["greedy", "beam_logprob", "best_exact", "best_semantic", "local_answer"]
    labels = ["Greedy", "Logprob beam", "Best exact value", "Best semantic value", "Answer repair"]
    values: Dict[str, List[float]] = {name: [] for name in series}
    for split in splits:
        values["greedy"].append(metric(metrics, split, "greedy"))
        values["beam_logprob"].append(metric(metrics, split, "beam_logprob"))
        exact = best_decoder(metrics, split, "beam_exact")
        semantic = best_decoder(metrics, split, "beam_semantic")
        values["best_exact"].append(float(exact["accuracy"]) if not exact.empty else float("nan"))
        values["best_semantic"].append(float(semantic["accuracy"]) if not semantic.empty else float("nan"))
        values["local_answer"].append(metric(metrics, split, "local_answer"))
    x = range(len(splits))
    width = 0.15
    plt.figure(figsize=(11.5, 6.0))
    for j, name in enumerate(series):
        plt.bar([i + (j - 2) * width for i in x], [100.0 * v for v in values[name]], width=width, label=labels[j])
    plt.xticks(list(x), split_labels)
    plt.ylabel("Top-1 executable accuracy (%)")
    plt.title("Main Run Accuracy by Decoder")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(FIGURES / "main_accuracy_by_decoder.png", dpi=180)
    plt.close()


def plot_oracle_gap(metrics: pd.DataFrame) -> None:
    splits = ["fresh_paired", "hard_composition"]
    labels = ["Fresh paired", "Hard composition"]
    plt.figure(figsize=(10.0, 5.6))
    for split, label in zip(splits, labels):
        rows = [
            ("greedy", metric(metrics, split, "greedy"), metric(metrics, split, "greedy", "oracle_accuracy")),
            ("logprob", metric(metrics, split, "beam_logprob"), metric(metrics, split, "beam_logprob", "oracle_accuracy")),
        ]
        exact = best_decoder(metrics, split, "beam_exact")
        semantic = best_decoder(metrics, split, "beam_semantic")
        rows.append(("exact", float(exact["accuracy"]), float(exact["oracle_accuracy"])))
        rows.append(("semantic", float(semantic["accuracy"]), float(semantic["oracle_accuracy"])))
        rows.append(("answer repair", metric(metrics, split, "local_answer"), metric(metrics, split, "local_answer", "oracle_accuracy")))
        xs = list(range(len(rows)))
        plt.plot(xs, [100 * r[1] for r in rows], marker="o", linewidth=2, label=f"{label} top-1")
        plt.plot(xs, [100 * r[2] for r in rows], marker="D", linestyle="--", linewidth=2, label=f"{label} oracle")
    plt.xticks(list(range(5)), ["greedy", "logprob", "exact", "semantic", "repair"], rotation=15)
    plt.ylabel("Accuracy (%)")
    plt.title("Top-1 Accuracy vs Candidate Oracle")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "oracle_gap.png", dpi=180)
    plt.close()


def plot_pilot_progress(metrics: pd.DataFrame, samples: pd.DataFrame) -> None:
    runs = [
        "pilot_semantic_prefix_value_s128_cached",
        "pilot_semantic_prefix_value_s128_ranked",
        "pilot_semantic_prefix_value_s128_top1",
        MAIN_RUN,
    ]
    rows: List[Dict[str, Any]] = []
    for run in runs:
        if run not in set(metrics["run"]):
            continue
        sample_row = samples[(samples["run_dir"].eq(run)) & (samples["split"].eq("train"))]
        semantic_rate = float(sample_row["semantic_positive_rate"].iloc[0]) if not sample_row.empty else float("nan")
        exact = best_decoder(metrics, "fresh_paired", "beam_exact", run=run)
        semantic = best_decoder(metrics, "fresh_paired", "beam_semantic", run=run)
        rows.append(
            {
                "run": run,
                "semantic_positive_rate": semantic_rate,
                "fresh_exact": float(exact["accuracy"]) if not exact.empty else float("nan"),
                "fresh_semantic": float(semantic["accuracy"]) if not semantic.empty else float("nan"),
                "fresh_logprob": metric(metrics, "fresh_paired", "beam_logprob", run=run),
            }
        )
    if not rows:
        return
    df = pd.DataFrame(rows)
    x = range(len(df))
    plt.figure(figsize=(10.5, 5.6))
    plt.plot(x, [100 * v for v in df["semantic_positive_rate"]], marker="o", linewidth=2, label="Semantic positive rate")
    plt.plot(x, [100 * v for v in df["fresh_semantic"]], marker="o", linewidth=2, label="Fresh paired semantic beam")
    plt.plot(x, [100 * v for v in df["fresh_exact"]], marker="o", linewidth=2, label="Fresh paired exact beam")
    plt.plot(x, [100 * v for v in df["fresh_logprob"]], marker="o", linewidth=2, label="Fresh paired logprob beam")
    plt.xticks(list(x), [r.replace("pilot_semantic_prefix_value_", "pilot_").replace("main_semantic_prefix_value_", "main_") for r in df["run"]], rotation=25, ha="right")
    plt.ylabel("Rate / accuracy (%)")
    plt.title("Pilot Iteration and Main Run")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "pilot_iteration.png", dpi=180)
    plt.close()


def make_summary(metrics: pd.DataFrame, verifier_logs: pd.DataFrame, samples: pd.DataFrame) -> str:
    fp_exact = best_decoder(metrics, "fresh_paired", "beam_exact")
    fp_sem = best_decoder(metrics, "fresh_paired", "beam_semantic")
    hard_exact = best_decoder(metrics, "hard_composition", "beam_exact")
    hard_sem = best_decoder(metrics, "hard_composition", "beam_semantic")
    main_samples = samples[(samples["run_dir"].eq(MAIN_RUN)) & (samples["split"].eq("train"))]
    main_auc = verifier_logs[verifier_logs["run"].eq(MAIN_RUN)]
    exact_auc = float(main_auc[main_auc["label_mode"].eq("exact")]["val_auc"].max()) if not main_auc.empty else float("nan")
    semantic_auc = float(main_auc[main_auc["label_mode"].eq("semantic")]["val_auc"].max()) if not main_auc.empty else float("nan")
    lines = [
        "# Analysis Summary",
        "",
        f"- Main compiler quick validation bytecode accuracy reached {pct(0.640625)} at the final logged checkpoint.",
        f"- Main train prefix labels: exact positives {pct(main_samples['exact_positive_rate'].iloc[0]) if not main_samples.empty else 'n/a'}, raw semantic positives {pct(main_samples['raw_semantic_positive_rate'].iloc[0]) if not main_samples.empty else 'n/a'}, filtered semantic positives {pct(main_samples['semantic_positive_rate'].iloc[0]) if not main_samples.empty else 'n/a'}.",
        f"- Exact-prefix value AUC reached {exact_auc:.3f}; semantic value AUC reached {semantic_auc:.3f}.",
        f"- Fresh paired: greedy/logprob were {pct(metric(metrics, 'fresh_paired', 'greedy'))}/{pct(metric(metrics, 'fresh_paired', 'beam_logprob'))}; best exact value was {pct(fp_exact['accuracy'])} (`{fp_exact['decoder']}`); best semantic value was {pct(fp_sem['accuracy'])} (`{fp_sem['decoder']}`); answer repair was {pct(metric(metrics, 'fresh_paired', 'local_answer'))}.",
        f"- Hard composition: greedy/logprob were {pct(metric(metrics, 'hard_composition', 'greedy'))}/{pct(metric(metrics, 'hard_composition', 'beam_logprob'))}; best exact value was {pct(hard_exact['accuracy'])} (`{hard_exact['decoder']}`); best semantic value was {pct(hard_sem['accuracy'])} (`{hard_sem['decoder']}`); answer repair was {pct(metric(metrics, 'hard_composition', 'local_answer'))}.",
        "- Conclusion: bounded semantic reachability creates a broader, learnable target, but in this setup it does not beat exact-prefix supervision for top-1 no-answer beam selection. The remaining gap is not candidate containment; it is ranking/calibration of reachable candidates.",
    ]
    return "\n".join(lines) + "\n"


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_csv(index=False)


def figure_md(name: str, caption: str) -> str:
    return f"![{caption}](../analysis/figures/{name})\n\n*{caption}*\n"


def make_report(metrics: pd.DataFrame, verifier_logs: pd.DataFrame, samples: pd.DataFrame, manifests: pd.DataFrame) -> str:
    fp_exact = best_decoder(metrics, "fresh_paired", "beam_exact")
    fp_sem = best_decoder(metrics, "fresh_paired", "beam_semantic")
    hard_exact = best_decoder(metrics, "hard_composition", "beam_exact")
    hard_sem = best_decoder(metrics, "hard_composition", "beam_semantic")
    main_samples = samples[(samples["run_dir"].eq(MAIN_RUN)) & (samples["split"].eq("train"))]
    main_auc = verifier_logs[verifier_logs["run"].eq(MAIN_RUN)]
    exact_auc = float(main_auc[main_auc["label_mode"].eq("exact")]["val_auc"].max()) if not main_auc.empty else float("nan")
    semantic_auc = float(main_auc[main_auc["label_mode"].eq("semantic")]["val_auc"].max()) if not main_auc.empty else float("nan")
    hardware = "unknown GPU"
    if not manifests.empty and "gpu_name" in manifests and not manifests["gpu_name"].dropna().empty:
        hardware = str(manifests[manifests["run"].eq(MAIN_RUN)]["gpu_name"].dropna().iloc[0])

    splits = ["fresh_standard", "fresh_paraphrase", "fresh_paired", "hard_composition"]
    rows: List[Dict[str, str]] = []
    for split in splits:
        exact = best_decoder(metrics, split, "beam_exact")
        semantic = best_decoder(metrics, split, "beam_semantic")
        rows.extend(
            [
                {"split": split, "decoder": "greedy", "accuracy": pct(metric(metrics, split, "greedy")), "oracle": pct(metric(metrics, split, "greedy", "oracle_accuracy")), "program_exact": pct(metric(metrics, split, "greedy", "program_exact"))},
                {"split": split, "decoder": "beam_logprob", "accuracy": pct(metric(metrics, split, "beam_logprob")), "oracle": pct(metric(metrics, split, "beam_logprob", "oracle_accuracy")), "program_exact": pct(metric(metrics, split, "beam_logprob", "program_exact"))},
                {"split": split, "decoder": str(exact["decoder"]), "accuracy": pct(exact["accuracy"]), "oracle": pct(exact["oracle_accuracy"]), "program_exact": pct(exact["program_exact"])},
                {"split": split, "decoder": str(semantic["decoder"]), "accuracy": pct(semantic["accuracy"]), "oracle": pct(semantic["oracle_accuracy"]), "program_exact": pct(semantic["program_exact"])},
                {"split": split, "decoder": "local_answer", "accuracy": pct(metric(metrics, split, "local_answer")), "oracle": pct(metric(metrics, split, "local_answer", "oracle_accuracy")), "program_exact": pct(metric(metrics, split, "local_answer", "program_exact"))},
            ]
        )
    result_table = markdown_table(pd.DataFrame(rows))

    label_rows = samples[(samples["run_dir"].isin(PILOT_RUNS + [MAIN_RUN])) & (samples["split"].eq("train"))][
        [
            "run_dir",
            "prefix_samples",
            "exact_positive_rate",
            "raw_semantic_positive_rate",
            "semantic_positive_rate",
            "semantic_extra_positive_rate",
        ]
    ].copy()
    for col in ["exact_positive_rate", "raw_semantic_positive_rate", "semantic_positive_rate", "semantic_extra_positive_rate"]:
        label_rows[col] = label_rows[col].map(pct)
    label_table = markdown_table(label_rows)

    return f"""# Qwen Semantic Prefix Value Model

## Abstract

This experiment tests whether a value model trained on semantic reachability labels can guide no-answer bytecode search better than an exact-prefix verifier. A frozen Qwen 4B model encodes natural-language prompts. A trained compiler head emits typed stack-machine bytecode distributions. A value model then scores partial bytecode actions during constrained beam search.

The central target is not whether a partial program matches a canonical trace. Instead, the semantic label asks whether bounded executable completion from the post-action VM state can still reach the target answer. Raw reachability was too broad, so the final main run trained on the top-1 reachable action per prefix by compiler prior while preserving canonical exact positives.

The main result is negative but informative. Exact-prefix value supervision reached AUC {exact_auc:.3f} and improved fresh-paired top-1 accuracy from {pct(metric(metrics, "fresh_paired", "greedy"))} greedy / {pct(metric(metrics, "fresh_paired", "beam_logprob"))} logprob beam to {pct(fp_exact["accuracy"])} with `{fp_exact["decoder"]}`. Semantic value supervision reached AUC {semantic_auc:.3f}, but its best fresh-paired beam was {pct(fp_sem["accuracy"])} with `{fp_sem["decoder"]}`. On hard composition, semantic value matched the logprob top-1 result at {pct(hard_sem["accuracy"])}, while answer-verified repair reached {pct(metric(metrics, "hard_composition", "local_answer"))}. The gap to repair remains mostly a ranking/calibration problem rather than a candidate-generation problem.

## Method

The task generator emits mixed natural-language tasks with executable bytecode over a compact stack VM. The opcode set includes arithmetic, comparison, min/max, modulus, and two lookup tables. Programs are normalized to a fixed length and executed invisibly for evaluation.

For each prompt, frozen Qwen hidden states are pooled by a trained compiler head. The head predicts opcode logits, argument logits, and an auxiliary answer head. Typed beam search expands only stack-valid actions. During training-data collection, every candidate action receives:

- `exact`: 1 if the action keeps the prefix equal to the canonical target program.
- `raw semantic`: 1 if bounded suffix search can still complete to the target answer from the post-action state.
- `filtered semantic`: 1 for canonical exact positives plus the top reachable action per prefix by compiler prior.

The value models do not see the final answer at decode time. Answer-verified local repair is included only as an upper-bound diagnostic because it chooses candidates by executing them against the known target answer.

## Runs

The experiment used a smoke run, three 128-example pilots, and one 512-example main run. The smoke run validated the full artifact path. The first pilot showed that raw semantic reachability was too dense. The second and third pilots introduced per-prefix rank filtering. The main run used the stricter top-1 semantic target.

Main run hardware: {hardware}.

## Label Density

{figure_md("label_density.png", "Exact labels are sparse; raw semantic reachability is broad; top-1 filtering reduces but does not eliminate semantic-only positives.")}

{label_table}

## Value Training

{figure_md("verifier_auc.png", "The exact-prefix value model reaches higher AUC than the semantic value model, but the semantic model is trained on a broader and noisier target.")}

In the main run, exact-prefix positives were {pct(main_samples["exact_positive_rate"].iloc[0]) if not main_samples.empty else "n/a"} of train prefix actions. Raw semantic positives were {pct(main_samples["raw_semantic_positive_rate"].iloc[0]) if not main_samples.empty else "n/a"}, and filtered semantic positives were {pct(main_samples["semantic_positive_rate"].iloc[0]) if not main_samples.empty else "n/a"}.

## Decoder Results

{figure_md("main_accuracy_by_decoder.png", "Exact-prefix value improves fresh paired accuracy; semantic value does not outperform exact-prefix value in the main run.")}

{result_table}

## Candidate Oracle Gap

{figure_md("oracle_gap.png", "Beam oracle accuracy remains much higher than top-1 no-answer selection, especially before answer-verified repair.")}

Fresh paired beam-logprob oracle accuracy was {pct(metric(metrics, "fresh_paired", "beam_logprob", "oracle_accuracy"))}, while top-1 logprob accuracy was {pct(metric(metrics, "fresh_paired", "beam_logprob"))}. Hard-composition beam-logprob oracle accuracy was {pct(metric(metrics, "hard_composition", "beam_logprob", "oracle_accuracy"))}, while top-1 logprob accuracy was {pct(metric(metrics, "hard_composition", "beam_logprob"))}. The value models did not reliably convert that oracle slack into top-1 gains.

## Iteration

{figure_md("pilot_iteration.png", "Pilot runs tightened semantic labels from raw reachability toward a top-1 reachable-action target before the main run.")}

The iteration changed the semantic label from raw reachability to rank-filtered reachability because raw reachability made too many actions positive. This improved target sharpness, but did not make semantic value dominate exact-prefix value in the final main run.

## Conclusion

Bounded semantic reachability is a real, learnable signal: it creates many non-canonical positive actions and the semantic value model reaches held-out AUC above 0.85 during training. However, this form of semantic value supervision is not enough to close the no-answer beam-ranking gap. In the main run, exact-prefix value produced the best fresh-paired top-1 result, and semantic value was mostly neutral relative to logprob search.

The next useful step is not another binary reachability classifier. The result points toward a calibrated action-value target: predict the best achievable completion score or success probability under the remaining search budget, not merely whether any bounded completion exists.
"""


def write_html(md_path: Path, html_path: Path, title: str) -> None:
    body = markdown.markdown(md_path.read_text(), extensions=["tables", "fenced_code"])
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; line-height: 1.55; color: #202124; margin: 0; background: #f7f7f5; }}
    main {{ max-width: 980px; margin: 0 auto; padding: 42px 24px 72px; background: #fff; }}
    h1, h2 {{ line-height: 1.15; }}
    h1 {{ font-size: 2.2rem; margin-bottom: 0.4rem; }}
    h2 {{ margin-top: 2.0rem; border-top: 1px solid #ddd; padding-top: 1.2rem; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 0.92rem; }}
    th, td {{ border: 1px solid #d8d8d8; padding: 6px 8px; text-align: left; }}
    th {{ background: #f0f2f4; }}
    img {{ max-width: 100%; display: block; margin: 1rem auto; border: 1px solid #ddd; }}
    code {{ background: #f1f3f4; padding: 0.1rem 0.25rem; border-radius: 4px; }}
  </style>
</head>
<body>
<main>
{body}
</main>
</body>
</html>
"""
    html_path.write_text(html)


def main() -> None:
    metrics = read_csvs("metrics.csv")
    train_logs = read_csvs("train_log.csv")
    verifier_logs = read_csvs("verifier_train_log.csv")
    samples = read_csvs("prefix_samples.csv")
    manifests = read_manifests()
    save_tables(metrics, train_logs, verifier_logs, samples, manifests)
    plot_label_density(samples)
    plot_verifier_auc(verifier_logs)
    plot_main_accuracy(metrics)
    plot_oracle_gap(metrics)
    plot_pilot_progress(metrics, samples)
    summary = make_summary(metrics, verifier_logs, samples)
    (ANALYSIS / "summary.md").write_text(summary)
    report = make_report(metrics, verifier_logs, samples, manifests)
    REPORTS.mkdir(parents=True, exist_ok=True)
    md_path = REPORTS / "qwen_semantic_prefix_value_model_paper.md"
    html_path = REPORTS / "qwen_semantic_prefix_value_model_paper.html"
    md_path.write_text(report)
    write_html(md_path, html_path, "Qwen Semantic Prefix Value Model")
    print(f"wrote {ANALYSIS / 'summary.md'}")
    print(f"wrote {md_path}")
    print(f"wrote {html_path}")


if __name__ == "__main__":
    main()
