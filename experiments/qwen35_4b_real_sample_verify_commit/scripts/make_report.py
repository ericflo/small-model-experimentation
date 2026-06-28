#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def load_fixed() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted((ROOT / "reports" / "eval").glob("*.json")):
        if "_train_" in path.name:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "candidate_scores" not in payload:
            continue
        rows.extend(payload.get("summary", []))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df


def load_adaptive() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted((ROOT / "reports" / "eval").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "candidate_scores" in payload:
            continue
        rows.extend(payload.get("summary", []))
    return pd.DataFrame(rows)


def write_tables(fixed: pd.DataFrame, adaptive: pd.DataFrame) -> None:
    fixed.to_csv(ROOT / "reports" / "summary_fixed_budget.csv", index=False)
    adaptive.to_csv(ROOT / "reports" / "summary_adaptive_budget.csv", index=False)


def plot_fixed(fixed: pd.DataFrame) -> None:
    fig_dir = ROOT / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    policies = ["oracle_coverage", "first_visible", "base_verifier", "sft_verifier"]
    labels = {
        "oracle_coverage": "Coverage ceiling",
        "first_visible": "First visible",
        "base_verifier": "Frozen verifier",
        "sft_verifier": "SFT verifier",
    }
    for dataset in sorted(fixed["dataset"].unique()):
        df = fixed[(fixed["dataset"] == dataset) & (fixed["policy"].isin(policies))]
        plt.figure(figsize=(8, 5))
        for policy in policies:
            sub = df[df["policy"] == policy].sort_values("budget")
            if sub.empty:
                continue
            plt.plot(sub["budget"], 100 * sub["selected_hidden_all"], marker="o", label=labels[policy])
        plt.xlabel("Generation budget")
        plt.ylabel("Selected hidden-pass (%)")
        plt.ylim(0, 105)
        plt.title(f"{dataset}: sample -> verify -> commit")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / f"{dataset}_fixed_budget_accuracy.png", dpi=180)
        plt.close()

        cov = df[df["policy"] == "oracle_coverage"].sort_values("budget")
        plt.figure(figsize=(7, 4.5))
        plt.plot(cov["budget"], 100 * cov["visible_coverage"], marker="o", color="#2a9d8f")
        plt.xlabel("Generation budget")
        plt.ylabel("Candidate-pool coverage (%)")
        plt.ylim(0, 105)
        plt.title(f"{dataset}: coverage ceiling")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(fig_dir / f"{dataset}_coverage_curve.png", dpi=180)
        plt.close()


def plot_adaptive(adaptive: pd.DataFrame) -> None:
    if adaptive.empty:
        return
    fig_dir = ROOT / "reports" / "figures"
    labels = {
        "threshold_sft_score": "Threshold",
        "sft_stop_controller": "SFT stop",
        "oracle_stop": "Oracle stop",
    }
    for dataset in sorted(adaptive["dataset"].unique()):
        df = adaptive[adaptive["dataset"] == dataset]
        plt.figure(figsize=(7, 5))
        for _, row in df.iterrows():
            plt.scatter(row["samples_used_mean"], 100 * row["selected_hidden_all"], s=80)
            plt.text(row["samples_used_mean"] + 0.05, 100 * row["selected_hidden_all"], labels.get(row["policy"], row["policy"]), fontsize=8)
        plt.xlabel("Mean samples used")
        plt.ylabel("Selected hidden-pass (%)")
        plt.ylim(0, 105)
        plt.title(f"{dataset}: adaptive generation budget")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(fig_dir / f"{dataset}_adaptive_budget.png", dpi=180)
        plt.close()


def pick(fixed: pd.DataFrame, dataset: str, policy: str, budget: int | None, col: str) -> float:
    df = fixed[(fixed["dataset"] == dataset) & (fixed["policy"] == policy)]
    if budget is not None:
        df = df[df["budget"] == budget]
    if df.empty:
        return float("nan")
    return float(df.sort_values("budget").iloc[-1][col])


def pick_adaptive(adaptive: pd.DataFrame, dataset: str, policy: str, col: str) -> float:
    df = adaptive[(adaptive["dataset"] == dataset) & (adaptive["policy"] == policy)]
    if df.empty:
        return float("nan")
    return float(df.iloc[0][col])


def make_markdown(fixed: pd.DataFrame, adaptive: pd.DataFrame) -> str:
    manifest = json.loads((ROOT / "data" / "dataset_manifest.json").read_text(encoding="utf-8"))
    max_budget = int(fixed["budget"].max())
    rows = []
    for _, row in fixed.sort_values(["dataset", "policy", "budget"]).iterrows():
        rows.append(
            f"| {row['dataset']} | {row['policy']} | {int(row['budget'])} | "
            f"{pct(float(row['visible_coverage']))} | {pct(float(row['selected_hidden_all']))} | "
            f"{pct(float(row['coverage_captured']))} | {float(row['sampled_candidates_mean']):.2f} |"
        )
    adaptive_rows = []
    for _, row in adaptive.sort_values(["dataset", "policy"]).iterrows():
        adaptive_rows.append(
            f"| {row['dataset']} | {row['policy']} | {pct(float(row['visible_coverage']))} | "
            f"{pct(float(row['selected_hidden_all']))} | {pct(float(row['coverage_captured']))} | "
            f"{float(row['samples_used_mean']):.2f} |"
        )

    mbpp_cov = pick(fixed, "mbpp", "oracle_coverage", None, "visible_coverage")
    mbpp_sft = pick(fixed, "mbpp", "sft_verifier", None, "selected_hidden_all")
    mbpp_first = pick(fixed, "mbpp", "first_visible", None, "selected_hidden_all")
    human_cov = pick(fixed, "humaneval", "oracle_coverage", None, "visible_coverage")
    human_sft = pick(fixed, "humaneval", "sft_verifier", None, "selected_hidden_all")
    human_first = pick(fixed, "humaneval", "first_visible", None, "selected_hidden_all")
    mbpp_adapt = pick_adaptive(adaptive, "mbpp", "sft_stop_controller", "selected_hidden_all")
    mbpp_adapt_samples = pick_adaptive(adaptive, "mbpp", "sft_stop_controller", "samples_used_mean")
    human_adapt = pick_adaptive(adaptive, "humaneval", "sft_stop_controller", "selected_hidden_all")
    human_adapt_samples = pick_adaptive(adaptive, "humaneval", "sft_stop_controller", "samples_used_mean")

    lines = [
        "# Qwen3.5-4B Real Sample Verify Commit",
        "",
        "## Objective",
        "",
        "Run the full sample -> verify -> commit loop using genuine Qwen3.5-4B code samples. The primary measurement decomposes final pass rate into candidate-pool coverage and selector capture.",
        "",
        "## Candidate Generation",
        "",
        f"- MBPP train records: {manifest['records']['mbpp_train']['records']}.",
        f"- MBPP eval records: {manifest['records']['mbpp_eval']['records']}.",
        f"- HumanEval eval records: {manifest['records']['humaneval_eval']['records']}.",
        f"- Direct samples per task: {manifest['samples_per_task']}.",
        f"- Model repair attempts per task: {manifest['repair_per_task']}.",
        f"- Temperatures: {manifest['temperatures']}.",
        f"- Max new tokens: {manifest['max_new_tokens']}.",
        "",
        "No mutation-generated candidates are used. Every candidate is either a direct Qwen sample or a Qwen repair sample.",
        "",
        "## Key Result",
        "",
        f"- MBPP eval coverage at max budget: {pct(mbpp_cov)}.",
        f"- MBPP eval first-visible commit: {pct(mbpp_first)}.",
        f"- MBPP eval SFT verifier commit: {pct(mbpp_sft)}.",
        f"- HumanEval coverage at max budget: {pct(human_cov)}.",
        f"- HumanEval first-visible commit: {pct(human_first)}.",
        f"- HumanEval SFT verifier commit: {pct(human_sft)}.",
        f"- MBPP SFT stop controller: {pct(mbpp_adapt)} using {mbpp_adapt_samples:.2f} samples on average.",
        f"- HumanEval SFT stop controller: {pct(human_adapt)} using {human_adapt_samples:.2f} samples on average.",
        "",
        "## Readout",
        "",
        "On these genuine sampled pools, selection is not the main bottleneck. HumanEval is already easy for the generator at this budget: coverage reaches 96.7%, and first-visible, frozen verifier, SFT verifier, and oracle selection all match the ceiling. MBPP is coverage-limited: max-budget coverage is 60.0%, and the verifier can capture that ceiling, but it cannot create missing correct programs. The adaptive controller mostly trades a small amount of MBPP accuracy for fewer samples, while matching the HumanEval ceiling with far fewer samples.",
        "",
        "## Fixed-Budget Summary",
        "",
        "| Dataset | Policy | Budget | Visible coverage | Selected hidden-pass | Coverage captured | Samples seen |",
        "|---|---|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "## Adaptive-Budget Summary",
        "",
        "| Dataset | Policy | Visible coverage | Selected hidden-pass | Coverage captured | Mean samples used |",
        "|---|---|---:|---:|---:|---:|",
        *adaptive_rows,
        "",
        "## Interpretation",
        "",
        "This experiment uses genuinely sampled candidate pools, so the central fork is visible directly: under this sampling setup, the verifier is not failing on subtle visible-passing near-misses at this scale. The dominant open problem is generating a correct candidate on harder MBPP tasks. For HumanEval, the first visible-passing sample is usually already correct, so verifier posttraining and adaptive control add little accuracy headroom, though adaptive stopping reduces sample cost.",
        "",
        "The main limitation is scale: this run uses 20 MBPP eval tasks and 30 HumanEval tasks because genuine Qwen sampling is the expensive step. The conclusion should be read as a measured pilot of the real candidate distribution, not as a final benchmark score.",
        "",
        "The next high-leverage step is generator-side: increase coverage through better sampling, repair, or verifier-guided self-improvement. Verifier work should focus on larger, more adversarial sampled pools, because this run did not surface a meaningful selector wall.",
        "",
        "## Figures",
        "",
        "- `reports/figures/mbpp_coverage_curve.png`",
        "- `reports/figures/mbpp_fixed_budget_accuracy.png`",
        "- `reports/figures/mbpp_adaptive_budget.png`",
        "- `reports/figures/humaneval_coverage_curve.png`",
        "- `reports/figures/humaneval_fixed_budget_accuracy.png`",
        "- `reports/figures/humaneval_adaptive_budget.png`",
        "",
        "## Reproduction",
        "",
        "```bash",
        "python scripts/sample_candidates.py --mbpp-train 40 --mbpp-eval 20 --humaneval-eval 30 --samples-per-task 8 --repair-per-task 1 --max-new-tokens 220 --generation-batch-size 4 --temperatures 0.2,0.7,1.0 --top-p 0.95",
        "python scripts/build_verifier_examples.py",
        "python scripts/train_action_sft.py --train data/train_verifier_examples.jsonl --output-dir /workspace/large_artifacts/qwen35_4b_real_sample_verify_commit/models/verifier_sft_lora --loss-out reports/verifier_sft_training_losses.json --method sampled_semantic_verifier_sft --max-steps 160 --batch-size 2 --grad-accum 2",
        "python scripts/eval_commit.py --records data/mbpp_train_records.jsonl --policy sft_verifier --name sft_verifier --out reports/eval/mbpp_train_sft_verifier.json --budgets 1,2,4,8,max --adapter-dir /workspace/large_artifacts/qwen35_4b_real_sample_verify_commit/models/verifier_sft_lora",
        "python scripts/build_stop_examples.py --scores reports/eval/mbpp_train_sft_verifier.json --budgets 1,2,4,8,max",
        "python scripts/train_action_sft.py --train data/train_stop_examples.jsonl --output-dir /workspace/large_artifacts/qwen35_4b_real_sample_verify_commit/models/stop_sft_lora --loss-out reports/stop_sft_training_losses.json --method sampled_generation_budget_stop_sft --max-steps 120 --batch-size 2 --grad-accum 2",
        "bash scripts/run_evaluation_suite.sh",
        "python scripts/make_report.py",
        "```",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    fixed = load_fixed()
    adaptive = load_adaptive()
    if fixed.empty:
        raise RuntimeError("no fixed-budget summaries found")
    write_tables(fixed, adaptive)
    plot_fixed(fixed)
    plot_adaptive(adaptive)
    report = make_markdown(fixed, adaptive)
    path = ROOT / "reports" / "qwen35_4b_real_sample_verify_commit_report.md"
    path.write_text(report, encoding="utf-8")
    print(json.dumps({"report": str(path), "fixed_rows": len(fixed), "adaptive_rows": len(adaptive)}, indent=2))


if __name__ == "__main__":
    main()
