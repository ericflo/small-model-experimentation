#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

POLICY_ORDER = [
    "first_visible",
    "shortest_visible",
    "random_visible",
    "oracle_coverage",
    "base_verifier",
    "sft_verifier",
]

POLICY_LABELS = {
    "first_visible": "First visible-pass",
    "shortest_visible": "Shortest visible-pass",
    "random_visible": "Random visible-pass",
    "oracle_coverage": "Oracle coverage",
    "base_verifier": "Base Qwen verifier",
    "sft_verifier": "SFT Qwen verifier",
}


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def load_eval() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []
    for path in sorted((ROOT / "reports" / "eval").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(payload.get("records", []))
        scores.extend(payload.get("candidate_scores", []))
    return pd.DataFrame(rows), pd.DataFrame(scores)


def ordered(df: pd.DataFrame) -> pd.DataFrame:
    order = {policy: i for i, policy in enumerate(POLICY_ORDER)}
    return df.assign(order=df["policy"].map(order).fillna(999)).sort_values(["dataset", "order", "policy"])


def write_tables(records: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    summary = (
        records.groupby(["dataset", "policy"], as_index=False)
        .agg(
            records=("record_id", "count"),
            candidate_pool_coverage=("candidate_pool_coverage", "mean"),
            selected_hidden_all=("selected_hidden_all", "mean"),
            visible_candidates_mean=("visible_candidate_count", "mean"),
            hidden_pass_visible_candidates_mean=("hidden_pass_visible_candidates", "mean"),
        )
        .sort_values(["dataset", "policy"])
    )
    summary["coverage_captured"] = summary.apply(
        lambda row: row["selected_hidden_all"] / row["candidate_pool_coverage"] if row["candidate_pool_coverage"] else 0.0,
        axis=1,
    )
    by_task = records.sort_values(["dataset", "policy", "task_id"])
    summary.to_csv(ROOT / "reports" / "summary_overall.csv", index=False)
    by_task.to_csv(ROOT / "reports" / "selection_by_task.csv", index=False)
    if not scores.empty:
        scores.to_csv(ROOT / "reports" / "candidate_scores.csv", index=False)
    return summary


def plot_summary(summary: pd.DataFrame) -> None:
    fig_dir = ROOT / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = ordered(summary)
    for metric, title, filename in [
        ("selected_hidden_all", "Selected Hidden-Pass Rate", "selected_hidden_pass.png"),
        ("coverage_captured", "Coverage Captured", "coverage_captured.png"),
    ]:
        pivot = 100 * df.pivot(index="policy", columns="dataset", values=metric)
        order = [policy for policy in POLICY_ORDER if policy in pivot.index]
        pivot = pivot.loc[order]
        ax = pivot.rename(index=POLICY_LABELS).plot(kind="bar", figsize=(9, 5.5), color=["#2a9d8f", "#457b9d"])
        ax.set_ylabel(f"{title} (%)")
        ax.set_ylim(0, 105)
        ax.set_title(title)
        ax.legend(title="")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(fig_dir / filename, dpi=180)
        plt.close()

    plt.figure(figsize=(8, 5.5))
    for _, row in df.iterrows():
        plt.scatter(row["candidate_pool_coverage"] * 100, row["selected_hidden_all"] * 100, s=70)
        plt.text(
            row["candidate_pool_coverage"] * 100 + 0.4,
            row["selected_hidden_all"] * 100,
            f"{row['dataset']} / {POLICY_LABELS.get(row['policy'], row['policy'])}",
            fontsize=7,
        )
    plt.xlabel("Candidate-pool coverage (%)")
    plt.ylabel("Selected hidden-pass rate (%)")
    plt.title("Selection vs Reachability")
    plt.xlim(0, 105)
    plt.ylim(0, 105)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(fig_dir / "selection_vs_coverage.png", dpi=180)
    plt.close()


def plot_training_curve() -> None:
    path = ROOT / "reports" / "verifier_sft_training_losses.json"
    if not path.exists():
        return
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not rows:
        return
    df = pd.DataFrame(rows)
    plt.figure(figsize=(7, 4))
    plt.plot(df["step"], df["loss"], color="#5c4d7d")
    plt.xlabel("Optimizer step")
    plt.ylabel("Cross-entropy loss")
    plt.title("Semantic Verifier SFT Loss")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(ROOT / "reports" / "figures" / "verifier_sft_loss.png", dpi=180)
    plt.close()


def value(summary: pd.DataFrame, dataset: str, policy: str, column: str) -> float:
    row = summary[(summary["dataset"] == dataset) & (summary["policy"] == policy)]
    if row.empty:
        return float("nan")
    return float(row.iloc[0][column])


def make_markdown(summary: pd.DataFrame) -> str:
    manifest = json.loads((ROOT / "data" / "dataset_manifest.json").read_text(encoding="utf-8"))
    rows: list[str] = []
    for _, row in ordered(summary).iterrows():
        rows.append(
            f"| {row['dataset']} | {POLICY_LABELS.get(row['policy'], row['policy'])} | "
            f"{pct(float(row['candidate_pool_coverage']))} | {pct(float(row['selected_hidden_all']))} | "
            f"{pct(float(row['coverage_captured']))} | {float(row['visible_candidates_mean']):.2f} | "
            f"{float(row['hidden_pass_visible_candidates_mean']):.2f} |"
        )
    human_sft = value(summary, "humaneval", "sft_verifier", "selected_hidden_all")
    human_base = value(summary, "humaneval", "base_verifier", "selected_hidden_all")
    human_first = value(summary, "humaneval", "first_visible", "selected_hidden_all")
    human_cov = value(summary, "humaneval", "oracle_coverage", "selected_hidden_all")
    mbpp_sft = value(summary, "mbpp", "sft_verifier", "selected_hidden_all")
    mbpp_base = value(summary, "mbpp", "base_verifier", "selected_hidden_all")
    mbpp_first = value(summary, "mbpp", "first_visible", "selected_hidden_all")

    lines = [
        "# Qwen3.5-4B Oracle-Distilled Semantic Verifier",
        "",
        "## Objective",
        "",
        "Train Qwen3.5-4B as a deployable verifier for Python candidate programs. The training oracle labels visible-test-passing candidates by hidden-test execution. At inference, the model sees only the task, public tests, public-test status, and candidate code; it ranks candidates by the probability that they pass hidden tests.",
        "",
        "## Data",
        "",
        f"- Train records: {manifest['records']['mbpp_train']['records']} MBPP tasks.",
        f"- Validation records: {manifest['records']['mbpp_valid']['records']} MBPP tasks.",
        f"- Transfer eval records: {manifest['records']['humaneval_eval']['records']} HumanEval tasks.",
        f"- Visible tests per task: {manifest['visible_tests']}.",
        f"- Candidate implementations per task: up to {manifest['candidate_count']}.",
        f"- Train verifier examples: {manifest['examples']['train']}.",
        "",
        "## Key Result",
        "",
        f"- HumanEval first-visible baseline: {pct(human_first)}.",
        f"- HumanEval base Qwen verifier: {pct(human_base)}.",
        f"- HumanEval candidate-pool coverage: {pct(human_cov)}.",
        f"- HumanEval SFT verifier: {pct(human_sft)}.",
        f"- MBPP validation first-visible baseline: {pct(mbpp_first)}.",
        f"- MBPP validation base Qwen verifier: {pct(mbpp_base)}.",
        f"- MBPP validation SFT verifier: {pct(mbpp_sft)}.",
        "",
        "## Readout",
        "",
        f"The SFT verifier improves MBPP validation selection by {100 * (mbpp_sft - mbpp_base):+.1f} points over the frozen base verifier and {100 * (mbpp_sft - mbpp_first):+.1f} points over first visible-pass selection. On HumanEval it improves {100 * (human_sft - human_first):+.1f} points over first visible-pass selection, but trails the frozen base verifier by {100 * (human_base - human_sft):.1f} points. This is a useful but not complete transfer result: oracle-distilled posttraining teaches an in-domain candidate verifier, while the frozen model is already a very strong out-of-domain verifier under this candidate-generation setup.",
        "",
        "## Overall",
        "",
        "| Dataset | Policy | Candidate-pool coverage | Selected hidden-pass | Coverage captured | Visible candidates | Hidden-pass candidates |",
        "|---|---|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "## Interpretation",
        "",
        "The primary question is whether oracle-labeled posttraining teaches a semantic candidate verifier that captures candidate-pool coverage under leak-free public evidence. The answer here is mixed. The adapter clearly improves in-domain MBPP selection, which means the hidden-test labels provide a learnable signal. The HumanEval result is weaker: SFT remains above simple selection baselines, but the frozen Qwen verifier is better on this eval set. The most likely reading is that this small MBPP-only SFT run partly overfits the mutation and task distribution instead of improving the model's general verifier prior. The next iteration should either train on a larger and more varied code corpus, or distill preferences from a stronger oracle while preserving the frozen model's general-code prior.",
        "",
        "## Figures",
        "",
        "- `reports/figures/selected_hidden_pass.png`",
        "- `reports/figures/coverage_captured.png`",
        "- `reports/figures/selection_vs_coverage.png`",
        "- `reports/figures/verifier_sft_loss.png`",
        "",
        "## Reproduction",
        "",
        "```bash",
        "python scripts/build_dataset.py --mbpp-train 90 --mbpp-valid 32 --humaneval-eval 31 --visible-tests 1 --candidate-count 18",
        "python scripts/eval_verifier.py --records data/mbpp_valid_records.jsonl --policy first --name first_visible --out reports/eval/mbpp_first_visible.json",
        "python scripts/eval_verifier.py --records data/mbpp_valid_records.jsonl --policy shortest --name shortest_visible --out reports/eval/mbpp_shortest_visible.json",
        "python scripts/eval_verifier.py --records data/mbpp_valid_records.jsonl --policy random --name random_visible --out reports/eval/mbpp_random_visible.json",
        "python scripts/eval_verifier.py --records data/mbpp_valid_records.jsonl --policy oracle --name oracle_coverage --out reports/eval/mbpp_oracle_coverage.json",
        "python scripts/eval_verifier.py --records data/humaneval_eval_records.jsonl --policy first --name first_visible --out reports/eval/humaneval_first_visible.json",
        "python scripts/eval_verifier.py --records data/humaneval_eval_records.jsonl --policy shortest --name shortest_visible --out reports/eval/humaneval_shortest_visible.json",
        "python scripts/eval_verifier.py --records data/humaneval_eval_records.jsonl --policy random --name random_visible --out reports/eval/humaneval_random_visible.json",
        "python scripts/eval_verifier.py --records data/humaneval_eval_records.jsonl --policy oracle --name oracle_coverage --out reports/eval/humaneval_oracle_coverage.json",
        "python scripts/eval_verifier.py --records data/mbpp_valid_records.jsonl --policy base --name base_verifier --out reports/eval/mbpp_base_verifier.json",
        "python scripts/eval_verifier.py --records data/humaneval_eval_records.jsonl --policy base --name base_verifier --out reports/eval/humaneval_base_verifier.json",
        "python scripts/train_verifier_sft.py --max-steps 220 --batch-size 2 --grad-accum 2",
        "python scripts/eval_verifier.py --records data/mbpp_valid_records.jsonl --policy adapter --name sft_verifier --out reports/eval/mbpp_sft_verifier.json --adapter-dir /workspace/large_artifacts/qwen35_4b_oracle_distilled_semantic_verifier/models/verifier_sft_lora",
        "python scripts/eval_verifier.py --records data/humaneval_eval_records.jsonl --policy adapter --name sft_verifier --out reports/eval/humaneval_sft_verifier.json --adapter-dir /workspace/large_artifacts/qwen35_4b_oracle_distilled_semantic_verifier/models/verifier_sft_lora",
        "python scripts/make_report.py",
        "```",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    (ROOT / "reports" / "figures").mkdir(parents=True, exist_ok=True)
    records, scores = load_eval()
    if records.empty:
        raise RuntimeError("no evaluation JSON files found")
    summary = write_tables(records, scores)
    plot_summary(summary)
    plot_training_curve()
    path = ROOT / "reports" / "qwen35_4b_oracle_distilled_semantic_verifier_report.md"
    path.write_text(make_markdown(summary), encoding="utf-8")
    print(json.dumps({"report": str(path), "rows": len(records)}, indent=2))


if __name__ == "__main__":
    main()
