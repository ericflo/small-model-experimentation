#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

POLICY_ORDER = [
    "fixed_budget0",
    "fixed_budget3",
    "fixed_budget6",
    "fixed_budget8",
    "threshold_70",
    "threshold_90",
    "oracle_stop",
    "base_budget_policy",
    "sft_budget_policy",
]

POLICY_LABELS = {
    "fixed_budget0": "Fixed budget 0",
    "fixed_budget3": "Fixed budget 3",
    "fixed_budget6": "Fixed budget 6",
    "fixed_budget8": "Fixed budget 8",
    "threshold_70": "Stop if cluster >=70%",
    "threshold_90": "Stop if cluster >=90%",
    "oracle_stop": "Oracle stop",
    "base_budget_policy": "Base Qwen stop/more",
    "sft_budget_policy": "SFT Qwen stop/more",
}


def load_eval() -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    for path in sorted((ROOT / "reports" / "eval").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        records.extend(payload.get("records", []))
        actions.extend(payload.get("actions", []))
    return pd.DataFrame(records), pd.DataFrame(actions)


def ordered(df: pd.DataFrame) -> pd.DataFrame:
    order = {policy: i for i, policy in enumerate(POLICY_ORDER)}
    return df.assign(order=df["policy"].map(order).fillna(999)).sort_values(["order", "policy"])


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def value(df: pd.DataFrame, policy: str, column: str) -> float:
    row = df[df["policy"] == policy]
    if row.empty:
        return float("nan")
    return float(row.iloc[0][column])


def write_tables(records: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    overall = (
        records.groupby("policy", as_index=False)
        .agg(
            records=("record_id", "count"),
            selected_hidden_correct=("selected_hidden_correct", "mean"),
            target_reachable=("target_reachable", "mean"),
            used_probes_mean=("used_probes", "mean"),
            candidate_count_mean=("candidate_count", "mean"),
            agreement_cluster_count_mean=("agreement_cluster_count", "mean"),
            selected_cluster_fraction_mean=("selected_cluster_fraction", "mean"),
            hidden_correct_survivors_mean=("hidden_correct_survivors", "mean"),
        )
        .sort_values("policy")
    )
    by_task = (
        records.groupby(["policy", "task_id"], as_index=False)
        .agg(
            selected_hidden_correct=("selected_hidden_correct", "mean"),
            used_probes=("used_probes", "mean"),
            candidate_count=("candidate_count", "mean"),
        )
        .sort_values(["policy", "task_id"])
    )
    overall.to_csv(ROOT / "reports" / "summary_overall.csv", index=False)
    by_task.to_csv(ROOT / "reports" / "summary_by_task.csv", index=False)
    if not actions.empty:
        actions.to_csv(ROOT / "reports" / "action_records.csv", index=False)
    return overall


def plot_pareto(overall: pd.DataFrame) -> None:
    fig_dir = ROOT / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = ordered(overall)
    plt.figure(figsize=(8, 5.5))
    for _, row in df.iterrows():
        plt.scatter(row["used_probes_mean"], 100 * row["selected_hidden_correct"], s=70)
        plt.text(
            row["used_probes_mean"] + 0.05,
            100 * row["selected_hidden_correct"],
            POLICY_LABELS.get(row["policy"], row["policy"]),
            fontsize=8,
        )
    plt.xlabel("Average executable probes used")
    plt.ylabel("Hidden-correct selected candidate (%)")
    plt.title("HumanEval Adaptive Evidence Budget")
    plt.ylim(0, 105)
    plt.xlim(left=-0.3)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(fig_dir / "accuracy_vs_probes.png", dpi=180)
    plt.close()


def plot_bars(overall: pd.DataFrame) -> None:
    fig_dir = ROOT / "reports" / "figures"
    df = ordered(overall)
    labels = [POLICY_LABELS.get(policy, policy) for policy in df["policy"]]
    plt.figure(figsize=(9, 5))
    plt.bar(labels, 100 * df["selected_hidden_correct"], color="#2a9d8f")
    plt.ylabel("Hidden-correct selected candidate (%)")
    plt.ylim(0, 105)
    plt.xticks(rotation=25, ha="right")
    plt.title("Accuracy by Policy")
    plt.tight_layout()
    plt.savefig(fig_dir / "accuracy_by_policy.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.bar(labels, df["used_probes_mean"], color="#457b9d")
    plt.ylabel("Average executable probes used")
    plt.xticks(rotation=25, ha="right")
    plt.title("Probe Use by Policy")
    plt.tight_layout()
    plt.savefig(fig_dir / "probes_by_policy.png", dpi=180)
    plt.close()


def plot_training_curve() -> None:
    path = ROOT / "reports" / "budget_sft_training_losses.json"
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
    plt.title("STOP/MORE SFT Loss")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(ROOT / "reports" / "figures" / "budget_sft_loss.png", dpi=180)
    plt.close()


def make_markdown(overall: pd.DataFrame) -> str:
    df = ordered(overall)
    rows = [
        f"| {POLICY_LABELS.get(row['policy'], row['policy'])} | {pct(float(row['selected_hidden_correct']))} | "
        f"{pct(float(row['target_reachable']))} | {float(row['used_probes_mean']):.2f} | "
        f"{float(row['agreement_cluster_count_mean']):.2f} | {pct(float(row['selected_cluster_fraction_mean']))} | "
        f"{float(row['hidden_correct_survivors_mean']):.2f} |"
        for _, row in df.iterrows()
    ]
    manifest = json.loads((ROOT / "data" / "dataset_manifest.json").read_text(encoding="utf-8"))
    sft_acc = value(overall, "sft_budget_policy", "selected_hidden_correct")
    sft_probe = value(overall, "sft_budget_policy", "used_probes_mean")
    fixed8_acc = value(overall, "fixed_budget8", "selected_hidden_correct")
    fixed8_probe = value(overall, "fixed_budget8", "used_probes_mean")
    oracle_acc = value(overall, "oracle_stop", "selected_hidden_correct")
    oracle_probe = value(overall, "oracle_stop", "used_probes_mean")
    base_acc = value(overall, "base_budget_policy", "selected_hidden_correct")
    coverage = value(overall, "fixed_budget0", "target_reachable")

    lines = [
        "# Qwen3.5-4B HumanEval Adaptive Evidence Budget",
        "",
        "## Objective",
        "",
        "This standalone experiment tests whether Qwen3.5-4B can be posttrained as a STOP/MORE controller for an executable Python verifier on HumanEval tasks. The verifier generates candidate implementations, chooses unlabeled probes by target-independent output-agreement split, and commits the first candidate in the largest output-agreement cluster. The model only decides whether to commit or spend one more executable probe.",
        "",
        "## Dataset",
        "",
        f"- Source: `{manifest['dataset']}`.",
        f"- Train tasks: {manifest['train_records']['records']}; eval tasks: {manifest['eval_records']['records']}.",
        f"- Visible tests per task: {manifest['visible_tests']}; probe pool: {manifest['probe_tests']}; generated hidden tests: {manifest['hidden_tests']}.",
        f"- Candidate implementations per task: {manifest['candidate_count']} maximum, from canonical-solution mutations and generic fallback bodies.",
        "- Public doctest examples are the only labeled visible tests. Generated probes are unlabeled and are used only to form candidate agreement clusters. Reference outputs for generated probes are stored only as audit metadata.",
        f"- STOP/MORE train states: {manifest['train_states']['states']}; eval states: {manifest['eval_states']['states']}.",
        "",
        "## Key Result",
        "",
        f"- SFT STOP/MORE reached {pct(sft_acc)} hidden-correct selection with {sft_probe:.2f} probes on average.",
        f"- Fixed budget 8 reached {pct(fixed8_acc)} with {fixed8_probe:.2f} probes.",
        f"- Oracle stopping reached {pct(oracle_acc)} with {oracle_probe:.2f} probes.",
        f"- Base Qwen STOP/MORE reached {pct(base_acc)}.",
        f"- Candidate-pool coverage was {pct(coverage)}: a hidden-correct candidate was present for every eval task, but the leak-free agreement selector usually did not choose it.",
        "",
        "## Overall",
        "",
        "| Policy | Hidden-correct selected | Candidate-pool coverage | Avg probes | Agreement clusters | Selected-cluster share | Hidden-correct survivors |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "## Interpretation",
        "",
        "This pilot is a clean negative for STOP/MORE budget control under this leak-free HumanEval evidence model. The candidate pool contains a hidden-correct implementation for every eval task, so the low final accuracy is not a coverage failure. The failure is that unlabeled generated probes split the visible-passing candidates into agreement clusters without grounding which cluster is correct. Spending more probes changes cluster structure but does not move the selected candidate onto the hidden-correct implementation, so even a hidden-aware oracle stopping rule has no accuracy to recover.",
        "",
        "The SFT controller learned a cheaper stopping behavior than fixed budget 8, but because the selected candidate remained wrong on 11 of 12 eval tasks, the cost saving is not useful. For this benchmark shape, the next useful lever is not a better STOP/MORE controller over unlabeled tests; it is either a stronger deployable candidate selector, a trustworthy labeled-test generator, or an adaptive generation budget that samples more candidate programs when the visible-passing pool is poorly grounded.",
        "",
        "This is a pilot-scale result. HumanEval public examples and randomly generated in-domain probes limited the usable split to 24 train and 12 eval tasks. The conclusion should be read as a substrate diagnostic, not as a benchmark-level pass-rate estimate.",
        "",
        "## Figures",
        "",
        "- `reports/figures/accuracy_vs_probes.png`",
        "- `reports/figures/accuracy_by_policy.png`",
        "- `reports/figures/probes_by_policy.png`",
        "- `reports/figures/budget_sft_loss.png`",
        "",
        "## Reproduction",
        "",
        "```bash",
        "python scripts/build_dataset.py --train-tasks 24 --eval-tasks 12 --visible-tests 1 --probe-tests 8 --hidden-tests 0 --candidate-count 16 --max-budget 8",
        "python scripts/eval_budget_policy.py --policy fixed --name fixed_budget0 --fixed-budget 0 --max-budget 8",
        "python scripts/eval_budget_policy.py --policy fixed --name fixed_budget3 --fixed-budget 3 --max-budget 8",
        "python scripts/eval_budget_policy.py --policy fixed --name fixed_budget6 --fixed-budget 6 --max-budget 8",
        "python scripts/eval_budget_policy.py --policy fixed --name fixed_budget8 --fixed-budget 8 --max-budget 8",
        "python scripts/eval_budget_policy.py --policy threshold --name threshold_70 --threshold 70 --max-budget 8",
        "python scripts/eval_budget_policy.py --policy threshold --name threshold_90 --threshold 90 --max-budget 8",
        "python scripts/eval_budget_policy.py --policy oracle_stop --name oracle_stop --max-budget 8",
        "python scripts/eval_budget_policy.py --policy base --name base_budget_policy --max-budget 8",
        "python scripts/train_budget_sft.py --max-steps 160 --batch-size 2 --grad-accum 2",
        "python scripts/eval_budget_policy.py --policy adapter --name sft_budget_policy --adapter-dir /workspace/large_artifacts/qwen35_4b_humaneval_adaptive_budget/models/budget_sft_lora --max-budget 8",
        "python scripts/make_report.py",
        "```",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    (ROOT / "reports" / "figures").mkdir(parents=True, exist_ok=True)
    records, actions = load_eval()
    if records.empty:
        raise RuntimeError("no evaluation JSON files found")
    overall = write_tables(records, actions)
    plot_pareto(overall)
    plot_bars(overall)
    plot_training_curve()
    path = ROOT / "reports" / "qwen35_4b_humaneval_adaptive_budget_report.md"
    path.write_text(make_markdown(overall), encoding="utf-8")
    print(json.dumps({"report": str(path), "rows": len(records)}, indent=2))


if __name__ == "__main__":
    main()
