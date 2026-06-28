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
    "fixed_budget3",
    "fixed_budget6",
    "fixed_budget10",
    "threshold_100",
    "threshold_1000",
    "oracle_stop",
    "base_budget_policy",
    "sft_budget_policy",
]

POLICY_LABELS = {
    "fixed_budget3": "Fixed budget 3",
    "fixed_budget6": "Fixed budget 6",
    "fixed_budget10": "Fixed budget 10",
    "threshold_100": "Threshold <=100",
    "threshold_1000": "Threshold <=1000",
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


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def write_tables(records: pd.DataFrame, actions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_cell = (
        records.groupby(["policy", "library_size", "template"], as_index=False)
        .agg(
            records=("record_id", "count"),
            selected_hidden_all=("selected_hidden_all", "mean"),
            used_probes_mean=("used_probes", "mean"),
            candidate_count_mean=("candidate_count", "mean"),
            hidden_equivalent_candidates_mean=("hidden_equivalent_candidates", "mean"),
        )
        .sort_values(["policy", "library_size", "template"])
    )
    by_template = (
        records.groupby(["policy", "template"], as_index=False)
        .agg(
            records=("record_id", "count"),
            selected_hidden_all=("selected_hidden_all", "mean"),
            used_probes_mean=("used_probes", "mean"),
            candidate_count_mean=("candidate_count", "mean"),
            hidden_equivalent_candidates_mean=("hidden_equivalent_candidates", "mean"),
        )
        .sort_values(["policy", "template"])
    )
    overall = (
        records.groupby(["policy"], as_index=False)
        .agg(
            records=("record_id", "count"),
            selected_hidden_all=("selected_hidden_all", "mean"),
            used_probes_mean=("used_probes", "mean"),
            candidate_count_mean=("candidate_count", "mean"),
            hidden_equivalent_candidates_mean=("hidden_equivalent_candidates", "mean"),
        )
        .sort_values(["policy"])
    )
    by_cell.to_csv(ROOT / "reports" / "summary_by_cell.csv", index=False)
    by_template.to_csv(ROOT / "reports" / "summary_by_template.csv", index=False)
    overall.to_csv(ROOT / "reports" / "summary_overall.csv", index=False)
    if not actions.empty:
        actions.to_csv(ROOT / "reports" / "action_records.csv", index=False)
    return by_cell, by_template, overall


def ordered(df: pd.DataFrame) -> pd.DataFrame:
    order = {policy: i for i, policy in enumerate(POLICY_ORDER)}
    return df.assign(order=df["policy"].map(order).fillna(999)).sort_values(["order", "policy"])


def plot_pareto(overall: pd.DataFrame) -> None:
    fig_dir = ROOT / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = ordered(overall)
    plt.figure(figsize=(8, 5.5))
    for _, row in df.iterrows():
        plt.scatter(row["used_probes_mean"], 100 * row["selected_hidden_all"], s=70)
        plt.text(
            row["used_probes_mean"] + 0.06,
            100 * row["selected_hidden_all"],
            POLICY_LABELS.get(row["policy"], row["policy"]),
            fontsize=8,
        )
    plt.xlabel("Average probes used")
    plt.ylabel("Hidden-all selected accuracy (%)")
    plt.title("Accuracy vs Evidence Budget")
    plt.ylim(0, 105)
    plt.xlim(left=-0.2)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(fig_dir / "accuracy_vs_probes.png", dpi=180)
    plt.close()


def plot_by_template(by_template: pd.DataFrame) -> None:
    fig_dir = ROOT / "reports" / "figures"
    df = ordered(by_template)
    pivot_acc = 100 * df.pivot(index="policy", columns="template", values="selected_hidden_all")
    pivot_probe = df.pivot(index="policy", columns="template", values="used_probes_mean")
    order = [policy for policy in POLICY_ORDER if policy in pivot_acc.index]
    pivot_acc = pivot_acc.loc[order]
    pivot_probe = pivot_probe.loc[order]

    ax = pivot_acc.rename(index=POLICY_LABELS).plot(kind="bar", figsize=(9, 5.5), color=["#2a9d8f", "#e76f51"])
    ax.set_ylabel("Hidden-all selected accuracy (%)")
    ax.set_title("Accuracy by Template")
    ax.set_ylim(0, 105)
    ax.legend(title="")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(fig_dir / "accuracy_by_template.png", dpi=180)
    plt.close()

    ax = pivot_probe.rename(index=POLICY_LABELS).plot(kind="bar", figsize=(9, 5.5), color=["#457b9d", "#f4a261"])
    ax.set_ylabel("Average probes used")
    ax.set_title("Probe Use by Template")
    ax.legend(title="")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(fig_dir / "probes_by_template.png", dpi=180)
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
    plt.title("Budget Policy SFT Loss")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(ROOT / "reports" / "figures" / "budget_sft_loss.png", dpi=180)
    plt.close()


def value(df: pd.DataFrame, policy: str, column: str) -> float:
    row = df[df["policy"] == policy]
    if row.empty:
        return float("nan")
    return float(row.iloc[0][column])


def make_markdown(overall: pd.DataFrame, by_template: pd.DataFrame) -> str:
    df = ordered(overall)
    overall_rows = [
        f"| {POLICY_LABELS.get(row['policy'], row['policy'])} | {pct(row['selected_hidden_all'])} | "
        f"{row['used_probes_mean']:.2f} | {row['candidate_count_mean']:.1f} | {row['hidden_equivalent_candidates_mean']:.1f} |"
        for _, row in df.iterrows()
    ]
    template_rows: list[str] = []
    for policy in POLICY_ORDER:
        subset = by_template[by_template["policy"] == policy]
        if subset.empty:
            continue
        vals = {row["template"]: row for _, row in subset.iterrows()}
        cells: list[str] = []
        for template in ["pair_affine_mod", "pair_compare_gate"]:
            if template not in vals:
                cells.append("n/a")
            else:
                row = vals[template]
                cells.append(f"{pct(float(row['selected_hidden_all']))} / {float(row['used_probes_mean']):.2f}")
        template_rows.append(f"| {POLICY_LABELS.get(policy, policy)} | {cells[0]} | {cells[1]} |")

    sft_acc = value(overall, "sft_budget_policy", "selected_hidden_all")
    sft_probe = value(overall, "sft_budget_policy", "used_probes_mean")
    fixed3_acc = value(overall, "fixed_budget3", "selected_hidden_all")
    fixed6_acc = value(overall, "fixed_budget6", "selected_hidden_all")
    fixed10_acc = value(overall, "fixed_budget10", "selected_hidden_all")
    oracle_acc = value(overall, "oracle_stop", "selected_hidden_all")
    oracle_avg_probes = value(overall, "oracle_stop", "used_probes_mean")

    manifest_path = ROOT / "data" / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    lines = [
        "# Qwen3.5-4B Adaptive Evidence Budget Policy",
        "",
        "## Objective",
        "",
        "This standalone experiment tests whether Qwen3.5-4B can be posttrained as a STOP/MORE controller for an executable verifier. The verifier supplies the best deployable next probe by target-independent expected split; the model decides whether to commit now or spend another probe, up to a maximum budget of ten.",
        "",
        "## Data",
        "",
        f"- Train records: {manifest.get('train_records', {}).get('records', 'unknown')}.",
        f"- Eval records: {manifest.get('eval_records', {}).get('records', 'unknown')}.",
        f"- Train STOP/MORE states: {manifest.get('train_states', {}).get('states', 'unknown')}.",
        f"- Eval STOP/MORE states: {manifest.get('eval_states', {}).get('states', 'unknown')}.",
        "",
        "## Key Result",
        "",
        f"- SFT STOP/MORE reached {pct(sft_acc)} accuracy using {sft_probe:.2f} probes on average.",
        f"- Fixed budget 3/6/10 reached {pct(fixed3_acc)} / {pct(fixed6_acc)} / {pct(fixed10_acc)}.",
        f"- Oracle stopping reached {pct(oracle_acc)} using {oracle_avg_probes:.2f} probes on average.",
        "",
        "## Overall",
        "",
        "| Policy | Hidden-all accuracy | Avg probes | Candidates left | Hidden-equivalent left |",
        "|---|---:|---:|---:|---:|",
        *overall_rows,
        "",
        "## By Template",
        "",
        "Cells show `accuracy / average probes`.",
        "",
        "| Policy | Affine-mod | Compare-gate |",
        "|---|---:|---:|",
        *template_rows,
        "",
        "## Interpretation",
        "",
        "This test separates the value of the adaptive inference loop from the value of learning the stop rule. If the SFT policy lies on or above the fixed-budget Pareto curve, posttraining learned useful budget control. If fixed budgets dominate it, the practical lever is simply allowing more executable observations and using a transparent budget rule.",
        "",
        "## Figures",
        "",
        "- `reports/figures/accuracy_vs_probes.png`",
        "- `reports/figures/accuracy_by_template.png`",
        "- `reports/figures/probes_by_template.png`",
        "- `reports/figures/budget_sft_loss.png`",
        "",
        "## Reproduction",
        "",
        "```bash",
        "python scripts/build_dataset.py --train-per-cell 40 --eval-per-cell 20 --query-pool-cases 96 --max-budget 10",
        "python scripts/eval_budget_policy.py --policy fixed --name fixed_budget3 --fixed-budget 3 --max-budget 10",
        "python scripts/eval_budget_policy.py --policy fixed --name fixed_budget6 --fixed-budget 6 --max-budget 10",
        "python scripts/eval_budget_policy.py --policy fixed --name fixed_budget10 --fixed-budget 10 --max-budget 10",
        "python scripts/eval_budget_policy.py --policy threshold --name threshold_100 --threshold 100 --max-budget 10",
        "python scripts/eval_budget_policy.py --policy threshold --name threshold_1000 --threshold 1000 --max-budget 10",
        "python scripts/eval_budget_policy.py --policy oracle_stop --name oracle_stop --max-budget 10",
        "python scripts/eval_budget_policy.py --policy base --name base_budget_policy --max-budget 10",
        "python scripts/train_budget_sft.py --max-steps 220 --batch-size 2 --grad-accum 2",
        "python scripts/eval_budget_policy.py --policy adapter --name sft_budget_policy --adapter-dir /workspace/large_artifacts/qwen35_4b_adaptive_evidence_budget_policy/models/budget_sft_lora --max-budget 10",
        "python scripts/make_report.py",
        "```",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    (ROOT / "reports" / "figures").mkdir(parents=True, exist_ok=True)
    records, actions = load_eval()
    if records.empty:
        raise RuntimeError("no evaluation JSON files found")
    by_cell, by_template, overall = write_tables(records, actions)
    plot_pareto(overall)
    plot_by_template(by_template)
    plot_training_curve()
    report = make_markdown(overall, by_template)
    path = ROOT / "reports" / "qwen35_4b_adaptive_evidence_budget_policy_report.md"
    path.write_text(report, encoding="utf-8")
    print(json.dumps({"report": str(path), "rows": len(records)}, indent=2))


if __name__ == "__main__":
    main()
