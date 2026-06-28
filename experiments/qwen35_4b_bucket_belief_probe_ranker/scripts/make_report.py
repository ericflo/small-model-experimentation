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


POLICY_ORDER = ["split_top1", "oracle_top8", "fullpool_oracle", "base_bucket_ranker", "sft_bucket_ranker"]
POLICY_LABELS = {
    "split_top1": "Split top-1",
    "oracle_top8": "Oracle over top-8",
    "fullpool_oracle": "Oracle over full pool",
    "base_bucket_ranker": "Base bucket ranker",
    "sft_bucket_ranker": "SFT bucket ranker",
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
    return f"{100.0 * value:.1f}%"


def write_csvs(records: pd.DataFrame, actions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = (
        records.groupby(["policy", "library_size", "template", "budget"], as_index=False)
        .agg(
            records=("record_id", "count"),
            selected_hidden_all=("selected_hidden_all", "mean"),
            selected_exact_pair=("selected_exact_pair", "mean"),
            target_reachable=("target_reachable", "mean"),
            candidate_count_mean=("candidate_count", "mean"),
            hidden_equivalent_candidates_mean=("hidden_equivalent_candidates", "mean"),
            mean_chosen_reward=("chosen_reward", "mean"),
        )
        .sort_values(["policy", "library_size", "template", "budget"])
    )
    overall = (
        records.groupby(["policy", "budget"], as_index=False)
        .agg(
            records=("record_id", "count"),
            selected_hidden_all=("selected_hidden_all", "mean"),
            selected_exact_pair=("selected_exact_pair", "mean"),
            target_reachable=("target_reachable", "mean"),
            candidate_count_mean=("candidate_count", "mean"),
            hidden_equivalent_candidates_mean=("hidden_equivalent_candidates", "mean"),
        )
        .sort_values(["budget", "policy"])
    )
    by_template = (
        records.groupby(["policy", "template", "budget"], as_index=False)
        .agg(
            records=("record_id", "count"),
            selected_hidden_all=("selected_hidden_all", "mean"),
            candidate_count_mean=("candidate_count", "mean"),
            hidden_equivalent_candidates_mean=("hidden_equivalent_candidates", "mean"),
        )
        .sort_values(["policy", "template", "budget"])
    )
    summary.to_csv(ROOT / "reports" / "summary_by_cell.csv", index=False)
    overall.to_csv(ROOT / "reports" / "summary_overall.csv", index=False)
    by_template.to_csv(ROOT / "reports" / "summary_by_template.csv", index=False)
    if not actions.empty:
        actions.to_csv(ROOT / "reports" / "probe_action_records.csv", index=False)
    return summary, overall, by_template


def plot_budget_curve(overall: pd.DataFrame) -> None:
    fig_dir = ROOT / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for policy in POLICY_ORDER:
        subset = overall[overall["policy"] == policy].sort_values("budget")
        if subset.empty:
            continue
        plt.plot(
            subset["budget"],
            100 * subset["selected_hidden_all"],
            marker="o",
            linewidth=2,
            label=POLICY_LABELS.get(policy, policy),
        )
    plt.xlabel("Probe budget")
    plt.ylabel("Hidden-all selected accuracy (%)")
    plt.title("Rollout Accuracy by Probe Budget")
    plt.ylim(0, 105)
    plt.grid(alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(fig_dir / "budget_curve.png", dpi=180)
    plt.close()


def plot_budget3_bar(overall: pd.DataFrame) -> None:
    fig_dir = ROOT / "reports" / "figures"
    b3 = overall[overall["budget"] == overall["budget"].max()].copy()
    b3["order"] = b3["policy"].map({policy: i for i, policy in enumerate(POLICY_ORDER)})
    b3 = b3.sort_values("order")
    plt.figure(figsize=(8, 4.8))
    plt.bar([POLICY_LABELS.get(p, p) for p in b3["policy"]], 100 * b3["selected_hidden_all"], color="#386fa4")
    plt.ylabel("Hidden-all selected accuracy (%)")
    plt.title("Budget-3 Accuracy")
    plt.xticks(rotation=25, ha="right")
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig(fig_dir / "budget3_accuracy.png", dpi=180)
    plt.close()


def plot_template_bars(by_template: pd.DataFrame) -> None:
    fig_dir = ROOT / "reports" / "figures"
    b3 = by_template[by_template["budget"] == by_template["budget"].max()].copy()
    b3["policy_label"] = b3["policy"].map(POLICY_LABELS).fillna(b3["policy"])
    pivot = b3.pivot(index="policy_label", columns="template", values="selected_hidden_all")
    order = [POLICY_LABELS[p] for p in POLICY_ORDER if POLICY_LABELS[p] in pivot.index]
    pivot = pivot.loc[order]
    ax = (100 * pivot).plot(kind="bar", figsize=(8, 5), color=["#2a9d8f", "#e76f51"])
    ax.set_ylabel("Hidden-all selected accuracy (%)")
    ax.set_title("Budget-3 Accuracy by Template")
    ax.set_ylim(0, 105)
    ax.legend(title="")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(fig_dir / "budget3_by_template.png", dpi=180)
    plt.close()


def plot_training_curve() -> None:
    path = ROOT / "reports" / "bucket_sft_training_losses.json"
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
    plt.title("Bucket-Belief SFT Loss")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(ROOT / "reports" / "figures" / "bucket_sft_loss.png", dpi=180)
    plt.close()


def metric(df: pd.DataFrame, policy: str, budget: int, column: str = "selected_hidden_all") -> float | None:
    row = df[(df["policy"] == policy) & (df["budget"] == budget)]
    if row.empty:
        return None
    return float(row.iloc[0][column])


def build_markdown(overall: pd.DataFrame, by_template: pd.DataFrame, summary: pd.DataFrame) -> str:
    max_budget = int(overall["budget"].max())
    b3_rows: list[str] = []
    for policy in POLICY_ORDER:
        val = metric(overall, policy, max_budget)
        cand = metric(overall, policy, max_budget, "candidate_count_mean")
        hidden = metric(overall, policy, max_budget, "hidden_equivalent_candidates_mean")
        if val is None:
            continue
        b3_rows.append(
            f"| {POLICY_LABELS.get(policy, policy)} | {pct(val)} | {cand:.1f} | {hidden:.1f} |"
        )

    template_rows: list[str] = []
    final_template = by_template[by_template["budget"] == max_budget]
    for policy in POLICY_ORDER:
        subset = final_template[final_template["policy"] == policy]
        if subset.empty:
            continue
        values = {row["template"]: float(row["selected_hidden_all"]) for _, row in subset.iterrows()}
        template_rows.append(
            f"| {POLICY_LABELS.get(policy, policy)} | {pct(values.get('pair_affine_mod', 0.0))} | {pct(values.get('pair_compare_gate', 0.0))} |"
        )

    key_lines: list[str] = []
    split = metric(overall, "split_top1", max_budget)
    sft = metric(overall, "sft_bucket_ranker", max_budget)
    top8 = metric(overall, "oracle_top8", max_budget)
    full = metric(overall, "fullpool_oracle", max_budget)
    base = metric(overall, "base_bucket_ranker", max_budget)
    if all(value is not None for value in [split, sft, top8, full, base]):
        key_lines.append(
            f"- At budget {max_budget}, the SFT bucket ranker reached {pct(sft)} hidden-all accuracy versus "
            f"{pct(split)} for target-independent split top-1, {pct(base)} for the base bucket ranker, "
            f"{pct(top8)} for an oracle over the same top-8 probe set, and {pct(full)} for the full-pool oracle."
        )
        key_lines.append(
            f"- The recoverable top-8 oracle gap is {100 * (top8 - split):.1f} points over split top-1; "
            f"the SFT ranker captured {100 * (sft - split):.1f} points of that gap."
        )
    else:
        key_lines.append("- One or more evaluation arms are missing; read the tables below as partial results.")

    manifest_path = ROOT / "data" / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    bucket_manifest_path = ROOT / "data" / "bucket_train_examples.manifest.json"
    bucket_manifest = json.loads(bucket_manifest_path.read_text(encoding="utf-8")) if bucket_manifest_path.exists() else {}

    lines = [
        "# Qwen3.5-4B Bucket-Belief Probe Ranker",
        "",
        "## Objective",
        "",
        "This standalone experiment tests whether a Qwen3.5-4B LoRA can turn verifier state into a deployable probe-ranking policy by predicting which candidate-output bucket contains the hidden target program. The model does not name operators and does not see the answer at inference time; it scores candidate probes by the expected survivors implied by its bucket probabilities.",
        "",
        "## Experimental Design",
        "",
        "- Substrate: two-operator typed program search with a 96-case probe pool, four visible observations, sixteen hidden checks, and library sizes 64, 128, 256, and held-out 512 at evaluation.",
        "- Candidate probes for the learned ranker: top 8 remaining probes by target-independent split statistics.",
        "- Training target: for each candidate probe, predict the output bucket that contains the true target program.",
        "- Rollout rule: choose the probe with the smallest model-predicted expected survivor count, observe its true output, update the verifier candidate set, and repeat for three probes.",
        "- Controls: target-independent split top-1, target-aware oracle over the same top-8 candidate probes, target-aware oracle over the full 96-case pool, and the untrained base model under the same bucket-scoring rule.",
        "",
        "## Data Summary",
        "",
        f"- Process records: train={manifest.get('train_records', {}).get('records', 'unknown')}, eval={manifest.get('eval_records', {}).get('records', 'unknown')}.",
        f"- Bucket SFT examples: {bucket_manifest.get('summary', {}).get('examples', 'unknown')}.",
        f"- Bucket SFT states: {bucket_manifest.get('summary', {}).get('states', 'unknown')}.",
        "",
        "## Results",
        "",
        *key_lines,
        "",
        "### Budget-3 Overall",
        "",
        "| Arm | Hidden-all accuracy | Candidates left | Hidden-equivalent left |",
        "|---|---:|---:|---:|",
        *b3_rows,
        "",
        "### Budget-3 by Template",
        "",
        "| Arm | Affine-mod | Compare-gate |",
        "|---|---:|---:|",
        *template_rows,
        "",
        "## Interpretation",
        "",
        "The experiment is intentionally decisive about whether target-aware oracle headroom can be converted into deployable ranking by a learned bucket-belief model. A lift over split top-1 means the model learned a useful non-uniform belief over output buckets. A result near split top-1 means the oracle gap is mostly unavailable without additional state, candidate representation, or truly generative probe construction.",
        "",
        "The base-model arm matters because it distinguishes learned bucket inference from prompt priors. The oracle-over-top-8 arm matters because it bounds what any ranker can gain when it is restricted to the same split-mined candidate probes. The full-pool oracle remains a headroom measurement, not a deployable result.",
        "",
        "## Figures",
        "",
        "- `reports/figures/budget_curve.png`",
        "- `reports/figures/budget3_accuracy.png`",
        "- `reports/figures/budget3_by_template.png`",
        "- `reports/figures/bucket_sft_loss.png`",
        "",
        "## Reproduction",
        "",
        "Run from this experiment directory:",
        "",
        "```bash",
        "python scripts/build_dataset.py --train-per-cell 50 --eval-per-cell 20 --states-per-record 3 --query-pool-cases 96 --action-source mined8",
        "python scripts/build_bucket_dataset.py",
        "python scripts/build_bucket_dataset.py --records data/eval_records.jsonl --states data/process_eval_states.jsonl --out data/bucket_eval_examples.jsonl",
        "python scripts/eval_bucket_ranker.py --policy split_top1 --name split_top1 --max-budget 3",
        "python scripts/eval_bucket_ranker.py --policy oracle_topk --name oracle_top8 --max-budget 3",
        "python scripts/eval_bucket_ranker.py --policy fullpool_oracle --name fullpool_oracle --max-budget 3",
        "python scripts/eval_bucket_ranker.py --policy base_bucket --name base_bucket_ranker --max-budget 3",
        "python scripts/train_bucket_sft.py --max-steps 220 --batch-size 2 --grad-accum 2",
        "python scripts/eval_bucket_ranker.py --policy adapter_bucket --name sft_bucket_ranker --adapter-dir /workspace/large_artifacts/qwen35_4b_bucket_belief_probe_ranker/models/bucket_sft_lora --max-budget 3",
        "python scripts/make_report.py",
        "```",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    (ROOT / "reports" / "figures").mkdir(parents=True, exist_ok=True)
    records, actions = load_eval()
    if records.empty:
        raise RuntimeError("no evaluation JSON files found under reports/eval")
    summary, overall, by_template = write_csvs(records, actions)
    plot_budget_curve(overall)
    plot_budget3_bar(overall)
    plot_template_bars(by_template)
    plot_training_curve()
    report = build_markdown(overall, by_template, summary)
    report_path = ROOT / "reports" / "qwen35_4b_bucket_belief_probe_ranker_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps({"report": str(report_path), "rows": len(records)}, indent=2))


if __name__ == "__main__":
    main()
