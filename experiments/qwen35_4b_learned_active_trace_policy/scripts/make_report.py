#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


POLICY_ORDER = [
    "visible_prior",
    "random_extra",
    "active_max_split",
    "learned_qwen_policy",
    "oracle_elimination",
]

POLICY_LABELS = {
    "visible_prior": "visible only",
    "random_extra": "random extra",
    "active_max_split": "active max-split",
    "learned_qwen_policy": "learned Qwen policy",
    "oracle_elimination": "oracle elimination",
}


def load_result(path: Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    policy = pd.DataFrame(data["policy_rows"])
    candidate = pd.DataFrame(data["candidate_rows"])
    policy["split"] = split
    candidate["split"] = split
    return policy, candidate, data


def summarize_policy(policy: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        policy.groupby(["split", "policy", "budget"], as_index=False)
        .agg(
            rows=("id", "count"),
            hidden_all_rate=("hidden_all", "mean"),
            observed_all_rate=("observed_all", "mean"),
            avg_hidden_passes=("hidden_passes", "mean"),
            avg_queries_used=("queries_used", "mean"),
        )
        .sort_values(["split", "policy", "budget"])
    )
    grouped["hidden_all_pct"] = grouped["hidden_all_rate"] * 100
    grouped["observed_all_pct"] = grouped["observed_all_rate"] * 100
    return grouped


def summarize_candidates(candidate: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        candidate.groupby("split", as_index=False)
        .agg(
            records=("id", "count"),
            candidate_oracle_hidden_all_rate=("candidate_oracle_hidden_all", "mean"),
            visible_selected_hidden_all_rate=("visible_selected_hidden_all", "mean"),
            target_program_synthesized_rate=("target_program_synthesized", "mean"),
            avg_synthesized_programs=("synthesized_program_count", "mean"),
            avg_visible_consistent_candidates=("visible_consistent_candidates", "mean"),
            avg_unique_sketches=("unique_sketch_count", "mean"),
        )
        .sort_values("split")
    )
    for column in [
        "candidate_oracle_hidden_all_rate",
        "visible_selected_hidden_all_rate",
        "target_program_synthesized_rate",
    ]:
        grouped[column.replace("_rate", "_pct")] = grouped[column] * 100
    return grouped


def learned_trace_rows(policy: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in policy[policy["policy"] == "learned_qwen_policy"].iterrows():
        for trace in row.get("query_trace", []) or []:
            rows.append(
                {
                    "split": row["split"],
                    "id": row["id"],
                    "budget": row["budget"],
                    "step": trace.get("step"),
                    "parse_ok": bool(trace.get("parse_ok")),
                    "fallback_used": bool(trace.get("fallback_used")),
                    "chosen_option_rank": trace.get("chosen_option_rank"),
                    "chosen_actual_eliminated": trace.get("chosen_actual_eliminated"),
                    "oracle_actual_eliminated": trace.get("oracle_actual_eliminated"),
                }
            )
    return pd.DataFrame(rows)


def plot_success_by_budget(summary: pd.DataFrame, split: str, out_path: Path) -> None:
    subset = summary[summary["split"] == split]
    plt.figure(figsize=(8.8, 5.2))
    for policy in POLICY_ORDER:
        rows = subset[subset["policy"] == policy].sort_values("budget")
        if rows.empty:
            continue
        plt.plot(rows["budget"], rows["hidden_all_pct"], marker="o", linewidth=2, label=POLICY_LABELS[policy])
    plt.xlabel("Extra queried execution cases")
    plt.ylabel("Hidden full-pass rate (%)")
    plt.title(f"{split}: hidden success vs query budget")
    plt.ylim(0, 105)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_budget_comparison(summary: pd.DataFrame, budget: int, out_path: Path) -> None:
    rows = summary[(summary["budget"].isin([0, budget])) & (summary["policy"].isin(POLICY_ORDER))].copy()
    rows = rows[(rows["policy"] == "visible_prior") | (rows["budget"] == budget)]
    rows["label"] = rows["policy"].map(
        {
            "visible_prior": "visible only",
            "random_extra": f"random +{budget}",
            "active_max_split": f"active +{budget}",
            "learned_qwen_policy": f"learned +{budget}",
            "oracle_elimination": f"oracle +{budget}",
        }
    )
    pivot = rows.pivot_table(index="split", columns="label", values="hidden_all_pct", aggfunc="mean")
    order = [
        label
        for label in [
            "visible only",
            f"random +{budget}",
            f"active +{budget}",
            f"learned +{budget}",
            f"oracle +{budget}",
        ]
        if label in pivot
    ]
    pivot = pivot[order]
    ax = pivot.plot(kind="bar", figsize=(9.2, 5.4), rot=0)
    ax.set_ylabel("Hidden full-pass rate (%)")
    ax.set_title(f"Split comparison at +{budget} queried cases")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_candidate_coverage(candidate_summary: pd.DataFrame, out_path: Path) -> None:
    rows = candidate_summary.set_index("split")[
        [
            "visible_selected_hidden_all_pct",
            "candidate_oracle_hidden_all_pct",
            "target_program_synthesized_pct",
        ]
    ].rename(
        columns={
            "visible_selected_hidden_all_pct": "visible selected",
            "candidate_oracle_hidden_all_pct": "candidate oracle",
            "target_program_synthesized_pct": "exact target synthesized",
        }
    )
    ax = rows.plot(kind="bar", figsize=(8.8, 5.2), rot=0)
    ax.set_ylabel("Records (%)")
    ax.set_title("Candidate coverage and visible selection")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_oracle_gap(summary: pd.DataFrame, budget: int, out_path: Path) -> None:
    rows = summary[summary["budget"] == budget]
    pivot = rows.pivot_table(index="split", columns="policy", values="hidden_all_pct", aggfunc="mean")
    needed = ["active_max_split", "learned_qwen_policy", "oracle_elimination"]
    if any(column not in pivot for column in needed):
        return
    gap = pd.DataFrame(
        {
            "active gap to oracle": pivot["oracle_elimination"] - pivot["active_max_split"],
            "learned gap to oracle": pivot["oracle_elimination"] - pivot["learned_qwen_policy"],
        }
    )
    ax = gap.plot(kind="bar", figsize=(8.8, 5.2), rot=0)
    ax.set_ylabel("Percentage-point gap")
    ax.set_title(f"Gap to oracle elimination at +{budget}")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_learned_rank(trace_df: pd.DataFrame, out_path: Path) -> None:
    if trace_df.empty or "chosen_option_rank" not in trace_df:
        return
    rows = trace_df.dropna(subset=["chosen_option_rank"]).copy()
    if rows.empty:
        return
    rows["chosen_option_rank"] = rows["chosen_option_rank"].astype(int)
    clipped = rows["chosen_option_rank"].clip(upper=9)
    labels = clipped.map(lambda value: "9+" if value >= 9 else str(value))
    counts = labels.value_counts().sort_index()
    ax = counts.plot(kind="bar", figsize=(8.4, 4.8), rot=0)
    ax.set_xlabel("Chosen option rank in displayed list")
    ax.set_ylabel("Learned-policy query count")
    ax.set_title("Learned policy choice ranks")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_policy_dataset(policy_manifest: dict[str, Any] | None, out_path: Path) -> None:
    if not policy_manifest:
        return
    train = policy_manifest.get("train_summary", {}).get("by_step", {})
    eval_rows = policy_manifest.get("eval_summary", {}).get("by_step", {})
    if not train and not eval_rows:
        return
    steps = sorted({int(key) for key in train} | {int(key) for key in eval_rows})
    frame = pd.DataFrame(
        {
            "step": steps,
            "train": [train.get(str(step), 0) for step in steps],
            "eval": [eval_rows.get(str(step), 0) for step in steps],
        }
    ).set_index("step")
    ax = frame.plot(kind="bar", figsize=(7.4, 4.8), rot=0)
    ax.set_xlabel("Oracle trajectory step")
    ax.set_ylabel("SFT examples")
    ax.set_title("Policy distillation examples by step")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def md_table(df: pd.DataFrame, columns: list[str], max_rows: int = 40) -> str:
    view = df[columns].head(max_rows).copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: f"{value:.3f}")
    return view.to_markdown(index=False)


def load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iid", type=Path, default=ROOT / "reports" / "eval" / "learned_iid.json")
    parser.add_argument("--support", type=Path, default=ROOT / "reports" / "eval" / "learned_support.json")
    parser.add_argument("--ceiling", type=Path, default=ROOT / "reports" / "eval" / "learned_ceiling.json")
    parser.add_argument("--policy-manifest", type=Path, default=ROOT / "data" / "policy" / "policy_dataset_manifest.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()

    policy_frames = []
    candidate_frames = []
    raw = {}
    for split, path in [("iid", args.iid), ("support", args.support), ("ceiling", args.ceiling)]:
        if not path.exists():
            continue
        policy, candidate, data = load_result(path, split)
        policy_frames.append(policy)
        candidate_frames.append(candidate)
        raw[split] = data
    if not policy_frames:
        raise SystemExit("No eval results found.")

    policy = pd.concat(policy_frames, ignore_index=True)
    candidate = pd.concat(candidate_frames, ignore_index=True)
    policy_summary = summarize_policy(policy)
    candidate_summary = summarize_candidates(candidate)
    trace_df = learned_trace_rows(policy)
    policy_manifest = load_optional_json(args.policy_manifest)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    policy_summary.to_csv(args.out_dir / "policy_summary.csv", index=False)
    candidate_summary.to_csv(args.out_dir / "candidate_summary.csv", index=False)
    policy.to_json(args.out_dir / "policy_rows.json", orient="records", indent=2)
    candidate.to_json(args.out_dir / "candidate_rows.json", orient="records", indent=2)
    if not trace_df.empty:
        trace_df.to_csv(args.out_dir / "learned_query_trace_summary.csv", index=False)

    fig_dir = args.out_dir / "figures"
    for split in sorted(policy["split"].unique()):
        plot_success_by_budget(policy_summary, split, fig_dir / f"{split}_success_by_budget.png")
    available_budgets = sorted(int(value) for value in policy_summary["budget"].unique())
    if 1 in available_budgets:
        plot_budget_comparison(policy_summary, 1, fig_dir / "split_comparison_budget1.png")
    if 3 in available_budgets:
        plot_budget_comparison(policy_summary, 3, fig_dir / "split_comparison_budget3.png")
    plot_candidate_coverage(candidate_summary, fig_dir / "candidate_coverage.png")
    plot_oracle_gap(policy_summary, 1, fig_dir / "oracle_gap_budget1.png")
    plot_oracle_gap(policy_summary, 3, fig_dir / "oracle_gap_budget3.png")
    plot_learned_rank(trace_df, fig_dir / "learned_choice_rank.png")
    plot_policy_dataset(policy_manifest, fig_dir / "policy_dataset_examples_by_step.png")

    parse_summary = {}
    if not trace_df.empty:
        parse_summary = (
            trace_df.groupby("split", as_index=False)
            .agg(
                learned_queries=("parse_ok", "count"),
                parse_ok_rate=("parse_ok", "mean"),
                fallback_rate=("fallback_used", "mean"),
                avg_chosen_rank=("chosen_option_rank", "mean"),
                avg_chosen_actual_eliminated=("chosen_actual_eliminated", "mean"),
                avg_oracle_actual_eliminated=("oracle_actual_eliminated", "mean"),
            )
            .sort_values("split")
        )
        parse_summary.to_csv(args.out_dir / "learned_parse_summary.csv", index=False)

    lines = [
        "# Qwen3.5-4B Learned Active Trace Policy Report",
        "",
        "## Summary",
        "",
        "This standalone experiment tests whether a Qwen3.5-4B LoRA can learn a low-budget active trace policy for selecting query inputs after typed-sketch candidate synthesis. The learned controller is trained by oracle-action distillation: each training state displays visible examples and candidate-output buckets for possible query inputs, while the target answer is the displayed query option that eliminates the most wrong candidates.",
        "",
        "The query prompt does not reveal the held-out expected output for candidate query inputs. The policy must infer which output bucket is likely correct from the visible examples and candidate-output structure.",
        "",
    ]
    if policy_manifest:
        lines.extend(
            [
                "## Policy Distillation Data",
                "",
                f"Train examples: `{policy_manifest.get('train_summary', {}).get('examples', 0)}`. Eval examples: `{policy_manifest.get('eval_summary', {}).get('examples', 0)}`.",
                "",
                "![Policy dataset examples](figures/policy_dataset_examples_by_step.png)",
                "",
            ]
        )
    lines.extend(
        [
            "## Candidate Coverage",
            "",
            md_table(
                candidate_summary,
                [
                    "split",
                    "records",
                    "visible_selected_hidden_all_pct",
                    "candidate_oracle_hidden_all_pct",
                    "target_program_synthesized_pct",
                    "avg_synthesized_programs",
                    "avg_visible_consistent_candidates",
                ],
            ),
            "",
            "![Candidate coverage](figures/candidate_coverage.png)",
            "",
            "## Policy Results",
            "",
            md_table(
                policy_summary,
                [
                    "split",
                    "policy",
                    "budget",
                    "rows",
                    "hidden_all_pct",
                    "observed_all_pct",
                    "avg_hidden_passes",
                    "avg_queries_used",
                ],
                max_rows=120,
            ),
            "",
            "![Ceiling success by budget](figures/ceiling_success_by_budget.png)",
            "",
            "![Support success by budget](figures/support_success_by_budget.png)",
            "",
            "![IID success by budget](figures/iid_success_by_budget.png)",
            "",
            "![Budget 1 comparison](figures/split_comparison_budget1.png)",
            "",
            "![Budget 3 comparison](figures/split_comparison_budget3.png)",
            "",
            "![Oracle gap budget 1](figures/oracle_gap_budget1.png)",
            "",
            "![Oracle gap budget 3](figures/oracle_gap_budget3.png)",
            "",
        ]
    )
    if isinstance(parse_summary, pd.DataFrame) and not parse_summary.empty:
        lines.extend(
            [
                "## Learned Policy Diagnostics",
                "",
                md_table(
                    parse_summary,
                    [
                        "split",
                        "learned_queries",
                        "parse_ok_rate",
                        "fallback_rate",
                        "avg_chosen_rank",
                        "avg_chosen_actual_eliminated",
                        "avg_oracle_actual_eliminated",
                    ],
                ),
                "",
                "![Learned choice rank](figures/learned_choice_rank.png)",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "The central test is whether learned query selection closes the low-budget gap between the hand-coded max-split heuristic and oracle elimination. A useful learned controller should improve especially at budgets `+1` to `+3`, where one or two high-value traces matter more than broad random coverage.",
            "",
            "## Reproducibility",
            "",
            "- Config: `configs/experiment.json`",
            "- Dataset manifest: `data/dataset_manifest.json`",
            "- Policy dataset manifest: `data/policy/policy_dataset_manifest.json`",
            "- Eval JSON files: `reports/eval/learned_iid.json`, `reports/eval/learned_support.json`, `reports/eval/learned_ceiling.json`",
            "- Policy summary CSV: `reports/policy_summary.csv`",
            "- Candidate summary CSV: `reports/candidate_summary.csv`",
            "- Large artifacts: `/workspace/large_artifacts/qwen35_4b_learned_active_trace_policy`",
        ]
    )
    report_path = args.out_dir / "qwen35_4b_learned_active_trace_policy_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
