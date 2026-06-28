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


def plot_success_by_budget(summary: pd.DataFrame, split: str, out_path: Path) -> None:
    subset = summary[summary["split"] == split]
    policies = ["visible_prior", "random_extra", "active_max_split", "oracle_elimination"]
    plt.figure(figsize=(8.5, 5.2))
    for policy in policies:
        rows = subset[subset["policy"] == policy].sort_values("budget")
        if rows.empty:
            continue
        label = {
            "visible_prior": "visible only",
            "random_extra": "random extra",
            "active_max_split": "active max-split",
            "oracle_elimination": "oracle elimination",
        }[policy]
        plt.plot(rows["budget"], rows["hidden_all_pct"], marker="o", linewidth=2, label=label)
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


def plot_split_comparison(summary: pd.DataFrame, budget: int, out_path: Path) -> None:
    policies = ["visible_prior", "random_extra", "active_max_split", "oracle_elimination"]
    rows = summary[(summary["budget"].isin([0, budget])) & (summary["policy"].isin(policies))].copy()
    rows = rows[(rows["policy"] == "visible_prior") | (rows["budget"] == budget)]
    rows["label"] = rows["policy"].map(
        {
            "visible_prior": "visible only",
            "random_extra": f"random +{budget}",
            "active_max_split": f"active +{budget}",
            "oracle_elimination": f"oracle +{budget}",
        }
    )
    pivot = rows.pivot_table(index="split", columns="label", values="hidden_all_pct", aggfunc="mean")
    order = [label for label in ["visible only", f"random +{budget}", f"active +{budget}", f"oracle +{budget}"] if label in pivot]
    pivot = pivot[order]
    ax = pivot.plot(kind="bar", figsize=(8.5, 5.2), rot=0)
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
    ax = rows.plot(kind="bar", figsize=(8.5, 5.2), rot=0)
    ax.set_ylabel("Records (%)")
    ax.set_title("Candidate coverage and visible selection")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_family_budget_heatmap(policy: pd.DataFrame, split: str, policy_name: str, out_path: Path) -> None:
    subset = policy[(policy["split"] == split) & (policy["policy"] == policy_name)]
    if subset.empty:
        return
    pivot = subset.pivot_table(index="family", columns="budget", values="hidden_all", aggfunc="mean").sort_index()
    plt.figure(figsize=(9.5, max(4.8, 0.38 * len(pivot))))
    plt.imshow(pivot.values * 100, aspect="auto", cmap="viridis", vmin=0, vmax=100)
    plt.colorbar(label="Hidden full-pass rate (%)")
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xlabel("Extra queried execution cases")
    plt.title(f"{split}: {policy_name} by family")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def md_table(df: pd.DataFrame, columns: list[str], max_rows: int = 30) -> str:
    view = df[columns].head(max_rows).copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: f"{value:.3f}")
    return view.to_markdown(index=False)


def best_active_line(summary: pd.DataFrame, split: str) -> dict[str, Any]:
    rows = summary[(summary["split"] == split) & (summary["policy"] == "active_max_split")].sort_values("hidden_all_rate")
    if rows.empty:
        return {}
    return rows.iloc[-1].to_dict()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iid", type=Path, default=ROOT / "reports" / "eval" / "active_iid.json")
    parser.add_argument("--support", type=Path, default=ROOT / "reports" / "eval" / "active_support.json")
    parser.add_argument("--ceiling", type=Path, default=ROOT / "reports" / "eval" / "active_ceiling.json")
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

    args.out_dir.mkdir(parents=True, exist_ok=True)
    policy_summary.to_csv(args.out_dir / "policy_summary.csv", index=False)
    candidate_summary.to_csv(args.out_dir / "candidate_summary.csv", index=False)
    policy.to_json(args.out_dir / "policy_rows.json", orient="records", indent=2)
    candidate.to_json(args.out_dir / "candidate_rows.json", orient="records", indent=2)

    fig_dir = args.out_dir / "figures"
    for split in sorted(policy["split"].unique()):
        plot_success_by_budget(policy_summary, split, fig_dir / f"{split}_success_by_budget.png")
    plot_split_comparison(policy_summary, 6, fig_dir / "split_comparison_budget6.png")
    plot_split_comparison(policy_summary, 12, fig_dir / "split_comparison_budget12.png")
    plot_candidate_coverage(candidate_summary, fig_dir / "candidate_coverage.png")
    if "ceiling" in set(policy["split"]):
        plot_family_budget_heatmap(policy, "ceiling", "active_max_split", fig_dir / "ceiling_active_family_heatmap.png")

    ceiling_best = best_active_line(policy_summary, "ceiling")
    lines = [
        "# Qwen3.5-4B Active Counterexample Trace Selection Report",
        "",
        "## Summary",
        "",
        "This standalone experiment tests whether a Qwen3.5-4B typed-sketch generator benefits from actively requested execution traces after candidate synthesis. The verifier first completes model-generated typed sketches into executable programs. Selection policies then choose whether to commit from the original visible trace or query additional cases from a held-out per-record pool.",
        "",
    ]
    if ceiling_best:
        lines.extend(
            [
                f"Best active ceiling result: `{ceiling_best['hidden_all_rate']:.3f}` hidden full-pass rate at `+{int(ceiling_best['budget'])}` queried cases.",
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
                max_rows=80,
            ),
            "",
            "![Ceiling success by budget](figures/ceiling_success_by_budget.png)",
            "",
            "![Support success by budget](figures/support_success_by_budget.png)",
            "",
            "![IID success by budget](figures/iid_success_by_budget.png)",
            "",
            "![Split comparison budget 6](figures/split_comparison_budget6.png)",
            "",
            "![Split comparison budget 12](figures/split_comparison_budget12.png)",
            "",
            "![Ceiling active family heatmap](figures/ceiling_active_family_heatmap.png)",
            "",
            "## Interpretation",
            "",
            "The primary question is whether extra traces close the gap between visible-trace selection and the candidate oracle. A large active improvement with a remaining oracle gap means the policy is useful but still leaves selection work. A small active improvement with a large oracle gap means the split heuristic is not finding the right discriminators. A small oracle gap means candidate synthesis is the current limiting factor.",
            "",
            "## Reproducibility",
            "",
            "- Dataset manifest: `data/dataset_manifest.json`",
            "- Config: `configs/experiment.json`",
            "- Eval JSON files: `reports/eval/active_iid.json`, `reports/eval/active_support.json`, `reports/eval/active_ceiling.json`",
            "- Policy summary CSV: `reports/policy_summary.csv`",
            "- Candidate summary CSV: `reports/candidate_summary.csv`",
            "- Large artifacts: `/workspace/large_artifacts/qwen35_4b_active_counterexample_trace_selection`",
        ]
    )
    report_path = args.out_dir / "qwen35_4b_active_counterexample_trace_selection_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
