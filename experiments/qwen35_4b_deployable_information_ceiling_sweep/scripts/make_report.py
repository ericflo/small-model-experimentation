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

POLICY_LABELS = {
    "greedy_uniform_split": "Greedy uniform split",
    "target_aware_oracle": "Target-aware oracle",
}


def load_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
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
        records.groupby(["policy", "visible_total", "library_size", "template", "budget"], as_index=False)
        .agg(
            records=("record_id", "count"),
            selected_hidden_all=("selected_hidden_all", "mean"),
            selected_exact_pair=("selected_exact_pair", "mean"),
            target_reachable=("target_reachable", "mean"),
            candidate_count_mean=("candidate_count", "mean"),
            hidden_equivalent_candidates_mean=("hidden_equivalent_candidates", "mean"),
        )
        .sort_values(["policy", "visible_total", "library_size", "template", "budget"])
    )
    by_template = (
        records.groupby(["policy", "visible_total", "template", "budget"], as_index=False)
        .agg(
            records=("record_id", "count"),
            selected_hidden_all=("selected_hidden_all", "mean"),
            selected_exact_pair=("selected_exact_pair", "mean"),
            candidate_count_mean=("candidate_count", "mean"),
            hidden_equivalent_candidates_mean=("hidden_equivalent_candidates", "mean"),
        )
        .sort_values(["visible_total", "template", "budget", "policy"])
    )
    overall = (
        records.groupby(["policy", "visible_total", "budget"], as_index=False)
        .agg(
            records=("record_id", "count"),
            selected_hidden_all=("selected_hidden_all", "mean"),
            selected_exact_pair=("selected_exact_pair", "mean"),
            candidate_count_mean=("candidate_count", "mean"),
            hidden_equivalent_candidates_mean=("hidden_equivalent_candidates", "mean"),
        )
        .sort_values(["visible_total", "budget", "policy"])
    )
    by_cell.to_csv(ROOT / "reports" / "summary_by_cell.csv", index=False)
    by_template.to_csv(ROOT / "reports" / "summary_by_template.csv", index=False)
    overall.to_csv(ROOT / "reports" / "summary_overall.csv", index=False)
    if not actions.empty:
        actions.to_csv(ROOT / "reports" / "action_records.csv", index=False)
    return by_cell, by_template, overall


def plot_budget_curves(by_template: pd.DataFrame) -> None:
    fig_dir = ROOT / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for template in sorted(by_template["template"].unique()):
        plt.figure(figsize=(9, 5.2))
        subset = by_template[by_template["template"] == template]
        for visible_total in sorted(subset["visible_total"].unique()):
            for policy in ["greedy_uniform_split", "target_aware_oracle"]:
                row = subset[(subset["visible_total"] == visible_total) & (subset["policy"] == policy)].sort_values("budget")
                if row.empty:
                    continue
                style = "-" if policy == "greedy_uniform_split" else "--"
                plt.plot(
                    row["budget"],
                    100 * row["selected_hidden_all"],
                    linestyle=style,
                    marker="o",
                    linewidth=2,
                    label=f"{POLICY_LABELS[policy]}, visible={visible_total}",
                )
        plt.xlabel("Active probe budget")
        plt.ylabel("Hidden-all selected accuracy (%)")
        plt.title(f"Information Sweep: {template}")
        plt.ylim(0, 105)
        plt.grid(alpha=0.25)
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(fig_dir / f"budget_curve_{template}.png", dpi=180)
        plt.close()


def plot_gap_by_budget(by_template: pd.DataFrame) -> None:
    fig_dir = ROOT / "reports" / "figures"
    rows: list[dict[str, Any]] = []
    for (visible_total, template, budget), group in by_template.groupby(["visible_total", "template", "budget"]):
        vals = {row["policy"]: float(row["selected_hidden_all"]) for _, row in group.iterrows()}
        if "greedy_uniform_split" in vals and "target_aware_oracle" in vals:
            rows.append(
                {
                    "visible_total": visible_total,
                    "template": template,
                    "budget": budget,
                    "gap": vals["target_aware_oracle"] - vals["greedy_uniform_split"],
                }
            )
    df = pd.DataFrame(rows)
    for template in sorted(df["template"].unique()):
        plt.figure(figsize=(8, 4.8))
        subset = df[df["template"] == template]
        for visible_total in sorted(subset["visible_total"].unique()):
            row = subset[subset["visible_total"] == visible_total].sort_values("budget")
            plt.plot(row["budget"], 100 * row["gap"], marker="o", linewidth=2, label=f"visible={visible_total}")
        plt.xlabel("Active probe budget")
        plt.ylabel("Oracle minus greedy gap (points)")
        plt.title(f"Non-Deployable Headroom Gap: {template}")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / f"oracle_gap_{template}.png", dpi=180)
        plt.close()


def value(df: pd.DataFrame, policy: str, visible_total: int, template: str, budget: int, column: str) -> float:
    row = df[
        (df["policy"] == policy)
        & (df["visible_total"] == visible_total)
        & (df["template"] == template)
        & (df["budget"] == budget)
    ]
    if row.empty:
        return float("nan")
    return float(row.iloc[0][column])


def make_markdown(by_template: pd.DataFrame) -> str:
    visible_values = sorted(int(v) for v in by_template["visible_total"].unique())
    max_budget = int(by_template["budget"].max())
    rows_b3: list[str] = []
    rows_b10: list[str] = []
    for template in ["pair_affine_mod", "pair_compare_gate"]:
        for visible_total in visible_values:
            for budget, out_rows in [(3, rows_b3), (max_budget, rows_b10)]:
                greedy = value(by_template, "greedy_uniform_split", visible_total, template, budget, "selected_hidden_all")
                oracle = value(by_template, "target_aware_oracle", visible_total, template, budget, "selected_hidden_all")
                cand = value(by_template, "greedy_uniform_split", visible_total, template, budget, "candidate_count_mean")
                hidden_eq = value(
                    by_template,
                    "greedy_uniform_split",
                    visible_total,
                    template,
                    budget,
                    "hidden_equivalent_candidates_mean",
                )
                out_rows.append(
                    f"| {template} | {visible_total} | {pct(greedy)} | {pct(oracle)} | {100 * (oracle - greedy):.1f} | {cand:.1f} | {hidden_eq:.1f} |"
                )

    cg_greedy_b3 = value(by_template, "greedy_uniform_split", 4, "pair_compare_gate", 3, "selected_hidden_all")
    cg_oracle_b3 = value(by_template, "target_aware_oracle", 4, "pair_compare_gate", 3, "selected_hidden_all")
    cg_greedy_b10 = value(by_template, "greedy_uniform_split", 4, "pair_compare_gate", max_budget, "selected_hidden_all")
    cg_oracle_b10 = value(by_template, "target_aware_oracle", 4, "pair_compare_gate", max_budget, "selected_hidden_all")
    cg_visible16_b3 = value(by_template, "greedy_uniform_split", 16, "pair_compare_gate", 3, "selected_hidden_all")
    cg_visible16_oracle_b3 = value(by_template, "target_aware_oracle", 16, "pair_compare_gate", 3, "selected_hidden_all")

    lines = [
        "# Qwen3.5-4B Deployable Information Ceiling Sweep",
        "",
        "## Objective",
        "",
        "This standalone diagnostic measures whether the hard low-information regime is limited by deployable information or by a trainable probe-selection policy. It uses no model training. The deployable policy is greedy max expected information gain under a uniform posterior over surviving verifier candidates. The oracle policy is target-aware and is included only as non-deployable headroom.",
        "",
        "## Key Checks",
        "",
        f"- With four visible observations and budget 3, compare-gate greedy accuracy is {pct(cg_greedy_b3)} versus {pct(cg_oracle_b3)} for the target-aware oracle.",
        f"- Keeping four visible observations but raising active budget to {max_budget}, compare-gate greedy accuracy is {pct(cg_greedy_b10)} versus {pct(cg_oracle_b10)} for the target-aware oracle.",
        f"- Raising initial visible observations to sixteen while keeping budget 3 gives compare-gate greedy accuracy {pct(cg_visible16_b3)} versus {pct(cg_visible16_oracle_b3)} for the target-aware oracle.",
        "",
        "## Interpretation",
        "",
        "The greedy policy is the deployable one-step Bayesian experiment-design rule for a uniform posterior over all candidates consistent with observed executions. If additional budget or additional visible observations lift greedy performance, the bottleneck is information volume. If the target-aware oracle remains far above greedy, that gap should be read as target-knowledge headroom rather than directly recoverable deployable headroom.",
        "",
        "## Budget-3 Summary",
        "",
        "| Template | Visible observations | Greedy | Target-aware oracle | Gap points | Greedy candidates left | Greedy hidden-equivalent left |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *rows_b3,
        "",
        f"## Budget-{max_budget} Summary",
        "",
        "| Template | Visible observations | Greedy | Target-aware oracle | Gap points | Greedy candidates left | Greedy hidden-equivalent left |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *rows_b10,
        "",
        "## Figures",
        "",
        "- `reports/figures/budget_curve_pair_affine_mod.png`",
        "- `reports/figures/budget_curve_pair_compare_gate.png`",
        "- `reports/figures/oracle_gap_pair_affine_mod.png`",
        "- `reports/figures/oracle_gap_pair_compare_gate.png`",
        "",
        "## Reproduction",
        "",
        "```bash",
        "python scripts/eval_information_sweep.py --max-budget 10 --visible-extra 0 4 8 12",
        "python scripts/make_report.py",
        "```",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    records, actions = load_rows()
    if records.empty:
        raise RuntimeError("no eval JSON files found")
    by_cell, by_template, overall = write_tables(records, actions)
    plot_budget_curves(by_template)
    plot_gap_by_budget(by_template)
    report = make_markdown(by_template)
    report_path = ROOT / "reports" / "qwen35_4b_deployable_information_ceiling_sweep_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps({"report": str(report_path), "rows": len(records)}, indent=2))


if __name__ == "__main__":
    main()
