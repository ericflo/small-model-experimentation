#!/usr/bin/env python
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_result() -> dict[str, Any]:
    return json.loads((ROOT / "reports" / "operator_search_results.json").read_text(encoding="utf-8"))


def pct(successes: int, records: int) -> float:
    return round(100.0 * successes / records, 3) if records else 0.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def flatten_candidate_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        out.append(
            {
                "arm": row["arm"],
                "operator_status": row["operator_status"],
                "operator": row.get("operator", ""),
                "template": row.get("template", ""),
                "records": row["records"],
                "target_raw_pct": pct(row["target_in_raw_candidates"]["successes"], row["target_in_raw_candidates"]["records"]),
                "target_visible_pct": pct(
                    row["target_in_visible_candidates"]["successes"], row["target_in_visible_candidates"]["records"]
                ),
                "oracle_hidden_all_pct": pct(row["candidate_oracle_hidden_all"]["successes"], row["candidate_oracle_hidden_all"]["records"]),
                "selected_hidden_all_pct": pct(row["selected_hidden_all"]["successes"], row["selected_hidden_all"]["records"]),
                "avg_raw_candidate_count": row["avg_raw_candidate_count"],
                "avg_visible_consistent_count": row["avg_visible_consistent_count"],
                "avg_visible_consistent_operator_count": row["avg_visible_consistent_operator_count"],
            }
        )
    return out


def flatten_active(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        out.append(
            {
                "arm": row["arm"],
                "operator_status": row["operator_status"],
                "policy": row["policy"],
                "budget": row["budget"],
                "records": row["records"],
                "selected_hidden_all_pct": pct(row["selected_hidden_all"]["successes"], row["selected_hidden_all"]["records"]),
                "avg_candidate_count": row["avg_candidate_count"],
                "avg_operator_candidate_count": row["avg_operator_candidate_count"],
                "avg_queries_used": row["avg_queries_used"],
            }
        )
    return out


def bar_status(summary: list[dict[str, Any]], metric: str, title: str, path: Path) -> None:
    arms = ["arm1_closed_vocab", "arm0_full_inventory"]
    statuses = ["in_bank", "held_out"]
    lookup = {(row["arm"], row["operator_status"]): row[metric] for row in summary}
    x = range(len(statuses))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for offset, arm in [(-width / 2, arms[0]), (width / 2, arms[1])]:
        ax.bar([i + offset for i in x], [lookup.get((arm, status), 0) for status in statuses], width=width, label=arm)
    ax.set_title(title)
    ax.set_ylabel("% records")
    ax.set_ylim(0, 105)
    ax.set_xticks(list(x))
    ax.set_xticklabels(statuses)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def operator_plot(summary: list[dict[str, Any]], path: Path) -> None:
    rows = [row for row in summary if row["arm"] == "arm0_full_inventory"]
    ops = [row["operator"] for row in rows]
    selected = [row["selected_hidden_all_pct"] for row in rows]
    target = [row["target_raw_pct"] for row in rows]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = range(len(ops))
    ax.bar([i - 0.18 for i in x], target, width=0.36, label="target coverage")
    ax.bar([i + 0.18 for i in x], selected, width=0.36, label="visible selected")
    ax.set_title("Full Inventory Results By Operator")
    ax.set_ylabel("% records")
    ax.set_ylim(0, 105)
    ax.set_xticks(list(x))
    ax.set_xticklabels(ops)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def active_plot(active: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, status in zip(axes, ["in_bank", "held_out"]):
        for arm in ["arm1_closed_vocab", "arm0_full_inventory"]:
            for policy in ["active_max_split", "oracle_elimination"]:
                rows = [row for row in active if row["operator_status"] == status and row["arm"] == arm and row["policy"] == policy]
                if not rows:
                    continue
                rows = sorted(rows, key=lambda row: row["budget"])
                linestyle = "-" if arm == "arm0_full_inventory" else "--"
                ax.plot(
                    [row["budget"] for row in rows],
                    [row["selected_hidden_all_pct"] for row in rows],
                    marker="o",
                    linestyle=linestyle,
                    label=f"{arm}/{policy}",
                )
        ax.set_title(status)
        ax.set_xlabel("query budget")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("selected hidden all-pass (%)")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def search_cost_plot(summary: list[dict[str, Any]], path: Path) -> None:
    arms = ["arm1_closed_vocab", "arm0_full_inventory"]
    statuses = ["in_bank", "held_out"]
    lookup = {(row["arm"], row["operator_status"]): row["avg_raw_candidate_count"] for row in summary}
    x = range(len(statuses))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for offset, arm in [(-width / 2, arms[0]), (width / 2, arms[1])]:
        ax.bar([i + offset for i in x], [lookup.get((arm, status), 0) for status in statuses], width=width, label=arm)
    ax.set_title("Raw Candidates Enumerated Per Record")
    ax.set_ylabel("candidate count")
    ax.set_xticks(list(x))
    ax.set_xticklabels(statuses)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def rows_by_key(rows: list[dict[str, Any]], *keys: str) -> dict[tuple[Any, ...], dict[str, Any]]:
    return {tuple(row[key] for key in keys): row for row in rows}


def report_text(status_rows: list[dict[str, Any]], operator_rows: list[dict[str, Any]], active_rows: list[dict[str, Any]]) -> str:
    status = rows_by_key(status_rows, "arm", "operator_status")
    active = rows_by_key(active_rows, "arm", "operator_status", "policy", "budget")
    held_closed = status[("arm1_closed_vocab", "held_out")]
    held_full = status[("arm0_full_inventory", "held_out")]
    inbank_full = status[("arm0_full_inventory", "in_bank")]
    held_full_b0 = active[("arm0_full_inventory", "held_out", "active_max_split", 0)]
    held_full_b2 = active[("arm0_full_inventory", "held_out", "active_max_split", 2)]
    held_oracle_b1 = active[("arm0_full_inventory", "held_out", "oracle_elimination", 1)]

    lines = [
        "# Qwen3.5-4B Operator Inventory Search Pilot Report",
        "",
        "## Summary",
        "",
        "This standalone no-training pilot tests the search-side ceiling for type-colliding operator identification. Every aggregate candidate has signature `list[int] -> int`; the task is to recover the correct operator from execution cases, not from type.",
        "",
        "Two arms are compared:",
        "",
        "- `arm1_closed_vocab`: closed operator set `sum`, `first`, `last`.",
        "- `arm0_full_inventory`: full operator inventory `sum`, `first`, `last`, `max`, `min`, `prod`, `gcd`.",
        "",
        "## Key Findings",
        "",
        f"- Closed vocabulary held-out target coverage was `{held_closed['target_raw_pct']:.1f}%`; full inventory held-out target coverage was `{held_full['target_raw_pct']:.1f}%`.",
        f"- Full inventory recovered the held-out target in visible-consistent candidates for `{held_full['target_visible_pct']:.1f}%` of records.",
        f"- Full inventory visible selection solved `{held_full['selected_hidden_all_pct']:.1f}%` of held-out records at budget 0, versus `{held_closed['selected_hidden_all_pct']:.1f}%` for closed vocabulary.",
        f"- Active max-split on full inventory held-out records improved from `{held_full_b0['selected_hidden_all_pct']:.1f}%` at budget 0 to `{held_full_b2['selected_hidden_all_pct']:.1f}%` at budget 2.",
        f"- Oracle-elimination querying reached `{held_oracle_b1['selected_hidden_all_pct']:.1f}%` on full inventory held-out records by budget 1.",
        f"- Search cost stayed small in this pilot: full inventory enumerated `{held_full['avg_raw_candidate_count']:.1f}` raw candidates per held-out record.",
        "",
        "Interpretation: the search/bank side can recover the held-out type-colliding operators in this substrate. The missing piece is not program-level search coverage; it is a deployable way for the model to name or shortlist inventory operators as the library scales.",
        "",
        "## Status Summary",
        "",
        "| arm | operator_status | records | target_raw_pct | target_visible_pct | oracle_hidden_all_pct | selected_hidden_all_pct | avg_raw_candidate_count | avg_visible_consistent_operator_count |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in status_rows:
        lines.append(
            f"| {row['arm']} | {row['operator_status']} | {row['records']} | {row['target_raw_pct']:.1f} | {row['target_visible_pct']:.1f} | {row['oracle_hidden_all_pct']:.1f} | {row['selected_hidden_all_pct']:.1f} | {row['avg_raw_candidate_count']:.1f} | {row['avg_visible_consistent_operator_count']:.2f} |"
        )
    lines.extend(
        [
            "",
            "![Target coverage by status](figures/target_coverage_by_status.png)",
            "",
            "![Selected hidden by status](figures/selected_hidden_by_status.png)",
            "",
            "![Search cost](figures/search_cost_by_status.png)",
            "",
            "## Operator Breakdown",
            "",
            "Full inventory by operator:",
            "",
            "| operator | status | records | target_raw_pct | selected_hidden_all_pct | avg_visible_consistent_operator_count |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in [r for r in operator_rows if r["arm"] == "arm0_full_inventory"]:
        lines.append(
            f"| {row['operator']} | {row['operator_status']} | {row['records']} | {row['target_raw_pct']:.1f} | {row['selected_hidden_all_pct']:.1f} | {row['avg_visible_consistent_operator_count']:.2f} |"
        )
    lines.extend(
        [
            "",
            "![Full inventory by operator](figures/full_inventory_by_operator.png)",
            "",
            "## Active Query Diagnostic",
            "",
            "| arm | status | policy | budget | records | selected_hidden_all_pct | avg_operator_candidate_count |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in active_rows:
        lines.append(
            f"| {row['arm']} | {row['operator_status']} | {row['policy']} | {row['budget']} | {row['records']} | {row['selected_hidden_all_pct']:.1f} | {row['avg_operator_candidate_count']:.2f} |"
        )
    lines.extend(
        [
            "",
            "![Active query lift](figures/active_query_lift.png)",
            "",
            "## Decision",
            "",
            "Arm 0 already reaches the coverage ceiling on held-out operators at small search cost. That means the immediate fix lives on the bank/search side for this substrate: grow the operator inventory and apply operator-level active disambiguation. A trained inventory-conditioned sketcher is still useful, but its job should be top-k operator shortlisting for larger libraries, not recovering coverage that search cannot find.",
            "",
            "## Artifacts",
            "",
            "- Dataset: `data/operator_inventory_eval.jsonl`",
            "- Dataset manifest: `data/dataset_manifest.json`",
            "- Full result JSON: `reports/operator_search_results.json`",
            "- CSVs: `reports/status_summary.csv`, `reports/operator_summary.csv`, `reports/template_summary.csv`, `reports/active_summary.csv`",
            "- Large artifacts: `/workspace/large_artifacts/qwen35_4b_operator_inventory_search_pilot`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    result = load_result()
    figures = ROOT / "reports" / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    status_rows = flatten_candidate_summary(result["summary_by_status"])
    operator_rows = flatten_candidate_summary(result["summary_by_operator"])
    template_rows = flatten_candidate_summary(result["summary_by_template"])
    active_rows = flatten_active(result["active_summary_by_status"])

    write_csv(ROOT / "reports" / "status_summary.csv", status_rows)
    write_csv(ROOT / "reports" / "operator_summary.csv", operator_rows)
    write_csv(ROOT / "reports" / "template_summary.csv", template_rows)
    write_csv(ROOT / "reports" / "active_summary.csv", active_rows)

    bar_status(status_rows, "target_raw_pct", "Target Coverage By Operator Status", figures / "target_coverage_by_status.png")
    bar_status(status_rows, "selected_hidden_all_pct", "Visible-Selected Hidden Success", figures / "selected_hidden_by_status.png")
    operator_plot(operator_rows, figures / "full_inventory_by_operator.png")
    active_plot(active_rows, figures / "active_query_lift.png")
    search_cost_plot(status_rows, figures / "search_cost_by_status.png")

    report = report_text(status_rows, operator_rows, active_rows)
    report_path = ROOT / "reports" / "qwen35_4b_operator_inventory_search_pilot_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()

