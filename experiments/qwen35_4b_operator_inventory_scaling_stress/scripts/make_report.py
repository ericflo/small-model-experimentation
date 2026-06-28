#!/usr/bin/env python
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "reports" / "operator_scaling_results.json"
REPORT_PATH = ROOT / "reports" / "qwen35_4b_operator_inventory_scaling_stress_report.md"
FIG_DIR = ROOT / "reports" / "figures"


def pct(metric: dict[str, Any]) -> float:
    return 100.0 * float(metric["rate"])


def flatten_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        flat = {}
        for key, value in row.items():
            if isinstance(value, dict) and {"successes", "records", "rate"} <= set(value):
                flat[f"{key}_successes"] = value["successes"]
                flat[f"{key}_records"] = value["records"]
                flat[f"{key}_pct"] = round(100.0 * value["rate"], 3)
            else:
                flat[key] = value
        out.append(flat)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rows_for(result: dict[str, Any], key: str, hole_count: int) -> list[dict[str, Any]]:
    return [row for row in result[key] if row["hole_count"] == hole_count]


def plot_cost(result: dict[str, Any]) -> None:
    rows = result["summary_by_library_and_depth"]
    plt.figure(figsize=(8, 5))
    for hole_count, label in [(1, "one operator hole"), (2, "two operator holes")]:
        subset = rows_for(result, "summary_by_library_and_depth", hole_count)
        xs = [row["library_size"] for row in subset]
        ys = [row["avg_raw_candidate_count"] for row in subset]
        plt.plot(xs, ys, marker="o", label=label)
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("operator library size")
    plt.ylabel("raw candidates per record")
    plt.title("Enumeration Cost")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "raw_candidate_cost.png", dpi=180)
    plt.close()


def plot_coverage(result: dict[str, Any]) -> None:
    rows = result["summary_by_library_and_depth"]
    plt.figure(figsize=(8, 5))
    for hole_count, label in [(1, "one hole target coverage"), (2, "two hole target coverage")]:
        subset = rows_for(result, "summary_by_library_and_depth", hole_count)
        xs = [row["library_size"] for row in subset]
        ys = [pct(row["target_in_visible_candidates"]) for row in subset]
        plt.plot(xs, ys, marker="o", label=label)
    for hole_count, label in [(1, "one hole selected"), (2, "two hole selected")]:
        subset = rows_for(result, "summary_by_library_and_depth", hole_count)
        xs = [row["library_size"] for row in subset]
        ys = [pct(row["selected_hidden_all"]) for row in subset]
        plt.plot(xs, ys, marker="s", linestyle="--", label=label)
    plt.xscale("log", base=2)
    plt.ylim(-2, 102)
    plt.xlabel("operator library size")
    plt.ylabel("percent")
    plt.title("Coverage And Selection")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "coverage_and_selection.png", dpi=180)
    plt.close()


def plot_ambiguity(result: dict[str, Any]) -> None:
    rows = result["summary_by_library_and_depth"]
    plt.figure(figsize=(8, 5))
    for hole_count, label in [(1, "one operator hole"), (2, "two operator holes")]:
        subset = rows_for(result, "summary_by_library_and_depth", hole_count)
        xs = [row["library_size"] for row in subset]
        ys = [row["avg_visible_consistent_count"] for row in subset]
        plt.plot(xs, ys, marker="o", label=label)
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xlabel("operator library size")
    plt.ylabel("visible-consistent candidates")
    plt.title("Residual Ambiguity After Six Cases")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "visible_ambiguity.png", dpi=180)
    plt.close()


def plot_active(result: dict[str, Any]) -> None:
    rows = [
        row
        for row in result["active_summary_by_library_and_depth"]
        if row["policy"] == "active_max_split" and row["hole_count"] == 2
    ]
    plt.figure(figsize=(8, 5))
    for budget in sorted({row["budget"] for row in rows}):
        subset = [row for row in rows if row["budget"] == budget]
        xs = [row["library_size"] for row in subset]
        ys = [pct(row["selected_hidden_all"]) for row in subset]
        plt.plot(xs, ys, marker="o", label=f"budget {budget}")
    plt.xscale("log", base=2)
    plt.ylim(-2, 102)
    plt.xlabel("operator library size")
    plt.ylabel("selected hidden-all percent")
    plt.title("Active Query Lift For Two-Hole Programs")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "active_query_lift_two_hole.png", dpi=180)
    plt.close()


def plot_prefix(result: dict[str, Any]) -> None:
    rows = [row for row in result["prefix_summary_by_library_and_depth"] if row["hole_count"] == 2]
    plt.figure(figsize=(8, 5))
    for budget in sorted({row["budget"] for row in rows}):
        subset = [row for row in rows if row["budget"] == budget]
        xs = [row["library_size"] for row in subset]
        ys = [pct(row["target_in_prefix"]) for row in subset]
        plt.plot(xs, ys, marker="o", label=f"{budget} candidates")
    plt.xscale("log", base=2)
    plt.ylim(-2, 102)
    plt.xlabel("operator library size")
    plt.ylabel("target in fixed prefix percent")
    plt.title("Fixed Candidate-Budget Coverage For Two-Hole Search")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fixed_budget_prefix_coverage.png", dpi=180)
    plt.close()


def table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for label, _key in columns) + " |"
    sep = "| " + " | ".join("---" for _label, _key in columns) + " |"
    body = []
    for row in rows:
        values = []
        for _label, key in columns:
            value = row[key]
            if isinstance(value, float):
                values.append(f"{value:.2f}")
            else:
                values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, sep, *body])


def key_status_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in result["summary_by_library_and_depth"]:
        if row["library_size"] in {8, 64, 512}:
            rows.append(
                {
                    "library_size": row["library_size"],
                    "holes": row["hole_count"],
                    "records": row["records"],
                    "raw_candidates": int(row["avg_raw_candidate_count"]),
                    "target_visible_pct": round(pct(row["target_in_visible_candidates"]), 1),
                    "oracle_pct": round(pct(row["candidate_oracle_hidden_all"]), 1),
                    "selected_pct": round(pct(row["selected_hidden_all"]), 1),
                    "visible_candidates": round(row["avg_visible_consistent_count"], 2),
                    "target_rank_p90": round(row["target_rank_p90"], 1),
                }
            )
    return rows


def active_key_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in result["active_summary_by_library_and_depth"]:
        if row["library_size"] in {64, 512} and row["hole_count"] == 2 and row["budget"] in {0, 1, 2, 3}:
            rows.append(
                {
                    "library_size": row["library_size"],
                    "policy": row["policy"],
                    "budget": row["budget"],
                    "selected_pct": round(pct(row["selected_hidden_all"]), 1),
                    "candidate_count": round(row["avg_candidate_count"], 2),
                }
            )
    return rows


def template_key_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in result["summary_by_library_template"]:
        if row["library_size"] in {128, 512} and row["hole_count"] == 2:
            rows.append(
                {
                    "library_size": row["library_size"],
                    "template": row["template"],
                    "selected_pct": round(pct(row["selected_hidden_all"]), 1),
                    "visible_candidates": round(row["avg_visible_consistent_count"], 2),
                    "target_rank_p90": round(row["target_rank_p90"], 1),
                }
            )
    return rows


def report_text(result: dict[str, Any]) -> str:
    status = key_status_rows(result)
    active = active_key_rows(result)
    templates = template_key_rows(result)
    max_pair = next(row for row in result["summary_by_library_and_depth"] if row["library_size"] == 512 and row["hole_count"] == 2)
    max_one = next(row for row in result["summary_by_library_and_depth"] if row["library_size"] == 512 and row["hole_count"] == 1)
    prefix_512 = [
        row
        for row in result["prefix_summary_by_library_and_depth"]
        if row["library_size"] == 512 and row["hole_count"] == 2
    ]
    prefix_line = ", ".join(f"{row['budget']}: {pct(row['target_in_prefix']):.1f}%" for row in prefix_512)
    return f"""# Qwen3.5-4B Operator Inventory Scaling Stress Report

## Summary

This standalone no-training experiment scales a same-signature `list[int] -> int` operator inventory from 8 to 512 operators. It measures one-hole templates, where exhaustive search scales as `N`, and two-hole templates, where exhaustive search scales as `N^2`.

At 512 operators, one-hole exhaustive search enumerates `{int(max_one['avg_raw_candidate_count'])}` candidates per record, while two-hole exhaustive search enumerates `{int(max_pair['avg_raw_candidate_count'])}` candidates per record. Target coverage remains `{pct(max_pair['target_in_visible_candidates']):.1f}%` for two-hole programs because the target is still in the library and visible cases retain it, but zero-query selection drops to `{pct(max_pair['selected_hidden_all']):.1f}%` as visible-consistent ambiguity grows.

The practical bottleneck is now compute budget and residual ambiguity, not target reachability. For two-hole programs at 512 operators, fixed prefix coverage is {prefix_line}; a deployable top-k shortlister must preserve target coverage while avoiding full `N^2` enumeration.

## Key Scaling Rows

{table(status, [
    ('library', 'library_size'),
    ('holes', 'holes'),
    ('records', 'records'),
    ('raw candidates', 'raw_candidates'),
    ('target visible %', 'target_visible_pct'),
    ('oracle %', 'oracle_pct'),
    ('selected %', 'selected_pct'),
    ('visible candidates', 'visible_candidates'),
    ('target rank p90', 'target_rank_p90'),
])}

![Raw candidate cost](figures/raw_candidate_cost.png)

![Coverage and selection](figures/coverage_and_selection.png)

![Visible ambiguity](figures/visible_ambiguity.png)

## Template Breakdown

The hard case is the low-information comparison template. At 512 operators, `pair_compare_gate` leaves far more visible-consistent candidates than `pair_affine_mod`, and zero-query selection falls accordingly.

{table(templates, [
    ('library', 'library_size'),
    ('template', 'template'),
    ('selected %', 'selected_pct'),
    ('visible candidates', 'visible_candidates'),
    ('target rank p90', 'target_rank_p90'),
])}

## Active Query Diagnostic

For two-hole programs, active querying reduces ambiguity but does not remove the search-cost issue. It helps after the full candidate set has already been generated and filtered. The oracle-elimination curve is a ceiling on query choice quality; max-split is the deployable heuristic.

{table(active, [
    ('library', 'library_size'),
    ('policy', 'policy'),
    ('budget', 'budget'),
    ('selected %', 'selected_pct'),
    ('candidate count', 'candidate_count'),
])}

![Active query lift two hole](figures/active_query_lift_two_hole.png)

## Fixed Candidate-Budget Diagnostic

The fixed-prefix diagnostic is deliberately simple: it asks whether a small canonical candidate budget would contain the target before semantic shortlisting. The answer becomes no as two-hole search grows, which quantifies the budget a learned shortlister has to beat.

![Fixed budget prefix coverage](figures/fixed_budget_prefix_coverage.png)

## Decision

The target remains reachable under exhaustive inventory search through 512 operators, including two-operator compositions. The reason to train an inventory-conditioned Qwen3.5-4B sketcher is now sharply defined: top-k shortlisting for large two-hole libraries. The training target should be coverage at fixed candidate budgets, especially `1024`, `4096`, and `16384`, with active querying retained as a post-shortlist disambiguator.

## Artifacts

- Dataset: `data/operator_scaling_eval.jsonl`
- Dataset manifest: `data/dataset_manifest.json`
- Full result JSON: `reports/operator_scaling_results.json`
- CSVs: `reports/library_depth_summary.csv`, `reports/library_template_summary.csv`, `reports/target_bucket_summary.csv`, `reports/prefix_summary.csv`, `reports/active_summary.csv`
- Large artifacts: `/workspace/large_artifacts/qwen35_4b_operator_inventory_scaling_stress`
"""


def main() -> None:
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(ROOT / "reports" / "library_depth_summary.csv", flatten_metric_rows(result["summary_by_library_and_depth"]))
    write_csv(ROOT / "reports" / "library_template_summary.csv", flatten_metric_rows(result["summary_by_library_template"]))
    write_csv(ROOT / "reports" / "target_bucket_summary.csv", flatten_metric_rows(result["summary_by_target_bucket"]))
    write_csv(ROOT / "reports" / "prefix_summary.csv", flatten_metric_rows(result["prefix_summary_by_library_and_depth"]))
    write_csv(ROOT / "reports" / "active_summary.csv", flatten_metric_rows(result["active_summary_by_library_and_depth"]))
    plot_cost(result)
    plot_coverage(result)
    plot_ambiguity(result)
    plot_active(result)
    plot_prefix(result)
    REPORT_PATH.write_text(report_text(result), encoding="utf-8")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
