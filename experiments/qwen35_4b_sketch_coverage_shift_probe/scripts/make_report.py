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


SKETCH_ORDER = ["auto", "manual", "erased"]
SHIFT_ORDER = ["control_in_bank", "name_shift", "primitive_shift"]
POLICY_ORDER = ["active_max_split", "oracle_elimination"]


def load_result() -> dict[str, Any]:
    return json.loads((ROOT / "reports" / "coverage_probe.json").read_text(encoding="utf-8"))


def pct(value: float) -> float:
    return round(100.0 * value, 3)


def group_rows(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    out = []
    for key_values, subset in sorted(groups.items()):
        data = dict(zip(keys, key_values))
        records = len(subset)
        data.update(
            {
                "records": records,
                "target_coverage_pct": pct(sum(row["target_program_synthesized"] for row in subset) / records),
                "oracle_hidden_all_pct": pct(sum(row["candidate_oracle_hidden_all"] for row in subset) / records),
                "selected_hidden_all_pct": pct(sum(row["selected_hidden_all"] for row in subset) / records),
                "selected_visible_all_pct": pct(sum(row["selected_visible_all"] for row in subset) / records),
                "avg_program_count": round(sum(row["program_count"] for row in subset) / records, 3),
                "avg_visible_consistent_count": round(sum(row["visible_consistent_count"] for row in subset) / records, 3),
            }
        )
        out.append(data)
    return out


def group_active(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    out = []
    for key_values, subset in sorted(groups.items()):
        data = dict(zip(keys, key_values))
        records = len(subset)
        data.update(
            {
                "records": records,
                "selected_hidden_all_pct": pct(sum(row["selected_hidden_all"] for row in subset) / records),
                "avg_candidate_count": round(sum(row["candidate_count"] for row in subset) / records, 3),
                "avg_queries_used": round(sum(row["queries_used"] for row in subset) / records, 3),
            }
        )
        out.append(data)
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


def bar_by_shift_sketch(rows: list[dict[str, Any]], metric: str, title: str, path: Path) -> None:
    cap_rows = [row for row in rows if row["hole_options"] == 28]
    lookup = {(row["shift_type"], row["sketch_mode"]): row[metric] for row in cap_rows}
    x = range(len(SHIFT_ORDER))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5.5))
    offsets = {"auto": -width, "manual": 0, "erased": width}
    for sketch_mode in SKETCH_ORDER:
        values = [lookup.get((shift, sketch_mode), 0.0) for shift in SHIFT_ORDER]
        ax.bar([i + offsets[sketch_mode] for i in x], values, width=width, label=sketch_mode)
    ax.set_title(title)
    ax.set_ylabel("records solved (%)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(SHIFT_ORDER, rotation=15, ha="right")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def cap_sensitivity(rows: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    for ax, shift in zip(axes, SHIFT_ORDER):
        subset = [row for row in rows if row["shift_type"] == shift]
        for sketch_mode in SKETCH_ORDER:
            xs = sorted({row["hole_options"] for row in subset})
            ys = []
            for cap in xs:
                matching = [row for row in subset if row["sketch_mode"] == sketch_mode and row["hole_options"] == cap]
                ys.append(sum(row["target_coverage_pct"] * row["records"] for row in matching) / sum(row["records"] for row in matching))
            ax.plot(xs, ys, marker="o", label=sketch_mode)
        ax.set_title(shift)
        ax.set_xlabel("hole option cap")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("target coverage (%)")
    axes[-1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def family_heatmap(rows: list[dict[str, Any]], path: Path) -> None:
    cap_rows = [row for row in rows if row["hole_options"] == 28]
    families = sorted({row["family"] for row in cap_rows})
    labels = [(family, sketch) for family in families for sketch in SKETCH_ORDER]
    values = []
    for family, sketch in labels:
        matching = [row for row in cap_rows if row["family"] == family and row["sketch_mode"] == sketch]
        values.append(matching[0]["target_coverage_pct"] if matching else 0.0)
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.imshow([values], aspect="auto", vmin=0, vmax=100, cmap="RdYlGn")
    ax.set_yticks([])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([f"{family}\n{sketch}" for family, sketch in labels], rotation=90, fontsize=7)
    ax.set_title("Target Coverage By Family And Sketch Condition, Cap 28")
    for index, value in enumerate(values):
        ax.text(index, 0, f"{value:.0f}", ha="center", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def active_plot(rows: list[dict[str, Any]], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    for ax, shift in zip(axes, SHIFT_ORDER):
        for sketch_mode in ["manual", "auto"]:
            for policy in POLICY_ORDER:
                subset = [
                    row
                    for row in rows
                    if row["shift_type"] == shift and row["sketch_mode"] == sketch_mode and row["policy"] == policy
                ]
                if not subset:
                    continue
                xs = sorted({row["budget"] for row in subset})
                ys = []
                for budget in xs:
                    matching = [row for row in subset if row["budget"] == budget]
                    ys.append(sum(row["selected_hidden_all_pct"] * row["records"] for row in matching) / sum(row["records"] for row in matching))
                linestyle = "-" if sketch_mode == "manual" else "--"
                ax.plot(xs, ys, marker="o", linestyle=linestyle, label=f"{sketch_mode}/{policy}")
        ax.set_title(shift)
        ax.set_xlabel("query budget")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("selected hidden all-pass (%)")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fmt_pct(rows: list[dict[str, Any]], shift: str, sketch: str, metric: str, cap: int = 28) -> str:
    matching = [row for row in rows if row["shift_type"] == shift and row["sketch_mode"] == sketch and row["hole_options"] == cap]
    if not matching:
        return "n/a"
    total = sum(row["records"] for row in matching)
    value = sum(row[metric] * row["records"] for row in matching) / total
    return f"{value:.1f}%"


def make_markdown(coverage_by_shift: list[dict[str, Any]], active_by_shift: list[dict[str, Any]]) -> str:
    primitive_auto = fmt_pct(coverage_by_shift, "primitive_shift", "auto", "target_coverage_pct")
    primitive_manual = fmt_pct(coverage_by_shift, "primitive_shift", "manual", "target_coverage_pct")
    primitive_erased = fmt_pct(coverage_by_shift, "primitive_shift", "erased", "target_coverage_pct")
    control_auto = fmt_pct(coverage_by_shift, "control_in_bank", "auto", "target_coverage_pct")
    name_erased = fmt_pct(coverage_by_shift, "name_shift", "erased", "target_coverage_pct")
    name_manual = fmt_pct(coverage_by_shift, "name_shift", "manual", "target_coverage_pct")

    lines = [
        "# Qwen3.5-4B Sketch Coverage Shift Probe Report",
        "",
        "## Summary",
        "",
        "This standalone experiment tests whether typed-sketch verified completion keeps the correct executable program in its bounded candidate set when the task substrate shifts. No new adapter is trained. The executor can score shifted primitives, while the completion bank is held fixed for the falsification pass.",
        "",
        "Each record is evaluated under three sketch conditions:",
        "",
        "- `auto`: generated by the typed target-sketch function.",
        "- `manual`: hand-typed sketch with the intended operator shape and typed holes.",
        "- `erased`: low-information sketch that preserves only output format or branch labels.",
        "",
        "## Key Findings",
        "",
        f"- Control target coverage at cap 28 with `auto` sketches was `{control_auto}`.",
        f"- Primitive-shift target coverage at cap 28 was `{primitive_auto}` for `auto`, `{primitive_manual}` for `manual`, and `{primitive_erased}` for `erased`.",
        f"- Name-shift target coverage at cap 28 was `{name_manual}` for `manual`, but only `{name_erased}` for `erased`.",
        "- The decisive failure mode is operator support in the sketch: if the sketch explicitly names and types the shifted operator, completion often works; if the operator must be discovered by the bank, coverage collapses.",
        "- Active querying helps only after coverage exists. It cannot recover a target program that was never synthesized.",
        "",
        "## Coverage Tables",
        "",
        "Primary cap-28 shift summary:",
        "",
        "| shift_type | sketch_mode | records | target_coverage_pct | oracle_hidden_all_pct | selected_hidden_all_pct | avg_program_count |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in [r for r in coverage_by_shift if r["hole_options"] == 28]:
        lines.append(
            f"| {row['shift_type']} | {row['sketch_mode']} | {row['records']} | {row['target_coverage_pct']:.1f} | {row['oracle_hidden_all_pct']:.1f} | {row['selected_hidden_all_pct']:.1f} | {row['avg_program_count']:.1f} |"
        )
    lines.extend(
        [
            "",
            "![Target coverage by shift](figures/target_coverage_by_shift.png)",
            "",
            "![Oracle hidden coverage by shift](figures/oracle_hidden_by_shift.png)",
            "",
            "![Selected hidden coverage by shift](figures/selected_hidden_by_shift.png)",
            "",
            "![Cap sensitivity](figures/cap_sensitivity.png)",
            "",
            "![Family heatmap](figures/family_target_coverage_heatmap.png)",
            "",
            "## Active Query Diagnostic",
            "",
            "Active selection is reported at hole option cap 28. These rows test disambiguation among synthesized candidates; they are not a substitute for coverage.",
            "",
            "| shift_type | sketch_mode | policy | budget | records | selected_hidden_all_pct |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in active_by_shift:
        if row["hole_options"] == 28 and row["sketch_mode"] in {"auto", "manual"}:
            lines.append(
                f"| {row['shift_type']} | {row['sketch_mode']} | {row['policy']} | {row['budget']} | {row['records']} | {row['selected_hidden_all_pct']:.1f} |"
            )
    lines.extend(
        [
            "",
            "![Active query diagnostic](figures/active_query_diagnostic.png)",
            "",
            "## Artifacts",
            "",
            "- Dataset: `data/shifted_coverage_eval.jsonl`",
            "- Dataset manifest: `data/dataset_manifest.json`",
            "- Full result JSON: `reports/coverage_probe.json`",
            "- Coverage CSVs: `reports/coverage_by_shift.csv`, `reports/coverage_by_family.csv`",
            "- Active CSV: `reports/active_by_shift.csv`",
            "- Large artifacts: `/workspace/large_artifacts/qwen35_4b_sketch_coverage_shift_probe`",
            "",
            "## Conclusion",
            "",
            "The coverage assumption does not survive all task shifts. The strongest result is conditional: typed completion can still work on shifted primitives when the sketch names the shifted operator and uses correctly typed holes. The weak result is equally important: erased sketches and automatically mistyped sketches often lose the target completely. The next scaling step should therefore widen and train sketch/operator coverage before investing in a stronger selector.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    result = load_result()
    figures = ROOT / "reports" / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    coverage_rows = result["coverage_rows"]
    active_rows = result["active_rows"]
    coverage_by_shift = group_rows(coverage_rows, ["shift_type", "sketch_mode", "hole_options"])
    coverage_by_family = group_rows(coverage_rows, ["shift_type", "family", "sketch_mode", "hole_options"])
    active_by_shift = group_active(active_rows, ["shift_type", "sketch_mode", "policy", "budget", "hole_options"])

    write_csv(ROOT / "reports" / "coverage_by_shift.csv", coverage_by_shift)
    write_csv(ROOT / "reports" / "coverage_by_family.csv", coverage_by_family)
    write_csv(ROOT / "reports" / "active_by_shift.csv", active_by_shift)

    bar_by_shift_sketch(coverage_by_shift, "target_coverage_pct", "Target Program Coverage, Cap 28", figures / "target_coverage_by_shift.png")
    bar_by_shift_sketch(coverage_by_shift, "oracle_hidden_all_pct", "Candidate Oracle Hidden Success, Cap 28", figures / "oracle_hidden_by_shift.png")
    bar_by_shift_sketch(coverage_by_shift, "selected_hidden_all_pct", "Visible-Selected Hidden Success, Cap 28", figures / "selected_hidden_by_shift.png")
    cap_sensitivity(coverage_by_shift, figures / "cap_sensitivity.png")
    family_heatmap(coverage_by_family, figures / "family_target_coverage_heatmap.png")
    active_plot(active_by_shift, figures / "active_query_diagnostic.png")

    report = make_markdown(coverage_by_shift, active_by_shift)
    report_path = ROOT / "reports" / "qwen35_4b_sketch_coverage_shift_probe_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
