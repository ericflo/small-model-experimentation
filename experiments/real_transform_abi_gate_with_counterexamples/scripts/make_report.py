#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def save_coverage_chart(summary: dict[str, Any], out: Path) -> None:
    domains = list(summary["by_domain"])
    raw = [summary["by_domain"][d]["raw_coverage"] for d in domains]
    filtered = [summary["by_domain"][d]["filtered_coverage"] for d in domains]
    labels = ["CSV/ETL", "Date/ID"] if domains == ["csv_etl", "date_id_irregular"] else domains
    x = range(len(domains))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([i - width / 2 for i in x], raw, width, label="Raw examples", color="#4C78A8")
    ax.bar([i + width / 2 for i in x], filtered, width, label="Counterexample filtered", color="#F58518")
    ax.set_xticks(list(x), labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("coverage")
    ax.set_title("Frozen ABI Coverage by Domain")
    ax.legend(frameon=False)
    for i, val in enumerate(raw):
        ax.text(i - width / 2, val + 0.025, pct(val), ha="center", fontsize=9)
    for i, val in enumerate(filtered):
        ax.text(i + width / 2, val + 0.025, pct(val), ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def save_depth_chart(summary: dict[str, Any], out: Path) -> None:
    raw = Counter({int(k): v for k, v in summary["raw_depth_counts"].items()})
    filtered = Counter({int(k): v for k, v in summary["filtered_depth_counts"].items()})
    depths = sorted(set(raw) | set(filtered))
    x = range(len(depths))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([i - width / 2 for i in x], [raw[d] for d in depths], width, label="Raw", color="#54A24B")
    ax.bar([i + width / 2 for i in x], [filtered[d] for d in depths], width, label="Filtered", color="#E45756")
    ax.set_xticks(list(x), [f"depth {d}" for d in depths])
    ax.set_ylabel("covered tasks")
    ax.set_title("Winning Program Depth")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def save_false_pass_chart(summary: dict[str, Any], out: Path) -> None:
    domains = list(summary["by_domain"])
    rates = [summary["by_domain"][d]["visible_hidden_wrong_rate"] for d in domains]
    labels = ["CSV/ETL", "Date/ID"] if domains == ["csv_etl", "date_id_irregular"] else domains

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, rates, color="#B279A2")
    ax.set_ylim(0, max(1.0, max(rates) + 0.1))
    ax.set_ylabel("visible-consistent candidates removed")
    ax.set_title("False-Pass Pressure")
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 0.025, pct(rate), ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def save_smoke_chart(summary: dict[str, Any], out: Path) -> None:
    smoke = summary["smoke"]["overall"]
    vals = [smoke["raw_coverage"], smoke["filtered_coverage"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Raw", "Filtered"], vals, color=["#4C78A8", "#F58518"])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("coverage")
    ax.set_title("Known-Coincidence Smoke Test")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.025, pct(val), ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def markdown_table(rows: list[list[str]]) -> str:
    header = rows[0]
    sep = ["---"] * len(header)
    body = rows[1:]
    return "\n".join("| " + " | ".join(row) + " |" for row in [header, sep, *body])


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    reports = root / "reports"
    figures = reports / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    summary = load_json(reports / "summary.json")
    rows = load_jsonl(root / "data" / "task_records.jsonl")
    smoke = load_jsonl(root / "data" / "smoke_records.jsonl")

    save_coverage_chart(summary, figures / "coverage_by_domain.png")
    save_depth_chart(summary, figures / "depth_counts.png")
    save_false_pass_chart(summary, figures / "false_pass_pressure.png")
    save_smoke_chart(summary, figures / "smoke_filter.png")

    domain_table = [["Domain", "n", "Raw coverage", "Filtered coverage", "Raw removed", "Visible false-pass rate"]]
    for domain, label in [("csv_etl", "CSV/ETL clean pipeline"), ("date_id_irregular", "Date/ID irregular")]:
        metrics = summary["by_domain"][domain]
        domain_table.append(
            [
                label,
                str(metrics["n"]),
                f'{metrics["raw_covered"]}/{metrics["n"]} ({pct(metrics["raw_coverage"])})',
                f'{metrics["filtered_covered"]}/{metrics["n"]} ({pct(metrics["filtered_coverage"])})',
                str(metrics["raw_removed_by_counterexamples"]),
                pct(metrics["visible_hidden_wrong_rate"]),
            ]
        )

    misses = [row for row in rows if not row["filtered_covered"] or row["raw_but_filtered_out"]]
    miss_table = [["Task", "Domain", "Raw", "Filtered", "Raw winning program", "Reason exposed by filter"]]
    for row in misses:
        if row["filtered_covered"]:
            continue
        reason = "No ABI candidate passed raw examples"
        if row["raw_but_filtered_out"]:
            reason = "Raw winner failed adversarial counterexample"
        miss_table.append(
            [
                row["task_id"],
                row["domain"],
                "yes" if row["raw_covered"] else "no",
                "yes" if row["filtered_covered"] else "no",
                json.dumps(row["winning_raw_program"], sort_keys=True) if row["winning_raw_program"] else "-",
                reason,
            ]
        )

    smoke_table = [["Smoke task", "Raw", "Filtered", "Raw winning program"]]
    for row in smoke:
        smoke_table.append(
            [
                row["task_id"],
                "yes" if row["raw_covered"] else "no",
                "yes" if row["filtered_covered"] else "no",
                json.dumps(row["winning_raw_program"], sort_keys=True) if row["winning_raw_program"] else "-",
            ]
        )

    overall = summary["overall"]
    report = f"""# Real Transformation ABI Gate With Counterexamples

## Summary

This no-training gate tested whether a frozen, generic transformation ABI covers two held-out-style deterministic transformation domains, and whether additional counterexamples remove thin-test coincidences.

Main result: the clean pipeline domain stayed fully covered after counterexample filtering, while the irregular date/ID/string domain lost coverage on edge cases. Overall filtered coverage was **{overall["filtered_covered"]}/{overall["n"]} ({pct(overall["filtered_coverage"])})**. The result supports the narrow claim that a generic ABI is useful for pipeline-shaped transformations, but it does not establish broad coverage of irregular transformation work.

The coincidence smoke test worked as intended: both known-wrong raw winners were removed by adversarial examples, so raw coverage alone is not a safe headline metric.

## Charts

![Coverage by domain](figures/coverage_by_domain.png)

![Known-coincidence smoke test](figures/smoke_filter.png)

![Winning program depth](figures/depth_counts.png)

![False-pass pressure](figures/false_pass_pressure.png)

## Method

- The ABI was frozen before evaluating the expanded 40-task suite.
- Coverage was measured twice: raw coverage on visible plus hidden examples, then filtered coverage after extra adversarial examples.
- The suite is curated and self-contained. It is not a public benchmark and should be treated as a gate for whether a larger benchmark build is worth doing.
- Counterexamples can refute a candidate program when expected behavior is available. They do not certify correctness in reference-free deployment.

## Domain Results

{markdown_table(domain_table)}

## Program Depth

- Raw covered depth counts: `{summary["raw_depth_counts"]}`
- Filtered covered depth counts: `{summary["filtered_depth_counts"]}`

Most covered tasks were depth-1 single-primitive programs. Depth-2 coverage appeared mainly in aggregation and parsing transforms. This is useful but limits the composition claim: the gate mainly validates reusable operation selection in clean transformations, not deep program synthesis.

## Counterexample Smoke

{markdown_table(smoke_table)}

The two smoke tasks were designed so a broad generic predicate can pass thin raw examples for the wrong reason. Counterexamples removed all raw smoke winners.

## Filtered Misses

{markdown_table(miss_table)}

The filtered misses are informative: they are irregular edge cases rather than broad pipeline failures. Examples include phone extensions, full month names, and hyphenated title casing. These are exactly the cases where a fixed generic ABI needs either richer primitives, task-specific logic, or a human/stronger-model expansion step.

## Interpretation

The gate gives a positive result for clean CSV/ETL-style transformations: filtered coverage was 100% on the curated clean domain. It gives a narrower result for irregular date/ID/string transformations: raw coverage was high, but counterexamples removed two tasks and one task had no raw ABI solution.

The practical takeaway is to split future work by domain shape:

- Clean row/column/filter/sort/group/normalize pipelines are a plausible target for a compiler-to-ABI system.
- Irregular extraction and formatting tasks need a stronger counterexample suite and a broader ABI before training a compiler would be justified.
- Any future model-training pilot should report depth-1 operation selection separately from depth-2+ composition, because this gate is dominated by single-primitive wins.
"""
    (reports / "report.md").write_text(report, encoding="utf-8")
    print(reports / "report.md")


if __name__ == "__main__":
    main()
