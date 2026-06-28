#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def save_bar(path: Path, labels: list[str], values: list[float], title: str, ylabel: str, color: str = "#4C78A8", ylim: float = 1.0) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, values, color=color)
    ax.set_ylim(0, ylim)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    for bar, val in zip(bars, values):
        text = pct(val) if val <= 1 else str(int(val))
        ax.text(bar.get_x() + bar.get_width() / 2, min(ylim * 0.97, val + ylim * 0.025), text, ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def table(rows: list[list[str]]) -> str:
    return "\n".join("| " + " | ".join(row) + " |" for row in [rows[0], ["---"] * len(rows[0]), *rows[1:]])


def main() -> None:
    summary = load_json(REPORTS / "summary.json")
    records = load_jsonl(ROOT / "data" / "case_records.jsonl")
    source_rev = Path("/workspace/large_artifacts/external_sources/foofah_benchmarks/.git/refs/heads/master")
    commit = "87c0f407e0881622acb02fb20893ca2506713a9a"

    overall = summary["overall"]
    save_bar(
        FIGURES / "coverage_overall.png",
        ["raw", "held-out", "first visible"],
        [overall["raw_coverage"], overall["heldout_coverage"], overall["first_visible_accuracy"]],
        "Frozen ABI Coverage on Foofah",
        "rate",
    )
    sample_labels = list(summary["by_num_samples"])
    save_bar(
        FIGURES / "coverage_by_num_samples.png",
        sample_labels,
        [summary["by_num_samples"][k]["heldout_coverage"] for k in sample_labels],
        "Held-Out Coverage by Number of Examples",
        "held-out coverage",
        color="#F58518",
    )
    covered_families = [(fam, m["heldout_coverage"], m["heldout_covered"], m["n"]) for fam, m in summary["by_family"].items() if m["heldout_covered"]]
    covered_families = sorted(covered_families, key=lambda x: (-x[1], x[0]))
    save_bar(
        FIGURES / "covered_families.png",
        [fam.replace("_", "\n") for fam, _, _, _ in covered_families],
        [cov for _, cov, _, _ in covered_families],
        "Families Covered by the Frozen ABI",
        "held-out coverage",
        color="#54A24B",
    )
    family_rows = [["Family", "Held-out covered", "Held-out coverage", "Mean candidates"]]
    for fam, cov, covered, n in covered_families:
        metrics = summary["by_family"][fam]
        family_rows.append([fam, f"{covered}/{n}", pct(cov), f'{metrics["candidate_count_mean"]:.1f}'])
    zero_count = sum(1 for m in summary["by_family"].values() if m["heldout_covered"] == 0)
    raw_only = [r for r in records if r["raw_covered"] and not r["heldout_covered"]]
    raw_only_rows = [["File", "Family", "Samples", "Candidate count", "First program"]]
    for r in raw_only[:8]:
        raw_only_rows.append([r["file"], r["family"], str(r["num_samples"]), str(r["candidate_count"]), json.dumps(r["first_program"], sort_keys=True)])

    report = f"""# External Foofah Transformation ABI Gate

## Summary

This standalone gate evaluated a frozen compact table-transformation ABI on the Foofah benchmark format. Each case provides an example pair (`InputTable` -> `OutputTable`) and a separate held-out check (`TestingTable` -> `TestAnswer`).

Main result: the frozen ABI covered **{overall["heldout_covered"]}/{overall["n"]} held-out cases ({pct(overall["heldout_coverage"])})**. Raw example coverage was **{overall["raw_covered"]}/{overall["n"]} ({pct(overall["raw_coverage"])})**. First-visible selection solved **{overall["first_visible_heldout"]}/{overall["n"]} ({pct(overall["first_visible_accuracy"])})**, nearly all held-out-covered cases, so selection is not the bottleneck.

The gate therefore fails at ABI expressivity on this external benchmark. Model training or constrained scoring would be hard to interpret because the correct program is absent for 82% of cases.

## Source

- Repository: `https://github.com/markjin1990/foofah_benchmarks`
- Local clone: `/workspace/large_artifacts/external_sources/foofah_benchmarks`
- Commit used: `{commit}`
- Files evaluated: {overall["n"]}

## Charts

![Overall coverage](figures/coverage_overall.png)

![Coverage by sample count](figures/coverage_by_num_samples.png)

![Covered families](figures/covered_families.png)

## Results

| Metric | Value |
| --- | ---: |
| Raw example coverage | {overall["raw_covered"]}/{overall["n"]} ({pct(overall["raw_coverage"])}) |
| Held-out coverage | {overall["heldout_covered"]}/{overall["n"]} ({pct(overall["heldout_coverage"])}) |
| First-visible held-out accuracy | {overall["first_visible_heldout"]}/{overall["n"]} ({pct(overall["first_visible_accuracy"])}) |
| Mean candidate count | {overall["candidate_count_mean"]:.1f} |
| Winner depth counts | `{summary["winner_depth_counts"]}` |
| Winner family counts | `{summary["winner_family_counts"]}` |

## Coverage by Example Count

| Num examples | n | Raw coverage | Held-out coverage | First-visible accuracy |
| ---: | ---: | ---: | ---: | ---: |
""" + "\n".join(
        f"| {k} | {m['n']} | {m['raw_covered']}/{m['n']} ({pct(m['raw_coverage'])}) | {m['heldout_covered']}/{m['n']} ({pct(m['heldout_coverage'])}) | {m['first_visible_heldout']}/{m['n']} ({pct(m['first_visible_accuracy'])}) |"
        for k, m in summary["by_num_samples"].items()
    ) + f"""

Held-out coverage stays essentially flat as examples increase from 1 to 5, which points to missing primitives rather than ambiguity from too few examples.

## Covered Families

{table(family_rows)}

Families with zero held-out coverage: {zero_count}/{len(summary["by_family"])}.

## Raw-Only False Coverage Examples

These cases had a candidate that fit the example pair but failed the held-out table. They are the external benchmark analogue of a counterexample filter removing coincidence fits.

{table(raw_only_rows)}

## Interpretation

This is a negative but useful breadth test. A compact frozen ABI handles a narrow slice of fold/unpivot and regex-extraction cases, but it does not cover most external Foofah transformations. Because first-visible selection nearly matches oracle held-out coverage, model-side selection is not the next bottleneck here.

The next useful experiment is not compiler training on this ABI. It is strict library expansion under a source-independent protocol: add generic table primitives from documentation or a training-only split, freeze them, then rerun this exact held-out gate. Adding task-specific primitives after inspecting the failed files would invalidate the gate.
"""
    (REPORTS / "report.md").write_text(report, encoding="utf-8")
    print(REPORTS / "report.md")


if __name__ == "__main__":
    main()
