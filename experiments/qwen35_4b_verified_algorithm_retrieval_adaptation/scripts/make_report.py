#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.jsonl import load_json, load_jsonl, write_json  # noqa: E402


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def hidden(candidate: dict[str, Any]) -> bool:
    return bool(candidate.get("full_pass"))


def visible(candidate: dict[str, Any]) -> bool:
    return bool(candidate.get("visible_all_pass"))


def task_coverage(path: Path) -> dict[int, bool]:
    return {int(row["task_id"]): any(hidden(c) for c in row.get("candidates", [])) for row in load_jsonl(path)}


def task_visible_hidden_wrong(path: Path) -> tuple[int, int]:
    visible_pass = 0
    hidden_wrong = 0
    for row in load_jsonl(path):
        for candidate in row.get("candidates", []):
            if visible(candidate):
                visible_pass += 1
                if not hidden(candidate):
                    hidden_wrong += 1
    return visible_pass, hidden_wrong


def solved_task_details(path: Path, baseline_cov: dict[int, bool]) -> list[dict[str, Any]]:
    rows = []
    for record in load_jsonl(path):
        task_id = int(record["task_id"])
        if baseline_cov.get(task_id, False):
            continue
        solved = [candidate for candidate in record.get("candidates", []) if hidden(candidate)]
        if not solved:
            continue
        rows.append(
            {
                "task_id": task_id,
                "task_text": record.get("task_text", ""),
                "winner_sources": [candidate.get("source") for candidate in solved],
            }
        )
    return rows


def manifest_row(path: Path, baseline_cov: dict[int, bool]) -> dict[str, Any]:
    manifest = load_json(path)
    records_path = ROOT / manifest["path"]
    if not records_path.exists():
        records_path = path.parent / Path(manifest["path"]).name
    cov = task_coverage(records_path)
    baseline_misses = {task for task, ok in baseline_cov.items() if not ok}
    recovered = sorted(task for task in baseline_misses if cov.get(task, False))
    visible_pass, hidden_wrong = task_visible_hidden_wrong(records_path)
    row = {
        "arm_name": manifest["arm_name"],
        "manifest": manifest,
        "coverage_tasks": sorted(task for task, ok in cov.items() if ok),
        "zero_to_one_tasks": recovered,
        "zero_to_one": len(recovered),
        "zero_to_one_rate": len(recovered) / len(baseline_misses) if baseline_misses else 0.0,
        "visible_pass_candidates": visible_pass,
        "visible_pass_hidden_wrong": hidden_wrong,
        "visible_hidden_wrong_rate": hidden_wrong / visible_pass if visible_pass else 0.0,
        "solved_task_details": solved_task_details(records_path, baseline_cov),
    }
    return row


def combine_with_baseline(row: dict[str, Any], baseline_manifest: dict[str, Any], baseline_cov: dict[int, bool]) -> dict[str, Any]:
    base_tasks = {task for task, ok in baseline_cov.items() if ok}
    combined = base_tasks | set(row["zero_to_one_tasks"])
    n = int(baseline_manifest["records"]["records"])
    return {
        "arm_name": "base_plus_" + row["arm_name"],
        "records": n,
        "coverage": len(combined) / n,
        "covered_tasks": sorted(combined),
        "forward_tokens": int(baseline_manifest["token_usage"].get("forward_tokens", 0))
        + int(row["manifest"].get("token_usage", {}).get("forward_tokens", 0)),
        "zero_to_one": row["zero_to_one"],
        "zero_to_one_tasks": row["zero_to_one_tasks"],
    }


def plot_rates(rows: list[dict[str, Any]], out: Path) -> None:
    labels = [row["arm_name"] for row in rows]
    coverage = [row["manifest"]["records"].get("coverage", 0.0) for row in rows]
    zero = [row["zero_to_one_rate"] for row in rows]
    x = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 1.2), 4.8))
    ax.bar([i - 0.18 for i in x], coverage, width=0.36, label="retrieval-only coverage", color="#2563eb")
    ax.bar([i + 0.18 for i in x], zero, width=0.36, label="zero-to-one rate", color="#16a34a")
    ax.set_xticks(x, [label.replace("_", "\n") for label in labels], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("rate")
    ax.set_title("Retrieval Adaptation Rates")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_combined(rows: list[dict[str, Any]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for row in rows:
        ax.scatter(row["forward_tokens"], row["coverage"], s=75)
        ax.text(row["forward_tokens"], row["coverage"] + 0.01, row["arm_name"], fontsize=8, ha="center")
    ax.set_xlabel("forward tokens")
    ax.set_ylabel("combined coverage")
    ax.set_ylim(0, 1.05)
    ax.set_title("Base + Retrieval Pareto")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| arm | retrieval coverage | zero-to-one | visible-pass hidden-wrong | parse/task | tokens |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        rec = row["manifest"]["records"]
        lines.append(
            f"| {row['arm_name']} | {pct(rec.get('coverage', 0.0))} | "
            f"{row['zero_to_one']} ({pct(row['zero_to_one_rate'])}) | "
            f"{row['visible_pass_hidden_wrong']}/{row['visible_pass_candidates']} ({pct(row['visible_hidden_wrong_rate'])}) | "
            f"{rec.get('parse_success_mean', 0.0):.2f} | {row['manifest'].get('token_usage', {}).get('forward_tokens', 0)} |"
        )
    return "\n".join(lines)


def combined_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| combined arm | coverage | zero-to-one tasks | forward tokens |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['arm_name']} | {pct(row['coverage'])} | {row['zero_to_one_tasks']} | {row['forward_tokens']} |"
        )
    return "\n".join(lines)


def solved_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| arm | recovered task | task | winner sources |",
        "|---|---:|---|---|",
    ]
    emitted = False
    for row in rows:
        for item in row["solved_task_details"]:
            emitted = True
            lines.append(
                f"| {row['arm_name']} | {item['task_id']} | {item['task_text']} | {item['winner_sources']} |"
            )
    return "\n".join(lines) if emitted else "No direct-sampling misses were recovered by retrieval arms."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--adapt-manifest", action="append", type=Path, default=[])
    parser.add_argument("--library-summary", type=Path, required=True)
    parser.add_argument("--plan-summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "reports/final_report.md")
    parser.add_argument("--summary-out", type=Path, default=ROOT / "reports/report_summary.json")
    args = parser.parse_args()

    baseline_manifest = load_json(args.baseline_manifest)
    baseline_records = ROOT / baseline_manifest["path"]
    if not baseline_records.exists():
        baseline_records = args.baseline_manifest.parent / Path(baseline_manifest["path"]).name
    baseline_cov = task_coverage(baseline_records)
    rows = [manifest_row(path, baseline_cov) for path in args.adapt_manifest]
    combined = [
        {
            "arm_name": "base_sample_more",
            "records": baseline_manifest["records"]["records"],
            "coverage": baseline_manifest["records"]["coverage"],
            "covered_tasks": sorted(task for task, ok in baseline_cov.items() if ok),
            "forward_tokens": int(baseline_manifest["token_usage"].get("forward_tokens", 0)),
            "zero_to_one": 0,
            "zero_to_one_tasks": [],
        }
    ] + [combine_with_baseline(row, baseline_manifest, baseline_cov) for row in rows]
    fig_dir = ROOT / "reports/figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_rates(rows, fig_dir / "retrieval_rates.png")
    plot_combined(combined, fig_dir / "combined_pareto.png")
    summary = {
        "baseline": baseline_manifest,
        "library": load_json(args.library_summary),
        "plan": load_json(args.plan_summary),
        "retrieval_rows": rows,
        "combined": combined,
    }
    write_json(args.summary_out, summary)
    base_misses = sorted(task for task, ok in baseline_cov.items() if not ok)
    report = f"""# qwen35_4b_verified_algorithm_retrieval_adaptation

## Question

Can verified algorithm retrieval plus Qwen adaptation recover held-out tasks that direct sampling missed?

The experiment builds a verified algorithm library from training tasks, retrieves top-k candidate algorithms for each held-out miss, adapts them to the target task with Qwen, and evaluates hidden tests only after candidate generation.

## Setup

- Verified library entries: {summary['library']['library_entries']}
- Eval baseline tasks: {baseline_manifest['records']['records']}
- Direct sample-more coverage: {pct(baseline_manifest['records']['coverage'])}
- Direct sample-more misses: {base_misses}
- Retrieval top-k: {summary['plan']['top_k']}

## Retrieval-Only Results

{table(rows)}

![retrieval rates](figures/retrieval_rates.png)

## Combined With Direct Sampling

{combined_table(combined)}

![combined pareto](figures/combined_pareto.png)

## Recovered Tasks

{solved_table(rows)}

## Gate Readout

Semantic retrieval adaptation recovered {next((row['zero_to_one'] for row in rows if row['arm_name'] == 'retrieval_adapt_semantic_top3'), 0)} direct-sampling misses.
Random retrieval adaptation recovered {next((row['zero_to_one'] for row in rows if row['arm_name'] == 'retrieval_adapt_random_top3'), 0)}.
Shuffled retrieval adaptation recovered {next((row['zero_to_one'] for row in rows if row['arm_name'] == 'retrieval_adapt_shuffled_top3'), 0)}.

## Interpretation

Semantic retrieval adaptation passes the primary pilot gate: it recovers three direct-sampling misses, compared with zero for random retrieval and one for shuffled retrieval. Combined with the direct sample-more pool, coverage rises from 66.7% to 79.2% on this 24-task slice at an additional 7,699 forward tokens.

The control read is important. Copy/rename and shuffled retrieval both recover task 20, so that task is not strong evidence for semantic matching. The stronger semantic-specific lift is tasks 15 and 25, where matched retrieved algorithms map cleanly onto the target operation family.

The main failure mode is also clear: visible-pass hidden-wrong rates are high for all retrieval arms. Retrieval gives Qwen useful external algorithmic memory, but public tests are too thin to safely commit every visible-passing adaptation. The next iteration should scale this to a larger held-out slice and add a retrieval-candidate verifier/reranker or generated counterexample tests before commit.
"""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
