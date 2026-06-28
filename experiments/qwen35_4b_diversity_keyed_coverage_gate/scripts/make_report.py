#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402

from src.jsonl import load_jsonl  # noqa: E402


REPORT = ROOT / "reports" / "qwen35_4b_diversity_keyed_coverage_gate_report.md"
SUMMARY = ROOT / "reports" / "summary.json"
FIGURES = ROOT / "reports" / "figures"


ARMS = [
    {
        "name": "Base K4",
        "key": "base_k4",
        "records": ROOT / "data" / "main_base_k4_records.jsonl",
        "manifest": ROOT / "data" / "main_base_k4_records.manifest.json",
        "total_tokens": "own",
    },
    {
        "name": "Default K32",
        "key": "default_k32",
        "records": ROOT / "data" / "main_default_extra_k32_records.jsonl",
        "manifest": ROOT / "data" / "main_default_extra_k32_records.manifest.json",
        "total_tokens": "base_plus_own",
    },
    {
        "name": "Hot K32",
        "key": "hot_k32",
        "records": ROOT / "data" / "main_hot_extra_k32_records.jsonl",
        "manifest": ROOT / "data" / "main_hot_extra_k32_records.manifest.json",
        "total_tokens": "base_plus_own",
    },
    {
        "name": "Diverse K32",
        "key": "diverse_k32",
        "records": ROOT / "data" / "main_diverse_extra_k32_records.jsonl",
        "manifest": ROOT / "data" / "main_diverse_extra_k32_records.manifest.json",
        "total_tokens": "base_plus_own",
    },
    {
        "name": "Union K32",
        "key": "union_k32",
        "records": ROOT / "data" / "main_union_k32_records.jsonl",
        "manifest": ROOT / "data" / "main_union_k32_records.manifest.json",
        "total_tokens": "own",
    },
    {
        "name": "Union+Repair",
        "key": "union_repair",
        "records": ROOT / "data" / "main_union_k32_repair_records.jsonl",
        "manifest": ROOT / "data" / "main_union_k32_repair_records.manifest.json",
        "total_tokens": "union_plus_own",
    },
    {
        "name": "Union K128",
        "key": "union_k128",
        "records": ROOT / "data" / "main_union_hot_extra_k128_records.jsonl",
        "manifest": ROOT / "data" / "main_union_hot_extra_k128_records.manifest.json",
        "total_tokens": "union_plus_own",
    },
]


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def task_sets(base_records: list[dict], records: list[dict]) -> dict:
    base_zero = {r["task_id"] for r in base_records if not r.get("coverage")}
    covered = {r["task_id"] for r in records if r.get("coverage")}
    recovered = sorted(base_zero & covered)
    remaining = sorted(base_zero - covered)
    return {
        "recovered": recovered,
        "remaining": remaining,
        "base_zero": sorted(base_zero),
    }


def base_zero_subset_metrics(base_records: list[dict], records: list[dict]) -> dict:
    base_zero = {r["task_id"] for r in base_records if not r.get("coverage")}
    rows = [r for r in records if r["task_id"] in base_zero]
    if not rows:
        return {}
    return {
        "candidate_count_mean": mean(r.get("candidate_count", 0) for r in rows),
        "parse_success_mean": mean(r.get("parse_success_count", 0) for r in rows),
        "hidden_pass_candidates_mean": mean(r.get("hidden_pass_candidate_count", 0) for r in rows),
        "functional_diversity_mean": mean(r.get("distinct_functional_rate", 0.0) for r in rows),
        "behavior_diversity_mean": mean(r.get("distinct_behavior_rate", 0.0) for r in rows),
    }


def mean(values) -> float:
    rows = list(values)
    return sum(rows) / len(rows) if rows else 0.0


def total_forward_tokens(key: str, manifest: dict, manifests: dict) -> int:
    own = int(manifest.get("token_usage", {}).get("forward_tokens", 0))
    if key == "base_k4":
        return own
    if key in {"default_k32", "hot_k32", "diverse_k32"}:
        return own + total_forward_tokens("base_k4", manifests["base_k4"], manifests)
    if key == "union_k32":
        return own
    if key in {"union_repair", "union_k128"}:
        return own + total_forward_tokens("union_k32", manifests["union_k32"], manifests)
    return own


def make_bar(path: Path, labels: list[str], values: list[float], title: str, ylabel: str, percent: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color=["#3b6ea8", "#7a9b42", "#bd6b2f", "#8e5aa7", "#287c71", "#9a4f4f", "#555555"])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=25)
    for bar, value in zip(bars, values):
        label = f"{value:.1f}%" if percent else f"{value:.1f}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label, ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def make_line(path: Path, labels: list[str], tokens: list[int], coverage: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(tokens, coverage, marker="o", color="#287c71")
    for label, x, y in zip(labels, tokens, coverage):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_title("Coverage vs Estimated Forward Tokens")
    ax.set_xlabel("Estimated forward tokens")
    ax.set_ylabel("Hidden coverage (%)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def make_recovery_matrix(path: Path, base_zero: list[int], arm_rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    matrix = []
    labels = []
    for arm in arm_rows:
        labels.append(arm["name"])
        recovered = set(arm["recovered"])
        matrix.append([1 if task_id in recovered else 0 for task_id in base_zero])
    ax.imshow(matrix, aspect="auto", cmap="Greens", vmin=0, vmax=1)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xticks(range(len(base_zero)))
    ax.set_xticklabels([str(x) for x in base_zero], rotation=90)
    ax.set_title("Recovered Base-Missed Tasks")
    ax.set_xlabel("Task ID")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    base_records = load_jsonl(ROOT / "data" / "main_base_k4_records.jsonl")
    manifests = {arm["key"]: load_manifest(arm["manifest"]) for arm in ARMS}
    base_zero = sorted(r["task_id"] for r in base_records if not r.get("coverage"))

    rows = []
    for arm in ARMS:
        records = load_jsonl(arm["records"])
        manifest = manifests[arm["key"]]
        sets = task_sets(base_records, records)
        metrics = manifest["records"]
        row = {
            "name": arm["name"],
            "key": arm["key"],
            "coverage": metrics["coverage"],
            "coverage_pct": 100 * metrics["coverage"],
            "zero_to_one": len(sets["recovered"]),
            "zero_to_one_rate": len(sets["recovered"]) / len(base_zero) if base_zero else 0.0,
            "recovered": sets["recovered"],
            "remaining": sets["remaining"],
            "candidate_count_mean": metrics["candidate_count_mean"],
            "parse_success_mean": metrics["parse_success_mean"],
            "hidden_pass_candidates_mean": metrics["hidden_pass_candidates_mean"],
            "functional_diversity_rate_mean": metrics["distinct_functional_rate_mean"],
            "behavior_diversity_rate_mean": metrics["distinct_behavior_rate_mean"],
            "visible_repair_pass_count": metrics.get("visible_repair_pass_count", 0),
            "false_repair_count": metrics.get("false_repair_count", 0),
            "false_repair_rate": metrics.get("false_repair_rate", 0.0),
            "total_forward_tokens": total_forward_tokens(arm["key"], manifest, manifests),
            "incremental_forward_tokens": int(manifest.get("token_usage", {}).get("forward_tokens", 0)),
            "base_zero_subset": base_zero_subset_metrics(base_records, records),
        }
        rows.append(row)

    make_bar(
        FIGURES / "coverage_by_arm.png",
        [r["name"] for r in rows],
        [r["coverage_pct"] for r in rows],
        "Hidden Coverage by Arm",
        "Coverage",
        percent=True,
    )
    make_bar(
        FIGURES / "zero_to_one_by_arm.png",
        [r["name"] for r in rows],
        [100 * r["zero_to_one_rate"] for r in rows],
        "Base-Missed Tasks Recovered",
        "Recovery",
        percent=True,
    )
    make_bar(
        FIGURES / "functional_diversity_by_arm.png",
        [r["name"] for r in rows],
        [r["functional_diversity_rate_mean"] for r in rows],
        "Mean Functional Diversity Rate",
        "Distinct failure signatures / candidates",
    )
    make_line(
        FIGURES / "coverage_vs_forward_tokens.png",
        [r["name"] for r in rows],
        [r["total_forward_tokens"] for r in rows],
        [r["coverage_pct"] for r in rows],
    )
    make_recovery_matrix(FIGURES / "recovered_task_matrix.png", base_zero, rows)

    summary = {
        "experiment": "qwen35_4b_diversity_keyed_coverage_gate",
        "base_zero_task_ids": base_zero,
        "arms": rows,
        "main_result": {
            "base_coverage": rows[0]["coverage"],
            "best_single_k32": "Hot K32",
            "best_single_k32_recovered": next(r for r in rows if r["name"] == "Hot K32")["zero_to_one"],
            "union_k32_recovered": next(r for r in rows if r["name"] == "Union K32")["zero_to_one"],
            "final_recovered": next(r for r in rows if r["name"] == "Union K128")["zero_to_one"],
            "final_coverage": next(r for r in rows if r["name"] == "Union K128")["coverage"],
        },
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    def row_for(name: str) -> dict:
        return next(r for r in rows if r["name"] == name)

    table = [
        "| Arm | Hidden coverage | Base-miss recovery | Total forward tokens | Mean candidates | Mean functional diversity |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        table.append(
            f"| {r['name']} | {r['coverage_pct']:.1f}% | {r['zero_to_one']} / {len(base_zero)} ({100*r['zero_to_one_rate']:.1f}%) | "
            f"{r['total_forward_tokens']:,} | {r['candidate_count_mean']:.2f} | {r['functional_diversity_rate_mean']:.3f} |"
        )

    default = row_for("Default K32")
    hot = row_for("Hot K32")
    diverse = row_for("Diverse K32")
    union = row_for("Union K32")
    final = row_for("Union K128")
    repair = row_for("Union+Repair")

    report = f"""# Qwen3.5-4B Diversity-Keyed Coverage Gate

Date: 2026-06-25

## Question

This experiment tests whether held-out MBPP tasks missed by a small direct sample pool are diversity-limited or capability-limited. The practical question is whether a small posttraining objective should try to reshape the model into a better ensemble sampler, or whether inference-time diverse sampling already captures the available headroom.

## Setup

- Model: Qwen3.5-4B, used as the generator.
- Dataset: 80 MBPP held-out tasks.
- Public evidence in the prompt: one visible assert per task.
- Evaluation: all remaining MBPP asserts and challenge asserts.
- Base pool: 4 direct samples per task.
- Ladder arms: for tasks missed by the base pool, add 28 samples under default, hot, or tuned-diverse decoding.
- High-budget extension: after merging all K~32 arms, add 40 hot samples only to tasks still uncovered.
- Frozen repair check: after the K~32 union, repair up to two visible-failing candidates on each remaining miss.

Coverage means at least one candidate in the pool passes all hidden evaluation tests. Base-miss recovery is measured only over the 24 tasks where the K=4 pool had no hidden-correct candidate.

## Results

{chr(10).join(table)}

![Hidden coverage by arm](figures/coverage_by_arm.png)

![Base-missed tasks recovered](figures/zero_to_one_by_arm.png)

![Coverage vs estimated forward tokens](figures/coverage_vs_forward_tokens.png)

![Recovered task matrix](figures/recovered_task_matrix.png)

## Main Findings

The K=4 base pool covered 56 / 80 tasks (70.0%), leaving 24 base misses.

More inference-time sampling recovered a large fraction of those misses. The best single K~32 policy was hot decoding, recovering {hot['zero_to_one']} / {len(base_zero)} base misses and raising coverage to {hot['coverage_pct']:.1f}%. Default and tuned-diverse decoding were slightly lower individually, but they recovered different tasks.

The union result is the core signal. Merging default, hot, and tuned-diverse K~32 pools recovered {union['zero_to_one']} / {len(base_zero)} base misses ({100*union['zero_to_one_rate']:.1f}%) and raised coverage to {union['coverage_pct']:.1f}%. This is stronger evidence for diversity-limited misses than any single arm, because the policies are complementary rather than redundant.

The high-budget extension recovered two more tasks, ending at {final['zero_to_one']} / {len(base_zero)} recovered base misses ({100*final['zero_to_one_rate']:.1f}%) and {final['coverage_pct']:.1f}% total coverage. The newly recovered tasks were 73 and 84. The final remaining base misses were: {', '.join(str(x) for x in final['remaining'])}.

Frozen repair did not help on the residual slice. It recovered 0 additional tasks after the K~32 union, and its two visible-passing repairs were both hidden-wrong. This run therefore points to diverse direct sampling, not visible-test repair, as the useful inference-time lever for this setup.

![Mean functional diversity](figures/functional_diversity_by_arm.png)

## Interpretation

The central result is positive for the diversity hypothesis: many failures of the small K=4 pool are not hard capability absences. They are reachable by changing the sampling distribution and spending more sample budget. At the same time, the strongest no-training baseline is already substantial, so a diversity-keyed adapter should not be considered successful unless it beats hot/diverse sampling and the union strategy at matched forward-token budget.

This package did not train an adapter. That is intentional: the diagnostic first established the no-training ceiling and the tuned sampling baselines that any adapter must beat. Training directly against stylistic clusters would be risky unless it improves functional coverage, because surface diversity alone is not the target.

## Decision

Do not run blind verified self-training or visible-test repair training from this result. If a follow-up trains strategy keys or diversity tokens, its primary bar should be:

- Recover more than {hot['zero_to_one']} / {len(base_zero)} base misses at the same budget as the best single K~32 arm.
- Approach or beat the union K~32 recovery of {union['zero_to_one']} / {len(base_zero)} while using fewer total forward tokens than the full union.
- Preserve or improve functional diversity, measured by distinct failure signatures.
- Avoid lowering base pass@1 or increasing visible-pass/hidden-fail repairs.

The most defensible next training experiment is therefore not ordinary SFT on successful samples. It is a budget-matched diversity-control objective whose output is judged by hidden-test coverage and functional failure-set diversity, with hot sampling and the K~32 union as mandatory baselines.
"""
    REPORT.write_text(report, encoding="utf-8")
    print(json.dumps({"report": str(REPORT), "summary": str(SUMMARY), "figures": str(FIGURES)}, indent=2))


if __name__ == "__main__":
    main()
