#!/usr/bin/env python
from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.jsonl import load_jsonl  # noqa: E402


REPORTS = ROOT / "reports"
DATA = ROOT / "data"
EVAL = REPORTS / "eval"
FIGURES = REPORTS / "figures"


DISPLAY: dict[str, str] = {
    "base_train": "base train",
    "selftrain_verified_r1_train": "verified r1 train",
    "base_heldout": "base held-out",
    "selftrain_verified_r1_heldout": "verified r1 held-out",
    "selftrain_unverified_heldout": "unverified held-out",
    "oracle_sft_heldout": "oracle SFT held-out",
    "sample_more_matched_compute_heldout": "sample-more held-out",
    "base_transfer": "base transfer",
    "selftrain_verified_r1_transfer": "verified r1 transfer",
    "smoke_base_heldout": "smoke base held-out",
    "smoke_selftrain_verified_r1_heldout": "smoke verified r1 held-out",
    "smoke_base_train": "smoke base train",
}

MAIN_ORDER = [
    "base_train",
    "selftrain_verified_r1_train",
    "base_heldout",
    "selftrain_verified_r1_heldout",
    "selftrain_unverified_heldout",
    "oracle_sft_heldout",
    "sample_more_matched_compute_heldout",
    "base_transfer",
    "selftrain_verified_r1_transfer",
]


@dataclass
class RecordSummary:
    key: str
    label: str
    split: str
    arm: str
    records: int
    coverage: float
    visible_coverage: float
    candidate_count_mean: float
    parse_success_mean: float
    visible_candidates_mean: float
    hidden_pass_candidates_mean: float
    distinct_program_rate: float
    mean_code_chars: float


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def fmt(value: float) -> str:
    return f"{value:.2f}"


def record_key(path: Path) -> str:
    name = path.name
    if name.endswith("_records.jsonl"):
        return name[: -len("_records.jsonl")]
    return path.stem


def split_arm_from_key(key: str) -> tuple[str, str]:
    if key.endswith("_train"):
        return "train", key[: -len("_train")]
    if key.endswith("_heldout"):
        return "heldout", key[: -len("_heldout")]
    if key.endswith("_transfer"):
        return "transfer", key[: -len("_transfer")]
    return "other", key


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_records(path: Path) -> RecordSummary:
    records = load_jsonl(path)
    key = record_key(path)
    split, arm = split_arm_from_key(key)
    n = len(records)
    coverage = mean([1.0 if record.get("coverage") else 0.0 for record in records])
    visible_coverage = mean([1.0 if record.get("visible_coverage") else 0.0 for record in records])
    candidate_count_mean = mean([float(record.get("candidate_count", len(record.get("candidates", [])))) for record in records])
    parse_success_mean = mean([float(record.get("parse_success_count", 0)) for record in records])
    visible_candidates_mean = mean([float(record.get("visible_candidate_count", 0)) for record in records])
    hidden_pass_candidates_mean = mean(
        [sum(1 for candidate in record.get("candidates", []) if candidate.get("full_pass")) for record in records]
    )
    distinct_rates: list[float] = []
    code_lens: list[float] = []
    for record in records:
        candidates = record.get("candidates", [])
        codes = [candidate.get("code", "") for candidate in candidates if candidate.get("code")]
        if candidates:
            distinct_rates.append(len(set(codes)) / len(candidates))
        code_lens.extend([len(code) for code in codes])
    return RecordSummary(
        key=key,
        label=DISPLAY.get(key, key.replace("_", " ")),
        split=split,
        arm=arm,
        records=n,
        coverage=coverage,
        visible_coverage=visible_coverage,
        candidate_count_mean=candidate_count_mean,
        parse_success_mean=parse_success_mean,
        visible_candidates_mean=visible_candidates_mean,
        hidden_pass_candidates_mean=hidden_pass_candidates_mean,
        distinct_program_rate=mean(distinct_rates),
        mean_code_chars=mean(code_lens),
    )


def load_record_summaries() -> dict[str, RecordSummary]:
    summaries: dict[str, RecordSummary] = {}
    for path in sorted(DATA.glob("*_records.jsonl")):
        summaries[record_key(path)] = summarize_records(path)
    return summaries


def load_commit_summaries() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(EVAL.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        records_path = Path(payload.get("records_path", ""))
        key = record_key(records_path) if records_path.name else path.stem
        for item in payload.get("summary", []):
            rows.append(
                {
                    "record_key": key,
                    "label": DISPLAY.get(key, key.replace("_", " ")),
                    "policy": item["policy"],
                    "budget": int(item["budget"]),
                    "records": int(item["records"]),
                    "coverage": float(item["coverage"]),
                    "visible_coverage": float(item["visible_coverage"]),
                    "selected_hidden_all": float(item["selected_hidden_all"]),
                    "coverage_captured": float(item["coverage_captured"]),
                    "visible_candidates_mean": float(item["visible_candidates_mean"]),
                    "sampled_candidates_mean": float(item["sampled_candidates_mean"]),
                    "path": str(path.relative_to(ROOT)),
                }
            )
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def bar_labels(ax: Any, bars: Any) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def save_coverage_fig(summaries: dict[str, RecordSummary]) -> None:
    keys = [key for key in MAIN_ORDER if key in summaries]
    labels = [summaries[key].label for key in keys]
    coverage = [100.0 * summaries[key].coverage for key in keys]
    visible = [100.0 * summaries[key].visible_coverage for key in keys]
    fig, ax = plt.subplots(figsize=(12, 5))
    x = list(range(len(keys)))
    width = 0.36
    bars1 = ax.bar([item - width / 2 for item in x], coverage, width, label="hidden coverage")
    bars2 = ax.bar([item + width / 2 for item in x], visible, width, label="visible coverage")
    bar_labels(ax, bars1)
    bar_labels(ax, bars2)
    ax.set_ylabel("coverage (%)")
    ax.set_ylim(0, max(100.0, max(coverage + visible + [0.0]) + 10.0))
    ax.set_title("Coverage by arm")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "coverage_by_arm.png", dpi=160)
    plt.close(fig)


def save_diversity_fig(summaries: dict[str, RecordSummary]) -> None:
    keys = [key for key in MAIN_ORDER if key in summaries and summaries[key].split in {"heldout", "transfer", "train"}]
    labels = [summaries[key].label for key in keys]
    candidate_counts = [summaries[key].candidate_count_mean for key in keys]
    distinct = [100.0 * summaries[key].distinct_program_rate for key in keys]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    x = list(range(len(keys)))
    bars = axes[0].bar(x, candidate_counts)
    bar_labels(axes[0], bars)
    axes[0].set_ylabel("mean candidates/task")
    axes[0].set_title("Candidate pool size")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=35, ha="right")
    bars = axes[1].bar(x, distinct)
    bar_labels(axes[1], bars)
    axes[1].set_ylabel("distinct code rate (%)")
    axes[1].set_title("Sample diversity")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(FIGURES / "diversity_and_pool_size.png", dpi=160)
    plt.close(fig)


def save_commit_fig(commit_rows: list[dict[str, Any]]) -> None:
    filtered = [
        row
        for row in commit_rows
        if row["record_key"]
        in {
            "base_heldout",
            "selftrain_verified_r1_heldout",
            "selftrain_unverified_heldout",
            "oracle_sft_heldout",
            "sample_more_matched_compute_heldout",
        }
    ]
    max_by_file: dict[tuple[str, str], dict[str, Any]] = {}
    for row in filtered:
        key = (row["record_key"], row["policy"])
        if key not in max_by_file or row["budget"] > max_by_file[key]["budget"]:
            max_by_file[key] = row
    policies = ["first_visible", "public_signature_majority", "base_verifier", "oracle_coverage"]
    keys = [key for key in MAIN_ORDER if key.endswith("_heldout") and any((key, policy) in max_by_file for policy in policies)]
    fig, ax = plt.subplots(figsize=(12, 5))
    x = list(range(len(keys)))
    width = 0.18
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    for policy, offset in zip(policies, offsets):
        values = [100.0 * max_by_file.get((key, policy), {}).get("selected_hidden_all", 0.0) for key in keys]
        bars = ax.bar([item + offset for item in x], values, width, label=policy)
        bar_labels(ax, bars)
    ax.set_ylabel("selected hidden-correct (%)")
    ax.set_title("Commit accuracy on MBPP held-out pools")
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY.get(key, key) for key in keys], rotation=30, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "commit_accuracy_heldout.png", dpi=160)
    plt.close(fig)


def save_loss_fig() -> None:
    loss_files = sorted(REPORTS.glob("*_losses.json"))
    if not loss_files:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    for path in loss_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        losses = payload if isinstance(payload, list) else payload.get("losses", [])
        xs: list[int] = []
        ys: list[float] = []
        for idx, item in enumerate(losses):
            if isinstance(item, dict):
                loss = item.get("loss")
                step = int(item.get("step", idx + 1))
            else:
                loss = item
                step = idx + 1
            if loss is not None and math.isfinite(float(loss)):
                xs.append(step)
                ys.append(float(loss))
        if xs and ys:
            ax.plot(xs, ys, label=path.stem.replace("_losses", ""))
    ax.set_xlabel("logged step")
    ax.set_ylabel("training loss")
    ax.set_title("Generator SFT losses")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "training_losses.png", dpi=160)
    plt.close(fig)


def write_tables(summaries: dict[str, RecordSummary], commit_rows: list[dict[str, Any]]) -> None:
    coverage_rows: list[dict[str, Any]] = []
    for key in sorted(summaries, key=lambda item: MAIN_ORDER.index(item) if item in MAIN_ORDER else 100 + hash(item) % 1000):
        item = summaries[key]
        coverage_rows.append(
            {
                "key": item.key,
                "label": item.label,
                "split": item.split,
                "arm": item.arm,
                "records": item.records,
                "coverage": item.coverage,
                "visible_coverage": item.visible_coverage,
                "candidate_count_mean": item.candidate_count_mean,
                "parse_success_mean": item.parse_success_mean,
                "visible_candidates_mean": item.visible_candidates_mean,
                "hidden_pass_candidates_mean": item.hidden_pass_candidates_mean,
                "distinct_program_rate": item.distinct_program_rate,
                "mean_code_chars": item.mean_code_chars,
            }
        )
    write_csv(
        REPORTS / "summary_coverage.csv",
        [
            "key",
            "label",
            "split",
            "arm",
            "records",
            "coverage",
            "visible_coverage",
            "candidate_count_mean",
            "parse_success_mean",
            "visible_candidates_mean",
            "hidden_pass_candidates_mean",
            "distinct_program_rate",
            "mean_code_chars",
        ],
        coverage_rows,
    )
    write_csv(
        REPORTS / "summary_commit.csv",
        [
            "record_key",
            "label",
            "policy",
            "budget",
            "records",
            "coverage",
            "visible_coverage",
            "selected_hidden_all",
            "coverage_captured",
            "visible_candidates_mean",
            "sampled_candidates_mean",
            "path",
        ],
        commit_rows,
    )


def table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    widths = [max(len(row[col]) for row in rows) for col in range(len(rows[0]))]
    lines = []
    for idx, row in enumerate(rows):
        lines.append("| " + " | ".join(cell.ljust(widths[col]) for col, cell in enumerate(row)) + " |")
        if idx == 0:
            lines.append("| " + " | ".join("-" * widths[col] for col in range(len(row))) + " |")
    return "\n".join(lines)


def best_commit_rows(commit_rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in commit_rows:
        key = (row["record_key"], row["policy"])
        if key not in out or row["budget"] > out[key]["budget"]:
            out[key] = row
    return out


def delta_line(summaries: dict[str, RecordSummary], a: str, b: str) -> str:
    if a not in summaries or b not in summaries:
        return "not available"
    delta = summaries[b].coverage - summaries[a].coverage
    return f"{pct(summaries[a].coverage)} -> {pct(summaries[b].coverage)} ({100.0 * delta:+.1f} pp)"


def write_report(summaries: dict[str, RecordSummary], commit_rows: list[dict[str, Any]]) -> None:
    best = best_commit_rows(commit_rows)
    main_keys = [key for key in MAIN_ORDER if key in summaries]
    coverage_table = [["Split", "Arm", "n", "Coverage", "Visible cov.", "Candidates", "Distinct"]]
    for key in main_keys:
        item = summaries[key]
        coverage_table.append(
            [
                item.split,
                item.arm,
                str(item.records),
                pct(item.coverage),
                pct(item.visible_coverage),
                fmt(item.candidate_count_mean),
                pct(item.distinct_program_rate),
            ]
        )

    heldout_policies = ["first_visible", "public_signature_majority", "base_verifier", "oracle_coverage"]
    commit_table = [["Arm", "Policy", "Budget", "Selected", "Coverage captured"]]
    for key in [item for item in MAIN_ORDER if item.endswith("_heldout") and item in summaries]:
        for policy in heldout_policies:
            row = best.get((key, policy))
            if not row:
                continue
            commit_table.append(
                [
                    summaries[key].arm,
                    policy,
                    str(row["budget"]),
                    pct(row["selected_hidden_all"]),
                    pct(row["coverage_captured"]),
                ]
            )

    verdict = [
        "# Qwen3.5-4B Verifier-Guided Self-Improvement Report",
        "",
        "Date: 2026-06-25",
        "",
        "## Executive Read",
        "",
        "The main result is negative for the central question. Verified self-training did not raise held-out generation coverage under this local LoRA/data budget. The 20-task smoke signal was positive, but the 150-task held-out run regressed slightly.",
        "",
        f"- MBPP held-out: {delta_line(summaries, 'base_heldout', 'selftrain_verified_r1_heldout')}.",
        f"- HumanEval transfer: {delta_line(summaries, 'base_transfer', 'selftrain_verified_r1_transfer')}.",
        f"- MBPP train: {delta_line(summaries, 'base_train', 'selftrain_verified_r1_train')}.",
        "",
        "Rounds 2 and 3 were intentionally stopped at the pre-registered gate because held-out coverage did not move in the right direction after round 1.",
        "",
        "The controls sharpen the read:",
        "",
        "- Unverified self-training is worse than verified self-training on MBPP held-out, so the execution filter is load-bearing.",
        "- Oracle/reference SFT on the same 80 train tasks also does not beat base on MBPP held-out, so the failure is not only noisy self-generated labels.",
    ]
    if "sample_more_matched_compute_heldout" in summaries:
        sample_more = summaries["sample_more_matched_compute_heldout"].coverage
        base = summaries["base_heldout"].coverage
        verdict.append(
            f"- More inference sampling beats the training arms on MBPP held-out: {pct(base)} -> {pct(sample_more)}."
        )
    else:
        verdict.append("- The matched-compute sampling baseline had not completed when this report was generated.")
    verdict.extend(
        [
            "",
            "## Coverage",
            "",
            table(coverage_table),
            "",
            "Primary metric: coverage is pass@K for the sampled pool, meaning at least one candidate passes hidden tests. Hidden tests were not used for self-training selection.",
            "",
            "## Commit Selection",
            "",
            table(commit_table),
            "",
            "Selection remains secondary here because coverage is the binding quantity. `oracle_coverage` is the diagnostic upper bound: if a hidden-correct candidate exists in the pool, it commits one.",
            "",
            "## Figures",
            "",
            "- [Coverage by arm](figures/coverage_by_arm.png)",
            "- [Diversity and pool size](figures/diversity_and_pool_size.png)",
            "- [Commit accuracy on held-out pools](figures/commit_accuracy_heldout.png)",
            "- [Generator SFT losses](figures/training_losses.png)",
            "",
            "## Interpretation",
            "",
            "This run does not support the hypothesis that one round of verified rejection-sampling SFT expands Qwen3.5-4B's coding frontier on held-out tasks. It mostly narrows the pool: candidate count, visible-passers, and transfer coverage all decrease slightly after verified SFT. The best current deployable lever in this package is not small-SFT self-improvement; it is preserving or increasing sample diversity and then using execution/selection to harvest coverage.",
            "",
            "A stronger future positive would need to change at least one of these constraints: substantially more train tasks, stronger multi-round data accumulation without diversity collapse, curriculumed repair data for tasks with zero initial coverage, or a generator objective that explicitly preserves pass@K diversity rather than only imitating passing samples.",
            "",
            "## Artifacts",
            "",
            f"- Experiment package: `{ROOT}`",
            "- Large adapters/checkpoints: `/workspace/large_artifacts/qwen35_4b_verifier_guided_self_improvement`",
            "- Coverage CSV: `reports/summary_coverage.csv`",
            "- Commit CSV: `reports/summary_commit.csv`",
        ]
    )
    (REPORTS / "qwen35_4b_verifier_guided_self_improvement_report.md").write_text("\n".join(verdict) + "\n", encoding="utf-8")


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    summaries = load_record_summaries()
    commit_rows = load_commit_summaries()
    write_tables(summaries, commit_rows)
    save_coverage_fig(summaries)
    save_diversity_fig(summaries)
    save_commit_fig(commit_rows)
    save_loss_fig()
    write_report(summaries, commit_rows)
    print(
        json.dumps(
            {
                "coverage_rows": len(summaries),
                "commit_rows": len(commit_rows),
                "report": str(REPORTS / "qwen35_4b_verifier_guided_self_improvement_report.md"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
