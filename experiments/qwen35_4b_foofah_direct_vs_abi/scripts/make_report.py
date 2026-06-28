#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def ratio(num: int, den: int) -> float:
    return num / den if den else 0.0


def metrics(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    n = len(rows)
    good = sum(bool(row[key]) for row in rows)
    return {"n": n, "count": good, "rate": ratio(good, n)}


def bucket_target_chars(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets = {
        "<=200": [],
        "201-500": [],
        "501-1000": [],
        ">1000": [],
    }
    for row in records:
        target_chars = len(json.dumps(row["target_table"], ensure_ascii=False))
        if target_chars <= 200:
            buckets["<=200"].append(row)
        elif target_chars <= 500:
            buckets["201-500"].append(row)
        elif target_chars <= 1000:
            buckets["501-1000"].append(row)
        else:
            buckets[">1000"].append(row)
    return {name: metrics(group, "exact") for name, group in buckets.items()}


def family_table(
    direct_records: list[dict[str, Any]],
    abi_records_by_file: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in direct_records:
        by_family[row["family"]].append(row)
    rows = []
    for family, group in sorted(by_family.items()):
        n = len(group)
        direct = sum(r["exact"] for r in group)
        parse = sum(r["parse_ok"] for r in group)
        abi = sum(abi_records_by_file[r["file"]]["heldout_covered"] for r in group)
        first = sum(abi_records_by_file[r["file"]]["first_visible_heldout"] for r in group)
        rows.append(
            {
                "family": family,
                "n": n,
                "direct_exact": direct,
                "direct_accuracy": ratio(direct, n),
                "direct_parse": parse,
                "direct_parse_rate": ratio(parse, n),
                "abi_oracle": abi,
                "abi_oracle_rate": ratio(abi, n),
                "abi_first_visible": first,
                "abi_first_visible_rate": ratio(first, n),
            }
        )
    return rows


def make_figures(summary: dict[str, Any], family_rows: list[dict[str, Any]]) -> None:
    fig_dir = ROOT / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    labels = ["Direct Qwen", "ABI oracle", "ABI first-visible"]
    rates = [
        summary["direct"]["exact_accuracy"],
        summary["abi"]["heldout_coverage"],
        summary["abi"]["first_visible_accuracy"],
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, rates, color=["#2f6f8f", "#6f8f2f", "#8f5a2f"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Exact held-out accuracy")
    ax.set_title("Foofah Direct Qwen vs Frozen ABI")
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 0.02, pct(rate), ha="center")
    fig.tight_layout()
    fig.savefig(fig_dir / "overall_comparison.png", dpi=160)
    plt.close(fig)

    sample_keys = sorted(summary["by_num_samples"].keys(), key=int)
    x = list(range(len(sample_keys)))
    direct = [summary["by_num_samples"][k]["direct_accuracy"] for k in sample_keys]
    abi = [summary["by_num_samples"][k]["abi_oracle_rate"] for k in sample_keys]
    width = 0.38
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([i - width / 2 for i in x], direct, width, label="Direct Qwen", color="#2f6f8f")
    ax.bar([i + width / 2 for i in x], abi, width, label="ABI oracle", color="#6f8f2f")
    ax.set_xticks(x)
    ax.set_xticklabels(sample_keys)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Foofah NumSamples")
    ax.set_ylabel("Exact held-out accuracy")
    ax.set_title("Accuracy by Number of Demonstration Rows")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "by_num_samples.png", dpi=160)
    plt.close(fig)

    top = sorted(family_rows, key=lambda r: r["direct_accuracy"] - r["abi_oracle_rate"], reverse=True)[:15]
    labels = [row["family"][:28] for row in top]
    y = list(range(len(top)))
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh([i - 0.18 for i in y], [row["direct_accuracy"] for row in top], 0.36, label="Direct", color="#2f6f8f")
    ax.barh([i + 0.18 for i in y], [row["abi_oracle_rate"] for row in top], 0.36, label="ABI", color="#6f8f2f")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set_xlabel("Exact held-out accuracy")
    ax.set_title("Families Where Direct Qwen Most Exceeds ABI")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "direct_advantage_families.png", dpi=160)
    plt.close(fig)


def main() -> None:
    direct_path = ROOT / "reports" / "direct_qwen_records.jsonl"
    if not direct_path.exists():
        raise SystemExit(f"Missing {direct_path}; run scripts/eval_direct_qwen.py first.")
    direct_records = load_jsonl(direct_path)
    abi_records = load_jsonl(ROOT / "data" / "abi_case_records.jsonl")
    abi_summary = load_json(ROOT / "data" / "abi_summary.json")
    abi_by_file = {row["file"]: row for row in abi_records}
    missing = [row["file"] for row in direct_records if row["file"] not in abi_by_file]
    if missing:
        raise SystemExit(f"ABI records missing {len(missing)} direct files, first={missing[0]}")

    n = len(direct_records)
    direct_exact = sum(row["exact"] for row in direct_records)
    direct_parse = sum(row["parse_ok"] for row in direct_records)
    abi_covered = sum(abi_by_file[row["file"]]["heldout_covered"] for row in direct_records)
    abi_first = sum(abi_by_file[row["file"]]["first_visible_heldout"] for row in direct_records)

    both = sum(row["exact"] and abi_by_file[row["file"]]["heldout_covered"] for row in direct_records)
    direct_only = sum(row["exact"] and not abi_by_file[row["file"]]["heldout_covered"] for row in direct_records)
    abi_only = sum((not row["exact"]) and abi_by_file[row["file"]]["heldout_covered"] for row in direct_records)
    neither = n - both - direct_only - abi_only
    union_oracle = sum(row["exact"] or abi_by_file[row["file"]]["heldout_covered"] for row in direct_records)
    union_first_visible = sum(row["exact"] or abi_by_file[row["file"]]["first_visible_heldout"] for row in direct_records)

    direct_on_abi_covered = [row for row in direct_records if abi_by_file[row["file"]]["heldout_covered"]]
    direct_on_abi_uncovered = [row for row in direct_records if not abi_by_file[row["file"]]["heldout_covered"]]

    family_rows = family_table(direct_records, abi_by_file)
    by_num_samples = {}
    for num in sorted({row["num_samples"] for row in direct_records}):
        group = [row for row in direct_records if row["num_samples"] == num]
        by_num_samples[str(num)] = {
            "n": len(group),
            "direct_exact": sum(row["exact"] for row in group),
            "direct_accuracy": ratio(sum(row["exact"] for row in group), len(group)),
            "direct_parse": sum(row["parse_ok"] for row in group),
            "direct_parse_rate": ratio(sum(row["parse_ok"] for row in group), len(group)),
            "abi_oracle": sum(abi_by_file[row["file"]]["heldout_covered"] for row in group),
            "abi_oracle_rate": ratio(sum(abi_by_file[row["file"]]["heldout_covered"] for row in group), len(group)),
        }

    overlap = {
        "both_direct_and_abi": both,
        "direct_only": direct_only,
        "abi_only": abi_only,
        "neither": neither,
        "direct_or_abi_oracle": union_oracle,
        "direct_or_abi_oracle_rate": ratio(union_oracle, n),
        "direct_or_abi_first_visible": union_first_visible,
        "direct_or_abi_first_visible_rate": ratio(union_first_visible, n),
        "direct_accuracy_on_abi_covered": metrics(direct_on_abi_covered, "exact"),
        "direct_accuracy_on_abi_uncovered": metrics(direct_on_abi_uncovered, "exact"),
    }

    summary = {
        "n": n,
        "direct": {
            "exact": direct_exact,
            "exact_accuracy": ratio(direct_exact, n),
            "parse_ok": direct_parse,
            "parse_rate": ratio(direct_parse, n),
            "max_new_tokens": 768,
        },
        "abi": {
            "heldout_covered": abi_covered,
            "heldout_coverage": ratio(abi_covered, n),
            "first_visible_heldout": abi_first,
            "first_visible_accuracy": ratio(abi_first, n),
            "candidate_count_mean": abi_summary["overall"]["candidate_count_mean"],
        },
        "overlap": overlap,
        "by_num_samples": by_num_samples,
        "by_family": family_rows,
        "target_char_buckets": bucket_target_chars(direct_records),
    }
    write_json(ROOT / "reports" / "comparison_summary.json", summary)
    write_json(ROOT / "reports" / "family_comparison.json", family_rows)
    make_figures(summary, family_rows)

    top_direct_only = [
        row["file"]
        for row in direct_records
        if row["exact"] and not abi_by_file[row["file"]]["heldout_covered"]
    ][:20]
    top_abi_only = [
        row["file"]
        for row in direct_records
        if (not row["exact"]) and abi_by_file[row["file"]]["heldout_covered"]
    ][:20]

    report = f"""# Foofah Direct Qwen vs Frozen ABI

## Question

Does the frozen Foofah table-transform ABI add value over directly asking Qwen3.5-4B to transform the held-out table?

This package compares exact held-out `TestAnswer` accuracy on the same 250 Foofah cases from `https://github.com/markjin1990/foofah_benchmarks`.

## Result

Direct Qwen is the stronger arm on this external structural-transform benchmark:

| arm | exact held-out | rate |
| --- | ---: | ---: |
| Direct Qwen greedy JSON generation | {direct_exact}/{n} | {pct(ratio(direct_exact, n))} |
| Frozen ABI oracle coverage | {abi_covered}/{n} | {pct(ratio(abi_covered, n))} |
| Frozen ABI first-visible selection | {abi_first}/{n} | {pct(ratio(abi_first, n))} |
| Direct Qwen OR ABI first-visible fallback | {union_first_visible}/{n} | {pct(ratio(union_first_visible, n))} |

Direct parse rate was {direct_parse}/{n} ({pct(ratio(direct_parse, n))}) with a 768-token generation cap.

## Overlap

| bucket | count |
| --- | ---: |
| Direct and ABI both correct/covered | {both} |
| Direct only | {direct_only} |
| ABI only | {abi_only} |
| Neither | {neither} |

Direct accuracy on ABI-covered cases: {overlap["direct_accuracy_on_abi_covered"]["count"]}/{overlap["direct_accuracy_on_abi_covered"]["n"]} ({pct(overlap["direct_accuracy_on_abi_covered"]["rate"])}).

Direct accuracy on ABI-uncovered cases: {overlap["direct_accuracy_on_abi_uncovered"]["count"]}/{overlap["direct_accuracy_on_abi_uncovered"]["n"]} ({pct(overlap["direct_accuracy_on_abi_uncovered"]["rate"])}).

The practical fallback union (`direct exact OR ABI first-visible`) reaches {union_first_visible}/{n} ({pct(ratio(union_first_visible, n))}), a +{union_first_visible - direct_exact} case lift over direct generation alone.

## Read

The remaining compiler niche did not appear as the dominant path on Foofah under this test. The ABI's structural table-transform coverage was only {pct(ratio(abi_covered, n))}, and direct Qwen solved many cases outside the ABI's expressivity (`direct_only={direct_only}`). The frozen ABI still has a small complementary slice (`abi_only={abi_only}`; first-visible adds {union_first_visible - direct_exact} deployable cases), but it is a fallback, not the main route.

The important interpretation is not that direct generation is perfect. It is not: exact accuracy is {pct(ratio(direct_exact, n))}, parse failures remain {n - direct_parse}, and long-output cases are penalized by the 768-token cap. The point is narrower and decisive for this gate: on an independent Foofah benchmark, the ABI/compiler route does not beat simply asking the base model to emit the transformed table.

## Diagnostics

By `NumSamples`, direct-vs-ABI accuracy is in `reports/comparison_summary.json` and `reports/figures/by_num_samples.png`.

Example direct-only files: {", ".join(top_direct_only) if top_direct_only else "none"}.

Example ABI-only files: {", ".join(top_abi_only) if top_abi_only else "none"}.

## Caveats

- This is greedy direct generation with a 768-token cap, not a best-possible direct-generation system.
- The ABI baseline is frozen from the prior Foofah gate and imported into `data/abi_*` for a standalone comparison.
- Exact table matching is strict string-table equality after normalizing cells to strings.
"""
    (ROOT / "reports" / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary["direct"], indent=2, sort_keys=True))
    print(json.dumps(summary["abi"], indent=2, sort_keys=True))
    print(json.dumps(summary["overlap"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
