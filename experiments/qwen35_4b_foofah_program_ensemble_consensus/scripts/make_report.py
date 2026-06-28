#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_program_ensemble import POLICIES, choose_output, summarize  # noqa: E402


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def visible_candidates(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        c for c in record["program_candidates"]
        if c["final"]["visible_pass"] and c["final"]["hidden_output_key"] is not None
    ]


def variant_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    variants = sorted({c["variant"] for r in records for c in r["program_candidates"]})
    rows = []
    for variant in variants:
        candidates = [c for r in records for c in r["program_candidates"] if c["variant"] == variant]
        visible = [c for c in candidates if c["final"]["visible_pass"]]
        correct_visible = [c for c in visible if c["final"]["hidden_exact"]]
        initial_visible = [c for c in candidates if c["attempts"][0]["visible_pass"]]
        repair_added = [c for c in candidates if (not c["attempts"][0]["visible_pass"]) and c["final"]["visible_pass"]]
        rows.append(
            {
                "variant": variant,
                "n": len(candidates),
                "visible_pass": len(visible),
                "visible_pass_rate": len(visible) / len(candidates) if candidates else 0,
                "visible_correct": len(correct_visible),
                "visible_precision": len(correct_visible) / len(visible) if visible else None,
                "initial_visible_pass": len(initial_visible),
                "repair_added_visible": len(repair_added),
            }
        )
    return rows


def family_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["family"]].append(record)
    rows = []
    for family, group in sorted(groups.items()):
        s = summarize(group)
        s["family"] = family
        rows.append(s)
    return rows


def prefix_summary(records: list[dict[str, Any]], step: int = 10) -> list[dict[str, Any]]:
    rows = []
    for n in range(step, len(records) + 1, step):
        s = summarize(records[:n])
        rows.append(
            {
                "n": n,
                "direct": s["policies"]["direct"]["accuracy"],
                "first_visible_program": s["policies"]["first_visible_program"]["accuracy"],
                "consensus_2": s["policies"]["consensus_2"]["accuracy"],
                "oracle_union": s["oracle_union"] / n,
            }
        )
    if rows and rows[-1]["n"] != len(records):
        s = summarize(records)
        rows.append(
            {
                "n": len(records),
                "direct": s["policies"]["direct"]["accuracy"],
                "first_visible_program": s["policies"]["first_visible_program"]["accuracy"],
                "consensus_2": s["policies"]["consensus_2"]["accuracy"],
                "oracle_union": s["oracle_union"] / len(records),
            }
        )
    return rows


def consensus_diagnostics(records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for record in records:
        visible = visible_candidates(record)
        counts = Counter(c["final"]["hidden_output_key"] for c in visible)
        top_count = counts.most_common(1)[0][1] if counts else 0
        rows.append(
            {
                "file": record["file"],
                "visible_count": len(visible),
                "top_cluster_size": top_count,
                "any_visible_correct": any(c["final"]["hidden_exact"] for c in visible),
                "first_visible_correct": bool(visible and visible[0]["final"]["hidden_exact"]),
                "direct_exact": record["direct_exact"],
            }
        )
    return {
        "tasks_with_visible_program": sum(r["visible_count"] > 0 for r in rows),
        "tasks_with_two_plus_visible_programs": sum(r["visible_count"] >= 2 for r in rows),
        "tasks_with_consensus_2_cluster": sum(r["top_cluster_size"] >= 2 for r in rows),
        "tasks_with_consensus_3_cluster": sum(r["top_cluster_size"] >= 3 for r in rows),
        "visible_program_only_oracle": sum((not r["direct_exact"]) and r["any_visible_correct"] for r in rows),
        "first_visible_missed_oracle": sum(r["any_visible_correct"] and not r["first_visible_correct"] for r in rows),
    }


def make_figures(summary: dict[str, Any], variants: list[dict[str, Any]], prefixes: list[dict[str, Any]], families: list[dict[str, Any]]) -> None:
    fig_dir = ROOT / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    policy_order = ["direct", "consensus_3", "consensus_2", "first_visible_program"]
    labels = ["Direct", "Consensus >=3", "Consensus >=2", "First visible program"]
    values = [summary["policies"][p]["accuracy"] for p in policy_order]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(labels, values, color=["#2f6f8f", "#8f7d2f", "#7a5fa8", "#2f8f5f"])
    ax.set_ylim(0, max(0.65, max(values) + 0.08))
    ax.set_ylabel("Exact held-out accuracy")
    ax.set_title("Program Ensemble Selector Accuracy")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, pct(value), ha="center")
    fig.tight_layout()
    fig.savefig(fig_dir / "policy_accuracy.png", dpi=160)
    plt.close(fig)

    x = list(range(len(policy_order)))
    commits = [summary["policies"][p]["program_commits"] for p in policy_order]
    recoveries = [summary["policies"][p]["direct_miss_recoveries"] for p in policy_order]
    precision = [summary["policies"][p]["program_commit_precision"] or 0 for p in policy_order]
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.bar([i - 0.18 for i in x], commits, 0.36, label="program commits", color="#627a9c")
    ax1.bar([i + 0.18 for i in x], recoveries, 0.36, label="direct-miss recoveries", color="#4f8f2f")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right")
    ax1.set_ylabel("Count")
    ax2 = ax1.twinx()
    ax2.plot(x, precision, marker="o", color="#a04f4f", label="commit precision")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Program commit precision")
    ax1.set_title("Selector Tradeoff")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")
    fig.tight_layout()
    fig.savefig(fig_dir / "selector_tradeoff.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    xs = [r["n"] for r in prefixes]
    for key, label, color in [
        ("direct", "Direct", "#2f6f8f"),
        ("first_visible_program", "First visible program", "#2f8f5f"),
        ("consensus_2", "Consensus >=2", "#7a5fa8"),
        ("oracle_union", "Oracle union", "#704c8f"),
    ]:
        ax.plot(xs, [r[key] for r in prefixes], marker="o", label=label, color=color)
    ax.set_ylim(0, 0.7)
    ax.set_xlabel("Cases evaluated")
    ax.set_ylabel("Exact held-out accuracy")
    ax.set_title("Accuracy Over Evaluation Prefix")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "prefix_progress.png", dpi=160)
    plt.close(fig)

    labels = [v["variant"] for v in variants]
    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([i - 0.18 for i in x], [v["visible_pass_rate"] for v in variants], 0.36, label="visible pass rate", color="#8f7d2f")
    ax.bar([i + 0.18 for i in x], [(v["visible_precision"] or 0) for v in variants], 0.36, label="visible precision", color="#2f8f5f")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Rate")
    ax.set_title("Variant Quality")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "variant_quality.png", dpi=160)
    plt.close(fig)

    ranked = sorted(families, key=lambda r: r["policies"]["first_visible_program"]["accuracy"] - r["policies"]["direct"]["accuracy"], reverse=True)[:15]
    y = list(range(len(ranked)))
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh([i - 0.18 for i in y], [r["policies"]["direct"]["accuracy"] for r in ranked], 0.36, label="Direct", color="#2f6f8f")
    ax.barh([i + 0.18 for i in y], [r["policies"]["first_visible_program"]["accuracy"] for r in ranked], 0.36, label="First visible program", color="#2f8f5f")
    ax.set_yticks(y)
    ax.set_yticklabels([r["family"][:32] for r in ranked])
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set_xlabel("Exact held-out accuracy")
    ax.set_title("Families With Largest Fallback Gain")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "family_gains.png", dpi=160)
    plt.close(fig)


def policy_table(summary: dict[str, Any]) -> str:
    names = [
        ("direct", "Direct JSON"),
        ("direct_or_program_agreement", "Direct/program agreement only"),
        ("consensus_3", "Program consensus >= 3"),
        ("consensus_2", "Program consensus >= 2"),
        ("first_visible_program", "First visible-passing program"),
    ]
    lines = []
    for key, label in names:
        p = summary["policies"][key]
        lines.append(
            f"| {label} | {p['exact']}/{summary['n']} | {pct(p['accuracy'])} | {p['program_commits']} | "
            f"{pct(p['program_commit_precision'])} | {p['direct_miss_recoveries']} | {p['direct_correct_losses']} |"
        )
    return "\n".join(lines)


def main() -> None:
    records = load_jsonl(ROOT / "reports" / "full_ensemble_records.jsonl")
    summary = summarize(records)
    variants = variant_summary(records)
    families = family_summary(records)
    prefixes = prefix_summary(records)
    consensus = consensus_diagnostics(records)
    smoke6 = load_json(ROOT / "reports" / "smoke6_v2_summary.json") if (ROOT / "reports" / "smoke6_v2_summary.json").exists() else None
    stride = load_json(ROOT / "reports" / "smoke_stride10_summary.json") if (ROOT / "reports" / "smoke_stride10_summary.json").exists() else None
    write_json(ROOT / "reports" / "final_summary.json", summary)
    write_json(ROOT / "reports" / "variant_summary.json", variants)
    write_json(ROOT / "reports" / "family_summary.json", families)
    write_json(ROOT / "reports" / "prefix_summary.json", prefixes)
    write_json(ROOT / "reports" / "consensus_diagnostics.json", consensus)
    make_figures(summary, variants, prefixes, families)

    p = summary["policies"]
    report = f"""# Foofah Program Ensemble Consensus

## Question

Can Qwen3.5-4B improve Foofah table-transformation accuracy by generating several independently prompted executable `transform(table)` programs, verifying them on the visible example, and selecting by output consensus on the held-out input?

The benchmark is Foofah (`https://github.com/markjin1990/foofah_benchmarks`). Hidden answers are used only for evaluation.

## Setup

Each of 250 tasks was evaluated with one direct JSON answer and three program variants:

- `verified_structural`
- `structural_python`
- `row_column_rule`

Each program variant received one visible-feedback repair attempt if the initial program failed the visible example. A program candidate was eligible for selection only if it passed the visible example and executed on the held-out input.

## Main Result

| selector | exact held-out | rate | program commits | program precision | direct-miss recoveries | direct-correct losses |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{policy_table(summary)}

The best deployed policy was **first visible-passing program**: {p['first_visible_program']['exact']}/{summary['n']} ({pct(p['first_visible_program']['accuracy'])}), versus direct JSON at {p['direct']['exact']}/{summary['n']} ({pct(p['direct']['accuracy'])}). It recovered {p['first_visible_program']['direct_miss_recoveries']} direct misses but lost {p['first_visible_program']['direct_correct_losses']} direct-correct cases.

The oracle union of direct JSON or any visible-correct program reached {summary['oracle_union']}/{summary['n']} ({pct(summary['oracle_union'] / summary['n'])}). That leaves {summary['oracle_union'] - p['first_visible_program']['exact']} cases of selector headroom after the best deployed policy.

## Consensus

Consensus was safer but too conservative:

- Consensus >= 2 committed {p['consensus_2']['program_commits']} times with {pct(p['consensus_2']['program_commit_precision'])} precision, recovering {p['consensus_2']['direct_miss_recoveries']} direct misses.
- Consensus >= 3 committed {p['consensus_3']['program_commits']} times with {pct(p['consensus_3']['program_commit_precision'])} precision, recovering {p['consensus_3']['direct_miss_recoveries']} direct misses.

There were {consensus['tasks_with_visible_program']} tasks with at least one visible-passing program, {consensus['tasks_with_two_plus_visible_programs']} with at least two visible-passing programs, and {consensus['tasks_with_consensus_2_cluster']} with an output cluster of size at least two. The ensemble had {consensus['visible_program_only_oracle']} direct-miss tasks where at least one visible program was hidden-correct.

## Variant Diagnostics

| variant | visible pass | visible precision | initial visible pass | repair-added visible |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(f"| {v['variant']} | {v['visible_pass']}/{v['n']} ({pct(v['visible_pass_rate'])}) | {pct(v['visible_precision'])} | {v['initial_visible_pass']} | {v['repair_added_visible']} |" for v in variants)}

## Iteration

The first six-case smoke exposed direct-output parsing fragility, so JSON extraction was updated to accept the first valid array prefix and the direct prompt was tightened to forbid prose or markdown.

After that fix, a six-case smoke reached direct 5/6 and first-visible program 5/6. A harder stride-10 smoke with the first prompt set showed no oracle gain and poor visible-program precision, so the weak minimal-code variant was replaced with `verified_structural`.

The full run used the revised three-variant ensemble with one repair round per variant.

## Read

The ensemble generated real additional candidate coverage. Direct JSON solved {p['direct']['exact']} tasks; direct plus any visible-correct program could solve {summary['oracle_union']}. The simple first-visible selector captured most of that gain, reaching {p['first_visible_program']['exact']}.

The specific hypothesis that independent program consensus would be the best selector did not hold. Consensus improved precision over first-visible fallback but under-recovered too many direct misses. On this benchmark, visible-example pass plus fixed prompt order was a better deployed selector than requiring agreement.

## Caveats

- The full run uses one greedy direct answer and one greedy generation per program variant, with one greedy repair attempt per variant.
- The ensemble has three prompt variants; larger or sampled ensembles may change the coverage/precision tradeoff.
- Program execution is sandboxed and limited to safe builtins plus `re`, `math`, `Counter`, and `defaultdict`.
- Exact table matching is strict after converting cells to strings.
"""
    (ROOT / "reports" / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"summary": summary, "variants": variants, "consensus": consensus, "smoke6": smoke6, "stride": stride}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

