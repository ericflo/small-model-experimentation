#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def ratio(num: int, den: int) -> float:
    return num / den if den else 0.0


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {"n": 0}
    final_visible = sum(r["final_visible_pass"] for r in rows)
    final_exact = sum(r["final_hidden_exact"] for r in rows)
    agreement = sum(r["direct_program_agreement"] for r in rows)
    agreement_correct = sum(r["agreement_correct"] for r in rows)
    union = sum(r["direct_exact"] or r["final_hidden_exact"] for r in rows)
    return {
        "n": n,
        "direct_exact": sum(r["direct_exact"] for r in rows),
        "direct_accuracy": ratio(sum(r["direct_exact"] for r in rows), n),
        "direct_parse_ok": sum(r["direct_parse_ok"] for r in rows),
        "direct_parse_rate": ratio(sum(r["direct_parse_ok"] for r in rows), n),
        "initial_hidden_exact": sum(r["initial_hidden_exact"] for r in rows),
        "initial_accuracy": ratio(sum(r["initial_hidden_exact"] for r in rows), n),
        "final_hidden_exact": final_exact,
        "final_program_accuracy": ratio(final_exact, n),
        "repair_added_program_correct": sum((not r["initial_hidden_exact"]) and r["final_hidden_exact"] for r in rows),
        "repair_lost_program_correct": sum(r["initial_hidden_exact"] and (not r["final_hidden_exact"]) for r in rows),
        "final_visible_pass": final_visible,
        "final_visible_pass_rate": ratio(final_visible, n),
        "visible_false_pass": final_visible - final_exact,
        "visible_false_pass_rate_among_visible": ratio(final_visible - final_exact, final_visible),
        "program_only": sum((not r["direct_exact"]) and r["final_hidden_exact"] for r in rows),
        "direct_only": sum(r["direct_exact"] and (not r["final_hidden_exact"]) for r in rows),
        "both_direct_and_program": sum(r["direct_exact"] and r["final_hidden_exact"] for r in rows),
        "neither": sum((not r["direct_exact"]) and (not r["final_hidden_exact"]) for r in rows),
        "oracle_union": union,
        "oracle_union_rate": ratio(union, n),
        "hybrid_exact": sum(r["hybrid_exact"] for r in rows),
        "hybrid_accuracy": ratio(sum(r["hybrid_exact"] for r in rows), n),
        "agreement": agreement,
        "agreement_correct": agreement_correct,
        "agreement_precision": ratio(agreement_correct, agreement),
        "mean_round_count": sum(r["round_count"] for r in rows) / n,
    }


def family_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)
    out = []
    for family, group in sorted(by_family.items()):
        row = metrics(group)
        row["family"] = family
        out.append(row)
    return out


def target_char_buckets(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets = {"<=200": [], "201-500": [], "501-1000": [], ">1000": []}
    for row in rows:
        target_chars = len(json.dumps(row["target_table"], ensure_ascii=False))
        if target_chars <= 200:
            buckets["<=200"].append(row)
        elif target_chars <= 500:
            buckets["201-500"].append(row)
        elif target_chars <= 1000:
            buckets["501-1000"].append(row)
        else:
            buckets[">1000"].append(row)
    return {name: metrics(group) for name, group in buckets.items()}


def by_num_samples(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(sample): metrics([row for row in rows if row["num_samples"] == sample])
        for sample in sorted({row["num_samples"] for row in rows})
    }


def prompt_iteration() -> list[dict[str, Any]]:
    specs = [
        ("prefix6 r2", ROOT / "reports" / "eval_summary_limit6.json"),
        ("standard spread25 r2", ROOT / "reports" / "eval_summary_stride10_limit25.json"),
        ("strict spread25 r2", ROOT / "reports" / "eval_summary_stride10_strict_limit25.json"),
    ]
    rows = []
    for name, path in specs:
        if not path.exists():
            continue
        overall = load_json(path)["overall"]
        rows.append(
            {
                "name": name,
                "n": overall["n"],
                "direct_accuracy": overall["direct_accuracy"],
                "initial_accuracy": overall["initial_accuracy"],
                "final_program_accuracy": overall["final_program_accuracy"],
                "oracle_union_rate": overall["oracle_union_rate"],
                "program_only": overall["program_only"],
                "visible_false_pass_rate_among_visible": overall["visible_false_pass_rate_among_visible"],
            }
        )
    return rows


def make_figures(summary: dict[str, Any], family_summary: list[dict[str, Any]]) -> None:
    fig_dir = ROOT / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    overall = summary["overall"]

    labels = ["Direct", "Initial program", "Repaired program", "Hybrid", "Oracle union"]
    values = [
        overall["direct_accuracy"],
        overall["initial_accuracy"],
        overall["final_program_accuracy"],
        overall["hybrid_accuracy"],
        overall["oracle_union_rate"],
    ]
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(labels, values, color=["#2f6f8f", "#78923f", "#4f8f2f", "#8f5a2f", "#704c8f"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Exact held-out accuracy")
    ax.set_title("Foofah Program Repair Agent")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, pct(value), ha="center")
    fig.tight_layout()
    fig.savefig(fig_dir / "overall_accuracy.png", dpi=160)
    plt.close(fig)

    round_stats = summary["round_stats"]
    rounds = sorted(round_stats.keys(), key=int)
    x = list(range(len(rounds)))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([i - width for i in x], [round_stats[r]["attempted"] for r in rounds], width, label="attempted", color="#777777")
    ax.bar(x, [round_stats[r]["visible_pass"] for r in rounds], width, label="visible pass", color="#8f7d2f")
    ax.bar([i + width for i in x], [round_stats[r]["visible_verified_hidden_exact"] for r in rounds], width, label="hidden correct", color="#4f8f2f")
    ax.set_xticks(x)
    ax.set_xticklabels(rounds)
    ax.set_xlabel("Round, 0 = initial")
    ax.set_ylabel("Count")
    ax.set_title("Repair Progress by Round")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "round_progress.png", dpi=160)
    plt.close(fig)

    labels = ["visible pass", "hidden correct", "false pass", "agreement", "agreement correct"]
    counts = [
        overall["final_visible_pass"],
        overall["final_hidden_exact"],
        overall["visible_false_pass"],
        overall["agreement"],
        overall["agreement_correct"],
    ]
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(labels, counts, color=["#8f7d2f", "#4f8f2f", "#a04f4f", "#627a9c", "#2f8f6f"])
    ax.set_ylabel("Count")
    ax.set_title("Verification and Agreement Diagnostics")
    for bar, value in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1, str(value), ha="center")
    fig.tight_layout()
    fig.savefig(fig_dir / "verification_diagnostics.png", dpi=160)
    plt.close(fig)

    samples = sorted(summary["by_num_samples"].keys(), key=int)
    x = list(range(len(samples)))
    width = 0.22
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([i - width for i in x], [summary["by_num_samples"][s]["direct_accuracy"] for s in samples], width, label="Direct", color="#2f6f8f")
    ax.bar(x, [summary["by_num_samples"][s]["final_program_accuracy"] for s in samples], width, label="Program", color="#4f8f2f")
    ax.bar([i + width for i in x], [summary["by_num_samples"][s]["oracle_union_rate"] for s in samples], width, label="Oracle union", color="#704c8f")
    ax.set_xticks(x)
    ax.set_xticklabels(samples)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Foofah NumSamples")
    ax.set_ylabel("Exact held-out accuracy")
    ax.set_title("Accuracy by Demonstration Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "by_num_samples.png", dpi=160)
    plt.close(fig)

    if summary["prompt_iteration"]:
        rows = summary["prompt_iteration"]
        labels = [r["name"] for r in rows]
        x = list(range(len(rows)))
        width = 0.22
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar([i - width for i in x], [r["initial_accuracy"] for r in rows], width, label="Initial program", color="#78923f")
        ax.bar(x, [r["final_program_accuracy"] for r in rows], width, label="Final program", color="#4f8f2f")
        ax.bar([i + width for i in x], [r["oracle_union_rate"] for r in rows], width, label="Oracle union", color="#704c8f")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Exact held-out accuracy")
        ax.set_title("Prompt Iteration Smokes")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "prompt_iteration.png", dpi=160)
        plt.close(fig)

    top = sorted(family_summary, key=lambda r: r["final_program_accuracy"] - r["initial_accuracy"], reverse=True)[:15]
    y = list(range(len(top)))
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh([i - 0.18 for i in y], [r["initial_accuracy"] for r in top], 0.36, label="Initial", color="#78923f")
    ax.barh([i + 0.18 for i in y], [r["final_program_accuracy"] for r in top], 0.36, label="Repaired", color="#4f8f2f")
    ax.set_yticks(y)
    ax.set_yticklabels([r["family"][:30] for r in top])
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set_xlabel("Exact held-out accuracy")
    ax.set_title("Families Where Repair Helps Most")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "repair_advantage_families.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, default=ROOT / "reports" / "eval_records.jsonl")
    parser.add_argument("--summary", type=Path, default=ROOT / "reports" / "eval_summary.json")
    args = parser.parse_args()
    records = load_jsonl(args.records)
    eval_summary = load_json(args.summary)
    family_summary = family_rows(records)
    report_summary = {
        "records": str(args.records),
        "overall": metrics(records),
        "round_stats": eval_summary["round_stats"],
        "by_family": family_summary,
        "by_num_samples": by_num_samples(records),
        "target_char_buckets": target_char_buckets(records),
        "prompt_iteration": prompt_iteration(),
        "final_visible_exec_errors": dict(Counter(str(r["rounds"][-1].get("visible_exec_error")) for r in records)),
    }
    write_json(ROOT / "reports" / "summary.json", report_summary)
    write_json(ROOT / "reports" / "family_summary.json", family_summary)
    make_figures(report_summary, family_summary)

    o = report_summary["overall"]
    program_only_files = [r["file"] for r in records if (not r["direct_exact"]) and r["final_hidden_exact"]]
    repair_added_files = [r["file"] for r in records if (not r["initial_hidden_exact"]) and r["final_hidden_exact"]]
    false_pass_files = [r["file"] for r in records if r["final_visible_pass"] and not r["final_hidden_exact"]]
    agreement_wrong_files = [r["file"] for r in records if r["direct_program_agreement"] and not r["agreement_correct"]]
    prompt_rows = report_summary["prompt_iteration"]
    prompt_table = "\n".join(
        f"| {r['name']} | {r['n']} | {pct(r['initial_accuracy'])} | {pct(r['final_program_accuracy'])} | {pct(r['oracle_union_rate'])} | {r['program_only']} | {pct(r['visible_false_pass_rate_among_visible'])} |"
        for r in prompt_rows
    )
    round_table = "\n".join(
        f"| {rnd} | {vals['attempted']} | {vals['visible_pass']} | {vals['visible_verified_hidden_exact']} |"
        for rnd, vals in sorted(report_summary["round_stats"].items(), key=lambda kv: int(kv[0]))
    )

    report = f"""# Foofah Program-Repair Agent

## Question

Can Qwen3.5-4B improve table transformation accuracy by writing an executable `transform(table)` program, observing visible-example failures, and repairing the program over several rounds before held-out execution?

The benchmark is Foofah (`https://github.com/markjin1990/foofah_benchmarks`), scored by exact equality to held-out `TestAnswer` tables.

## Result

| arm | exact held-out | rate |
| --- | ---: | ---: |
| Direct JSON generation | {o['direct_exact']}/{o['n']} | {pct(o['direct_accuracy'])} |
| Initial visible-verified program | {o['initial_hidden_exact']}/{o['n']} | {pct(o['initial_accuracy'])} |
| Final repaired visible-verified program | {o['final_hidden_exact']}/{o['n']} | {pct(o['final_program_accuracy'])} |
| Direct with program fallback on direct parse failure | {o['hybrid_exact']}/{o['n']} | {pct(o['hybrid_accuracy'])} |
| Oracle union: direct OR final program | {o['oracle_union']}/{o['n']} | {pct(o['oracle_union_rate'])} |

Repair raised visible-verified program correctness from {o['initial_hidden_exact']} to {o['final_hidden_exact']} cases, adding {o['repair_added_program_correct']} program-correct cases while losing {o['repair_lost_program_correct']}.

The final program arm contributed {o['program_only']} direct-miss recoveries. The oracle union reached {o['oracle_union']}/{o['n']}, a +{o['oracle_union'] - o['direct_exact']} case headroom over direct JSON generation.

## Verification Risk

Final programs passed the visible example on {o['final_visible_pass']}/{o['n']} cases. Of those, {o['visible_false_pass']} were hidden-wrong, a false-pass rate of {pct(o['visible_false_pass_rate_among_visible'])}.

Direct/program agreement occurred on {o['agreement']} cases, with {o['agreement_correct']} correct ({pct(o['agreement_precision'])} precision).

## Repair By Round

| round | attempted | visible pass | visible-pass and hidden-correct |
| ---: | ---: | ---: | ---: |
{round_table}

Mean code-generation rounds per case: {o['mean_round_count']:.2f}. Including direct JSON generation, the mean model-generation calls per case were about {1 + o['mean_round_count']:.2f}.

## Iteration

Before the full run, the repair prompt was tested and revised:

| smoke | n | initial program | final program | oracle union | program-only | visible false-pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{prompt_table}

The first hard-spread repair prompt increased visible-pass but added no hidden-correct programs. The stricter repair prompt explicitly warned against visible-output hardcoding and improved the same spread from 2 to 4 final program-correct cases, with 2 program-only recoveries.

## Diagnostics

Program-only recoveries: {", ".join(program_only_files) if program_only_files else "none"}.

Repair-added program-correct files: {", ".join(repair_added_files[:40]) if repair_added_files else "none"}.

Visible-pass hidden-wrong files: {", ".join(false_pass_files[:40]) if false_pass_files else "none"}.

Agreement-hidden-wrong files: {", ".join(agreement_wrong_files[:40]) if agreement_wrong_files else "none"}.

## Read

The repair loop produced a real coverage gain over one-shot program induction: execution feedback converted additional failed programs into held-out-correct programs. It is not a standalone replacement for direct generation, but it is a complementary tool path.

The deployability issue remains selection. Visible-example verification is useful but incomplete; about one fifth of visible-passing final programs were hidden-wrong. The clean positive signal is the oracle union and the direct-miss recoveries, not naive commit-on-visible-pass.

## Caveats

- This is greedy single-sample direct generation and greedy repair generation.
- The loop stops at the first visible-passing program, matching deployment where hidden answers are unavailable.
- Generated code is sandboxed and limited to safe builtins plus `re`, `math`, `Counter`, and `defaultdict`.
- Exact table matching is strict after converting cells to strings.
"""
    (ROOT / "reports" / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(o, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
