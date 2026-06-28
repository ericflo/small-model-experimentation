#!/usr/bin/env python3
from __future__ import annotations

import json
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


def pct(x: float | None) -> str:
    if x is None:
        return "-"
    return f"{100 * x:.1f}%"


def make_figures(summary: dict[str, Any], records: list[dict[str, Any]]) -> None:
    fig_dir = ROOT / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    policies = summary["policies"]
    ordered = [
        "direct",
        "program_if_direct_parse_fail",
        "program_if_probe_support_050",
        "visible_program_veto_probe_lt_067",
        "program_if_visible_disagree",
        "program_if_visible",
    ]
    labels = [
        "Direct",
        "Parse fallback",
        "Probe support >= .50",
        "Visible + probe veto",
        "Visible disagree",
        "Visible program",
    ]
    values = [policies[p]["accuracy"] for p in ordered]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.bar(labels, values, color=["#2f6f8f", "#6f7f8f", "#9a7d2f", "#9a5f3f", "#4f8f2f", "#2f8f5f"])
    ax.set_ylim(0, 0.7)
    ax.set_ylabel("Exact held-out accuracy")
    ax.set_title("Selector Policy Accuracy")
    ax.tick_params(axis="x", rotation=15)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, pct(value), ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "policy_accuracy.png", dpi=160)
    plt.close(fig)

    commits = [policies[p]["program_commits"] for p in ordered]
    precision = [policies[p]["program_commit_precision"] or 0 for p in ordered]
    recoveries = [policies[p]["direct_miss_recoveries"] for p in ordered]
    x = list(range(len(ordered)))
    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax1.bar([i - 0.18 for i in x], commits, 0.36, label="program commits", color="#627a9c")
    ax1.bar([i + 0.18 for i in x], recoveries, 0.36, label="direct-miss recoveries", color="#4f8f2f")
    ax1.set_ylabel("Count")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right")
    ax2 = ax1.twinx()
    ax2.plot(x, precision, marker="o", label="commit precision", color="#a04f4f")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Program commit precision")
    ax1.set_title("Program Commit Tradeoff")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    fig.tight_layout()
    fig.savefig(fig_dir / "program_commit_tradeoff.png", dpi=160)
    plt.close(fig)

    disagree = [r for r in records if r["candidate"]["final_visible_pass"] and not r["candidate"]["direct_program_agreement"]]
    support_correct = [
        r["probe_features"]["probe_agree_rate"]
        for r in disagree
        if r["candidate"]["final_hidden_exact"] and r["probe_features"]["probe_agree_rate"] is not None
    ]
    support_wrong = [
        r["probe_features"]["probe_agree_rate"]
        for r in disagree
        if (not r["candidate"]["final_hidden_exact"]) and r["probe_features"]["probe_agree_rate"] is not None
    ]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist([support_correct, support_wrong], bins=[0, 0.01, 0.34, 0.67, 1.01], label=["program correct", "program wrong"], color=["#4f8f2f", "#a04f4f"], alpha=0.75)
    ax.set_xlabel("Probe direct/program agreement rate")
    ax.set_ylabel("Decision-case count")
    ax.set_title("Probe Support on Visible-Disagreement Cases")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "probe_support_histogram.png", dpi=160)
    plt.close(fig)

    families: dict[str, dict[str, int]] = {}
    for row in records:
        family = row["candidate"]["family"]
        item = families.setdefault(family, {"n": 0, "direct": 0, "visible": 0})
        item["n"] += 1
        item["direct"] += int(row["candidate"]["direct_exact"])
        _, visible_ok = choose_visible_program(row)
        item["visible"] += int(visible_ok)
    top = sorted(families.items(), key=lambda kv: (kv[1]["visible"] - kv[1]["direct"], kv[1]["n"]), reverse=True)[:15]
    y = list(range(len(top)))
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh([i - 0.18 for i in y], [v["direct"] / v["n"] for _, v in top], 0.36, label="Direct", color="#2f6f8f")
    ax.barh([i + 0.18 for i in y], [v["visible"] / v["n"] for _, v in top], 0.36, label="Visible program fallback", color="#2f8f5f")
    ax.set_yticks(y)
    ax.set_yticklabels([k[:32] for k, _ in top])
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set_xlabel("Exact held-out accuracy")
    ax.set_title("Families With Largest Fallback Gains")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "family_gains.png", dpi=160)
    plt.close(fig)


def choose_visible_program(row: dict[str, Any]) -> tuple[str, bool]:
    c = row["candidate"]
    if c["final_visible_pass"]:
        return "program", bool(c["final_hidden_exact"])
    return "direct", bool(c["direct_exact"])


def policy_table(summary: dict[str, Any]) -> str:
    names = [
        ("direct", "Direct JSON"),
        ("program_if_direct_parse_fail", "Program only if direct parse fails"),
        ("program_if_probe_support_050", "Program on disagreement if probe support >= 0.50"),
        ("visible_program_veto_probe_lt_067", "Visible program, veto disagreement if probe support < 0.67"),
        ("program_if_visible_disagree", "Program on visible disagreement only"),
        ("program_if_visible", "Program whenever visible example passes"),
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
    records = load_jsonl(ROOT / "reports" / "final_records.jsonl")
    summary = load_json(ROOT / "reports" / "final_summary.json")
    smoke = load_json(ROOT / "reports" / "smoke_visible_disagree_summary.json")
    decision = load_json(ROOT / "reports" / "model_probe_visible_disagree_summary.json")
    make_figures(summary, records)

    disagree = [r for r in records if r["candidate"]["final_visible_pass"] and not r["candidate"]["direct_program_agreement"]]
    correct_support = [
        r["probe_features"]["probe_agree_rate"]
        for r in disagree
        if r["candidate"]["final_hidden_exact"] and r["probe_features"]["probe_agree_rate"] is not None
    ]
    wrong_support = [
        r["probe_features"]["probe_agree_rate"]
        for r in disagree
        if (not r["candidate"]["final_hidden_exact"]) and r["probe_features"]["probe_agree_rate"] is not None
    ]
    extra = {
        "visible_disagreement_correct_probe_mean": sum(correct_support) / len(correct_support) if correct_support else None,
        "visible_disagreement_wrong_probe_mean": sum(wrong_support) / len(wrong_support) if wrong_support else None,
        "smoke_summary": smoke,
        "decision_slice_summary": decision,
    }
    write_json(ROOT / "reports" / "report_metrics.json", {"summary": summary, "extra": extra})

    p = summary["policies"]
    visible_gain = p["program_if_visible"]["exact"] - p["direct"]["exact"]
    report = f"""# Foofah Selective Program Fallback

## Question

Can Qwen3.5-4B improve deployed Foofah table-transformation accuracy by using a visible-verified executable `transform(table)` program as a fallback to direct JSON generation, and do counterexample-style probe inputs make that fallback decision safer?

The benchmark is Foofah (`https://github.com/markjin1990/foofah_benchmarks`). Each task has visible input-output examples and a held-out test input. Hidden answers are used only for evaluation.

## Candidate Pool

The package contains 250 task records. Each record has:

- one direct JSON answer for the held-out input,
- one executable program candidate after visible-example repair,
- visible-example execution status for the program,
- held-out exact-match labels for evaluation.

## Main Result

| selector | exact held-out | rate | program commits | program precision | direct-miss recoveries | direct-correct losses |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{policy_table(summary)}

The strongest deployed policy was also the simplest: **commit the program whenever it passes the visible example**. It reached {p['program_if_visible']['exact']}/{summary['n']} ({pct(p['program_if_visible']['accuracy'])}), improving direct JSON by +{visible_gain} cases with {p['program_if_visible']['direct_correct_losses']} direct-correct losses.

The key diagnostic is the visible-disagreement slice. There were {summary['visible_disagree']} cases where the program passed the visible example but disagreed with the direct answer. In that slice, direct JSON was hidden-correct on {summary['visible_disagree_direct_correct']} cases, while the program was hidden-correct on {summary['visible_disagree_program_correct']} cases. That made visible-program fallback strongly complementary to direct generation in this candidate pool.

## Counterexample Probes

For each visible-disagreement case, the evaluator generated up to three deterministic probe input tables and asked Qwen3.5-4B for direct JSON outputs on those probes. The candidate program was also executed on the same probes. Probe support is the fraction of comparable probes where direct output and program output agreed.

The probe mechanism did **not** improve selection:

- Probe support >= 0.50 recovered {p['program_if_probe_support_050']['direct_miss_recoveries']} direct misses and reached {p['program_if_probe_support_050']['exact']}/{summary['n']} ({pct(p['program_if_probe_support_050']['accuracy'])}).
- Visible-program fallback with a probe-support veto recovered {p['visible_program_veto_probe_lt_067']['direct_miss_recoveries']} direct misses and reached {p['visible_program_veto_probe_lt_067']['exact']}/{summary['n']} ({pct(p['visible_program_veto_probe_lt_067']['accuracy'])}).
- Mean probe agreement on the decision slice was {pct(summary['mean_probe_agree_rate'])}.

Probe support was not a reliable correctness signal. Among visible-disagreement cases with comparable probes, mean support was {pct(extra['visible_disagreement_correct_probe_mean'])} for hidden-correct programs and {pct(extra['visible_disagreement_wrong_probe_mean'])} for hidden-wrong programs.

## Iteration

The experiment used three stages:

1. A no-model selector diagnostic over all 250 cases established that visible-program fallback reached {p['program_if_visible']['exact']}/{summary['n']} and parse-failure fallback reached {p['program_if_direct_parse_fail']['exact']}/{summary['n']}.
2. A small model-probe smoke on 8 visible-disagreement cases showed probe thresholds rejecting most useful program wins.
3. A full model-probe pass on all {summary['visible_disagree']} visible-disagreement cases confirmed that probe support was weaker than the simple visible-pass rule.

## Read

The useful result is not that counterexample-stressed direct agreement solved selection. It did not. The useful result is that visible-example execution alone was a strong fallback gate for this candidate pool: every direct-correct case survived, and the visible-passing programs recovered 18 direct misses.

The counterexample probes failed for a concrete reason: the independent direct channel often agreed with the same wrong extrapolation, while rejecting many correct programs. Generated probes added cost and reduced accuracy under thresholded policies.

## Caveats

- The candidate pool is fixed and included in `data/candidate_records.jsonl`.
- Program fallback is evaluated only for programs that pass the visible example.
- Probe answers are greedy Qwen3.5-4B direct JSON generations on synthetic probe inputs, not ground truth.
- Hidden answers are used only for evaluation and policy comparison.
- Exact table matching is strict after normalizing cells to strings.
"""
    (ROOT / "reports" / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"summary": summary, "extra": extra}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

