#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def ratio(num: int, den: int) -> float:
    return num / den if den else 0.0


def count(rows: list[dict[str, Any]], key: str) -> int:
    return sum(bool(row[key]) for row in rows)


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    visible = count(rows, "visible_pass")
    agreement = count(rows, "agreement")
    direct = count(rows, "direct_exact")
    program = count(rows, "program_hidden_exact")
    hybrid = count(rows, "hybrid_exact")
    union = sum(row["direct_exact"] or row["program_hidden_exact"] for row in rows)
    return {
        "n": n,
        "direct_exact": direct,
        "direct_accuracy": ratio(direct, n),
        "direct_parse_ok": count(rows, "direct_parse_ok"),
        "direct_parse_rate": ratio(count(rows, "direct_parse_ok"), n),
        "code_found": count(rows, "code_found"),
        "code_found_rate": ratio(count(rows, "code_found"), n),
        "visible_exec_ok": count(rows, "visible_exec_ok"),
        "visible_exec_ok_rate": ratio(count(rows, "visible_exec_ok"), n),
        "visible_pass": visible,
        "visible_pass_rate": ratio(visible, n),
        "program_hidden_exact": program,
        "program_accuracy": ratio(program, n),
        "visible_false_pass": visible - program,
        "visible_false_pass_rate_among_visible": ratio(visible - program, visible),
        "agreement": agreement,
        "agreement_rate": ratio(agreement, n),
        "agreement_correct": count(rows, "agreement_correct"),
        "agreement_precision": ratio(count(rows, "agreement_correct"), agreement),
        "hybrid_exact": hybrid,
        "hybrid_accuracy": ratio(hybrid, n),
        "oracle_union": union,
        "oracle_union_rate": ratio(union, n),
    }


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


def sample_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    for sample in sorted({row["num_samples"] for row in rows}):
        out[str(sample)] = metrics([row for row in rows if row["num_samples"] == sample])
    return out


def overlap(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "both_direct_and_program": sum(row["direct_exact"] and row["program_hidden_exact"] for row in rows),
        "direct_only": sum(row["direct_exact"] and not row["program_hidden_exact"] for row in rows),
        "program_only": sum((not row["direct_exact"]) and row["program_hidden_exact"] for row in rows),
        "neither": sum((not row["direct_exact"]) and (not row["program_hidden_exact"]) for row in rows),
        "visible_pass_hidden_wrong": sum(row["visible_pass"] and not row["program_hidden_exact"] for row in rows),
        "agreement_hidden_wrong": sum(row["agreement"] and not row["agreement_correct"] for row in rows),
    }


def make_figures(summary: dict[str, Any], families: list[dict[str, Any]]) -> None:
    fig_dir = ROOT / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    overall = summary["overall"]

    labels = ["Direct", "Program", "Hybrid", "Oracle union"]
    values = [
        overall["direct_accuracy"],
        overall["program_accuracy"],
        overall["hybrid_accuracy"],
        overall["oracle_union_rate"],
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, values, color=["#2f6f8f", "#6f8f2f", "#8f5a2f", "#704c8f"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Exact held-out accuracy")
    ax.set_title("Foofah Ephemeral Program Induction")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, pct(val), ha="center")
    fig.tight_layout()
    fig.savefig(fig_dir / "overall_accuracy.png", dpi=160)
    plt.close(fig)

    labels = ["visible pass", "program correct", "visible false-pass", "agreement", "agreement correct"]
    counts = [
        overall["visible_pass"],
        overall["program_hidden_exact"],
        overall["visible_false_pass"],
        overall["agreement"],
        overall["agreement_correct"],
    ]
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(labels, counts, color=["#627a9c", "#6f8f2f", "#a04f4f", "#8f7d2f", "#2f8f6f"])
    ax.set_ylabel("Count")
    ax.set_title("Verification and Agreement Diagnostics")
    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1, str(val), ha="center")
    fig.tight_layout()
    fig.savefig(fig_dir / "verification_diagnostics.png", dpi=160)
    plt.close(fig)

    sample_keys = sorted(summary["by_num_samples"].keys(), key=int)
    x = list(range(len(sample_keys)))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([i - width for i in x], [summary["by_num_samples"][k]["direct_accuracy"] for k in sample_keys], width, label="Direct", color="#2f6f8f")
    ax.bar(x, [summary["by_num_samples"][k]["program_accuracy"] for k in sample_keys], width, label="Program", color="#6f8f2f")
    ax.bar([i + width for i in x], [summary["by_num_samples"][k]["oracle_union_rate"] for k in sample_keys], width, label="Oracle union", color="#704c8f")
    ax.set_xticks(x)
    ax.set_xticklabels(sample_keys)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Foofah NumSamples")
    ax.set_ylabel("Exact held-out accuracy")
    ax.set_title("Accuracy by Demonstration Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "by_num_samples.png", dpi=160)
    plt.close(fig)

    top = sorted(families, key=lambda row: row["program_accuracy"] - row["direct_accuracy"], reverse=True)[:15]
    fig, ax = plt.subplots(figsize=(9, 6))
    y = list(range(len(top)))
    labels = [row["family"][:30] for row in top]
    ax.barh([i - 0.18 for i in y], [row["direct_accuracy"] for row in top], 0.36, label="Direct", color="#2f6f8f")
    ax.barh([i + 0.18 for i in y], [row["program_accuracy"] for row in top], 0.36, label="Program", color="#6f8f2f")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set_xlabel("Exact held-out accuracy")
    ax.set_title("Families Where Program Arm Helps Most")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "program_advantage_families.png", dpi=160)
    plt.close(fig)

    if summary.get("prompt_iteration"):
        rows = summary["prompt_iteration"]
        labels = [row["name"] for row in rows]
        x = list(range(len(rows)))
        width = 0.22
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar([i - width for i in x], [row["direct_accuracy"] for row in rows], width, label="Direct", color="#2f6f8f")
        ax.bar(x, [row["program_accuracy"] for row in rows], width, label="Program", color="#6f8f2f")
        ax.bar([i + width for i in x], [row["oracle_union_rate"] for row in rows], width, label="Oracle union", color="#704c8f")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Exact held-out accuracy")
        ax.set_title("Prompt Iteration Smokes")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "prompt_iteration.png", dpi=160)
        plt.close(fig)


def prompt_iteration_rows() -> list[dict[str, Any]]:
    candidates = [
        ("induce prefix8", ROOT / "reports" / "eval_summary_induce_limit8.json"),
        ("context prefix8", ROOT / "reports" / "eval_summary_context_limit8.json"),
        ("context spread25", ROOT / "reports" / "eval_summary_context_stride10_limit25.json"),
        ("context_v2 spread25", ROOT / "reports" / "eval_summary_context_v2_stride10_limit25.json"),
    ]
    rows = []
    for name, path in candidates:
        if not path.exists():
            continue
        overall = load_json(path)["overall"]
        rows.append(
            {
                "name": name,
                "n": overall["n"],
                "direct_accuracy": overall["direct_accuracy"],
                "program_accuracy": overall["program_accuracy"],
                "oracle_union_rate": overall["oracle_union_rate"],
                "agreement_precision": overall["agreement_precision"],
                "visible_pass_rate": overall["visible_pass_rate"],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, default=ROOT / "reports" / "eval_records_context.jsonl")
    parser.add_argument("--prompt-name", type=str, default="context")
    args = parser.parse_args()
    rows = load_jsonl(args.records)
    overall = metrics(rows)
    families = family_rows(rows)
    summary = {
        "records": str(args.records),
        "prompt_name": args.prompt_name,
        "overall": overall,
        "overlap": overlap(rows),
        "by_num_samples": sample_rows(rows),
        "by_family": families,
        "target_char_buckets": target_char_buckets(rows),
        "visible_exec_errors": dict(Counter(str(row["visible_exec_error"]) for row in rows)),
        "prompt_iteration": prompt_iteration_rows(),
    }
    write_json(ROOT / "reports" / "summary.json", summary)
    write_json(ROOT / "reports" / "family_summary.json", families)
    make_figures(summary, families)

    ov = summary["overlap"]
    direct = overall["direct_exact"]
    program = overall["program_hidden_exact"]
    union = overall["oracle_union"]
    program_only_files = [row["file"] for row in rows if (not row["direct_exact"]) and row["program_hidden_exact"]]
    agreement_wrong_files = [row["file"] for row in rows if row["agreement"] and not row["agreement_correct"]]

    report = f"""# Foofah Ephemeral Program Induction

## Question

Can Qwen3.5-4B improve external table transformations by writing a bespoke executable `transform(table)` function, verifying it on the visible example, and executing it on the held-out input?

The benchmark is Foofah (`https://github.com/markjin1990/foofah_benchmarks`), scored by exact equality to held-out `TestAnswer` tables.

## Result

| arm | exact held-out | rate |
| --- | ---: | ---: |
| Direct JSON generation | {direct}/{overall["n"]} | {pct(overall["direct_accuracy"])} |
| Visible-verified generated program | {program}/{overall["n"]} | {pct(overall["program_accuracy"])} |
| Direct with program fallback on direct parse failure | {overall["hybrid_exact"]}/{overall["n"]} | {pct(overall["hybrid_accuracy"])} |
| Oracle union: direct OR program | {union}/{overall["n"]} | {pct(overall["oracle_union_rate"])} |

Program induction found code for {overall["code_found"]}/{overall["n"]} cases and executed on the visible example for {overall["visible_exec_ok"]}/{overall["n"]}.
It passed the visible example on {overall["visible_pass"]}/{overall["n"]}, but {overall["visible_false_pass"]} of those visible-pass programs were hidden-wrong.

## Overlap

| bucket | count |
| --- | ---: |
| Both direct and program correct | {ov["both_direct_and_program"]} |
| Direct only | {ov["direct_only"]} |
| Program only | {ov["program_only"]} |
| Neither | {ov["neither"]} |

Agreement between direct output and program execution occurred on {overall["agreement"]}/{overall["n"]} cases, with precision {overall["agreement_correct"]}/{overall["agreement"]} ({pct(overall["agreement_precision"])}).

Program-only recoveries: {", ".join(program_only_files) if program_only_files else "none"}.

Agreement-hidden-wrong cases: {", ".join(agreement_wrong_files[:20]) if agreement_wrong_files else "none"}.

## Iteration

Before the full run, four prompt smokes were run:

| prompt smoke | n | direct | program | oracle union | agreement precision |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(f"| {row['name']} | {row['n']} | {pct(row['direct_accuracy'])} | {pct(row['program_accuracy'])} | {pct(row['oracle_union_rate'])} | {pct(row['agreement_precision'])} |" for row in summary['prompt_iteration'])}

The full run used `context`: it had lower agreement precision than `context_v2` on the hard spread, but it preserved the only direct-failure recovery in that spread and therefore had higher coverage headroom.

## Read

The generated-program route tests a tool-use idea: the model emits a bespoke executable artifact, the artifact is checked on the visible example, and the checked artifact is executed on the held-out input. The result should be read through coverage and selection separately.

The executable-program arm is real but weak on this benchmark. It creates some correct programs outside direct generation (`program_only={ov["program_only"]}`), but visible-example verification is not enough to make it deployable: false-pass among visible-pass programs is {overall["visible_false_pass"]}/{overall["visible_pass"]} ({pct(overall["visible_false_pass_rate_among_visible"])}).

The decisive number is the oracle union. If it is meaningfully above direct generation, there is headroom for a better selector or verifier over direct-vs-program outputs. If it is close to direct generation, ephemeral program induction is not adding much capability on Foofah.

## Caveats

- This is greedy single-sample direct generation and greedy single-sample code generation.
- Generated code is sandboxed and limited to safe builtins plus `re`, `math`, `Counter`, and `defaultdict`.
- The program is verified only on the visible example before held-out execution; visible-pass hidden-wrong is expected and measured.
- Exact table matching is strict after converting cells to strings.
"""
    (ROOT / "reports" / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(overall, indent=2, sort_keys=True))
    print(json.dumps(ov, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
