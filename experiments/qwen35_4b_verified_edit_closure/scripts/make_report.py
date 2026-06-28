#!/usr/bin/env python
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
EVAL_DIR = REPORT_DIR / "eval"
FIG_DIR = REPORT_DIR / "figures"


SPLITS = {
    "IID": {
        "baseline": EVAL_DIR / "static60_iid.json",
        "closure": EVAL_DIR / "edit_closure_iid.json",
    },
    "Support": {
        "baseline": EVAL_DIR / "static60_support.json",
        "closure": EVAL_DIR / "edit_closure_support.json",
    },
    "Ceiling": {
        "baseline": EVAL_DIR / "static60_ceiling.json",
        "closure": EVAL_DIR / "edit_closure_ceiling.json",
    },
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_text(metric: dict[str, Any]) -> str:
    return f"{metric['successes']}/{metric['records']} ({metric['rate'] * 100:.1f}%)"


def metric_success(metric: dict[str, Any]) -> int:
    return int(metric["successes"])


def pct(metric: dict[str, Any]) -> float:
    return float(metric["rate"]) * 100


def p90(values: list[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(0.9 * (len(ordered) - 1)))
    return float(ordered[index])


def summarize_counts(rows: list[dict[str, Any]]) -> dict[str, float]:
    generated = [int(row["closure_generated_count"]) for row in rows]
    valid = [int(row["closure_valid_count"]) for row in rows]
    return {
        "generated_median": statistics.median(generated) if generated else 0.0,
        "generated_p90": p90(generated),
        "valid_median": statistics.median(valid) if valid else 0.0,
        "valid_p90": p90(valid),
    }


def make_ceiling_overall_chart(closure: dict[str, Any]) -> None:
    overall = closure["summary"]["overall"]
    labels = ["Baseline", "Closure", "Conservative", "Strict", "Oracle"]
    keys = [
        "base_rerank_hidden_all",
        "closure_hidden_all",
        "conservative_closure_hidden_all",
        "strict_closure_hidden_all",
        "closure_oracle_hidden_all",
    ]
    values = [metric_success(overall[key]) for key in keys]
    records = overall["base_rerank_hidden_all"]["records"]

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2"]
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylim(0, records)
    ax.set_ylabel("Hidden all-pass records")
    ax.set_title("Ceiling Hidden Success")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1, str(value), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ceiling_hidden_success.png", dpi=180)
    plt.close(fig)


def make_family_chart(closure: dict[str, Any]) -> None:
    by_family = closure["summary"]["by_family"]
    families = sorted(by_family)
    y = list(range(len(families)))
    width = 0.2
    series = [
        ("Baseline", "base_rerank_hidden_all", "#4C78A8", -1.5 * width),
        ("Closure", "closure_hidden_all", "#F58518", -0.5 * width),
        ("Strict", "strict_closure_hidden_all", "#E45756", 0.5 * width),
        ("Oracle", "closure_oracle_hidden_all", "#B279A2", 1.5 * width),
    ]

    fig, ax = plt.subplots(figsize=(10, 7.2))
    for label, key, color, offset in series:
        values = [metric_success(by_family[family][key]) for family in families]
        ax.barh([item + offset for item in y], values, height=width, label=label, color=color)
    ax.set_yticks(y)
    ax.set_yticklabels(families)
    ax.set_xlim(0, 12)
    ax.set_xlabel("Hidden all-pass records out of 12")
    ax.set_title("Ceiling Hidden Success by Family")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "closure_by_family.png", dpi=180)
    plt.close(fig)


def make_candidate_count_chart(closure: dict[str, Any]) -> None:
    rows = closure["rows"]
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_family.setdefault(row["family"], []).append(row)

    families = sorted(by_family)
    generated = [statistics.mean(row["closure_generated_count"] for row in by_family[family]) for family in families]
    valid = [statistics.mean(row["closure_valid_count"] for row in by_family[family]) for family in families]

    x = list(range(len(families)))
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    ax.bar([item - 0.18 for item in x], generated, width=0.36, label="Generated", color="#4C78A8")
    ax.bar([item + 0.18 for item in x], valid, width=0.36, label="Valid", color="#54A24B")
    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=45, ha="right")
    ax.set_ylabel("Mean programs per record")
    ax.set_title("Ceiling Closure Search Size by Family")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "closure_candidate_counts.png", dpi=180)
    plt.close(fig)


def table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def split_summary_rows(results: dict[str, dict[str, Any]]) -> list[list[str]]:
    rows = []
    for name, data in results.items():
        baseline = data["baseline"]["summary"]["overall"]
        closure = data["closure"]["summary"]["overall"]
        rows.append(
            [
                name,
                str(closure["base_rerank_hidden_all"]["records"]),
                metric_text(baseline["greedy_hidden_all"]),
                metric_text(baseline["rerank_hidden_all"]),
                metric_text(closure["closure_hidden_all"]),
                metric_text(closure["strict_closure_hidden_all"]),
                metric_text(closure["closure_oracle_hidden_all"]),
            ]
        )
    return rows


def policy_rows(ceiling: dict[str, Any]) -> list[list[str]]:
    overall = ceiling["summary"]["overall"]
    return [
        [
            "Baseline visible rerank",
            metric_text(overall["base_rerank_hidden_all"]),
            metric_text(overall["base_rerank_visible_all"]),
            "0",
            "0",
            "0",
        ],
        [
            "Closure visible select",
            metric_text(overall["closure_hidden_all"]),
            metric_text(overall["closure_visible_all"]),
            str(overall["closure_improved_hidden"]["successes"]),
            str(overall["closure_damaged_hidden"]["successes"]),
            "120",
        ],
        [
            "Conservative accept visible gain",
            metric_text(overall["conservative_closure_hidden_all"]),
            metric_text(overall["conservative_closure_visible_all"]),
            str(overall["conservative_closure_improved_hidden"]["successes"]),
            str(overall["conservative_closure_damaged_hidden"]["successes"]),
            str(overall["conservative_closure_accepted"]["successes"]),
        ],
        [
            "Strict accept visible all",
            metric_text(overall["strict_closure_hidden_all"]),
            metric_text(overall["strict_closure_visible_all"]),
            str(overall["strict_closure_improved_hidden"]["successes"]),
            str(overall["strict_closure_damaged_hidden"]["successes"]),
            str(overall["strict_closure_accepted"]["successes"]),
        ],
        [
            "Hidden oracle diagnostic",
            metric_text(overall["closure_oracle_hidden_all"]),
            "hidden-only",
            "diagnostic",
            "diagnostic",
            "120",
        ],
    ]


def family_rows(ceiling: dict[str, Any]) -> list[list[str]]:
    by_family = ceiling["summary"]["by_family"]
    rows = []
    for family in sorted(by_family):
        metrics = by_family[family]
        rows.append(
            [
                family,
                str(metrics["base_rerank_hidden_all"]["successes"]),
                str(metrics["closure_hidden_all"]["successes"]),
                str(metrics["strict_closure_hidden_all"]["successes"]),
                str(metrics["closure_oracle_hidden_all"]["successes"]),
                str(metrics["strict_closure_accepted"]["successes"]),
                str(metrics["strict_closure_damaged_hidden"]["successes"]),
            ]
        )
    return rows


def search_rows(results: dict[str, dict[str, Any]]) -> list[list[str]]:
    rows = []
    for name, data in results.items():
        counts = summarize_counts(data["closure"]["rows"])
        rows.append(
            [
                name,
                f"{counts['generated_median']:.0f}",
                f"{counts['generated_p90']:.0f}",
                f"{counts['valid_median']:.0f}",
                f"{counts['valid_p90']:.0f}",
            ]
        )
    return rows


def write_report(results: dict[str, dict[str, Any]]) -> None:
    ceiling = results["Ceiling"]["closure"]
    overall = ceiling["summary"]["overall"]
    baseline_success = overall["base_rerank_hidden_all"]["successes"]
    closure_success = overall["closure_hidden_all"]["successes"]
    strict_success = overall["strict_closure_hidden_all"]["successes"]
    oracle_success = overall["closure_oracle_hidden_all"]["successes"]

    lines = [
        "# Qwen3.5-4B Verified DSL Edit Closure",
        "",
        "## Executive Summary",
        "",
        (
            "A fresh `Qwen/Qwen3.5-4B` LoRA adapter solved IID and support DSL evals perfectly. "
            "On the held-out ceiling split, normal visible reranking reached "
            f"{baseline_success}/120 hidden all-pass records. Bounded symbolic edit closure raised "
            f"visible-selected hidden all-pass to {closure_success}/120, while the hidden oracle inside "
            f"the same closure neighborhoods reached {oracle_success}/120."
        ),
        "",
        (
            "The main positive result is that local verified edits added real recoverable capability on "
            "hard held-out compositions. The main negative result is that six visible cases are not enough "
            "to choose safely in every family: the strict visible-all acceptance policy kept "
            f"{strict_success}/120 hidden all-pass and reduced pass-count damage to "
            f"{overall['strict_closure_damaged_hidden']['successes']} records, but it did not eliminate it."
        ),
        "",
        "## Setup",
        "",
        "- Base model: `Qwen/Qwen3.5-4B`.",
        "- Adapter: LoRA, trained for 2 epochs on 240 static DSL records.",
        "- Data: 60 IID eval records, 120 support eval records, 120 held-out ceiling records.",
        "- Each eval record has 6 visible cases and 18 hidden cases.",
        "- Normal baseline: greedy plus three sampled candidates for support and ceiling, selected by visible execution.",
        "- Closure: bounded two-round local DSL edits from up to four model candidates, selected by visible execution.",
        "- Strict policy: accept a closure program only when it reaches all visible cases and the baseline did not.",
        "- Hidden oracle: best hidden-case result inside the closure candidate set, used only as a diagnostic.",
        "",
        "## Split Results",
        "",
        table(
            ["Split", "Records", "Greedy Hidden", "Rerank Hidden", "Closure Hidden", "Strict Hidden", "Oracle Hidden"],
            split_summary_rows(results),
        ),
        "",
        "## Ceiling Policies",
        "",
        table(
            ["Policy", "Hidden All-Pass", "Visible All-Pass", "Hidden Pass-Count Improved", "Hidden Pass-Count Damaged", "Accepted"],
            policy_rows(ceiling),
        ),
        "",
        "![Ceiling hidden success](figures/ceiling_hidden_success.png)",
        "",
        "## Ceiling Families",
        "",
        table(
            ["Family", "Base", "Closure", "Strict", "Oracle", "Strict Accepted", "Strict Damaged"],
            family_rows(ceiling),
        ),
        "",
        "![Ceiling family results](figures/closure_by_family.png)",
        "",
        "## Search Budget",
        "",
        table(
            ["Split", "Generated Median", "Generated P90", "Valid Median", "Valid P90"],
            search_rows(results),
        ),
        "",
        "![Closure candidate counts](figures/closure_candidate_counts.png)",
        "",
        "## Interpretation",
        "",
        (
            "The edit closure helped most when the model produced a structurally nearby but semantically "
            "wrong program. The clearest gains were in `tuple_value_mod_label`, "
            "`sorted_index_sum_branch_label`, and `token_count_mod_length_code`. These are cases where the "
            "symbolic neighborhood contained useful repairs and visible execution usually moved selection "
            "in the right direction."
        ),
        "",
        (
            "The ceiling oracle result, 69/120, is only seven records above pure visible closure at 62/120. "
            "That means the local edit space is a real constraint, not just the selector. Families such as "
            "`sorted_join_contains_code` and `sum_len_mod_label` had zero hidden-oracle all-pass records, so "
            "the current edit operators do not generate the needed programs for those records."
        ),
        "",
        (
            "The strict policy is the better deployable readout than pure closure: it gives nearly the same "
            "hidden all-pass count as conservative closure, accepts fewer edits, and halves pass-count damage. "
            "The remaining damage is concentrated in `tuple_sum_mod_gate_label`, where visible cases are "
            "ambiguous among several plausible tuple and sum predicates."
        ),
        "",
        "## Iteration Notes",
        "",
        (
            "The first closure selector used shortest-program tie-breaking among visible-equivalent programs. "
            "That caused IID hidden regressions by selecting degenerate but visible-perfect simplifications. "
            "The selector was changed to stable first-seen tie-breaking, which preserves the nearest seed "
            "candidate under ties. After that change, IID and support closure both stayed perfect."
        ),
        "",
        (
            "A second policy layer was then added: conservative acceptance requires visible pass-count gain, "
            "and strict acceptance additionally requires all visible cases. Strict acceptance is the cleanest "
            "summary of what a visible-only repair policy can safely claim here."
        ),
        "",
        "## Conclusion",
        "",
        (
            "This experiment supports the hypothesis that a symbolic program-edit region can amplify a small "
            "LLM's held-out executable-task performance, but not enough by itself for a step-change result. "
            "The next most direct improvement is not more LoRA training. It is stronger visible discrimination: "
            "adaptive counterexample generation or active visible-case expansion targeted at closure ties."
        ),
        "",
    ]
    (REPORT_DIR / "qwen35_4b_verified_edit_closure_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        name: {
            "baseline": load_json(paths["baseline"]),
            "closure": load_json(paths["closure"]),
        }
        for name, paths in SPLITS.items()
    }
    make_ceiling_overall_chart(results["Ceiling"]["closure"])
    make_family_chart(results["Ceiling"]["closure"])
    make_candidate_count_chart(results["Ceiling"]["closure"])
    write_report(results)
    print(REPORT_DIR / "qwen35_4b_verified_edit_closure_report.md")


if __name__ == "__main__":
    main()
