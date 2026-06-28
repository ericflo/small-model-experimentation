#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

POLICIES = ["direct", "first_visible_program", "consensus_2", "consensus_3"]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def visible_candidates(record: dict[str, Any], variants: list[str]) -> list[dict[str, Any]]:
    order = {v: i for i, v in enumerate(variants)}
    return sorted(
        [
            c
            for c in record["program_candidates"]
            if c["variant"] in order and c["final"]["visible_pass"] and c["final"].get("hidden_output_key") is not None
        ],
        key=lambda c: order[c["variant"]],
    )


def choose(record: dict[str, Any], variants: list[str], policy: str) -> tuple[str, bool]:
    direct_ok = bool(record["direct_exact"])
    if policy == "direct" or not variants:
        return "direct", direct_ok
    visible = visible_candidates(record, variants)
    if policy == "first_visible_program":
        if visible:
            c = visible[0]
            return f"program:{c['variant']}", bool(c["final"]["hidden_exact"])
        return "direct", direct_ok
    if policy in {"consensus_2", "consensus_3"}:
        threshold = int(policy.rsplit("_", 1)[1])
        counts = Counter(c["final"]["hidden_output_key"] for c in visible)
        if counts:
            key, count = counts.most_common(1)[0]
            if count >= threshold:
                ok = any(c["final"]["hidden_output_key"] == key and c["final"]["hidden_exact"] for c in visible)
                return policy, ok
        return "direct", direct_ok
    raise ValueError(policy)


def metrics(records: list[dict[str, Any]], variants: list[str], policy: str) -> dict[str, Any]:
    n = len(records)
    exact = 0
    program_commits = 0
    program_correct = 0
    recoveries = 0
    losses = 0
    for record in records:
        source, ok = choose(record, variants, policy)
        exact += int(ok)
        direct_ok = bool(record["direct_exact"])
        if source.startswith("program") or source.startswith("consensus"):
            program_commits += 1
            program_correct += int(ok)
            recoveries += int((not direct_ok) and ok)
            losses += int(direct_ok and not ok)
    variant_set = set(variants)
    oracle_visible = sum(any(c["variant"] in variant_set and c["final"]["visible_verified_hidden_exact"] for c in r["program_candidates"]) for r in records)
    oracle_union = sum(r["direct_exact"] or any(c["variant"] in variant_set and c["final"]["visible_verified_hidden_exact"] for c in r["program_candidates"]) for r in records)
    tokens = sum(r.get("direct_total_tokens", 0) + sum(c.get("total_tokens", 0) for c in r["program_candidates"] if c["variant"] in variant_set) for r in records)
    return {
        "n": n,
        "exact": exact,
        "accuracy": exact / n if n else 0,
        "direct_exact": sum(r["direct_exact"] for r in records),
        "direct_accuracy": sum(r["direct_exact"] for r in records) / n if n else 0,
        "program_commits": program_commits,
        "program_commit_correct": program_correct,
        "program_commit_precision": program_correct / program_commits if program_commits else None,
        "direct_miss_recoveries": recoveries,
        "direct_correct_losses": losses,
        "oracle_visible_program": oracle_visible,
        "oracle_union": oracle_union,
        "oracle_union_accuracy": oracle_union / n if n else 0,
        "total_forward_tokens": tokens,
    }


def by_split(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["split"]].append(record)
    return dict(grouped)


def all_variants(records: list[dict[str, Any]]) -> list[str]:
    variants = []
    for record in records:
        for c in record["program_candidates"]:
            if c["variant"] not in variants:
                variants.append(c["variant"])
    return variants


def variant_quality(records: list[dict[str, Any]], variants: list[str]) -> list[dict[str, Any]]:
    rows = []
    for variant in variants:
        candidates = []
        for record in records:
            candidates.extend(c for c in record["program_candidates"] if c["variant"] == variant)
        visible = [c for c in candidates if c["final"]["visible_pass"]]
        correct_visible = [c for c in visible if c["final"]["hidden_exact"]]
        initial_visible = [c for c in candidates if c["attempts"][0]["visible_pass"]]
        rows.append(
            {
                "variant": variant,
                "n": len(candidates),
                "visible_pass": len(visible),
                "visible_pass_rate": len(visible) / len(candidates) if candidates else 0,
                "visible_precision": len(correct_visible) / len(visible) if visible else None,
                "initial_visible_pass": len(initial_visible),
                "repair_added_visible": len(visible) - len(initial_visible),
            }
        )
    return rows


def family_metrics(records: list[dict[str, Any]], variants: list[str], policy: str) -> dict[str, Any]:
    out = {}
    for family in sorted({r["family"] for r in records}):
        group = [r for r in records if r["family"] == family]
        out[family] = metrics(group, variants, policy)
    return out


def save_accuracy_chart(path: Path, split_metrics: dict[str, dict[str, Any]]) -> None:
    labels = list(split_metrics)
    direct = [split_metrics[s]["direct"]["accuracy"] for s in labels]
    selected = [split_metrics[s]["selected"]["accuracy"] for s in labels]
    oracle = [split_metrics[s]["selected_oracle"]["oracle_union_accuracy"] for s in labels]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.25
    ax.bar([i - width for i in x], direct, width, label="Direct")
    ax.bar(list(x), selected, width, label="Selected portfolio")
    ax.bar([i + width for i in x], oracle, width, label="Direct OR program oracle")
    ax.set_xticks(list(x), labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Exact held-out accuracy")
    ax.set_title("Accuracy by split")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_prefix_chart(path: Path, selection_summary: dict[str, Any]) -> None:
    rows = selection_summary["prefix_grid"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for split in ["train", "dev"]:
        for policy in ["first_visible_program", "consensus_2"]:
            xs, ys = [], []
            for row in rows:
                m = row["metrics"]
                if row["split"] == split and m["policy"] == policy:
                    xs.append(row["k"])
                    ys.append(m["accuracy"])
            if xs:
                ax.plot(xs, ys, marker="o", label=f"{split} {policy}")
    ax.set_xlabel("Portfolio prefix size")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Portfolio prefix search")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_variant_chart(path: Path, rows: list[dict[str, Any]]) -> None:
    labels = [r["variant"] for r in rows]
    visible = [r["visible_pass_rate"] for r in rows]
    precision = [r["visible_precision"] or 0 for r in rows]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    width = 0.35
    ax.bar([i - width / 2 for i in x], visible, width, label="Visible pass rate")
    ax.bar([i + width / 2 for i in x], precision, width, label="Visible-pass hidden precision")
    ax.set_xticks(list(x), labels, rotation=35, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Program strategy quality")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_policy_chart(path: Path, policies: dict[str, dict[str, Any]]) -> None:
    labels = list(policies)
    accuracy = [policies[p]["accuracy"] for p in labels]
    precision = [policies[p]["program_commit_precision"] or 0 for p in labels]
    commits = [policies[p]["program_commits"] for p in labels]
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    x = range(len(labels))
    ax1.bar(x, accuracy, label="Accuracy", color="#4777b3")
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Accuracy / precision")
    ax1.plot(x, precision, marker="o", color="#c2514a", label="Program precision")
    ax2 = ax1.twinx()
    ax2.plot(x, commits, marker="s", color="#4b8f5a", label="Program commits")
    ax2.set_ylabel("Program commits")
    ax1.set_xticks(list(x), labels, rotation=20, ha="right")
    ax1.set_title("Held-out selector tradeoff")
    ax1.grid(axis="y", alpha=0.25)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-records", type=Path, required=True)
    parser.add_argument("--test-records", type=Path, required=True)
    parser.add_argument("--selection-summary", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, required=True)
    args = parser.parse_args()

    pilot_records = load_jsonl(args.pilot_records)
    test_records = load_jsonl(args.test_records)
    selection_summary = load_json(args.selection_summary)
    selection = load_json(args.selection)
    selected_variants = selection["selected_variants"]
    selected_policy = selection["selected_policy"]

    split_records = by_split(pilot_records)
    split_records["test"] = test_records
    split_metrics: dict[str, dict[str, Any]] = {}
    for split in ["train", "dev", "test"]:
        records = split_records.get(split, [])
        split_metrics[split] = {
            "direct": metrics(records, [], "direct"),
            "selected": metrics(records, selected_variants, selected_policy),
            "selected_first_visible": metrics(records, selected_variants, "first_visible_program"),
            "selected_consensus_2": metrics(records, selected_variants, "consensus_2"),
            "selected_oracle": metrics(records, selected_variants, "first_visible_program"),
        }

    test_policy_metrics = {
        policy: metrics(test_records, selected_variants, policy)
        for policy in POLICIES
        if selected_variants or policy == "direct"
    }
    pilot_variants = all_variants(pilot_records)
    variant_rows = variant_quality(pilot_records + test_records, sorted(set(pilot_variants + selected_variants)))
    test_family = family_metrics(test_records, selected_variants, selected_policy)

    save_accuracy_chart(args.figures_dir / "accuracy_by_split.png", split_metrics)
    save_prefix_chart(args.figures_dir / "portfolio_prefix_search.png", selection_summary)
    save_variant_chart(args.figures_dir / "variant_quality.png", variant_rows)
    save_policy_chart(args.figures_dir / "test_selector_tradeoff.png", test_policy_metrics)

    final = {
        "selection": selection,
        "split_metrics": split_metrics,
        "test_policy_metrics": test_policy_metrics,
        "variant_quality": variant_rows,
        "test_family_metrics": test_family,
    }
    write_json(args.summary_out, final)

    test_selected = split_metrics["test"]["selected"]
    test_direct = split_metrics["test"]["direct"]
    test_oracle = split_metrics["test"]["selected_oracle"]
    lines = [
        "# Qwen3.5-4B Foofah Program Strategy Portfolio",
        "",
        "## Question",
        "",
        "Can a small, searched portfolio of executable program-generation strategies improve Foofah table-transformation accuracy over direct JSON generation on held-out task families?",
        "",
        "The experiment searches strategy prompts on train/dev families, freezes the selected portfolio, and evaluates it once on held-out families. Hidden answers are used for measurement and for train/dev strategy selection only, never as inputs to generation.",
        "",
        "## Selected Portfolio",
        "",
        f"- Selected variants: `{', '.join(selected_variants) if selected_variants else 'none'}`",
        f"- Selected policy: `{selected_policy}`",
        f"- Selection rule: {selection['selection_rule']}",
        "",
        "## Main Held-Out Result",
        "",
        "| arm | exact | rate | program commits | program precision | direct-miss recoveries | direct-correct losses | forward tokens |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| Direct JSON | {test_direct['exact']}/{test_direct['n']} | {pct(test_direct['accuracy'])} | 0 | - | 0 | 0 | {test_direct['total_forward_tokens']} |",
        f"| Selected portfolio | {test_selected['exact']}/{test_selected['n']} | {pct(test_selected['accuracy'])} | {test_selected['program_commits']} | {pct(test_selected['program_commit_precision']) if test_selected['program_commit_precision'] is not None else '-'} | {test_selected['direct_miss_recoveries']} | {test_selected['direct_correct_losses']} | {test_selected['total_forward_tokens']} |",
        f"| Direct OR selected-program oracle | {test_oracle['oracle_union']}/{test_oracle['n']} | {pct(test_oracle['oracle_union_accuracy'])} | - | - | - | - | - |",
        "",
        "## Held-Out Selector Tradeoff",
        "",
        "| selector | exact | rate | program commits | program precision | recoveries | losses |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for policy, m in test_policy_metrics.items():
        lines.append(
            f"| `{policy}` | {m['exact']}/{m['n']} | {pct(m['accuracy'])} | {m['program_commits']} | "
            f"{pct(m['program_commit_precision']) if m['program_commit_precision'] is not None else '-'} | "
            f"{m['direct_miss_recoveries']} | {m['direct_correct_losses']} |"
        )
    lines.extend(
        [
            "",
            "## Figures",
            "",
            "- `reports/figures/accuracy_by_split.png`",
            "- `reports/figures/portfolio_prefix_search.png`",
            "- `reports/figures/variant_quality.png`",
            "- `reports/figures/test_selector_tradeoff.png`",
            "",
            "## Read",
            "",
            "This report is generated directly from the recorded model outputs and safe-execution results. The key read is whether the frozen portfolio improves the held-out-family accuracy/token tradeoff over direct JSON generation, and whether any gain comes from real direct-miss recoveries rather than sacrificing direct-correct cases.",
            "",
            "## Caveats",
            "",
            "- Family-heldout evaluation is stricter than a random case split but still uses one external benchmark.",
            "- Program candidates are verified on the visible example only; hidden answers are used solely for evaluation and train/dev strategy selection.",
            "- The selected strategy portfolio is greedy and small; it is not a global optimum over all possible prompts.",
            "- Exact table matching normalizes all cells to strings and requires exact row/column equality.",
        ]
    )
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "summary": str(args.summary_out)}, indent=2))


if __name__ == "__main__":
    main()

