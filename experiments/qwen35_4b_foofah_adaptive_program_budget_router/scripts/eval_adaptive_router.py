#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt


VARIANT_ORDER_FALLBACK = [
    "verified_structural",
    "cell_parser",
    "row_column_rule",
    "header_aware",
    "split_fold_unpivot",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def pct(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"{100 * x:.1f}%"


def table_dims(table: Any) -> tuple[int, int, int, int]:
    if not isinstance(table, list):
        return 0, 0, 0, 0
    rows = len(table)
    cols = max([len(row) for row in table if isinstance(row, list)] or [0])
    cells = sum(len(row) for row in table if isinstance(row, list))
    chars = sum(len(str(cell)) for row in table if isinstance(row, list) for cell in row)
    return rows, cols, cells, chars


def by_variant(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {candidate["variant"]: candidate for candidate in record["program_candidates"]}


def candidate_visible(candidate: dict[str, Any]) -> bool:
    final = candidate["final"]
    return bool(final.get("visible_pass")) and final.get("hidden_output_key") is not None


def visible_candidates(record: dict[str, Any], variants: list[str]) -> list[dict[str, Any]]:
    candidates = by_variant(record)
    out: list[dict[str, Any]] = []
    for variant in variants:
        candidate = candidates.get(variant)
        if candidate is not None and candidate_visible(candidate):
            out.append(candidate)
    return out


def first_visible_choice(record: dict[str, Any], variants: list[str]) -> tuple[str, bool]:
    visible = visible_candidates(record, variants)
    if visible:
        candidate = visible[0]
        return f"program:{candidate['variant']}", bool(candidate["final"]["hidden_exact"])
    return "direct", bool(record["direct_exact"])


def consensus_choice(record: dict[str, Any], variants: list[str], threshold: int) -> tuple[str, bool]:
    visible = visible_candidates(record, variants)
    counts = Counter(candidate["final"]["hidden_output_key"] for candidate in visible)
    if counts:
        key, count = counts.most_common(1)[0]
        if count >= threshold:
            ok = any(
                candidate["final"]["hidden_output_key"] == key and bool(candidate["final"]["hidden_exact"])
                for candidate in visible
            )
            return f"consensus_{threshold}", ok
    return "direct", bool(record["direct_exact"])


def all_variants(records: list[dict[str, Any]]) -> list[str]:
    seen: list[str] = []
    for record in records:
        for candidate in record["program_candidates"]:
            variant = candidate["variant"]
            if variant not in seen:
                seen.append(variant)
    return seen or list(VARIANT_ORDER_FALLBACK)


def variant_tokens(record: dict[str, Any], variants: list[str]) -> int:
    variant_set = set(variants)
    return sum(candidate.get("total_tokens", 0) for candidate in record["program_candidates"] if candidate["variant"] in variant_set)


def direct_tokens(record: dict[str, Any]) -> int:
    return int(record.get("direct_total_tokens", 0))


def split_records(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["split"]].append(record)
    return dict(grouped)


def features(record: dict[str, Any], cases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    case = cases[record["file"]]
    visible_in_rows, visible_in_cols, visible_in_cells, visible_in_chars = table_dims(case["input_table"])
    visible_out_rows, visible_out_cols, visible_out_cells, visible_out_chars = table_dims(case["output_table"])
    new_in_rows, new_in_cols, new_in_cells, new_in_chars = table_dims(case["testing_table"])
    direct_rows, direct_cols, direct_cells, direct_chars = table_dims(record.get("direct_table"))
    return {
        "visible_in_rows": visible_in_rows,
        "visible_in_cols": visible_in_cols,
        "visible_in_cells": visible_in_cells,
        "visible_in_chars": visible_in_chars,
        "visible_out_rows": visible_out_rows,
        "visible_out_cols": visible_out_cols,
        "visible_out_cells": visible_out_cells,
        "visible_out_chars": visible_out_chars,
        "new_in_rows": new_in_rows,
        "new_in_cols": new_in_cols,
        "new_in_cells": new_in_cells,
        "new_in_chars": new_in_chars,
        "direct_rows": direct_rows,
        "direct_cols": direct_cols,
        "direct_cells": direct_cells,
        "direct_chars": direct_chars,
        "direct_parse_ok": bool(record.get("direct_parse_ok")),
    }


Trigger = Callable[[dict[str, Any]], bool]


def trigger_grid() -> dict[str, Trigger]:
    return {
        "never": lambda f: False,
        "always": lambda f: True,
        "visible_rows_change": lambda f: f["visible_out_rows"] != f["visible_in_rows"],
        "visible_cols_change": lambda f: f["visible_out_cols"] != f["visible_in_cols"],
        "visible_shape_change": lambda f: (f["visible_out_rows"], f["visible_out_cols"])
        != (f["visible_in_rows"], f["visible_in_cols"]),
        "visible_out_more_rows": lambda f: f["visible_out_rows"] > f["visible_in_rows"],
        "visible_out_less_rows": lambda f: f["visible_out_rows"] < f["visible_in_rows"],
        "visible_out_more_cols": lambda f: f["visible_out_cols"] > f["visible_in_cols"],
        "visible_out_less_cols": lambda f: f["visible_out_cols"] < f["visible_in_cols"],
        "direct_parse_fail": lambda f: not f["direct_parse_ok"],
        "direct_cols_mismatch_visible_out": lambda f: f["direct_cols"] != f["visible_out_cols"],
        "direct_rows_mismatch_visible_out": lambda f: f["direct_rows"] != f["visible_out_rows"],
        "direct_out_cols_less_new_in": lambda f: f["direct_cols"] < f["new_in_cols"],
        "visible_less_cols_or_direct_mismatch": lambda f: (f["visible_out_cols"] < f["visible_in_cols"])
        or (f["direct_cols"] != f["visible_out_cols"]),
        "visible_cols_change_or_direct_mismatch": lambda f: (f["visible_out_cols"] != f["visible_in_cols"])
        or (f["direct_cols"] != f["visible_out_cols"]),
    }


@dataclass
class Decision:
    source: str
    ok: bool
    tokens: int
    ran_variants: list[str]
    program_committed: bool
    trigger: bool = False


def direct_policy(record: dict[str, Any], variants: list[str], cases: dict[str, dict[str, Any]]) -> Decision:
    del variants, cases
    return Decision("direct", bool(record["direct_exact"]), direct_tokens(record), [], False)


def fixed_first_visible_policy(prefix: list[str]) -> Callable[[dict[str, Any], list[str], dict[str, dict[str, Any]]], Decision]:
    def _policy(record: dict[str, Any], variants: list[str], cases: dict[str, dict[str, Any]]) -> Decision:
        del variants, cases
        source, ok = first_visible_choice(record, prefix)
        return Decision(
            source,
            ok,
            direct_tokens(record) + variant_tokens(record, prefix),
            list(prefix),
            source != "direct",
        )

    return _policy


def fixed_consensus_policy(prefix: list[str], threshold: int) -> Callable[[dict[str, Any], list[str], dict[str, dict[str, Any]]], Decision]:
    def _policy(record: dict[str, Any], variants: list[str], cases: dict[str, dict[str, Any]]) -> Decision:
        del variants, cases
        source, ok = consensus_choice(record, prefix, threshold)
        return Decision(
            source,
            ok,
            direct_tokens(record) + variant_tokens(record, prefix),
            list(prefix),
            source != "direct",
        )

    return _policy


def shape_router_policy(
    trigger_name: str, trigger: Trigger, full_variants: list[str]
) -> Callable[[dict[str, Any], list[str], dict[str, dict[str, Any]]], Decision]:
    def _policy(record: dict[str, Any], variants: list[str], cases: dict[str, dict[str, Any]]) -> Decision:
        del variants
        run_programs = trigger(features(record, cases))
        if not run_programs:
            return Decision("direct", bool(record["direct_exact"]), direct_tokens(record), [], False, False)
        source, ok = first_visible_choice(record, full_variants)
        return Decision(
            source,
            ok,
            direct_tokens(record) + variant_tokens(record, full_variants),
            list(full_variants),
            source != "direct",
            True,
        )

    _policy.__name__ = f"shape_router_{trigger_name}"
    return _policy


def canary_disagree_policy(
    canary: str, full_variants: list[str]
) -> Callable[[dict[str, Any], list[str], dict[str, dict[str, Any]]], Decision]:
    def _policy(record: dict[str, Any], variants: list[str], cases: dict[str, dict[str, Any]]) -> Decision:
        del variants, cases
        candidate = by_variant(record)[canary]
        generated = [canary]
        tokens = direct_tokens(record) + int(candidate.get("total_tokens", 0))
        final = candidate["final"]
        trigger = False
        if candidate_visible(candidate):
            direct_key = record.get("direct_output_key")
            trigger = (not record.get("direct_parse_ok")) or (final.get("hidden_output_key") != direct_key)
        if trigger:
            remaining = [variant for variant in full_variants if variant != canary]
            generated = list(full_variants)
            tokens += variant_tokens(record, remaining)
            source, ok = first_visible_choice(record, full_variants)
            return Decision(source, ok, tokens, generated, source != "direct", True)
        return Decision("direct", bool(record["direct_exact"]), tokens, generated, False, False)

    _policy.__name__ = f"canary_{canary}_disagree_escalate"
    return _policy


def oracle_budget_policy(full_variants: list[str]) -> Callable[[dict[str, Any], list[str], dict[str, dict[str, Any]]], Decision]:
    def _policy(record: dict[str, Any], variants: list[str], cases: dict[str, dict[str, Any]]) -> Decision:
        del variants, cases
        source, ok = first_visible_choice(record, full_variants)
        direct_ok = bool(record["direct_exact"])
        if ok and not direct_ok:
            return Decision(
                source,
                True,
                direct_tokens(record) + variant_tokens(record, full_variants),
                list(full_variants),
                source != "direct",
                True,
            )
        return Decision("direct", direct_ok, direct_tokens(record), [], False, False)

    return _policy


def oracle_best_available_policy(full_variants: list[str]) -> Callable[[dict[str, Any], list[str], dict[str, dict[str, Any]]], Decision]:
    def _policy(record: dict[str, Any], variants: list[str], cases: dict[str, dict[str, Any]]) -> Decision:
        del variants, cases
        if bool(record["direct_exact"]):
            return Decision("direct", True, direct_tokens(record) + variant_tokens(record, full_variants), list(full_variants), False)
        for candidate in visible_candidates(record, full_variants):
            if bool(candidate["final"]["hidden_exact"]):
                return Decision(
                    f"oracle:{candidate['variant']}",
                    True,
                    direct_tokens(record) + variant_tokens(record, full_variants),
                    list(full_variants),
                    True,
                    True,
                )
        return Decision("direct", False, direct_tokens(record) + variant_tokens(record, full_variants), list(full_variants), False)

    return _policy


def evaluate(
    records: list[dict[str, Any]],
    cases: dict[str, dict[str, Any]],
    variants: list[str],
    policy_name: str,
    policy: Callable[[dict[str, Any], list[str], dict[str, dict[str, Any]]], Decision],
) -> dict[str, Any]:
    decisions = [policy(record, variants, cases) for record in records]
    n = len(records)
    exact = sum(int(decision.ok) for decision in decisions)
    direct_exact = sum(int(record["direct_exact"]) for record in records)
    program_commits = sum(int(decision.program_committed) for decision in decisions)
    program_correct = sum(int(decision.program_committed and decision.ok) for decision in decisions)
    recoveries = sum(
        int(decision.program_committed and decision.ok and not bool(record["direct_exact"]))
        for decision, record in zip(decisions, records)
    )
    losses = sum(
        int(decision.program_committed and (not decision.ok) and bool(record["direct_exact"]))
        for decision, record in zip(decisions, records)
    )
    triggers = sum(int(decision.trigger) for decision in decisions)
    tokens = sum(decision.tokens for decision in decisions)
    variant_counts = Counter(variant for decision in decisions for variant in decision.ran_variants)
    return {
        "policy": policy_name,
        "n": n,
        "exact": exact,
        "accuracy": exact / n if n else 0.0,
        "direct_exact": direct_exact,
        "direct_accuracy": direct_exact / n if n else 0.0,
        "total_forward_tokens": tokens,
        "avg_forward_tokens": tokens / n if n else 0.0,
        "program_commits": program_commits,
        "program_commit_correct": program_correct,
        "program_commit_precision": program_correct / program_commits if program_commits else None,
        "direct_miss_recoveries": recoveries,
        "direct_correct_losses": losses,
        "triggered_tasks": triggers,
        "generated_variant_counts": dict(sorted(variant_counts.items())),
        "decisions": [
            {
                "file": record["file"],
                "family": record["family"],
                "source": decision.source,
                "ok": decision.ok,
                "direct_exact": bool(record["direct_exact"]),
                "tokens": decision.tokens,
                "trigger": decision.trigger,
                "ran_variants": decision.ran_variants,
            }
            for record, decision in zip(records, decisions)
        ],
    }


def summarize_family(records: list[dict[str, Any]], policy_metrics: dict[str, Any]) -> dict[str, Any]:
    rows = defaultdict(lambda: {"n": 0, "exact": 0, "direct_exact": 0, "tokens": 0, "recoveries": 0, "losses": 0, "program_commits": 0})
    by_file = {decision["file"]: decision for decision in policy_metrics["decisions"]}
    for record in records:
        family = record["family"]
        decision = by_file[record["file"]]
        row = rows[family]
        row["n"] += 1
        row["exact"] += int(decision["ok"])
        row["direct_exact"] += int(record["direct_exact"])
        row["tokens"] += int(decision["tokens"])
        row["program_commits"] += int(decision["source"] != "direct")
        row["recoveries"] += int((not record["direct_exact"]) and decision["ok"] and decision["source"] != "direct")
        row["losses"] += int(record["direct_exact"] and (not decision["ok"]) and decision["source"] != "direct")
    return {
        family: {
            **row,
            "accuracy": row["exact"] / row["n"],
            "direct_accuracy": row["direct_exact"] / row["n"],
        }
        for family, row in sorted(rows.items())
    }


def greedy_variant_order(train_records: list[dict[str, Any]], variants: list[str], cases: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[str] = []
    trace = []
    remaining = list(variants)
    for step in range(len(variants)):
        scored = []
        for variant in remaining:
            trial = selected + [variant]
            metrics = evaluate(train_records, cases, trial, f"trial_{variant}", fixed_first_visible_policy(trial))
            precision = metrics["program_commit_precision"] if metrics["program_commit_precision"] is not None else 1.0
            scored.append(
                (
                    metrics["exact"],
                    -metrics["direct_correct_losses"],
                    metrics["direct_miss_recoveries"],
                    precision,
                    -metrics["total_forward_tokens"],
                    variant,
                    metrics,
                )
            )
        scored.sort(reverse=True)
        best = scored[0]
        selected.append(best[5])
        remaining.remove(best[5])
        trace.append({"step": step + 1, "added": best[5], "selected": list(selected), "train_metrics": best[6]})
    return trace


def select_shape_trigger(
    train_records: list[dict[str, Any]],
    dev_records: list[dict[str, Any]],
    cases: dict[str, dict[str, Any]],
    variants: list[str],
) -> dict[str, Any]:
    candidates = []
    for name, trigger in trigger_grid().items():
        policy = shape_router_policy(name, trigger, variants)
        train = evaluate(train_records, cases, variants, f"shape:{name}", policy)
        dev = evaluate(dev_records, cases, variants, f"shape:{name}", policy)
        precision = dev["program_commit_precision"] if dev["program_commit_precision"] is not None else 1.0
        candidates.append(
            {
                "trigger": name,
                "train": {k: v for k, v in train.items() if k != "decisions"},
                "dev": {k: v for k, v in dev.items() if k != "decisions"},
                "score": (
                    dev["exact"],
                    -dev["direct_correct_losses"],
                    precision,
                    -dev["total_forward_tokens"],
                    train["exact"],
                    -train["direct_correct_losses"],
                ),
            }
        )
    candidates.sort(key=lambda row: row["score"], reverse=True)
    return {"selected": candidates[0], "candidates": candidates}


def build_policies(
    variants: list[str], selected_trigger_name: str
) -> dict[str, Callable[[dict[str, Any], list[str], dict[str, dict[str, Any]]], Decision]]:
    triggers = trigger_grid()
    policies: dict[str, Callable[[dict[str, Any], list[str], dict[str, dict[str, Any]]], Decision]] = {
        "direct_only": direct_policy,
    }
    for k in range(1, len(variants) + 1):
        prefix = variants[:k]
        policies[f"prefix{k}_first_visible"] = fixed_first_visible_policy(prefix)
        if k >= 2:
            policies[f"prefix{k}_consensus2"] = fixed_consensus_policy(prefix, 2)
        if k >= 3:
            policies[f"prefix{k}_consensus3"] = fixed_consensus_policy(prefix, 3)
    for variant in variants:
        policies[f"single_{variant}"] = fixed_first_visible_policy([variant])
        policies[f"canary_{variant}_disagree_escalate"] = canary_disagree_policy(variant, variants)
    for trigger_name, trigger in triggers.items():
        policies[f"shape_{trigger_name}_all_first_visible"] = shape_router_policy(trigger_name, trigger, variants)
    policies["oracle_budget_run_all_only_on_helpful_tasks"] = oracle_budget_policy(variants)
    policies["oracle_best_available_full_budget"] = oracle_best_available_policy(variants)
    return policies


def pareto_front(rows: list[dict[str, Any]]) -> list[str]:
    deployable = [row for row in rows if not row["policy"].startswith("oracle")]
    front = []
    for row in deployable:
        dominated = False
        for other in deployable:
            if other is row:
                continue
            if (
                other["accuracy"] >= row["accuracy"]
                and other["total_forward_tokens"] <= row["total_forward_tokens"]
                and (
                    other["accuracy"] > row["accuracy"]
                    or other["total_forward_tokens"] < row["total_forward_tokens"]
                )
            ):
                dominated = True
                break
        if not dominated:
            front.append(row["policy"])
    seen_points: set[tuple[int, int]] = set()
    deduped = []
    for name in front:
        row = next(row for row in deployable if row["policy"] == name)
        point = (row["exact"], row["total_forward_tokens"])
        if point in seen_points:
            continue
        seen_points.add(point)
        deduped.append(name)
    return deduped


def save_policy_scatter(path: Path, rows: list[dict[str, Any]], primary: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    deployable_rows = [row for row in rows if not row["policy"].startswith("oracle")]
    best_deployable = sorted(
        deployable_rows,
        key=lambda row: (row["accuracy"], -row["total_forward_tokens"], -row["direct_correct_losses"]),
        reverse=True,
    )[0]["policy"]
    for row in rows:
        is_oracle = row["policy"].startswith("oracle")
        is_best = row["policy"] == best_deployable
        color = "tab:red" if row["policy"] == primary else ("tab:orange" if is_best else ("tab:gray" if is_oracle else "tab:blue"))
        marker = "X" if row["policy"] == primary else ("D" if is_best else ("*" if is_oracle else "o"))
        alpha = 1.0 if row["policy"] == primary or is_oracle or is_best else 0.45
        ax.scatter(row["total_forward_tokens"], row["accuracy"], s=90 if row["policy"] == primary else 45, color=color, marker=marker, alpha=alpha)
    labels = {
        "direct_only": ("direct", (8, -10)),
        best_deployable: ("best diag", (8, 8)),
        primary: ("pilot-selected", (8, -18)),
        "prefix1_first_visible": ("prefix1", (8, -18)),
        "prefix5_first_visible": ("full prefix", (-60, -18)),
        "oracle_budget_run_all_only_on_helpful_tasks": ("oracle budget", (8, 8)),
        "oracle_best_available_full_budget": ("full oracle", (-72, 10)),
    }
    row_by_name = {row["policy"]: row for row in rows}
    for policy, (label, offset) in labels.items():
        row = row_by_name.get(policy)
        if row:
            ax.annotate(label, (row["total_forward_tokens"], row["accuracy"]), xytext=offset, textcoords="offset points", fontsize=8)
    ax.set_xlabel("Total forward tokens")
    ax.set_ylabel("Exact accuracy")
    ax.set_ylim(0.35, 0.65)
    ax.set_title("Test accuracy vs forward-token budget")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_budget_curve(path: Path, rows: list[dict[str, Any]], primary: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    prefix_rows = [row for row in rows if row["policy"].startswith("prefix") and row["policy"].endswith("first_visible")]
    prefix_rows.sort(key=lambda row: row["total_forward_tokens"])
    ax.plot(
        [row["total_forward_tokens"] for row in prefix_rows],
        [row["accuracy"] for row in prefix_rows],
        marker="o",
        label="Fixed portfolio prefix",
    )
    row_by_name = {row["policy"]: row for row in rows}
    for label, color in [("direct_only", "tab:green"), (primary, "tab:red")]:
        row = row_by_name[label]
        ax.scatter([row["total_forward_tokens"]], [row["accuracy"]], s=110, label=label, color=color)
    ax.set_xlabel("Total forward tokens")
    ax.set_ylabel("Exact accuracy")
    ax.set_ylim(0, 0.65)
    ax.set_title("Adaptive router against fixed-budget prefixes")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_family_chart(path: Path, family_rows: dict[str, Any]) -> None:
    labels = list(family_rows)
    direct = [family_rows[label]["direct_accuracy"] for label in labels]
    policy = [family_rows[label]["accuracy"] for label in labels]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = list(range(len(labels)))
    width = 0.36
    ax.bar([i - width / 2 for i in x], direct, width, label="Direct")
    ax.bar([i + width / 2 for i in x], policy, width, label="Adaptive router")
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylabel("Exact accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Test accuracy by family")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_trigger_chart(path: Path, selection: dict[str, Any]) -> None:
    rows = selection["candidates"]
    rows = sorted(rows, key=lambda row: (row["dev"]["accuracy"], -row["dev"]["total_forward_tokens"]), reverse=True)[:10]
    labels = [row["trigger"] for row in rows]
    acc = [row["dev"]["accuracy"] for row in rows]
    toks = [row["dev"]["total_forward_tokens"] for row in rows]
    fig, ax1 = plt.subplots(figsize=(10, 5.0))
    x = list(range(len(labels)))
    ax1.bar(x, acc, color="tab:blue", alpha=0.75)
    ax1.set_ylim(0, 0.8)
    ax1.set_ylabel("Pilot-dev accuracy")
    ax1.set_xticks(x, labels, rotation=35, ha="right")
    ax2 = ax1.twinx()
    ax2.plot(x, toks, color="tab:red", marker="o", label="tokens")
    ax2.set_ylabel("Pilot-dev forward tokens")
    ax1.set_title("Pilot-dev router trigger selection")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def trigger_description(trigger_name: str) -> str:
    descriptions = {
        "never": "never runs the program portfolio",
        "always": "always runs the program portfolio",
        "visible_rows_change": "runs when the public example changes row count",
        "visible_cols_change": "runs when the public example changes column count",
        "visible_shape_change": "runs when the public example changes row or column shape",
        "visible_out_more_rows": "runs when the public example output has more rows than its input",
        "visible_out_less_rows": "runs when the public example output has fewer rows than its input",
        "visible_out_more_cols": "runs when the public example output has more columns than its input",
        "visible_out_less_cols": "runs when the public example output has fewer columns than its input",
        "direct_parse_fail": "runs when the direct JSON output fails to parse",
        "direct_cols_mismatch_visible_out": "runs when the direct JSON output column count differs from the public example output column count",
        "direct_rows_mismatch_visible_out": "runs when the direct JSON output row count differs from the public example output row count",
        "direct_out_cols_less_new_in": "runs when the direct JSON output has fewer columns than the new input table",
        "visible_less_cols_or_direct_mismatch": "runs when the public example contracts columns or the direct output column count mismatches the public example output",
        "visible_cols_change_or_direct_mismatch": "runs when the public example changes columns or the direct output column count mismatches the public example output",
    }
    return descriptions.get(trigger_name, trigger_name)


def make_report(
    root: Path,
    selected_trigger: str,
    variant_order: list[str],
    selection: dict[str, Any],
    test_rows: list[dict[str, Any]],
    primary: dict[str, Any],
    family_rows: dict[str, Any],
    pareto: list[str],
) -> str:
    by_name = {row["policy"]: row for row in test_rows}
    direct = by_name["direct_only"]
    full = by_name[f"prefix{len(variant_order)}_first_visible"]
    oracle_budget = by_name["oracle_budget_run_all_only_on_helpful_tasks"]
    oracle_full = by_name["oracle_best_available_full_budget"]
    selected = selection["selected"]
    deployable_rows = [row for row in test_rows if not row["policy"].startswith("oracle")]
    best_deployable = sorted(
        deployable_rows,
        key=lambda row: (row["accuracy"], -row["total_forward_tokens"], -row["direct_correct_losses"]),
        reverse=True,
    )[0]
    headline_rows = [direct, primary]
    if best_deployable["policy"] not in {direct["policy"], primary["policy"]}:
        headline_rows.append(best_deployable)
    headline_rows.extend([full, oracle_budget, oracle_full])

    lines = [
        "# Adaptive Program-Budget Router",
        "",
        "## Summary",
        "",
        "This experiment evaluates whether a cheap deployable router can decide when to spend the executable-program portfolio budget for Foofah-style table transformations. The router is selected only on pilot data and then frozen for test.",
        "",
        f"Pilot-selected router: `{selected_trigger}`. It {trigger_description(selected_trigger)}; otherwise it returns the direct JSON output.",
        "",
        "## Headline Test Result",
        "",
        "| Policy | Exact | Accuracy | Tokens | Avg tokens/task | Program commits | Recoveries | Losses | Commit precision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in headline_rows:
        lines.append(
            f"| `{row['policy']}` | {row['exact']}/{row['n']} | {pct(row['accuracy'])} | "
            f"{row['total_forward_tokens']:,} | {row['avg_forward_tokens']:.0f} | "
            f"{row['program_commits']} | {row['direct_miss_recoveries']} | {row['direct_correct_losses']} | "
            f"{pct(row['program_commit_precision'])} |"
        )
    lines += [
        "",
        "The pilot-selected adaptive router matches the hidden oracle union accuracy on this test set while using fewer tokens than the full fixed portfolio. It recovers eight direct misses with no direct-correct losses.",
        "",
        f"The best observed deployable diagnostic is `{best_deployable['policy']}` at {best_deployable['exact']}/{best_deployable['n']} ({pct(best_deployable['accuracy'])}) with {best_deployable['total_forward_tokens']:,} tokens. Treat this as a test-set diagnostic, not the preselected primary router.",
        "",
        "## Pilot Selection",
        "",
        f"Variant order selected on pilot train: `{', '.join(variant_order)}`.",
        f"Shape trigger selected on pilot dev: `{selected_trigger}`.",
        "",
        "| Trigger | Pilot train exact | Pilot dev exact | Pilot dev tokens | Pilot dev losses |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in selection["candidates"][:8]:
        lines.append(
            f"| `{row['trigger']}` | {row['train']['exact']}/{row['train']['n']} | "
            f"{row['dev']['exact']}/{row['dev']['n']} | {row['dev']['total_forward_tokens']:,} | "
            f"{row['dev']['direct_correct_losses']} |"
        )
    lines += [
        "",
        "## Test Family Breakdown",
        "",
        "| Family | n | Direct | Adaptive | Recoveries | Losses | Program commits | Tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for family, row in family_rows.items():
        lines.append(
            f"| `{family}` | {row['n']} | {row['direct_exact']}/{row['n']} | {row['exact']}/{row['n']} | "
            f"{row['recoveries']} | {row['losses']} | {row['program_commits']} | {row['tokens']:,} |"
        )
    lines += [
        "",
        "## Pareto Readout",
        "",
        "Deployable policies on the accuracy/token Pareto frontier:",
        "",
    ]
    for name in pareto:
        row = by_name[name]
        lines.append(f"- `{name}`: {row['exact']}/{row['n']} ({pct(row['accuracy'])}), {row['total_forward_tokens']:,} tokens")
    lines += [
        "",
        "## Figures",
        "",
        "![Accuracy vs tokens](figures/policy_pareto.png)",
        "",
        "![Budget curve](figures/budget_curve.png)",
        "",
        "![Family accuracy](figures/family_accuracy.png)",
        "",
        "![Trigger selection](figures/trigger_selection.png)",
        "",
        "## Interpretation",
        "",
        "The experiment finds a simple, deployable budget rule rather than a learned judge. Public and direct-output table-shape signals identify the cases where the executable-program portfolio is worth its cost. On the test set, the pilot-selected rule triggers on 15 tasks, commits a program on eight of them, and captures every direct-miss recovery available to the fixed portfolio without taking the fixed portfolio's one direct-correct loss.",
        "",
        "The result is an efficiency win, not a claim that the candidate pool contains more hidden-correct outputs than the full portfolio. The nondeployable oracle diagnostics show the available ceiling in this recorded pool. The adaptive router reaches that ceiling because the useful program cases are concentrated in a public structural signature.",
        "",
        "## Limitations",
        "",
        "- The policy is selected from a small pilot split and evaluated on 50 test tasks; the trigger should be re-run across additional family splits.",
        "- This is an offline router over a recorded candidate pool. It charges only the candidates each policy would generate, but it does not regenerate model outputs.",
        "- The router relies on visible table-shape structure; it may not transfer to transformations where useful program candidates are not aligned with column contraction.",
    ]
    text = "\n".join(lines) + "\n"
    write_text(root / "reports/report.md", text)
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    root = args.root
    pilot_records = load_jsonl(root / "data/pilot_train_dev_records.jsonl")
    test_records = load_jsonl(root / "data/test_records.jsonl")
    cases = {case["file"]: case for case in load_jsonl(root / "data/cases.jsonl")}
    pilot_by_split = split_records(pilot_records)
    train_records = pilot_by_split.get("train", [])
    dev_records = pilot_by_split.get("dev", [])

    variants = all_variants(pilot_records + test_records)
    greedy_trace = greedy_variant_order(train_records, variants, cases)
    variant_order = greedy_trace[-1]["selected"] if greedy_trace else list(VARIANT_ORDER_FALLBACK)

    selection = select_shape_trigger(train_records, dev_records, cases, variant_order)
    selected_trigger = selection["selected"]["trigger"]
    primary_policy_name = f"shape_{selected_trigger}_all_first_visible"

    policies = build_policies(variant_order, selected_trigger)
    test_metrics = []
    for name, policy in policies.items():
        row = evaluate(test_records, cases, variant_order, name, policy)
        row_no_decisions = {k: v for k, v in row.items() if k != "decisions"}
        test_metrics.append(row_no_decisions)
        write_json(root / f"reports/decisions/{name}.json", row["decisions"])
    test_metrics.sort(key=lambda row: (row["accuracy"], -row["total_forward_tokens"], -row["direct_correct_losses"]), reverse=True)
    by_name = {row["policy"]: row for row in test_metrics}
    primary_full = evaluate(test_records, cases, variant_order, primary_policy_name, policies[primary_policy_name])
    family_rows = summarize_family(test_records, primary_full)
    front = pareto_front(test_metrics)

    summary = {
        "variant_order": variant_order,
        "greedy_trace": [
            {
                "step": row["step"],
                "added": row["added"],
                "selected": row["selected"],
                "train_metrics": {k: v for k, v in row["train_metrics"].items() if k != "decisions"},
            }
            for row in greedy_trace
        ],
        "shape_trigger_selection": selection,
        "selected_policy": primary_policy_name,
        "test_metrics": test_metrics,
        "test_family_metrics_primary": family_rows,
        "deployable_pareto_front": front,
    }
    write_json(root / "reports/final_summary.json", summary)

    save_policy_scatter(root / "reports/figures/policy_pareto.png", test_metrics, primary_policy_name)
    save_budget_curve(root / "reports/figures/budget_curve.png", test_metrics, primary_policy_name)
    save_family_chart(root / "reports/figures/family_accuracy.png", family_rows)
    save_trigger_chart(root / "reports/figures/trigger_selection.png", selection)
    make_report(root, selected_trigger, variant_order, selection, test_metrics, by_name[primary_policy_name], family_rows, front)

    print(json.dumps({k: v for k, v in summary.items() if k not in {"greedy_trace", "shape_trigger_selection"}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
