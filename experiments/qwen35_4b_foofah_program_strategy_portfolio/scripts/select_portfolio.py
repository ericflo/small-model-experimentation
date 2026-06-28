#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

POLICIES = ["direct", "first_visible_program", "consensus_2", "consensus_3"]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def visible_candidates(record: dict[str, Any], variants: list[str]) -> list[dict[str, Any]]:
    by_variant = {c["variant"]: c for c in record["program_candidates"]}
    out = []
    for variant in variants:
        cand = by_variant.get(variant)
        if cand and cand["final"]["visible_pass"] and cand["final"].get("hidden_output_key") is not None:
            out.append(cand)
    return out


def choose(record: dict[str, Any], variants: list[str], policy: str) -> tuple[str, bool]:
    direct_ok = bool(record["direct_exact"])
    if policy == "direct" or not variants:
        return "direct", direct_ok
    visible = visible_candidates(record, variants)
    if policy == "first_visible_program":
        if not visible:
            return "direct", direct_ok
        c = visible[0]
        return f"program:{c['variant']}", bool(c["final"]["hidden_exact"])
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
    visible_program = sum(any(c["final"]["visible_verified_hidden_exact"] and c["variant"] in set(variants) for c in r["program_candidates"]) for r in records)
    oracle_union = sum(r["direct_exact"] or any(c["final"]["visible_verified_hidden_exact"] and c["variant"] in set(variants) for c in r["program_candidates"]) for r in records)
    tokens = sum(r.get("direct_total_tokens", 0) + sum(c.get("total_tokens", 0) for c in r["program_candidates"] if c["variant"] in set(variants)) for r in records)
    return {
        "n": n,
        "variants": variants,
        "policy": policy,
        "exact": exact,
        "accuracy": exact / n if n else 0,
        "direct_exact": sum(r["direct_exact"] for r in records),
        "direct_accuracy": sum(r["direct_exact"] for r in records) / n if n else 0,
        "program_commits": program_commits,
        "program_commit_correct": program_correct,
        "program_commit_precision": program_correct / program_commits if program_commits else None,
        "direct_miss_recoveries": recoveries,
        "direct_correct_losses": losses,
        "oracle_visible_program": visible_program,
        "oracle_union": oracle_union,
        "oracle_union_accuracy": oracle_union / n if n else 0,
        "total_forward_tokens": tokens,
    }


def records_by_split(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["split"]].append(record)
    return dict(grouped)


def all_variants(records: list[dict[str, Any]]) -> list[str]:
    variants = []
    for record in records:
        for cand in record["program_candidates"]:
            if cand["variant"] not in variants:
                variants.append(cand["variant"])
    return variants


def greedy_order(train_records: list[dict[str, Any]], variants: list[str], max_size: int) -> list[dict[str, Any]]:
    selected: list[str] = []
    trace = []
    remaining = list(variants)
    for step in range(min(max_size, len(variants))):
        scored = []
        for variant in remaining:
            trial = selected + [variant]
            m = metrics(train_records, trial, "first_visible_program")
            scored.append(
                (
                    m["exact"],
                    m["direct_miss_recoveries"],
                    -m["direct_correct_losses"],
                    m["program_commit_precision"] if m["program_commit_precision"] is not None else -1,
                    variant,
                    m,
                )
            )
        scored.sort(reverse=True)
        best = scored[0]
        selected.append(best[4])
        remaining.remove(best[4])
        trace.append({"step": step + 1, "added": best[4], "selected": list(selected), "train_first_visible": best[5]})
    return trace


def prefix_grid(records_by_name: dict[str, list[dict[str, Any]]], order: list[str]) -> list[dict[str, Any]]:
    rows = []
    prefixes = [[]] + [order[:i] for i in range(1, len(order) + 1)]
    for split, records in records_by_name.items():
        for prefix in prefixes:
            for policy in POLICIES:
                if not prefix and policy != "direct":
                    continue
                rows.append({"split": split, "k": len(prefix), "metrics": metrics(records, prefix, policy)})
    return rows


def choose_from_dev(grid: list[dict[str, Any]]) -> dict[str, Any]:
    dev_rows = [row for row in grid if row["split"] == "dev"]
    scored = []
    for row in dev_rows:
        m = row["metrics"]
        precision = m["program_commit_precision"] if m["program_commit_precision"] is not None else 1.0
        scored.append(
            (
                m["exact"],
                -m["direct_correct_losses"],
                precision,
                -m["total_forward_tokens"],
                m["policy"],
                row["k"],
                row,
            )
        )
    scored.sort(reverse=True)
    return scored[0][-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-records", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--max-size", type=int, default=5)
    parser.add_argument("--selection-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    args = parser.parse_args()

    records = load_jsonl(args.pilot_records)
    split_records = records_by_split(records)
    variants = all_variants(records)
    order_trace = greedy_order(split_records.get("train", []), variants, args.max_size)
    order = order_trace[-1]["selected"] if order_trace else []
    grid = prefix_grid({"train": split_records.get("train", []), "dev": split_records.get("dev", [])}, order)
    chosen = choose_from_dev(grid)
    chosen_variants = chosen["metrics"]["variants"]
    selection = {
        "family_splits": load_json(args.splits),
        "candidate_variants": variants,
        "greedy_order": order,
        "selected_variants": chosen_variants,
        "selected_policy": chosen["metrics"]["policy"],
        "selected_k": len(chosen_variants),
        "selection_rule": "greedy ordered variants on train; choose prefix and selector maximizing dev exact accuracy with loss/precision/token tie-breaks",
    }
    summary = {
        "selection": selection,
        "greedy_trace": order_trace,
        "prefix_grid": grid,
        "train_selected": metrics(split_records.get("train", []), chosen_variants, selection["selected_policy"]),
        "dev_selected": metrics(split_records.get("dev", []), chosen_variants, selection["selected_policy"]),
        "train_direct": metrics(split_records.get("train", []), [], "direct"),
        "dev_direct": metrics(split_records.get("dev", []), [], "direct"),
        "train_all_variants_first_visible": metrics(split_records.get("train", []), variants, "first_visible_program"),
        "dev_all_variants_first_visible": metrics(split_records.get("dev", []), variants, "first_visible_program"),
    }
    write_json(args.selection_out, selection)
    write_json(args.summary_out, summary)
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
