#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.code_env import run_python_script, static_safety_check  # noqa: E402
from src.coverage_utils import EXPERIMENT, add_usage, empty_usage, summarize_records, write_manifest  # noqa: E402
from src.jsonl import load_jsonl, write_json, write_jsonl  # noqa: E402


def literal_variants(value: Any) -> list[Any]:
    if isinstance(value, bool):
        variants = [value, not value]
    elif isinstance(value, int):
        variants = [value, 0, 1, -1, 2, value + 1, value - 1, value * 2, abs(value) + 3]
    elif isinstance(value, float):
        variants = [value, 0.0, 1.0, -1.0, value + 1.0]
    elif isinstance(value, str):
        variants = [value, "", value[::-1], value.lower(), value.upper(), value + value[:1], "abc", "AaBb", "a_b"]
    elif isinstance(value, (list, tuple)):
        seq = list(value)
        variants = [type(value)(seq), type(value)([]), type(value)(list(reversed(seq)))]
        try:
            variants.append(type(value)(sorted(seq)))
        except Exception:
            pass
        if seq:
            variants.append(type(value)(seq[: max(1, len(seq) // 2)]))
            variants.append(type(value)(seq + [seq[0]]))
        else:
            variants.append(type(value)([0]))
    else:
        variants = [value]
    out = []
    seen = set()
    for item in variants:
        key = repr(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out[:8]


def call_from_args(entry_point: str, args: list[Any]) -> str | None:
    try:
        call = ast.Call(
            func=ast.Name(id=entry_point, ctx=ast.Load()),
            args=[
                ast.Constant(arg)
                if isinstance(arg, (str, int, float, bool, type(None)))
                else ast.parse(repr(arg), mode="eval").body
                for arg in args
            ],
            keywords=[],
        )
        ast.fix_missing_locations(call)
        return ast.unparse(call)
    except Exception:
        return None


def probe_calls(record: dict[str, Any], limit: int) -> list[str]:
    calls: list[str] = []
    seen: set[str] = set()
    for case in record.get("public_cases", []):
        try:
            tree = ast.parse(case.get("call_expr", ""), mode="eval").body
        except SyntaxError:
            continue
        if not isinstance(tree, ast.Call) or tree.keywords:
            continue
        try:
            args = [ast.literal_eval(arg) for arg in tree.args]
        except Exception:
            continue
        variants_by_arg = [literal_variants(arg) for arg in args]
        candidate_args = [args]
        for idx, variants in enumerate(variants_by_arg):
            for variant in variants:
                row = list(args)
                row[idx] = variant
                candidate_args.append(row)
        for combo in itertools.product(*[vals[:4] for vals in variants_by_arg[: min(3, len(variants_by_arg))]]):
            row = list(args)
            for idx, variant in enumerate(combo):
                row[idx] = variant
            candidate_args.append(row)
        for row in candidate_args:
            expr = call_from_args(record["entry_point"], row)
            if expr and expr not in seen:
                seen.add(expr)
                calls.append(expr)
            if len(calls) >= limit:
                return calls
    return calls


def execute_calls(code: str, record: dict[str, Any], calls: list[str]) -> list[str]:
    if not calls:
        return []
    safe, reason = static_safety_check(code)
    if not safe:
        return [f"UNSAFE:{reason}" for _ in calls]
    script = f"""
import collections
import functools
import heapq
import itertools
import math
import re
import string
from typing import *

{record.get("setup_code", "")}

{code}

calls = {calls!r}
rows = []
for call in calls:
    try:
        rows.append("OK:" + repr(eval(call, globals())))
    except BaseException as exc:
        rows.append("ERR:" + type(exc).__name__)
print(json.dumps(rows))
"""
    result = run_python_script("import json\n" + script, timeout_s=float(record.get("timeout_s", 5.0)))
    if not result["ok"]:
        return ["ERR:runtime"] * len(calls)
    try:
        rows = json.loads(result["stdout"].strip().splitlines()[-1])
        if isinstance(rows, list) and len(rows) == len(calls):
            return [str(row) for row in rows]
    except Exception:
        pass
    return ["ERR:bad_json"] * len(calls)


def entropy(values: list[str]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    total = len(values)
    return -sum((count / total) * math.log(count / total + 1e-12) for count in counts.values())


def choose_disagreement_calls(calls: list[str], visible_candidates: list[dict[str, Any]], budget: int) -> list[int]:
    scored = []
    for idx, _call in enumerate(calls):
        values = [candidate["probe_outputs"][idx] for candidate in visible_candidates]
        scored.append((entropy(values), len(set(values)), idx))
    scored.sort(key=lambda row: (-row[0], -row[1], row[2]))
    return [idx for ent, distinct, idx in scored[:budget] if distinct > 1 and ent > 0.0]


def clone_candidate(candidate: dict[str, Any], pool_name: str, index: int) -> dict[str, Any]:
    row = dict(candidate)
    row["pool_name"] = pool_name
    row["pool_candidate_id"] = f"{pool_name}:{candidate.get('candidate_id', index)}:{index}"
    row["source_id"] = candidate.get("retrieved_library_id") or candidate.get("source") or row["pool_candidate_id"]
    return row


def records_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["record_id"]: row for row in rows}


def pool_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}
    visible_pass = 0
    visible_hidden_wrong = 0
    total_candidates = 0
    total_visible = 0
    total_hidden = 0
    coverage = 0
    for record in records:
        candidates = record.get("candidates", [])
        total_candidates += len(candidates)
        total_visible += sum(1 for cand in candidates if cand.get("visible_all_pass"))
        total_hidden += sum(1 for cand in candidates if cand.get("full_pass"))
        coverage += int(any(cand.get("full_pass") for cand in candidates))
        for cand in candidates:
            if cand.get("visible_all_pass"):
                visible_pass += 1
                visible_hidden_wrong += int(not cand.get("full_pass"))
    return {
        "records": len(records),
        "candidate_count_mean": total_candidates / len(records),
        "visible_pass_mean": total_visible / len(records),
        "hidden_correct_mean": total_hidden / len(records),
        "coverage": coverage / len(records),
        "visible_pass_count": visible_pass,
        "visible_hidden_wrong_count": visible_hidden_wrong,
        "visible_hidden_wrong_rate": visible_hidden_wrong / visible_pass if visible_pass else 0.0,
    }


def choose_first(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    visible = [cand for cand in candidates if cand.get("visible_all_pass")]
    if not visible:
        return None
    return sorted(visible, key=lambda cand: (cand.get("retrieved_rank", 999), cand.get("order", 0)))[0]


def choose_oracle(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    winners = [cand for cand in candidates if cand.get("full_pass")]
    if not winners:
        return None
    return sorted(winners, key=lambda cand: (cand.get("retrieved_rank", 999), cand.get("order", 0)))[0]


def choose_consensus(candidates: list[dict[str, Any]], min_sources: int, require_probe: bool = True) -> dict[str, Any] | None:
    visible = [
        cand
        for cand in candidates
        if cand.get("visible_all_pass") and (not require_probe or int(cand.get("consensus_probe_count", 0)) > 0)
    ]
    if not visible:
        return None
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cand in visible:
        clusters[cand.get("consensus_signature", "no_signature")].append(cand)
    ranked = []
    for signature, members in clusters.items():
        sources = {member.get("source_id") for member in members}
        source_count = len(sources)
        hidden_correct = sum(1 for member in members if member.get("full_pass"))
        avg_rank = sum(float(member.get("retrieved_rank", 999)) for member in members) / len(members)
        ranked.append((source_count, len(members), hidden_correct, -avg_rank, signature, members))
    ranked.sort(key=lambda row: (-row[0], -row[1], row[3], row[4]))
    if not ranked or ranked[0][0] < min_sources:
        return None
    members = ranked[0][5]
    chosen = sorted(members, key=lambda cand: (cand.get("retrieved_rank", 999), cand.get("order", 0)))[0]
    chosen = dict(chosen)
    chosen["consensus_source_count"] = ranked[0][0]
    chosen["consensus_cluster_size"] = ranked[0][1]
    chosen["consensus_cluster_hidden_correct_count"] = ranked[0][2]
    return chosen


def selector_row(chosen: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "selected": chosen is not None,
        "pool_candidate_id": chosen.get("pool_candidate_id") if chosen else None,
        "pool_name": chosen.get("pool_name") if chosen else None,
        "source": chosen.get("source") if chosen else None,
        "source_id": chosen.get("source_id") if chosen else None,
        "visible_all_pass": bool(chosen.get("visible_all_pass")) if chosen else False,
        "full_pass": bool(chosen.get("full_pass")) if chosen else False,
        "consensus_source_count": int(chosen.get("consensus_source_count", 1)) if chosen else 0,
        "consensus_cluster_size": int(chosen.get("consensus_cluster_size", 1)) if chosen else 0,
        "consensus_cluster_hidden_correct_count": int(chosen.get("consensus_cluster_hidden_correct_count", 0)) if chosen else 0,
    }


def selected_metrics(rows: list[dict[str, Any]], selector: str) -> dict[str, Any]:
    selected = [row["selectors"][selector] for row in rows]
    commits = [item for item in selected if item.get("selected")]
    correct = sum(1 for item in commits if item.get("full_pass"))
    wrong = sum(1 for item in commits if item.get("visible_all_pass") and not item.get("full_pass"))
    return {
        "records": len(rows),
        "commit_count": len(commits),
        "commit_rate": len(commits) / len(rows) if rows else 0.0,
        "selected_hidden_correct": correct,
        "selected_recovery_rate": correct / len(rows) if rows else 0.0,
        "selected_visible_hidden_wrong": wrong,
        "selected_false_pass_rate": wrong / len(commits) if commits else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--independent-records", type=Path, required=True)
    parser.add_argument("--same-records", type=Path, required=True)
    parser.add_argument("--direct-records", type=Path)
    parser.add_argument("--probe-pool", type=int, default=64)
    parser.add_argument("--probe-budget", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    plan_rows = load_jsonl(args.plan)
    records = {row["record"]["record_id"]: row["record"] for row in plan_rows}
    independent_rows = load_jsonl(args.independent_records)
    same_rows = load_jsonl(args.same_records)
    independent_by_id = records_by_id(independent_rows)
    same_by_id = records_by_id(same_rows)
    out_rows = []

    for record_id, record in records.items():
        independent = [clone_candidate(cand, "independent", idx) for idx, cand in enumerate(independent_by_id[record_id].get("candidates", []))]
        same = [clone_candidate(cand, "same_neighborhood", idx) for idx, cand in enumerate(same_by_id[record_id].get("candidates", []))]
        calls = probe_calls(record, args.probe_pool)
        all_candidates = independent + same
        for candidate in all_candidates:
            candidate["probe_outputs"] = execute_calls(candidate.get("code", ""), record, calls) if candidate.get("parse_status") == "parsed" else []
        selected_call_indices_by_pool = {}
        for pool_name, pool_candidates in [("independent", independent), ("same_neighborhood", same)]:
            visible = [candidate for candidate in pool_candidates if candidate.get("visible_all_pass")]
            selected = choose_disagreement_calls(calls, visible, args.probe_budget)
            selected_call_indices_by_pool[pool_name] = selected
        for candidate in pool_candidates:
            candidate["consensus_probe_count"] = len(selected)
            if selected and candidate.get("probe_outputs"):
                candidate["consensus_signature"] = json.dumps([candidate["probe_outputs"][idx] for idx in selected], sort_keys=True)
            else:
                    candidate["consensus_signature"] = candidate.get("public_signature", "no_signature")
        selectors = {
            "first_visible_independent": selector_row(choose_first(independent)),
            "first_visible_same": selector_row(choose_first(same)),
            "consensus_independent_min2": selector_row(choose_consensus(independent, min_sources=2)),
            "consensus_independent_min3": selector_row(choose_consensus(independent, min_sources=3)),
            "consensus_same_min2": selector_row(choose_consensus(same, min_sources=2)),
            "consensus_same_min3": selector_row(choose_consensus(same, min_sources=3)),
            "oracle_independent": selector_row(choose_oracle(independent)),
            "oracle_same": selector_row(choose_oracle(same)),
            "oracle_union": selector_row(choose_oracle(independent + same)),
        }
        out_rows.append(
            {
                "record_id": record_id,
                "task_id": record["task_id"],
                "task_text": record["task_text"],
                "entry_point": record["entry_point"],
                "probe_calls": calls,
                "selected_probe_indices": selected_call_indices_by_pool,
                "independent_candidate_count": len(independent),
                "same_candidate_count": len(same),
                "independent_visible_count": sum(1 for cand in independent if cand.get("visible_all_pass")),
                "same_visible_count": sum(1 for cand in same if cand.get("visible_all_pass")),
                "independent_hidden_correct_count": sum(1 for cand in independent if cand.get("full_pass")),
                "same_hidden_correct_count": sum(1 for cand in same if cand.get("full_pass")),
                "independent_coverage": any(cand.get("full_pass") for cand in independent),
                "same_coverage": any(cand.get("full_pass") for cand in same),
                "selectors": selectors,
                "candidates": [
                    {
                        "pool_candidate_id": cand.get("pool_candidate_id"),
                        "pool_name": cand.get("pool_name"),
                        "source_id": cand.get("source_id"),
                        "source": cand.get("source"),
                        "retrieved_rank": cand.get("retrieved_rank"),
                        "visible_all_pass": cand.get("visible_all_pass"),
                        "full_pass": cand.get("full_pass"),
                        "consensus_signature": cand.get("consensus_signature"),
                    }
                    for cand in all_candidates
                ],
            }
        )

    direct_rows = load_jsonl(args.direct_records) if args.direct_records else []
    direct_summary = summarize_records(direct_rows) if direct_rows else None
    direct_usage = empty_usage()
    for row in direct_rows:
        direct_usage = add_usage(direct_usage, row.get("token_usage", {}))

    independent_records_for_summary = [{"record_id": row["record_id"], "candidates": row.get("candidates", [])} for row in independent_rows]
    same_records_for_summary = [{"record_id": row["record_id"], "candidates": row.get("candidates", [])} for row in same_rows]
    summary = {
        "experiment": EXPERIMENT,
        "records": len(out_rows),
        "probe_pool": args.probe_pool,
        "probe_budget": args.probe_budget,
        "independent_pool": pool_metrics(independent_records_for_summary),
        "same_neighborhood_pool": pool_metrics(same_records_for_summary),
        "selectors": {
            name: selected_metrics(out_rows, name)
            for name in [
                "first_visible_independent",
                "first_visible_same",
                "consensus_independent_min2",
                "consensus_independent_min3",
                "consensus_same_min2",
                "consensus_same_min3",
                "oracle_independent",
                "oracle_same",
                "oracle_union",
            ]
        },
        "direct_sample_more": {"summary": direct_summary, "token_usage": direct_usage} if direct_rows else None,
        "token_usage": {
            "independent": add_usage(empty_usage(), json.load(open(args.independent_records.with_suffix(".manifest.json"))).get("token_usage", {}))
            if args.independent_records.with_suffix(".manifest.json").exists()
            else empty_usage(),
            "same_neighborhood": add_usage(empty_usage(), json.load(open(args.same_records.with_suffix(".manifest.json"))).get("token_usage", {}))
            if args.same_records.with_suffix(".manifest.json").exists()
            else empty_usage(),
        },
        "out": str(args.out),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, out_rows)
    write_json(args.summary, summary)
    write_manifest(args.out.with_suffix(".manifest.json"), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
