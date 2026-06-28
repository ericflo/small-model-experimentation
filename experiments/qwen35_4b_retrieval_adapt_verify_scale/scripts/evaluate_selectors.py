#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import itertools
import json
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
    variants: list[Any] = []
    if isinstance(value, bool):
        variants = [value, not value]
    elif isinstance(value, int):
        variants = [value, 0, 1, -1, 2, value + 1, value - 1, value * 2, abs(value) + 3]
    elif isinstance(value, float):
        variants = [value, 0.0, 1.0, -1.0, value + 1.0]
    elif isinstance(value, str):
        variants = [value, "", value[::-1], value.lower(), value.upper(), value + value[:1], "abc"]
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
    out: list[Any] = []
    seen: set[str] = set()
    for item in variants:
        key = repr(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out[:6]


def call_from_args(entry_point: str, args: list[Any]) -> str | None:
    try:
        call = ast.Call(
            func=ast.Name(id=entry_point, ctx=ast.Load()),
            args=[ast.Constant(arg) if isinstance(arg, (str, int, float, bool, type(None))) else ast.parse(repr(arg), mode="eval").body for arg in args],
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
        call_expr = case.get("call_expr", "")
        try:
            tree = ast.parse(call_expr, mode="eval").body
        except SyntaxError:
            continue
        if not isinstance(tree, ast.Call) or tree.keywords:
            continue
        args: list[Any] = []
        try:
            args = [ast.literal_eval(arg) for arg in tree.args]
        except Exception:
            continue
        per_arg = [literal_variants(arg) for arg in args]
        candidate_args: list[list[Any]] = [args]
        for idx, variants in enumerate(per_arg):
            for variant in variants:
                row = list(args)
                row[idx] = variant
                candidate_args.append(row)
        for combo in itertools.product(*[vals[:3] for vals in per_arg[: min(3, len(per_arg))]]):
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


def probe_signature(code: str, record: dict[str, Any], calls: list[str]) -> str:
    if not calls:
        return "no_probes"
    safe, reason = static_safety_check(code)
    if not safe:
        return f"unsafe:{reason}"
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
        value = eval(call, globals())
        rows.append("OK:" + repr(value))
    except BaseException as exc:
        rows.append("ERR:" + type(exc).__name__)
print(json.dumps(rows))
"""
    result = run_python_script("import json\n" + script, timeout_s=float(record.get("timeout_s", 5.0)))
    if not result["ok"]:
        return "runtime_error:" + str(result.get("stderr", ""))[-80:]
    try:
        rows = json.loads(result["stdout"].strip().splitlines()[-1])
    except Exception:
        return "bad_json"
    return json.dumps(rows, sort_keys=True)


def clone_candidate(candidate: dict[str, Any], pool_name: str, index: int) -> dict[str, Any]:
    row = dict(candidate)
    row["pool_candidate_id"] = f"{pool_name}:{candidate.get('candidate_id', index)}:{index}"
    row["pool_name"] = pool_name
    return row


def records_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["record_id"]: row for row in rows}


def selector_first_visible(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    visible = [cand for cand in candidates if cand.get("visible_all_pass")]
    if not visible:
        return None
    return sorted(visible, key=lambda cand: (cand.get("source_rank", 999), cand.get("pool_name", ""), cand.get("order", 0)))[0]


def selector_shortest_visible(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    visible = [cand for cand in candidates if cand.get("visible_all_pass")]
    if not visible:
        return None
    return sorted(visible, key=lambda cand: (len(cand.get("code", "")), cand.get("source_rank", 999), cand.get("order", 0)))[0]


def selector_consensus_visible(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    visible = [cand for cand in candidates if cand.get("visible_all_pass")]
    if not visible:
        return None
    counts = Counter(cand.get("probe_signature", "") for cand in visible)
    return sorted(
        visible,
        key=lambda cand: (
            -counts[cand.get("probe_signature", "")],
            cand.get("source_rank", 999),
            len(cand.get("code", "")),
            cand.get("order", 0),
        ),
    )[0]


def selector_oracle(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    winners = [cand for cand in candidates if cand.get("full_pass")]
    if not winners:
        return None
    return sorted(winners, key=lambda cand: (cand.get("source_rank", 999), cand.get("order", 0)))[0]


def pool_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}
    visible_pass = 0
    visible_hidden_wrong = 0
    total_candidates = 0
    total_visible = 0
    total_hidden_correct = 0
    for record in records:
        candidates = record.get("candidates", [])
        total_candidates += len(candidates)
        total_visible += sum(1 for cand in candidates if cand.get("visible_all_pass"))
        total_hidden_correct += sum(1 for cand in candidates if cand.get("full_pass"))
        for cand in candidates:
            if cand.get("visible_all_pass"):
                visible_pass += 1
                if not cand.get("full_pass"):
                    visible_hidden_wrong += 1
    return {
        "records": len(records),
        "candidate_count_mean": total_candidates / len(records),
        "visible_pass_mean": total_visible / len(records),
        "hidden_correct_mean": total_hidden_correct / len(records),
        "coverage": sum(1 for row in records if any(cand.get("full_pass") for cand in row.get("candidates", []))) / len(records),
        "visible_any_rate": sum(1 for row in records if any(cand.get("visible_all_pass") for cand in row.get("candidates", []))) / len(records),
        "visible_pass_count": visible_pass,
        "visible_hidden_wrong_count": visible_hidden_wrong,
        "visible_hidden_wrong_rate": visible_hidden_wrong / visible_pass if visible_pass else 0.0,
    }


def selected_metrics(rows: list[dict[str, Any]], selector: str) -> dict[str, Any]:
    selected = [row["selectors"][selector] for row in rows]
    commits = [item for item in selected if item.get("selected")]
    return {
        "records": len(rows),
        "commit_count": len(commits),
        "commit_rate": len(commits) / len(rows) if rows else 0.0,
        "selected_hidden_correct": sum(1 for item in commits if item.get("full_pass")),
        "selected_recovery_rate": sum(1 for item in commits if item.get("full_pass")) / len(rows) if rows else 0.0,
        "selected_visible_hidden_wrong": sum(1 for item in commits if item.get("visible_all_pass") and not item.get("full_pass")),
        "selected_false_pass_rate": (
            sum(1 for item in commits if item.get("visible_all_pass") and not item.get("full_pass")) / len(commits) if commits else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-records", type=Path, required=True)
    parser.add_argument("--candidate-records", type=Path, nargs="+", required=True)
    parser.add_argument("--pool-names", type=str, nargs="+", required=True)
    parser.add_argument("--max-probes", type=int, default=24)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    if len(args.candidate_records) != len(args.pool_names):
        raise ValueError("--candidate-records and --pool-names must have the same length")

    baseline = load_jsonl(args.baseline_records)
    base_by_id = records_by_id(baseline)
    base_miss_ids = {
        row["record_id"]
        for row in baseline
        if not any(candidate.get("full_pass") for candidate in row.get("candidates", []))
    }
    pool_by_record: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pool_inputs: dict[str, list[dict[str, Any]]] = {}
    for pool_index, (path, pool_name) in enumerate(zip(args.candidate_records, args.pool_names)):
        rows = load_jsonl(path)
        pool_inputs[pool_name] = rows
        for row in rows:
            if row["record_id"] not in base_miss_ids:
                continue
            for idx, candidate in enumerate(row.get("candidates", [])):
                cand = clone_candidate(candidate, pool_name, idx)
                cand["source_rank"] = pool_index
                pool_by_record[row["record_id"]].append(cand)

    out_rows: list[dict[str, Any]] = []
    for record_id in sorted(base_miss_ids, key=lambda rid: int(base_by_id[rid]["task_id"])):
        record = base_by_id[record_id]
        candidates = pool_by_record.get(record_id, [])
        calls = probe_calls(record, args.max_probes)
        for candidate in candidates:
            if candidate.get("parse_status") == "parsed":
                candidate["probe_signature"] = probe_signature(candidate.get("code", ""), record, calls)
            else:
                candidate["probe_signature"] = "parse_failed"
        selectors = {}
        for name, fn in [
            ("first_visible", selector_first_visible),
            ("shortest_visible", selector_shortest_visible),
            ("consensus_visible", selector_consensus_visible),
            ("oracle_hidden", selector_oracle),
        ]:
            chosen = fn(candidates)
            selectors[name] = {
                "selected": chosen is not None,
                "pool_candidate_id": chosen.get("pool_candidate_id") if chosen else None,
                "pool_name": chosen.get("pool_name") if chosen else None,
                "source": chosen.get("source") if chosen else None,
                "visible_all_pass": bool(chosen.get("visible_all_pass")) if chosen else False,
                "full_pass": bool(chosen.get("full_pass")) if chosen else False,
                "probe_cluster_size": (
                    sum(1 for cand in candidates if cand.get("probe_signature") == chosen.get("probe_signature")) if chosen else 0
                ),
            }
        out_rows.append(
            {
                "record_id": record_id,
                "task_id": record["task_id"],
                "task_text": record["task_text"],
                "entry_point": record["entry_point"],
                "probe_calls": calls,
                "candidate_count": len(candidates),
                "visible_candidate_count": sum(1 for cand in candidates if cand.get("visible_all_pass")),
                "hidden_correct_candidate_count": sum(1 for cand in candidates if cand.get("full_pass")),
                "coverage": any(cand.get("full_pass") for cand in candidates),
                "visible_hidden_wrong_count": sum(1 for cand in candidates if cand.get("visible_all_pass") and not cand.get("full_pass")),
                "candidates": [
                    {
                        "pool_candidate_id": cand.get("pool_candidate_id"),
                        "pool_name": cand.get("pool_name"),
                        "source": cand.get("source"),
                        "parse_status": cand.get("parse_status"),
                        "visible_all_pass": cand.get("visible_all_pass"),
                        "full_pass": cand.get("full_pass"),
                        "probe_signature": cand.get("probe_signature"),
                        "code_chars": len(cand.get("code", "")),
                    }
                    for cand in candidates
                ],
                "selectors": selectors,
            }
        )

    records_for_summary = []
    for row in out_rows:
        records_for_summary.append({"record_id": row["record_id"], "candidates": pool_by_record.get(row["record_id"], [])})

    summary = {
        "experiment": EXPERIMENT,
        "baseline_records": str(args.baseline_records),
        "base_records": len(baseline),
        "base_coverage": sum(1 for row in baseline if any(c.get("full_pass") for c in row.get("candidates", []))) / len(baseline),
        "base_miss_records": len(base_miss_ids),
        "base_miss_tasks": [base_by_id[rid]["task_id"] for rid in sorted(base_miss_ids, key=lambda rid: int(base_by_id[rid]["task_id"]))],
        "pool_names": args.pool_names,
        "candidate_records": [str(path) for path in args.candidate_records],
        "max_probes": args.max_probes,
        "residual_pool": pool_metrics(records_for_summary),
        "selectors": {name: selected_metrics(out_rows, name) for name in ["first_visible", "shortest_visible", "consensus_visible", "oracle_hidden"]},
        "per_input_pool": {},
        "token_usage": empty_usage(),
        "out": str(args.out),
    }
    for pool_name, rows in pool_inputs.items():
        residual_rows = [row for row in rows if row["record_id"] in base_miss_ids]
        summary["per_input_pool"][pool_name] = summarize_records(residual_rows)
        for row in residual_rows:
            summary["token_usage"] = add_usage(summary["token_usage"], row.get("token_usage", {}))

    write_jsonl(args.out, out_rows)
    write_json(args.summary, summary)
    write_manifest(args.out.with_suffix(".manifest.json"), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
