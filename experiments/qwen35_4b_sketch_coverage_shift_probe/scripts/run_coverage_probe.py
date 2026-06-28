#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dsl import execute, program_is_valid, program_pass_count  # noqa: E402
from src.sketch import complete_sketch, select_by_visible  # noqa: E402


SKETCH_FIELDS = {
    "auto": "target_sketch_auto",
    "manual": "target_sketch_manual",
    "erased": "target_sketch_erased",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def output_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def program_output_key(program: str, case: dict[str, Any]) -> str:
    try:
        return output_key(execute(program, case["input"]))
    except Exception as exc:
        return output_key({"__error__": type(exc).__name__})


def visible_consistent(programs: list[str], cases: list[dict[str, Any]]) -> list[str]:
    return [program for program in programs if program_is_valid(program) and program_pass_count(program, cases) == len(cases)]


def choose_query(
    candidates: list[str],
    query_pool: list[dict[str, Any]],
    used_indexes: set[int],
    *,
    oracle: bool,
) -> tuple[int | None, dict[str, Any]]:
    best_index = None
    best_score = None
    best_meta: dict[str, Any] = {}
    if not candidates:
        return None, {"reason": "empty_candidates"}
    for index, case in enumerate(query_pool):
        if index in used_indexes:
            continue
        counts = Counter(program_output_key(program, case) for program in candidates)
        if not counts:
            continue
        true_key = output_key(case["expected"])
        if oracle:
            kept = counts.get(true_key, 0)
            eliminated = len(candidates) - kept
            score = (eliminated, len(counts), -index)
        else:
            largest_bucket = max(counts.values())
            eliminated_if_majority_wrong = len(candidates) - largest_bucket
            score = (eliminated_if_majority_wrong, len(counts), -largest_bucket, -index)
        if best_score is None or score > best_score:
            best_score = score
            best_index = index
            best_meta = {
                "candidate_count": len(candidates),
                "bucket_count": len(counts),
                "largest_bucket": max(counts.values()),
                "true_bucket": counts.get(true_key, 0),
                "actual_eliminated": len(candidates) - counts.get(true_key, 0),
                "split_eliminated_if_majority_wrong": len(candidates) - max(counts.values()),
            }
    return best_index, best_meta


def run_active_policy(
    *,
    policy: str,
    programs: list[str],
    record: dict[str, Any],
    budgets: list[int],
) -> list[dict[str, Any]]:
    observed = list(record["visible"])
    query_pool = list(record["query_pool"])
    used_indexes: set[int] = set()
    candidates = visible_consistent(programs, observed)
    if not candidates:
        candidates = [program for program in programs if program_is_valid(program)]
    rows = []
    max_budget = max(budgets)
    for step in range(max_budget + 1):
        if step in budgets:
            selection = select_by_visible(candidates or programs, observed, record["hidden"])
            rows.append(
                {
                    "id": record["id"],
                    "family": record["family"],
                    "shift_type": record["shift_type"],
                    "policy": policy,
                    "budget": step,
                    "queries_used": len(used_indexes),
                    "candidate_count": len(candidates),
                    "selected_program": selection["program"],
                    "selected_visible_passes": selection["visible_passes"],
                    "selected_hidden_passes": selection["hidden_passes"] or 0,
                    "visible_total": len(observed),
                    "base_visible_total": len(record["visible"]),
                    "hidden_total": len(record["hidden"]),
                    "selected_visible_all": selection["visible_passes"] == len(observed),
                    "selected_hidden_all": (selection["hidden_passes"] or 0) == len(record["hidden"]),
                }
            )
        if step == max_budget:
            break
        query_index, meta = choose_query(
            candidates,
            query_pool,
            used_indexes,
            oracle=(policy == "oracle_elimination"),
        )
        if query_index is None:
            continue
        used_indexes.add(query_index)
        case = query_pool[query_index]
        true_key = output_key(case["expected"])
        candidates = [program for program in candidates if program_output_key(program, case) == true_key]
        observed.append(case)
        if rows:
            rows[-1].setdefault("next_query_meta", meta)
    return rows


def evaluate_record(
    record: dict[str, Any],
    *,
    sketch_mode: str,
    hole_options: int,
    max_programs_per_sketch: int,
    active_hole_options: int,
    budgets: list[int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sketch = record[SKETCH_FIELDS[sketch_mode]]
    programs = complete_sketch(
        sketch,
        record["visible"],
        max_programs_per_sketch=max_programs_per_sketch,
        max_hole_options=hole_options,
    )
    selected = select_by_visible(programs, record["visible"], record["hidden"])
    hidden_passes = [program_pass_count(program, record["hidden"]) for program in programs if program_is_valid(program)]
    visible_passes = [program_pass_count(program, record["visible"]) for program in programs if program_is_valid(program)]
    target_program_synthesized = record["target_program"] in programs
    row = {
        "id": record["id"],
        "family": record["family"],
        "shift_type": record["shift_type"],
        "sketch_mode": sketch_mode,
        "hole_options": hole_options,
        "sketch": sketch,
        "target_program": record["target_program"],
        "program_count": len(programs),
        "valid_program_count": len(hidden_passes),
        "visible_consistent_count": sum(1 for count in visible_passes if count == len(record["visible"])),
        "target_program_synthesized": target_program_synthesized,
        "candidate_oracle_visible_passes": max(visible_passes, default=0),
        "candidate_oracle_hidden_passes": max(hidden_passes, default=0),
        "candidate_oracle_visible_all": max(visible_passes, default=0) == len(record["visible"]),
        "candidate_oracle_hidden_all": max(hidden_passes, default=0) == len(record["hidden"]),
        "selected_program": selected["program"],
        "selected_visible_passes": selected["visible_passes"],
        "selected_hidden_passes": selected["hidden_passes"] or 0,
        "visible_total": len(record["visible"]),
        "hidden_total": len(record["hidden"]),
        "selected_visible_all": selected["visible_passes"] == len(record["visible"]),
        "selected_hidden_all": (selected["hidden_passes"] or 0) == len(record["hidden"]),
    }
    active_rows: list[dict[str, Any]] = []
    if hole_options == active_hole_options:
        for policy in ["active_max_split", "oracle_elimination"]:
            for active_row in run_active_policy(policy=policy, programs=programs, record=record, budgets=budgets):
                active_row["sketch_mode"] = sketch_mode
                active_row["hole_options"] = hole_options
                active_rows.append(active_row)
    return row, active_rows


def rate(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    successes = sum(1 for row in rows if row.get(key))
    return {"successes": successes, "records": len(rows), "rate": successes / len(rows) if rows else 0.0}


def avg(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row.get(key, 0)) for row in rows) / len(rows) if rows else 0.0


def summarize_coverage(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["shift_type"], row["family"], row["sketch_mode"], row["hole_options"])].append(row)
    out = []
    for (shift_type, family, sketch_mode, hole_options), subset in sorted(groups.items()):
        out.append(
            {
                "shift_type": shift_type,
                "family": family,
                "sketch_mode": sketch_mode,
                "hole_options": hole_options,
                "records": len(subset),
                "target_coverage": rate(subset, "target_program_synthesized"),
                "oracle_hidden_all": rate(subset, "candidate_oracle_hidden_all"),
                "selected_hidden_all": rate(subset, "selected_hidden_all"),
                "selected_visible_all": rate(subset, "selected_visible_all"),
                "avg_program_count": round(avg(subset, "program_count"), 3),
                "avg_visible_consistent_count": round(avg(subset, "visible_consistent_count"), 3),
            }
        )
    return out


def summarize_active(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["shift_type"], row["sketch_mode"], row["policy"], row["budget"], row["hole_options"])].append(row)
    out = []
    for (shift_type, sketch_mode, policy, budget, hole_options), subset in sorted(groups.items()):
        out.append(
            {
                "shift_type": shift_type,
                "sketch_mode": sketch_mode,
                "policy": policy,
                "budget": budget,
                "hole_options": hole_options,
                "records": len(subset),
                "selected_hidden_all": rate(subset, "selected_hidden_all"),
                "avg_candidate_count": round(avg(subset, "candidate_count"), 3),
                "avg_queries_used": round(avg(subset, "queries_used"), 3),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sketch-modes", default="auto,manual,erased")
    parser.add_argument("--hole-options", default="8,16,28")
    parser.add_argument("--active-hole-options", type=int, default=28)
    parser.add_argument("--budgets", default="0,1,2,3")
    parser.add_argument("--max-programs-per-sketch", type=int, default=4000)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    random.seed(args.seed)
    records = load_jsonl(args.data)
    if args.max_records:
        records = records[: args.max_records]
    sketch_modes = [item.strip() for item in args.sketch_modes.split(",") if item.strip()]
    hole_options = [int(item) for item in args.hole_options.split(",") if item.strip()]
    budgets = [int(item) for item in args.budgets.split(",") if item.strip()]

    coverage_rows: list[dict[str, Any]] = []
    active_rows: list[dict[str, Any]] = []
    jobs = [(record, sketch_mode, cap) for record in records for sketch_mode in sketch_modes for cap in hole_options]
    for record, sketch_mode, cap in tqdm(jobs, desc="coverage-probe"):
        row, active = evaluate_record(
            record,
            sketch_mode=sketch_mode,
            hole_options=cap,
            max_programs_per_sketch=args.max_programs_per_sketch,
            active_hole_options=args.active_hole_options,
            budgets=budgets,
        )
        coverage_rows.append(row)
        active_rows.extend(active)

    result = {
        "data": str(args.data),
        "records": len(records),
        "sketch_modes": sketch_modes,
        "hole_options": hole_options,
        "active_hole_options": args.active_hole_options,
        "budgets": budgets,
        "max_programs_per_sketch": args.max_programs_per_sketch,
        "coverage_summary": summarize_coverage(coverage_rows),
        "active_summary": summarize_active(active_rows),
        "coverage_rows": coverage_rows,
        "active_rows": active_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"coverage_groups": len(result["coverage_summary"]), "active_groups": len(result["active_summary"])}, indent=2))


if __name__ == "__main__":
    main()
