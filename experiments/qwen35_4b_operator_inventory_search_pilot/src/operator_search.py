from __future__ import annotations

import json
from collections import Counter
from typing import Any

from .dsl import execute, program_case_passes, program_is_valid, program_pass_count


def _first_env(visible: list[dict[str, Any]]) -> dict[str, Any]:
    return dict(visible[0]["input"]) if visible else {}


def _seq_vars(visible: list[dict[str, Any]]) -> list[str]:
    env = _first_env(visible)
    return [
        name
        for name, value in env.items()
        if isinstance(value, list) and value and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    ]


def _num_terms(visible: list[dict[str, Any]], template: str) -> list[str]:
    env = _first_env(visible)
    names = [name for name, value in env.items() if isinstance(value, int) and not isinstance(value, bool)]
    preferred = {
        "mod_format": ["m"],
        "offset_format": ["offset"],
        "threshold_gate": ["threshold"],
    }.get(template, [])
    ordered = preferred + [name for name in names if name not in preferred]
    for constant in ["0", "1", "2", "3", "5", "7", "11"]:
        if constant not in ordered:
            ordered.append(constant)
    return ordered


def render_program(template: str, operator: str, seq_term: str, num_term: str) -> str:
    if template == "mod_format":
        return f'(format "M{{}}" (mod ({operator} {seq_term}) {num_term}))'
    if template == "offset_format":
        return f'(format "A{{}}" (add ({operator} {seq_term}) {num_term}))'
    if template == "threshold_gate":
        return f"(if (gt ({operator} {seq_term}) {num_term}) hi lo)"
    raise KeyError(template)


def complete_operator_hole(
    record: dict[str, Any],
    operator_names: list[str],
    *,
    max_candidates: int = 10000,
) -> list[str]:
    programs: list[str] = []
    seen: set[str] = set()
    seq_terms = _seq_vars(record["visible"])
    num_terms = _num_terms(record["visible"], record["template"])
    for operator in operator_names:
        for seq_term in seq_terms:
            for num_term in num_terms:
                program = render_program(record["template"], operator, seq_term, num_term)
                if program in seen:
                    continue
                seen.add(program)
                if not program_is_valid(program):
                    continue
                programs.append(program)
                if len(programs) >= max_candidates:
                    return programs
    return programs


def visible_consistent(programs: list[str], cases: list[dict[str, Any]]) -> list[str]:
    return [program for program in programs if program_pass_count(program, cases) == len(cases)]


def output_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True)


def program_output_key(program: str, case: dict[str, Any]) -> str:
    try:
        return output_key(execute(program, case["input"]))
    except Exception as exc:
        return output_key({"__error__": type(exc).__name__})


def operator_name(program: str) -> str | None:
    for op in ["sum", "first", "last", "max", "min", "prod", "gcd"]:
        marker = f"({op} "
        if marker in program:
            return op
    return None


def select_by_visible_then_prior(programs: list[str], visible: list[dict[str, Any]], hidden: list[dict[str, Any]]) -> dict[str, Any]:
    best_score = None
    best_row = None
    for index, program in enumerate(programs):
        visible_passes = program_pass_count(program, visible)
        hidden_passes = program_pass_count(program, hidden)
        op = operator_name(program) or ""
        # Keep a stable, cheap prior: prefer fewer chars after visible score,
        # then original enumeration order. This exposes ambiguity instead of
        # smuggling in hidden knowledge.
        score = (visible_passes, -len(program), -index)
        row = {
            "program": program,
            "operator": op,
            "visible_passes": visible_passes,
            "hidden_passes": hidden_passes,
            "visible_all": visible_passes == len(visible),
            "hidden_all": hidden_passes == len(hidden),
            "candidate_index": index,
        }
        if best_score is None or score > best_score:
            best_score = score
            best_row = row
    if best_row is None:
        return {
            "program": "",
            "operator": None,
            "visible_passes": 0,
            "hidden_passes": 0,
            "visible_all": False,
            "hidden_all": False,
            "candidate_index": None,
        }
    return best_row


def choose_query(
    candidates: list[str],
    query_pool: list[dict[str, Any]],
    used_indexes: set[int],
    *,
    oracle: bool,
) -> tuple[int | None, dict[str, Any]]:
    best_score = None
    best_index = None
    best_meta: dict[str, Any] = {}
    for index, case in enumerate(query_pool):
        if index in used_indexes:
            continue
        buckets = Counter(program_output_key(program, case) for program in candidates)
        if not buckets:
            continue
        true_key = output_key(case["expected"])
        if oracle:
            true_bucket = buckets.get(true_key, 0)
            score = (len(candidates) - true_bucket, len(buckets), -index)
        else:
            largest_bucket = max(buckets.values())
            score = (len(candidates) - largest_bucket, len(buckets), -largest_bucket, -index)
        if best_score is None or score > best_score:
            best_score = score
            best_index = index
            best_meta = {
                "candidate_count": len(candidates),
                "bucket_count": len(buckets),
                "largest_bucket": max(buckets.values()),
                "true_bucket": buckets.get(true_key, 0),
                "actual_eliminated": len(candidates) - buckets.get(true_key, 0),
                "split_eliminated_if_majority_wrong": len(candidates) - max(buckets.values()),
            }
    return best_index, best_meta


def active_trace(
    record: dict[str, Any],
    programs: list[str],
    *,
    budgets: list[int],
    policy: str,
) -> list[dict[str, Any]]:
    observed = list(record["visible"])
    used_indexes: set[int] = set()
    candidates = visible_consistent(programs, observed)
    rows = []
    max_budget = max(budgets)
    for step in range(max_budget + 1):
        selected = select_by_visible_then_prior(candidates, observed, record["hidden"])
        if step in budgets:
            candidate_ops = sorted({operator_name(program) for program in candidates if operator_name(program)})
            rows.append(
                {
                    "budget": step,
                    "policy": policy,
                    "queries_used": len(used_indexes),
                    "candidate_count": len(candidates),
                    "operator_candidate_count": len(candidate_ops),
                    "operator_candidates": candidate_ops,
                    "selected_program": selected["program"],
                    "selected_operator": selected["operator"],
                    "selected_hidden_passes": selected["hidden_passes"],
                    "selected_hidden_all": selected["hidden_all"],
                    "selected_visible_passes": selected["visible_passes"],
                    "observed_count": len(observed),
                }
            )
        if step == max_budget:
            break
        query_index, _meta = choose_query(candidates, record["query_pool"], used_indexes, oracle=policy == "oracle_elimination")
        if query_index is None:
            continue
        used_indexes.add(query_index)
        case = record["query_pool"][query_index]
        true_key = output_key(case["expected"])
        candidates = [program for program in candidates if program_output_key(program, case) == true_key]
        observed.append(case)
    return rows


def candidate_summary(record: dict[str, Any], programs: list[str]) -> dict[str, Any]:
    visible_candidates = visible_consistent(programs, record["visible"])
    target = record["target_program"]
    target_in_raw = target in programs
    target_in_visible = target in visible_candidates
    hidden_passes = [program_pass_count(program, record["hidden"]) for program in visible_candidates]
    selected = select_by_visible_then_prior(visible_candidates, record["visible"], record["hidden"])
    visible_ops = sorted({operator_name(program) for program in visible_candidates if operator_name(program)})
    return {
        "raw_candidate_count": len(programs),
        "visible_consistent_count": len(visible_candidates),
        "visible_consistent_operator_count": len(visible_ops),
        "visible_consistent_operators": visible_ops,
        "target_in_raw_candidates": target_in_raw,
        "target_in_visible_candidates": target_in_visible,
        "candidate_oracle_hidden_passes": max(hidden_passes, default=0),
        "candidate_oracle_hidden_all": max(hidden_passes, default=0) == len(record["hidden"]),
        "selected_program": selected["program"],
        "selected_operator": selected["operator"],
        "selected_hidden_passes": selected["hidden_passes"],
        "selected_hidden_all": selected["hidden_all"],
        "selected_visible_passes": selected["visible_passes"],
        "selected_visible_all": selected["visible_all"],
    }

