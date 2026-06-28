from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
from typing import Any

from .dsl import DslError, Symbol, eval_expr, parse_expr


DSL_OPS = {
    "sum",
    "len",
    "mod",
    "add",
    "sub",
    "format",
    "contains",
    "count_eq",
    "tuple_get",
    "sort",
    "first",
    "last",
    "gt",
    "ge",
    "lt",
    "eq",
    "and",
    "or",
    "not",
    "if",
    "join",
}


POLICY_SYSTEM_PROMPT = (
    "You choose the next execution query for a program-selection agent. "
    "You are given visible input/output examples and candidate-output buckets "
    "for possible query inputs. The true output for each query option is not "
    "shown. Pick the option most likely to eliminate wrong candidate programs. "
    "Answer with exactly one option id such as Q03."
)


def compact_json(value: Any, *, max_chars: int = 220) -> str:
    text = json.dumps(value, sort_keys=True)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def case_key(case: dict[str, Any]) -> str:
    return json.dumps(case["input"], sort_keys=True)


def dedupe_pool(record: dict[str, Any]) -> list[dict[str, Any]]:
    seen = {case_key(case) for case in record["visible"]}
    out = []
    for case in record.get("case_pool", []):
        key = case_key(case)
        if key in seen:
            continue
        seen.add(key)
        out.append(case)
    return out


def _op_name(expr: Any) -> str | None:
    if isinstance(expr, list) and expr and isinstance(expr[0], Symbol):
        return expr[0].name
    return None


def program_prior_features(program: str, observed: list[dict[str, Any]]) -> tuple[int, int]:
    if not observed:
        return (0, 0)
    env = observed[0].get("input", {})
    data_symbols: set[str] = set()
    op_count = 0
    try:
        expr = parse_expr(program)
    except (DslError, ValueError, TypeError):
        return (0, 0)

    def visit(node: Any) -> None:
        nonlocal op_count
        if isinstance(node, Symbol):
            if node.name in env and not node.name.endswith("_label"):
                data_symbols.add(node.name)
            return
        if isinstance(node, list):
            op = _op_name(node)
            if op in DSL_OPS:
                op_count += 1
            for child in node:
                visit(child)

    visit(expr)
    return (len(data_symbols), op_count)


class ProgramBank:
    def __init__(self, programs: list[str], record: dict[str, Any]) -> None:
        self.programs = programs
        self.record = record
        self.exprs: list[Any | None] = []
        self.valid: list[bool] = []
        self.features: list[tuple[int, int]] = []
        self.output_cache: dict[tuple[int, str], str] = {}
        for program in programs:
            try:
                expr = parse_expr(program)
            except Exception:
                expr = None
            self.exprs.append(expr)
            self.valid.append(expr is not None)
            self.features.append(program_prior_features(program, record["visible"]) if expr is not None else (0, 0))
        self.hidden_passes = [self.pass_count_index(index, record["hidden"]) for index in range(len(programs))]
        self.visible_passes = [self.pass_count_index(index, record["visible"]) for index in range(len(programs))]

    def output_key_index(self, index: int, case: dict[str, Any]) -> str:
        key = (index, case_key(case))
        cached = self.output_cache.get(key)
        if cached is not None:
            return cached
        expr = self.exprs[index]
        if expr is None:
            value = "<invalid>"
        else:
            try:
                value = eval_expr(expr, case["input"])
            except Exception as exc:
                value = f"<error:{type(exc).__name__}:{exc}>"
        out = json.dumps(value, sort_keys=True)
        self.output_cache[key] = out
        return out

    def case_passes_index(self, index: int, case: dict[str, Any]) -> bool:
        return self.output_key_index(index, case) == json.dumps(case["expected"], sort_keys=True)

    def pass_count_index(self, index: int, cases: list[dict[str, Any]]) -> int:
        if not self.valid[index]:
            return 0
        return sum(1 for case in cases if self.case_passes_index(index, case))

    def score_index(self, index: int, observed: list[dict[str, Any]]) -> tuple[int, int, int, int, int, int, str]:
        passes = self.pass_count_index(index, observed)
        data_symbol_count, op_count = self.features[index]
        program = self.programs[index]
        return (passes, int(self.valid[index]), data_symbol_count, op_count, -index, -len(program), program)

    def ranked_indexes(self, observed: list[dict[str, Any]]) -> list[tuple[tuple[int, int, int, int, int, int, str], int]]:
        ranked = [(self.score_index(index, observed), index) for index in range(len(self.programs))]
        ranked.sort(reverse=True)
        return ranked

    def select(self, observed: list[dict[str, Any]]) -> dict[str, Any]:
        ranked = self.ranked_indexes(observed)
        if not ranked:
            return {
                "program": "",
                "valid": False,
                "visible_passes": 0,
                "hidden_passes": 0,
                "candidate_index": -1,
            }
        _, index = ranked[0]
        return {
            "program": self.programs[index],
            "valid": self.valid[index],
            "visible_passes": self.pass_count_index(index, observed),
            "hidden_passes": self.hidden_passes[index],
            "candidate_index": index,
        }

    def viable_indexes(self, observed: list[dict[str, Any]], *, max_policy_candidates: int) -> list[int]:
        ranked = self.ranked_indexes(observed)
        if not ranked:
            return []
        max_passes = ranked[0][0][0]
        perfect = len(observed)
        threshold = perfect if max_passes == perfect else max_passes
        viable = [index for score, index in ranked if score[0] == threshold and score[1] == 1]
        return viable[:max_policy_candidates]


def viable_candidates(
    bank: ProgramBank,
    observed: list[dict[str, Any]],
    *,
    max_policy_candidates: int,
) -> list[int]:
    return bank.viable_indexes(observed, max_policy_candidates=max_policy_candidates)


def entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    value = 0.0
    for count in counts.values():
        p = count / total
        value -= p * math.log2(p)
    return value


def choose_random_case(pool: list[dict[str, Any]], used: set[str], rng: random.Random) -> dict[str, Any] | None:
    choices = [case for case in pool if case_key(case) not in used]
    if not choices:
        return None
    return rng.choice(choices)


def choose_active_split_case(
    bank: ProgramBank,
    observed: list[dict[str, Any]],
    pool: list[dict[str, Any]],
    used: set[str],
    *,
    max_policy_candidates: int,
    rng: random.Random,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    viable = viable_candidates(bank, observed, max_policy_candidates=max_policy_candidates)
    if not viable:
        return None, {"viable_candidates": 0}
    best: tuple[tuple[float, int, int, float], dict[str, Any], dict[str, Any]] | None = None
    for case in pool:
        key = case_key(case)
        if key in used:
            continue
        outputs = Counter(bank.output_key_index(index, case) for index in viable)
        if not outputs:
            continue
        largest_bucket = max(outputs.values())
        split_score = len(viable) - largest_bucket
        details = {
            "viable_candidates": len(viable),
            "output_buckets": len(outputs),
            "largest_bucket": largest_bucket,
            "max_possible_eliminated": split_score,
            "entropy": entropy(outputs),
        }
        score = (details["entropy"], split_score, len(outputs), rng.random())
        if best is None or score > best[0]:
            best = (score, case, details)
    if best is None:
        return None, {"viable_candidates": len(viable)}
    return best[1], best[2]


def choose_oracle_elimination_case(
    bank: ProgramBank,
    observed: list[dict[str, Any]],
    pool: list[dict[str, Any]],
    used: set[str],
    *,
    max_policy_candidates: int,
    rng: random.Random,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    viable = viable_candidates(bank, observed, max_policy_candidates=max_policy_candidates)
    if not viable:
        return None, {"viable_candidates": 0}
    best: tuple[tuple[int, float, int, float], dict[str, Any], dict[str, Any]] | None = None
    for case in pool:
        key = case_key(case)
        if key in used:
            continue
        eliminated = sum(1 for index in viable if not bank.case_passes_index(index, case))
        outputs = Counter(bank.output_key_index(index, case) for index in viable)
        details = {
            "viable_candidates": len(viable),
            "output_buckets": len(outputs),
            "actual_eliminated": eliminated,
            "entropy": entropy(outputs),
        }
        score = (eliminated, details["entropy"], len(outputs), rng.random())
        if best is None or score > best[0]:
            best = (score, case, details)
    if best is None:
        return None, {"viable_candidates": len(viable)}
    return best[1], best[2]


def query_option_rows(
    *,
    bank: ProgramBank,
    observed: list[dict[str, Any]],
    pool: list[dict[str, Any]],
    used: set[str],
    max_policy_candidates: int,
    max_options: int,
    max_buckets: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    viable = viable_candidates(bank, observed, max_policy_candidates=max_policy_candidates)
    if not viable:
        return []

    rows = []
    for case in pool:
        key = case_key(case)
        if key in used:
            continue
        outputs = Counter(bank.output_key_index(index, case) for index in viable)
        if not outputs:
            continue
        largest_bucket = max(outputs.values())
        expected_key = json.dumps(case["expected"], sort_keys=True)
        actual_eliminated = sum(1 for index in viable if not bank.case_passes_index(index, case))
        sorted_buckets = sorted(outputs.items(), key=lambda item: (-item[1], item[0]))
        rows.append(
            {
                "case": case,
                "input": case["input"],
                "expected": case["expected"],
                "viable_candidates": len(viable),
                "output_buckets": len(outputs),
                "largest_bucket": largest_bucket,
                "max_possible_eliminated": len(viable) - largest_bucket,
                "actual_eliminated": actual_eliminated,
                "target_bucket_size": outputs.get(expected_key, 0),
                "entropy": entropy(outputs),
                "buckets": [
                    {"value": value, "count": count}
                    for value, count in sorted_buckets[:max_buckets]
                ],
                "case_key": key,
                "tie": rng.random(),
            }
        )
    rows.sort(
        key=lambda row: (
            row["entropy"],
            row["max_possible_eliminated"],
            row["output_buckets"],
            -row["largest_bucket"],
            row["tie"],
        ),
        reverse=True,
    )
    selected = rows[:max_options]
    for index, row in enumerate(selected):
        row["option_id"] = f"Q{index:02d}"
    return selected


def best_oracle_option(options: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not options:
        return None
    return max(
        options,
        key=lambda row: (
            row["actual_eliminated"],
            row["entropy"],
            row["max_possible_eliminated"],
            row["output_buckets"],
        ),
    )


def render_policy_prompt(
    *,
    record: dict[str, Any],
    observed: list[dict[str, Any]],
    options: list[dict[str, Any]],
    step: int,
    max_visible: int = 8,
) -> str:
    lines = [
        "Task: choose the next query input for candidate-program selection.",
        f"Record family: {record.get('family', 'unknown')}",
        f"Query step: {step}",
        "",
        "Observed input/output cases:",
    ]
    for index, case in enumerate(observed[:max_visible]):
        lines.append(
            f"V{index:02d} input={compact_json(case['input'])} output={compact_json(case['expected'])}"
        )
    if len(observed) > max_visible:
        lines.append(f"... {len(observed) - max_visible} additional observed cases omitted")
    lines.extend(["", "Candidate query options:"])
    for option in options:
        bucket_text = "; ".join(
            f"{bucket['value']}:{bucket['count']}" for bucket in option["buckets"]
        )
        lines.append(
            " ".join(
                [
                    f"{option['option_id']}",
                    f"input={compact_json(option['input'])}",
                    f"candidate_outputs=[{bucket_text}]",
                    f"buckets={option['output_buckets']}",
                    f"largest={option['largest_bucket']}",
                    f"entropy={option['entropy']:.3f}",
                ]
            )
        )
    lines.extend(["", "Answer with exactly one option id."])
    return "\n".join(lines)


def parse_policy_action(text: str, valid_ids: set[str]) -> str | None:
    match = re.search(r"\bQ\d{2}\b", text.strip())
    if not match:
        return None
    action = match.group(0)
    if action not in valid_ids:
        return None
    return action


def selection_snapshot(
    *,
    record: dict[str, Any],
    bank: ProgramBank,
    observed: list[dict[str, Any]],
    policy: str,
    budget: int,
    repeat: int,
    query_trace: list[dict[str, Any]],
) -> dict[str, Any]:
    selected = bank.select(observed)
    hidden_passes = int(selected["hidden_passes"] or 0)
    observed_passes = int(selected["visible_passes"] or 0)
    visible_passes = bank.pass_count_index(selected["candidate_index"], record["visible"]) if selected["candidate_index"] >= 0 else 0
    return {
        "id": record["id"],
        "family": record["family"],
        "policy": policy,
        "budget": budget,
        "repeat": repeat,
        "queries_used": len(query_trace),
        "observed_total": len(observed),
        "visible_total": len(record["visible"]),
        "hidden_total": len(record["hidden"]),
        "visible_passes": visible_passes,
        "observed_passes": observed_passes,
        "hidden_passes": hidden_passes,
        "visible_all": visible_passes == len(record["visible"]),
        "observed_all": observed_passes == len(observed),
        "hidden_all": hidden_passes == len(record["hidden"]),
        "selected_program": selected["program"],
        "query_trace": [dict(item) for item in query_trace],
    }


def run_policy(
    *,
    record: dict[str, Any],
    bank: ProgramBank,
    query_pool: list[dict[str, Any]],
    policy: str,
    budgets: list[int],
    repeat: int,
    max_policy_candidates: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    max_budget = max(budgets)
    observed = list(record["visible"])
    used = {case_key(case) for case in observed}
    query_trace: list[dict[str, Any]] = []
    results = []
    budget_set = set(budgets)
    if 0 in budget_set:
        results.append(
            selection_snapshot(
                record=record,
                bank=bank,
                observed=observed,
                policy=policy,
                budget=0,
                repeat=repeat,
                query_trace=query_trace,
            )
        )
    for step in range(1, max_budget + 1):
        details: dict[str, Any] = {}
        if policy == "random_extra":
            chosen = choose_random_case(query_pool, used, rng)
        elif policy == "active_max_split":
            chosen, details = choose_active_split_case(
                bank,
                observed,
                query_pool,
                used,
                max_policy_candidates=max_policy_candidates,
                rng=rng,
            )
        elif policy == "oracle_elimination":
            chosen, details = choose_oracle_elimination_case(
                bank,
                observed,
                query_pool,
                used,
                max_policy_candidates=max_policy_candidates,
                rng=rng,
            )
        else:
            raise ValueError(f"unknown policy: {policy}")
        if chosen is None:
            if step in budget_set:
                results.append(
                    selection_snapshot(
                        record=record,
                        bank=bank,
                        observed=observed,
                        policy=policy,
                        budget=step,
                        repeat=repeat,
                        query_trace=query_trace,
                    )
                )
            continue
        used.add(case_key(chosen))
        observed.append(chosen)
        query_trace.append(
            {
                "step": step,
                "input": chosen["input"],
                "expected": chosen["expected"],
                **details,
            }
        )
        if step in budget_set:
            results.append(
                selection_snapshot(
                    record=record,
                    bank=bank,
                    observed=observed,
                    policy=policy,
                    budget=step,
                    repeat=repeat,
                    query_trace=query_trace,
                )
            )
    return results


def summarize_policy_rows(policy_rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    family_grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in policy_rows:
        grouped[(row["policy"], int(row["budget"]))].append(row)
        family_grouped[(row["policy"], int(row["budget"]), row["family"])].append(row)

    def block(rows: list[dict[str, Any]]) -> dict[str, Any]:
        records = len(rows)
        return {
            "records": records,
            "hidden_all_successes": sum(1 for row in rows if row["hidden_all"]),
            "hidden_all_rate": round(sum(1 for row in rows if row["hidden_all"]) / records, 6) if records else 0.0,
            "observed_all_rate": round(sum(1 for row in rows if row["observed_all"]) / records, 6) if records else 0.0,
            "avg_hidden_passes": round(sum(row["hidden_passes"] for row in rows) / records, 3) if records else 0.0,
            "avg_queries_used": round(sum(row["queries_used"] for row in rows) / records, 3) if records else 0.0,
        }

    return {
        "overall": {f"{policy}@{budget}": block(rows) for (policy, budget), rows in sorted(grouped.items())},
        "by_family": {
            f"{policy}@{budget}/{family}": block(rows)
            for (policy, budget, family), rows in sorted(family_grouped.items())
        },
    }


def summarize_candidates(candidate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        by_family[row["family"]].append(row)

    def block(rows: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(rows)
        return {
            "records": n,
            "candidate_oracle_hidden_all_successes": sum(1 for row in rows if row["candidate_oracle_hidden_all"]),
            "candidate_oracle_hidden_all_rate": round(
                sum(1 for row in rows if row["candidate_oracle_hidden_all"]) / n,
                6,
            )
            if n
            else 0.0,
            "target_program_synthesized_successes": sum(1 for row in rows if row["target_program_synthesized"]),
            "target_program_synthesized_rate": round(sum(1 for row in rows if row["target_program_synthesized"]) / n, 6)
            if n
            else 0.0,
            "avg_synthesized_programs": round(sum(row["synthesized_program_count"] for row in rows) / n, 3) if n else 0.0,
            "avg_visible_consistent_candidates": round(
                sum(row["visible_consistent_candidates"] for row in rows) / n,
                3,
            )
            if n
            else 0.0,
            "avg_unique_sketches": round(sum(row["unique_sketch_count"] for row in rows) / n, 3) if n else 0.0,
        }

    return {
        "overall": block(candidate_rows),
        "by_family": {family: block(rows) for family, rows in sorted(by_family.items())},
    }
