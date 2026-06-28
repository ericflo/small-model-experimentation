#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dsl import (  # noqa: E402
    DslError,
    Symbol,
    execute,
    eval_expr,
    normalize_program,
    parse_expr,
    program_case_passes,
    program_is_valid,
    program_pass_count,
)
from src.modeling import load_generation_model, load_jsonl, load_tokenizer  # noqa: E402
from src.prompts import messages_for_sketch_record, shuffled_visible  # noqa: E402
from src.sketch import complete_sketch, sketch_hole_count, sketch_variants  # noqa: E402


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


def prompt_text(tokenizer, record: dict[str, Any], prompt_mode: str, trace_override=None) -> str:
    messages = messages_for_sketch_record(record, prompt_mode=prompt_mode, trace_override=trace_override)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def decode_new(tokenizer, output_ids, prompt_len: int) -> str:
    return tokenizer.decode(output_ids[prompt_len:], skip_special_tokens=True)


def generate_sketches(model, tokenizer, prompt: str, args) -> list[dict[str, Any]]:
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    candidates = []
    with torch.no_grad():
        greedy = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )[0]
    raw = decode_new(tokenizer, greedy, inputs.input_ids.shape[1])
    candidates.append({"kind": "greedy", "raw": raw, "sketch": normalize_program(raw)})
    if args.num_samples:
        with torch.no_grad():
            sampled = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                num_return_sequences=args.num_samples,
                pad_token_id=tokenizer.pad_token_id,
            )
        for i, output_ids in enumerate(sampled):
            raw = decode_new(tokenizer, output_ids, inputs.input_ids.shape[1])
            candidates.append({"kind": f"sample_{i:02d}", "raw": raw, "sketch": normalize_program(raw)})
    return candidates


def synthesize_from_sketches(
    sketches: list[dict[str, Any]],
    record: dict[str, Any],
    args,
) -> tuple[list[str], list[dict[str, Any]]]:
    programs: list[str] = []
    seen: set[str] = set()
    sketch_rows = []
    for sketch_row in sketches:
        variants = sketch_variants(sketch_row["sketch"], max_variants=args.max_sketch_variants)
        for variant_index, sketch in enumerate(variants):
            valid_syntax = program_is_valid(sketch)
            hole_count = sketch_hole_count(sketch)
            completed = complete_sketch(
                sketch,
                record["visible"],
                max_programs_per_sketch=args.max_programs_per_sketch,
                max_hole_options=args.max_hole_options,
            )
            added = 0
            for program in completed:
                if program in seen:
                    continue
                if len(programs) >= args.max_total_programs_per_record:
                    break
                seen.add(program)
                programs.append(program)
                added += 1
            sketch_rows.append(
                {
                    "kind": sketch_row["kind"],
                    "variant_index": variant_index,
                    "is_original_variant": variant_index == 0,
                    "sketch": sketch,
                    "raw": sketch_row["raw"] if variant_index == 0 else None,
                    "valid_syntax": valid_syntax,
                    "hole_count": hole_count,
                    "completed_programs": len(completed),
                    "added_programs": added,
                }
            )
            if len(programs) >= args.max_total_programs_per_record:
                break
        if len(programs) >= args.max_total_programs_per_record:
            break
    return programs, sketch_rows


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


def output_key(program: str, case: dict[str, Any]) -> str:
    try:
        value = execute(program, case["input"])
    except Exception as exc:
        value = f"<error:{type(exc).__name__}:{exc}>"
    return json.dumps(value, sort_keys=True)


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
    pool = query_pool
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
            chosen = choose_random_case(pool, used, rng)
            details = {}
        elif policy == "active_max_split":
            chosen, details = choose_active_split_case(
                bank,
                observed,
                pool,
                used,
                max_policy_candidates=max_policy_candidates,
                rng=rng,
            )
        elif policy == "oracle_elimination":
            chosen, details = choose_oracle_elimination_case(
                bank,
                observed,
                pool,
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
                sum(1 for row in rows if row["candidate_oracle_hidden_all"]) / n, 6
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


def unique_sketch_candidates(sketch_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    seen = set()
    for candidate in sketch_candidates:
        if candidate["sketch"] in seen:
            continue
        seen.add(candidate["sketch"])
        out.append(candidate)
    return out


def target_sketch_row(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"kind": "target_sketch", "raw": record["target_sketch"], "sketch": record["target_sketch"]}]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-mode", choices=["trace", "no_trace", "shuffled_trace"], default="trace")
    parser.add_argument("--sketch-source", choices=["model", "target"], default="model")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--revision", default="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
    parser.add_argument("--cache-dir", default="/workspace/.cache/huggingface")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--max-programs-per-sketch", type=int, default=5000)
    parser.add_argument("--max-total-programs-per-record", type=int, default=16000)
    parser.add_argument("--max-hole-options", type=int, default=28)
    parser.add_argument("--max-sketch-variants", type=int, default=16)
    parser.add_argument("--budgets", default="0,1,2,3,6,12")
    parser.add_argument("--random-repeats", type=int, default=5)
    parser.add_argument("--max-policy-candidates", type=int, default=4096)
    parser.add_argument("--max-query-pool-cases", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20260701)
    args = parser.parse_args()

    budgets = sorted({int(item) for item in args.budgets.split(",") if item.strip()})
    if not budgets or budgets[0] != 0:
        budgets = sorted(set([0] + budgets))

    records = load_jsonl(args.data)
    if args.max_records:
        records = records[: args.max_records]

    tokenizer = None
    model = None
    trace_overrides = [None] * len(records)
    if args.sketch_source == "model":
        if args.adapter is None:
            raise ValueError("--adapter is required with --sketch-source model")
        tokenizer = load_tokenizer(args.model_id, args.revision, args.cache_dir)
        model = load_generation_model(
            args.model_id,
            args.revision,
            adapter=str(args.adapter),
            cache_dir=args.cache_dir,
        )
        trace_overrides = shuffled_visible(records, seed=99) if args.prompt_mode == "shuffled_trace" else [None] * len(records)

    candidate_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    for index, record in enumerate(tqdm(records, desc=f"eval-active:{args.data.stem}")):
        if args.sketch_source == "target":
            sketches = target_sketch_row(record)
        else:
            assert model is not None and tokenizer is not None
            prompt = prompt_text(tokenizer, record, args.prompt_mode, trace_override=trace_overrides[index])
            sketches = unique_sketch_candidates(generate_sketches(model, tokenizer, prompt, args))
        programs, sketch_rows = synthesize_from_sketches(sketches, record, args)
        bank = ProgramBank(programs, record)
        query_pool = dedupe_pool(record)
        if args.max_query_pool_cases and len(query_pool) > args.max_query_pool_cases:
            query_pool = random.Random(args.seed + index * 31).sample(query_pool, args.max_query_pool_cases)
        visible_selected = bank.select(record["visible"])
        candidate_oracle_hidden = max(bank.hidden_passes, default=0)
        candidate_oracle_visible = max(bank.visible_passes, default=0)
        visible_consistent = sum(1 for passes in bank.visible_passes if passes == len(record["visible"]))
        candidate_rows.append(
            {
                "id": record["id"],
                "family": record["family"],
                "target_program": record["target_program"],
                "target_sketch": record.get("target_sketch"),
                "visible_selected_program": visible_selected["program"],
                "visible_selected_hidden_passes": int(visible_selected["hidden_passes"] or 0),
                "visible_selected_hidden_all": int(visible_selected["hidden_passes"] or 0) == len(record["hidden"]),
                "candidate_oracle_visible_passes": candidate_oracle_visible,
                "candidate_oracle_hidden_passes": candidate_oracle_hidden,
                "candidate_oracle_visible_all": candidate_oracle_visible == len(record["visible"]),
                "candidate_oracle_hidden_all": candidate_oracle_hidden == len(record["hidden"]),
                "target_program_synthesized": record["target_program"] in programs,
                "unique_sketch_count": len(sketches),
                "synthesized_program_count": len(programs),
                "visible_consistent_candidates": visible_consistent,
                "case_pool_count": len(record.get("case_pool", [])),
                "query_pool_count": len(query_pool),
                "sketches": sketch_rows,
            }
        )
        policy_rows.extend(
            run_policy(
                record=record,
                bank=bank,
                query_pool=query_pool,
                policy="active_max_split",
                budgets=budgets,
                repeat=0,
                max_policy_candidates=args.max_policy_candidates,
                rng=random.Random(args.seed + index * 17 + 1),
            )
        )
        policy_rows.extend(
            run_policy(
                record=record,
                bank=bank,
                query_pool=query_pool,
                policy="oracle_elimination",
                budgets=budgets,
                repeat=0,
                max_policy_candidates=args.max_policy_candidates,
                rng=random.Random(args.seed + index * 17 + 2),
            )
        )
        for repeat in range(args.random_repeats):
            policy_rows.extend(
                run_policy(
                    record=record,
                    bank=bank,
                    query_pool=query_pool,
                    policy="random_extra",
                    budgets=budgets,
                    repeat=repeat,
                    max_policy_candidates=args.max_policy_candidates,
                    rng=random.Random(args.seed + index * 1009 + repeat),
                )
            )

    visible_rows = [
        {
            **row,
            "policy": "visible_prior",
            "budget": 0,
            "repeat": 0,
            "queries_used": 0,
            "observed_total": row["visible_total"],
            "observed_passes": row["visible_passes"],
            "observed_all": row["visible_all"],
            "query_trace": [],
        }
        for row in policy_rows
        if row["policy"] == "active_max_split" and row["budget"] == 0
    ]
    policy_rows.extend(visible_rows)

    result = {
        "experiment": "qwen35_4b_active_counterexample_trace_selection",
        "model_id": args.model_id,
        "revision": args.revision,
        "adapter": str(args.adapter) if args.adapter else None,
        "prompt_mode": args.prompt_mode,
        "sketch_source": args.sketch_source,
        "data": str(args.data),
        "records": len(records),
        "budgets": budgets,
        "random_repeats": args.random_repeats,
        "max_policy_candidates": args.max_policy_candidates,
        "max_query_pool_cases": args.max_query_pool_cases,
        "num_samples": args.num_samples,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_programs_per_sketch": args.max_programs_per_sketch,
        "max_total_programs_per_record": args.max_total_programs_per_record,
        "max_hole_options": args.max_hole_options,
        "max_sketch_variants": args.max_sketch_variants,
        "candidate_summary": summarize_candidates(candidate_rows),
        "policy_summary": summarize_policy_rows(policy_rows),
        "candidate_rows": candidate_rows,
        "policy_rows": policy_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"candidate_summary": result["candidate_summary"], "policy_summary": result["policy_summary"]["overall"]}, indent=2))


if __name__ == "__main__":
    main()
