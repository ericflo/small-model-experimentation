#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.active_core import (  # noqa: E402
    POLICY_SYSTEM_PROMPT,
    ProgramBank,
    best_oracle_option,
    case_key,
    dedupe_pool,
    parse_policy_action,
    query_option_rows,
    render_policy_prompt,
    run_policy,
    selection_snapshot,
    summarize_candidates,
    summarize_policy_rows,
)
from src.dsl import normalize_program, program_is_valid  # noqa: E402
from src.modeling import load_generation_model, load_jsonl, load_tokenizer  # noqa: E402
from src.prompts import messages_for_sketch_record, shuffled_visible  # noqa: E402
from src.sketch import complete_sketch, sketch_hole_count, sketch_variants  # noqa: E402


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


def policy_chat_prompt(tokenizer, prompt: str) -> str:
    messages = [
        {"role": "system", "content": POLICY_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_policy_action(model, tokenizer, prompt: str, args) -> str:
    chat_prompt = policy_chat_prompt(tokenizer, prompt)
    inputs = tokenizer(chat_prompt, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=args.policy_max_length).to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=args.policy_max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )[0]
    return decode_new(tokenizer, output, inputs.input_ids.shape[1]).strip()


def choose_learned_case(
    *,
    record: dict[str, Any],
    bank: ProgramBank,
    observed: list[dict[str, Any]],
    pool: list[dict[str, Any]],
    used: set[str],
    step: int,
    model,
    tokenizer,
    args,
    rng: random.Random,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    options = query_option_rows(
        bank=bank,
        observed=observed,
        pool=pool,
        used=used,
        max_policy_candidates=args.max_policy_candidates,
        max_options=args.policy_max_options,
        max_buckets=args.policy_max_buckets,
        rng=rng,
    )
    if not options:
        return None, {"policy_options": 0, "parse_ok": False}
    prompt = render_policy_prompt(record=record, observed=observed, options=options, step=step)
    raw_action = generate_policy_action(model, tokenizer, prompt, args)
    valid_ids = {option["option_id"] for option in options}
    parsed_action = parse_policy_action(raw_action, valid_ids)
    parse_ok = parsed_action is not None
    if parsed_action is None:
        chosen = options[0]
        parsed_action = chosen["option_id"]
    else:
        chosen = next(option for option in options if option["option_id"] == parsed_action)
    oracle = best_oracle_option(options)
    chosen_rank = next(index for index, option in enumerate(options) if option["option_id"] == chosen["option_id"])
    details = {
        "policy_options": len(options),
        "raw_action": raw_action,
        "parsed_action": parsed_action,
        "parse_ok": parse_ok,
        "fallback_used": not parse_ok,
        "chosen_option_rank": chosen_rank,
        "chosen_entropy": round(chosen["entropy"], 6),
        "chosen_output_buckets": chosen["output_buckets"],
        "chosen_largest_bucket": chosen["largest_bucket"],
        "chosen_max_possible_eliminated": chosen["max_possible_eliminated"],
        "chosen_actual_eliminated": chosen["actual_eliminated"],
        "oracle_option_id": oracle["option_id"] if oracle else None,
        "oracle_actual_eliminated": oracle["actual_eliminated"] if oracle else None,
        "oracle_entropy": round(oracle["entropy"], 6) if oracle else None,
    }
    return chosen["case"], details


def run_learned_policy(
    *,
    record: dict[str, Any],
    bank: ProgramBank,
    query_pool: list[dict[str, Any]],
    budgets: list[int],
    max_policy_candidates: int,
    model,
    tokenizer,
    args,
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
                policy="learned_qwen_policy",
                budget=0,
                repeat=0,
                query_trace=query_trace,
            )
        )
    for step in range(1, max_budget + 1):
        chosen, details = choose_learned_case(
            record=record,
            bank=bank,
            observed=observed,
            pool=query_pool,
            used=used,
            step=step,
            model=model,
            tokenizer=tokenizer,
            args=args,
            rng=rng,
        )
        if chosen is None:
            if step in budget_set:
                results.append(
                    selection_snapshot(
                        record=record,
                        bank=bank,
                        observed=observed,
                        policy="learned_qwen_policy",
                        budget=step,
                        repeat=0,
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
                    policy="learned_qwen_policy",
                    budget=step,
                    repeat=0,
                    query_trace=query_trace,
                )
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--sketch-adapter", type=Path)
    parser.add_argument("--policy-adapter", type=Path)
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
    parser.add_argument("--max-total-programs-per-record", type=int, default=4000)
    parser.add_argument("--max-hole-options", type=int, default=28)
    parser.add_argument("--max-sketch-variants", type=int, default=16)
    parser.add_argument("--budgets", default="0,1,2,3,6")
    parser.add_argument("--random-repeats", type=int, default=2)
    parser.add_argument("--max-policy-candidates", type=int, default=128)
    parser.add_argument("--max-query-pool-cases", type=int, default=48)
    parser.add_argument("--policy-max-options", type=int, default=24)
    parser.add_argument("--policy-max-buckets", type=int, default=6)
    parser.add_argument("--policy-max-length", type=int, default=4096)
    parser.add_argument("--policy-max-new-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260702)
    args = parser.parse_args()

    budgets = sorted({int(item) for item in args.budgets.split(",") if item.strip()})
    if not budgets or budgets[0] != 0:
        budgets = sorted(set([0] + budgets))

    records = load_jsonl(args.data)
    if args.max_records:
        records = records[: args.max_records]

    sketch_tokenizer = None
    sketch_model = None
    trace_overrides = [None] * len(records)
    if args.sketch_source == "model":
        if args.sketch_adapter is None:
            raise ValueError("--sketch-adapter is required with --sketch-source model")
        sketch_tokenizer = load_tokenizer(args.model_id, args.revision, args.cache_dir)
        sketch_model = load_generation_model(
            args.model_id,
            args.revision,
            adapter=str(args.sketch_adapter),
            cache_dir=args.cache_dir,
        )
        trace_overrides = shuffled_visible(records, seed=99) if args.prompt_mode == "shuffled_trace" else [None] * len(records)

    learned_enabled = args.policy_adapter is not None
    policy_tokenizer = None
    policy_model = None
    if learned_enabled:
        policy_tokenizer = load_tokenizer(args.model_id, args.revision, args.cache_dir)
        policy_model = load_generation_model(
            args.model_id,
            args.revision,
            adapter=str(args.policy_adapter),
            cache_dir=args.cache_dir,
        )

    candidate_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    for index, record in enumerate(tqdm(records, desc=f"eval-learned:{args.data.stem}")):
        if args.sketch_source == "target":
            sketches = target_sketch_row(record)
        else:
            assert sketch_model is not None and sketch_tokenizer is not None
            prompt = prompt_text(sketch_tokenizer, record, args.prompt_mode, trace_override=trace_overrides[index])
            sketches = unique_sketch_candidates(generate_sketches(sketch_model, sketch_tokenizer, prompt, args))
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
        if learned_enabled:
            assert policy_model is not None and policy_tokenizer is not None
            policy_rows.extend(
                run_learned_policy(
                    record=record,
                    bank=bank,
                    query_pool=query_pool,
                    budgets=budgets,
                    max_policy_candidates=args.max_policy_candidates,
                    model=policy_model,
                    tokenizer=policy_tokenizer,
                    args=args,
                    rng=random.Random(args.seed + index * 7919),
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

    learned_traces = [
        trace
        for row in policy_rows
        if row["policy"] == "learned_qwen_policy"
        for trace in row.get("query_trace", [])
    ]
    learned_parse_rate = (
        sum(1 for trace in learned_traces if trace.get("parse_ok")) / len(learned_traces)
        if learned_traces
        else None
    )
    result = {
        "experiment": "qwen35_4b_learned_active_trace_policy",
        "model_id": args.model_id,
        "revision": args.revision,
        "sketch_adapter": str(args.sketch_adapter) if args.sketch_adapter else None,
        "policy_adapter": str(args.policy_adapter) if args.policy_adapter else None,
        "prompt_mode": args.prompt_mode,
        "sketch_source": args.sketch_source,
        "data": str(args.data),
        "records": len(records),
        "budgets": budgets,
        "random_repeats": args.random_repeats,
        "max_policy_candidates": args.max_policy_candidates,
        "max_query_pool_cases": args.max_query_pool_cases,
        "policy_max_options": args.policy_max_options,
        "policy_max_buckets": args.policy_max_buckets,
        "policy_max_length": args.policy_max_length,
        "learned_parse_rate": learned_parse_rate,
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
    print(
        json.dumps(
            {
                "candidate_summary": result["candidate_summary"],
                "policy_summary": result["policy_summary"]["overall"],
                "learned_parse_rate": learned_parse_rate,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
