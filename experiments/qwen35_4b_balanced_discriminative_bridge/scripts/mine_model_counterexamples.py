#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_gen import (  # noqa: E402
    BRIDGE_TOTAL,
    FRONTIER_FAMILIES,
    STATIC_BRIDGE_ALLOCATION,
    attach_got,
    alias_selector_programs,
    make_specs,
    select_counterexample_cases,
    write_jsonl,
)
from src.dsl import execute, normalize_program, program_is_valid, program_pass_count  # noqa: E402
from src.modeling import load_generation_model, load_jsonl, load_tokenizer  # noqa: E402
from src.prompts import messages_for_record  # noqa: E402


def prompt_text(tokenizer, record: dict[str, Any]) -> str:
    messages = messages_for_record(record, prompt_mode="trace")
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def decode_new(tokenizer, output_ids, prompt_len: int) -> str:
    return tokenizer.decode(output_ids[prompt_len:], skip_special_tokens=True)


def generate_candidates(model, tokenizer, prompt: str, args) -> list[dict[str, Any]]:
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
    candidates.append({"kind": "greedy", "raw": raw, "program": normalize_program(raw)})
    for i in range(args.num_samples):
        with torch.no_grad():
            sampled = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.pad_token_id,
            )[0]
        raw = decode_new(tokenizer, sampled, inputs.input_ids.shape[1])
        candidates.append({"kind": f"sample_{i:02d}", "raw": raw, "program": normalize_program(raw)})
    return candidates


def candidate_stats(record: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    program = candidate["program"]
    case_pool = record.get("case_pool") or record["hidden"]
    valid = program_is_valid(program)
    executable_cases = 0
    if valid:
        for case in case_pool:
            try:
                execute(program, case["input"])
            except Exception:
                continue
            executable_cases += 1
    executes_all = valid and executable_cases == len(case_pool)
    pool_passes = program_pass_count(program, case_pool) if valid else 0
    visible_passes = program_pass_count(program, record["visible"]) if valid else 0
    is_target = program.strip() == record["target_program"].strip()
    is_wrong = executes_all and not is_target and pool_passes < len(case_pool)
    return {
        **candidate,
        "valid": valid,
        "executable_cases": executable_cases,
        "executes_all": executes_all,
        "is_target": is_target,
        "is_wrong": is_wrong,
        "visible_passes": visible_passes,
        "visible_total": len(record["visible"]),
        "pool_passes": pool_passes,
        "pool_total": len(case_pool),
    }


def choose_wrong_programs(rows: list[dict[str, Any]]) -> list[str]:
    counts = Counter()
    for row in rows:
        for candidate in row["candidates"]:
            if candidate["is_wrong"]:
                counts[candidate["program"]] += 1
    return [program for program, _ in counts.most_common()]


def wrong_candidate_score(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows for candidate in row["candidates"] if candidate["is_wrong"])


def compute_allocation(
    rows_by_family: dict[str, list[dict[str, Any]]],
) -> dict[str, int]:
    missing = [family for family in FRONTIER_FAMILIES if len(rows_by_family[family]) < STATIC_BRIDGE_ALLOCATION[family]]
    if missing:
        raise RuntimeError(f"not enough mining rows for fixed allocation: {missing}")
    return dict(STATIC_BRIDGE_ALLOCATION)


def build_model_mined_record(
    *,
    source: dict[str, Any],
    family_index: int,
    wrong_programs: list[str],
    trace_strategy: str,
    rng: random.Random,
    visible_count: int,
    selector_pool_size: int,
    selector_case_mode: str,
    model_wrong_program_count: int,
) -> dict[str, Any]:
    spec = make_specs()[source["family"]]
    selector_programs = wrong_programs or alias_selector_programs(spec)
    chosen_wrong = selector_programs[0]
    visible, selection_stats = select_counterexample_cases(
        spec,
        rng,
        selector_programs,
        count=visible_count,
        pool_size=selector_pool_size,
        case_mode=selector_case_mode,
    )
    return {
        "id": f"train_{trace_strategy}_{source['family']}_{family_index:03d}",
        "split": "train_bridge",
        "trace_strategy": trace_strategy if model_wrong_program_count else f"{trace_strategy}_fallback_alias",
        "family": source["family"],
        "schema": spec.schema,
        "wrong_program": chosen_wrong,
        "target_program": spec.program,
        "visible": attach_got(chosen_wrong, visible),
        "hidden": source["hidden"],
        "static_distractors": list(spec.static_distractors),
        "selector_programs": selector_programs,
        "selection_stats": selection_stats,
        "mined_wrong_program_count": model_wrong_program_count,
        "selector_program_count": len(selector_programs),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mining-data", type=Path, default=ROOT / "data" / "mining" / "dsl_mining_pool.jsonl")
    parser.add_argument("--base-anchor", type=Path, default=ROOT / "data" / "base_anchor" / "dsl_train_base_anchor.jsonl")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--allocation-mode", choices=["fixed"], default="fixed")
    parser.add_argument("--trace-strategy", choices=["model_discriminative"], default="model_discriminative")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--revision", default="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
    parser.add_argument("--cache-dir", default="/workspace/.cache/huggingface")
    parser.add_argument("--output-train", type=Path, default=ROOT / "data" / "model_discriminative" / "dsl_train.jsonl")
    parser.add_argument("--output-bridge", type=Path, default=ROOT / "data" / "model_discriminative" / "bridge_records.jsonl")
    parser.add_argument("--output-report", type=Path, default=ROOT / "reports" / "mining" / "model_discriminative_mining.json")
    parser.add_argument("--bridge-total", type=int, default=BRIDGE_TOTAL)
    parser.add_argument("--selector-case-mode", choices=["normal", "hard", "mixed"], default="hard")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--visible-cases", type=int, default=6)
    parser.add_argument("--selector-pool-size", type=int, default=192)
    parser.add_argument("--seed", type=int, default=20260624)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    records = load_jsonl(args.mining_data)
    tokenizer = load_tokenizer(args.model_id, args.revision, args.cache_dir)
    model = load_generation_model(
        args.model_id,
        args.revision,
        adapter=str(args.adapter),
        cache_dir=args.cache_dir,
    )

    mined_rows = []
    for record in tqdm(records, desc="mine"):
        prompt = prompt_text(tokenizer, record)
        candidates = [
            candidate_stats(record, candidate)
            for candidate in generate_candidates(model, tokenizer, prompt, args)
        ]
        mined_rows.append(
            {
                "id": record["id"],
                "family": record["family"],
                "target_program": record["target_program"],
                "prompt_wrong_program": record["wrong_program"],
                "candidates": candidates,
            }
        )

    rows_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in mined_rows:
        rows_by_family[row["family"]].append(row)

    source_by_id = {record["id"]: record for record in records}
    model_bridge = []
    allocation = compute_allocation(
        rows_by_family,
    )
    family_summaries = {}
    for family, wanted in allocation.items():
        family_rows = rows_by_family[family]
        global_wrong_programs = choose_wrong_programs(family_rows)
        usable_rows = [
            row
            for row in family_rows
            if any(candidate["is_wrong"] for candidate in row["candidates"])
        ]
        fallback_rows = [row for row in family_rows if row not in usable_rows]
        ordered_rows = usable_rows + fallback_rows
        if len(ordered_rows) < wanted:
            raise RuntimeError(f"not enough mining rows for {family}: need {wanted}, have {len(ordered_rows)}")
        selected_rows = ordered_rows[:wanted]
        family_summaries[family] = {
            "mining_records": len(family_rows),
            "bridge_records_requested": wanted,
            "bridge_records_with_model_wrong": len(usable_rows[:wanted]),
            "wrong_candidate_score": wrong_candidate_score(family_rows),
            "unique_wrong_programs": len(global_wrong_programs),
            "top_wrong_programs": [
                {"program": program, "count": count}
                for program, count in Counter(
                    candidate["program"]
                    for row in family_rows
                    for candidate in row["candidates"]
                    if candidate["is_wrong"]
                ).most_common(8)
            ],
        }
        for index, row in enumerate(selected_rows):
            spec = make_specs()[family]
            row_wrong_programs = [
                candidate["program"]
                for candidate in row["candidates"]
                if candidate["is_wrong"]
            ]
            # Add family-level wrong programs so selection is robust when a row has only one sampled error.
            model_selector_programs = list(dict.fromkeys(row_wrong_programs + global_wrong_programs[:8]))
            selector_programs = list(dict.fromkeys(model_selector_programs + alias_selector_programs(spec)))
            model_bridge.append(
                build_model_mined_record(
                    source=source_by_id[row["id"]],
                    family_index=index,
                    wrong_programs=selector_programs,
                    trace_strategy=args.trace_strategy,
                    rng=rng,
                    visible_count=args.visible_cases,
                    selector_pool_size=args.selector_pool_size,
                    selector_case_mode=args.selector_case_mode,
                    model_wrong_program_count=len(model_selector_programs),
                )
            )

    base_anchor = load_jsonl(args.base_anchor)
    train_rows = base_anchor + model_bridge
    write_jsonl(args.output_bridge, model_bridge)
    write_jsonl(args.output_train, train_rows)

    report = {
        "model_id": args.model_id,
        "revision": args.revision,
        "adapter": str(args.adapter),
        "mining_data": str(args.mining_data),
        "records": len(records),
        "num_samples": args.num_samples,
        "max_new_tokens": args.max_new_tokens,
        "allocation_mode": args.allocation_mode,
        "trace_strategy": args.trace_strategy,
        "selector_case_mode": args.selector_case_mode,
        "bridge_allocation": allocation,
        "train_records": len(train_rows),
        "base_anchor_records": len(base_anchor),
        "mined_bridge_records": len(model_bridge),
        "family_summaries": family_summaries,
        "rows": mined_rows,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    (args.output_train.parent / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "base_anchor_records": len(base_anchor),
                "mined_bridge_records": len(model_bridge),
                "train_records": len(train_rows),
                "allocation_mode": args.allocation_mode,
                "trace_strategy": args.trace_strategy,
                "selector_case_mode": args.selector_case_mode,
                "bridge_allocation": allocation,
                "family_summaries": family_summaries,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(json.dumps(report["family_summaries"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
