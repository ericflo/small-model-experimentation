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
    *,
    mode: str,
    bridge_total: int,
    adaptive_min_per_family: int,
) -> dict[str, int]:
    if mode == "fixed":
        return dict(STATIC_BRIDGE_ALLOCATION)
    scores = {family: wrong_candidate_score(rows_by_family[family]) for family in FRONTIER_FAMILIES}
    if sum(scores.values()) == 0:
        return dict(STATIC_BRIDGE_ALLOCATION)
    allocation = {
        family: min(adaptive_min_per_family, len(rows_by_family[family]))
        for family in FRONTIER_FAMILIES
    }
    remaining = bridge_total - sum(allocation.values())
    while remaining > 0:
        candidates = [
            family
            for family in FRONTIER_FAMILIES
            if allocation[family] < len(rows_by_family[family])
        ]
        if not candidates:
            break
        family = max(
            candidates,
            key=lambda item: (
                scores[item] / max(1, allocation[item]),
                scores[item],
                -allocation[item],
                item,
            ),
        )
        allocation[family] += 1
        remaining -= 1
    return allocation


def build_model_mined_record(
    *,
    source: dict[str, Any],
    family_index: int,
    wrong_programs: list[str],
    trace_strategy: str,
    rng: random.Random,
    visible_count: int,
    selector_pool_size: int,
) -> dict[str, Any]:
    spec = make_specs()[source["family"]]
    selector_programs = wrong_programs or list(spec.static_distractors)
    chosen_wrong = selector_programs[0]
    visible, selection_stats = select_counterexample_cases(
        spec,
        rng,
        selector_programs,
        count=visible_count,
        pool_size=selector_pool_size,
    )
    return {
        "id": f"train_{trace_strategy}_{source['family']}_{family_index:03d}",
        "split": "train_bridge",
        "trace_strategy": trace_strategy if wrong_programs else f"{trace_strategy}_fallback_static",
        "family": source["family"],
        "schema": spec.schema,
        "wrong_program": chosen_wrong,
        "target_program": spec.program,
        "visible": attach_got(chosen_wrong, visible),
        "hidden": source["hidden"],
        "static_distractors": list(spec.static_distractors),
        "selector_programs": selector_programs,
        "selection_stats": selection_stats,
        "mined_wrong_program_count": len(wrong_programs),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mining-data", type=Path, default=ROOT / "data" / "mining" / "dsl_mining_pool.jsonl")
    parser.add_argument("--base-anchor", type=Path, default=ROOT / "data" / "base_anchor" / "dsl_train_base_anchor.jsonl")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--allocation-mode", choices=["fixed", "adaptive"], default="fixed")
    parser.add_argument("--trace-strategy", choices=["seed_mined", "adaptive_mined"], default="seed_mined")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--revision", default="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
    parser.add_argument("--cache-dir", default="/workspace/.cache/huggingface")
    parser.add_argument("--output-train", type=Path, default=ROOT / "data" / "seed_mined" / "dsl_train.jsonl")
    parser.add_argument("--output-bridge", type=Path, default=ROOT / "data" / "seed_mined" / "bridge_records.jsonl")
    parser.add_argument("--output-report", type=Path, default=ROOT / "reports" / "mining" / "seed_mined_mining.json")
    parser.add_argument("--bridge-total", type=int, default=BRIDGE_TOTAL)
    parser.add_argument("--adaptive-min-per-family", type=int, default=2)
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
        mode=args.allocation_mode,
        bridge_total=args.bridge_total,
        adaptive_min_per_family=args.adaptive_min_per_family,
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
            row_wrong_programs = [
                candidate["program"]
                for candidate in row["candidates"]
                if candidate["is_wrong"]
            ]
            # Add family-level wrong programs so selection is robust when a row has only one sampled error.
            selector_programs = list(dict.fromkeys(row_wrong_programs + global_wrong_programs[:8]))
            model_bridge.append(
                build_model_mined_record(
                    source=source_by_id[row["id"]],
                    family_index=index,
                    wrong_programs=selector_programs,
                    trace_strategy=args.trace_strategy,
                    rng=rng,
                    visible_count=args.visible_cases,
                    selector_pool_size=args.selector_pool_size,
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
