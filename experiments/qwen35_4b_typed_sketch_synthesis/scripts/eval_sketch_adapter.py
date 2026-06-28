#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dsl import normalize_program, program_is_valid, program_pass_count  # noqa: E402
from src.modeling import load_generation_model, load_jsonl, load_tokenizer  # noqa: E402
from src.prompts import messages_for_sketch_record, shuffled_visible  # noqa: E402
from src.sketch import complete_sketch, select_by_visible, sketch_hole_count, sketch_variants  # noqa: E402


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


def load_baseline_rows(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {row["id"]: row for row in data.get("rows", [])}


def synthesize_from_sketches(sketches: list[dict[str, Any]], record: dict[str, Any], args) -> tuple[list[str], list[dict[str, Any]]]:
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


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_family = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)

    def metric(key: str, subset: list[dict[str, Any]]) -> dict[str, Any]:
        successes = sum(1 for row in subset if row[key])
        return {"successes": successes, "records": len(subset), "rate": successes / len(subset) if subset else 0.0}

    def avg(key: str, subset: list[dict[str, Any]]) -> float:
        return sum(float(row.get(key, 0)) for row in subset) / len(subset) if subset else 0.0

    def block(subset: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "base_rerank_hidden_all": metric("base_rerank_hidden_all", subset),
            "sketch_synth_hidden_all": metric("sketch_synth_hidden_all", subset),
            "sketch_synth_oracle_hidden_all": metric("sketch_synth_oracle_hidden_all", subset),
            "hybrid_hidden_all": metric("hybrid_hidden_all", subset),
            "base_rerank_visible_all": metric("base_rerank_visible_all", subset),
            "sketch_synth_visible_all": metric("sketch_synth_visible_all", subset),
            "hybrid_visible_all": metric("hybrid_visible_all", subset),
            "target_program_synthesized": metric("target_program_synthesized", subset),
            "avg_synthesized_programs": round(avg("synthesized_program_count", subset), 3),
            "avg_unique_sketches": round(avg("unique_sketch_count", subset), 3),
            "improved_over_base_hidden": sum(1 for row in subset if row["sketch_synth_hidden_passes"] > row["base_rerank_hidden_passes"]),
            "damaged_vs_base_hidden": sum(1 for row in subset if row["sketch_synth_hidden_passes"] < row["base_rerank_hidden_passes"]),
            "hybrid_improved_over_base_hidden": sum(1 for row in subset if row["hybrid_hidden_passes"] > row["base_rerank_hidden_passes"]),
            "hybrid_damaged_vs_base_hidden": sum(1 for row in subset if row["hybrid_hidden_passes"] < row["base_rerank_hidden_passes"]),
        }

    return {
        "overall": block(rows),
        "by_family": {family: block(subset) for family, subset in sorted(by_family.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--baseline-result", type=Path)
    parser.add_argument("--prompt-mode", choices=["trace", "no_trace", "shuffled_trace"], default="trace")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--revision", default="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
    parser.add_argument("--cache-dir", default="/workspace/.cache/huggingface")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--max-programs-per-sketch", type=int, default=5000)
    parser.add_argument("--max-total-programs-per-record", type=int, default=16000)
    parser.add_argument("--max-hole-options", type=int, default=28)
    parser.add_argument("--max-sketch-variants", type=int, default=16)
    args = parser.parse_args()

    records = load_jsonl(args.data)
    if args.max_records:
        records = records[: args.max_records]
    baseline_rows = load_baseline_rows(args.baseline_result)
    tokenizer = load_tokenizer(args.model_id, args.revision, args.cache_dir)
    model = load_generation_model(
        args.model_id,
        args.revision,
        adapter=str(args.adapter),
        cache_dir=args.cache_dir,
    )
    trace_overrides = shuffled_visible(records, seed=99) if args.prompt_mode == "shuffled_trace" else [None] * len(records)
    rows = []
    for index, record in enumerate(tqdm(records, desc="eval-sketch")):
        prompt = prompt_text(tokenizer, record, args.prompt_mode, trace_override=trace_overrides[index])
        sketch_candidates = generate_sketches(model, tokenizer, prompt, args)
        unique_sketches = []
        seen_sketches = set()
        for candidate in sketch_candidates:
            if candidate["sketch"] in seen_sketches:
                continue
            seen_sketches.add(candidate["sketch"])
            unique_sketches.append(candidate)
        synthesized_programs, sketch_rows = synthesize_from_sketches(unique_sketches, record, args)
        selected = select_by_visible(synthesized_programs, record["visible"], record["hidden"])
        selected_hidden = selected["hidden_passes"] or 0
        oracle_hidden = max((program_pass_count(program, record["hidden"]) for program in synthesized_programs), default=0)
        oracle_visible = max((program_pass_count(program, record["visible"]) for program in synthesized_programs), default=0)
        baseline = baseline_rows.get(record["id"], {})
        base_hidden = int(baseline.get("selected_hidden_passes", 0))
        base_visible = int(baseline.get("selected_visible_passes", 0))
        if base_visible == len(record["visible"]):
            hybrid_source = "base_visible_all"
            hybrid_visible = base_visible
            hybrid_hidden = base_hidden
            hybrid_program = baseline.get("selected_program")
        elif selected["visible_passes"] == len(record["visible"]):
            hybrid_source = "sketch_visible_all"
            hybrid_visible = selected["visible_passes"]
            hybrid_hidden = selected_hidden
            hybrid_program = selected["program"]
        elif selected["visible_passes"] > base_visible:
            hybrid_source = "sketch_more_visible"
            hybrid_visible = selected["visible_passes"]
            hybrid_hidden = selected_hidden
            hybrid_program = selected["program"]
        else:
            hybrid_source = "base_more_or_tie_visible"
            hybrid_visible = base_visible
            hybrid_hidden = base_hidden
            hybrid_program = baseline.get("selected_program")
        row = {
            "id": record["id"],
            "family": record["family"],
            "trace_strategy": record["trace_strategy"],
            "target_program": record["target_program"],
            "target_sketch": record.get("target_sketch"),
            "base_selected_program": baseline.get("selected_program"),
            "sketch_selected_program": selected["program"],
            "base_rerank_visible_passes": base_visible,
            "base_rerank_hidden_passes": base_hidden,
            "sketch_synth_visible_passes": selected["visible_passes"],
            "sketch_synth_hidden_passes": selected_hidden,
            "sketch_synth_oracle_visible_passes": oracle_visible,
            "sketch_synth_oracle_hidden_passes": oracle_hidden,
            "hybrid_source": hybrid_source,
            "hybrid_program": hybrid_program,
            "hybrid_visible_passes": hybrid_visible,
            "hybrid_hidden_passes": hybrid_hidden,
            "visible_total": len(record["visible"]),
            "hidden_total": len(record["hidden"]),
            "base_rerank_visible_all": base_visible == len(record["visible"]),
            "base_rerank_hidden_all": base_hidden == len(record["hidden"]),
            "sketch_synth_visible_all": selected["visible_passes"] == len(record["visible"]),
            "sketch_synth_hidden_all": selected_hidden == len(record["hidden"]),
            "sketch_synth_oracle_visible_all": oracle_visible == len(record["visible"]),
            "sketch_synth_oracle_hidden_all": oracle_hidden == len(record["hidden"]),
            "hybrid_visible_all": hybrid_visible == len(record["visible"]),
            "hybrid_hidden_all": hybrid_hidden == len(record["hidden"]),
            "target_program_synthesized": record["target_program"] in synthesized_programs,
            "unique_sketch_count": len(unique_sketches),
            "synthesized_program_count": len(synthesized_programs),
            "sketches": sketch_rows,
            "selected": {key: value for key, value in selected.items() if key != "ranked"},
        }
        rows.append(row)

    result = {
        "model_id": args.model_id,
        "revision": args.revision,
        "adapter": str(args.adapter),
        "prompt_mode": args.prompt_mode,
        "data": str(args.data),
        "baseline_result": str(args.baseline_result) if args.baseline_result else None,
        "records": len(records),
        "num_samples": args.num_samples,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_programs_per_sketch": args.max_programs_per_sketch,
        "max_total_programs_per_record": args.max_total_programs_per_record,
        "max_hole_options": args.max_hole_options,
        "max_sketch_variants": args.max_sketch_variants,
        "summary": summarize(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
