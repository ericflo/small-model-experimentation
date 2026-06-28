#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.graphir import graph_is_valid, graph_pass_count, normalize_graph, safe_execute_graph  # noqa: E402
from src.modeling import load_generation_model, load_jsonl, load_tokenizer  # noqa: E402
from src.prompts import messages_for_record, shuffled_visible  # noqa: E402


def prompt_text(tokenizer, record: dict[str, Any], prompt_mode: str, trace_override=None) -> str:
    messages = messages_for_record(record, prompt_mode=prompt_mode, trace_override=trace_override)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def decode_new(tokenizer, output_ids, prompt_len: int) -> str:
    return tokenizer.decode(output_ids[prompt_len:], skip_special_tokens=True)


def generate_candidates(model, tokenizer, prompt: str, args, *, prefix: str) -> list[dict[str, Any]]:
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
    candidates.append({"kind": f"{prefix}_greedy", "raw": raw, "graph": normalize_graph(raw)})
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
            candidates.append({"kind": f"{prefix}_sample_{i:02d}", "raw": raw, "graph": normalize_graph(raw)})
    return candidates


def select_graph(candidates: list[dict[str, Any]], visible: list[dict[str, Any]]) -> dict[str, Any]:
    best = None
    for candidate in candidates:
        graph = candidate["graph"]
        candidate["valid"] = graph_is_valid(graph)
        candidate["visible_passes"] = graph_pass_count(graph, visible)
        candidate["visible_total"] = len(visible)
        score = (candidate["visible_passes"], int(candidate["valid"]), -len(graph))
        if best is None or score > best[0]:
            best = (score, candidate)
    assert best is not None
    return best[1]


def add_candidate_got(cases: list[dict[str, Any]], graph: str) -> list[dict[str, Any]]:
    return [{**case, "candidate_got": safe_execute_graph(graph, case["input"])} for case in cases]


def make_repair_record(record: dict[str, Any], candidate_graph: str) -> dict[str, Any]:
    row = dict(record)
    row["task"] = "graph_repair"
    row["candidate_graph"] = candidate_graph
    row["visible"] = add_candidate_got(record["visible"], candidate_graph)
    row["target_output"] = record["target_graph"]
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_family = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)

    def metric(key: str, subset: list[dict[str, Any]]) -> dict[str, Any]:
        successes = sum(1 for row in subset if row[key])
        return {"successes": successes, "records": len(subset), "rate": successes / len(subset) if subset else 0.0}

    keys = [
        "construct_greedy_hidden_all",
        "construct_rerank_hidden_all",
        "repair_hidden_all",
        "construct_greedy_visible_all",
        "construct_rerank_visible_all",
        "repair_visible_all",
    ]
    return {
        "overall": {key: metric(key, rows) for key in keys},
        "by_family": {
            family: {key: metric(key, subset) for key in keys}
            for family, subset in sorted(by_family.items())
        },
    }


def unload(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--construct-adapter", type=Path, required=True)
    parser.add_argument("--repair-adapter", type=Path)
    parser.add_argument("--prompt-mode", choices=["trace", "no_trace", "shuffled_trace"], default="trace")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--revision", default="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
    parser.add_argument("--cache-dir", default="/workspace/.cache/huggingface")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--skip-repair-if-visible-pass", action="store_true")
    args = parser.parse_args()

    records = load_jsonl(args.data)
    if args.max_records:
        records = records[: args.max_records]
    tokenizer = load_tokenizer(args.model_id, args.revision, args.cache_dir)
    trace_overrides = shuffled_visible(records, seed=99) if args.prompt_mode == "shuffled_trace" else [None] * len(records)

    construct_model = load_generation_model(
        args.model_id,
        args.revision,
        adapter=str(args.construct_adapter),
        cache_dir=args.cache_dir,
    )
    rows = []
    for index, record in enumerate(tqdm(records, desc="construct")):
        prompt = prompt_text(tokenizer, record, args.prompt_mode, trace_override=trace_overrides[index])
        candidates = generate_candidates(construct_model, tokenizer, prompt, args, prefix="construct")
        selected = select_graph(candidates, record["visible"])
        greedy = candidates[0]
        greedy_visible = graph_pass_count(greedy["graph"], record["visible"])
        greedy_hidden = graph_pass_count(greedy["graph"], record["hidden"])
        selected_hidden = graph_pass_count(selected["graph"], record["hidden"])
        rows.append(
            {
                "id": record["id"],
                "family": record["family"],
                "target_graph": record["target_graph"],
                "target_program": record["target_program"],
                "wrong_program": record["wrong_program"],
                "construct_greedy_graph": greedy["graph"],
                "construct_selected_graph": selected["graph"],
                "construct_greedy_visible_passes": greedy_visible,
                "construct_greedy_hidden_passes": greedy_hidden,
                "construct_selected_visible_passes": selected["visible_passes"],
                "construct_selected_hidden_passes": selected_hidden,
                "visible_total": len(record["visible"]),
                "hidden_total": len(record["hidden"]),
                "construct_greedy_visible_all": greedy_visible == len(record["visible"]),
                "construct_greedy_hidden_all": greedy_hidden == len(record["hidden"]),
                "construct_rerank_visible_all": selected["visible_passes"] == len(record["visible"]),
                "construct_rerank_hidden_all": selected_hidden == len(record["hidden"]),
                "construct_selected_kind": selected["kind"],
                "construct_candidate_count": len(candidates),
                "construct_candidates": candidates,
                "repair_graph": selected["graph"],
                "repair_visible_passes": selected["visible_passes"],
                "repair_hidden_passes": selected_hidden,
                "repair_visible_all": selected["visible_passes"] == len(record["visible"]),
                "repair_hidden_all": selected_hidden == len(record["hidden"]),
                "repair_candidate_count": 0,
                "repair_candidates": [],
            }
        )
    unload(construct_model)

    if args.repair_adapter:
        repair_model = load_generation_model(
            args.model_id,
            args.revision,
            adapter=str(args.repair_adapter),
            cache_dir=args.cache_dir,
        )
        for index, (record, row) in enumerate(tqdm(list(zip(records, rows)), desc="repair")):
            if args.skip_repair_if_visible_pass and row["construct_rerank_visible_all"]:
                continue
            repair_record = make_repair_record(record, row["construct_selected_graph"])
            repair_override = None
            if args.prompt_mode == "shuffled_trace":
                repair_override = add_candidate_got(trace_overrides[index], row["construct_selected_graph"])
            prompt = prompt_text(tokenizer, repair_record, args.prompt_mode, trace_override=repair_override)
            repair_candidates = [
                {
                    "kind": "pre_repair_selected",
                    "raw": row["construct_selected_graph"],
                    "graph": row["construct_selected_graph"],
                }
            ]
            repair_candidates.extend(generate_candidates(repair_model, tokenizer, prompt, args, prefix="repair"))
            selected = select_graph(repair_candidates, record["visible"])
            selected_hidden = graph_pass_count(selected["graph"], record["hidden"])
            row["repair_graph"] = selected["graph"]
            row["repair_visible_passes"] = selected["visible_passes"]
            row["repair_hidden_passes"] = selected_hidden
            row["repair_visible_all"] = selected["visible_passes"] == len(record["visible"])
            row["repair_hidden_all"] = selected_hidden == len(record["hidden"])
            row["repair_selected_kind"] = selected["kind"]
            row["repair_candidate_count"] = len(repair_candidates)
            row["repair_candidates"] = repair_candidates
        unload(repair_model)

    result = {
        "model_id": args.model_id,
        "revision": args.revision,
        "construct_adapter": str(args.construct_adapter),
        "repair_adapter": str(args.repair_adapter) if args.repair_adapter else None,
        "prompt_mode": args.prompt_mode,
        "data": str(args.data),
        "records": len(records),
        "num_samples": args.num_samples,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "summary": summarize(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
