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

from src.graphir import graph_is_valid, graph_pass_count, normalize_graph  # noqa: E402
from src.modeling import load_generation_model, load_jsonl, load_tokenizer  # noqa: E402
from src.prompts import messages_for_record, shuffled_visible  # noqa: E402


def prompt_text(tokenizer, record: dict[str, Any], prompt_mode: str, trace_override=None) -> str:
    messages = messages_for_record(record, prompt_mode=prompt_mode, trace_override=trace_override)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def decode_new(tokenizer, output_ids, prompt_len: int) -> str:
    return tokenizer.decode(output_ids[prompt_len:], skip_special_tokens=True)


def generate_candidates(model, tokenizer, prompt: str, args) -> list[dict[str, Any]]:
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    candidates = [
        {"kind": "candidate_input", "raw": "", "graph": None},
    ]
    with torch.no_grad():
        greedy = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )[0]
    raw = decode_new(tokenizer, greedy, inputs.input_ids.shape[1])
    candidates.append({"kind": "greedy", "raw": raw, "graph": normalize_graph(raw)})
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
            candidates.append({"kind": f"sample_{i:02d}", "raw": raw, "graph": normalize_graph(raw)})
    return candidates


def select_graph(candidates: list[dict[str, Any]], visible: list[dict[str, Any]], original_graph: str) -> dict[str, Any]:
    best = None
    for candidate in candidates:
        graph = original_graph if candidate["graph"] is None else candidate["graph"]
        candidate["graph"] = graph
        candidate["valid"] = graph_is_valid(graph)
        candidate["visible_passes"] = graph_pass_count(graph, visible)
        candidate["visible_total"] = len(visible)
        score = (candidate["visible_passes"], int(candidate["valid"]), -len(graph))
        if best is None or score > best[0]:
            best = (score, candidate)
    assert best is not None
    return best[1]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_family = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)

    def metric(key: str, subset: list[dict[str, Any]]) -> dict[str, Any]:
        successes = sum(1 for row in subset if row[key])
        return {"successes": successes, "records": len(subset), "rate": successes / len(subset) if subset else 0.0}

    return {
        "overall": {
            "input_hidden_all": metric("input_hidden_all", rows),
            "repair_hidden_all": metric("repair_hidden_all", rows),
            "input_visible_all": metric("input_visible_all", rows),
            "repair_visible_all": metric("repair_visible_all", rows),
        },
        "by_family": {
            family: {
                "input_hidden_all": metric("input_hidden_all", subset),
                "repair_hidden_all": metric("repair_hidden_all", subset),
                "input_visible_all": metric("input_visible_all", subset),
                "repair_visible_all": metric("repair_visible_all", subset),
            }
            for family, subset in sorted(by_family.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
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
    args = parser.parse_args()

    records = load_jsonl(args.data)
    if args.max_records:
        records = records[: args.max_records]
    tokenizer = load_tokenizer(args.model_id, args.revision, args.cache_dir)
    model = load_generation_model(
        args.model_id,
        args.revision,
        adapter=str(args.adapter),
        cache_dir=args.cache_dir,
    )
    trace_overrides = shuffled_visible(records, seed=99) if args.prompt_mode == "shuffled_trace" else [None] * len(records)
    rows = []
    for index, record in enumerate(tqdm(records, desc="repair")):
        prompt = prompt_text(tokenizer, record, args.prompt_mode, trace_override=trace_overrides[index])
        candidates = generate_candidates(model, tokenizer, prompt, args)
        selected = select_graph(candidates, record["visible"], record["candidate_graph"])
        input_visible = graph_pass_count(record["candidate_graph"], record["visible"])
        input_hidden = graph_pass_count(record["candidate_graph"], record["hidden"])
        repair_hidden = graph_pass_count(selected["graph"], record["hidden"])
        rows.append(
            {
                "id": record["id"],
                "family": record["family"],
                "target_graph": record["target_graph"],
                "candidate_graph": record["candidate_graph"],
                "selected_graph": selected["graph"],
                "input_visible_passes": input_visible,
                "input_hidden_passes": input_hidden,
                "repair_visible_passes": selected["visible_passes"],
                "repair_hidden_passes": repair_hidden,
                "visible_total": len(record["visible"]),
                "hidden_total": len(record["hidden"]),
                "input_visible_all": input_visible == len(record["visible"]),
                "input_hidden_all": input_hidden == len(record["hidden"]),
                "repair_visible_all": selected["visible_passes"] == len(record["visible"]),
                "repair_hidden_all": repair_hidden == len(record["hidden"]),
                "selected_kind": selected["kind"],
                "candidate_count": len(candidates),
                "candidates": candidates,
            }
        )

    result = {
        "model_id": args.model_id,
        "revision": args.revision,
        "adapter": str(args.adapter),
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
