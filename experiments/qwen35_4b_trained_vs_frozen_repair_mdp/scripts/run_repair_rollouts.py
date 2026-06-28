#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.jsonl import load_jsonl, write_jsonl  # noqa: E402
from src.model_utils import DEFAULT_MODEL_PATH, attach_existing_lora, code_chat_prompt, load_generation_model, load_tokenizer  # noqa: E402
from src.repair_utils import (  # noqa: E402
    EXPERIMENT,
    add_usage,
    candidate_from_completion,
    choose_repair_sources,
    dedupe_candidates,
    empty_usage,
    estimate_text_tokens,
    recompute_record_metrics,
    repair_prompt,
    sample_prompt_with_usage,
    summarize_records,
    write_manifest,
)


def source_summary(candidate: dict[str, Any]) -> str:
    status = "visible_pass" if candidate.get("visible_all_pass") else "visible_fail"
    return f"{candidate.get('candidate_id')}: {status}, public_signature={candidate.get('public_signature', '')}"


def repair_record(record: dict[str, Any], model: Any, tokenizer: Any, args: argparse.Namespace, seed: int, budget_state: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(record)
    candidates = list(out.get("candidates", []))
    usage = empty_usage()
    continuation_prompt = out.get("continuation_prompt")
    order = len(candidates)
    frontier_ids: set[str] | None = None

    for round_index in range(1, args.max_rounds + 1):
        if frontier_ids is None:
            pool_record = dict(out, candidates=candidates)
        else:
            pool_record = dict(out, candidates=[candidate for candidate in candidates if candidate.get("candidate_id") in frontier_ids])
        sources = choose_repair_sources(pool_record, args.max_sources_per_record, include_visible_fail_only=not args.repair_visible_passers)
        if not sources:
            break
        new_ids: set[str] = set()
        prior = [source_summary(candidate) for candidate in candidates[-args.prior_summary_limit :]]
        for source_index, source in enumerate(sources):
            for attempt in range(args.repairs_per_source):
                if args.target_forward_tokens and budget_state["forward_tokens"] >= args.target_forward_tokens:
                    out["repair_budget_exhausted"] = True
                    out["candidates"] = dedupe_candidates(candidates)
                    recompute_record_metrics(out)
                    out["repair_usage"] = usage
                    return out
                prompt_text = repair_prompt(out, source, prior_summaries=prior)
                prompt = code_chat_prompt(tokenizer, prompt_text)
                completions, batch_usage = sample_prompt_with_usage(
                    model,
                    tokenizer,
                    prompt,
                    count=1,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    batch_size=1,
                    seed=seed + round_index * 100000 + source_index * 1000 + attempt,
                )
                usage = add_usage(usage, batch_usage)
                budget_state.update(add_usage(budget_state, batch_usage))
                completion = completions[0]
                candidate = candidate_from_completion(
                    completion,
                    out,
                    source=f"repair_r{round_index}_of_{source['candidate_id']}_{attempt}",
                    order=order,
                    continuation_prompt=continuation_prompt,
                    parent_id=source["candidate_id"],
                    round_index=round_index,
                    prompt_tokens=estimate_text_tokens(tokenizer, prompt),
                    completion_tokens=estimate_text_tokens(tokenizer, completion),
                )
                candidates.append(candidate)
                new_ids.add(candidate["candidate_id"])
                prior.append(source_summary(candidate))
                order += 1
        frontier_ids = new_ids

    out["candidates"] = dedupe_candidates(candidates)
    recompute_record_metrics(out)
    out["repair_usage"] = usage
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--round-name", type=str, required=True)
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--max-rounds", type=int, default=1)
    parser.add_argument("--max-sources-per-record", type=int, default=2)
    parser.add_argument("--repairs-per-source", type=int, default=1)
    parser.add_argument("--repair-visible-passers", action="store_true")
    parser.add_argument("--prior-summary-limit", type=int, default=4)
    parser.add_argument("--target-forward-tokens", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=260)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    random.seed(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    base_records = load_jsonl(args.records)
    tokenizer = load_tokenizer(args.model_path, padding_side="left")
    model = load_generation_model(args.model_path)
    if args.adapter_dir:
        model = attach_existing_lora(model, args.adapter_dir, is_trainable=False)
        model.eval()

    budget_state = empty_usage()
    records: list[dict[str, Any]] = []
    for index, record in enumerate(tqdm(base_records, desc=f"repair-{args.round_name}")):
        repaired = repair_record(record, model, tokenizer, args, seed=args.seed + index * 1009, budget_state=budget_state)
        repaired["round_name"] = args.round_name
        repaired["model_adapter"] = str(args.adapter_dir) if args.adapter_dir else "base"
        records.append(repaired)
        if args.target_forward_tokens and budget_state["forward_tokens"] >= args.target_forward_tokens:
            for untouched in base_records[index + 1 :]:
                row = deepcopy(untouched)
                row["round_name"] = args.round_name
                row["model_adapter"] = str(args.adapter_dir) if args.adapter_dir else "base"
                row["repair_usage"] = empty_usage()
                records.append(row)
            break

    write_jsonl(args.out, records)
    manifest = {
        "experiment": EXPERIMENT,
        "round_name": args.round_name,
        "records_in": str(args.records),
        "adapter_dir": str(args.adapter_dir) if args.adapter_dir else None,
        "max_rounds": args.max_rounds,
        "max_sources_per_record": args.max_sources_per_record,
        "repairs_per_source": args.repairs_per_source,
        "repair_visible_passers": args.repair_visible_passers,
        "target_forward_tokens": args.target_forward_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "token_usage": budget_state,
        "records": summarize_records(records, base_records=base_records),
        "path": str(args.out),
    }
    write_manifest(args.out.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
