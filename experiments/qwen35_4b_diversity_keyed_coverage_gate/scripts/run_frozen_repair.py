#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from copy import deepcopy
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.diversity_utils import (  # noqa: E402
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
from src.jsonl import load_jsonl, write_jsonl  # noqa: E402
from src.model_utils import DEFAULT_MODEL_PATH, code_chat_prompt, load_generation_model, load_tokenizer  # noqa: E402


def source_summary(candidate: dict) -> str:
    status = "visible_pass" if candidate.get("visible_all_pass") else "visible_fail"
    return f"{candidate.get('candidate_id')}: {status}, public_signature={candidate.get('public_signature', '')}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--round-name", type=str, required=True)
    parser.add_argument("--max-sources-per-record", type=int, default=2)
    parser.add_argument("--repairs-per-source", type=int, default=1)
    parser.add_argument("--repair-visible-passers", action="store_true")
    parser.add_argument("--prior-summary-limit", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=260)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    random.seed(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    base_records = load_jsonl(args.records)
    records = [deepcopy(record) for record in base_records]
    tokenizer = load_tokenizer(args.model_path, padding_side="left")
    model = load_generation_model(args.model_path)
    usage = empty_usage()

    for index, record in enumerate(tqdm(records, desc=f"repair-{args.round_name}")):
        if record.get("coverage"):
            record["round_name"] = args.round_name
            continue
        sources = choose_repair_sources(record, args.max_sources_per_record, visible_fail_only=not args.repair_visible_passers)
        if not sources:
            record["round_name"] = args.round_name
            continue
        order = len(record.get("candidates", []))
        prior = [source_summary(candidate) for candidate in record.get("candidates", [])[-args.prior_summary_limit :]]
        for source_index, source in enumerate(sources):
            for attempt in range(args.repairs_per_source):
                prompt_text = repair_prompt(record, source, prior_summaries=prior)
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
                    seed=args.seed + index * 100003 + source_index * 1000 + attempt,
                )
                usage = add_usage(usage, batch_usage)
                completion = completions[0]
                candidate = candidate_from_completion(
                    completion,
                    record,
                    source=f"repair_of_{source['candidate_id']}_{attempt}",
                    order=order,
                    prompt_tokens=estimate_text_tokens(tokenizer, prompt),
                    completion_tokens=estimate_text_tokens(tokenizer, completion),
                    parent_id=source["candidate_id"],
                    repair_round=1,
                )
                record.setdefault("candidates", []).append(candidate)
                prior.append(source_summary(candidate))
                order += 1
        record["round_name"] = args.round_name
        record["repair_usage"] = usage
        record["candidates"] = dedupe_candidates(record.get("candidates", []))
        recompute_record_metrics(record)

    for record in records:
        record["round_name"] = args.round_name
        recompute_record_metrics(record)

    write_jsonl(args.out, records)
    manifest = {
        "experiment": EXPERIMENT,
        "round_name": args.round_name,
        "records_in": str(args.records),
        "max_sources_per_record": args.max_sources_per_record,
        "repairs_per_source": args.repairs_per_source,
        "repair_visible_passers": args.repair_visible_passers,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "token_usage": usage,
        "records": summarize_records(records, base_records=base_records),
        "path": str(args.out),
    }
    write_manifest(args.out.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
