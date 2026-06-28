#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.jsonl import load_jsonl, write_jsonl  # noqa: E402
from src.model_utils import DEFAULT_MODEL_PATH, code_chat_prompt, load_tokenizer  # noqa: E402
from src.repair_utils import EXPERIMENT, repair_prompt, write_manifest  # noqa: E402


def by_id(candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {candidate["candidate_id"]: candidate for candidate in candidates}


def repair_candidates(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [candidate for candidate in record.get("candidates", []) if str(candidate.get("source", "")).startswith("repair")]


def build_prompt(tokenizer: Any, record: dict[str, Any], parent: dict[str, Any]) -> str:
    return code_chat_prompt(tokenizer, repair_prompt(record, parent, prior_summaries=[]))


def build_sft(records: list[dict[str, Any]], tokenizer: Any, max_per_task: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_targets: set[tuple[str, str]] = set()
    for record in records:
        candidates = by_id(record.get("candidates", []))
        task_rows: list[dict[str, Any]] = []
        for candidate in repair_candidates(record):
            if not candidate.get("full_pass"):
                continue
            parent = candidates.get(candidate.get("parent_id", ""))
            if parent is None or parent.get("full_pass"):
                continue
            key = (record["record_id"], candidate.get("code", ""))
            if key in seen_targets:
                continue
            seen_targets.add(key)
            task_rows.append(
                {
                    "record_id": record["record_id"],
                    "task_id": record["task_id"],
                    "dataset": record["dataset"],
                    "parent_id": parent["candidate_id"],
                    "candidate_id": candidate["candidate_id"],
                    "zero_to_one_parent": not any(
                        item.get("full_pass") and not str(item.get("source", "")).startswith("repair")
                        for item in record.get("candidates", [])
                    ),
                    "prompt": build_prompt(tokenizer, record, parent),
                    "target": candidate["code"],
                    "source_public_signature": parent.get("public_signature", ""),
                    "target_public_signature": candidate.get("public_signature", ""),
                }
            )
        rows.extend(task_rows[:max_per_task])
    return rows


def build_dpo(records: list[dict[str, Any]], tokenizer: Any, max_per_task: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        candidates = by_id(record.get("candidates", []))
        by_parent: dict[str, list[dict[str, Any]]] = {}
        for candidate in repair_candidates(record):
            parent_id = candidate.get("parent_id")
            if parent_id:
                by_parent.setdefault(parent_id, []).append(candidate)
        task_rows: list[dict[str, Any]] = []
        for parent_id, attempts in by_parent.items():
            parent = candidates.get(parent_id)
            if parent is None or parent.get("full_pass"):
                continue
            chosen = [candidate for candidate in attempts if candidate.get("full_pass")]
            rejected = [candidate for candidate in attempts if not candidate.get("full_pass") and candidate.get("code")]
            if not chosen:
                continue
            if rejected:
                reject = rejected[0]
                rejected_text = reject["code"]
                rejected_id = reject["candidate_id"]
            else:
                rejected_text = parent.get("code", "")
                rejected_id = parent["candidate_id"]
            if not rejected_text:
                continue
            task_rows.append(
                {
                    "record_id": record["record_id"],
                    "task_id": record["task_id"],
                    "dataset": record["dataset"],
                    "parent_id": parent_id,
                    "chosen_id": chosen[0]["candidate_id"],
                    "rejected_id": rejected_id,
                    "zero_to_one_parent": not any(
                        item.get("full_pass") and not str(item.get("source", "")).startswith("repair")
                        for item in record.get("candidates", [])
                    ),
                    "prompt": build_prompt(tokenizer, record, parent),
                    "chosen": chosen[0]["code"],
                    "rejected": rejected_text,
                }
            )
        rows.extend(task_rows[:max_per_task])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--sft-out", type=Path, required=True)
    parser.add_argument("--dpo-out", type=Path, required=True)
    parser.add_argument("--max-per-task", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    random.seed(args.seed)
    records = load_jsonl(args.records)
    tokenizer = load_tokenizer(args.model_path)
    sft_rows = build_sft(records, tokenizer, args.max_per_task)
    dpo_rows = build_dpo(records, tokenizer, args.max_per_task)
    random.shuffle(sft_rows)
    random.shuffle(dpo_rows)
    write_jsonl(args.sft_out, sft_rows)
    write_jsonl(args.dpo_out, dpo_rows)
    manifest = {
        "experiment": EXPERIMENT,
        "records": str(args.records),
        "sft_examples": len(sft_rows),
        "dpo_examples": len(dpo_rows),
        "sft_zero_to_one_examples": sum(1 for row in sft_rows if row.get("zero_to_one_parent")),
        "dpo_zero_to_one_examples": sum(1 for row in dpo_rows if row.get("zero_to_one_parent")),
        "max_per_task": args.max_per_task,
    }
    write_manifest(args.sft_out.with_suffix(".manifest.json"), {**manifest, "path": str(args.sft_out), "kind": "sft"})
    write_manifest(args.dpo_out.with_suffix(".manifest.json"), {**manifest, "path": str(args.dpo_out), "kind": "dpo"})
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
