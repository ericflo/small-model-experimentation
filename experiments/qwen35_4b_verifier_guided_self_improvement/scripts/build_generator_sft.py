#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.code_env import mbpp_sampling_prompt  # noqa: E402
from src.jsonl import load_jsonl, write_jsonl  # noqa: E402
from src.model_utils import DEFAULT_MODEL_PATH, code_chat_prompt, load_tokenizer  # noqa: E402


def train_prompt(record: dict[str, Any], tokenizer: Any) -> str:
    public_tests = [case["assert_src"] for case in record.get("public_cases", [])]
    return code_chat_prompt(tokenizer, mbpp_sampling_prompt(record, record["entry_point"], public_tests))


def candidate_examples(record: dict[str, Any], mode: str, tokenizer: Any, max_per_task: int) -> list[dict[str, Any]]:
    prompt = train_prompt(record, tokenizer)
    rows: list[dict[str, Any]] = []
    if mode == "oracle":
        code = (record.get("reference_code") or "").strip()
        if code:
            rows.append(
                {
                    "prompt": prompt,
                    "target": code.rstrip() + "\n",
                    "record_id": record["record_id"],
                    "task_id": record["task_id"],
                    "source": "reference_solution",
                    "visible_all_pass": True,
                    "hidden_all_pass": True,
                }
            )
        return rows

    for candidate in sorted(record.get("candidates", []), key=lambda item: item.get("order", 0)):
        if candidate.get("parse_status") != "parsed" or not candidate.get("safe") or not candidate.get("code", "").strip():
            continue
        if mode == "verified" and not candidate.get("visible_all_pass"):
            continue
        rows.append(
            {
                "prompt": prompt,
                "target": candidate["code"].rstrip() + "\n",
                "record_id": record["record_id"],
                "task_id": record["task_id"],
                "candidate_id": candidate["candidate_id"],
                "source": candidate.get("source", ""),
                "visible_all_pass": bool(candidate.get("visible_all_pass")),
                "hidden_all_pass": bool(candidate.get("full_pass")),
                "public_signature": candidate.get("public_signature", ""),
            }
        )
        if len(rows) >= max_per_task:
            break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, nargs="+", required=True)
    parser.add_argument("--mode", choices=["verified", "unverified", "oracle"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path)
    parser.add_argument("--max-per-task", type=int, default=2)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model_path)
    examples: list[dict[str, Any]] = []
    source_records = 0
    for path in args.records:
        records = load_jsonl(path)
        source_records += len(records)
        for record in records:
            if record.get("dataset") != "mbpp":
                continue
            examples.extend(candidate_examples(record, args.mode, tokenizer, args.max_per_task))

    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for row in examples:
        key = (row["prompt"], row["target"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, deduped)
    manifest = {
        "mode": args.mode,
        "records": [str(path) for path in args.records],
        "source_records": source_records,
        "examples": len(deduped),
        "visible_positive_examples": sum(1 for row in deduped if row.get("visible_all_pass")),
        "hidden_positive_examples": sum(1 for row in deduped if row.get("hidden_all_pass")),
        "max_per_task": args.max_per_task,
        "out": str(args.out),
    }
    manifest_path = args.manifest_out or args.out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
