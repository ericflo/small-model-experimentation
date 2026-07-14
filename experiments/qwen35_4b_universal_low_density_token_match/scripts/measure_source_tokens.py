#!/usr/bin/env python3
"""Freeze per-row Qwen token lengths for the two copied source corpora."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer


EXP = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
MAX_LENGTH = 4096
SOURCES = {
    "designed": (
        EXP / "data" / "sft_universal_fast.jsonl",
        "4a0833756b5497fcbccf278476ece8c98bcbfce80e900ecc2489150e496b27c4",
        800,
    ),
    "replay": (
        EXP / "data" / "sft_blend.jsonl",
        "25a9595f2e70e4d5cab0a730f0e2613d314843f2a5dfe96187bc30d5d2abf0c2",
        2240,
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path, expected_sha256: str, expected_rows: int) -> list[dict]:
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"frozen source changed: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != expected_rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"unexpected source rows: {path}")
    return rows


def measure(row: dict, tokenizer) -> dict[str, int]:
    prompt = tokenizer.apply_chat_template(
        row["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    think_part = row["think"].strip() + "\n</think>\n\n"
    answer_part = row["answer"].strip() + tokenizer.eos_token
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    middle_ids = tokenizer(prompt + think_part, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt + think_part + answer_part, add_special_tokens=False)["input_ids"]
    if full_ids[: len(prompt_ids)] != prompt_ids or full_ids[: len(middle_ids)] != middle_ids:
        raise ValueError(f"tokenizer boundary merge for {row.get('task_id')}")
    if len(full_ids) > MAX_LENGTH:
        raise ValueError(f"over-length row for {row.get('task_id')}: {len(full_ids)}")
    return {
        "forward": len(full_ids),
        "prompt": len(prompt_ids),
        "think_target": len(middle_ids) - len(prompt_ids),
        "answer_target": len(full_ids) - len(middle_ids),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=EXP / "data" / "source_token_lengths.json"
    )
    args = parser.parse_args()
    if args.out.exists():
        parser.error("refusing to overwrite source token receipt")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    sources = {}
    for name, (path, expected_sha256, expected_rows) in SOURCES.items():
        rows = load_rows(path, expected_sha256, expected_rows)
        lengths = [measure(row, tokenizer) for row in rows]
        sources[name] = {
            "path": path.relative_to(EXP).as_posix(),
            "sha256": expected_sha256,
            "rows": len(rows),
            "lengths": lengths,
            "totals": {
                field: sum(item[field] for item in lengths)
                for field in ("forward", "prompt", "think_target", "answer_target")
            },
        }
    payload = {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "max_length": MAX_LENGTH,
        "sources": sources,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "out": str(args.out),
        "sha256": sha256_file(args.out),
        "sources": {name: value["totals"] for name, value in sources.items()},
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
