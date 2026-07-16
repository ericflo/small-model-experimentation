#!/usr/bin/env python3
"""Measure exact Qwen training spans for the gym-mix treatment corpus and replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer


EXP = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
MAX_LENGTH = 4096
SOURCES = {
    "gym_mix": (
        EXP / "data" / "sft_gym_mix.jsonl",
        "6295011622096992e889b58a1a004fee26f4f9787bd952d348c0bf8593564a89",
        160,
    ),
    "replay": (
        EXP / "data" / "sft_blend.jsonl",
        "25a9595f2e70e4d5cab0a730f0e2613d314843f2a5dfe96187bc30d5d2abf0c2",
        2240,
    ),
}
OUT = EXP / "data" / "source_token_lengths.json"
sys.path.insert(0, str(EXP / "scripts"))

from train_think import encode_row  # noqa: E402


FIELDS = (
    "forward",
    "prompt",
    "parent_prefix",
    "masked_context",
    "think_target",
    "close_target",
    "answer_target",
    "target_span",
    "nonzero_target",
    "absolute_loss_mass_x5",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path: Path, expected_sha256: str, expected_rows: int) -> list[dict]:
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise ValueError(f"frozen source changed: {path}")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(rows) != expected_rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"unexpected source rows: {path}")
    return rows


def training_span_boundaries(row: dict, tokenizer, encoded: dict) -> tuple[int, int, int]:
    prompt = tokenizer.apply_chat_template(
        row["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    think_part = row["think"].strip() + "\n"
    close_part = "</think>\n\n"
    answer_part = row["answer"].strip() + tokenizer.eos_token
    prefix_ids = row.get("assistant_prefix_token_ids")
    if prefix_ids is not None:
        think_ids = tokenizer(think_part, add_special_tokens=False)["input_ids"]
        close_ids = tokenizer(close_part, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(answer_part, add_special_tokens=False)["input_ids"]
        expected = prompt_ids + prefix_ids + think_ids + close_ids + answer_ids
        masked = len(prompt_ids) + len(prefix_ids)
        think_end = masked + len(think_ids)
        close_end = think_end + len(close_ids)
    else:
        think_ids = tokenizer(prompt + think_part, add_special_tokens=False)["input_ids"]
        close_ids = tokenizer(
            prompt + think_part + close_part, add_special_tokens=False
        )["input_ids"]
        expected = tokenizer(
            prompt + think_part + close_part + answer_part,
            add_special_tokens=False,
        )["input_ids"]
        masked = len(prompt_ids)
        think_end = len(think_ids)
        close_end = len(close_ids)
    if encoded["input_ids"] != expected:
        raise ValueError(f"span reconstruction diverged: {row.get('task_id')}")
    return masked, think_end, close_end


def measure(row: dict, tokenizer) -> dict[str, int]:
    encoded = encode_row(row, tokenizer, MAX_LENGTH, 0.2, 0.2)
    if encoded is None:
        raise ValueError(f"untrainable row: {row.get('task_id')}")
    forward = len(encoded["input_ids"])
    masked, think_end, close_end = training_span_boundaries(row, tokenizer, encoded)
    parent_prefix = len(row.get("assistant_prefix_token_ids", []))
    weights = encoded["loss_weights"]
    scaled_weights = [abs(float(weight)) * 5 for weight in weights]
    if any(abs(value - round(value)) > 1e-6 for value in scaled_weights):
        raise ValueError(f"loss weight is not representable in fifths: {row.get('task_id')}")
    value = {
        "forward": forward,
        "prompt": masked - parent_prefix,
        "parent_prefix": parent_prefix,
        "masked_context": masked,
        "think_target": think_end - masked,
        "close_target": close_end - think_end,
        "answer_target": forward - close_end,
        "target_span": forward - masked,
        "nonzero_target": sum(weight != 0.0 for weight in weights),
        "absolute_loss_mass_x5": sum(round(item) for item in scaled_weights),
    }
    if (
        min(value.values()) < 0
        or value["prompt"] + value["parent_prefix"] + value["target_span"] != forward
        or len(weights) != forward
        or any(weight != 0.0 for weight in weights[:masked])
    ):
        raise ValueError(f"invalid token-span accounting: {row.get('task_id')}")
    return value


def build_payload() -> dict:
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
            "totals": {field: sum(item[field] for item in lengths) for field in FIELDS},
            "min_forward": min(item["forward"] for item in lengths),
            "max_forward": max(item["forward"] for item in lengths),
        }
    if any(source["totals"]["parent_prefix"] != 0 for source in sources.values()):
        raise ValueError("a frozen source gained an on-policy parent prefix")
    if sources["gym_mix"]["rows"] != 160:
        raise ValueError("gym-mix treatment source lost rows")
    return {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "max_length": MAX_LENGTH,
        "encoder": "scripts/train_think.py:encode_row",
        "encoder_sha256": sha256_file(EXP / "scripts" / "train_think.py"),
        "thought_weight": 0.2,
        "close_weight": 0.2,
        "absolute_loss_mass_scale": 5,
        "match_axes": ["forward", "nonzero_target", "absolute_loss_mass_x5"],
        "sources": sources,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    payload = build_payload()
    value = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    if args.check:
        if not OUT.is_file() or OUT.read_bytes() != value:
            parser.error("source token receipt is absent or changed")
    else:
        if OUT.exists():
            parser.error("refusing to overwrite source token receipt")
        OUT.write_bytes(value)
    print(json.dumps({
        "out": str(OUT),
        "sha256": hashlib.sha256(value).hexdigest(),
        "totals": {name: source["totals"] for name, source in payload["sources"].items()},
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
