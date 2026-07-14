#!/usr/bin/env python3
"""Verify derived-dose bytes and record exact Qwen training-token exposure."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
sys.path.insert(0, str(EXP / "scripts"))

from materialize_doses import build_outputs  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: row is not an object")
        rows.append(row)
    if not rows:
        raise ValueError(f"empty JSONL: {path}")
    return rows


def token_receipt(path: Path, rows: list[dict], tokenizer, max_length: int) -> dict:
    lengths: list[int] = []
    prompt_tokens = 0
    think_tokens = 0
    answer_tokens = 0
    skipped: list[dict] = []
    for index, row in enumerate(rows):
        prompt = tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )
        think_part = row["think"].strip() + "\n</think>\n\n"
        answer_part = row["answer"].strip() + tokenizer.eos_token
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        middle_ids = tokenizer(prompt + think_part, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(prompt + think_part + answer_part, add_special_tokens=False)["input_ids"]
        reason = None
        if full_ids[: len(prompt_ids)] != prompt_ids or full_ids[: len(middle_ids)] != middle_ids:
            reason = "tokenizer_boundary_merge"
        elif len(full_ids) > max_length:
            reason = "over_length"
        if reason:
            skipped.append({
                "index": index,
                "task_id": row.get("task_id"),
                "kind": row.get("kind"),
                "length": len(full_ids),
                "reason": reason,
            })
            continue
        lengths.append(len(full_ids))
        prompt_tokens += len(prompt_ids)
        think_tokens += len(middle_ids) - len(prompt_ids)
        answer_tokens += len(full_ids) - len(middle_ids)
    return {
        "path": portable_path(path),
        "sha256": sha256_file(path),
        "rows": len(rows),
        "kinds": dict(sorted(Counter(row.get("kind", "missing") for row in rows).items())),
        "families": dict(sorted(Counter(row.get("family", "missing") for row in rows).items())),
        "encoded_rows": len(lengths),
        "skipped_rows": len(skipped),
        "skipped": skipped,
        "min_sequence_tokens": min(lengths) if lengths else None,
        "max_sequence_tokens": max(lengths) if lengths else None,
        "total_forward_tokens_per_epoch": sum(lengths),
        "prompt_tokens_per_epoch": prompt_tokens,
        "think_target_tokens_per_epoch": think_tokens,
        "answer_target_tokens_per_epoch": answer_tokens,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument(
        "--receipt", type=Path, default=EXP / "data" / "dose_token_receipt.json"
    )
    args = parser.parse_args()
    if args.receipt.exists():
        parser.error("refusing to overwrite token receipt")

    expected_outputs, expected_manifest = build_outputs()
    manifest_path = EXP / "data" / "dose_manifest.json"
    expected_manifest_bytes = (
        json.dumps(expected_manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    if manifest_path.read_bytes() != expected_manifest_bytes:
        raise SystemExit("dose manifest is absent or changed")
    paths = [EXP / "data" / "warm_union.jsonl", EXP / "data" / "replay_refresh.jsonl"]
    for path in paths:
        if path.read_bytes() != expected_outputs[path.name]:
            raise SystemExit(f"derived dose is absent or changed: {path}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    files = [token_receipt(path, load_jsonl(path), tokenizer, args.max_length) for path in paths]
    failures = [row for receipt in files for row in receipt["skipped"]]
    if failures:
        raise SystemExit(
            f"derived doses have {len(failures)} untrainable rows:\n"
            + json.dumps(failures[:10], indent=2, ensure_ascii=False)
        )
    payload = {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "max_length": args.max_length,
        "dose_manifest": portable_path(manifest_path),
        "dose_manifest_sha256": sha256_file(manifest_path),
        "files": files,
        "total_rows": sum(row["rows"] for row in files),
        "skipped_rows": 0,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
