#!/usr/bin/env python3
"""Verify derived streams and freeze exact Qwen training-token exposure."""

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
MAX_LENGTH = 4096
EXPECTED_FORWARD_TOKENS = 304313
EXPECTED_ROWS = 320
sys.path.insert(0, str(EXP / "scripts"))

from materialize_streams import build_outputs  # noqa: E402
from measure_source_tokens import training_span_boundaries  # noqa: E402
from train_think import encode_row  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def load_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"empty or malformed JSONL: {path}")
    return rows


def token_receipt(path: Path, rows: list[dict], tokenizer) -> dict:
    spans = Counter()
    lengths: list[int] = []
    nonzero_weight_tokens = 0
    absolute_weight_mass = 0.0
    skipped = []
    for index, row in enumerate(rows):
        encoded = encode_row(row, tokenizer, MAX_LENGTH, 0.2, 0.2)
        if encoded is None:
            skipped.append({
                "index": index,
                "task_id": row.get("task_id"),
                "kind": row.get("kind"),
                "reason": "encode_row_rejected",
            })
            continue
        forward = len(encoded["input_ids"])
        masked, think_end, close_end = training_span_boundaries(
            row, tokenizer, encoded
        )
        parent_prefix = len(row.get("assistant_prefix_token_ids", []))
        if (
            forward > MAX_LENGTH
            or masked < parent_prefix
            or not all(label == -100 for label in encoded["labels"][:masked])
            or len(encoded["loss_weights"]) != forward
        ):
            raise ValueError(f"invalid encoded row contract: {row.get('task_id')}")
        lengths.append(forward)
        spans.update({
            "forward": forward,
            "prompt": masked - parent_prefix,
            "parent_prefix": parent_prefix,
            "masked_context": masked,
            "think_target": think_end - masked,
            "close_target": close_end - think_end,
            "answer_target": forward - close_end,
            "target_span": forward - masked,
        })
        nonzero_weight_tokens += sum(weight != 0.0 for weight in encoded["loss_weights"])
        absolute_weight_mass += sum(abs(weight) for weight in encoded["loss_weights"])
    return {
        "path": portable_path(path),
        "sha256": sha256_file(path),
        "rows": len(rows),
        "kinds": dict(sorted(Counter(row.get("kind", "missing") for row in rows).items())),
        "families": dict(
            sorted(Counter(row.get("family", "missing") for row in rows).items())
        ),
        "encoded_rows": len(lengths),
        "skipped_rows": len(skipped),
        "skipped": skipped,
        "min_sequence_tokens": min(lengths) if lengths else None,
        "max_sequence_tokens": max(lengths) if lengths else None,
        "spans_per_epoch": dict(sorted(spans.items())),
        "nonzero_weight_tokens_per_epoch": nonzero_weight_tokens,
        "absolute_weight_mass_per_epoch": round(absolute_weight_mass, 6),
    }


def build_payload() -> dict:
    expected_outputs, expected_manifest = build_outputs()
    manifest_path = EXP / "data" / "stream_manifest.json"
    expected_manifest_bytes = (
        json.dumps(expected_manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()
    if not manifest_path.is_file() or manifest_path.read_bytes() != expected_manifest_bytes:
        raise ValueError("stream manifest is absent or changed")
    paths = [
        EXP / "data" / "replay_after_close.jsonl",
        EXP / "data" / "prefix_repair_after_close.jsonl",
    ]
    for path in paths:
        if not path.is_file() or path.read_bytes() != expected_outputs[path.name]:
            raise ValueError(f"derived stream is absent or changed: {path}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    files = [token_receipt(path, load_jsonl(path), tokenizer) for path in paths]
    failures = [row for receipt in files for row in receipt["skipped"]]
    if failures:
        raise ValueError(
            f"derived streams have {len(failures)} untrainable rows: "
            + json.dumps(failures[:10], sort_keys=True)
        )
    if any(row["rows"] != EXPECTED_ROWS for row in files):
        raise ValueError("derived stream row count changed")
    forward_tokens = {
        Path(row["path"]).name: row["spans_per_epoch"]["forward"] for row in files
    }
    if set(forward_tokens.values()) != {EXPECTED_FORWARD_TOKENS}:
        raise ValueError(f"derived arms are not exactly forward-token matched: {forward_tokens}")

    control_rows = paths[0].read_text(encoding="utf-8").splitlines()
    candidate_rows = paths[1].read_text(encoding="utf-8").splitlines()
    aligned_positions = [
        index for index, pair in enumerate(zip(control_rows, candidate_rows, strict=True))
        if pair[0] == pair[1]
    ]
    if len(aligned_positions) != 200:
        raise ValueError(f"expected 200 byte-identical aligned replay slots, got {len(aligned_positions)}")
    candidate = files[1]
    expected_kinds = {
        "u_prefix_repair_bounded_induction",
        "u_prefix_repair_commit_serialization",
        "u_prefix_repair_declaration_operation",
        "u_prefix_repair_probe_scoring",
        "u_prefix_repair_repair_propagation",
        "u_prefix_repair_state_transition",
    }
    if any(candidate["kinds"].get(kind) != 10 for kind in expected_kinds):
        raise ValueError("candidate stream lost its six balanced prefix-repair classes")
    if candidate["spans_per_epoch"]["parent_prefix"] != 47123:
        raise ValueError("candidate stream lost exact parent-prefix tokens")

    control_spans = files[0]["spans_per_epoch"]
    candidate_spans = files[1]["spans_per_epoch"]
    span_delta = {
        key: candidate_spans[key] - control_spans[key] for key in sorted(control_spans)
    }
    return {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "max_length": MAX_LENGTH,
        "encoder": "scripts/train_think.py:encode_row",
        "encoder_sha256": sha256_file(EXP / "scripts" / "train_think.py"),
        "tokenizer": {
            "class": tokenizer.__class__.__name__,
            "vocab_size": len(tokenizer),
            "eos_token_id": tokenizer.eos_token_id,
            "think_close_token_id": tokenizer.convert_tokens_to_ids("</think>"),
        },
        "stream_manifest": portable_path(manifest_path),
        "stream_manifest_sha256": sha256_file(manifest_path),
        "source_token_receipt": portable_path(EXP / "data" / "source_token_lengths.json"),
        "source_token_receipt_sha256": sha256_file(EXP / "data" / "source_token_lengths.json"),
        "files": files,
        "arm_data": {
            "replay_after_close": portable_path(paths[0]),
            "prefix_repair_after_close": portable_path(paths[1]),
        },
        "position_aligned_streams": True,
        "shared_position_aligned_rows": len(aligned_positions),
        "rows_per_arm": EXPECTED_ROWS,
        "forward_tokens_per_arm": EXPECTED_FORWARD_TOKENS,
        "forward_token_delta": 0,
        "skipped_rows": 0,
        "candidate_minus_control_spans": span_delta,
        "training": expected_manifest["training"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--receipt", type=Path, default=EXP / "data" / "stream_token_receipt.json"
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    payload = build_payload()
    value = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()
    if args.check:
        if not args.receipt.is_file() or args.receipt.read_bytes() != value:
            parser.error("stream token receipt is absent or changed")
    else:
        if args.receipt.exists():
            parser.error("refusing to overwrite token receipt")
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_bytes(value)
    print(json.dumps({
        "receipt": str(args.receipt),
        "sha256": hashlib.sha256(value).hexdigest(),
        "rows_per_arm": payload["rows_per_arm"],
        "forward_tokens_per_arm": payload["forward_tokens_per_arm"],
        "shared_position_aligned_rows": payload["shared_position_aligned_rows"],
        "skipped_rows": payload["skipped_rows"],
        "candidate_minus_control_spans": payload["candidate_minus_control_spans"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
