#!/usr/bin/env python3
"""Independently encode and validate exact-exposure training streams."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer


EXP = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
MAX_LENGTH = 4096
MANIFEST = EXP / "data" / "stream_manifest.json"
MANIFEST_SHA256 = "7ba55045e72371e3675ba67bcf0bd72f6a0bf645c3ad7d0e92f7282e59d91de1"
STREAMS = {
    "replay_control": (
        EXP / "data" / "replay_control.jsonl",
        "7a8d45666000cbb6bffabf6faab8f9d61006bf3a80275a631238a23cd03b5078",
    ),
    "counterfactual_restart_candidate": (
        EXP / "data" / "counterfactual_restart_candidate.jsonl",
        "28deb20e6bfca81f760549b071d0d0df39bfa561c4d09fde0580d81699413190",
    ),
}
OUT = EXP / "data" / "stream_token_receipt.json"
sys.path.insert(0, str(EXP / "scripts"))

from measure_source_tokens import training_span_boundaries  # noqa: E402
from train_think import encode_row  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path: Path, expected_sha256: str) -> list[dict]:
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise ValueError(f"materialized stream changed: {path}")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(rows) != 320:
        raise ValueError(f"stream row count changed: {path}")
    return rows


def encode_receipt(path: Path, rows: list[dict], tokenizer) -> dict:
    spans = Counter()
    lengths: list[int] = []
    skipped: list[dict] = []
    for index, row in enumerate(rows):
        encoded = encode_row(row, tokenizer, MAX_LENGTH, 0.2, 0.2)
        if encoded is None:
            skipped.append({"index": index, "task_id": row.get("task_id"), "kind": row.get("kind")})
            continue
        forward = len(encoded["input_ids"])
        masked, think_end, close_end = training_span_boundaries(row, tokenizer, encoded)
        parent_prefix = len(row.get("assistant_prefix_token_ids", []))
        weights = encoded["loss_weights"]
        scaled = [abs(float(weight)) * 5 for weight in weights]
        if (
            len(weights) != forward
            or any(weight != 0.0 for weight in weights[:masked])
            or any(abs(value - round(value)) > 1e-6 for value in scaled)
        ):
            raise ValueError(f"encoded weight contract changed: {row.get('task_id')}")
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
            "nonzero_target": sum(weight != 0.0 for weight in weights),
            "absolute_loss_mass_x5": sum(round(value) for value in scaled),
        })
    return {
        "path": path.relative_to(EXP).as_posix(),
        "sha256": sha256_file(path),
        "rows": len(rows),
        "encoded_rows": len(lengths),
        "skipped_rows": len(skipped),
        "skipped": skipped,
        "min_sequence_tokens": min(lengths) if lengths else None,
        "max_sequence_tokens": max(lengths) if lengths else None,
        "kinds": dict(sorted(Counter(row.get("kind", "missing") for row in rows).items())),
        "families": dict(sorted(Counter(row.get("family", "missing") for row in rows).items())),
        "spans_per_epoch": dict(sorted(spans.items())),
        "absolute_loss_mass": spans["absolute_loss_mass_x5"] / 5,
    }


def build_payload() -> dict:
    if not MANIFEST.is_file() or sha256_file(MANIFEST) != MANIFEST_SHA256:
        raise ValueError("stream manifest changed")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = {name: load_rows(path, expected) for name, (path, expected) in STREAMS.items()}
    files = {
        name: encode_receipt(STREAMS[name][0], value, tokenizer)
        for name, value in rows.items()
    }
    if any(receipt["skipped_rows"] for receipt in files.values()):
        raise ValueError("a materialized stream has tokenizer skips")
    control = files["replay_control"]["spans_per_epoch"]
    candidate = files["counterfactual_restart_candidate"]["spans_per_epoch"]
    deltas = {key: candidate[key] - control[key] for key in sorted(control)}
    expected_deltas = manifest["exposure"]["candidate_minus_control"]
    if deltas != expected_deltas:
        raise ValueError(f"independent stream spans disagree with manifest: {deltas}")
    for axis in manifest["match_axes"]:
        if deltas[axis] != 0:
            raise ValueError(f"exact exposure axis is not matched: {axis}={deltas[axis]}")
    control_lines = STREAMS["replay_control"][0].read_text(encoding="utf-8").splitlines()
    candidate_lines = STREAMS["counterfactual_restart_candidate"][0].read_text(encoding="utf-8").splitlines()
    aligned = [
        index
        for index, pair in enumerate(zip(control_lines, candidate_lines, strict=True))
        if pair[0] == pair[1]
    ]
    candidate_rows = rows["counterfactual_restart_candidate"]
    restart_kinds = {
        f"u_counterfactual_restart_{skill}"
        for skill in (
            "induct", "execute", "select", "trace", "verify", "count", "repair",
            "optimize", "abstain", "state", "order", "probe", "route",
        )
    }
    if (
        len(aligned) != 200
        or any(files["counterfactual_restart_candidate"]["kinds"].get(kind) != 4 for kind in restart_kinds)
        or any("assistant_prefix_token_ids" in row for row in candidate_rows)
        or files["counterfactual_restart_candidate"]["spans_per_epoch"]["parent_prefix"] != 0
    ):
        raise ValueError("candidate balance, alignment, or clean-restart contract changed")
    return {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "stage": "independent_exact_exposure_validation",
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
        "stream_manifest": MANIFEST.relative_to(EXP).as_posix(),
        "stream_manifest_sha256": MANIFEST_SHA256,
        "files": files,
        "candidate_minus_control_spans": deltas,
        "match_axes": manifest["match_axes"],
        "forward_tokens_per_arm": control["forward"],
        "nonzero_target_tokens_per_arm": control["nonzero_target"],
        "absolute_loss_mass_x5_per_arm": control["absolute_loss_mass_x5"],
        "shared_position_aligned_rows": len(aligned),
        "rows_per_arm": 320,
        "skipped_rows": 0,
        "targets_modified_for_matching": False,
        "rows_duplicated_for_matching": False,
        "rows_truncated_for_matching": False,
        "training_authorized": False,
        "benchmark_data_read": False,
        "aggregate_seed_open": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    payload = build_payload()
    value = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    if args.check:
        if not OUT.is_file() or OUT.read_bytes() != value:
            parser.error("stream token receipt is absent or changed")
    else:
        if OUT.exists():
            parser.error("refusing to overwrite stream token receipt")
        OUT.write_bytes(value)
    print(json.dumps({
        "out": str(OUT),
        "sha256": hashlib.sha256(value).hexdigest(),
        "forward_tokens_per_arm": payload["forward_tokens_per_arm"],
        "nonzero_target_tokens_per_arm": payload["nonzero_target_tokens_per_arm"],
        "absolute_loss_mass_x5_per_arm": payload["absolute_loss_mass_x5_per_arm"],
        "candidate_minus_control_spans": payload["candidate_minus_control_spans"],
        "skipped_rows": payload["skipped_rows"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
