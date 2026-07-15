#!/usr/bin/env python3
"""Independently encode and validate exact-exposure de-stacked streams."""

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
ROWS_PER_ARM = 1520
CORE_ROWS = 1280
MANIFEST = EXP / "data" / "stream_manifest.json"
# TODO-PIN: the orchestrator fills the manifest and stream pins after
# materialize_streams.py publishes them; loading fails closed while None.
MANIFEST_SHA256 = "16b5c1a86aabcae67e3115e24d72072f42b8b2d4169835f6979901fe1e65baac"
STREAMS = {
    "replay_clean": (
        EXP / "data" / "replay_clean.jsonl",
        "2189d160c9c3007ccbaf935478b3354b597c71945f6dbf1c32bf4e670ace0017",
    ),
    "hygiene_explore": (
        EXP / "data" / "hygiene_explore.jsonl",
        "82aa1a78c0a429a48c3db6b94ac84397cea001b041477e7b137b38c21354112f",
    ),
}
OUT = EXP / "data" / "stream_token_receipt.json"
REPLAY_KINDS = {"atom", "atom_fc", "episode", "oracle_trace"}
CANDIDATE_TREATMENT_KINDS = {
    "u_explore": 40,
    "u_hygiene": 40,
}
sys.path.insert(0, str(EXP / "scripts"))

from measure_source_tokens import training_span_boundaries  # noqa: E402
from train_think import encode_row  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path: Path, expected_sha256: str | None) -> list[dict]:
    if expected_sha256 is None:
        raise ValueError(f"materialized stream pin is unfilled (TODO-PIN): {path}")
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise ValueError(f"materialized stream changed: {path}")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(rows) != ROWS_PER_ARM:
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


def check_arm_kinds(files: dict[str, dict]) -> None:
    for name, receipt in files.items():
        kinds = receipt["kinds"]
        treatment_kinds = {
            kind: count for kind, count in kinds.items() if kind not in REPLAY_KINDS
        }
        if any(not kind.startswith("u_") for kind in treatment_kinds):
            raise ValueError(f"unexpected non-replay non-treatment kind in {name}")
        if name == "replay_clean":
            if treatment_kinds:
                raise ValueError("replay_clean stream contains treatment kinds")
        elif name == "hygiene_explore":
            if treatment_kinds != CANDIDATE_TREATMENT_KINDS:
                raise ValueError("hygiene_explore arm kind quotas changed")
        replay_rows = sum(count for kind, count in kinds.items() if kind in REPLAY_KINDS)
        if replay_rows + sum(treatment_kinds.values()) != ROWS_PER_ARM:
            raise ValueError(f"arm row accounting changed for {name}")


def build_payload() -> dict:
    if MANIFEST_SHA256 is None:
        raise ValueError("stream manifest pin is unfilled (TODO-PIN)")
    if not MANIFEST.is_file() or sha256_file(MANIFEST) != MANIFEST_SHA256:
        raise ValueError("stream manifest changed")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if (
        manifest.get("experiment_id") != EXP.name
        or manifest.get("outcome") != "PASS_EXPOSURE_MATCH"
        or manifest.get("match_axes") != ["forward", "nonzero_target", "absolute_loss_mass_x5"]
        or manifest.get("selection", {}).get("rows_per_arm") != ROWS_PER_ARM
    ):
        raise ValueError("stream manifest identity changed")
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
    if any(
        "assistant_prefix_token_ids" in row for value in rows.values() for row in value
    ) or any(receipt["spans_per_epoch"]["parent_prefix"] != 0 for receipt in files.values()):
        raise ValueError("a stream gained an on-policy parent prefix")
    control = files["replay_clean"]["spans_per_epoch"]
    deltas = {
        "hygiene_explore_minus_replay_clean": {
            key: files["hygiene_explore"]["spans_per_epoch"][key] - control[key]
            for key in sorted(control)
        }
    }
    manifest_deltas = manifest["exposure"]["deltas"]
    for name, pair_deltas in deltas.items():
        if pair_deltas != manifest_deltas[name]:
            raise ValueError(f"independent stream spans disagree with manifest: {name}")
        for axis in manifest["match_axes"]:
            if pair_deltas[axis] != 0:
                raise ValueError(f"exact exposure axis is not matched: {name}:{axis}={pair_deltas[axis]}")
    for name, receipt in files.items():
        if receipt["spans_per_epoch"] != manifest["exposure"]["arms"][name]:
            raise ValueError(f"independent arm spans disagree with manifest: {name}")
    stream_lines = {
        name: STREAMS[name][0].read_text(encoding="utf-8").splitlines()
        for name in STREAMS
    }
    aligned = [
        index
        for index, lines in enumerate(
            zip(
                stream_lines["replay_clean"],
                stream_lines["hygiene_explore"],
                strict=True,
            )
        )
        if len(set(lines)) == 1
    ]
    if len(aligned) != CORE_ROWS:
        raise ValueError(f"expected exactly {CORE_ROWS} aligned shared rows, got {len(aligned)}")
    check_arm_kinds(files)
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
        "deltas": deltas,
        "match_axes": manifest["match_axes"],
        "forward_tokens_per_arm": control["forward"],
        "nonzero_target_tokens_per_arm": control["nonzero_target"],
        "absolute_loss_mass_x5_per_arm": control["absolute_loss_mass_x5"],
        "shared_position_aligned_rows": len(aligned),
        "rows_per_arm": ROWS_PER_ARM,
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
        "deltas": {
            name: {axis: pair[axis] for axis in payload["match_axes"]}
            for name, pair in payload["deltas"].items()
        },
        "shared_position_aligned_rows": payload["shared_position_aligned_rows"],
        "skipped_rows": payload["skipped_rows"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
