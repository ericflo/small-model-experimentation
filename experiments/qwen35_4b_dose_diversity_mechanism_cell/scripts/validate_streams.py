#!/usr/bin/env python3
"""Independently encode and validate the single exact-exposure candidate stream.

Mechanism-cell shape: only ``axis160_direct`` trains, so only it is encoded.
The notional control block never materializes as a stream, but its exactness
bookkeeping is still independently re-verified here: the control-block source
indices recorded in the stream manifest are re-summed from the pinned source
token receipt and must equal the candidate variable block on every match axis.
"""

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
TREATMENT_ROWS = 160
FILLER_ROWS = 80
CONTROL_ROWS = 240
MANIFEST = EXP / "data" / "stream_manifest.json"
MANIFEST_SHA256 = "49cbe33112bc2c671b527edbd63a667985c03abd18d6320643053be2e6f5421a"
STREAMS = {
    "axis160_direct": (
        EXP / "data" / "axis160_direct.jsonl",
        "0cbf3abf04781f3217189dd7b2bdf173d83772b0200b283a7c608dc3d386c1ee",
    ),
}
TOKENS = EXP / "data" / "source_token_lengths.json"
OUT = EXP / "data" / "stream_token_receipt.json"
REPLAY_KINDS = {"atom", "atom_fc", "episode", "oracle_trace"}
CANDIDATE_TREATMENT_KINDS = {
    "u_explore": 40,
    "u_hygiene": 40,
    "u_protocol": 40,
    "u_tracefix": 40,
}
MATCH_AXES = ("forward", "nonzero_target", "absolute_loss_mass_x5")
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
        if name == "axis160_direct":
            if treatment_kinds != CANDIDATE_TREATMENT_KINDS:
                raise ValueError("axis160_direct arm kind quotas changed")
        else:
            raise ValueError(f"unexpected arm: {name}")
        replay_rows = sum(count for kind, count in kinds.items() if kind in REPLAY_KINDS)
        if replay_rows + sum(treatment_kinds.values()) != ROWS_PER_ARM:
            raise ValueError(f"arm row accounting changed for {name}")


def notional_control_bookkeeping(manifest: dict) -> dict:
    """Re-derive the never-materialized control block from primary sources.

    The manifest's recorded control indices are re-summed against the pinned
    source token receipt; the result must equal both the manifest's recorded
    control block and the candidate variable block on every match axis.
    """
    tokens_pin = (
        manifest.get("sources", {}).get("source_token_receipt", {}).get("sha256")
    )
    if not TOKENS.is_file() or sha256_file(TOKENS) != tokens_pin:
        raise ValueError("pinned source token receipt is absent or changed")
    token_payload = json.loads(TOKENS.read_text(encoding="utf-8"))
    replay_lengths = token_payload["sources"]["replay"]["lengths"]
    selection = manifest.get("selection", {})
    control = selection.get("replay_control_source_indices", [])
    filler = selection.get("candidate_replay_filler_source_indices", [])
    core = selection.get("replay_core_source_indices", [])
    if (
        len(control) != CONTROL_ROWS
        or len(filler) != FILLER_ROWS
        or len(core) != CORE_ROWS
        or len(set(control) | set(filler) | set(core))
        != CONTROL_ROWS + FILLER_ROWS + CORE_ROWS
        or selection.get("control_stream_materialized") is not False
    ):
        raise ValueError("manifest control/filler/core index accounting changed")
    control_sums = {
        axis: sum(replay_lengths[index][axis] for index in control)
        for axis in MATCH_AXES
    }
    manifest_control = manifest["exposure"]["blocks"]["notional_control"]
    manifest_candidate = manifest["exposure"]["blocks"]["axis160_direct"]
    for axis in MATCH_AXES:
        if control_sums[axis] != manifest_control[axis]:
            raise ValueError(
                f"recomputed notional control block disagrees with manifest: {axis}"
            )
        if control_sums[axis] != manifest_candidate[axis]:
            raise ValueError(
                f"exact exposure axis is not matched: notional_control:{axis}"
            )
    return {
        "control_rows": len(control),
        "recomputed_from": "sources.source_token_receipt + selection indices",
        "block_axis_sums": control_sums,
        "equals_candidate_variable_block_on_match_axes": True,
        "stream_materialized": False,
    }


def build_payload() -> dict:
    if MANIFEST_SHA256 is None:
        raise ValueError("stream manifest pin is unfilled (TODO-PIN)")
    if not MANIFEST.is_file() or sha256_file(MANIFEST) != MANIFEST_SHA256:
        raise ValueError("stream manifest changed")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if (
        manifest.get("experiment_id") != EXP.name
        or manifest.get("outcome") != "PASS_EXPOSURE_MATCH"
        or manifest.get("match_axes") != list(MATCH_AXES)
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
    candidate = files["axis160_direct"]["spans_per_epoch"]
    if candidate != manifest["exposure"]["arms"]["axis160_direct"]:
        raise ValueError("independent arm spans disagree with manifest: axis160_direct")
    notional_control = notional_control_bookkeeping(manifest)
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
        "notional_control_block": notional_control,
        "match_axes": manifest["match_axes"],
        "forward_tokens_per_arm": candidate["forward"],
        "nonzero_target_tokens_per_arm": candidate["nonzero_target"],
        "absolute_loss_mass_x5_per_arm": candidate["absolute_loss_mass_x5"],
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
        "notional_control_block": payload["notional_control_block"]["block_axis_sums"],
        "skipped_rows": payload["skipped_rows"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
