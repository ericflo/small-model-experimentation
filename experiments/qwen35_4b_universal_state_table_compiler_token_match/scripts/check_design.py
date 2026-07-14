#!/usr/bin/env python3
"""Recompute the model-free design, truth, compute, and gate-reachability receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))

import gen_state_table_curriculum as generator  # noqa: E402
import materialize_streams as streams  # noqa: E402


SOURCE = EXP / "data" / "state_table_curriculum_source.jsonl"
SOURCE_LENGTHS = EXP / "data" / "source_token_lengths.json"
TOKEN_RECEIPT = EXP / "data" / "stream_token_receipt.json"
STREAM_MANIFEST = EXP / "data" / "stream_manifest.json"
EXPECTED_SOURCE_SHA256 = "a7b453afa0d2273b7008a96a8460086b62ae7004fa7aa4557493728cd87e88bb"
EXPECTED_SOURCE_LENGTHS_SHA256 = "5f9b22d21b110ca5758cd5d45296d38f1e41f4d0e9f4bb0496d29770d743c33d"
EXPECTED_TOKEN_RECEIPT_SHA256 = "163e40a61d0b3f4dc541f56ea32510bacb8ce64f658e00f47e5867da4a45f0b8"
EXPECTED_STREAM_MANIFEST_SHA256 = "69f37b2bcc4854f831fa8cb915374b55a5b63641ece579018222ed99b1f47a61"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"not an object: {path}")
    return payload


def build_receipt() -> dict:
    if (
        sha256_file(SOURCE) != EXPECTED_SOURCE_SHA256
        or sha256_file(SOURCE_LENGTHS) != EXPECTED_SOURCE_LENGTHS_SHA256
        or sha256_file(TOKEN_RECEIPT) != EXPECTED_TOKEN_RECEIPT_SHA256
        or sha256_file(STREAM_MANIFEST) != EXPECTED_STREAM_MANIFEST_SHA256
    ):
        raise ValueError("one or more frozen data identities changed")

    rows = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines()]
    regenerated = generator.generate()
    if generator.render_rows(regenerated) != SOURCE.read_bytes() or rows != regenerated:
        raise ValueError("source bytes do not equal deterministic generator output")
    for row in rows:
        generator.validate_row(row)

    source_lengths = load_json(SOURCE_LENGTHS)["sources"]["curriculum"]["lengths"]
    if len(source_lengths) != len(rows):
        raise ValueError("source token lengths lost row alignment")
    stage_tokens: dict[str, Counter] = defaultdict(Counter)
    for row, lengths in zip(rows, source_lengths, strict=True):
        stage = row["_audit"]["stage"]
        stage_tokens[stage].update(
            {
                "rows": 1,
                "forward": lengths["forward"],
                "prompt": lengths["prompt"],
                "think_target": lengths["think_target"],
                "close_target": lengths["close_target"],
                "answer_target": lengths["answer_target"],
            }
        )

    expected_outputs, expected_manifest = streams.build_outputs()
    manifest = load_json(STREAM_MANIFEST)
    if manifest != expected_manifest:
        raise ValueError("stream manifest does not equal deterministic materialization")
    output_paths = {
        name: EXP / "data" / name for name in expected_outputs
    }
    if any(path.read_bytes() != expected_outputs[name] for name, path in output_paths.items()):
        raise ValueError("stream bytes do not equal deterministic materialization")
    replay = expected_outputs["replay_after_close.jsonl"].decode().splitlines()
    candidate = expected_outputs["state_table_after_close.jsonl"].decode().splitlines()
    identical_positions = sum(
        left == right for left, right in zip(replay, candidate, strict=True)
    )

    token_receipt = load_json(TOKEN_RECEIPT)
    files = {Path(row["path"]).name: row for row in token_receipt["files"]}
    control = files["replay_after_close.jsonl"]
    intervention = files["state_table_after_close.jsonl"]
    fields = (
        "prompt_tokens_per_epoch",
        "think_target_tokens_per_epoch",
        "close_target_tokens_per_epoch",
        "answer_target_tokens_per_epoch",
    )
    target_deltas = {
        field: intervention[field] - control[field] for field in fields
    }

    local_rows = 26
    kind_rows = 2
    absolute = {
        "accuracy": {
            "threshold": 0.65,
            "minimum_correct": math.ceil(0.65 * local_rows),
            "maximum_correct": local_rows,
            "reachable": math.ceil(0.65 * local_rows) <= local_rows,
        },
        "parse_rate": {
            "threshold": 0.90,
            "minimum_parsed": math.ceil(0.90 * local_rows),
            "maximum_parsed": local_rows,
            "reachable": math.ceil(0.90 * local_rows) <= local_rows,
        },
        "cap_contacts": {"maximum_allowed": 2, "minimum_possible": 0, "reachable": True},
        "route_abstentions": {"maximum_allowed": 1, "minimum_possible": 0, "reachable": True},
        "execute": {"threshold": 0.50, "minimum_correct": 1, "rows": kind_rows, "reachable": True},
        "induct": {"threshold": 0.50, "minimum_correct": 1, "rows": kind_rows, "reachable": True},
        "probe": {"threshold": 0.50, "minimum_correct": 1, "rows": kind_rows, "reachable": True},
    }
    if not all(value["reachable"] for value in absolute.values()):
        raise ValueError("an absolute local gate is structurally unreachable")

    run_benchmark = (EXP / "scripts" / "run_benchmark.py").read_text(encoding="utf-8")
    run_source = (EXP / "scripts" / "run.py").read_text(encoding="utf-8")
    if "benchmarks/" in run_benchmark or "benchmarks/" in run_source:
        raise ValueError("benchmark implementation path leaked into experiment harness")

    return {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "stage": "model_free_design_freeze",
        "model": {
            "id": "Qwen/Qwen3.5-4B",
            "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            "loaded": False,
            "calls": 0,
        },
        "seeds": {
            "construction": 77112,
            "training": 46,
            "local": 88008,
            "conditional_aggregate": 78138,
        },
        "parent": expected_manifest["parent"],
        "curriculum": {
            "rows": len(rows),
            "sha256": EXPECTED_SOURCE_SHA256,
            "stages": dict(sorted(Counter(row["_audit"]["stage"] for row in rows).items())),
            "depths": dict(sorted(Counter(str(row["depth"]) for row in rows).items())),
            "surfaces": dict(sorted(Counter(row["surface"] for row in rows).items())),
            "correct_hypothesis_positions": dict(sorted(Counter(
                str(row["_audit"]["correct_index"] + 1)
                for row in rows if row["_audit"]["stage"] == "score"
            ).items())),
            "stage_tokens": {
                stage: dict(sorted(values.items()))
                for stage, values in sorted(stage_tokens.items())
            },
            "all_truth_audits_recomputed": True,
            "predecessor_canonical_vocabulary_absent": True,
        },
        "compute": {
            "arms": ["replay_after_close", "state_table_after_close"],
            "rows_per_arm": token_receipt["rows_per_arm"],
            "forward_tokens_per_arm": token_receipt["forward_tokens_per_arm"],
            "skipped_rows": token_receipt["skipped_rows"],
            "optimizer_steps": 40,
            "shared_position_aligned_rows": identical_positions,
            "block_forward_tokens": expected_manifest["selection"]["block_forward_tokens"],
            "candidate_minus_control_target_token_deltas": target_deltas,
        },
        "local_admission": {
            "rows": local_rows,
            "absolute_gates": absolute,
            "all_absolute_gates_reachable": True,
            "target_kinds": ["u_execute", "u_induct", "u_probe"],
            "target_rows": 6,
            "relative_gates": {
                "candidate_must_strictly_beat_each_control_total": True,
                "candidate_maximum_total": 26,
                "control_maximum_total_allowing_a_strict_win": 25,
                "candidate_must_strictly_beat_each_control_on_target": True,
                "candidate_maximum_target": 6,
                "control_maximum_target_allowing_a_strict_win": 5,
                "admissible_region_nonempty": True,
                "perfect_control_means_fail_closed": True,
            },
        },
        "aggregate_admission": {
            "conditionally_available": True,
            "seed": 78138,
            "tier": "quick",
            "think_budget": 1024,
            "backend": "qwen_vllm",
            "public_family_count": 10,
            "strictly_positive_every_family_required": True,
            "must_meet_or_beat_all_named_controls": True,
            "benchmark_data_read_during_design": False,
            "gateway_only": True,
        },
        "checkpoint_policy": {
            "one_expensive_stage_per_invocation": True,
            "clean_worktree_required": True,
            "preceding_receipt_must_be_committed_at_head": True,
        },
        "source_identities": {
            "source_token_lengths_sha256": EXPECTED_SOURCE_LENGTHS_SHA256,
            "stream_manifest_sha256": EXPECTED_STREAM_MANIFEST_SHA256,
            "token_receipt_sha256": EXPECTED_TOKEN_RECEIPT_SHA256,
            "streams": {name: sha256_file(path) for name, path in sorted(output_paths.items())},
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--out", type=Path, default=EXP / "data" / "design_receipt.json"
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    value = (
        json.dumps(build_receipt(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()
    if args.check:
        if not args.out.is_file() or args.out.read_bytes() != value:
            parser.error("design receipt is absent or changed")
    else:
        if args.out.exists():
            parser.error("refusing to overwrite design receipt")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(value)
    print(json.dumps({"out": str(args.out), "sha256": hashlib.sha256(value).hexdigest()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
