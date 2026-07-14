#!/usr/bin/env python3
"""Freeze the fresh benchmark-free local gate and its model-facing input."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "scripts"))

import gen_curriculum as curriculum  # noqa: E402


SEED = 88009
MIX = curriculum.SMOKE_MIX
ROWS = 26
SOURCE = EXP / "data" / "local_tasks_seed88009.jsonl"
RUNNER_INPUT = EXP / "data" / "local_input_seed88009.jsonl"
RECEIPT = EXP / "data" / "local_design_receipt.json"
TRAINING_SOURCES = (
    EXP / "data" / "replay_after_close.jsonl",
    EXP / "data" / "prefix_repair_after_close.jsonl",
    EXP / "data" / "rollout_tasks.jsonl",
)
PRIOR_LOCAL_SEEDS = tuple(range(88000, 88009))
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
PARENT_MERGE_RECEIPT = EXP / "runs" / "merges" / "close_xi_parent.json"
CONTROL_TRAINING_RECEIPT = EXP / "runs" / "training" / "replay_after_close.json"
CANDIDATE_TRAINING_RECEIPT = (
    EXP / "runs" / "training" / "prefix_repair_after_close.json"
)
EXPECTED_PARENT_MERGE_RECEIPT_SHA256 = (
    "10c3870deefb638fcbf2f7980fe39e35be4c08f0ab0cbabceafbf87a5231895b"
)
EXPECTED_CONTROL_RECEIPT_SHA256 = (
    "f78f2069fd1c7b37bbd0b13b581df0ce7360de92256323fcf5f3c7b0936ed6de"
)
EXPECTED_CANDIDATE_RECEIPT_SHA256 = (
    "846d8107ecadad458c18cd985d54feb42748e87677dd708c14a99e84cf4e7098"
)
CODE_FILES = {
    "generator": Path(__file__),
    "curriculum": EXP / "scripts" / "gen_curriculum.py",
    "gate": EXP / "scripts" / "check_local.py",
    "merge": EXP / "scripts" / "merge_trained_arm.py",
    "external_merger": (
        ROOT
        / "experiments"
        / "qwen35_4b_same_prefix_advantage_routing"
        / "scripts"
        / "merge_adapter.py"
    ),
    "training_authenticator": EXP / "scripts" / "train_trial.py",
    "evaluator": EXP / "scripts" / "eval_local_vllm.py",
    "runner": EXP / "src" / "vllm_runner.py",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonl_bytes(rows: list[dict]) -> bytes:
    return "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows
    ).encode()


def message_bytes(row: dict) -> bytes:
    return json.dumps(
        row["messages"], sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def build_rows() -> tuple[list[dict], list[dict]]:
    source_rows = curriculum.generate_curriculum(MIX, SEED)
    for row in source_rows:
        row["task_id"] = f"local{SEED}_{row['task_id']}"
    summary = curriculum.validate_generated(source_rows)
    expected_kinds = {f"u_{name}": 2 for name in curriculum.SKILLS}
    if summary["rows"] != ROWS or summary["kinds"] != expected_kinds:
        raise ValueError("fresh local mix no longer has two rows per registered skill")
    runner_rows = [
        {
            "id": row["task_id"],
            "messages": row["messages"],
            "meta": {
                "kind": row["kind"],
                "surface": row["surface"],
                "seed": SEED,
            },
        }
        for row in source_rows
    ]
    if any(set(row) != {"id", "messages", "meta"} for row in runner_rows):
        raise ValueError("local runner input schema leaked hidden fields")
    return source_rows, runner_rows


def overlap_receipt(source_rows: list[dict]) -> dict:
    local_messages = {message_bytes(row) for row in source_rows}
    training_messages = {
        message_bytes(row)
        for path in TRAINING_SOURCES
        for row in load_jsonl(path)
        if row.get("messages")
    }
    prior_messages = {
        message_bytes(row)
        for seed in PRIOR_LOCAL_SEEDS
        for row in curriculum.generate_curriculum(MIX, seed)
    }
    training_overlap = len(local_messages & training_messages)
    prior_local_overlap = len(local_messages & prior_messages)
    if training_overlap or prior_local_overlap:
        raise ValueError("fresh local prompts overlap training or prior reserved local seeds")
    return {
        "message_sha256s": sorted(sha256_bytes(value) for value in local_messages),
        "unique_local_messages": len(local_messages),
        "training_messages_compared": len(training_messages),
        "training_overlap": training_overlap,
        "prior_local_seeds_compared": list(PRIOR_LOCAL_SEEDS),
        "prior_local_messages_compared": len(prior_messages),
        "prior_local_overlap": prior_local_overlap,
    }


def build_outputs() -> dict[Path, bytes]:
    for path in (*TRAINING_SOURCES, *CODE_FILES.values()):
        if not path.is_file():
            raise ValueError(f"required local-design input is absent: {path}")
    pinned_receipts = {
        PARENT_MERGE_RECEIPT: EXPECTED_PARENT_MERGE_RECEIPT_SHA256,
        CONTROL_TRAINING_RECEIPT: EXPECTED_CONTROL_RECEIPT_SHA256,
        CANDIDATE_TRAINING_RECEIPT: EXPECTED_CANDIDATE_RECEIPT_SHA256,
    }
    for path, expected in pinned_receipts.items():
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"published prerequisite changed: {path}")
    source_rows, runner_rows = build_rows()
    source = jsonl_bytes(source_rows)
    runner_input = jsonl_bytes(runner_rows)
    kinds = Counter(row["kind"] for row in source_rows)
    receipt = {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "stage": "fresh_local_gate_design",
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "loaded": False,
            "calls": 0,
        },
        "seed": SEED,
        "mix": MIX,
        "rows": ROWS,
        "kinds": dict(sorted(kinds.items())),
        "source": {
            "path": SOURCE.relative_to(ROOT).as_posix(),
            "sha256": sha256_bytes(source),
            "contains_executable_truth": True,
        },
        "runner_input": {
            "path": RUNNER_INPUT.relative_to(ROOT).as_posix(),
            "sha256": sha256_bytes(runner_input),
            "schema": ["id", "messages", "meta"],
            "contains_answer": False,
            "contains_oracle": False,
        },
        "freshness": overlap_receipt(source_rows),
        "backend": {
            "name": "vllm_merged_composite",
            "thinking": "natural",
            "greedy": True,
            "samples_per_task": 1,
            "max_tokens": 1024,
            "max_model_len": 4096,
            "max_num_seqs": 16,
            "max_num_batched_tokens": 8192,
            "same_runner_and_geometry_for_every_arm": True,
            "runtime_lora_forbidden": True,
        },
        "arms": [
            "close_xi_parent",
            "replay_after_close",
            "prefix_repair_after_close",
        ],
        "gates": {
            "parsed_at_least": 24,
            "correct_at_least": 17,
            "cap_contacts_at_most": 2,
            "route_abstentions_at_most": 1,
            "execute_correct_at_least": 1,
            "induct_correct_at_least": 1,
            "probe_correct_at_least": 1,
            "candidate_strictly_beats_parent_total": True,
            "candidate_strictly_beats_replay_total": True,
            "candidate_strictly_beats_parent_execute_induct_probe": True,
            "candidate_strictly_beats_replay_execute_induct_probe": True,
        },
        "prerequisites": {
            "parent_merge_receipt_sha256": EXPECTED_PARENT_MERGE_RECEIPT_SHA256,
            "control_training_receipt_sha256": EXPECTED_CONTROL_RECEIPT_SHA256,
            "candidate_training_receipt_sha256": EXPECTED_CANDIDATE_RECEIPT_SHA256,
        },
        "code_sha256": {
            name: sha256_file(path) for name, path in sorted(CODE_FILES.items())
        },
        "firewall": {
            "benchmark_data_read": False,
            "benchmark_gateway_exposed": False,
            "aggregate_seed_sealed": True,
        },
        "next_authorized_stage": "merge-control",
    }
    rendered_receipt = (
        json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()
    return {SOURCE: source, RUNNER_INPUT: runner_input, RECEIPT: rendered_receipt}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    outputs = build_outputs()
    if args.check:
        changed = [path for path, value in outputs.items() if not path.is_file() or path.read_bytes() != value]
        if changed:
            parser.error("local-gate artifacts are absent or changed: " + ", ".join(map(str, changed)))
    else:
        existing = [path for path in outputs if path.exists()]
        if existing:
            parser.error("refusing to overwrite local-gate artifacts: " + ", ".join(map(str, existing)))
        for path, value in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
    print(
        json.dumps(
            {
                "rows": ROWS,
                "source_sha256": sha256_bytes(outputs[SOURCE]),
                "runner_input_sha256": sha256_bytes(outputs[RUNNER_INPUT]),
                "receipt_sha256": sha256_bytes(outputs[RECEIPT]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
