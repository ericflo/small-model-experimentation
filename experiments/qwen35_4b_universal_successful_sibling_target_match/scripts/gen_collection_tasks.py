#!/usr/bin/env python3
"""Materialize fresh tasks for greedy-failure and successful-sibling collection."""

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


CONSTRUCTION_SEED = 77115
ROWS_PER_SKILL = 48
MIX = ",".join(f"{skill}={ROWS_PER_SKILL}" for skill in curriculum.SKILLS)
ROWS = ROWS_PER_SKILL * len(curriculum.SKILLS)
SOURCE = EXP / "data" / "collection_tasks_seed77115.jsonl"
RUNNER_INPUT = EXP / "data" / "greedy_input_seed66115.jsonl"
MANIFEST = EXP / "data" / "collection_task_manifest.json"
PREFIX_EXP = ROOT / "experiments" / "qwen35_4b_universal_on_policy_prefix_repair_token_match"
RESTART_EXP = ROOT / "experiments" / "qwen35_4b_universal_failure_selected_restart_target_match"
PRIOR_SOURCES = (
    PREFIX_EXP / "data" / "rollout_tasks.jsonl",
    PREFIX_EXP / "data" / "local_tasks_seed88009.jsonl",
    RESTART_EXP / "data" / "rollout_tasks_seed77114.jsonl",
    RESTART_EXP / "data" / "local_tasks_seed88010.jsonl",
)
DISJOINT_LOCAL_SEEDS = tuple(range(88000, 88012))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_message(row: dict) -> bytes:
    return json.dumps(
        row["messages"], sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def jsonl_bytes(rows: list[dict]) -> bytes:
    return (
        "\n".join(
            json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            for row in rows
        )
        + "\n"
    ).encode()


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def generate_source() -> list[dict]:
    rows = curriculum.generate_curriculum(MIX, CONSTRUCTION_SEED)
    for row in rows:
        row["task_id"] = f"sibling{CONSTRUCTION_SEED}_{row['task_id']}"
        row["selection_skill"] = row["kind"].removeprefix("u_")
        row["construction_seed"] = CONSTRUCTION_SEED
    return rows


def validate(rows: list[dict]) -> dict:
    base_rows = [
        {key: value for key, value in row.items() if key not in {"selection_skill", "construction_seed"}}
        for row in rows
    ]
    summary = curriculum.validate_generated(base_rows)
    counts = Counter(row["selection_skill"] for row in rows)
    expected = Counter({skill: ROWS_PER_SKILL for skill in curriculum.SKILLS})
    if len(rows) != ROWS or counts != expected:
        raise ValueError(f"fresh rollout skill balance changed: {counts}")
    if any(row["kind"] != f"u_{row['selection_skill']}" for row in rows):
        raise ValueError("selection skill diverged from universal kind")
    if len({row["task_id"] for row in rows}) != ROWS:
        raise ValueError("fresh rollout task ids are not unique")
    messages = [canonical_message(row) for row in rows]
    if len(set(messages)) != ROWS:
        raise ValueError("fresh rollout messages are not unique")
    return {**summary, "skills": dict(sorted(counts.items()))}


def runner_rows(source_rows: list[dict]) -> list[dict]:
    rows = [
        {
            "id": row["task_id"],
            "messages": row["messages"],
            "meta": {
                "kind": row["kind"],
                "skill": row["selection_skill"],
                "surface": row["surface"],
                "level": row["level"],
                "construction_seed": CONSTRUCTION_SEED,
            },
        }
        for row in source_rows
    ]
    if any(set(row) != {"id", "messages", "meta"} for row in rows):
        raise ValueError("runner input schema leaked hidden fields")
    raw = jsonl_bytes(rows)
    forbidden = (b'"answer"', b'"think"', b'"_audit"', b'"truth_valid"')
    if any(value in raw for value in forbidden):
        raise ValueError("runner input contains an oracle field")
    return rows


def overlap_receipt(source_rows: list[dict]) -> dict:
    current = {canonical_message(row) for row in source_rows}
    prior_rows = [row for path in PRIOR_SOURCES for row in load_jsonl(path)]
    prior = {canonical_message(row) for row in prior_rows}
    prior_local = {
        canonical_message(row)
        for seed in DISJOINT_LOCAL_SEEDS
        for row in curriculum.generate_curriculum(curriculum.SMOKE_MIX, seed)
    }
    prior_overlap = current & prior
    seed_overlap = current & prior_local
    if prior_overlap or seed_overlap:
        raise ValueError("fresh rollout messages overlap a prior collection or local gate")
    return {
        "fresh_unique_messages": len(current),
        "prior_source_paths": [path.relative_to(ROOT).as_posix() for path in PRIOR_SOURCES],
        "prior_source_messages_compared": len(prior),
        "prior_source_overlap": 0,
        "local_seeds_compared": list(DISJOINT_LOCAL_SEEDS),
        "prior_local_messages_compared": len(prior_local),
        "prior_local_overlap": 0,
        "fresh_message_sha256s": sorted(sha256_bytes(value) for value in current),
    }


def build_outputs() -> dict[Path, bytes]:
    for path in PRIOR_SOURCES:
        if not path.is_file():
            raise ValueError(f"freshness input is absent: {path}")
    source_rows = generate_source()
    summary = validate(source_rows)
    public_rows = runner_rows(source_rows)
    source_bytes = jsonl_bytes(source_rows)
    runner_bytes = jsonl_bytes(public_rows)
    manifest = {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "stage": "fresh_successful_sibling_collection_substrate",
        "construction_seed": CONSTRUCTION_SEED,
        "rows": ROWS,
        "rows_per_skill": ROWS_PER_SKILL,
        "skills": list(curriculum.SKILLS),
        "mix": MIX,
        "source": {
            "path": SOURCE.relative_to(ROOT).as_posix(),
            "sha256": sha256_bytes(source_bytes),
            "contains_executable_truth": True,
            "summary": summary,
        },
        "runner_input": {
            "path": RUNNER_INPUT.relative_to(ROOT).as_posix(),
            "sha256": sha256_bytes(runner_bytes),
            "schema": ["id", "messages", "meta"],
            "excludes_oracle_think_answer_and_audit": True,
        },
        "freshness": overlap_receipt(source_rows),
        "reserved_local_seed": 88011,
        "local_seed_materialized": False,
        "reserved_conditional_aggregate_seed": 78141,
        "benchmark_data_read": False,
    }
    return {
        SOURCE: source_bytes,
        RUNNER_INPUT: runner_bytes,
        MANIFEST: (
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    outputs = build_outputs()
    if args.check:
        for path, expected in outputs.items():
            if not path.is_file() or path.read_bytes() != expected:
                parser.error(f"derived rollout substrate is absent or changed: {path}")
    else:
        conflict = next((path for path in outputs if path.exists()), None)
        if conflict is not None:
            parser.error(f"refusing to overwrite rollout substrate: {conflict}")
        for path, value in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
    print(
        json.dumps(
            {
                path.relative_to(ROOT).as_posix(): sha256_bytes(value)
                for path, value in outputs.items()
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
