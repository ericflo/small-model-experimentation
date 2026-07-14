#!/usr/bin/env python3
"""Generate fresh, truth-audited tasks for on-policy parent rollouts.

The model-facing input never contains the oracle continuation or expected answer.
Those fields remain in the experiment-owned source so a later model-free miner can
classify failures and attach corrective continuations without benchmark access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))

import gen_curriculum as curriculum  # noqa: E402


SEED = 77113
ROWS_PER_CLASS = 48
FAILURE_CLASSES = (
    "declaration_operation",
    "state_transition",
    "bounded_induction",
    "probe_scoring",
    "repair_propagation",
    "commit_serialization",
)
SOURCE = EXP / "data" / "rollout_tasks.jsonl"
RUNNER_INPUT = EXP / "data" / "parent_rollout_input.jsonl"
MANIFEST = EXP / "data" / "rollout_task_manifest.json"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def stable_class_seed(name: str) -> int:
    digest = hashlib.sha256(f"{SEED}:{name}".encode()).digest()
    return SEED + int.from_bytes(digest[:4], "big")


def _source_row(row: dict, failure_class: str, index: int) -> dict:
    if row.get("_audit", {}).get("truth_valid") is not True:
        raise ValueError(f"source builder lost its executable truth audit: {failure_class}")
    expected = row["answer"].removeprefix("ANSWER: ").strip()
    if not expected or "\n" in expected or not row.get("think", "").strip():
        raise ValueError(f"malformed oracle target: {failure_class}")
    return {
        "messages": row["messages"],
        "oracle_think": row["think"].strip(),
        "answer": f"ANSWER: {expected}",
        "kind": f"u_on_policy_{failure_class}",
        "family": "universal_on_policy_prefix_repair",
        "failure_class": failure_class,
        "surface": row["surface"],
        "level": row["level"],
        "task_id": f"uop_{failure_class}_{index:05d}",
        "_audit": {
            "schema_version": 1,
            "truth_valid": True,
            "source_kind": row["kind"],
            "source_audit": row["_audit"],
            "expected": expected,
        },
    }


def _declaration_row(rng: random.Random) -> dict:
    """Make the cycle line pure reference data, never an executable step."""
    for _ in range(500):
        row = curriculum.execute_lesson(rng)
        procedure = row["messages"][0]["content"].partition("Cycle order:")[0]
        if row["level"] == 4 and "advance every item" not in procedure:
            row["_audit"] = {
                **row["_audit"],
                "cycle_is_reference_only": True,
                "listed_operations": 4,
            }
            return row
    raise RuntimeError("could not synthesize a declaration-only cycle task")


def _state_transition_row(rng: random.Random) -> dict:
    for _ in range(300):
        row = curriculum.execute_lesson(rng)
        if row["level"] == 4:
            return row
    raise RuntimeError("could not synthesize a four-step execution task")


def _repair_row(rng: random.Random) -> dict:
    for _ in range(300):
        row = curriculum.repair_lesson(rng)
        if row["level"] == 4:
            return row
    raise RuntimeError("could not synthesize a four-step repair task")


def _commit_row(rng: random.Random) -> dict:
    """Expose verified work and require immediate exact serialization."""
    source = _state_transition_row(rng)
    expected = source["answer"].removeprefix("ANSWER: ").strip()
    prompt = (
        "The scratch work below has already been independently verified. Do not redo any "
        "operation. Read its final-state sentence, close immediately, and return exactly one "
        "line of the form ANSWER: <final state>.\nVerified scratch work:\n"
        + source["think"]
    )
    return {
        **source,
        "messages": [{"role": "user", "content": prompt}],
        "think": (
            f"The verified final state is {expected}. No operation remains; commit exactly now."
        ),
        "_audit": {
            **source["_audit"],
            "verified_work_in_prompt": True,
            "immediate_commit_required": True,
        },
    }


BUILDERS = {
    "declaration_operation": _declaration_row,
    "state_transition": _state_transition_row,
    "bounded_induction": curriculum.induct_lesson,
    "probe_scoring": curriculum.probe_lesson,
    "repair_propagation": _repair_row,
    "commit_serialization": _commit_row,
}


def generate() -> list[dict]:
    rows: list[dict] = []
    for failure_class in FAILURE_CLASSES:
        rng = random.Random(stable_class_seed(failure_class))
        builder = BUILDERS[failure_class]
        for index in range(ROWS_PER_CLASS):
            rows.append(_source_row(builder(rng), failure_class, index))
    order = list(range(len(rows)))
    random.Random(SEED + 1).shuffle(order)
    rows = [rows[index] for index in order]
    validate(rows)
    return rows


def validate(rows: list[dict]) -> None:
    expected_fields = {
        "messages", "oracle_think", "answer", "kind", "family", "failure_class",
        "surface", "level", "task_id", "_audit",
    }
    if len(rows) != ROWS_PER_CLASS * len(FAILURE_CLASSES):
        raise ValueError("rollout task count changed")
    if Counter(row["failure_class"] for row in rows) != Counter(
        {name: ROWS_PER_CLASS for name in FAILURE_CLASSES}
    ):
        raise ValueError("failure-class balance changed")
    prompts: set[str] = set()
    task_ids: set[str] = set()
    for row in rows:
        if set(row) != expected_fields:
            raise ValueError(f"task schema changed: {sorted(row)}")
        if (
            len(row["messages"]) != 1
            or row["messages"][0].get("role") != "user"
            or row["_audit"].get("truth_valid") is not True
            or row["_audit"].get("expected")
            != row["answer"].removeprefix("ANSWER: ").strip()
        ):
            raise ValueError(f"invalid truth/message contract: {row['task_id']}")
        prompt = row["messages"][0]["content"]
        if prompt in prompts or row["task_id"] in task_ids:
            raise ValueError("duplicate prompt or task id")
        prompts.add(prompt)
        task_ids.add(row["task_id"])
        if row["failure_class"] == "declaration_operation":
            procedure = prompt.partition("Cycle order:")[0]
            if "advance every item" in procedure:
                raise ValueError("declaration task contains a cycle-advance operation")
        if row["failure_class"] == "commit_serialization" and not row["_audit"][
            "source_audit"
        ].get("immediate_commit_required"):
            raise ValueError("commit task lost its policy audit")


def render_source(rows: list[dict]) -> bytes:
    return (
        "\n".join(json.dumps(row, sort_keys=True, ensure_ascii=False) for row in rows)
        + "\n"
    ).encode()


def render_runner_input(rows: list[dict]) -> bytes:
    public = [
        {
            "id": row["task_id"],
            "messages": row["messages"],
            "meta": {
                "failure_class": row["failure_class"],
                "surface": row["surface"],
                "level": row["level"],
            },
        }
        for row in rows
    ]
    return (
        "\n".join(json.dumps(row, sort_keys=True, ensure_ascii=False) for row in public)
        + "\n"
    ).encode()


def build_outputs() -> dict[Path, bytes]:
    rows = generate()
    source = render_source(rows)
    runner_input = render_runner_input(rows)
    manifest = {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "stage": "model_free_rollout_task_freeze",
        "construction_seed": SEED,
        "rows": len(rows),
        "rows_per_failure_class": ROWS_PER_CLASS,
        "failure_classes": list(FAILURE_CLASSES),
        "source": {"path": SOURCE.relative_to(EXP).as_posix(), "sha256": sha256_bytes(source)},
        "runner_input": {
            "path": RUNNER_INPUT.relative_to(EXP).as_posix(),
            "sha256": sha256_bytes(runner_input),
        },
        "runner_input_excludes_hidden_oracle_fields": True,
        "local_seed_materialized": False,
        "benchmark_data_read": False,
    }
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()
    return {SOURCE: source, RUNNER_INPUT: runner_input, MANIFEST: manifest_bytes}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    outputs = build_outputs()
    if args.check:
        for path, expected in outputs.items():
            if not path.is_file() or path.read_bytes() != expected:
                parser.error(f"frozen rollout task artifact is absent or changed: {path}")
    else:
        conflicts = [path for path in outputs if path.exists()]
        if conflicts:
            parser.error(f"refusing to overwrite rollout task artifact: {conflicts[0]}")
        for path, value in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
    manifest = json.loads(outputs[MANIFEST])
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
