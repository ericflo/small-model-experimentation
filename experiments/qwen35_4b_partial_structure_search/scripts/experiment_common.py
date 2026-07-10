#!/usr/bin/env python3
"""Shared serialization, prompt, oracle-map, fill, and selection helpers."""

from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))
import families as F  # noqa: E402


DSL_TASK_TEXT = """A list is transformed by a pipeline applied from left to right.
The legal operation TYPES are:
reverse; sort_asc; sort_desc; unique_stable; dedup_adjacent; abs_all; square; negate;
running_sum; adjacent_diff; add_k; mul_k; mod_k; take_k; drop_k; rotate_k.
Parameterized types use one shared legal integer at their position. The parameter is not shown in a type
skeleton: add_k has k in {-3,-2,-1,1,2,3}; mul_k in {-2,2,3}; mod_k in {2,3,4};
take_k in {1,2,3,4}; drop_k in {1,2,3}; rotate_k in {1,2,3}.
One parameter choice per parameterized position must work for every visible example."""


def load_config() -> dict[str, Any]:
    return yaml.safe_load((EXP / "configs" / "default.yaml").read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")


def success_rows(oracle_record: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(row) for row in oracle_record.get("successful_skeletons", [])]


def live_prefix_maps(
    successful: Sequence[Mapping[str, Any]], depth: int
) -> tuple[dict[tuple[str, ...], int], dict[tuple[str, ...], int]]:
    """Return exact successful-skeleton and parameter-fill counts by prefix."""
    skeleton_counts: dict[tuple[str, ...], int] = defaultdict(int)
    fill_counts: dict[tuple[str, ...], int] = defaultdict(int)
    for row in successful:
        skeleton = tuple(str(name) for name in row["skeleton"])
        if len(skeleton) != depth:
            raise ValueError("oracle skeleton length does not match task depth")
        parameter_count = int(row["parameter_fill_count"])
        for length in range(depth + 1):
            prefix = skeleton[:length]
            skeleton_counts[prefix] += 1
            fill_counts[prefix] += parameter_count
    return dict(skeleton_counts), dict(fill_counts)


def calibration_record(
    task: Mapping[str, Any],
    prefix: Sequence[str],
    *,
    record_id: str,
    **meta: Any,
) -> dict[str, Any]:
    """Whitelist the only fields the model layer may see."""
    depth = int(task["depth"])
    normalized = [str(name) for name in prefix]
    return {
        "id": record_id,
        "task_text": DSL_TASK_TEXT,
        "visible_examples": task["visible"],
        "candidate_prefix": normalized,
        "remaining_steps": depth - len(normalized),
        "task_id": str(task["task_id"]),
        "prefix_len": len(normalized),
        **meta,
    }


def prompt_boundary_audit(task: Mapping[str, Any], prefix: Sequence[str]) -> bool:
    """Oracle-field deletion cannot change the whitelisted model record."""
    original = calibration_record(task, prefix, record_id="audit")
    stripped = dict(task)
    for key in (
        "label_probe",
        "hidden",
        "target_pipeline",
        "target_skeleton",
        "target_ops",
        "min_depth_audit",
    ):
        stripped.pop(key, None)
    return original == calibration_record(stripped, prefix, record_id="audit")


def task_cases(task: Mapping[str, Any], split: str) -> tuple[list[list[int]], list[list[int]]]:
    rows = list(task[split])
    return [row["input"] for row in rows], [row["output"] for row in rows]


def fills_passing(
    skeleton: Sequence[str],
    task: Mapping[str, Any],
    *,
    splits: Sequence[str] = ("visible",),
    fill_cap: int | None = None,
) -> tuple[list[F.Pipeline], dict[str, int]]:
    """Enumerate parameter fills in frozen order and retain all case-passers."""
    inputs, outputs = F.task_cases(task, splits)
    attempted = primitive_applications = 0
    passers: list[F.Pipeline] = []
    for pipeline in F.enumerate_parameter_fills(tuple(skeleton)):
        if fill_cap is not None and attempted >= fill_cap:
            break
        attempted += 1
        primitive_applications += len(pipeline) * len(inputs)
        if F.pipeline_solves(pipeline, inputs, outputs):
            passers.append(pipeline)
    return passers, {
        "parameter_fills_attempted": attempted,
        "case_pipeline_executions": attempted * len(inputs),
        "primitive_applications": primitive_applications,
    }


def _selector_inputs(task_id: str, count: int = 16) -> list[list[int]]:
    seed = int.from_bytes(hashlib.sha256(task_id.encode("utf-8")).digest()[:8], "big")
    rng = random.Random(seed)
    return [[rng.randint(-9, 9) for _ in range(rng.randint(5, 8))] for _ in range(count)]


def consensus_select(
    candidates: Sequence[F.Pipeline], task: Mapping[str, Any]
) -> tuple[F.Pipeline | None, dict[str, Any]]:
    """Select a plurality behavior on unlabeled deterministic probe inputs."""
    if not candidates:
        return None, {"abstained": True, "candidate_count": 0, "signature_groups": 0}
    probes = _selector_inputs(str(task["task_id"]))
    groups: dict[tuple[Any, ...], list[F.Pipeline]] = defaultdict(list)
    for pipeline in candidates:
        signature = tuple(
            tuple(output) if (output := F.execute_pipeline(pipeline, value)) is not None else None
            for value in probes
        )
        groups[signature].append(pipeline)
    # Stable tie-break by signature JSON and then pipeline repr, never hidden correctness.
    best_signature = min(
        groups,
        key=lambda signature: (-len(groups[signature]), json.dumps(signature, sort_keys=True)),
    )
    chosen = min(groups[best_signature], key=lambda pipeline: tuple(F.format_op(op) for op in pipeline))
    return chosen, {
        "abstained": False,
        "candidate_count": len(candidates),
        "signature_groups": len(groups),
        "plurality_size": len(groups[best_signature]),
    }


def hidden_grade(pipeline: F.Pipeline | None, task: Mapping[str, Any]) -> bool:
    if pipeline is None:
        return False
    inputs, outputs = task_cases(task, "hidden")
    return F.pipeline_solves(pipeline, inputs, outputs)


def aggregate_counts(rows: Iterable[Mapping[str, int]]) -> dict[str, int]:
    total: Counter[str] = Counter()
    for row in rows:
        total.update({key: int(value) for key, value in row.items()})
    return dict(total)
