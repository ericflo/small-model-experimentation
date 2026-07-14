#!/usr/bin/env python3
"""Solve and materialize exact three-axis replay/restart training streams."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, vstack


EXP = Path(__file__).resolve().parents[1]
RESTART = EXP / "data" / "counterfactual_restart_source.jsonl"
REPLAY = EXP / "data" / "sft_blend.jsonl"
TOKENS = EXP / "data" / "source_token_lengths.json"
PREDECESSOR = EXP / "data" / "predecessor_stream_manifest.json"
MANIFEST = EXP / "data" / "stream_manifest.json"
CONTROL_OUT = EXP / "data" / "replay_control.jsonl"
CANDIDATE_OUT = EXP / "data" / "counterfactual_restart_candidate.jsonl"
RESTART_SHA256 = "022b1ea4cfe2bb50fca7f5fdc472a0bf228a5d7a7adb637b221b8efe434d951f"
REPLAY_SHA256 = "25a9595f2e70e4d5cab0a730f0e2613d314843f2a5dfe96187bc30d5d2abf0c2"
TOKENS_SHA256 = "ac9b9c8a3c9bfc66699781c96792ea72c37701b11719772764e74b35dba10bd6"
PREDECESSOR_SHA256 = "abf8b5055e68c0fb2bb6e32a29f7be3b3677a0dd179e77397647777a2aa0966f"
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
COMMON_ROWS = 200
RESTART_ROWS = 52
FILLER_ROWS = 68
CONTROL_ROWS = 120
ROWS_PER_ARM = 320
STREAM_ORDER_SEED = 55115
MATCH_AXES = ("forward", "nonzero_target", "absolute_loss_mass_x5")
SOLVER_TIME_LIMIT_SECONDS = 300.0


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_lines(path: Path, expected_sha256: str, expected_rows: int) -> list[tuple[str, dict]]:
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise ValueError(f"frozen source changed: {path}")
    rows = [
        (line, json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(rows) != expected_rows:
        raise ValueError(f"unexpected source rows: {path}")
    return rows


def load_lengths() -> tuple[list[dict[str, int]], list[dict[str, int]], dict]:
    if not TOKENS.is_file() or sha256_file(TOKENS) != TOKENS_SHA256:
        raise ValueError("source token receipt changed")
    payload = json.loads(TOKENS.read_text(encoding="utf-8"))
    if (
        payload.get("experiment_id") != EXP.name
        or payload.get("model_id") != MODEL_ID
        or payload.get("model_revision") != MODEL_REVISION
        or payload.get("max_length") != 4096
        or payload.get("match_axes") != list(MATCH_AXES)
    ):
        raise ValueError("source token receipt identity changed")
    restart = payload["sources"]["restart"]
    replay = payload["sources"]["replay"]
    if (
        restart.get("sha256") != RESTART_SHA256
        or restart.get("rows") != RESTART_ROWS
        or replay.get("sha256") != REPLAY_SHA256
        or replay.get("rows") != 2240
    ):
        raise ValueError("source token receipt lineage changed")
    return restart["lengths"], replay["lengths"], payload


def span_sum(indices: list[int], lengths: list[dict[str, int]]) -> dict[str, int]:
    fields = tuple(lengths[0])
    return {field: sum(lengths[index][field] for index in indices) for field in fields}


def vector_sum(indices: list[int], lengths: list[dict[str, int]]) -> tuple[int, ...]:
    return tuple(sum(lengths[index][axis] for index in indices) for axis in MATCH_AXES)


def load_common_core(replay_lengths: list[dict[str, int]]) -> list[int]:
    if not PREDECESSOR.is_file() or sha256_file(PREDECESSOR) != PREDECESSOR_SHA256:
        raise ValueError("predecessor stream manifest changed")
    payload = json.loads(PREDECESSOR.read_text(encoding="utf-8"))
    core = payload.get("selection", {}).get("replay_core_source_indices", [])
    if (
        payload.get("experiment_id") != "qwen35_4b_universal_close_weight_token_match"
        or len(core) != COMMON_ROWS
        or len(set(core)) != COMMON_ROWS
        or min(core) < 0
        or max(core) >= len(replay_lengths)
    ):
        raise ValueError("predecessor replay core identity changed")
    return core


def solve_exact_match(
    available: list[int],
    replay_lengths: list[dict[str, int]],
    restart_vector: tuple[int, ...],
) -> tuple[list[int], list[int], dict]:
    """Find disjoint 68/120 replay subsets with exact integer residuals."""
    count = len(available)
    variables = count * 2
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    lower: list[float] = []
    upper: list[float] = []

    # Exact filler and control cardinalities.
    for local in range(count):
        rows.append(0)
        cols.append(local)
        data.append(1.0)
        rows.append(1)
        cols.append(count + local)
        data.append(1.0)
    lower.extend((FILLER_ROWS, CONTROL_ROWS))
    upper.extend((FILLER_ROWS, CONTROL_ROWS))

    # control - filler = fixed restart exposure on all three axes.
    for axis_offset, axis in enumerate(MATCH_AXES, start=2):
        for local, source_index in enumerate(available):
            value = replay_lengths[source_index][axis]
            rows.extend((axis_offset, axis_offset))
            cols.extend((local, count + local))
            data.extend((-float(value), float(value)))
        target = restart_vector[axis_offset - 2]
        lower.append(float(target))
        upper.append(float(target))

    equality = coo_matrix((data, (rows, cols)), shape=(5, variables)).tocsr()
    # A replay row cannot appear in both variable blocks.
    disjoint_rows = np.repeat(np.arange(count), 2)
    disjoint_cols = np.column_stack((np.arange(count), count + np.arange(count))).ravel()
    disjoint = coo_matrix(
        (np.ones(count * 2), (disjoint_rows, disjoint_cols)),
        shape=(count, variables),
    ).tocsr()
    matrix = vstack((equality, disjoint), format="csr")
    constraint = LinearConstraint(
        matrix,
        np.asarray(lower + [-np.inf] * count),
        np.asarray(upper + [1.0] * count),
    )
    started = time.perf_counter()
    result = milp(
        c=np.zeros(variables),
        integrality=np.ones(variables),
        bounds=Bounds(np.zeros(variables), np.ones(variables)),
        constraints=constraint,
        options={
            "time_limit": SOLVER_TIME_LIMIT_SECONDS,
            "mip_rel_gap": 0.0,
            "presolve": True,
        },
    )
    elapsed = time.perf_counter() - started
    solver = {
        "name": "scipy.optimize.milp_highs",
        "scipy_version": scipy.__version__,
        "status": int(result.status),
        "success": bool(result.success),
        "message": str(result.message),
        "wall_seconds": elapsed,
        "time_limit_seconds": SOLVER_TIME_LIMIT_SECONDS,
        "objective": float(result.fun) if result.fun is not None else None,
        "mip_node_count": int(result.mip_node_count) if result.mip_node_count is not None else None,
        "mip_gap": float(result.mip_gap) if result.mip_gap is not None else None,
    }
    if not result.success or result.x is None:
        return [], [], solver
    rounded = np.rint(result.x).astype(int)
    if np.max(np.abs(result.x - rounded)) > 1e-6:
        raise ValueError("MILP returned a non-integral solution")
    filler = sorted(available[local] for local in np.flatnonzero(rounded[:count]))
    control = sorted(available[local] for local in np.flatnonzero(rounded[count:]))
    if (
        len(filler) != FILLER_ROWS
        or len(control) != CONTROL_ROWS
        or set(filler) & set(control)
        or tuple(b - a for a, b in zip(vector_sum(filler, replay_lengths), vector_sum(control, replay_lengths)))
        != restart_vector
    ):
        raise ValueError("MILP solution failed exact integer postvalidation")
    return filler, control, solver


def slot_order() -> list[int]:
    return sorted(
        range(ROWS_PER_ARM),
        key=lambda index: hashlib.sha256(
            f"{STREAM_ORDER_SEED}:stream-slot:{index}".encode()
        ).digest(),
    )


def render(lines: list[str]) -> bytes:
    if len(lines) != ROWS_PER_ARM:
        raise ValueError("training stream row count changed")
    order = slot_order()
    return ("\n".join(lines[index] for index in order) + "\n").encode()


def summarize(value: bytes) -> dict:
    rows = [json.loads(line) for line in value.decode().splitlines() if line]
    return {
        "rows": len(rows),
        "sha256": sha256_bytes(value),
        "kinds": dict(sorted(Counter(row.get("kind", "missing") for row in rows).items())),
        "families": dict(sorted(Counter(row.get("family", "missing") for row in rows).items())),
    }


def build() -> tuple[dict[Path, bytes], dict]:
    restarts = load_lines(RESTART, RESTART_SHA256, RESTART_ROWS)
    replay = load_lines(REPLAY, REPLAY_SHA256, 2240)
    restart_lengths, replay_lengths, token_receipt = load_lengths()
    core = load_common_core(replay_lengths)
    available = [index for index in range(len(replay)) if index not in set(core)]
    restart_indices = list(range(RESTART_ROWS))
    restart_vector = vector_sum(restart_indices, restart_lengths)
    filler, control, solver = solve_exact_match(available, replay_lengths, restart_vector)
    if not solver["success"]:
        manifest = {
            "schema_version": 1,
            "experiment_id": EXP.name,
            "stage": "exact_three_axis_exposure_feasibility",
            "outcome": "STOP_EXPOSURE_MATCH_INFEASIBLE",
            "solver": solver,
            "match_axes": list(MATCH_AXES),
            "restart_vector": dict(zip(MATCH_AXES, restart_vector)),
            "training_authorized": False,
            "benchmark_data_read": False,
            "aggregate_seed_open": False,
        }
        return {}, manifest

    core_lines = [replay[index][0] for index in core]
    control_bytes = render(core_lines + [replay[index][0] for index in control])
    candidate_bytes = render(
        core_lines + [line for line, _ in restarts] + [replay[index][0] for index in filler]
    )
    control_total = span_sum(core + control, replay_lengths)
    candidate_replay_total = span_sum(core + filler, replay_lengths)
    restart_total = span_sum(restart_indices, restart_lengths)
    candidate_total = {
        field: candidate_replay_total[field] + restart_total[field]
        for field in candidate_replay_total
    }
    deltas = {field: candidate_total[field] - control_total[field] for field in control_total}
    if any(deltas[axis] != 0 for axis in MATCH_AXES):
        raise ValueError(f"three-axis exposure match failed: {deltas}")
    control_lines = control_bytes.decode().splitlines()
    candidate_lines = candidate_bytes.decode().splitlines()
    aligned = [
        index
        for index, pair in enumerate(zip(control_lines, candidate_lines, strict=True))
        if pair[0] == pair[1]
    ]
    if len(aligned) != COMMON_ROWS:
        raise ValueError(f"expected exactly {COMMON_ROWS} aligned shared rows, got {len(aligned)}")
    manifest = {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "stage": "exact_three_axis_exposure_feasibility",
        "outcome": "PASS_EXPOSURE_MATCH",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "stream_order_seed": STREAM_ORDER_SEED,
        "match_axes": list(MATCH_AXES),
        "solver": solver,
        "sources": {
            "restart": {"path": RESTART.relative_to(EXP).as_posix(), "sha256": RESTART_SHA256, "rows": RESTART_ROWS},
            "replay": {"path": REPLAY.relative_to(EXP).as_posix(), "sha256": REPLAY_SHA256, "rows": len(replay)},
            "source_token_receipt": {"path": TOKENS.relative_to(EXP).as_posix(), "sha256": TOKENS_SHA256},
            "predecessor_stream_manifest": {"path": PREDECESSOR.relative_to(EXP).as_posix(), "sha256": PREDECESSOR_SHA256},
        },
        "selection": {
            "shared_replay_rows": len(core),
            "restart_rows": RESTART_ROWS,
            "candidate_replay_filler_rows": len(filler),
            "control_variable_replay_rows": len(control),
            "rows_per_arm": ROWS_PER_ARM,
            "shared_position_aligned_rows": len(aligned),
            "replay_core_source_indices": core,
            "candidate_replay_filler_source_indices": filler,
            "replay_control_source_indices": control,
        },
        "exposure": {
            "control": control_total,
            "candidate": candidate_total,
            "candidate_minus_control": deltas,
            "restart_block": restart_total,
            "candidate_replay_block": candidate_replay_total,
        },
        "training": {
            "authorized": False,
            "reason": "second adversarial compute review must be committed and green",
            "rows_per_arm": ROWS_PER_ARM,
            "optimizer_steps": 40,
            "batch_size": 1,
            "gradient_accumulation": 8,
            "epochs": 1,
            "learning_rate": 1e-5,
            "seed": 48,
            "max_length": 4096,
            "thought_weight": 0.2,
            "close_weight": 0.2,
        },
        "outputs": {
            CONTROL_OUT.name: summarize(control_bytes),
            CANDIDATE_OUT.name: summarize(candidate_bytes),
        },
        "tokenizer_receipt_totals": {
            name: value["totals"] for name, value in token_receipt["sources"].items()
        },
        "targets_modified_for_matching": False,
        "rows_duplicated_for_matching": False,
        "rows_truncated_for_matching": False,
        "training_authorized": False,
        "benchmark_data_read": False,
        "aggregate_seed_open": False,
    }
    return {CONTROL_OUT: control_bytes, CANDIDATE_OUT: candidate_bytes}, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        if not MANIFEST.is_file():
            parser.error("stream manifest is absent")
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if manifest.get("outcome") != "PASS_EXPOSURE_MATCH":
            parser.error("existing stream manifest is not a feasibility pass")
        for path in (CONTROL_OUT, CANDIDATE_OUT):
            expected = manifest["outputs"][path.name]["sha256"]
            if not path.is_file() or sha256_file(path) != expected:
                parser.error(f"materialized stream changed: {path}")
        print(json.dumps({"manifest": str(MANIFEST), "sha256": sha256_file(MANIFEST)}, indent=2))
        return 0
    if any(path.exists() for path in (MANIFEST, CONTROL_OUT, CANDIDATE_OUT)):
        parser.error("refusing to overwrite an exposure artifact")
    outputs, manifest = build()
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    MANIFEST.write_bytes(manifest_bytes)
    for path, value in outputs.items():
        path.write_bytes(value)
    print(json.dumps({
        "outcome": manifest["outcome"],
        "manifest": str(MANIFEST),
        "manifest_sha256": sha256_bytes(manifest_bytes),
        "solver": manifest["solver"],
        "outputs": manifest.get("outputs", {}),
        "match_deltas": manifest.get("exposure", {}).get("candidate_minus_control"),
    }, indent=2, sort_keys=True))
    return 0 if outputs else 2


if __name__ == "__main__":
    raise SystemExit(main())
