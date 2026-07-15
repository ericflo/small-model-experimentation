#!/usr/bin/env python3
"""Solve and materialize exact three-axis axis-v2 arm training streams."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, vstack


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
TREATMENT = EXP / "data" / "sft_axis_v2.jsonl"
REPLAY = EXP / "data" / "sft_blend.jsonl"
TOKENS = EXP / "data" / "source_token_lengths.json"
MANIFEST = EXP / "data" / "stream_manifest.json"
CONTROL_OUT = EXP / "data" / "replay_repeat3.jsonl"
CANDIDATE_OUT = EXP / "data" / "axis_v2.jsonl"
TREATMENT_SHA256 = "28d9be20180b017e64eab4749d79eb659089b2bcc12985efbb753f4a66479e79"
REPLAY_SHA256 = "25a9595f2e70e4d5cab0a730f0e2613d314843f2a5dfe96187bc30d5d2abf0c2"
# TODO-PIN filled by the orchestrator from the published source token receipt.
TOKENS_SHA256 = "5a88c4ea9b0999ce35cbbc552c0dfff67d4d4f398e323117d56c484a61633765"
WARM_START_ADAPTER = (
    ROOT
    / "large_artifacts"
    / "qwen35_4b_axis_replay_stack_medium_target_match"
    / "adapters"
    / "axis_on_replay"
)
WARM_START_ADAPTER_SHA256 = "87cdebde17d6151f440dd5c8fe28abc69ff074036c889eb1e4732775a76f3801"
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
CORE_ROWS = 1280
TREATMENT_ROWS = 160
FILLER_ROWS = 80
CONTROL_ROWS = 240
VARIABLE_BLOCK_ROWS = 240
ROWS_PER_ARM = 1520
STREAM_ORDER_SEED = 55120
MATCH_AXES = ("forward", "nonzero_target", "absolute_loss_mass_x5")
FIELDS = (
    "forward",
    "prompt",
    "parent_prefix",
    "masked_context",
    "think_target",
    "close_target",
    "answer_target",
    "target_span",
    "nonzero_target",
    "absolute_loss_mass_x5",
)
SOLVER_TIME_LIMIT_SECONDS = 600.0


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_lines(path: Path, expected_sha256: str, expected_rows: int) -> list[tuple[str, dict]]:
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise ValueError(f"frozen source changed: {path}")
    rows = [
        (line, json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(rows) != expected_rows or not all(isinstance(row, dict) for _, row in rows):
        raise ValueError(f"unexpected source rows: {path}")
    return rows


def load_lengths() -> tuple[dict[str, list[dict[str, int]]], dict]:
    if TOKENS_SHA256 is None:
        raise ValueError("source token receipt pin is unfilled (TODO-PIN)")
    if not TOKENS.is_file() or sha256_file(TOKENS) != TOKENS_SHA256:
        raise ValueError("source token receipt changed")
    payload = json.loads(TOKENS.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != 1
        or payload.get("experiment_id") != EXP.name
        or payload.get("model_id") != MODEL_ID
        or payload.get("model_revision") != MODEL_REVISION
        or payload.get("max_length") != 4096
        or payload.get("match_axes") != list(MATCH_AXES)
    ):
        raise ValueError("source token receipt identity changed")
    lengths: dict[str, list[dict[str, int]]] = {}
    for name, expected_path, expected_sha, expected_rows in (
        ("axis", TREATMENT, TREATMENT_SHA256, TREATMENT_ROWS),
        ("replay", REPLAY, REPLAY_SHA256, 2240),
    ):
        source = payload.get("sources", {}).get(name, {})
        items = source.get("lengths", [])
        if (
            source.get("path") != expected_path.relative_to(EXP).as_posix()
            or source.get("sha256") != expected_sha
            or source.get("rows") != expected_rows
            or len(items) != expected_rows
            or not all(set(item) == set(FIELDS) for item in items)
        ):
            raise ValueError(f"source token receipt lineage changed for {name}")
        lengths[name] = items
    return lengths, payload


def span_sum(indices: list[int], lengths: list[dict[str, int]]) -> dict[str, int]:
    return {field: sum(lengths[index][field] for index in indices) for field in FIELDS}


def vector_sum(indices: list[int], lengths: list[dict[str, int]]) -> tuple[int, ...]:
    return tuple(sum(lengths[index][axis] for index in indices) for axis in MATCH_AXES)


def add_spans(*spans: dict[str, int]) -> dict[str, int]:
    return {field: sum(span[field] for span in spans) for field in FIELDS}


def allocation(group_sizes: dict[tuple[str, ...], int], total: int) -> dict[tuple[str, ...], int]:
    population = sum(group_sizes.values())
    if not 0 <= total <= population:
        raise ValueError("invalid stratified target")
    quotas = {key: size * total // population for key, size in group_sizes.items()}
    order = sorted(
        group_sizes,
        key=lambda key: (-(group_sizes[key] * total % population), key),
    )
    for key in order[: total - sum(quotas.values())]:
        quotas[key] += 1
    return quotas


def deterministic_rank(namespace: str, index: int, line: str) -> bytes:
    return hashlib.sha256(
        f"{STREAM_ORDER_SEED}:{namespace}:{index}:".encode() + line.encode()
    ).digest()


def select_stratified(
    rows: list[tuple[str, dict]],
    indices: list[int],
    total: int,
    group_fn: Callable[[dict], tuple[str, ...]],
    namespace: str,
) -> list[int]:
    groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index in indices:
        groups[group_fn(rows[index][1])].append(index)
    quotas = allocation({key: len(value) for key, value in groups.items()}, total)
    selected = []
    for key, members in sorted(groups.items()):
        ranked = sorted(
            members,
            key=lambda index: deterministic_rank(namespace, index, rows[index][0]),
        )
        selected.extend(ranked[: quotas[key]])
    return sorted(selected)


def solve_exact_match(
    available: list[int],
    replay_lengths: list[dict[str, int]],
    treatment_vector: tuple[int, ...],
) -> tuple[list[int], list[int], dict]:
    """Jointly pick disjoint 80/240 replay subsets with exact block equality.

    Variable layout: x[role * count + local] for roles
    (0=filler_candidate, 1=control) over `available` replay rows.
    Equality: control - filler_candidate == axis_v2_160 on every match axis,
    which makes the two 240-row variable blocks identical on those axes.
    """
    count = len(available)
    role_counts = (FILLER_ROWS, CONTROL_ROWS)
    variables = count * 2
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    lower: list[float] = []
    upper: list[float] = []

    # Exact per-role cardinalities.
    for role, cardinality in enumerate(role_counts):
        for local in range(count):
            rows.append(role)
            cols.append(role * count + local)
            data.append(1.0)
        lower.append(float(cardinality))
        upper.append(float(cardinality))

    # control - filler = fixed axis-arm exposure on all three axes.
    row_offset = len(role_counts)
    for axis_offset, axis in enumerate(MATCH_AXES):
        constraint_row = row_offset
        row_offset += 1
        for local, source_index in enumerate(available):
            value = float(replay_lengths[source_index][axis])
            rows.extend((constraint_row, constraint_row))
            cols.extend((local, count + local))
            data.extend((-value, value))
        lower.append(float(treatment_vector[axis_offset]))
        upper.append(float(treatment_vector[axis_offset]))

    equality = coo_matrix((data, (rows, cols)), shape=(row_offset, variables)).tocsr()
    # A replay row cannot appear in more than one variable block.
    disjoint_rows = np.repeat(np.arange(count), 2)
    disjoint_cols = np.column_stack(
        (np.arange(count), count + np.arange(count))
    ).ravel()
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
        "time_limit_seconds": SOLVER_TIME_LIMIT_SECONDS,
        "mip_rel_gap_limit": 0.0,
    }
    print(json.dumps({"solver_wall_seconds": elapsed}, sort_keys=True))
    if not result.success or result.x is None:
        return [], [], solver
    rounded = np.rint(result.x).astype(int)
    if np.max(np.abs(result.x - rounded)) > 1e-6:
        raise ValueError("MILP returned a non-integral solution")
    filler = sorted(available[local] for local in np.flatnonzero(rounded[:count]))
    control = sorted(available[local] for local in np.flatnonzero(rounded[count:]))
    control_vector = vector_sum(control, replay_lengths)
    if (
        len(filler) != FILLER_ROWS
        or len(control) != CONTROL_ROWS
        or set(filler) & set(control)
        or tuple(
            control_axis - filler_axis
            for filler_axis, control_axis in zip(vector_sum(filler, replay_lengths), control_vector)
        )
        != treatment_vector
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


def training_section() -> dict:
    warm_start_file = WARM_START_ADAPTER / "adapter_model.safetensors"
    if not warm_start_file.is_file() or sha256_file(warm_start_file) != WARM_START_ADAPTER_SHA256:
        raise ValueError("warm-start parent adapter changed")
    return {
        "authorized": False,
        "reason": "streams must pass independent validation before any GPU step",
        "arms": {
            "control": CONTROL_OUT.name,
            "candidate_axis_v2": CANDIDATE_OUT.name,
        },
        "rows_per_arm": ROWS_PER_ARM,
        "optimizer_steps": ROWS_PER_ARM // 8,
        "batch_size": 1,
        "gradient_accumulation": 8,
        "epochs": 1,
        "learning_rate": 1e-5,
        "lora_rank": 32,
        "lora_alpha": 64,
        "seed": 54,
        "max_length": 4096,
        "thought_weight": 0.2,
        "close_weight": 0.2,
        "zero_skipped_rows_required": True,
        "warm_start_adapter": {
            "arm": "axis_on_replay",
            "path": WARM_START_ADAPTER.relative_to(ROOT).as_posix(),
            "adapter_model_sha256": WARM_START_ADAPTER_SHA256,
        },
    }


def build() -> tuple[dict[Path, bytes], dict]:
    treatment = load_lines(TREATMENT, TREATMENT_SHA256, TREATMENT_ROWS)
    replay = load_lines(REPLAY, REPLAY_SHA256, 2240)
    lengths, token_receipt = load_lengths()
    treatment_lengths = lengths["axis"]
    replay_lengths = lengths["replay"]

    core = select_stratified(
        replay,
        list(range(len(replay))),
        CORE_ROWS,
        lambda row: (str(row.get("family")), str(row.get("kind"))),
        "replay-core",
    )
    core_set = set(core)
    available = [index for index in range(len(replay)) if index not in core_set]
    if len(core) != CORE_ROWS or len(available) != len(replay) - CORE_ROWS:
        raise ValueError("shared replay core selection changed size")

    treatment_indices = list(range(TREATMENT_ROWS))
    treatment_vector = vector_sum(treatment_indices, treatment_lengths)
    filler, control, solver = solve_exact_match(
        available, replay_lengths, treatment_vector
    )
    if not solver["success"]:
        manifest = {
            "schema_version": 1,
            "experiment_id": EXP.name,
            "stage": "exact_three_axis_exposure_feasibility",
            "outcome": "STOP_EXPOSURE_MATCH_INFEASIBLE",
            "solver": solver,
            "match_axes": list(MATCH_AXES),
            "treatment_vector": dict(zip(MATCH_AXES, treatment_vector)),
            "training_authorized": False,
            "benchmark_data_read": False,
            "aggregate_seed_open": False,
        }
        return {}, manifest

    core_lines = [replay[index][0] for index in core]
    control_bytes = render(core_lines + [replay[index][0] for index in control])
    candidate_bytes = render(
        core_lines
        + [line for line, _ in treatment]
        + [replay[index][0] for index in filler]
    )

    core_total = span_sum(core, replay_lengths)
    control_block = span_sum(control, replay_lengths)
    candidate_block = add_spans(
        span_sum(treatment_indices, treatment_lengths),
        span_sum(filler, replay_lengths),
    )
    arm_totals = {
        "replay_repeat3": add_spans(core_total, control_block),
        "axis_v2": add_spans(core_total, candidate_block),
    }
    deltas = {
        "axis_v2_minus_replay_repeat3": {
            field: arm_totals["axis_v2"][field] - arm_totals["replay_repeat3"][field]
            for field in FIELDS
        },
    }
    if any(
        pair_deltas[axis] != 0 for pair_deltas in deltas.values() for axis in MATCH_AXES
    ):
        raise ValueError(f"three-axis exposure match failed: {deltas}")

    stream_lines = {
        path: value.decode().splitlines()
        for path, value in (
            (CONTROL_OUT, control_bytes),
            (CANDIDATE_OUT, candidate_bytes),
        )
    }
    aligned = [
        index
        for index, lines in enumerate(zip(*stream_lines.values(), strict=True))
        if len(set(lines)) == 1
    ]
    if len(aligned) != CORE_ROWS:
        raise ValueError(f"expected exactly {CORE_ROWS} aligned shared rows, got {len(aligned)}")

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
            "axis": {
                "path": TREATMENT.relative_to(EXP).as_posix(),
                "sha256": TREATMENT_SHA256,
                "rows": TREATMENT_ROWS,
            },
            "replay": {
                "path": REPLAY.relative_to(EXP).as_posix(),
                "sha256": REPLAY_SHA256,
                "rows": len(replay),
            },
            "source_token_receipt": {
                "path": TOKENS.relative_to(EXP).as_posix(),
                "sha256": TOKENS_SHA256,
            },
        },
        "selection": {
            "shared_replay_rows": len(core),
            "treatment_rows": TREATMENT_ROWS,
            "candidate_replay_filler_rows": len(filler),
            "control_variable_replay_rows": len(control),
            "variable_block_rows": VARIABLE_BLOCK_ROWS,
            "rows_per_arm": ROWS_PER_ARM,
            "shared_position_aligned_rows": len(aligned),
            "core_stratification": "family_kind_largest_remainder",
            "replay_core_source_indices": core,
            "candidate_replay_filler_source_indices": filler,
            "replay_control_source_indices": control,
        },
        "exposure": {
            "shared_core": core_total,
            "blocks": {
                "control": control_block,
                "axis_v2": candidate_block,
            },
            "arms": arm_totals,
            "deltas": deltas,
        },
        "training": training_section(),
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
    return {
        CONTROL_OUT: control_bytes,
        CANDIDATE_OUT: candidate_bytes,
    }, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    outputs, manifest = build()
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    targets: dict[Path, bytes] = {MANIFEST: manifest_bytes, **outputs}
    if args.check:
        if manifest["outcome"] != "PASS_EXPOSURE_MATCH":
            parser.error("regenerated stream manifest is not a feasibility pass")
        for path, expected in targets.items():
            if not path.is_file() or path.read_bytes() != expected:
                parser.error(f"materialized stream artifact is absent or changed: {path}")
    else:
        conflicts = [path for path in targets if path.exists()]
        if conflicts:
            parser.error(f"refusing to overwrite an exposure artifact: {conflicts[0]}")
        for path, value in targets.items():
            path.write_bytes(value)
    print(json.dumps({
        "outcome": manifest["outcome"],
        "manifest": str(MANIFEST),
        "manifest_sha256": sha256_bytes(manifest_bytes),
        "solver": manifest["solver"],
        "outputs": {
            name: value["sha256"] for name, value in manifest.get("outputs", {}).items()
        },
        "block_axis_sums": {
            name: {axis: block[axis] for axis in MATCH_AXES}
            for name, block in manifest.get("exposure", {}).get("blocks", {}).items()
        },
        "arm_forward_tokens": {
            name: arm["forward"]
            for name, arm in manifest.get("exposure", {}).get("arms", {}).items()
        },
    }, indent=2, sort_keys=True))
    return 0 if outputs else 2


if __name__ == "__main__":
    raise SystemExit(main())
