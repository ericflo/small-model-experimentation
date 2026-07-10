#!/usr/bin/env python3
"""Fail-closed CPU audit for frozen task and semantic-oracle artifacts.

This gate independently recomputes every behavioral minimum-depth receipt,
checks task/oracle pairing and the hidden-label boundary, and verifies pairwise
behavioral disjointness among calibration, oracle-development, and primary task
functions on one experiment-owned common probe bank. It never loads a model and
never reads final hidden labels for oracle construction or scoring; hidden cases
are used only to verify that a serialized target pipeline generated its task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))
sys.path.insert(0, str(EXP / "scripts"))
import experiment_common as C  # noqa: E402
import families as F  # noqa: E402


COMMON_PROBE_SEED = 20_260_710
COMMON_PROBE_COUNT = 64
SPLIT_NAMES = ("calibration", "development", "primary")
COMMON_PROBE_PREFIX: tuple[tuple[int, ...], ...] = (
    (0, 1, 2, 3, 4),
    (4, 3, 2, 1, 0),
    (0, 0, 0, 0, 0),
    (1, 1, 2, 2, 3),
    (-9, 9, -9, 9, -9),
    (9, -8, 7, -6, 5, -4, 3, -2),
    (-3, -2, -1, 0, 1, 2, 3),
    (3, 3, 3, -3, -3, -3),
)


def _make_common_probe_bank() -> tuple[tuple[int, ...], ...]:
    probes = list(COMMON_PROBE_PREFIX)
    seen = set(probes)
    rng = random.Random(COMMON_PROBE_SEED)
    while len(probes) < COMMON_PROBE_COUNT:
        candidate = tuple(rng.randint(-9, 9) for _ in range(rng.randint(5, 8)))
        if candidate not in seen:
            seen.add(candidate)
            probes.append(candidate)
    return tuple(probes)


COMMON_PROBE_BANK = _make_common_probe_bank()
COMMON_PROBE_BANK_SHA256 = hashlib.sha256(
    json.dumps(COMMON_PROBE_BANK, separators=(",", ":")).encode("utf-8")
).hexdigest()


def _error(code: str, message: str, **context: Any) -> dict[str, Any]:
    return {"code": code, "message": message, **context}


def _behavior_vector(pipeline: F.Pipeline) -> tuple[tuple[int, ...] | None, ...]:
    rows: list[tuple[int, ...] | None] = []
    for probe in COMMON_PROBE_BANK:
        output = F.execute_pipeline(pipeline, probe)
        rows.append(None if output is None else tuple(output))
    return tuple(rows)


def _behavior_digest(vector: Sequence[Sequence[int] | None]) -> str:
    return hashlib.sha256(
        json.dumps(vector, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _audit_task(payload: tuple[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Worker-safe exact audit of one serialized task."""

    split_name, task = payload
    task_id = str(task.get("task_id", "<missing>"))
    errors: list[dict[str, Any]] = []
    try:
        depth = int(task["depth"])
        pipeline = F.normalize_pipeline(task["target_pipeline"])
    except Exception as exc:
        return {
            "task_id": task_id,
            "split": split_name,
            "errors": [
                _error(
                    "invalid_task_pipeline",
                    f"could not parse task depth/target pipeline: {exc}",
                    split=split_name,
                    task_id=task_id,
                )
            ],
        }

    if task.get("schema_version") != 1:
        errors.append(
            _error("task_schema", "task schema_version must be 1", split=split_name, task_id=task_id)
        )
    if task.get("split") != split_name:
        errors.append(
            _error(
                "task_split",
                f"task split is {task.get('split')!r}, expected {split_name!r}",
                split=split_name,
                task_id=task_id,
            )
        )
    if len(pipeline) != depth:
        errors.append(
            _error(
                "target_depth",
                f"target pipeline length {len(pipeline)} != task depth {depth}",
                split=split_name,
                task_id=task_id,
            )
        )
    target_skeleton = [name for name, _parameter in pipeline]
    if task.get("target_skeleton") != target_skeleton:
        errors.append(
            _error(
                "target_skeleton",
                "target_skeleton does not match target_pipeline",
                split=split_name,
                task_id=task_id,
            )
        )

    split_inputs: dict[str, set[tuple[int, ...]]] = {}
    all_inputs: list[list[int]] = []
    all_outputs: list[list[int]] = []
    for case_split in ("visible", "label_probe", "hidden"):
        rows = task.get(case_split)
        if not isinstance(rows, list) or not rows:
            errors.append(
                _error(
                    "missing_case_split",
                    f"{case_split} must be a non-empty list",
                    split=split_name,
                    task_id=task_id,
                )
            )
            split_inputs[case_split] = set()
            continue
        keys: set[tuple[int, ...]] = set()
        for case in rows:
            try:
                value = list(case["input"])
                expected = list(case["output"])
                actual = F.execute_pipeline(pipeline, value)
            except Exception as exc:
                errors.append(
                    _error(
                        "invalid_case",
                        f"could not execute {case_split} case: {exc}",
                        split=split_name,
                        task_id=task_id,
                    )
                )
                continue
            if actual != expected:
                errors.append(
                    _error(
                        "target_case_mismatch",
                        f"target pipeline does not reproduce a {case_split} label",
                        split=split_name,
                        task_id=task_id,
                    )
                )
            key = tuple(value)
            if key in keys:
                errors.append(
                    _error(
                        "duplicate_case_input",
                        f"duplicate input inside {case_split}",
                        split=split_name,
                        task_id=task_id,
                    )
                )
            keys.add(key)
            all_inputs.append(value)
            all_outputs.append(expected)
        split_inputs[case_split] = keys
    for left_index, left in enumerate(("visible", "label_probe", "hidden")):
        for right in ("visible", "label_probe", "hidden")[left_index + 1 :]:
            overlap = split_inputs[left] & split_inputs[right]
            if overlap:
                errors.append(
                    _error(
                        "case_split_overlap",
                        f"{len(overlap)} inputs occur in both {left} and {right}",
                        split=split_name,
                        task_id=task_id,
                    )
                )

    stored = task.get("min_depth_audit")
    if not isinstance(stored, Mapping):
        errors.append(
            _error(
                "missing_depth_receipt",
                "min_depth_audit is missing or malformed",
                split=split_name,
                task_id=task_id,
            )
        )
        stored = {}
    expected_levels = list(range(1, depth))
    receipt_expectations = {
        "algorithm": "uncapped_behavioral_bfs",
        "seen_cap": None,
        "max_depth": depth - 1,
        "found_depth": None,
        "representative_pipeline": None,
        "within_limit": False,
        "exhaustive_decision": True,
        "levels_fully_exhausted": expected_levels,
    }
    for field, expected in receipt_expectations.items():
        if stored.get(field) != expected:
            errors.append(
                _error(
                    "depth_receipt_field",
                    f"min_depth_audit.{field}={stored.get(field)!r}, expected {expected!r}",
                    split=split_name,
                    task_id=task_id,
                    field=field,
                )
            )

    recomputed: dict[str, Any] | None = None
    if all_inputs and len(all_inputs) == len(all_outputs):
        recomputed = F.exhaustive_min_depth_leq(all_inputs, all_outputs, depth - 1).to_dict()
        if dict(stored) != recomputed:
            differing = sorted(
                key
                for key in set(stored) | set(recomputed)
                if stored.get(key) != recomputed.get(key)
            )
            errors.append(
                _error(
                    "depth_receipt_recompute_mismatch",
                    f"stored receipt differs from exact recomputation at fields {differing}",
                    split=split_name,
                    task_id=task_id,
                    differing_fields=differing,
                )
            )

    try:
        prompt_boundary_ok = C.prompt_boundary_audit(task, target_skeleton[:1])
    except Exception as exc:
        prompt_boundary_ok = False
        errors.append(
            _error(
                "prompt_boundary_error",
                f"could not render the whitelisted model record: {exc}",
                split=split_name,
                task_id=task_id,
            )
        )
    if not prompt_boundary_ok:
        errors.append(
            _error(
                "prompt_boundary",
                "oracle-field deletion changed the whitelisted model record",
                split=split_name,
                task_id=task_id,
            )
        )

    vector = _behavior_vector(pipeline)
    return {
        "task_id": task_id,
        "split": split_name,
        "depth": depth,
        "common_behavior_signature_sha256": _behavior_digest(vector),
        "undefined_common_probe_outputs": sum(row is None for row in vector),
        "behavior_vector": vector,  # Removed before writing; retained for exact collision checks.
        "exact_receipt_recomputed": recomputed is not None,
        "recomputed_transitions_considered": int(
            recomputed.get("transitions_considered", 0) if recomputed else 0
        ),
        "recomputed_case_operation_applications": int(
            recomputed.get("case_operation_applications", 0) if recomputed else 0
        ),
        "errors": errors,
    }


def _validate_oracle_pair(
    split_name: str, task: Mapping[str, Any], oracle: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    task_id = str(task.get("task_id", "<missing>"))
    errors: list[dict[str, Any]] = []
    if oracle.get("schema_version") != 1:
        errors.append(_error("oracle_schema", "oracle schema_version must be 1", split=split_name, task_id=task_id))
    if str(oracle.get("task_id")) != task_id:
        errors.append(
            _error("oracle_task_id", "oracle task_id does not match task", split=split_name, task_id=task_id)
        )
    try:
        oracle_depth_matches = int(oracle.get("depth", -1)) == int(task.get("depth", -2))
    except (TypeError, ValueError):
        oracle_depth_matches = False
    if not oracle_depth_matches:
        errors.append(_error("oracle_depth", "oracle depth does not match task", split=split_name, task_id=task_id))
    if oracle.get("label_source_splits") != ["visible", "label_probe"]:
        errors.append(
            _error(
                "oracle_label_splits",
                "oracle labels must use exactly visible + label_probe",
                split=split_name,
                task_id=task_id,
            )
        )
    if oracle.get("hidden_cases_used_for_labels") is not False:
        errors.append(
            _error(
                "oracle_hidden_boundary",
                "hidden_cases_used_for_labels must be the boolean false",
                split=split_name,
                task_id=task_id,
            )
        )
    unexpected_hidden_keys = sorted(
        key
        for key in oracle
        if "hidden" in str(key).lower() and key != "hidden_cases_used_for_labels"
    )
    if unexpected_hidden_keys:
        errors.append(
            _error(
                "oracle_hidden_fields",
                f"oracle contains unexpected hidden-named fields: {unexpected_hidden_keys}",
                split=split_name,
                task_id=task_id,
            )
        )

    successes = oracle.get("successful_skeletons")
    if not isinstance(successes, list) or not successes:
        errors.append(
            _error("oracle_successes", "oracle successful_skeletons must be non-empty", split=split_name, task_id=task_id)
        )
        successes = []
    if oracle.get("successful_skeleton_count") != len(successes):
        errors.append(
            _error("oracle_skeleton_count", "successful_skeleton_count is inconsistent", split=split_name, task_id=task_id)
        )
    parameter_fill_count = 0
    semantic_rows_checked = 0
    try:
        if not isinstance(task.get("visible"), list) or not task.get("visible"):
            raise ValueError("visible cases are missing")
        if not isinstance(task.get("label_probe"), list) or not task.get("label_probe"):
            raise ValueError("label_probe cases are missing")
        inputs, outputs = F.task_cases(task, ("visible", "label_probe"))
        depth = int(task.get("depth", 0))
        case_bank_valid = True
    except Exception as exc:
        inputs, outputs, depth, case_bank_valid = [], [], 0, False
        errors.append(
            _error(
                "oracle_case_bank",
                f"could not build the visible + label_probe case bank: {exc}",
                split=split_name,
                task_id=task_id,
            )
        )
    for index, success in enumerate(successes):
        try:
            skeleton = tuple(success["skeleton"])
            pipeline = F.normalize_pipeline(success["representative_pipeline"])
            fill_count = int(success["parameter_fill_count"])
        except Exception as exc:
            errors.append(
                _error(
                    "oracle_success_row",
                    f"malformed successful skeleton row {index}: {exc}",
                    split=split_name,
                    task_id=task_id,
                )
            )
            continue
        parameter_fill_count += fill_count
        semantic_rows_checked += 1
        if fill_count < 1:
            errors.append(_error("oracle_fill_count", "parameter_fill_count must be positive", split=split_name, task_id=task_id))
        if len(skeleton) != depth or tuple(name for name, _parameter in pipeline) != skeleton:
            errors.append(_error("oracle_skeleton_shape", "successful skeleton/pipeline shape mismatch", split=split_name, task_id=task_id))
        if case_bank_valid and not F.pipeline_solves(pipeline, inputs, outputs):
            errors.append(
                _error(
                    "oracle_semantics",
                    "representative pipeline does not solve visible + label_probe",
                    split=split_name,
                    task_id=task_id,
                )
            )
    if oracle.get("successful_parameter_fill_count") != parameter_fill_count:
        errors.append(
            _error("oracle_parameter_count", "successful_parameter_fill_count is inconsistent", split=split_name, task_id=task_id)
        )
    accounting = oracle.get("accounting")
    if not isinstance(accounting, Mapping):
        errors.append(_error("oracle_accounting", "oracle accounting receipt is missing", split=split_name, task_id=task_id))
    else:
        if accounting.get("successful_type_skeletons") != len(successes):
            errors.append(_error("oracle_accounting_skeletons", "accounting successful_type_skeletons mismatch", split=split_name, task_id=task_id))
        if accounting.get("successful_concrete_pipelines") != parameter_fill_count:
            errors.append(_error("oracle_accounting_fills", "accounting successful_concrete_pipelines mismatch", split=split_name, task_id=task_id))
    return {"semantic_success_rows_checked": semantic_rows_checked}, errors


def _duplicates(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


def audit_dataset(
    calibration_tasks: Sequence[Mapping[str, Any]],
    calibration_oracles: Sequence[Mapping[str, Any]],
    development_tasks: Sequence[Mapping[str, Any]],
    development_oracles: Sequence[Mapping[str, Any]],
    primary_tasks: Sequence[Mapping[str, Any]],
    primary_oracles: Sequence[Mapping[str, Any]],
    *,
    workers: int = 1,
    smoke: bool = False,
) -> dict[str, Any]:
    """Audit already-loaded artifacts; useful both to the CLI and unit tests."""

    if workers < 1:
        raise ValueError("workers must be positive")
    task_sets = {
        "calibration": list(calibration_tasks),
        "development": list(development_tasks),
        "primary": list(primary_tasks),
    }
    oracle_sets = {
        "calibration": list(calibration_oracles),
        "development": list(development_oracles),
        "primary": list(primary_oracles),
    }
    payloads = [
        (split_name, task)
        for split_name in SPLIT_NAMES
        for task in task_sets[split_name]
    ]
    if workers == 1:
        task_reports = [_audit_task(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            task_reports = list(executor.map(_audit_task, payloads))

    reports_by_key = {
        (str(report.get("split")), str(report.get("task_id"))): report
        for report in task_reports
    }
    errors: list[dict[str, Any]] = []
    split_reports: dict[str, dict[str, Any]] = {}
    oracle_rows_checked = 0
    for split_name in SPLIT_NAMES:
        tasks = task_sets[split_name]
        oracles = oracle_sets[split_name]
        task_ids = [str(task.get("task_id", "<missing>")) for task in tasks]
        oracle_ids = [str(row.get("task_id", "<missing>")) for row in oracles]
        duplicate_task_ids = _duplicates(task_ids)
        duplicate_oracle_ids = _duplicates(oracle_ids)
        if duplicate_task_ids:
            errors.append(_error("duplicate_task_ids", f"duplicate task IDs: {duplicate_task_ids}", split=split_name))
        if duplicate_oracle_ids:
            errors.append(_error("duplicate_oracle_ids", f"duplicate oracle IDs: {duplicate_oracle_ids}", split=split_name))
        if task_ids != oracle_ids:
            errors.append(
                _error(
                    "task_oracle_id_order",
                    "task and oracle IDs differ in membership or serialized order",
                    split=split_name,
                    task_only=sorted(set(task_ids) - set(oracle_ids)),
                    oracle_only=sorted(set(oracle_ids) - set(task_ids)),
                )
            )
        oracle_by_id = {str(row.get("task_id")): row for row in oracles}
        semantic_rows = 0
        for task in tasks:
            task_id = str(task.get("task_id", "<missing>"))
            report = reports_by_key.get((split_name, task_id))
            if report:
                errors.extend(report.get("errors", []))
            oracle = oracle_by_id.get(task_id)
            if oracle is None:
                continue
            pair_report, pair_errors = _validate_oracle_pair(split_name, task, oracle)
            semantic_rows += pair_report["semantic_success_rows_checked"]
            errors.extend(pair_errors)
            oracle_rows_checked += 1
        split_reports[split_name] = {
            "task_count": len(tasks),
            "oracle_count": len(oracles),
            "task_oracle_ids_equal_and_ordered": task_ids == oracle_ids,
            "duplicate_task_ids": duplicate_task_ids,
            "duplicate_oracle_ids": duplicate_oracle_ids,
            "semantic_success_rows_checked": semantic_rows,
        }

    signature_groups: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    vectors_by_digest: dict[str, tuple[tuple[int, ...] | None, ...]] = {}
    public_task_reports: list[dict[str, Any]] = []
    for report in task_reports:
        vector = report.pop("behavior_vector", None)
        digest = report.get("common_behavior_signature_sha256")
        if vector is not None and isinstance(digest, str):
            previous = vectors_by_digest.setdefault(digest, vector)
            if previous != vector:
                errors.append(_error("behavior_hash_collision", "different common behaviors share a SHA-256 digest"))
            signature_groups[digest][str(report["split"])].append(str(report["task_id"]))
        public_task_reports.append(report)

    within_split_duplicates: list[dict[str, Any]] = []
    cross_split_collisions: list[dict[str, Any]] = []
    for digest in sorted(signature_groups):
        groups = signature_groups[digest]
        for split_name in SPLIT_NAMES:
            ids = sorted(groups.get(split_name, []))
            if len(ids) > 1:
                within_split_duplicates.append(
                    {"signature_sha256": digest, "split": split_name, "task_ids": ids}
                )
        ids_by_split = {
            split_name: sorted(groups.get(split_name, []))
            for split_name in SPLIT_NAMES
            if groups.get(split_name)
        }
        if len(ids_by_split) > 1:
            cross_split_collisions.append(
                {
                    "signature_sha256": digest,
                    "task_ids_by_split": ids_by_split,
                }
            )
    if within_split_duplicates:
        errors.append(
            _error(
                "within_split_behavior_duplicates",
                f"found {len(within_split_duplicates)} duplicate common-probe behaviors",
            )
        )
    if cross_split_collisions:
        errors.append(
            _error(
                "cross_split_behavior_collision",
                f"found {len(cross_split_collisions)} pairwise cross-split behavior collisions",
            )
        )

    exact_reports = [row for row in public_task_reports if row.get("exact_receipt_recomputed")]
    result = {
        "schema_version": 1,
        "audit": "task_oracle_data_integrity",
        "smoke": bool(smoke),
        "passed": not errors,
        "common_probe_bank": {
            "seed": COMMON_PROBE_SEED,
            "count": len(COMMON_PROBE_BANK),
            "sha256": COMMON_PROBE_BANK_SHA256,
            "structured_prefix_count": len(COMMON_PROBE_PREFIX),
        },
        "splits": split_reports,
        "exact_depth_receipts": {
            "recomputed": len(exact_reports),
            "expected": len(task_reports),
            "transitions_considered": sum(
                int(row.get("recomputed_transitions_considered", 0)) for row in exact_reports
            ),
            "case_operation_applications": sum(
                int(row.get("recomputed_case_operation_applications", 0)) for row in exact_reports
            ),
            "seen_cap": None,
        },
        "task_oracle_pairs_checked": oracle_rows_checked,
        "hidden_label_boundary": {
            "required_label_source_splits": ["visible", "label_probe"],
            "hidden_cases_used_for_oracle_labels": False,
        },
        "behavioral_disjointness": {
            "definition": "target-pipeline outputs on the frozen common probe bank",
            "unique_signatures": len(signature_groups),
            "within_split_duplicate_groups": within_split_duplicates,
            "cross_split_collision_groups": cross_split_collisions,
        },
        "task_rows": public_task_reports,
        "errors": errors,
    }
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    suffix = "_smoke" if args.smoke else ""
    paths = {
        f"{split}_{kind}": EXP / "data" / f"{split}_{kind}{suffix}.jsonl"
        for split in SPLIT_NAMES
        for kind in ("tasks", "oracle")
    }
    missing = [str(path.relative_to(EXP)) for path in paths.values() if not path.exists()]
    if missing:
        result: dict[str, Any] = {
            "schema_version": 1,
            "audit": "task_oracle_data_integrity",
            "smoke": bool(args.smoke),
            "passed": False,
            "errors": [_error("missing_artifacts", f"missing required data files: {missing}")],
        }
    else:
        result = audit_dataset(
            C.load_jsonl(paths["calibration_tasks"]),
            C.load_jsonl(paths["calibration_oracle"]),
            C.load_jsonl(paths["development_tasks"]),
            C.load_jsonl(paths["development_oracle"]),
            C.load_jsonl(paths["primary_tasks"]),
            C.load_jsonl(paths["primary_oracle"]),
            workers=args.workers,
            smoke=args.smoke,
        )
        result["source_artifacts"] = {
            str(path.relative_to(EXP)): {"sha256": _sha256_file(path)}
            for path in paths.values()
        }
    output = EXP / "runs" / f"data_audit{suffix}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "output": str(output),
                "errors": len(result.get("errors", [])),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
