"""Crossed paired analysis and fail-closed verdict assignment."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import config_sha256, is_confirmatory_config, source_contract_sha256
from .mechanics import (
    crossed_paired_bootstrap_interval,
    gate_reachability,
    paired_bootstrap_interval,
    recurrent_compute_receipt,
)
from .receipt_contracts import validate_gate_lineage


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _checkpoint_identity(metadata: Mapping[str, Any]) -> str:
    return _canonical_sha256(
        {
            key: value
            for key, value in metadata.items()
            if key != "checkpoint_identity_sha256"
        }
    )


def _verify_receipt_identity(receipt: Mapping[str, Any], *, kind: str, path: Path) -> None:
    identity = receipt.get("receipt_identity_sha256")
    payload = {key: value for key, value in receipt.items() if key != "receipt_identity_sha256"}
    if not _is_sha256(identity) or identity != _canonical_sha256(payload):
        raise RuntimeError(f"{kind} receipt identity digest mismatch: {path}")


def _current_requirements_lock_sha256() -> str:
    path = Path(__file__).resolve().parents[3] / "requirements-training.lock.txt"
    if not path.is_file():
        raise RuntimeError(f"requirements training lock is missing: {path}")
    return _sha256(path)


def _resolve_receipted_rows(
    summary_path: Path,
    path_value: Any,
    expected_hash: Any,
    *,
    kind: str,
) -> tuple[Path, list[dict[str, Any]]]:
    if not isinstance(path_value, str) or not path_value:
        raise RuntimeError(f"{kind} has no row-file receipt: {summary_path}")
    if not _is_sha256(expected_hash):
        raise RuntimeError(f"{kind} has no valid row hash: {summary_path}")
    rows_path = Path(path_value)
    if not rows_path.is_absolute():
        rows_path = summary_path.parent / rows_path
    if not rows_path.is_file():
        raise RuntimeError(f"{kind} row file is missing: {rows_path}")
    if _sha256(rows_path) != expected_hash:
        raise RuntimeError(f"{kind} row hash mismatch: {rows_path}")
    return rows_path, _read_jsonl(rows_path)


def _index_rows(bundle: Mapping[str, Any]) -> dict[tuple[str, int, str], dict[str, Any]]:
    indexed: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in bundle["rows"]:
        try:
            key = (str(row["id"]), int(row["k"]), str(row["split"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"malformed evaluation row in {bundle.get('summary_path', '<memory>')}"
            ) from exc
        if key in indexed:
            raise RuntimeError(
                f"duplicate evaluation row key {key} in "
                f"{bundle.get('summary_path', '<memory>')}"
            )
        indexed[key] = row
    return indexed


def _evaluation_bundles(
    runs_dir: Path,
    expected_config_sha256: str,
    config: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Load evaluation bundles only after verifying every scientific receipt."""
    bundles: list[dict[str, Any]] = []
    expected_source_contract = source_contract_sha256()
    expected_requirements_lock = _current_requirements_lock_sha256()
    seen_bundle_keys: set[tuple[bool, int, str, str]] = set()
    for summary_path in sorted(runs_dir.rglob("summary.json")):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if summary.get("status") != "EVALUATION_COMPLETE":
            continue
        if summary.get("config_sha256") != expected_config_sha256:
            continue
        _verify_receipt_identity(summary, kind="evaluation", path=summary_path)
        _, rows = _resolve_receipted_rows(
            summary_path,
            summary.get("row_file"),
            summary.get("row_file_sha256"),
            kind="evaluation",
        )
        metadata = summary.get("setup", {}).get("checkpoint_metadata")
        if not isinstance(metadata, Mapping):
            raise RuntimeError(f"evaluation has no checkpoint metadata: {summary_path}")
        for receipt_key in (
            "data_manifest_sha256",
            "source_contract_sha256",
            "requirements_training_lock_sha256",
        ):
            summary_value = summary.get(receipt_key)
            metadata_value = metadata.get(receipt_key)
            if not _is_sha256(summary_value) or summary_value != metadata_value:
                raise RuntimeError(
                    f"checkpoint/evaluation {receipt_key} mismatch: {summary_path}"
                )
            if (
                receipt_key == "source_contract_sha256"
                and summary_value != expected_source_contract
            ):
                raise RuntimeError(
                    f"evaluation source contract is stale: {summary_path}"
                )
            if (
                receipt_key == "requirements_training_lock_sha256"
                and summary_value != expected_requirements_lock
            ):
                raise RuntimeError(
                    f"evaluation requirements lock is stale: {summary_path}"
                )
        expected_checkpoint_fields = {
            "experiment_id": config["experiment_id"] if config is not None else None,
            "model_id": config["model"]["id"] if config is not None else None,
            "model_revision": config["model"]["revision"] if config is not None else None,
            "backend": config["model"]["backend"] if config is not None else None,
            "config_sha256": expected_config_sha256,
        }
        for key, expected_value in expected_checkpoint_fields.items():
            if expected_value is not None and metadata.get(key) != expected_value:
                raise RuntimeError(
                    f"evaluation checkpoint {key} mismatch: {summary_path}"
                )
        identity = metadata.get("checkpoint_identity_sha256")
        if not _is_sha256(identity) or identity != _checkpoint_identity(metadata):
            raise RuntimeError(f"checkpoint identity digest mismatch: {summary_path}")
        if summary.get("checkpoint_identity_sha256") != identity:
            raise RuntimeError(
                f"checkpoint/evaluation identity mismatch: {summary_path}"
            )
        validate_gate_lineage(
            metadata.get("gate_lineage"), checkpoint_phase=str(metadata.get("phase"))
        )
        try:
            train_seed = int(metadata["train_seed"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"evaluation has no valid train seed: {summary_path}") from exc
        train_arm = str(summary.get("train_arm"))
        eval_mode = str(summary.get("eval_mode"))
        if train_arm != metadata.get("train_arm"):
            raise RuntimeError(f"checkpoint/evaluation arm mismatch: {summary_path}")
        pilot = bool(summary.get("pilot", False))
        bundle_key = (pilot, train_seed, train_arm, eval_mode)
        if bundle_key in seen_bundle_keys:
            raise RuntimeError(f"duplicate evaluation bundle {bundle_key}")
        seen_bundle_keys.add(bundle_key)

        if config is not None:
            expected_phase = "pilot" if pilot else "full"
            expected_step = int(
                config["training"]["pilot_steps" if pilot else "train_steps"]
            )
            expected_seeds = (
                {int(config["training"]["pilot_seed"])}
                if pilot
                else set(map(int, config["training"]["train_seeds"]))
            )
            if metadata.get("phase") != expected_phase:
                raise RuntimeError(
                    f"checkpoint phase mismatch for {summary_path}: "
                    f"{metadata.get('phase')!r} != {expected_phase!r}"
                )
            if summary.get("phase") != expected_phase:
                raise RuntimeError(f"evaluation summary phase mismatch: {summary_path}")
            if metadata.get("pilot") is not pilot:
                raise RuntimeError(f"checkpoint pilot flag mismatch: {summary_path}")
            if int(metadata.get("step", -1)) != expected_step:
                raise RuntimeError(
                    f"checkpoint is not the registered final {expected_phase} step: "
                    f"{summary_path}"
                )
            if train_seed not in expected_seeds:
                raise RuntimeError(
                    f"unexpected {expected_phase} model seed {train_seed}: {summary_path}"
                )
            if int(summary.get("expected_seed", -1)) != train_seed:
                raise RuntimeError(f"evaluation expected-seed mismatch: {summary_path}")

        bundle = {
            "summary_path": summary_path,
            "summary": summary,
            "rows": rows,
            "checkpoint_metadata": dict(metadata),
            "checkpoint_identity_sha256": identity,
            "data_manifest_sha256": summary["data_manifest_sha256"],
            "source_contract_sha256": summary["source_contract_sha256"],
            "requirements_training_lock_sha256": summary[
                "requirements_training_lock_sha256"
            ],
            "train_seed": train_seed,
            "train_arm": train_arm,
            "eval_mode": eval_mode,
            "pilot": pilot,
        }
        _index_rows(bundle)
        bundles.append(bundle)

    for receipt_key in (
        "data_manifest_sha256",
        "source_contract_sha256",
        "requirements_training_lock_sha256",
    ):
        values = {bundle[receipt_key] for bundle in bundles}
        if len(values) > 1:
            raise RuntimeError(
                f"scientific evaluation bundles disagree on {receipt_key}"
            )
    return bundles


def _prefer_full_bundles(bundles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Never let retained pilot rows overwrite or pool with full evaluations."""
    full = [bundle for bundle in bundles if not bundle["pilot"]]
    return full if full else bundles


def _expected_seeds(config: Mapping[str, Any], pilot: bool) -> set[int]:
    return (
        {int(config["training"]["pilot_seed"])}
        if pilot
        else set(map(int, config["training"]["train_seeds"]))
    )


def _expected_depth_tasks(config: Mapping[str, Any], pilot: bool) -> int:
    total = int(
        config["substrate"][
            "pilot_examples_per_split" if pilot else "evaluation_examples_per_split"
        ]
    )
    depths = len(config["substrate"]["extrapolation_depths"])
    if total % depths:
        raise RuntimeError(
            f"registered item count {total} is not divisible across {depths} depths"
        )
    return total // depths


def _assert_exact_keys(
    left: Mapping[Any, Any], right: Mapping[Any, Any], *, comparison: str
) -> None:
    left_keys, right_keys = set(left), set(right)
    if left_keys != right_keys:
        missing = sorted(left_keys - right_keys, key=str)
        extra = sorted(right_keys - left_keys, key=str)
        raise RuntimeError(
            f"{comparison} requires exact paired key equality; "
            f"right missing={missing[:5]} right extra={extra[:5]}"
        )


def _assert_paired_row_contract(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    comparison: str,
    fields: Sequence[str] = (
        "depth",
        "family",
        "template",
        "query_kind",
        "correct_choice",
        "prompt_tokens",
        "layer_token_applications",
    ),
) -> None:
    for field in fields:
        if field not in left or field not in right or left[field] != right[field]:
            raise RuntimeError(
                f"{comparison} immutable row mismatch for {field}: "
                f"{left.get(field)!r} != {right.get(field)!r}"
            )


def _crossed_summary(
    matrix: Mapping[int, Mapping[str, float]],
    *,
    config: Mapping[str, Any],
    bootstrap_seed_offset: int,
    effect_name: str,
) -> dict[str, Any]:
    mean, lower, upper = crossed_paired_bootstrap_interval(
        matrix,
        resamples=int(config["evaluation"]["bootstrap_resamples"]),
        seed=int(config["evaluation"]["bootstrap_seed"]) + bootstrap_seed_offset,
    )
    first = next(iter(matrix.values()))
    return {
        effect_name: mean,
        "ci95": [lower, upper],
        "unique_tasks": len(first),
        "model_seeds": sorted(map(int, matrix)),
        "model_seed_count": len(matrix),
        "per_seed": {
            str(seed): sum(values.values()) / len(values)
            for seed, values in sorted(matrix.items())
        },
    }


def _paired_carry_bag(
    bundles: list[dict[str, Any]], config: Mapping[str, Any], *, pilot: bool = False
) -> dict[str, Any]:
    expected_seeds = _expected_seeds(config, pilot)
    by_seed: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for bundle in bundles:
        seed, arm, mode = bundle["train_seed"], bundle["train_arm"], bundle["eval_mode"]
        if arm not in {"carry", "bag"} or mode != arm:
            continue
        if seed not in expected_seeds:
            raise RuntimeError(f"unexpected Carry/Bag model seed {seed}")
        if arm in by_seed[seed]:
            raise RuntimeError(f"duplicate {arm} evaluation for seed {seed}")
        by_seed[seed][arm] = bundle

    split = "pilot_depth" if pilot else "depth_extrapolation"
    primary_depths = set(map(int, config["evaluation"]["primary_depths"]))
    differences_by_seed: dict[int, dict[str, float]] = {}
    per_depth_seed: dict[int, dict[int, dict[str, float]]] = defaultdict(dict)
    per_query_seed: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    interface_by_seed: dict[int, list[float]] = defaultdict(list)
    bag_accuracy_by_seed: dict[int, list[float]] = defaultdict(list)
    seed_receipts = []
    for seed, arms in sorted(by_seed.items()):
        if set(arms) != {"carry", "bag"}:
            continue
        carry_metadata = arms["carry"]["checkpoint_metadata"]
        bag_metadata = arms["bag"]["checkpoint_metadata"]
        allowed_k1_error = float(config["gates"]["k1_max_logit_abs_error"])
        for arm_name, bundle in arms.items():
            parity_error = bundle["summary"].get("checkpoint_k1_max_logit_abs_error")
            if (
                parity_error is None
                or not math.isfinite(float(parity_error))
                or float(parity_error) > allowed_k1_error
            ):
                raise RuntimeError(
                    f"missing or failed post-checkpoint K=1 parity for {arm_name} seed {seed}"
                )
        for receipt_key in (
            "data_manifest_sha256",
            "source_contract_sha256",
            "training_prompt_tokens",
            "training_layer_token_applications",
            "training_order_sha256",
        ):
            if (
                receipt_key not in carry_metadata
                or carry_metadata.get(receipt_key) != bag_metadata.get(receipt_key)
            ):
                raise RuntimeError(
                    f"Carry/Bag training receipt mismatch for seed {seed}: {receipt_key}"
                )
        carry_parameters = carry_metadata.get("trainable_parameters", {})
        bag_parameters = bag_metadata.get("trainable_parameters", {})
        for receipt_key in ("total", "names_sha256", "values_sha256"):
            if (
                receipt_key not in carry_parameters
                or carry_parameters.get(receipt_key) != bag_parameters.get(receipt_key)
            ):
                raise RuntimeError(
                    f"Carry/Bag initialization receipt mismatch for seed {seed}: "
                    f"{receipt_key}"
                )

        carry = _index_rows(arms["carry"])
        bag = _index_rows(arms["bag"])
        _assert_exact_keys(carry, bag, comparison=f"Carry/Bag seed {seed}")
        seed_differences: dict[str, float] = {}
        depth_values: dict[int, dict[str, float]] = defaultdict(dict)
        query_values: dict[str, dict[str, float]] = defaultdict(dict)
        for key in sorted(carry):
            c, b = carry[key], bag[key]
            _assert_paired_row_contract(
                c, b, comparison=f"Carry/Bag seed {seed}, task {c.get('id')}"
            )
            for receipt_key in ("prompt_tokens", "layer_token_applications"):
                if c.get(receipt_key) != b.get(receipt_key):
                    raise RuntimeError(
                        f"Carry/Bag evaluation-compute mismatch for seed {seed}, "
                        f"item {c['id']}: {receipt_key}"
                    )
            depth = int(c["depth"])
            if c["split"] != split or depth not in primary_depths or int(c["k"]) != depth:
                continue
            item_id = str(c["id"])
            if item_id in seed_differences:
                raise RuntimeError(f"duplicate primary task id {item_id} for seed {seed}")
            if c.get("query_kind") != b.get("query_kind") or c.get("query_kind") not in {
                "node",
                "checksum",
            }:
                raise RuntimeError(f"missing/mismatched query kind for task {item_id}")
            difference = float(bool(c["correct"])) - float(bool(b["correct"]))
            seed_differences[item_id] = difference
            depth_values[depth][item_id] = difference
            query_values[str(c["query_kind"])][item_id] = difference
            top_is_answer = c.get("full_top_is_answer")
            if top_is_answer is None:
                raise RuntimeError(f"Carry row has no full-vocabulary answer-mode receipt: {item_id}")
            interface_by_seed[seed].append(float(bool(top_is_answer)))
            bag_accuracy_by_seed[seed].append(float(bool(b["correct"])))
        if seed_differences:
            differences_by_seed[seed] = seed_differences
            for depth, values in depth_values.items():
                per_depth_seed[depth][seed] = values
            for query_kind, values in query_values.items():
                per_query_seed[query_kind][seed] = values
        seed_receipts.append(
            {
                "seed": seed,
                "unique_primary_tasks": len(seed_differences),
                "checkpoint_phase": carry_metadata.get("phase"),
                "training_compute_equal": True,
                "training_order_equal": True,
                "initialization_equal": True,
                "post_checkpoint_k1_parity": True,
            }
        )

    observed_seeds = set(differences_by_seed)
    if not differences_by_seed:
        return {
            "available": False,
            "reason": "no separately trained carry/bag matched-depth rows",
            "expected_model_seeds": sorted(expected_seeds),
            "observed_model_seeds": sorted(observed_seeds),
            "seed_receipts": seed_receipts,
        }
    overall = _crossed_summary(
        differences_by_seed,
        config=config,
        bootstrap_seed_offset=0,
        effect_name="carry_minus_bag",
    )
    depth_results: dict[str, Any] = {}
    for depth, matrix in sorted(per_depth_seed.items()):
        depth_results[str(depth)] = _crossed_summary(
            matrix,
            config=config,
            bootstrap_seed_offset=depth,
            effect_name="difference",
        )
    query_results: dict[str, Any] = {}
    for index, (query_kind, matrix) in enumerate(sorted(per_query_seed.items())):
        query_results[query_kind] = _crossed_summary(
            matrix,
            config=config,
            bootstrap_seed_offset=40 + index,
            effect_name="difference",
        )

    expected_per_depth = _expected_depth_tasks(config, pilot)
    complete_depths = 0
    for depth in primary_depths:
        matrix = per_depth_seed.get(depth, {})
        if set(matrix) != expected_seeds:
            continue
        first_ids = set(next(iter(matrix.values())))
        if len(first_ids) != expected_per_depth:
            continue
        if all(set(values) == first_ids for values in matrix.values()):
            complete_depths += 1
    query_kinds_complete = set(query_results) == {"node", "checksum"}
    query_kinds_positive = query_kinds_complete and all(
        query_results[kind]["difference"] > 0 for kind in ("node", "checksum")
    )
    carry_answer_mode_rate = sum(map(sum, interface_by_seed.values())) / sum(
        map(len, interface_by_seed.values())
    )
    answer_mode_per_seed = {
        str(seed): sum(values) / len(values)
        for seed, values in sorted(interface_by_seed.items())
    }
    bag_primary_accuracy = sum(map(sum, bag_accuracy_by_seed.values())) / sum(
        map(len, bag_accuracy_by_seed.values())
    )
    return {
        "available": True,
        **overall,
        "expected_model_seeds": sorted(expected_seeds),
        "seed_set_complete": observed_seeds == expected_seeds,
        "per_depth": depth_results,
        "positive_depths": sum(
            result["difference"] > 0 for result in depth_results.values()
        ),
        "expected_unique_tasks_per_primary_depth": expected_per_depth,
        "complete_primary_depths": complete_depths,
        "required_primary_depths": len(primary_depths),
        "per_query_kind": query_results,
        "query_kinds_complete": query_kinds_complete,
        "query_kinds_positive": query_kinds_positive,
        "carry_full_top_is_answer_rate": carry_answer_mode_rate,
        "carry_full_top_is_answer_rate_per_seed": answer_mode_per_seed,
        "bag_primary_accuracy": bag_primary_accuracy,
        "carry_answer_interface_valid": all(
            value >= float(config["gates"]["min_carry_answer_mode_rate"])
            for value in answer_mode_per_seed.values()
        ),
        "seed_receipts": seed_receipts,
    }


def _unseen_k_scaling(
    bundles: list[dict[str, Any]], config: Mapping[str, Any], *, pilot: bool = False
) -> dict[str, Any]:
    train_k = int(config["training"]["train_k"])
    split = "pilot_depth" if pilot else "depth_extrapolation"
    expected_seeds = _expected_seeds(config, pilot)
    gains_by_seed: dict[int, dict[str, float]] = {}
    seen_seeds: set[int] = set()
    for bundle in bundles:
        if bundle["train_arm"] != "carry" or bundle["eval_mode"] != "carry":
            continue
        seed = int(bundle["train_seed"])
        if seed not in expected_seeds:
            raise RuntimeError(f"unexpected Carry scaling seed {seed}")
        if seed in seen_seeds:
            raise RuntimeError(f"duplicate Carry scaling bundle for seed {seed}")
        seen_seeds.add(seed)
        rows_by_id: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
        for row in bundle["rows"]:
            if row["split"] != split:
                continue
            item_id, k = str(row["id"]), int(row["k"])
            if k in rows_by_id[item_id]:
                raise RuntimeError(f"duplicate K={k} scaling row for task {item_id}")
            rows_by_id[item_id][k] = row
        gains: dict[str, float] = {}
        for item_id, item in rows_by_id.items():
            depth = int(next(iter(item.values()))["depth"])
            if depth <= train_k:
                continue
            if train_k not in item or depth not in item:
                raise RuntimeError(
                    f"scaling task {item_id} lacks exact K={train_k}/K=depth pair"
                )
            gains[item_id] = float(bool(item[depth]["correct"])) - float(
                bool(item[train_k]["correct"])
            )
        if gains:
            gains_by_seed[seed] = gains
    if not gains_by_seed:
        return {
            "available": False,
            "reason": "no paired K=train_k versus K=depth carry rows",
        }
    result = _crossed_summary(
        gains_by_seed,
        config=config,
        bootstrap_seed_offset=100,
        effect_name="gain",
    )
    expected_tasks = int(
        config["substrate"][
            "pilot_examples_per_split" if pilot else "evaluation_examples_per_split"
        ]
    )
    return {
        "available": True,
        **result,
        "expected_unique_tasks": expected_tasks,
        "complete": set(gains_by_seed) == expected_seeds
        and result["unique_tasks"] == expected_tasks,
    }


def _state_sufficiency(
    bundles: list[dict[str, Any]], config: Mapping[str, Any], *, pilot: bool = False
) -> dict[str, Any]:
    split = "pilot_depth" if pilot else "depth_extrapolation"
    expected_seeds = _expected_seeds(config, pilot)
    primary_depths = set(map(int, config["evaluation"]["primary_depths"]))
    by_seed: dict[int, dict[str, tuple[float, float]]] = defaultdict(dict)
    for bundle in bundles:
        if bundle["train_arm"] != "carry" or bundle["eval_mode"] != "carry":
            continue
        seed = int(bundle["train_seed"])
        if seed not in expected_seeds:
            raise RuntimeError(f"unexpected state-sufficiency seed {seed}")
        for row in bundle["rows"]:
            depth = int(row["depth"])
            if row["split"] == split and depth in primary_depths and int(row["k"]) == depth:
                item_id = str(row["id"])
                if item_id in by_seed[seed]:
                    raise RuntimeError(f"duplicate state-sufficiency task {item_id}")
                by_seed[seed][item_id] = (
                    float(row["node_step_accuracy"]),
                    float(row["joint_step_accuracy"]),
                )
    if not by_seed:
        return {"available": False}
    task_ids = set(next(iter(by_seed.values())))
    if any(set(values) != task_ids for values in by_seed.values()):
        raise RuntimeError("state sufficiency requires common task ids across model seeds")
    node = sum(value[0] for values in by_seed.values() for value in values.values()) / (
        len(by_seed) * len(task_ids)
    )
    joint = sum(value[1] for values in by_seed.values() for value in values.values()) / (
        len(by_seed) * len(task_ids)
    )
    return {
        "available": True,
        "unique_tasks": len(task_ids),
        "model_seeds": sorted(by_seed),
        "node_step_accuracy": node,
        "joint_step_accuracy": joint,
        # Joint node+phase+checksum sufficiency is mandatory; node alone cannot pass.
        "passes": joint >= float(config["gates"]["min_state_joint_accuracy"]),
    }


def _joint_holdout_summary(
    bundles: list[dict[str, Any]], config: Mapping[str, Any], *, pilot: bool = False
) -> dict[str, Any]:
    split = "pilot_joint" if pilot else "joint_holdout"
    expected_seeds = _expected_seeds(config, pilot)
    by_seed: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for bundle in bundles:
        if bundle["train_arm"] not in {"carry", "bag"} or bundle["eval_mode"] != bundle["train_arm"]:
            continue
        seed = int(bundle["train_seed"])
        if seed not in expected_seeds:
            raise RuntimeError(f"unexpected joint-holdout seed {seed}")
        arm = str(bundle["train_arm"])
        if arm in by_seed[seed]:
            raise RuntimeError(f"duplicate joint-holdout {arm} bundle for seed {seed}")
        by_seed[seed][arm] = bundle
    matrix: dict[int, dict[str, float]] = {}
    per_depth: dict[int, dict[int, dict[str, float]]] = defaultdict(dict)
    for seed, arms in sorted(by_seed.items()):
        if set(arms) != {"carry", "bag"}:
            continue
        carry, bag = _index_rows(arms["carry"]), _index_rows(arms["bag"])
        _assert_exact_keys(carry, bag, comparison=f"joint holdout seed {seed}")
        values: dict[str, float] = {}
        depth_values: dict[int, dict[str, float]] = defaultdict(dict)
        for key in sorted(carry):
            c, b = carry[key], bag[key]
            _assert_paired_row_contract(
                c, b, comparison=f"joint holdout seed {seed}, task {c.get('id')}"
            )
            if c["split"] != split or int(c["k"]) != int(c["depth"]):
                continue
            for receipt_key in ("prompt_tokens", "layer_token_applications"):
                if c.get(receipt_key) != b.get(receipt_key):
                    raise RuntimeError(
                        f"joint-holdout compute mismatch for seed {seed}, task {c['id']}"
                    )
            item_id = str(c["id"])
            if item_id in values:
                raise RuntimeError(f"duplicate joint-holdout task {item_id}")
            difference = float(bool(c["correct"])) - float(bool(b["correct"]))
            values[item_id] = difference
            depth_values[int(c["depth"])][item_id] = difference
        if values:
            matrix[seed] = values
            for depth, depth_matrix in depth_values.items():
                per_depth[depth][seed] = depth_matrix
    if not matrix:
        return {"available": False}
    result = _crossed_summary(
        matrix,
        config=config,
        bootstrap_seed_offset=150,
        effect_name="carry_minus_bag",
    )
    expected_per_depth = (
        _expected_depth_tasks(config, True)
        if pilot
        else int(config["evaluation"]["holdout_items_per_depth"])
    )
    complete_depths = sum(
        set(seed_matrix) == expected_seeds
        and len(next(iter(seed_matrix.values()))) == expected_per_depth
        and all(
            set(values) == set(next(iter(seed_matrix.values())))
            for values in seed_matrix.values()
        )
        for seed_matrix in per_depth.values()
    )
    threshold = float(config["gates"]["min_joint_holdout_carry_minus_bag"])
    return {
        "available": True,
        **result,
        "expected_unique_tasks_per_depth": expected_per_depth,
        "complete_depths": complete_depths,
        "required_depths": len(config["evaluation"]["primary_depths"]),
        "seed_set_complete": set(matrix) == expected_seeds,
        "non_reversal_all_seeds": all(value >= 0 for value in result["per_seed"].values()),
        "passes": result["carry_minus_bag"] >= threshold
        and result["ci95"][0] > 0
        and all(value >= 0 for value in result["per_seed"].values()),
    }


def _swap_summary(
    bundles: list[dict[str, Any]], config: Mapping[str, Any], *, pilot: bool = False
) -> dict[str, Any]:
    expected_seeds = _expected_seeds(config, pilot)
    expected_pairs = int(
        config["substrate"][
            "pilot_counterfactual_pairs" if pilot else "counterfactual_pairs"
        ]
    )
    per_seed = []
    seen_seeds: set[int] = set()
    for bundle in bundles:
        if bundle["train_arm"] != "carry" or bundle["eval_mode"] != "carry":
            continue
        seed = int(bundle["train_seed"])
        if seed not in expected_seeds:
            raise RuntimeError(f"unexpected counterfactual-swap seed {seed}")
        if seed in seen_seeds:
            raise RuntimeError(f"duplicate swap bundle for seed {seed}")
        seen_seeds.add(seed)
        swap = bundle["summary"].get("counterfactual_swaps")
        if not isinstance(swap, Mapping):
            continue
        path_value = swap.get("counterfactual_swap_row_file")
        hash_value = swap.get("counterfactual_swap_row_file_sha256")
        # Also accept top-level receipts while the runner schema is migrated.
        if path_value is None:
            path_value = bundle["summary"].get("counterfactual_swap_row_file")
            hash_value = bundle["summary"].get("counterfactual_swap_row_file_sha256")
        _, rows = _resolve_receipted_rows(
            bundle["summary_path"], path_value, hash_value, kind="counterfactual swap"
        )
        indexed: dict[tuple[str, str], dict[str, Any]] = {}
        directions_by_pair: dict[str, set[str]] = defaultdict(set)
        differences_by_pair: dict[str, list[float]] = defaultdict(list)
        donor_over_recipient_by_pair: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            key = (str(row["pair_id"]), str(row["direction"]))
            if key in indexed:
                raise RuntimeError(f"duplicate counterfactual swap row {key}")
            indexed[key] = row
            directions_by_pair[key[0]].add(key[1])
            for required in (
                "baseline_prediction",
                "baseline_correct",
                "baseline_donor_follow",
                "baseline_recipient_correct",
                "donor_choice_in_recipient",
                "donor_follow",
                "recipient_preserve",
                "geometry_equal",
            ):
                if required not in row:
                    raise RuntimeError(f"swap row {key} lacks {required}")
            if row["geometry_equal"] is not True:
                raise RuntimeError(f"swap row {key} has unequal token/state geometry")
            baseline_donor_follow = (
                int(row["baseline_prediction"])
                == int(row["donor_choice_in_recipient"])
            )
            if bool(row["baseline_donor_follow"]) is not baseline_donor_follow:
                raise RuntimeError(f"swap row {key} baseline-donor receipt mismatch")
            if bool(row["baseline_recipient_correct"]) is not bool(
                row["baseline_correct"]
            ):
                raise RuntimeError(f"swap row {key} baseline-recipient receipt mismatch")
            differences_by_pair[key[0]].append(
                float(bool(row["donor_follow"]))
                - float(baseline_donor_follow)
            )
            donor_over_recipient_by_pair[key[0]].append(
                float(bool(row["donor_follow"]))
                - float(bool(row["recipient_preserve"]))
            )
        if len(rows) != 2 * expected_pairs or len(directions_by_pair) != expected_pairs:
            raise RuntimeError(
                f"swap seed {seed} has {len(rows)} directions across "
                f"{len(directions_by_pair)} pairs; expected {2 * expected_pairs} directions"
            )
        expected_directions = {"a_to_b", "b_to_a"}
        if any(directions != expected_directions for directions in directions_by_pair.values()):
            raise RuntimeError(f"swap seed {seed} is not bidirectional for every pair")
        if int(swap.get("directions", -1)) != 2 * expected_pairs:
            raise RuntimeError(f"swap summary direction count mismatch for seed {seed}")
        pair_differences = [
            sum(values) / len(values)
            for _, values in sorted(differences_by_pair.items())
        ]
        pair_donor_over_recipient = [
            sum(values) / len(values)
            for _, values in sorted(donor_over_recipient_by_pair.items())
        ]
        mean, lower, upper = paired_bootstrap_interval(
            pair_differences,
            resamples=int(config["evaluation"]["bootstrap_resamples"]),
            seed=int(config["evaluation"]["bootstrap_seed"]) + 300 + seed,
        )
        donor_recipient_mean = sum(pair_donor_over_recipient) / len(
            pair_donor_over_recipient
        )
        raw_rates = {
            "baseline_donor_follow_rate": sum(
                int(row["baseline_prediction"])
                == int(row["donor_choice_in_recipient"])
                for row in rows
            )
            / len(rows),
            "donor_follow_rate": sum(bool(row["donor_follow"]) for row in rows)
            / len(rows),
            "recipient_preserve_rate": sum(
                bool(row["recipient_preserve"]) for row in rows
            )
            / len(rows),
        }
        for summary_key, recomputed in raw_rates.items():
            if not math.isclose(
                float(swap.get(summary_key, float("nan"))),
                recomputed,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise RuntimeError(
                    f"swap summary/raw mismatch for seed {seed}: {summary_key}"
                )
        per_seed.append(
            {
                "train_seed": seed,
                "unique_pairs": expected_pairs,
                "directions": len(rows),
                "bootstrap_unit": "counterfactual_pair_mean_over_two_directions",
                "baseline_donor_follow_rate": raw_rates[
                    "baseline_donor_follow_rate"
                ],
                "post_swap_donor_follow_rate": raw_rates["donor_follow_rate"],
                "post_swap_recipient_preserve_rate": raw_rates[
                    "recipient_preserve_rate"
                ],
                "donor_follow_gain": mean,
                "donor_follow_minus_recipient_preserve": donor_recipient_mean,
                "ci95": [lower, upper],
            }
        )
    threshold = float(config["gates"]["min_donor_follow_gain"])
    complete = {entry["train_seed"] for entry in per_seed} == expected_seeds
    return {
        "available": bool(per_seed),
        "model_seeds": sorted(entry["train_seed"] for entry in per_seed),
        "expected_model_seeds": sorted(expected_seeds),
        "complete": complete,
        "seeds": per_seed,
        "passes": complete
        and all(
            entry["donor_follow_gain"] >= threshold
            and entry["donor_follow_minus_recipient_preserve"] >= threshold
            and entry["ci95"][0] > 0
            for entry in per_seed
        ),
    }


def _parse_raw_sample_choice(text: str) -> int | None:
    if "</think>" not in text:
        return None
    visible = text.rsplit("</think>", 1)[1]
    matches = re.findall(
        r"(?:Answer\s*:\s*)?\b([ABCD])\b", visible, flags=re.IGNORECASE
    )
    return "ABCD".index(matches[-1].upper()) if matches else None


def _edge_cut_summary(
    bundles: list[dict[str, Any]], config: Mapping[str, Any], *, pilot: bool = False
) -> dict[str, Any]:
    """Pair each Carry checkpoint with itself after cutting the carried edge."""
    expected_seeds = _expected_seeds(config, pilot)
    split = "pilot_depth" if pilot else "depth_extrapolation"
    by_seed: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for bundle in bundles:
        if bundle["train_arm"] != "carry" or bundle["eval_mode"] not in {"carry", "bag"}:
            continue
        seed = int(bundle["train_seed"])
        if seed not in expected_seeds:
            raise RuntimeError(f"unexpected edge-cut seed {seed}")
        mode = str(bundle["eval_mode"])
        if mode in by_seed[seed]:
            raise RuntimeError(f"duplicate Carry-checkpoint {mode} evaluation for seed {seed}")
        by_seed[seed][mode] = bundle

    primary_depths = set(map(int, config["evaluation"]["primary_depths"]))
    matrix: dict[int, dict[str, float]] = {}
    per_depth: dict[int, dict[int, dict[str, float]]] = defaultdict(dict)
    identities: dict[int, str] = {}
    for seed, modes in sorted(by_seed.items()):
        if set(modes) != {"carry", "bag"}:
            continue
        intact_bundle, cut_bundle = modes["carry"], modes["bag"]
        if (
            intact_bundle["checkpoint_identity_sha256"]
            != cut_bundle["checkpoint_identity_sha256"]
        ):
            raise RuntimeError(f"edge cut seed {seed} did not use the same checkpoint")
        identities[seed] = intact_bundle["checkpoint_identity_sha256"]
        intact_all, cut_all = _index_rows(intact_bundle), _index_rows(cut_bundle)
        intact = {
            key: row
            for key, row in intact_all.items()
            if row["split"] == split
            and int(row["depth"]) in primary_depths
            and int(row["k"]) == int(row["depth"])
        }
        cut = {
            key: row
            for key, row in cut_all.items()
            if row["split"] == split
            and int(row["depth"]) in primary_depths
            and int(row["k"]) == int(row["depth"])
        }
        _assert_exact_keys(intact, cut, comparison=f"edge cut seed {seed}")
        values: dict[str, float] = {}
        depth_values: dict[int, dict[str, float]] = defaultdict(dict)
        for key in sorted(intact):
            intact_row, cut_row = intact[key], cut[key]
            _assert_paired_row_contract(
                intact_row,
                cut_row,
                comparison=f"edge cut seed {seed}, task {intact_row.get('id')}",
            )
            depth = int(intact_row["depth"])
            if intact_row["split"] != split or depth not in primary_depths or int(intact_row["k"]) != depth:
                continue
            for receipt_key in ("prompt_tokens", "layer_token_applications"):
                if intact_row.get(receipt_key) != cut_row.get(receipt_key):
                    raise RuntimeError(
                        f"edge-cut compute mismatch for seed {seed}, task {intact_row['id']}"
                    )
            item_id = str(intact_row["id"])
            if item_id in values:
                raise RuntimeError(f"duplicate edge-cut task {item_id}")
            difference = float(bool(intact_row["correct"])) - float(
                bool(cut_row["correct"])
            )
            values[item_id] = difference
            depth_values[depth][item_id] = difference
        if values:
            matrix[seed] = values
            for depth, depth_matrix in depth_values.items():
                per_depth[depth][seed] = depth_matrix
    if not matrix:
        return {"available": False, "model_seeds": []}
    result = _crossed_summary(
        matrix,
        config=config,
        bootstrap_seed_offset=250,
        effect_name="intact_minus_edge_cut",
    )
    expected_per_depth = _expected_depth_tasks(config, pilot)
    complete_depths = sum(
        set(seed_matrix) == expected_seeds
        and len(next(iter(seed_matrix.values()))) == expected_per_depth
        and all(
            set(values) == set(next(iter(seed_matrix.values())))
            for values in seed_matrix.values()
        )
        for seed_matrix in per_depth.values()
    )
    threshold = float(config["gates"]["min_edge_cut_gain"])
    complete = (
        set(matrix) == expected_seeds
        and complete_depths == len(primary_depths)
    )
    return {
        "available": True,
        **result,
        "checkpoint_identities": {str(seed): identity for seed, identity in identities.items()},
        "expected_unique_tasks_per_depth": expected_per_depth,
        "complete_primary_depths": complete_depths,
        "required_primary_depths": len(primary_depths),
        "complete": complete,
        "positive_all_seeds": all(value > 0 for value in result["per_seed"].values()),
        "passes": complete
        and all(value > 0 for value in result["per_seed"].values())
        and result["ci95"][0] > threshold,
    }


def _write_curve_csv(path: Path, bundles: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, str, str, int, int], list[float]] = defaultdict(list)
    for bundle in bundles:
        for row in bundle["rows"]:
            groups[
                (
                    str(bundle["train_arm"]),
                    str(bundle["eval_mode"]),
                    str(row["split"]),
                    int(row["depth"]),
                    int(row["k"]),
                )
            ].append(float(bool(row["correct"])))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("train_arm", "eval_mode", "split", "depth", "k", "n", "accuracy"),
            lineterminator="\n",
        )
        writer.writeheader()
        for (train_arm, eval_mode, split, depth, k), values in sorted(groups.items()):
            writer.writerow(
                {
                    "train_arm": train_arm,
                    "eval_mode": eval_mode,
                    "split": split,
                    "depth": depth,
                    "k": k,
                    "n": len(values),
                    "accuracy": sum(values) / len(values),
                }
            )


def analyze_runs(
    config: Mapping[str, Any], runs_dir: Path, output: Path
) -> dict[str, Any]:
    expected_config = config_sha256(config)
    if not is_confirmatory_config(config):
        summary = {
            "schema_version": 2,
            "experiment_id": config["experiment_id"],
            "config_sha256": expected_config,
            "source_contract_sha256": source_contract_sha256(),
            "requirements_training_lock_sha256": _current_requirements_lock_sha256(),
            "data_manifest_sha256": None,
            "phase": "setup",
            "verdict": "NONCONFIRMATORY_SMOKE_ONLY",
            "warning": (
                "Reduced/smoke configurations may test mechanics and data plumbing but "
                "cannot emit pilot promotion or scientific evidence."
            ),
        }
        summary["receipt_identity_sha256"] = _canonical_sha256(summary)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _write_curve_csv(output.parent / "k_depth_curves.csv", [])
        return summary
    bundles = _prefer_full_bundles(
        _evaluation_bundles(runs_dir, expected_config, config)
    )
    pilot = bool(bundles) and all(bundle["pilot"] for bundle in bundles)
    carry_bag = _paired_carry_bag(bundles, config, pilot=pilot)
    scaling = _unseen_k_scaling(bundles, config, pilot=pilot)
    state_sufficiency = _state_sufficiency(bundles, config, pilot=pilot)
    joint_holdout = _joint_holdout_summary(bundles, config, pilot=pilot)
    swaps = _swap_summary(bundles, config, pilot=pilot)
    edge_cut = _edge_cut_summary(bundles, config, pilot=pilot)
    manifest_hash = next((bundle["data_manifest_sha256"] for bundle in bundles), None)
    source_hash = next(
        (bundle["source_contract_sha256"] for bundle in bundles),
        source_contract_sha256(),
    )
    requirements_hash = next(
        (bundle["requirements_training_lock_sha256"] for bundle in bundles),
        _current_requirements_lock_sha256(),
    )
    sample_more = {
        "available": False,
        "reason": "G4 is explicitly deferred; this successor resolves adaptation capacity through G3",
    }
    deployment = {
        "available": False,
        "reason": "no deployment claim is licensed by this capacity-only successor",
    }
    gate = config["gates"]

    pilot_gate: dict[str, Any] | None = None
    if pilot:
        reachability = (
            gate_reachability(
                float(carry_bag["bag_primary_accuracy"]),
                float(gate["min_carry_minus_bag"]),
            )
            if carry_bag.get("available")
            else None
        )
        checks = {
            "complete_registered_cells": bool(
                carry_bag.get("seed_set_complete")
                and carry_bag.get("complete_primary_depths")
                == carry_bag.get("required_primary_depths")
            ),
            "k4_diagnostic_complete": bool(
                scaling.get("available") and scaling.get("complete")
            ),
            "joint_holdout_diagnostic_complete": bool(
                joint_holdout.get("available")
                and joint_holdout.get("seed_set_complete")
                and joint_holdout.get("complete_depths")
                == joint_holdout.get("required_depths")
            ),
            "swap_diagnostic_complete": bool(
                swaps.get("available") and swaps.get("complete")
            ),
            "positive_carry_minus_bag": bool(
                carry_bag.get("available") and carry_bag.get("carry_minus_bag", 0) > 0
            ),
            "joint_state_sufficient": bool(
                state_sufficiency.get("available") and state_sufficiency.get("passes")
            ),
            "query_kinds_positive": bool(carry_bag.get("query_kinds_positive")),
            "answer_interface_valid": bool(carry_bag.get("carry_answer_interface_valid")),
            "gate_reachable": bool(reachability and reachability["reachable"]),
        }
        complete = all(
            checks[key]
            for key in (
                "complete_registered_cells",
                "k4_diagnostic_complete",
                "joint_holdout_diagnostic_complete",
                "swap_diagnostic_complete",
            )
        )
        promote = complete and all(checks.values())
        if not complete:
            pilot_status = "PILOT_INCOMPLETE"
            capacity_conclusion = "not_licensed"
        elif not checks["gate_reachable"]:
            pilot_status = "PILOT_PROMOTION_BLOCKED"
            capacity_conclusion = "not_licensed"
        elif not checks["joint_state_sufficient"]:
            pilot_status = "PILOT_STATE_FORMATION_MISS"
            capacity_conclusion = "full_rank_joint_state_formation_failed"
        elif promote:
            pilot_status = "PILOT_PROMOTION_READY"
            capacity_conclusion = "full_rank_joint_state_formed"
        else:
            pilot_status = "PILOT_PROMOTION_BLOCKED"
            capacity_conclusion = "full_rank_joint_state_formed_but_mechanism_gate_failed"
        pilot_gate = {
            "status": pilot_status,
            "complete": complete,
            "promote": promote,
            "capacity_branch_closed": pilot_status == "PILOT_STATE_FORMATION_MISS",
            "capacity_conclusion": capacity_conclusion,
            "expected_model_seed": int(config["training"]["pilot_seed"]),
            "reachability": reachability,
            "failure_reason": (
                "GATE_INFEASIBLE"
                if reachability is not None and not reachability["reachable"]
                else None
            ),
            "checks": checks,
        }
        verdict = pilot_status
    elif not carry_bag["available"]:
        verdict = "SETUP_ONLY"
    elif (
        not carry_bag.get("seed_set_complete")
        or carry_bag["complete_primary_depths"] < carry_bag["required_primary_depths"]
    ):
        verdict = "UNDER_REPLICATED"
    elif (
        carry_bag["carry_minus_bag"] < float(gate["min_carry_minus_bag"])
        or carry_bag["ci95"][0] <= 0
    ):
        verdict = "NO_SERIAL_STATE_ADVANTAGE"
    elif not scaling["available"] or not scaling.get("complete") or scaling["gain"] <= 0 or scaling["ci95"][0] <= 0:
        verdict = "TRAINED_UNROLLING_ONLY"
    elif not state_sufficiency["available"] or not state_sufficiency["passes"]:
        verdict = "SERIAL_BUT_STATE_NOT_SUFFICIENT"
    elif (
        carry_bag["positive_depths"] < int(gate["min_positive_primary_depths"])
        or not carry_bag["query_kinds_positive"]
        or not joint_holdout.get("passes", False)
        or joint_holdout.get("complete_depths") != joint_holdout.get("required_depths")
    ):
        verdict = "DEPTH_NOT_ROBUST"
    elif not edge_cut.get("passes", False) or not swaps.get("passes", False):
        verdict = "DEEP_BUT_NOT_CAUSALLY_IDENTIFIED"
    else:
        verdict = "FULLRANK_CAUSAL_DEPTH_POSITIVE"

    summary = {
        "schema_version": 2,
        "experiment_id": config["experiment_id"],
        "config_sha256": expected_config,
        "source_contract_sha256": source_hash,
        "requirements_training_lock_sha256": requirements_hash,
        "data_manifest_sha256": manifest_hash,
        "phase": "pilot" if pilot else ("full" if bundles else "setup"),
        "verdict": verdict,
        "pilot_gate": pilot_gate,
        "evaluation_bundles": len(bundles),
        "carry_vs_bag": carry_bag,
        "unseen_k_scaling": scaling,
        "joint_holdout_carry_vs_bag": joint_holdout,
        "state_sufficiency": state_sufficiency,
        "trained_checkpoint_edge_cut": edge_cut,
        "counterfactual_swaps": swaps,
        "sample_more": sample_more,
        "deployment_comparison": deployment,
        "warning": (
            "This successor resolves the LoRA-capacity counterfactual through causal G3 only; "
            "it cannot license a deployment or sample-more claim."
        ),
    }
    summary["receipt_identity_sha256"] = _canonical_sha256(summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_curve_csv(output.parent / "k_depth_curves.csv", bundles)
    return summary
