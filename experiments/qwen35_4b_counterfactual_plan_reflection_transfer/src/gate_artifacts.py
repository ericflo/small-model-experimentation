"""Exact, replayable gate artifacts used by staged execution authorization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from analyze import (
    evaluate_calibration,
    evaluate_positive_control,
    evaluate_retention,
    evaluate_seed_block,
)
from eval_inputs import task_metadata
from score_artifacts import validate_score_artifact


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_ref(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"gate source artifact does not exist: {resolved}")
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _path_from_ref(value: Any, label: str) -> Path:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"gate artifact has malformed {label} reference")
    path = Path(value["path"])
    if not path.is_absolute() or not path.is_file() or sha256_file(path) != value["sha256"]:
        raise ValueError(f"gate artifact {label} source is absent or changed")
    return path


def _producer(experiment_root: Path, script_name: str) -> dict[str, Any]:
    from stages import git_commit

    return {
        "script_sha256": sha256_file(experiment_root / "scripts" / script_name),
        "analysis_module_sha256": sha256_file(experiment_root / "src" / "analyze.py"),
        "gate_module_sha256": sha256_file(Path(__file__).resolve()),
        "score_module_sha256": sha256_file(experiment_root / "src" / "score_artifacts.py"),
        "git_commit": git_commit(),
    }


def _validated_score_rows(
    paths: list[Path],
    *,
    config: dict[str, Any],
    config_path: Path,
    experiment_root: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            raise ValueError("gate score artifact path is duplicated")
        seen.add(resolved)
        rows.extend(
            validate_score_artifact(
                resolved,
                config=config,
                config_path=config_path,
                experiment_root=experiment_root,
            )
        )
    return rows


def build_calibration_artifact(
    *,
    scores_path: Path,
    config: dict[str, Any],
    config_path: Path,
    experiment_root: Path,
) -> dict[str, Any]:
    rows = _validated_score_rows(
        [scores_path], config=config, config_path=config_path, experiment_root=experiment_root
    )
    if {row["split"] for row in rows} != {"calibration"}:
        raise ValueError("calibration score bundle has the wrong split")
    if any(
        row["arm"] != "frozen_action" or row.get("training_seed") is not None
        for row in rows
    ):
        raise ValueError("calibration must contain only receipt-bound frozen rows")
    return {
        "schema_version": 2,
        "experiment_id": config["experiment_id"],
        "config_sha256": sha256_file(config_path),
        "producer": _producer(experiment_root, "calibration_gate.py"),
        "invocation": {"scores": _artifact_ref(scores_path)},
        "gate": evaluate_calibration(
            rows,
            config["decision_gates"]["calibration_before_training"],
            task_metadata(config, "calibration"),
        ),
    }


def build_decision_artifact(
    *,
    block: str,
    seed: int,
    stage_receipt_path: Path,
    score_paths: list[Path],
    config: dict[str, Any],
    config_path: Path,
    experiment_root: Path,
) -> dict[str, Any]:
    if block not in {"qualification", "confirmation"}:
        raise ValueError("decision block is not preregistered")
    if seed not in set(config["training"]["staged_seeds"].values()):
        raise ValueError("decision seed is not preregistered")
    screen = int(config["training"]["staged_seeds"]["screen"])
    expected_stage = (
        "confirmation"
        if block == "confirmation"
        else ("screen_training" if seed == screen else "replication_training")
    )
    from stages import read_and_validate_stage_receipt

    read_and_validate_stage_receipt(
        stage_receipt_path,
        config=config,
        config_path=config_path,
        expected_stage=expected_stage,
    )
    rows = _validated_score_rows(
        score_paths, config=config, config_path=config_path, experiment_root=experiment_root
    )
    if {row["split"] for row in rows} != {block}:
        raise ValueError("score bundle contains the wrong evaluation block")
    for row in rows:
        expected_seed = None if row["arm"] == "frozen_action" else seed
        if row.get("training_seed") != expected_seed:
            raise ValueError("score bundle training seed differs from requested decision seed")
    decision = config["decision_gates"]
    thresholds = dict(decision["per_seed_qualification_and_confirmation"])
    thresholds["reflection_specific_correct_minus_auxiliary_min"] = decision[
        "reflection_specific_mechanism"
    ]["correct_minus_auxiliary_min"]
    expected_task_metadata = task_metadata(config, block)
    capability = evaluate_seed_block(
        rows, thresholds, decision["bootstrap"], expected_task_metadata
    )
    result: dict[str, Any] = {
        "schema_version": 2,
        "experiment_id": config["experiment_id"],
        "config_sha256": sha256_file(config_path),
        "producer": _producer(experiment_root, "analyze.py"),
        "invocation": {
            "block": block,
            "seed": seed,
            "stage_receipt": _artifact_ref(stage_receipt_path),
            "scores": [_artifact_ref(path) for path in score_paths],
        },
        "block": block,
        "seed": seed,
        "capability": capability,
    }
    positive_arm = "direct_plan_answer_positive_control_action"
    if seed == screen and any(row["arm"] == positive_arm for row in rows):
        result["positive_control"] = evaluate_positive_control(
            rows,
            decision["positive_control_sanity_on_qualification"],
            decision["bootstrap"],
            expected_task_metadata,
        )
    return result


def build_retention_artifact(
    *,
    arm: str,
    seed: int,
    stage_receipt_path: Path,
    score_paths: list[Path],
    config: dict[str, Any],
    config_path: Path,
    experiment_root: Path,
) -> dict[str, Any]:
    allowed = {
        "reflection_correct_action",
        "reflection_shuffled_action",
        "auxiliary_plan_label_correct_action",
    }
    if arm not in allowed:
        raise ValueError("retention arm is not preregistered")
    if seed not in set(config["training"]["staged_seeds"].values()):
        raise ValueError("retention seed is not preregistered")
    screen = int(config["training"]["staged_seeds"]["screen"])
    expected_stage = "screen_training" if seed == screen else "replication_training"
    from stages import read_and_validate_stage_receipt

    read_and_validate_stage_receipt(
        stage_receipt_path,
        config=config,
        config_path=config_path,
        expected_stage=expected_stage,
    )
    rows = _validated_score_rows(
        score_paths, config=config, config_path=config_path, experiment_root=experiment_root
    )
    if {row["split"] for row in rows} != {"retention"}:
        raise ValueError("retention score bundle has the wrong split")
    for row in rows:
        expected_seed = None if row["arm"] == "frozen_action" else seed
        if row.get("training_seed") != expected_seed:
            raise ValueError("retention score seed differs from requested adapter seed")
    thresholds = config["decision_gates"]["retention_noninferiority"]
    return {
        "schema_version": 2,
        "experiment_id": config["experiment_id"],
        "config_sha256": sha256_file(config_path),
        "producer": _producer(experiment_root, "retention_gate.py"),
        "invocation": {
            "arm": arm,
            "seed": seed,
            "stage_receipt": _artifact_ref(stage_receipt_path),
            "scores": [_artifact_ref(path) for path in score_paths],
        },
        "arm": arm,
        "seed": seed,
        "gate": evaluate_retention(
            rows,
            arm,
            depth_min=float(thresholds["each_depth_delta_min"]),
            family_min=float(thresholds["each_family_delta_min"]),
            expected_task_metadata=task_metadata(config, "retention"),
        ),
    }


def validate_gate_artifact(
    path: Path,
    *,
    kind: str,
    config: dict[str, Any],
    config_path: Path,
    experiment_root: Path,
) -> dict[str, Any]:
    if kind not in {"calibration_gate", "decision", "retention"}:
        raise ValueError("unknown gate artifact kind")
    observed = json.loads(path.read_text())
    if not isinstance(observed, dict) or observed.get("schema_version") != 2:
        raise ValueError("gate artifact lacks the exact replayable schema")
    invocation = observed.get("invocation")
    if not isinstance(invocation, dict):
        raise ValueError("gate artifact lacks an exact invocation")
    if kind == "calibration_gate":
        if set(invocation) != {"scores"}:
            raise ValueError("calibration gate invocation schema changed")
        expected = build_calibration_artifact(
            scores_path=_path_from_ref(invocation["scores"], "calibration scores"),
            config=config,
            config_path=config_path,
            experiment_root=experiment_root,
        )
    elif kind == "decision":
        if set(invocation) != {"block", "seed", "stage_receipt", "scores"}:
            raise ValueError("decision invocation schema changed")
        if not isinstance(invocation["scores"], list) or not invocation["scores"]:
            raise ValueError("decision lacks score artifact references")
        expected = build_decision_artifact(
            block=str(invocation["block"]),
            seed=int(invocation["seed"]),
            stage_receipt_path=_path_from_ref(invocation["stage_receipt"], "stage receipt"),
            score_paths=[
                _path_from_ref(value, "decision scores") for value in invocation["scores"]
            ],
            config=config,
            config_path=config_path,
            experiment_root=experiment_root,
        )
    else:
        if set(invocation) != {"arm", "seed", "stage_receipt", "scores"}:
            raise ValueError("retention invocation schema changed")
        if not isinstance(invocation["scores"], list) or not invocation["scores"]:
            raise ValueError("retention lacks score artifact references")
        expected = build_retention_artifact(
            arm=str(invocation["arm"]),
            seed=int(invocation["seed"]),
            stage_receipt_path=_path_from_ref(invocation["stage_receipt"], "stage receipt"),
            score_paths=[
                _path_from_ref(value, "retention scores") for value in invocation["scores"]
            ],
            config=config,
            config_path=config_path,
            experiment_root=experiment_root,
        )
    if observed != expected:
        raise ValueError("gate artifact differs from exact source-evidence recomputation")
    return observed
