"""Reconstruct and authenticate task-level score artifacts from raw generation files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from eval_inputs import task_metadata
from provenance import validate_action_inputs, validate_generation_protocol, validate_sampling
from scoring import score_generation_rows
from vllm_runner import SamplingConfig


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_ref(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"score source artifact does not exist: {resolved}")
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def jsonl_payload(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )


def _validate_adapter_gate(
    *,
    gate_path: Path,
    metadata: dict[str, Any],
    config: dict[str, Any],
    config_path: Path,
    arm: str,
    training_seed: int,
    runtime_protocol_sha256: str,
) -> None:
    from adapter_gate_artifacts import validate_adapter_gate_artifact

    gate = validate_adapter_gate_artifact(
        gate_path,
        config=config,
        config_path=config_path,
        experiment_root=config_path.parents[1],
    )
    training_arm = {
        "reflection_correct_action": "reflection_correct",
        "reflection_shuffled_action": "reflection_shuffled",
        "auxiliary_plan_label_correct_action": "auxiliary_plan_label_correct",
        "direct_plan_answer_positive_control_action": "direct_plan_answer_positive_control",
    }[arm]
    if (
        gate.get("schema_version") != 3
        or gate.get("pass") is not True
        or gate.get("experiment_id") != config["experiment_id"]
        or gate.get("config_sha256") != sha256_file(config_path)
        or gate.get("arm") != training_arm
        or gate.get("seed") != training_seed
        or gate.get("model_override") != metadata["model_override"]
        or gate.get("runtime_protocol_sha256") != runtime_protocol_sha256
        or int(gate.get("changed_tasks", 0)) < 1
        or int(gate.get("total_tasks", -1))
        != len(task_metadata(config, "calibration"))
    ):
        raise ValueError("adapter ON/OFF gate does not bind this arm/seed/merged model")


def build_score_rows(
    *,
    config: dict[str, Any],
    config_path: Path,
    experiment_root: Path,
    generated_path: Path,
    metadata_path: Path,
    input_receipt_path: Path,
    labels_path: Path,
    arm: str,
    training_seed: int | None,
    adapter_gate_receipt_path: Path | None,
) -> list[dict[str, Any]]:
    """Re-run the complete score contract and return its exact canonical rows."""
    evaluation = config["evaluation"]
    if arm not in set(evaluation["arms"]):
        raise ValueError("arm is not preregistered")
    metadata = json.loads(metadata_path.read_text())
    input_receipt = json.loads(input_receipt_path.read_text())
    split, expected_task_metadata, sealed = validate_action_inputs(
        config=config,
        config_path=config_path,
        receipt_path=input_receipt_path,
        labels_path=labels_path,
    )
    expected_seed = int(evaluation["sample_seeds"][split])
    validate_sampling(
        metadata,
        SamplingConfig(
            thinking="budget",
            thinking_budget=int(evaluation["thinking_budget"]),
            n=int(evaluation["primary_candidate_count"]),
            answer_max_tokens=int(evaluation["answer_max_tokens"]),
            temperature=float(evaluation["temperature"]),
            top_p=float(evaluation["top_p"]),
            top_k=int(evaluation["top_k"]),
            run_seed=expected_seed,
        ),
    )
    frozen = arm == "frozen_action"
    screen = int(config["training"]["staged_seeds"]["screen"])
    if split == "confirmation":
        expected_stage = "confirmation"
    elif split == "calibration" and frozen:
        expected_stage = "calibration_generation"
    elif frozen or training_seed == screen:
        expected_stage = "screen_training"
    else:
        expected_stage = "replication_training"
    runtime_protocol_sha256 = validate_generation_protocol(
        metadata=metadata,
        config=config,
        experiment_root=experiment_root,
        generated_path=generated_path,
        expected_rows=int(input_receipt["rows"]),
        expect_merged=not frozen,
        expected_stage=expected_stage,
        expected_split=split,
        expected_input_kind="action",
        expected_source_seed=None if frozen else training_seed,
    )
    if metadata["input"]["sha256"] != sealed["prompt_sha256"]:
        raise ValueError("generation input differs from sealed reconstruction")
    if frozen:
        if (
            training_seed is not None
            or metadata["model_override"] is not None
            or adapter_gate_receipt_path is not None
        ):
            raise ValueError("frozen arm must use the base model without a training seed")
    else:
        if training_seed not in set(config["training"]["staged_seeds"].values()):
            raise ValueError("adapter arm lacks a preregistered training seed")
        if metadata["model_override"] is None:
            raise ValueError("trained arm did not use a receipt-bound merged checkpoint")
        if adapter_gate_receipt_path is None:
            raise ValueError("trained arm lacks its adapter ON/OFF gate receipt")
        _validate_adapter_gate(
            gate_path=adapter_gate_receipt_path,
            metadata=metadata,
            config=config,
            config_path=config_path,
            arm=arm,
            training_seed=int(training_seed),
            runtime_protocol_sha256=runtime_protocol_sha256,
        )
        if arm == "direct_plan_answer_positive_control_action" and training_seed != screen:
            raise ValueError("positive-control replication seed is not authorized")
    counts = tuple(
        sorted(
            {
                int(evaluation["primary_candidate_count"]),
                *[int(value) for value in evaluation["descriptive_candidate_counts"]],
            }
        )
    )
    scored = score_generation_rows(
        _read_jsonl(generated_path),
        _read_jsonl(labels_path),
        arm=arm,
        candidate_counts=counts,
        answer_max_tokens=int(evaluation["answer_max_tokens"]),
        loop_detector=evaluation["loop_detector"],
    )
    score_script = experiment_root / "scripts" / "score.py"
    provenance = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "config_sha256": sha256_file(config_path),
        "score_script_sha256": sha256_file(score_script),
        "score_module_sha256": sha256_file(Path(__file__).resolve()),
        "scoring_module_sha256": sha256_file(experiment_root / "src" / "scoring.py"),
        "arm": arm,
        "training_seed": training_seed,
        "generated": _artifact_ref(generated_path),
        "metadata": _artifact_ref(metadata_path),
        "input_receipt": _artifact_ref(input_receipt_path),
        "labels": _artifact_ref(labels_path),
        "adapter_gate_receipt": (
            None if adapter_gate_receipt_path is None else _artifact_ref(adapter_gate_receipt_path)
        ),
    }
    generated_sha256 = provenance["generated"]["sha256"]
    metadata_sha256 = provenance["metadata"]["sha256"]
    for row in scored:
        expected_family, expected_depth = expected_task_metadata[row["task_id"]]
        if row["family"] != expected_family or int(row["depth"]) != expected_depth:
            raise ValueError("scored task metadata differs from sealed reconstruction")
        row["training_seed"] = training_seed
        row["generated_sha256"] = generated_sha256
        row["metadata_sha256"] = metadata_sha256
        row["runtime_protocol_sha256"] = runtime_protocol_sha256
        row["adapter_gate_receipt_sha256"] = (
            None
            if adapter_gate_receipt_path is None
            else provenance["adapter_gate_receipt"]["sha256"]
        )
        row["score_provenance"] = provenance
    return scored


def _path_from_ref(value: Any, label: str) -> Path:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"score provenance has malformed {label} reference")
    path = Path(value["path"])
    if not path.is_absolute() or not path.is_file() or sha256_file(path) != value["sha256"]:
        raise ValueError(f"score provenance {label} artifact is absent or changed")
    return path


def validate_score_artifact(
    path: Path,
    *,
    config: dict[str, Any],
    config_path: Path,
    experiment_root: Path,
) -> list[dict[str, Any]]:
    observed = _read_jsonl(path)
    if not observed:
        raise ValueError("score artifact is empty")
    provenances = [row.get("score_provenance") for row in observed]
    if any(value != provenances[0] for value in provenances):
        raise ValueError("score rows do not share one exact provenance object")
    provenance = provenances[0]
    required = {
        "schema_version", "experiment_id", "config_sha256", "score_script_sha256",
        "score_module_sha256", "scoring_module_sha256", "arm", "training_seed",
        "generated", "metadata", "input_receipt", "labels", "adapter_gate_receipt",
    }
    if (
        not isinstance(provenance, dict)
        or set(provenance) != required
        or provenance["schema_version"] != 1
        or provenance["experiment_id"] != config["experiment_id"]
        or provenance["config_sha256"] != sha256_file(config_path)
        or provenance["score_script_sha256"]
        != sha256_file(experiment_root / "scripts" / "score.py")
        or provenance["score_module_sha256"] != sha256_file(Path(__file__).resolve())
        or provenance["scoring_module_sha256"]
        != sha256_file(experiment_root / "src" / "scoring.py")
    ):
        raise ValueError("score artifact producer identity differs from current implementation")
    gate_ref = provenance["adapter_gate_receipt"]
    gate_path = None if gate_ref is None else _path_from_ref(gate_ref, "adapter gate")
    expected = build_score_rows(
        config=config,
        config_path=config_path,
        experiment_root=experiment_root,
        generated_path=_path_from_ref(provenance["generated"], "generated"),
        metadata_path=_path_from_ref(provenance["metadata"], "metadata"),
        input_receipt_path=_path_from_ref(provenance["input_receipt"], "input receipt"),
        labels_path=_path_from_ref(provenance["labels"], "labels"),
        arm=str(provenance["arm"]),
        training_seed=provenance["training_seed"],
        adapter_gate_receipt_path=gate_path,
    )
    if observed != expected or path.read_bytes() != jsonl_payload(expected):
        raise ValueError("score artifact differs from exact raw-generation reconstruction")
    return observed
