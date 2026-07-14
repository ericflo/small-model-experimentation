"""Replayable ON/OFF adapter gates anchored to raw base and merged generations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from eval_inputs import task_metadata
from provenance import validate_action_inputs, validate_generation_protocol, validate_sampling
from vllm_runner import SamplingConfig


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_ref(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"adapter-gate source artifact does not exist: {resolved}")
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _path_from_ref(value: Any, label: str) -> Path:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"adapter-gate artifact has malformed {label} reference")
    path = Path(value["path"])
    if not path.is_absolute() or not path.is_file() or sha256_file(path) != value["sha256"]:
        raise ValueError(f"adapter-gate {label} source is absent or changed")
    return path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build_adapter_gate_artifact(
    *,
    arm: str,
    seed: int,
    base_generated_path: Path,
    base_metadata_path: Path,
    merged_generated_path: Path,
    merged_metadata_path: Path,
    input_receipt_path: Path,
    labels_path: Path,
    config: dict[str, Any],
    config_path: Path,
    experiment_root: Path,
) -> dict[str, Any]:
    if arm not in config["training"]["arms"]:
        raise ValueError("adapter gate arm is not preregistered")
    if seed not in set(config["training"]["staged_seeds"].values()):
        raise ValueError("adapter gate seed is not preregistered")
    screen = int(config["training"]["staged_seeds"]["screen"])
    if arm == config["training"]["positive_control"]["arm"] and seed != screen:
        raise ValueError("positive-control replication seed is not authorized")
    base_meta = json.loads(base_metadata_path.read_text())
    merged_meta = json.loads(merged_metadata_path.read_text())
    input_receipt = json.loads(input_receipt_path.read_text())
    _, expected_task_metadata, sealed = validate_action_inputs(
        config=config,
        config_path=config_path,
        receipt_path=input_receipt_path,
        labels_path=labels_path,
        expected_split="calibration",
    )
    if base_meta["input"]["sha256"] != sealed["prompt_sha256"]:
        raise ValueError("base adapter-gate input differs from sealed reconstruction")
    if merged_meta["input"]["sha256"] != sealed["prompt_sha256"]:
        raise ValueError("merged adapter-gate input differs from sealed reconstruction")
    if base_meta["model_override"] is not None or merged_meta["model_override"] is None:
        raise ValueError("adapter gate did not compare base against a merged override")
    if (
        merged_meta["model_override"].get("source_arm") != arm
        or merged_meta["model_override"].get("source_seed") != seed
    ):
        raise ValueError("merged override source arm/seed differs from adapter gate")
    gate = config["evaluation"]["adapter_gate"]
    expected_sampling = SamplingConfig(
        thinking="budget",
        thinking_budget=int(gate["thinking_budget"]),
        answer_max_tokens=int(gate["answer_max_tokens"]),
        n=int(gate["candidate_count"]),
        greedy=bool(gate["greedy"]),
        logprobs=int(gate["logprobs"]),
        run_seed=int(gate["run_seed"]),
    )
    validate_sampling(base_meta, expected_sampling)
    validate_sampling(merged_meta, expected_sampling)
    protocols = {
        validate_generation_protocol(
            metadata=base_meta,
            config=config,
            experiment_root=experiment_root,
            generated_path=base_generated_path,
            expected_rows=int(input_receipt["rows"]),
            expect_merged=False,
            expected_stage="calibration_generation",
            expected_split="calibration",
            expected_input_kind="action",
            expected_source_seed=None,
        ),
        validate_generation_protocol(
            metadata=merged_meta,
            config=config,
            experiment_root=experiment_root,
            generated_path=merged_generated_path,
            expected_rows=int(input_receipt["rows"]),
            expect_merged=True,
            expected_stage=("screen_training" if seed == screen else "replication_training"),
            expected_split="calibration",
            expected_input_kind="action",
            expected_source_seed=seed,
        ),
    }
    if len(protocols) != 1:
        raise ValueError("adapter gate base/merged runtime protocols differ")
    runtime_protocol_sha256 = protocols.pop()
    base_rows = {row["id"]: row for row in _read_jsonl(base_generated_path)}
    merged_rows = {row["id"]: row for row in _read_jsonl(merged_generated_path)}
    if set(base_rows) != set(merged_rows) or len(base_rows) != int(input_receipt["rows"]):
        raise ValueError("adapter-gate task IDs differ")
    if set(base_rows) != set(expected_task_metadata):
        raise ValueError("adapter-gate task IDs differ from sealed calibration")
    for task_id, (family, depth) in expected_task_metadata.items():
        for row in (base_rows[task_id], merged_rows[task_id]):
            if (
                row.get("meta", {}).get("family") != family
                or int(row.get("meta", {}).get("depth", -1)) != depth
            ):
                raise ValueError("adapter-gate task metadata differs from sealed calibration")
    changed = []
    for task_id in sorted(base_rows):
        base_output = base_rows[task_id]["outputs"][0]
        merged_output = merged_rows[task_id]["outputs"][0]
        if (
            base_output["token_ids"] != merged_output["token_ids"]
            or base_output["sampled_cumulative_logprob"]
            != merged_output["sampled_cumulative_logprob"]
        ):
            changed.append(task_id)
    if not changed:
        raise ValueError("merged adapter is an exact ON/OFF no-op on the frozen gate")
    invocation = {
        "arm": arm,
        "seed": seed,
        "base_generated": _artifact_ref(base_generated_path),
        "base_metadata": _artifact_ref(base_metadata_path),
        "merged_generated": _artifact_ref(merged_generated_path),
        "merged_metadata": _artifact_ref(merged_metadata_path),
        "input_receipt": _artifact_ref(input_receipt_path),
        "labels": _artifact_ref(labels_path),
    }
    return {
        "schema_version": 3,
        "experiment_id": config["experiment_id"],
        "config_sha256": sha256_file(config_path),
        "producer": {
            "script_sha256": sha256_file(experiment_root / "scripts" / "adapter_behavior_gate.py"),
            "module_sha256": sha256_file(Path(__file__).resolve()),
        },
        "invocation": invocation,
        "arm": arm,
        "seed": seed,
        "pass": True,
        "changed_tasks": len(changed),
        "total_tasks": len(base_rows),
        "model_override": merged_meta["model_override"],
        "runtime_protocol_sha256": runtime_protocol_sha256,
        "base_generated_sha256": invocation["base_generated"]["sha256"],
        "merged_generated_sha256": invocation["merged_generated"]["sha256"],
        "base_metadata_sha256": invocation["base_metadata"]["sha256"],
        "merged_metadata_sha256": invocation["merged_metadata"]["sha256"],
    }


def validate_adapter_gate_artifact(
    path: Path,
    *,
    config: dict[str, Any],
    config_path: Path,
    experiment_root: Path,
) -> dict[str, Any]:
    observed = json.loads(path.read_text())
    invocation = observed.get("invocation") if isinstance(observed, dict) else None
    required = {
        "arm", "seed", "base_generated", "base_metadata", "merged_generated",
        "merged_metadata", "input_receipt", "labels",
    }
    if not isinstance(invocation, dict) or set(invocation) != required:
        raise ValueError("adapter gate lacks an exact replayable invocation")
    expected = build_adapter_gate_artifact(
        arm=str(invocation["arm"]),
        seed=int(invocation["seed"]),
        base_generated_path=_path_from_ref(invocation["base_generated"], "base generation"),
        base_metadata_path=_path_from_ref(invocation["base_metadata"], "base metadata"),
        merged_generated_path=_path_from_ref(
            invocation["merged_generated"], "merged generation"
        ),
        merged_metadata_path=_path_from_ref(invocation["merged_metadata"], "merged metadata"),
        input_receipt_path=_path_from_ref(invocation["input_receipt"], "input receipt"),
        labels_path=_path_from_ref(invocation["labels"], "labels"),
        config=config,
        config_path=config_path,
        experiment_root=experiment_root,
    )
    if observed != expected:
        raise ValueError("adapter gate differs from exact base/merged reconstruction")
    return observed
