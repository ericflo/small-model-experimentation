"""Exact generation, environment, and sealed-input provenance checks."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from eval_inputs import action_bundles, action_receipt, jsonl_payload, task_metadata
from vllm_runner import EngineConfig, SamplingConfig


LOCK_LINE = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s;]+)$")


def canonical(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return canonical(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical(item) for item in value]
    return value


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        canonical(value), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def expected_sampling(config: SamplingConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    config.validate()
    return canonical(dataclasses.asdict(config)), canonical(config.resolved_sampling())


def validate_sampling(metadata: dict[str, Any], expected: SamplingConfig) -> None:
    expected_raw, expected_resolved = expected_sampling(expected)
    if metadata.get("sampling") != expected_raw:
        raise ValueError("generation sampling dictionary differs from preregistration")
    if metadata.get("resolved_sampling") != expected_resolved:
        raise ValueError("generation resolved-sampling dictionary differs from preregistration")


def validate_action_inputs(
    *,
    config: dict[str, Any],
    config_path: Path,
    receipt_path: Path,
    labels_path: Path,
    expected_split: str | None = None,
) -> tuple[str, dict[str, tuple[str, int]], dict[str, Any]]:
    observed_receipt = json.loads(receipt_path.read_text())
    split = str(observed_receipt.get("split", ""))
    if expected_split is not None and split != expected_split:
        raise ValueError("action receipt has the wrong sealed split")
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    expected_receipt = action_receipt(config, config_sha256, split)
    if observed_receipt != expected_receipt:
        raise ValueError("action input receipt differs from sealed reconstruction")
    prompts, labels = action_bundles(config, split)
    if hashlib.sha256(labels_path.read_bytes()).hexdigest() != hashlib.sha256(
        jsonl_payload(labels)
    ).hexdigest():
        raise ValueError("oracle labels differ from sealed reconstruction")
    return split, task_metadata(config, split), {
        "receipt": expected_receipt,
        "prompt_sha256": hashlib.sha256(jsonl_payload(prompts)).hexdigest(),
        "label_sha256": hashlib.sha256(jsonl_payload(labels)).hexdigest(),
    }


def _locked_versions(lock_path: Path) -> dict[str, str]:
    versions: dict[str, str] = {}
    for line in lock_path.read_text().splitlines():
        match = LOCK_LINE.fullmatch(line)
        if match:
            versions[match.group(1).lower().replace("_", "-")] = match.group(2)
    if not versions:
        raise ValueError("vLLM lock contains no exact package pins")
    return versions


def validate_runtime_packages(runtime: dict[str, Any], lock_path: Path) -> None:
    lock_sha256 = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    if runtime.get("environment_lock", {}).get("sha256") != lock_sha256:
        raise ValueError("generation environment lock differs from repository pin")
    packages = runtime.get("packages")
    if not isinstance(packages, dict):
        raise ValueError("generation runtime lacks the installed-package inventory")
    mismatches = {
        name: (version, packages.get(name))
        for name, version in _locked_versions(lock_path).items()
        if packages.get(name) != version
    }
    if mismatches:
        first = sorted(mismatches.items())[:5]
        raise ValueError(f"installed packages differ from vLLM lock: {first}")


def validate_generation_protocol(
    *,
    metadata: dict[str, Any],
    config: dict[str, Any],
    experiment_root: Path,
    generated_path: Path,
    expected_rows: int,
    expect_merged: bool,
    expected_stage: str,
    expected_split: str,
    expected_input_kind: str,
    expected_source_seed: int | None,
) -> str:
    runner_path = experiment_root / "src" / "vllm_runner.py"
    generated_sha256 = hashlib.sha256(generated_path.read_bytes()).hexdigest()
    if metadata.get("output") != {
        "description": "generated JSONL",
        "sha256": generated_sha256,
        "rows": expected_rows,
    }:
        raise ValueError("generation output differs from exact runner metadata")
    if (
        metadata.get("runner_sha256") != hashlib.sha256(runner_path.read_bytes()).hexdigest()
        or metadata.get("base_model") != config["model"]["id"]
        or metadata.get("model_revision") != config["model"]["revision"]
    ):
        raise ValueError("generation used the wrong runner or model identity")
    model_override = metadata.get("model_override")
    if expect_merged != (model_override is not None):
        raise ValueError("generation base/merged model status differs from its arm")
    if metadata.get("adapter") is not None:
        raise ValueError("runtime LoRA adapter use is forbidden")
    evaluation = config["evaluation"]
    frozen_engine = evaluation["engine"]
    override_path = None if model_override is None else Path(model_override["path"])
    expected_engine = canonical(
        EngineConfig(
            max_model_len=int(frozen_engine["max_model_len"]),
            gpu_memory_utilization=float(frozen_engine["gpu_memory_utilization"]),
            max_num_seqs=int(frozen_engine["max_num_seqs"]),
            max_num_batched_tokens=int(frozen_engine["max_num_batched_tokens"]),
            cudagraph_capture_sizes=tuple(frozen_engine["cudagraph_capture_sizes"]),
            enable_prefix_caching=bool(frozen_engine["prefix_caching"]),
            enforce_eager=False,
            model_override=override_path,
        )
    )
    if metadata.get("engine") != expected_engine:
        raise ValueError("generation engine dictionary differs from preregistration")
    expected_engine_args = {
        "model": config["model"]["id"] if override_path is None else str(override_path),
        "tokenizer": config["model"]["id"],
        "tokenizer_revision": config["model"]["revision"],
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "max_model_len": int(frozen_engine["max_model_len"]),
        "gpu_memory_utilization": float(frozen_engine["gpu_memory_utilization"]),
        "max_num_seqs": int(frozen_engine["max_num_seqs"]),
        "max_num_batched_tokens": int(frozen_engine["max_num_batched_tokens"]),
        "language_model_only": True,
        "enable_prefix_caching": bool(frozen_engine["prefix_caching"]),
        "mamba_cache_mode": "none",
        "enforce_eager": False,
        "generation_config": "vllm",
        "max_logprobs": 20,
        "seed": 0,
        "async_scheduling": bool(frozen_engine["async_scheduling"]),
        "cudagraph_capture_sizes": list(frozen_engine["cudagraph_capture_sizes"]),
        "max_cudagraph_capture_size": max(frozen_engine["cudagraph_capture_sizes"]),
    }
    if override_path is None:
        expected_engine_args["revision"] = config["model"]["revision"]
    if metadata.get("engine_args") != expected_engine_args:
        raise ValueError("generation engine arguments differ from preregistration")
    runtime = metadata.get("runtime", {})
    if runtime.get("git_dirty"):
        raise ValueError("generation ran from a dirty worktree")
    current_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if runtime.get("git_commit") != current_commit:
        raise ValueError("generation Git commit differs from the scoring checkout")
    validate_runtime_packages(runtime, experiment_root.parents[1] / "requirements-vllm.lock.txt")
    generation_stage = metadata.get("generation_stage")
    if (
        not isinstance(generation_stage, dict)
        or set(generation_stage)
        != {
            "authorized_stage",
            "stage_receipt_sha256",
            "config_sha256",
            "issuer_git_commit",
            "split",
            "input_kind",
            "source_seed",
        }
        or generation_stage["authorized_stage"] != expected_stage
        or not isinstance(generation_stage["stage_receipt_sha256"], str)
        or len(generation_stage["stage_receipt_sha256"]) != 64
        or generation_stage["config_sha256"]
        != hashlib.sha256((experiment_root / "configs" / "default.yaml").read_bytes()).hexdigest()
        or generation_stage["issuer_git_commit"] != runtime["git_commit"]
        or generation_stage["split"] != expected_split
        or generation_stage["input_kind"] != expected_input_kind
        or generation_stage["source_seed"] != expected_source_seed
    ):
        raise ValueError("generation stage metadata differs from exact ancestry")
    preflight = metadata.get("capacity_preflight")
    if not isinstance(preflight, dict) or preflight.get("decision") != "LIVE_KV_CAPACITY_PASS":
        raise ValueError("generation lacks a passing live KV-capacity preflight")
    if preflight.get("engine") != expected_engine:
        raise ValueError("live KV preflight is not bound to the frozen engine")
    protocol = {
        "base_model": metadata["base_model"],
        "model_revision": metadata["model_revision"],
        "runner_sha256": metadata["runner_sha256"],
        "engine": {**expected_engine, "model_override": None},
        "engine_args": {
            key: value
            for key, value in expected_engine_args.items()
            if key not in {"model", "revision"}
        },
        "resolved_cudagraph": metadata.get("resolved_cudagraph"),
        "capacity_geometry": preflight.get("live_cache"),
        "runtime": runtime,
    }
    return canonical_sha256(protocol)
