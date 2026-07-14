"""Exact generation, environment, and sealed-input provenance checks."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename

from eval_inputs import action_bundles, action_receipt, jsonl_payload, task_metadata
from scoring import validate_generation_counters
from vllm_runner import EngineConfig, SamplingConfig, _validate_model_override


LOCK_LINE = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s;]+)$")
TRAINING_RUNTIME_EXTRAS = {
    "causal-conv1d": "1.6.2.post1",
    "iniconfig": "2.3.0",
    "pluggy": "1.6.0",
    "pytest": "9.1.1",
}


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


def _locked_versions(
    lock_path: Path, *, required_backend: str = "vllm"
) -> dict[str, str]:
    versions: dict[str, str] = {}
    for line in lock_path.read_text().splitlines():
        match = LOCK_LINE.fullmatch(line)
        if match:
            versions[canonicalize_name(match.group(1))] = match.group(2)
            continue
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or line[:1].isspace():
            continue
        requirement = Requirement(stripped)
        if requirement.url is None:
            continue
        wheel_name = Path(unquote(urlparse(requirement.url).path)).name
        try:
            distribution, version, _build, _tags = parse_wheel_filename(wheel_name)
        except ValueError as error:
            raise ValueError(
                f"direct-URL lock requirement is not an exact wheel pin: {stripped}"
            ) from error
        required_name = canonicalize_name(requirement.name)
        if canonicalize_name(distribution) != required_name:
            raise ValueError(
                f"direct-URL wheel distribution differs from requirement: {stripped}"
            )
        versions[required_name] = str(version)
    if not versions:
        raise ValueError("runtime lock contains no exact package pins")
    if required_backend == "vllm":
        if versions.get("vllm") != "0.24.0+cu129":
            raise ValueError("vLLM lock lacks the exact 0.24.0+cu129 backend wheel pin")
    elif required_backend == "training":
        if versions.get("peft") != "0.19.1" or versions.get("bitsandbytes") != "0.49.2":
            raise ValueError("training lock lacks exact PEFT/bitsandbytes pins")
        versions.update(TRAINING_RUNTIME_EXTRAS)
    else:
        raise ValueError(f"unknown runtime backend: {required_backend}")
    return versions


def validate_runtime_packages(
    runtime: dict[str, Any],
    lock_path: Path,
    *,
    required_backend: str = "vllm",
) -> None:
    lock_sha256 = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    if runtime.get("environment_lock", {}).get("sha256") != lock_sha256:
        raise ValueError("generation environment lock differs from repository pin")
    packages = runtime.get("packages")
    if not isinstance(packages, dict):
        raise ValueError("generation runtime lacks the installed-package inventory")
    locked = _locked_versions(lock_path, required_backend=required_backend)
    if set(packages) != set(locked):
        missing = sorted(set(locked) - set(packages))[:5]
        extra = sorted(set(packages) - set(locked))[:5]
        raise ValueError(
            "installed package surface differs from the exact vLLM lock: "
            f"missing={missing}, extra={extra}"
        )
    mismatches = {
        name: (version, packages.get(name))
        for name, version in locked.items()
        if packages.get(name) != version
    }
    if mismatches:
        first = sorted(mismatches.items())[:5]
        raise ValueError(f"installed packages differ from vLLM lock: {first}")


def validate_interpreter_runtime(runtime: dict[str, Any], repo_root: Path) -> None:
    executable_value = runtime.get("python_executable")
    if not isinstance(executable_value, str):
        raise ValueError("runtime lacks an exact interpreter path")
    executable = Path(executable_value)
    repo_root = repo_root.resolve()
    if (
        not executable.is_absolute()
        or not executable.is_file()
        or executable.is_symlink()
        or repo_root == executable.resolve()
        or repo_root in executable.resolve().parents
        or runtime.get("python_executable_sha256")
        != hashlib.sha256(executable.read_bytes()).hexdigest()
        or runtime.get("python_isolated") is not True
        or runtime.get("python_dont_write_bytecode") is not True
        or runtime.get("python_no_site") is not True
    ):
        raise ValueError("runtime interpreter is mutable, unauthenticated, or non-isolated")


def validate_runtime_bootstrap(
    runtime: dict[str, Any], repo_root: Path, expected_backend: str
) -> None:
    from runtime_contract import (
        _bootstrap_locked_versions,
        _initial_import_path_allowed,
        _runtime_pin,
        authenticate_site_packages,
    )

    bootstrap = runtime.get("bootstrap")
    required = {
        "schema_version",
        "backend",
        "worktree",
        "environment_root",
        "invoked_python",
        "resolved_python",
        "resolved_python_sha256",
        "site_packages",
        "initial_sys_path",
        "startup_files",
        "environment_authentication",
        "lock_file",
        "lock_sha256",
        "python_isolated",
        "python_dont_write_bytecode",
        "python_no_site",
    }
    pin = _runtime_pin(expected_backend)
    environment_root = Path(pin["environment_root"])
    invoked_python = environment_root / "bin" / "python"
    site_packages = (
        environment_root
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    startup_paths = {
        path.name: path
        for pattern in ("*.pth", "sitecustomize.py", "usercustomize.py")
        for path in site_packages.glob(pattern)
    }
    observed_startup = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in sorted(startup_paths.items())
    }
    lock_path = repo_root / pin["lock_file"]
    environment_authentication = authenticate_site_packages(
        site_packages,
        _bootstrap_locked_versions(lock_path, expected_backend),
        pin["record_surface_sha256"],
        pin["site_surface_sha256"],
    )
    expected_worktree = runtime.get("worktree")
    if expected_worktree is None:
        expected_worktree = {
            "repo_root": runtime.get("git_root"),
            "git_commit": runtime.get("git_commit"),
            "head_mode": runtime.get("git_head_mode"),
            "cwd": runtime.get("cwd"),
        }
    if (
        not isinstance(bootstrap, dict)
        or set(bootstrap) != required
        or bootstrap.get("schema_version") != 1
        or bootstrap.get("backend") != expected_backend
        or bootstrap.get("worktree") != expected_worktree
        or bootstrap.get("environment_root") != str(environment_root)
        or bootstrap.get("invoked_python") != str(invoked_python)
        or bootstrap.get("resolved_python") != str(invoked_python.resolve())
        or bootstrap.get("resolved_python_sha256")
        != hashlib.sha256(invoked_python.resolve().read_bytes()).hexdigest()
        or bootstrap.get("site_packages") != str(site_packages.resolve())
        or not isinstance(bootstrap.get("initial_sys_path"), list)
        or not bootstrap["initial_sys_path"]
        or any(not _initial_import_path_allowed(item) for item in bootstrap["initial_sys_path"])
        or bootstrap.get("startup_files") != pin["startup_files"]
        or observed_startup != pin["startup_files"]
        or bootstrap.get("environment_authentication") != environment_authentication
        or bootstrap.get("lock_file") != pin["lock_file"]
        or bootstrap.get("lock_sha256")
        != hashlib.sha256(lock_path.read_bytes()).hexdigest()
        or bootstrap.get("python_isolated") is not True
        or bootstrap.get("python_dont_write_bytecode") is not True
        or bootstrap.get("python_no_site") is not True
    ):
        raise ValueError("runtime bootstrap/import/startup surface is invalid or stale")


def validate_gpu_identity(value: Any) -> dict[str, Any]:
    required = {
        "cuda_visible_devices",
        "physical_index",
        "name",
        "uuid",
        "driver_version",
        "memory_total_mib",
    }
    if (
        not isinstance(value, dict)
        or set(value) != required
        or not isinstance(value.get("uuid"), str)
        or not value["uuid"].startswith("GPU-")
        or value.get("cuda_visible_devices") != value["uuid"]
        or type(value.get("physical_index")) is not int
        or value["physical_index"] < 0
        or not isinstance(value.get("name"), str)
        or not value["name"]
        or not isinstance(value.get("driver_version"), str)
        or not value["driver_version"]
        or type(value.get("memory_total_mib")) is not int
        or value["memory_total_mib"] < 1
    ):
        raise ValueError("runtime lacks one exact selected GPU UUID identity")
    return value


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
    if type(metadata.get("schema_version")) is not int or metadata["schema_version"] != 6:
        raise ValueError("generation runner schema differs from the reviewed version")
    if metadata.get("output") != {
        "description": "generated JSONL",
        "sha256": generated_sha256,
        "rows": expected_rows,
    }:
        raise ValueError("generation output differs from exact runner metadata")
    try:
        generated_rows = [
            json.loads(line)
            for line in generated_path.read_text().splitlines()
            if line.strip()
        ]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("generated output is not valid UTF-8 JSONL") from error
    if len(generated_rows) != expected_rows:
        raise ValueError("generated output row count differs from protocol expectation")
    validate_generation_counters(generated_rows, metadata)
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
    from merge_replay import authenticate_base_snapshot, base_snapshot_commitment
    from load_window_guard import validate_load_window_receipt
    from tokenizer_lineage import (
        authenticate_tokenizer_snapshot,
        ensure_closed_tokenizer_view,
    )

    exact_base_root, _base_index, _base_structure = authenticate_base_snapshot()
    exact_base = base_snapshot_commitment(exact_base_root)
    closed_tokenizer_path, exact_tokenizer = ensure_closed_tokenizer_view()
    if authenticate_tokenizer_snapshot() != exact_tokenizer:
        raise ValueError("closed/source tokenizer commitments differ")
    exact_override = _validate_model_override(
        None if model_override is None else Path(model_override["path"])
    )
    override_path = None if model_override is None else Path(model_override["path"])
    post_load = metadata.get("post_load_integrity")
    if (
        metadata.get("base_snapshot") != exact_base
        or metadata.get("tokenizer_snapshot") != exact_tokenizer
        or not isinstance(post_load, dict)
        or set(post_load)
        != {
            "base_snapshot",
            "tokenizer_snapshot",
            "model_override",
            "load_window_guards",
            "decision",
        }
        or post_load.get("base_snapshot") != exact_base
        or post_load.get("tokenizer_snapshot") != exact_tokenizer
        or post_load.get("model_override") != exact_override
        or post_load.get("decision")
        != "LOAD_WINDOWS_IMMUTABLE_AND_POSTLOAD_BYTES_MATCH"
    ):
        raise ValueError("generation model/tokenizer load commitments differ from exact bytes")
    load_guards = post_load["load_window_guards"]
    if not isinstance(load_guards, dict) or set(load_guards) != {
        "tokenizer_and_config",
        "engine",
    }:
        raise ValueError("generation lacks both load-window guard receipts")
    validate_load_window_receipt(
        load_guards["tokenizer_and_config"],
        [exact_base_root, closed_tokenizer_path],
        expected_content={"base": exact_base, "tokenizer": exact_tokenizer},
    )
    validate_load_window_receipt(
        load_guards["engine"],
        [closed_tokenizer_path, exact_base_root if override_path is None else override_path],
        expected_content={
            "tokenizer": exact_tokenizer,
            "engine_model": exact_base if override_path is None else exact_override,
        },
    )
    evaluation = config["evaluation"]
    frozen_engine = evaluation["engine"]
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
        "model": str(exact_base_root if override_path is None else override_path),
        "tokenizer": str(closed_tokenizer_path),
        "trust_remote_code": False,
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
    if metadata.get("engine_args") != expected_engine_args:
        raise ValueError("generation engine arguments differ from preregistration")
    runtime = metadata.get("runtime", {})
    repo_root = experiment_root.parents[1].resolve()
    expected_runtime_keys = {
        "schema_version",
        "bootstrap",
        "python",
        "python_executable",
        "python_executable_sha256",
        "python_isolated",
        "python_dont_write_bytecode",
        "python_no_site",
        "platform",
        "packages",
        "packages_sha256",
        "environment_lock",
        "uv",
        "cuda_toolkit",
        "gpu",
        "vllm_enable_v1_multiprocessing",
        "git_commit",
        "git_dirty",
        "git_root",
        "cwd",
        "git_head_mode",
    }
    if (
        not isinstance(runtime, dict)
        or set(runtime) != expected_runtime_keys
        or type(runtime.get("schema_version")) is not int
        or runtime["schema_version"] != 3
        or runtime.get("git_dirty")
        or runtime.get("git_root") != str(repo_root)
        or runtime.get("cwd") != str(repo_root)
        or runtime.get("git_head_mode") != "detached"
        or runtime.get("python_isolated") is not True
        or runtime.get("python_dont_write_bytecode") is not True
        or runtime.get("python_no_site") is not True
        or not isinstance(runtime.get("python_executable"), str)
        or not isinstance(runtime.get("python_executable_sha256"), str)
        or len(runtime["python_executable_sha256"]) != 64
        or runtime.get("packages_sha256") != canonical_sha256(runtime.get("packages"))
    ):
        raise ValueError("generation did not run from the clean detached execution worktree")
    validate_gpu_identity(runtime.get("gpu"))
    if exact_override is not None and runtime["gpu"] != exact_override["source_training_gpu"]:
        raise ValueError("trained generation hardware differs from its training hardware")
    validate_interpreter_runtime(runtime, repo_root)
    validate_runtime_bootstrap(runtime, repo_root, "vllm")
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
            "stage_receipt_path",
            "stage_receipt_sha256",
            "config_sha256",
            "issuer_git_commit",
            "split",
            "input_kind",
            "source_seed",
        }
        or generation_stage["authorized_stage"] != expected_stage
        or not isinstance(generation_stage["stage_receipt_path"], str)
        or not Path(generation_stage["stage_receipt_path"]).is_absolute()
        or not Path(generation_stage["stage_receipt_path"]).is_file()
        or hashlib.sha256(
            Path(generation_stage["stage_receipt_path"]).read_bytes()
        ).hexdigest()
        != generation_stage["stage_receipt_sha256"]
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
    from stages import read_and_validate_stage_receipt

    read_and_validate_stage_receipt(
        Path(generation_stage["stage_receipt_path"]),
        config=config,
        config_path=experiment_root / "configs" / "default.yaml",
        expected_stage=expected_stage,
    )
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
