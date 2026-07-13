"""Configuration loading and immutable experiment-contract validation."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
BACKEND = "transformers"
EXPERIMENT_ID = "qwen35_4b_state_formation_capacity_adjudication"
SOURCE_CONTRACT_VERSION = 6
CONFIRMATORY_CONFIG_SHA256 = "eeb4e828526f750dce1258bcc91d03114c80688d300112e03d18c9d911489393"
SOURCE_CONTRACT_FILES = (
    "scripts/archive_failed_attempt.py",
    "scripts/archive_invalidated_setup.py",
    "scripts/run.py",
    "reports/implementation_review.md",
    "src/__init__.py",
    "src/adaptation.py",
    "src/analysis.py",
    "src/config.py",
    "src/data_pipeline.py",
    "src/design_boundary.py",
    "src/gpu_runner.py",
    "src/initialization.py",
    "src/mechanics.py",
    "src/optimizer_receipts.py",
    "src/state_loop_model.py",
    "src/substrate.py",
    "tests/__init__.py",
    "tests/test_archive_failed_attempt.py",
    "tests/test_archive_invalidated_setup.py",
    "tests/test_analysis.py",
    "tests/test_config.py",
    "tests/test_data_parity.py",
    "tests/test_design_boundary.py",
    "tests/test_fullrank_delta.py",
    "tests/test_initialization.py",
    "tests/test_mechanics.py",
    "tests/test_model_smoke_failure.py",
    "tests/test_objectives.py",
    "tests/test_optimizer_receipts.py",
    "tests/test_positive_control.py",
    "tests/test_receipt_contracts.py",
    "tests/test_state_loop_aggregation.py",
    "tests/test_static_contracts.py",
    "tests/test_substrate.py",
)

# Durable failed-G0 receipts are resumable setup state, not free-form
# diagnostics.  The producer and invalidated-setup archiver share this exact
# progress language so an archived failure cannot claim impossible work.
G0_COMPLETED_CHECKS = (
    "branch_authorization",
    "train_only_data_manifest",
    "shared_initialization",
    "pinned_model_and_wrapper_setup",
    "registered_setup_rows_and_encoding",
    "pre_optimizer_k1_parity",
    "zero_function_and_k4_call_geometry",
    "two_step_state_only_optimizer_probe",
    "live_joint_backward_and_optimizer_probe",
    "optimizer_state_receipt",
    "timed_ten_step_probe",
    "post_optimizer_k1_parity",
    "worst_depth_probe",
    "common_initialization_rng_isolation",
    "checkpoint_roundtrip",
    "memory_headroom",
)
G0_FAILURE_STAGE_PREFIX_LENGTHS = {
    "branch_authorization": (0,),
    "data_manifest": (1,),
    # Initialization is captured inside the canonical loader, while the full
    # setup binding is installed immediately after that loader returns.
    "model_setup": (2, 3, 4),
    "setup_rows_and_encoding": (4,),
    "pre_optimizer_k1_parity": (5,),
    "zero_function_and_k4_call_geometry": (6,),
    "two_step_state_only_optimizer_probe": (7,),
    "live_joint_backward_probe": (8,),
    "live_joint_optimizer_step": (8,),
    "optimizer_state_receipt": (9,),
    "timed_ten_step_probe": (10,),
    "post_optimizer_k1_parity": (11,),
    "worst_depth_probe": (12,),
    "common_initialization_rng_isolation": (13,),
    "checkpoint_roundtrip": (14,),
    "memory_headroom": (15,),
    "receipt_construction": (16,),
}


def _merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key == "inherits":
            continue
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge(dict(merged[key]), value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _config_digest(config: Mapping[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def is_confirmatory_config(config: Mapping[str, Any]) -> bool:
    return (
        config.get("evidence_profile") == "confirmatory"
        and _config_digest(config) == CONFIRMATORY_CONFIG_SHA256
    )


def require_confirmatory_config(config: Mapping[str, Any]) -> None:
    if not is_confirmatory_config(config):
        raise RuntimeError(
            "model-bearing and verdict stages require the exact frozen default config"
        )


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    parent = raw.get("inherits")
    config = _merge(load_config(path.parent / str(parent)), raw) if parent else raw
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError(f"experiment_id must be {EXPERIMENT_ID}")
    profile = config.get("evidence_profile")
    if profile not in {"confirmatory", "smoke"}:
        raise ValueError("evidence_profile must be confirmatory or smoke")
    model = config["model"]
    frozen_model = {
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        "backend": BACKEND,
        "transformers_version": "5.13.0",
        "dtype": "bfloat16",
        "attention_implementation": "sdpa",
        "trust_remote_code": True,
        "use_cache": False,
    }
    if dict(model) != frozen_model:
        raise ValueError("the Qwen3.5-4B Transformers model contract is frozen")

    arch = config["architecture"]
    start, end = int(arch["loop_start"]), int(arch["loop_end"])
    layers = int(arch["expected_num_layers"])
    pattern = list(arch["expected_layer_pattern"])
    if (start, end, layers, int(arch["expected_hidden_size"])) != (12, 20, 32, 2560):
        raise ValueError("Qwen loop geometry is frozen")
    if len(pattern) != 4 or start % 4 or end % 4:
        raise ValueError("loop boundaries must align to complete four-layer motifs")
    if arch["semantic_echo"]["mode"] != "continuous":
        raise ValueError("only the continuous state interface is registered")
    if int(arch["max_recurrence"]) < max(map(int, config["evaluation"]["k_values"])):
        raise ValueError("max_recurrence must cover every evaluated K")
    frozen_lora = {
        "mode": "matched_hook_lora",
        "rank": 32,
        "alpha": 64,
        "dropout": 0.05,
        "scale": 2.0,
        "expected_targets": 62,
        "expected_parameters": 16232448,
        "target_loop_only": True,
        "active_on_extra_calls_only": True,
    }
    frozen_fullrank = {
        "mode": "direct_weight_delta",
        "dropout": 0.05,
        "scale": 2.0,
        "expected_targets": 62,
        "expected_parameters": 892272640,
        "parameter_dtype": "float32",
        "initialization": "zeros",
        "target_loop_only": True,
        "active_on_extra_calls_only": True,
    }
    adaptation = arch["adaptation"]
    if dict(adaptation["lora"]) != frozen_lora or dict(adaptation["fullrank"]) != frozen_fullrank:
        raise ValueError("the matched adaptation parameterizations are frozen")

    substrate = config["substrate"]
    if substrate["heldout_family"] in set(substrate["train_families"]):
        raise ValueError("held-out family overlaps training")
    if substrate["heldout_template"] in set(substrate["train_templates"]):
        raise ValueError("held-out template overlaps training")
    train_depths = list(map(int, substrate["train_depths"]))
    extrapolation = list(map(int, substrate["extrapolation_depths"]))
    if max(train_depths) >= min(extrapolation):
        raise ValueError("depth extrapolation must begin beyond training")
    if set(substrate["seeds"]) != {
        "train", "validation", "depth", "joint", "contrast_validation",
        "contrast_depth", "contrast_joint",
    }:
        raise ValueError("exactly seven fresh data seeds are required")
    if len(set(map(int, substrate["seeds"].values()))) != 7:
        raise ValueError("data seeds must be pairwise distinct")
    for key in (
        "train_examples", "validation_examples", "depth_examples", "joint_examples",
        "contrast_validation_examples", "contrast_depth_examples", "contrast_joint_examples",
    ):
        if int(substrate[key]) <= 0 or int(substrate[key]) % 2:
            raise ValueError(f"{key} must be positive and even")

    training = config["training"]
    if int(training["train_k"]) != max(train_depths) or int(training["batch_size"]) != 1:
        raise ValueError("training geometry is frozen to K=4 and microbatch one")
    if list(map(int, training["train_seeds"])) != [7411, 7412, 7413]:
        raise ValueError("train seeds are frozen to 7411--7413")
    if dict(training["g0_control"]) != {"worst_depth_seed": 73992}:
        raise ValueError("the setup-only G0 worst-depth seed is frozen")
    frozen_objectives = {
        "joint": {
            "answer_loss_weight": 1.0,
            "state_loss_weight": 0.5,
            "fixed_point_loss_weight": 0.05,
        },
        "state_only": {
            "answer_loss_weight": 0.0,
            "state_loss_weight": 0.5,
            "fixed_point_loss_weight": 0.05,
        },
    }
    if dict(training["objectives"]) != frozen_objectives:
        raise ValueError("joint and state-only objective definitions are frozen")
    frozen_positive_control = {
        "rows": 48,
        "updates": 256,
        "seed": 73991,
        "depths": [2, 3, 4],
        "examples_per_cell": 2,
        "min_overfit_final_joint_accuracy": 0.95,
        "min_oracle_readout_accuracy": 0.99,
    }
    if profile == "confirmatory" and dict(training["positive_control"]) != frozen_positive_control:
        raise ValueError("the fresh positive-control grid is frozen")
    if profile == "confirmatory":
        frozen_training = {
            "train_steps": 1500,
            "gradient_accumulation": 16,
            "learning_rate": 0.0002,
            "weight_decay": 0.01,
            "warmup_fraction": 0.05,
            "adaptation_gradient_clip": 1.0,
            "common_gradient_clip": 1.0,
            "eval_every_steps": 250,
            "save_every_steps": 1500,
        }
        for key, expected in frozen_training.items():
            if training[key] != expected:
                raise ValueError(f"{key} is frozen to {expected}")

    evaluation = config["evaluation"]
    if list(evaluation["trigger_splits"]) != [
        "validation", "depth_extrapolation", "joint_holdout"
    ] or list(evaluation["sealed_contrast_splits"]) != [
        "contrast_validation", "contrast_depth", "contrast_joint"
    ]:
        raise ValueError("the trigger and sealed contrast cells are frozen")
    if evaluation["require_same_backend"] is not True or int(evaluation["bootstrap_resamples"]) < 10000:
        raise ValueError("evaluation backend/bootstrap contract violated")
    frozen_gates = {
        "k1_max_logit_abs_error": 0.00001,
        "exact_parameter_count_match": True,
        "min_final_joint_accuracy_each_seed_depth": 0.40,
        "min_adaptation_intact_minus_disabled_each_seed": 0.0,
        "require_adaptation_gain_crossed_lcb_above_zero": True,
    }
    if dict(config["gates"]) != frozen_gates:
        raise ValueError("state-formation gates are frozen")
    if config["paths"].get("design_receipt") != "reports/design_receipt.json":
        raise ValueError("the canonical design-boundary receipt path is frozen")
    if profile == "confirmatory" and not is_confirmatory_config(config):
        raise ValueError("confirmatory evidence requires the frozen default config digest")


def canonical_json(config: Mapping[str, Any]) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def config_sha256(config: Mapping[str, Any]) -> str:
    return _config_digest(config)


def source_contract_sha256(root: str | Path | None = None) -> str:
    experiment_root = Path(root).resolve() if root else Path(__file__).resolve().parents[1]
    files = []
    for relative_path in SOURCE_CONTRACT_FILES:
        path = experiment_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"source-contract file is missing: {path}")
        files.append({"path": relative_path, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    payload = {"version": SOURCE_CONTRACT_VERSION, "files": files}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def resolved_config_receipt(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "schema_version": config["schema_version"],
        "evidence_profile": config["evidence_profile"],
        "config_sha256": config_sha256(config),
        "source_contract_version": SOURCE_CONTRACT_VERSION,
        "source_contract_sha256": source_contract_sha256(),
        "model_id": config["model"]["id"],
        "model_revision": config["model"]["revision"],
        "backend": config["model"]["backend"],
        "capacities": ["lora", "fullrank"],
        "objectives": ["joint", "state_only"],
        "train_seeds": config["training"]["train_seeds"],
    }
