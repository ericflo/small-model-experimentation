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
EXPERIMENT_ID = "qwen35_4b_state_carry_vs_state_bag"
SOURCE_CONTRACT_VERSION = 1
CONFIRMATORY_CONFIG_SHA256 = "70e4a2d6df7acb0c5a21c7c945c66499a0ede8e98321c7b56da1c080c819744b"
SOURCE_CONTRACT_FILES = (
    "scripts/run.py",
    "src/analysis.py",
    "src/config.py",
    "src/data_pipeline.py",
    "src/gpu_runner.py",
    "src/mechanics.py",
    "src/state_loop_model.py",
    "src/substrate.py",
)


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
            "model-bearing and scientific-verdict stages require the exact frozen "
            "confirmatory default config; smoke/reduced configs are setup-only"
        )


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    parent = raw.get("inherits")
    if parent:
        parent_path = (path.parent / str(parent)).resolve()
        config = _merge(load_config(parent_path), raw)
    else:
        config = raw
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError(f"experiment_id must be {EXPERIMENT_ID}")
    if config.get("evidence_profile") not in {"confirmatory", "smoke"}:
        raise ValueError("evidence_profile must be confirmatory or smoke")
    model = config["model"]
    if model.get("id") != MODEL_ID:
        raise ValueError(f"the only permitted model is {MODEL_ID}")
    if model.get("revision") != MODEL_REVISION:
        raise ValueError(f"model revision must be pinned to {MODEL_REVISION}")
    if model.get("backend") != BACKEND:
        raise ValueError("all result-bearing arms must use the Transformers backend")
    if model.get("transformers_version") != "5.13.0":
        raise ValueError("the manual Qwen forward is pinned to Transformers 5.13.0")
    if model.get("dtype") != "bfloat16":
        raise ValueError("all model-bearing arms are frozen to bfloat16")
    if model.get("attention_implementation") != "sdpa":
        raise ValueError("the K=1 parity contract is frozen to SDPA")
    if model.get("use_cache") is not False:
        raise ValueError("arbitrary repeated middle-layer passes require use_cache=false")

    architecture = config["architecture"]
    start, end = int(architecture["loop_start"]), int(architecture["loop_end"])
    layers = int(architecture["expected_num_layers"])
    pattern = list(architecture["expected_layer_pattern"])
    if not (0 < start < end < layers):
        raise ValueError("loop boundaries must leave non-empty prelude and coda")
    if len(pattern) != 4 or (start % len(pattern)) or (end % len(pattern)):
        raise ValueError("loop boundaries must align to complete four-layer Qwen motifs")
    if end - start < len(pattern):
        raise ValueError("the recurrent block must contain at least one complete motif")
    if int(architecture["state_slots"]) < 2:
        raise ValueError("state_slots must be at least two")
    if int(architecture["max_recurrence"]) < max(config["evaluation"]["k_values"]):
        raise ValueError("max_recurrence must cover every evaluated K")
    if architecture["semantic_echo"]["mode"] != "continuous":
        raise ValueError("continuous semantic echo is the only registered mode")
    if architecture["lora"].get("target_loop_only") is not True:
        raise ValueError("LoRA must remain confined to the recurrent block")

    substrate = config["substrate"]
    train_families = set(substrate["train_families"])
    if substrate["heldout_family"] in train_families:
        raise ValueError("held-out family cannot be a training family")
    if substrate["heldout_template"] in set(substrate["train_templates"]):
        raise ValueError("held-out surface template cannot be a training template")
    train_depths = list(map(int, substrate["train_depths"]))
    extrapolation = list(map(int, substrate["extrapolation_depths"]))
    if max(train_depths) >= min(extrapolation):
        raise ValueError("depth extrapolation must begin strictly beyond training depth")
    if int(config["training"]["train_k"]) != max(train_depths):
        raise ValueError("train_k must equal the maximum trained semantic depth")
    if int(config["training"]["batch_size"]) != 1:
        raise ValueError("the implemented training path is frozen to microbatch one")
    if int(substrate["num_choices"]) != 4:
        raise ValueError("the frozen answer interface is four single-token letters")
    required_data_seeds = {
        "train",
        "validation",
        "depth",
        "family",
        "template",
        "joint",
        "counterfactual",
        "pilot_depth",
        "pilot_joint",
        "pilot_counterfactual",
        "pilot_validation",
    }
    data_seeds = substrate["seeds"]
    if set(data_seeds) != required_data_seeds:
        raise ValueError("every registered data split must have exactly one explicit seed")
    if len(set(map(int, data_seeds.values()))) != len(required_data_seeds):
        raise ValueError("all registered data-split seeds must be pairwise distinct")
    balanced_counts = (
        "train_examples",
        "validation_examples",
        "evaluation_examples_per_split",
        "pilot_examples_per_split",
        "counterfactual_pairs",
        "pilot_counterfactual_pairs",
        "pilot_validation_examples",
    )
    for key in balanced_counts:
        if int(substrate[key]) <= 0 or int(substrate[key]) % 2:
            raise ValueError(f"{key} must be a positive even count for query balance")

    training = config["training"]
    registered_train_seeds = list(map(int, training["train_seeds"]))
    if registered_train_seeds != [7411, 7412, 7413]:
        raise ValueError("train_seeds are frozen to [7411, 7412, 7413]")
    if int(training["pilot_seed"]) != 7401:
        raise ValueError("the independent pilot seed is frozen to 7401")
    if int(training["pilot_seed"]) in registered_train_seeds:
        raise ValueError("the pilot seed must be disjoint from full training seeds")

    evaluation = config["evaluation"]
    if evaluation.get("require_same_backend") is not True:
        raise ValueError("backend equality is a mandatory scientific invariant")
    if evaluation.get("sample_more_compute_unit") != "decoder_layer_token_applications":
        raise ValueError("the preregistered compute unit cannot be changed in-place")
    if int(evaluation["bootstrap_resamples"]) < 10000:
        raise ValueError("paired bootstrap must use at least 10,000 resamples")
    sample_more = evaluation["sample_more"]
    frozen_sample_more = {
        "desired_tokens_per_transition": 24,
        "fixed_overhead_tokens": 32,
        "max_new_tokens": 512,
        "max_samples": 8,
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
    }
    if dict(sample_more) != frozen_sample_more:
        raise ValueError("sample-more allocation and sampling parameters are frozen")

    gates = config["gates"]
    frozen_gates = {
        "exact_parameter_count_match": True,
        "min_state_joint_accuracy": 0.40,
        "min_edge_cut_gain": 0.0,
        "min_joint_holdout_carry_minus_bag": 0.0,
        "min_carry_answer_mode_rate": 0.95,
        "min_sample_more_parse_rate": 0.95,
        "max_sample_more_cap_contact_rate": 0.05,
        "require_paired_lower_bound_above_zero": True,
        "require_unseen_k_gain": True,
        "require_sample_more_win_for_deployable_claim": True,
    }
    for key, expected in frozen_gates.items():
        if gates[key] != expected:
            raise ValueError(f"{key} is frozen to {expected}")
    if config.get("evidence_profile") == "confirmatory" and not is_confirmatory_config(config):
        raise ValueError(
            "confirmatory evidence requires the exact frozen default configuration"
        )


def canonical_json(config: Mapping[str, Any]) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def config_sha256(config: Mapping[str, Any]) -> str:
    return _config_digest(config)


def source_contract_sha256(root: str | Path | None = None) -> str:
    """Hash the versioned, result-bearing runtime implementation.

    The explicit allowlist makes source changes fail closed without allowing
    caches, reports, or generated artifacts to perturb the contract.
    """
    experiment_root = (
        Path(root).resolve()
        if root is not None
        else Path(__file__).resolve().parents[1]
    )
    files = []
    for relative_path in SOURCE_CONTRACT_FILES:
        path = experiment_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"source-contract file is missing: {path}")
        files.append(
            {
                "path": relative_path,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
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
        "loop_layers": [
            config["architecture"]["loop_start"],
            config["architecture"]["loop_end"],
        ],
        "train_depths": config["substrate"]["train_depths"],
        "extrapolation_depths": config["substrate"]["extrapolation_depths"],
        "train_families": config["substrate"]["train_families"],
        "heldout_family": config["substrate"]["heldout_family"],
    }
