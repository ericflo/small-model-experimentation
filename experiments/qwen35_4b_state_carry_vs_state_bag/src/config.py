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
    if architecture["semantic_echo"]["mode"] not in {"continuous", "mixed"}:
        raise ValueError("semantic echo mode must be continuous or mixed")
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

    evaluation = config["evaluation"]
    if evaluation.get("require_same_backend") is not True:
        raise ValueError("backend equality is a mandatory scientific invariant")
    if evaluation.get("sample_more_compute_unit") != "decoder_layer_token_applications":
        raise ValueError("the preregistered compute unit cannot be changed in-place")
    if int(evaluation["bootstrap_resamples"]) < 10000:
        raise ValueError("paired bootstrap must use at least 10,000 resamples")


def canonical_json(config: Mapping[str, Any]) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def config_sha256(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def resolved_config_receipt(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "schema_version": config["schema_version"],
        "config_sha256": config_sha256(config),
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
