"""Shared semantic validation for matched-control training receipts."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import torch

from io_utils import sha256_file
from model_provenance import safe_existing_path, validate_model_checkpoint
from training_units import (
    fit_prompt_around_completion,
    offpolicy_prompt_and_completion,
    prompt_and_student_completion,
)


CONTROL_ARMS = ("non_advantage_route", "wrong_teacher", "offpolicy_sft")


def validate_parameter_control_model(
    model: Path,
    *,
    quick_adapter: Path,
    deep_adapter: Path,
    expected_deep_weight: float,
    expected_model_id: str,
    expected_revision: str,
) -> dict[str, Any]:
    """Authenticate one weighted-adapter control and every inference-artifact byte."""

    provenance = validate_model_checkpoint(
        model,
        profile="source",
        require_recorded_inference_inventory=True,
    )
    model = Path(provenance["model"])
    quick_adapter = safe_existing_path(
        quick_adapter, label="quick adapter directory", directory=True
    )
    deep_adapter = safe_existing_path(
        deep_adapter, label="deep adapter directory", directory=True
    )
    quick_config = safe_existing_path(
        quick_adapter / "adapter_config.json",
        label="quick adapter config",
        directory=False,
    )
    quick_weights = safe_existing_path(
        quick_adapter / "adapter_model.safetensors",
        label="quick adapter weights",
        directory=False,
    )
    deep_config = safe_existing_path(
        deep_adapter / "adapter_config.json",
        label="deep adapter config",
        directory=False,
    )
    deep_weights = safe_existing_path(
        deep_adapter / "adapter_model.safetensors",
        label="deep adapter weights",
        directory=False,
    )
    receipt = provenance["receipt"]

    deep_weight = receipt.get("deep_weight")
    applied = receipt.get("applied_lora_modules")
    nonzero = receipt.get("nonzero_lora_modules")
    norm_sum = receipt.get("delta_frobenius_norm_sum")
    if (
        receipt.get("method") != "explicit_convex_lora_delta_merge"
        or receipt.get("model") != expected_model_id
        or receipt.get("revision") != expected_revision
        or receipt.get("quick_adapter") != str(quick_adapter)
        or receipt.get("deep_adapter") != str(deep_adapter)
        or receipt.get("quick_config_sha256") != sha256_file(quick_config)
        or receipt.get("quick_weights_sha256") != sha256_file(quick_weights)
        or receipt.get("deep_config_sha256") != sha256_file(deep_config)
        or receipt.get("deep_weights_sha256") != sha256_file(deep_weights)
        or type(deep_weight) not in (int, float)
        or float(deep_weight) != float(expected_deep_weight)
        or type(applied) is not int
        or applied < 1
        or type(nonzero) is not int
        or nonzero != applied
        or type(norm_sum) not in (int, float)
        or not math.isfinite(float(norm_sum))
        or float(norm_sum) <= 0.0
    ):
        raise ValueError("parameter-control merge provenance is stale")

    return receipt


def _load_round_manifest(
    source_manifest: Path,
    *,
    receipt: Mapping[str, Any],
    round_index: int,
) -> dict[str, Any]:
    source_manifest = Path(source_manifest)
    if not source_manifest.is_file() or source_manifest.is_symlink():
        raise ValueError("control source round manifest is missing or unsafe")
    try:
        manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("control source round manifest is unreadable") from exc
    if not isinstance(manifest, dict):
        raise ValueError("control source round manifest is malformed")
    if (
        manifest.get("stage") != "online_advantage_training_round"
        or int(manifest.get("round", -1)) != int(round_index)
        or manifest.get("config_sha256") != receipt.get("config_sha256")
    ):
        raise ValueError("control source round manifest provenance mismatch")
    return manifest


def _canonical_mopd_ledger(
    *,
    receipt: Mapping[str, Any],
    config: Mapping[str, Any],
    arm: str,
    source_manifest: Path,
    target_cache: Path,
    round_index: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Replay the exact dense-control assignment and training order."""

    _load_round_manifest(
        source_manifest, receipt=receipt, round_index=round_index
    )
    target_cache = Path(target_cache)
    if not target_cache.is_file() or target_cache.is_symlink():
        raise ValueError("control target cache is missing or unsafe")
    if (
        Path(str(receipt.get("target_cache", ""))).resolve()
        != target_cache.resolve()
        or receipt.get("target_cache_sha256") != sha256_file(target_cache)
    ):
        raise ValueError("control target cache receipt binding mismatch")
    try:
        payload = torch.load(target_cache, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("control target cache is unreadable") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("control target cache payload is malformed")
    if (
        payload.get("round_manifest_sha256") != sha256_file(source_manifest)
        or int(payload.get("round", -1)) != int(round_index)
        or payload.get("config_sha256") != receipt.get("config_sha256")
    ):
        raise ValueError("control target cache is not bound to the source manifest")
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise ValueError("control target cache lacks its sample inventory")

    capability_role = "route_control" if arm == "non_advantage_route" else "capability"
    selected = [
        sample
        for sample in samples
        if isinstance(sample, Mapping)
        and isinstance(sample.get("meta"), Mapping)
        and sample["meta"].get("role") in {capability_role, "anchor"}
    ]
    cfg = config["mopd"]
    expected_units = int(cfg["updates_per_round"]) * int(cfg["grad_accum"])
    if len(selected) != expected_units or len(
        {str(sample.get("id")) for sample in selected}
    ) != expected_units:
        raise ValueError("control target cache has the wrong consume-once inventory")

    units: list[dict[str, Any]] = []
    for sample in selected:
        sample_id = sample.get("id")
        meta = sample["meta"]
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("control target cache has a malformed sample ID")
        if meta.get("role") == "anchor":
            target = "soup"
        elif arm == "wrong_teacher":
            target = "quick"
        else:
            target = "deep"
        targets = sample.get("targets")
        if not isinstance(targets, Mapping) or target not in targets:
            raise ValueError(f"control target {target} is absent for {sample_id}")
        positions = sample.get("positions")
        if not isinstance(positions, torch.Tensor):
            raise ValueError(f"control target positions are malformed for {sample_id}")
        units.append({"sample": sample, "target": target})
    random.Random(int(seed) + int(round_index) * 1000).shuffle(units)
    return [
        {
            "micro_step": index + 1,
            "sample_id": str(unit["sample"]["id"]),
            "target": str(unit["target"]),
            "role": str(unit["sample"]["meta"]["role"]),
            "kind": str(unit["sample"]["meta"]["kind"]),
            "level": int(unit["sample"]["meta"]["level"]),
            "prompt_tokens_truncated": int(
                unit["sample"]["meta"].get("prompt_tokens_truncated", 0)
            ),
            "target_positions": int(unit["sample"]["positions"].numel()),
        }
        for index, unit in enumerate(units)
    ]


def _canonical_offpolicy_ledger(
    *,
    receipt: Mapping[str, Any],
    config: Mapping[str, Any],
    source_manifest: Path,
    base_model: Path,
    round_index: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Replay exact tokenizer-dependent off-policy unit construction and order."""

    manifest = _load_round_manifest(
        source_manifest, receipt=receipt, round_index=round_index
    )
    source_manifest = Path(source_manifest)
    base_model = Path(base_model)
    if (
        Path(str(receipt.get("round_manifest", ""))).resolve()
        != source_manifest.resolve()
        or receipt.get("round_manifest_sha256") != sha256_file(source_manifest)
    ):
        raise ValueError("off-policy round-manifest receipt binding mismatch")
    merge_receipt = base_model / "merge_receipt.json"
    if (
        not base_model.is_dir()
        or base_model.is_symlink()
        or Path(str(receipt.get("base_model", ""))).resolve()
        != base_model.resolve()
        or not merge_receipt.is_file()
        or receipt.get("base_merge_receipt_sha256") != sha256_file(merge_receipt)
    ):
        raise ValueError("off-policy base-model receipt binding mismatch")
    units_source = manifest.get("units")
    cfg = config["mopd"]
    expected_units = int(cfg["updates_per_round"]) * int(cfg["grad_accum"])
    if not isinstance(units_source, list) or len(units_source) != expected_units:
        raise ValueError("off-policy manifest has the wrong unit inventory")

    # Keep Transformers optional for CPU-only import and test discovery.  The
    # actual audit loads only the tokenizer from the exact local round base.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        local_files_only=True,
        trust_remote_code=True,
        use_fast=True,
    )
    units: list[dict[str, Any]] = []
    for row in units_source:
        if not isinstance(row, Mapping):
            raise ValueError("off-policy manifest contains a malformed unit")
        role = row.get("role")
        if role == "capability":
            prompt, completion, active = offpolicy_prompt_and_completion(row, tokenizer)
            target = row.get("offpolicy_target")
            if not isinstance(target, Mapping):
                raise ValueError("off-policy capability target is malformed")
            target_policy = str(target["policy"])
            terminal_score = float(target["terminal_score"])
        elif role == "anchor":
            prompt, completion, active = prompt_and_student_completion(row, tokenizer)
            target_policy = "student_anchor"
            terminal_score = float(row["state"]["student_terminal_score"])
        else:
            raise ValueError(f"unknown off-policy role: {role!r}")
        positions = active[-int(cfg["max_target_positions"]):]
        _, prompt_tokens_truncated = fit_prompt_around_completion(
            prompt,
            completion,
            max_length=int(cfg["max_length"]),
            state_id=str(row["state_id"]),
        )
        if prompt_tokens_truncated:
            raise ValueError("off-policy canonical unit would truncate its prefix")
        units.append(
            {
                "id": str(row["state_id"]),
                "role": str(role),
                "kind": str(row["kind"]),
                "target_policy": target_policy,
                "target_terminal_score": terminal_score,
                "prompt_tokens_truncated": prompt_tokens_truncated,
                "positions": positions,
            }
        )
    if len({unit["id"] for unit in units}) != expected_units:
        raise ValueError("off-policy units are not consume-once unique")
    random.Random(int(seed) + int(round_index) * 1000).shuffle(units)
    return [
        {
            "micro_step": index + 1,
            "sample_id": unit["id"],
            "role": unit["role"],
            "kind": unit["kind"],
            "target_policy": unit["target_policy"],
            "prompt_tokens_truncated": int(unit["prompt_tokens_truncated"]),
            "target_positions": len(unit["positions"]),
            "target_terminal_score": unit["target_terminal_score"],
        }
        for index, unit in enumerate(units)
    ]


def _finite_float(value: object, field: str, *, positive: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"control receipt {field} is not numeric") from exc
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"control receipt {field} must be {qualifier}")
    return result


def _exact_counts(values: list[str], expected: Mapping[str, int], field: str) -> None:
    observed = Counter(values)
    if observed != Counter({str(key): int(value) for key, value in expected.items()}):
        raise ValueError(
            f"control receipt {field} mismatch: "
            f"observed={dict(sorted(observed.items()))} "
            f"expected={dict(sorted(expected.items()))}"
        )


def _probe(
    receipt: Mapping[str, Any],
    *,
    ledger: list[Mapping[str, Any]],
    arm: str,
) -> None:
    probe = receipt.get("pressure_probe")
    if not isinstance(probe, Mapping):
        raise ValueError("control receipt lacks the frozen pressure probe")
    unit_ids = probe.get("unit_ids")
    if not isinstance(unit_ids, list) or not all(
        isinstance(value, str) and value for value in unit_ids
    ):
        raise ValueError("control pressure-probe unit IDs are malformed")

    ordered = sorted(ledger, key=lambda row: str(row["sample_id"]))
    capability_role = "route_control" if arm == "non_advantage_route" else "capability"
    capability = [
        str(row["sample_id"]) for row in ordered if row["role"] == capability_role
    ][:6]
    anchors = [str(row["sample_id"]) for row in ordered if row["role"] == "anchor"][:2]
    expected_ids = capability + anchors
    if unit_ids != expected_ids or len(set(unit_ids)) != 8:
        raise ValueError("control pressure-probe identity/geometry mismatch")

    if arm == "offpolicy_sft":
        expected_geometry = "6_capability_2_anchor"
        expected_roles = {"capability": 6, "anchor": 2}
        expected_targets = {"deep": 6, "student_anchor": 2}
    else:
        expected_geometry = "6_teacher_2_anchor"
        expected_roles = {
            "capability": 6 if arm == "wrong_teacher" else 0,
            "route_control": 6 if arm == "non_advantage_route" else 0,
            "anchor": 2,
        }
        expected_targets = {
            "quick": 6 if arm == "wrong_teacher" else 0,
            "deep": 6 if arm == "non_advantage_route" else 0,
            "soup": 2,
        }
    if probe.get("geometry") != expected_geometry:
        raise ValueError("control pressure-probe geometry tag mismatch")
    if probe.get("role_counts") != expected_roles:
        raise ValueError("control pressure-probe role counts mismatch")
    if probe.get("target_counts") != expected_targets:
        raise ValueError("control pressure-probe target counts mismatch")

    for field in ("initial_probe", "final_probe"):
        value = receipt.get(field)
        if not isinstance(value, Mapping) or int(value.get("unit_count", -1)) != 8:
            raise ValueError(f"control receipt {field} geometry mismatch")
        _finite_float(
            value.get("mean_loss"),
            f"{field}.mean_loss",
            positive=field == "initial_probe",
        )


def validate_control_training_receipt(
    receipt: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    arm: str,
    expected_target_initial_loss: float,
    source_manifest: Path,
    round_index: int,
    seed: int,
    target_cache: Path | None = None,
    base_model: Path | None = None,
    require_completed_updates: bool = True,
) -> None:
    """Fail closed unless one control receipt realizes the frozen design exactly."""

    if arm not in CONTROL_ARMS:
        raise ValueError(f"unknown matched-control arm: {arm}")
    expected_method = (
        "offpolicy_best_selection_continuation_sft"
        if arm == "offpolicy_sft"
        else "deep_advantage_routed_corrected_teacher_topk_reverse_kl"
    )
    expected_schema = 1 if arm == "offpolicy_sft" else 2
    if (
        int(receipt.get("schema_version", -1)) != expected_schema
        or receipt.get("method") != expected_method
        or receipt.get("arm") != arm
        or int(receipt.get("round", -1)) != int(round_index)
        or int(receipt.get("seed", -1)) != int(seed)
    ):
        raise ValueError("control receipt method/arm schema mismatch")

    cfg = config["mopd"]
    updates = int(cfg["updates_per_round"])
    grad_accum = int(cfg["grad_accum"])
    expected_units = updates * grad_accum
    capability_units = int(cfg["capability_units_per_round"])
    anchor_units = int(cfg["anchor_units_per_round"])
    if expected_units != capability_units + anchor_units:
        raise ValueError("frozen control unit geometry is internally inconsistent")
    completed_updates = int(receipt.get("completed_updates", -1))
    if (
        int(receipt.get("requested_updates", -1)) != updates
        or not 0 <= completed_updates <= updates
        or (require_completed_updates and completed_updates != updates)
        or int(receipt.get("consume_once_units", -1)) != expected_units
        or receipt.get("consume_once_verified") is not True
    ):
        raise ValueError("control receipt update/consume-once geometry mismatch")
    round_gate = receipt.get("round_gate")
    if not isinstance(round_gate, Mapping) or bool(
        round_gate.get("completed_all_updates")
    ) != (completed_updates == updates):
        raise ValueError("control receipt completed-update gate is inconsistent")

    ledger = receipt.get("unit_ledger")
    if not isinstance(ledger, list) or len(ledger) != expected_units or not all(
        isinstance(row, Mapping) for row in ledger
    ):
        raise ValueError("control receipt unit ledger has the wrong size")
    if [int(row.get("micro_step", -1)) for row in ledger] != list(
        range(1, expected_units + 1)
    ):
        raise ValueError("control receipt micro-step inventory is not contiguous")
    sample_ids = [row.get("sample_id") for row in ledger]
    if not all(isinstance(value, str) and value for value in sample_ids) or len(
        set(sample_ids)
    ) != expected_units:
        raise ValueError("control receipt sample IDs are not consume-once unique")
    if any(int(row.get("prompt_tokens_truncated", -1)) != 0 for row in ledger):
        raise ValueError("control receipt contains a truncated training prefix")
    maximum_positions = int(cfg["max_target_positions"])
    if any(
        not 1 <= int(row.get("target_positions", -1)) <= maximum_positions
        for row in ledger
    ):
        raise ValueError("control receipt target-position geometry mismatch")

    if arm == "offpolicy_sft":
        if target_cache is not None or base_model is None:
            raise ValueError("off-policy canonical replay inputs are incomplete")
        canonical_ledger = _canonical_offpolicy_ledger(
            receipt=receipt,
            config=config,
            source_manifest=source_manifest,
            base_model=base_model,
            round_index=round_index,
            seed=seed,
        )
    else:
        if target_cache is None or base_model is not None:
            raise ValueError("MOPD canonical replay inputs are incomplete")
        canonical_ledger = _canonical_mopd_ledger(
            receipt=receipt,
            config=config,
            arm=arm,
            source_manifest=source_manifest,
            target_cache=target_cache,
            round_index=round_index,
            seed=seed,
        )
    if ledger != canonical_ledger:
        raise ValueError(
            "control receipt unit ledger differs from canonical source replay"
        )

    capability_role = "route_control" if arm == "non_advantage_route" else "capability"
    _exact_counts(
        [str(row.get("role")) for row in ledger],
        {capability_role: capability_units, "anchor": anchor_units},
        "role counts",
    )
    if arm == "non_advantage_route":
        expected_targets = {"quick": 0, "deep": capability_units, "soup": anchor_units}
        target_field = "target"
    elif arm == "wrong_teacher":
        expected_targets = {"quick": capability_units, "deep": 0, "soup": anchor_units}
        target_field = "target"
    else:
        expected_targets = {"deep": capability_units, "student_anchor": anchor_units}
        target_field = "target_policy"
    _exact_counts(
        [str(row.get(target_field)) for row in ledger],
        expected_targets,
        "target counts",
    )
    if receipt.get("target_counts") != expected_targets:
        raise ValueError("control receipt top-level target counts mismatch")

    if arm != "offpolicy_sft":
        assignment_payload = sorted(
            (str(row["sample_id"]), str(row["target"]))
            for row in canonical_ledger
        )
        expected_assignment = hashlib.sha256(
            json.dumps(assignment_payload, separators=(",", ":")).encode()
        ).hexdigest()
        if receipt.get("assignment_sha256") != expected_assignment:
            raise ValueError("control receipt target assignment hash mismatch")
    else:
        expected_assignment = hashlib.sha256(
            json.dumps(
                canonical_ledger, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        if receipt.get("assignment_sha256") != expected_assignment:
            raise ValueError("off-policy control unit-ledger hash mismatch")

    expected_pressure = _finite_float(
        expected_target_initial_loss,
        "expected_target_initial_loss",
        positive=True,
    )
    recorded_pressure = _finite_float(
        receipt.get("target_initial_loss"), "target_initial_loss", positive=True
    )
    if recorded_pressure != expected_pressure:
        raise ValueError("control receipt target initial pressure mismatch")
    initial_loss = _finite_float(
        (receipt.get("initial_probe") or {}).get("mean_loss"),
        "initial_probe.mean_loss",
        positive=True,
    )
    expected_scale = expected_pressure / initial_loss
    recorded_scale = _finite_float(
        receipt.get("backward_loss_scale"), "backward_loss_scale", positive=True
    )
    if recorded_scale != expected_scale:
        raise ValueError("control receipt backward loss scale mismatch")

    _probe(receipt, ledger=canonical_ledger, arm=arm)
