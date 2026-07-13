#!/usr/bin/env python3
"""Measure NF4 training-surrogate versus merged-bf16 parity on fixed probes.

This is a post-registration, interpretation-only diagnostic.  It is
deliberately standalone and emits no downstream authorization.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch


EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
sys.path.insert(0, str(EXP / "src"))

from io_utils import (  # noqa: E402
    load_config,
    resolve_repo_path,
    sha256_file,
    validate_policy_cache_provenance,
)
from mopd_loss import sparse_teacher_topk_reverse_kl  # noqa: E402
from precision_parity import (  # noqa: E402
    DIAGNOSTIC_ROUNDS,
    DIAGNOSTIC_SEED,
    PROBE_TARGET_COUNTS,
    canonical_tensor_sha256,
    endpoint_logit_metrics,
    mean_numeric_metrics,
    position_objective_metrics,
    probe_identity_sha256,
    replay_comparison,
    select_registered_probe_units,
    summarize_objective_rows,
    update_logit_metrics,
)


PROTOCOL = EXP / "reports" / "nf4_bf16_parity_protocol.md"
DEFAULT_OUT = EXP / "analysis" / "nf4_bf16_parity_seed42.json"
TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
)
UPDATE_BOOLEAN_KEYS = {
    "nf4_update_degenerate",
    "bf16_update_degenerate",
    "update_cosine_defined",
    "update_norm_ratio_defined",
}
UPDATE_NULLABLE_KEYS = {
    "update_cosine_similarity",
    "bf16_to_nf4_update_norm_ratio",
}
SOURCE_FILES = (
    PROTOCOL,
    Path(__file__).resolve(),
    EXP / "src" / "precision_parity.py",
    EXP / "src" / "mopd_loss.py",
    EXP / "src" / "training_units.py",
    EXP / "src" / "io_utils.py",
    EXP / "scripts" / "cache_policy_targets.py",
    EXP / "scripts" / "train_mopd_round.py",
    EXP / "scripts" / "merge_adapter.py",
    EXP / "tests" / "test_precision_parity.py",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise SystemExit(f"missing {description}: {path}")
    return path.resolve()


def _require_dir(path: Path, description: str) -> Path:
    if not path.is_dir():
        raise SystemExit(f"missing {description}: {path}")
    return path.resolve()


def _path_equal(recorded: Any, expected: Path) -> bool:
    try:
        return Path(str(recorded)).resolve() == expected.resolve()
    except (TypeError, ValueError):
        return False


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def _exact_tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    header = json.dumps(
        {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _adapter_state_structure(
    state: Mapping[str, torch.Tensor],
) -> list[dict[str, Any]]:
    rows = []
    for key in sorted(state):
        tensor = state[key]
        if not isinstance(tensor, torch.Tensor):
            raise SystemExit(f"adapter state contains a non-tensor value: {key}")
        rows.append(
            {
                "key": str(key),
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
            }
        )
    if not rows:
        raise SystemExit("adapter state inventory is empty")
    return rows


def _validate_adapter_attachment(
    expected_structure: list[dict[str, Any]],
    saved_state: Mapping[str, torch.Tensor],
    loaded_state: Mapping[str, torch.Tensor],
    load_result: Any,
) -> dict[str, Any]:
    """Prove every saved LoRA tensor reached the already-attached PEFT modules."""

    saved_structure = _adapter_state_structure(saved_state)
    loaded_structure = _adapter_state_structure(loaded_state)
    if saved_structure != expected_structure:
        raise SystemExit("saved adapter does not match the expected PEFT key/shape/dtype inventory")
    if loaded_structure != saved_structure:
        raise SystemExit("loaded adapter does not match the saved key/shape/dtype inventory")
    unexpected = sorted(str(key) for key in (getattr(load_result, "unexpected_keys", []) or []))
    missing = sorted(str(key) for key in (getattr(load_result, "missing_keys", []) or []))
    missing_adapter = [key for key in missing if "lora_" in key or "modules_to_save" in key]
    if unexpected:
        raise SystemExit(f"saved adapter has unexpected keys: {unexpected[:3]}")
    if missing_adapter:
        raise SystemExit(f"saved adapter left LoRA keys missing: {missing_adapter[:3]}")

    tensor_rows = []
    for key in sorted(saved_state):
        saved = saved_state[key].detach().to(device="cpu").contiguous()
        loaded = loaded_state[key].detach().to(device="cpu").contiguous()
        saved_sha = _exact_tensor_sha256(saved)
        loaded_sha = _exact_tensor_sha256(loaded)
        exact = bool(torch.equal(saved, loaded))
        if not exact or saved_sha != loaded_sha:
            raise SystemExit(f"loaded adapter tensor differs from saved tensor: {key}")
        tensor_rows.append(
            {
                "key": key,
                "shape": list(saved.shape),
                "dtype": str(saved.dtype),
                "saved_sha256": saved_sha,
                "loaded_sha256": loaded_sha,
                "exact": True,
            }
        )
    return {
        "tensor_count": len(tensor_rows),
        "expected_structure_sha256": _canonical_json_sha256(expected_structure),
        "saved_structure_sha256": _canonical_json_sha256(saved_structure),
        "loaded_structure_sha256": _canonical_json_sha256(loaded_structure),
        "saved_value_inventory_sha256": _canonical_json_sha256(
            [{"key": row["key"], "sha256": row["saved_sha256"]} for row in tensor_rows]
        ),
        "loaded_value_inventory_sha256": _canonical_json_sha256(
            [{"key": row["key"], "sha256": row["loaded_sha256"]} for row in tensor_rows]
        ),
        "unexpected_keys": unexpected,
        "missing_adapter_keys": missing_adapter,
        "non_adapter_missing_key_count": len(missing) - len(missing_adapter),
        "all_tensors_exact": True,
        "tensors": tensor_rows,
    }


def _adapter_file_inventory(path: Path) -> dict[str, Any]:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as tensors:
        keys = sorted(tensors.keys())
    a_suffix = ".lora_A.weight"
    b_suffix = ".lora_B.weight"
    a_cores = {key[: -len(a_suffix)] for key in keys if key.endswith(a_suffix)}
    b_cores = {key[: -len(b_suffix)] for key in keys if key.endswith(b_suffix)}
    recognized = len(a_cores) + len(b_cores)
    if not keys or recognized != len(keys) or a_cores != b_cores:
        raise SystemExit(f"adapter A/B key inventory is incomplete or contains extras: {path}")
    return {
        "tensor_count": len(keys),
        "module_count": len(a_cores),
        "key_inventory_sha256": _canonical_json_sha256(keys),
    }


def _validated_shard_index(path: Path, weight_names: set[str]) -> dict[str, Any] | None:
    index_path = path / "model.safetensors.index.json"
    if not index_path.is_file():
        if len(weight_names) > 1:
            raise SystemExit(f"sharded model is missing model.safetensors.index.json: {path}")
        return None
    index = _read_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise SystemExit(f"invalid safetensors shard index: {index_path}")
    mapped_names = {str(value) for value in weight_map.values()}
    if mapped_names != weight_names or any(
        not isinstance(key, str) or not key for key in weight_map
    ):
        raise SystemExit(f"safetensors shard-index inventory mismatch: {index_path}")
    return {
        "path": str(index_path.resolve()),
        "sha256": sha256_file(index_path),
        "parameter_count": len(weight_map),
        "weight_files": sorted(mapped_names),
    }


def _committed_source_receipt() -> dict[str, Any]:
    """Require every diagnostic-defining source to match the current commit."""

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise SystemExit("diagnostic requires a readable git HEAD") from error
    rows = []
    for path in SOURCE_FILES:
        path = _require_file(path, "diagnostic source")
        relative = path.relative_to(REPO).as_posix()
        try:
            committed = subprocess.run(
                ["git", "show", f"HEAD:{relative}"],
                cwd=REPO,
                check=True,
                capture_output=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as error:
            raise SystemExit(
                f"diagnostic source is not committed at HEAD: {relative}"
            ) from error
        worktree_sha = sha256_file(path)
        committed_sha = _sha256_bytes(committed)
        if worktree_sha != committed_sha:
            raise SystemExit(
                f"diagnostic source differs from HEAD; commit before execution: {relative}"
            )
        rows.append(
            {
                "path": relative,
                "sha256": worktree_sha,
                "committed_blob_sha256": committed_sha,
                "matches_head": True,
            }
        )
    return {"git_commit": commit, "files": rows, "all_match_head": True}


def _verified_model_provenance(path: Path) -> dict[str, Any]:
    path = path.resolve()
    config_path = _require_file(path / "config.json", "model config")
    merge_path = _require_file(path / "merge_receipt.json", "model merge receipt")
    merge = _read_json(merge_path)
    recorded = merge.get("weight_files")
    if not isinstance(recorded, list) or not recorded:
        raise SystemExit(f"merge receipt has no weight inventory: {merge_path}")
    weights = []
    names = set()
    for row in recorded:
        if not isinstance(row, dict) or not row.get("name") or not row.get("sha256"):
            raise SystemExit(f"invalid weight inventory row: {merge_path}")
        name = str(row["name"])
        if name in names or Path(name).name != name or not name.endswith(".safetensors"):
            raise SystemExit(f"invalid or duplicate model weight name: {name}")
        names.add(name)
        weight_path = _require_file(path / name, "model weight")
        actual = sha256_file(weight_path)
        if actual != str(row["sha256"]):
            raise SystemExit(f"model weight hash mismatch: {weight_path}")
        weights.append(
            {"name": name, "sha256": actual, "bytes": weight_path.stat().st_size}
        )
    actual_names = {value.name for value in path.glob("*.safetensors")}
    if actual_names != names:
        raise SystemExit(
            f"model weight inventory mismatch at {path}: {actual_names} != {names}"
        )
    shard_index = _validated_shard_index(path, names)
    return {
        "path": str(path),
        "config_sha256": sha256_file(config_path),
        "merge_receipt_sha256": sha256_file(merge_path),
        "merge_method": merge.get("method"),
        "weights": weights,
        "shard_index": shard_index,
    }


def _round_paths(config: dict[str, Any], round_index: int) -> dict[str, Path]:
    root = resolve_repo_path(config["model"]["artifacts_root"]).resolve()
    initial = resolve_repo_path(config["model"]["student_checkpoint"]).resolve()
    base = (
        initial
        if round_index == 0
        else root / "merged" / "primary" / f"seed_{DIAGNOSTIC_SEED}" / f"round_{round_index - 1}"
    )
    adapter = (
        root / "adapters" / "primary" / f"seed_{DIAGNOSTIC_SEED}"
        / f"round_{round_index}"
    )
    merged = (
        root / "merged" / "primary" / f"seed_{DIAGNOSTIC_SEED}"
        / f"round_{round_index}"
    )
    online = (
        root / "online" / "primary" / f"seed_{DIAGNOSTIC_SEED}"
        / f"round_{round_index}"
    )
    return {
        "base": base.resolve(),
        "adapter": adapter.resolve(),
        "merged": merged.resolve(),
        "cache": (online / "all_policy_targets.pt").resolve(),
        "cache_receipt": (online / "all_policy_targets.pt.receipt.json").resolve(),
        "training_receipt": (adapter / "training_receipt.json").resolve(),
        "adapter_config": (adapter / "adapter_config.json").resolve(),
        "adapter_weights": (adapter / "adapter_model.safetensors").resolve(),
    }


def _validate_round_files(
    config: dict[str, Any], config_path: Path, round_index: int
) -> dict[str, Any]:
    paths = _round_paths(config, round_index)
    for name in ("base", "adapter", "merged"):
        _require_dir(paths[name], f"round-{round_index} {name}")
    for name in (
        "cache", "cache_receipt", "training_receipt", "adapter_config",
        "adapter_weights",
    ):
        path = paths[name]
        _require_file(path, f"round-{round_index} {name}")
    cache_sha = sha256_file(paths["cache"])
    cache_receipt = _read_json(paths["cache_receipt"])
    try:
        validate_policy_cache_provenance(cache_receipt, config, config_path)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if cache_receipt.get("cache_sha256") != cache_sha:
        raise SystemExit(f"round-{round_index} cache checksum mismatch")
    if int(cache_receipt.get("round", -1)) != round_index:
        raise SystemExit(f"round-{round_index} cache round mismatch")
    try:
        round_manifest = Path(str(cache_receipt["round_manifest"])).resolve()
    except (KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"round-{round_index} cache manifest path invalid") from error
    if (
        not round_manifest.is_file()
        or cache_receipt.get("round_manifest_sha256") != sha256_file(round_manifest)
    ):
        raise SystemExit(f"round-{round_index} cache manifest provenance mismatch")

    training = _read_json(paths["training_receipt"])
    required_updates = int(config["mopd"]["updates_per_round"])
    required_units = required_updates * int(config["mopd"]["grad_accum"])
    unit_ledger = list(training.get("unit_ledger") or [])
    unit_ids = [str(row.get("sample_id")) for row in unit_ledger]
    ledger_micro_steps = [int(row.get("micro_step", -1)) for row in unit_ledger]
    if (
        training.get("method")
        != "deep_advantage_routed_corrected_teacher_topk_reverse_kl"
        or training.get("arm") != "primary"
        or training.get("config_sha256") != sha256_file(config_path)
        or int(training.get("seed", -1)) != DIAGNOSTIC_SEED
        or int(training.get("round", -1)) != round_index
        or int(training.get("requested_updates", -1)) != required_updates
        or int(training.get("completed_updates", -1)) != required_updates
        or not bool(training.get("round_gate", {}).get("passed"))
        or not _path_equal(training.get("base_model"), paths["base"])
        or training.get("base_merge_receipt_sha256")
        != sha256_file(paths["base"] / "merge_receipt.json")
        or training.get("target_cache_sha256") != cache_sha
        or len(unit_ledger) != required_units
        or len(unit_ids) != len(set(unit_ids))
        or ledger_micro_steps != list(range(1, required_units + 1))
        or not bool(training.get("consume_once_verified"))
    ):
        raise SystemExit(f"round-{round_index} training receipt contract mismatch")
    expected_counts = {
        "quick": 0,
        "deep": int(config["mopd"]["capability_units_per_round"]),
        "soup": int(config["mopd"]["anchor_units_per_round"]),
    }
    if training.get("target_counts") != expected_counts:
        raise SystemExit(f"round-{round_index} target-count mismatch")
    for endpoint in ("initial_probe", "final_probe"):
        probe = training.get(endpoint) or {}
        if int(probe.get("unit_count", -1)) != sum(PROBE_TARGET_COUNTS.values()):
            raise SystemExit(f"round-{round_index} {endpoint} unit count mismatch")
        losses = probe.get("unit_losses") or []
        if len(losses) != sum(PROBE_TARGET_COUNTS.values()) or not all(
            math.isfinite(float(value)) for value in losses
        ):
            raise SystemExit(f"round-{round_index} {endpoint} losses invalid")

    adapter_config = _read_json(paths["adapter_config"])
    if (
        not _path_equal(adapter_config.get("base_model_name_or_path"), paths["base"])
        or int(adapter_config.get("r", -1)) != int(config["mopd"]["rank"])
        or int(adapter_config.get("lora_alpha", -1)) != int(config["mopd"]["alpha"])
        or abs(float(adapter_config.get("lora_dropout", -1.0)) - 0.05) > 1e-12
        or set(adapter_config.get("target_modules") or []) != set(TARGET_MODULES)
    ):
        raise SystemExit(f"round-{round_index} adapter config mismatch")
    adapter_sha = sha256_file(paths["adapter_weights"])
    adapter_inventory = _adapter_file_inventory(paths["adapter_weights"])
    after_merge = _read_json(paths["merged"] / "merge_receipt.json")
    expected_scale = float(config["mopd"]["alpha"]) / float(config["mopd"]["rank"])
    recorded_scale = float(after_merge.get("scale", float("nan")))
    if (
        after_merge.get("method") != "explicit_composite_lora_merge"
        or not _path_equal(after_merge.get("base_model"), paths["base"])
        or not _path_equal(after_merge.get("adapter"), paths["adapter"])
        or after_merge.get("adapter_config_sha256")
        != sha256_file(paths["adapter_config"])
        or after_merge.get("adapter_weights_sha256") != adapter_sha
        or int(after_merge.get("applied_lora_modules", -1))
        != int(adapter_inventory["module_count"])
        or int(after_merge.get("nonzero_lora_modules", -1))
        != int(adapter_inventory["module_count"])
        or not math.isfinite(recorded_scale)
        or abs(recorded_scale - expected_scale) > 1e-12
    ):
        raise SystemExit(f"round-{round_index} explicit merge provenance mismatch")

    return {
        "paths": paths,
        "cache_sha256": cache_sha,
        "cache_receipt": cache_receipt,
        "training": training,
        "provenance": {
            "round": round_index,
            "base_model": _verified_model_provenance(paths["base"]),
            "merged_model": _verified_model_provenance(paths["merged"]),
            "target_cache": {
                "path": str(paths["cache"]),
                "sha256": cache_sha,
                "bytes": paths["cache"].stat().st_size,
            },
            "target_cache_receipt": {
                "path": str(paths["cache_receipt"]),
                "sha256": sha256_file(paths["cache_receipt"]),
            },
            "round_manifest": {
                "path": str(round_manifest),
                "sha256": sha256_file(round_manifest),
            },
            "training_receipt": {
                "path": str(paths["training_receipt"]),
                "sha256": sha256_file(paths["training_receipt"]),
            },
            "adapter_config": {
                "path": str(paths["adapter_config"]),
                "sha256": sha256_file(paths["adapter_config"]),
            },
            "adapter_weights": {
                "path": str(paths["adapter_weights"]),
                "sha256": adapter_sha,
                "bytes": paths["adapter_weights"].stat().st_size,
                **adapter_inventory,
            },
        },
    }


def _load_round_probes(
    round_info: dict[str, Any], config: dict[str, Any], config_path: Path
) -> list[dict[str, Any]]:
    payload = torch.load(
        round_info["paths"]["cache"], map_location="cpu", weights_only=False
    )
    if not isinstance(payload, dict):
        raise SystemExit("target cache payload is not a mapping")
    try:
        validate_policy_cache_provenance(payload, config, config_path)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if payload.get("models") != round_info["cache_receipt"].get("models"):
        raise SystemExit("target cache payload/receipt model provenance mismatch")
    if int(payload.get("round", -1)) != int(round_info["training"]["round"]):
        raise SystemExit("target cache payload round mismatch")
    samples = list(payload.get("samples") or [])
    probes = select_registered_probe_units(
        samples, list(round_info["training"]["unit_ledger"])
    )
    del payload, samples
    gc.collect()
    return probes


def _validate_integration_receipt(
    config_path: Path, round_infos: list[dict[str, Any]]
) -> dict[str, Any]:
    path = _require_file(
        EXP / "runs" / "integration" / f"seed_{DIAGNOSTIC_SEED}.json",
        "completed seed-42 integration receipt",
    )
    receipt = _read_json(path)
    rows = list(receipt.get("rounds") or [])
    if (
        receipt.get("stage") != "four_round_deep_advantage_routed_mopd"
        or receipt.get("config_sha256") != sha256_file(config_path)
        or int(receipt.get("seed", -1)) != DIAGNOSTIC_SEED
        or int(receipt.get("completed_rounds", -1)) != len(DIAGNOSTIC_ROUNDS)
        or not bool((receipt.get("gate") or {}).get("passed"))
        or len(rows) != len(DIAGNOSTIC_ROUNDS)
        or not _path_equal(
            receipt.get("final_model"), round_infos[-1]["paths"]["merged"]
        )
    ):
        raise SystemExit("seed-42 integration receipt is not complete and authoritative")
    for round_index, (row, info) in enumerate(zip(rows, round_infos)):
        provenance = info["provenance"]
        if (
            int(row.get("round", -1)) != round_index
            or not _path_equal(
                row.get("round_manifest"), Path(provenance["round_manifest"]["path"])
            )
            or row.get("round_manifest_sha256") != provenance["round_manifest"]["sha256"]
            or not _path_equal(row.get("target_cache"), info["paths"]["cache"])
            or row.get("target_cache_sha256") != info["cache_sha256"]
            or not _path_equal(row.get("training_receipt"), info["paths"]["training_receipt"])
            or row.get("training_receipt_sha256")
            != provenance["training_receipt"]["sha256"]
            or not _path_equal(row.get("merged"), info["paths"]["merged"])
            or row.get("merge_receipt_sha256")
            != provenance["merged_model"]["merge_receipt_sha256"]
            or not bool((row.get("round_gate") or {}).get("passed"))
        ):
            raise SystemExit(f"integration receipt round-{round_index} binding mismatch")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "stage": receipt["stage"],
        "seed": DIAGNOSTIC_SEED,
        "round_count": len(rows),
        "gate_passed": True,
    }


def _score_model(
    model: Any, probes: list[dict[str, Any]], *, top_k: int, view_name: str
) -> dict[str, dict[str, Any]]:
    model.eval()
    scored: dict[str, dict[str, Any]] = {}
    with torch.inference_mode():
        for index, unit in enumerate(probes):
            sample = unit["sample"]
            sample_id = str(sample["id"])
            target = sample["targets"][unit["target"]]
            positions = sample["positions"].to(dtype=torch.long).tolist()
            if not positions or positions != sorted(set(positions)):
                raise SystemExit(f"invalid registered target positions: {sample_id}")
            first = int(positions[0])
            end = int(positions[-1]) + 1
            prompt = sample["prompt_ids"].to(dtype=torch.long).tolist()
            completion = sample["completion_ids"].to(dtype=torch.long).tolist()[:end]
            ids = torch.tensor([prompt + completion], dtype=torch.long, device=model.device)
            tail = end - first
            outputs = model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                logits_to_keep=tail + 1,
                use_cache=False,
            )
            prediction = outputs.logits[0, -(tail + 1):-1]
            relative = torch.tensor(
                [position - first for position in positions],
                dtype=torch.long,
                device=model.device,
            )
            selected = prediction.index_select(0, relative)
            objective = sparse_teacher_topk_reverse_kl(
                selected,
                target["indices"],
                target["log_probs"],
                reduction="none",
            ).detach().float().cpu()
            if objective.ndim != 1 or objective.numel() != len(positions):
                raise SystemExit(f"objective shape mismatch for {sample_id}")
            if not bool(torch.isfinite(objective).all()):
                raise SystemExit(f"non-finite objective for {sample_id} in {view_name}")
            midpoint_index = len(positions) // 2
            midpoint_logits = selected[midpoint_index].detach().float().cpu()
            if not bool(torch.isfinite(midpoint_logits).all()):
                raise SystemExit(f"non-finite logits for {sample_id} in {view_name}")
            scored[sample_id] = {
                "objective": objective,
                "objective_mean": float(objective.mean()),
                "midpoint_logits": midpoint_logits,
                "midpoint_position": int(positions[midpoint_index]),
                "target_positions": len(positions),
                "forward_input_tokens": int(ids.shape[1]),
            }
            del ids, outputs, prediction, relative, selected, objective, midpoint_logits
            print(
                f"[nf4-bf16-parity] {view_name}: {index + 1}/{len(probes)}",
                flush=True,
            )
    return scored


def _score_nf4_views(
    base_path: Path,
    adapter_path: Path,
    probes: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    from peft import (
        LoraConfig,
        get_peft_model_state_dict,
        get_peft_model,
        prepare_model_for_kbit_training,
        set_peft_model_state_dict,
    )
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    random.seed(DIAGNOSTIC_SEED)
    torch.manual_seed(DIAGNOSTIC_SEED)
    torch.cuda.manual_seed_all(DIAGNOSTIC_SEED)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        local_files_only=True,
        trust_remote_code=True,
        device_map="cuda",
        dtype=torch.bfloat16,
        quantization_config=bnb,
        attn_implementation="sdpa",
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(config["mopd"]["rank"]),
            lora_alpha=int(config["mopd"]["alpha"]),
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(TARGET_MODULES),
        ),
    )
    model.config.use_cache = False
    expected_state = get_peft_model_state_dict(model, adapter_name="default")
    expected_structure = _adapter_state_structure(expected_state)
    del expected_state
    before = _score_model(
        model, probes, top_k=int(config["mopd"]["top_k"]), view_name="nf4_before"
    )
    adapter_state = load_file(str(adapter_path / "adapter_model.safetensors"), device="cpu")
    load_result = set_peft_model_state_dict(model, adapter_state, adapter_name="default")
    loaded_state = get_peft_model_state_dict(model, adapter_name="default")
    attachment = _validate_adapter_attachment(
        expected_structure, adapter_state, loaded_state, load_result
    )
    del adapter_state, loaded_state
    after = _score_model(
        model, probes, top_k=int(config["mopd"]["top_k"]), view_name="nf4_after"
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return before, after, attachment


def _score_bf16_view(
    model_path: Path,
    probes: list[dict[str, Any]],
    config: dict[str, Any],
    view_name: str,
) -> dict[str, dict[str, Any]]:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        device_map="cuda",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    scored = _score_model(
        model, probes, top_k=int(config["mopd"]["top_k"]), view_name=view_name
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return scored


def _round_measurement(
    round_info: dict[str, Any],
    probes: list[dict[str, Any]],
    views: dict[str, dict[str, dict[str, Any]]],
    config: dict[str, Any],
) -> dict[str, Any]:
    top_k = int(config["mopd"]["top_k"])
    rows = []
    for unit in probes:
        sample = unit["sample"]
        sample_id = str(sample["id"])
        target = sample["targets"][unit["target"]]
        values = {view: views[view][sample_id] for view in views}
        midpoint = int(values["nf4_before"]["midpoint_position"])
        if any(int(value["midpoint_position"]) != midpoint for value in values.values()):
            raise SystemExit(f"midpoint mismatch across views: {sample_id}")
        position_count = int(values["nf4_before"]["target_positions"])
        if any(int(value["target_positions"]) != position_count for value in values.values()):
            raise SystemExit(f"position-count mismatch across views: {sample_id}")
        positions = sample["positions"].to(dtype=torch.long).tolist()
        midpoint_index = len(positions) // 2
        teacher_indices = target["indices"][midpoint_index]
        objective = {
            view: float(values[view]["objective_mean"])
            for view in ("nf4_before", "nf4_after", "bf16_before", "bf16_after")
        }
        objective.update(
            {
                "bf16_minus_nf4_before": objective["bf16_before"]
                - objective["nf4_before"],
                "bf16_minus_nf4_after": objective["bf16_after"]
                - objective["nf4_after"],
                "nf4_gain": objective["nf4_before"] - objective["nf4_after"],
                "bf16_gain": objective["bf16_before"] - objective["bf16_after"],
            }
        )
        objective["bf16_minus_nf4_gain"] = objective["bf16_gain"] - objective["nf4_gain"]
        row = {
            "sample_id": sample_id,
            "role": str(sample["meta"]["role"]),
            "target": str(unit["target"]),
            "target_positions": position_count,
            "midpoint_position": midpoint,
            "objective": objective,
            "objective_vector_sha256": {
                view: canonical_tensor_sha256(values[view]["objective"])
                for view in values
            },
            "position_objective_parity": position_objective_metrics(
                values["nf4_before"]["objective"],
                values["nf4_after"]["objective"],
                values["bf16_before"]["objective"],
                values["bf16_after"]["objective"],
            ),
            "midpoint_logits_sha256": {
                view: canonical_tensor_sha256(values[view]["midpoint_logits"])
                for view in values
            },
            "endpoint_logit_parity": {
                "before": endpoint_logit_metrics(
                    values["nf4_before"]["midpoint_logits"],
                    values["bf16_before"]["midpoint_logits"],
                    teacher_indices,
                    top_k=top_k,
                ),
                "after": endpoint_logit_metrics(
                    values["nf4_after"]["midpoint_logits"],
                    values["bf16_after"]["midpoint_logits"],
                    teacher_indices,
                    top_k=top_k,
                ),
            },
            "update_logit_parity": update_logit_metrics(
                values["nf4_before"]["midpoint_logits"],
                values["nf4_after"]["midpoint_logits"],
                values["bf16_before"]["midpoint_logits"],
                values["bf16_after"]["midpoint_logits"],
            ),
        }
        rows.append(row)

    training = round_info["training"]
    replay_before = replay_comparison(
        [row["objective"]["nf4_before"] for row in rows],
        list(training["initial_probe"]["unit_losses"]),
    )
    replay_after = replay_comparison(
        [row["objective"]["nf4_after"] for row in rows],
        list(training["final_probe"]["unit_losses"]),
    )
    endpoint_before = mean_numeric_metrics(
        [row["endpoint_logit_parity"]["before"] for row in rows],
        boolean_keys={"top1_agreement"},
    )
    endpoint_after = mean_numeric_metrics(
        [row["endpoint_logit_parity"]["after"] for row in rows],
        boolean_keys={"top1_agreement"},
    )
    update = mean_numeric_metrics(
        [row["update_logit_parity"] for row in rows],
        boolean_keys=UPDATE_BOOLEAN_KEYS,
        nullable_keys=UPDATE_NULLABLE_KEYS,
    )
    target_counts = {
        target: sum(row["target"] == target for row in rows)
        for target in PROBE_TARGET_COUNTS
    }
    return {
        "round": int(training["round"]),
        "provenance": round_info["provenance"],
        "probe_identity_sha256": probe_identity_sha256(probes),
        "probe_units": len(rows),
        "target_counts": target_counts,
        "target_positions": sum(int(row["target_positions"]) for row in rows),
        "replay": {"nf4_before": replay_before, "nf4_after": replay_after},
        "diagnostic_validity": {
            "registered_probe_mixture": target_counts == PROBE_TARGET_COUNTS,
            "nf4_before_replays_training_receipt": bool(replay_before["passed"]),
            "nf4_after_replays_training_receipt": bool(replay_after["passed"]),
        },
        "objective_parity": summarize_objective_rows(rows),
        "position_objective_parity": mean_numeric_metrics(
            [row["position_objective_parity"] for row in rows]
        ),
        "midpoint_endpoint_logit_parity": {
            "before": endpoint_before,
            "after": endpoint_after,
        },
        "midpoint_update_logit_parity": update,
        "rows": rows,
    }


def _runtime_receipt() -> dict[str, Any]:
    import bitsandbytes
    import peft
    import transformers

    lock = _require_file(REPO / "requirements-training.lock.txt", "training lock")
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "bitsandbytes": bitsandbytes.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "training_lock": {"path": str(lock), "sha256": sha256_file(lock)},
        "loader_contract": {
            "nf4": {
                "quant_type": "nf4",
                "double_quant": True,
                "compute_dtype": "bfloat16",
                "model_dtype": "bfloat16",
                "attention": "sdpa",
                "prepare_model_for_kbit_training": True,
                "gradient_checkpointing_setup": True,
                "adapter_eval_mode": True,
                "lora_dropout_active": False,
            },
            "bf16": {
                "model_dtype": "bfloat16",
                "attention": "sdpa",
                "explicit_merged_composite": True,
            },
        },
    }


def _artifact_snapshot(
    *,
    source_receipt: dict[str, Any],
    config_path: Path,
    prereg_path: Path,
    integration_receipt: dict[str, Any],
    runtime: dict[str, Any],
    round_infos: list[dict[str, Any]],
) -> dict[str, Any]:
    bound: dict[str, dict[str, Any]] = {}

    def add(path: Path, expected_sha256: str) -> None:
        resolved = _require_file(path, "bound diagnostic artifact")
        key = str(resolved)
        row = {
            "path": key,
            "sha256": str(expected_sha256),
            "bytes": resolved.stat().st_size,
        }
        previous = bound.get(key)
        if previous is not None and previous != row:
            raise SystemExit(f"conflicting hashes for bound artifact: {resolved}")
        bound[key] = row

    for row in source_receipt["files"]:
        add(REPO / row["path"], row["sha256"])
    add(config_path, sha256_file(config_path))
    add(prereg_path, sha256_file(prereg_path))
    add(Path(integration_receipt["path"]), integration_receipt["sha256"])
    add(Path(runtime["training_lock"]["path"]), runtime["training_lock"]["sha256"])
    for info in round_infos:
        provenance = info["provenance"]
        for model_name in ("base_model", "merged_model"):
            model = provenance[model_name]
            model_path = Path(model["path"])
            add(model_path / "config.json", model["config_sha256"])
            add(model_path / "merge_receipt.json", model["merge_receipt_sha256"])
            for weight in model["weights"]:
                add(model_path / weight["name"], weight["sha256"])
            if model["shard_index"] is not None:
                add(
                    Path(model["shard_index"]["path"]),
                    model["shard_index"]["sha256"],
                )
        for name in (
            "target_cache",
            "target_cache_receipt",
            "round_manifest",
            "training_receipt",
            "adapter_config",
            "adapter_weights",
        ):
            add(Path(provenance[name]["path"]), provenance[name]["sha256"])
    rows = sorted(bound.values(), key=lambda row: row["path"])
    return {
        "file_count": len(rows),
        "inventory_sha256": _canonical_json_sha256(rows),
        "files": rows,
    }


def _verify_artifact_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    changed = []
    for row in snapshot["files"]:
        path = Path(row["path"])
        if not path.is_file():
            changed.append({"path": str(path), "reason": "missing"})
            continue
        actual_bytes = path.stat().st_size
        actual_sha = sha256_file(path)
        if actual_bytes != int(row["bytes"]) or actual_sha != row["sha256"]:
            changed.append(
                {
                    "path": str(path),
                    "reason": "content_changed",
                    "expected_bytes": int(row["bytes"]),
                    "actual_bytes": actual_bytes,
                    "expected_sha256": row["sha256"],
                    "actual_sha256": actual_sha,
                }
            )
    if changed:
        raise SystemExit(
            "bound artifacts changed during parity scoring: "
            + json.dumps(changed[:3], sort_keys=True)
        )
    return {
        "file_count": int(snapshot["file_count"]),
        "inventory_sha256": str(snapshot["inventory_sha256"]),
        "verified_unchanged_after_scoring": True,
    }


def _write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
    except FileExistsError as error:
        raise SystemExit(f"refusing to overwrite existing diagnostic receipt: {path}") from error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    requested_out = args.out
    if requested_out.exists() or requested_out.is_symlink():
        raise SystemExit(
            f"refusing to overwrite existing diagnostic receipt: {requested_out}"
        )
    args.out = requested_out.resolve()
    if args.out.exists() or args.out.is_symlink():
        raise SystemExit(f"refusing to overwrite existing diagnostic receipt: {args.out}")
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    source_receipt = _committed_source_receipt()
    config, config_path = load_config(args.config)
    config_path = config_path.resolve()
    prereg_path = _require_file(
        EXP / "runs" / "preregistration_receipt.json", "preregistration receipt"
    )
    prereg = _read_json(prereg_path)
    frozen_config_sha = (prereg.get("frozen_files") or {}).get("configs/default.yaml")
    config_sha = sha256_file(config_path)
    if config_path != (EXP / "configs" / "default.yaml").resolve():
        raise SystemExit("diagnostic is frozen to the experiment default config")
    if config_sha != frozen_config_sha:
        raise SystemExit("default config differs from the preregistered frozen hash")
    if (
        int(config["mopd"]["rounds"]) != len(DIAGNOSTIC_ROUNDS)
        or list(map(int, config["seeds"]["integration_training"]))[:1]
        != [DIAGNOSTIC_SEED]
        or int(config["mopd"]["top_k"]) != 50
    ):
        raise SystemExit("frozen seed/round/top-k contract changed")

    # Validate and hash every required round before allocating the GPU.  This
    # also makes omission of a failed or inconvenient round impossible.
    round_infos = [
        _validate_round_files(config, config_path, round_index)
        for round_index in DIAGNOSTIC_ROUNDS
    ]
    integration_receipt = _validate_integration_receipt(config_path, round_infos)
    runtime = _runtime_receipt()
    artifact_snapshot = _artifact_snapshot(
        source_receipt=source_receipt,
        config_path=config_path,
        prereg_path=prereg_path,
        integration_receipt=integration_receipt,
        runtime=runtime,
        round_infos=round_infos,
    )
    torch.cuda.reset_peak_memory_stats()
    measurements = []
    for round_info in round_infos:
        round_index = int(round_info["training"]["round"])
        print(f"[nf4-bf16-parity] round {round_index}: loading fixed probes", flush=True)
        probes = _load_round_probes(round_info, config, config_path)
        nf4_before, nf4_after, attachment = _score_nf4_views(
            round_info["paths"]["base"],
            round_info["paths"]["adapter"],
            probes,
            config,
        )
        round_info["provenance"]["nf4_adapter_attachment"] = attachment
        bf16_before = _score_bf16_view(
            round_info["paths"]["base"], probes, config, "bf16_before"
        )
        bf16_after = _score_bf16_view(
            round_info["paths"]["merged"], probes, config, "bf16_after"
        )
        measurement = _round_measurement(
            round_info,
            probes,
            {
                "nf4_before": nf4_before,
                "nf4_after": nf4_after,
                "bf16_before": bf16_before,
                "bf16_after": bf16_after,
            },
            config,
        )
        measurements.append(measurement)
        del probes, nf4_before, nf4_after, bf16_before, bf16_after
        gc.collect()
        torch.cuda.empty_cache()

    post_score_artifact_verification = _verify_artifact_snapshot(artifact_snapshot)

    all_rows = [row for measurement in measurements for row in measurement["rows"]]
    all_validity = [
        value
        for measurement in measurements
        for value in measurement["diagnostic_validity"].values()
    ]
    diagnostic_valid = all(all_validity)
    result = {
        "schema_version": 1,
        "stage": "nf4_bf16_training_deployment_parity_diagnostic",
        "status": "interpretation_only",
        "added_after_preregistration": True,
        "scope": (
            "NF4 training surrogate versus exact merged-bf16 checkpoint under "
            "Transformers SDPA; not HF/vLLM kernel parity"
        ),
        "decision_contract": {
            "scientific_measurements_have_gate": False,
            "can_stop_or_rescue_frozen_experiment": False,
            "can_select_seed_round_checkpoint_or_state": False,
            "can_retroactively_reclassify_round_gate": False,
            "downstream_authorization": None,
            "authoritative_outcome": "frozen same-vLLM procedural confirmation",
        },
        "protocol": {
            "path": str(PROTOCOL.resolve()),
            "sha256": sha256_file(PROTOCOL),
        },
        "source": source_receipt,
        "config": str(config_path),
        "config_sha256": config_sha,
        "preregistration_receipt": str(prereg_path),
        "preregistration_receipt_sha256": sha256_file(prereg_path),
        "integration_receipt": integration_receipt,
        "artifact_snapshot": {
            **artifact_snapshot,
            **post_score_artifact_verification,
        },
        "cohort_contract": {
            "seed": DIAGNOSTIC_SEED,
            "rounds": list(DIAGNOSTIC_ROUNDS),
            "units_per_round": sum(PROBE_TARGET_COUNTS.values()),
            "target_counts_per_round": PROBE_TARGET_COUNTS,
            "selection": (
                "existing consumed trainer probe: lexicographic first 6 deep and 2 soup"
            ),
            "objective_positions": "every registered natural target position",
            "full_logit_position": "positions[len(positions) // 2]",
        },
        "runtime": runtime,
        "started_at_utc": started_at,
        "rounds": measurements,
        "aggregate": {
            "round_count": len(measurements),
            "unit_count": len(all_rows),
            "target_positions": sum(int(row["target_positions"]) for row in all_rows),
            "objective_parity": summarize_objective_rows(all_rows),
            "position_objective_parity": mean_numeric_metrics(
                [row["position_objective_parity"] for row in all_rows]
            ),
            "midpoint_endpoint_logit_parity": {
                endpoint: mean_numeric_metrics(
                    [row["endpoint_logit_parity"][endpoint] for row in all_rows],
                    boolean_keys={"top1_agreement"},
                )
                for endpoint in ("before", "after")
            },
            "midpoint_update_logit_parity": mean_numeric_metrics(
                [row["update_logit_parity"] for row in all_rows],
                boolean_keys=UPDATE_BOOLEAN_KEYS,
                nullable_keys=UPDATE_NULLABLE_KEYS,
            ),
        },
        "diagnostic_valid": diagnostic_valid,
        "diagnostic_validity": {
            "source_committed_before_execution": source_receipt["all_match_head"],
            "all_four_rounds_present": len(measurements) == len(DIAGNOSTIC_ROUNDS),
            "all_round_provenance_validated_before_gpu": True,
            "integration_receipt_bound": integration_receipt["gate_passed"],
            "artifacts_unchanged_after_scoring": post_score_artifact_verification[
                "verified_unchanged_after_scoring"
            ],
            "all_round_probe_and_replay_checks_passed": diagnostic_valid,
        },
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "wall_seconds": time.perf_counter() - started,
    }
    _write_json_exclusive(args.out, result)
    print(
        json.dumps(
            {
                "stage": result["stage"],
                "status": result["status"],
                "diagnostic_valid": result["diagnostic_valid"],
                "round_count": result["aggregate"]["round_count"],
                "unit_count": result["aggregate"]["unit_count"],
                "out": str(args.out.resolve()),
                "downstream_authorization": None,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if diagnostic_valid else 4


if __name__ == "__main__":
    raise SystemExit(main())
