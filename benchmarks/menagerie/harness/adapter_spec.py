# Validation logic adapted from templates/experiment/src/vllm_runner.py (_validate_adapter), the repo's reusable experiment runner.

import hashlib
import json
import os
import struct
from pathlib import Path


VLLM_SUPPORTED_RANKS = {1, 8, 16, 32, 64, 128, 256, 320, 512}
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
HF_CACHE_MODEL_DIR = "models--Qwen--Qwen3.5-4B"
LOCAL_MODEL_DIR = "Qwen3.5-4B"


def _path_segments(path: Path) -> set[str]:
    return {segment for segment in os.fspath(path).split(os.sep) if segment}


def _is_allowed_base(base: str) -> bool:
    if base == MODEL_ID:
        return True
    if not base:
        return False

    try:
        path = Path(base).expanduser()
    except (OSError, RuntimeError, ValueError):
        return False
    segments = _path_segments(path)
    try:
        resolved = path.resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        resolved = None
    if resolved is not None:
        segments.update(_path_segments(resolved))

    return (
        HF_CACHE_MODEL_DIR in segments
        or LOCAL_MODEL_DIR in segments
        or MODEL_REVISION in segments
    )


def _validate_safetensors_lora_rank(path: Path, rank: int) -> None:
    try:
        with path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise ValueError(f"safetensors file has an invalid header prefix: {path}")
            header_length = struct.unpack("<Q", prefix)[0]
            handle.seek(0, os.SEEK_END)
            file_size = handle.tell()
            if header_length > file_size - 8:
                raise ValueError(
                    f"safetensors file declares a header longer than the file: {path}"
                )
            handle.seek(8)
            header_bytes = handle.read(header_length)
    except OSError as exc:
        raise ValueError(f"could not read safetensors file {path}: {exc}") from exc

    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"safetensors file has an invalid JSON header: {path}: {exc}") from exc
    if not isinstance(header, dict):
        raise ValueError(f"safetensors header must contain a JSON object: {path}")

    for tensor_name, tensor_info in header.items():
        if not tensor_name.endswith(".lora_A.weight"):
            continue
        shape = tensor_info.get("shape") if isinstance(tensor_info, dict) else None
        actual_rank = shape[0] if isinstance(shape, list) and shape else None
        if actual_rank != rank:
            raise ValueError(
                f"adapter rank mismatch: config r={rank}, but {tensor_name!r} "
                f"in {path.name} has shape[0]={actual_rank!r}"
            )


def validate_adapter(adapter_path) -> dict:
    try:
        adapter = Path(adapter_path).expanduser()
    except TypeError as exc:
        raise ValueError(f"adapter path is invalid: {adapter_path!r}") from exc
    if not adapter.exists():
        raise ValueError(f"adapter path does not exist: {adapter}")
    if not adapter.is_dir():
        raise ValueError(f"adapter path is not a directory: {adapter}")
    adapter = adapter.resolve()

    config_path = adapter / "adapter_config.json"
    if not config_path.is_file():
        raise ValueError(f"adapter is missing adapter_config.json: {adapter}")
    config_bytes = config_path.read_bytes()
    try:
        config = json.loads(config_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"adapter_config.json is invalid JSON: {config_path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError("adapter_config.json must contain a JSON object")

    if str(config.get("peft_type", "")).upper() != "LORA":
        raise ValueError("only PEFT LoRA adapters are supported")

    base = str(config.get("base_model_name_or_path", "")).strip()
    if not _is_allowed_base(base):
        raise ValueError(
            f"adapter base model {base!r} violates the repo one-model rule "
            f"(AGENTS.md Non-Negotiables): only {MODEL_ID} adapters may be evaluated"
        )

    if config.get("use_dora"):
        raise ValueError("DoRA adapters are not supported by this runner")
    if config.get("modules_to_save"):
        raise ValueError("adapters with modules_to_save are not supported")
    if str(config.get("bias", "none")).lower() != "none":
        raise ValueError("LoRA bias weights are not supported")
    if config.get("rank_pattern"):
        raise ValueError("per-module rank_pattern adapters are not supported")
    if config.get("alpha_pattern"):
        raise ValueError("per-module alpha_pattern adapters are not supported")

    try:
        rank = int(config.get("r", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"adapter rank {config.get('r')!r} is not an integer") from exc
    if rank < 1 or rank not in VLLM_SUPPORTED_RANKS:
        raise ValueError(f"adapter rank {rank} is not supported by vLLM 0.24")

    weights = sorted(adapter.glob("*.safetensors"))
    if not weights:
        raise ValueError(f"adapter has no .safetensors weights: {adapter}")
    digest = hashlib.sha256()
    for path in weights:
        _validate_safetensors_lora_rank(path, rank)
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)

    return {
        "path": str(adapter),
        "rank": rank,
        "base_model_name_or_path": base,
        "target_modules": config.get("target_modules"),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "weights_sha256": digest.hexdigest(),
    }
