"""Model-free durability helpers for the preregistered full-run shard protocol.

The full matrix is too large for whole-arm atomic files or repository-local raw
rows.  This module owns the canonical task sharding, receipt format, path
containment, and byte-level validation used by both ``scripts/run.py`` and
``scripts/analyze.py``.  It deliberately has no vLLM, Torch, tokenizer, or
experiment-domain imports.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


FULL_ARTIFACT_SCHEMA_VERSION = 1
RECEIPT_FILE = "receipt.json"
PAYLOAD_FILES = ("preflight.json", "rows.jsonl", "runner.meta.json")
SHARD_DIR_RE = re.compile(r"^shard_[0-9]{3}$")
ARM_RE = re.compile(r"^[a-z][a-z0-9_]*$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PROTOCOL_RUNTIME_KEYS = (
    "python",
    "python_executable",
    "platform",
    "packages",
    "environment_lock",
    "uv",
    "cuda_toolkit",
    "gpu",
    "vllm_enable_v1_multiprocessing",
)


class FullArtifactError(ValueError):
    """A canonical full-run artifact is absent, malformed, or inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FullArtifactError(message)


def canonical_bytes(value: Any, *, pretty: bool = False) -> bytes:
    """Return the one registered JSON serialization."""

    options: dict[str, Any] = {
        "ensure_ascii": False,
        "allow_nan": False,
        "sort_keys": True,
    }
    if pretty:
        return (json.dumps(value, indent=2, **options) + "\n").encode("utf-8")
    return json.dumps(value, separators=(",", ":"), **options).encode("utf-8")


def value_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_integrity(path: Path) -> dict[str, Any]:
    """Stream a file once and return its SHA-256 and exact byte size."""

    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
                size += len(block)
    except FileNotFoundError as exc:
        raise FullArtifactError(f"missing full artifact file: {path}") from exc
    return {"sha256": digest.hexdigest(), "bytes": size}


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FullArtifactError(f"missing full artifact JSON: {path}") from exc
    except json.JSONDecodeError as exc:
        raise FullArtifactError(f"invalid full artifact JSON {path}: {exc}") from exc


def contained_path(root: Path, candidate: Path, *, must_exist: bool = True) -> Path:
    """Resolve ``candidate`` and prove that it is strictly beneath ``root``."""

    resolved_root = root.resolve(strict=must_exist)
    resolved = candidate.resolve(strict=must_exist)
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise FullArtifactError(
            f"full artifact path escapes canonical root {resolved_root}: {resolved}"
        ) from exc
    _require(relative != Path("."), "a shard path may not equal the external artifact root")
    return resolved


def canonical_task_triplets(
    tasks: Sequence[Mapping[str, Any]],
) -> list[list[str]]:
    """Return the frozen 40 ``[no_i, reuse_2i, reuse_2i+1]`` triplets."""

    seen: set[str] = set()
    by_split: dict[str, list[str]] = {"no_reuse": [], "reuse": []}
    for index, task in enumerate(tasks):
        task_id = task.get("id")
        split = task.get("split")
        _require(isinstance(task_id, str) and bool(task_id), f"task {index} lacks an id")
        _require(task_id not in seen, f"duplicate full task id: {task_id}")
        _require(split in by_split, f"full task {task_id} has unsupported split {split!r}")
        seen.add(task_id)
        by_split[str(split)].append(task_id)
    no_reuse = sorted(by_split["no_reuse"])
    reuse = sorted(by_split["reuse"])
    _require(len(no_reuse) == 40, f"full plan requires 40 no_reuse tasks, got {len(no_reuse)}")
    _require(len(reuse) == 80, f"full plan requires 80 reuse tasks, got {len(reuse)}")
    return [
        [no_reuse[index], reuse[2 * index], reuse[2 * index + 1]]
        for index in range(40)
    ]


def build_shard_plan(
    tasks: Sequence[Mapping[str, Any]],
    arms: Sequence[str],
    *,
    base_k: int = 24,
    macro_k: int = 12,
) -> dict[str, Any]:
    """Build the Amendment-8 task/K/batch plan without touching model outputs."""

    arm_order = [str(arm) for arm in arms]
    _require(bool(arm_order), "full shard plan requires at least one arm")
    _require(len(set(arm_order)) == len(arm_order), "full shard plan has duplicate arms")
    _require(all(ARM_RE.fullmatch(arm) for arm in arm_order), "full shard plan has invalid arm id")
    _require(arm_order[0] == "base", "base must be the first full arm")
    _require(base_k == 24, "base K must remain 24")
    _require(macro_k == 12, "macro-arm K must remain 12")
    triplets = canonical_task_triplets(tasks)
    per_arm: dict[str, Any] = {}
    for arm in arm_order:
        k = base_k if arm == "base" else macro_k
        triplets_per_shard = 2 if arm == "base" else 4
        shards: list[dict[str, Any]] = []
        for start in range(0, len(triplets), triplets_per_shard):
            grouped = triplets[start : start + triplets_per_shard]
            task_ids = [task_id for triplet in grouped for task_id in triplet]
            shard_index = len(shards)
            shards.append(
                {
                    "shard_index": shard_index,
                    "triplet_indices": list(range(start, start + triplets_per_shard)),
                    "task_ids": task_ids,
                    "record_ids": [f"{task_id}::{arm}" for task_id in task_ids],
                    "tasks": len(task_ids),
                    "k": k,
                    "completions": len(task_ids) * k,
                }
            )
        expected_shards = 20 if arm == "base" else 10
        _require(len(shards) == expected_shards, f"{arm} shard count drifted")
        _require(
            all(shard["completions"] == 144 for shard in shards),
            f"{arm} shards must each contain 144 completions",
        )
        per_arm[arm] = {
            "k": k,
            "shard_count": len(shards),
            "tasks_per_shard": 6 if arm == "base" else 12,
            "completions_per_shard": 144,
            "shards": shards,
        }
    return {
        "schema_version": FULL_ARTIFACT_SCHEMA_VERSION,
        "protocol": "amendment_8_balanced_full_shards",
        "triplet_rule": "[sorted_no_reuse_i, sorted_reuse_2i, sorted_reuse_2i+1]",
        "triplets": triplets,
        "arm_order": arm_order,
        "arms": per_arm,
    }


def plan_sha256(plan: Mapping[str, Any]) -> str:
    return value_sha256(dict(plan))


def shard_spec(plan: Mapping[str, Any], arm: str, shard_index: int) -> dict[str, Any]:
    raw_arms = plan.get("arms")
    _require(isinstance(raw_arms, Mapping) and arm in raw_arms, f"arm absent from shard plan: {arm}")
    arm_plan = raw_arms[arm]
    _require(isinstance(arm_plan, Mapping), f"invalid shard plan for arm {arm}")
    shards = arm_plan.get("shards")
    _require(isinstance(shards, list), f"invalid shard list for arm {arm}")
    _require(0 <= shard_index < len(shards), f"invalid {arm} shard index {shard_index}")
    spec = shards[shard_index]
    _require(isinstance(spec, dict), f"invalid {arm} shard {shard_index} spec")
    _require(spec.get("shard_index") == shard_index, f"{arm} shard index/order drifted")
    return dict(spec)


def shard_directory(root: Path, *, budget: int, arm: str, shard_index: int) -> Path:
    _require(isinstance(budget, int) and budget > 0, "full shard budget must be positive")
    _require(ARM_RE.fullmatch(arm) is not None, f"invalid full arm id: {arm!r}")
    _require(0 <= shard_index <= 999, "full shard index must fit three digits")
    return root / f"think_{budget}" / arm / f"shard_{shard_index:03d}"


def runner_provenance(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Extract full receipt provenance, including non-protocol Git audit fields."""

    keys = (
        "schema_version",
        "model",
        "model_revision",
        "runner_sha256",
        "adapter",
        "sampling",
        "resolved_sampling",
        "engine",
        "engine_args",
        "runtime",
        "rng_isolation",
        "termination",
        "think_token_ids",
    )
    provenance = {key: copy.deepcopy(summary.get(key)) for key in keys}
    for key in (
        "schema_version",
        "model",
        "model_revision",
        "runner_sha256",
        "sampling",
        "resolved_sampling",
        "engine",
        "engine_args",
        "runtime",
        "rng_isolation",
        "termination",
        "think_token_ids",
    ):
        _require(
            provenance[key] is not None,
            f"runner metadata lacks receipt provenance field {key}",
        )
    runtime = provenance["runtime"]
    _require(isinstance(runtime, Mapping), "runner runtime provenance must be an object")
    for key in (*PROTOCOL_RUNTIME_KEYS, "git_commit", "git_dirty"):
        _require(key in runtime, f"runner runtime provenance lacks {key}")
    _require(
        isinstance(provenance["schema_version"], int)
        and not isinstance(provenance["schema_version"], bool)
        and int(provenance["schema_version"]) >= 1,
        "runner schema version must be a positive integer",
    )
    for key in ("model", "model_revision", "runner_sha256"):
        _require(
            isinstance(provenance[key], str) and bool(provenance[key]),
            f"runner provenance {key} must be a non-empty string",
        )
    _require(
        SHA256_RE.fullmatch(str(provenance["runner_sha256"])) is not None,
        "runner provenance runner_sha256 must be a SHA-256",
    )
    for key in (
        "sampling",
        "resolved_sampling",
        "engine",
        "engine_args",
        "rng_isolation",
        "termination",
        "think_token_ids",
    ):
        _require(isinstance(provenance[key], Mapping), f"runner provenance {key} must be an object")
    _require(
        provenance["adapter"] is None or isinstance(provenance["adapter"], Mapping),
        "runner adapter provenance must be null or an object",
    )
    _require(
        isinstance(runtime["packages"], Mapping) and bool(runtime["packages"]),
        "runner runtime packages must be a non-empty object",
    )
    _require(
        isinstance(runtime["environment_lock"], Mapping),
        "runner runtime environment_lock must be an object",
    )
    for key in (
        "python",
        "python_executable",
        "platform",
        "uv",
        "cuda_toolkit",
        "gpu",
        "git_commit",
    ):
        _require(isinstance(runtime[key], str), f"runner runtime {key} must be a string")
    _require(isinstance(runtime["git_dirty"], bool), "runner runtime git_dirty must be bool")
    return provenance


def protocol_identity(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Return only fields that must be equal for shards to share a protocol.

    ``git_commit`` and ``git_dirty`` remain in :func:`runner_provenance` for
    audit, but they do not affect inference and therefore cannot invalidate an
    otherwise exact resumed shard.  Package/lock, hardware/runtime, model,
    runner, sampling, engine, token-boundary, and RNG fields remain strict.
    """

    provenance = runner_provenance(summary)
    runtime = provenance["runtime"]
    _require(isinstance(runtime, Mapping), "runner runtime provenance must be an object")
    result = {
        key: copy.deepcopy(value)
        for key, value in provenance.items()
        if key != "runtime"
    }
    result["runtime"] = {
        key: copy.deepcopy(runtime[key]) for key in PROTOCOL_RUNTIME_KEYS
    }
    return result


def require_protocol_identity(
    summary: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    where: str = "runner metadata",
) -> dict[str, Any]:
    """Validate a summary against a precomputed equality-critical identity."""

    actual = protocol_identity(summary)
    _require(actual == dict(expected), f"{where} protocol identity drift")
    return actual


def receipt_protocol_identity(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return the equality-critical binding from a receipt."""

    identity = receipt.get("protocol_identity")
    _require(isinstance(identity, Mapping), "full shard receipt lacks protocol identity")
    normalized = copy.deepcopy(dict(identity))
    _require(
        receipt.get("protocol_identity_sha256") == value_sha256(normalized),
        "full shard receipt protocol identity hash mismatch",
    )
    return normalized


def protocol_binding(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Return receipt fields that bind provenance and equality-critical protocol.

    ``identity`` is a temporary schema-1 compatibility alias for
    ``protocol_identity``; critically, it no longer contains Git audit state.
    """

    provenance = runner_provenance(summary)
    identity = protocol_identity(summary)
    return {
        "provenance": provenance,
        "provenance_sha256": value_sha256(provenance),
        "protocol_identity": identity,
        "protocol_identity_sha256": value_sha256(identity),
        "identity": copy.deepcopy(identity),
        "identity_sha256": value_sha256(identity),
    }


def runner_identity(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Compatibility alias for the equality-critical protocol identity."""

    return protocol_identity(summary)


def _prompt_bindings(preflight: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = preflight.get("records")
    _require(isinstance(rows, list) and bool(rows), "full shard preflight lacks records")
    bindings: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        _require(isinstance(row, Mapping), f"preflight record {index} is not an object")
        binding = {
            "id": row.get("id"),
            "input_record_sha256": row.get("input_record_sha256"),
            "rendered_prompt_sha256": row.get("rendered_prompt_sha256"),
            "prompt_tokens": row.get("prompt_tokens"),
            "prompt_plus_reserve_tokens": row.get("prompt_plus_reserve_tokens"),
        }
        _require(
            isinstance(binding["id"], str)
            and isinstance(binding["input_record_sha256"], str)
            and SHA256_RE.fullmatch(binding["input_record_sha256"]) is not None
            and isinstance(binding["rendered_prompt_sha256"], str)
            and SHA256_RE.fullmatch(binding["rendered_prompt_sha256"]) is not None
            and isinstance(binding["prompt_tokens"], int)
            and not isinstance(binding["prompt_tokens"], bool)
            and binding["prompt_tokens"] >= 0
            and isinstance(binding["prompt_plus_reserve_tokens"], int)
            and not isinstance(binding["prompt_plus_reserve_tokens"], bool)
            and binding["prompt_plus_reserve_tokens"] >= 0,
            f"preflight record {index} lacks a prompt binding field",
        )
        bindings.append(binding)
    return bindings


def _integer(value: Any, *, where: str, minimum: int = 0) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
        f"{where} must be an integer >= {minimum}",
    )
    return int(value)


def _integer_token_list(value: Any, *, where: str) -> list[int]:
    _require(
        isinstance(value, list)
        and all(isinstance(token, int) and not isinstance(token, bool) for token in value),
        f"{where} must be an integer-token list",
    )
    return [int(token) for token in value]


def _validate_preflight(
    preflight: Mapping[str, Any],
    summary: Mapping[str, Any],
    *,
    budget: int,
    expected_record_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Recompute the full prompt/context arithmetic from metadata."""

    _require(
        preflight.get("schema_version") == FULL_ARTIFACT_SCHEMA_VERSION,
        "full shard preflight schema mismatch",
    )
    _require(preflight.get("pass") is True, "full shard preflight did not pass")
    _require(
        preflight.get("n_records") == len(expected_record_ids),
        "full shard preflight record count mismatch",
    )
    sampling = summary.get("sampling")
    engine = summary.get("engine")
    think_ids = summary.get("think_token_ids")
    _require(isinstance(sampling, Mapping), "runner metadata lacks sampling")
    _require(isinstance(engine, Mapping), "runner metadata lacks engine")
    _require(isinstance(think_ids, Mapping), "runner metadata lacks thinking token ids")
    _require(sampling.get("thinking") == "budget", "full shards require budget thinking")
    _require(
        _integer(
            sampling.get("thinking_budget"),
            where="runner sampling.thinking_budget",
            minimum=1,
        )
        == budget,
        "runner thinking budget disagrees with shard budget",
    )
    answer_max_tokens = _integer(
        sampling.get("answer_max_tokens"),
        where="runner sampling.answer_max_tokens",
        minimum=1,
    )
    _require(
        sampling.get("max_tokens") == answer_max_tokens,
        "full runner max_tokens must equal answer_max_tokens",
    )
    forced_close = _integer_token_list(
        think_ids.get("forced_close_sequence"),
        where="runner think_token_ids.forced_close_sequence",
    )
    _require(bool(forced_close), "forced-close sequence cannot be empty")
    expected_reserve = budget + len(forced_close) + answer_max_tokens
    _require(
        preflight.get("generation_reserve_tokens") == expected_reserve,
        "full shard preflight generation reserve mismatch",
    )
    max_model_len = _integer(
        engine.get("max_model_len"), where="runner engine.max_model_len", minimum=1
    )
    _require(
        preflight.get("max_model_len") == max_model_len,
        "full shard preflight max_model_len mismatch",
    )
    prompts = _prompt_bindings(preflight)
    _require(
        [prompt["id"] for prompt in prompts] == list(expected_record_ids),
        "full shard preflight record order mismatch",
    )
    prompt_counts = [int(prompt["prompt_tokens"]) for prompt in prompts]
    _require(
        preflight.get("min_prompt_tokens") == min(prompt_counts),
        "full shard preflight min prompt count mismatch",
    )
    _require(
        preflight.get("max_prompt_tokens") == max(prompt_counts),
        "full shard preflight max prompt count mismatch",
    )
    _require(
        preflight.get("max_prompt_plus_reserve_tokens")
        == max(prompt_counts) + expected_reserve,
        "full shard preflight max prompt-plus-reserve mismatch",
    )
    for prompt in prompts:
        expected_total = int(prompt["prompt_tokens"]) + expected_reserve
        _require(
            prompt["prompt_plus_reserve_tokens"] == expected_total,
            f"full shard preflight per-record reserve mismatch for {prompt['id']}",
        )
        _require(
            expected_total <= max_model_len,
            f"full shard preflight context overflow for {prompt['id']}",
        )
    return prompts


def make_receipt(
    shard_dir: Path,
    *,
    shard_plan_sha256: str,
    budget: int,
    arm: str,
    shard_index: int,
    task_ids: Sequence[str],
    k: int,
) -> dict[str, Any]:
    """Construct a last-written receipt for three already-complete payload files."""

    normalized_task_ids = [str(task_id) for task_id in task_ids]
    _require(
        all(normalized_task_ids)
        and len(set(normalized_task_ids)) == len(normalized_task_ids),
        "receipt task ids are empty or duplicated",
    )
    preflight = read_json(shard_dir / "preflight.json")
    summary = read_json(shard_dir / "runner.meta.json")
    _require(isinstance(preflight, Mapping), "full shard preflight must be an object")
    _require(isinstance(summary, Mapping), "full shard runner metadata must be an object")
    prompts, _ = _validate_payload_structure(
        shard_dir / "rows.jsonl",
        preflight=preflight,
        summary=summary,
        budget=budget,
        arm=arm,
        task_ids=normalized_task_ids,
        k=k,
    )
    ordered_ids = [str(prompt["id"]) for prompt in prompts]
    expected_ids = [f"{task_id}::{arm}" for task_id in normalized_task_ids]
    _require(ordered_ids == expected_ids, "receipt record order disagrees with shard task order")
    return {
        "schema_version": FULL_ARTIFACT_SCHEMA_VERSION,
        "status": "complete",
        "shard_plan_sha256": shard_plan_sha256,
        "budget": budget,
        "arm": arm,
        "shard_index": shard_index,
        "task_ids": normalized_task_ids,
        "ordered_record_ids": ordered_ids,
        "ordered_prompts": prompts,
        "k": k,
        "completions": len(ordered_ids) * k,
        **protocol_binding(summary),
        "files": {name: file_integrity(shard_dir / name) for name in PAYLOAD_FILES},
    }


def _stream_row_bindings(
    path: Path,
    *,
    arm: str,
    k: int,
    budget: int,
    prompts: Sequence[Mapping[str, Any]],
    forced_close_sequence: Sequence[int],
    think_close_token_id: int,
    hf_model_eos_token_id: int,
) -> tuple[list[str], list[str], dict[str, int]]:
    record_ids: list[str] = []
    task_ids: list[str] = []
    prompt_by_id = {str(prompt["id"]): prompt for prompt in prompts}
    totals = {
        "requests": 0,
        "completions": 0,
        "unique_input_prompt_tokens": 0,
        "stage1_logical_prompt_tokens": 0,
        "stage2_logical_prompt_tokens": 0,
        "sampled_tokens": 0,
        "injected_tokens": 0,
    }
    try:
        handle = path.open("r", encoding="utf-8")
    except FileNotFoundError as exc:
        raise FullArtifactError(f"missing full shard rows: {path}") from exc
    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise FullArtifactError(f"invalid shard JSONL {path}:{line_number}: {exc}") from exc
            _require(isinstance(row, Mapping), f"{path}:{line_number} must be an object")
            record_id = row.get("id")
            meta = row.get("meta")
            outputs = row.get("outputs")
            _require(isinstance(record_id, str), f"{path}:{line_number} lacks id")
            _require(isinstance(meta, Mapping), f"{path}:{line_number} lacks meta")
            task_id = meta.get("task_id")
            _require(isinstance(task_id, str), f"{path}:{line_number} lacks task_id")
            _require(record_id == f"{task_id}::{arm}", f"{path}:{line_number} id/arm mismatch")
            _require(meta.get("arm") == arm, f"{path}:{line_number} meta arm mismatch")
            _require(record_id in prompt_by_id, f"{path}:{line_number} lacks preflight binding")
            prompt = prompt_by_id[record_id]
            prompt_tokens = _integer(
                row.get("n_prompt_tokens"), where=f"{path}:{line_number}.n_prompt_tokens"
            )
            _require(
                prompt_tokens == prompt["prompt_tokens"],
                f"{path}:{line_number} prompt token count mismatch",
            )
            _require(
                row.get("prompt_sha256") == prompt["rendered_prompt_sha256"],
                f"{path}:{line_number} rendered prompt hash mismatch",
            )
            _require(
                isinstance(outputs, list) and len(outputs) == k,
                f"{path}:{line_number} K mismatch",
            )
            sample_indices = [
                output.get("sample_index") if isinstance(output, Mapping) else None
                for output in outputs
            ]
            _require(
                sample_indices == list(range(k)),
                f"{path}:{line_number} sample indices are not exact and ordered",
            )
            totals["requests"] += 1
            totals["unique_input_prompt_tokens"] += prompt_tokens
            for sample_index, output in enumerate(outputs):
                _require(
                    isinstance(output, Mapping),
                    f"{path}:{line_number} output {sample_index} is not an object",
                )
                stage1_prompt = _integer(
                    output.get("n_stage1_prompt_tokens"),
                    where=f"{path}:{line_number} output {sample_index}.n_stage1_prompt_tokens",
                )
                stage2_prompt = _integer(
                    output.get("n_stage2_prompt_tokens"),
                    where=f"{path}:{line_number} output {sample_index}.n_stage2_prompt_tokens",
                )
                sampled = _integer(
                    output.get("n_sampled_tokens"),
                    where=f"{path}:{line_number} output {sample_index}.n_sampled_tokens",
                )
                injected = _integer(
                    output.get("n_injected_tokens"),
                    where=f"{path}:{line_number} output {sample_index}.n_injected_tokens",
                )
                completion = _integer(
                    output.get("n_completion_tokens"),
                    where=f"{path}:{line_number} output {sample_index}.n_completion_tokens",
                )
                thinking = _integer(
                    output.get("n_thinking_tokens"),
                    where=f"{path}:{line_number} output {sample_index}.n_thinking_tokens",
                )
                answer = _integer(
                    output.get("n_answer_tokens"),
                    where=f"{path}:{line_number} output {sample_index}.n_answer_tokens",
                )
                terminal_trimmed = _integer(
                    output.get("n_terminal_tokens_trimmed"),
                    where=(
                        f"{path}:{line_number} output "
                        f"{sample_index}.n_terminal_tokens_trimmed"
                    ),
                )
                _require(
                    stage1_prompt == prompt_tokens,
                    f"{path}:{line_number} output {sample_index} stage-one prompt mismatch",
                )
                _require(
                    thinking <= budget,
                    f"{path}:{line_number} output {sample_index} exceeds thinking budget",
                )
                token_ids = _integer_token_list(
                    output.get("token_ids"),
                    where=f"{path}:{line_number} output {sample_index}.token_ids",
                )
                stage1_ids = _integer_token_list(
                    output.get("stage1_token_ids"),
                    where=f"{path}:{line_number} output {sample_index}.stage1_token_ids",
                )
                stage2_ids = _integer_token_list(
                    output.get("stage2_token_ids"),
                    where=f"{path}:{line_number} output {sample_index}.stage2_token_ids",
                )
                injected_ids = _integer_token_list(
                    output.get("injected_token_ids"),
                    where=f"{path}:{line_number} output {sample_index}.injected_token_ids",
                )
                _require(
                    completion == len(token_ids),
                    f"{path}:{line_number} output {sample_index} completion count mismatch",
                )
                _require(
                    sampled == len(stage1_ids) + len(stage2_ids),
                    f"{path}:{line_number} output {sample_index} sampled count mismatch",
                )
                _require(
                    injected == len(injected_ids),
                    f"{path}:{line_number} output {sample_index} injected count mismatch",
                )
                forced_close = output.get("forced_close")
                _require(
                    isinstance(forced_close, bool),
                    f"{path}:{line_number} output {sample_index}.forced_close must be bool",
                )
                _require(
                    isinstance(output.get("text"), str),
                    f"{path}:{line_number} output {sample_index}.text must be string",
                )
                finish_reason = output.get("finish_reason")
                stage1_finish_reason = output.get("stage1_finish_reason")
                truncated = output.get("truncated")
                _require(
                    isinstance(finish_reason, str)
                    and isinstance(stage1_finish_reason, str),
                    f"{path}:{line_number} output {sample_index} lacks finish reasons",
                )
                _require(
                    isinstance(truncated, bool)
                    and truncated == (finish_reason == "length"),
                    f"{path}:{line_number} output {sample_index} truncation mismatch",
                )
                seed_stage2 = output.get("seed_stage2")
                _require(
                    seed_stage2 is None
                    or (
                        isinstance(seed_stage2, int)
                        and not isinstance(seed_stage2, bool)
                        and seed_stage2 >= 0
                    ),
                    f"{path}:{line_number} output {sample_index}.seed_stage2 is invalid",
                )
                continued = seed_stage2 is not None
                trimmed_stage1 = (
                    stage1_ids[:-1]
                    if stage1_ids and stage1_ids[-1] == hf_model_eos_token_id
                    else stage1_ids
                )
                trimmed_stage2 = (
                    stage2_ids[:-1]
                    if stage2_ids and stage2_ids[-1] == hf_model_eos_token_id
                    else stage2_ids
                )
                expected_terminal_trimmed = (
                    len(stage1_ids)
                    - len(trimmed_stage1)
                    + len(stage2_ids)
                    - len(trimmed_stage2)
                )
                _require(
                    terminal_trimmed == expected_terminal_trimmed,
                    f"{path}:{line_number} output {sample_index} terminal trim mismatch",
                )
                if continued:
                    retained_ids = _integer_token_list(
                        output.get("retained_thinking_token_ids"),
                        where=(
                            f"{path}:{line_number} output "
                            f"{sample_index}.retained_thinking_token_ids"
                        ),
                    )
                    _require(
                        stage2_prompt
                        == prompt_tokens + len(retained_ids) + len(injected_ids),
                        f"{path}:{line_number} output {sample_index} stage-two prompt arithmetic mismatch",
                    )
                    _require(
                        injected_ids == list(forced_close_sequence),
                        f"{path}:{line_number} output {sample_index} close injection mismatch",
                    )
                    _require(
                        token_ids
                        == retained_ids + injected_ids + trimmed_stage2,
                        f"{path}:{line_number} output {sample_index} continuation token structure mismatch",
                    )
                    _require(
                        thinking == len(retained_ids)
                        and answer == len(trimmed_stage2),
                        f"{path}:{line_number} output {sample_index} continuation channel counts mismatch",
                    )
                else:
                    _require(
                        stage2_prompt == 0
                        and not stage2_ids
                        and not injected_ids
                        and forced_close is False,
                        f"{path}:{line_number} output {sample_index} unexpected stage-two structure",
                    )
                    _require(
                        token_ids == trimmed_stage1,
                        f"{path}:{line_number} output {sample_index} stage-one token structure mismatch",
                    )
                    _require(
                        think_close_token_id in token_ids,
                        f"{path}:{line_number} output {sample_index} natural answer lacks thinking close",
                    )
                    close_index = token_ids.index(think_close_token_id)
                    _require(
                        thinking == close_index
                        and answer == len(token_ids) - close_index - 1,
                        f"{path}:{line_number} output {sample_index} natural channel counts mismatch",
                    )
                _require(
                    not forced_close or continued,
                    f"{path}:{line_number} output {sample_index} forced close lacks continuation",
                )
                _require(
                    output.get("thinking_closed") is True,
                    f"{path}:{line_number} output {sample_index} thinking channel is not closed",
                )
                totals["completions"] += 1
                totals["stage1_logical_prompt_tokens"] += stage1_prompt
                totals["stage2_logical_prompt_tokens"] += stage2_prompt
                totals["sampled_tokens"] += sampled
                totals["injected_tokens"] += injected
            record_ids.append(record_id)
            task_ids.append(task_id)
    _require(bool(record_ids), f"full shard rows are empty: {path}")
    return record_ids, task_ids, totals


def _validate_payload_structure(
    rows_path: Path,
    *,
    preflight: Mapping[str, Any],
    summary: Mapping[str, Any],
    budget: int,
    arm: str,
    task_ids: Sequence[str],
    k: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    expected_record_ids = [f"{task_id}::{arm}" for task_id in task_ids]
    prompts = _validate_preflight(
        preflight,
        summary,
        budget=budget,
        expected_record_ids=expected_record_ids,
    )
    sampling = summary.get("sampling")
    think_ids = summary.get("think_token_ids")
    termination = summary.get("termination")
    _require(isinstance(sampling, Mapping), "runner metadata lacks sampling")
    _require(isinstance(think_ids, Mapping), "runner metadata lacks thinking token ids")
    _require(isinstance(termination, Mapping), "runner metadata lacks termination")
    _require(sampling.get("n") == k, "runner sampling n disagrees with shard K")
    forced_close_sequence = _integer_token_list(
        think_ids.get("forced_close_sequence"),
        where="runner think_token_ids.forced_close_sequence",
    )
    think_close_token_id = _integer(
        think_ids.get("close"),
        where="runner think_token_ids.close",
    )
    hf_model_eos_token_id = _integer(
        termination.get("hf_model_eos_token_id"),
        where="runner termination.hf_model_eos_token_id",
    )
    row_ids, row_task_ids, totals = _stream_row_bindings(
        rows_path,
        arm=arm,
        k=k,
        budget=budget,
        prompts=prompts,
        forced_close_sequence=forced_close_sequence,
        think_close_token_id=think_close_token_id,
        hf_model_eos_token_id=hf_model_eos_token_id,
    )
    _require(row_ids == expected_record_ids, "full shard row record order mismatch")
    _require(row_task_ids == list(task_ids), "full shard row task order mismatch")
    counts = summary.get("counts")
    _require(isinstance(counts, Mapping), "full shard runner metadata lacks counts")
    expected_counts = {
        **totals,
        "logical_model_input_tokens": (
            totals["stage1_logical_prompt_tokens"]
            + totals["stage2_logical_prompt_tokens"]
        ),
    }
    for key, expected in expected_counts.items():
        _require(
            counts.get(key) == expected,
            f"full shard runner summary count {key} mismatch",
        )
    return prompts, totals


def validate_shard_directory(
    shard_dir: Path,
    *,
    root: Path,
    shard_plan_sha256: str,
    budget: int,
    arm: str,
    shard_index: int,
    task_ids: Sequence[str],
    k: int,
    expected_protocol_identity: Mapping[str, Any] | None = None,
    expected_identity: Mapping[str, Any] | None = None,
    allow_temporary_name: bool = False,
) -> dict[str, Any]:
    """Fail closed unless a final shard exactly matches its receipt and plan."""

    resolved = contained_path(root, shard_dir)
    _require(resolved.is_dir(), f"full shard is not a directory: {resolved}")
    if not allow_temporary_name:
        _require(
            SHARD_DIR_RE.fullmatch(resolved.name) is not None,
            f"invalid shard directory name: {resolved}",
        )
    names = {path.name for path in resolved.iterdir()}
    expected_names = set(PAYLOAD_FILES) | {RECEIPT_FILE}
    _require(names == expected_names, f"full shard has missing or unexpected files: {resolved}")
    for name in expected_names:
        child = resolved / name
        _require(not child.is_symlink(), f"full shard file may not be a symlink: {child}")
        _require(child.is_file(), f"full shard entry is not a file: {child}")
        contained_path(resolved, child)

    receipt = read_json(resolved / RECEIPT_FILE)
    _require(isinstance(receipt, dict), f"full shard receipt must be an object: {resolved}")
    expected_scalars = {
        "schema_version": FULL_ARTIFACT_SCHEMA_VERSION,
        "status": "complete",
        "shard_plan_sha256": shard_plan_sha256,
        "budget": budget,
        "arm": arm,
        "shard_index": shard_index,
        "k": k,
        "completions": len(task_ids) * k,
    }
    for key, value in expected_scalars.items():
        _require(receipt.get(key) == value, f"full shard receipt {key} mismatch: {resolved}")
    expected_task_ids = [str(task_id) for task_id in task_ids]
    _require(
        all(expected_task_ids) and len(set(expected_task_ids)) == len(expected_task_ids),
        f"full shard expected task ids are empty or duplicated: {resolved}",
    )
    expected_record_ids = [f"{task_id}::{arm}" for task_id in expected_task_ids]
    _require(receipt.get("task_ids") == expected_task_ids, f"full shard task order mismatch: {resolved}")
    _require(
        receipt.get("ordered_record_ids") == expected_record_ids,
        f"full shard record order mismatch: {resolved}",
    )
    files = receipt.get("files")
    _require(isinstance(files, Mapping), f"full shard receipt lacks files: {resolved}")
    _require(set(files) == set(PAYLOAD_FILES), f"full shard receipt file set mismatch: {resolved}")
    for name in PAYLOAD_FILES:
        _require(
            files.get(name) == file_integrity(resolved / name),
            f"full shard payload hash/size mismatch for {resolved / name}",
        )

    preflight = read_json(resolved / "preflight.json")
    summary = read_json(resolved / "runner.meta.json")
    _require(isinstance(preflight, Mapping), f"full shard preflight is invalid: {resolved}")
    _require(isinstance(summary, Mapping), f"full shard metadata is invalid: {resolved}")
    prompts, _ = _validate_payload_structure(
        resolved / "rows.jsonl",
        preflight=preflight,
        summary=summary,
        budget=budget,
        arm=arm,
        task_ids=expected_task_ids,
        k=k,
    )
    _require(receipt.get("ordered_prompts") == prompts, f"full shard prompt receipt mismatch: {resolved}")
    _require(
        [prompt["id"] for prompt in prompts] == expected_record_ids,
        f"full shard preflight record order mismatch: {resolved}",
    )
    provenance = runner_provenance(summary)
    _require(
        receipt.get("provenance") == provenance,
        f"full shard runner provenance mismatch: {resolved}",
    )
    _require(
        receipt.get("provenance_sha256") == value_sha256(provenance),
        f"full shard provenance hash mismatch: {resolved}",
    )
    identity = protocol_identity(summary)
    _require(
        receipt.get("protocol_identity") == identity,
        f"full shard protocol identity mismatch: {resolved}",
    )
    _require(
        receipt.get("protocol_identity_sha256") == value_sha256(identity),
        f"full shard protocol identity hash mismatch: {resolved}",
    )
    _require(
        receipt.get("identity") == identity
        and receipt.get("identity_sha256") == value_sha256(identity),
        f"full shard legacy protocol-identity alias mismatch: {resolved}",
    )
    _require(
        not (
            expected_protocol_identity is not None
            and expected_identity is not None
        ),
        "provide only expected_protocol_identity, not the legacy expected_identity alias",
    )
    expected = (
        expected_protocol_identity
        if expected_protocol_identity is not None
        else expected_identity
    )
    if expected is not None:
        _require(
            identity == dict(expected),
            f"full shard protocol identity drift: {resolved}",
        )
    return receipt


def catalog_shard_entry(root: Path, shard_dir: Path, receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Return the compact catalog entry for one already validated final shard."""

    resolved = contained_path(root, shard_dir)
    relative = resolved.relative_to(root.resolve(strict=True)).as_posix()
    return {
        "budget": receipt["budget"],
        "arm": receipt["arm"],
        "shard_index": receipt["shard_index"],
        "relative_path": relative,
        "receipt": file_integrity(resolved / RECEIPT_FILE),
        "provenance_sha256": receipt["provenance_sha256"],
        "protocol_identity_sha256": receipt["protocol_identity_sha256"],
        "payload_files": copy.deepcopy(receipt["files"]),
    }


def require_catalog_shard_entry(
    root: Path,
    shard_dir: Path,
    receipt: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    """Rebuild and compare one catalog entry rather than trusting its hashes."""

    actual = catalog_shard_entry(root, shard_dir, receipt)
    _require(actual == dict(expected), f"full catalog shard entry drift: {shard_dir}")
    return actual


def aggregate_runner_summaries(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Synthesize one accounting-compatible arm summary from fixed shards."""

    _require(bool(summaries), "cannot aggregate zero runner summaries")
    identity = protocol_identity(summaries[0])
    _require(
        all(protocol_identity(summary) == identity for summary in summaries[1:]),
        "runner protocol identity differs across shards in one arm",
    )
    result = copy.deepcopy(dict(summaries[0]))
    count_maps = [summary.get("counts") for summary in summaries]
    _require(all(isinstance(counts, Mapping) for counts in count_maps), "shard summary lacks counts")
    count_keys = set(count_maps[0])  # type: ignore[arg-type]
    _require(
        all(set(counts) == count_keys for counts in count_maps[1:]),  # type: ignore[arg-type]
        "shard summary count fields differ",
    )
    result["counts"] = {
        key: sum(int(counts[key]) for counts in count_maps)  # type: ignore[index]
        for key in sorted(count_keys)
    }
    timings = [summary.get("timing") for summary in summaries]
    _require(all(isinstance(timing, Mapping) for timing in timings), "shard summary lacks timing")
    generation_seconds = sum(
        float(timing.get("generation_seconds", 0.0))  # type: ignore[union-attr]
        for timing in timings
    )
    sampled_tokens = int(result["counts"].get("sampled_tokens", 0))
    result["timing"] = {
        "generation_seconds": generation_seconds,
        "model_load_seconds": float(timings[0].get("model_load_seconds", 0.0)),  # type: ignore[union-attr]
        "sampled_tokens_per_second": (
            sampled_tokens / generation_seconds if generation_seconds > 0.0 else None
        ),
        "aggregation": "sum generation seconds across fixed full shards; model load from first shard",
        "shards": len(summaries),
    }
    result["full_shard_aggregation"] = {
        "schema_version": FULL_ARTIFACT_SCHEMA_VERSION,
        "shards": len(summaries),
    }
    return result


def fsync_directory(path: Path) -> None:
    """Durably order file creation before a directory rename on POSIX."""

    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
