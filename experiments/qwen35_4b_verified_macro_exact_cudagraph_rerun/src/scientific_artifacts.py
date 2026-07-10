"""Fail-closed external storage for scientific-smoke runner artifacts.

The scientific smoke can produce individual JSONL files larger than GitHub's
hard file limit.  This module keeps those files in one explicit external root
without weakening the experiment's cache identity.  Runner-native artifacts
retain their existing flat names::

    smoke_tiers/think_49152/base.preflight.json
    smoke_tiers/think_49152/base.jsonl
    smoke_tiers/think_49152/base.meta.json
    smoke_tiers/think_49152/base.receipt.json

The receipt is the last-written commit marker.  A preflight by itself is a
valid resumable state; every other partial combination fails closed.  Selected
tiers are represented only by a logical pointer in the tracked catalog.  This
module never creates a physical ``smoke/`` promotion directory.

Only the Python standard library is used.  Row validation checks identities,
prompt hashes/counts, sample order, and K; it never parses or grades decoded
model output.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


EXPERIMENT_ID = "qwen35_4b_verified_macro_exact_cudagraph_rerun"
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
RUNNER_SHA256 = "3a98eb8da787054aded56a1ec3fd040ee2edaacc7d0694b4aec5a0309488774a"
FORCED_CLOSE_TOKENS = 2
ANSWER_MAX_TOKENS = 512
DEFAULT_ARTIFACT_ROOT = Path(
    "/workspace/large_artifacts/"
    "qwen35_4b_verified_macro_exact_cudagraph_rerun/scientific_smoke_v1"
)
LOCK_PATH = DEFAULT_ARTIFACT_ROOT.parent / ".exact_cudagraph.lock"
PREDECESSOR_ARTIFACT_ROOT = Path(
    "/workspace/large_artifacts/"
    "qwen35_4b_verified_macro_capacity_fit_rerun/scientific_smoke_v1"
)
OTHER_FORBIDDEN_ARTIFACT_ROOT = Path(
    "/workspace/large_artifacts/"
    "qwen35_4b_verified_macro_long_context_rerun/scientific_smoke_v1"
)
ARTIFACT_ROOT_ENV = "QWEN35_MACRO_EXACT_CUDAGRAPH_ARTIFACT_ROOT"
CATALOG_LOGICAL_PATH = "analysis/scientific_smoke_artifact_catalog.json"
SELECTION_LOGICAL_PATH = "analysis/smoke_budget_selection.json"
RECEIPT_SCHEMA_VERSION = 1
CATALOG_SCHEMA_VERSION = 2
PROTOCOL_BINDING_SCHEMA_VERSION = 1
RUNNER_SCHEMA_VERSION = 4

# Exact-CUDA-graph scientific geometry is an independent storage invariant, not
# a value delegated to whichever orchestration config happens to call us.
SCIENTIFIC_BUDGETS = (49152, 61440)
MAX_NUM_SEQS_BY_BUDGET = {49152: 19, 61440: 15}
CUDAGRAPH_CAPTURE_SIZES_BY_BUDGET = {
    49152: (1, 2, 4, 8, 16, 19),
    61440: (1, 2, 4, 8, 15),
}
SCIENTIFIC_MATRIX_ARMS = ("base", "designed_ceiling")
SCIENTIFIC_MATRIX_K = 12
SCIENTIFIC_PROBE_K = 4
SCIENTIFIC_N_RECORDS = 12
SCIENTIFIC_RUN_SEED = 2701

def expected_max_num_seqs(budget: int) -> int:
    """Return the preregistered live-KV-safe concurrency for one independent rung."""

    _require(
        budget in MAX_NUM_SEQS_BY_BUDGET,
        f"unregistered scientific thinking budget: {budget}",
    )
    return MAX_NUM_SEQS_BY_BUDGET[budget]


def expected_cudagraph_capture_sizes(budget: int) -> tuple[int, ...]:
    """Return the explicit graph shapes whose maximum is the active batch width."""

    _require(
        budget in CUDAGRAPH_CAPTURE_SIZES_BY_BUDGET,
        f"unregistered scientific thinking budget: {budget}",
    )
    sizes = CUDAGRAPH_CAPTURE_SIZES_BY_BUDGET[budget]
    _require(
        sizes[-1] == expected_max_num_seqs(budget),
        "registered CUDA-graph maximum differs from max_num_seqs",
    )
    return sizes


def _expected_engine(budget: int) -> dict[str, Any]:
    return {
        "max_model_len": 65536,
        "gpu_memory_utilization": 0.9,
        "max_num_seqs": expected_max_num_seqs(budget),
        "max_num_batched_tokens": 32768,
        "cudagraph_capture_sizes": list(expected_cudagraph_capture_sizes(budget)),
        "enable_prefix_caching": False,
        "enforce_eager": False,
        "adapter": None,
    }


def _expected_engine_args(budget: int) -> dict[str, Any]:
    max_num_seqs = expected_max_num_seqs(budget)
    return {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "language_model_only": True,
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.9,
        "max_model_len": 65536,
        "max_num_seqs": max_num_seqs,
        "max_num_batched_tokens": 32768,
        "cudagraph_capture_sizes": list(expected_cudagraph_capture_sizes(budget)),
        "max_cudagraph_capture_size": max_num_seqs,
        "enable_prefix_caching": False,
        "enforce_eager": False,
        "generation_config": "vllm",
        "seed": 0,
        "max_logprobs": 20,
        "async_scheduling": False,
        "mamba_cache_mode": "none",
    }


def _expected_resolved_cudagraph(budget: int) -> dict[str, Any]:
    return {
        "source": "llm_engine.vllm_config.compilation_config",
        "cudagraph_capture_sizes": list(expected_cudagraph_capture_sizes(budget)),
        "max_cudagraph_capture_size": expected_max_num_seqs(budget),
        "mode": "FULL_DECODE_ONLY",
        "decode_mode": "FULL",
        "mixed_mode": "NONE",
        "has_full_cudagraphs": True,
    }


def _validate_resolved_cudagraph(
    value: Mapping[str, Any], *, budget: int, where: str
) -> dict[str, Any]:
    resolved = _json_clone(dict(value), where=where)
    expected_fields = {
        "source",
        "cudagraph_capture_sizes",
        "max_cudagraph_capture_size",
        "mode",
        "decode_mode",
        "mixed_mode",
        "has_full_cudagraphs",
    }
    _require(set(resolved) == expected_fields, f"{where} fields mismatch")
    _require(
        resolved.get("source") == "llm_engine.vllm_config.compilation_config",
        f"{where} source mismatch",
    )
    _require(
        resolved.get("cudagraph_capture_sizes")
        == list(expected_cudagraph_capture_sizes(budget))
        and resolved.get("max_cudagraph_capture_size")
        == expected_max_num_seqs(budget),
        f"{where} sizes or maximum drifted",
    )
    mode_geometry = {
        "FULL": ("FULL", "FULL"),
        "FULL_DECODE_ONLY": ("FULL", "NONE"),
        "FULL_AND_PIECEWISE": ("FULL", "PIECEWISE"),
    }
    mode = resolved.get("mode")
    _require(mode in mode_geometry, f"{where} lacks a full-decode CUDA-graph mode")
    expected_decode, expected_mixed = mode_geometry[mode]
    _require(
        resolved.get("decode_mode") == expected_decode
        and resolved.get("mixed_mode") == expected_mixed
        and resolved.get("has_full_cudagraphs") is True,
        f"{where} resolved mode does not enable full decode CUDA graphs",
    )
    return resolved


_EXPECTED_RESOLVED_SAMPLING = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "repetition_penalty": 1.0,
}
_EXPECTED_THINK_TOKEN_IDS = {
    "open": 248068,
    "close": 248069,
    "forced_close_sequence": [248069, 271],
    "thinking_prompt_suffix": [248045, 74455, 198, 248068, 198],
    "no_thinking_prompt_suffix": [248045, 74455, 198, 248068, 271, 248069, 271],
}
_EXPECTED_TERMINATION = {
    "hf_model_eos_token_id": 248044,
    "vllm_tokenizer_eos_ignored": 248046,
}
_EXPECTED_RNG_ISOLATION = {
    "engine_seed": 0,
    "caller_global_rng_state_restored": True,
}

_PROTOCOL_FILES = (
    "configs/default.yaml",
    "data/tasks.json",
    "data/demonstrations.json",
    "data/prompt_manifest.json",
    "data/source_provenance.json",
)
_PROTOCOL_SOURCES = (
    "scripts/analyze.py",
    "scripts/run.py",
    "src/macro_domain.py",
    "src/model_harness.py",
    "src/scientific_artifacts.py",
    "src/vllm_runner.py",
)
_SMOKE_LIBRARY_ARMS = SCIENTIFIC_MATRIX_ARMS

_NAMESPACES = {"smoke_tiers", "smoke_budget_probes"}
_ARM_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_THINK_RE = re.compile(r"^think_([1-9][0-9]*)$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FILE_SUFFIXES = (
    ".preflight.json",
    ".jsonl",
    ".meta.json",
    ".receipt.json",
)


class ScientificArtifactError(ValueError):
    """A scientific artifact is missing, unsafe, partial, or identity-invalid."""


@dataclass(frozen=True)
class BundlePaths:
    """The four flat paths that form one committed scientific artifact."""

    prefix: str
    preflight: Path
    rows: Path
    metadata: Path
    receipt: Path


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ScientificArtifactError(message)


def _canonical_bytes(value: Any, *, pretty: bool = False) -> bytes:
    options: dict[str, Any] = {
        "ensure_ascii": False,
        "allow_nan": False,
        "sort_keys": True,
    }
    if pretty:
        text = json.dumps(value, indent=2, **options) + "\n"
    else:
        text = json.dumps(value, separators=(",", ":"), **options)
    return text.encode("utf-8")


def _json_clone(value: Any, *, where: str) -> Any:
    try:
        return json.loads(_canonical_bytes(value).decode("utf-8"))
    except (TypeError, ValueError) as exc:
        raise ScientificArtifactError(f"{where} is not canonical JSON: {exc}") from exc


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_value(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _assert_absolute_no_symlinks(path: Path, *, where: str) -> None:
    _require(path.is_absolute(), f"{where} must be an absolute path: {path}")
    current = Path(path.anchor)
    if _lexists(current):
        _require(not current.is_symlink(), f"{where} traverses a symlink: {current}")
    for part in path.parts[1:]:
        current = current / part
        if _lexists(current):
            _require(not current.is_symlink(), f"{where} traverses a symlink: {current}")


def _normalize_relative(value: str | Path, *, where: str) -> str:
    raw = value.as_posix() if isinstance(value, Path) else str(value)
    _require(bool(raw), f"{where} must not be empty")
    _require("\\" not in raw, f"{where} must use POSIX separators: {raw!r}")
    relative = PurePosixPath(raw)
    _require(not relative.is_absolute(), f"{where} must be relative: {raw!r}")
    _require(
        all(part not in {"", ".", ".."} for part in relative.parts),
        f"{where} contains a traversal or empty component: {raw!r}",
    )
    canonical = relative.as_posix()
    _require(canonical == raw, f"{where} is not canonical: {raw!r}")
    return canonical


def resolve_artifact_root(
    explicit: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Resolve an explicit root, then the environment override, then the default.

    Roots are deliberately absolute and may not contain a symlink component.
    The returned path is not created by this read-only resolver.
    """

    env = os.environ if environ is None else environ
    configured = explicit if explicit is not None else env.get(ARTIFACT_ROOT_ENV)
    root = Path(configured) if configured is not None else DEFAULT_ARTIFACT_ROOT
    root = root.expanduser()
    _assert_absolute_no_symlinks(root, where="scientific artifact root")
    resolved_root = root.resolve(strict=False)
    for forbidden in (PREDECESSOR_ARTIFACT_ROOT, OTHER_FORBIDDEN_ARTIFACT_ROOT):
        resolved_forbidden = forbidden.resolve(strict=False)
        _require(
            not resolved_root.is_relative_to(resolved_forbidden)
            and not resolved_forbidden.is_relative_to(resolved_root),
            "exact-CUDA-graph artifacts may not equal, contain, or nest inside a "
            "predecessor root",
        )
    return root


def safe_path(root: str | Path, relative: str | Path) -> Path:
    """Resolve a canonical root-relative path without following any symlink."""

    root_path = resolve_artifact_root(root)
    normalized = _normalize_relative(relative, where="scientific artifact path")
    candidate = root_path.joinpath(*PurePosixPath(normalized).parts)
    _assert_absolute_no_symlinks(candidate, where="scientific artifact path")
    resolved_root = root_path.resolve(strict=False)
    resolved_candidate = candidate.resolve(strict=False)
    _require(
        resolved_candidate.is_relative_to(resolved_root),
        f"scientific artifact path escapes its root: {normalized}",
    )
    return candidate


def _geometry(namespace: str, budget: int, arm: str) -> tuple[int, int]:
    _require(
        budget in SCIENTIFIC_BUDGETS,
        f"unregistered scientific thinking budget: {budget}",
    )
    if namespace == "smoke_tiers":
        _require(
            arm in SCIENTIFIC_MATRIX_ARMS,
            f"scientific matrix arm must be one of {SCIENTIFIC_MATRIX_ARMS}: {arm!r}",
        )
        return SCIENTIFIC_MATRIX_K, SCIENTIFIC_N_RECORDS
    _require(namespace == "smoke_budget_probes", f"invalid scientific namespace: {namespace}")
    _require(arm == "base", f"termination probes are base-only: {arm!r}")
    return SCIENTIFIC_PROBE_K, SCIENTIFIC_N_RECORDS


def _expected_sampling(*, budget: int, k: int) -> dict[str, Any]:
    return {
        "thinking": "budget",
        "thinking_budget": budget,
        "n": k,
        "max_tokens": ANSWER_MAX_TOKENS,
        "answer_max_tokens": ANSWER_MAX_TOKENS,
        "greedy": False,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        "run_seed": SCIENTIFIC_RUN_SEED,
        "shuffle_thinking": False,
        "logprobs": None,
        "prompt_logprobs": None,
        "logprob_token_ids": [],
        "allow_custom_prompts": False,
    }


def _stable_seed(run_seed: int, record_id: str, sample_index: int, stage: str) -> int:
    payload = f"{run_seed}\0{record_id}\0{sample_index}\0{stage}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") % (2**31)


def _validate_prefix(relative_prefix: str | Path) -> tuple[str, int, str, str]:
    normalized = _normalize_relative(relative_prefix, where="artifact bundle prefix")
    parts = PurePosixPath(normalized).parts
    _require(
        len(parts) == 3 and parts[0] in _NAMESPACES,
        "artifact bundle prefix must be "
        "smoke_tiers/think_<budget>/<arm> or "
        f"smoke_budget_probes/think_<budget>/<arm>: {normalized}",
    )
    match = _THINK_RE.fullmatch(parts[1])
    _require(match is not None, f"invalid thinking-budget directory: {parts[1]}")
    arm = parts[2]
    _require(_ARM_RE.fullmatch(arm) is not None, f"invalid scientific arm: {arm!r}")
    namespace = parts[0]
    budget = int(match.group(1))
    _geometry(namespace, budget, arm)
    return normalized, budget, arm, namespace


def bundle_paths(root: str | Path, relative_prefix: str | Path) -> BundlePaths:
    """Return the only permitted flat scientific-smoke artifact paths."""

    prefix, _, _, _ = _validate_prefix(relative_prefix)
    return BundlePaths(
        prefix=prefix,
        preflight=safe_path(root, prefix + ".preflight.json"),
        rows=safe_path(root, prefix + ".jsonl"),
        metadata=safe_path(root, prefix + ".meta.json"),
        receipt=safe_path(root, prefix + ".receipt.json"),
    )


def _ensure_parent(path: Path) -> None:
    _assert_absolute_no_symlinks(path, where="artifact write path")
    path.parent.mkdir(parents=True, exist_ok=True)
    _assert_absolute_no_symlinks(path.parent, where="artifact write directory")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_tree_and_parent(root: str | Path) -> None:
    """Durably flush a validated staged tree, its directories, and its parent.

    Migration uses a directory rename as its commit point.  Fsyncing only the
    files is insufficient: the copied directory entries and the later rename
    also need durable directory metadata.  The helper is model-free and rejects
    the same symlinks/unknown layout as catalog construction.
    """

    root_path = resolve_artifact_root(root)
    _require(root_path.is_dir(), f"scientific artifact tree is missing: {root_path}")
    files = _walk_files(root_path)
    for path in files:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    directories = {root_path}
    for path in files:
        current = path.parent
        while True:
            directories.add(current)
            if current == root_path:
                break
            current = current.parent
    for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        _fsync_directory(directory)
    _assert_absolute_no_symlinks(root_path.parent, where="scientific artifact tree parent")
    _fsync_directory(root_path.parent)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    """Write one file durably without ever following a target or temp symlink."""

    _ensure_parent(path)
    _require(not path.is_symlink(), f"refusing to replace symlink: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    _require(not _lexists(temporary), f"temporary artifact unexpectedly exists: {temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _require(not path.is_symlink(), f"artifact target became a symlink: {path}")
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if _lexists(temporary):
            temporary.unlink()


def _read_json(path: Path, *, where: str) -> dict[str, Any]:
    _assert_absolute_no_symlinks(path, where=where)
    _require(path.is_file(), f"missing {where}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ScientificArtifactError(f"invalid JSON in {where} {path}: {exc}") from exc
    _require(isinstance(value, dict), f"{where} must contain a JSON object: {path}")
    return value


def _file_digest(root: Path, path: Path) -> dict[str, Any]:
    _assert_absolute_no_symlinks(path, where="artifact file")
    _require(path.is_file(), f"missing artifact file: {path}")
    relative = path.relative_to(root).as_posix()
    return {
        "relative_path": relative,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _protocol_file_digest(experiment_root: Path, relative: str) -> dict[str, Any]:
    path = experiment_root / relative
    _assert_absolute_no_symlinks(path, where=f"protocol file {relative}")
    _require(path.is_file(), f"missing protocol file: {path}")
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def build_protocol_binding(experiment_root: str | Path) -> dict[str, Any]:
    """Bind every repository input and implementation that can change smoke scoring.

    The whole ``libraries.json`` is intentionally not hashed: train-only Qwen
    proposal construction may append Qwen-only arms after smoke passes.  The
    canonical base and designed-ceiling library payloads are bound separately,
    so those later additions cannot invalidate an already frozen smoke.
    """

    root = Path(experiment_root).expanduser()
    _assert_absolute_no_symlinks(root, where="experiment protocol root")
    _require(root.is_dir(), f"experiment protocol root is missing: {root}")
    files = [
        _protocol_file_digest(root, relative)
        for relative in (*_PROTOCOL_FILES, *_PROTOCOL_SOURCES)
    ]
    library_path = root / "data" / "libraries.json"
    libraries_payload = _read_json(library_path, where="prepared libraries")
    raw_libraries = libraries_payload.get("libraries")
    _require(isinstance(raw_libraries, Mapping), "prepared libraries lack a libraries object")
    libraries: dict[str, Any] = {}
    for arm in _SMOKE_LIBRARY_ARMS:
        library = raw_libraries.get(arm)
        _require(isinstance(library, Mapping), f"prepared libraries lack smoke arm {arm}")
        canonical = _json_clone(dict(library), where=f"smoke library {arm}")
        library_id = _string(canonical.get("id"), where=f"smoke library {arm}.id")
        libraries[arm] = {
            "library_id": library_id,
            "content_sha256": _sha256_value(canonical),
        }
    core = {
        "schema_version": PROTOCOL_BINDING_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "files": files,
        "smoke_libraries": libraries,
        "library_scope": (
            "base and designed_ceiling only; later Qwen-only library additions are excluded"
        ),
    }
    return {**core, "binding_sha256": _sha256_value(core)}


def _validate_protocol_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _json_clone(dict(binding), where="scientific protocol binding")
    _require(
        normalized.get("schema_version") == PROTOCOL_BINDING_SCHEMA_VERSION,
        "scientific protocol binding schema mismatch",
    )
    _require(normalized.get("experiment_id") == EXPERIMENT_ID, "protocol experiment mismatch")
    binding_sha256 = normalized.pop("binding_sha256", None)
    _require(
        binding_sha256 == _sha256_value(normalized),
        "scientific protocol binding hash mismatch",
    )
    normalized["binding_sha256"] = binding_sha256
    return normalized


def _sha256(value: Any, *, where: str) -> str:
    _require(
        isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None,
        f"{where} must be a lowercase SHA-256 digest",
    )
    return value


def _integer(value: Any, *, where: str, minimum: int = 0) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
        f"{where} must be an integer >= {minimum}",
    )
    return value


def _string(value: Any, *, where: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{where} must be a non-empty string")
    return value


def _ordered_records(preflight: Mapping[str, Any]) -> list[dict[str, Any]]:
    _require(preflight.get("schema_version") == 1, "preflight schema mismatch")
    _require(preflight.get("pass") is True, "preflight did not pass")
    _validate_protocol_binding(
        preflight.get("protocol_binding")
        if isinstance(preflight.get("protocol_binding"), Mapping)
        else {}
    )
    max_model_len = _integer(
        preflight.get("max_model_len"), where="preflight.max_model_len", minimum=1
    )
    _require(max_model_len == 65536, "scientific preflight max_model_len must be 65536")
    reserve = _integer(
        preflight.get("generation_reserve_tokens"),
        where="preflight.generation_reserve_tokens",
        minimum=1,
    )
    raw_records = preflight.get("records")
    _require(isinstance(raw_records, list) and bool(raw_records), "preflight has no records")
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_records):
        _require(isinstance(raw, Mapping), f"preflight record {index} must be an object")
        record_id = _string(raw.get("id"), where=f"preflight record {index}.id")
        _require(record_id not in seen, f"duplicate preflight record id: {record_id}")
        seen.add(record_id)
        prompt_tokens = _integer(
            raw.get("prompt_tokens"),
            where=f"preflight record {record_id}.prompt_tokens",
            minimum=1,
        )
        prompt_plus_reserve = _integer(
            raw.get("prompt_plus_reserve_tokens"),
            where=f"preflight record {record_id}.prompt_plus_reserve_tokens",
            minimum=1,
        )
        _require(
            prompt_plus_reserve == prompt_tokens + reserve,
            f"preflight reserve arithmetic mismatch for {record_id}",
        )
        _require(
            prompt_plus_reserve <= max_model_len,
            f"preflight context overflow for {record_id}",
        )
        ordered.append(
            {
                "id": record_id,
                "input_record_sha256": _sha256(
                    raw.get("input_record_sha256"),
                    where=f"preflight record {record_id}.input_record_sha256",
                ),
                "rendered_prompt_sha256": _sha256(
                    raw.get("rendered_prompt_sha256"),
                    where=f"preflight record {record_id}.rendered_prompt_sha256",
                ),
                "prompt_tokens": prompt_tokens,
            }
        )
    _require(
        preflight.get("n_records") == len(ordered),
        "preflight n_records disagrees with its ordered records",
    )
    prompt_counts = [record["prompt_tokens"] for record in ordered]
    _require(preflight.get("min_prompt_tokens") == min(prompt_counts), "preflight min mismatch")
    _require(preflight.get("max_prompt_tokens") == max(prompt_counts), "preflight max mismatch")
    _require(
        preflight.get("max_prompt_plus_reserve_tokens") == max(prompt_counts) + reserve,
        "preflight maximum reserve arithmetic mismatch",
    )
    return ordered


def _validate_capacity_fit(
    preflight: Mapping[str, Any], *, budget: int, k: int
) -> dict[str, Any]:
    """Verify the measured engine cache can hold every active sequence at its cap."""

    raw = preflight.get("capacity_fit")
    _require(isinstance(raw, Mapping), "preflight lacks a live capacity-fit audit")
    audit = _json_clone(dict(raw), where="preflight capacity-fit audit")
    expected_fields = {
        "source",
        "kv_cache_size_tokens",
        "block_size",
        "live_max_model_len",
        "max_num_seqs",
        "logical_sequences",
        "active_sequences",
        "max_prompt_plus_reserve_tokens",
        "rounded_tokens_per_sequence",
        "required_cache_tokens",
        "pass",
    }
    _require(set(audit) == expected_fields, "capacity-fit audit fields mismatch")
    _require(
        audit.get("source") == "vllm_config.cache_config.kv_cache_size_tokens",
        "capacity-fit audit source mismatch",
    )
    capacity = _integer(
        audit.get("kv_cache_size_tokens"),
        where="capacity-fit kv_cache_size_tokens",
        minimum=1,
    )
    block_size = _integer(
        audit.get("block_size"), where="capacity-fit block_size", minimum=1
    )
    _require(
        audit.get("live_max_model_len") == 65536,
        "live vLLM model context differs from 65536",
    )
    max_num_seqs = expected_max_num_seqs(budget)
    _require(
        audit.get("max_num_seqs") == max_num_seqs,
        "capacity-fit max_num_seqs differs from the registered rung mapping",
    )
    logical_sequences = SCIENTIFIC_N_RECORDS * k
    active_sequences = min(logical_sequences, max_num_seqs)
    _require(
        audit.get("logical_sequences") == logical_sequences
        and audit.get("active_sequences") == active_sequences,
        "capacity-fit active sequence geometry mismatch",
    )
    max_total = _integer(
        preflight.get("max_prompt_plus_reserve_tokens"),
        where="preflight.max_prompt_plus_reserve_tokens",
        minimum=1,
    )
    rounded = ((max_total + block_size - 1) // block_size) * block_size
    required = active_sequences * rounded
    _require(
        audit.get("max_prompt_plus_reserve_tokens") == max_total
        and audit.get("rounded_tokens_per_sequence") == rounded
        and audit.get("required_cache_tokens") == required,
        "capacity-fit cache arithmetic mismatch",
    )
    _require(
        audit.get("pass") is True and required <= capacity,
        "live vLLM KV cache is insufficient for the registered concurrency",
    )
    return audit


def _validate_cudagraph_geometry(
    preflight: Mapping[str, Any], *, budget: int, k: int
) -> dict[str, Any]:
    """Verify vLLM kept the explicit graph list and covers the active width."""

    raw = preflight.get("cudagraph_geometry")
    _require(isinstance(raw, Mapping), "preflight lacks a CUDA-graph geometry audit")
    audit = _json_clone(dict(raw), where="preflight CUDA-graph geometry audit")
    expected_fields = {
        "source",
        "requested_capture_sizes",
        "resolved_capture_sizes",
        "requested_max_capture_size",
        "resolved_max_capture_size",
        "resolved_mode",
        "resolved_decode_mode",
        "resolved_mixed_mode",
        "has_full_cudagraphs",
        "active_sequences",
        "active_width_covered",
        "pass",
    }
    _require(set(audit) == expected_fields, "CUDA-graph geometry audit fields mismatch")
    _require(
        audit.get("source") == "llm_engine.vllm_config.compilation_config",
        "CUDA-graph geometry audit source mismatch",
    )
    expected_sizes = list(expected_cudagraph_capture_sizes(budget))
    expected_max = expected_max_num_seqs(budget)
    active = min(SCIENTIFIC_N_RECORDS * k, expected_max)
    _require(
        audit.get("requested_capture_sizes") == expected_sizes
        and audit.get("resolved_capture_sizes") == expected_sizes,
        "requested or resolved CUDA-graph capture sizes drifted",
    )
    _require(
        audit.get("requested_max_capture_size") == expected_max
        and audit.get("resolved_max_capture_size") == expected_max,
        "requested or resolved CUDA-graph maximum drifted",
    )
    _validate_resolved_cudagraph(
        {
            "source": audit.get("source"),
            "cudagraph_capture_sizes": audit.get("resolved_capture_sizes"),
            "max_cudagraph_capture_size": audit.get("resolved_max_capture_size"),
            "mode": audit.get("resolved_mode"),
            "decode_mode": audit.get("resolved_decode_mode"),
            "mixed_mode": audit.get("resolved_mixed_mode"),
            "has_full_cudagraphs": audit.get("has_full_cudagraphs"),
        },
        budget=budget,
        where="preflight resolved CUDA-graph geometry",
    )
    _require(
        audit.get("active_sequences") == active,
        "CUDA-graph active sequence width mismatch",
    )
    _require(
        audit.get("active_width_covered") is True
        and any(size >= active for size in expected_sizes)
        and audit.get("resolved_decode_mode") == "FULL"
        and audit.get("has_full_cudagraphs") is True
        and audit.get("pass") is True,
        "active decode width is not covered by the frozen CUDA-graph list",
    )
    return audit


def _validate_ordered_geometry(
    ordered: Sequence[Mapping[str, Any]], *, arm: str, expected_n_records: int
) -> None:
    _require(
        len(ordered) == expected_n_records == SCIENTIFIC_N_RECORDS,
        f"scientific bundle must contain exactly {SCIENTIFIC_N_RECORDS} records",
    )
    task_ids: list[str] = []
    for record in ordered:
        record_id = str(record["id"])
        suffix = f"::{arm}"
        _require(record_id.endswith(suffix), f"record id/arm mismatch: {record_id}")
        task_id = record_id[: -len(suffix)]
        _require(bool(task_id), f"record id lacks task identity: {record_id}")
        task_ids.append(task_id)
    _require(len(set(task_ids)) == len(task_ids), "scientific task ids must be unique")


def write_preflight_only(
    root: str | Path,
    relative_prefix: str | Path,
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze a preflight as the one valid incomplete/resumable bundle state."""

    root_path = resolve_artifact_root(root)
    prefix, budget, arm, namespace = _validate_prefix(relative_prefix)
    _, expected_n_records = _geometry(namespace, budget, arm)
    paths = bundle_paths(root_path, prefix)
    ordered = _ordered_records(preflight)
    _validate_ordered_geometry(
        ordered, arm=arm, expected_n_records=expected_n_records
    )
    expected_k, _ = _geometry(namespace, budget, arm)
    _validate_capacity_fit(preflight, budget=budget, k=expected_k)
    _validate_cudagraph_geometry(preflight, budget=budget, k=expected_k)
    _require(
        preflight.get("generation_reserve_tokens")
        == budget + FORCED_CLOSE_TOKENS + ANSWER_MAX_TOKENS,
        "preflight generation reserve differs from its scientific tier",
    )
    for forbidden in (paths.rows, paths.metadata, paths.receipt):
        _require(not _lexists(forbidden), f"cannot write preflight over partial bundle: {forbidden}")
    payload = _canonical_bytes(dict(preflight), pretty=True)
    if _lexists(paths.preflight):
        _require(not paths.preflight.is_symlink(), f"preflight is a symlink: {paths.preflight}")
        _require(
            paths.preflight.read_bytes() == payload,
            f"frozen preflight differs from regeneration: {paths.preflight}",
        )
    else:
        _atomic_bytes(paths.preflight, payload)
    return _file_digest(root_path, paths.preflight)


def _validate_rows_identity(
    rows_path: Path,
    *,
    ordered_records: Sequence[Mapping[str, Any]],
    k: int,
    arm: str,
) -> None:
    row_count = 0
    seen_task_ids: set[str] = set()
    try:
        handle = rows_path.open("r", encoding="utf-8")
    except FileNotFoundError as exc:
        raise ScientificArtifactError(f"missing runner rows: {rows_path}") from exc
    with handle:
        for line_number, line in enumerate(handle, 1):
            _require(bool(line.strip()), f"blank JSONL row at {rows_path}:{line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ScientificArtifactError(
                    f"invalid JSONL at {rows_path}:{line_number}: {exc}"
                ) from exc
            _require(isinstance(row, dict), f"runner row {line_number} must be an object")
            _require(row_count < len(ordered_records), f"too many runner rows in {rows_path}")
            expected = ordered_records[row_count]
            record_id = str(expected["id"])
            _require(row.get("id") == record_id, f"runner row order/id mismatch for {record_id}")
            meta = row.get("meta")
            _require(isinstance(meta, Mapping), f"runner row {record_id} lacks task metadata")
            task_id = _string(meta.get("task_id"), where=f"runner row {record_id}.meta.task_id")
            _require(task_id not in seen_task_ids, f"duplicate runner task id: {task_id}")
            seen_task_ids.add(task_id)
            _require(record_id == f"{task_id}::{arm}", f"runner row task/id mismatch: {record_id}")
            _require(meta.get("arm") == arm, f"runner row arm mismatch for {record_id}")
            _string(meta.get("library_id"), where=f"runner row {record_id}.meta.library_id")
            _require(
                meta.get("split") in {"smoke_reuse", "smoke_no_reuse"},
                f"runner row split mismatch for {record_id}",
            )
            _require(
                meta.get("prompt_kind") == "solve_program",
                f"runner row prompt kind mismatch for {record_id}",
            )
            _require(
                meta.get("max_surface_calls") == 5
                and meta.get("max_expanded_primitive_depth") == 5,
                f"runner row solver geometry mismatch for {record_id}",
            )
            _require(
                isinstance(meta.get("macros_callable"), bool),
                f"runner row macros_callable missing for {record_id}",
            )
            _require(
                row.get("prompt_sha256") == expected["rendered_prompt_sha256"],
                f"runner prompt hash mismatch for {record_id}",
            )
            _require(
                row.get("n_prompt_tokens") == expected["prompt_tokens"],
                f"runner prompt-token count mismatch for {record_id}",
            )
            _require(
                row.get("prompt_channel") == "thinking",
                f"runner prompt channel mismatch for {record_id}",
            )
            _require(
                row.get("prompt_logprobs") is None,
                f"scientific runner prompt logprobs must be disabled for {record_id}",
            )
            outputs = row.get("outputs")
            _require(
                isinstance(outputs, list) and len(outputs) == k,
                f"runner K mismatch for {record_id}",
            )
            indices: list[Any] = []
            stage1_parent_seed = _stable_seed(
                SCIENTIFIC_RUN_SEED, record_id, -1, "stage1"
            )
            for output in outputs:
                _require(isinstance(output, Mapping), f"runner output for {record_id} is invalid")
                sample_index = output.get("sample_index")
                sample_index_value = _integer(
                    sample_index,
                    where=f"runner output {record_id}.sample_index",
                )
                indices.append(sample_index)
                _require(
                    output.get("stage1_parent_seed") == stage1_parent_seed,
                    f"runner stage-1 parent seed mismatch for {record_id}/{sample_index}",
                )
                _require(
                    output.get("seed_stage1") == stage1_parent_seed + sample_index_value,
                    f"runner stage-1 seed mismatch for {record_id}/{sample_index}",
                )
                seed_stage2 = output.get("seed_stage2")
                if seed_stage2 is None:
                    _require(
                        output.get("injected_token_ids") == [],
                        f"natural close has injected tokens for {record_id}/{sample_index}",
                    )
                else:
                    _require(
                        seed_stage2
                        == _stable_seed(
                            SCIENTIFIC_RUN_SEED, record_id, sample_index_value, "stage2"
                        ),
                        f"runner stage-2 seed mismatch for {record_id}/{sample_index}",
                    )
                    _require(
                        output.get("injected_token_ids")
                        == _EXPECTED_THINK_TOKEN_IDS["forced_close_sequence"],
                        f"stage-2 close sequence mismatch for {record_id}/{sample_index}",
                    )
                _require(
                    output.get("thinking_closed") is True,
                    f"scientific output lacks a closed thinking region for {record_id}/{sample_index}",
                )
                _require(
                    isinstance(output.get("forced_close"), bool),
                    f"scientific output lacks forced-close metadata for {record_id}/{sample_index}",
                )
                for token_field in ("token_ids", "stage1_token_ids", "injected_token_ids", "stage2_token_ids"):
                    token_ids = output.get(token_field)
                    _require(
                        isinstance(token_ids, list)
                        and all(isinstance(token, int) and not isinstance(token, bool) for token in token_ids),
                        f"runner {token_field} is invalid for {record_id}/{sample_index}",
                    )
                for count_field in (
                    "n_thinking_tokens",
                    "n_answer_tokens",
                    "n_sampled_tokens",
                    "n_injected_tokens",
                    "n_completion_tokens",
                    "n_terminal_tokens_trimmed",
                    "n_stage1_prompt_tokens",
                    "n_stage2_prompt_tokens",
                ):
                    _integer(
                        output.get(count_field),
                        where=f"runner output {record_id}/{sample_index}.{count_field}",
                    )
                _require(
                    output.get("n_stage1_prompt_tokens") == expected["prompt_tokens"],
                    f"runner stage-1 prompt count mismatch for {record_id}/{sample_index}",
                )
                _require(
                    output.get("n_injected_tokens") == len(output["injected_token_ids"]),
                    f"runner injected-token accounting mismatch for {record_id}/{sample_index}",
                )
            _require(indices == list(range(k)), f"runner sample order mismatch for {record_id}")
            row_count += 1
    _require(
        row_count == len(ordered_records),
        f"runner row count {row_count} != preflight count {len(ordered_records)}",
    )


def _identity_from_metadata(
    metadata: Mapping[str, Any],
    *,
    n_records: int,
    k: int,
    thinking_budget: int,
) -> dict[str, Any]:
    _require(
        metadata.get("schema_version") == RUNNER_SCHEMA_VERSION,
        f"runner metadata schema must be {RUNNER_SCHEMA_VERSION}",
    )
    model = _string(metadata.get("model"), where="runner metadata.model")
    _require(model == MODEL_ID, f"runner metadata used forbidden model: {model}")
    revision = _string(
        metadata.get("model_revision"), where="runner metadata.model_revision"
    )
    _require(revision == MODEL_REVISION, "runner metadata model revision mismatch")
    runner_sha256 = _sha256(
        metadata.get("runner_sha256"), where="runner metadata.runner_sha256"
    )
    _require(runner_sha256 == RUNNER_SHA256, "runner metadata runner hash mismatch")
    _require(metadata.get("adapter") is None, "scientific smoke must not use an adapter")
    sampling = metadata.get("sampling")
    resolved_sampling = metadata.get("resolved_sampling")
    engine = metadata.get("engine")
    engine_args = metadata.get("engine_args")
    resolved_cudagraph = metadata.get("resolved_cudagraph")
    think_token_ids = metadata.get("think_token_ids")
    termination = metadata.get("termination")
    rng_isolation = metadata.get("rng_isolation")
    runtime = metadata.get("runtime")
    _require(isinstance(sampling, Mapping), "runner metadata.sampling must be an object")
    _require(
        isinstance(resolved_sampling, Mapping),
        "runner metadata.resolved_sampling must be an object",
    )
    _require(isinstance(engine, Mapping), "runner metadata.engine must be an object")
    _require(isinstance(engine_args, Mapping), "runner metadata.engine_args must be an object")
    _require(
        isinstance(resolved_cudagraph, Mapping),
        "runner metadata.resolved_cudagraph must be an object",
    )
    _require(
        isinstance(think_token_ids, Mapping),
        "runner metadata.think_token_ids must be an object",
    )
    _require(isinstance(termination, Mapping), "runner metadata.termination must be an object")
    _require(
        isinstance(rng_isolation, Mapping),
        "runner metadata.rng_isolation must be an object",
    )
    _require(isinstance(runtime, Mapping), "runner metadata.runtime must be an object")
    sampling_json = _json_clone(dict(sampling), where="runner metadata.sampling")
    resolved_sampling_json = _json_clone(
        dict(resolved_sampling), where="runner metadata.resolved_sampling"
    )
    engine_json = _json_clone(dict(engine), where="runner metadata.engine")
    engine_args_json = _json_clone(dict(engine_args), where="runner metadata.engine_args")
    resolved_cudagraph_json = _json_clone(
        dict(resolved_cudagraph), where="runner metadata.resolved_cudagraph"
    )
    think_token_ids_json = _json_clone(
        dict(think_token_ids), where="runner metadata.think_token_ids"
    )
    termination_json = _json_clone(
        dict(termination), where="runner metadata.termination"
    )
    rng_isolation_json = _json_clone(
        dict(rng_isolation), where="runner metadata.rng_isolation"
    )
    runtime_protocol = _json_clone(dict(runtime), where="runner metadata.runtime")
    runtime_protocol.pop("git_commit", None)
    runtime_protocol.pop("git_dirty", None)
    _require(bool(runtime_protocol), "runner runtime protocol identity is empty")
    _require(
        sampling_json == _expected_sampling(budget=thinking_budget, k=k),
        "runner metadata sampling protocol mismatch",
    )
    _require(
        resolved_sampling_json == _EXPECTED_RESOLVED_SAMPLING,
        "runner metadata resolved sampling protocol mismatch",
    )
    _require(
        engine_json == _expected_engine(thinking_budget),
        "runner metadata engine protocol mismatch",
    )
    _require(
        engine_args_json == _expected_engine_args(thinking_budget),
        "runner metadata engine_args protocol mismatch",
    )
    resolved_cudagraph_json = _validate_resolved_cudagraph(
        resolved_cudagraph_json,
        budget=thinking_budget,
        where="runner metadata resolved CUDA-graph geometry",
    )
    _require(
        think_token_ids_json == _EXPECTED_THINK_TOKEN_IDS,
        "runner metadata think-token protocol mismatch",
    )
    _require(
        termination_json == _EXPECTED_TERMINATION,
        "runner metadata termination protocol mismatch",
    )
    _require(
        rng_isolation_json == _EXPECTED_RNG_ISOLATION,
        "runner metadata RNG-isolation protocol mismatch",
    )
    counts = metadata.get("counts")
    _require(isinstance(counts, Mapping), "runner metadata.counts must be an object")
    expected_count_fields = {
        "requests",
        "completions",
        "unique_input_prompt_tokens",
        "stage1_logical_prompt_tokens",
        "stage2_logical_prompt_tokens",
        "logical_model_input_tokens",
        "sampled_tokens",
        "injected_tokens",
    }
    _require(set(counts) == expected_count_fields, "runner metadata count fields mismatch")
    for field in expected_count_fields:
        _integer(counts.get(field), where=f"runner metadata.counts.{field}")
    _require(counts.get("requests") == n_records, "runner metadata request count mismatch")
    _require(
        counts.get("completions") == n_records * k,
        "runner metadata completion count mismatch",
    )
    _require(
        counts.get("logical_model_input_tokens")
        == counts.get("stage1_logical_prompt_tokens")
        + counts.get("stage2_logical_prompt_tokens"),
        "runner metadata logical prompt accounting mismatch",
    )
    return {
        "runner_schema_version": RUNNER_SCHEMA_VERSION,
        "model": model,
        "model_revision": revision,
        "runner_sha256": runner_sha256,
        "adapter": None,
        "sampling": sampling_json,
        "sampling_sha256": _sha256_value(sampling_json),
        "resolved_sampling": resolved_sampling_json,
        "resolved_sampling_sha256": _sha256_value(resolved_sampling_json),
        "engine": engine_json,
        "engine_sha256": _sha256_value(engine_json),
        "engine_args": engine_args_json,
        "engine_args_sha256": _sha256_value(engine_args_json),
        "resolved_cudagraph": resolved_cudagraph_json,
        "resolved_cudagraph_sha256": _sha256_value(resolved_cudagraph_json),
        "think_token_ids": think_token_ids_json,
        "termination": termination_json,
        "rng_isolation": rng_isolation_json,
        "runtime_protocol": runtime_protocol,
        "runtime_protocol_sha256": _sha256_value(runtime_protocol),
    }


def _validate_protocol_cross_bindings(
    preflight: Mapping[str, Any],
    identity: Mapping[str, Any],
    *,
    thinking_budget: int,
    k: int,
) -> None:
    sampling = identity.get("sampling")
    engine = identity.get("engine")
    resolved_cudagraph = identity.get("resolved_cudagraph")
    preflight_cudagraph = preflight.get("cudagraph_geometry")
    _require(isinstance(sampling, Mapping), "receipt sampling identity is missing")
    _require(isinstance(engine, Mapping), "receipt engine identity is missing")
    _require(
        isinstance(resolved_cudagraph, Mapping),
        "receipt resolved CUDA-graph identity is missing",
    )
    _require(
        isinstance(preflight_cudagraph, Mapping),
        "preflight resolved CUDA-graph identity is missing",
    )
    _require(sampling.get("thinking") == "budget", "scientific smoke must use budget thinking")
    _require(
        sampling.get("thinking_budget") == thinking_budget,
        "sampling thinking budget differs from the tier",
    )
    _require(sampling.get("n") == k, "sampling n differs from receipt K")
    answer_max_tokens = _integer(
        sampling.get("answer_max_tokens"),
        where="sampling.answer_max_tokens",
        minimum=1,
    )
    _require(
        answer_max_tokens == ANSWER_MAX_TOKENS,
        "scientific smoke answer allowance differs from 512",
    )
    _require(
        preflight.get("generation_reserve_tokens")
        == thinking_budget + FORCED_CLOSE_TOKENS + answer_max_tokens,
        "preflight generation reserve differs from sampling",
    )
    _require(
        engine.get("max_model_len") == preflight.get("max_model_len"),
        "preflight max_model_len differs from engine identity",
    )
    crosswalk = {
        "source": "source",
        "cudagraph_capture_sizes": "resolved_capture_sizes",
        "max_cudagraph_capture_size": "resolved_max_capture_size",
        "mode": "resolved_mode",
        "decode_mode": "resolved_decode_mode",
        "mixed_mode": "resolved_mixed_mode",
        "has_full_cudagraphs": "has_full_cudagraphs",
    }
    _require(
        all(
            resolved_cudagraph.get(metadata_key)
            == preflight_cudagraph.get(preflight_key)
            for metadata_key, preflight_key in crosswalk.items()
        ),
        "preflight and runner metadata resolved CUDA-graph identities differ",
    )


def _validate_expected(receipt: Mapping[str, Any], expected: Mapping[str, Any] | None) -> None:
    if expected is None:
        return
    identity = receipt.get("identity")
    _require(isinstance(identity, Mapping), "receipt identity is missing")
    for key, expected_value in expected.items():
        actual = identity.get(key) if key in identity else receipt.get(key)
        normalized = _json_clone(expected_value, where=f"expected receipt {key}")
        _require(actual == normalized, f"receipt {key} differs from the frozen expectation")


def comparable_protocol_identity(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Return equality-critical same-rung identity, ignoring only the expected K change."""

    raw_identity = receipt.get("identity")
    _require(isinstance(raw_identity, Mapping), "receipt lacks protocol identity")
    identity = _json_clone(dict(raw_identity), where="comparable protocol identity")
    sampling = identity.get("sampling")
    _require(isinstance(sampling, dict), "receipt identity lacks sampling")
    sampling.pop("n", None)
    identity.pop("sampling_sha256", None)
    return identity


def _role_for_namespace(namespace: str) -> tuple[str, str]:
    if namespace == "smoke_tiers":
        return "complete_matrix_arm", "complete_k12_matrix"
    return "termination_probe", "termination_probe_only"


def commit_receipt(
    root: str | Path,
    relative_prefix: str | Path,
    *,
    role: str,
    tier_mode: str,
    thinking_budget: int,
    arm: str,
    k: int,
    expected_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a complete flat bundle and atomically write its receipt last.

    Rows, metadata, and preflight must already exist.  The function never
    rewrites them.  An existing receipt is immutable and is verified instead.
    """

    root_path = resolve_artifact_root(root)
    prefix, prefix_budget, prefix_arm, namespace = _validate_prefix(relative_prefix)
    paths = bundle_paths(root_path, prefix)
    expected_k, expected_n_records = _geometry(
        namespace, prefix_budget, prefix_arm
    )
    expected_role, expected_mode = _role_for_namespace(namespace)
    _require(role == expected_role, f"role {role!r} does not match namespace {namespace}")
    _require(tier_mode == expected_mode, f"tier mode {tier_mode!r} does not match namespace")
    _require(thinking_budget == prefix_budget, "receipt budget differs from its directory")
    _require(arm == prefix_arm, "receipt arm differs from its filename")
    _require(k == expected_k, f"receipt K must be {expected_k} for {namespace}")

    if _lexists(paths.receipt):
        return verify_receipt(
            root_path,
            prefix,
            expected={
                "role": role,
                "tier_mode": tier_mode,
                "thinking_budget": thinking_budget,
                "arm": arm,
                "k": k,
                **dict(expected_identity or {}),
            },
        )

    preflight = _read_json(paths.preflight, where="runner preflight")
    metadata = _read_json(paths.metadata, where="runner metadata")
    _assert_absolute_no_symlinks(paths.rows, where="runner rows")
    _require(paths.rows.is_file(), f"missing runner rows: {paths.rows}")
    ordered = _ordered_records(preflight)
    _validate_ordered_geometry(
        ordered, arm=arm, expected_n_records=expected_n_records
    )
    _validate_capacity_fit(preflight, budget=thinking_budget, k=k)
    _validate_cudagraph_geometry(preflight, budget=thinking_budget, k=k)
    identity = _identity_from_metadata(
        metadata,
        n_records=len(ordered),
        k=k,
        thinking_budget=thinking_budget,
    )
    _validate_protocol_cross_bindings(
        preflight, identity, thinking_budget=thinking_budget, k=k
    )
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "relative_prefix": prefix,
        "role": role,
        "run": "smoke",
        "tier_mode": tier_mode,
        "thinking_budget": thinking_budget,
        "arm": arm,
        "k": k,
        "n_records": len(ordered),
        "n_completions": len(ordered) * k,
        "ordered_records": ordered,
        "protocol_binding": preflight["protocol_binding"],
        "identity": identity,
        "files": {
            "preflight": _file_digest(root_path, paths.preflight),
            "rows": _file_digest(root_path, paths.rows),
            "metadata": _file_digest(root_path, paths.metadata),
        },
        "commit_state": "complete",
    }
    _validate_expected(receipt, expected_identity)
    _validate_rows_identity(paths.rows, ordered_records=ordered, k=k, arm=arm)
    _atomic_bytes(paths.receipt, _canonical_bytes(receipt, pretty=True))
    return verify_receipt(root_path, prefix, expected=expected_identity)


def _expected_file_entries(root: Path, paths: BundlePaths) -> dict[str, dict[str, Any]]:
    return {
        "preflight": _file_digest(root, paths.preflight),
        "rows": _file_digest(root, paths.rows),
        "metadata": _file_digest(root, paths.metadata),
    }


def verify_receipt(
    root: str | Path,
    relative_prefix: str | Path,
    *,
    expected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify every receipt field and underlying byte before returning it."""

    root_path = resolve_artifact_root(root)
    prefix, budget, arm, namespace = _validate_prefix(relative_prefix)
    paths = bundle_paths(root_path, prefix)
    expected_k, expected_n_records = _geometry(namespace, budget, arm)
    receipt = _read_json(paths.receipt, where="scientific receipt")
    _require(receipt.get("schema_version") == RECEIPT_SCHEMA_VERSION, "receipt schema mismatch")
    _require(receipt.get("experiment_id") == EXPERIMENT_ID, "receipt experiment mismatch")
    _require(receipt.get("relative_prefix") == prefix, "receipt prefix mismatch")
    expected_role, expected_mode = _role_for_namespace(namespace)
    _require(receipt.get("role") == expected_role, "receipt role/namespace mismatch")
    _require(receipt.get("tier_mode") == expected_mode, "receipt tier mode mismatch")
    _require(receipt.get("run") == "smoke", "receipt run must be smoke")
    _require(receipt.get("thinking_budget") == budget, "receipt budget mismatch")
    _require(receipt.get("arm") == arm, "receipt arm mismatch")
    k = _integer(receipt.get("k"), where="receipt.k", minimum=1)
    _require(k == expected_k, f"receipt K must be {expected_k} for {namespace}")
    _require(receipt.get("commit_state") == "complete", "receipt is not committed")

    preflight = _read_json(paths.preflight, where="runner preflight")
    metadata = _read_json(paths.metadata, where="runner metadata")
    ordered = _ordered_records(preflight)
    _validate_ordered_geometry(
        ordered, arm=arm, expected_n_records=expected_n_records
    )
    _validate_capacity_fit(preflight, budget=budget, k=k)
    _validate_cudagraph_geometry(preflight, budget=budget, k=k)
    _require(receipt.get("ordered_records") == ordered, "receipt ordered prompt identity mismatch")
    _require(
        receipt.get("protocol_binding") == preflight.get("protocol_binding"),
        "receipt/preflight protocol binding mismatch",
    )
    _require(receipt.get("n_records") == len(ordered), "receipt record count mismatch")
    _require(receipt.get("n_completions") == len(ordered) * k, "receipt completion count mismatch")
    identity = _identity_from_metadata(
        metadata, n_records=len(ordered), k=k, thinking_budget=budget
    )
    _validate_protocol_cross_bindings(preflight, identity, thinking_budget=budget, k=k)
    _require(receipt.get("identity") == identity, "receipt runner identity mismatch")
    _require(
        receipt.get("files") == _expected_file_entries(root_path, paths),
        "receipt file size/hash/path mismatch",
    )
    _validate_rows_identity(paths.rows, ordered_records=ordered, k=k, arm=arm)
    _validate_expected(receipt, expected)
    return receipt


def bundle_state(root: str | Path, relative_prefix: str | Path) -> dict[str, Any]:
    """Return absent, preflight-only, or verified-complete; reject all else."""

    root_path = resolve_artifact_root(root)
    prefix, budget, arm, namespace = _validate_prefix(relative_prefix)
    paths = bundle_paths(root_path, prefix)
    expected_k, expected_n_records = _geometry(namespace, budget, arm)
    present = {
        "preflight": _lexists(paths.preflight),
        "rows": _lexists(paths.rows),
        "metadata": _lexists(paths.metadata),
        "receipt": _lexists(paths.receipt),
    }
    if not any(present.values()):
        return {"status": "absent", "relative_prefix": prefix}
    if present == {"preflight": True, "rows": False, "metadata": False, "receipt": False}:
        preflight = _read_json(paths.preflight, where="runner preflight")
        ordered = _ordered_records(preflight)
        _validate_ordered_geometry(
            ordered, arm=arm, expected_n_records=expected_n_records
        )
        _validate_capacity_fit(preflight, budget=budget, k=expected_k)
        _validate_cudagraph_geometry(preflight, budget=budget, k=expected_k)
        _require(
            preflight.get("generation_reserve_tokens")
            == budget + FORCED_CLOSE_TOKENS + ANSWER_MAX_TOKENS,
            "preflight generation reserve differs from its scientific tier",
        )
        role, tier_mode = _role_for_namespace(namespace)
        return {
            "status": "preflight_only",
            "relative_prefix": prefix,
            "role": role,
            "tier_mode": tier_mode,
            "thinking_budget": budget,
            "arm": arm,
            "k": expected_k,
            "n_records": len(ordered),
            "ordered_records": ordered,
            "files": {"preflight": _file_digest(root_path, paths.preflight)},
        }
    if all(present.values()):
        receipt = verify_receipt(root_path, prefix)
        return {
            "status": "complete",
            "relative_prefix": prefix,
            "role": receipt["role"],
            "tier_mode": receipt["tier_mode"],
            "thinking_budget": receipt["thinking_budget"],
            "arm": receipt["arm"],
            "k": receipt["k"],
            "n_records": receipt["n_records"],
            "protocol_binding": receipt["protocol_binding"],
            "receipt": _file_digest(root_path, paths.receipt),
            "files": receipt["files"],
        }
    raise ScientificArtifactError(
        f"partial scientific bundle fails closed at {prefix}: {present}"
    )


def _walk_files(root: Path) -> list[Path]:
    _assert_absolute_no_symlinks(root, where="scientific artifact root")
    if not _lexists(root):
        return []
    _require(root.is_dir(), f"scientific artifact root is not a directory: {root}")
    files: list[Path] = []
    for directory, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        directory_path = Path(directory)
        _assert_absolute_no_symlinks(directory_path, where="scientific artifact directory")
        if directory_path != root:
            relative_directory = directory_path.relative_to(root)
            parts = relative_directory.parts
            _require(
                (len(parts) == 1 and parts[0] in _NAMESPACES)
                or (
                    len(parts) == 2
                    and parts[0] in _NAMESPACES
                    and _THINK_RE.fullmatch(parts[1]) is not None
                ),
                f"unexpected directory in scientific artifact root: "
                f"{relative_directory.as_posix()}",
            )
            _require(
                bool(dirnames or filenames),
                f"empty directory is not a scientific artifact state: "
                f"{relative_directory.as_posix()}",
            )
        for dirname in dirnames:
            child = directory_path / dirname
            _require(not child.is_symlink(), f"symlink directory forbidden in artifact root: {child}")
            relative_child = child.relative_to(root)
            parts = relative_child.parts
            _require(
                (len(parts) == 1 and parts[0] in _NAMESPACES)
                or (
                    len(parts) == 2
                    and parts[0] in _NAMESPACES
                    and _THINK_RE.fullmatch(parts[1]) is not None
                ),
                f"unexpected directory in scientific artifact root: "
                f"{relative_child.as_posix()}",
            )
        for filename in filenames:
            child = directory_path / filename
            _require(not child.is_symlink(), f"symlink file forbidden in artifact root: {child}")
            _require(child.is_file(), f"non-regular artifact entry: {child}")
            files.append(child)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def discover_bundle_prefixes(root: str | Path) -> list[str]:
    """List every permitted flat bundle prefix, including uncommitted triplets."""

    root_path = resolve_artifact_root(root)
    return sorted({_prefix_from_file(root_path, path) for path in _walk_files(root_path)})


def _prefix_from_file(root: Path, path: Path) -> str:
    relative = path.relative_to(root).as_posix()
    for suffix in _FILE_SUFFIXES:
        if relative.endswith(suffix):
            prefix = relative[: -len(suffix)]
            _validate_prefix(prefix)
            return prefix
    raise ScientificArtifactError(f"unexpected file in scientific artifact root: {relative}")


def _entry_id(namespace: str, budget: int, arm: str) -> str:
    kind = "matrix" if namespace == "smoke_tiers" else "probe"
    return f"{kind}/think_{budget}/{arm}"


def _tree_digest(root: Path, files: Sequence[Path]) -> dict[str, Any]:
    records: list[bytes] = []
    total = 0
    for path in files:
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        digest = _sha256_file(path)
        total += size
        records.append(relative.encode("utf-8") + b"\0" + digest.encode("ascii") + b"\n")
    return {
        "files": len(files),
        "bytes": total,
        "sha256": _sha256_bytes(b"".join(records)),
    }


def _entry_artifact_hashes(entry: Mapping[str, Any], *, where: str) -> dict[str, str]:
    files = entry.get("files")
    receipt = entry.get("receipt")
    _require(isinstance(files, Mapping), f"{where} lacks file identities")
    _require(isinstance(receipt, Mapping), f"{where} lacks receipt identity")
    result: dict[str, str] = {}
    for selection_name, entry_name in (
        ("rows", "rows"),
        ("meta", "metadata"),
        ("preflight", "preflight"),
    ):
        identity = files.get(entry_name)
        _require(isinstance(identity, Mapping), f"{where} lacks {entry_name} identity")
        result[selection_name] = _sha256(
            identity.get("sha256"), where=f"{where}.{entry_name}.sha256"
        )
    result["receipt"] = _sha256(
        receipt.get("sha256"), where=f"{where}.receipt.sha256"
    )
    return result


def _selection_artifacts_match_entry(
    artifacts: Any,
    entry: Mapping[str, Any],
    *,
    where: str,
) -> None:
    _require(isinstance(artifacts, Mapping), f"{where} lacks artifact hashes")
    normalized = {
        str(key): _sha256(value, where=f"{where}.artifacts.{key}")
        for key, value in artifacts.items()
    }
    _require(
        normalized == _entry_artifact_hashes(entry, where=where),
        f"{where} artifact hashes differ from the scientific catalog",
    )


def validate_selection(
    selection_file: str | Path,
    catalog: Mapping[str, Any],
    *,
    budget_ladder: Sequence[int],
    arms: Sequence[str],
) -> dict[str, Any]:
    """Validate a hashed smoke selection as a coherent first-adequate history.

    Callers pass their config values deliberately; this helper proves those
    values still equal the independently frozen storage geometry.  It then
    binds every completed arm/probe audit back to the corresponding catalog
    receipt instead of treating a selection-file hash as semantic validation.
    """

    ladder = tuple(
        _integer(value, where="selection budget ladder entry", minimum=1)
        for value in budget_ladder
    )
    arm_order = tuple(str(arm) for arm in arms)
    _require(ladder == SCIENTIFIC_BUDGETS, "selection budget ladder drifted")
    _require(arm_order == SCIENTIFIC_MATRIX_ARMS, "selection arm order drifted")

    selection_path = Path(selection_file)
    _assert_absolute_no_symlinks(selection_path, where="smoke budget selection")
    selection = _read_json(selection_path, where="smoke budget selection")
    _require(selection.get("schema_version") == 1, "smoke selection schema mismatch")
    _require(selection.get("run") == "smoke", "smoke selection run mismatch")
    _require(
        selection.get("selection_uses_output_content") is False,
        "smoke selection must be content-blind",
    )
    _require(
        selection.get("lower_tiers_excluded_from_scoring") is True,
        "smoke selection must exclude lower tiers",
    )
    _require(
        selection.get("scientific_probe_k") == SCIENTIFIC_PROBE_K,
        "smoke selection probe K mismatch",
    )
    _require(
        selection.get("probes_excluded_from_promotion_scoring_and_prefix_pooling") is True,
        "smoke selection did not exclude probes",
    )

    selected_pointer = catalog.get("selected")
    passed = selection.get("pass")
    _require(isinstance(passed, bool), "smoke selection pass flag must be boolean")
    selected_budget = selection.get("selected_thinking_budget")
    if passed:
        selected_budget = _integer(
            selected_budget, where="smoke selected thinking budget", minimum=1
        )
        _require(selected_budget in ladder, "smoke selected budget is unregistered")
        _require(
            isinstance(selected_pointer, Mapping),
            "passing smoke selection lacks a catalog pointer",
        )
        _require(
            selected_pointer.get("thinking_budget") == selected_budget,
            "selection/catalog selected budget mismatch",
        )
        _require(
            selected_pointer.get("selection_path") == SELECTION_LOGICAL_PATH,
            "selection/catalog logical path mismatch",
        )
        _require(
            selected_pointer.get("selection_bytes") == selection_path.stat().st_size
            and selected_pointer.get("selection_sha256") == _sha256_file(selection_path),
            "selection/catalog file identity mismatch",
        )
    else:
        _require(selected_budget is None, "failed smoke selection has a selected budget")
        _require(selected_pointer is None, "failed smoke selection has a catalog pointer")

    entries = catalog.get("entries")
    _require(isinstance(entries, list), "scientific catalog entries are missing")
    entry_by_id: dict[str, Mapping[str, Any]] = {}
    for raw_entry in entries:
        _require(isinstance(raw_entry, Mapping), "scientific catalog entry is invalid")
        entry_id = _string(raw_entry.get("id"), where="scientific catalog entry.id")
        _require(entry_id not in entry_by_id, f"duplicate scientific catalog entry: {entry_id}")
        entry_by_id[entry_id] = raw_entry

    raw_tiers = selection.get("tiers")
    _require(isinstance(raw_tiers, list) and bool(raw_tiers), "smoke selection has no tiers")
    tier_budgets = [
        _integer(
            tier.get("budget") if isinstance(tier, Mapping) else None,
            where=f"smoke selection tier {index}.budget",
            minimum=1,
        )
        for index, tier in enumerate(raw_tiers)
    ]
    _require(
        tuple(tier_budgets) == ladder[: len(tier_budgets)],
        "smoke selection tiers are not an exact contiguous ladder prefix",
    )
    _require(len(raw_tiers) <= len(ladder), "smoke selection has too many tiers")

    adequate_indices: list[int] = []
    for tier_index, raw_tier in enumerate(raw_tiers):
        _require(isinstance(raw_tier, Mapping), f"smoke tier {tier_index} is invalid")
        tier = raw_tier
        budget = tier_budgets[tier_index]
        complete = tier.get("complete")
        adequate = tier.get("adequate")
        _require(isinstance(complete, bool), f"smoke tier {budget} complete flag is invalid")
        _require(isinstance(adequate, bool), f"smoke tier {budget} adequate flag is invalid")
        if adequate:
            adequate_indices.append(tier_index)
            _require(complete is True, f"adequate smoke tier {budget} is incomplete")
            _require(tier.get("status") == "selectable", f"adequate smoke tier {budget} is not selectable")
            _require(
                tier.get("tier_mode") == "complete_k12_matrix",
                f"adequate smoke tier {budget} has the wrong mode",
            )

        raw_arm_states = tier.get("arms")
        _require(isinstance(raw_arm_states, Mapping), f"smoke tier {budget} lacks arms")
        _require(
            tuple(raw_arm_states.keys()) == arm_order,
            f"smoke tier {budget} arm order/set mismatch",
        )
        arm_statuses: list[str] = []
        completed_arm_adequacy: list[bool] = []
        for arm in arm_order:
            raw_arm_state = raw_arm_states.get(arm)
            _require(
                isinstance(raw_arm_state, Mapping),
                f"smoke tier {budget}/{arm} state is invalid",
            )
            status = raw_arm_state.get("status")
            _require(
                status in {"complete", "skipped"},
                f"smoke tier {budget}/{arm} status is invalid",
            )
            arm_statuses.append(str(status))
            if status == "complete":
                entry_id = f"matrix/think_{budget}/{arm}"
                entry = entry_by_id.get(entry_id)
                _require(entry is not None, f"selection references missing {entry_id}")
                _require(
                    entry.get("status") == "complete"
                    and entry.get("role") == "complete_matrix_arm"
                    and entry.get("k") == SCIENTIFIC_MATRIX_K
                    and entry.get("n_records") == SCIENTIFIC_N_RECORDS,
                    f"selection references invalid matrix entry {entry_id}",
                )
                termination = raw_arm_state.get("termination")
                _require(
                    isinstance(termination, Mapping)
                    and isinstance(termination.get("adequate"), bool),
                    f"smoke tier {budget}/{arm} termination audit is invalid",
                )
                completed_arm_adequacy.append(bool(termination.get("adequate")))
                _selection_artifacts_match_entry(
                    raw_arm_state.get("artifacts"),
                    entry,
                    where=f"smoke tier {budget}/{arm}",
                )
                if adequate:
                    _require(
                        termination.get("adequate") is True,
                        f"selected smoke tier {budget}/{arm} is termination-inadequate",
                    )
            elif adequate:
                raise ScientificArtifactError(
                    f"adequate smoke tier {budget} skipped required arm {arm}"
                )

        computed_complete = all(status == "complete" for status in arm_statuses)
        computed_adequate = computed_complete and all(completed_arm_adequacy)
        _require(
            complete is computed_complete,
            f"smoke tier {budget} complete flag disagrees with its arms",
        )
        _require(
            adequate is computed_adequate,
            f"smoke tier {budget} adequate flag disagrees with its arms",
        )
        if not adequate:
            _require(
                tier.get("status") in {"rejected", "probe_only_rejected"},
                f"inadequate smoke tier {budget} has an invalid status",
            )

        probe = tier.get("scientific_probe")
        if budget <= 32768:
            _require(probe is None, f"smoke tier {budget} must not use a probe")
        else:
            _require(probe is not None, f"higher smoke tier {budget} lacks its base probe")
        if probe is not None:
            _require(isinstance(probe, Mapping), f"smoke tier {budget} probe is invalid")
            _require(
                probe.get("status") == "complete"
                and probe.get("role") == "termination_only_budget_probe"
                and probe.get("budget") == budget
                and probe.get("arm") == "base"
                and probe.get("k") == SCIENTIFIC_PROBE_K
                and probe.get("records") == SCIENTIFIC_N_RECORDS,
                f"smoke tier {budget} probe geometry mismatch",
            )
            termination = probe.get("termination")
            _require(
                isinstance(termination, Mapping)
                and isinstance(termination.get("adequate"), bool),
                f"smoke tier {budget} probe termination audit is invalid",
            )
            if tier.get("status") == "probe_only_rejected":
                _require(
                    termination.get("adequate") is False,
                    f"probe-rejected tier {budget} has an adequate probe",
                )
                _require(
                    tier.get("tier_mode") == "termination_probe_only",
                    f"probe-rejected tier {budget} has the wrong mode",
                )
            else:
                _require(
                    termination.get("adequate") is True,
                    f"matrix tier {budget} followed an inadequate probe",
                )
            probe_entry_id = f"probe/think_{budget}/base"
            probe_entry = entry_by_id.get(probe_entry_id)
            _require(probe_entry is not None, f"selection references missing {probe_entry_id}")
            _require(
                probe_entry.get("status") == "complete"
                and probe_entry.get("role") == "termination_probe"
                and probe_entry.get("k") == SCIENTIFIC_PROBE_K
                and probe_entry.get("n_records") == SCIENTIFIC_N_RECORDS,
                f"selection references invalid probe entry {probe_entry_id}",
            )
            _selection_artifacts_match_entry(
                probe.get("artifacts"),
                probe_entry,
                where=f"smoke tier {budget} probe",
            )
        if tier.get("status") != "probe_only_rejected":
            _require(
                tier.get("tier_mode") == "complete_k12_matrix",
                f"matrix tier {budget} has the wrong mode",
            )

    if passed:
        _require(
            adequate_indices == [len(raw_tiers) - 1],
            "passing smoke selection did not choose the first/only adequate tier",
        )
        _require(
            tier_budgets[-1] == selected_budget,
            "selected budget is not the final adequate tier",
        )
        selected_arms = selected_pointer.get("arms")  # type: ignore[union-attr]
        _require(isinstance(selected_arms, Mapping), "catalog selected arms are missing")
        _require(tuple(selected_arms.keys()) == arm_order, "catalog selected arm order/set mismatch")
        _require(
            selected_arms
            == {arm: f"matrix/think_{selected_budget}/{arm}" for arm in arm_order},
            "catalog selected arm pointers are not the exact K12 matrix",
        )
    else:
        _require(not adequate_indices, "failed smoke selection contains an adequate tier")
    return selection


def build_catalog(
    root: str | Path,
    *,
    protocol_binding: Mapping[str, Any],
    selection_file: str | Path | None = None,
    selected_budget: int | None = None,
    selected_entries: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build a deterministic per-file catalog and optional logical selection.

    ``selection_file`` is hashed by the catalog.  The selection file never
    hashes the catalog, so the dependency has one direction and no cycle.
    """

    root_path = resolve_artifact_root(root)
    frozen_protocol = _validate_protocol_binding(protocol_binding)
    files = _walk_files(root_path)
    prefixes = sorted({_prefix_from_file(root_path, path) for path in files})
    entries: list[dict[str, Any]] = []
    for prefix in prefixes:
        state = bundle_state(root_path, prefix)
        _, budget, arm, namespace = _validate_prefix(prefix)
        entry = {
            "id": _entry_id(namespace, budget, arm),
            **state,
        }
        if state.get("status") == "complete":
            _require(
                state.get("protocol_binding") == frozen_protocol,
                f"completed artifact {prefix} belongs to a different frozen protocol",
            )
        entries.append(entry)
    entries.sort(key=lambda entry: str(entry["id"]))

    provided = (selection_file is not None, selected_budget is not None, selected_entries is not None)
    _require(
        len(set(provided)) == 1,
        "selection_file, selected_budget, and selected_entries must be provided together",
    )
    selected: dict[str, Any] | None = None
    selection_path: Path | None = None
    if all(provided):
        selection_path = Path(selection_file)  # type: ignore[arg-type]
        _assert_absolute_no_symlinks(selection_path, where="smoke budget selection")
        _require(selection_path.is_file(), f"missing smoke budget selection: {selection_path}")
        budget_value = _integer(selected_budget, where="selected budget", minimum=1)
        _require(budget_value in SCIENTIFIC_BUDGETS, "selected budget is unregistered")
        arm_entries = {
            str(arm): str(entry_id)
            for arm, entry_id in sorted(dict(selected_entries or {}).items())
        }
        expected_arm_entries = {
            arm: f"matrix/think_{budget_value}/{arm}"
            for arm in SCIENTIFIC_MATRIX_ARMS
        }
        _require(
            arm_entries == expected_arm_entries,
            "selected entries must be the exact base/designed K12 matrix",
        )
        entry_by_id = {str(entry["id"]): entry for entry in entries}
        for arm, entry_id in arm_entries.items():
            _require(_ARM_RE.fullmatch(arm) is not None, f"invalid selected arm: {arm!r}")
            _require(entry_id in entry_by_id, f"selected catalog entry is missing: {entry_id}")
            entry = entry_by_id[entry_id]
            _require(entry["status"] == "complete", f"selected entry is incomplete: {entry_id}")
            _require(
                entry["role"] == "complete_matrix_arm",
                f"termination probe cannot be selected: {entry_id}",
            )
            _require(entry["thinking_budget"] == budget_value, "selected entry budget mismatch")
            _require(entry["arm"] == arm, "selected entry arm mismatch")
            _require(
                entry["k"] == SCIENTIFIC_MATRIX_K
                and entry["n_records"] == SCIENTIFIC_N_RECORDS,
                "selected entry scientific geometry mismatch",
            )
        selected = {
            "thinking_budget": budget_value,
            "selection_path": SELECTION_LOGICAL_PATH,
            "selection_bytes": selection_path.stat().st_size,
            "selection_sha256": _sha256_file(selection_path),
            "arms": arm_entries,
        }

    catalog = {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "storage": {
            "default_root": str(DEFAULT_ARTIFACT_ROOT),
            "override_environment_variable": ARTIFACT_ROOT_ENV,
        },
        "checksum_scheme": (
            "sha256 over sorted relative-path NUL file-sha256 newline records"
        ),
        "protocol_binding": frozen_protocol,
        "entries": entries,
        "selected": selected,
        "tree": _tree_digest(root_path, files),
    }
    if selection_path is not None:
        validate_selection(
            selection_path,
            catalog,
            budget_ladder=SCIENTIFIC_BUDGETS,
            arms=SCIENTIFIC_MATRIX_ARMS,
        )
    return catalog


def write_catalog(path: str | Path, catalog: Mapping[str, Any]) -> None:
    """Atomically write a deterministic tracked catalog."""

    target = Path(path)
    _assert_absolute_no_symlinks(target, where="scientific artifact catalog")
    payload = _canonical_bytes(dict(catalog), pretty=True)
    _atomic_bytes(target, payload)


def verify_catalog(
    path: str | Path,
    root: str | Path,
    *,
    protocol_binding: Mapping[str, Any],
    selection_file: str | Path | None = None,
) -> dict[str, Any]:
    """Rebuild the catalog from external bytes and require exact equality."""

    catalog_path = Path(path)
    stored = _read_json(catalog_path, where="scientific artifact catalog")
    _require(stored.get("schema_version") == CATALOG_SCHEMA_VERSION, "catalog schema mismatch")
    _require(stored.get("experiment_id") == EXPERIMENT_ID, "catalog experiment mismatch")
    selected = stored.get("selected")
    if selected is None:
        _require(selection_file is None, "unselected catalog received a selection file")
        rebuilt = build_catalog(root, protocol_binding=protocol_binding)
    else:
        _require(isinstance(selected, Mapping), "catalog selected pointer must be an object")
        _require(selection_file is not None, "selected catalog requires its selection file")
        _require(
            selected.get("selection_path") == SELECTION_LOGICAL_PATH,
            "catalog selection path mismatch",
        )
        arms = selected.get("arms")
        _require(isinstance(arms, Mapping), "catalog selected arms must be an object")
        rebuilt = build_catalog(
            root,
            protocol_binding=protocol_binding,
            selection_file=selection_file,
            selected_budget=_integer(
                selected.get("thinking_budget"), where="catalog selected budget", minimum=1
            ),
            selected_entries={str(key): str(value) for key, value in arms.items()},
        )
    _require(stored == rebuilt, "scientific artifact catalog differs from external storage")
    return stored


def selected_bundle_prefixes(
    catalog: Mapping[str, Any], expected_arms: Sequence[str]
) -> tuple[int, dict[str, str]]:
    """Resolve a verified catalog's logical selected arms without copying bytes."""

    selected = catalog.get("selected")
    _require(isinstance(selected, Mapping), "scientific catalog has no selected smoke tier")
    budget = _integer(
        selected.get("thinking_budget"), where="selected smoke budget", minimum=1
    )
    _require(budget in SCIENTIFIC_BUDGETS, "selected smoke budget is unregistered")
    raw_arms = selected.get("arms")
    _require(isinstance(raw_arms, Mapping), "scientific catalog selected arms are missing")
    arm_order = [str(arm) for arm in expected_arms]
    _require(
        tuple(arm_order) == SCIENTIFIC_MATRIX_ARMS,
        "selected smoke expected arms drifted from the fixed matrix",
    )
    _require(
        tuple(raw_arms.keys()) == SCIENTIFIC_MATRIX_ARMS,
        "selected smoke arm order/set mismatch",
    )
    entries = catalog.get("entries")
    _require(isinstance(entries, list), "scientific catalog entries are missing")
    by_id = {
        str(entry.get("id")): entry
        for entry in entries
        if isinstance(entry, Mapping)
    }
    prefixes: dict[str, str] = {}
    for arm in arm_order:
        entry_id = raw_arms.get(arm)
        _require(isinstance(entry_id, str) and entry_id in by_id, f"missing selected entry for {arm}")
        entry = by_id[entry_id]
        _require(entry.get("status") == "complete", f"selected {arm} bundle is incomplete")
        _require(entry.get("role") == "complete_matrix_arm", f"selected {arm} is a probe")
        _require(entry.get("thinking_budget") == budget, f"selected {arm} budget mismatch")
        _require(entry.get("arm") == arm, f"selected {arm} identity mismatch")
        _require(
            entry.get("k") == SCIENTIFIC_MATRIX_K
            and entry.get("n_records") == SCIENTIFIC_N_RECORDS,
            f"selected {arm} scientific geometry mismatch",
        )
        prefix = entry.get("relative_prefix")
        _require(isinstance(prefix, str), f"selected {arm} prefix is missing")
        _validate_prefix(prefix)
        prefixes[arm] = prefix
    return budget, prefixes


__all__ = [
    "ARTIFACT_ROOT_ENV",
    "ANSWER_MAX_TOKENS",
    "BundlePaths",
    "CATALOG_LOGICAL_PATH",
    "CUDAGRAPH_CAPTURE_SIZES_BY_BUDGET",
    "DEFAULT_ARTIFACT_ROOT",
    "EXPERIMENT_ID",
    "FORCED_CLOSE_TOKENS",
    "LOCK_PATH",
    "MAX_NUM_SEQS_BY_BUDGET",
    "MODEL_ID",
    "MODEL_REVISION",
    "RUNNER_SHA256",
    "RUNNER_SCHEMA_VERSION",
    "SCIENTIFIC_BUDGETS",
    "SCIENTIFIC_MATRIX_ARMS",
    "SCIENTIFIC_MATRIX_K",
    "SCIENTIFIC_N_RECORDS",
    "SCIENTIFIC_PROBE_K",
    "SCIENTIFIC_RUN_SEED",
    "SELECTION_LOGICAL_PATH",
    "ScientificArtifactError",
    "build_catalog",
    "build_protocol_binding",
    "bundle_paths",
    "bundle_state",
    "commit_receipt",
    "comparable_protocol_identity",
    "discover_bundle_prefixes",
    "expected_max_num_seqs",
    "expected_cudagraph_capture_sizes",
    "fsync_tree_and_parent",
    "resolve_artifact_root",
    "safe_path",
    "selected_bundle_prefixes",
    "validate_selection",
    "verify_catalog",
    "verify_receipt",
    "write_catalog",
    "write_preflight_only",
]
