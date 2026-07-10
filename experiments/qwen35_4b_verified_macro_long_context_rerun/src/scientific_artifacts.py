"""Fail-closed external storage for scientific-smoke runner artifacts.

The scientific smoke can produce individual JSONL files larger than GitHub's
hard file limit.  This module keeps those files in one explicit external root
without weakening the experiment's cache identity.  Runner-native artifacts
retain their existing flat names::

    smoke_tiers/think_32768/base.preflight.json
    smoke_tiers/think_32768/base.jsonl
    smoke_tiers/think_32768/base.meta.json
    smoke_tiers/think_32768/base.receipt.json

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


EXPERIMENT_ID = "qwen35_4b_verified_macro_long_context_rerun"
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
RUNNER_SHA256 = "fd9972bdcb3a9e8b9841b45ed8e2849017a6e80b601e924817cdaaa5144b8782"
FORCED_CLOSE_TOKENS = 2
ANSWER_MAX_TOKENS = 512
DEFAULT_ARTIFACT_ROOT = Path(
    "/workspace/large_artifacts/"
    "qwen35_4b_verified_macro_long_context_rerun/scientific_smoke_v1"
)
ARTIFACT_ROOT_ENV = "QWEN35_MACRO_SCIENTIFIC_ARTIFACT_ROOT"
CATALOG_LOGICAL_PATH = "analysis/scientific_smoke_artifact_catalog.json"
SELECTION_LOGICAL_PATH = "analysis/smoke_budget_selection.json"
RECEIPT_SCHEMA_VERSION = 1
CATALOG_SCHEMA_VERSION = 2
PROTOCOL_BINDING_SCHEMA_VERSION = 1

_PROTOCOL_FILES = (
    "configs/default.yaml",
    "data/tasks.json",
    "data/demonstrations.json",
)
_PROTOCOL_SOURCES = (
    "scripts/analyze.py",
    "scripts/run.py",
    "src/macro_domain.py",
    "src/model_harness.py",
    "src/scientific_artifacts.py",
    "src/vllm_runner.py",
)
_SMOKE_LIBRARY_ARMS = ("base", "designed_ceiling")

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
    return normalized, int(match.group(1)), arm, namespace


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
    _require(preflight.get("pass") is True, "preflight did not pass")
    max_model_len = _integer(
        preflight.get("max_model_len"), where="preflight.max_model_len", minimum=1
    )
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


def write_preflight_only(
    root: str | Path,
    relative_prefix: str | Path,
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze a preflight as the one valid incomplete/resumable bundle state."""

    root_path = resolve_artifact_root(root)
    prefix, budget, _, _ = _validate_prefix(relative_prefix)
    paths = bundle_paths(root_path, prefix)
    _ordered_records(preflight)
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
            _require(
                row.get("prompt_sha256") == expected["rendered_prompt_sha256"],
                f"runner prompt hash mismatch for {record_id}",
            )
            _require(
                row.get("n_prompt_tokens") == expected["prompt_tokens"],
                f"runner prompt-token count mismatch for {record_id}",
            )
            outputs = row.get("outputs")
            _require(
                isinstance(outputs, list) and len(outputs) == k,
                f"runner K mismatch for {record_id}",
            )
            indices = [
                output.get("sample_index") if isinstance(output, Mapping) else None
                for output in outputs
            ]
            _require(indices == list(range(k)), f"runner sample order mismatch for {record_id}")
            meta = row.get("meta")
            if isinstance(meta, Mapping) and "arm" in meta:
                _require(meta.get("arm") == arm, f"runner row arm mismatch for {record_id}")
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
) -> dict[str, Any]:
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
    sampling = metadata.get("sampling")
    engine = metadata.get("engine")
    _require(isinstance(sampling, Mapping), "runner metadata.sampling must be an object")
    _require(isinstance(engine, Mapping), "runner metadata.engine must be an object")
    sampling_json = _json_clone(dict(sampling), where="runner metadata.sampling")
    engine_json = _json_clone(dict(engine), where="runner metadata.engine")
    counts = metadata.get("counts")
    _require(isinstance(counts, Mapping), "runner metadata.counts must be an object")
    _require(counts.get("requests") == n_records, "runner metadata request count mismatch")
    _require(
        counts.get("completions") == n_records * k,
        "runner metadata completion count mismatch",
    )
    return {
        "model": model,
        "model_revision": revision,
        "runner_sha256": runner_sha256,
        "sampling": sampling_json,
        "sampling_sha256": _sha256_value(sampling_json),
        "engine": engine_json,
        "engine_sha256": _sha256_value(engine_json),
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
    _require(isinstance(sampling, Mapping), "receipt sampling identity is missing")
    _require(isinstance(engine, Mapping), "receipt engine identity is missing")
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


def _validate_expected(receipt: Mapping[str, Any], expected: Mapping[str, Any] | None) -> None:
    if expected is None:
        return
    identity = receipt.get("identity")
    _require(isinstance(identity, Mapping), "receipt identity is missing")
    for key, expected_value in expected.items():
        actual = identity.get(key) if key in identity else receipt.get(key)
        normalized = _json_clone(expected_value, where=f"expected receipt {key}")
        _require(actual == normalized, f"receipt {key} differs from the frozen expectation")


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
    expected_role, expected_mode = _role_for_namespace(namespace)
    _require(role == expected_role, f"role {role!r} does not match namespace {namespace}")
    _require(tier_mode == expected_mode, f"tier mode {tier_mode!r} does not match namespace")
    _require(thinking_budget == prefix_budget, "receipt budget differs from its directory")
    _require(arm == prefix_arm, "receipt arm differs from its filename")
    _integer(k, where="receipt K", minimum=1)

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
    identity = _identity_from_metadata(metadata, n_records=len(ordered), k=k)
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
    _require(receipt.get("commit_state") == "complete", "receipt is not committed")

    preflight = _read_json(paths.preflight, where="runner preflight")
    metadata = _read_json(paths.metadata, where="runner metadata")
    ordered = _ordered_records(preflight)
    _require(receipt.get("ordered_records") == ordered, "receipt ordered prompt identity mismatch")
    _require(receipt.get("n_records") == len(ordered), "receipt record count mismatch")
    _require(receipt.get("n_completions") == len(ordered) * k, "receipt completion count mismatch")
    identity = _identity_from_metadata(metadata, n_records=len(ordered), k=k)
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
            "k": None,
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
        entries.append(entry)
    entries.sort(key=lambda entry: str(entry["id"]))

    provided = (selection_file is not None, selected_budget is not None, selected_entries is not None)
    _require(
        len(set(provided)) == 1,
        "selection_file, selected_budget, and selected_entries must be provided together",
    )
    selected: dict[str, Any] | None = None
    if all(provided):
        selection_path = Path(selection_file)  # type: ignore[arg-type]
        _assert_absolute_no_symlinks(selection_path, where="smoke budget selection")
        _require(selection_path.is_file(), f"missing smoke budget selection: {selection_path}")
        budget_value = _integer(selected_budget, where="selected budget", minimum=1)
        arm_entries = {
            str(arm): str(entry_id)
            for arm, entry_id in sorted(dict(selected_entries or {}).items())
        }
        _require(bool(arm_entries), "selected entries must not be empty")
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
        selected = {
            "thinking_budget": budget_value,
            "selection_path": SELECTION_LOGICAL_PATH,
            "selection_bytes": selection_path.stat().st_size,
            "selection_sha256": _sha256_file(selection_path),
            "arms": arm_entries,
        }

    return {
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
    raw_arms = selected.get("arms")
    _require(isinstance(raw_arms, Mapping), "scientific catalog selected arms are missing")
    arm_order = [str(arm) for arm in expected_arms]
    _require(bool(arm_order) and len(set(arm_order)) == len(arm_order), "invalid expected arms")
    _require(set(raw_arms) == set(arm_order), "selected smoke arm set mismatch")
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
    "DEFAULT_ARTIFACT_ROOT",
    "EXPERIMENT_ID",
    "FORCED_CLOSE_TOKENS",
    "MODEL_ID",
    "MODEL_REVISION",
    "RUNNER_SHA256",
    "SELECTION_LOGICAL_PATH",
    "ScientificArtifactError",
    "build_catalog",
    "build_protocol_binding",
    "bundle_paths",
    "bundle_state",
    "commit_receipt",
    "discover_bundle_prefixes",
    "resolve_artifact_root",
    "safe_path",
    "selected_bundle_prefixes",
    "verify_catalog",
    "verify_receipt",
    "write_catalog",
    "write_preflight_only",
]
