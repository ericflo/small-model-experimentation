"""Immutable pre-run design boundary and validation.

The receipt freezes the files that define scientific interpretation.  Runtime
source, tests, and the environment lock are recorded as provenance but remain
separately content-addressed by every downstream artifact.  That separation
allows a setup-only mechanical repair to invalidate and regenerate data/init/G0
without changing the frozen scientific design or erasing the failed artifact.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping

from .config import (
    SOURCE_CONTRACT_VERSION,
    SOURCE_CONTRACT_FILES,
    config_sha256,
    requirements_training_lock_bytes,
    require_confirmatory_config,
    reviewed_implementation_snapshot,
    source_contract_execution_snapshot,
    source_contract_sha256,
)
from .safe_io import (
    StableArtifactError,
    open_stable_regular,
    publish_new_bytes,
    read_stable_bytes,
)


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
REQUIREMENTS_LOCK = REPO_ROOT / "requirements-training.lock.txt"
DESIGN_FILES = (
    "configs/default.yaml",
    "idea_intake.md",
    "reports/preregistration.md",
    "reports/design_review.md",
    "docs/architecture.md",
    "docs/gpu_runbook.md",
    "docs/research_handoff.md",
)
STATUS = "DESIGN_FROZEN"
PHASE = "design_boundary"
IMPLEMENTATION_REVIEW = ROOT / "reports" / "implementation_review.md"
FROZEN_DESIGN_RECEIPT = ROOT / "reports" / "design_receipt.json"
FROZEN_DESIGN_RECEIPT_SHA256 = (
    "490035fe77519739e18eb3494ad693544c423a62d4222cf4ea0b4e3d5684d45e"
)
FROZEN_DESIGN_RECEIPT_IDENTITY = (
    "d943b909250c2ebd377b8094bb55324a5d5ccf555c7559736526f013d248ac52"
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_GIT_HEAD_RE = re.compile(r"[0-9a-f]{40}\Z")


def _sha256(path: Path) -> str:
    lexical = Path(os.path.abspath(os.fspath(path)))
    trusted_root = REPO_ROOT if lexical.is_relative_to(REPO_ROOT) else lexical.parent
    return hashlib.sha256(read_stable_bytes(trusted_root, lexical)).hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _relative(path: Path) -> str:
    raw = os.fspath(path)
    candidate = Path(raw)
    lexical = Path(os.path.abspath(raw))
    if candidate.is_absolute() and raw != lexical.as_posix():
        raise RuntimeError(f"registered path is not canonical: {raw}")
    try:
        return lexical.relative_to(REPO_ROOT).as_posix()
    except ValueError as exc:
        raise RuntimeError(f"registered path escapes repository: {lexical}") from exc


def _contract_paths() -> list[Path]:
    paths = [ROOT / item for item in DESIGN_FILES]
    if not paths or len({_relative(path) for path in paths}) != len(paths):
        raise RuntimeError("design-boundary path set is empty or duplicated")
    return sorted(paths, key=_relative)


def _manifest() -> list[dict[str, Any]]:
    paths = _contract_paths()
    rows: list[dict[str, Any]] = []
    try:
        with contextlib.ExitStack() as stack:
            handles = {
                path: stack.enter_context(open_stable_regular(REPO_ROOT, path))
                for path in paths
            }
            for path in paths:
                raw = handles[path].read()
                rows.append(
                    {
                        "path": _relative(path),
                        "bytes": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                    }
                )
    except StableArtifactError as exc:
        raise RuntimeError("design-boundary files are missing or aliased") from exc
    return rows


def _git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
        text=True, capture_output=True,
    )
    return completed.stdout.strip()


def _require_tracked_clean_inputs() -> None:
    paths = sorted(
        {
            REQUIREMENTS_LOCK,
            *_contract_paths(),
            *[ROOT / item for item in SOURCE_CONTRACT_FILES],
        },
        key=_relative,
    )
    relative = [_relative(path) for path in paths]
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", *relative],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    if tracked.returncode != 0:
        raise RuntimeError("design freeze requires every registered input to be tracked at HEAD")
    dirty = subprocess.run(
        ["git", "status", "--porcelain=v1", "--", *relative],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError(f"design freeze requires registered inputs clean at HEAD: {dirty}")


@contextlib.contextmanager
def _implementation_go_snapshot() -> Any:
    review = Path(os.path.abspath(os.fspath(IMPLEMENTATION_REVIEW)))
    review_root = REPO_ROOT if review.is_relative_to(REPO_ROOT) else review.parent
    try:
        with open_stable_regular(review_root, review) as review_handle:
            with reviewed_implementation_snapshot() as current_digest:
                raw = review_handle.read()
                try:
                    text = raw.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise RuntimeError("implementation review is not UTF-8") from exc
                versions = re.findall(
                    r"^\*\*Source-contract version:\*\*\s+`([0-9]+)`(?:\s|$)",
                    text,
                    flags=re.MULTILINE,
                )
                digests = re.findall(
                    r"^\*\*Reviewed implementation SHA-256:\*\*\s+`([0-9a-f]{64})`(?:\s|$)",
                    text,
                    flags=re.MULTILINE,
                )
                statuses = re.findall(
                    r"^\*\*Status:\*\*\s+`(GO|NO_GO)`(?:\s|$)",
                    text,
                    flags=re.MULTILINE,
                )
                if (
                    versions != [str(SOURCE_CONTRACT_VERSION)]
                    or digests != [current_digest]
                    or statuses != ["GO"]
                ):
                    raise RuntimeError(
                        "implementation review has not authorized this source with exact GO"
                    )
                yield
    except StableArtifactError as exc:
        raise RuntimeError("implementation review is missing or aliased") from exc


def require_implementation_go() -> None:
    with _implementation_go_snapshot():
        return None


@contextlib.contextmanager
def authorized_source_execution_snapshot() -> Any:
    """Hold reviewed source, tests, review, and lock through one command."""

    with _implementation_go_snapshot():
        with source_contract_execution_snapshot() as source_digest:
            yield source_digest


def canonical_design_receipt_path(config: Mapping[str, Any]) -> Path:
    return Path(os.path.abspath(ROOT / str(config["paths"]["design_receipt"])))


def freeze_design(config: Mapping[str, Any], output: Path) -> dict[str, Any]:
    require_confirmatory_config(config)
    require_implementation_go()
    _require_tracked_clean_inputs()
    head_before = _git_head()
    output = Path(os.path.abspath(os.fspath(output)))
    if output != canonical_design_receipt_path(config):
        raise RuntimeError("design boundary must use the canonical receipt path")
    manifest = _manifest()
    source_digest = source_contract_sha256(ROOT)
    requirements_digest = hashlib.sha256(
        requirements_training_lock_bytes()
    ).hexdigest()
    _require_tracked_clean_inputs()
    head_after = _git_head()
    if head_after != head_before:
        raise RuntimeError("repository HEAD changed during design freeze")
    payload = {
        "schema_version": 1,
        "status": STATUS,
        "phase": PHASE,
        "experiment_id": config["experiment_id"],
        "config_sha256": config_sha256(config),
        "implementation_provenance_at_freeze": {
            "source_contract_sha256": source_digest,
            "requirements_training_lock_sha256": requirements_digest,
            "git_head": head_before,
        },
        "frozen_files": manifest,
        "frozen_files_sha256": _canonical_sha256({"files": manifest}),
        "benchmark_files_read": 0,
        "scientific_evidence": False,
    }
    payload["receipt_identity_sha256"] = _canonical_sha256(payload)
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    publication_root = REPO_ROOT if output.is_relative_to(REPO_ROOT) else output.parent
    try:
        publish_new_bytes(publication_root, output, encoded)
    except StableArtifactError as exc:
        raise RuntimeError(f"refusing to overwrite or alias frozen design receipt: {output}") from exc
    return payload


def _strict_json_object(raw: bytes, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant: {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeError, ValueError) as exc:
        raise RuntimeError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is not a JSON object")
    return value


def _validate_design_payload(
    config: Mapping[str, Any], expected_path: Path, raw: bytes
) -> dict[str, Any]:
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    receipt = _strict_json_object(raw, "design receipt")
    required_fields = {
        "schema_version",
        "status",
        "phase",
        "experiment_id",
        "config_sha256",
        "implementation_provenance_at_freeze",
        "frozen_files",
        "frozen_files_sha256",
        "benchmark_files_read",
        "scientific_evidence",
        "receipt_identity_sha256",
    }
    if set(receipt) != required_fields:
        raise RuntimeError("design receipt fields changed")
    provenance = receipt.get("implementation_provenance_at_freeze")
    if not isinstance(provenance, dict) or set(provenance) != {
        "source_contract_sha256",
        "requirements_training_lock_sha256",
        "git_head",
    }:
        raise RuntimeError("design receipt freeze provenance changed")
    if (
        _SHA256_RE.fullmatch(str(provenance.get("source_contract_sha256"))) is None
        or _SHA256_RE.fullmatch(
            str(provenance.get("requirements_training_lock_sha256"))
        )
        is None
        or _GIT_HEAD_RE.fullmatch(str(provenance.get("git_head"))) is None
    ):
        raise RuntimeError("design receipt freeze provenance is malformed")
    claimed = receipt.get("receipt_identity_sha256")
    identity_payload = {
        key: value for key, value in receipt.items() if key != "receipt_identity_sha256"
    }
    if claimed != _canonical_sha256(identity_payload):
        raise RuntimeError("design receipt identity mismatch")
    expected = {
        "schema_version": 1,
        "status": STATUS,
        "phase": PHASE,
        "experiment_id": config["experiment_id"],
        "config_sha256": config_sha256(config),
        "benchmark_files_read": 0,
        "scientific_evidence": False,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise RuntimeError(f"design receipt {key} mismatch")
    manifest = _manifest()
    if receipt.get("frozen_files") != manifest:
        raise RuntimeError("a design-boundary file changed after freeze")
    if receipt.get("frozen_files_sha256") != _canonical_sha256({"files": manifest}):
        raise RuntimeError("design-boundary manifest digest mismatch")
    if expected_path == FROZEN_DESIGN_RECEIPT and (
        raw_sha256 != FROZEN_DESIGN_RECEIPT_SHA256
        or receipt.get("receipt_identity_sha256") != FROZEN_DESIGN_RECEIPT_IDENTITY
    ):
        raise RuntimeError("canonical frozen design receipt bytes changed")
    return receipt


def _validated_design_snapshot(
    config: Mapping[str, Any], path: Path | None = None
) -> tuple[dict[str, Any], str]:
    expected_path = canonical_design_receipt_path(config)
    supplied = Path(os.path.abspath(os.fspath(path or expected_path)))
    if supplied != expected_path:
        raise RuntimeError(f"canonical design receipt is missing: {expected_path}")
    try:
        with _implementation_go_snapshot():
            with open_stable_regular(
                REPO_ROOT if supplied.is_relative_to(REPO_ROOT) else supplied.parent,
                supplied,
            ) as handle:
                raw = handle.read()
                receipt = _validate_design_payload(config, expected_path, raw)
                return receipt, hashlib.sha256(raw).hexdigest()
    except StableArtifactError as exc:
        raise RuntimeError(
            f"canonical design receipt is missing or aliased: {expected_path}"
        ) from exc


def validate_design_receipt(
    config: Mapping[str, Any], path: Path | None = None
) -> dict[str, Any]:
    return _validated_design_snapshot(config, path)[0]


def design_lineage(config: Mapping[str, Any]) -> dict[str, Any]:
    path = canonical_design_receipt_path(config)
    receipt, raw_sha256 = _validated_design_snapshot(config, path)
    return {
        "path": _relative(path),
        "sha256": raw_sha256,
        "receipt_identity_sha256": receipt["receipt_identity_sha256"],
        "status": STATUS,
        "phase": PHASE,
    }
