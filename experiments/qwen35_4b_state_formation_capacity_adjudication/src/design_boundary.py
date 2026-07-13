"""Immutable pre-run design boundary and validation.

The receipt freezes the files that define scientific interpretation.  Runtime
source, tests, and the environment lock are recorded as provenance but remain
separately content-addressed by every downstream artifact.  That separation
allows a setup-only mechanical repair to invalidate and regenerate data/init/G0
without changing the frozen scientific design or erasing the failed artifact.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping

from .config import (
    SOURCE_CONTRACT_FILES,
    config_sha256,
    require_confirmatory_config,
    source_contract_sha256,
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def _contract_paths() -> list[Path]:
    paths = [ROOT / item for item in DESIGN_FILES]
    resolved = sorted({path.resolve() for path in paths}, key=_relative)
    if not resolved or any(not path.is_file() for path in resolved):
        missing = [_relative(path) for path in resolved if not path.is_file()]
        raise RuntimeError(f"design-boundary files are missing: {missing}")
    return resolved


def _manifest() -> list[dict[str, Any]]:
    return [
        {"path": _relative(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in _contract_paths()
    ]


def _git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
        text=True, capture_output=True,
    )
    return completed.stdout.strip()


def _require_tracked_clean_inputs() -> None:
    paths = sorted(
        {
            REQUIREMENTS_LOCK.resolve(),
            *[path.resolve() for path in _contract_paths()],
            *[(ROOT / item).resolve() for item in SOURCE_CONTRACT_FILES],
        },
        key=_relative,
    )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"prefreeze registered inputs are missing: {[_relative(path) for path in missing]}"
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


def require_implementation_go() -> None:
    if not IMPLEMENTATION_REVIEW.is_file():
        raise RuntimeError("implementation review is missing")
    matches = re.findall(
        r"^\*\*Status:\*\*\s+`(GO|NO_GO)`(?:\s|$)",
        IMPLEMENTATION_REVIEW.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    if matches != ["GO"]:
        raise RuntimeError("implementation review has not authorized execution with exact GO")


def canonical_design_receipt_path(config: Mapping[str, Any]) -> Path:
    return (ROOT / str(config["paths"]["design_receipt"])).resolve()


def freeze_design(config: Mapping[str, Any], output: Path) -> dict[str, Any]:
    require_confirmatory_config(config)
    require_implementation_go()
    _require_tracked_clean_inputs()
    output = output.resolve()
    if output != canonical_design_receipt_path(config):
        raise RuntimeError("design boundary must use the canonical receipt path")
    if output.exists():
        raise RuntimeError(f"refusing to overwrite frozen design receipt: {output}")
    manifest = _manifest()
    payload = {
        "schema_version": 1,
        "status": STATUS,
        "phase": PHASE,
        "experiment_id": config["experiment_id"],
        "config_sha256": config_sha256(config),
        "implementation_provenance_at_freeze": {
            "source_contract_sha256": source_contract_sha256(ROOT),
            "requirements_training_lock_sha256": _sha256(REQUIREMENTS_LOCK),
            "git_head": _git_head(),
        },
        "frozen_files": manifest,
        "frozen_files_sha256": _canonical_sha256({"files": manifest}),
        "benchmark_files_read": 0,
        "scientific_evidence": False,
    }
    payload["receipt_identity_sha256"] = _canonical_sha256(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def validate_design_receipt(
    config: Mapping[str, Any], path: Path | None = None
) -> dict[str, Any]:
    require_implementation_go()
    expected_path = canonical_design_receipt_path(config)
    path = (path or expected_path).resolve()
    if path != expected_path or not path.is_file():
        raise RuntimeError(f"canonical design receipt is missing: {expected_path}")
    receipt = json.loads(path.read_text(encoding="utf-8"))
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
    return receipt


def design_lineage(config: Mapping[str, Any]) -> dict[str, Any]:
    path = canonical_design_receipt_path(config)
    receipt = validate_design_receipt(config, path)
    return {
        "path": _relative(path),
        "sha256": _sha256(path),
        "receipt_identity_sha256": receipt["receipt_identity_sha256"],
        "status": STATUS,
        "phase": PHASE,
    }
