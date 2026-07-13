#!/usr/bin/env python3
"""Content-address and preserve an incomplete canonical attempt before retry."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    MODEL_ID,
    MODEL_REVISION,
    config_sha256,
    load_config,
    source_contract_sha256,
)
from src.design_boundary import design_lineage, validate_design_receipt  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(REPO_ROOT):
        raise RuntimeError(f"failed attempt path escapes repository: {resolved}")
    return resolved.relative_to(REPO_ROOT).as_posix()


def _manifest(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        raise RuntimeError(f"failed attempt directory is missing: {path}")
    files = []
    for item in sorted(path.rglob("*")):
        if item.is_symlink():
            raise RuntimeError(f"failed attempt contains a prohibited symlink: {item}")
        if item.is_file():
            files.append(
                {
                    "path": item.relative_to(path).as_posix(),
                    "bytes": item.stat().st_size,
                    "sha256": _sha256(item),
                }
            )
    payload = {
        "source_path": _repo_relative(path),
        "files": files,
        "files_sha256": _canonical_sha256(files),
    }
    payload["tree_identity_sha256"] = _canonical_sha256(payload)
    return payload


def _allowed_paths(config: dict[str, Any]) -> set[Path]:
    large = (ROOT / config["paths"]["large_artifacts_dir"]).resolve()
    paths = set()
    for capacity in ("lora", "fullrank"):
        for objective in ("joint", "state_only"):
            for seed in map(int, config["training"]["train_seeds"]):
                cell = f"{capacity}_{objective}_seed{seed}"
                paths.add(large / cell)
                paths.add((ROOT / "runs" / "training" / cell).resolve())
                paths.add((ROOT / "runs" / f"{cell}_trigger").resolve())
                if objective == "joint":
                    paths.add((ROOT / "runs" / f"{cell}_contrast").resolve())
    return {path.resolve() for path in paths}


def archive_failed_attempt(
    config: dict[str, Any], paths: list[Path]
) -> dict[str, Any]:
    validate_design_receipt(config)
    if not 1 <= len(paths) <= 2:
        raise RuntimeError("archive exactly one canonical attempt and at most one companion")
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise RuntimeError("failed attempt paths must be distinct")
    allowed = _allowed_paths(config)
    if any(path not in allowed for path in resolved):
        raise RuntimeError("refusing to archive a noncanonical or unregistered path")
    evaluation_paths = [
        path for path in resolved
        if path.parent == (ROOT / "runs").resolve()
        and path.name.endswith(("_trigger", "_contrast"))
    ]
    if any((path / "summary.json").is_file() for path in evaluation_paths):
        raise RuntimeError("refusing to archive a completed evaluation as a failed attempt")
    if len(resolved) == 2:
        large = (ROOT / config["paths"]["large_artifacts_dir"]).resolve()
        tracked = (ROOT / "runs" / "training").resolve()
        large_cells = [path for path in resolved if path.parent == large]
        tracked_cells = [path for path in resolved if path.parent == tracked]
        if (
            len(large_cells) != 1
            or len(tracked_cells) != 1
            or large_cells[0].name != tracked_cells[0].name
        ):
            raise RuntimeError(
                "two-path archive must contain the exact same-cell tracked companion"
            )
    manifests = [_manifest(path) for path in resolved]
    attempt_identity = _canonical_sha256({"attempts": manifests})
    label = resolved[0].name
    archive_root = (
        ROOT
        / config["paths"]["large_artifacts_dir"]
        / "failed_attempts"
        / f"{label}-{attempt_identity[:16]}"
    ).resolve()
    tracked_receipt = (
        ROOT / "runs" / "failures" / f"{label}-{attempt_identity[:16]}.json"
    ).resolve()
    if archive_root.exists() or tracked_receipt.exists():
        raise RuntimeError("refusing to overwrite an archived failed attempt")
    receipt = {
        "schema_version": 1,
        "status": "FAILED_ATTEMPT_ARCHIVED",
        "experiment_id": config["experiment_id"],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "backend": "transformers",
        "config_sha256": config_sha256(config),
        "source_contract_sha256": source_contract_sha256(ROOT),
        "requirements_training_lock_sha256": _sha256(
            REPO_ROOT / "requirements-training.lock.txt"
        ),
        "design_lineage": design_lineage(config),
        "attempt_identity_sha256": attempt_identity,
        "archive_path": _repo_relative(archive_root),
        "attempts": manifests,
        "scientific_evidence": False,
    }
    receipt["receipt_identity_sha256"] = _canonical_sha256(receipt)
    archive_root.mkdir(parents=True, exist_ok=False)
    try:
        for index, source in enumerate(resolved, start=1):
            destination = archive_root / f"source_{index}_{source.name}"
            shutil.move(str(source), str(destination))
        archive_receipt = archive_root / "archive_receipt.json"
        encoded = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
        archive_receipt.write_bytes(encoded)
        tracked_receipt.parent.mkdir(parents=True, exist_ok=True)
        temporary = tracked_receipt.with_name(
            f".{tracked_receipt.name}.tmp-{os.getpid()}"
        )
        temporary.write_bytes(encoded)
        os.replace(temporary, tracked_receipt)
    except Exception:
        # Never delete or roll back preserved bytes. A partial archive is itself
        # evidence and must be inspected manually.
        raise
    return receipt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--path", action="append", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    receipt = archive_failed_attempt(config, [Path(item) for item in args.path])
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
