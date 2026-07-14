"""Immutable detached-worktree and runtime provenance contracts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def _run(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)


def require_detached_execution_worktree(repo_root: Path) -> dict[str, str]:
    """Require commands to run from one clean, detached, exact-SHA worktree."""
    repo_root = repo_root.resolve()
    if Path.cwd().resolve() != repo_root:
        raise ValueError("execution must be invoked with the detached worktree root as cwd")
    top = _run(["git", "rev-parse", "--show-toplevel"], cwd=repo_root)
    head = _run(["git", "rev-parse", "HEAD"], cwd=repo_root)
    status = _run(["git", "status", "--porcelain"], cwd=repo_root)
    branch = _run(["git", "symbolic-ref", "-q", "HEAD"], cwd=repo_root)
    if (
        top.returncode != 0
        or Path(top.stdout.strip()).resolve() != repo_root
        or head.returncode != 0
        or len(head.stdout.strip()) != 40
        or status.returncode != 0
        or status.stdout
        or branch.returncode == 0
    ):
        raise ValueError("execution requires a clean detached exact-SHA Git worktree")
    return {
        "repo_root": str(repo_root),
        "git_commit": head.stdout.strip(),
        "head_mode": "detached",
        "cwd": str(Path.cwd().resolve()),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def installed_packages() -> dict[str, str]:
    values: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            normalized = name.lower().replace("_", "-")
            values[normalized] = importlib.metadata.version(name)
    return dict(sorted(values.items()))


def runtime_metadata(repo_root: Path, lock_path: Path) -> dict[str, Any]:
    """Record the complete installed/runtime/hardware identity for training."""
    worktree = require_detached_execution_worktree(repo_root)
    gpu = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        cwd=repo_root,
    )
    nvcc = _run(["nvcc", "--version"], cwd=repo_root)
    packages = installed_packages()
    value = {
        "schema_version": 1,
        "worktree": worktree,
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "packages_sha256": hashlib.sha256(
            json.dumps(packages, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "environment_lock": {
            "path": str(lock_path.resolve()),
            "sha256": _sha256_file(lock_path),
        },
        "gpu": gpu.stdout.strip() if gpu.returncode == 0 else "",
        "cuda_toolkit": nvcc.stdout.strip() if nvcc.returncode == 0 else "",
    }
    if not value["gpu"]:
        raise ValueError("training runtime lacks an exact visible GPU identity")
    return value
