"""Canonical fresh identities and the one authenticated parent hash manifest."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


EXPERIMENT_ID = "qwen35_4b_tokenizer_eos_residual_mechanics_fresh_replay"
TASK_NAMESPACE = "tokenizer-eos-residual-mechanics-fresh-replay-v1"
REQUEST_NAMESPACE = TASK_NAMESPACE
TRANSPORT_REQUEST_NAMESPACE = "tokenizer-eos-residual-mechanics-fresh-replay-transport-v1"
_NAMESPACE_RE = re.compile(r"\A[a-z0-9]+(?:-[a-z0-9]+)*-v[1-9][0-9]*\Z")

PARENT_COLLISION_MANIFEST = {
    "path": (
        "experiments/qwen35_4b_tokenizer_eos_residual_mechanics_fresh_replay/"
        "runs/parent_lineage/collision_manifest.json"
    ),
    "sha256": "450eb55d41b09aabff33967d5c75e6315e41cf093bd043004045a8ae5d0d07ef",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verified_regular_file(root: Path, relative: str, digest: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise RuntimeError(f"authenticated path escaped the repository: {relative}")
    cursor = root.resolve()
    for part in candidate.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise RuntimeError(f"authenticated path contains a symlink: {relative}")
    if not cursor.is_file() or file_sha256(cursor) != digest:
        raise RuntimeError(f"authenticated file differs: {relative}")
    return cursor


def verified_parent_collision_manifest(root: Path) -> Path:
    return verified_regular_file(
        root,
        PARENT_COLLISION_MANIFEST["path"],
        PARENT_COLLISION_MANIFEST["sha256"],
    )


def namespaced_task_id(namespace: str, split: str, index: int) -> str:
    if not _NAMESPACE_RE.fullmatch(namespace):
        raise ValueError("task namespace has the wrong versioned form")
    if split not in {"calibration", "mechanics"}:
        raise ValueError("unknown task split")
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError("task index must be a non-negative integer")
    return f"{namespace}/{split}/{index:05d}"


def public_instance_payload(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "depth": task["depth"],
        "visible": task["visible"],
        "unlabeled_probe_inputs": task["unlabeled_probe_inputs"],
    }


def public_instance_fingerprint(task: dict[str, Any]) -> str:
    return canonical_sha256(public_instance_payload(task))


def request_seed_key(
    family: str,
    task_id: str,
    *,
    candidate: str | None = None,
    sample_index: int | None = None,
) -> list[str]:
    if family not in {"calibration", "transport", "suffix", "direct"}:
        raise ValueError("unknown request family")
    if not task_id.startswith(TASK_NAMESPACE + "/"):
        raise ValueError("task ID is outside the successor namespace")
    if family == "suffix" and candidate is None:
        raise ValueError("suffix requests require a candidate")
    if family != "suffix" and candidate is not None:
        raise ValueError("only suffix requests are candidate-bound")
    if family == "direct" and sample_index is None:
        raise ValueError("direct master-pool rows require a sample index")
    if family != "direct" and sample_index is not None:
        raise ValueError("only direct requests may include a sample index")
    namespace = (
        TRANSPORT_REQUEST_NAMESPACE if family == "transport" else REQUEST_NAMESPACE
    )
    return [
        namespace,
        family,
        task_id,
        *([] if candidate is None else [candidate]),
        *([] if sample_index is None else [f"{sample_index:05d}"]),
    ]


def request_id(seed_key: list[str]) -> str:
    return canonical_sha256(seed_key)
