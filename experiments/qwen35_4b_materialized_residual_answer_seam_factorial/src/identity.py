"""Canonical fresh identities and narrow authenticated parent reads."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


EXPERIMENT_ID = "qwen35_4b_materialized_residual_answer_seam_factorial"
TASK_NAMESPACE = "materialized-residual-answer-seam-factorial-v1"
REQUEST_NAMESPACE = TASK_NAMESPACE
_NAMESPACE_RE = re.compile(r"\A[a-z0-9]+(?:-[a-z0-9]+)*-v[1-9][0-9]*\Z")

PARENT_PUBLIC_FILES = {
    "mechanics": {
        "path": (
            "experiments/qwen35_4b_materialized_residual_sibling_search_fresh_replication/"
            "data/procedural/mechanics_public.jsonl"
        ),
        "sha256": "8f4ed72908579195918b0dc4ecf0b22f5fa96dfdf771e17d46b5fd795e2e00cb",
    },
    "qualification": {
        "path": (
            "experiments/qwen35_4b_materialized_residual_sibling_search_fresh_replication/"
            "data/procedural/qualification_public.jsonl"
        ),
        "sha256": "68252c389fae3a547142046374472b29e7b8b512485e8d20fea89cbecbbacf88",
    },
    "confirmation": {
        "path": (
            "experiments/qwen35_4b_materialized_residual_sibling_search_fresh_replication/"
            "data/procedural/confirmation_public.jsonl"
        ),
        "sha256": "af54c99dcc3a0039fff26b767204c98283f7c2671a7327adff2ca72a168eaed5",
    },
}
PARENT_PREOUTCOME = {
    "path": (
        "experiments/qwen35_4b_materialized_residual_sibling_search_fresh_replication/"
        "runs/mechanics/prepared/preoutcome_receipt_v2.json"
    ),
    "sha256": "04d8ba59d212adac3193d88c19a38f58298fa18cbdd41321bf9e312bea72fe72",
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


def verified_parent_public_files(root: Path) -> dict[str, Path]:
    return {
        split: verified_regular_file(root, row["path"], row["sha256"])
        for split, row in sorted(PARENT_PUBLIC_FILES.items())
    }


def verified_parent_preoutcome(root: Path) -> Path:
    return verified_regular_file(
        root, PARENT_PREOUTCOME["path"], PARENT_PREOUTCOME["sha256"]
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
    if not task_id.startswith(REQUEST_NAMESPACE + "/"):
        raise ValueError("task ID is outside the successor namespace")
    if family == "suffix" and candidate is None:
        raise ValueError("suffix requests require a candidate")
    if family != "suffix" and candidate is not None:
        raise ValueError("only suffix requests are candidate-bound")
    if family == "direct" and sample_index is None:
        raise ValueError("direct master-pool rows require a sample index")
    if family != "direct" and sample_index is not None:
        raise ValueError("only direct requests may include a sample index")
    return [
        REQUEST_NAMESPACE,
        family,
        task_id,
        *([] if candidate is None else [candidate]),
        *([] if sample_index is None else [f"{sample_index:05d}"]),
    ]


def request_id(seed_key: list[str]) -> str:
    return canonical_sha256(seed_key)
