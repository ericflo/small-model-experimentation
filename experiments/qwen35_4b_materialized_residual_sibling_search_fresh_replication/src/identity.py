"""Frozen successor identities and authenticated parent-lineage reads."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


EXPERIMENT_ID = "qwen35_4b_materialized_residual_sibling_search_fresh_replication"
SCIENTIFIC_PARENT_ID = "qwen35_4b_materialized_residual_sibling_search"
TASK_NAMESPACE = "materialized-residual-fresh-replication-v1"
REQUEST_NAMESPACE = TASK_NAMESPACE
_NAMESPACE_RE = re.compile(r"\A[a-z0-9]+(?:-[a-z0-9]+)*-v[1-9][0-9]*\Z")

# These hashes authenticate the complete incident chain that makes replay of
# the parent's terminal invocation impermissible.  The historical files stay
# in the parent experiment; the successor records and rechecks them by value.
PARENT_LINEAGE: dict[str, dict[str, str]] = {
    "construction_manifest": {
        "path": (
            "experiments/qwen35_4b_materialized_residual_sibling_search/"
            "data/procedural/manifest.json"
        ),
        "sha256": "a566129343a4c09473005b355f2ea7aa35549efe101dc14dc8d73ab5a0323857",
    },
    "original_preoutcome": {
        "path": (
            "experiments/qwen35_4b_materialized_residual_sibling_search/"
            "runs/mechanics/prepared/preoutcome_receipt.json"
        ),
        "sha256": "3de86e8b08bf37174cf687e4e7220ff802386c61652bafb6554cf1e772c89b88",
    },
    "original_implementation_lock": {
        "path": (
            "experiments/qwen35_4b_materialized_residual_sibling_search/"
            "runs/mechanics/implementation_lock.json"
        ),
        "sha256": "896c4cc64e157627eaf35a8a4365af766971644905fbde5c72e4d35ea72792e0",
    },
    "attempt_1_live_preflight": {
        "path": (
            "experiments/qwen35_4b_materialized_residual_sibling_search/"
            "runs/mechanics/raw/live_preflight.json"
        ),
        "sha256": "a438ecf190e4006e8f19368f907d37595a05d543ca8158c33d27333c816f14b5",
    },
    "attempt_1_incident": {
        "path": (
            "experiments/qwen35_4b_materialized_residual_sibling_search/"
            "runs/mechanics/preflight_attempt_1_incident.json"
        ),
        "sha256": "424fadd111d81b253dfbd79c8cbbec0f9b541eb1dd2437e97f880d3c01a6e5ce",
    },
    "v2_preoutcome": {
        "path": (
            "experiments/qwen35_4b_materialized_residual_sibling_search/"
            "runs/mechanics/preoutcome_receipt_v2.json"
        ),
        "sha256": "118546faaa5c18ada5ca57aeee047e90c0633ffd50bddfc4b6daf3b899270372",
    },
    "v2_implementation_lock": {
        "path": (
            "experiments/qwen35_4b_materialized_residual_sibling_search/"
            "runs/mechanics/implementation_lock_v2.json"
        ),
        "sha256": "953da4e9ba5b4d19f5d1b785d907b7d78379705af50ab124a0027b7ce79a1264",
    },
    "attempt_2_live_preflight": {
        "path": (
            "experiments/qwen35_4b_materialized_residual_sibling_search/"
            "runs/mechanics/raw_v2/live_preflight.json"
        ),
        "sha256": "16961d80437835fdfdcad0fa78d482a847c00a243d25fa0a5d86062dc5fcea25",
    },
    "attempt_2_terminal_started": {
        "path": (
            "experiments/qwen35_4b_materialized_residual_sibling_search/"
            "runs/mechanics/raw_v2/suffix_materialized.started.json"
        ),
        "sha256": "f6aa447b1936fac397a353fc13183f008e31884b5006ed7fc50ac78deed3387a",
    },
    "attempt_2_terminal_prepared_input": {
        "path": (
            "experiments/qwen35_4b_materialized_residual_sibling_search/"
            "runs/mechanics/prepared/suffix_materialized_requests.jsonl"
        ),
        "sha256": "b33d66bc25124f0d7eebc6ded30fcbfa759f0ba76ada30968c3ac4cb43f32d76",
    },
    "attempt_2_incident": {
        "path": (
            "experiments/qwen35_4b_materialized_residual_sibling_search/"
            "runs/mechanics/mechanics_attempt_2_incident.json"
        ),
        "sha256": "48ae3f49addc43b435fd5c2b121d57b9498223488a5251999910f56c50e9d4d2",
    },
    "attempt_2_incident_report": {
        "path": (
            "experiments/qwen35_4b_materialized_residual_sibling_search/"
            "reports/mechanics_termination_incident.md"
        ),
        "sha256": "df3b42a4aae4e0f437acf4fca8164c716770fa312a6f59a831af79796085f478",
    },
    "attempt_2_adversarial_review": {
        "path": (
            "experiments/qwen35_4b_materialized_residual_sibling_search/"
            "reports/mechanics_attempt_2_adversarial_review.md"
        ),
        "sha256": "1227e67749b7e2314d7ce2f93fe85feb3ee2c79208b124776466d89d7fc1d9eb",
    },
}


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_regular_file(root: Path, relative: str, expected_sha256: str) -> Path:
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise RuntimeError(f"parent lineage path escaped the repository: {relative}")
    root = root.resolve()
    path = root / relative
    cursor = root
    for part in Path(relative).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise RuntimeError(f"parent lineage path contains a symlink: {relative}")
    if not path.is_file():
        raise RuntimeError(f"parent lineage file is absent or non-regular: {relative}")
    observed = file_sha256(path)
    if observed != expected_sha256:
        raise RuntimeError(
            f"parent lineage hash drift for {relative}: {observed} != {expected_sha256}"
        )
    return path


def verify_parent_lineage(root: Path) -> dict[str, dict[str, str]]:
    if len(PARENT_LINEAGE) != 13:
        raise RuntimeError("parent lineage inventory changed")
    verified: dict[str, dict[str, str]] = {}
    for name, row in sorted(PARENT_LINEAGE.items()):
        _verified_regular_file(root, row["path"], row["sha256"])
        verified[name] = dict(row)
    return verified


def verified_manifest_file(root: Path, row: Any) -> Path:
    if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
        raise RuntimeError("authenticated manifest file row has the wrong schema")
    path = row["path"]
    digest = row["sha256"]
    if not isinstance(path, str) or not isinstance(digest, str):
        raise RuntimeError("authenticated manifest file row has non-string fields")
    return _verified_regular_file(root, path, digest)


def namespaced_task_id(namespace: str, split: str, index: int) -> str:
    if not _NAMESPACE_RE.fullmatch(namespace):
        raise ValueError("task namespace has the wrong versioned form")
    if split not in {"mechanics", "qualification", "confirmation"}:
        raise ValueError("unknown task split")
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError("task index must be a non-negative integer")
    return f"{namespace}/{split}/{index:05d}"


def public_instance_payload(task: dict[str, Any]) -> dict[str, Any]:
    """Identity-free public substrate; orientation is intentionally omitted."""
    return {
        "depth": task["depth"],
        "visible": task["visible"],
        "unlabeled_probe_inputs": task["unlabeled_probe_inputs"],
    }


def public_instance_fingerprint(task: dict[str, Any]) -> str:
    return canonical_sha256(public_instance_payload(task))


def request_seed_key(
    namespace: str,
    family: str,
    task_id: str,
    candidate: str | None = None,
) -> list[str]:
    if namespace != REQUEST_NAMESPACE:
        raise ValueError("request namespace changed")
    if family not in {"suffix", "viability", "direct", "listwise"}:
        raise ValueError("unknown request family")
    if not task_id.startswith(namespace + "/"):
        raise ValueError("request task ID is outside the successor namespace")
    if family in {"suffix", "viability"} and not candidate:
        raise ValueError("candidate-bound family requires a candidate")
    if family in {"direct", "listwise"} and candidate is not None:
        raise ValueError("candidate-blind family cannot include a candidate")
    return [namespace, family, task_id] + ([] if candidate is None else [candidate])


def request_id(seed_key: list[str]) -> str:
    return canonical_sha256(seed_key)
