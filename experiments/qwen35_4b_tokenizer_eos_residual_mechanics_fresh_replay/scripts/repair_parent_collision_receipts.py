#!/usr/bin/env python3
"""Migrate the administrative collision-manifest receipt chain without key access."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
REL = "experiments/qwen35_4b_tokenizer_eos_residual_mechanics_fresh_replay"
MANIFEST_REL = f"{REL}/runs/parent_lineage/collision_manifest.json"
CONSTRUCTION_REL = f"{REL}/runs/construction/summary.json"
PREOUTCOME_REL = f"{REL}/runs/prepared/preoutcome_receipt.json"
TOKENIZER_REL = f"{REL}/runs/tokenizer/receipt.json"
REPAIR_REL = f"{REL}/runs/construction/parent_collision_receipt_repair.json"
MANIFEST = ROOT / MANIFEST_REL
CONSTRUCTION = ROOT / CONSTRUCTION_REL
PREOUTCOME = ROOT / PREOUTCOME_REL
TOKENIZER = ROOT / TOKENIZER_REL
REPAIR = ROOT / REPAIR_REL
OLD_COMMIT = "98e9e9f6cac0eade7fd352157b32f62b67d55ef0"
OLD_MANIFEST_SHA256 = (
    "72faacf5bebf4a8964faaba81d5088dc9602e1a228f818f8e998a03dc145e8e5"
)
NEW_MANIFEST_SHA256 = (
    "450eb55d41b09aabff33967d5c75e6315e41cf093bd043004045a8ae5d0d07ef"
)
OLD_RECEIPT_SHA256S = {
    CONSTRUCTION_REL: "7e1ff08290dd3b963e312fcec21c88ef2c5290d72984c1fb8d26d1d2138942b6",
    PREOUTCOME_REL: "41c7dd32a1edcf95f1afe7c6ea3eabf8b680a54cd2e77e5723978a3ca84a7331",
    TOKENIZER_REL: "da57c1a7da9d35aef45676167d57fcea704b74612ab92ee8c0302cf6a366bd16",
}
ADDED_PARENT_SOURCE = (
    "experiments/qwen35_4b_tokenizer_eos_answer_commit_factorial/src/protocol.py"
)
ADDED_PARENT_SOURCE_SHA256 = (
    "628b1235bfa84e476b5bad62c899e8f2279c6a6edb0c5617f98d59af6ad297ec"
)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def exact_json_equal(observed: Any, expected: Any) -> bool:
    if type(observed) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(observed) == set(expected) and all(
            exact_json_equal(observed[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(observed) == len(expected) and all(
            exact_json_equal(left, right)
            for left, right in zip(observed, expected, strict=True)
        )
    return observed == expected


def read_exact_json(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"receipt path is unsafe or absent: {path}")
    payload = path.read_bytes()
    if expected_sha256 is not None and sha256_bytes(payload) != expected_sha256:
        raise RuntimeError(f"receipt changed before administrative migration: {path}")
    value = json.loads(payload)
    if not isinstance(value, dict) or payload != canonical_bytes(value):
        raise RuntimeError(f"receipt is not canonical JSON: {path}")
    return value


def old_manifest_from_git() -> dict[str, Any]:
    completed = subprocess.run(
        ["git", "show", f"{OLD_COMMIT}:{MANIFEST_REL}"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if sha256_bytes(completed.stdout) != OLD_MANIFEST_SHA256:
        raise RuntimeError("frozen old collision manifest Git object changed")
    value = json.loads(completed.stdout)
    if not isinstance(value, dict) or completed.stdout != canonical_bytes(value):
        raise RuntimeError("frozen old collision manifest is not canonical JSON")
    return value


def validate_manifest_delta(old: dict[str, Any], new: dict[str, Any]) -> None:
    if set(old) != set(new) or "administrative_sources" not in old:
        raise RuntimeError("collision manifest schema changed during migration")
    old_domains = {key: value for key, value in old.items() if key != "administrative_sources"}
    new_domains = {key: value for key, value in new.items() if key != "administrative_sources"}
    if not exact_json_equal(old_domains, new_domains):
        raise RuntimeError("a scientific parent collision domain changed")
    old_sources = old["administrative_sources"]
    new_sources = new["administrative_sources"]
    if (
        not isinstance(old_sources, dict)
        or not isinstance(new_sources, dict)
        or len(old_sources) != 7
        or len(new_sources) != 8
        or any(new_sources.get(path) != digest for path, digest in old_sources.items())
        or set(new_sources) != {*old_sources, ADDED_PARENT_SOURCE}
        or new_sources.get(ADDED_PARENT_SOURCE) != ADDED_PARENT_SOURCE_SHA256
    ):
        raise RuntimeError("administrative source correction is not the exact one-file delta")


def write_atomic(path: Path, payload: bytes) -> None:
    if path.is_symlink() or not path.parent.is_dir():
        raise RuntimeError(f"unsafe repair destination: {path}")
    temporary = path.with_name(path.name + ".repair-tmp")
    if temporary.exists() or temporary.is_symlink():
        raise RuntimeError(f"unsafe stale repair temporary: {temporary}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def validate_completed_repair() -> dict[str, Any] | None:
    if not REPAIR.exists() and not REPAIR.is_symlink():
        return None
    receipt = read_exact_json(REPAIR)
    if receipt.get("decision") != "PARENT_COLLISION_RECEIPT_REPAIR_PASS":
        raise RuntimeError("existing repair receipt is not passing")
    if receipt.get("new_manifest_sha256") != NEW_MANIFEST_SHA256:
        raise RuntimeError("existing repair receipt names a different manifest")
    for relative, path in (
        (CONSTRUCTION_REL, CONSTRUCTION),
        (PREOUTCOME_REL, PREOUTCOME),
        (TOKENIZER_REL, TOKENIZER),
    ):
        entry = receipt.get("receipt_migrations", {}).get(relative)
        if (
            not isinstance(entry, dict)
            or entry.get("old_sha256") != OLD_RECEIPT_SHA256S[relative]
            or entry.get("new_sha256") != sha256_bytes(path.read_bytes())
        ):
            raise RuntimeError(f"existing repair no longer binds {relative}")
    if sha256_bytes(MANIFEST.read_bytes()) != NEW_MANIFEST_SHA256:
        raise RuntimeError("corrected manifest changed after receipt repair")
    return receipt


def main() -> int:
    completed = validate_completed_repair()
    if completed is not None:
        print(json.dumps(completed, indent=2, sort_keys=True))
        return 0

    old_manifest = old_manifest_from_git()
    new_manifest = read_exact_json(MANIFEST, NEW_MANIFEST_SHA256)
    validate_manifest_delta(old_manifest, new_manifest)

    construction = read_exact_json(
        CONSTRUCTION, OLD_RECEIPT_SHA256S[CONSTRUCTION_REL]
    )
    preoutcome = read_exact_json(PREOUTCOME, OLD_RECEIPT_SHA256S[PREOUTCOME_REL])
    tokenizer = read_exact_json(TOKENIZER, OLD_RECEIPT_SHA256S[TOKENIZER_REL])
    expected_parent_read = {
        MANIFEST_REL: {
            "purpose": "authenticated_hash_only_parent_collision_domains",
            "sha256": OLD_MANIFEST_SHA256,
        }
    }
    if not exact_json_equal(construction.get("parent_read_receipt"), expected_parent_read):
        raise RuntimeError("old construction parent read receipt changed")
    if preoutcome.get("construction_summary_sha256") != OLD_RECEIPT_SHA256S[CONSTRUCTION_REL]:
        raise RuntimeError("old preoutcome construction binding changed")
    manifest_read = tokenizer.get("read_receipt", {}).get(MANIFEST_REL)
    preoutcome_read = tokenizer.get("read_receipt", {}).get(PREOUTCOME_REL)
    if (
        tokenizer.get("preoutcome_sha256") != OLD_RECEIPT_SHA256S[PREOUTCOME_REL]
        or not isinstance(manifest_read, dict)
        or manifest_read.get("sha256") != OLD_MANIFEST_SHA256
        or not isinstance(preoutcome_read, dict)
        or preoutcome_read.get("sha256") != OLD_RECEIPT_SHA256S[PREOUTCOME_REL]
    ):
        raise RuntimeError("old tokenizer administrative bindings changed")

    construction["parent_read_receipt"][MANIFEST_REL]["sha256"] = NEW_MANIFEST_SHA256
    construction_payload = canonical_bytes(construction)
    construction_sha256 = sha256_bytes(construction_payload)
    preoutcome["construction_summary_sha256"] = construction_sha256
    preoutcome_payload = canonical_bytes(preoutcome)
    preoutcome_sha256 = sha256_bytes(preoutcome_payload)
    tokenizer["preoutcome_sha256"] = preoutcome_sha256
    tokenizer["read_receipt"][MANIFEST_REL]["sha256"] = NEW_MANIFEST_SHA256
    tokenizer["read_receipt"][PREOUTCOME_REL]["sha256"] = preoutcome_sha256
    tokenizer_payload = canonical_bytes(tokenizer)
    tokenizer_sha256 = sha256_bytes(tokenizer_payload)

    repair = {
        "schema_version": 1,
        "stage": "outcome_blind_parent_collision_receipt_repair",
        "decision": "PARENT_COLLISION_RECEIPT_REPAIR_PASS",
        "old_git_object": OLD_COMMIT,
        "old_manifest_sha256": OLD_MANIFEST_SHA256,
        "new_manifest_sha256": NEW_MANIFEST_SHA256,
        "collision_domains_exactly_unchanged": True,
        "administrative_source_added": {
            "path": ADDED_PARENT_SOURCE,
            "sha256": ADDED_PARENT_SOURCE_SHA256,
        },
        "receipt_migrations": {
            CONSTRUCTION_REL: {
                "old_sha256": OLD_RECEIPT_SHA256S[CONSTRUCTION_REL],
                "new_sha256": construction_sha256,
            },
            PREOUTCOME_REL: {
                "old_sha256": OLD_RECEIPT_SHA256S[PREOUTCOME_REL],
                "new_sha256": preoutcome_sha256,
            },
            TOKENIZER_REL: {
                "old_sha256": OLD_RECEIPT_SHA256S[TOKENIZER_REL],
                "new_sha256": tokenizer_sha256,
            },
        },
        "files_rewritten": [CONSTRUCTION_REL, PREOUTCOME_REL, TOKENIZER_REL],
        "model_loaded": False,
        "model_calls": 0,
        "sampled_model_outputs_read": 0,
        "hidden_files_read": [],
        "local_key_files_read": [],
        "benchmark_files_read": [],
        "parent_raw_sampled_bundles_read": [],
    }

    write_atomic(CONSTRUCTION, construction_payload)
    write_atomic(PREOUTCOME, preoutcome_payload)
    write_atomic(TOKENIZER, tokenizer_payload)
    write_atomic(REPAIR, canonical_bytes(repair))
    print(json.dumps(repair, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
