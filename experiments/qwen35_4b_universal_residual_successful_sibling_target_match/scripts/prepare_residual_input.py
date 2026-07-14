#!/usr/bin/env python3
"""Derive the residual-only, oracle-free sibling input from published lineage."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "scripts"))

import sibling_policy as policy  # noqa: E402


SOURCE = EXP / "data" / "inherited_collection_tasks_seed77115.jsonl"
INVENTORY = EXP / "data" / "inherited_greedy_failure_inventory_seed66115.json"
GREEDY_RECEIPT = EXP / "data" / "inherited_greedy_collection_receipt.json"
STOP_RECEIPT = EXP / "data" / "inherited_balanced_stop_receipt.json"
OUT = EXP / "data" / "residual_sibling_input_seed66117.jsonl"
MANIFEST = EXP / "data" / "residual_collection_manifest.json"
SOURCE_SHA256 = "9071ce57e98e95ff9c0555e236abea5e71ab7e114883e3de68742c4e667603e9"
INVENTORY_SHA256 = "8e21caf82b29ee800a4846401a7561e6624461457324e19f8a50a42175eed783"
GREEDY_RECEIPT_SHA256 = "cee1f19d1319e01a9aa69cedb1e9261de29a3c24d9e07477b77a1f5d9cc94962"
STOP_RECEIPT_SHA256 = "3397b7738eefcc12a0c6a8c6687e627e9a6125d7186d48f4a5c73e6e2a9d2a6e"
ORIGIN_EXPERIMENT = "qwen35_4b_universal_successful_sibling_target_match"
EXCLUDED_SKILLS = ("select", "count", "route")
EXPECTED_ROWS = 225


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def jsonl_bytes(rows: list[dict]) -> bytes:
    return (
        "\n".join(
            json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            for row in rows
        )
        + "\n"
    ).encode()


def build_outputs() -> dict[Path, bytes]:
    expected_hashes = {
        SOURCE: SOURCE_SHA256,
        INVENTORY: INVENTORY_SHA256,
        GREEDY_RECEIPT: GREEDY_RECEIPT_SHA256,
        STOP_RECEIPT: STOP_RECEIPT_SHA256,
    }
    for path, expected in expected_hashes.items():
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"inherited lineage changed: {path}")
    source_rows = load_jsonl(SOURCE)
    inventory = load_json(INVENTORY)
    greedy_receipt = load_json(GREEDY_RECEIPT)
    stop_receipt = load_json(STOP_RECEIPT)
    items = inventory.get("items")
    if (
        len(source_rows) != 624
        or not isinstance(items, list)
        or len(items) != 624
        or inventory.get("experiment_id") != ORIGIN_EXPERIMENT
        or inventory.get("hard_failure_rows") != 227
        or greedy_receipt.get("experiment_id") != ORIGIN_EXPERIMENT
        or greedy_receipt.get("stage") != "authenticated_replay_parent_greedy_failure_collection"
        or greedy_receipt.get("seed") != 66115
        or stop_receipt.get("experiment_id") != ORIGIN_EXPERIMENT
        or stop_receipt.get("outcome") != "STOP_INSUFFICIENT_GREEDY_FAILURES"
        or stop_receipt.get("inventory_sha256") != INVENTORY_SHA256
    ):
        raise ValueError("inherited collection identity changed")
    source_by_id = {row["task_id"]: row for row in source_rows}
    if len(source_by_id) != 624:
        raise ValueError("inherited task identities are not unique")
    selected = [
        item
        for item in items
        if item.get("hard_failure") is True and item.get("skill") in policy.EXPECTED_SKILLS
    ]
    counts = Counter(item["skill"] for item in selected)
    if (
        len(selected) != EXPECTED_ROWS
        or set(counts) != set(policy.EXPECTED_SKILLS)
        or any(counts[skill] < policy.QUOTA_PER_SKILL for skill in policy.EXPECTED_SKILLS)
        or any(item.get("skill") in EXCLUDED_SKILLS for item in selected)
    ):
        raise ValueError(f"prospective residual set changed: {counts}")
    rows = []
    for item in selected:
        source = source_by_id[item["task_id"]]
        rows.append({
            "id": item["task_id"],
            "messages": source["messages"],
            "meta": {
                "kind": source["kind"],
                "skill": source["selection_skill"],
                "surface": source["surface"],
                "level": source["level"],
                "construction_seed": 77115,
                "greedy_failure_reasons": item["reasons"],
            },
        })
    input_bytes = jsonl_bytes(rows)
    forbidden = (b'"answer"', b'"think"', b'"_audit"', b'"truth_valid"', b'"expected_answer"')
    if any(value in input_bytes for value in forbidden):
        raise ValueError("residual sibling input contains an oracle field")
    manifest = {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "stage": "inherited_residual_sibling_collection_substrate",
        "origin_experiment": ORIGIN_EXPERIMENT,
        "origin": {
            "construction_seed": 77115,
            "greedy_seed": 66115,
            "source_sha256": SOURCE_SHA256,
            "failure_inventory_sha256": INVENTORY_SHA256,
            "greedy_receipt_sha256": GREEDY_RECEIPT_SHA256,
            "balanced_stop_receipt_sha256": STOP_RECEIPT_SHA256,
            "hard_failure_rows": 227,
        },
        "residual_policy": {
            "skills": list(policy.EXPECTED_SKILLS),
            "excluded_skills": list(EXCLUDED_SKILLS),
            "definition": "published skills with at least four hard greedy failures",
            "quota_per_skill": policy.QUOTA_PER_SKILL,
            "hard_failure_rows": len(rows),
            "hard_failure_rows_by_skill": dict(sorted(counts.items())),
        },
        "sibling_collection": {
            "seed": 66117,
            "samples_per_failure": policy.SAMPLES_PER_FAILURE,
            "max_sibling_thinking_tokens": policy.MAX_SIBLING_THINKING_TOKENS,
            "input_path": OUT.relative_to(ROOT).as_posix(),
            "input_sha256": sha256_bytes(input_bytes),
            "input_rows": len(rows),
            "oracle_fields_in_input": False,
        },
        "retention_policy": {
            "saturated_or_thin_skills": list(EXCLUDED_SKILLS),
            "protected_by_active_replay": True,
            "unchanged_all_skill_local_gate_required": True,
        },
        "benchmark_data_read": False,
        "aggregate_seed_open": False,
    }
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()
    return {OUT: input_bytes, MANIFEST: manifest_bytes}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    outputs = build_outputs()
    if args.check:
        for path, expected in outputs.items():
            if not path.is_file() or path.read_bytes() != expected:
                parser.error(f"derived residual substrate is absent or changed: {path}")
    else:
        conflict = next((path for path in outputs if path.exists()), None)
        if conflict is not None:
            parser.error(f"refusing to overwrite residual substrate: {conflict}")
        for path, value in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
    print(json.dumps({
        path.relative_to(ROOT).as_posix(): sha256_bytes(value)
        for path, value in outputs.items()
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
