#!/usr/bin/env python3
"""Freeze greedy failures and materialize oracle-free sibling-sampling input."""

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


SOURCE = EXP / "data" / "collection_tasks_seed77115.jsonl"
MANIFEST = EXP / "data" / "collection_task_manifest.json"
GREEDY = EXP / "runs" / "greedy_collection" / "seed66115.jsonl"
GREEDY_RECEIPT = EXP / "runs" / "greedy_collection" / "seed66115.receipt.json"
INVENTORY = EXP / "data" / "greedy_failure_inventory_seed66115.json"
SIBLING_INPUT = EXP / "data" / "sibling_input_seed66116.jsonl"
RECEIPT = EXP / "data" / "greedy_failure_selection_receipt.json"


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


def authenticate() -> tuple[list[dict], list[dict], dict, dict]:
    manifest = load_json(MANIFEST)
    receipt = load_json(GREEDY_RECEIPT)
    source = load_jsonl(SOURCE)
    greedy = load_jsonl(GREEDY)
    if (
        manifest.get("experiment_id") != EXP.name
        or manifest.get("construction_seed") != 77115
        or manifest.get("rows") != 624
        or manifest.get("source", {}).get("sha256") != sha256_file(SOURCE)
        or receipt.get("experiment_id") != EXP.name
        or receipt.get("stage") != "authenticated_replay_parent_greedy_failure_collection"
        or receipt.get("seed") != 66115
        or receipt.get("rows") != 624
        or receipt.get("rollouts_sha256") != sha256_file(GREEDY)
        or receipt.get("sampling") != {
            "greedy": True,
            "max_model_len": 4096,
            "max_tokens": 1024,
            "n": 1,
            "thinking": "natural",
        }
        or receipt.get("benchmark_data_read") is not False
        or len(source) != 624
        or len(greedy) != 624
    ):
        raise ValueError("greedy collection or source failed authentication")
    return source, greedy, manifest, receipt


def build_outputs() -> tuple[dict[Path, bytes], bool]:
    source_rows, greedy_rows, manifest, greedy_receipt = authenticate()
    source_by_id = {row["task_id"]: row for row in source_rows}
    greedy_by_id = {row.get("id"): row for row in greedy_rows}
    if set(source_by_id) != set(greedy_by_id):
        raise ValueError("greedy task identities differ from source identities")
    items = [
        policy.grade_greedy(source_by_id[task_id], greedy_by_id[task_id])
        for task_id in source_by_id
    ]
    failures = [item for item in items if item["hard_failure"]]
    availability = Counter(item["skill"] for item in failures)
    quota_possible = all(
        availability[skill] >= policy.QUOTA_PER_SKILL for skill in policy.EXPECTED_SKILLS
    )
    failure_ids = {item["task_id"] for item in failures}
    input_rows = [
        {
            "id": row["task_id"],
            "messages": row["messages"],
            "meta": {
                "kind": row["kind"],
                "skill": row["selection_skill"],
                "surface": row["surface"],
                "level": row["level"],
                "construction_seed": 77115,
                "greedy_failure_reasons": next(
                    item["reasons"] for item in failures if item["task_id"] == row["task_id"]
                ),
            },
        }
        for row in source_rows
        if row["task_id"] in failure_ids
    ]
    input_bytes = jsonl_bytes(input_rows) if input_rows else b""
    forbidden = (b'"answer"', b'"think"', b'"_audit"', b'"truth_valid"', b'"expected_answer"')
    if any(value in input_bytes for value in forbidden):
        raise ValueError("sibling-sampling input contains an oracle field")
    reasons = Counter(reason for item in items for reason in item["reasons"])
    inventory = {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "stage": "greedy_failure_inventory",
        "rows": len(items),
        "hard_failure_rows": len(failures),
        "availability_by_skill": {
            skill: availability[skill] for skill in policy.EXPECTED_SKILLS
        },
        "failure_reasons": dict(sorted(reasons.items())),
        "minimum_required_per_skill": policy.QUOTA_PER_SKILL,
        "failure_availability_pass": quota_possible,
        "items": items,
        "source_sha256": manifest["source"]["sha256"],
        "greedy_rollout_sha256": greedy_receipt["rollouts_sha256"],
        "benchmark_data_read": False,
    }
    inventory_bytes = (
        json.dumps(inventory, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()
    receipt = {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "stage": "greedy_failure_selection",
        "outcome": "PASS_FAILURE_AVAILABILITY" if quota_possible else "STOP_INSUFFICIENT_GREEDY_FAILURES",
        "greedy_seed": 66115,
        "sibling_sampling_seed": 66116,
        "hard_failure_rows": len(failures),
        "hard_failure_rows_by_skill": {
            skill: availability[skill] for skill in policy.EXPECTED_SKILLS
        },
        "inventory_sha256": sha256_bytes(inventory_bytes),
        "sibling_input_rows": len(input_rows) if quota_possible else 0,
        "sibling_input_sha256": sha256_bytes(input_bytes) if quota_possible else None,
        "sibling_samples_per_failure": policy.SAMPLES_PER_FAILURE,
        "oracle_fields_in_sibling_input": False,
        "next_authorized_stage": "collect-siblings" if quota_possible else None,
        "training_authorized": False,
        "benchmark_data_read": False,
        "aggregate_seed_open": False,
    }
    receipt_bytes = (
        json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()
    outputs = {INVENTORY: inventory_bytes, RECEIPT: receipt_bytes}
    if quota_possible:
        outputs[SIBLING_INPUT] = input_bytes
    return outputs, quota_possible


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    outputs, passed = build_outputs()
    candidates = (INVENTORY, SIBLING_INPUT, RECEIPT)
    if args.check:
        for path, expected in outputs.items():
            if not path.is_file() or path.read_bytes() != expected:
                parser.error(f"derived greedy-failure artifact is absent or changed: {path}")
        if not passed and SIBLING_INPUT.exists():
            parser.error("sibling input exists after failed failure-availability gate")
    else:
        conflict = next((path for path in candidates if path.exists()), None)
        if conflict is not None:
            parser.error(f"refusing to overwrite greedy-failure artifact: {conflict}")
        for path, value in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
    print(json.dumps(json.loads(outputs[RECEIPT]), indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
