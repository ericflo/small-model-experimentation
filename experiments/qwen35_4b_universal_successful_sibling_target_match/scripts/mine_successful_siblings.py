#!/usr/bin/env python3
"""Select shortest verified same-parent siblings under the frozen per-skill gate."""

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
GREEDY_INVENTORY = EXP / "data" / "greedy_failure_inventory_seed66115.json"
FAILURE_RECEIPT = EXP / "data" / "greedy_failure_selection_receipt.json"
SIBLING_ROLLOUT = EXP / "runs" / "sibling_collection" / "seed66116.jsonl"
SIBLING_RECEIPT = EXP / "runs" / "sibling_collection" / "seed66116.receipt.json"
INVENTORY = EXP / "data" / "successful_sibling_inventory_seed66116.json"
TRAINING_SOURCE = EXP / "data" / "successful_sibling_source.jsonl"
SELECTION_RECEIPT = EXP / "data" / "successful_sibling_selection_receipt.json"


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


def authenticate() -> tuple[list[dict], dict, list[dict], dict]:
    source = load_jsonl(SOURCE)
    greedy_inventory = load_json(GREEDY_INVENTORY)
    failure_receipt = load_json(FAILURE_RECEIPT)
    sibling_rows = load_jsonl(SIBLING_ROLLOUT)
    sibling_receipt = load_json(SIBLING_RECEIPT)
    expected_ids = {
        item["task_id"]
        for item in greedy_inventory.get("items", [])
        if item.get("hard_failure") is True
    }
    if (
        len(source) != 624
        or failure_receipt.get("outcome") != "PASS_FAILURE_AVAILABILITY"
        or failure_receipt.get("inventory_sha256") != sha256_file(GREEDY_INVENTORY)
        or failure_receipt.get("sibling_input_rows") != len(expected_ids)
        or sibling_receipt.get("experiment_id") != EXP.name
        or sibling_receipt.get("stage") != "authenticated_replay_parent_successful_sibling_collection"
        or sibling_receipt.get("seed") != 66116
        or sibling_receipt.get("rows") != len(expected_ids)
        or sibling_receipt.get("samples") != len(expected_ids) * policy.SAMPLES_PER_FAILURE
        or sibling_receipt.get("rollouts_sha256") != sha256_file(SIBLING_ROLLOUT)
        or sibling_receipt.get("sampling") != {
            "greedy": False,
            "max_model_len": 4096,
            "max_tokens": 1024,
            "n": 16,
            "temperature": 0.6,
            "thinking": "natural",
            "top_k": 20,
            "top_p": 0.95,
        }
        or sibling_receipt.get("benchmark_data_read") is not False
        or len(sibling_rows) != len(expected_ids)
        or {row.get("id") for row in sibling_rows} != expected_ids
    ):
        raise ValueError("sibling collection or greedy inventory failed authentication")
    return source, greedy_inventory, sibling_rows, sibling_receipt


def compact_grade(item: dict) -> dict:
    return {key: value for key, value in item.items() if key not in {"think", "answer"}}


def build_outputs() -> tuple[dict[Path, bytes], bool]:
    source_rows, greedy_inventory, sibling_rows, sibling_receipt = authenticate()
    source_by_id = {row["task_id"]: row for row in source_rows}
    greedy_by_id = {item["task_id"]: item for item in greedy_inventory["items"]}
    rollout_by_id = {row["id"]: row for row in sibling_rows}
    best_by_task: list[dict] = []
    inventory_items: list[dict] = []
    rejection_reasons: Counter[str] = Counter()
    qualified_samples = 0
    for task_id in sorted(rollout_by_id):
        source = source_by_id[task_id]
        if greedy_by_id[task_id].get("hard_failure") is not True:
            raise ValueError("sibling rollout contains a nonfailure task")
        best, grades = policy.choose_best_sibling(source, rollout_by_id[task_id])
        qualified = [item for item in grades if item["qualified"]]
        qualified_samples += len(qualified)
        rejection_reasons.update(reason for item in grades for reason in item["reasons"])
        record = {
            "task_id": task_id,
            "skill": source["selection_skill"],
            "kind": source["kind"],
            "greedy_failure_reasons": greedy_by_id[task_id]["reasons"],
            "qualified_samples": len(qualified),
            "best": best,
        }
        if best is not None:
            best_by_task.append(record)
        inventory_items.append(
            {
                **record,
                "best": compact_grade(best) if best is not None else None,
                "samples": [compact_grade(item) for item in grades],
            }
        )
    selected, availability = policy.select_balanced(best_by_task)
    passed = bool(selected)
    selected_by_skill = Counter(item["skill"] for item in selected)
    training_bytes: bytes | None = None
    if passed:
        training_rows = [
            policy.training_row(
                source_by_id[item["task_id"]], item, sibling_receipt["rollouts_sha256"]
            )
            for item in selected
        ]
        training_bytes = jsonl_bytes(training_rows)
        if (
            len(training_rows) != len(policy.EXPECTED_SKILLS) * policy.QUOTA_PER_SKILL
            or selected_by_skill
            != Counter({skill: policy.QUOTA_PER_SKILL for skill in policy.EXPECTED_SKILLS})
            or any(row["teacher"]["oracle_trace_used"] is not False for row in training_rows)
        ):
            raise ValueError("successful-sibling training source lost its balance or provenance")
    inventory = {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "stage": "successful_sibling_inventory",
        "greedy_failure_tasks": len(sibling_rows),
        "samples_per_task": policy.SAMPLES_PER_FAILURE,
        "samples": len(sibling_rows) * policy.SAMPLES_PER_FAILURE,
        "qualified_samples": qualified_samples,
        "tasks_with_qualified_sibling": len(best_by_task),
        "availability_by_skill": availability,
        "quota_per_skill": policy.QUOTA_PER_SKILL,
        "max_sibling_thinking_tokens": policy.MAX_SIBLING_THINKING_TOKENS,
        "selection_seed": policy.SELECTION_SEED,
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "selected_task_ids": [item["task_id"] for item in selected],
        "selected_rows_by_skill": dict(sorted(selected_by_skill.items())),
        "items": inventory_items,
        "sibling_rollout_sha256": sibling_receipt["rollouts_sha256"],
        "benchmark_data_read": False,
    }
    inventory_bytes = (
        json.dumps(inventory, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()
    receipt = {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "stage": "policy_supported_successful_sibling_selection",
        "outcome": "PASS_SUCCESSFUL_SIBLING_QUOTAS" if passed else "STOP_INSUFFICIENT_SUCCESSFUL_SIBLINGS",
        "selected_rows": len(selected),
        "selected_rows_by_skill": dict(sorted(selected_by_skill.items())),
        "availability_by_skill": availability,
        "inventory_sha256": sha256_bytes(inventory_bytes),
        "training_source_sha256": sha256_bytes(training_bytes) if training_bytes else None,
        "training_targets_from_sampled_parent_only": True,
        "hand_authored_oracle_targets": False,
        "shortest_qualified_sibling_per_selected_task": True,
        "exact_exposure_match_pending": passed,
        "training_authorized": False,
        "next_required_review": "exact exposure feasibility and adversarial compute review" if passed else None,
        "benchmark_data_read": False,
        "aggregate_seed_open": False,
    }
    receipt_bytes = (
        json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()
    outputs = {INVENTORY: inventory_bytes, SELECTION_RECEIPT: receipt_bytes}
    if training_bytes is not None:
        outputs[TRAINING_SOURCE] = training_bytes
    return outputs, passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    outputs, passed = build_outputs()
    candidates = (INVENTORY, TRAINING_SOURCE, SELECTION_RECEIPT)
    if args.check:
        for path, expected in outputs.items():
            if not path.is_file() or path.read_bytes() != expected:
                parser.error(f"derived sibling-selection artifact is absent or changed: {path}")
        if not passed and TRAINING_SOURCE.exists():
            parser.error("training source exists after failed successful-sibling gate")
    else:
        conflict = next((path for path in candidates if path.exists()), None)
        if conflict is not None:
            parser.error(f"refusing to overwrite sibling-selection artifact: {conflict}")
        for path, value in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
    print(json.dumps(json.loads(outputs[SELECTION_RECEIPT]), indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
