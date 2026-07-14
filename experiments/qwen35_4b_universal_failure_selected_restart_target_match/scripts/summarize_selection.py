#!/usr/bin/env python3
"""Summarize the frozen restart selection without changing its algorithm."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
INVENTORY = EXP / "data" / "failure_inventory_seed66114.json"
SOURCE = EXP / "data" / "counterfactual_restart_source.jsonl"
RECEIPT = EXP / "data" / "restart_selection_receipt.json"
OUT = EXP / "data" / "selection_summary.json"
EXPECTED = {
    INVENTORY: "c19d3de700c1ccab931298816c259b587ae0476d5105e3a29b75d93007966240",
    SOURCE: "022b1ea4cfe2bb50fca7f5fdc472a0bf228a5d7a7adb637b221b8efe434d951f",
    RECEIPT: "567d6b020b9120c82bd19fdc7992dc49b927df2b604978ab3d6ae64e2c05b662",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_payload() -> dict:
    for path, expected in EXPECTED.items():
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"frozen selection artifact changed: {path}")
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    source = [
        json.loads(line)
        for line in SOURCE.read_text(encoding="utf-8").splitlines()
        if line
    ]
    selected_ids = set(inventory["selected_task_ids"])
    selected = [item for item in inventory["items"] if item["task_id"] in selected_ids]
    hard_availability = Counter(
        item["skill"] for item in inventory["items"] if item["hard_failure"]
    )
    selected_reasons = Counter(
        reason for item in selected for reason in item["reasons"]
    )
    selected_hard_by_skill = Counter(
        item["skill"] for item in selected if item["hard_failure"]
    )
    source_skills = Counter(row["kind"].removeprefix("u_counterfactual_restart_") for row in source)
    if (
        receipt.get("outcome") != "PASS_RESTART_QUOTAS"
        or len(source) != 52
        or len(selected) != 52
        or any("assistant_prefix_token_ids" in row for row in source)
        or any(row.get("failure_selection", {}).get("parent_prefix_in_training_context") is not False for row in source)
        or set(selected_ids) != {row["failure_selection"]["parent_task_id"] for row in source}
        or set(source_skills.values()) != {4}
    ):
        raise ValueError("frozen restart selection failed its analysis contract")
    skills = receipt["selected_rows_by_skill"]
    return {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "stage": "post_selection_model_free_summary",
        "source_artifacts": {
            path.relative_to(EXP).as_posix(): expected
            for path, expected in EXPECTED.items()
        },
        "pool": {
            "rows": inventory["rows"],
            "eligible_rows": inventory["eligible_rows"],
            "hard_failure_rows": inventory["hard_failure_rows"],
            "failure_reasons": inventory["failure_reasons"],
            "availability_by_skill": inventory["availability_by_skill"],
            "hard_availability_by_skill": {
                skill: hard_availability[skill] for skill in skills
            },
        },
        "selected": {
            "rows": len(selected),
            "rows_by_skill": skills,
            "hard_failure_rows": sum(item["hard_failure"] for item in selected),
            "budget_only_rows": sum(not item["hard_failure"] for item in selected),
            "hard_rows_by_skill": {
                skill: selected_hard_by_skill[skill] for skill in skills
            },
            "failure_reasons": dict(sorted(selected_reasons.items())),
            "parent_prefix_rows": 0,
            "full_oracle_restart_rows": len(source),
        },
        "training_authorized": False,
        "next_required_gate": "exact three-axis exposure feasibility and second adversarial compute review",
        "benchmark_data_read": False,
        "aggregate_seed_open": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    payload = build_payload()
    value = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    if args.check:
        if not OUT.is_file() or OUT.read_bytes() != value:
            parser.error("selection summary is absent or changed")
    else:
        if OUT.exists():
            parser.error("refusing to overwrite selection summary")
        OUT.write_bytes(value)
    print(json.dumps({"out": str(OUT), "sha256": hashlib.sha256(value).hexdigest()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
