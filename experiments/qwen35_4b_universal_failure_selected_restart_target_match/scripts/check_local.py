#!/usr/bin/env python3
"""Apply the frozen absolute and control-relative fresh-local promotion gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SEED = 88010
ROWS = 26
PARENT = "replay_parent"
CONTROL = "replay_control"
CANDIDATE = "counterfactual_restart_candidate"
ARMS = (PARENT, CONTROL, CANDIDATE)
TARGET_KINDS = {"u_execute", "u_induct", "u_probe"}
EXPECTED_KINDS = {
    "u_abstain",
    "u_count",
    "u_execute",
    "u_induct",
    "u_optimize",
    "u_order",
    "u_probe",
    "u_repair",
    "u_route",
    "u_select",
    "u_state",
    "u_trace",
    "u_verify",
}
ABSTENTION_ANSWERS = {
    "",
    "ABSTAIN",
    "INSUFFICIENT",
    "N/A",
    "NO ANSWER",
    "NONE",
    "NULL",
    "UNDECIDABLE",
    "UNKNOWN",
}


def is_abstention(value: object) -> bool:
    return value is None or str(value).strip().upper() in ABSTENTION_ANSWERS


def selected_rows(payload: dict, label: str) -> list[dict]:
    rows = [row for row in payload.get("rows", []) if row.get("adapter") == label]
    if label not in ARMS or len(rows) != ROWS:
        raise ValueError(f"local receipt does not contain {ROWS} rows for {label}")
    return rows


def correct_count(payload: dict, label: str, kinds: set[str] | None = None) -> int:
    return sum(
        bool(row.get("correct"))
        for row in selected_rows(payload, label)
        if kinds is None or row.get("kind") in kinds
    )


def absolute_gate(payload: dict, label: str) -> dict:
    rows = selected_rows(payload, label)
    parsed = sum(row.get("parsed") is not None for row in rows)
    correct = sum(bool(row.get("correct")) for row in rows)
    cap_contacts = sum(bool(row.get("cap_contact")) for row in rows)
    route_abstentions = sum(
        row.get("kind") == "u_route" and is_abstention(row.get("parsed"))
        for row in rows
    )
    per_kind_correct = {
        kind: sum(row.get("kind") == kind and bool(row.get("correct")) for row in rows)
        for kind in sorted(TARGET_KINDS)
    }
    checks = {
        "parsed_at_least_24_of_26": parsed >= 24,
        "correct_at_least_17_of_26": correct >= 17,
        "cap_contacts_at_most_2": cap_contacts <= 2,
        "route_abstentions_at_most_1": route_abstentions <= 1,
        "execute_correct_at_least_1_of_2": per_kind_correct["u_execute"] >= 1,
        "induct_correct_at_least_1_of_2": per_kind_correct["u_induct"] >= 1,
        "probe_correct_at_least_1_of_2": per_kind_correct["u_probe"] >= 1,
    }
    return {
        "adapter": label,
        "parsed": parsed,
        "correct": correct,
        "cap_contacts": cap_contacts,
        "route_abstentions": route_abstentions,
        "target_correct": sum(per_kind_correct.values()),
        "per_kind_correct": per_kind_correct,
        "checks": checks,
        "passes": all(checks.values()),
    }


def relative_checks(payload: dict) -> dict[str, bool]:
    candidate_total = correct_count(payload, CANDIDATE)
    candidate_target = correct_count(payload, CANDIDATE, TARGET_KINDS)
    return {
        "beats_parent_total_correct": candidate_total > correct_count(payload, PARENT),
        "beats_replay_total_correct": candidate_total > correct_count(payload, CONTROL),
        "beats_parent_target_correct": candidate_target
        > correct_count(payload, PARENT, TARGET_KINDS),
        "beats_replay_target_correct": candidate_target
        > correct_count(payload, CONTROL, TARGET_KINDS),
    }


def validate_receipt_layout(payload: dict) -> None:
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != ROWS * len(ARMS):
        raise ValueError("local receipt must contain exactly 78 graded rows")
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("local receipt rows must be JSON objects")
    if {row.get("adapter") for row in rows} != set(ARMS):
        raise ValueError("local receipt arm set changed")
    task_sets: dict[str, set[str]] = {}
    kind_maps: dict[str, dict[str, str]] = {}
    for label in ARMS:
        selected = [row for row in rows if row.get("adapter") == label]
        task_ids = [row.get("task_id") for row in selected]
        if (
            len(selected) != ROWS
            or any(not isinstance(task_id, str) or not task_id for task_id in task_ids)
            or len(set(task_ids)) != ROWS
            or any(row.get("kind") not in EXPECTED_KINDS for row in selected)
            or any(type(row.get("correct")) is not bool for row in selected)
            or any(type(row.get("cap_contact")) is not bool for row in selected)
        ):
            raise ValueError(f"local receipt row schema changed for {label}")
        kind_counts = {
            kind: sum(row.get("kind") == kind for row in selected)
            for kind in EXPECTED_KINDS
        }
        if set(kind_counts.values()) != {2}:
            raise ValueError(f"local receipt kind balance changed for {label}")
        task_sets[label] = set(task_ids)
        kind_maps[label] = {row["task_id"]: row["kind"] for row in selected}
    if len({frozenset(value) for value in task_sets.values()}) != 1:
        raise ValueError("local arms do not share the same task ids")
    if any(kind_maps[label] != kind_maps[PARENT] for label in ARMS[1:]):
        raise ValueError("local task-to-kind mapping differs across arms")


def evaluate_promotion(payload: dict) -> dict:
    if payload.get("seed") != SEED or payload.get("rows_per_arm") != ROWS:
        raise ValueError("local receipt seed or row count changed")
    if payload.get("labels") != list(ARMS):
        raise ValueError("local receipt label order changed")
    validate_receipt_layout(payload)
    gates = {label: absolute_gate(payload, label) for label in ARMS}
    relative = relative_checks(payload)
    passes = gates[CANDIDATE]["passes"] and all(relative.values())
    return {
        "schema_version": 1,
        "seed": SEED,
        "candidate": CANDIDATE,
        "controls": [CONTROL, PARENT],
        "target_kinds": sorted(TARGET_KINDS),
        "gates": gates,
        "relative_checks": relative,
        "eligible": [CANDIDATE] if passes else [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        result = evaluate_promotion(
            json.loads(args.receipt.read_text(encoding="utf-8"))
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    rendered = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.out:
        if args.out.exists():
            parser.error("refusing to overwrite local promotion receipt")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["eligible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
