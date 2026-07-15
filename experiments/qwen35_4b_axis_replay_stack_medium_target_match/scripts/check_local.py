#!/usr/bin/env python3
"""Apply the frozen two-instrument (axis holdout + retention) local gate.

The candidate promotes iff ALL of:
- installability on the 40 axis-holdout rows: candidate total correct STRICTLY
  above the parent's total AND the replay control's total, and candidate
  strictly beats both controls on at least 3 of the 4 axis kinds (a kind counts
  only when candidate correct > max(parent, replay) for that kind; ties fail);
- retention non-inferiority on the 104 retention rows: correct >= parent - 5
  and >= replay - 5, cap contacts <= parent + 3 and <= replay + 3, parsed >=
  parent - 3 and >= replay - 3;
- sanity: candidate abstains on at most 4 of the 8 feasible retention u_route
  rows.

There are NO absolute per-kind floors anywhere (the predecessor's induct floor
was structurally unpassable).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SEED = 88015
AGGREGATE_SEED = 78145
AXIS_ROWS = 40
RETENTION_ROWS = 104
ROWS = AXIS_ROWS + RETENTION_ROWS
AXIS_PER_KIND = 10
RETENTION_PER_KIND = 8
PARENT = "replay_parent"
CONTROL = "replay_squared"
CANDIDATE = "axis_on_replay"
CANDIDATES = (CANDIDATE,)
ARMS = (PARENT, CONTROL, CANDIDATE)
AXIS_KINDS = frozenset({"u_explore", "u_hygiene", "u_protocol", "u_tracefix"})
AXIS_KIND_WINS_REQUIRED = 3
RETENTION_KINDS = frozenset({
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
})
EXPECTED_KINDS = AXIS_KINDS | RETENTION_KINDS
RETENTION_CORRECT_BAND = 5
RETENTION_CAP_BAND = 3
RETENTION_PARSED_BAND = 3
ROUTE_ABSTENTIONS_MAX = 4
ABSTENTION_ANSWERS = {
    "",
    "ABSTAIN",
    # A trained arm dodging a feasible route task with a budget-exhaustion
    # token is an abstention, not an answer.
    "BUDGET",
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


def arm_summary(payload: dict, label: str) -> dict:
    rows = selected_rows(payload, label)
    axis_rows = [row for row in rows if row.get("kind") in AXIS_KINDS]
    retention_rows = [row for row in rows if row.get("kind") in RETENTION_KINDS]
    if len(axis_rows) != AXIS_ROWS or len(retention_rows) != RETENTION_ROWS:
        raise ValueError(f"instrument row accounting changed for {label}")
    return {
        "adapter": label,
        "axis": {
            "rows": len(axis_rows),
            "correct": sum(bool(row.get("correct")) for row in axis_rows),
            "per_kind_correct": {
                kind: sum(
                    row.get("kind") == kind and bool(row.get("correct"))
                    for row in axis_rows
                )
                for kind in sorted(AXIS_KINDS)
            },
        },
        "retention": {
            "rows": len(retention_rows),
            "correct": sum(bool(row.get("correct")) for row in retention_rows),
            "parsed": sum(row.get("parsed") is not None for row in retention_rows),
            "cap_contacts": sum(bool(row.get("cap_contact")) for row in retention_rows),
            "route_abstentions": sum(
                row.get("kind") == "u_route" and is_abstention(row.get("parsed"))
                for row in retention_rows
            ),
        },
        "total_correct": sum(bool(row.get("correct")) for row in rows),
        "cap_contacts": sum(bool(row.get("cap_contact")) for row in rows),
    }


def validate_receipt_layout(payload: dict) -> None:
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != ROWS * len(ARMS):
        raise ValueError(
            f"local receipt must contain exactly {ROWS * len(ARMS)} graded rows"
        )
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
        if any(kind_counts[kind] != AXIS_PER_KIND for kind in AXIS_KINDS) or any(
            kind_counts[kind] != RETENTION_PER_KIND for kind in RETENTION_KINDS
        ):
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
    summaries = {label: arm_summary(payload, label) for label in ARMS}
    candidate = summaries[CANDIDATE]
    parent = summaries[PARENT]
    control = summaries[CONTROL]
    kind_wins = {
        kind: candidate["axis"]["per_kind_correct"][kind]
        > max(
            parent["axis"]["per_kind_correct"][kind],
            control["axis"]["per_kind_correct"][kind],
        )
        for kind in sorted(AXIS_KINDS)
    }
    axis_kind_wins = sum(kind_wins.values())
    checks = {
        "axis_total_strictly_beats_parent": candidate["axis"]["correct"]
        > parent["axis"]["correct"],
        "axis_total_strictly_beats_replay": candidate["axis"]["correct"]
        > control["axis"]["correct"],
        f"axis_kind_wins_at_least_{AXIS_KIND_WINS_REQUIRED}_of_4": axis_kind_wins
        >= AXIS_KIND_WINS_REQUIRED,
        f"retention_correct_within_{RETENTION_CORRECT_BAND}_of_parent": (
            candidate["retention"]["correct"]
            >= parent["retention"]["correct"] - RETENTION_CORRECT_BAND
        ),
        f"retention_correct_within_{RETENTION_CORRECT_BAND}_of_replay": (
            candidate["retention"]["correct"]
            >= control["retention"]["correct"] - RETENTION_CORRECT_BAND
        ),
        f"retention_cap_contacts_within_{RETENTION_CAP_BAND}_of_parent": (
            candidate["retention"]["cap_contacts"]
            <= parent["retention"]["cap_contacts"] + RETENTION_CAP_BAND
        ),
        f"retention_cap_contacts_within_{RETENTION_CAP_BAND}_of_replay": (
            candidate["retention"]["cap_contacts"]
            <= control["retention"]["cap_contacts"] + RETENTION_CAP_BAND
        ),
        f"retention_parsed_within_{RETENTION_PARSED_BAND}_of_parent": (
            candidate["retention"]["parsed"]
            >= parent["retention"]["parsed"] - RETENTION_PARSED_BAND
        ),
        f"retention_parsed_within_{RETENTION_PARSED_BAND}_of_replay": (
            candidate["retention"]["parsed"]
            >= control["retention"]["parsed"] - RETENTION_PARSED_BAND
        ),
        f"route_abstentions_at_most_{ROUTE_ABSTENTIONS_MAX}_of_8": (
            candidate["retention"]["route_abstentions"] <= ROUTE_ABSTENTIONS_MAX
        ),
    }
    promoted = CANDIDATE if all(checks.values()) else None
    return {
        "schema_version": 1,
        "seed": SEED,
        "candidates": list(CANDIDATES),
        "controls": [CONTROL, PARENT],
        "axis_kinds": sorted(AXIS_KINDS),
        "axis_rows": AXIS_ROWS,
        "retention_rows": RETENTION_ROWS,
        "summaries": summaries,
        "kind_wins": kind_wins,
        "axis_kind_wins": axis_kind_wins,
        "checks": checks,
        "no_absolute_per_kind_floors": True,
        "eligible": [CANDIDATE] if promoted else [],
        "promoted": promoted,
    }


def finalize_promotion(
    result: dict, receipt_path: Path, raw: bytes, design_receipt: Path | None = None
) -> dict:
    """Shared writer fields: keeps the eval-internal and recovery receipts
    schema-identical; run_benchmark.py hard-requires these fields."""
    exp = Path(__file__).resolve().parents[1]
    if design_receipt is None:
        design_receipt = exp / "data" / "local_design_receipt.json"
    finalized = dict(result)
    finalized.update({
        "experiment_id": exp.name,
        "local_receipt": str(receipt_path.resolve()),
        "local_receipt_sha256": hashlib.sha256(raw).hexdigest(),
        "design_receipt_sha256": hashlib.sha256(design_receipt.read_bytes()).hexdigest(),
        "backend": "vllm_merged_composite",
        "aggregate_seed": AGGREGATE_SEED,
        "aggregate_seed_open": result["promoted"] is not None,
        "benchmark_data_read": False,
    })
    return finalized


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        raw = args.receipt.read_bytes()
        result = evaluate_promotion(json.loads(raw.decode("utf-8")))
        result = finalize_promotion(result, args.receipt, raw)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    rendered = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.out:
        if args.out.exists():
            parser.error("refusing to overwrite local promotion receipt")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["promoted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
