#!/usr/bin/env python3
"""Apply the frozen four-arm dose-diversity MECHANISM readout (no promotion).

This is a mechanism cell: nothing promotes and there is no aggregate seed to
open. The gate evaluates one freshly trained composite (``axis160_direct``,
the 160-row four-kind axis dose trained directly on the clean parent) against
THREE already-published composites re-judged on the same fresh instrument:

- ``clean_parent`` — the designed_fresh parent (warm-start lineage);
- ``hygiene_explore_direct`` — the 80-row two-kind dose trained directly on
  the same clean parent (seed-88018 event measured retention ~= parent - 10);
- ``replay_clean`` — the destack trial's replay-only control composite.

Three preregistered readings, all on the fresh 104-row retention screen:

(a) ``retention_delta_axis160``  = axis160_direct retention correct minus
    clean_parent retention correct;
(b) ``retention_delta_hygexp``   = hygiene_explore_direct retention correct
    minus clean_parent retention correct (the seed-88018 ~-10, re-measured
    fresh on this same screen);
(c) verdict ``diversity_mechanism``:
    - ``SCREEN_FORTUNE_SUSPECT`` if (b) >= -5 (the known ~-10 forgetting does
      not reproduce on this screen, so the screen cannot adjudicate the
      mechanism);
    - otherwise ``SUPPORTED`` if (a) >= -5 (the four-kind 160-row dose
      retains where the two-kind 80-row dose forgets: diversity/volume of the
      dose, not direct-dosing per se, is implicated);
    - otherwise ``REFUTED_INTRINSIC`` (both direct doses forget: forgetting
      is intrinsic to direct dosing on this parent).

The receipt records everything (per-arm axis totals, per-kind counts,
retention correct/parsed/cap-contacts, both deltas, the verdict) and the
process exits 0 on ANY complete event: there is no seed to open and no
promotion to grant or refuse.

ANSWER NORMALIZATION (frozen grading delta, applied identically to every arm
and both instruments by the evaluator): both the parsed and the expected
answer pass through ``normalize_answer`` before comparison — collapse runs of
whitespace to a single space, strip, then remove any spaces immediately
adjacent to '>' or ';'. Rationale: 21 correct-but-rejected whitespace rows
across the three prior gate events (seeds 88014/88015/88016), recorded in the
re-adjudication experiment's analysis/three_event_failure_forensics.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


SEED = 88020
AXIS_ROWS = 40
RETENTION_ROWS = 104
ROWS = AXIS_ROWS + RETENTION_ROWS
AXIS_PER_KIND = 10
RETENTION_PER_KIND = 8
PARENT = "clean_parent"
HYGEXP = "hygiene_explore_direct"
REPLAY = "replay_clean"
CANDIDATE = "axis160_direct"
CANDIDATES = (CANDIDATE,)
ARMS = (PARENT, HYGEXP, REPLAY, CANDIDATE)
AXIS_KINDS = frozenset({"u_explore", "u_hygiene", "u_protocol", "u_tracefix"})
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
# Integer retention-delta thresholds (candidate-or-hygexp minus clean_parent):
# >= RETAINED_AT_LEAST reads as "retention held"; <= FORGOT_AT_MOST reads as
# "forgetting reproduced". The two bands partition the integers exactly.
RETAINED_AT_LEAST = -5
FORGOT_AT_MOST = -6
VERDICTS = ("SUPPORTED", "REFUTED_INTRINSIC", "SCREEN_FORTUNE_SUSPECT")
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
ANSWER_NORMALIZATION = {
    "function": "check_local.normalize_answer",
    "definition": (
        "re.sub(r'\\s+', ' ', s).strip(), then re.sub(r'\\s*>\\s*', '>', s), "
        "then re.sub(r'\\s*;\\s*', ';', s)"
    ),
    "applied_to": (
        "both the parsed and the expected answer, every arm, both instruments"
    ),
    "rationale": (
        "21 correct-but-rejected whitespace rows across the three prior gate "
        "events (seeds 88014/88015/88016), recorded in experiments/"
        "qwen35_4b_axis_stack_readjudication_medium_pilot/analysis/"
        "three_event_failure_forensics.md"
    ),
    "prospective": True,
}


def normalize_answer(value: str) -> str:
    """Frozen grading normalization; see ANSWER_NORMALIZATION."""
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r"\s*>\s*", ">", value)
    value = re.sub(r"\s*;\s*", ";", value)
    return value


def diversity_mechanism_verdict(delta_axis160: int, delta_hygexp: int) -> str:
    """The frozen three-way verdict over the two integer retention deltas."""
    if delta_hygexp >= RETAINED_AT_LEAST:
        return "SCREEN_FORTUNE_SUSPECT"
    if delta_axis160 >= RETAINED_AT_LEAST:
        return "SUPPORTED"
    return "REFUTED_INTRINSIC"


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


def evaluate_mechanism(payload: dict) -> dict:
    if payload.get("seed") != SEED or payload.get("rows_per_arm") != ROWS:
        raise ValueError("local receipt seed or row count changed")
    if payload.get("labels") != list(ARMS):
        raise ValueError("local receipt label order changed")
    validate_receipt_layout(payload)
    summaries = {label: arm_summary(payload, label) for label in ARMS}
    parent_retention = summaries[PARENT]["retention"]["correct"]
    retention_delta_axis160 = (
        summaries[CANDIDATE]["retention"]["correct"] - parent_retention
    )
    retention_delta_hygexp = (
        summaries[HYGEXP]["retention"]["correct"] - parent_retention
    )
    retention_delta_replay_clean = (
        summaries[REPLAY]["retention"]["correct"] - parent_retention
    )
    verdict = diversity_mechanism_verdict(
        retention_delta_axis160, retention_delta_hygexp
    )
    if verdict not in VERDICTS:
        raise ValueError(f"verdict left the frozen space: {verdict}")
    return {
        "schema_version": 1,
        "seed": SEED,
        "mechanism_cell": True,
        "candidates": list(CANDIDATES),
        "controls": [PARENT, HYGEXP, REPLAY],
        "axis_kinds": sorted(AXIS_KINDS),
        "axis_rows": AXIS_ROWS,
        "retention_rows": RETENTION_ROWS,
        "summaries": summaries,
        "answer_normalization": ANSWER_NORMALIZATION,
        "readings": {
            "retention_delta_axis160": retention_delta_axis160,
            "retention_delta_hygexp": retention_delta_hygexp,
            "retention_delta_replay_clean": retention_delta_replay_clean,
            "diversity_mechanism": verdict,
        },
        "thresholds": {
            "retained_at_least": RETAINED_AT_LEAST,
            "forgot_at_most": FORGOT_AT_MOST,
            "rule": (
                "SCREEN_FORTUNE_SUSPECT if retention_delta_hygexp >= -5; else "
                "SUPPORTED if retention_delta_axis160 >= -5; else "
                "REFUTED_INTRINSIC"
            ),
        },
        "diversity_mechanism": verdict,
        "no_promotion_in_mechanism_cell": True,
        "outcome": "MECHANISM_READ_COMPLETE",
        "eligible": [],
        "promoted": None,
    }


def finalize_mechanism(
    result: dict, receipt_path: Path, raw: bytes, design_receipt: Path | None = None
) -> dict:
    """Shared writer fields: keeps the eval-internal and recovery receipts
    schema-identical; the smoke harness hard-requires these fields."""
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
        "aggregate_seed": None,
        "aggregate_seed_open": False,
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
        result = evaluate_mechanism(json.loads(raw.decode("utf-8")))
        result = finalize_mechanism(result, args.receipt, raw)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    rendered = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.out:
        if args.out.exists():
            parser.error("refusing to overwrite local mechanism receipt")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    # A complete mechanism event always exits 0: there is no seed to open and
    # no promotion to grant; the verdict lives in the receipt.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
