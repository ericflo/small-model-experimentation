#!/usr/bin/env python3
"""Apply the frozen two-arm replay-compound retention gate (pooled_k3).

There is NO axis instrument in this cell: the stage-8 treatment is the FULL
replay pool itself (no new kind exists to hold out), so the local gate is a
pure RETENTION NON-DRIFT screen between exactly two arms — the ``count_walk``
parent composite and the ``replay_compound`` candidate — and the aggregate
question (does replay compounding still add aggregate at stage 8?) belongs
exclusively to the sealed benchmark event.

The candidate (``replay_compound``) promotes iff ALL of the frozen TWO-SIDED
pooled_k3 retention bands hold, read on the POOLED MEAN over THREE fresh
104-row retention screens (seeds 88060/88061/88062; 8 rows per each of the
13 original skills):

- pooled correct within +-5 of the parent (means) — with the same number of
  screens per arm every pooled-mean band is evaluated in exact integer
  arithmetic on the screen SUMS: |sum_c - sum_p| <= 15;
- pooled cap contacts within +-3 of the parent (means) — |sums| <= 9;
- pooled parsed within +-3 of the parent (means) — |sums| <= 9.

The bands are deliberately TWO-SIDED: this is a drift screen, not a win
gate — a candidate that moved retention correctness by more than 5 pooled
points in EITHER direction is no longer the same-behavior replay refresh
this stage preregistered, and the sealed benchmark event (not the local
gate) is the only instrument allowed to price aggregate movement. There are
NO absolute per-kind floors anywhere; per-kind counts and across-screen
delta SDs are reported descriptively via the calibration cell's pooled_sd
machinery and never gate.

ANSWER NORMALIZATION (frozen grading rule, byte-identical to the
retention-calibration cell's ``normalize_answer`` and applied identically to
every arm and every input file by the evaluator): both the parsed and the
expected answer pass through ``normalize_answer`` before comparison —
collapse runs of whitespace to a single space, strip, then remove any spaces
immediately adjacent to '>' or ';'.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


SCREEN_SEEDS = (88060, 88061, 88062)
AGGREGATE_SEED = 78168
RETENTION_ROWS_PER_SCREEN = 104
RETENTION_PER_KIND = 8
ROWS_PER_ARM = RETENTION_ROWS_PER_SCREEN * len(SCREEN_SEEDS)
PARENT = "count_walk"
CANDIDATE = "replay_compound"
CANDIDATES = (CANDIDATE,)
ARMS = (PARENT, CANDIDATE)
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
RETENTION_CORRECT_BAND = 5
RETENTION_CAP_BAND = 3
RETENTION_PARSED_BAND = 3
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
    "NOWHERE",
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
        "both the parsed and the expected answer, every arm, all three "
        "retention screens"
    ),
    "rationale": (
        "21 correct-but-rejected whitespace rows across the three prior gate "
        "events (seeds 88014/88015/88016), recorded in experiments/"
        "qwen35_4b_axis_stack_readjudication_medium_pilot/analysis/"
        "three_event_failure_forensics.md; unchanged since the seed-88021 gate"
    ),
    "prospective": True,
}


def normalize_answer(value: str) -> str:
    """Frozen grading normalization; see ANSWER_NORMALIZATION."""
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r"\s*>\s*", ">", value)
    value = re.sub(r"\s*;\s*", ";", value)
    return value


def sample_mean(values: list[int] | list[float]) -> float:
    return sum(values) / len(values)


def sample_sd(values: list[int] | list[float]) -> float:
    """Across-screen sample standard deviation (ddof=1)."""
    if len(values) < 2:
        raise ValueError("sample SD needs at least two screens")
    mean = sample_mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def pooled_sd(values_by_arm: dict[str, list[int]]) -> float:
    """Pooled within-arm across-screen SD over equal-df arms."""
    if not values_by_arm:
        raise ValueError("pooled SD needs at least one arm")
    total_ss = 0.0
    total_df = 0
    for values in values_by_arm.values():
        if len(values) < 2:
            raise ValueError("pooled SD needs at least two screens per arm")
        mean = sample_mean(values)
        total_ss += sum((value - mean) ** 2 for value in values)
        total_df += len(values) - 1
    return math.sqrt(total_ss / total_df)


def is_abstention(value: object) -> bool:
    return value is None or str(value).strip().upper() in ABSTENTION_ANSWERS


def selected_rows(payload: dict, label: str, screen: int) -> list[dict]:
    rows = [
        row
        for row in payload.get("rows", [])
        if row.get("adapter") == label and row.get("screen") == screen
    ]
    if (
        label not in ARMS
        or screen not in SCREEN_SEEDS
        or len(rows) != RETENTION_ROWS_PER_SCREEN
    ):
        raise ValueError(
            f"local receipt does not contain {RETENTION_ROWS_PER_SCREEN} rows "
            f"for {label} at {screen}"
        )
    return rows


def retention_screen_summary(payload: dict, label: str, screen: int) -> dict:
    rows = selected_rows(payload, label, screen)
    if any(row.get("kind") not in RETENTION_KINDS for row in rows):
        raise ValueError(f"retention instrument kind set changed for {label} at {screen}")
    return {
        "rows": len(rows),
        "correct": sum(bool(row.get("correct")) for row in rows),
        "parsed": sum(row.get("parsed") is not None for row in rows),
        "cap_contacts": sum(bool(row.get("cap_contact")) for row in rows),
        "route_abstentions": sum(
            row.get("kind") == "u_route" and is_abstention(row.get("parsed"))
            for row in rows
        ),
        "per_kind_correct": {
            kind: sum(
                row.get("kind") == kind and bool(row.get("correct")) for row in rows
            )
            for kind in sorted(RETENTION_KINDS)
        },
    }


def arm_summary(payload: dict, label: str) -> dict:
    screens = {
        str(screen): retention_screen_summary(payload, label, screen)
        for screen in SCREEN_SEEDS
    }
    sums = {
        field: sum(screens[str(screen)][field] for screen in SCREEN_SEEDS)
        for field in ("correct", "parsed", "cap_contacts", "route_abstentions")
    }
    return {
        "adapter": label,
        "retention": {
            "screens": screens,
            "sums": sums,
            "pooled_means": {
                field: sums[field] / len(SCREEN_SEEDS) for field in sums
            },
            "correct_by_screen": [
                screens[str(screen)]["correct"] for screen in SCREEN_SEEDS
            ],
        },
    }


def validate_receipt_layout(payload: dict) -> None:
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != ROWS_PER_ARM * len(ARMS):
        raise ValueError(
            f"local receipt must contain exactly {ROWS_PER_ARM * len(ARMS)} graded rows"
        )
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("local receipt rows must be JSON objects")
    if {row.get("adapter") for row in rows} != set(ARMS):
        raise ValueError("local receipt arm set changed")
    if {row.get("screen") for row in rows} != set(SCREEN_SEEDS):
        raise ValueError("local receipt screen set changed")
    for screen in SCREEN_SEEDS:
        prefix = f"ret{screen}_"
        task_sets: dict[str, set[str]] = {}
        kind_maps: dict[str, dict[str, str]] = {}
        for label in ARMS:
            selected = [
                row
                for row in rows
                if row.get("adapter") == label and row.get("screen") == screen
            ]
            task_ids = [row.get("task_id") for row in selected]
            if (
                len(selected) != RETENTION_ROWS_PER_SCREEN
                or any(
                    not isinstance(task_id, str) or not task_id.startswith(prefix)
                    for task_id in task_ids
                )
                or len(set(task_ids)) != RETENTION_ROWS_PER_SCREEN
                or any(row.get("kind") not in RETENTION_KINDS for row in selected)
                or any(type(row.get("correct")) is not bool for row in selected)
                or any(type(row.get("cap_contact")) is not bool for row in selected)
                # `parsed` feeds a GATED retention band: every row must carry
                # the key explicitly as None (unparsed) or a string; a missing
                # key or non-string garbage must abort, never default.
                or any("parsed" not in row for row in selected)
                or any(
                    row["parsed"] is not None and not isinstance(row["parsed"], str)
                    for row in selected
                )
            ):
                raise ValueError(
                    f"local receipt row schema changed for {label} at {screen}"
                )
            kind_counts = {
                kind: sum(row.get("kind") == kind for row in selected)
                for kind in RETENTION_KINDS
            }
            if any(
                kind_counts[kind] != RETENTION_PER_KIND for kind in RETENTION_KINDS
            ):
                raise ValueError(
                    f"local receipt kind balance changed for {label} at {screen}"
                )
            task_sets[label] = set(task_ids)
            kind_maps[label] = {row["task_id"]: row["kind"] for row in selected}
        if len({frozenset(value) for value in task_sets.values()}) != 1:
            raise ValueError(f"local arms do not share the same task ids at {screen}")
        if any(kind_maps[label] != kind_maps[PARENT] for label in ARMS[1:]):
            raise ValueError(
                f"local task-to-kind mapping differs across arms at {screen}"
            )
    combined: set[str] = set()
    for screen in SCREEN_SEEDS:
        screen_ids = {
            row["task_id"]
            for row in rows
            if row.get("adapter") == PARENT and row.get("screen") == screen
        }
        if combined & screen_ids:
            raise ValueError("instruments share task ids across input files")
        combined |= screen_ids


def pooled_band_checks(summaries: dict[str, dict]) -> dict:
    """TWO-SIDED pooled-mean retention bands, evaluated exactly on screen sums.

    With the same number of screens for every arm,
    ``|mean_c - mean_p| <= band`` is exactly
    ``|sum_c - sum_p| <= band * n_screens`` — integer arithmetic, no float
    boundary ambiguity at the preregistered edges (correct +-15, cap
    contacts +-9, parsed +-9 on the three-screen sums).
    """
    n_screens = len(SCREEN_SEEDS)
    candidate = summaries[CANDIDATE]["retention"]["sums"]
    parent = summaries[PARENT]["retention"]["sums"]
    return {
        f"retention_pooled_correct_within_{RETENTION_CORRECT_BAND}_of_parent": (
            abs(candidate["correct"] - parent["correct"])
            <= RETENTION_CORRECT_BAND * n_screens
        ),
        f"retention_pooled_cap_contacts_within_{RETENTION_CAP_BAND}_of_parent": (
            abs(candidate["cap_contacts"] - parent["cap_contacts"])
            <= RETENTION_CAP_BAND * n_screens
        ),
        f"retention_pooled_parsed_within_{RETENTION_PARSED_BAND}_of_parent": (
            abs(candidate["parsed"] - parent["parsed"])
            <= RETENTION_PARSED_BAND * n_screens
        ),
    }


def evaluate_promotion(payload: dict) -> dict:
    if (
        payload.get("screen_seeds") != list(SCREEN_SEEDS)
        or payload.get("rows_per_arm") != ROWS_PER_ARM
    ):
        raise ValueError("local receipt seed layout or row count changed")
    if payload.get("labels") != list(ARMS):
        raise ValueError("local receipt label order changed")
    validate_receipt_layout(payload)
    summaries = {label: arm_summary(payload, label) for label in ARMS}
    checks = pooled_band_checks(summaries)
    correct_by_arm = {
        label: summaries[label]["retention"]["correct_by_screen"] for label in ARMS
    }
    by_screen = [
        correct_by_arm[CANDIDATE][index] - correct_by_arm[PARENT][index]
        for index in range(len(SCREEN_SEEDS))
    ]
    deltas = {
        CANDIDATE: {
            "vs": PARENT,
            "by_screen": by_screen,
            "pooled_mean": sample_mean(by_screen),
            "sd": sample_sd(by_screen),
        }
    }
    promoted = CANDIDATE if all(checks.values()) else None
    return {
        "schema_version": 1,
        "screen_seeds": list(SCREEN_SEEDS),
        "candidates": list(CANDIDATES),
        "controls": [PARENT],
        "retention_rows_per_screen": RETENTION_ROWS_PER_SCREEN,
        "retention_screens": len(SCREEN_SEEDS),
        "adjudication_protocol": "pooled_k3",
        "bands_two_sided": True,
        "no_axis_instrument": (
            "the stage-8 treatment is the full replay pool itself; no new "
            "kind exists to hold out, so the local gate is a pure retention "
            "non-drift screen and the aggregate question belongs to the "
            "sealed benchmark event"
        ),
        "summaries": summaries,
        "checks": checks,
        "descriptive_noise": {
            "deltas_vs_parent": deltas,
            "screen_sd_pooled_levels": pooled_sd(correct_by_arm),
            "reported_not_gated": True,
        },
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
