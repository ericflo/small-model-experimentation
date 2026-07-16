#!/usr/bin/env python3
"""Apply the frozen single-kind feedloop dose-scale local gate with pooled retention.

The candidate (``feedloop_scale``) promotes iff ALL of:

- installability on the 40 axis-holdout rows (gate seed 88037; all
  ``u_feedloop``, 5 per formalism across all eight): candidate total correct
  STRICTLY above the parent's total AND the replay control's total. This is
  a SINGLE-KIND dose, so there is NO per-kind split and NO per-kind win
  requirement (per-formalism correctness is reported descriptively, never
  gated);
- retention non-inferiority under the calibrated pooled_k3 protocol, read on
  the POOLED MEAN over THREE fresh 104-row retention screens (seeds
  88038/88039/88040): candidate pooled correct >= parent pooled - 5 AND >=
  replay pooled - 5; candidate pooled cap contacts <= parent pooled + 3 AND
  <= replay pooled + 3; candidate pooled parsed >= parent pooled - 3 AND >=
  replay pooled - 3. Means, never per-screen; with the same number of
  screens per arm every pooled-mean band is evaluated in exact integer
  arithmetic on the screen SUMS (band x 3 screens: correct -15, caps +9,
  parsed -9).

There are NO absolute per-kind floors anywhere. Across-screen delta SDs are
reported descriptively via the calibration cell's pooled_sd machinery.

PREREGISTERED NON-GATING DOSE-RESPONSE READING: the candidate's axis total
is additionally compared against the reference cell's frozen 80-row
baseline (``feedloop_state`` scored u_feedloop 0/20 on fresh instances at
gate seed 88026; promotion receipt sha d232a1be…). Rendered per formalism.
Any nonzero fresh-instance transfer at 10x is the mechanism answer even
without promotion; a 0 at 10x closes the dose-scale mechanism class for
this skill. Both consequence statements are recorded either way; the
reading never feeds the promotion verdict.

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


SEED = 88037
SCREEN_SEEDS = (88038, 88039, 88040)
AGGREGATE_SEED = 78158
AXIS_ROWS = 40
AXIS_SURFACES = (
    "balesled",
    "barrowyoke",
    "crankwheel",
    "millround",
    "sigilslate",
    "skeinreel",
    "trinketcord",
    "troughline",
)
AXIS_PER_SURFACE = 5
RETENTION_ROWS_PER_SCREEN = 104
RETENTION_PER_KIND = 8
ROWS_PER_ARM = AXIS_ROWS + RETENTION_ROWS_PER_SCREEN * len(SCREEN_SEEDS)
PARENT = "hygiene_explore_parent"
CONTROL = "replay_ctl3"
CANDIDATE = "feedloop_scale"
CANDIDATES = (CANDIDATE,)
ARMS = (PARENT, CONTROL, CANDIDATE)
AXIS_KINDS = frozenset({"u_feedloop"})
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
# The frozen 80-row baseline this cell's dose-response reading is anchored
# to: the reference cell's candidate scored u_feedloop 0/20 on fresh
# instances. The receipt is sha-pinned so a drifted baseline fails loudly in
# the unit tests, never silently.
DOSE_RESPONSE_BASELINE = {
    "cell": "qwen35_4b_feedback_loop_state_chain_install",
    "candidate_arm": "feedloop_state",
    "dose_rows": 80,
    "axis_feedloop_rows": 20,
    "candidate_feedloop_correct": 0,
    "gate_seed": 88026,
    "promotion_receipt": (
        "experiments/qwen35_4b_feedback_loop_state_chain_install"
        "/runs/local/seed88026_promotion.json"
    ),
    "promotion_receipt_sha256": (
        "d232a1be86b53a2f6d295cd735409e13b2343fa0e8403056b183fed14f650fa1"
    ),
}
# DOSE x DIVERSITY CONFOUND (stated in the frozen consequences): formalism
# diversity doubled simultaneously with dose (4 -> 8 formalisms; 20 -> 100
# rows per formalism), so a nonzero reading is NOT a pure dose-response
# isolate — the 10x dose is the dominant delta, but diversity moved with it.
DOSE_RESPONSE_CONSEQUENCES = {
    "if_nonzero": (
        "nonzero fresh-instance u_feedloop transfer at 10x the failed dose "
        "is evidence that SCALE-PLUS-DIVERSITY reopens the family (C43: the "
        "80-row install was data-limited; the 10x dose is the dominant "
        "delta, but formalism diversity doubled 4->8 with it, so this is "
        "not a pure dose-response isolate), even without promotion"
    ),
    "if_zero": (
        "a 0 at 10x the failed dose closes the dose-scale mechanism class "
        "AND the added-diversity variant together for the feedback-loop "
        "skill on this parent"
    ),
}
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
        "both the parsed and the expected answer, every arm, the axis holdout "
        "and all three retention screens"
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
    """Pooled within-arm across-screen SD over equal-df arms.

    sqrt( sum_i ss_i / sum_i df_i ) with df_i = n_i - 1; with every arm at
    the same three screens this equals sqrt(mean per-arm sample variance).
    """
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
    expected = AXIS_ROWS if screen == SEED else RETENTION_ROWS_PER_SCREEN
    if label not in ARMS or screen not in (SEED, *SCREEN_SEEDS) or len(rows) != expected:
        raise ValueError(
            f"local receipt does not contain {expected} rows for {label} at {screen}"
        )
    return rows


def axis_summary(payload: dict, label: str) -> dict:
    rows = selected_rows(payload, label, SEED)
    if any(row.get("kind") not in AXIS_KINDS for row in rows):
        raise ValueError(f"axis instrument kind set changed for {label}")
    if any(row.get("surface") not in AXIS_SURFACES for row in rows):
        raise ValueError(f"axis instrument surface set changed for {label}")
    return {
        "rows": len(rows),
        "correct": sum(bool(row.get("correct")) for row in rows),
        "parsed": sum(row.get("parsed") is not None for row in rows),
        "cap_contacts": sum(bool(row.get("cap_contact")) for row in rows),
        # Descriptive only: this single-kind dose has NO per-formalism gate.
        "per_surface_correct": {
            surface: sum(
                row.get("surface") == surface and bool(row.get("correct"))
                for row in rows
            )
            for surface in AXIS_SURFACES
        },
    }


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
        "axis": axis_summary(payload, label),
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
    if {row.get("screen") for row in rows} != {SEED, *SCREEN_SEEDS}:
        raise ValueError("local receipt screen set changed")
    for screen in (SEED, *SCREEN_SEEDS):
        prefix = f"axis{SEED}_" if screen == SEED else f"ret{screen}_"
        expected_rows = AXIS_ROWS if screen == SEED else RETENTION_ROWS_PER_SCREEN
        expected_kinds = AXIS_KINDS if screen == SEED else RETENTION_KINDS
        task_sets: dict[str, set[str]] = {}
        kind_maps: dict[str, dict[str, str]] = {}
        surface_maps: dict[str, dict[str, str]] = {}
        for label in ARMS:
            selected = [
                row
                for row in rows
                if row.get("adapter") == label and row.get("screen") == screen
            ]
            task_ids = [row.get("task_id") for row in selected]
            if (
                len(selected) != expected_rows
                or any(
                    not isinstance(task_id, str) or not task_id.startswith(prefix)
                    for task_id in task_ids
                )
                or len(set(task_ids)) != expected_rows
                or any(row.get("kind") not in expected_kinds for row in selected)
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
            if screen == SEED:
                kind_counts = {
                    kind: sum(row.get("kind") == kind for row in selected)
                    for kind in expected_kinds
                }
                surface_counts = {
                    surface: sum(row.get("surface") == surface for row in selected)
                    for surface in AXIS_SURFACES
                }
                if (
                    kind_counts != {"u_feedloop": AXIS_ROWS}
                    or any(
                        surface_counts[surface] != AXIS_PER_SURFACE
                        for surface in AXIS_SURFACES
                    )
                    or sum(surface_counts.values()) != AXIS_ROWS
                ):
                    raise ValueError(
                        f"local receipt axis balance changed for {label} at {screen}"
                    )
                surface_maps[label] = {
                    row["task_id"]: row.get("surface") for row in selected
                }
            else:
                kind_counts = {
                    kind: sum(row.get("kind") == kind for row in selected)
                    for kind in expected_kinds
                }
                if any(
                    kind_counts[kind] != RETENTION_PER_KIND for kind in expected_kinds
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
        if screen == SEED and any(
            surface_maps[label] != surface_maps[PARENT] for label in ARMS[1:]
        ):
            raise ValueError(
                f"local task-to-surface mapping differs across arms at {screen}"
            )
    combined: set[str] = set()
    for screen in (SEED, *SCREEN_SEEDS):
        screen_ids = {
            row["task_id"]
            for row in rows
            if row.get("adapter") == PARENT and row.get("screen") == screen
        }
        if combined & screen_ids:
            raise ValueError("instruments share task ids across input files")
        combined |= screen_ids


def pooled_band_checks(summaries: dict[str, dict]) -> dict:
    """Pooled-mean retention bands, evaluated exactly on screen sums.

    With the same number of screens for every arm, ``mean_c >= mean_p - band``
    is exactly ``sum_c >= sum_p - band * n_screens`` — integer arithmetic, no
    float boundary ambiguity at the preregistered edges.
    """
    n_screens = len(SCREEN_SEEDS)
    candidate = summaries[CANDIDATE]["retention"]["sums"]
    checks = {}
    for reference_name, reference in (
        ("parent", summaries[PARENT]["retention"]["sums"]),
        ("replay", summaries[CONTROL]["retention"]["sums"]),
    ):
        checks[f"retention_pooled_correct_within_{RETENTION_CORRECT_BAND}_of_{reference_name}"] = (
            candidate["correct"]
            >= reference["correct"] - RETENTION_CORRECT_BAND * n_screens
        )
        checks[f"retention_pooled_cap_contacts_within_{RETENTION_CAP_BAND}_of_{reference_name}"] = (
            candidate["cap_contacts"]
            <= reference["cap_contacts"] + RETENTION_CAP_BAND * n_screens
        )
        checks[f"retention_pooled_parsed_within_{RETENTION_PARSED_BAND}_of_{reference_name}"] = (
            candidate["parsed"]
            >= reference["parsed"] - RETENTION_PARSED_BAND * n_screens
        )
    return checks


def dose_response_reading(summaries: dict[str, dict]) -> dict:
    """Preregistered NON-GATING 10x dose-response reading vs the 0/20 baseline.

    Never feeds the promotion verdict. Records the candidate's fresh-instance
    u_feedloop axis total (and the per-formalism table) against the reference
    cell's frozen 80-row 0/20 baseline, plus BOTH frozen consequence
    statements and which one applies.
    """
    candidate_axis = summaries[CANDIDATE]["axis"]
    correct = candidate_axis["correct"]
    nonzero = correct > 0
    return {
        "gating": False,
        "baseline": dict(DOSE_RESPONSE_BASELINE),
        "this_cell": {
            "dose_rows": 800,
            "dose_multiple_vs_baseline": 10,
            "axis_feedloop_rows": AXIS_ROWS,
            "candidate_feedloop_correct": correct,
            "per_surface_correct": dict(candidate_axis["per_surface_correct"]),
        },
        "nonzero_transfer_at_10x": nonzero,
        "consequence_statements": dict(DOSE_RESPONSE_CONSEQUENCES),
        "consequence": (
            DOSE_RESPONSE_CONSEQUENCES["if_nonzero"]
            if nonzero
            else DOSE_RESPONSE_CONSEQUENCES["if_zero"]
        ),
    }


def evaluate_promotion(payload: dict) -> dict:
    if (
        payload.get("seed") != SEED
        or payload.get("screen_seeds") != list(SCREEN_SEEDS)
        or payload.get("rows_per_arm") != ROWS_PER_ARM
    ):
        raise ValueError("local receipt seed layout or row count changed")
    if payload.get("labels") != list(ARMS):
        raise ValueError("local receipt label order changed")
    validate_receipt_layout(payload)
    summaries = {label: arm_summary(payload, label) for label in ARMS}
    candidate = summaries[CANDIDATE]
    parent = summaries[PARENT]
    control = summaries[CONTROL]
    checks = {
        "axis_total_strictly_beats_parent": candidate["axis"]["correct"]
        > parent["axis"]["correct"],
        "axis_total_strictly_beats_replay": candidate["axis"]["correct"]
        > control["axis"]["correct"],
        **pooled_band_checks(summaries),
    }
    correct_by_arm = {
        label: summaries[label]["retention"]["correct_by_screen"] for label in ARMS
    }
    deltas = {}
    for label in (CONTROL, CANDIDATE):
        by_screen = [
            correct_by_arm[label][index] - correct_by_arm[PARENT][index]
            for index in range(len(SCREEN_SEEDS))
        ]
        deltas[label] = {
            "vs": PARENT,
            "by_screen": by_screen,
            "pooled_mean": sample_mean(by_screen),
            "sd": sample_sd(by_screen),
        }
    promoted = CANDIDATE if all(checks.values()) else None
    return {
        "schema_version": 1,
        "seed": SEED,
        "screen_seeds": list(SCREEN_SEEDS),
        "candidates": list(CANDIDATES),
        "controls": [CONTROL, PARENT],
        "axis_kinds": sorted(AXIS_KINDS),
        "axis_surfaces": list(AXIS_SURFACES),
        "axis_rows": AXIS_ROWS,
        "retention_rows_per_screen": RETENTION_ROWS_PER_SCREEN,
        "retention_screens": len(SCREEN_SEEDS),
        "adjudication_protocol": "pooled_k3",
        "summaries": summaries,
        "single_kind_dose_no_per_kind_split": True,
        "per_surface_reported_not_gated": True,
        "checks": checks,
        "dose_response_reading": dose_response_reading(summaries),
        "descriptive_noise": {
            "deltas_vs_parent": deltas,
            "screen_sd_pooled_levels": pooled_sd(correct_by_arm),
            "delta_sd_pooled": pooled_sd(
                {label: deltas[label]["by_screen"] for label in deltas}
            ),
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
