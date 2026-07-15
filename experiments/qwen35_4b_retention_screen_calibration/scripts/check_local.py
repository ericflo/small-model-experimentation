#!/usr/bin/env python3
"""Apply the frozen five-arm four-screen CALIBRATION readout (no promotion).

This is a pure instrument-calibration cell: nothing trains, nothing merges,
nothing promotes, and there is no aggregate seed to open. Five already
published composites are re-measured on FOUR fresh retention-only screens
(seeds 88022/88023/88024/88025, 104 rows each, 8 per original skill):

- ``clean_parent`` — the designed_fresh parent composite;
- ``replay_clean`` — the de-stack trial's replay-only composite;
- ``hygiene_explore_direct`` — the de-stack trial's hygiene+explore composite;
- ``axis160_direct`` — the dose-diversity cell's rank-32 arm;
- ``axis160_r64`` — the rank-capacity cell's fresh rank-64 arm.

Preregistered outputs, all from the per-arm per-screen retention-correct
table (across-screen statistics use the sample SD, ddof=1, n=4 screens):

(a) ``delta_sd_pooled`` — THE GOVERNING ESTIMAND: the pooled across-screen
    SD of the per-screen retention-correct delta versus ``clean_parent``,
    pooled over the four non-parent arms (df=3 each): sqrt(mean over the
    four arms of the per-arm across-screen sample variance of its delta
    series). Every retention band this program adjudicates is a same-screen
    delta versus a parent or control, so the band must be calibrated on the
    delta noise process: common screen-difficulty variance inflates level
    SD but cancels exactly in same-screen deltas, while independent
    per-arm noise makes delta SD ~ sqrt(2) x level SD (adversarial design
    review finding, corrected before freeze);
(b) ``recommended_band`` — ceil(2 * delta_sd_pooled) with a minimum of 5;
(c) ``adjudication_protocol`` (ordered total partition of the reals over
    delta_sd_pooled):
    - ``single_screen`` if delta_sd_pooled <= 2;
    - ``pooled_k2``      if 2 < delta_sd_pooled <= 3.5;
    - ``pooled_k3``      otherwise;
(d) ``screen_sd_pooled`` — reported DESCRIPTIVELY, governs nothing: the
    pooled within-arm across-screen SD of retention-correct LEVELS over
    all five arms;
(e) stability flags — whether each frozen historical single-screen delta
    reading (axis160_direct -9 at 88020; hygiene_explore_direct -10 at 88018
    and -10 at 88020; replay_clean -5 at 88020; axis160_r64 -7 at 88021)
    falls inside its arm's pooled interval from this study: the arm's mean
    delta-vs-clean_parent across the four screens +/- 2 x the across-screen
    sample SD of that delta;
(f) the paused vehicle reading, resumed DESCRIPTIVELY (reported, not gated):
    axis160_r64's pooled delta versus axis160_direct's, with the same
    measured-noise intervals.

The receipt records everything (per-arm per-screen correct/parsed/cap
contacts, per-arm across-screen mean/SD, per-arm per-screen deltas versus
clean_parent with pooled mean and SD, the four preregistered outputs) and
the process exits 0 on ANY complete event: there is no seed to open and no
promotion to grant or refuse.

ANSWER NORMALIZATION (frozen grading rule, unchanged from the seed-88021
gate and applied identically to every arm and every screen by the
evaluator): both the parsed and the expected answer pass through
``normalize_answer`` before comparison — collapse runs of whitespace to a
single space, strip, then remove any spaces immediately adjacent to '>' or
';'.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


SEEDS = (88022, 88023, 88024, 88025)
ROWS = 104
RETENTION_PER_KIND = 8
PARENT = "clean_parent"
# Frozen run-order alphabet: the five published composites, alphabetical.
ARMS = (
    "axis160_direct",
    "axis160_r64",
    "clean_parent",
    "hygiene_explore_direct",
    "replay_clean",
)
DELTA_ARMS = tuple(label for label in ARMS if label != PARENT)
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
# Frozen protocol thresholds over screen_sd_pooled (a total ordered
# partition of the reals) and the band floor.
SINGLE_SCREEN_MAX_SD = 2.0
POOLED_K2_MAX_SD = 3.5
BAND_MINIMUM = 5
PROTOCOLS = ("single_screen", "pooled_k2", "pooled_k3")
# Frozen historical single-screen retention-delta readings (arm, gate seed,
# delta vs clean_parent), copied from the committed predecessor receipts:
# the de-stack 88018 event, the dose-diversity 88020 event, and the
# rank-capacity 88021 event.
HISTORICAL_READINGS = (
    ("axis160_direct", 88020, -9),
    ("axis160_r64", 88021, -7),
    ("hygiene_explore_direct", 88018, -10),
    ("hygiene_explore_direct", 88020, -10),
    ("replay_clean", 88020, -5),
)
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
        "both the parsed and the expected answer, every arm, all four screens"
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
    the same four screens this equals sqrt(mean per-arm sample variance).
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


def recommended_band(screen_sd: float) -> int:
    """The frozen band formula: ceil(2 * SD), floored at BAND_MINIMUM."""
    if screen_sd < 0 or not math.isfinite(screen_sd):
        raise ValueError(f"screen SD out of range: {screen_sd}")
    return max(BAND_MINIMUM, math.ceil(2 * screen_sd))


def adjudication_protocol(screen_sd: float) -> str:
    """The frozen ordered three-way protocol over screen_sd_pooled."""
    if screen_sd < 0 or not math.isfinite(screen_sd):
        raise ValueError(f"screen SD out of range: {screen_sd}")
    if screen_sd <= SINGLE_SCREEN_MAX_SD:
        return "single_screen"
    if screen_sd <= POOLED_K2_MAX_SD:
        return "pooled_k2"
    return "pooled_k3"


def is_abstention(value: object) -> bool:
    return value is None or str(value).strip().upper() in ABSTENTION_ANSWERS


def selected_rows(payload: dict, label: str, seed: int) -> list[dict]:
    rows = [
        row
        for row in payload.get("rows", [])
        if row.get("adapter") == label and row.get("screen") == seed
    ]
    if label not in ARMS or seed not in SEEDS or len(rows) != ROWS:
        raise ValueError(
            f"local receipt does not contain {ROWS} rows for {label} at {seed}"
        )
    return rows


def screen_summary(payload: dict, label: str, seed: int) -> dict:
    rows = selected_rows(payload, label, seed)
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


def validate_receipt_layout(payload: dict) -> None:
    rows = payload.get("rows")
    expected_total = ROWS * len(ARMS) * len(SEEDS)
    if not isinstance(rows, list) or len(rows) != expected_total:
        raise ValueError(
            f"local receipt must contain exactly {expected_total} graded rows"
        )
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("local receipt rows must be JSON objects")
    if {row.get("adapter") for row in rows} != set(ARMS):
        raise ValueError("local receipt arm set changed")
    if {row.get("screen") for row in rows} != set(SEEDS):
        raise ValueError("local receipt screen set changed")
    task_sets: dict[int, dict[str, set[str]]] = {}
    kind_maps: dict[int, dict[str, dict[str, str]]] = {}
    for seed in SEEDS:
        task_sets[seed] = {}
        kind_maps[seed] = {}
        for label in ARMS:
            selected = [
                row
                for row in rows
                if row.get("adapter") == label and row.get("screen") == seed
            ]
            task_ids = [row.get("task_id") for row in selected]
            if (
                len(selected) != ROWS
                or any(
                    not isinstance(task_id, str)
                    or not task_id.startswith(f"ret{seed}_")
                    for task_id in task_ids
                )
                or len(set(task_ids)) != ROWS
                or any(row.get("kind") not in RETENTION_KINDS for row in selected)
                or any(type(row.get("correct")) is not bool for row in selected)
                or any(type(row.get("cap_contact")) is not bool for row in selected)
            ):
                raise ValueError(
                    f"local receipt row schema changed for {label} at {seed}"
                )
            kind_counts = {
                kind: sum(row.get("kind") == kind for row in selected)
                for kind in RETENTION_KINDS
            }
            if any(
                kind_counts[kind] != RETENTION_PER_KIND for kind in RETENTION_KINDS
            ):
                raise ValueError(
                    f"local receipt kind balance changed for {label} at {seed}"
                )
            task_sets[seed][label] = set(task_ids)
            kind_maps[seed][label] = {row["task_id"]: row["kind"] for row in selected}
        if len({frozenset(value) for value in task_sets[seed].values()}) != 1:
            raise ValueError(f"local arms do not share the same task ids at {seed}")
        if any(
            kind_maps[seed][label] != kind_maps[seed][PARENT] for label in ARMS
        ):
            raise ValueError(f"local task-to-kind mapping differs across arms at {seed}")
    combined: set[str] = set()
    for seed in SEEDS:
        screen_ids = task_sets[seed][PARENT]
        if combined & screen_ids:
            raise ValueError("screens share task ids across seeds")
        combined |= screen_ids


def evaluate_calibration(payload: dict) -> dict:
    if (
        payload.get("seeds") != list(SEEDS)
        or payload.get("rows_per_arm_per_screen") != ROWS
    ):
        raise ValueError("local receipt seeds or row count changed")
    if payload.get("labels") != list(ARMS):
        raise ValueError("local receipt label order changed")
    validate_receipt_layout(payload)
    summaries = {
        label: {str(seed): screen_summary(payload, label, seed) for seed in SEEDS}
        for label in ARMS
    }
    correct_by_arm = {
        label: [summaries[label][str(seed)]["correct"] for seed in SEEDS]
        for label in ARMS
    }
    across_screens = {
        label: {
            "correct_by_screen": correct_by_arm[label],
            "mean_correct": sample_mean(correct_by_arm[label]),
            "sd_correct": sample_sd(correct_by_arm[label]),
        }
        for label in ARMS
    }
    deltas = {}
    for label in DELTA_ARMS:
        by_screen = [
            correct_by_arm[label][index] - correct_by_arm[PARENT][index]
            for index in range(len(SEEDS))
        ]
        deltas[label] = {
            "vs": PARENT,
            "by_screen": by_screen,
            "pooled_mean": sample_mean(by_screen),
            "sd": sample_sd(by_screen),
        }
    screen_sd_pooled = pooled_sd(correct_by_arm)
    delta_sd_pooled = pooled_sd(
        {label: deltas[label]["by_screen"] for label in DELTA_ARMS}
    )
    band = recommended_band(delta_sd_pooled)
    protocol = adjudication_protocol(delta_sd_pooled)
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol left the frozen space: {protocol}")
    stability_flags = []
    for arm, gate_seed, historical in HISTORICAL_READINGS:
        interval_low = deltas[arm]["pooled_mean"] - 2 * deltas[arm]["sd"]
        interval_high = deltas[arm]["pooled_mean"] + 2 * deltas[arm]["sd"]
        stability_flags.append(
            {
                "arm": arm,
                "gate_seed": gate_seed,
                "historical_delta": historical,
                "interval_low": interval_low,
                "interval_high": interval_high,
                "inside": interval_low <= historical <= interval_high,
            }
        )
    vehicle = {
        "reported_not_gated": True,
        "r64_pooled_delta": deltas["axis160_r64"]["pooled_mean"],
        "r64_interval": [
            deltas["axis160_r64"]["pooled_mean"] - 2 * deltas["axis160_r64"]["sd"],
            deltas["axis160_r64"]["pooled_mean"] + 2 * deltas["axis160_r64"]["sd"],
        ],
        "r32_pooled_delta": deltas["axis160_direct"]["pooled_mean"],
        "r32_interval": [
            deltas["axis160_direct"]["pooled_mean"]
            - 2 * deltas["axis160_direct"]["sd"],
            deltas["axis160_direct"]["pooled_mean"]
            + 2 * deltas["axis160_direct"]["sd"],
        ],
        "r64_minus_r32": (
            deltas["axis160_r64"]["pooled_mean"]
            - deltas["axis160_direct"]["pooled_mean"]
        ),
    }
    return {
        "schema_version": 1,
        "seeds": list(SEEDS),
        "calibration_cell": True,
        "candidates": [],
        "controls": list(ARMS),
        "parent": PARENT,
        "retention_rows_per_screen": ROWS,
        "summaries": summaries,
        "across_screens": across_screens,
        "deltas_vs_clean_parent": deltas,
        "answer_normalization": ANSWER_NORMALIZATION,
        "readings": {
            "delta_sd_pooled": delta_sd_pooled,
            "band_and_protocol_basis": "delta_sd_pooled",
            "screen_sd_pooled": screen_sd_pooled,
            "recommended_band": band,
            "adjudication_protocol": protocol,
            "stability_flags": stability_flags,
            "vehicle_descriptive": vehicle,
        },
        "thresholds": {
            "single_screen_max_sd": SINGLE_SCREEN_MAX_SD,
            "pooled_k2_max_sd": POOLED_K2_MAX_SD,
            "band_minimum": BAND_MINIMUM,
            "rule": (
                "delta_sd_pooled = sqrt(mean per-arm across-screen sample "
                "variance, ddof=1, of the per-screen delta vs clean_parent "
                "over the four non-parent arms); recommended_band = max(5, "
                "ceil(2 * delta_sd_pooled)); single_screen if "
                "delta_sd_pooled <= 2; else pooled_k2 if <= 3.5; else "
                "pooled_k3; screen_sd_pooled (levels, all five arms) is "
                "reported descriptively and governs nothing"
            ),
            "stability_rule": (
                "a historical reading is stable when it lies inside the arm's "
                "pooled delta vs clean_parent +/- 2 x across-screen sample SD "
                "of that delta"
            ),
        },
        "adjudication_protocol": protocol,
        "no_promotion_in_calibration_cell": True,
        "outcome": "CALIBRATION_READ_COMPLETE",
        "eligible": [],
        "promoted": None,
    }


def finalize_calibration(
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
        result = evaluate_calibration(json.loads(raw.decode("utf-8")))
        result = finalize_calibration(result, args.receipt, raw)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    rendered = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.out:
        if args.out.exists():
            parser.error("refusing to overwrite local calibration receipt")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    # A complete calibration event always exits 0: there is no seed to open
    # and no promotion to grant; the outputs live in the receipt.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
