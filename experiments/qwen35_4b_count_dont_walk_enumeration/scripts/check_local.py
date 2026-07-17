#!/usr/bin/env python3
"""Apply the frozen single-kind count-walk local gate with pooled retention.

The candidate (``count_walk``) promotes iff ALL of:

- installability on the 40 axis-holdout rows (gate seed 88056; all
  ``u_count_walk``, 5 per formalism across all eight): candidate total
  correct STRICTLY above the parent's total AND the replay control's
  total (ties fail). This is a SINGLE-KIND dose (the design rule
  hardened by the gym-mix cell: one kind per dose at full
  concentration), so there is NO per-kind split and NO per-kind win
  requirement (per-formalism correctness is reported descriptively,
  never gated);
- retention non-inferiority under the calibrated pooled_k3 protocol,
  read on the POOLED MEAN over THREE fresh 104-row retention screens
  (seeds 88057/88058/88059): candidate pooled correct >= parent pooled
  - 5 AND >= replay pooled - 5; candidate pooled cap contacts <= parent
  pooled + 3 AND <= replay pooled + 3; candidate pooled parsed >=
  parent pooled - 3 AND >= replay pooled - 3. Means, never per-screen;
  with the same number of screens per arm every pooled-mean band is
  evaluated in exact integer arithmetic on the screen SUMS (band x 3
  screens: correct -15, caps +9, parsed -9).

There are NO absolute per-kind floors anywhere. Across-screen delta SDs
are reported descriptively via the calibration cell's pooled_sd
machinery.

PREREGISTERED NON-GATING MECHANISM READING (enumeration fidelity): for
every axis row the evaluator records three booleans about the model's
proposed candidate — (a) LEGAL (a bounded single-step change that
differs from the written step), (b) UNTRIED (not among the already-tried
candidates), (c) CANONICAL-NEXT (exactly the frozen-order target). This
gate summarizes the decomposition per arm; it NEVER feeds the promotion
verdict. The paired analytic reading — the number of turns a PERFECT
canonical enumerator needs per holdout episode — is computed model-free
in the local-gate design receipt.

NEW NON-GATING GATE READING (expression cost — the reading this lineage
now owes after the reference cell's truncation forensics): for every
arm this gate summarizes the per-arm THINK-TOKEN-LENGTH distribution
over the 40 axis rows (min/median/mean/max of ``n_thinking_tokens``)
plus the TRUNCATION COUNT (axis rows at the generation cap). A
count-don't-walk install should show short, k-independent thinking and
zero truncations where the reference cell's walker truncated; recorded
per arm, it NEVER feeds the promotion verdict.

ANSWER NORMALIZATION (frozen grading rule, byte-identical to the
retention-calibration cell's ``normalize_answer`` and applied identically
to every arm and every input file by the evaluator): both the parsed and
the expected answer pass through ``normalize_answer`` before comparison —
collapse runs of whitespace to a single space, strip, then remove any
spaces immediately adjacent to '>' or ';'.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


SEED = 88056
SCREEN_SEEDS = (88057, 88058, 88059)
AGGREGATE_SEED = 78163
AXIS_ROWS = 40
AXIS_KIND_COUNTS = {"u_count_walk": 40}
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
PARENT = "zero_root_parent"
CONTROL = "replay_ctl7"
CANDIDATE = "count_walk"
CANDIDATES = (CANDIDATE,)
ARMS = (PARENT, CONTROL, CANDIDATE)
AXIS_KINDS = frozenset(AXIS_KIND_COUNTS)
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
FIDELITY_FIELDS = ("parseable", "legal", "untried", "canonical_next")
EXPRESSION_COST_FIELDS = (
    "rows_with_readout",
    "think_tokens_min",
    "think_tokens_median",
    "think_tokens_mean",
    "think_tokens_max",
    "truncations",
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


def fidelity_summary(rows: list[dict]) -> dict:
    """The NON-GATING mechanism decomposition: how many axis proposals
    were parseable / legal / untried / canonical-next per arm."""
    present = [row for row in rows if isinstance(row.get("enumeration_fidelity"), dict)]
    summary = {
        "rows_with_readout": len(present),
        **{
            field: sum(
                bool(row["enumeration_fidelity"].get(field)) for row in present
            )
            for field in FIDELITY_FIELDS
        },
        "legal_but_already_tried": sum(
            bool(row["enumeration_fidelity"].get("legal"))
            and not row["enumeration_fidelity"].get("untried")
            for row in present
        ),
        "legal_untried_but_not_next": sum(
            bool(row["enumeration_fidelity"].get("legal"))
            and bool(row["enumeration_fidelity"].get("untried"))
            and not row["enumeration_fidelity"].get("canonical_next")
            for row in present
        ),
        "reported_not_gated": True,
    }
    return summary


def expression_cost_summary(rows: list[dict]) -> dict:
    """The NEW NON-GATING expression-cost reading: per-arm think-token
    length distribution + truncation count over the axis rows."""
    lengths = sorted(row["n_thinking_tokens"] for row in rows)
    count = len(lengths)
    if count % 2 == 0:
        median = (lengths[count // 2 - 1] + lengths[count // 2]) / 2
    else:
        median = lengths[count // 2]
    return {
        "rows_with_readout": count,
        "think_tokens_min": lengths[0],
        "think_tokens_median": median,
        "think_tokens_mean": sum(lengths) / count,
        "think_tokens_max": lengths[-1],
        "truncations": sum(bool(row.get("cap_contact")) for row in rows),
        "reported_not_gated": True,
    }


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
        # Descriptive only: this is a SINGLE-KIND gate — NO per-formalism
        # (surface) gate exists anywhere.
        "per_surface_correct": {
            surface: sum(
                row.get("surface") == surface and bool(row.get("correct"))
                for row in rows
            )
            for surface in AXIS_SURFACES
        },
        # The preregistered NON-GATING mechanism reading.
        "enumeration_fidelity": fidelity_summary(rows),
        # The NEW NON-GATING expression-cost reading (think-token length
        # distribution + truncation count) this lineage now owes.
        "expression_cost": expression_cost_summary(rows),
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
                if kind_counts != AXIS_KIND_COUNTS:
                    raise ValueError(
                        f"local receipt axis kind balance changed for {label} at {screen}"
                    )
                surface_counts = {
                    surface: sum(row.get("surface") == surface for row in selected)
                    for surface in AXIS_SURFACES
                }
                if any(
                    surface_counts[surface] != AXIS_PER_SURFACE
                    for surface in AXIS_SURFACES
                ):
                    raise ValueError(
                        f"local receipt axis surface balance changed for {label} at {screen}"
                    )
                # The fidelity readout is REQUIRED on every axis row (its
                # booleans are recorded either way); it stays non-gating.
                if any(
                    not isinstance(row.get("enumeration_fidelity"), dict)
                    or set(row["enumeration_fidelity"]) != set(FIDELITY_FIELDS)
                    or any(
                        type(row["enumeration_fidelity"][field]) is not bool
                        for field in FIDELITY_FIELDS
                    )
                    for row in selected
                ):
                    raise ValueError(
                        f"axis rows lack the enumeration-fidelity readout for {label}"
                    )
                # The expression-cost reading needs the per-row think-token
                # count on every axis row (recorded either way; non-gating).
                if any(
                    not isinstance(row.get("n_thinking_tokens"), int)
                    or isinstance(row.get("n_thinking_tokens"), bool)
                    or row["n_thinking_tokens"] < 0
                    for row in selected
                ):
                    raise ValueError(
                        f"axis rows lack the think-token-length readout for {label}"
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
        "axis_kind_counts": dict(sorted(AXIS_KIND_COUNTS.items())),
        "axis_surfaces": list(AXIS_SURFACES),
        "axis_rows": AXIS_ROWS,
        "retention_rows_per_screen": RETENTION_ROWS_PER_SCREEN,
        "retention_screens": len(SCREEN_SEEDS),
        "adjudication_protocol": "pooled_k3",
        "summaries": summaries,
        "single_kind_gate": {
            "kind": CANDIDATE,
            "no_per_kind_split_exists": True,
            "ties_fail_the_axis_total": True,
        },
        "per_surface_reported_not_gated": True,
        "mechanism_reading": {
            "enumeration_fidelity_per_arm": {
                label: summaries[label]["axis"]["enumeration_fidelity"]
                for label in ARMS
            },
            "expression_cost_per_arm": {
                label: summaries[label]["axis"]["expression_cost"]
                for label in ARMS
            },
            "expression_cost_reading": (
                "per-arm think-token-length distribution over the 40 axis "
                "rows plus the truncation count — the expression-cost "
                "reading owed after the reference cell's truncation "
                "forensics (20 of 21 unparseable rows were cap truncations "
                "mid-correct-walk); a count-don't-walk install should show "
                "short, k-independent thinking and zero truncations; "
                "recorded per arm, never gated"
            ),
            "decomposition": (
                "legal / untried / canonical-next booleans per axis row: a "
                "mechanism decomposition beyond raw correctness — an arm can "
                "propose legal-but-already-tried candidates (no crossing "
                "off), legal-untried-but-out-of-order candidates (no frozen "
                "order), or the canonical-next candidate (the installed "
                "enumerator)"
            ),
            "reported_not_gated": True,
        },
        "checks": checks,
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
