#!/usr/bin/env python3
"""Compute the four preregistered budget-probe readings from the gateway receipts.

This is a MEASUREMENT read, not a gate: nothing passes or fails the
experiment here, nothing is promoted, and the process exits 0 on any
complete readout. The four frozen readings, computed only from the four
authenticated aggregate-gateway receipts of the one sealed tb4096 event:

1. budget_movement: per (arm, family) over ``menders`` and ``rites``,
   whether the family moved from EXACTLY ZERO at the pinned tb1024
   seed-78150 event to above zero at tb4096 — the goal-gate ceiling
   question (are 9 or 10 families reachable?) — plus the full per-family
   table. Premise from the pinned contrast source: menders was 0.0 for
   ALL FOUR arms at tb1024; rites was 0.0 for base, replay_repeat, and
   hygiene_explore, and 0.1 for designed_fresh. Pairs already nonzero at
   tb1024 (designed_fresh's rites) never fire the movement booleans and
   are reported separately as ``already_nonzero_at_tb1024``, so a
   status-quo repeat cannot answer the budget question.
2. budget_contrast: per arm per family, the delta versus the committed
   tb1024 event at seed 78150 (summary sha256-pinned; fail closed if it
   changed). The contrast additionally FAILS CLOSED unless the two
   events share the same benchmark implementation (runner sha256, source
   inventory sha256, file count) — a preregistered-contrast integrity
   condition, not a soft flag; both signatures are surfaced in the
   block. The block is labeled ``cross_seed_confound: true`` — the
   remaining confounds are seed AND budget, so it is a movement reading,
   never a causal isolation.
3. goal_gate: strict wins / ties / losses versus base across the ten
   public families per treated arm; goal_gate_pass = ten strict wins.
   Recorded, never gated on.
4. budget_integrity: per arm, the gateway receipt's ``within_budget`` flag
   and ``wall_seconds``. If ANY arm has within_budget false the readout
   sets ``paired_comparison_valid: false`` with the reason; scores are
   still recorded either way.

The FAMILIES tuple and the strict-win logic are reused byte-for-byte from
the tier forensics' analyze_constants.py so the goal-gate reading is the
same statistic the venue-moving evidence was computed with. The benchmark
suite directory is never read; only gateway receipts are consumed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]

FROZEN_NAME = "measurement"
FROZEN_TIER = "medium"
FROZEN_THINK_BUDGET = 4096
FROZEN_SEED = 78153
MODEL_ORDER = ("base", "designed_fresh", "replay_repeat", "hygiene_explore")
TREATED_ARMS = ("designed_fresh", "replay_repeat", "hygiene_explore")
FROZEN_MODEL_PATHS = {
    "base": (
        ROOT / "large_artifacts" / "qwen35_4b_universal_curriculum"
        / "merged" / "base_reserialized"
    ),
    "designed_fresh": (
        ROOT / "large_artifacts"
        / "qwen35_4b_universal_fresh_surface_budget_commit_target_match"
        / "merged" / "designed_fresh"
    ),
    "replay_repeat": (
        ROOT / "large_artifacts" / "qwen35_4b_goal_gap_axis_curriculum_target_match"
        / "merged" / "replay_repeat"
    ),
    "hygiene_explore": (
        ROOT / "large_artifacts" / "qwen35_4b_hygiene_explore_destack_medium"
        / "merged" / "hygiene_explore"
    ),
}
EVENT_DIR = (
    EXP / "runs" / "benchmark"
    / f"{FROZEN_TIER}_tb{FROZEN_THINK_BUDGET}_seed{FROZEN_SEED}_{FROZEN_NAME}"
)
READOUT = EVENT_DIR / "measurement_readout.json"
DESIGN_RECEIPT = EXP / "data" / "design_receipt.json"
GATEWAY_KEYS = {
    "schema_version", "stage", "tier", "think_budget", "seed", "backend", "model",
    "model_merge_receipt_sha256", "benchmark_runner_sha256",
    "benchmark_source_inventory_sha256", "benchmark_source_file_count",
    "aggregate", "per_family", "within_budget", "wall_seconds",
}
# Byte-for-byte the tier forensics' FAMILIES tuple
# (experiments/qwen35_4b_menders_sirens_tier_forensics/scripts/analyze_constants.py).
FAMILIES = (
    "chronicle",
    "lockpick",
    "menders",
    "mirage",
    "rites",
    "siftstack",
    "sirens",
    "stockade",
    "toolsmith",
    "warren",
)
# The probe families. Premise from the pinned tb1024 seed-78150 event:
# menders was 0.0 for all four arms; rites was 0.0 for every arm EXCEPT
# designed_fresh (0.1). The movement booleans are therefore scoped to the
# zero-at-tb1024 (arm, family) pairs — a status-quo repeat of an already
# nonzero score can never fire them.
BUDGET_FAMILIES = ("menders", "rites")
IMPLEMENTATION_KEYS = ("runner_sha256", "source_inventory_sha256", "source_file_count")
# The committed tb1024 medium event at seed 78150 (a different, already
# consumed seed) this probe contrasts against. Sha-pinned; fail closed.
TB1024_REFERENCE = {
    "seed": 78150,
    "tier": "medium",
    "think_budget": 1024,
    "summary": (
        "experiments/qwen35_4b_universal_medium_tier_measurement"
        "/runs/benchmark/medium_tb1024_seed78150_measurement/summary.json"
    ),
    "summary_sha256": (
        "a927fc838ca8b1eaa3083d6034ba09ad0659c21a2a13b22c525487cf95a6fb43"
    ),
    "cross_seed_confound": True,
}
CROSS_SEED_NOTE = (
    "valid as a preregistered contrast only because benchmark-"
    "implementation equality (runner sha256, source inventory sha256, "
    "file count) was verified fail-closed at read time; remaining "
    "confounds: seed AND budget — a movement reading, "
    "not a causal isolation"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_score(value: object) -> bool:
    """A gateway score must be a finite float in [0, 1]; NaN never passes.

    NaN compares unequal to everything, so an unguarded NaN would silently
    drop its family from every strict-win partition. Fail loudly instead.
    """
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0.0 <= value <= 1.0
    )


def _valid_wall_seconds(value: object) -> bool:
    """Wall time must be a finite non-negative number (recorded, never gated)."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0.0
    )


def load_event(path: Path, model: Path) -> dict:
    """Authenticate one aggregate-gateway receipt against the frozen event.

    ``within_budget`` must be a strict bool but is RECORDED, never required
    to be true: the budget_integrity reading scopes the paired comparison
    (paired_comparison_valid) instead of rejecting an over-budget arm.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        set(payload) != GATEWAY_KEYS
        or payload.get("schema_version") != 1
        or payload.get("stage") != "menagerie_aggregate_gateway"
        or payload.get("tier") != FROZEN_TIER
        or payload.get("think_budget") != FROZEN_THINK_BUDGET
        or payload.get("seed") != FROZEN_SEED
        or payload.get("backend") != "qwen_vllm"
        or Path(payload.get("model", "")).resolve() != model.resolve()
        or not isinstance(payload.get("within_budget"), bool)
        or not _valid_wall_seconds(payload.get("wall_seconds"))
        or set(payload.get("per_family", {})) != set(FAMILIES)
        or not _valid_score(payload.get("aggregate"))
        or any(
            not _valid_score(value)
            for value in payload.get("per_family", {}).values()
        )
        or payload.get("model_merge_receipt_sha256")
        != sha256_file(model / "merge_receipt.json")
    ):
        raise ValueError(f"aggregate gateway event failed authentication: {path}")
    return payload


def _valid_implementation(block: object) -> bool:
    """The benchmark_implementation signature: two sha256 hexes and a count."""
    return (
        isinstance(block, dict)
        and set(block) == set(IMPLEMENTATION_KEYS)
        and all(
            isinstance(block[key], str)
            and re.fullmatch(r"[0-9a-f]{64}", block[key]) is not None
            for key in ("runner_sha256", "source_inventory_sha256")
        )
        and isinstance(block["source_file_count"], int)
        and not isinstance(block["source_file_count"], bool)
        and block["source_file_count"] > 0
    )


def load_tb1024_reference() -> dict:
    """Load the sha-pinned tb1024 contrast source; fail closed on any drift.

    Returns both the per-arm scores and the reference event's
    benchmark_implementation signature: the budget_contrast reading is
    only preregistered-valid if the tb4096 receipts share that signature.
    """
    summary = ROOT / TB1024_REFERENCE["summary"]
    if not summary.is_file() or sha256_file(summary) != TB1024_REFERENCE["summary_sha256"]:
        raise ValueError(
            f"pinned tb1024 contrast-source summary is absent or changed: {summary}"
        )
    payload = json.loads(summary.read_text(encoding="utf-8"))
    scores = payload.get("scores", {})
    implementation = payload.get("benchmark_implementation")
    if (
        payload.get("seed") != TB1024_REFERENCE["seed"]
        or payload.get("tier") != TB1024_REFERENCE["tier"]
        or payload.get("think_budget") != TB1024_REFERENCE["think_budget"]
        or payload.get("model_order") != list(MODEL_ORDER)
        or set(scores) != set(MODEL_ORDER)
        or not _valid_implementation(implementation)
    ):
        raise ValueError("tb1024 contrast source is not the frozen seed-78150 event")
    for label in MODEL_ORDER:
        row = scores[label]
        if (
            set(row.get("per_family", {})) != set(FAMILIES)
            or not _valid_score(row.get("aggregate"))
            or any(not _valid_score(value) for value in row["per_family"].values())
        ):
            raise ValueError(f"tb1024 contrast source violates the score shape: {label}")
    return {
        "scores": {
            label: {
                "aggregate": scores[label]["aggregate"],
                "per_family": scores[label]["per_family"],
            }
            for label in MODEL_ORDER
        },
        "benchmark_implementation": dict(implementation),
    }


def require_implementation_equality(
    implementation: dict, reference_implementation: dict
) -> None:
    """MAJOR-2 contract: the cross-budget contrast aborts on implementation drift.

    The confound label enumerates seed and budget ONLY because everything
    else is held fixed; if the benchmark implementation (runner, source
    inventory, file count) differs between the two events the preregistered
    contrast is invalid and the readout must not be written.
    """
    if implementation != reference_implementation:
        raise ValueError(
            "benchmark implementation differs from the pinned tb1024 contrast "
            f"source (tb4096={implementation}, tb1024={reference_implementation}); "
            "the preregistered budget contrast is invalid"
        )


def budget_movement(
    per_family_by_label: dict[str, dict[str, float]],
    reference_per_family_by_label: dict[str, dict[str, float]],
) -> dict:
    """Reading 1: did menders or rites move from zero at tb1024 to > 0 at tb4096.

    ``moved`` fires for an (arm, family) pair only when the arm's pinned
    tb1024 value for that family is exactly zero AND its tb4096 value is
    above zero. Pairs already nonzero at tb1024 (per the pinned contrast
    source: designed_fresh's rites, 0.1) are excluded from every boolean
    and reported descriptively in ``already_nonzero_at_tb1024`` — a
    status-quo repeat can never fire ``any_arm_moved``.
    """
    per_arm = {}
    already_nonzero = []
    for label in MODEL_ORDER:
        values = per_family_by_label[label]
        reference = reference_per_family_by_label[label]
        moved = {}
        for family in BUDGET_FAMILIES:
            eligible = reference[family] == 0.0
            moved[family] = eligible and values[family] > 0.0
            if not eligible:
                already_nonzero.append(
                    {
                        "arm": label,
                        "family": family,
                        "tb1024": reference[family],
                        "tb4096": values[family],
                    }
                )
        per_arm[label] = {
            "menders": values["menders"],
            "rites": values["rites"],
            "menders_tb1024": reference["menders"],
            "rites_tb1024": reference["rites"],
            "menders_moved": moved["menders"],
            "rites_moved": moved["rites"],
            "either_moved": any(moved.values()),
        }
    return {
        "question": (
            "at medium/tb4096 does menders or rites move from zero at "
            "tb1024/seed-78150 to above zero for any arm (goal-gate "
            "ceiling: 9 vs 10 reachable families)"
        ),
        "premise": (
            "at medium/tb1024 seed 78150, menders was 0.0 for all four "
            "arms; rites was 0.0 for base, replay_repeat, and "
            "hygiene_explore, and 0.1 for designed_fresh"
        ),
        "movement_rule": (
            "moved(arm, family) := tb1024[arm][family] == 0 and "
            "tb4096[arm][family] > 0; pairs already nonzero at tb1024 "
            "never fire the booleans and are listed in "
            "already_nonzero_at_tb1024"
        ),
        "families_probed": list(BUDGET_FAMILIES),
        "per_arm": per_arm,
        "already_nonzero_at_tb1024": already_nonzero,
        "any_arm_moved": any(row["either_moved"] for row in per_arm.values()),
        "per_family": {
            label: dict(per_family_by_label[label]) for label in MODEL_ORDER
        },
    }


def budget_contrast(
    scores: dict[str, dict],
    reference_scores: dict[str, dict],
    implementation: dict,
    reference_implementation: dict,
) -> dict:
    """Reading 2: per arm per family delta vs the tb1024 seed-78150 event.

    Aborts (fail closed) unless both events share the same benchmark
    implementation signature; both signatures are surfaced in the block.
    """
    require_implementation_equality(implementation, reference_implementation)
    per_arm = {}
    for label in MODEL_ORDER:
        new = scores[label]
        old = reference_scores[label]
        per_arm[label] = {
            "aggregate_tb4096": new["aggregate"],
            "aggregate_tb1024": old["aggregate"],
            "aggregate_delta": new["aggregate"] - old["aggregate"],
            "per_family_delta": {
                family: new["per_family"][family] - old["per_family"][family]
                for family in FAMILIES
            },
        }
    return {
        "cross_seed_confound": True,
        "note": CROSS_SEED_NOTE,
        "reference": dict(TB1024_REFERENCE),
        "benchmark_implementation": {
            "tb4096": dict(implementation),
            "tb1024": dict(reference_implementation),
            "identical": implementation == reference_implementation,
        },
        "per_arm": per_arm,
    }


def goal_gate_table(per_family_by_label: dict[str, dict[str, float]]) -> dict:
    """Reading 3: strict wins / ties / losses vs base; pass = ten strict wins."""
    base = per_family_by_label["base"]
    table = {}
    for arm in TREATED_ARMS:
        values = per_family_by_label[arm]
        wins = [f for f in FAMILIES if values[f] > base[f]]
        losses = [f for f in FAMILIES if values[f] < base[f]]
        ties = [f for f in FAMILIES if values[f] == base[f]]
        table[arm] = {
            "strict_wins": len(wins),
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "goal_gate_pass": len(wins) == len(FAMILIES),
        }
    return table


def budget_integrity(budget: dict[str, dict]) -> dict:
    """Reading 4: within_budget/wall_seconds per arm; scope the comparison."""
    per_arm = {
        label: {
            "within_budget": budget[label]["within_budget"],
            "wall_seconds": budget[label]["wall_seconds"],
        }
        for label in MODEL_ORDER
    }
    over = [label for label in MODEL_ORDER if not per_arm[label]["within_budget"]]
    return {
        "per_arm": per_arm,
        "all_within_budget": not over,
        "paired_comparison_valid": not over,
        "reason": (
            None
            if not over
            else (
                f"arms exceeded the gateway budget: {over}; the four-arm "
                "paired comparison is not budget-matched (scores recorded, "
                "not compared)"
            )
        ),
    }


def build_readout(
    scores: dict[str, dict],
    budget: dict[str, dict],
    implementation: dict,
    reference_scores: dict[str, dict],
    reference_implementation: dict,
    receipts: dict[str, dict],
    design_receipt_sha256: str,
) -> dict:
    """Assemble the readout from pure inputs (unit-testable, no file IO)."""
    per_family = {label: scores[label]["per_family"] for label in MODEL_ORDER}
    reference_per_family = {
        label: reference_scores[label]["per_family"] for label in MODEL_ORDER
    }
    integrity = budget_integrity(budget)
    return {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "stage": "medium_intermediate_budget_probe_readout",
        "name": FROZEN_NAME,
        "tier": FROZEN_TIER,
        "think_budget": FROZEN_THINK_BUDGET,
        "seed": FROZEN_SEED,
        "benchmark_data_read": False,
        "promoted": None,
        "outcome": "MEASUREMENT_READ_COMPLETE",
        "paired_comparison_valid": integrity["paired_comparison_valid"],
        "design_receipt_sha256": design_receipt_sha256,
        "tb1024_reference": dict(TB1024_REFERENCE),
        "receipts": receipts,
        "scores": scores,
        "budget": budget,
        "readings": {
            "budget_movement": budget_movement(per_family, reference_per_family),
            "budget_contrast": budget_contrast(
                scores, reference_scores, implementation, reference_implementation
            ),
            "goal_gate": goal_gate_table(per_family),
            "budget_integrity": integrity,
        },
    }


def render() -> bytes:
    """Authenticate every input, then render the readout bytes."""
    reference = load_tb1024_reference()
    if not DESIGN_RECEIPT.is_file():
        raise ValueError("design receipt is absent; readout stays unwritten")
    events = {}
    receipts = {}
    for label in MODEL_ORDER:
        path = EVENT_DIR / f"{label}.json"
        if not path.is_file():
            raise ValueError(f"gateway receipt is absent for {label}: {path}")
        events[label] = load_event(path, FROZEN_MODEL_PATHS[label])
        receipts[label] = {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(path),
        }
    signatures = {
        (
            event["benchmark_runner_sha256"],
            event["benchmark_source_inventory_sha256"],
            event["benchmark_source_file_count"],
        )
        for event in events.values()
    }
    if len(signatures) != 1:
        raise ValueError("benchmark implementation changed between paired arms")
    runner_sha, inventory_sha, file_count = next(iter(signatures))
    implementation = {
        "runner_sha256": runner_sha,
        "source_inventory_sha256": inventory_sha,
        "source_file_count": file_count,
    }
    scores = {
        label: {"aggregate": event["aggregate"], "per_family": event["per_family"]}
        for label, event in events.items()
    }
    budget = {
        label: {
            "within_budget": event["within_budget"],
            "wall_seconds": event["wall_seconds"],
        }
        for label, event in events.items()
    }
    readout = build_readout(
        scores,
        budget,
        implementation,
        reference["scores"],
        reference["benchmark_implementation"],
        receipts,
        sha256_file(DESIGN_RECEIPT),
    )
    return (
        json.dumps(readout, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--out", type=Path, help="write the readout (refuses overwrite)")
    args = parser.parse_args()
    try:
        value = render()
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))
    if args.out is not None:
        if args.out.exists():
            parser.error("refusing to overwrite measurement readout")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(value)
        target = args.out
    else:
        if not READOUT.is_file() or READOUT.read_bytes() != value:
            parser.error("published measurement readout is absent or changed")
        target = READOUT
    print(
        json.dumps(
            {
                "out": str(target),
                "sha256": hashlib.sha256(value).hexdigest(),
                "outcome": "MEASUREMENT_READ_COMPLETE",
            },
            indent=1,
            sort_keys=True,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
