#!/usr/bin/env python3
"""Compute the four preregistered medium-tier readings from the gateway receipts.

This is a MEASUREMENT read, not a gate: nothing passes or fails the
experiment here, nothing is promoted, and the process exits 0 on any
complete readout. The four frozen readings, computed only from the four
authenticated aggregate-gateway receipts of the one sealed event:

1. Aggregate ordering at medium, compared against the frozen quick
   ordering from seed 78144 (replay_repeat 0.5081 > designed_fresh 0.4644
   > base 0.1085; hygiene_explore has no quick aggregate and is recorded
   as null).
2. Goal-gate table per treated arm: strict wins / ties / losses versus
   base across the ten public families; goal_gate_pass = ten strict wins.
   Recorded, never gated on.
3. Base sanity envelope: per family, whether base's medium score falls
   inside the historical base [min, max] from the tier-forensics analysis
   (file pinned by sha256; an out-of-envelope base flags instrument drift
   and scopes every same-event comparison).
4. Blocking families per arm: the families not strictly won, i.e. the
   design constraints for the next dose.

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
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]

FROZEN_NAME = "measurement"
FROZEN_TIER = "medium"
FROZEN_THINK_BUDGET = 1024
FROZEN_SEED = 78150
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
FORENSICS_ANALYSIS = (
    ROOT / "experiments" / "qwen35_4b_menders_sirens_tier_forensics"
    / "runs" / "constants_analysis.json"
)
FORENSICS_ANALYSIS_SHA256 = (
    "62aaa80cf71ebfb0510b5a7c892d4bcf04d6b81b6fa559c401f2ee82a9d8868f"
)
# The frozen quick-tier reference event (a different, already-consumed seed).
QUICK_MEASURED_ARMS = ("base", "designed_fresh", "replay_repeat")
QUICK_REFERENCE = {
    "seed": 78144,
    "tier": "quick",
    "think_budget": 1024,
    "summary": (
        "experiments/qwen35_4b_goal_gap_axis_curriculum_target_match"
        "/runs/benchmark/quick_tb1024_seed78144_pilot/summary.json"
    ),
    "summary_sha256": (
        "4e28ba21a0c25e7bf46cabd42152a011fc86f3c0f4ba24c23ec1bf18beb78f23"
    ),
    # This cell's designed_fresh composite ran at quick under the label
    # designed_fresh_parent; hygiene_explore never ran at quick aggregate.
    "quick_labels": {
        "base": "base",
        "designed_fresh": "designed_fresh_parent",
        "hygiene_explore": None,
        "replay_repeat": "replay_repeat",
    },
    "aggregates": {
        "base": 0.10851063829787233,
        "designed_fresh": 0.46443720816263595,
        "hygiene_explore": None,
        "replay_repeat": 0.508134008201943,
    },
}


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


def load_event(path: Path, model: Path) -> dict:
    """Authenticate one aggregate-gateway receipt against the frozen event."""
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
        or payload.get("within_budget") is not True
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


def load_forensics_profile() -> dict:
    """Load the pinned historical base medium profile; fail closed on drift."""
    if (
        not FORENSICS_ANALYSIS.is_file()
        or sha256_file(FORENSICS_ANALYSIS) != FORENSICS_ANALYSIS_SHA256
    ):
        raise ValueError(
            f"pinned forensics analysis is absent or changed: {FORENSICS_ANALYSIS}"
        )
    payload = json.loads(FORENSICS_ANALYSIS.read_text(encoding="utf-8"))
    profile = payload["base_profile"]["medium"]
    families = profile.get("families", {})
    if set(families) != set(FAMILIES) or any(
        not isinstance(families[family].get(bound), (int, float))
        or isinstance(families[family].get(bound), bool)
        for family in FAMILIES
        for bound in ("min", "max")
    ):
        raise ValueError("forensics base medium profile violates the frozen shape")
    return profile


def verify_quick_reference() -> None:
    """Cross-check the embedded frozen quick aggregates against their receipt."""
    summary = ROOT / QUICK_REFERENCE["summary"]
    if not summary.is_file() or sha256_file(summary) != QUICK_REFERENCE["summary_sha256"]:
        raise ValueError(f"pinned quick reference summary is absent or changed: {summary}")
    payload = json.loads(summary.read_text(encoding="utf-8"))
    scores = payload.get("scores", {})
    if (
        payload.get("seed") != QUICK_REFERENCE["seed"]
        or payload.get("tier") != QUICK_REFERENCE["tier"]
        or payload.get("think_budget") != QUICK_REFERENCE["think_budget"]
    ):
        raise ValueError("quick reference summary is not the frozen quick event")
    for label in MODEL_ORDER:
        quick_label = QUICK_REFERENCE["quick_labels"][label]
        expected = QUICK_REFERENCE["aggregates"][label]
        if quick_label is None:
            if expected is not None or label in scores:
                raise ValueError(f"quick reference unexpectedly measured {label}")
            continue
        if scores.get(quick_label, {}).get("aggregate") != expected:
            raise ValueError(f"frozen quick aggregate changed for {label}")


def goal_gate_table(per_family_by_label: dict[str, dict[str, float]]) -> dict:
    """Strict wins / ties / losses vs base; pass = all ten strict wins."""
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


def blocking_families(table: dict) -> dict:
    """Reading 4: per arm, the families not strictly won."""
    return {arm: sorted(row["losses"] + row["ties"]) for arm, row in table.items()}


def base_envelope(base_per_family: dict[str, float], profile: dict) -> dict:
    """Reading 3: base's medium score against the historical [min, max]."""
    rows = {}
    for family in FAMILIES:
        value = base_per_family[family]
        low = profile["families"][family]["min"]
        high = profile["families"][family]["max"]
        rows[family] = {
            "base_medium": value,
            "envelope_min": low,
            "envelope_max": high,
            "inside": bool(low <= value <= high),
        }
    return {
        "historical_base_events": profile["events"],
        "families": rows,
        "all_inside": all(row["inside"] for row in rows.values()),
    }


def ordering_reading(aggregates: dict[str, float]) -> dict:
    """Reading 1: medium ordering vs the frozen quick ordering."""
    medium_ranking = sorted(MODEL_ORDER, key=lambda label: (-aggregates[label], label))
    quick = QUICK_REFERENCE["aggregates"]
    quick_ranking = sorted(
        QUICK_MEASURED_ARMS, key=lambda label: (-quick[label], label)
    )
    medium_ranking_quick_arms = [
        label for label in medium_ranking if label in QUICK_MEASURED_ARMS
    ]
    return {
        "medium_aggregates": {label: aggregates[label] for label in MODEL_ORDER},
        "medium_ranking": medium_ranking,
        "medium_strictly_ordered": (
            len({aggregates[label] for label in MODEL_ORDER}) == len(MODEL_ORDER)
        ),
        "quick_reference_seed": QUICK_REFERENCE["seed"],
        "quick_aggregates": dict(quick),
        "quick_ranking_measured_arms": quick_ranking,
        "medium_ranking_quick_measured_arms": medium_ranking_quick_arms,
        "ordering_matches_quick_on_measured_arms": (
            medium_ranking_quick_arms == quick_ranking
        ),
    }


def build_readout(
    scores: dict[str, dict],
    profile: dict,
    receipts: dict[str, dict],
    design_receipt_sha256: str,
) -> dict:
    """Assemble the readout from pure inputs (unit-testable, no file IO)."""
    aggregates = {label: scores[label]["aggregate"] for label in MODEL_ORDER}
    per_family = {label: scores[label]["per_family"] for label in MODEL_ORDER}
    table = goal_gate_table(per_family)
    return {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "stage": "medium_tier_measurement_readout",
        "name": FROZEN_NAME,
        "tier": FROZEN_TIER,
        "think_budget": FROZEN_THINK_BUDGET,
        "seed": FROZEN_SEED,
        "benchmark_data_read": False,
        "promoted": None,
        "outcome": "MEASUREMENT_READ_COMPLETE",
        "design_receipt_sha256": design_receipt_sha256,
        "forensics_analysis_sha256": FORENSICS_ANALYSIS_SHA256,
        "quick_reference": QUICK_REFERENCE,
        "receipts": receipts,
        "scores": scores,
        "readings": {
            "aggregate_ordering": ordering_reading(aggregates),
            "goal_gate": table,
            "base_sanity_envelope": base_envelope(per_family["base"], profile),
            "blocking_families": blocking_families(table),
        },
    }


def render() -> bytes:
    """Authenticate every input, then render the readout bytes."""
    verify_quick_reference()
    profile = load_forensics_profile()
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
    scores = {
        label: {"aggregate": event["aggregate"], "per_family": event["per_family"]}
        for label, event in events.items()
    }
    readout = build_readout(scores, profile, receipts, sha256_file(DESIGN_RECEIPT))
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
