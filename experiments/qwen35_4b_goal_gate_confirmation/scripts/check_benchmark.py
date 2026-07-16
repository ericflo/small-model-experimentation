#!/usr/bin/env python3
"""Compute the four preregistered confirmation readings from the gateway receipts.

This is a MEASUREMENT read with a preregistered verdict, not a promotion:
nothing is promoted, no training exists anywhere in this cell, and the
process exits 0 on any complete readout. The four frozen readings are
computed only from the SIX authenticated aggregate-gateway receipts of
the three sealed per-seed events (78155, 78156, 78157; two arms each,
base then hygiene_explore) plus the sha-pinned committed discovery-seed
summary:

1. per_seed: per seed, both aggregates, the full per-family table, and
   the goal gate — strict wins / ties / losses of hygiene_explore versus
   base over the ten public families; pass = TEN strict wins. The
   FAMILIES tuple and the strict-win logic are byte-for-byte the tier
   forensics' (analyze_constants.py), so the statistic is exactly the
   one the recorded seed-78154 pass was computed with.
2. confirmation_verdict — an ORDERED TOTAL PARTITION:
   ``CONFIRMED`` iff hygiene_explore's aggregate strictly beats base on
   ALL THREE seeds AND the goal gate passes on AT LEAST TWO of three;
   ``AGGREGATE_ONLY`` iff the aggregate strictly beats base on all three
   seeds but the goal-gate majority fails; ``NOT_REPLICATED`` otherwise.
   Exact ties are never strict wins. The discovery seed 78154 is
   REPORTED alongside (from the sha-pinned committed summary, fail
   closed on any drift) but NEVER counted in the verdict.
3. fragility: per seed, the menders and warren margins (hygiene_explore
   minus base — the two single-item margins that carried the discovery
   pass) and which families block any seed that does not pass.
4. budget_integrity: per arm per seed, the gateway receipt's
   ``within_budget`` flag and ``wall_seconds``. If ANY arm at ANY seed
   has within_budget false the readout sets ``paired_comparison_valid:
   false`` with the reason; scores are still recorded either way.

Implementation-signature integrity is a fail-closed precondition, not a
reading: all six receipts must share one (runner sha256, source
inventory sha256, file count) signature AND match the pinned discovery
summary's benchmark_implementation block, or the readout is never
written.

PROVENANCE ANCHORING (fail closed, both --out and verify modes): the
verdict inputs are only ever read through the k-seed write-ahead
ledger. The ledger must contain EXACTLY the complete canonical sequence
— three well-formed closed records for the three frozen seeds, in
order, each preceded by its opened record, with no trailing crashed
record; a missing, incomplete, or crashed ledger refuses. Each seed's
sealed summary must match the sha256 its closed record pinned; each
per-arm gateway receipt must match the receipt sha256 its closed record
pinned at close time; and each receipt's scores/budget/implementation
must equal the sealed summary's recorded blocks. A forged or swapped
receipt therefore fails against the write-ahead pins instead of
flowing into the verdict. The benchmark suite directory is never read;
only ledger-pinned gateway receipts and the pinned committed summary
are consumed.
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

FROZEN_NAME = "confirmation"
FROZEN_TIER = "medium"
FROZEN_THINK_BUDGET = 1024
SEED_ORDER = (78155, 78156, 78157)
MODEL_ORDER = ("base", "hygiene_explore")
TREATED_ARM = "hygiene_explore"
FROZEN_MODEL_PATHS = {
    "base": (
        ROOT / "large_artifacts" / "qwen35_4b_universal_curriculum"
        / "merged" / "base_reserialized"
    ),
    "hygiene_explore": (
        ROOT / "large_artifacts" / "qwen35_4b_hygiene_explore_destack_medium"
        / "merged" / "hygiene_explore"
    ),
}
FROZEN_TREE_SHA256 = {
    "base": "26d8ee48583adb0fb557d0ff668664949adff0068fa5baafe6f0af68e22fb677",
    "hygiene_explore": (
        "9eb653d78f05546ca594a831c989fa906d12f3eb7a5a8550d1afcd6bfccc4971"
    ),
}
FROZEN_WEIGHTS_SHA256 = {
    "base": "b654e033d525d87cbbd746bb681d80813c4b00d8e6202cb3edcfb6dfa3b416db",
    "hygiene_explore": (
        "e21123443a230ada2c73ded411e0b5b7c2b1459856b2c38e4f1beea8958dc02f"
    ),
}
EVENT_DIRS = {
    seed: (
        EXP / "runs" / "benchmark"
        / f"{FROZEN_TIER}_tb{FROZEN_THINK_BUDGET}_seed{seed}_{FROZEN_NAME}"
    )
    for seed in SEED_ORDER
}
READOUT = EXP / "runs" / "benchmark" / "confirmation_readout.json"
LEDGER = EXP / "runs" / "benchmark_events.jsonl"
DESIGN_RECEIPT = EXP / "data" / "design_receipt.json"
GATEWAY_SHA256 = "53cf6533dbd710eb167503363c39f73dbf7559a0d91f40a00436a3c218a01c17"
CLOSED_RECORD_KEYS = frozenset(
    {
        "name", "phase", "tier", "think_budget", "seed",
        "summary", "summary_sha256", "receipts",
    }
)
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
# The two single-item margins that carried the discovery pass: menders
# 0.0167 and warren 0.050 at seed 78154.
FRAGILITY_FAMILIES = ("menders", "warren")
IMPLEMENTATION_KEYS = ("runner_sha256", "source_inventory_sha256", "source_file_count")
# The committed discovery event at seed 78154 (already consumed by the
# statechain dose pilot): the recorded 10/10 goal-gate pass this cell
# replicates. Sha-pinned; fail closed; REPORTED alongside the verdict,
# NEVER counted in it.
DISCOVERY = {
    "seed": 78154,
    "tier": "medium",
    "think_budget": 1024,
    "summary": (
        "experiments/qwen35_4b_statechain_only_dose"
        "/runs/benchmark/medium_tb1024_seed78154_pilot/summary.json"
    ),
    "summary_sha256": (
        "6b1a43869f013e24a048a45a04e5603b45fe59488912194eb3e76a43679255fa"
    ),
    "base_arm": "base",
    "treated_arm": "hygiene_explore_parent",
    "counted_in_verdict": False,
}
DISCOVERY_MODEL_ORDER = (
    "base", "hygiene_explore_parent", "replay_ctl2", "statechain_only"
)
DISCOVERY_IMPLEMENTATION = {
    "runner_sha256": (
        "a3beecd8b5c89ccfd99a172a6d85321d39b9feb6c29d12f10b2f4d7499e273cb"
    ),
    "source_inventory_sha256": (
        "218b8615a95f24da962c931e9cd2dba58d853a7bdcd2847cd8e2c42fc2c05f42"
    ),
    "source_file_count": 56,
}
VERDICTS = ("CONFIRMED", "AGGREGATE_ONLY", "NOT_REPLICATED")
VERDICT_RULE = (
    "ordered total partition: CONFIRMED iff hygiene_explore's aggregate "
    "strictly beats base on ALL THREE seeds AND the goal gate (ten strict "
    "family wins) passes on AT LEAST TWO of three; AGGREGATE_ONLY iff the "
    "aggregate strictly beats base on all three seeds but the goal-gate "
    "majority fails; NOT_REPLICATED otherwise; exact ties are never strict "
    "wins; the discovery seed 78154 is reported alongside and NEVER counted"
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


def opened_record(seed: int) -> dict:
    return {
        "name": FROZEN_NAME,
        "phase": "opened",
        "seed": seed,
        "think_budget": FROZEN_THINK_BUDGET,
        "tier": FROZEN_TIER,
    }


def is_closed_record(row: object, seed: int) -> bool:
    """A well-formed per-seed closed record pinning summary AND receipts."""
    return (
        isinstance(row, dict)
        and set(row) == set(CLOSED_RECORD_KEYS)
        and row["name"] == FROZEN_NAME
        and row["phase"] == "closed"
        and row["tier"] == FROZEN_TIER
        and row["think_budget"] == FROZEN_THINK_BUDGET
        and row["seed"] == seed
        and row["summary"] == str(EVENT_DIRS[seed] / "summary.json")
        and isinstance(row["summary_sha256"], str)
        and re.fullmatch(r"[0-9a-f]{64}", row["summary_sha256"]) is not None
        and isinstance(row["receipts"], dict)
        and set(row["receipts"]) == set(MODEL_ORDER)
        and all(
            isinstance(value, str)
            and re.fullmatch(r"[0-9a-f]{64}", value) is not None
            for value in row["receipts"].values()
        )
    )


def authenticate_ledger(rows: list[object]) -> dict[int, dict]:
    """The verdict may only be read through a COMPLETE k-seed ledger.

    Requires EXACTLY the canonical seed-major sequence — opened(78155),
    closed(78155), opened(78156), closed(78156), opened(78157),
    closed(78157) — and returns the three closed records (whose pinned
    shas anchor every summary and receipt). A missing or empty ledger, an
    incomplete event, a trailing crashed opened record, out-of-order
    seeds, malformed rows, or trailing extras all refuse: the readout is
    never computed from unanchored receipt files.
    """
    if not rows:
        raise ValueError(
            "benchmark ledger is absent or empty; the confirmation readout "
            "requires the complete three-seed write-ahead ledger"
        )
    closed = {}
    index = 0
    for seed in SEED_ORDER:
        if index == len(rows) or rows[index] != opened_record(seed):
            raise ValueError(
                f"benchmark ledger row {index + 1} is not the frozen opened "
                f"record for seed {seed}; the ledger is incomplete or corrupt"
            )
        index += 1
        if index == len(rows):
            raise ValueError(
                f"benchmark ledger ends with a crashed opened record for seed "
                f"{seed}; the confirmation readout requires all three seeds "
                "closed"
            )
        if not is_closed_record(rows[index], seed):
            raise ValueError(
                f"benchmark ledger row {index + 1} is not the closed record "
                f"for seed {seed}"
            )
        closed[seed] = rows[index]
        index += 1
    if index != len(rows):
        raise ValueError(
            "benchmark ledger has rows beyond the frozen three-seed event"
        )
    return closed


def require_summary_consistency(
    seed: int, summary: object, events: dict[str, dict]
) -> None:
    """Receipts must equal the sealed summary's recorded blocks, exactly.

    The summary bytes are pinned by the closed ledger record; requiring
    each authenticated receipt's scores/budget/implementation to equal the
    summary's recorded blocks binds the verdict inputs to the sealed
    event — a receipt swapped after close time cannot both match its
    pinned sha and diverge here.
    """
    if (
        not isinstance(summary, dict)
        or summary.get("schema_version") != 1
        or summary.get("name") != FROZEN_NAME
        or summary.get("tier") != FROZEN_TIER
        or summary.get("think_budget") != FROZEN_THINK_BUDGET
        or summary.get("seed") != seed
        or summary.get("model_order") != list(MODEL_ORDER)
        or summary.get("promoted") is not None
        or summary.get("benchmark_data_read") is not False
        or summary.get("gateway_sha256") != GATEWAY_SHA256
        or summary.get("discovery_summary_sha256") != DISCOVERY["summary_sha256"]
    ):
        raise ValueError(f"sealed summary failed authentication for seed {seed}")
    for label in MODEL_ORDER:
        event = events[label]
        if summary.get("scores", {}).get(label) != {
            "aggregate": event["aggregate"],
            "per_family": event["per_family"],
        }:
            raise ValueError(
                f"receipt scores diverge from the sealed summary for seed "
                f"{seed} arm {label}"
            )
        if summary.get("budget", {}).get(label) != {
            "within_budget": event["within_budget"],
            "wall_seconds": event["wall_seconds"],
        }:
            raise ValueError(
                f"receipt budget diverges from the sealed summary for seed "
                f"{seed} arm {label}"
            )
        if summary.get("benchmark_implementation") != {
            "runner_sha256": event["benchmark_runner_sha256"],
            "source_inventory_sha256": event["benchmark_source_inventory_sha256"],
            "source_file_count": event["benchmark_source_file_count"],
        }:
            raise ValueError(
                f"receipt implementation diverges from the sealed summary for "
                f"seed {seed} arm {label}"
            )


def load_event(path: Path, model: Path, seed: int) -> dict:
    """Authenticate one aggregate-gateway receipt against its frozen seed.

    ``within_budget`` must be a strict bool but is RECORDED, never required
    to be true: the budget_integrity reading scopes the paired comparison
    (paired_comparison_valid) instead of rejecting an over-budget arm.
    """
    if seed not in SEED_ORDER:
        raise ValueError(f"receipt seed is not one of the frozen three: {seed}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        set(payload) != GATEWAY_KEYS
        or payload.get("schema_version") != 1
        or payload.get("stage") != "menagerie_aggregate_gateway"
        or payload.get("tier") != FROZEN_TIER
        or payload.get("think_budget") != FROZEN_THINK_BUDGET
        or payload.get("seed") != seed
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


def goal_gate_row(
    base_per_family: dict[str, float], treated_per_family: dict[str, float]
) -> dict:
    """The forensics strict-win partition; pass = ten strict wins."""
    wins = [f for f in FAMILIES if treated_per_family[f] > base_per_family[f]]
    losses = [f for f in FAMILIES if treated_per_family[f] < base_per_family[f]]
    ties = [f for f in FAMILIES if treated_per_family[f] == base_per_family[f]]
    return {
        "strict_wins": len(wins),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "goal_gate_pass": len(wins) == len(FAMILIES),
    }


def load_discovery_reference() -> dict:
    """Load the sha-pinned discovery-seed summary; fail closed on any drift.

    Beyond the byte pin, the loader authenticates that the summary IS the
    recorded discovery: seed 78154 at medium/tb1024, the four-arm pilot
    order, the same base and hygiene_explore composites (tree AND weights
    hashes equal to this cell's frozen pins — the treated arm was labeled
    ``hygiene_explore_parent`` there), a valid benchmark-implementation
    signature equal to the pinned block, and the recorded 10/10 goal-gate
    pass, re-derived from the pinned scores with this cell's own
    strict-win partition.
    """
    summary = ROOT / DISCOVERY["summary"]
    if not summary.is_file() or sha256_file(summary) != DISCOVERY["summary_sha256"]:
        raise ValueError(
            f"pinned discovery-seed summary is absent or changed: {summary}"
        )
    payload = json.loads(summary.read_text(encoding="utf-8"))
    scores = payload.get("scores", {})
    implementation = payload.get("benchmark_implementation")
    trees = payload.get("model_tree_sha256s", {})
    weights = payload.get("model_weight_sha256s", {})
    if (
        payload.get("seed") != DISCOVERY["seed"]
        or payload.get("tier") != DISCOVERY["tier"]
        or payload.get("think_budget") != DISCOVERY["think_budget"]
        or payload.get("model_order") != list(DISCOVERY_MODEL_ORDER)
        or not _valid_implementation(implementation)
        or implementation != DISCOVERY_IMPLEMENTATION
        or trees.get(DISCOVERY["base_arm"]) != FROZEN_TREE_SHA256["base"]
        or trees.get(DISCOVERY["treated_arm"]) != FROZEN_TREE_SHA256[TREATED_ARM]
        or weights.get(DISCOVERY["base_arm"]) != FROZEN_WEIGHTS_SHA256["base"]
        or weights.get(DISCOVERY["treated_arm"])
        != FROZEN_WEIGHTS_SHA256[TREATED_ARM]
    ):
        raise ValueError("discovery summary is not the frozen seed-78154 event")
    for label in (DISCOVERY["base_arm"], DISCOVERY["treated_arm"]):
        row = scores.get(label, {})
        if (
            set(row.get("per_family", {})) != set(FAMILIES)
            or not _valid_score(row.get("aggregate"))
            or any(not _valid_score(value) for value in row["per_family"].values())
        ):
            raise ValueError(f"discovery summary violates the score shape: {label}")
    recorded = (
        payload.get("goal_gate", {})
        .get("per_arm", {})
        .get(DISCOVERY["treated_arm"], {})
    )
    recomputed = goal_gate_row(
        scores[DISCOVERY["base_arm"]]["per_family"],
        scores[DISCOVERY["treated_arm"]]["per_family"],
    )
    if (
        recorded.get("goal_gate_pass") is not True
        or recorded.get("strict_wins") != len(FAMILIES)
        or recorded.get("wins") != recomputed["wins"]
        or recomputed["goal_gate_pass"] is not True
    ):
        raise ValueError(
            "discovery summary does not carry the recorded 10/10 goal-gate pass"
        )
    return {
        "scores": {
            label: {
                "aggregate": scores[label]["aggregate"],
                "per_family": dict(scores[label]["per_family"]),
            }
            for label in (DISCOVERY["base_arm"], DISCOVERY["treated_arm"])
        },
        "benchmark_implementation": dict(implementation),
    }


def require_implementation_equality(
    implementation: dict, discovery_implementation: dict
) -> None:
    """Fail-closed comparability anchor to the discovery event.

    All six new receipts already share ``implementation`` (enforced by the
    caller); this guard additionally requires the shared signature to
    equal the pinned discovery summary's block, or the whole readout is
    invalid and must not be written.
    """
    if implementation != discovery_implementation:
        raise ValueError(
            "benchmark implementation differs from the pinned discovery event "
            f"(confirmation={implementation}, "
            f"discovery={discovery_implementation}); "
            "the preregistered confirmation is not comparable"
        )


def per_seed_reading(scores_by_seed: dict[int, dict]) -> dict:
    """Reading 1: aggregates, full per-family table, and goal gate per seed."""
    reading = {}
    for seed in SEED_ORDER:
        scores = scores_by_seed[seed]
        base = scores["base"]
        treated = scores[TREATED_ARM]
        reading[str(seed)] = {
            "aggregates": {
                "base": base["aggregate"],
                TREATED_ARM: treated["aggregate"],
            },
            "aggregate_margin": treated["aggregate"] - base["aggregate"],
            "treated_beats_base_aggregate": treated["aggregate"] > base["aggregate"],
            "per_family": {
                label: dict(scores[label]["per_family"]) for label in MODEL_ORDER
            },
            "goal_gate": goal_gate_row(base["per_family"], treated["per_family"]),
        }
    return reading


def discovery_report(discovery_scores: dict[str, dict]) -> dict:
    """The discovery seed, reported alongside the verdict, never counted."""
    base = discovery_scores[DISCOVERY["base_arm"]]
    treated = discovery_scores[DISCOVERY["treated_arm"]]
    return {
        "seed": DISCOVERY["seed"],
        "tier": DISCOVERY["tier"],
        "think_budget": DISCOVERY["think_budget"],
        "summary": DISCOVERY["summary"],
        "summary_sha256": DISCOVERY["summary_sha256"],
        "treated_arm_label": DISCOVERY["treated_arm"],
        "aggregates": {
            DISCOVERY["base_arm"]: base["aggregate"],
            DISCOVERY["treated_arm"]: treated["aggregate"],
        },
        "aggregate_margin": treated["aggregate"] - base["aggregate"],
        "goal_gate": goal_gate_row(base["per_family"], treated["per_family"]),
        "fragility_margins": {
            family: treated["per_family"][family] - base["per_family"][family]
            for family in FRAGILITY_FAMILIES
        },
        "counted_in_verdict": False,
    }


def confirmation_verdict(
    scores_by_seed: dict[int, dict], discovery_scores: dict[str, dict]
) -> dict:
    """Reading 2: the ordered total partition over the three sealed seeds.

    The discovery block is embedded for the report but contributes NOTHING
    to the verdict: only the three sealed seeds' aggregate strict wins and
    goal-gate passes are counted.
    """
    aggregate_wins = {}
    gate_passes = {}
    for seed in SEED_ORDER:
        scores = scores_by_seed[seed]
        aggregate_wins[str(seed)] = (
            scores[TREATED_ARM]["aggregate"] > scores["base"]["aggregate"]
        )
        gate_passes[str(seed)] = goal_gate_row(
            scores["base"]["per_family"], scores[TREATED_ARM]["per_family"]
        )["goal_gate_pass"]
    aggregate_all = all(aggregate_wins.values())
    pass_count = sum(1 for passed in gate_passes.values() if passed)
    majority = pass_count >= 2
    if aggregate_all and majority:
        verdict = "CONFIRMED"
    elif aggregate_all:
        verdict = "AGGREGATE_ONLY"
    else:
        verdict = "NOT_REPLICATED"
    return {
        "rule": VERDICT_RULE,
        "aggregate_strict_wins": aggregate_wins,
        "aggregate_wins_all_three_seeds": aggregate_all,
        "goal_gate_passes": gate_passes,
        "goal_gate_pass_count": pass_count,
        "goal_gate_majority": majority,
        "verdict": verdict,
        "discovery": discovery_report(discovery_scores),
    }


def fragility(scores_by_seed: dict[int, dict]) -> dict:
    """Reading 3: the discovery-carrying margins and the blocking families."""
    per_seed = {}
    for seed in SEED_ORDER:
        scores = scores_by_seed[seed]
        base = scores["base"]["per_family"]
        treated = scores[TREATED_ARM]["per_family"]
        gate = goal_gate_row(base, treated)
        per_seed[str(seed)] = {
            "menders_margin": treated["menders"] - base["menders"],
            "warren_margin": treated["warren"] - base["warren"],
            "goal_gate_pass": gate["goal_gate_pass"],
            "blocking_families": sorted(gate["ties"] + gate["losses"]),
        }
    return {
        "question": (
            "do the two single-item margins that carried the discovery pass "
            "(menders 0.0167 and warren 0.050 at seed 78154) survive fresh "
            "seeds, and which families block any seed that does not pass "
            "the goal gate"
        ),
        "families": list(FRAGILITY_FAMILIES),
        "per_seed": per_seed,
    }


def budget_integrity(budget_by_seed: dict[int, dict]) -> dict:
    """Reading 4: within_budget/wall_seconds per arm per seed; scope only."""
    per_seed = {}
    over_all = []
    for seed in SEED_ORDER:
        budget = budget_by_seed[seed]
        per_arm = {
            label: {
                "within_budget": budget[label]["within_budget"],
                "wall_seconds": budget[label]["wall_seconds"],
            }
            for label in MODEL_ORDER
        }
        over = [label for label in MODEL_ORDER if not per_arm[label]["within_budget"]]
        per_seed[str(seed)] = {
            "per_arm": per_arm,
            "all_within_budget": not over,
            "paired_comparison_valid": not over,
            "reason": (
                None
                if not over
                else (
                    f"arms exceeded the gateway budget at seed {seed}: {over}; "
                    "the paired comparison at this seed is not budget-matched "
                    "(scores recorded, not compared)"
                )
            ),
        }
        over_all.extend(f"seed {seed}: {label}" for label in over)
    return {
        "per_seed": per_seed,
        "all_within_budget": not over_all,
        "paired_comparison_valid": not over_all,
        "reason": (
            None
            if not over_all
            else (
                f"over-budget arms: {over_all}; the three-seed confirmation "
                "comparison is not budget-matched (scores recorded, not "
                "compared)"
            )
        ),
    }


def build_readout(
    scores_by_seed: dict[int, dict],
    budget_by_seed: dict[int, dict],
    implementation: dict,
    discovery: dict,
    receipts: dict[int, dict],
    design_receipt_sha256: str,
) -> dict:
    """Assemble the readout from pure inputs (unit-testable, no file IO)."""
    require_implementation_equality(
        implementation, discovery["benchmark_implementation"]
    )
    integrity = budget_integrity(budget_by_seed)
    verdict = confirmation_verdict(scores_by_seed, discovery["scores"])
    return {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "stage": "goal_gate_confirmation_readout",
        "name": FROZEN_NAME,
        "tier": FROZEN_TIER,
        "think_budget": FROZEN_THINK_BUDGET,
        "seeds": list(SEED_ORDER),
        "benchmark_data_read": False,
        "promoted": None,
        "outcome": "CONFIRMATION_READ_COMPLETE",
        "verdict": verdict["verdict"],
        "paired_comparison_valid": integrity["paired_comparison_valid"],
        "design_receipt_sha256": design_receipt_sha256,
        "provenance": {
            "ledger": "runs/benchmark_events.jsonl",
            "ledger_complete_sequence_required": True,
            "receipt_sha256s_pinned_in_closed_records": True,
            "summaries_verified_against_ledger_pins": True,
            "receipts_verified_against_ledger_pins": True,
            "receipt_blocks_verified_against_sealed_summaries": True,
        },
        "discovery_reference": dict(DISCOVERY),
        "benchmark_implementation": {
            "signature": dict(implementation),
            "discovery": dict(discovery["benchmark_implementation"]),
            "identical_across_all_six_receipts_and_discovery": True,
        },
        "receipts": {str(seed): receipts[seed] for seed in SEED_ORDER},
        "scores": {str(seed): scores_by_seed[seed] for seed in SEED_ORDER},
        "budget": {str(seed): budget_by_seed[seed] for seed in SEED_ORDER},
        "readings": {
            "per_seed": per_seed_reading(scores_by_seed),
            "confirmation_verdict": verdict,
            "fragility": fragility(scores_by_seed),
            "budget_integrity": integrity,
        },
    }


def render() -> bytes:
    """Authenticate every input through the ledger, then render the readout.

    Order of anchoring: the complete ledger first (three closed records,
    in order, nothing crashed or trailing), then per seed the sealed
    summary against its pinned sha, then each receipt against the sha its
    closed record pinned at close time, then structural receipt
    authentication, then receipt-versus-summary block equality. Only
    inputs that survive all five layers reach the verdict.
    """
    discovery = load_discovery_reference()
    if not DESIGN_RECEIPT.is_file():
        raise ValueError("design receipt is absent; readout stays unwritten")
    ledger_rows = (
        [
            json.loads(line)
            for line in LEDGER.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if LEDGER.is_file()
        else []
    )
    closed_records = authenticate_ledger(ledger_rows)
    design_receipt_sha256 = sha256_file(DESIGN_RECEIPT)
    scores_by_seed = {}
    budget_by_seed = {}
    receipts = {}
    signatures = set()
    for seed in SEED_ORDER:
        record = closed_records[seed]
        summary_path = EVENT_DIRS[seed] / "summary.json"
        if (
            not summary_path.is_file()
            or sha256_file(summary_path) != record["summary_sha256"]
        ):
            raise ValueError(
                f"sealed summary is absent or does not match its closed "
                f"ledger pin for seed {seed}"
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("design_receipt_sha256") != design_receipt_sha256:
            raise ValueError(
                f"sealed summary for seed {seed} was produced under a "
                "different design receipt"
            )
        events = {}
        receipts[seed] = {}
        for label in MODEL_ORDER:
            path = EVENT_DIRS[seed] / f"{label}.json"
            if not path.is_file():
                raise ValueError(
                    f"gateway receipt is absent for seed {seed} arm {label}: {path}"
                )
            digest = sha256_file(path)
            if digest != record["receipts"][label]:
                raise ValueError(
                    f"gateway receipt does not match its closed ledger pin "
                    f"for seed {seed} arm {label}"
                )
            events[label] = load_event(path, FROZEN_MODEL_PATHS[label], seed)
            receipts[seed][label] = {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": digest,
            }
            signatures.add(
                (
                    events[label]["benchmark_runner_sha256"],
                    events[label]["benchmark_source_inventory_sha256"],
                    events[label]["benchmark_source_file_count"],
                )
            )
        require_summary_consistency(seed, summary, events)
        scores_by_seed[seed] = {
            label: {"aggregate": event["aggregate"], "per_family": event["per_family"]}
            for label, event in events.items()
        }
        budget_by_seed[seed] = {
            label: {
                "within_budget": event["within_budget"],
                "wall_seconds": event["wall_seconds"],
            }
            for label, event in events.items()
        }
    if len(signatures) != 1:
        raise ValueError("benchmark implementation changed between the six receipts")
    runner_sha, inventory_sha, file_count = next(iter(signatures))
    implementation = {
        "runner_sha256": runner_sha,
        "source_inventory_sha256": inventory_sha,
        "source_file_count": file_count,
    }
    readout = build_readout(
        scores_by_seed,
        budget_by_seed,
        implementation,
        discovery,
        receipts,
        design_receipt_sha256,
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
            parser.error("refusing to overwrite confirmation readout")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(value)
        target = args.out
    else:
        if not READOUT.is_file() or READOUT.read_bytes() != value:
            parser.error("published confirmation readout is absent or changed")
        target = READOUT
    print(
        json.dumps(
            {
                "out": str(target),
                "sha256": hashlib.sha256(value).hexdigest(),
                "outcome": "CONFIRMATION_READ_COMPLETE",
                "verdict": json.loads(value.decode("utf-8"))["verdict"],
            },
            indent=1,
            sort_keys=True,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
