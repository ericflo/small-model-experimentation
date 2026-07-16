#!/usr/bin/env python3
"""Compute the frozen zero-root readings from the three gateway receipts.

This is a MEASUREMENT read with a preregistered consequence, not a
promotion: nothing is promoted and the process exits 0 on any complete
readout. The frozen readings are computed only from the THREE
authenticated aggregate-gateway receipts of the ONE sealed event (seed
78159 at medium/tb1024; arms base, hygiene_explore_original,
zero_root_hygiene_explore):

1. per_family: both composite aggregates, the base aggregate, and the
   full per-family table for all three arms.
2. goal_gate: for BOTH composites versus base — strict wins / ties /
   losses over the ten public families; pass = TEN strict wins. The
   FAMILIES tuple and the strict-win logic are byte-for-byte the tier
   forensics' (analyze_constants.py), so the statistic is exactly the
   one the original composite's recorded sweeps were computed with.
3. prefix_contribution: zero-root minus original, per family and
   aggregate — the frozen framing: "the gym-era root's contribution at
   medium, one seed, cross-arm same-seed paired".
4. budget_integrity: per arm, the gateway receipt's ``within_budget``
   flag and ``wall_seconds``. If ANY arm has within_budget false the
   readout sets ``paired_comparison_valid: false`` with the reason;
   scores are still recorded either way.
5. margins: the menders / rites / warren single-family margins versus
   base for both composites. The statechain-to-rites conversion question
   does NOT apply here (this lineage has no statechain stage), but the
   rites and warren margins matter for the sweep reading.

CONSEQUENCE (ordered total partition, frozen before the event):
``ZERO_ROOT_COMPARABLE`` iff the zero-root aggregate strictly beats base
AND its goal-gate strict wins are at least the original's strict wins on
this seed minus one — "the documented stages alone carry the
demonstrated position; the headline model is contamination-clean
end-to-end". ``ZERO_ROOT_DEGRADED`` otherwise — "the undocumented prefix
is load-bearing at medium; its contribution is the recorded contrast".
Exact ties are never strict wins.

PROVENANCE ANCHORING (fail closed, both --out and verify modes): the
consequence inputs are only ever read through the single-seed
write-ahead ledger. The ledger must contain EXACTLY the canonical
sequence — the opened record then the closed record for seed 78159; a
missing, incomplete, or crashed ledger refuses. The sealed summary must
match the sha256 its closed record pinned; each per-arm gateway receipt
must match the receipt sha256 the closed record pinned at close time;
each receipt's scores/budget/implementation must equal the sealed
summary's recorded blocks; and the summary's zero-root arm identity must
match the COMMITTED rebuild merge receipt (recomputed sha256 plus tree
equality), so the measured composite is provenance-anchored to the
six committed stage receipts. The benchmark suite directory is never
read; only ledger-pinned gateway receipts are consumed.
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

FROZEN_NAME = "zero_root"
FROZEN_TIER = "medium"
FROZEN_THINK_BUDGET = 1024
FROZEN_SEED = 78159
MODEL_ORDER = ("base", "hygiene_explore_original", "zero_root_hygiene_explore")
ORIGINAL_ARM = "hygiene_explore_original"
ZERO_ROOT_ARM = "zero_root_hygiene_explore"
COMPOSITE_ARMS = (ORIGINAL_ARM, ZERO_ROOT_ARM)
FROZEN_MODEL_PATHS = {
    "base": (
        ROOT / "large_artifacts" / "qwen35_4b_universal_curriculum"
        / "merged" / "base_reserialized"
    ),
    "hygiene_explore_original": (
        ROOT / "large_artifacts" / "qwen35_4b_hygiene_explore_destack_medium"
        / "merged" / "hygiene_explore"
    ),
    "zero_root_hygiene_explore": (
        ROOT / "large_artifacts" / "qwen35_4b_zero_root_lineage_rebuild"
        / "merged" / "zero_root_hygiene_explore"
    ),
}
# Base and original pins are the goal-gate events'; the zero-root arm is
# anchored through the COMMITTED rebuild merge receipt instead of a code
# pin (this file carries no TODO-PINs and is hash-pinned by the design
# receipt).
FROZEN_TREE_SHA256 = {
    "base": "26d8ee48583adb0fb557d0ff668664949adff0068fa5baafe6f0af68e22fb677",
    "hygiene_explore_original": (
        "9eb653d78f05546ca594a831c989fa906d12f3eb7a5a8550d1afcd6bfccc4971"
    ),
}
FROZEN_WEIGHTS_SHA256 = {
    "base": "b654e033d525d87cbbd746bb681d80813c4b00d8e6202cb3edcfb6dfa3b416db",
    "hygiene_explore_original": (
        "e21123443a230ada2c73ded411e0b5b7c2b1459856b2c38e4f1beea8958dc02f"
    ),
}
ZERO_ROOT_MERGE_RECEIPT = EXP / "runs" / "lineage" / "merge.json"
EVENT_DIR = (
    EXP / "runs" / "benchmark"
    / f"{FROZEN_TIER}_tb{FROZEN_THINK_BUDGET}_seed{FROZEN_SEED}_{FROZEN_NAME}"
)
READOUT = EXP / "runs" / "benchmark" / "zero_root_readout.json"
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
# Single-family margins recorded specifically: menders and warren carried
# the original discovery pass on single-item margins; rites matters for the
# sweep reading (the statechain-to-rites conversion question does NOT apply
# here — this lineage has no statechain stage).
MARGIN_FAMILIES = ("menders", "rites", "warren")
# The signature of the discovery/confirmation events under which the
# original composite recorded its two 10/10 sweeps; all three receipts must
# match it (enforced at event time, re-checked here).
REFERENCE_IMPLEMENTATION = {
    "runner_sha256": (
        "a3beecd8b5c89ccfd99a172a6d85321d39b9feb6c29d12f10b2f4d7499e273cb"
    ),
    "source_inventory_sha256": (
        "218b8615a95f24da962c931e9cd2dba58d853a7bdcd2847cd8e2c42fc2c05f42"
    ),
    "source_file_count": 56,
}
CONSEQUENCES = ("ZERO_ROOT_COMPARABLE", "ZERO_ROOT_DEGRADED")
CONSEQUENCE_RULE = (
    "ordered total partition: ZERO_ROOT_COMPARABLE iff the zero-root "
    "composite's aggregate strictly beats base AND its goal-gate strict "
    "wins over the ten public families are at least the original "
    "composite's strict wins on this seed minus one; ZERO_ROOT_DEGRADED "
    "otherwise; exact ties are never strict wins"
)
CONSEQUENCE_STATEMENTS = {
    "ZERO_ROOT_COMPARABLE": (
        "the documented stages alone carry the demonstrated position; the "
        "headline model is contamination-clean end-to-end"
    ),
    "ZERO_ROOT_DEGRADED": (
        "the undocumented prefix is load-bearing at medium; its "
        "contribution is the recorded contrast"
    ),
}
PREFIX_CONTRIBUTION_FRAMING = (
    "the gym-era root's contribution at medium, one seed, cross-arm "
    "same-seed paired"
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


def _sha_hex(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def opened_record() -> dict:
    return {
        "name": FROZEN_NAME,
        "phase": "opened",
        "seed": FROZEN_SEED,
        "think_budget": FROZEN_THINK_BUDGET,
        "tier": FROZEN_TIER,
    }


def is_closed_record(row: object) -> bool:
    """A well-formed closed record pinning the summary AND all three receipts."""
    return (
        isinstance(row, dict)
        and set(row) == set(CLOSED_RECORD_KEYS)
        and row["name"] == FROZEN_NAME
        and row["phase"] == "closed"
        and row["tier"] == FROZEN_TIER
        and row["think_budget"] == FROZEN_THINK_BUDGET
        and row["seed"] == FROZEN_SEED
        and row["summary"] == str(EVENT_DIR / "summary.json")
        and _sha_hex(row["summary_sha256"])
        and isinstance(row["receipts"], dict)
        and set(row["receipts"]) == set(MODEL_ORDER)
        and all(_sha_hex(value) for value in row["receipts"].values())
    )


def authenticate_ledger(rows: list[object]) -> dict:
    """The consequence may only be read through a COMPLETE ledger.

    Requires EXACTLY the canonical sequence — opened(78159) then
    closed(78159) — and returns the closed record (whose pinned shas
    anchor the summary and all three receipts). A missing or empty
    ledger, a crashed opened record, malformed rows, or trailing extras
    all refuse: the readout is never computed from unanchored receipts.
    """
    if not rows:
        raise ValueError(
            "benchmark ledger is absent or empty; the zero-root readout "
            "requires the complete single-seed write-ahead ledger"
        )
    if rows[0] != opened_record():
        raise ValueError(
            "benchmark ledger row 1 is not the frozen opened record for "
            f"seed {FROZEN_SEED}; the ledger is incomplete or corrupt"
        )
    if len(rows) == 1:
        raise ValueError(
            "benchmark ledger ends with a crashed opened record; the "
            "zero-root readout requires the seed closed"
        )
    if not is_closed_record(rows[1]):
        raise ValueError(
            f"benchmark ledger row 2 is not the closed record for seed {FROZEN_SEED}"
        )
    if len(rows) != 2:
        raise ValueError(
            "benchmark ledger has rows beyond the frozen single-seed event"
        )
    return rows[1]


def load_zero_root_anchor() -> dict:
    """Anchor the zero-root arm to the committed rebuild merge receipt.

    check_benchmark carries no TODO-PINs: the zero-root arm's identity is
    read from runs/lineage/merge.json (committed by --stage rebuild),
    whose recomputed sha256 the sealed summary must also record.
    """
    if not ZERO_ROOT_MERGE_RECEIPT.is_file():
        raise ValueError(
            f"zero-root rebuild merge receipt is absent: {ZERO_ROOT_MERGE_RECEIPT}"
        )
    payload = json.loads(ZERO_ROOT_MERGE_RECEIPT.read_text(encoding="utf-8"))
    if (
        payload.get("experiment_id") != EXP.name
        or payload.get("stage") != "merge"
        or payload.get("name") != ZERO_ROOT_ARM
        or Path(payload.get("merged", "")).resolve()
        != FROZEN_MODEL_PATHS[ZERO_ROOT_ARM].resolve()
        or not _sha_hex(payload.get("output_tree_sha256"))
        or not _sha_hex(payload.get("weights_sha256"))
        or not _sha_hex(payload.get("inner_merge_receipt_sha256"))
    ):
        raise ValueError(
            "zero-root rebuild merge receipt does not describe the frozen arm"
        )
    return {
        "sha256": sha256_file(ZERO_ROOT_MERGE_RECEIPT),
        "tree_sha256": payload["output_tree_sha256"],
        "weights_sha256": payload["weights_sha256"],
    }


def require_summary_consistency(summary: object, events: dict, anchor: dict) -> None:
    """Receipts must equal the sealed summary's recorded blocks, exactly.

    The summary bytes are pinned by the closed ledger record; requiring
    each authenticated receipt's scores/budget/implementation to equal the
    summary's recorded blocks binds the consequence inputs to the sealed
    event — a receipt swapped after close time cannot both match its
    pinned sha and diverge here. The summary's arm identities must match
    the frozen pins (base/original) and the committed rebuild merge
    receipt (zero-root).
    """
    if (
        not isinstance(summary, dict)
        or summary.get("schema_version") != 1
        or summary.get("name") != FROZEN_NAME
        or summary.get("tier") != FROZEN_TIER
        or summary.get("think_budget") != FROZEN_THINK_BUDGET
        or summary.get("seed") != FROZEN_SEED
        or summary.get("model_order") != list(MODEL_ORDER)
        or summary.get("promoted") is not None
        or summary.get("benchmark_data_read") is not False
        or summary.get("gateway_sha256") != GATEWAY_SHA256
        or summary.get("zero_root_merge_receipt_sha256") != anchor["sha256"]
        or summary.get("model_tree_sha256s", {}).get("base")
        != FROZEN_TREE_SHA256["base"]
        or summary.get("model_tree_sha256s", {}).get(ORIGINAL_ARM)
        != FROZEN_TREE_SHA256[ORIGINAL_ARM]
        or summary.get("model_tree_sha256s", {}).get(ZERO_ROOT_ARM)
        != anchor["tree_sha256"]
        or summary.get("model_weight_sha256s", {}).get("base")
        != FROZEN_WEIGHTS_SHA256["base"]
        or summary.get("model_weight_sha256s", {}).get(ORIGINAL_ARM)
        != FROZEN_WEIGHTS_SHA256[ORIGINAL_ARM]
        or summary.get("model_weight_sha256s", {}).get(ZERO_ROOT_ARM)
        != anchor["weights_sha256"]
    ):
        raise ValueError("sealed summary failed authentication")
    for label in MODEL_ORDER:
        event = events[label]
        if summary.get("scores", {}).get(label) != {
            "aggregate": event["aggregate"],
            "per_family": event["per_family"],
        }:
            raise ValueError(
                f"receipt scores diverge from the sealed summary for arm {label}"
            )
        if summary.get("budget", {}).get(label) != {
            "within_budget": event["within_budget"],
            "wall_seconds": event["wall_seconds"],
        }:
            raise ValueError(
                f"receipt budget diverges from the sealed summary for arm {label}"
            )
        if summary.get("benchmark_implementation") != {
            "runner_sha256": event["benchmark_runner_sha256"],
            "source_inventory_sha256": event["benchmark_source_inventory_sha256"],
            "source_file_count": event["benchmark_source_file_count"],
        }:
            raise ValueError(
                f"receipt implementation diverges from the sealed summary for "
                f"arm {label}"
            )


def load_event(path: Path, model: Path) -> dict:
    """Authenticate one aggregate-gateway receipt against the frozen seed."""
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


def per_family_reading(scores: dict[str, dict]) -> dict:
    """Reading 1: aggregates and the full per-family table, all three arms."""
    return {
        "aggregates": {label: scores[label]["aggregate"] for label in MODEL_ORDER},
        "per_family": {
            label: dict(scores[label]["per_family"]) for label in MODEL_ORDER
        },
    }


def goal_gate_reading(scores: dict[str, dict]) -> dict:
    """Reading 2: the strict-win partition versus base for BOTH composites."""
    base = scores["base"]["per_family"]
    return {
        "statistic": (
            "strict wins / ties / losses over the ten public families, "
            "byte-identical to the tier forensics FAMILIES and strict-win "
            "logic; pass = ten strict wins; exact ties are never strict wins"
        ),
        "per_arm": {
            label: goal_gate_row(base, scores[label]["per_family"])
            for label in COMPOSITE_ARMS
        },
    }


def prefix_contribution_reading(scores: dict[str, dict]) -> dict:
    """Reading 3: zero-root minus original, per family and aggregate."""
    original = scores[ORIGINAL_ARM]
    zero_root = scores[ZERO_ROOT_ARM]
    return {
        "framing": PREFIX_CONTRIBUTION_FRAMING,
        "direction": (
            "zero_root minus original: a NEGATIVE value is capability the "
            "undocumented gym-era root was carrying"
        ),
        "aggregate": zero_root["aggregate"] - original["aggregate"],
        "per_family": {
            family: zero_root["per_family"][family] - original["per_family"][family]
            for family in FAMILIES
        },
    }


def budget_integrity_reading(budget: dict[str, dict]) -> dict:
    """Reading 4: within_budget/wall_seconds per arm; scope only, never a gate."""
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
                f"arms exceeded the gateway budget: {over}; the three-arm "
                "paired comparison is not budget-matched (scores recorded, "
                "not compared)"
            )
        ),
    }


def margins_reading(scores: dict[str, dict]) -> dict:
    """Reading 5: menders/rites/warren margins versus base, both composites."""
    base = scores["base"]["per_family"]
    return {
        "question": (
            "do the fragile single-family margins survive without the "
            "undocumented root: menders and warren carried the original "
            "discovery pass on single-item margins, and rites matters for "
            "the sweep reading"
        ),
        "statechain_note": (
            "the statechain-to-rites conversion question does NOT apply "
            "here: this lineage has no statechain stage"
        ),
        "families": list(MARGIN_FAMILIES),
        "per_arm": {
            label: {
                family: scores[label]["per_family"][family] - base[family]
                for family in MARGIN_FAMILIES
            }
            for label in COMPOSITE_ARMS
        },
    }


def consequence_reading(scores: dict[str, dict]) -> dict:
    """The frozen ordered total partition over the one sealed seed."""
    base = scores["base"]
    original_gate = goal_gate_row(
        base["per_family"], scores[ORIGINAL_ARM]["per_family"]
    )
    zero_root_gate = goal_gate_row(
        base["per_family"], scores[ZERO_ROOT_ARM]["per_family"]
    )
    aggregate_beats_base = (
        scores[ZERO_ROOT_ARM]["aggregate"] > base["aggregate"]
    )
    wins_bar = original_gate["strict_wins"] - 1
    wins_met = zero_root_gate["strict_wins"] >= wins_bar
    consequence = (
        "ZERO_ROOT_COMPARABLE"
        if aggregate_beats_base and wins_met
        else "ZERO_ROOT_DEGRADED"
    )
    return {
        "rule": CONSEQUENCE_RULE,
        "zero_root_aggregate_beats_base": aggregate_beats_base,
        "original_strict_wins": original_gate["strict_wins"],
        "zero_root_strict_wins": zero_root_gate["strict_wins"],
        "required_strict_wins": wins_bar,
        "strict_wins_bar_met": wins_met,
        "consequence": consequence,
        "statement": CONSEQUENCE_STATEMENTS[consequence],
    }


def build_readout(
    scores: dict[str, dict],
    budget: dict[str, dict],
    implementation: dict,
    receipts: dict[str, dict],
    anchor: dict,
    design_receipt_sha256: str,
    summary_sha256: str,
) -> dict:
    """Assemble the readout from pure inputs (unit-testable, no file IO)."""
    if implementation != REFERENCE_IMPLEMENTATION:
        raise ValueError(
            "benchmark implementation differs from the pinned reference "
            f"events ({implementation} != {REFERENCE_IMPLEMENTATION}); the "
            "cross-event framing is not comparable"
        )
    integrity = budget_integrity_reading(budget)
    consequence = consequence_reading(scores)
    return {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "stage": "zero_root_readout",
        "name": FROZEN_NAME,
        "tier": FROZEN_TIER,
        "think_budget": FROZEN_THINK_BUDGET,
        "seed": FROZEN_SEED,
        "benchmark_data_read": False,
        "promoted": None,
        "outcome": "ZERO_ROOT_READ_COMPLETE",
        "consequence": consequence["consequence"],
        "consequence_statement": consequence["statement"],
        "paired_comparison_valid": integrity["paired_comparison_valid"],
        "design_receipt_sha256": design_receipt_sha256,
        "summary_sha256": summary_sha256,
        "zero_root_anchor": {
            "merge_receipt": "runs/lineage/merge.json",
            "merge_receipt_sha256": anchor["sha256"],
            "tree_sha256": anchor["tree_sha256"],
            "weights_sha256": anchor["weights_sha256"],
        },
        "provenance": {
            "ledger": "runs/benchmark_events.jsonl",
            "ledger_complete_sequence_required": True,
            "receipt_sha256s_pinned_in_closed_record": True,
            "summary_verified_against_ledger_pin": True,
            "receipts_verified_against_ledger_pins": True,
            "receipt_blocks_verified_against_sealed_summary": True,
            "zero_root_arm_anchored_to_committed_merge_receipt": True,
        },
        "benchmark_implementation": {
            "signature": dict(implementation),
            "reference": dict(REFERENCE_IMPLEMENTATION),
            "identical_across_all_three_receipts_and_reference": True,
        },
        "receipts": receipts,
        "scores": {label: scores[label] for label in MODEL_ORDER},
        "budget": {label: budget[label] for label in MODEL_ORDER},
        "readings": {
            "per_family": per_family_reading(scores),
            "goal_gate": goal_gate_reading(scores),
            "prefix_contribution": prefix_contribution_reading(scores),
            "budget_integrity": integrity,
            "margins": margins_reading(scores),
            "consequence": consequence,
        },
    }


def render() -> bytes:
    """Authenticate every input through the ledger, then render the readout.

    Order of anchoring: the complete ledger first (opened then closed,
    nothing trailing), then the sealed summary against its pinned sha,
    then each receipt against the sha its closed record pinned at close
    time, then structural receipt authentication, then receipt-versus-
    summary block equality, then the zero-root anchor to the committed
    rebuild merge receipt. Only inputs that survive all layers reach the
    consequence.
    """
    anchor = load_zero_root_anchor()
    if not DESIGN_RECEIPT.is_file():
        raise ValueError("design receipt is absent; readout stays unwritten")
    rows = (
        [
            json.loads(line)
            for line in LEDGER.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if LEDGER.is_file()
        else []
    )
    closed = authenticate_ledger(rows)
    design_receipt_sha256 = sha256_file(DESIGN_RECEIPT)
    summary_path = EVENT_DIR / "summary.json"
    if (
        not summary_path.is_file()
        or sha256_file(summary_path) != closed["summary_sha256"]
    ):
        raise ValueError(
            "sealed summary is absent or does not match its closed ledger pin"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("design_receipt_sha256") != design_receipt_sha256:
        raise ValueError(
            "sealed summary was produced under a different design receipt"
        )
    events = {}
    receipts = {}
    signatures = set()
    for label in MODEL_ORDER:
        path = EVENT_DIR / f"{label}.json"
        if not path.is_file():
            raise ValueError(f"gateway receipt is absent for arm {label}: {path}")
        digest = sha256_file(path)
        if digest != closed["receipts"][label]:
            raise ValueError(
                f"gateway receipt does not match its closed ledger pin for "
                f"arm {label}"
            )
        events[label] = load_event(path, FROZEN_MODEL_PATHS[label])
        receipts[label] = {
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
    require_summary_consistency(summary, events, anchor)
    if len(signatures) != 1:
        raise ValueError("benchmark implementation changed between the three receipts")
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
        receipts,
        anchor,
        design_receipt_sha256,
        closed["summary_sha256"],
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
            parser.error("refusing to overwrite zero-root readout")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(value)
        target = args.out
    else:
        if not READOUT.is_file() or READOUT.read_bytes() != value:
            parser.error("published zero-root readout is absent or changed")
        target = READOUT
    payload = json.loads(value.decode("utf-8"))
    print(
        json.dumps(
            {
                "out": str(target),
                "sha256": hashlib.sha256(value).hexdigest(),
                "outcome": "ZERO_ROOT_READ_COMPLETE",
                "consequence": payload["consequence"],
                "statement": payload["consequence_statement"],
            },
            indent=1,
            sort_keys=True,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
