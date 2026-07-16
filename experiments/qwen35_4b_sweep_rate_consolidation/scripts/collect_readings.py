#!/usr/bin/env python3
"""Collect the program's SIX all-time goal-gate readings from committed receipts.

Analysis-only terminal bookkeeping: no model, no GPU, no seed is consumed,
and the `benchmarks/` directory is never read. The inputs are the six
committed benchmark summaries in which the headline hygiene_explore
composite was paired against base at a sealed medium/tb1024 seed — the
complete all-time record of goal-gate events for that composite. Each
summary is byte-copied into `data/source_summaries/` so this cell
self-contains its inputs; this script reads the LOCAL copies, verifies
them against the hard-pinned sha256s AND against the original files when
those experiment directories are still present, recomputes each goal gate
from the per_family scores (strict wins/ties/losses vs base, FAMILIES
byte-identical to the tier-forensics analyzer), and cross-checks against
any goal-gate block the summary already records (they must agree; the
script aborts otherwise).

Output: `runs/readings_table.json` — one row per seed with aggregates,
per-family scores and deltas, wins/ties/losses, blockers, and the
provenance sha. The companion `analyze_sweep_rate.py` consumes this
table; this script only collects.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
OUT = EXP / "runs" / "readings_table.json"

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

# The six goal-gate readings, hard-pinned. `original` is the committed
# summary in its source experiment; `local` is this cell's byte-identical
# copy; `sha256` is the provenance pin both must match.
READINGS = (
    {
        "seed": 78150,
        "arm": "hygiene_explore",
        "source_experiment": "qwen35_4b_universal_medium_tier_measurement",
        "original": "experiments/qwen35_4b_universal_medium_tier_measurement/runs/benchmark/medium_tb1024_seed78150_measurement/summary.json",
        "local": "data/source_summaries/medium_tb1024_seed78150_measurement_summary.json",
        "sha256": "a927fc838ca8b1eaa3083d6034ba09ad0659c21a2a13b22c525487cf95a6fb43",
    },
    {
        "seed": 78154,
        "arm": "hygiene_explore_parent",
        "source_experiment": "qwen35_4b_statechain_only_dose",
        "original": "experiments/qwen35_4b_statechain_only_dose/runs/benchmark/medium_tb1024_seed78154_pilot/summary.json",
        "local": "data/source_summaries/medium_tb1024_seed78154_pilot_summary.json",
        "sha256": "6b1a43869f013e24a048a45a04e5603b45fe59488912194eb3e76a43679255fa",
    },
    {
        "seed": 78155,
        "arm": "hygiene_explore",
        "source_experiment": "qwen35_4b_goal_gate_confirmation",
        "original": "experiments/qwen35_4b_goal_gate_confirmation/runs/benchmark/medium_tb1024_seed78155_confirmation/summary.json",
        "local": "data/source_summaries/medium_tb1024_seed78155_confirmation_summary.json",
        "sha256": "482260548d936f6ddd51401328861fd99a67be044f917ffee917348e41b3123b",
    },
    {
        "seed": 78156,
        "arm": "hygiene_explore",
        "source_experiment": "qwen35_4b_goal_gate_confirmation",
        "original": "experiments/qwen35_4b_goal_gate_confirmation/runs/benchmark/medium_tb1024_seed78156_confirmation/summary.json",
        "local": "data/source_summaries/medium_tb1024_seed78156_confirmation_summary.json",
        "sha256": "604b755497a104b3f0337a1c25a36b6996c4c5ccd01ae9ed9e0e9041747fd19a",
    },
    {
        "seed": 78157,
        "arm": "hygiene_explore",
        "source_experiment": "qwen35_4b_goal_gate_confirmation",
        "original": "experiments/qwen35_4b_goal_gate_confirmation/runs/benchmark/medium_tb1024_seed78157_confirmation/summary.json",
        "local": "data/source_summaries/medium_tb1024_seed78157_confirmation_summary.json",
        "sha256": "0ac2c412cc09375446cc1fcee594aedf96bcd7ef9cd6a2214d6b30cf195e0fa3",
    },
    {
        "seed": 78159,
        "arm": "hygiene_explore_original",
        "source_experiment": "qwen35_4b_zero_root_lineage_rebuild",
        "original": "experiments/qwen35_4b_zero_root_lineage_rebuild/runs/benchmark/medium_tb1024_seed78159_zero_root/summary.json",
        "local": "data/source_summaries/medium_tb1024_seed78159_zero_root_summary.json",
        "sha256": "c83586f0bf1e98cf0e01ebf3918f3d28c98ae8bee7d8f9361dcbcaaf83da8b4d",
    },
)


class ProvenanceError(SystemExit):
    """Raised (fail closed) when a pinned input drifts."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_pinned_summary(spec: dict, exp: Path = EXP, repo: Path = REPO) -> dict:
    """Read the local copy, enforce both provenance pins, return the payload.

    Fail-closed rules:
    - the local copy must exist and hash to the pinned sha256;
    - when the original file still exists in its source experiment, it
      must hash to the same pin (byte-identical to the local copy). A
      missing original is allowed — that is the standalone point of the
      local copies — but a PRESENT-and-different original is drift.
    """
    local = exp / spec["local"]
    if not local.is_file():
        raise ProvenanceError(f"missing pinned local copy: {local}")
    data = local.read_bytes()
    digest = sha256_bytes(data)
    if digest != spec["sha256"]:
        raise ProvenanceError(
            f"provenance drift in {spec['local']}: pinned {spec['sha256']}, "
            f"found {digest}"
        )
    original = repo / spec["original"]
    original_present = original.is_file()
    if original_present:
        original_digest = sha256_bytes(original.read_bytes())
        if original_digest != spec["sha256"]:
            raise ProvenanceError(
                f"provenance drift in original {spec['original']}: pinned "
                f"{spec['sha256']}, found {original_digest}"
            )
    payload = json.loads(data.decode("utf-8"))
    payload["__original_present__"] = original_present
    return payload


def extract_scores(payload: dict, arm: str, seed: int, spec_path: str) -> tuple[dict, dict]:
    """Return (base_block, arm_block); validate the event's identity."""
    if payload.get("tier") != "medium":
        raise ProvenanceError(f"{spec_path}: tier is {payload.get('tier')!r}, not medium")
    if payload.get("think_budget") != 1024:
        raise ProvenanceError(
            f"{spec_path}: think_budget is {payload.get('think_budget')!r}, not 1024"
        )
    if payload.get("seed") != seed:
        raise ProvenanceError(f"{spec_path}: seed is {payload.get('seed')!r}, not {seed}")
    scores = payload.get("scores")
    if not isinstance(scores, dict):
        raise ProvenanceError(f"{spec_path}: no scores block")
    for label in ("base", arm):
        block = scores.get(label)
        if not isinstance(block, dict) or not isinstance(block.get("per_family"), dict):
            raise ProvenanceError(f"{spec_path}: arm {label!r} missing per_family scores")
        per_family = block["per_family"]
        for family in FAMILIES:
            value = per_family.get(family)
            if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
                raise ProvenanceError(
                    f"{spec_path}: arm {label!r} family {family!r} is not a score in [0, 1]: {value!r}"
                )
        if not isinstance(block.get("aggregate"), (int, float)):
            raise ProvenanceError(f"{spec_path}: arm {label!r} missing aggregate")
    return scores["base"], scores[arm]


def goal_gate(base_families: dict, arm_families: dict) -> dict:
    """Strict-win goal gate, FAMILIES semantics identical to the forensics analyzer."""
    wins = [f for f in FAMILIES if arm_families[f] > base_families[f]]
    losses = [f for f in FAMILIES if arm_families[f] < base_families[f]]
    ties = [f for f in FAMILIES if arm_families[f] == base_families[f]]
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "strict_wins": len(wins),
        "goal_gate_pass": len(wins) == len(FAMILIES),
    }


def cross_check_recorded_gate(payload: dict, arm: str, computed: dict, spec_path: str) -> str:
    """Compare the recomputed gate to any goal-gate block already in the summary.

    Returns "agrees" when a recorded block exists and matches, "absent"
    when the summary records none for this arm; aborts on disagreement.
    """
    recorded = payload.get("goal_gate")
    if not isinstance(recorded, dict):
        return "absent"
    per_arm = recorded.get("per_arm")
    if not isinstance(per_arm, dict) or arm not in per_arm:
        return "absent"
    block = per_arm[arm]
    mismatches = []
    for key, cast in (
        ("wins", sorted),
        ("losses", sorted),
        ("ties", sorted),
        ("strict_wins", None),
        ("goal_gate_pass", None),
    ):
        if key not in block:
            continue
        recorded_value = cast(block[key]) if cast else block[key]
        computed_value = cast(computed[key]) if cast else computed[key]
        if recorded_value != computed_value:
            mismatches.append(f"{key}: recorded {recorded_value!r} != computed {computed_value!r}")
    if mismatches:
        raise ProvenanceError(
            f"{spec_path}: recorded goal_gate block disagrees with recomputation for "
            f"arm {arm!r}: " + "; ".join(mismatches)
        )
    return "agrees"


def compute_reading(spec: dict, payload: dict) -> dict:
    base, arm = extract_scores(payload, spec["arm"], spec["seed"], spec["local"])
    base_families = {f: float(base["per_family"][f]) for f in FAMILIES}
    arm_families = {f: float(arm["per_family"][f]) for f in FAMILIES}
    gate = goal_gate(base_families, arm_families)
    cross_check = cross_check_recorded_gate(payload, spec["arm"], gate, spec["local"])
    return {
        "seed": spec["seed"],
        "arm": spec["arm"],
        "source_experiment": spec["source_experiment"],
        "original_path": spec["original"],
        "original_present": bool(payload.get("__original_present__", False)),
        "local_copy": spec["local"],
        "summary_sha256": spec["sha256"],
        "tier": "medium",
        "think_budget": 1024,
        "base_aggregate": float(base["aggregate"]),
        "arm_aggregate": float(arm["aggregate"]),
        "aggregate_delta": float(arm["aggregate"]) - float(base["aggregate"]),
        "base_per_family": base_families,
        "arm_per_family": arm_families,
        "per_family_delta": {f: arm_families[f] - base_families[f] for f in FAMILIES},
        "wins": gate["wins"],
        "losses": gate["losses"],
        "ties": gate["ties"],
        "strict_wins": gate["strict_wins"],
        "goal_gate_pass": gate["goal_gate_pass"],
        "blockers": sorted(gate["ties"] + gate["losses"]),
        "recorded_goal_gate_cross_check": cross_check,
    }


def build_table(exp: Path = EXP, repo: Path = REPO) -> dict:
    rows = []
    for spec in sorted(READINGS, key=lambda s: s["seed"]):
        payload = load_pinned_summary(spec, exp=exp, repo=repo)
        rows.append(compute_reading(spec, payload))
    seeds = [row["seed"] for row in rows]
    if len(set(seeds)) != len(READINGS):
        raise ProvenanceError(f"duplicate seeds in READINGS: {seeds}")
    return {
        "schema_version": 1,
        "benchmark_data_read": False,
        "source": (
            "the six committed hygiene_explore-vs-base goal-gate summaries at "
            "sealed medium/tb1024 seeds, byte-copied into data/source_summaries "
            "and verified against hard-pinned sha256s (and against the originals "
            "when their experiment directories are present)"
        ),
        "families": list(FAMILIES),
        "readings": rows,
    }


def serialize(table: dict) -> str:
    return (
        json.dumps(table, indent=1, ensure_ascii=False, sort_keys=True) + "\n"
    )


def main() -> int:
    table = build_table()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(serialize(table), encoding="utf-8")
    passes = sum(1 for row in table["readings"] if row["goal_gate_pass"])
    print(
        f"collected {len(table['readings'])} goal-gate readings "
        f"({passes} passes) -> {OUT}"
    )
    for row in table["readings"]:
        print(
            f"  seed {row['seed']}: {row['strict_wins']}/10"
            f" ties={row['ties']} losses={row['losses']}"
            f" cross-check={row['recorded_goal_gate_cross_check']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
