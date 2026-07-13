#!/usr/bin/env python3
"""Qualify replicated non-saturated semantic-policy axes without training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

EXP = Path(__file__).resolve().parents[1]


def load(path: Path, block: str) -> dict:
    payload = json.loads(path.read_text())
    if payload["block"] != block or payload["scenario_set"] != "recovery":
        raise SystemExit(f"wrong headroom receipt: {path}")
    return payload


def subset_success(payload: dict, families: set[str], scenario: str) -> float:
    rows = [
        row for row in payload["aggregate"]["cases"]
        if row["family"] in families and row["scenario"] == scenario
    ]
    if not rows:
        raise SystemExit((families, scenario))
    return sum(float(row["success"]) for row in rows) / len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXP / "configs" / "default.yaml")
    parser.add_argument("--block-a", type=Path, required=True)
    parser.add_argument("--block-b", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    gates = cfg["qualification"]
    families = cfg["families"]
    blocks = {
        "headroom_a": load(args.block_a, "headroom_a"),
        "headroom_b": load(args.block_b, "headroom_b"),
    }
    if blocks["headroom_a"]["task_content_manifest_sha256"] == blocks["headroom_b"]["task_content_manifest_sha256"]:
        raise SystemExit("headroom blocks have identical content manifests")

    explicit = set(families["explicit_controls"])
    axis_families = {
        axis: set(families[f"inferred_{axis}"])
        for axis in ("negative", "noninteger", "blank")
    }
    lower = float(gates["inferred_axis_failed_test_success_min"])
    upper = float(gates["inferred_axis_failed_test_success_max"])
    minimum_shapes = int(gates["minimum_shapes_in_band_per_axis_per_block"])
    block_metrics = {}
    for block_name, payload in blocks.items():
        per_family_failed = {
            family: subset_success(payload, {family}, "failed_test")
            for family in families["all_headroom"]
        }
        axes = {}
        for axis, members in axis_families.items():
            failed_macro = subset_success(payload, members, "failed_test")
            rejected_macro = subset_success(payload, members, "rejected_patch")
            shapes_in_band = sum(
                lower <= per_family_failed[family] <= upper for family in members
            )
            axes[axis] = {
                "failed_test_success": failed_macro,
                "rejected_patch_success": rejected_macro,
                "per_family_failed_test_success": {
                    family: per_family_failed[family] for family in sorted(members)
                },
                "shapes_in_band": shapes_in_band,
                "eligible_in_block": (
                    lower <= failed_macro <= upper and shapes_in_band >= minimum_shapes
                ),
            }
        block_metrics[block_name] = {
            "task_manifest_sha256": payload["task_manifest_sha256"],
            "task_content_manifest_sha256": payload["task_content_manifest_sha256"],
            "n_cases": payload["aggregate"]["n_cases"],
            "explicit_failed_test_success": subset_success(payload, explicit, "failed_test"),
            "invalid_action_rate_per_turn": payload["aggregate"]["invalid_action_rate_per_turn"],
            "answer_cap_hit_rate_per_turn": payload["aggregate"]["answer_cap_hit_rate_per_turn"],
            "axes": axes,
        }

    eligible_axes = [
        axis for axis in axis_families
        if all(block_metrics[name]["axes"][axis]["eligible_in_block"] for name in blocks)
    ]
    checks = {
        "content_disjoint": (
            block_metrics["headroom_a"]["task_content_manifest_sha256"]
            != block_metrics["headroom_b"]["task_content_manifest_sha256"]
        ),
        "explicit_controls": all(
            row["explicit_failed_test_success"]
            >= float(gates["explicit_failed_test_success_min"])
            for row in block_metrics.values()
        ),
        "invalid_actions": all(
            float(row["invalid_action_rate_per_turn"])
            <= float(gates["invalid_action_rate_per_turn_max"])
            for row in block_metrics.values()
        ),
        "answer_cap": all(
            float(row["answer_cap_hit_rate_per_turn"])
            <= float(gates["answer_cap_hit_rate_per_turn_max"])
            for row in block_metrics.values()
        ),
        "eligible_axis_count": len(eligible_axes) >= int(gates["minimum_eligible_axes"]),
    }
    result = {
        "schema_version": 1,
        "stage": "semantic_policy_headroom_qualification",
        "model": cfg["model"],
        "eligibility_band": [lower, upper],
        "minimum_shapes_in_band_per_axis_per_block": minimum_shapes,
        "blocks": block_metrics,
        "eligible_axes": eligible_axes,
        "eligible_families": {
            axis: sorted(axis_families[axis]) for axis in eligible_axes
        },
        "checks": checks,
        "gate": {"passed": all(checks.values())},
        "menagerie_authorized": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["gate"]["passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
