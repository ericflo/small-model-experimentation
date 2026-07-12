#!/usr/bin/env python3
"""Select one recovery arm using only the registered trained-family calibration block."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

EXP = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if payload["block"] != "calibration" or payload["scenario_set"] != "recovery":
        raise SystemExit(f"not a recovery calibration result: {path}")
    return payload


def transition_score(aggregate: dict) -> float:
    scenarios = aggregate["per_scenario"]
    return (
        float(scenarios["rejected_patch"]["immediate_transition_rate"])
        + float(scenarios["failed_test"]["immediate_transition_rate"])
        + float(scenarios["failed_test"]["changed_patch_within_two"])
    ) / 3.0


def score(payload: dict) -> tuple[float, float, float]:
    aggregate = payload["aggregate"]
    scenario_success = [
        float(row["success"]) for row in aggregate["per_scenario"].values()
    ]
    return (
        float(aggregate["success"]),
        min(scenario_success),
        transition_score(aggregate),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXP / "configs" / "default.yaml")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--happy", type=Path, required=True)
    parser.add_argument("--action", type=Path, required=True)
    parser.add_argument("--reason", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    gates = cfg["selection_gates"]
    payloads = {
        "base": load(args.base),
        "happy_action": load(args.happy),
        "recovery_action": load(args.action),
        "recovery_reason": load(args.reason),
    }
    manifests = {item["task_manifest_sha256"] for item in payloads.values()}
    if len(manifests) != 1:
        raise SystemExit("calibration manifests differ")
    # Lexicographic score is frozen. Exact ties prefer action-only recovery to
    # avoid adding plan supervision without evidence that it helps.
    candidates = ("recovery_action", "recovery_reason")
    selected = max(candidates, key=lambda name: (score(payloads[name]), name == "recovery_action"))
    selected_aggregate = payloads[selected]["aggregate"]
    base_aggregate = payloads["base"]["aggregate"]
    happy_aggregate = payloads["happy_action"]["aggregate"]
    success_delta_base = selected_aggregate["success"] - base_aggregate["success"]
    success_delta_happy = selected_aggregate["success"] - happy_aggregate["success"]
    transition_delta_happy = (
        transition_score(selected_aggregate) - transition_score(happy_aggregate)
    )
    checks = {
        "success_not_worse_than_base": (
            success_delta_base >= float(gates["success_delta_vs_base_min"])
        ),
        "conditional_mechanism_signal": (
            success_delta_happy >= float(gates["success_delta_vs_happy_min"])
            or transition_delta_happy >= float(gates["transition_delta_vs_happy_min"])
        ),
    }
    result = {
        "schema_version": 1,
        "selection_block": "calibration",
        "selected_arm": selected,
        "rule": "lexicographic(overall_success,min_scenario_success,transition_score); ties prefer recovery_action",
        "scores": {name: score(payload) for name, payload in payloads.items()},
        "success_delta_vs_base": success_delta_base,
        "success_delta_vs_happy": success_delta_happy,
        "transition_delta_vs_happy": transition_delta_happy,
        "checks": checks,
        "gate": {"passed": all(checks.values())},
        "inputs": {name: str(path.resolve()) for name, path in {
            "base": args.base, "happy_action": args.happy,
            "recovery_action": args.action, "recovery_reason": args.reason,
        }.items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["gate"]["passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
