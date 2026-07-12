#!/usr/bin/env python3
"""Select one locality-screened interpolation on the frozen calibration block."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

EXP = Path(__file__).resolve().parents[1]


def named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("candidate must be NAME=PATH")
    name, raw = value.split("=", 1)
    return name, Path(raw)


def load_eval(path: Path) -> dict:
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


def candidate_lambda(name: str) -> float:
    if name == "action_anchor":
        return 0.0
    prefix = "reason_mix_"
    if not name.startswith(prefix):
        raise SystemExit(f"unexpected candidate name: {name}")
    return int(name[len(prefix):]) / 100.0


def score(payload: dict, name: str) -> tuple[float, float, float, float, float]:
    aggregate = payload["aggregate"]
    scenario_success = [
        float(row["success"]) for row in aggregate["per_scenario"].values()
    ]
    return (
        float(aggregate["success"]),
        min(scenario_success),
        -float(aggregate["invalid_action_rate_per_turn"]),
        transition_score(aggregate),
        -candidate_lambda(name),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXP / "configs" / "default.yaml")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--happy", type=Path, required=True)
    parser.add_argument("--candidate", action="append", type=named_path, required=True)
    parser.add_argument("--locality-screen", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    gates = cfg["selection_gates"]
    base = load_eval(args.base)
    happy = load_eval(args.happy)
    candidates = {name: load_eval(path) for name, path in args.candidate}
    if len(candidates) != len(args.candidate):
        raise SystemExit("candidate names must be unique")
    expected_names = {"action_anchor"} | {
        f"reason_mix_{round(float(value) * 100):03d}"
        for value in cfg["interpolation"]["candidate_lambdas"]
    }
    locality = json.loads(args.locality_screen.read_text())
    if set(locality["eligible_candidates"]) != expected_names:
        raise SystemExit("locality eligible set differs from preregistered candidates")
    expected_evaluated = set(locality["passing_eligible_candidates"])
    if set(candidates) != expected_evaluated:
        raise SystemExit(
            f"behavior inputs must be exactly the locality-pass set: "
            f"{sorted(candidates)} != {sorted(expected_evaluated)}"
        )
    for name, payload in candidates.items():
        expected_arm = "recovery_action" if name == "action_anchor" else name
        if payload.get("arm") != expected_arm:
            raise SystemExit(f"candidate arm mismatch for {name}: {payload.get('arm')}")
        locality_model = Path(locality["candidates"][name]["model"]).resolve()
        behavior_model = Path(payload["model"]).resolve()
        if locality_model != behavior_model:
            raise SystemExit(f"locality/behavior model mismatch for {name}")
        if name != "action_anchor":
            receipt = json.loads((behavior_model / "interpolation_receipt.json").read_text())
            if abs(float(receipt["reason_lambda"]) - candidate_lambda(name)) > 1e-12:
                raise SystemExit(f"interpolation receipt lambda mismatch for {name}")
    manifests = {
        payload["task_manifest_sha256"] for payload in (base, happy, *candidates.values())
    }
    if len(manifests) != 1:
        raise SystemExit("calibration task manifests differ")
    base_aggregate = base["aggregate"]
    happy_aggregate = happy["aggregate"]
    details = {}
    for name, payload in candidates.items():
        aggregate = payload["aggregate"]
        success_delta_base = float(aggregate["success"]) - float(base_aggregate["success"])
        success_delta_happy = float(aggregate["success"]) - float(happy_aggregate["success"])
        transition_delta_happy = transition_score(aggregate) - transition_score(happy_aggregate)
        invalid_delta_base = (
            float(aggregate["invalid_action_rate_per_turn"])
            - float(base_aggregate["invalid_action_rate_per_turn"])
        )
        scenarios = aggregate["per_scenario"]
        checks = {
            "locality_screen": bool(locality["candidates"][name]["gate"]["passed"]),
            "recovery_vs_base": success_delta_base >= float(
                gates["recovery_delta_vs_base_min"]
            ),
            "conditional_signal_vs_happy": (
                success_delta_happy >= float(gates["recovery_delta_vs_happy_min"])
                or transition_delta_happy >= float(gates["transition_delta_vs_happy_min"])
            ),
            "invalid_action_retention": invalid_delta_base <= float(
                gates["invalid_action_delta_vs_base_max"]
            ),
            "rejected_patch_transition": float(
                scenarios["rejected_patch"]["immediate_transition_rate"]
            ) >= float(gates["rejected_patch_transition_absolute_min"]),
            "failed_test_changed_patch": float(
                scenarios["failed_test"]["changed_patch_within_two"]
            ) >= float(gates["failed_test_changed_patch_within_two_absolute_min"]),
        }
        details[name] = {
            "reason_lambda": candidate_lambda(name),
            "model": payload.get("model"),
            "score": score(payload, name),
            "success_delta_vs_base": success_delta_base,
            "success_delta_vs_happy": success_delta_happy,
            "transition_delta_vs_happy": transition_delta_happy,
            "invalid_action_delta_vs_base": invalid_delta_base,
            "checks": checks,
            "eligible": all(checks.values()),
        }
    eligible = [name for name, item in details.items() if item["eligible"]]
    selected = max(eligible, key=lambda name: score(candidates[name], name)) if eligible else None
    result = {
        "schema_version": 1,
        "selection_block": "calibration",
        "rule": (
            "filter by locality, recovery, invalid-action, and transition gates; then "
            "lexicographic(overall success, worst-scenario success, -invalid rate, "
            "transition score, -reason lambda)"
        ),
        "task_manifest_sha256": next(iter(manifests)),
        "eligible_candidates": eligible,
        "selected_arm": selected,
        "selected_model": details[selected]["model"] if selected else None,
        "candidates": details,
        "gate": {"passed": selected is not None},
        "inputs": {
            "base": str(args.base.resolve()),
            "happy": str(args.happy.resolve()),
            "locality_screen": str(args.locality_screen.resolve()),
            "candidates": {
                name: str(path.resolve()) for name, path in args.candidate
            },
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["gate"]["passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
