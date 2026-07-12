#!/usr/bin/env python3
"""Fail closed when a frozen specialist gain bar exceeds score headroom."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from io_utils import domain_families, load_config, write_json  # noqa: E402


DOMAINS = ("discover", "control", "tools", "compose")
SCORE_CEILING = 1.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=EXP / "analysis" / "specialist_headroom_gate.json",
    )
    args = parser.parse_args()
    config, _ = load_config(args.config)
    scores = json.loads(args.scores.read_text(encoding="utf-8"))
    expected_families = list(config["split"]["train_families"]) + list(
        config["split"]["transfer_families"]
    )
    protocol_checks = {
        "scope": scores.get("scope") == "calibration",
        "decode": scores.get("decode") == "greedy",
        "families": scores.get("families") == expected_families,
        "levels": scores.get("levels")
        == [int(value) for value in config["proxy_eval"]["levels"]],
        "episodes_per_level": int(scores.get("episodes_per_level", -1))
        == int(config["proxy_eval"]["calibration_episodes_per_level"]),
        "episode_seed_base": int(scores.get("episode_seed_base", -1))
        == int(config["seeds"]["proxy_eval_base"]),
        "atoms_enabled": scores.get("atoms_enabled") is True,
    }
    by_family = scores.get("episode_summary", {}).get("by_family", {})
    required_delta = float(config["gates"]["specialist_incumbent_delta"])
    domains = {}
    for domain in DOMAINS:
        families = domain_families(config, domain)
        missing = set(families) - set(by_family)
        if missing:
            raise SystemExit(f"baseline missing {domain} families: {sorted(missing)}")
        baseline = sum(float(by_family[name]["mean_score"]) for name in families) / len(
            families
        )
        max_gain = SCORE_CEILING - baseline
        domains[domain] = {
            "families": families,
            "incumbent_macro": baseline,
            "required_absolute_score": baseline + required_delta,
            "score_ceiling": SCORE_CEILING,
            "maximum_possible_gain": max_gain,
            "required_gain": required_delta,
            "feasible": max_gain + 1e-12 >= required_delta,
        }
    impossible = [name for name in DOMAINS if not domains[name]["feasible"]]
    passed = all(protocol_checks.values()) and not impossible
    result = {
        "stage": "specialist_headroom_gate",
        "scores": str(args.scores.resolve()),
        "score_contract": "all registered environment scores are bounded in [0, 1]",
        "protocol_checks": protocol_checks,
        "domains": domains,
        "impossible_domains": impossible,
        "gate": {"passed": passed},
        "downstream_authorization": (
            "best8_then_specialist_production"
            if passed
            else "stop_before_best8_and_specialist_production"
        ),
        "interpretation": (
            "all frozen pass-one gain bars have sufficient score headroom"
            if passed
            else "at least one frozen pass-one gain bar is mathematically unreachable"
        ),
    }
    write_json(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 4


if __name__ == "__main__":
    raise SystemExit(main())
