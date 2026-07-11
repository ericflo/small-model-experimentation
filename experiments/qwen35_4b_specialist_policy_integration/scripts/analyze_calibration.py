#!/usr/bin/env python3
"""Freeze the incumbent compound-headroom gate before specialist training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from io_utils import load_config, write_json  # noqa: E402


COMPOUNDS = ("cipherkiln", "mazeferry", "patchferry", "tripleforge")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=EXP / "analysis" / "calibration_gate.json")
    args = parser.parse_args()
    config, _ = load_config(args.config)
    result = json.loads(args.scores.read_text(encoding="utf-8"))
    by_family = result["episode_summary"]["by_family"]
    missing = set(COMPOUNDS) - set(by_family)
    if missing:
        raise SystemExit(f"calibration output missing compound families: {sorted(missing)}")
    means = {family: float(by_family[family]["mean_score"]) for family in COMPOUNDS}
    macro = sum(means.values()) / len(means)
    threshold = float(config["gates"]["calibration_max_incumbent_compound_score"])
    payload = {
        "stage": "incumbent_compound_calibration",
        "scores": str(args.scores.resolve()),
        "compound_family_scores": means,
        "compound_macro_score": macro,
        "threshold_exclusive": threshold,
        "gate": {"passed": macro < threshold},
        "frozen_confirmatory_levels": list(config["proxy_eval"]["levels"]),
    }
    write_json(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["gate"]["passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
