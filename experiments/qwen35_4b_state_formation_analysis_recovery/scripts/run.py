#!/usr/bin/env python3
"""Run the frozen state-formation analysis recovery consumer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.recovery import PHASE_OUTPUTS, run_analysis, run_smoke  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", action="store_true")
    group.add_argument("--phase", choices=tuple(PHASE_OUTPUTS))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_smoke() if args.smoke else run_analysis(args.phase)
    public = {
        key: result[key]
        for key in (
            "experiment_id",
            "status",
            "receipt_identity_sha256",
            "phase",
            "producer_status",
            "producer_verdict",
            "producer_next_stage",
        )
        if key in result
    }
    print(json.dumps(public, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
