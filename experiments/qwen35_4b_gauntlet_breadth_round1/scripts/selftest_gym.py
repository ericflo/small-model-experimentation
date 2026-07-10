#!/usr/bin/env python3
"""Run every gym family selftest (CPU-only, any python3)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from gym.families import ALL_FAMILIES, load  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", nargs="*", default=list(ALL_FAMILIES))
    args = parser.parse_args()

    failures = []
    for name in args.families:
        started = time.perf_counter()
        try:
            module = load(name)
            stats = module.selftest()
            elapsed = time.perf_counter() - started
            print(f"PASS {name} ({elapsed:.1f}s): {json.dumps(stats)}", flush=True)
        except Exception as exc:  # noqa: BLE001 - report and continue
            elapsed = time.perf_counter() - started
            print(f"FAIL {name} ({elapsed:.1f}s): {exc}", flush=True)
            failures.append(name)
    if failures:
        print(f"\n{len(failures)} failing families: {', '.join(failures)}")
        return 1
    print(f"\nall {len(args.families)} families pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
