#!/usr/bin/env python3
"""Harness for the analysis-only sweep-rate consolidation cell.

`--smoke`: verify the six provenance pins (local copies hash to the
hard-pinned sha256s, and the originals — when their experiment
directories are present — hash to the same pins), then re-derive the
analysis from the committed readings table and require byte-identity
with the committed `runs/sweep_rate_analysis.json`. No model event
exists in this cell.

`--full`: additionally re-collect the readings table from the pinned
summaries (recompute + cross-check every goal gate) and require the
regenerated `runs/readings_table.json` to match the committed table
byte-for-byte, then re-verify the analysis. Still zero GPU, zero seeds,
`benchmarks/` never read.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
SCRIPTS = EXP / "scripts"
TABLE = EXP / "runs" / "readings_table.json"
ANALYSIS = EXP / "runs" / "sweep_rate_analysis.json"

sys.path.insert(0, str(SCRIPTS))
import collect_readings  # noqa: E402


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_pins() -> None:
    for spec in collect_readings.READINGS:
        collect_readings.load_pinned_summary(spec)
    print(f"provenance pins verified for all {len(collect_readings.READINGS)} summaries")


def rerun_and_compare(script: str, target: Path, extra: tuple[str, ...] = ()) -> None:
    if not target.exists():
        raise SystemExit(f"missing committed artifact: {target}")
    before = sha256_file(target)
    with tempfile.TemporaryDirectory() as scratch:
        backup = Path(scratch) / target.name
        backup.write_bytes(target.read_bytes())
        try:
            subprocess.run(
                [sys.executable, "-B", str(SCRIPTS / script), *extra],
                check=True,
                capture_output=True,
            )
            after = sha256_file(target)
        finally:
            target.write_bytes(backup.read_bytes())
    if after != before:
        raise SystemExit(
            f"{script} no longer reproduces the committed {target.name}: "
            f"committed {before}, regenerated {after}"
        )
    print(f"{script} reproduces {target.name} byte-identically")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", action="store_true")
    group.add_argument("--full", action="store_true")
    args = parser.parse_args()

    verify_pins()
    rerun_and_compare("analyze_sweep_rate.py", ANALYSIS)
    if args.full:
        rerun_and_compare("collect_readings.py", TABLE)
    print(
        "PASS: analysis-only sweep-rate consolidation reproduces its committed "
        f"artifacts ({'collect + analysis' if args.full else 'analysis'} verified)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
