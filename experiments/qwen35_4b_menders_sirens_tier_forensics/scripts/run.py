#!/usr/bin/env python3
"""Harness for the analysis-only tier-forensics cell.

`--smoke`: re-run the analyzer against the committed receipt table and
require the regenerated analysis to be byte-identical to the committed
`runs/constants_analysis.json` — the analysis is a pure function of the
committed table. No model event exists in this cell.

Full mode (`--full`): additionally re-run the sweep over the repository's
committed gateway receipts and require the regenerated table to match the
committed `runs/receipt_table.json` byte-for-byte. This re-hashes ~2,300
JSON files (about a minute) and still touches no model.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
TABLE = EXP / "runs" / "receipt_table.json"
ANALYSIS = EXP / "runs" / "constants_analysis.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rerun_and_compare(script: str, target: Path) -> None:
    before = sha256_file(target)
    with tempfile.TemporaryDirectory() as scratch:
        backup = Path(scratch) / target.name
        backup.write_bytes(target.read_bytes())
        try:
            subprocess.run(
                [sys.executable, "-B", str(EXP / "scripts" / script)],
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", action="store_true")
    group.add_argument("--full", action="store_true")
    args = parser.parse_args()

    for target in (TABLE, ANALYSIS):
        if not target.exists():
            raise SystemExit(f"missing committed artifact: {target}")
    rerun_and_compare("analyze_constants.py", ANALYSIS)
    if args.full:
        rerun_and_compare("sweep_receipts.py", TABLE)
    print(
        "PASS: analysis-only forensics reproduces its committed artifacts "
        f"({'sweep + analysis' if args.full else 'analysis'} verified)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
