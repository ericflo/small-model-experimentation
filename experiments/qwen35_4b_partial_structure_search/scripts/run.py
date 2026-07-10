#!/usr/bin/env python3
"""Run the fail-closed Recognize -> Search experiment lifecycle."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
SCRIPTS = EXP / "scripts"


def _call(script: str, *arguments: str, check: bool = True) -> int:
    command = [sys.executable, str(SCRIPTS / script), *arguments]
    print("[orchestrator] " + " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=EXP, check=False)
    if check and completed.returncode:
        raise RuntimeError(f"{script} failed with exit code {completed.returncode}")
    return int(completed.returncode)


def _passed(path: Path) -> bool:
    if not path.exists():
        raise RuntimeError(f"expected gate receipt was not written: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    return bool(value.get("passed"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    suffix = "_smoke" if args.smoke else ""
    mode = ["--smoke"] if args.smoke else []
    workers = ["--workers", str(args.workers)]

    _call("build_data.py", *mode, *workers)
    _call("data_audit.py", *mode, *workers)

    oracle_code = _call("oracle_gate.py", *mode, check=False)
    oracle_receipt = EXP / "runs" / f"oracle_gate{suffix}.json"
    if oracle_code and not _passed(oracle_receipt):
        print("[orchestrator] stopped at G0: oracle state is not useful", flush=True)
        return 0
    if oracle_code:
        raise RuntimeError("oracle gate errored without a valid stop receipt")

    # This model-free depth-5 reference remains informative even when the
    # recognition gate later stops model-guided search.
    _call("full_brute.py", *mode, *workers)
    _call("build_calibration.py", *mode)
    _call("run_calibration.py", *mode)
    calibration_code = _call("analyze_calibration.py", *mode, check=False)
    calibration_receipt = EXP / "runs" / f"calibration_verdict{suffix}.json"
    if calibration_code and not _passed(calibration_receipt):
        print(
            "[orchestrator] stopped at recognition calibration; primary search is not authorized",
            flush=True,
        )
        return 0
    if calibration_code:
        raise RuntimeError("calibration analysis errored without a valid stop receipt")

    _call("run_search.py", *mode)
    _call("analyze_search.py", *mode)
    print("[orchestrator] gated lifecycle complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
