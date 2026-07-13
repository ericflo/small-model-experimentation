#!/usr/bin/env python3
"""Publish, execute, or analyze the sealed answer-seam calibration."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


EXP = Path(__file__).resolve().parents[1]
SRC = EXP / "src"
sys.path.insert(0, str(SRC))

from calibration_lock import (  # noqa: E402
    DECISION,
    IMPLEMENTATION_LOCK,
    LIVE_PREFLIGHT,
    RAW_DIR,
    publish_calibration_lock,
    publish_or_verify_live_preflight,
    verify_calibration_lock,
    verify_recorded_live_preflight,
)
from calibration_stage import (  # noqa: E402
    INVOCATION_ORDER,
    analyze_calibration,
    engine_config,
    load_calibration_inputs,
    run_calibration_transactions,
)
from transactions import (  # noqa: E402
    inventory_state,
    read_canonical,
    sha256_file,
    write_exclusive_durable,
)
from vllm_runner import VLLMRunner  # noqa: E402


RUN_LOCK = EXP / "runs/calibration/run.lock"
RUNNER_PATH = SRC / "vllm_runner.py"


@contextmanager
def calibration_process_lock() -> Iterator[None]:
    RUN_LOCK.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(RUN_LOCK, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another calibration process holds the live lock") from error
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
        try:
            RUN_LOCK.unlink()
        except FileNotFoundError:
            pass


def _write_or_verify_decision(value: dict[str, Any]) -> dict[str, Any]:
    if DECISION.exists() or DECISION.is_symlink():
        if read_canonical(DECISION) != value:
            raise RuntimeError("recorded calibration decision changed")
    else:
        write_exclusive_durable(DECISION, value)
    return value


def analyze() -> dict[str, Any]:
    inputs = load_calibration_inputs()
    verify_calibration_lock()
    verify_recorded_live_preflight(inputs=inputs)
    value = {
        **analyze_calibration(inputs=inputs, raw_dir=RAW_DIR),
        "implementation_lock_sha256": sha256_file(IMPLEMENTATION_LOCK),
        "live_preflight_sha256": sha256_file(LIVE_PREFLIGHT),
    }
    return _write_or_verify_decision(value)


def run_live() -> dict[str, Any]:
    inputs = load_calibration_inputs()
    verify_calibration_lock()
    states = [inventory_state(RAW_DIR, name) for name in INVOCATION_ORDER]
    if all(state == "complete" for state in states):
        return analyze()
    with VLLMRunner(engine_config(inputs)) as runner:
        publish_or_verify_live_preflight(runner=runner, inputs=inputs)
        run_calibration_transactions(
            inputs=inputs,
            runner=runner,
            raw_dir=RAW_DIR,
            implementation_lock_path=IMPLEMENTATION_LOCK,
            live_preflight_path=LIVE_PREFLIGHT,
            runner_path=RUNNER_PATH,
        )
    return analyze()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--stage", choices=("lock", "run", "analyze"), required=True)
    args = parser.parse_args(argv)
    if args.stage == "lock":
        value = publish_calibration_lock()
    else:
        with calibration_process_lock():
            value = run_live() if args.stage == "run" else analyze()
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
