#!/usr/bin/env python3
"""Publish, execute, or analyze the sealed answer-seam calibration."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
EXP_REL = EXP.relative_to(ROOT)
SRC = EXP / "src"
_BOOTSTRAP_IMPORT_FILES = (
    str(EXP_REL / "scripts/run_calibration.py"),
    str(EXP_REL / "src/calibration_lock.py"),
    str(EXP_REL / "src/calibration_stage.py"),
    str(EXP_REL / "src/identity.py"),
    str(EXP_REL / "src/interface_analysis.py"),
    str(EXP_REL / "src/protocol.py"),
    str(EXP_REL / "src/task_data.py"),
    str(EXP_REL / "src/transactions.py"),
    str(EXP_REL / "src/vllm_runner.py"),
)
_MODEL_ID = "Qwen/Qwen3.5-4B"
_MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


def _bootstrap_stage() -> str | None:
    found: str | None = None
    index = 1
    while index < len(sys.argv):
        argument = sys.argv[index]
        value: str | None = None
        if argument == "--stage":
            if index + 1 >= len(sys.argv) or sys.argv[index + 1].startswith("--"):
                raise RuntimeError("bootstrap --stage lacks a value")
            value = sys.argv[index + 1]
            index += 1
        elif argument.startswith("--stage="):
            value = argument.split("=", 1)[1]
            if not value:
                raise RuntimeError("bootstrap --stage lacks a value")
        if value is not None:
            if found is not None:
                raise RuntimeError("duplicate bootstrap --stage is forbidden")
            found = value
        index += 1
    return found


def _bootstrap_sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"pre-import critical file is unsafe: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bootstrap_strict_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("local imports require a real implementation lock")

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError("duplicate key in pre-import implementation lock")
            result[key] = value
        return result

    value = json.loads(path.read_bytes(), object_pairs_hook=no_duplicates)
    if not isinstance(value, dict):
        raise RuntimeError("pre-import implementation lock schema changed")
    return value


def _install_calibration_path_audit(allowed_relative: list[str]) -> None:
    root = ROOT.resolve()
    benchmark_root = (ROOT / "benchmarks").resolve()
    calibration_runs = (EXP / "runs/calibration").resolve()
    git_root = (ROOT / ".git").resolve()
    environment_root = (ROOT / ".venv-vllm").resolve()
    allowed = {(ROOT / relative).resolve() for relative in allowed_relative}

    def audit(event: str, arguments: tuple[Any, ...]) -> None:
        if event != "open" or not arguments:
            return
        raw_path = arguments[0]
        if not isinstance(raw_path, (str, bytes, os.PathLike)):
            return
        try:
            path = Path(raw_path).resolve()
        except (OSError, TypeError, ValueError):
            return
        if path.is_relative_to(benchmark_root):
            raise PermissionError(f"calibration process forbids benchmark access: {path}")
        if not path.is_relative_to(root):
            return
        if (
            path in allowed
            or path.is_relative_to(calibration_runs)
            or path.is_relative_to(git_root)
            or path.is_relative_to(environment_root)
        ):
            return
        raise PermissionError(f"calibration process forbids unregistered repository access: {path}")

    sys.addaudithook(audit)


def _bootstrap_normalize_distribution(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _bootstrap_validate_environment() -> None:
    executable = Path(sys.executable)
    if executable.parent != ROOT / ".venv-vllm/bin":
        raise RuntimeError("live calibration requires the pinned .venv-vllm interpreter")
    expected = {"vllm": "0.24.0+cu129", "torch": "2.11.0+cu129", "transformers": "5.13.0"}
    observed = {
        name: importlib.metadata.version(_bootstrap_normalize_distribution(name))
        for name in expected
    }
    if observed != expected:
        raise RuntimeError(f"live calibration package versions changed: {observed}")


def _bootstrap_verify_before_local_imports() -> None:
    sys.dont_write_bytecode = True
    stage = _bootstrap_stage()
    if stage not in {"run", "analyze"}:
        return
    if any(EXP.rglob("__pycache__")):
        raise RuntimeError("pre-import calibration refuses local Python caches")
    lock_path = EXP / "runs/calibration/implementation_lock.json"
    lock = _bootstrap_strict_json(lock_path)
    critical = lock.get("critical_files")
    allowed = lock.get("calibration_runtime_files")
    implementation = lock.get("implementation_commit")
    if (
        lock.get("stage") != "calibration_implementation_lock"
        or lock.get("authorization") != "interface_calibration_only"
        or lock.get("model") != _MODEL_ID
        or lock.get("revision") != _MODEL_REVISION
        or not isinstance(critical, dict)
        or not isinstance(allowed, list)
        or not isinstance(implementation, str)
        or len(implementation) != 40
        or any(relative not in allowed for relative in _BOOTSTRAP_IMPORT_FILES)
    ):
        raise RuntimeError("pre-import implementation lock schema changed")
    _bootstrap_validate_environment()
    _install_calibration_path_audit(allowed)
    for relative in _BOOTSTRAP_IMPORT_FILES:
        path = ROOT / relative
        expected = critical.get(relative)
        if not isinstance(expected, str) or _bootstrap_sha256(path) != expected:
            raise RuntimeError(f"pre-import critical file changed: {relative}")
        blob = subprocess.check_output(
            ["git", "show", f"{implementation}:{relative}"], cwd=ROOT
        )
        if hashlib.sha256(blob).hexdigest() != expected:
            raise RuntimeError(f"pre-import committed file changed: {relative}")


_bootstrap_verify_before_local_imports()

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
