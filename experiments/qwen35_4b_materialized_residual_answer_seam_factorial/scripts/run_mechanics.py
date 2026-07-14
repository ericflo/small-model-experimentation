#!/usr/bin/env python3
"""Authorize, run, visibly select, or hidden-score frozen mechanics."""

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
    str(EXP_REL / "scripts/run_mechanics.py"),
    str(EXP_REL / "src/calibration_lock.py"),
    str(EXP_REL / "src/calibration_stage.py"),
    str(EXP_REL / "src/interface_analysis.py"),
    str(EXP_REL / "src/identity.py"),
    str(EXP_REL / "src/mechanics_lock.py"),
    str(EXP_REL / "src/mechanics_stage.py"),
    str(EXP_REL / "src/plans.py"),
    str(EXP_REL / "src/protocol.py"),
    str(EXP_REL / "src/stats.py"),
    str(EXP_REL / "src/task_data.py"),
    str(EXP_REL / "src/transactions.py"),
    str(EXP_REL / "src/vllm_runner.py"),
)
_MECHANICS_FROZEN_IMPORT_FILES = (
    str(EXP_REL / "scripts/run_mechanics.py"),
    str(EXP_REL / "src/identity.py"),
    str(EXP_REL / "src/mechanics_lock.py"),
    str(EXP_REL / "src/mechanics_stage.py"),
    str(EXP_REL / "src/plans.py"),
    str(EXP_REL / "src/protocol.py"),
    str(EXP_REL / "src/stats.py"),
    str(EXP_REL / "src/task_data.py"),
    str(EXP_REL / "src/transactions.py"),
    str(EXP_REL / "src/vllm_runner.py"),
)
_MODEL_ID = "Qwen/Qwen3.5-4B"
_MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
_BOOTSTRAP_GOLD_AUTHORIZED = False


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


def _strict_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"pre-import {label} is unsafe or absent")

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"duplicate key in pre-import {label}")
            result[key] = value
        return result

    value = json.loads(path.read_bytes(), object_pairs_hook=no_duplicates)
    if not isinstance(value, dict):
        raise RuntimeError(f"pre-import {label} schema changed")
    return value


def _install_mechanics_path_audit(allowed_relative: list[str]) -> None:
    root = ROOT.resolve()
    benchmark_root = (ROOT / "benchmarks").resolve()
    calibration_runs = (EXP / "runs/calibration").resolve()
    mechanics_runs = (EXP / "runs/mechanics").resolve()
    procedural = (EXP / "data/procedural").resolve()
    gold = (EXP / "data/procedural/mechanics_gold.jsonl").resolve()
    allowed_procedural = {
        (procedural / "mechanics_public.jsonl").resolve(),
        (procedural / "mechanics_audit.jsonl").resolve(),
    }
    prepared = (EXP / "runs/prepared").resolve()
    mechanics_prepared = {
        (prepared / name).resolve()
        for name in (
            "transport_requests.jsonl",
            "direct_requests.jsonl",
            "suffix_materialized_requests.jsonl",
            "suffix_name_only_requests.jsonl",
            "suffix_shuffled_requests.jsonl",
        )
    }
    allowed = {(ROOT / relative).resolve() for relative in allowed_relative}
    git_root = (ROOT / ".git").resolve()
    environment_root = (ROOT / ".venv-vllm").resolve()

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
            raise PermissionError(f"mechanics process forbids benchmark access: {path}")
        if not path.is_relative_to(root):
            return
        if path == gold:
            if _BOOTSTRAP_GOLD_AUTHORIZED:
                return
            raise PermissionError("mechanics gold remains sealed before publication gate")
        if path.is_relative_to(procedural) and path not in allowed_procedural:
            raise PermissionError(f"mechanics process forbids sealed procedural access: {path}")
        if path.is_relative_to(prepared) and path not in mechanics_prepared and path not in allowed:
            raise PermissionError(f"mechanics process forbids unregistered prepared access: {path}")
        if (
            path in allowed
            or path in allowed_procedural
            or path in mechanics_prepared
            or path.is_relative_to(calibration_runs)
            or path.is_relative_to(mechanics_runs)
            or path.is_relative_to(git_root)
            or path.is_relative_to(environment_root)
        ):
            return
        raise PermissionError(f"mechanics process forbids unregistered repository access: {path}")

    sys.addaudithook(audit)


def _validate_environment() -> None:
    executable = Path(sys.executable)
    if executable.parent != ROOT / ".venv-vllm/bin":
        raise RuntimeError("live mechanics requires the pinned .venv-vllm interpreter")
    expected = {"vllm": "0.24.0+cu129", "torch": "2.11.0+cu129", "transformers": "5.13.0"}
    observed = {
        name: importlib.metadata.version(re.sub(r"[-_.]+", "-", name).lower())
        for name in expected
    }
    if observed != expected:
        raise RuntimeError(f"live mechanics package versions changed: {observed}")


def _bootstrap_verify_before_local_imports() -> None:
    sys.dont_write_bytecode = True
    stage = _bootstrap_stage()
    if stage not in {"lock", "run", "analyze-visible", "score-hidden"}:
        return
    if any(EXP.rglob("__pycache__")):
        raise RuntimeError("pre-import mechanics refuses local Python caches")
    calibration = _strict_json(
        EXP / "runs/calibration/implementation_lock.json", "calibration lock"
    )
    critical = calibration.get("critical_files")
    runtime = calibration.get("calibration_runtime_files")
    implementation = calibration.get("implementation_commit")
    frozen = calibration.get("frozen_mechanics_blobs")
    mechanics = None
    if stage != "lock":
        mechanics = _strict_json(
            EXP / "runs/mechanics/implementation_lock.json", "mechanics lock"
        )
    if (
        calibration.get("model") != _MODEL_ID
        or calibration.get("revision") != _MODEL_REVISION
        or not isinstance(critical, dict)
        or not isinstance(runtime, list)
        or not isinstance(frozen, dict)
        or not isinstance(implementation, str)
        or len(implementation) != 40
        or any(relative not in critical for relative in _BOOTSTRAP_IMPORT_FILES)
        or any(relative not in frozen for relative in _MECHANICS_FROZEN_IMPORT_FILES)
        or (
            mechanics is not None
            and (
                mechanics.get("model") != _MODEL_ID
                or mechanics.get("revision") != _MODEL_REVISION
                or mechanics.get("calibration_implementation_commit")
                != implementation
                or mechanics.get("frozen_mechanics_blobs") != frozen
            )
        )
    ):
        raise RuntimeError("pre-import mechanics lock schema changed")
    _validate_environment()
    _install_mechanics_path_audit(list(set(runtime) | set(_BOOTSTRAP_IMPORT_FILES)))
    for relative in _BOOTSTRAP_IMPORT_FILES:
        path = ROOT / relative
        expected = critical.get(relative)
        if (
            not isinstance(expected, str)
            or path.is_symlink()
            or not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise RuntimeError(f"pre-import mechanics file changed: {relative}")
        blob = subprocess.check_output(
            ["git", "show", f"{implementation}:{relative}"], cwd=ROOT
        )
        if hashlib.sha256(blob).hexdigest() != expected:
            raise RuntimeError(f"pre-import committed mechanics file changed: {relative}")
    for relative in _MECHANICS_FROZEN_IMPORT_FILES:
        observed_blob = subprocess.check_output(
            ["git", "rev-parse", f"{implementation}:{relative}"],
            cwd=ROOT,
            text=True,
        ).strip()
        if observed_blob != frozen[relative]:
            raise RuntimeError(f"pre-import frozen mechanics blob changed: {relative}")


def _bootstrap_authorize_gold_path() -> None:
    global _BOOTSTRAP_GOLD_AUTHORIZED
    _BOOTSTRAP_GOLD_AUTHORIZED = True


_bootstrap_verify_before_local_imports()

sys.path.insert(0, str(SRC))

from calibration_lock import DECISION as CALIBRATION_DECISION  # noqa: E402
from calibration_stage import (  # noqa: E402
    engine_config,
    load_analysis_tokenizer,
    load_calibration_inputs,
)
from mechanics_lock import (  # noqa: E402
    MECHANICS_LOCK,
    MECHANICS_PREFLIGHT,
    authorize_hidden_read,
    publish_mechanics_lock,
    publish_or_verify_mechanics_preflight,
    verify_mechanics_lock,
)
from mechanics_stage import (  # noqa: E402
    HIDDEN_RESULT,
    MECHANICS_INVOCATION_ORDER,
    RAW_DIR,
    TRANSPORT_DECISION,
    VISIBLE_SELECTION,
    analyze_transport,
    analyze_visible,
    run_generation_transactions,
    run_transport_transaction,
    score_hidden,
)
from transactions import inventory_state, read_canonical, write_exclusive_durable  # noqa: E402
from vllm_runner import VLLMRunner  # noqa: E402


PROCESS_LOCK = EXP / "runs/mechanics/run.lock"
RUNNER_PATH = SRC / "vllm_runner.py"


@contextmanager
def mechanics_process_lock() -> Iterator[None]:
    PROCESS_LOCK.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(PROCESS_LOCK, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another mechanics process holds the live lock") from error
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
        try:
            PROCESS_LOCK.unlink()
        except FileNotFoundError:
            pass


def _write_or_verify(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    if path.exists() or path.is_symlink():
        if read_canonical(path) != value:
            raise RuntimeError(f"recorded mechanics artifact changed: {path.name}")
    else:
        write_exclusive_durable(path, value)
    return value


def visible_analysis() -> dict[str, Any]:
    inputs = load_calibration_inputs()
    tokenizer = load_analysis_tokenizer(inputs)
    decision = read_canonical(CALIBRATION_DECISION)
    verify_mechanics_lock()
    value = analyze_visible(
        decision=decision,
        inputs=inputs,
        mechanics_lock_path=MECHANICS_LOCK,
        live_preflight_path=MECHANICS_PREFLIGHT,
        runner_path=RUNNER_PATH,
        tokenizer=tokenizer,
    )
    return _write_or_verify(VISIBLE_SELECTION, value)


def run_live() -> dict[str, Any]:
    inputs = load_calibration_inputs()
    tokenizer = load_analysis_tokenizer(inputs)
    decision = read_canonical(CALIBRATION_DECISION)
    verify_mechanics_lock()
    states = [inventory_state(RAW_DIR, name) for name in MECHANICS_INVOCATION_ORDER]
    if VISIBLE_SELECTION.exists() or all(state == "complete" for state in states):
        return visible_analysis()
    if states[0] == "complete":
        transport = _write_or_verify(
            TRANSPORT_DECISION,
            analyze_transport(
                decision=decision,
                inputs=inputs,
                mechanics_lock_path=MECHANICS_LOCK,
                live_preflight_path=MECHANICS_PREFLIGHT,
                runner_path=RUNNER_PATH,
                tokenizer=tokenizer,
            ),
        )
        if transport["decision"] != "SELECTED_INTERFACE_TRANSPORT_PASS":
            return transport
    with VLLMRunner(engine_config(inputs)) as runner:
        publish_or_verify_mechanics_preflight(runner=runner, inputs=inputs)
        if states[0] != "complete":
            run_transport_transaction(
                decision=decision,
                inputs=inputs,
                runner=runner,
                mechanics_lock_path=MECHANICS_LOCK,
                live_preflight_path=MECHANICS_PREFLIGHT,
                runner_path=RUNNER_PATH,
            )
            transport = _write_or_verify(
                TRANSPORT_DECISION,
                analyze_transport(
                    decision=decision,
                    inputs=inputs,
                    mechanics_lock_path=MECHANICS_LOCK,
                    live_preflight_path=MECHANICS_PREFLIGHT,
                    runner_path=RUNNER_PATH,
                    tokenizer=tokenizer,
                ),
            )
            if transport["decision"] != "SELECTED_INTERFACE_TRANSPORT_PASS":
                return transport
        run_generation_transactions(
            decision=decision,
            transport=transport,
            inputs=inputs,
            runner=runner,
            mechanics_lock_path=MECHANICS_LOCK,
            live_preflight_path=MECHANICS_PREFLIGHT,
            runner_path=RUNNER_PATH,
            tokenizer=tokenizer,
        )
    return visible_analysis()


def hidden_analysis() -> dict[str, Any]:
    authorization = authorize_hidden_read()
    _bootstrap_authorize_gold_path()
    inputs = load_calibration_inputs()
    visible = read_canonical(VISIBLE_SELECTION)
    value = {
        **score_hidden(visible=visible, config=inputs.config),
        "hidden_read_authorization": authorization,
    }
    return _write_or_verify(HIDDEN_RESULT, value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--stage",
        choices=("lock", "run", "analyze-visible", "score-hidden"),
        required=True,
    )
    args = parser.parse_args(argv)
    if args.stage == "lock":
        value = publish_mechanics_lock()
    else:
        with mechanics_process_lock():
            if args.stage == "run":
                value = run_live()
            elif args.stage == "analyze-visible":
                value = visible_analysis()
            else:
                value = hidden_analysis()
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
