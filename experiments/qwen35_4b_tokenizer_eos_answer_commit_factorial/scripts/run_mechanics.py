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
    str(EXP_REL / "src/mechanics_protocol.py"),
    str(EXP_REL / "src/mechanics_runtime.py"),
    str(EXP_REL / "src/mechanics_stage.py"),
    str(EXP_REL / "src/mechanics_transactions.py"),
    str(EXP_REL / "src/plans.py"),
    str(EXP_REL / "src/protocol.py"),
    str(EXP_REL / "src/stats.py"),
    str(EXP_REL / "src/task_data.py"),
    str(EXP_REL / "src/transactions.py"),
    str(EXP_REL / "src/vllm_runner.py"),
)
_BOOTSTRAP_RUNTIME_FILES = (
    "requirements-vllm.lock.txt",
    str(EXP_REL / "configs/default.yaml"),
    str(EXP_REL / "reports/design_review.md"),
    str(EXP_REL / "reports/preregistration.md"),
    str(EXP_REL / "scripts/mechanics_launcher"),
    str(EXP_REL / "scripts/mechanics_launcher.S"),
    *_BOOTSTRAP_IMPORT_FILES,
)
_BOOTSTRAP_SUPPORT_FILES = (
    str(EXP_REL / "reports/calibration_implementation_review.json"),
    str(EXP_REL / "reports/calibration_implementation_review.md"),
    str(EXP_REL / "reports/mechanics_implementation_review.json"),
    str(EXP_REL / "reports/mechanics_implementation_review.md"),
    str(EXP_REL / "runs/prepared/calibration_requests.jsonl"),
    str(EXP_REL / "runs/prepared/preoutcome_receipt.json"),
    str(EXP_REL / "runs/tokenizer/receipt.json"),
    str(EXP_REL / "scripts/calibration_launcher"),
    str(EXP_REL / "scripts/calibration_launcher.S"),
    str(EXP_REL / "scripts/run_calibration.py"),
    str(EXP_REL / "src/process_lock.py"),
)
_MODEL_ID = "Qwen/Qwen3.5-4B"
_MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
_BOOTSTRAP_GOLD_AUTHORIZED = False
_STATIC_LAUNCHER = EXP / "scripts/mechanics_launcher"
_STATIC_LAUNCHER_PROOF_FD = 198
_STATIC_LAUNCHER_SHA256 = (
    "6fdfb46399c7880da2be42b93b78975cc3354301840dde79de74569e5e4cc4f2"
)


if not (
    sys.flags.isolated == 1
    and sys.flags.ignore_environment == 1
    and sys.flags.safe_path
    and sys.flags.dont_write_bytecode == 1
):
    raise RuntimeError("sealed mechanics requires static-launcher isolated Python")


def _bootstrap_authenticate_static_launcher() -> None:
    parent_fd: int | None = None
    try:
        parent_pid = os.getppid()
        if parent_pid <= 1:
            raise RuntimeError("mechanics static-launcher parent is absent")
        parent_fd = os.open(f"/proc/{parent_pid}/exe", os.O_RDONLY)
        if not os.get_inheritable(_STATIC_LAUNCHER_PROOF_FD):
            raise RuntimeError("mechanics launcher proof descriptor is not inheritable")
        parent_before = os.fstat(parent_fd)
        fd_before = os.fstat(_STATIC_LAUNCHER_PROOF_FD)
        path_before = os.stat(_STATIC_LAUNCHER, follow_symlinks=False)
        if (
            _STATIC_LAUNCHER.is_symlink()
            or (parent_before.st_mode & 0o170000) != 0o100000
            or (fd_before.st_mode & 0o170000) != 0o100000
            or (path_before.st_mode & 0o170000) != 0o100000
            or (parent_before.st_dev, parent_before.st_ino)
            != (path_before.st_dev, path_before.st_ino)
            or (fd_before.st_dev, fd_before.st_ino)
            != (path_before.st_dev, path_before.st_ino)
        ):
            raise RuntimeError("mechanics launcher proof names different bytes")
        os.lseek(_STATIC_LAUNCHER_PROOF_FD, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        while True:
            block = os.read(_STATIC_LAUNCHER_PROOF_FD, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        parent_after = os.fstat(parent_fd)
        fd_after = os.fstat(_STATIC_LAUNCHER_PROOF_FD)
        path_after = os.stat(_STATIC_LAUNCHER, follow_symlinks=False)
    except OSError as error:
        raise RuntimeError("mechanics launcher proof is unavailable") from error
    finally:
        if parent_fd is not None:
            os.close(parent_fd)
    stable = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns")
    if (
        any(getattr(parent_before, key) != getattr(parent_after, key) for key in stable)
        or any(getattr(fd_before, key) != getattr(fd_after, key) for key in stable)
        or any(getattr(path_before, key) != getattr(path_after, key) for key in stable)
        or (fd_after.st_dev, fd_after.st_ino)
        != (path_after.st_dev, path_after.st_ino)
        or digest.hexdigest() != _STATIC_LAUNCHER_SHA256
    ):
        raise RuntimeError("mechanics launcher proof bytes changed")


def _bootstrap_sanitize_environment() -> None:
    _bootstrap_authenticate_static_launcher()
    if any(
        os.environ.get(key)
        for key in ("LD_PRELOAD", "LD_AUDIT", "LD_DEBUG", "GLIBC_TUNABLES")
    ):
        raise RuntimeError("sealed mechanics forbids dynamic-loader injection")
    for key in tuple(os.environ):
        if (
            key in {"PYTHONPATH", "PYTHONHOME", "GH_REPO", "GH_HOST"}
            or key.startswith("GIT_")
        ):
            os.environ.pop(key, None)
    os.environ["PATH"] = (
        f"{ROOT}/.venv-vllm/bin:/usr/local/cuda/bin:/usr/bin:/bin"
    )
    os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64"
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ["GIT_CONFIG_NOSYSTEM"] = "1"
    os.environ["GIT_CONFIG_GLOBAL"] = "/dev/null"
    os.environ["GIT_TERMINAL_PROMPT"] = "0"


_bootstrap_sanitize_environment()


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

    raw = path.read_bytes()

    def reject_constant(value: str) -> Any:
        raise RuntimeError(f"non-finite value in pre-import {label}: {value}")

    try:
        value = json.loads(
            raw,
            object_pairs_hook=no_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid pre-import {label}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"pre-import {label} schema changed")
    canonical = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if raw != canonical:
        raise RuntimeError(f"noncanonical pre-import {label}")
    return value


def _install_mechanics_path_audit(
    allowed_relative: list[str], *, mechanics_authorized: bool
) -> None:
    root = ROOT.resolve()
    benchmark_root = (ROOT / "benchmarks").resolve()
    calibration_runs = (EXP / "runs/calibration").resolve()
    mechanics_runs = (EXP / "runs/mechanics").resolve()
    procedural = (EXP / "data/procedural").resolve()
    hidden_ciphertext = (
        EXP / "data/procedural/mechanics_gold.jsonl.aesgcm"
    ).resolve()
    hidden_key = (EXP / ".secrets/mechanics_gold.aes256.key").resolve()
    sealed_hidden = {hidden_ciphertext, hidden_key}
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
    allowed = {
        (ROOT / relative).resolve()
        for relative in (*allowed_relative, *_BOOTSTRAP_SUPPORT_FILES)
    }
    allowed.add((EXP / "runs/construction/summary.json").resolve())
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
        if path in sealed_hidden:
            if _BOOTSTRAP_GOLD_AUTHORIZED:
                return
            raise PermissionError("mechanics hidden data remains sealed before publication gate")
        if path in allowed_procedural and not mechanics_authorized:
            raise PermissionError("mechanics lock stage forbids mechanics data reads")
        if path.is_relative_to(procedural) and path not in allowed_procedural:
            raise PermissionError(
                f"mechanics process forbids sealed procedural access: {path}"
            )
        if path in mechanics_prepared and not mechanics_authorized:
            raise PermissionError("mechanics lock stage forbids prepared-request reads")
        if (
            path.is_relative_to(prepared)
            and path not in mechanics_prepared
            and path not in allowed
        ):
            raise PermissionError(
                f"mechanics process forbids unregistered prepared access: {path}"
            )
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
        raise RuntimeError("sealed mechanics stage is absent or invalid")
    if any(EXP.rglob("__pycache__")):
        raise RuntimeError("pre-import mechanics refuses local Python caches")
    calibration = _strict_json(
        EXP / "runs/calibration/implementation_lock.json", "calibration lock"
    )
    frozen = calibration.get("frozen_mechanics_blobs")
    if (
        calibration.get("model") != _MODEL_ID
        or calibration.get("revision") != _MODEL_REVISION
        or not isinstance(frozen, dict)
    ):
        raise RuntimeError("pre-import calibration lock boundary changed")

    if stage == "lock":
        review = _strict_json(
            EXP / "reports/mechanics_implementation_review.json",
            "mechanics implementation review",
        )
        implementation = review.get("reviewed_commit")
        report_path = EXP / "reports/mechanics_implementation_review.md"
        if (
            review.get("verdict") != "PASS_IMPLEMENTATION"
            or not isinstance(implementation, str)
            or len(implementation) != 40
            or any(character not in "0123456789abcdef" for character in implementation)
            or report_path.is_symlink()
            or not report_path.is_file()
            or hashlib.sha256(report_path.read_bytes()).hexdigest()
            != review.get("review_report_sha256")
        ):
            raise RuntimeError("pre-import mechanics review boundary changed")
        critical = {
            relative: hashlib.sha256(
                subprocess.check_output(
                    [
                        "/usr/bin/git",
                        "--no-replace-objects",
                        "-c",
                        "core.hooksPath=/dev/null",
                        "show",
                        f"{implementation}:{relative}",
                    ],
                    cwd=ROOT,
                )
            ).hexdigest()
            for relative in _BOOTSTRAP_RUNTIME_FILES
        }
    else:
        mechanics = _strict_json(
            EXP / "runs/mechanics/implementation_lock.json", "mechanics lock"
        )
        critical = mechanics.get("critical_files")
        implementation = mechanics.get("implementation_commit")
        if (
            mechanics.get("model") != _MODEL_ID
            or mechanics.get("revision") != _MODEL_REVISION
            or mechanics.get("selected_interface")
            != "tokenizer_eos_no_think_program_slot"
            or mechanics.get("calibration_implementation_commit")
            != calibration.get("implementation_commit")
            or mechanics.get("frozen_mechanics_blobs") != frozen
            or not isinstance(critical, dict)
            or set(_BOOTSTRAP_RUNTIME_FILES) - set(critical)
            or not isinstance(implementation, str)
            or len(implementation) != 40
            or any(character not in "0123456789abcdef" for character in implementation)
        ):
            raise RuntimeError("pre-import mechanics lock schema changed")

    _validate_environment()
    _install_mechanics_path_audit(
        list(_BOOTSTRAP_RUNTIME_FILES), mechanics_authorized=stage != "lock"
    )
    for relative in _BOOTSTRAP_RUNTIME_FILES:
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
            [
                "/usr/bin/git",
                "--no-replace-objects",
                "-c",
                "core.hooksPath=/dev/null",
                "show",
                f"{implementation}:{relative}",
            ],
            cwd=ROOT,
        )
        if hashlib.sha256(blob).hexdigest() != expected:
            raise RuntimeError(f"pre-import committed mechanics file changed: {relative}")
    for relative, expected_blob in frozen.items():
        observed_blob = subprocess.check_output(
            [
                "/usr/bin/git",
                "--no-replace-objects",
                "-c",
                "core.hooksPath=/dev/null",
                "rev-parse",
                f"HEAD:{relative}",
            ],
            cwd=ROOT,
            text=True,
        ).strip()
        if observed_blob != expected_blob:
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
    RESOURCE_DECISION,
    TRANSPORT_DECISION,
    VISIBLE_SELECTION,
    analyze_transport,
    analyze_visible,
    decrypt_hidden_gold,
    run_generation_transactions,
    run_transport_transaction,
    score_hidden,
)
from mechanics_transactions import (  # noqa: E402
    exact_json_equal,
    inventory_state,
    read_canonical,
    write_exclusive_durable,
)
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
        if not exact_json_equal(read_canonical(path), value):
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
    if value["decision"] == "DIRECT_RESOURCE_MATCH_POOL_EXHAUSTED":
        if VISIBLE_SELECTION.exists() or VISIBLE_SELECTION.is_symlink():
            raise RuntimeError("resource exhaustion cannot coexist with visible selection")
        return _write_or_verify(RESOURCE_DECISION, value)
    if RESOURCE_DECISION.exists() or RESOURCE_DECISION.is_symlink():
        raise RuntimeError("visible selection cannot coexist with resource exhaustion")
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
    gold_rows, gold_receipt = decrypt_hidden_gold()
    value = {
        **score_hidden(
            visible=visible,
            gold_rows=gold_rows,
            gold_receipt=gold_receipt,
            config=inputs.config,
        ),
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
