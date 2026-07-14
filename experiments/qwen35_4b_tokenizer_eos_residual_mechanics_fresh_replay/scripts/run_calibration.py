#!/usr/bin/env python3
"""Publish, execute, or analyze the sealed answer-seam calibration."""

from __future__ import annotations

import sys


if not (
    sys.flags.isolated == 1
    and sys.flags.ignore_environment == 1
    and sys.flags.safe_path
):
    raise RuntimeError(
        "sealed calibration requires isolated Python; invoke .venv-vllm/bin/python -I"
    )

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any


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
    str(EXP_REL / "src/process_lock.py"),
    str(EXP_REL / "src/protocol.py"),
    str(EXP_REL / "src/task_data.py"),
    str(EXP_REL / "src/transactions.py"),
    str(EXP_REL / "src/vllm_runner.py"),
)
_BOOTSTRAP_RUNTIME_FILES = (
    "requirements-vllm.lock.txt",
    str(EXP_REL / "configs/default.yaml"),
    str(EXP_REL / "scripts/calibration_launcher"),
    str(EXP_REL / "scripts/calibration_launcher.S"),
    str(EXP_REL / "runs/prepared/calibration_requests.jsonl"),
    str(EXP_REL / "runs/prepared/preoutcome_receipt.json"),
    str(EXP_REL / "runs/tokenizer/receipt.json"),
    *_BOOTSTRAP_IMPORT_FILES,
)
_BOOTSTRAP_REVIEW_FILES = (
    str(EXP_REL / "reports/calibration_implementation_review.md"),
    str(EXP_REL / "reports/calibration_implementation_review.json"),
)
_BOOTSTRAP_AUDIT_FILES = (*_BOOTSTRAP_RUNTIME_FILES, *_BOOTSTRAP_REVIEW_FILES)
_BOOTSTRAP_CRITICAL_FILES = (
    "requirements-vllm.lock.txt",
    str(EXP_REL / "configs/default.yaml"),
    str(EXP_REL / "reports/design_review.md"),
    str(EXP_REL / "reports/preregistration.md"),
    str(EXP_REL / "reports/calibration_implementation_review.md"),
    str(EXP_REL / "scripts/calibration_launcher"),
    str(EXP_REL / "scripts/calibration_launcher.S"),
    str(EXP_REL / "runs/prepared/calibration_requests.jsonl"),
    str(EXP_REL / "runs/prepared/preoutcome_receipt.json"),
    str(EXP_REL / "runs/tokenizer/receipt.json"),
    *_BOOTSTRAP_IMPORT_FILES,
    str(EXP_REL / "scripts/tokenizer_receipt.py"),
    str(EXP_REL / "tests/test_calibration_lock.py"),
    str(EXP_REL / "tests/test_calibration_process_lock.py"),
    str(EXP_REL / "tests/test_calibration_stage.py"),
    str(EXP_REL / "tests/test_calibration_bootstrap.py"),
    str(EXP_REL / "tests/test_construction.py"),
    str(EXP_REL / "tests/test_interface_analysis.py"),
    str(EXP_REL / "tests/test_protocol.py"),
    str(EXP_REL / "tests/test_tokenizer_receipt.py"),
    str(EXP_REL / "tests/test_transactions.py"),
    str(EXP_REL / "tests/test_vllm_runner.py"),
)
_BOOTSTRAP_FROZEN_MECHANICS = (
    str(EXP_REL / "data/procedural/mechanics_public.jsonl"),
    str(EXP_REL / "data/procedural/mechanics_audit.jsonl"),
    str(EXP_REL / "data/procedural/mechanics_gold.jsonl.aesgcm"),
    str(EXP_REL / "runs/construction/summary.json"),
    str(EXP_REL / "runs/prepared/transport_requests.jsonl"),
    str(EXP_REL / "runs/prepared/direct_requests.jsonl"),
    str(EXP_REL / "runs/prepared/suffix_materialized_requests.jsonl"),
    str(EXP_REL / "runs/prepared/suffix_name_only_requests.jsonl"),
    str(EXP_REL / "runs/prepared/suffix_shuffled_requests.jsonl"),
)
_BOOTSTRAP_REQUIRED_WORKFLOWS = ("Validate Repository", "Publish Research Site")
_BOOTSTRAP_LOCK_KEYS = {
    "schema_version", "stage", "authorization", "model", "revision",
    "implementation_commit", "critical_files", "calibration_runtime_files",
    "frozen_mechanics_blobs", "calibration_inputs", "invocation_order",
    "expected_source_rows", "expected_answer_pairs", "expected_answer_requests",
    "engine", "sampling", "implementation_ci", "implementation_review",
    "release_commit", "release_ci", "experimental_model_requests_before_lock",
    "sampled_model_outputs_before_lock", "hidden_files_read",
    "qualification_files_read", "confirmation_files_read", "benchmark_files_read",
}
_BOOTSTRAP_REVIEW_KEYS = {
    "schema_version", "verdict", "reviewed_commit", "reviewer",
    "review_report_sha256", "reviewed_ci", "adversarial_review_rounds",
    "experimental_model_requests_reviewed", "sampled_model_outputs_reviewed",
    "hidden_files_read", "qualification_files_read", "confirmation_files_read",
    "benchmark_files_read",
}
_MODEL_ID = "Qwen/Qwen3.5-4B"
_MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
_CANONICAL_REPOSITORY = "ericflo/small-model-experimentation"
_CANONICAL_ORIGIN = "https://github.com/ericflo/small-model-experimentation.git"
_GIT_EXECUTABLE = "/usr/bin/git"
_GH_EXECUTABLE = "/usr/bin/gh"
_PINNED_PATH = f"{ROOT}/.venv-vllm/bin:/usr/local/cuda/bin:/usr/bin:/bin"
_STATIC_LAUNCHER = EXP / "scripts/calibration_launcher"
_STATIC_LAUNCHER_PROOF_FD = 198
_STATIC_LAUNCHER_SHA256 = (
    "f7a62bb63a49e8ff430e04a07c27b3e6e225ec1b1c15d2cf5071147ffdb77f6b"
)


def _bootstrap_authenticate_static_launcher() -> None:
    parent_fd: int | None = None
    try:
        parent_pid = os.getppid()
        if parent_pid <= 1:
            raise RuntimeError("static-launcher parent is absent")
        parent_fd = os.open(f"/proc/{parent_pid}/exe", os.O_RDONLY)
        if not os.get_inheritable(_STATIC_LAUNCHER_PROOF_FD):
            raise RuntimeError("static-launcher proof descriptor will not survive re-exec")
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
            raise RuntimeError("static-launcher proof names different executable bytes")
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
        raise RuntimeError(
            "sealed calibration requires live static-launcher parent and proof descriptor"
        ) from error
    finally:
        if parent_fd is not None:
            os.close(parent_fd)
    stable_fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns")
    if (
        any(
            getattr(parent_before, field) != getattr(parent_after, field)
            for field in stable_fields
        )
        or any(
            getattr(fd_before, field) != getattr(fd_after, field)
            for field in stable_fields
        )
        or any(
            getattr(path_before, field) != getattr(path_after, field)
            for field in stable_fields
        )
        or (fd_after.st_dev, fd_after.st_ino)
        != (path_after.st_dev, path_after.st_ino)
        or digest.hexdigest() != _STATIC_LAUNCHER_SHA256
    ):
        raise RuntimeError("sealed calibration static-launcher proof bytes changed")


def _bootstrap_sanitize_process_environment() -> None:
    _bootstrap_authenticate_static_launcher()
    if any(
        os.environ.get(key)
        for key in ("LD_PRELOAD", "LD_AUDIT", "LD_DEBUG", "GLIBC_TUNABLES")
    ):
        raise RuntimeError("sealed calibration forbids dynamic-loader injection")
    for key in tuple(os.environ):
        if (
            key in {"PYTHONPATH", "PYTHONHOME", "GH_REPO", "GH_HOST"}
            or key.startswith("GIT_")
        ):
            os.environ.pop(key, None)
    os.environ["PATH"] = _PINNED_PATH
    os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64"
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ["GH_HOST"] = "github.com"
    os.environ["GIT_CONFIG_NOSYSTEM"] = "1"
    os.environ["GIT_CONFIG_GLOBAL"] = "/dev/null"
    os.environ["GIT_TERMINAL_PROMPT"] = "0"


def _bootstrap_child_environment() -> dict[str, str]:
    result = {
        "HOME": "/root",
        "PATH": _PINNED_PATH,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "GH_CONFIG_DIR": "/root/.config/gh",
        "GH_HOST": "github.com",
        "NO_COLOR": "1",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_TERMINAL_PROMPT": "0",
    }
    if os.environ.get("GITHUB_TOKEN"):
        result["GITHUB_TOKEN"] = os.environ["GITHUB_TOKEN"]
    return result


_bootstrap_sanitize_process_environment()


def _bootstrap_exact_int(value: Any, expected: int) -> bool:
    return type(value) is int and value == expected


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
        raise RuntimeError(f"pre-import JSON is unsafe or absent: {path}")

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"duplicate key in pre-import JSON: {path}")
            result[key] = value
        return result

    raw = path.read_bytes()
    value = json.loads(raw, object_pairs_hook=no_duplicates)
    if not isinstance(value, dict):
        raise RuntimeError(f"pre-import JSON schema changed: {path}")
    canonical = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode()
    if raw != canonical:
        raise RuntimeError(f"pre-import JSON is noncanonical: {path}")
    return value


def _bootstrap_commit(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"pre-import {label} is not a full commit ID")
    return value


def _bootstrap_git(*arguments: str) -> str:
    command = [
        _GIT_EXECUTABLE,
        "--no-replace-objects",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "core.fsmonitor=false",
        *arguments,
    ]
    return subprocess.check_output(
        command,
        cwd=ROOT,
        text=True,
        env=_bootstrap_child_environment(),
    ).strip()


def _bootstrap_git_bytes(*arguments: str) -> bytes:
    return subprocess.check_output(
        [
            _GIT_EXECUTABLE,
            "--no-replace-objects",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "core.fsmonitor=false",
            *arguments,
        ],
        cwd=ROOT,
        env=_bootstrap_child_environment(),
    )


def _bootstrap_ancestor(older: str, newer: str) -> bool:
    return subprocess.run(
        [
            _GIT_EXECUTABLE,
            "--no-replace-objects",
            "-c",
            "core.hooksPath=/dev/null",
            "merge-base",
            "--is-ancestor",
            older,
            newer,
        ],
        cwd=ROOT,
        capture_output=True,
        env=_bootstrap_child_environment(),
    ).returncode == 0


def _bootstrap_tracked_commit(relative: str) -> str:
    commit = _bootstrap_git("log", "-1", "--format=%H", "--", relative)
    if not commit:
        raise RuntimeError(f"pre-import artifact is uncommitted: {relative}")
    return _bootstrap_commit(commit, label=f"{relative} commit")


def _bootstrap_validate_ci(
    commit: str, recorded: Any | None = None
) -> dict[str, dict[str, Any]]:
    rows = json.loads(
        subprocess.check_output(
            [
                _GH_EXECUTABLE, "run", "list", "--repo", _CANONICAL_REPOSITORY,
                "--commit", commit, "--limit", "20",
                "--json", "databaseId,headSha,status,conclusion,workflowName,url",
            ],
            cwd=ROOT,
            text=True,
            env=_bootstrap_child_environment(),
        )
    )
    result: dict[str, dict[str, Any]] = {}
    if recorded is not None and (
        not isinstance(recorded, dict)
        or set(recorded) != set(_BOOTSTRAP_REQUIRED_WORKFLOWS)
    ):
        raise RuntimeError("pre-import workflow inventory changed")
    for workflow in _BOOTSTRAP_REQUIRED_WORKFLOWS:
        candidates = [
            row for row in rows
            if row.get("workflowName") == workflow and row.get("headSha") == commit
        ]
        if not candidates:
            raise RuntimeError(f"pre-import workflow has no run: {workflow}")
        if recorded is None:
            row = max(candidates, key=lambda item: int(item["databaseId"]))
        else:
            recorded_id = recorded[workflow].get("database_id")
            matches = [
                row for row in candidates
                if int(row.get("databaseId", -1)) == recorded_id
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    f"pre-import recorded workflow disappeared: {workflow}"
                )
            row = matches[0]
        result[workflow] = {
            "database_id": int(row["databaseId"]),
            "head_sha": commit,
            "status": str(row["status"]),
            "conclusion": str(row["conclusion"]),
            "url": str(row["url"]),
        }
        if result[workflow]["status"] != "completed" or result[workflow]["conclusion"] != "success":
            raise RuntimeError(f"pre-import workflow is not green: {workflow}")
    if recorded is not None and recorded != result:
        raise RuntimeError("pre-import recorded workflow evidence changed")
    return result


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


def _bootstrap_authenticate_review_release() -> dict[str, Any]:
    if any(EXP.rglob("__pycache__")):
        raise RuntimeError("pre-import calibration refuses local Python caches")
    review_path = EXP / "reports/calibration_implementation_review.json"
    report_path = EXP / "reports/calibration_implementation_review.md"
    review = _bootstrap_strict_json(review_path)
    implementation = _bootstrap_commit(
        review.get("reviewed_commit"), label="reviewed implementation commit"
    )
    if (
        set(review) != _BOOTSTRAP_REVIEW_KEYS
        or not _bootstrap_exact_int(review.get("schema_version"), 1)
        or review.get("verdict") != "PASS_IMPLEMENTATION"
        or not isinstance(review.get("reviewer"), str)
        or not review["reviewer"].strip()
        or not isinstance(review.get("adversarial_review_rounds"), int)
        or isinstance(review.get("adversarial_review_rounds"), bool)
        or review["adversarial_review_rounds"] < 3
        or not _bootstrap_exact_int(
            review.get("experimental_model_requests_reviewed"), 0
        )
        or not _bootstrap_exact_int(review.get("sampled_model_outputs_reviewed"), 0)
        or any(
            review.get(field) != []
            for field in (
                "hidden_files_read", "qualification_files_read",
                "confirmation_files_read", "benchmark_files_read",
            )
        )
    ):
        raise RuntimeError("pre-import implementation review changed")
    review_relative = str(review_path.relative_to(ROOT))
    report_relative = str(report_path.relative_to(ROOT))
    review_commit = _bootstrap_tracked_commit(review_relative)
    head = _bootstrap_commit(_bootstrap_git("rev-parse", "HEAD"), label="HEAD")
    if (
        review_path.read_bytes()
        != _bootstrap_git_bytes("show", f"{review_commit}:{review_relative}")
        or _bootstrap_sha256(report_path) != review.get("review_report_sha256")
        or report_path.read_bytes()
        != _bootstrap_git_bytes("show", f"{review_commit}:{report_relative}")
        or not _bootstrap_ancestor(implementation, review_commit)
        or not _bootstrap_ancestor(review_commit, head)
    ):
        raise RuntimeError("pre-import review provenance changed")
    for relative in _BOOTSTRAP_RUNTIME_FILES:
        implementation_blob = _bootstrap_git_bytes(
            "show", f"{implementation}:{relative}"
        )
        if _bootstrap_sha256(ROOT / relative) != hashlib.sha256(
            implementation_blob
        ).hexdigest():
            raise RuntimeError(f"pre-import reviewed runtime changed: {relative}")
    if _bootstrap_git("remote", "get-url", "origin") != _CANONICAL_ORIGIN:
        raise RuntimeError("pre-import Git origin URL changed")
    subprocess.run(
        [
            _GIT_EXECUTABLE,
            "--no-replace-objects",
            "-c",
            "core.hooksPath=/dev/null",
            "fetch",
            "--quiet",
            "--no-tags",
            _CANONICAL_ORIGIN,
            "+refs/heads/main:refs/remotes/origin/main",
        ],
        cwd=ROOT,
        check=True,
        env=_bootstrap_child_environment(),
    )
    origin = _bootstrap_commit(
        _bootstrap_git("rev-parse", "origin/main"), label="origin/main"
    )
    if not _bootstrap_ancestor(head, origin):
        raise RuntimeError("pre-import live HEAD is not published on main")
    _bootstrap_validate_ci(implementation, review.get("reviewed_ci"))
    for published_commit in dict.fromkeys((review_commit, head)):
        _bootstrap_validate_ci(published_commit)
    _bootstrap_validate_environment()
    _install_calibration_path_audit(list(_BOOTSTRAP_AUDIT_FILES))
    return {
        "value": review,
        "implementation_commit": implementation,
        "review_commit": review_commit,
        "receipt_sha256": _bootstrap_sha256(review_path),
        "head": head,
    }


def _bootstrap_verify_before_local_imports() -> None:
    sys.dont_write_bytecode = True
    stage = _bootstrap_stage()
    if stage not in {"lock", "run", "analyze"}:
        raise RuntimeError("pre-import calibration stage is invalid or absent")
    review_release = _bootstrap_authenticate_review_release()
    if stage == "lock":
        return
    lock_path = EXP / "runs/calibration/implementation_lock.json"
    lock = _bootstrap_strict_json(lock_path)
    critical = lock.get("critical_files")
    allowed = lock.get("calibration_runtime_files")
    frozen = lock.get("frozen_mechanics_blobs")
    review_binding = lock.get("implementation_review")
    if (
        set(lock) != _BOOTSTRAP_LOCK_KEYS
        or not _bootstrap_exact_int(lock.get("schema_version"), 2)
        or lock.get("stage") != "calibration_implementation_lock"
        or lock.get("authorization") != "interface_calibration_only"
        or lock.get("model") != _MODEL_ID
        or lock.get("revision") != _MODEL_REVISION
        or not isinstance(critical, dict)
        or set(critical) != set(_BOOTSTRAP_CRITICAL_FILES)
        or allowed != list(_BOOTSTRAP_RUNTIME_FILES)
        or not isinstance(frozen, dict)
        or set(frozen) != set(_BOOTSTRAP_FROZEN_MECHANICS)
        or not isinstance(review_binding, dict)
        or set(review_binding)
        != {"verdict", "reviewer", "reviewed_commit", "receipt_sha256", "receipt_commit"}
        or review_binding.get("verdict") != "PASS_IMPLEMENTATION"
        or not _bootstrap_exact_int(lock.get("expected_source_rows"), 48)
        or not _bootstrap_exact_int(lock.get("expected_answer_pairs"), 192)
        or not _bootstrap_exact_int(lock.get("expected_answer_requests"), 384)
        or not _bootstrap_exact_int(
            lock.get("experimental_model_requests_before_lock"), 0
        )
        or not _bootstrap_exact_int(lock.get("sampled_model_outputs_before_lock"), 0)
        or any(
            lock.get(field) != []
            for field in (
                "hidden_files_read", "qualification_files_read",
                "confirmation_files_read", "benchmark_files_read",
            )
        )
    ):
        raise RuntimeError("pre-import implementation lock schema changed")
    implementation = _bootstrap_commit(
        lock["implementation_commit"], label="implementation commit"
    )
    release = _bootstrap_commit(lock["release_commit"], label="release commit")
    review_commit = _bootstrap_commit(
        review_binding["receipt_commit"], label="review receipt commit"
    )
    if (
        implementation != review_release["implementation_commit"]
        or review_commit != review_release["review_commit"]
        or review_binding["reviewed_commit"] != implementation
        or review_binding.get("receipt_sha256") != review_release["receipt_sha256"]
    ):
        raise RuntimeError("pre-import review names another implementation")
    review_path = EXP / "reports/calibration_implementation_review.json"
    review = _bootstrap_strict_json(review_path)
    if (
        set(review) != _BOOTSTRAP_REVIEW_KEYS
        or not _bootstrap_exact_int(review.get("schema_version"), 1)
        or review.get("verdict") != "PASS_IMPLEMENTATION"
        or review.get("reviewed_commit") != implementation
        or review.get("reviewer") != review_binding.get("reviewer")
        or review.get("reviewed_ci") != lock.get("implementation_ci")
        or not _bootstrap_exact_int(
            review.get("experimental_model_requests_reviewed"), 0
        )
        or not _bootstrap_exact_int(review.get("sampled_model_outputs_reviewed"), 0)
        or any(
            review.get(field) != []
            for field in (
                "hidden_files_read", "qualification_files_read",
                "confirmation_files_read", "benchmark_files_read",
            )
        )
        or _bootstrap_sha256(review_path) != review_binding.get("receipt_sha256")
    ):
        raise RuntimeError("pre-import implementation review changed")
    review_relative = str(review_path.relative_to(ROOT))
    report_path = EXP / "reports/calibration_implementation_review.md"
    report_relative = str(report_path.relative_to(ROOT))
    lock_relative = str(lock_path.relative_to(ROOT))
    if (
        _bootstrap_tracked_commit(review_relative) != review_commit
        or review_path.read_bytes()
        != _bootstrap_git_bytes("show", f"{review_commit}:{review_relative}")
        or _bootstrap_sha256(report_path) != review.get("review_report_sha256")
        or report_path.read_bytes()
        != _bootstrap_git_bytes("show", f"{review_commit}:{report_relative}")
    ):
        raise RuntimeError("pre-import review provenance changed")
    lock_commit = _bootstrap_tracked_commit(lock_relative)
    head = _bootstrap_commit(_bootstrap_git("rev-parse", "HEAD"), label="HEAD")
    if (
        head != review_release["head"]
        or lock_path.read_bytes()
        != _bootstrap_git_bytes("show", f"{lock_commit}:{lock_relative}")
        or not _bootstrap_ancestor(implementation, review_commit)
        or not _bootstrap_ancestor(review_commit, release)
        or not _bootstrap_ancestor(release, lock_commit)
        or not _bootstrap_ancestor(lock_commit, head)
    ):
        raise RuntimeError("pre-import implementation ancestry changed")
    for relative in _BOOTSTRAP_CRITICAL_FILES:
        expected = critical.get(relative)
        implementation_blob = _bootstrap_git_bytes(
            "show", f"{implementation}:{relative}"
        )
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or hashlib.sha256(implementation_blob).hexdigest() != expected
        ):
            raise RuntimeError(f"pre-import reviewed blob changed: {relative}")
        if relative in _BOOTSTRAP_RUNTIME_FILES and _bootstrap_sha256(
            ROOT / relative
        ) != expected:
            raise RuntimeError(f"pre-import runtime changed: {relative}")
    for relative in _BOOTSTRAP_FROZEN_MECHANICS:
        if _bootstrap_git("rev-parse", f"{head}:{relative}") != frozen[relative]:
            raise RuntimeError(f"pre-import frozen mechanics changed: {relative}")
    _bootstrap_validate_ci(release, lock.get("release_ci"))
    _bootstrap_validate_ci(lock_commit)


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
    calibration_decision_value,
    engine_config,
    load_analysis_tokenizer,
    load_calibration_inputs,
    run_calibration_transactions,
)
from process_lock import calibration_process_lock  # noqa: E402
from transactions import (  # noqa: E402
    inventory_state,
    read_canonical,
    write_exclusive_durable,
)
from vllm_runner import VLLMRunner  # noqa: E402


RUN_LOCK = EXP / "runs/calibration/run.lock"
RUNNER_PATH = SRC / "vllm_runner.py"


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
    tokenizer = load_analysis_tokenizer(inputs)
    value = calibration_decision_value(
        inputs=inputs,
        raw_dir=RAW_DIR,
        tokenizer=tokenizer,
    )
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
        with calibration_process_lock(RUN_LOCK):
            value = run_live() if args.stage == "run" else analyze()
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
