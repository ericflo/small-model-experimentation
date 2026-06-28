from __future__ import annotations

import py_compile
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .patching import write_files


def run_pytest(
    files: dict[str, str],
    visible_tests: dict[str, str],
    hidden_tests: dict[str, str] | None = None,
    *,
    which: str = "visible",
    timeout: int = 20,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_files(root, files)
        write_files(root, visible_tests)
        if hidden_tests:
            write_files(root, hidden_tests)
        test_paths = sorted(visible_tests)
        if which == "hidden" and hidden_tests:
            test_paths = sorted(hidden_tests)
        elif which == "all" and hidden_tests:
            test_paths = sorted(set(visible_tests) | set(hidden_tests))
        cmd = ["python", "-m", "pytest", "-q", *test_paths]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(root / "src")
        proc = subprocess.run(
            cmd,
            cwd=root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        return {
            "returncode": proc.returncode,
            "passed": proc.returncode == 0,
            "output": proc.stdout[-12000:],
            "cmd": " ".join(cmd),
        }


def syntax_valid(files: dict[str, str], timeout: int = 20) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_files(root, files)
        for path in sorted(files):
            if not path.endswith(".py"):
                continue
            try:
                py_compile.compile(str(root / path), doraise=True)
            except Exception as exc:  # noqa: BLE001
                return False, str(exc)
    return True, ""


def classify_failure(output: str, visible_passed: bool, hidden_passed: bool, apply_status: str) -> str:
    lowered = output.lower()
    if apply_status != "applied":
        return "apply_error"
    if "syntaxerror" in lowered:
        return "syntax"
    if "modulenotfounderror" in lowered or "importerror" in lowered:
        return "import"
    if "timeout" in lowered:
        return "timeout"
    if visible_passed and not hidden_passed:
        return "visible_pass_hidden_fail"
    if "assert" in lowered or "failed" in lowered:
        return "assertion"
    if "traceback" in lowered or "error" in lowered:
        return "traceback"
    return "other"
