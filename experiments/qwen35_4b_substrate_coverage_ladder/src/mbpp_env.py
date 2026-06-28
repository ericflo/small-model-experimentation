from __future__ import annotations

import ast
import json
import re
import resource
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


SAFE_IMPORT_ROOTS = {
    "bisect",
    "collections",
    "copy",
    "dataclasses",
    "functools",
    "heapq",
    "itertools",
    "math",
    "operator",
    "re",
    "string",
    "typing",
}

DANGEROUS_NAMES = {
    "__import__",
    "compile",
    "eval",
    "exec",
    "exit",
    "globals",
    "input",
    "locals",
    "open",
    "quit",
}

DANGEROUS_ATTR_ROOTS = {
    "builtins",
    "ctypes",
    "importlib",
    "multiprocessing",
    "os",
    "pathlib",
    "pickle",
    "shutil",
    "signal",
    "socket",
    "subprocess",
    "sys",
}


def parse_entry_from_assert(assertion: str) -> str | None:
    try:
        tree = ast.parse(assertion)
    except SyntaxError:
        return None
    if not tree.body or not isinstance(tree.body[0], ast.Assert):
        return None
    test = tree.body[0].test
    call: ast.Call | None = None
    if isinstance(test, ast.Compare) and isinstance(test.left, ast.Call):
        call = test.left
    elif isinstance(test, ast.Call):
        call = test
    if call is None:
        return None
    if isinstance(call.func, ast.Name):
        return call.func.id
    return None


def parse_assert_case(assertion: str) -> dict[str, Any] | None:
    try:
        tree = ast.parse(assertion)
    except SyntaxError:
        return None
    if not tree.body or not isinstance(tree.body[0], ast.Assert):
        return None
    test = tree.body[0].test
    if not isinstance(test, ast.Compare) or not isinstance(test.left, ast.Call) or not test.comparators:
        return None
    try:
        return {
            "assert_src": assertion,
            "call_expr": ast.unparse(test.left),
            "expected_expr": ast.unparse(test.comparators[0]),
        }
    except Exception:
        return None


def infer_entry_point(test_list: list[str]) -> str | None:
    for assertion in test_list:
        entry = parse_entry_from_assert(assertion)
        if entry:
            return entry
    return None


def infer_arity_from_cases(public_cases: list[dict[str, Any]], entry_point: str) -> int:
    for case in public_cases:
        try:
            expr = ast.parse(case["call_expr"], mode="eval").body
        except SyntaxError:
            continue
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id == entry_point:
            return len(expr.args)
    return 1


def static_safety_check(code: str) -> tuple[bool, str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False, "syntax_error"
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [node.module or ""] if isinstance(node, ast.ImportFrom) else [alias.name for alias in node.names]
            for name in names:
                root = name.split(".")[0]
                if root not in SAFE_IMPORT_ROOTS:
                    return False, f"blocked_import:{root}"
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in DANGEROUS_NAMES:
                return False, f"blocked_call:{func.id}"
            if isinstance(func, ast.Attribute):
                root = func
                while isinstance(root, ast.Attribute):
                    root = root.value
                if isinstance(root, ast.Name) and root.id in DANGEROUS_ATTR_ROOTS:
                    return False, f"blocked_attr:{root.id}"
        elif isinstance(node, ast.Attribute):
            root = node
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id in DANGEROUS_ATTR_ROOTS:
                return False, f"blocked_attr:{root.id}"
    return True, "ok"


def _limit_child() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
    resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024, 1024 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))


def run_python_script(script: str, timeout_s: float = 5.0) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="substrate_ladder_exec_") as tmp:
        path = Path(tmp) / "candidate.py"
        path.write_text(script, encoding="utf-8")
        try:
            result = subprocess.run(
                [sys.executable, "-I", str(path)],
                cwd=tmp,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_s,
                check=False,
                preexec_fn=_limit_child,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "timeout": True, "stdout": "", "stderr": "TIMEOUT"}
    if result.returncode != 0:
        return {"ok": False, "timeout": False, "stdout": result.stdout[-1000:], "stderr": result.stderr[-1000:]}
    return {"ok": True, "timeout": False, "stdout": result.stdout, "stderr": result.stderr[-1000:]}


def execute_public_and_asserts(
    code: str,
    public_cases: list[dict[str, Any]],
    hidden_asserts: list[str],
    setup_code: str = "",
    timeout_s: float = 5.0,
) -> dict[str, Any]:
    safe, reason = static_safety_check(code)
    if not safe:
        return {
            "safe": False,
            "safety_reason": reason,
            "public_passed": [False for _ in public_cases],
            "public_outputs": [],
            "visible_all_pass": False,
            "full_pass": False,
        }
    script = f"""
import collections
import functools
import heapq
import itertools
import math
import re
import string
from typing import *

{setup_code}

{code}

public_cases = {public_cases!r}
hidden_asserts = {hidden_asserts!r}
public_results = []
for case in public_cases:
    try:
        value = eval(case["call_expr"], globals())
        expected = eval(case["expected_expr"], globals())
        public_results.append({{"passed": value == expected, "output": repr(value), "error": ""}})
    except BaseException as exc:
        public_results.append({{"passed": False, "output": "", "error": type(exc).__name__}})

hidden_ok = True
for assertion in hidden_asserts:
    try:
        exec(assertion, globals())
    except BaseException:
        hidden_ok = False
        break

print(json.dumps({{"public": public_results, "hidden_ok": hidden_ok}}))
"""
    result = run_python_script("import json\n" + script, timeout_s=timeout_s)
    if not result["ok"]:
        return {
            "safe": True,
            "safety_reason": "ok",
            "public_passed": [False for _ in public_cases],
            "public_outputs": [],
            "visible_all_pass": False,
            "full_pass": False,
            "runtime_error": result.get("stderr", ""),
            "timeout": result.get("timeout", False),
        }
    try:
        payload = json.loads(result["stdout"].strip().splitlines()[-1])
    except Exception:
        return {
            "safe": True,
            "safety_reason": "ok",
            "public_passed": [False for _ in public_cases],
            "public_outputs": [],
            "visible_all_pass": False,
            "full_pass": False,
            "runtime_error": "bad_json",
        }
    public_passed = [bool(row["passed"]) for row in payload["public"]]
    public_outputs = [str(row.get("output", "")) for row in payload["public"]]
    visible_all = all(public_passed) if public_cases else True
    return {
        "safe": True,
        "safety_reason": "ok",
        "public_passed": public_passed,
        "public_outputs": public_outputs,
        "visible_all_pass": visible_all,
        "full_pass": bool(visible_all and payload["hidden_ok"]),
    }


def sanitize_reference_function(code: str, target_entry: str, target_arity: int) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    fn: ast.FunctionDef | None = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            if len(node.args.args) == target_arity:
                fn = node
                break
    if fn is None:
        return None
    fn.name = target_entry
    tree.body = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef))]
    try:
        return ast.unparse(tree) + "\n"
    except Exception:
        return None


def tokenize_text(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[a-zA-Z_][a-zA-Z_0-9]+", text.lower()) if len(tok) > 2}
