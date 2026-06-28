from __future__ import annotations

import ast
import doctest
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


def indent_block(text: str, prefix: str) -> str:
    return "\n".join(prefix + line if line.strip() else line for line in text.splitlines())


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


def extract_doctest_public_tests(prompt: str, entry_point: str, limit: int) -> list[dict[str, Any]]:
    try:
        tree = ast.parse(prompt)
        fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == entry_point)
        text = ast.get_docstring(fn) or ""
    except Exception:
        text = prompt
    try:
        examples = doctest.DocTestParser().get_examples(text)
    except ValueError:
        return []
    tests: list[dict[str, Any]] = []
    for example in examples:
        source = example.source.strip()
        if not source:
            continue
        try:
            expr = ast.parse(source, mode="eval").body
        except SyntaxError:
            continue
        if not isinstance(expr, ast.Call) or not isinstance(expr.func, ast.Name) or expr.func.id != entry_point:
            continue
        if expr.keywords:
            continue
        expected_expr = example.want.strip()
        try:
            ast.parse(expected_expr, mode="eval")
        except SyntaxError:
            continue
        tests.append(
            {
                "assert_src": f"assert {source} == {expected_expr}",
                "call_expr": source,
                "expected_expr": expected_expr,
            }
        )
        if len(tests) >= limit:
            break
    return tests


def strip_markdown(text: str) -> str:
    fence = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1)
    text = text.replace("\r\n", "\n")
    lines = text.splitlines()
    start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(("def ", "class ", "from ", "import ")):
            start = i
            break
    return "\n".join(lines[start:]).strip() + "\n"


def contains_entry_function(code: str, entry_point: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != entry_point:
            continue
        body = list(node.body)
        if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
            body = body[1:]
        if body:
            return True
    return False


def longest_parseable_prefix(text: str, entry_point: str) -> str | None:
    lines = text.splitlines()
    for end in range(len(lines), 0, -1):
        candidate = "\n".join(lines[:end]).strip() + "\n"
        if contains_entry_function(candidate, entry_point):
            return candidate
    return None


def extract_candidate_code(raw_completion: str, entry_point: str, continuation_prompt: str | None = None) -> tuple[str | None, str]:
    cleaned = strip_markdown(raw_completion)
    attempts = [cleaned]
    if continuation_prompt:
        attempts.append(continuation_prompt.rstrip() + "\n" + raw_completion.strip() + "\n")
        attempts.append(continuation_prompt.rstrip() + "\n" + cleaned)
    for attempt in attempts:
        prefix = longest_parseable_prefix(strip_markdown(attempt), entry_point)
        if prefix is not None:
            return prefix, "parsed"
    return None, "parse_failed"


def static_safety_check(code: str) -> tuple[bool, str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False, "syntax_error"
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                names = [alias.name for alias in node.names]
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
    with tempfile.TemporaryDirectory(prefix="real_sample_exec_") as tmp:
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
    full_pass = bool(visible_all and payload["hidden_ok"])
    return {
        "safe": True,
        "safety_reason": "ok",
        "public_passed": public_passed,
        "public_outputs": public_outputs,
        "visible_all_pass": visible_all,
        "full_pass": full_pass,
    }


def execute_humaneval(
    code: str,
    public_cases: list[dict[str, Any]],
    entry_point: str,
    official_test: str,
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

{code}

public_cases = {public_cases!r}
public_results = []
for case in public_cases:
    try:
        value = eval(case["call_expr"], globals())
        expected = eval(case["expected_expr"], globals())
        public_results.append({{"passed": value == expected, "output": repr(value), "error": ""}})
    except BaseException as exc:
        public_results.append({{"passed": False, "output": "", "error": type(exc).__name__}})

official_ok = False
try:
{indent_block(official_test, "    ")}
    check({entry_point})
    official_ok = True
except BaseException:
    official_ok = False

print(json.dumps({{"public": public_results, "official_ok": official_ok}}))
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
        "full_pass": bool(visible_all and payload["official_ok"]),
    }


def public_signature(candidate: dict[str, Any]) -> str:
    if candidate.get("public_outputs"):
        return "|".join(candidate["public_outputs"])
    if candidate.get("public_passed"):
        return "|".join("1" if item else "0" for item in candidate["public_passed"])
    return "no_public_tests"


def mbpp_sampling_prompt(raw: dict[str, Any], entry_point: str, visible_tests: list[str]) -> str:
    tests = "\n".join(visible_tests)
    task_text = raw.get("task_text", raw.get("text", ""))
    return (
        "Return only Python code. Do not use markdown.\n"
        f"Task: {task_text}\n"
        f"Define a function named `{entry_point}` and any helper classes or functions needed.\n"
        "The code should satisfy these public tests:\n"
        f"{tests}\n"
    )


def humaneval_sampling_prompt(raw: dict[str, Any], public_cases: list[dict[str, Any]]) -> str:
    tests = "\n".join(case["assert_src"] for case in public_cases) or "(no public tests in prompt)"
    prompt_text = raw.get("task_text", raw.get("prompt", ""))
    return (
        "Return only Python code. Do not use markdown.\n"
        "Complete the function below as a full, runnable Python definition. Do not include tests.\n\n"
        f"{prompt_text}\n"
        "Public examples:\n"
        f"{tests}\n"
    )


def verifier_prompt(record: dict[str, Any], candidate: dict[str, Any]) -> str:
    public_tests = "\n".join(case["assert_src"] for case in record["public_cases"]) or "(none)"
    status = "PASS" if candidate["visible_all_pass"] else "FAIL"
    return (
        "You are a semantic verifier for Python benchmark solutions.\n"
        "Choose A if the candidate is likely to pass the hidden tests. Choose B if it is likely to fail hidden tests.\n\n"
        f"Dataset: {record['dataset']}\n"
        f"Task id: {record['task_id']}\n"
        f"Task:\n{record['task_text']}\n\n"
        f"Public tests:\n{public_tests}\n\n"
        f"Candidate public-test status: {status}\n\n"
        f"Candidate code:\n{candidate['code']}\n\n"
        "Answer with one letter.\nAnswer: "
    )


def stop_prompt(record: dict[str, Any], state: dict[str, Any]) -> str:
    return (
        "You control a code-generation budget. Choose A to STOP and commit the current top candidate. "
        "Choose B to sample more code.\n\n"
        f"Dataset: {record['dataset']}\n"
        f"Task id: {record['task_id']}\n"
        f"Samples seen: {state['budget']}\n"
        f"Visible-passing candidates seen: {state['visible_count']}\n"
        f"Top verifier score: {state.get('top_score', 0.0):.4f}\n"
        f"Verifier score margin: {state.get('score_margin', 0.0):.4f}\n"
        f"Top candidate source: {state.get('selected_source', 'none')}\n"
        f"Top candidate public-test status: {state.get('selected_public_status', 'NONE')}\n\n"
        f"Task:\n{record['task_text']}\n\n"
        f"Top candidate code:\n{state.get('selected_code', '')}\n\n"
        "Answer with one letter.\nAnswer: "
    )
