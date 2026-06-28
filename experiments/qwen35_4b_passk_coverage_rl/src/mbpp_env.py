from __future__ import annotations

import ast
import doctest
import hashlib
import json
import re
import resource
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from .model_utils import code_chat_prompt, estimate_tokens


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

DANGEROUS_NAMES = {"__import__", "compile", "eval", "exec", "exit", "globals", "input", "locals", "open", "quit"}
DANGEROUS_ATTR_ROOTS = {"builtins", "ctypes", "importlib", "multiprocessing", "os", "pathlib", "pickle", "shutil", "signal", "socket", "subprocess", "sys"}


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
    return call.func.id if isinstance(call.func, ast.Name) else None


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
        return {"assert_src": assertion, "call_expr": ast.unparse(test.left), "expected_expr": ast.unparse(test.comparators[0])}
    except Exception:
        return None


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
        return bool(body)
    return False


def longest_parseable_prefix(text: str, entry_point: str) -> str | None:
    lines = text.splitlines()
    for end in range(len(lines), 0, -1):
        candidate = "\n".join(lines[:end]).strip() + "\n"
        if contains_entry_function(candidate, entry_point):
            return candidate
    return None


def extract_candidate_code(raw_completion: str, entry_point: str) -> tuple[str | None, str]:
    cleaned = strip_markdown(raw_completion)
    prefix = longest_parseable_prefix(cleaned, entry_point)
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
    with tempfile.TemporaryDirectory(prefix="passk_rl_exec_") as tmp:
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


def execute_candidate(
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
            "failure_bits": "S",
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

hidden_results = []
for assertion in hidden_asserts:
    try:
        exec(assertion, globals())
        hidden_results.append({{"passed": True, "error": ""}})
    except BaseException as exc:
        hidden_results.append({{"passed": False, "error": type(exc).__name__}})

print(json.dumps({{"public": public_results, "hidden": hidden_results}}))
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
            "failure_bits": "T" if result.get("timeout") else "E",
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
            "failure_bits": "J",
            "runtime_error": "bad_json",
        }
    public_passed = [bool(row["passed"]) for row in payload["public"]]
    public_outputs = [str(row.get("output", "")) for row in payload["public"]]
    hidden_passed = [bool(row["passed"]) for row in payload["hidden"]]
    visible_all = all(public_passed) if public_cases else True
    full_pass = bool(visible_all and all(hidden_passed))
    return {
        "safe": True,
        "safety_reason": "ok",
        "public_passed": public_passed,
        "public_outputs": public_outputs,
        "hidden_passed": hidden_passed,
        "visible_all_pass": visible_all,
        "full_pass": full_pass,
        "failure_bits": "".join("1" if item else "0" for item in hidden_passed),
    }


def candidate_from_completion(raw_completion: str, record: dict[str, Any], source: str, order: int, tokenizer: Any | None = None, prompt: str = "") -> dict[str, Any]:
    code, parse_status = extract_candidate_code(raw_completion, record["entry_point"])
    if code is None:
        result = {
            "safe": False,
            "safety_reason": "parse_failed",
            "public_passed": [False for _ in record["public_cases"]],
            "public_outputs": [],
            "visible_all_pass": False,
            "full_pass": False,
            "failure_bits": "P",
        }
        code = ""
    else:
        result = execute_candidate(code, record["public_cases"], record["hidden_asserts"], record.get("setup_code", ""), record.get("timeout_s", 5.0))
    prompt_tokens = estimate_tokens(tokenizer, prompt) if tokenizer is not None else 0
    completion_tokens = estimate_tokens(tokenizer, raw_completion) if tokenizer is not None else 0
    return {
        "candidate_id": f"cand_{order:04d}",
        "source": source,
        "order": order,
        "raw_completion": raw_completion,
        "code": code,
        "code_hash": hashlib.sha1(code.encode("utf-8")).hexdigest()[:16],
        "parse_status": parse_status,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "forward_tokens": prompt_tokens + completion_tokens,
        **result,
    }


def load_mbpp_records(split: str, count: int, offset: int, visible_tests: int, timeout_s: float, tokenizer: Any | None = None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/mbpp", "full", split=split)
    rows: list[dict[str, Any]] = []
    for item in ds:
        tests = list(item.get("test_list", []))
        entry = None
        for assertion in tests:
            entry = parse_entry_from_assert(assertion)
            if entry:
                break
        if not entry:
            continue
        public_cases = [case for assertion in tests[:visible_tests] if (case := parse_assert_case(assertion))]
        hidden_asserts = tests[visible_tests:] + list(item.get("challenge_test_list", []))
        prompt = code_chat_prompt(tokenizer, item.get("text", ""), entry, [case["assert_src"] for case in public_cases]) if tokenizer is not None else ""
        rows.append(
            {
                "dataset": "mbpp",
                "split": split,
                "task_id": int(item["task_id"]),
                "record_id": f"mbpp_{split}_{item['task_id']}",
                "task_text": item.get("text", ""),
                "entry_point": entry,
                "public_cases": public_cases,
                "hidden_asserts": hidden_asserts,
                "all_asserts": tests + list(item.get("challenge_test_list", [])),
                "setup_code": item.get("test_setup_code", ""),
                "timeout_s": timeout_s,
                "prompt": prompt,
            }
        )
    return rows[offset : offset + count]


def recompute_metrics(record: dict[str, Any]) -> None:
    candidates = record.get("candidates", [])
    record["candidate_count"] = len(candidates)
    record["parse_success_count"] = sum(1 for cand in candidates if cand.get("parse_status") == "parsed")
    record["visible_candidate_count"] = sum(1 for cand in candidates if cand.get("visible_all_pass"))
    record["hidden_pass_candidate_count"] = sum(1 for cand in candidates if cand.get("full_pass"))
    record["coverage"] = any(cand.get("full_pass") for cand in candidates)
    record["visible_coverage"] = any(cand.get("visible_all_pass") for cand in candidates)
    hashes = {cand.get("code_hash") for cand in candidates if cand.get("code")}
    failure = {cand.get("failure_bits", "") for cand in candidates}
    record["distinct_program_count"] = len(hashes)
    record["distinct_program_rate"] = len(hashes) / len(candidates) if candidates else 0.0
    record["distinct_functional_count"] = len(failure)
    record["distinct_functional_rate"] = len(failure) / len(candidates) if candidates else 0.0
    record["pass1_proxy"] = bool(candidates and candidates[0].get("full_pass"))


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    if n == 0:
        return {}
    return {
        "task_count": n,
        "coverage": sum(1 for row in records if row.get("coverage")) / n,
        "visible_coverage": sum(1 for row in records if row.get("visible_coverage")) / n,
        "pass1_proxy": sum(1 for row in records if row.get("pass1_proxy")) / n,
        "mean_candidates": sum(row.get("candidate_count", 0) for row in records) / n,
        "mean_distinct_program_rate": sum(row.get("distinct_program_rate", 0.0) for row in records) / n,
        "mean_distinct_functional_rate": sum(row.get("distinct_functional_rate", 0.0) for row in records) / n,
        "hidden_pass_candidates": sum(row.get("hidden_pass_candidate_count", 0) for row in records),
        "visible_candidates": sum(row.get("visible_candidate_count", 0) for row in records),
        "forward_tokens": sum(sum(cand.get("forward_tokens", 0) for cand in row.get("candidates", [])) for row in records),
    }
