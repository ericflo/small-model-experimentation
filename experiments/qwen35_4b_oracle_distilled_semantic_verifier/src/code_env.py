from __future__ import annotations

import ast
import doctest
import json
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable


PASS_LETTER = "A"
FAIL_LETTER = "B"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def indent_block(text: str, prefix: str) -> str:
    return "\n".join(prefix + line if line.strip() else line for line in text.splitlines())


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))[:80]


def execute_asserts(code: str, tests: list[str], setup_code: str = "", timeout_s: float = 5.0) -> dict[str, Any]:
    script = f"""
import collections
import functools
import itertools
import math
import re
import string
import typing
from typing import *

{setup_code}

{code}

TESTS = {tests!r}
results = []
for test in TESTS:
    try:
        exec(test, globals())
        results.append({{"passed": True, "error": ""}})
    except BaseException as exc:
        results.append({{"passed": False, "error": type(exc).__name__}})
print(json.dumps({{"results": results}}))
"""
    script = "import json\n" + script
    with tempfile.TemporaryDirectory(prefix="oracle_verifier_exec_") as tmp:
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
            )
        except subprocess.TimeoutExpired:
            return {"passed": [False for _ in tests], "errors": ["TIMEOUT" for _ in tests], "timeout": True}
    if result.returncode != 0:
        return {"passed": [False for _ in tests], "errors": ["SCRIPT_ERROR" for _ in tests], "stderr": result.stderr[-1000:]}
    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        rows = payload["results"]
        return {"passed": [bool(row["passed"]) for row in rows], "errors": [str(row.get("error", "")) for row in rows]}
    except Exception:
        return {"passed": [False for _ in tests], "errors": ["BAD_JSON" for _ in tests], "stdout": result.stdout[-1000:]}


def execute_humaneval(code: str, entry_point: str, public_tests: list[dict[str, Any]], official_test: str, timeout_s: float = 5.0) -> dict[str, Any]:
    public_asserts = [
        f"assert {entry_point}(*{test['args']!r}) == {test['expected_expr']}"
        for test in public_tests
    ]
    official_script = f"""
import collections
import functools
import itertools
import math
import re
import string
import typing
from typing import *

{code}

public_tests = {public_asserts!r}
public_results = []
for test in public_tests:
    try:
        exec(test, globals())
        public_results.append({{"passed": True, "error": ""}})
    except BaseException as exc:
        public_results.append({{"passed": False, "error": type(exc).__name__}})

official_ok = False
try:
{indent_block(official_test, "    ")}
    check({entry_point})
    official_ok = True
except BaseException:
    official_ok = False
print(json.dumps({{"public": public_results, "official_ok": official_ok}}))
"""
    script = "import json\n" + official_script
    with tempfile.TemporaryDirectory(prefix="oracle_verifier_heval_") as tmp:
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
            )
        except subprocess.TimeoutExpired:
            return {"public_passed": [False for _ in public_tests], "official_ok": False, "timeout": True}
    if result.returncode != 0:
        return {"public_passed": [False for _ in public_tests], "official_ok": False, "stderr": result.stderr[-1000:]}
    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        return {
            "public_passed": [bool(row["passed"]) for row in payload["public"]],
            "official_ok": bool(payload["official_ok"]),
        }
    except Exception:
        return {"public_passed": [False for _ in public_tests], "official_ok": False, "stdout": result.stdout[-1000:]}


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


def extract_function_source(code: str, entry_point: str) -> tuple[str, ast.FunctionDef] | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    lines = code.splitlines(keepends=True)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == entry_point:
            start = node.lineno - 1
            end = node.end_lineno or node.lineno
            return "".join(lines[start:end]), node
    return None


def default_function_variant(code: str, entry_point: str, return_expr: str) -> str | None:
    found = extract_function_source(code, entry_point)
    if found is None:
        return None
    fn_source, node = found
    indent = " " * node.col_offset
    body_indent = indent + "    "
    header = fn_source.splitlines()[0]
    replacement = f"{header}\n{body_indent}return {return_expr}\n"
    return code.replace(fn_source, replacement, 1)


def mutate_code(code: str) -> list[tuple[str, str]]:
    variants: list[tuple[str, str]] = []
    replacements = [
        ("<=", "<"),
        (">=", ">"),
        ("==", "!="),
        ("!=", "=="),
        ("<", "<="),
        (">", ">="),
        ("+", "-"),
        ("-", "+"),
        ("*", "+"),
        ("//", "%"),
        ("%", "+"),
        (" and ", " or "),
        (" or ", " and "),
        ("True", "False"),
        ("False", "True"),
        ("min(", "max("),
        ("max(", "min("),
        ("sorted(", "list("),
    ]
    for old, new in replacements:
        if old in code:
            variants.append((f"replace_{safe_name(old)}_with_{safe_name(new)}", code.replace(old, new, 1)))
    for match in list(re.finditer(r"\b-?\d+\b", code))[:16]:
        value = int(match.group(0))
        for delta in (-1, 1):
            variants.append((f"const_{value}_to_{value + delta}", code[: match.start()] + str(value + delta) + code[match.end() :]))
    lines = code.splitlines(keepends=True)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("return ") and len(stripped) > len("return "):
            mutated = list(lines)
            mutated[i] = line[: len(line) - len(line.lstrip())] + "return None\n"
            variants.append((f"return_none_line_{i}", "".join(mutated)))
            break
    return variants


def make_candidates(code: str, max_candidates: int, seed: int, entry_point: str | None = None) -> list[dict[str, str]]:
    raw: list[tuple[str, str]] = [("seed_solution", code)]
    raw.extend(mutate_code(code))
    if entry_point:
        for expr in ["None", "0", "1", "-1", "False", "True", "[]", "''"]:
            variant = default_function_variant(code, entry_point, expr)
            if variant is not None:
                raw.append((f"default_return_{safe_name(expr)}", variant))
    dedup: dict[str, tuple[str, str]] = {}
    for source, candidate_code in raw:
        try:
            ast.parse(candidate_code)
        except SyntaxError:
            continue
        dedup.setdefault(candidate_code, (source, candidate_code))
    items = list(dedup.values())
    rng = random.Random(seed)
    rng.shuffle(items)
    if code not in {candidate_code for _, candidate_code in items[:max_candidates]}:
        chosen = items[: max_candidates - 1] + [("seed_solution", code)]
    else:
        chosen = items[:max_candidates]
    return [
        {"candidate_id": f"cand_{i:02d}", "source": source, "code": candidate_code}
        for i, (source, candidate_code) in enumerate(chosen)
    ]


def extract_doctest_public_tests(prompt: str, entry_point: str, limit: int) -> list[dict[str, Any]]:
    tests: list[dict[str, Any]] = []
    try:
        tree = ast.parse(prompt)
        fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == entry_point)
        text = ast.get_docstring(fn) or ""
    except Exception:
        text = prompt
    parser = doctest.DocTestParser()
    try:
        examples = parser.get_examples(text)
    except ValueError:
        return []
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
        try:
            args = [ast.literal_eval(arg) for arg in expr.args]
            expected_expr = example.want.strip()
            ast.parse(expected_expr, mode="eval")
        except Exception:
            continue
        tests.append({"args": args, "expected_expr": expected_expr, "source": source})
        if len(tests) >= limit:
            break
    return tests


def build_mbpp_record(raw: dict[str, Any], split: str, index: int, visible_tests: int, max_candidates: int, seed: int) -> tuple[dict[str, Any] | None, str | None]:
    tests = list(raw.get("test_list") or [])
    if len(tests) < visible_tests + 1:
        return None, "too_few_tests"
    entry_point = parse_entry_from_assert(tests[0])
    if entry_point is None:
        return None, "entry_parse_failed"
    code = raw["code"].replace("\r\n", "\n")
    visible = tests[:visible_tests]
    hidden = tests[visible_tests:] + list(raw.get("challenge_test_list") or [])
    candidates = make_candidates(code, max_candidates=max_candidates, seed=seed + index * 97, entry_point=entry_point)
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        public_result = execute_asserts(candidate["code"], visible, setup_code=raw.get("test_setup_code") or "")
        hidden_result = execute_asserts(candidate["code"], visible + hidden, setup_code=raw.get("test_setup_code") or "")
        rows.append(
            {
                **candidate,
                "public_passed": public_result["passed"],
                "public_all_pass": all(public_result["passed"]),
                "hidden_all_pass": all(hidden_result["passed"]),
            }
        )
    if not any(row["hidden_all_pass"] for row in rows):
        return None, "no_positive"
    if not any(row["public_all_pass"] and not row["hidden_all_pass"] for row in rows):
        return None, "no_hard_negative"
    return {
        "record_id": f"{split}_mbpp_{raw['task_id']}_{index:04d}",
        "dataset": "mbpp",
        "split": split,
        "task_id": str(raw["task_id"]),
        "entry_point": entry_point,
        "task_text": raw["text"],
        "public_tests": visible,
        "hidden_test_count": len(hidden),
        "candidates": rows,
    }, None


def build_humaneval_record(raw: dict[str, Any], split: str, index: int, visible_tests: int, max_candidates: int, seed: int) -> tuple[dict[str, Any] | None, str | None]:
    prompt = raw["prompt"]
    solution = raw["canonical_solution"]
    entry_point = raw["entry_point"]
    public = extract_doctest_public_tests(prompt, entry_point, visible_tests)
    if len(public) < visible_tests:
        return None, f"too_few_public_tests:{len(public)}"
    code = prompt + solution
    candidates = make_candidates(code, max_candidates=max_candidates, seed=seed + index * 101, entry_point=entry_point)
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        result = execute_humaneval(candidate["code"], entry_point, public, raw["test"])
        rows.append(
            {
                **candidate,
                "public_passed": result["public_passed"],
                "public_all_pass": all(result["public_passed"]),
                "hidden_all_pass": bool(result["official_ok"]),
            }
        )
    if not any(row["hidden_all_pass"] for row in rows):
        return None, "no_positive"
    if not any(row["public_all_pass"] and not row["hidden_all_pass"] for row in rows):
        return None, "no_hard_negative"
    return {
        "record_id": f"{split}_humaneval_{safe_name(raw['task_id'])}_{index:04d}",
        "dataset": "humaneval",
        "split": split,
        "task_id": str(raw["task_id"]),
        "entry_point": entry_point,
        "task_text": prompt,
        "public_tests": [test["source"] + " == " + test["expected_expr"] for test in public],
        "hidden_test_count": 1,
        "candidates": rows,
    }, None


def candidate_prompt(record: dict[str, Any], candidate: dict[str, Any], include_public_status: bool = True) -> str:
    lines: list[str] = []
    lines.append("You are a semantic verifier for Python programming tasks.")
    lines.append("Decide whether the candidate implementation will pass the hidden tests.")
    lines.append("A = PASS hidden tests. B = FAIL hidden tests.")
    lines.append("Reply with exactly A or B.")
    lines.append("")
    lines.append(f"Dataset: {record['dataset']}")
    lines.append(f"Task id: {record['task_id']}")
    lines.append("Task:")
    lines.append(str(record["task_text"]).strip())
    lines.append("")
    lines.append("Public tests:")
    for test in record["public_tests"]:
        lines.append(f"- {test}")
    if include_public_status:
        status = "PASS" if candidate["public_all_pass"] else "FAIL"
        lines.append(f"Candidate public-test status: {status}")
    lines.append("")
    lines.append("Candidate implementation:")
    lines.append("```python")
    code = candidate["code"]
    if len(code) > 2400:
        code = code[:2400] + "\n# ... truncated ..."
    lines.append(code)
    lines.append("```")
    lines.append("")
    lines.append("Answer: ")
    return "\n".join(lines)


def build_candidate_examples(records: list[dict[str, Any]], split: str, balance: bool, seed: int) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    rng = random.Random(seed)
    for record in records:
        candidates = [candidate for candidate in record["candidates"] if candidate["public_all_pass"]]
        positives = [candidate for candidate in candidates if candidate["hidden_all_pass"]]
        negatives = [candidate for candidate in candidates if not candidate["hidden_all_pass"]]
        if balance and positives and negatives:
            if len(negatives) > len(positives) * 2:
                negatives = rng.sample(negatives, len(positives) * 2)
            selected = positives + negatives
        else:
            selected = candidates
        for candidate in selected:
            label = PASS_LETTER if candidate["hidden_all_pass"] else FAIL_LETTER
            examples.append(
                {
                    "record_id": record["record_id"],
                    "candidate_id": candidate["candidate_id"],
                    "dataset": record["dataset"],
                    "split": split,
                    "label": label,
                    "label_index": 0 if label == PASS_LETTER else 1,
                    "hidden_all_pass": candidate["hidden_all_pass"],
                    "public_all_pass": candidate["public_all_pass"],
                    "prompt": candidate_prompt(record, candidate),
                }
            )
    rng.shuffle(examples)
    return examples


def visible_candidates(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [candidate for candidate in record["candidates"] if candidate["public_all_pass"]]


def record_coverage(record: dict[str, Any]) -> bool:
    return any(candidate["hidden_all_pass"] for candidate in visible_candidates(record))

