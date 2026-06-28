#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import math
import re
import sys
from collections import Counter, defaultdict
from typing import Any


FORBIDDEN_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "dir",
    "eval",
    "exec",
    "exit",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "quit",
    "setattr",
    "vars",
}


SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


def normalize_table(table: Any) -> list[list[str]]:
    if not isinstance(table, list) or not all(isinstance(row, list) for row in table):
        raise ValueError("transform must return list[list]")
    return [[str(cell) for cell in row] for row in table]


def strip_safe_imports(code: str) -> str:
    kept = []
    safe_lines = {
        "import re",
        "import math",
        "from collections import Counter",
        "from collections import defaultdict",
        "from collections import Counter, defaultdict",
        "from collections import defaultdict, Counter",
    }
    for line in code.splitlines():
        if line.strip() in safe_lines:
            continue
        kept.append(line)
    return "\n".join(kept)


def guard_ast(code: str) -> None:
    tree = ast.parse(code)
    if not any(isinstance(node, ast.FunctionDef) and node.name == "transform" for node in tree.body):
        raise ValueError("missing transform(table)")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("imports are not allowed")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("dunder attributes are not allowed")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("dunder names are not allowed")
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in FORBIDDEN_CALLS:
                raise ValueError(f"forbidden call: {func.id}")
            if isinstance(func, ast.Attribute) and func.attr in FORBIDDEN_CALLS:
                raise ValueError(f"forbidden method call: {func.attr}")


def execute(code: str, table: list[list[str]]) -> list[list[str]]:
    code = strip_safe_imports(code)
    guard_ast(code)
    namespace: dict[str, Any] = {
        "__builtins__": SAFE_BUILTINS,
        "Counter": Counter,
        "defaultdict": defaultdict,
        "math": math,
        "re": re,
    }
    exec(compile(code, "<generated_transform>", "exec"), namespace, namespace)
    transform = namespace.get("transform")
    if not callable(transform):
        raise ValueError("transform is not callable")
    return normalize_table(transform(table))


def main() -> None:
    try:
        payload = json.loads(sys.stdin.read())
        result = execute(payload["code"], payload["table"])
        print(json.dumps({"ok": True, "result": result}, ensure_ascii=False))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": type(exc).__name__, "message": str(exc)[:500]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
