from __future__ import annotations

import ast
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

from src.code_env import mbpp_sampling_prompt
from src.diversity_utils import sampling_prompt
from src.model_utils import code_chat_prompt


STRATEGIES: list[dict[str, str]] = [
    {
        "key": "DIRECT",
        "description": "Use the simplest direct implementation with minimal helper structure.",
    },
    {
        "key": "LOOP",
        "description": "Use explicit loops and state variables to build the answer step by step.",
    },
    {
        "key": "COMPREHENSION",
        "description": "Use comprehensions, generator expressions, or compact functional iteration.",
    },
    {
        "key": "SORTING",
        "description": "Sort, order, or rank the data before computing the result.",
    },
    {
        "key": "SET_DICT",
        "description": "Use sets, dictionaries, counters, or maps to track relationships.",
    },
    {
        "key": "RECURSION",
        "description": "Use recursion or a decomposed helper function.",
    },
    {
        "key": "STRING_REGEX",
        "description": "Use string parsing, character operations, or regular expressions.",
    },
    {
        "key": "MATH",
        "description": "Use arithmetic, formulas, numeric loops, or math-library operations.",
    },
]

STRATEGY_BY_KEY = {row["key"]: row for row in STRATEGIES}
STRATEGY_KEYS = [row["key"] for row in STRATEGIES]


def strategy_prefix(strategy_key: str) -> str:
    strategy = STRATEGY_BY_KEY[strategy_key]
    return f"Strategy key: {strategy['key']}\nStrategy instruction: {strategy['description']}\n"


def strategy_prompt(record: dict[str, Any], strategy_key: str, tokenizer: Any) -> str:
    visible_tests = [case["assert_src"] for case in record.get("public_cases", [])]
    base = mbpp_sampling_prompt(record, record["entry_point"], visible_tests)
    prompt = (
        "Return only Python code. Do not use markdown.\n"
        "Follow the requested strategy key. Different strategy keys should produce genuinely different valid approaches when possible.\n\n"
        f"{strategy_prefix(strategy_key)}\n"
        f"{base}"
    )
    return code_chat_prompt(tokenizer, prompt)


def plain_prompt(record: dict[str, Any], tokenizer: Any) -> str:
    return sampling_prompt(record, tokenizer)


def normalized_code_hash(code: str) -> str:
    compact = re.sub(r"\s+", " ", code.strip())
    return hashlib.sha1(compact.encode("utf-8")).hexdigest()[:12]


def structural_features(code: str, entry_point: str) -> dict[str, bool]:
    features = {
        "recursion": False,
        "regex": bool(re.search(r"\bre\b|import re|from re import", code)),
        "sort": ("sorted(" in code or ".sort(" in code),
        "set_dict": any(token in code for token in ("set(", "dict(", "Counter(", "defaultdict(", "{}")),
        "comprehension": False,
        "loop": False,
        "math": any(token in code for token in ("import math", "from math import", "math.", "sum(", "min(", "max(", "abs(")),
        "string_regex": any(
            token in code
            for token in (".split(", ".join(", ".replace(", ".strip(", ".lower(", ".upper(", ".isalpha(", ".isdigit(")
        ),
    }
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return features
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.While)):
            features["loop"] = True
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            features["comprehension"] = True
        if isinstance(node, (ast.Set, ast.Dict)):
            features["set_dict"] = True
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == entry_point:
                features["recursion"] = True
            if isinstance(node.func, ast.Name) and node.func.id in {"set", "dict", "sum", "min", "max", "abs", "round", "pow"}:
                if node.func.id in {"set", "dict"}:
                    features["set_dict"] = True
                else:
                    features["math"] = True
            if isinstance(node.func, ast.Attribute):
                attr = node.func.attr
                if attr in {"sort"}:
                    features["sort"] = True
                if attr in {"split", "join", "replace", "strip", "lower", "upper", "isalpha", "isdigit"}:
                    features["string_regex"] = True
    return features


def classify_strategy(code: str, entry_point: str) -> str:
    features = structural_features(code, entry_point)
    if features["recursion"]:
        return "RECURSION"
    if features["regex"] or features["string_regex"]:
        return "STRING_REGEX"
    if features["sort"]:
        return "SORTING"
    if features["set_dict"]:
        return "SET_DICT"
    if features["comprehension"]:
        return "COMPREHENSION"
    if features["math"]:
        return "MATH"
    if features["loop"]:
        return "LOOP"
    return "DIRECT"


def shuffled_key(original_key: str, rng: random.Random) -> str:
    choices = [key for key in STRATEGY_KEYS if key != original_key]
    return rng.choice(choices)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else ""), encoding="utf-8")
