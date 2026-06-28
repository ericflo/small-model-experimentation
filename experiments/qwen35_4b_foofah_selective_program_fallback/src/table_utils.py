from __future__ import annotations

import json
from typing import Any


def normalize_table(table: list[list[Any]] | None) -> list[list[str]] | None:
    if table is None:
        return None
    return [[str(cell) for cell in row] for row in table]


def equal_table(a: list[list[Any]] | None, b: list[list[Any]] | None) -> bool:
    aa = normalize_table(a)
    bb = normalize_table(b)
    return aa is not None and bb is not None and aa == bb


def extract_json_table(text: str) -> tuple[bool, list[list[str]] | None, str]:
    stripped = text.strip()
    candidates = []
    if stripped:
        candidates.append(stripped)
    start = stripped.find("[")
    end = stripped.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidates.append(stripped[start : end + 1])
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        if isinstance(obj, list) and all(isinstance(row, list) for row in obj):
            return True, normalize_table(obj), cand
    return False, None, stripped[:500]


def direct_prompt_for_table(row: dict[str, Any], table: list[list[str]]) -> str:
    return (
        "You are given one or more examples of a table transformation and a new input table. "
        "Return only the transformed table as valid JSON, with no explanation, no markdown, and no prose. "
        "The first character of your answer must be '[' and the last character must be ']'. "
        "The table must be a JSON array of rows, where each row is an array of strings.\n\n"
        f"Example input table:\n{json.dumps(row['input_table'], ensure_ascii=False)}\n\n"
        f"Example output table:\n{json.dumps(row['output_table'], ensure_ascii=False)}\n\n"
        f"New input table:\n{json.dumps(table, ensure_ascii=False)}\n\n"
        "Transformed output table:"
    )

