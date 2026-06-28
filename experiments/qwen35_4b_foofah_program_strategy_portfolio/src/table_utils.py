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


def table_key(table: list[list[Any]] | None) -> str | None:
    norm = normalize_table(table)
    if norm is None:
        return None
    return json.dumps(norm, sort_keys=True, ensure_ascii=False)


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
            try:
                obj, _end = json.JSONDecoder().raw_decode(cand)
            except Exception:
                continue
        if isinstance(obj, list) and all(isinstance(row, list) for row in obj):
            return True, normalize_table(obj), cand
    return False, None, stripped[:500]


def extract_code(text: str) -> tuple[bool, str, str]:
    stripped = text.strip()
    candidates = []
    if "```" in stripped:
        parts = stripped.split("```")
        for idx in range(1, len(parts), 2):
            block = parts[idx]
            if block.lstrip().startswith("python"):
                block = block.lstrip()[6:]
            candidates.append(block.strip())
    candidates.append(stripped)
    marker = "def transform"
    if marker in stripped:
        candidates.append(stripped[stripped.find(marker) :].strip())
    for cand in candidates:
        if marker in cand:
            return True, cand, "found_transform"
    return False, stripped, "missing_transform"


def compact_table(table: list[list[str]] | None, max_chars: int = 1800) -> str:
    if table is None:
        return "null"
    text = json.dumps(table, ensure_ascii=False)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"... <truncated {len(text) - max_chars} chars>"


def diff_tables(expected: list[list[str]], actual: list[list[str]] | None, max_rows: int = 8) -> str:
    if actual is None:
        return "No actual table was produced."
    lines = [
        f"Expected row count: {len(expected)}",
        f"Actual row count: {len(actual)}",
    ]
    mismatches = 0
    for idx in range(max(len(expected), len(actual))):
        exp = expected[idx] if idx < len(expected) else None
        act = actual[idx] if idx < len(actual) else None
        if exp != act:
            lines.append(f"First mismatch row {idx}: expected={json.dumps(exp, ensure_ascii=False)} actual={json.dumps(act, ensure_ascii=False)}")
            mismatches += 1
            if mismatches >= max_rows:
                break
    if mismatches == 0:
        lines.append("Rows match exactly.")
    return "\n".join(lines)
