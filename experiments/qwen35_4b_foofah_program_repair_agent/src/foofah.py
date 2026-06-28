from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def normalize_table(table: list[list[Any]]) -> list[list[str]]:
    return [[str(cell) for cell in row] for row in table]


def equal_table(a: list[list[Any]], b: list[list[Any]]) -> bool:
    return normalize_table(a) == normalize_table(b)


def load_cases(source: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(source.glob("*.txt")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if not all(k in data for k in ["InputTable", "OutputTable", "TestingTable", "TestAnswer"]):
            continue
        rows.append(
            {
                "file": path.name,
                "test_name": str(data.get("TestName", path.stem)),
                "num_samples": int(data.get("NumSamples", 0)),
                "input_table": normalize_table(data["InputTable"]),
                "output_table": normalize_table(data["OutputTable"]),
                "testing_table": normalize_table(data["TestingTable"]),
                "test_answer": normalize_table(data["TestAnswer"]),
            }
        )
    return rows


def family_name(file_name: str, test_name: str) -> str:
    stem = file_name.replace(".txt", "")
    parts = stem.split("_")
    if len(parts) >= 3 and parts[1].isdigit():
        return f"synthetic_{parts[1]}"
    if stem.startswith("exp0_"):
        return "_".join(parts[1:-1]) or test_name
    return test_name


def direct_prompt(row: dict[str, Any]) -> str:
    return (
        "You are given one or more examples of a table transformation and a new input table. "
        "Return only the transformed table as valid JSON, with no explanation, no markdown, and no prose. "
        "The first character of your answer must be '[' and the last character must be ']'. "
        "The table must be a JSON array of rows, where each row is an array of strings.\n\n"
        f"Example input table:\n{json.dumps(row['input_table'], ensure_ascii=False)}\n\n"
        f"Example output table:\n{json.dumps(row['output_table'], ensure_ascii=False)}\n\n"
        f"New input table:\n{json.dumps(row['testing_table'], ensure_ascii=False)}\n\n"
        "Transformed output table:"
    )


def initial_program_prompt(row: dict[str, Any]) -> str:
    return (
        "Write a Python function that performs the table transformation shown by the example. "
        "Output only Python code, with no markdown and no explanation. "
        "Define exactly one function named transform(table). "
        "The function must return a list of rows, where each row is a list of strings. "
        "Do not read files, do not print, and do not use input(). "
        "Do not include import statements; helper names are already available as globals: re, math, Counter, defaultdict.\n\n"
        f"Example input table:\n{json.dumps(row['input_table'], ensure_ascii=False)}\n\n"
        f"Example output table:\n{json.dumps(row['output_table'], ensure_ascii=False)}\n\n"
        "The function will be executed on this new input table after it is verified on the example. "
        "Use the new table only to understand the shape of the data; still write a general transform(table) function.\n\n"
        f"New input table:\n{json.dumps(row['testing_table'], ensure_ascii=False)}\n\n"
        "Python code:"
    )


def repair_prompt(row: dict[str, Any], code: str, feedback: str, round_index: int) -> str:
    new_rows = row["testing_table"]
    visible_rows = row["input_table"]
    visible_out = row["output_table"]
    return (
        "Repair the Python table-transform function below. "
        "It was executed on the visible example and failed. "
        "Return a full replacement implementation of exactly one function named transform(table). "
        "Output only Python code, with no markdown, no explanation, and no comments. "
        "Do not include import statements; helper names are already available as globals: re, math, Counter, defaultdict. "
        "The function must return list[list[str]]. "
        "Do not hard-code the visible input, visible output, or specific row values. "
        "A repair that only passes the visible example but does not generalize is wrong. "
        "Infer the transformation rule from input-output structure: row counts, column counts, repeated fields, split/fold/unpivot patterns, headers, and string normalization. "
        "Pay special attention to how the output size should scale from the visible input to the new input.\n\n"
        f"Repair round: {round_index}\n\n"
        f"Visible input shape: rows={len(visible_rows)}, columns={[len(r) for r in visible_rows[:8]]}\n"
        f"Visible output shape: rows={len(visible_out)}, columns={[len(r) for r in visible_out[:8]]}\n"
        f"New input shape: rows={len(new_rows)}, columns={[len(r) for r in new_rows[:12]]}\n\n"
        f"Visible example input:\n{json.dumps(row['input_table'], ensure_ascii=False)}\n\n"
        f"Expected visible output:\n{json.dumps(row['output_table'], ensure_ascii=False)}\n\n"
        f"New input shape to generalize to:\n{json.dumps(row['testing_table'], ensure_ascii=False)[:3500]}\n\n"
        f"Failure feedback:\n{feedback}\n\n"
        f"Previous code:\n{code[:6000]}\n\n"
        "Replacement Python code:"
    )


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
