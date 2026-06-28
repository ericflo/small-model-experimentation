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


def program_prompt(row: dict[str, Any], mode: str) -> str:
    base = (
        "Write a small Python function that performs the table transformation shown by the example. "
        "Output only Python code, with no markdown and no explanation. "
        "Define exactly one function named transform(table). "
        "The function must return a list of rows, where each row is a list of strings. "
        "Do not read files, do not print, and do not use input(). "
        "Do not include import statements; helper names are already available as globals: re, math, Counter, defaultdict.\n\n"
        f"Example input table:\n{json.dumps(row['input_table'], ensure_ascii=False)}\n\n"
        f"Example output table:\n{json.dumps(row['output_table'], ensure_ascii=False)}\n\n"
    )
    if mode in {"context", "context_v2"}:
        base += (
            "The function will be executed on this new input table after it is verified on the example. "
            "Use the new table only to understand the shape of the data; still write a general transform(table) function.\n\n"
            f"New input table:\n{json.dumps(row['testing_table'], ensure_ascii=False)}\n\n"
        )
    if mode == "context_v2":
        base += (
            "Infer the actual transformation, not just a table that happens to pass the example. "
            "Common transformations include selecting columns, filtering rows, splitting cells, extracting text, "
            "folding or unpivoting columns into rows, transposing, grouping repeated fields, and normalizing strings. "
            "Do not return the input table unchanged unless the example output is exactly the same transformation for all future inputs. "
            "Write clear loops over rows and columns and convert returned cells to strings.\n\n"
        )
    return base + "Python code:"


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
