from __future__ import annotations

import json
from typing import Any

from src.table_utils import compact_table


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


def program_prompt(row: dict[str, Any], variant: str) -> str:
    if variant == "verified_structural":
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
    common = (
        "Output only Python code. Define exactly one function named transform(table). "
        "The function must return a list of rows, where each row is a list of strings. "
        "Do not print, do not read files, and do not call input(). "
        "Do not include import statements; helper names are already available: re, math, Counter, defaultdict.\n\n"
        f"Example input table:\n{json.dumps(row['input_table'], ensure_ascii=False)}\n\n"
        f"Example output table:\n{json.dumps(row['output_table'], ensure_ascii=False)}\n\n"
        f"New input table:\n{json.dumps(row['testing_table'], ensure_ascii=False)}\n\n"
    )
    if variant == "structural_python":
        lead = (
            "Write a general Python table transformation function. "
            "Infer the rule from row counts, column counts, headers, repeated fields, split/fold/unpivot patterns, and string normalization. "
            "Use the new input only to understand the data shape; do not hard-code its final answer.\n\n"
        )
    elif variant == "minimal_python":
        lead = (
            "Write the shortest clear implementation of the transformation rule. "
            "Prefer simple loops, comprehensions, slicing, and dictionary/counter operations. "
            "Do not special-case the visible table values.\n\n"
        )
    elif variant == "row_column_rule":
        lead = (
            "First infer how each output row is constructed from input rows and columns, then implement only that rule in Python. "
            "Pay attention to whether headers are kept, whether rows are expanded or grouped, and whether cells are split, merged, sorted, counted, or normalized. "
            "Return only the code, not the reasoning.\n\n"
        )
    else:
        raise ValueError(f"unknown variant: {variant}")
    return lead + common + "Python code:"


def repair_prompt(row: dict[str, Any], variant: str, code: str, feedback: str, repair_round: int) -> str:
    return (
        "Repair the Python function below so it implements the table transformation generally. "
        "It failed on the visible example. Return a full replacement implementation of exactly one function named transform(table). "
        "Output only Python code, with no markdown, no explanation, and no comments. "
        "Do not include import statements; helper names are already available: re, math, Counter, defaultdict. "
        "Do not hard-code the visible input, visible output, new input, or specific row values. "
        "A repair that only passes the visible example but does not generalize is wrong.\n\n"
        f"Variant: {variant}\n"
        f"Repair round: {repair_round}\n"
        f"Visible input shape: rows={len(row['input_table'])}, columns={[len(r) for r in row['input_table'][:8]]}\n"
        f"Visible output shape: rows={len(row['output_table'])}, columns={[len(r) for r in row['output_table'][:8]]}\n"
        f"New input shape: rows={len(row['testing_table'])}, columns={[len(r) for r in row['testing_table'][:12]]}\n\n"
        f"Visible example input:\n{json.dumps(row['input_table'], ensure_ascii=False)}\n\n"
        f"Expected visible output:\n{json.dumps(row['output_table'], ensure_ascii=False)}\n\n"
        f"New input table for shape reference:\n{compact_table(row['testing_table'], 3500)}\n\n"
        f"Failure feedback:\n{feedback}\n\n"
        f"Previous code:\n{code[:6000]}\n\n"
        "Replacement Python code:"
    )
