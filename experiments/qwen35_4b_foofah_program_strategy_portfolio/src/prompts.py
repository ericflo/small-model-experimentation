from __future__ import annotations

import json
from typing import Any

from src.table_utils import compact_table


def direct_prompt(row: dict[str, Any]) -> str:
    return (
        "You are given examples of a table transformation and a new input table. "
        "Return only the transformed table as valid JSON, with no explanation, no markdown, and no prose. "
        "The first character of your answer must be '[' and the last character must be ']'. "
        "The table must be a JSON array of rows, where each row is an array of strings.\n\n"
        f"Example input table:\n{json.dumps(row['input_table'], ensure_ascii=False)}\n\n"
        f"Example output table:\n{json.dumps(row['output_table'], ensure_ascii=False)}\n\n"
        f"New input table:\n{json.dumps(row['testing_table'], ensure_ascii=False)}\n\n"
        "Transformed output table:"
    )


def _common(row: dict[str, Any]) -> str:
    return (
        "Output only Python code. Define exactly one function named transform(table). "
        "The function must return a list of rows, where each row is a list of strings. "
        "Do not print, do not read files, and do not call input(). "
        "Do not include import statements; helper names are already available: re, math, Counter, defaultdict.\n\n"
        f"Example input table:\n{json.dumps(row['input_table'], ensure_ascii=False)}\n\n"
        f"Example output table:\n{json.dumps(row['output_table'], ensure_ascii=False)}\n\n"
        f"New input table:\n{json.dumps(row['testing_table'], ensure_ascii=False)}\n\n"
    )


def program_prompt(row: dict[str, Any], variant: str) -> str:
    common = _common(row)
    leads = {
        "verified_structural": (
            "Write a Python function that performs the table transformation shown by the example. "
            "Use the new table only to understand the shape of the data; still write a general transform(table) function. "
            "Prefer simple deterministic table logic over memorizing values.\n\n"
        ),
        "row_column_rule": (
            "Infer how output rows are constructed from input rows and columns, then implement only that rule. "
            "Pay close attention to headers, row expansion, grouping, splitting, merging, sorting, counting, and normalization. "
            "Do not special-case the visible table values.\n\n"
        ),
        "shape_first": (
            "Focus first on the output shape: number of rows, number of columns, whether headers are kept, and how each output row maps to input cells. "
            "Then implement the general rule in clear Python loops and list operations.\n\n"
        ),
        "split_fold_unpivot": (
            "Assume the transformation may be a table reshape: fold, unfold, unpivot, pivot, split one cell into many rows, or merge many cells into one row. "
            "Write code that captures the general reshape pattern rather than the specific example values.\n\n"
        ),
        "header_aware": (
            "Use column headers and row labels as semantic clues. "
            "Infer which columns are identifiers, measures, categories, or generated headers, then implement the transformation generically.\n\n"
        ),
        "cell_parser": (
            "Focus on cell-level parsing and normalization. "
            "Use string operations and re when cells need to be split, cleaned, reordered, joined, or extracted before forming the output table.\n\n"
        ),
        "aggregation_grouping": (
            "Assume the transformation may require grouping rows, counting repeated values, aggregating fields, sorting groups, or expanding grouped records. "
            "Implement a general grouping or counting rule if the example supports one.\n\n"
        ),
        "transpose_restructure": (
            "Look for row/column transposition, matrix rotation, header-to-row conversion, row-to-header conversion, and other structural rearrangements. "
            "Implement the table restructuring rule using loops and index arithmetic.\n\n"
        ),
    }
    if variant not in leads:
        raise ValueError(f"unknown variant: {variant}")
    return leads[variant] + common + "Python code:"


def repair_prompt(row: dict[str, Any], variant: str, code: str, feedback: str, repair_round: int) -> str:
    return (
        "Repair the Python function below so it implements the table transformation generally. "
        "It failed on the visible example. Return a full replacement implementation of exactly one function named transform(table). "
        "Output only Python code, with no markdown, no explanation, and no comments. "
        "Do not include import statements; helper names are already available: re, math, Counter, defaultdict. "
        "Do not hard-code the visible input, visible output, new input, or specific row values. "
        "A repair that only passes the visible example but does not generalize is wrong.\n\n"
        f"Strategy variant: {variant}\n"
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

