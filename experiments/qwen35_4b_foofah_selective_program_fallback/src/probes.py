from __future__ import annotations

import copy
import json
from typing import Any


def _dedupe(tables: list[tuple[str, list[list[str]]]]) -> list[dict[str, Any]]:
    seen = set()
    out = []
    for name, table in tables:
        key = json.dumps(table, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        out.append({"name": name, "table": table})
    return out


def _reverse_data(table: list[list[str]]) -> list[list[str]] | None:
    if len(table) <= 2:
        return None
    return [table[0]] + list(reversed(table[1:]))


def _head_subset(table: list[list[str]]) -> list[list[str]] | None:
    if len(table) <= 2:
        return None
    return table[: max(2, min(len(table), 4))]


def _tail_subset(table: list[list[str]]) -> list[list[str]] | None:
    if len(table) <= 3:
        return None
    return [table[0]] + table[-min(3, len(table) - 1) :]


def _duplicate_data_row(table: list[list[str]]) -> list[list[str]] | None:
    if len(table) <= 1:
        return None
    out = copy.deepcopy(table)
    out.append(copy.deepcopy(table[1]))
    return out


def _mix_visible_header_with_test_rows(visible: list[list[str]], test: list[list[str]]) -> list[list[str]] | None:
    if not visible or not test:
        return None
    width = len(visible[0])
    rows = [r for r in test[1:] if len(r) == width]
    if not rows:
        return None
    return [visible[0]] + rows[: min(3, len(rows))]


def make_probe_inputs(row: dict[str, Any], max_probes: int = 3) -> list[dict[str, Any]]:
    visible = row["input_table"]
    test = row["testing_table"]
    candidates: list[tuple[str, list[list[str]]]] = []
    for name, fn, source in [
        ("reverse_test_data_rows", _reverse_data, test),
        ("head_test_subset", _head_subset, test),
        ("tail_test_subset", _tail_subset, test),
        ("duplicate_test_first_data_row", _duplicate_data_row, test),
        ("reverse_visible_data_rows", _reverse_data, visible),
        ("duplicate_visible_first_data_row", _duplicate_data_row, visible),
    ]:
        value = fn(source)
        if value is not None:
            candidates.append((name, value))
    mixed = _mix_visible_header_with_test_rows(visible, test)
    if mixed is not None:
        candidates.append(("visible_header_test_rows", mixed))
    return _dedupe(candidates)[:max_probes]

