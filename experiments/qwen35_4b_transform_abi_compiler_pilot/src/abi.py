from __future__ import annotations

import csv
import io
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from typing import Any


DATE_FORMATS = [
    "%Y-%m-%d",
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%Y%m%d",
    "%d-%b-%Y",
    "%b %d, %Y",
    "%B %d, %Y",
]

REGEXES = {
    "order_id": r"(?:ORD|ORDER)[- ]?(\d{3,8})",
    "ticket": r"#([A-Za-z0-9-]+)",
    "email": r"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})",
    "zip5": r"\b(\d{5})(?:-\d{4})?\b",
    "digits": r"(\d+)",
    "sku_like": r"([A-Za-z]{2,4}[-_ ]?\d{2,6})",
}


@dataclass(frozen=True)
class Example:
    inp: Any
    out: Any
    kind: str


@dataclass(frozen=True)
class Task:
    task_id: str
    domain: str
    depth: int
    description: str
    visible: tuple[Example, ...]
    hidden: tuple[Example, ...]
    adversarial: tuple[Example, ...]
    target_program: dict[str, Any]


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def rows_to_csv(rows: list[dict[str, Any]]) -> str:
    out = io.StringIO()
    fields = list(rows[0])
    writer = csv.DictWriter(out, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue().strip()


def parse_csv_text(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))


def as_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        return parse_csv_text(value)
    return [dict(row) for row in value]


def as_float(value: Any) -> float:
    if value in ("", None):
        raise ValueError("missing numeric")
    return float(str(value).replace("$", "").replace(",", ""))


def maybe_int(value: float) -> int | float:
    return int(value) if value == int(value) else round(value, 6)


def normalize_scalar(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, list):
        return [normalize_scalar(x) for x in value]
    if isinstance(value, dict):
        return {k: normalize_scalar(v) for k, v in sorted(value.items())}
    return value


def equal(a: Any, b: Any) -> bool:
    return normalize_scalar(a) == normalize_scalar(b)


def safe_execute(program: dict[str, Any], inp: Any) -> tuple[bool, Any]:
    try:
        return True, execute_program(program, inp)
    except Exception as exc:
        return False, type(exc).__name__


def parse_date(text: str) -> datetime:
    clean = str(text).strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(clean, fmt)
        except ValueError:
            pass
    raise ValueError(f"unparsed date {text!r}")


def normalize_phone(inp: Any) -> str:
    text = str(inp)
    text = re.sub(r"(?:ext|x)\.?\s*\d+\s*$", "", text, flags=re.I)
    digits = re.sub(r"\D", "", text)
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    if len(digits) != 10:
        return ""
    return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"


def title_case(text: Any) -> str:
    def cap_piece(piece: str) -> str:
        return "-".join(part.capitalize() for part in piece.split("-"))

    return " ".join(cap_piece(part) for part in str(text).split())


def apply_op(op: dict[str, Any], value: Any) -> Any:
    name = op["op"]
    if name == "identity":
        return value
    if name == "filter_eq":
        col, target = op["col"], str(op["value"])
        return [row for row in as_rows(value) if str(row.get(col, "")) == target]
    if name == "filter_nonempty":
        col = op["col"]
        return [row for row in as_rows(value) if str(row.get(col, "")).strip() != ""]
    if name == "filter_num_cmp":
        col, cmp, threshold = op["col"], op["cmp"], float(op["threshold"])
        rows = as_rows(value)
        if cmp == "gt":
            return [row for row in rows if as_float(row.get(col)) > threshold]
        if cmp == "ge":
            return [row for row in rows if as_float(row.get(col)) >= threshold]
        if cmp == "lt":
            return [row for row in rows if as_float(row.get(col)) < threshold]
        if cmp == "le":
            return [row for row in rows if as_float(row.get(col)) <= threshold]
        raise ValueError(f"bad cmp {cmp}")
    if name == "select_cols":
        cols = list(op["cols"])
        return [{col: row.get(col, "") for col in cols} for row in as_rows(value)]
    if name == "sort_by":
        col = op["col"]
        reverse = bool(op.get("reverse", False))
        numeric = bool(op.get("numeric", False))
        key = (lambda row: as_float(row.get(col))) if numeric else (lambda row: str(row.get(col, "")))
        return sorted(as_rows(value), key=key, reverse=reverse)
    if name == "group_count":
        col = op["col"]
        return dict(Counter(str(row.get(col, "")) for row in as_rows(value)))
    if name == "group_sum":
        key_col, value_col = op["key_col"], op["value_col"]
        out: dict[str, float] = defaultdict(float)
        for row in as_rows(value):
            out[str(row.get(key_col, ""))] += as_float(row.get(value_col))
        return {k: maybe_int(v) for k, v in sorted(out.items())}
    if name == "group_avg":
        key_col, value_col = op["key_col"], op["value_col"]
        sums: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        for row in as_rows(value):
            key = str(row.get(key_col, ""))
            sums[key] += as_float(row.get(value_col))
            counts[key] += 1
        return {k: round(sums[k] / counts[k], 3) for k in sorted(sums)}
    if name == "normalize_col":
        col, mode = op["col"], op["mode"]
        rows = []
        for row in as_rows(value):
            row = dict(row)
            text = str(row.get(col, ""))
            if mode == "strip":
                row[col] = text.strip()
            elif mode == "lower":
                row[col] = text.lower()
            elif mode == "strip_lower":
                row[col] = text.strip().lower()
            elif mode == "upper":
                row[col] = text.upper()
            else:
                raise ValueError(f"bad mode {mode}")
            rows.append(row)
        return rows
    if name == "split_full_name":
        rows = []
        for row in as_rows(value):
            row = dict(row)
            first, _, last = str(row.get("name", "")).strip().partition(" ")
            row["first_name"] = first
            row["last_name"] = last
            rows.append(row)
        return rows
    if name == "parse_money_col":
        col = op["col"]
        out_col = op.get("out_col", f"{col}_num")
        rows = []
        for row in as_rows(value):
            row = dict(row)
            row[out_col] = maybe_int(as_float(row.get(col)))
            rows.append(row)
        return rows
    if name == "normalize_date_iso":
        return parse_date(str(value)).strftime("%Y-%m-%d")
    if name == "normalize_date_us":
        return parse_date(str(value)).strftime("%m/%d/%Y")
    if name == "extract_regex":
        match = re.search(REGEXES[op["pattern"]], str(value), re.I)
        return match.group(1) if match else ""
    if name == "bool_regex":
        return bool(re.search(REGEXES[op["pattern"]], str(value), re.I))
    if name == "normalize_phone":
        return normalize_phone(value)
    if name == "normalize_sku":
        return re.sub(r"[^A-Za-z0-9]", "", str(value)).upper()
    if name == "mask_account":
        digits = re.sub(r"\D", "", str(value))
        return "****" + digits[-4:]
    if name == "slugify":
        text = re.sub(r"[^A-Za-z0-9]+", "-", str(value).strip().lower())
        return text.strip("-")
    if name == "title_case":
        return title_case(value)
    if name == "extract_state_zip":
        match = re.search(r"\b([A-Z]{2})\s+(\d{5})(?:-\d{4})?\b", str(value))
        return {"state": match.group(1), "zip": match.group(2)} if match else {"state": "", "zip": ""}
    if name == "luhn_valid":
        digits = [int(ch) for ch in re.sub(r"\D", "", str(value))]
        total = 0
        parity = len(digits) % 2
        for idx, digit in enumerate(digits):
            if idx % 2 == parity:
                digit *= 2
                if digit > 9:
                    digit -= 9
            total += digit
        return total % 10 == 0 and bool(digits)
    raise ValueError(f"unknown op {name}")


def execute_program(program: dict[str, Any], inp: Any) -> Any:
    value = inp
    for step in program["steps"]:
        value = apply_op(step, value)
    return value


def observed_columns(task: Task) -> list[str]:
    rows: list[dict[str, Any]] = []
    for ex in task.visible:
        try:
            rows.extend(as_rows(ex.inp))
        except Exception:
            pass
    return sorted({key for row in rows for key in row})


def observed_values(task: Task, col: str) -> list[str]:
    values = set()
    for ex in task.visible:
        try:
            for row in as_rows(ex.inp):
                text = str(row.get(col, ""))
                if text:
                    values.add(text)
        except Exception:
            pass
    return sorted(values)


def primitive_steps(task: Task) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = [{"op": "identity"}]
    columns = observed_columns(task)
    for col in columns:
        for value in observed_values(task, col):
            steps.append({"op": "filter_eq", "col": col, "value": value})
        steps.append({"op": "filter_nonempty", "col": col})
        for threshold in [0, 1, 10, 50, 100, 1000]:
            for cmp in ["gt", "ge", "lt", "le"]:
                steps.append({"op": "filter_num_cmp", "col": col, "cmp": cmp, "threshold": threshold})
        for reverse in [False, True]:
            steps.append({"op": "sort_by", "col": col, "numeric": False, "reverse": reverse})
            steps.append({"op": "sort_by", "col": col, "numeric": True, "reverse": reverse})
        steps.append({"op": "group_count", "col": col})
        for mode in ["strip", "lower", "strip_lower", "upper"]:
            steps.append({"op": "normalize_col", "col": col, "mode": mode})
        steps.append({"op": "parse_money_col", "col": col, "out_col": f"{col}_num"})
        for value_col in columns:
            if value_col != col:
                steps.append({"op": "group_sum", "key_col": col, "value_col": value_col})
                steps.append({"op": "group_avg", "key_col": col, "value_col": value_col})
    for width in [1, 2, 3]:
        for cols in combinations(columns, width):
            steps.append({"op": "select_cols", "cols": list(cols)})
    steps.append({"op": "split_full_name"})
    for pattern in REGEXES:
        steps.append({"op": "extract_regex", "pattern": pattern})
        steps.append({"op": "bool_regex", "pattern": pattern})
    for op in [
        "normalize_date_iso",
        "normalize_date_us",
        "normalize_phone",
        "normalize_sku",
        "mask_account",
        "slugify",
        "title_case",
        "extract_state_zip",
        "luhn_valid",
    ]:
        steps.append({"op": op})
    # Deduplicate while preserving order.
    seen = set()
    out = []
    for step in steps:
        key = canonical_json(step)
        if key not in seen:
            out.append(step)
            seen.add(key)
    return out


def valid_on_visible(program: dict[str, Any], task: Task) -> bool:
    for ex in task.visible:
        ok, out = safe_execute(program, ex.inp)
        if not ok or not equal(out, ex.out):
            return False
    return True


def candidate_programs(task: Task, max_depth: int = 3, limit: int = 192) -> list[dict[str, Any]]:
    target_key = canonical_json(task.target_program)
    candidates: dict[str, dict[str, Any]] = {}
    prims = primitive_steps(task)
    # All depth-1 visible-consistent programs.
    for step in prims:
        prog = {"steps": [step]}
        if valid_on_visible(prog, task):
            candidates.setdefault(canonical_json(prog), prog)
    # Composition candidates are built around visible-consistent prefixes to keep the set bounded.
    if max_depth >= 2:
        prefixes = [prog for prog in candidates.values() if len(prog["steps"]) == 1][:64]
        for prefix in prefixes:
            for step in prims:
                prog = {"steps": [*prefix["steps"], step]}
                if valid_on_visible(prog, task):
                    candidates.setdefault(canonical_json(prog), prog)
                    if len(candidates) >= limit:
                        break
    if max_depth >= 3:
        prefixes = [prog for prog in candidates.values() if len(prog["steps"]) == 2][:48]
        for prefix in prefixes:
            for step in prims:
                prog = {"steps": [*prefix["steps"], step]}
                if valid_on_visible(prog, task):
                    candidates.setdefault(canonical_json(prog), prog)
                    if len(candidates) >= limit:
                        break
    candidates[target_key] = task.target_program
    ordered = [candidates[key] for key in sorted(candidates)]
    if len(ordered) <= limit:
        return ordered
    target = task.target_program
    trimmed = ordered[: limit - 1]
    if canonical_json(target) not in {canonical_json(item) for item in trimmed}:
        trimmed.append(target)
    return trimmed


def examples_pass(program: dict[str, Any], examples: list[Example] | tuple[Example, ...]) -> bool:
    for ex in examples:
        ok, out = safe_execute(program, ex.inp)
        if not ok or not equal(out, ex.out):
            return False
    return True


def prompt_for_task(task: Task) -> str:
    visible = []
    for ex in task.visible:
        visible.append({"input": ex.inp, "output": ex.out})
    ops = [
        "filter_eq",
        "filter_nonempty",
        "filter_num_cmp",
        "select_cols",
        "sort_by",
        "group_count",
        "group_sum",
        "group_avg",
        "normalize_col",
        "split_full_name",
        "parse_money_col",
        "normalize_date_iso",
        "normalize_date_us",
        "extract_regex",
        "bool_regex",
        "normalize_phone",
        "normalize_sku",
        "mask_account",
        "slugify",
        "title_case",
        "extract_state_zip",
        "luhn_valid",
    ]
    return (
        "Compile the task into a compact JSON ABI program. "
        "A program has the form {\"steps\":[...]} and steps run in order. "
        f"Available ops: {', '.join(ops)}.\n"
        f"Task: {task.description}\n"
        f"Examples: {json.dumps(visible, sort_keys=True)}\n"
        "Program:"
    )
