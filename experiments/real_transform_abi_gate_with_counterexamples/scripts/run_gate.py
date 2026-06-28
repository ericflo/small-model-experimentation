#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


@dataclass
class Example:
    inp: Any
    out: Any
    kind: str


@dataclass
class Task:
    task_id: str
    domain: str
    description: str
    visible: list[Example]
    hidden: list[Example]
    adversarial: list[Example]


@dataclass
class Candidate:
    name: str
    category: str
    depth: int
    func: Callable[[Any], Any]
    program: dict[str, Any]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


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
    return int(value) if value == int(value) else value


def normalize_scalar(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, list):
        return [normalize_scalar(x) for x in value]
    if isinstance(value, dict):
        return {k: normalize_scalar(v) for k, v in value.items()}
    return value


def equal(a: Any, b: Any) -> bool:
    return normalize_scalar(a) == normalize_scalar(b)


def safe_call(func: Callable[[Any], Any], inp: Any) -> tuple[bool, Any]:
    try:
        return True, func(inp)
    except Exception as exc:
        return False, type(exc).__name__


def all_distinct_chars(inp: Any) -> bool:
    text = str(inp)
    return len(set(text)) == len(text)


def filter_eq(col: str, value: str) -> Callable[[Any], list[dict[str, Any]]]:
    return lambda inp: [row for row in as_rows(inp) if str(row.get(col, "")) == value]


def filter_nonempty(col: str) -> Callable[[Any], list[dict[str, Any]]]:
    return lambda inp: [row for row in as_rows(inp) if str(row.get(col, "")).strip() != ""]


def filter_num_cmp(col: str, op: str, threshold: float) -> Callable[[Any], list[dict[str, Any]]]:
    def f(inp: Any) -> list[dict[str, Any]]:
        rows = as_rows(inp)
        if op == "gt":
            return [row for row in rows if as_float(row.get(col)) > threshold]
        if op == "ge":
            return [row for row in rows if as_float(row.get(col)) >= threshold]
        if op == "lt":
            return [row for row in rows if as_float(row.get(col)) < threshold]
        return [row for row in rows if as_float(row.get(col)) <= threshold]
    return f


def select_cols(cols: tuple[str, ...]) -> Callable[[Any], list[dict[str, Any]]]:
    return lambda inp: [{col: row.get(col, "") for col in cols} for row in as_rows(inp)]


def sort_by(col: str, numeric: bool = False, reverse: bool = False) -> Callable[[Any], list[dict[str, Any]]]:
    def f(inp: Any) -> list[dict[str, Any]]:
        rows = as_rows(inp)
        key = (lambda row: as_float(row.get(col))) if numeric else (lambda row: str(row.get(col, "")))
        return sorted(rows, key=key, reverse=reverse)
    return f


def group_count(col: str) -> Callable[[Any], dict[str, int]]:
    def f(inp: Any) -> dict[str, int]:
        return dict(Counter(str(row.get(col, "")) for row in as_rows(inp)))
    return f


def group_sum(key_col: str, value_col: str) -> Callable[[Any], dict[str, int | float]]:
    def f(inp: Any) -> dict[str, int | float]:
        out: dict[str, float] = defaultdict(float)
        for row in as_rows(inp):
            out[str(row.get(key_col, ""))] += as_float(row.get(value_col))
        return {k: maybe_int(v) for k, v in sorted(out.items())}
    return f


def group_avg(key_col: str, value_col: str) -> Callable[[Any], dict[str, float]]:
    def f(inp: Any) -> dict[str, float]:
        sums: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        for row in as_rows(inp):
            key = str(row.get(key_col, ""))
            sums[key] += as_float(row.get(value_col))
            counts[key] += 1
        return {k: round(sums[k] / counts[k], 3) for k in sorted(sums)}
    return f


def normalize_column(col: str, mode: str) -> Callable[[Any], list[dict[str, Any]]]:
    def f(inp: Any) -> list[dict[str, Any]]:
        out = []
        for row in as_rows(inp):
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
            out.append(row)
        return out
    return f


def split_full_name(inp: Any) -> list[dict[str, Any]]:
    rows = []
    for row in as_rows(inp):
        row = dict(row)
        first, _, last = str(row.get("name", "")).strip().partition(" ")
        row["first_name"] = first
        row["last_name"] = last
        rows.append(row)
    return rows


def parse_money_column(col: str, out_col: str) -> Callable[[Any], list[dict[str, Any]]]:
    def f(inp: Any) -> list[dict[str, Any]]:
        rows = []
        for row in as_rows(inp):
            row = dict(row)
            row[out_col] = maybe_int(as_float(row.get(col)))
            rows.append(row)
        return rows
    return f


DATE_FORMATS = ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y%m%d", "%d-%b-%Y", "%b %d, %Y"]


def parse_date(text: str) -> datetime:
    clean = text.strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(clean, fmt)
        except ValueError:
            pass
    raise ValueError(f"unparsed date {text!r}")


def normalize_date_iso(inp: Any) -> str:
    return parse_date(str(inp)).strftime("%Y-%m-%d")


def normalize_date_us(inp: Any) -> str:
    return parse_date(str(inp)).strftime("%m/%d/%Y")


def extract_regex(pattern: str, group: int = 0, flags: int = 0) -> Callable[[Any], str]:
    def f(inp: Any) -> str:
        match = re.search(pattern, str(inp), flags)
        return match.group(group) if match else ""
    return f


def regex_bool(pattern: str, flags: int = 0) -> Callable[[Any], bool]:
    return lambda inp: bool(re.search(pattern, str(inp), flags))


def normalize_phone(inp: Any) -> str:
    digits = re.sub(r"\D", "", str(inp))
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    if len(digits) != 10:
        return ""
    return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"


def normalize_sku(inp: Any) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(inp)).upper()


def mask_account(inp: Any) -> str:
    digits = re.sub(r"\D", "", str(inp))
    return "****" + digits[-4:]


def slugify(inp: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "-", str(inp).strip().lower())
    return text.strip("-")


def title_case(inp: Any) -> str:
    return " ".join(part.capitalize() for part in str(inp).split())


def extract_state_zip(inp: Any) -> dict[str, str]:
    match = re.search(r"\b([A-Z]{2})\s+(\d{5})(?:-\d{4})?\b", str(inp))
    return {"state": match.group(1), "zip": match.group(2)} if match else {"state": "", "zip": ""}


def luhn_valid(inp: Any) -> bool:
    digits = [int(ch) for ch in re.sub(r"\D", "", str(inp))]
    total = 0
    parity = len(digits) % 2
    for idx, digit in enumerate(digits):
        if idx % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0 and bool(digits)


def generate_candidates(task: Task) -> list[Candidate]:
    rows: list[Candidate] = []

    def add(name: str, category: str, depth: int, func: Callable[[Any], Any], **program: Any) -> None:
        rows.append(Candidate(name, category, depth, func, {"op": name, **program}))

    sample_inputs = [ex.inp for ex in task.visible]
    observed_rows: list[dict[str, Any]] = []
    for inp in sample_inputs:
        try:
            observed_rows.extend(as_rows(inp))
        except Exception:
            pass
    columns = sorted({key for row in observed_rows for key in row})
    values_by_col: dict[str, set[str]] = defaultdict(set)
    for row in observed_rows:
        for col in columns:
            values_by_col[col].add(str(row.get(col, "")))

    add("identity", "generic", 0, lambda x: x)
    add("len", "generic", 1, lambda x: len(x))
    add("all_distinct_chars", "generic", 1, all_distinct_chars)

    for col in columns:
        for value in sorted(values_by_col[col]):
            if value:
                add(f"filter_eq_{col}_{value}", "csv_etl", 1, filter_eq(col, value), col=col, value=value)
        add(f"filter_nonempty_{col}", "csv_etl", 1, filter_nonempty(col), col=col)
        for threshold in [0, 1, 10, 50, 100, 1000]:
            for op in ["gt", "ge", "lt", "le"]:
                add(f"filter_{col}_{op}_{threshold}", "csv_etl", 1, filter_num_cmp(col, op, threshold), col=col, threshold=threshold)
        add(f"sort_{col}", "csv_etl", 1, sort_by(col), col=col)
        add(f"sort_{col}_desc", "csv_etl", 1, sort_by(col, reverse=True), col=col)
        add(f"sort_{col}_numeric", "csv_etl", 1, sort_by(col, numeric=True), col=col)
        add(f"group_count_{col}", "csv_etl", 1, group_count(col), col=col)
        for mode in ["strip", "lower", "strip_lower", "upper"]:
            add(f"normalize_{col}_{mode}", "csv_etl", 1, normalize_column(col, mode), col=col, mode=mode)
        for value_col in columns:
            if value_col != col:
                add(f"group_sum_{col}_{value_col}", "csv_etl", 2, group_sum(col, value_col), key_col=col, value_col=value_col)
                add(f"group_avg_{col}_{value_col}", "csv_etl", 2, group_avg(col, value_col), key_col=col, value_col=value_col)

    for width in [1, 2, 3]:
        for cols in [tuple(columns[i : i + width]) for i in range(max(0, len(columns) - width + 1))]:
            if cols:
                add(f"select_{'_'.join(cols)}", "csv_etl", 1, select_cols(cols), cols=cols)
    add("split_full_name", "csv_etl", 2, split_full_name)
    for col in columns:
        add(f"parse_money_{col}", "csv_etl", 2, parse_money_column(col, f"{col}_num"), col=col)

    regexes = {
        "order_id": r"(?:ORD|ORDER)[- ]?(\d{3,8})",
        "ticket": r"#([A-Za-z0-9-]+)",
        "email": r"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})",
        "zip5": r"\b(\d{5})(?:-\d{4})?\b",
        "letters": r"([A-Za-z]+)",
        "digits": r"(\d+)",
        "sku_like": r"([A-Za-z]{2,4}[-_ ]?\d{2,6})",
    }
    for name, pattern in regexes.items():
        add(f"extract_{name}", "date_id", 1, extract_regex(pattern, 1, re.I), pattern=pattern)
        add(f"bool_{name}", "date_id", 1, regex_bool(pattern, re.I), pattern=pattern)
    for name, func in [
        ("normalize_date_iso", normalize_date_iso),
        ("normalize_date_us", normalize_date_us),
        ("normalize_phone", normalize_phone),
        ("normalize_sku", normalize_sku),
        ("mask_account", mask_account),
        ("slugify", slugify),
        ("title_case", title_case),
        ("extract_state_zip", extract_state_zip),
        ("luhn_valid", luhn_valid),
    ]:
        add(name, "date_id", 1, func)
    return rows


def csv_tasks() -> list[Task]:
    def ex(rows: list[dict[str, Any]], out: Any, kind: str) -> Example:
        return Example(rows, out, kind)

    return [
        Task(
            "csv_01",
            "csv_etl",
            "Keep only rows whose status is active.",
            [ex([{"id": "1", "status": "active"}, {"id": "2", "status": "paused"}], [{"id": "1", "status": "active"}], "visible")],
            [ex([{"id": "3", "status": "active"}, {"id": "4", "status": "inactive"}], [{"id": "3", "status": "active"}], "hidden")],
            [ex([{"id": "5", "status": "active "}, {"id": "6", "status": "active"}], [{"id": "6", "status": "active"}], "adversarial")],
        ),
        Task(
            "csv_02",
            "csv_etl",
            "Sum revenue by region.",
            [ex([{"region": "E", "revenue": "10"}, {"region": "E", "revenue": "5"}, {"region": "W", "revenue": "7"}], {"E": 15, "W": 7}, "visible")],
            [ex([{"region": "N", "revenue": "2"}, {"region": "N", "revenue": "3"}], {"N": 5}, "hidden")],
            [ex([{"region": "E", "revenue": "$1,200"}, {"region": "E", "revenue": "50"}], {"E": 1250}, "adversarial")],
        ),
        Task(
            "csv_03",
            "csv_etl",
            "Normalize email cells by trimming whitespace and lowercasing.",
            [ex([{"email": " A@EXAMPLE.COM "}], [{"email": "a@example.com"}], "visible")],
            [ex([{"email": "Bob@Example.Org"}], [{"email": "bob@example.org"}], "hidden")],
            [ex([{"email": "  CAROL+X@Example.NET  "}], [{"email": "carol+x@example.net"}], "adversarial")],
        ),
        Task(
            "csv_04",
            "csv_etl",
            "Return only name and city columns.",
            [ex([{"name": "Ada", "city": "NY", "age": "30"}], [{"name": "Ada", "city": "NY"}], "visible")],
            [ex([{"name": "Ben", "city": "LA", "age": "41"}], [{"name": "Ben", "city": "LA"}], "hidden")],
            [ex([{"city": "SF", "name": "Cy", "age": ""}], [{"name": "Cy", "city": "SF"}], "adversarial")],
        ),
        Task(
            "csv_05",
            "csv_etl",
            "Count rows by category.",
            [ex([{"cat": "a"}, {"cat": "a"}, {"cat": "b"}], {"a": 2, "b": 1}, "visible")],
            [ex([{"cat": "x"}, {"cat": "y"}, {"cat": "x"}], {"x": 2, "y": 1}, "hidden")],
            [ex([{"cat": ""}, {"cat": ""}, {"cat": "x"}], {"": 2, "x": 1}, "adversarial")],
        ),
        Task(
            "csv_06",
            "csv_etl",
            "Sort rows by numeric score ascending.",
            [ex([{"name": "a", "score": "10"}, {"name": "b", "score": "2"}], [{"name": "b", "score": "2"}, {"name": "a", "score": "10"}], "visible")],
            [ex([{"name": "x", "score": "3"}, {"name": "y", "score": "11"}], [{"name": "x", "score": "3"}, {"name": "y", "score": "11"}], "hidden")],
            [ex([{"name": "n", "score": "-1"}, {"name": "p", "score": "0"}], [{"name": "n", "score": "-1"}, {"name": "p", "score": "0"}], "adversarial")],
        ),
        Task(
            "csv_07",
            "csv_etl",
            "Drop rows with an empty sku field.",
            [ex([{"sku": "A1"}, {"sku": ""}], [{"sku": "A1"}], "visible")],
            [ex([{"sku": "B2"}, {"sku": ""}, {"sku": "C3"}], [{"sku": "B2"}, {"sku": "C3"}], "hidden")],
            [ex([{"sku": "   "}, {"sku": "D4"}], [{"sku": "D4"}], "adversarial")],
        ),
        Task(
            "csv_08",
            "csv_etl",
            "Split full name into first_name and last_name columns.",
            [ex([{"name": "Ada Lovelace"}], [{"name": "Ada Lovelace", "first_name": "Ada", "last_name": "Lovelace"}], "visible")],
            [ex([{"name": "Grace Hopper"}], [{"name": "Grace Hopper", "first_name": "Grace", "last_name": "Hopper"}], "hidden")],
            [ex([{"name": "Prince"}], [{"name": "Prince", "first_name": "Prince", "last_name": ""}], "adversarial")],
        ),
        Task(
            "csv_09",
            "csv_etl",
            "Average rating by product.",
            [ex([{"product": "a", "rating": "4"}, {"product": "a", "rating": "2"}], {"a": 3.0}, "visible")],
            [ex([{"product": "b", "rating": "5"}, {"product": "b", "rating": "4"}], {"b": 4.5}, "hidden")],
            [ex([{"product": "x", "rating": "1"}, {"product": "y", "rating": "5"}], {"x": 1.0, "y": 5.0}, "adversarial")],
        ),
        Task(
            "csv_10",
            "csv_etl",
            "Parse amount strings into numeric amount_num column.",
            [ex([{"amount": "$10"}], [{"amount": "$10", "amount_num": 10}], "visible")],
            [ex([{"amount": "12.50"}], [{"amount": "12.50", "amount_num": 12.5}], "hidden")],
            [ex([{"amount": "$1,200"}], [{"amount": "$1,200", "amount_num": 1200}], "adversarial")],
        ),
        Task(
            "csv_11",
            "csv_etl",
            "Keep rows with score at least 50.",
            [ex([{"id": "a", "score": "49"}, {"id": "b", "score": "50"}], [{"id": "b", "score": "50"}], "visible")],
            [ex([{"id": "c", "score": "70"}, {"id": "d", "score": "10"}], [{"id": "c", "score": "70"}], "hidden")],
            [ex([{"id": "e", "score": "50.5"}, {"id": "f", "score": "-1"}], [{"id": "e", "score": "50.5"}], "adversarial")],
        ),
        Task(
            "csv_12",
            "csv_etl",
            "Sort rows by name descending.",
            [ex([{"name": "Ada"}, {"name": "Grace"}], [{"name": "Grace"}, {"name": "Ada"}], "visible")],
            [ex([{"name": "Linus"}, {"name": "Barbara"}], [{"name": "Linus"}, {"name": "Barbara"}], "hidden")],
            [ex([{"name": "zoe"}, {"name": "amy"}], [{"name": "zoe"}, {"name": "amy"}], "adversarial")],
        ),
        Task(
            "csv_13",
            "csv_etl",
            "Uppercase state abbreviations in the state column.",
            [ex([{"state": "ny"}], [{"state": "NY"}], "visible")],
            [ex([{"state": "ca"}, {"state": "Tx"}], [{"state": "CA"}, {"state": "TX"}], "hidden")],
            [ex([{"state": " dc "}], [{"state": " DC "}], "adversarial")],
        ),
        Task(
            "csv_14",
            "csv_etl",
            "Return only product and price columns.",
            [ex([{"product": "pen", "price": "2", "qty": "4"}], [{"product": "pen", "price": "2"}], "visible")],
            [ex([{"product": "pad", "price": "5", "qty": "1"}], [{"product": "pad", "price": "5"}], "hidden")],
            [ex([{"qty": "9", "price": "1", "product": "clip"}], [{"product": "clip", "price": "1"}], "adversarial")],
        ),
        Task(
            "csv_15",
            "csv_etl",
            "Sum quantity by sku.",
            [ex([{"sku": "A", "qty": "2"}, {"sku": "A", "qty": "3"}], {"A": 5}, "visible")],
            [ex([{"sku": "B", "qty": "4"}, {"sku": "C", "qty": "1"}], {"B": 4, "C": 1}, "hidden")],
            [ex([{"sku": "A", "qty": "1,000"}, {"sku": "A", "qty": "2"}], {"A": 1002}, "adversarial")],
        ),
        Task(
            "csv_16",
            "csv_etl",
            "Keep rows where country is US.",
            [ex([{"country": "US", "id": "1"}, {"country": "CA", "id": "2"}], [{"country": "US", "id": "1"}], "visible")],
            [ex([{"country": "US", "id": "3"}, {"country": "MX", "id": "4"}], [{"country": "US", "id": "3"}], "hidden")],
            [ex([{"country": "USA", "id": "5"}, {"country": "US", "id": "6"}], [{"country": "US", "id": "6"}], "adversarial")],
        ),
        Task(
            "csv_17",
            "csv_etl",
            "Sort rows by ISO date ascending.",
            [ex([{"date": "2024-02-01"}, {"date": "2024-01-01"}], [{"date": "2024-01-01"}, {"date": "2024-02-01"}], "visible")],
            [ex([{"date": "2025-01-01"}, {"date": "2023-12-31"}], [{"date": "2023-12-31"}, {"date": "2025-01-01"}], "hidden")],
            [ex([{"date": "2024-10-01"}, {"date": "2024-02-01"}], [{"date": "2024-02-01"}, {"date": "2024-10-01"}], "adversarial")],
        ),
        Task(
            "csv_18",
            "csv_etl",
            "Lowercase status values.",
            [ex([{"status": "OPEN"}], [{"status": "open"}], "visible")],
            [ex([{"status": "Closed"}], [{"status": "closed"}], "hidden")],
            [ex([{"status": "  Paused  "}], [{"status": "  paused  "}], "adversarial")],
        ),
        Task(
            "csv_19",
            "csv_etl",
            "Keep rows with amount greater than 100.",
            [ex([{"amount": "99"}, {"amount": "101"}], [{"amount": "101"}], "visible")],
            [ex([{"amount": "$250"}, {"amount": "100"}], [{"amount": "$250"}], "hidden")],
            [ex([{"amount": "100.01"}, {"amount": "-5"}], [{"amount": "100.01"}], "adversarial")],
        ),
        Task(
            "csv_20",
            "csv_etl",
            "Count rows by normalized status exactly as written.",
            [ex([{"status": "ok"}, {"status": "fail"}, {"status": "ok"}], {"ok": 2, "fail": 1}, "visible")],
            [ex([{"status": "new"}, {"status": "new"}], {"new": 2}, "hidden")],
            [ex([{"status": "OK"}, {"status": "ok"}], {"OK": 1, "ok": 1}, "adversarial")],
        ),
    ]


def date_id_tasks() -> list[Task]:
    def ex(inp: Any, out: Any, kind: str) -> Example:
        return Example(inp, out, kind)

    return [
        Task("id_01", "date_id_irregular", "Normalize dates to ISO YYYY-MM-DD.", [ex("01/02/2024", "2024-01-02", "visible")], [ex("2024-03-04", "2024-03-04", "hidden")], [ex("03/04/2024", "2024-03-04", "adversarial")]),
        Task("id_02", "date_id_irregular", "Extract order id digits from order text.", [ex("Order ORD-12345 shipped", "12345", "visible")], [ex("ORDER 987 ready", "987", "hidden")], [ex("preorder 42 ORD-777", "777", "adversarial")]),
        Task("id_03", "date_id_irregular", "Normalize US phone numbers to 555-123-4567.", [ex("(555) 123-4567", "555-123-4567", "visible")], [ex("1-212-555-7890", "212-555-7890", "hidden")], [ex("+1 303.555.0100 ext 9", "303-555-0100", "adversarial")]),
        Task("id_04", "date_id_irregular", "Normalize SKU by removing separators and uppercasing.", [ex("ab-123", "AB123", "visible")], [ex("Xy_999", "XY999", "hidden")], [ex(" q r 42 ", "QR42", "adversarial")]),
        Task("id_05", "date_id_irregular", "Mask account number keeping last four digits.", [ex("acct 123456789", "****6789", "visible")], [ex("9876-5432-1000", "****1000", "hidden")], [ex("id: 00001234", "****1234", "adversarial")]),
        Task("id_06", "date_id_irregular", "Slugify title text.", [ex("Hello, World!", "hello-world", "visible")], [ex(" A/B test ", "a-b-test", "hidden")], [ex("Already---Slug", "already-slug", "adversarial")]),
        Task("id_07", "date_id_irregular", "Extract state and five-digit ZIP from address.", [ex("Austin TX 78701", {"state": "TX", "zip": "78701"}, "visible")], [ex("Boston MA 02108-1234", {"state": "MA", "zip": "02108"}, "hidden")], [ex("No zip here", {"state": "", "zip": ""}, "adversarial")]),
        Task("id_08", "date_id_irregular", "Validate Luhn-style account id.", [ex("79927398713", True, "visible")], [ex("79927398710", False, "hidden")], [ex("7992 7398 713", True, "adversarial")]),
        Task("id_09", "date_id_irregular", "Title-case person names.", [ex("ada lovelace", "Ada Lovelace", "visible")], [ex("GRACE HOPPER", "Grace Hopper", "hidden")], [ex("  alan   turing ", "Alan Turing", "adversarial")]),
        Task("id_10", "date_id_irregular", "Extract email address from message.", [ex("contact a@example.com now", "a@example.com", "visible")], [ex("Email: bob.smith+q@foo.co", "bob.smith+q@foo.co", "hidden")], [ex("none", "", "adversarial")]),
        Task("id_11", "date_id_irregular", "Normalize compact dates to ISO YYYY-MM-DD.", [ex("20240131", "2024-01-31", "visible")], [ex("31-Jan-2024", "2024-01-31", "hidden")], [ex("September 5, 2024", "2024-09-05", "adversarial")]),
        Task("id_12", "date_id_irregular", "Return whether text contains an email address.", [ex("mail a@example.com", True, "visible")], [ex("no address here", False, "hidden")], [ex("first.last+tag@sub.example.org", True, "adversarial")]),
        Task("id_13", "date_id_irregular", "Extract the first five-digit ZIP code.", [ex("Ship to 94105 today", "94105", "visible")], [ex("zip: 02108-1234", "02108", "hidden")], [ex("no postal code", "", "adversarial")]),
        Task("id_14", "date_id_irregular", "Extract ticket id after a # marker.", [ex("ticket #ABC-123 opened", "ABC-123", "visible")], [ex("see #z9", "z9", "hidden")], [ex("no hash", "", "adversarial")]),
        Task("id_15", "date_id_irregular", "Normalize dotted US phone numbers.", [ex("415.555.0101", "415-555-0101", "visible")], [ex("2125550199", "212-555-0199", "hidden")], [ex("x2125550199", "212-555-0199", "adversarial")]),
        Task("id_16", "date_id_irregular", "Mask any identifier by keeping the last four digits.",
             [ex("card 4444333322221111", "****1111", "visible")], [ex("0000", "****0000", "hidden")], [ex("abc-12", "****12", "adversarial")]),
        Task("id_17", "date_id_irregular", "Normalize part codes by stripping punctuation and uppercasing.", [ex("aa/bb-12", "AABB12", "visible")], [ex("x y z 9", "XYZ9", "hidden")], [ex("part: q-7!", "PARTQ7", "adversarial")]),
        Task("id_18", "date_id_irregular", "Title-case hyphenated names while preserving the hyphen.", [ex("jean-luc picard", "Jean-Luc Picard", "visible")], [ex("mary-jane watson", "Mary-Jane Watson", "hidden")], [ex("  ana-maria  lopez ", "Ana-Maria Lopez", "adversarial")]),
        Task("id_19", "date_id_irregular", "Normalize dates to US MM/DD/YYYY.", [ex("2024-12-25", "12/25/2024", "visible")], [ex("25-Dec-2024", "12/25/2024", "hidden")], [ex("Dec 5, 2024", "12/05/2024", "adversarial")]),
        Task("id_20", "date_id_irregular", "Extract the first run of digits.", [ex("abc123def", "123", "visible")], [ex("no 42 here", "42", "hidden")], [ex("none", "", "adversarial")]),
    ]


def smoke_tasks() -> list[Task]:
    return [
        Task(
            "smoke_parentheses",
            "smoke",
            "Verify balanced parentheses, not merely distinct characters.",
            [Example("()", True, "visible")],
            [Example("[]", True, "hidden")],
            [Example("(()", False, "adversarial"), Example(")(", False, "adversarial")],
        ),
        Task(
            "smoke_month_30",
            "smoke",
            "Check whether month has 30 days.",
            [Example("April", True, "visible")],
            [Example("June", True, "hidden")],
            [Example("September", True, "adversarial"), Example("February", False, "adversarial")],
        ),
    ]


def candidate_results(candidate: Candidate, examples: list[Example]) -> list[dict[str, Any]]:
    rows = []
    for ex in examples:
        ok, value = safe_call(candidate.func, ex.inp)
        rows.append({"ok": ok, "value_repr": repr(value), "passed": ok and equal(value, ex.out)})
    return rows


def run_task(task: Task) -> dict[str, Any]:
    candidates = generate_candidates(task)
    raw_examples = task.visible + task.hidden
    all_examples = raw_examples + task.adversarial
    raw_winners: list[Candidate] = []
    filtered_winners: list[Candidate] = []
    visible_winners: list[Candidate] = []
    visible_hidden_wrong = 0
    candidate_rows = []
    for cand in candidates:
        raw_results = candidate_results(cand, raw_examples)
        all_results = candidate_results(cand, all_examples)
        visible_pass = all(row["passed"] for row in raw_results[: len(task.visible)])
        raw_pass = all(row["passed"] for row in raw_results)
        filtered_pass = all(row["passed"] for row in all_results)
        if visible_pass:
            visible_winners.append(cand)
            if not filtered_pass:
                visible_hidden_wrong += 1
        if raw_pass:
            raw_winners.append(cand)
        if filtered_pass:
            filtered_winners.append(cand)
        if visible_pass or raw_pass or filtered_pass:
            candidate_rows.append(
                {
                    "name": cand.name,
                    "category": cand.category,
                    "depth": cand.depth,
                    "program": cand.program,
                    "visible_pass": visible_pass,
                    "raw_pass": raw_pass,
                    "filtered_pass": filtered_pass,
                    "raw_outputs": [row["value_repr"] for row in raw_results],
                    "all_outputs": [row["value_repr"] for row in all_results],
                }
            )
    return {
        "task_id": task.task_id,
        "domain": task.domain,
        "description": task.description,
        "candidate_count": len(candidates),
        "visible_consistent_count": len(visible_winners),
        "visible_hidden_wrong_count": visible_hidden_wrong,
        "raw_winner_count": len(raw_winners),
        "filtered_winner_count": len(filtered_winners),
        "raw_covered": bool(raw_winners),
        "filtered_covered": bool(filtered_winners),
        "raw_but_filtered_out": bool(raw_winners) and not bool(filtered_winners),
        "winning_raw_program": raw_winners[0].program if raw_winners else None,
        "winning_raw_depth": raw_winners[0].depth if raw_winners else None,
        "winning_filtered_program": filtered_winners[0].program if filtered_winners else None,
        "winning_filtered_depth": filtered_winners[0].depth if filtered_winners else None,
        "visible_or_winning_candidates": candidate_rows[:20],
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def metrics(group: list[dict[str, Any]]) -> dict[str, Any]:
        if not group:
            return {"n": 0}
        raw = sum(row["raw_covered"] for row in group)
        filt = sum(row["filtered_covered"] for row in group)
        visible_any = sum(row["visible_consistent_count"] > 0 for row in group)
        return {
            "n": len(group),
            "raw_covered": raw,
            "raw_coverage": raw / len(group),
            "filtered_covered": filt,
            "filtered_coverage": filt / len(group),
            "raw_removed_by_counterexamples": sum(row["raw_but_filtered_out"] for row in group),
            "visible_any": visible_any,
            "candidate_count_mean": sum(row["candidate_count"] for row in group) / len(group),
            "visible_consistent_mean": sum(row["visible_consistent_count"] for row in group) / len(group),
            "visible_hidden_wrong_candidates": sum(row["visible_hidden_wrong_count"] for row in group),
            "visible_hidden_wrong_rate": sum(row["visible_hidden_wrong_count"] for row in group) / max(1, sum(row["visible_consistent_count"] for row in group)),
        }

    domains = sorted(set(row["domain"] for row in rows))
    return {
        "overall": metrics(rows),
        "by_domain": {domain: metrics([row for row in rows if row["domain"] == domain]) for domain in domains},
        "raw_depth_counts": {str(k): v for k, v in sorted(Counter(row["winning_raw_depth"] for row in rows if row["raw_covered"]).items())},
        "filtered_depth_counts": {str(k): v for k, v in sorted(Counter(row["winning_filtered_depth"] for row in rows if row["filtered_covered"]).items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()

    smoke = [run_task(task) for task in smoke_tasks()]
    domain_tasks = csv_tasks() + date_id_tasks()
    rows = [run_task(task) for task in domain_tasks]
    summary = summarize(rows)
    smoke_summary = summarize(smoke)
    summary["smoke"] = smoke_summary
    summary["smoke_records"] = smoke

    write_jsonl(args.data_dir / "task_records.jsonl", rows)
    write_jsonl(args.data_dir / "smoke_records.jsonl", smoke)
    write_json(args.reports_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
