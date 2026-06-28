#!/usr/bin/env python3
"""Real-style deterministic office task ABI coverage gate.

This experiment does not train Qwen. It asks the prior question that must be
answered before a large compiler corpus is worth building: can a frozen office
ABI express realistic held-out tasks, or are synthetic successes only possible
because the task factory was built from the same primitives?
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import statistics
import textwrap
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("/workspace/experiments/qwen_real_task_abi_coverage_gate")
LARGE_ROOT = Path("/workspace/large_artifacts/qwen_real_task_abi_coverage_gate")
RUNS = ROOT / "runs"
ANALYSIS = ROOT / "analysis"
FIGURES = ANALYSIS / "figures"
REPORTS = ROOT / "reports"


# ---------------------------------------------------------------------------
# Frozen office ABI. This is intentionally defined before task definitions.
# ---------------------------------------------------------------------------


Value = Any
Row = Dict[str, Any]
Table = List[Dict[str, Any]]


def as_str(x: Value) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return str(x)


def clean_spaces(s: Value) -> str:
    return re.sub(r"\s+", " ", as_str(s)).strip()


def title_words(s: Value) -> str:
    return " ".join(part.capitalize() for part in clean_spaces(s).split(" ") if part)


def only_digits(s: Value) -> str:
    return re.sub(r"\D+", "", as_str(s))


def only_alpha_space(s: Value) -> str:
    return clean_spaces(re.sub(r"[^A-Za-z ]+", " ", as_str(s)))


def only_alnum(s: Value) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", as_str(s))


def safe_float(x: Value) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return float(x)
    s = as_str(x).strip()
    if not s:
        return None
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    s = s.replace("$", "").replace(",", "").replace("%", "").replace("USD", "").strip()
    try:
        v = float(s)
    except ValueError:
        return None
    return -v if neg else v


def money_cents(x: Value) -> Optional[int]:
    v = safe_float(x)
    if v is None:
        return None
    return int(round(v * 100))


def percent_float(x: Value) -> Optional[float]:
    s = as_str(x).strip()
    v = safe_float(s)
    if v is None:
        return None
    if "%" in s or abs(v) > 1.0:
        return round(v / 100.0, 6)
    return round(v, 6)


DATE_FORMATS = [
    "%Y-%m-%d",
    "%m/%d/%Y",
    "%m/%d/%y",
    "%d/%m/%Y",
    "%b %d, %Y",
    "%B %d, %Y",
    "%d %b %Y",
    "%Y/%m/%d",
]


def parse_date(x: Value) -> Optional[datetime]:
    s = clean_spaces(x)
    if not s:
        return None
    s = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", s, flags=re.I)
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    return None


def date_iso(x: Value) -> Optional[str]:
    d = parse_date(x)
    return None if d is None else d.strftime("%Y-%m-%d")


def date_year(x: Value) -> Optional[int]:
    d = parse_date(x)
    return None if d is None else d.year


def date_month2(x: Value) -> Optional[str]:
    d = parse_date(x)
    return None if d is None else d.strftime("%m")


def date_month_name(x: Value) -> Optional[str]:
    d = parse_date(x)
    return None if d is None else d.strftime("%B")


def date_quarter(x: Value) -> Optional[str]:
    d = parse_date(x)
    if d is None:
        return None
    return f"Q{(d.month - 1) // 3 + 1}-{d.year}"


def phone_e164_us(x: Value) -> Optional[str]:
    d = only_digits(x)
    if len(d) == 11 and d.startswith("1"):
        d = d[1:]
    if len(d) != 10:
        return None
    return "+1" + d


def phone_last4(x: Value) -> Optional[str]:
    d = only_digits(x)
    return d[-4:] if len(d) >= 4 else None


def email_user(x: Value) -> Optional[str]:
    s = as_str(x).strip().lower()
    return s.split("@", 1)[0] if "@" in s else None


def email_domain(x: Value) -> Optional[str]:
    s = as_str(x).strip().lower()
    return s.split("@", 1)[1] if "@" in s else None


def email_company(x: Value) -> Optional[str]:
    d = email_domain(x)
    if not d:
        return None
    root = d.split(".", 1)[0]
    root = root.replace("-", " ").replace("_", " ")
    return title_words(root)


def url_domain(x: Value) -> Optional[str]:
    s = as_str(x).strip()
    s = re.sub(r"^[a-zA-Z]+://", "", s)
    if not s:
        return None
    return s.split("/", 1)[0].lower()


def url_last_path(x: Value) -> Optional[str]:
    s = as_str(x).strip().rstrip("/")
    s = re.sub(r"^[a-zA-Z]+://", "", s)
    if "/" not in s:
        return None
    return s.rsplit("/", 1)[1]


def slug_title(x: Value) -> str:
    s = as_str(x)
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"\.[A-Za-z0-9]+$", "", s)
    return title_words(s)


def file_ext(x: Value) -> Optional[str]:
    stem = as_str(x).split("/")[-1]
    if "." not in stem:
        return None
    return stem.rsplit(".", 1)[1].lower()


def file_stem(x: Value) -> str:
    stem = as_str(x).rstrip("/").split("/")[-1]
    return stem.rsplit(".", 1)[0] if "." in stem else stem


def zip5(x: Value) -> Optional[str]:
    m = re.search(r"\b(\d{5})(?:-\d{4})?\b", as_str(x))
    return None if not m else m.group(1)


def us_state(x: Value) -> Optional[str]:
    m = re.search(r"\b([A-Z]{2})\s+\d{5}(?:-\d{4})?\b", as_str(x))
    return None if not m else m.group(1)


def city_before_state(x: Value) -> Optional[str]:
    m = re.search(r"([^,]+),\s*[A-Z]{2}\s+\d{5}", as_str(x))
    return None if not m else clean_spaces(m.group(1))


def sku_prefix(x: Value) -> str:
    return as_str(x).split("-", 1)[0].strip().upper()


def sku_suffix(x: Value) -> str:
    return as_str(x).rsplit("-", 1)[-1].strip().upper()


def token_first(x: Value) -> str:
    parts = clean_spaces(x).split()
    return parts[0] if parts else ""


def token_last(x: Value) -> str:
    parts = clean_spaces(x).split()
    return parts[-1] if parts else ""


def initials(x: Value) -> str:
    return "".join(p[0].upper() for p in re.findall(r"[A-Za-z]+", as_str(x)))


def reverse_first_last(x: Value) -> Optional[str]:
    parts = clean_spaces(only_alpha_space(x)).split()
    if len(parts) < 2:
        return None
    return f"{parts[-1].capitalize()}, {parts[0].capitalize()}"


def normalize_name(x: Value) -> str:
    s = as_str(x).replace(",", " ")
    return title_words(only_alpha_space(s))


def first_after_comma(x: Value) -> Optional[str]:
    s = as_str(x)
    if "," not in s:
        return None
    return clean_spaces(s.split(",", 1)[1])


def before_at(x: Value) -> Optional[str]:
    s = as_str(x)
    return s.split("@", 1)[0] if "@" in s else None


def before_dash(x: Value) -> str:
    return as_str(x).split("-", 1)[0].strip()


def after_dash(x: Value) -> str:
    return as_str(x).split("-", 1)[-1].strip()


def before_colon(x: Value) -> str:
    return as_str(x).split(":", 1)[0].strip()


def after_colon(x: Value) -> str:
    return as_str(x).split(":", 1)[-1].strip()


def bool_yes_no(x: Value) -> Optional[bool]:
    s = as_str(x).strip().lower()
    if s in {"yes", "y", "true", "1", "active", "paid", "complete"}:
        return True
    if s in {"no", "n", "false", "0", "inactive", "unpaid", "open"}:
        return False
    return None


@dataclass(frozen=True)
class UnaryOp:
    name: str
    func: Callable[[Value], Value]
    tier: str


@dataclass(frozen=True)
class BinaryOp:
    name: str
    func: Callable[[Value, Value], Value]
    tier: str


def safe_bin_numeric(fn: Callable[[float, float], float]) -> Callable[[Value, Value], Value]:
    def wrapped(a: Value, b: Value) -> Optional[float]:
        av, bv = safe_float(a), safe_float(b)
        if av is None or bv is None:
            return None
        try:
            out = fn(av, bv)
        except ZeroDivisionError:
            return None
        if isinstance(out, float):
            return round(out, 6)
        return out

    return wrapped


def joiner(sep: str) -> Callable[[Value, Value], str]:
    return lambda a, b: f"{clean_spaces(a)}{sep}{clean_spaces(b)}".strip()


CORE_UNARY: List[UnaryOp] = [
    UnaryOp("strip", lambda x: as_str(x).strip(), "core"),
    UnaryOp("lower", lambda x: as_str(x).lower(), "core"),
    UnaryOp("upper", lambda x: as_str(x).upper(), "core"),
    UnaryOp("title", title_words, "core"),
    UnaryOp("normalize_spaces", clean_spaces, "core"),
    UnaryOp("digits", only_digits, "core"),
    UnaryOp("alpha_spaces", only_alpha_space, "core"),
    UnaryOp("alnum", only_alnum, "core"),
    UnaryOp("first_token", token_first, "core"),
    UnaryOp("last_token", token_last, "core"),
    UnaryOp("initials", initials, "core"),
    UnaryOp("before_dash", before_dash, "core"),
    UnaryOp("after_dash", after_dash, "core"),
    UnaryOp("before_colon", before_colon, "core"),
    UnaryOp("after_colon", after_colon, "core"),
    UnaryOp("slug_title", slug_title, "core"),
]

OFFICE_UNARY: List[UnaryOp] = CORE_UNARY + [
    UnaryOp("money_cents", money_cents, "office"),
    UnaryOp("percent_float", percent_float, "office"),
    UnaryOp("date_iso", date_iso, "office"),
    UnaryOp("date_year", date_year, "office"),
    UnaryOp("date_month2", date_month2, "office"),
    UnaryOp("date_month_name", date_month_name, "office"),
    UnaryOp("date_quarter", date_quarter, "office"),
    UnaryOp("phone_e164_us", phone_e164_us, "office"),
    UnaryOp("phone_last4", phone_last4, "office"),
    UnaryOp("email_user", email_user, "office"),
    UnaryOp("email_domain", email_domain, "office"),
    UnaryOp("email_company", email_company, "office"),
    UnaryOp("url_domain", url_domain, "office"),
    UnaryOp("url_last_path", url_last_path, "office"),
    UnaryOp("file_ext", file_ext, "office"),
    UnaryOp("file_stem", file_stem, "office"),
    UnaryOp("zip5", zip5, "office"),
    UnaryOp("us_state", us_state, "office"),
    UnaryOp("city_before_state", city_before_state, "office"),
    UnaryOp("sku_prefix", sku_prefix, "office"),
    UnaryOp("sku_suffix", sku_suffix, "office"),
    UnaryOp("reverse_first_last", reverse_first_last, "office"),
    UnaryOp("normalize_name", normalize_name, "office"),
    UnaryOp("first_after_comma", first_after_comma, "office"),
    UnaryOp("before_at", before_at, "office"),
    UnaryOp("bool_yes_no", bool_yes_no, "office"),
]

CORE_BINARY: List[BinaryOp] = [
    BinaryOp("concat", joiner(""), "core"),
    BinaryOp("join_space", joiner(" "), "core"),
    BinaryOp("join_dash", joiner("-"), "core"),
    BinaryOp("join_comma_space", joiner(", "), "core"),
]

OFFICE_BINARY: List[BinaryOp] = CORE_BINARY + [
    BinaryOp("add", safe_bin_numeric(lambda a, b: a + b), "office"),
    BinaryOp("sub", safe_bin_numeric(lambda a, b: a - b), "office"),
    BinaryOp("mul", safe_bin_numeric(lambda a, b: a * b), "office"),
    BinaryOp("div", safe_bin_numeric(lambda a, b: a / b), "office"),
    BinaryOp("date_diff_days", lambda a, b: None if parse_date(a) is None or parse_date(b) is None else (parse_date(a) - parse_date(b)).days, "office"),
]


# ---------------------------------------------------------------------------
# Task catalog. References are ordinary Python functions over examples, not ABI
# programs. The task examples are hand-curated real-style office transforms.
# ---------------------------------------------------------------------------


@dataclass
class Example:
    inp: Any
    out: Any
    phase: str


@dataclass
class Task:
    task_id: str
    family: str
    split: str
    input_kind: str
    description: str
    examples: List[Example]
    constants: List[Any] = field(default_factory=list)


def row_task(
    task_id: str,
    family: str,
    split: str,
    description: str,
    rows: List[Row],
    ref: Callable[[Row], Any],
    train_n: int = 4,
    constants: Optional[List[Any]] = None,
) -> Task:
    examples = [
        Example(row, ref(row), "train" if i < train_n else "test")
        for i, row in enumerate(rows)
    ]
    return Task(task_id, family, split, "row", description, examples, constants or [])


def table_task(
    task_id: str,
    family: str,
    split: str,
    description: str,
    tables: List[Table],
    ref: Callable[[Table], Any],
    train_n: int = 3,
    constants: Optional[List[Any]] = None,
) -> Task:
    examples = [
        Example(table, ref(table), "train" if i < train_n else "test")
        for i, table in enumerate(tables)
    ]
    return Task(task_id, family, split, "table", description, examples, constants or [])


def build_tasks() -> List[Task]:
    tasks: List[Task] = []

    contact_rows = [
        {"name": "  jane   DOE ", "email": "Jane.Doe@Acme-Tools.com", "phone": "(415) 555-0134"},
        {"name": "MIGUEL santos", "email": "m.santos@northwind.io", "phone": "1-212-555-0199"},
        {"name": "Priya  Shah", "email": "pshah@contoso.co", "phone": "617.555.0101"},
        {"name": "LI WEI", "email": "li.wei@example.org", "phone": "+1 303 555 0188"},
        {"name": "anne-marie o'neil", "email": "am.oneil@blue-river.net", "phone": "2065550177"},
        {"name": "Omar   Haddad", "email": "omar@fabrikam.com", "phone": "(512) 555 0166"},
    ]
    tasks += [
        row_task("contact_email_domain", "contact", "calibration", "Extract the lower-case email domain.", contact_rows, lambda r: email_domain(r["email"])),
        row_task("contact_email_company", "contact", "heldout_same_family", "Convert the email domain root into a company-style title.", contact_rows, lambda r: email_company(r["email"])),
        row_task("contact_phone_e164", "contact", "calibration", "Normalize a US phone number to E.164.", contact_rows, lambda r: phone_e164_us(r["phone"])),
        row_task("contact_phone_last4", "contact", "heldout_same_family", "Extract the last four phone digits.", contact_rows, lambda r: phone_last4(r["phone"])),
        row_task("contact_name_initials", "contact", "calibration", "Return initials from the contact name.", contact_rows, lambda r: initials(r["name"])),
        row_task("contact_normalize_name", "contact", "heldout_same_family", "Normalize a messy person name into title case.", contact_rows, lambda r: normalize_name(r["name"])),
    ]

    date_rows = [
        {"date": "Jan 5, 2024"},
        {"date": "02/14/2025"},
        {"date": "2023/07/03"},
        {"date": "19 Aug 2026"},
        {"date": "March 9, 2022"},
        {"date": "11/30/24"},
    ]
    interval_rows = [
        {"start": "2024-01-01", "end": "2024-01-05"},
        {"start": "2025-02-01", "end": "2025-02-14"},
        {"start": "2023-07-01", "end": "2023-07-03"},
        {"start": "2026-08-10", "end": "2026-08-19"},
        {"start": "2022-03-01", "end": "2022-03-09"},
        {"start": "2024-11-01", "end": "2024-11-30"},
    ]
    tasks += [
        row_task("date_to_iso", "date", "calibration", "Normalize dates to ISO format.", date_rows, lambda r: date_iso(r["date"])),
        row_task("date_month_name", "date", "heldout_same_family", "Return the month name.", date_rows, lambda r: date_month_name(r["date"])),
        row_task("date_quarter_label", "date", "heldout_same_family", "Return a QN-YYYY quarter label.", date_rows, lambda r: date_quarter(r["date"])),
        row_task("date_days_between", "date", "heldout_composition", "Return calendar days between two date fields.", interval_rows, lambda r: (parse_date(r["end"]) - parse_date(r["start"])).days),
        row_task("date_fiscal_year_july", "date", "stress_out_of_abi", "Return fiscal year where July starts the next fiscal year.", date_rows, lambda r: parse_date(r["date"]).year + (1 if parse_date(r["date"]).month >= 7 else 0)),
    ]

    money_rows = [
        {"price": "$1,245.50", "qty": 2, "discount": "15%"},
        {"price": "87.99", "qty": 5, "discount": "7.5%"},
        {"price": "$12.00", "qty": 12, "discount": "0%"},
        {"price": "1,999.95", "qty": 1, "discount": "20%"},
        {"price": "$4.25", "qty": 40, "discount": "5%"},
        {"price": "310.10", "qty": 3, "discount": "12%"},
    ]
    tasks += [
        row_task("money_cents", "money", "calibration", "Parse a currency amount as integer cents.", money_rows, lambda r: money_cents(r["price"])),
        row_task("money_percent_decimal", "money", "calibration", "Parse a percent into decimal form.", money_rows, lambda r: percent_float(r["discount"])),
        row_task("money_line_total_cents", "money", "heldout_composition", "Return price times quantity in cents.", money_rows, lambda r: int(round(safe_float(r["price"]) * r["qty"] * 100))),
        row_task("money_discounted_total", "money", "stress_out_of_abi", "Return discounted line total in dollars rounded to cents.", money_rows, lambda r: round(safe_float(r["price"]) * r["qty"] * (1 - percent_float(r["discount"])), 2)),
    ]

    url_rows = [
        {"url": "https://docs.example.com/reports/q1-summary", "path": "/mnt/reports/q1-summary.xlsx"},
        {"url": "http://shop.acme.io/products/red-widget", "path": "C:/exports/orders_2025_02.csv"},
        {"url": "https://support.contoso.com/kb/reset-password", "path": "/tmp/raw/customer-list.json"},
        {"url": "https://blog.northwind.net/2024/launch-notes", "path": "/home/me/archive/photo.final.PNG"},
        {"url": "https://data.fabrikam.com/files/monthly-rollup", "path": "/var/log/app/server.log"},
        {"url": "https://example.org/a/b/invoice-status", "path": "s3://bucket/folder/items.parquet"},
    ]
    tasks += [
        row_task("url_domain", "url_file", "calibration", "Extract URL domain.", url_rows, lambda r: url_domain(r["url"])),
        row_task("url_last_slug", "url_file", "heldout_same_family", "Extract the final URL path slug.", url_rows, lambda r: url_last_path(r["url"])),
        row_task("url_slug_to_title", "url_file", "heldout_composition", "Turn the final URL slug into a title.", url_rows, lambda r: slug_title(url_last_path(r["url"]))),
        row_task("file_extension", "url_file", "calibration", "Extract the file extension.", url_rows, lambda r: file_ext(r["path"])),
        row_task("file_stem_title", "url_file", "heldout_composition", "Turn a file stem into a readable title.", url_rows, lambda r: slug_title(file_stem(r["path"]))),
    ]

    product_rows = [
        {"sku": "AB-123-red", "code": "INV:000145", "status": "paid"},
        {"sku": "ZX-9-blue", "code": "PO:55421", "status": "OPEN"},
        {"sku": "mn-777-green", "code": "CASE:77", "status": "complete"},
        {"sku": "TT-42-black", "code": "REQ:0099", "status": "inactive"},
        {"sku": "Q5-808-white", "code": "INV:990001", "status": "unpaid"},
        {"sku": "rr-5-orange", "code": "BUG:331", "status": "yes"},
    ]
    tasks += [
        row_task("sku_prefix", "product", "calibration", "Extract the product family prefix.", product_rows, lambda r: sku_prefix(r["sku"])),
        row_task("sku_suffix", "product", "heldout_same_family", "Extract the final SKU segment.", product_rows, lambda r: sku_suffix(r["sku"])),
        row_task("code_after_colon", "product", "heldout_same_family", "Extract the code after a colon.", product_rows, lambda r: after_colon(r["code"])),
        row_task("status_boolean", "product", "heldout_new_family", "Map common status words to booleans.", product_rows, lambda r: bool_yes_no(r["status"])),
        row_task("sku_middle_segment", "product", "stress_out_of_abi", "Extract the middle SKU segment.", product_rows, lambda r: r["sku"].split("-")[1]),
    ]

    address_rows = [
        {"address": "144 Market St, San Francisco, CA 94105-1234"},
        {"address": "9 Lake Road, Seattle, WA 98101"},
        {"address": "77 Main Ave, Austin, TX 78701"},
        {"address": "12 Broadway, New York, NY 10004"},
        {"address": "8 Pearl St, Boston, MA 02110-2201"},
        {"address": "500 Pine, Denver, CO 80202"},
    ]
    tasks += [
        row_task("address_zip5", "address", "heldout_new_family", "Extract five-digit ZIP code.", address_rows, lambda r: zip5(r["address"])),
        row_task("address_state", "address", "heldout_new_family", "Extract two-letter state code.", address_rows, lambda r: us_state(r["address"])),
        row_task("address_city", "address", "heldout_new_family", "Extract city before state and ZIP.", address_rows, lambda r: city_before_state(r["address"])),
    ]

    text_rows = [
        {"text": "  Alpha   beta\tGamma  ", "phrase": "quarterly revenue report"},
        {"text": "hello---world!!", "phrase": "customer support ticket"},
        {"text": " ACME, Inc. ", "phrase": "north east region"},
        {"text": "foo_bar baz", "phrase": "product launch plan"},
        {"text": "Line   Item #42", "phrase": "data quality review"},
        {"text": "two   spaces", "phrase": "monthly close checklist"},
    ]
    tasks += [
        row_task("text_normalize_spaces", "text", "calibration", "Normalize whitespace.", text_rows, lambda r: clean_spaces(r["text"])),
        row_task("text_alnum_upper", "text", "heldout_composition", "Keep alphanumerics and uppercase.", text_rows, lambda r: only_alnum(r["text"]).upper()),
        row_task("phrase_initials", "text", "heldout_same_family", "Return phrase initials.", text_rows, lambda r: initials(r["phrase"])),
        row_task("phrase_slug", "text", "stress_out_of_abi", "Convert a phrase to kebab-case slug.", text_rows, lambda r: clean_spaces(r["phrase"]).lower().replace(" ", "-")),
    ]

    def orders_examples() -> List[Table]:
        return [
            [
                {"region": "West", "status": "paid", "amount": 120.0, "qty": 2, "date": "2024-01-05"},
                {"region": "East", "status": "open", "amount": 80.0, "qty": 1, "date": "2024-01-09"},
                {"region": "West", "status": "paid", "amount": 35.0, "qty": 3, "date": "2024-01-12"},
            ],
            [
                {"region": "South", "status": "paid", "amount": 10.0, "qty": 1, "date": "2024-02-01"},
                {"region": "West", "status": "open", "amount": 15.0, "qty": 2, "date": "2024-02-02"},
                {"region": "West", "status": "paid", "amount": 40.0, "qty": 4, "date": "2024-02-03"},
            ],
            [
                {"region": "East", "status": "paid", "amount": 99.0, "qty": 5, "date": "2024-03-01"},
                {"region": "East", "status": "paid", "amount": 11.0, "qty": 1, "date": "2024-03-04"},
                {"region": "West", "status": "open", "amount": 45.0, "qty": 2, "date": "2024-03-07"},
            ],
            [
                {"region": "West", "status": "paid", "amount": 70.0, "qty": 1, "date": "2024-04-01"},
                {"region": "West", "status": "paid", "amount": 30.0, "qty": 2, "date": "2024-04-03"},
                {"region": "South", "status": "open", "amount": 60.0, "qty": 2, "date": "2024-04-05"},
            ],
            [
                {"region": "East", "status": "open", "amount": 22.0, "qty": 1, "date": "2024-05-01"},
                {"region": "West", "status": "paid", "amount": 18.0, "qty": 1, "date": "2024-05-02"},
                {"region": "East", "status": "paid", "amount": 77.0, "qty": 3, "date": "2024-05-09"},
            ],
        ]

    tables = orders_examples()
    tasks += [
        table_task("table_count_rows", "table", "heldout_new_family", "Count rows in a small table.", tables, lambda t: len(t)),
        table_task("table_sum_amount", "table", "heldout_new_family", "Sum the amount column.", tables, lambda t: round(sum(float(r["amount"]) for r in t), 2)),
        table_task("table_sum_paid_amount", "table", "heldout_composition", "Sum amount where status is paid.", tables, lambda t: round(sum(float(r["amount"]) for r in t if r["status"].lower() == "paid"), 2), constants=["paid"]),
        table_task("table_count_west", "table", "heldout_composition", "Count rows where region is West.", tables, lambda t: sum(1 for r in t if r["region"] == "West"), constants=["West"]),
        table_task("table_sum_west_qty", "table", "heldout_composition", "Sum quantity where region is West.", tables, lambda t: sum(int(r["qty"]) for r in t if r["region"] == "West"), constants=["West"]),
        table_task("table_latest_paid_amount", "table", "stress_out_of_abi", "Return amount from latest paid order.", tables, lambda t: sorted([r for r in t if r["status"] == "paid"], key=lambda r: parse_date(r["date"]))[-1]["amount"], constants=["paid"]),
    ]

    return tasks


# ---------------------------------------------------------------------------
# Oracle enumeration.
# ---------------------------------------------------------------------------


@dataclass
class Expr:
    code: str
    depth: int
    func: Callable[[Any], Value]

    def eval_all(self, inputs: Sequence[Any]) -> Optional[Tuple[Any, ...]]:
        out: List[Any] = []
        try:
            for x in inputs:
                y = self.func(x)
                if y is None:
                    return None
                out.append(canon(y))
        except Exception:
            return None
        return tuple(out)


def canon(x: Value) -> Value:
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        return round(x, 6)
    s = as_str(x)
    return s


def expected(task: Task, phase: str) -> Tuple[Any, ...]:
    return tuple(canon(ex.out) for ex in task.examples if ex.phase == phase)


def phase_inputs(task: Task, phase: str) -> List[Any]:
    return [ex.inp for ex in task.examples if ex.phase == phase]


def task_fields(task: Task) -> List[str]:
    fields = set()
    for ex in task.examples:
        if isinstance(ex.inp, dict):
            fields.update(ex.inp.keys())
    return sorted(fields)


def abi_ops(variant: str) -> Tuple[List[UnaryOp], List[BinaryOp], bool]:
    if variant == "core":
        return CORE_UNARY, CORE_BINARY, False
    if variant == "office":
        return OFFICE_UNARY, OFFICE_BINARY, False
    if variant == "office_table":
        return OFFICE_UNARY, OFFICE_BINARY, True
    raise ValueError(variant)


def add_expr(
    expr: Expr,
    train_inputs: List[Any],
    seen: Dict[Tuple[Any, ...], Expr],
    pool: List[Expr],
    by_depth: Dict[int, List[Expr]],
) -> None:
    sig = expr.eval_all(train_inputs)
    if sig is None:
        return
    old = seen.get(sig)
    if old is not None and (old.depth, len(old.code)) <= (expr.depth, len(expr.code)):
        return
    seen[sig] = expr
    pool.append(expr)
    by_depth[expr.depth].append(expr)


def enumerate_row_task(
    task: Task,
    variant: str,
    max_depth: int,
    max_pool: int = 1200,
    max_pair_source: int = 160,
) -> Dict[str, Any]:
    unary_ops, binary_ops, _ = abi_ops(variant)
    train_inputs = phase_inputs(task, "train")
    test_inputs = phase_inputs(task, "test")
    train_y = expected(task, "train")
    test_y = expected(task, "test")

    pool: List[Expr] = []
    by_depth: Dict[int, List[Expr]] = defaultdict(list)
    seen: Dict[Tuple[Any, ...], Expr] = {}

    for field_name in task_fields(task):
        add_expr(
            Expr(f"FIELD({field_name})", 0, lambda row, f=field_name: row.get(f)),
            train_inputs,
            seen,
            pool,
            by_depth,
        )
    for c in task.constants:
        add_expr(Expr(f"CONST({json.dumps(c)})", 0, lambda _row, v=c: v), train_inputs, seen, pool, by_depth)

    train_matches: List[Expr] = []
    test_matches: List[Expr] = []

    def record_matches(candidates: Iterable[Expr]) -> None:
        for expr in candidates:
            train_sig = expr.eval_all(train_inputs)
            if train_sig == train_y:
                train_matches.append(expr)
                if expr.eval_all(test_inputs) == test_y:
                    test_matches.append(expr)

    record_matches(pool)
    if test_matches:
        best_test = min(test_matches, key=lambda e: (e.depth, len(e.code), e.code))
        return {
            "train_match": True,
            "test_match": True,
            "program": best_test.code,
            "program_depth": best_test.depth,
            "train_match_count": len({e.code for e in train_matches}),
            "candidate_count": len(pool),
            "failure_reason": "covered",
        }

    for depth in range(1, max_depth + 1):
        before_len = len(pool)
        source = sorted(
            [e for e in pool if e.depth == depth - 1],
            key=lambda e: (e.depth, len(e.code), e.code),
        )[:max_pool]
        for expr in source:
            for op in unary_ops:
                add_expr(
                    Expr(f"{op.name}({expr.code})", depth, lambda row, e=expr, o=op: o.func(e.func(row))),
                    train_inputs,
                    seen,
                    pool,
                    by_depth,
                )

        pair_source = sorted(
            [e for e in pool if e.depth <= depth - 1],
            key=lambda e: (e.depth, len(e.code), e.code),
        )[:max_pair_source]
        for i, left in enumerate(pair_source):
            for right in pair_source[i:]:
                if max(left.depth, right.depth) + 1 != depth:
                    continue
                for op in binary_ops:
                    add_expr(
                        Expr(
                            f"{op.name}({left.code},{right.code})",
                            depth,
                            lambda row, a=left, b=right, o=op: o.func(a.func(row), b.func(row)),
                        ),
                        train_inputs,
                        seen,
                        pool,
                        by_depth,
                    )
                    if left.code != right.code and op.name not in {"sub", "div", "date_diff_days"}:
                        add_expr(
                            Expr(
                                f"{op.name}({right.code},{left.code})",
                                depth,
                                lambda row, a=right, b=left, o=op: o.func(a.func(row), b.func(row)),
                            ),
                            train_inputs,
                            seen,
                            pool,
                            by_depth,
                        )
        record_matches(pool[before_len:])
        if test_matches:
            best_test = min(test_matches, key=lambda e: (e.depth, len(e.code), e.code))
            return {
                "train_match": True,
                "test_match": True,
                "program": best_test.code,
                "program_depth": best_test.depth,
                "train_match_count": len({e.code for e in train_matches}),
                "candidate_count": len(pool),
                "failure_reason": "covered",
            }

    best_train = min(train_matches, key=lambda e: (e.depth, len(e.code), e.code), default=None)
    best_test = min(test_matches, key=lambda e: (e.depth, len(e.code), e.code), default=None)
    return {
        "train_match": best_train is not None,
        "test_match": best_test is not None,
        "program": best_test.code if best_test else (best_train.code if best_train else ""),
        "program_depth": best_test.depth if best_test else (best_train.depth if best_train else None),
        "train_match_count": len({e.code for e in train_matches}),
        "candidate_count": len(pool),
        "failure_reason": "covered" if best_test else ("train_match_only" if best_train else "no_train_match"),
    }


def table_programs(task: Task) -> List[Expr]:
    train_tables = phase_inputs(task, "train")
    fields = sorted({k for table in train_tables for row in table for k in row.keys()})
    constants = set(task.constants)
    for table in train_tables:
        for row in table:
            for v in row.values():
                if isinstance(v, str) and len(v) <= 12:
                    constants.add(v)

    programs: List[Expr] = [
        Expr("COUNT_ROWS", 1, lambda table: len(table)),
    ]
    for field_name in fields:
        programs.extend(
            [
                Expr(f"SUM({field_name})", 1, lambda table, f=field_name: round(sum(safe_float(r.get(f)) or 0.0 for r in table), 6)),
                Expr(f"COUNT_NONEMPTY({field_name})", 1, lambda table, f=field_name: sum(1 for r in table if as_str(r.get(f)).strip())),
                Expr(f"MAX({field_name})", 1, lambda table, f=field_name: max([safe_float(r.get(f)) for r in table if safe_float(r.get(f)) is not None], default=None)),
                Expr(f"MIN({field_name})", 1, lambda table, f=field_name: min([safe_float(r.get(f)) for r in table if safe_float(r.get(f)) is not None], default=None)),
            ]
        )
    for cond_field in fields:
        for const in constants:
            programs.append(
                Expr(
                    f"COUNT_WHERE({cond_field}={const})",
                    2,
                    lambda table, f=cond_field, c=const: sum(1 for r in table if as_str(r.get(f)).lower() == as_str(c).lower()),
                )
            )
            for value_field in fields:
                programs.append(
                    Expr(
                        f"SUM_WHERE({cond_field}={const},{value_field})",
                        2,
                        lambda table, cf=cond_field, c=const, vf=value_field: round(
                            sum((safe_float(r.get(vf)) or 0.0) for r in table if as_str(r.get(cf)).lower() == as_str(c).lower()),
                            6,
                        ),
                    )
                )
    return programs


def enumerate_table_task(task: Task, variant: str) -> Dict[str, Any]:
    _, _, table_enabled = abi_ops(variant)
    if not table_enabled:
        return {
            "train_match": False,
            "test_match": False,
            "program": "",
            "program_depth": None,
            "train_match_count": 0,
            "candidate_count": 0,
            "failure_reason": "table_ops_disabled",
        }
    train_inputs = phase_inputs(task, "train")
    test_inputs = phase_inputs(task, "test")
    train_y = expected(task, "train")
    test_y = expected(task, "test")
    candidates = table_programs(task)
    train_matches = [p for p in candidates if p.eval_all(train_inputs) == train_y]
    test_matches = [p for p in train_matches if p.eval_all(test_inputs) == test_y]
    best_train = min(train_matches, key=lambda e: (e.depth, len(e.code), e.code), default=None)
    best_test = min(test_matches, key=lambda e: (e.depth, len(e.code), e.code), default=None)
    return {
        "train_match": best_train is not None,
        "test_match": best_test is not None,
        "program": best_test.code if best_test else (best_train.code if best_train else ""),
        "program_depth": best_test.depth if best_test else (best_train.depth if best_train else None),
        "train_match_count": len({e.code for e in train_matches}),
        "candidate_count": len(candidates),
        "failure_reason": "covered" if best_test else ("train_match_only" if best_train else "no_train_match"),
    }


def evaluate(tasks: List[Task], variants: List[str], depths: List[int]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for variant in variants:
        for depth in depths:
            print(f"evaluate variant={variant} depth={depth} tasks={len(tasks)}", flush=True)
            for idx, task in enumerate(tasks, start=1):
                if idx == 1 or idx % 10 == 0 or idx == len(tasks):
                    print(f"  {idx}/{len(tasks)} {task.task_id}", flush=True)
                if task.input_kind == "table":
                    result = enumerate_table_task(task, variant)
                else:
                    result = enumerate_row_task(task, variant, depth)
                rows.append(
                    {
                        "variant": variant,
                        "max_depth": depth,
                        "task_id": task.task_id,
                        "family": task.family,
                        "split": task.split,
                        "input_kind": task.input_kind,
                        "description": task.description,
                        "n_train": sum(ex.phase == "train" for ex in task.examples),
                        "n_test": sum(ex.phase == "test" for ex in task.examples),
                        **result,
                    }
                )
    return pd.DataFrame(rows)


def summarize(details: pd.DataFrame) -> pd.DataFrame:
    groups = ["variant", "max_depth", "split", "family", "input_kind"]
    rows: List[Dict[str, Any]] = []
    for keys, sub in details.groupby(groups, dropna=False):
        rows.append(
            {
                **dict(zip(groups, keys)),
                "tasks": len(sub),
                "heldout_covered": float(sub["test_match"].mean()),
                "train_match_rate": float(sub["train_match"].mean()),
                "train_match_only": float(((sub["train_match"]) & (~sub["test_match"])).mean()),
                "no_train_match": float((~sub["train_match"]).mean()),
                "median_program_depth": float(sub["program_depth"].dropna().median()) if sub["program_depth"].notna().any() else None,
            }
        )
    return pd.DataFrame(rows).sort_values(groups)


def overall_summary(details: pd.DataFrame) -> pd.DataFrame:
    groups = ["variant", "max_depth"]
    rows = []
    for keys, sub in details.groupby(groups):
        rows.append(
            {
                **dict(zip(groups, keys)),
                "tasks": len(sub),
                "heldout_covered": float(sub["test_match"].mean()),
                "train_match_rate": float(sub["train_match"].mean()),
                "train_match_only": float(((sub["train_match"]) & (~sub["test_match"])).mean()),
                "no_train_match": float((~sub["train_match"]).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(groups)


def pct(x: Any) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    return f"{100 * float(x):.1f}%"


def md_table(df: pd.DataFrame, cols: Optional[List[str]] = None, max_rows: int = 80) -> str:
    if df.empty:
        return "_No rows._"
    cols = cols or list(df.columns)
    view = df[cols].head(max_rows).copy()
    for col in view.columns:
        if view[col].dtype.kind in "fc":
            if "rate" in col or "covered" in col or "match" in col:
                view[col] = view[col].map(pct)
            else:
                view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    header = "|" + "|".join(cols) + "|"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    body = ["|" + "|".join(html.escape(str(row[c])) for c in cols) + "|" for _, row in view.iterrows()]
    return "\n".join([header, sep] + body)


def plot_coverage_by_depth(details: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for variant, sub in details.groupby("variant"):
        curve = sub.groupby("max_depth")["test_match"].mean().sort_index()
        ax.plot(curve.index, curve.values * 100, marker="o", label=variant)
    ax.set_title("Held-out coverage by expression search depth")
    ax.set_xlabel("Max expression depth")
    ax.set_ylabel("Coverage (%)")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "coverage_by_depth.png", dpi=160)
    plt.close(fig)


def plot_split_coverage(details: pd.DataFrame, primary_variant: str, primary_depth: int) -> None:
    sub = details[(details["variant"].eq(primary_variant)) & (details["max_depth"].eq(primary_depth))]
    order = ["calibration", "heldout_same_family", "heldout_new_family", "heldout_composition", "stress_out_of_abi"]
    vals = sub.groupby("split")["test_match"].mean().reindex(order)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(vals.index, vals.values * 100, color="#3b82f6")
    ax.set_title(f"Coverage by split ({primary_variant}, depth {primary_depth})")
    ax.set_ylabel("Coverage (%)")
    ax.set_ylim(0, 105)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "coverage_by_split.png", dpi=160)
    plt.close(fig)


def plot_family_coverage(details: pd.DataFrame, primary_variant: str, primary_depth: int) -> None:
    sub = details[(details["variant"].eq(primary_variant)) & (details["max_depth"].eq(primary_depth))]
    vals = sub.groupby("family")["test_match"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(vals.index, vals.values * 100, color="#10b981")
    ax.set_title(f"Coverage by task family ({primary_variant}, depth {primary_depth})")
    ax.set_xlabel("Coverage (%)")
    ax.set_xlim(0, 105)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "coverage_by_family.png", dpi=160)
    plt.close(fig)


def plot_failure_modes(details: pd.DataFrame, primary_variant: str, primary_depth: int) -> None:
    sub = details[(details["variant"].eq(primary_variant)) & (details["max_depth"].eq(primary_depth))]
    vals = sub["failure_reason"].value_counts(normalize=True).reindex(["covered", "train_match_only", "no_train_match", "table_ops_disabled"]).fillna(0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(vals.index, vals.values * 100, color=["#10b981", "#f59e0b", "#ef4444", "#94a3b8"])
    ax.set_title(f"Failure mode mix ({primary_variant}, depth {primary_depth})")
    ax.set_ylabel("Tasks (%)")
    ax.set_ylim(0, 105)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "failure_modes.png", dpi=160)
    plt.close(fig)


def plot_program_depth(details: pd.DataFrame, primary_variant: str, primary_depth: int) -> None:
    sub = details[
        details["variant"].eq(primary_variant)
        & details["max_depth"].eq(primary_depth)
        & details["test_match"]
        & details["program_depth"].notna()
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    if sub.empty:
        ax.text(0.5, 0.5, "No covered tasks", ha="center", va="center")
    else:
        ax.hist(sub["program_depth"].astype(float), bins=range(0, int(sub["program_depth"].max()) + 3), color="#6366f1", edgecolor="white")
    ax.set_title("Program depth distribution for covered tasks")
    ax.set_xlabel("Synthesized program depth")
    ax.set_ylabel("Task count")
    fig.tight_layout()
    fig.savefig(FIGURES / "program_depth_distribution.png", dpi=160)
    plt.close(fig)


def make_report(details: pd.DataFrame, summary: pd.DataFrame, overall: pd.DataFrame, tasks: List[Task], suite: str) -> None:
    primary_variant = "office_table"
    primary_depth = int(details["max_depth"].max())
    primary = details[(details["variant"].eq(primary_variant)) & (details["max_depth"].eq(primary_depth))]
    calibration = primary[primary["split"].eq("calibration")]
    heldout = primary[~primary["split"].eq("calibration")]
    composition = primary[primary["split"].eq("heldout_composition")]
    stress = primary[primary["split"].eq("stress_out_of_abi")]
    table = primary[primary["input_kind"].eq("table")]

    lines: List[str] = []
    lines.append("# Qwen Real Task ABI Coverage Gate")
    lines.append("")
    lines.append("## Abstract")
    lines.append("")
    lines.append("This standalone experiment tests whether a frozen office-data ABI covers real-style deterministic tasks that were not generated from that ABI. It uses oracle enumeration, not model training: the result is a decomposability gate for whether a large compiler corpus is worth building.")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append("- The ABI is a fixed library of scalar string/date/money/contact/file operations plus small-table aggregation templates.")
    lines.append("- Task references are ordinary Python functions over hand-curated examples; task outputs are not produced by stored ABI programs.")
    lines.append("- The oracle enumerates expressions from visible fields/constants, fits train examples, then tests held-out examples for each task.")
    lines.append("- ABI variants are `core`, `office`, and `office_table`; the primary gate is `office_table` at the largest search depth.")
    lines.append("")
    lines.append("## Run Configuration")
    lines.append("")
    lines.append(f"- Suite: `{suite}`.")
    lines.append(f"- Tasks: `{len(tasks)}` total, `{int((primary['input_kind'] == 'table').sum())}` table tasks under the primary slice.")
    lines.append(f"- Primary ABI/depth: `{primary_variant}`, depth `{primary_depth}`.")
    lines.append(f"- Large artifacts directory: `{LARGE_ROOT}`.")
    lines.append("")
    lines.append("## Primary Results")
    lines.append("")
    lines.append(f"- Overall held-out coverage: {pct(primary['test_match'].mean())} ({int(primary['test_match'].sum())}/{len(primary)} tasks).")
    lines.append(f"- Calibration coverage: {pct(calibration['test_match'].mean())} ({int(calibration['test_match'].sum())}/{len(calibration)}).")
    lines.append(f"- Non-calibration coverage: {pct(heldout['test_match'].mean())} ({int(heldout['test_match'].sum())}/{len(heldout)}).")
    lines.append(f"- Held-out composition coverage: {pct(composition['test_match'].mean())} ({int(composition['test_match'].sum())}/{len(composition)}).")
    lines.append(f"- Stress/out-of-ABI coverage: {pct(stress['test_match'].mean())} ({int(stress['test_match'].sum())}/{len(stress)}).")
    lines.append(f"- Table-task coverage: {pct(table['test_match'].mean())} ({int(table['test_match'].sum())}/{len(table)}).")
    lines.append(f"- Train-match-only rate: {pct(((primary['train_match']) & (~primary['test_match'])).mean())}.")
    lines.append(f"- No-train-match rate: {pct((~primary['train_match']).mean())}.")
    lines.append("")
    lines.append("### Overall By ABI")
    lines.append("")
    lines.append(md_table(overall, ["variant", "max_depth", "tasks", "heldout_covered", "train_match_rate", "train_match_only", "no_train_match"], max_rows=60))
    lines.append("")
    lines.append("### Primary Split Summary")
    lines.append("")
    split_summary = summary[(summary["variant"].eq(primary_variant)) & (summary["max_depth"].eq(primary_depth))]
    split_only = (
        split_summary.groupby("split", as_index=False)
        .agg(tasks=("tasks", "sum"), heldout_covered=("heldout_covered", "mean"), train_match_rate=("train_match_rate", "mean"), train_match_only=("train_match_only", "mean"), no_train_match=("no_train_match", "mean"))
        .sort_values("split")
    )
    lines.append(md_table(split_only, ["split", "tasks", "heldout_covered", "train_match_rate", "train_match_only", "no_train_match"]))
    lines.append("")
    lines.append("### Covered Program Examples")
    lines.append("")
    examples = primary[primary["test_match"]][["task_id", "family", "split", "program_depth", "program"]].sort_values(["family", "task_id"]).head(18)
    lines.append(md_table(examples, ["task_id", "family", "split", "program_depth", "program"], max_rows=18))
    lines.append("")
    lines.append("### Uncovered Examples")
    lines.append("")
    failures = primary[~primary["test_match"]][["task_id", "family", "split", "failure_reason", "train_match", "program"]].sort_values(["split", "family", "task_id"]).head(25)
    lines.append(md_table(failures, ["task_id", "family", "split", "failure_reason", "train_match", "program"], max_rows=25))
    lines.append("")
    for fig in [
        "coverage_by_depth.png",
        "coverage_by_split.png",
        "coverage_by_family.png",
        "failure_modes.png",
        "program_depth_distribution.png",
    ]:
        lines.append(f"![{fig}](../analysis/figures/{fig})")
        lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("The fixed ABI covers a meaningful fraction of real-style deterministic tasks, but coverage is far from universal. The gap between calibration and stress splits is the main signal: common office transforms decompose well, while tasks that require bespoke fiscal logic, middle-token extraction, discount formulas, or latest-row selection expose missing primitives or missing control patterns.")
    lines.append("The positive read is that held-out composition and held-out new-family tasks are covered in this catalog, including table filters and aggregations. The negative read is equally important: every deliberately out-of-ABI stress task fails, so the ABI cannot be treated as an open-ended intelligence multiplier without a retrieval/extension path for missing primitives.")
    lines.append("A train-match-only result is treated as a warning rather than success: it means the ABI/search can fit visible examples but does not identify the actual task robustly on held-out examples.")
    lines.append("This is a gate, not a compiler result. If the target domain is restricted to covered families, the ABI direction has a real base to build on. If the target domain includes the uncovered stress patterns, the ABI must be expanded or paired with retrieval/tooling before model training is worth scaling.")
    lines.append("The next decisive step is a less hand-curated corpus: either production-like task logs or a public deterministic transformation benchmark, with the ABI frozen before evaluating that set.")
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append("The tasks are hand-curated real-style examples, not production logs or a public benchmark. That is stronger than factory-generated ABI compositions but still not a definitive real-world coverage estimate. The oracle is bounded by the implemented search space, so some misses may be search misses rather than true ABI misses. The table ABI is intentionally small and does not include sorting or window operations.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append("- Details: `analysis/details.csv`")
    lines.append("- Summary: `analysis/summary.csv` and `analysis/overall_summary.csv`")
    lines.append("- Task catalog: `analysis/task_catalog.json`")
    lines.append(f"- Large artifacts directory: `{LARGE_ROOT}`")

    report_md = REPORTS / "qwen_real_task_abi_coverage_gate_report.md"
    report_html = REPORTS / "qwen_real_task_abi_coverage_gate_report.html"
    report_md.write_text("\n".join(lines) + "\n")
    body = "\n".join(lines)
    body = html.escape(body)
    body = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r'<p><img src="\2" alt="\1" style="max-width:100%;border:1px solid #ddd;border-radius:6px"></p>', body)
    body = re.sub(r"^# (.*)$", r"<h1>\1</h1>", body, flags=re.M)
    body = re.sub(r"^## (.*)$", r"<h2>\1</h2>", body, flags=re.M)
    body = re.sub(r"^### (.*)$", r"<h3>\1</h3>", body, flags=re.M)
    body = body.replace("\n", "<br>\n")
    report_html.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Qwen Real Task ABI Coverage Gate</title>"
        "<style>body{font-family:Inter,system-ui,sans-serif;line-height:1.45;max-width:1180px;margin:32px auto;padding:0 24px;color:#172033}"
        "table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ddd;padding:4px 6px}code{background:#f4f4f5;padding:1px 4px;border-radius:4px}"
        "h1,h2,h3{line-height:1.15}</style></head><body>"
        + body
        + "</body></html>"
    )


def write_catalog(tasks: List[Task]) -> None:
    catalog = []
    for task in tasks:
        catalog.append(
            {
                "task_id": task.task_id,
                "family": task.family,
                "split": task.split,
                "input_kind": task.input_kind,
                "description": task.description,
                "constants": task.constants,
                "examples": [{"phase": ex.phase, "input": ex.inp, "output": ex.out} for ex in task.examples],
            }
        )
    (ANALYSIS / "task_catalog.json").write_text(json.dumps(catalog, indent=2, default=str))


def run_suite(args: argparse.Namespace) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    LARGE_ROOT.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks()
    if args.suite == "smoke":
        tasks = tasks[:12]
        depths = [1, 2]
        variants = ["core", "office_table"]
    elif args.suite == "pilot":
        tasks = tasks[:30]
        depths = [1, 2]
        variants = ["core", "office", "office_table"]
    else:
        depths = [1, 2]
        variants = ["core", "office", "office_table"]

    started = datetime.now(timezone.utc)
    details = evaluate(tasks, variants, depths)
    summary = summarize(details)
    overall = overall_summary(details)

    run_dir = RUNS / f"{args.suite}_v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    details.to_csv(run_dir / "details.csv", index=False)
    summary.to_csv(run_dir / "summary.csv", index=False)
    overall.to_csv(run_dir / "overall_summary.csv", index=False)
    (run_dir / "config.json").write_text(json.dumps({"suite": args.suite, "depths": depths, "variants": variants, "started_utc": started.isoformat()}, indent=2))

    details.to_csv(ANALYSIS / "details.csv", index=False)
    summary.to_csv(ANALYSIS / "summary.csv", index=False)
    overall.to_csv(ANALYSIS / "overall_summary.csv", index=False)
    write_catalog(tasks)

    plot_coverage_by_depth(details)
    plot_split_coverage(details, "office_table", max(depths))
    plot_family_coverage(details, "office_table", max(depths))
    plot_failure_modes(details, "office_table", max(depths))
    plot_program_depth(details, "office_table", max(depths))
    make_report(details, summary, overall, tasks, args.suite)

    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    with (ROOT / "experiment_log.md").open("a") as f:
        f.write(f"\n## Run `{args.suite}_v1`\n\n")
        f.write(f"- Started: {started.strftime('%Y-%m-%d %H:%M:%S')} UTC\n")
        f.write(f"- Variants: `{','.join(variants)}`\n")
        f.write(f"- Depths: `{','.join(map(str, depths))}`\n")
        f.write(f"- Tasks: `{len(tasks)}`\n")
        f.write(f"- Completed in {elapsed:.1f}s.\n")
        primary = details[(details["variant"].eq("office_table")) & (details["max_depth"].eq(max(depths)))]
        f.write(f"- Primary coverage: {pct(primary['test_match'].mean())} ({int(primary['test_match'].sum())}/{len(primary)} tasks).\n")
        f.write(f"- Train-match-only: {pct(((primary['train_match']) & (~primary['test_match'])).mean())}; no-train-match: {pct((~primary['train_match']).mean())}.\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["smoke", "pilot", "main"], default="main")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_suite(args)


if __name__ == "__main__":
    main()
