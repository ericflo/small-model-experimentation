from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


Table = list[list[str]]


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    depth: int
    program: dict[str, Any]
    func: Callable[[Table], Table]


def normalize_table(table: Table) -> Table:
    return [[str(cell) for cell in row] for row in table]


def equal(a: Table, b: Table) -> bool:
    return normalize_table(a) == normalize_table(b)


def safe_call(func: Callable[[Table], Table], table: Table) -> tuple[bool, Table | str]:
    try:
        return True, normalize_table(func(normalize_table(table)))
    except Exception as exc:
        return False, type(exc).__name__


def project(cols: tuple[int, ...]) -> Callable[[Table], Table]:
    return lambda table: [[row[i] if i < len(row) else "" for i in cols] for row in table]


def drop_rows(n: int) -> Callable[[Table], Table]:
    return lambda table: table[n:]


def take_rows(start: int, end: int | None) -> Callable[[Table], Table]:
    return lambda table: table[start:end]


def unpivot(id_cols: int, start_col: int, keep_header: bool = True, skip_empty: bool = False) -> Callable[[Table], Table]:
    def f(table: Table) -> Table:
        header = table[0] if keep_header and table else [str(i) for i in range(max((len(r) for r in table), default=0))]
        rows = table[1:] if keep_header else table
        out: Table = []
        for row in rows:
            ids = row[:id_cols]
            for j in range(start_col, len(row)):
                value = row[j]
                if skip_empty and value == "":
                    continue
                out.append([*ids, header[j] if j < len(header) else str(j), value])
        return out
    return f


def unpivot_no_header(id_cols: int, start_col: int) -> Callable[[Table], Table]:
    def f(table: Table) -> Table:
        out: Table = []
        for row in table:
            ids = row[:id_cols]
            for j in range(start_col, len(row)):
                out.append([*ids, row[j]])
        return out
    return f


def key_value_fold() -> Callable[[Table], Table]:
    def f(table: Table) -> Table:
        if not table:
            return []
        keys = []
        values_rows = []
        for row in table:
            row_keys = []
            row_values = []
            for cell in row[1:]:
                if ":" not in cell:
                    continue
                key, value = cell.split(":", 1)
                row_keys.append(key)
                row_values.append(value)
            if not keys:
                keys = row_keys
            values_rows.append([row[0], *row_values])
        return [["", *keys], *values_rows]
    return f


def regex_extract(pattern: str, groups: int) -> Callable[[Table], Table]:
    compiled = re.compile(pattern, re.I)

    def f(table: Table) -> Table:
        out: Table = []
        for row in table:
            text = " ".join(row)
            match = compiled.search(text)
            if match:
                out.append([match.group(i) for i in range(1, groups + 1)])
            else:
                out.append([""] * groups)
        return out
    return f


def split_cells(delim: str) -> Callable[[Table], Table]:
    def f(table: Table) -> Table:
        out: Table = []
        for row in table:
            new = []
            for cell in row:
                new.extend(cell.split(delim))
            out.append(new)
        return out
    return f


def transpose() -> Callable[[Table], Table]:
    def f(table: Table) -> Table:
        width = max((len(row) for row in table), default=0)
        padded = [row + [""] * (width - len(row)) for row in table]
        return [list(col) for col in zip(*padded)]
    return f


def compose(a: Candidate, b: Candidate) -> Candidate:
    def f(table: Table) -> Table:
        return b.func(a.func(table))

    return Candidate(
        name=f"{a.name}__then__{b.name}",
        family="compose",
        depth=a.depth + b.depth,
        program={"steps": [*a.program["steps"], *b.program["steps"]]},
        func=f,
    )


def primitive_candidates(max_width: int) -> list[Candidate]:
    cands: list[Candidate] = []

    def add(name: str, family: str, func: Callable[[Table], Table], **program: Any) -> None:
        cands.append(Candidate(name, family, 1, {"steps": [{"op": name, **program}]}, func))

    add("identity", "generic", lambda table: table)
    for n in range(1, min(3, max_width) + 1):
        add(f"drop_rows_{n}", "row", drop_rows(n), n=n)
    for start in range(0, min(3, max_width)):
        for end in [None, start + 1, start + 2, start + 3]:
            if end is None or end > start:
                add(f"take_rows_{start}_{end}", "row", take_rows(start, end), start=start, end=end)
    for width in range(1, min(max_width, 4) + 1):
        cols = tuple(range(width))
        add(f"project_first_{width}", "column", project(cols), cols=cols)
    if max_width >= 2:
        for i in range(max_width):
            for j in range(max_width):
                if i != j:
                    add(f"project_{i}_{j}", "column", project((i, j)), cols=[i, j])
    for id_cols in [0, 1, 2]:
        for start in range(id_cols, min(max_width, id_cols + 4)):
            add(f"unpivot_h_id{id_cols}_s{start}", "unpivot", unpivot(id_cols, start, True, False), id_cols=id_cols, start=start, header=True)
            add(f"unpivot_h_skip_id{id_cols}_s{start}", "unpivot", unpivot(id_cols, start, True, True), id_cols=id_cols, start=start, header=True, skip_empty=True)
            add(f"unpivot_noh_id{id_cols}_s{start}", "unpivot", unpivot_no_header(id_cols, start), id_cols=id_cols, start=start, header=False)
    add("key_value_fold", "fold", key_value_fold())
    for delim in [":", "-", "/", ",", " "]:
        add(f"split_{delim}", "split", split_cells(delim), delim=delim)
    add("transpose", "reshape", transpose())
    regexes = {
        "money_bedrooms": (r"(\$\d+)\s*/\s*(\d+)br", 2),
        "money": (r"(\$\d+)", 1),
        "paren": (r"\(([^)]+)\)", 1),
        "first_number": (r"(\d+)", 1),
        "two_numbers": (r"(\d+).*?(\d+)", 2),
        "word_number": (r"([A-Za-z]+).*?(\d+)", 2),
    }
    for name, (pattern, groups) in regexes.items():
        add(f"regex_{name}", "regex", regex_extract(pattern, groups), pattern=pattern, groups=groups)
    return cands


def candidate_programs(input_table: Table, output_table: Table, max_candidates: int = 256) -> list[Candidate]:
    max_width = max((len(row) for row in input_table), default=1)
    prims = primitive_candidates(max_width)
    visible: dict[str, Candidate] = {}
    for cand in prims:
        ok, out = safe_call(cand.func, input_table)
        if ok and equal(out, output_table):
            visible[json.dumps(cand.program, sort_keys=True)] = cand
    prefixes = list(visible.values())[:80]
    for prefix in prefixes:
        for cand in prims:
            comp = compose(prefix, cand)
            ok, out = safe_call(comp.func, input_table)
            if ok and equal(out, output_table):
                visible[json.dumps(comp.program, sort_keys=True)] = comp
                if len(visible) >= max_candidates:
                    return sorted(visible.values(), key=lambda c: json.dumps(c.program, sort_keys=True))
    return sorted(visible.values(), key=lambda c: json.dumps(c.program, sort_keys=True))


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


def family_name(row: dict[str, Any]) -> str:
    stem = row["file"].replace(".txt", "")
    parts = stem.split("_")
    if len(parts) >= 3 and parts[1].isdigit():
        return f"synthetic_{parts[1]}"
    if stem.startswith("exp0_"):
        return "_".join(parts[1:-1]) or row["test_name"]
    return row["test_name"]


def run_case(row: dict[str, Any]) -> dict[str, Any]:
    cands = candidate_programs(row["input_table"], row["output_table"])
    heldout = []
    for cand in cands:
        ok, out = safe_call(cand.func, row["testing_table"])
        if ok and equal(out, row["test_answer"]):
            heldout.append(cand)
    first = cands[0] if cands else None
    first_heldout = False
    if first:
        ok, out = safe_call(first.func, row["testing_table"])
        first_heldout = ok and equal(out, row["test_answer"])
    return {
        "file": row["file"],
        "family": family_name(row),
        "num_samples": row["num_samples"],
        "raw_covered": bool(cands),
        "heldout_covered": bool(heldout),
        "candidate_count": len(cands),
        "heldout_winner_count": len(heldout),
        "first_visible_heldout": first_heldout,
        "winning_program": heldout[0].program if heldout else None,
        "winning_family": heldout[0].family if heldout else None,
        "winning_depth": heldout[0].depth if heldout else None,
        "first_program": first.program if first else None,
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    def metrics(group: list[dict[str, Any]]) -> dict[str, Any]:
        if not group:
            return {"n": 0}
        return {
            "n": len(group),
            "raw_covered": sum(r["raw_covered"] for r in group),
            "raw_coverage": sum(r["raw_covered"] for r in group) / len(group),
            "heldout_covered": sum(r["heldout_covered"] for r in group),
            "heldout_coverage": sum(r["heldout_covered"] for r in group) / len(group),
            "first_visible_heldout": sum(r["first_visible_heldout"] for r in group),
            "first_visible_accuracy": sum(r["first_visible_heldout"] for r in group) / len(group),
            "candidate_count_mean": sum(r["candidate_count"] for r in group) / len(group),
        }

    families = sorted({r["family"] for r in records})
    samples = sorted({r["num_samples"] for r in records})
    return {
        "overall": metrics(records),
        "by_family": {fam: metrics([r for r in records if r["family"] == fam]) for fam in families},
        "by_num_samples": {str(n): metrics([r for r in records if r["num_samples"] == n]) for n in samples},
        "winner_depth_counts": {str(k): v for k, v in sorted(Counter(r["winning_depth"] for r in records if r["heldout_covered"]).items())},
        "winner_family_counts": dict(sorted(Counter(r["winning_family"] for r in records if r["heldout_covered"]).items())),
    }

