from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class OperatorSpec:
    index: int
    alias: str
    description: str
    family: str
    fn: Callable[[list[int]], int]
    signature: str = "list[int] -> int"

    def eval(self, xs: list[int]) -> int:
        return int(self.fn(xs))


def _prod(xs: list[int]) -> int:
    out = 1
    for value in xs:
        out *= value
    return out


def _gcd(xs: list[int]) -> int:
    out = 0
    for value in xs:
        out = math.gcd(out, int(value))
    return abs(out)


def _median_low(xs: list[int]) -> int:
    values = sorted(xs)
    return values[(len(values) - 1) // 2]


def _median_high(xs: list[int]) -> int:
    values = sorted(xs)
    return values[len(values) // 2]


def _mode_smallest(xs: list[int]) -> int:
    counts: dict[int, int] = {}
    for value in xs:
        counts[value] = counts.get(value, 0) + 1
    return min(counts, key=lambda value: (-counts[value], value))


def _alt_sum(xs: list[int]) -> int:
    return sum(value if index % 2 == 0 else -value for index, value in enumerate(xs))


def _clamp_index(index: int, xs: list[int]) -> int:
    return min(max(index, 0), len(xs) - 1)


def _add(
    rows: list[tuple[str, str, Callable[[list[int]], int]]],
    names: set[str],
    description: str,
    family: str,
    fn: Callable[[list[int]], int],
) -> None:
    if description in names:
        return
    names.add(description)
    rows.append((description, family, fn))


def build_operator_library(max_size: int = 512) -> list[OperatorSpec]:
    rows: list[tuple[str, str, Callable[[list[int]], int]]] = []
    names: set[str] = set()

    core: list[tuple[str, str, Callable[[list[int]], int]]] = [
        ("sum", "core", sum),
        ("first", "core", lambda xs: xs[0]),
        ("last", "core", lambda xs: xs[-1]),
        ("max", "core", max),
        ("min", "core", min),
        ("prod", "core", _prod),
        ("gcd", "core", _gcd),
        ("len", "core", len),
        ("range=max-min", "core", lambda xs: max(xs) - min(xs)),
        ("uniq_count", "core", lambda xs: len(set(xs))),
        ("median_low", "core", _median_low),
        ("median_high", "core", _median_high),
        ("mode_smallest", "core", _mode_smallest),
        ("count_even", "core", lambda xs: sum(1 for value in xs if value % 2 == 0)),
        ("count_odd", "core", lambda xs: sum(1 for value in xs if value % 2 == 1)),
        ("sum_even", "core", lambda xs: sum(value for value in xs if value % 2 == 0)),
        ("sum_odd", "core", lambda xs: sum(value for value in xs if value % 2 == 1)),
        ("alt_sum", "core", _alt_sum),
        ("abs_alt_sum", "core", lambda xs: abs(_alt_sum(xs))),
    ]
    for description, family, fn in core:
        _add(rows, names, description, family, fn)
        if len(rows) >= max_size:
            return _with_aliases(rows)

    for k in range(10):
        _add(rows, names, f"sort[{k}]", "order_stat", lambda xs, k=k: sorted(xs)[_clamp_index(k, xs)])
        _add(rows, names, f"rsort[{k}]", "order_stat", lambda xs, k=k: sorted(xs, reverse=True)[_clamp_index(k, xs)])
        if len(rows) >= max_size:
            return _with_aliases(rows[:max_size])

    for value in range(0, 21):
        _add(rows, names, f"count_eq({value})", "count_eq", lambda xs, value=value: sum(1 for item in xs if item == value))
        _add(rows, names, f"count_gt({value})", "count_threshold", lambda xs, value=value: sum(1 for item in xs if item > value))
        _add(rows, names, f"count_lt({value})", "count_threshold", lambda xs, value=value: sum(1 for item in xs if item < value))
        if len(rows) >= max_size:
            return _with_aliases(rows[:max_size])

    for modulus in range(2, 98):
        _add(rows, names, f"sum%{modulus}", "sum_mod", lambda xs, modulus=modulus: sum(xs) % modulus)
        _add(rows, names, f"prod%{modulus}", "prod_mod", lambda xs, modulus=modulus: _prod(xs) % modulus)
        _add(rows, names, f"first%{modulus}", "edge_mod", lambda xs, modulus=modulus: xs[0] % modulus)
        _add(rows, names, f"last%{modulus}", "edge_mod", lambda xs, modulus=modulus: xs[-1] % modulus)
        _add(rows, names, f"max%{modulus}", "extreme_mod", lambda xs, modulus=modulus: max(xs) % modulus)
        _add(rows, names, f"min%{modulus}", "extreme_mod", lambda xs, modulus=modulus: min(xs) % modulus)
        if len(rows) >= max_size:
            return _with_aliases(rows[:max_size])

    raise RuntimeError(f"operator generator produced only {len(rows)} rows")


def _with_aliases(rows: list[tuple[str, str, Callable[[list[int]], int]]]) -> list[OperatorSpec]:
    return [
        OperatorSpec(index=index, alias=f"op_{index:03d}", description=description, family=family, fn=fn)
        for index, (description, family, fn) in enumerate(rows)
    ]
