from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class OperatorSpec:
    name: str
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


def _add_unique(specs: list[OperatorSpec], names: set[str], name: str, family: str, fn: Callable[[list[int]], int]) -> None:
    if name in names:
        return
    specs.append(OperatorSpec(name=name, family=family, fn=fn))
    names.add(name)


def build_operator_library(max_size: int = 512) -> list[OperatorSpec]:
    specs: list[OperatorSpec] = []
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
        ("range", "core", lambda xs: max(xs) - min(xs)),
        ("unique_count", "core", lambda xs: len(set(xs))),
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
    for name, family, fn in core:
        _add_unique(specs, names, name, family, fn)
        if len(specs) >= max_size:
            return specs

    for k in range(10):
        _add_unique(specs, names, f"sorted_{k}", "order_stat", lambda xs, k=k: sorted(xs)[_clamp_index(k, xs)])
        _add_unique(specs, names, f"revsorted_{k}", "order_stat", lambda xs, k=k: sorted(xs, reverse=True)[_clamp_index(k, xs)])
        if len(specs) >= max_size:
            return specs[:max_size]

    for value in range(0, 21):
        _add_unique(specs, names, f"count_eq_{value}", "count_eq", lambda xs, value=value: sum(1 for item in xs if item == value))
        _add_unique(specs, names, f"count_gt_{value}", "count_threshold", lambda xs, value=value: sum(1 for item in xs if item > value))
        _add_unique(specs, names, f"count_lt_{value}", "count_threshold", lambda xs, value=value: sum(1 for item in xs if item < value))
        if len(specs) >= max_size:
            return specs[:max_size]

    for modulus in range(2, 98):
        _add_unique(specs, names, f"sum_mod_{modulus}", "sum_mod", lambda xs, modulus=modulus: sum(xs) % modulus)
        _add_unique(specs, names, f"prod_mod_{modulus}", "prod_mod", lambda xs, modulus=modulus: _prod(xs) % modulus)
        _add_unique(specs, names, f"first_mod_{modulus}", "edge_mod", lambda xs, modulus=modulus: xs[0] % modulus)
        _add_unique(specs, names, f"last_mod_{modulus}", "edge_mod", lambda xs, modulus=modulus: xs[-1] % modulus)
        _add_unique(specs, names, f"max_mod_{modulus}", "extreme_mod", lambda xs, modulus=modulus: max(xs) % modulus)
        _add_unique(specs, names, f"min_mod_{modulus}", "extreme_mod", lambda xs, modulus=modulus: min(xs) % modulus)
        if len(specs) >= max_size:
            return specs[:max_size]

    for threshold in range(4, 164, 4):
        _add_unique(specs, names, f"sum_clip_{threshold}", "clip", lambda xs, threshold=threshold: min(sum(xs), threshold))
        _add_unique(specs, names, f"prod_clip_{threshold}", "clip", lambda xs, threshold=threshold: min(_prod(xs), threshold))
        if len(specs) >= max_size:
            return specs[:max_size]

    for divisor in range(2, 32):
        _add_unique(
            specs,
            names,
            f"count_divisible_{divisor}",
            "divisibility",
            lambda xs, divisor=divisor: sum(1 for item in xs if item % divisor == 0),
        )
        for remainder in range(divisor):
            _add_unique(
                specs,
                names,
                f"count_mod_{divisor}_{remainder}",
                "remainder_count",
                lambda xs, divisor=divisor, remainder=remainder: sum(1 for item in xs if item % divisor == remainder),
            )
            if len(specs) >= max_size:
                return specs[:max_size]
        if len(specs) >= max_size:
            return specs[:max_size]

    raise RuntimeError(f"operator generator produced only {len(specs)} specs")

