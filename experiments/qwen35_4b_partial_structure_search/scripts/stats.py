#!/usr/bin/env python3
"""Small dependency-free paired statistics for the experiment.

All uncertainty resamples whole tasks. Prefixes and sibling groups from the same task are
not independent observations.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Callable, Hashable, Iterable, Mapping, Sequence


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    """Tie-aware Mann-Whitney AUROC."""
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have equal length")
    positives = sum(int(y) for y in labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    order = sorted(range(len(scores)), key=lambda i: float(scores[i]))
    rank_sum = 0.0
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        value = float(scores[order[cursor]])
        while end < len(order) and float(scores[order[end]]) == value:
            end += 1
        average_rank = ((cursor + 1) + end) / 2.0
        rank_sum += average_rank * sum(int(labels[order[j]]) for j in range(cursor, end))
        cursor = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def macro_group_auroc(
    rows: Iterable[Mapping[str, object]],
    score_key: str,
    label_key: str = "live",
    group_keys: Sequence[str] = ("task_id", "prefix_len"),
) -> tuple[float | None, int]:
    groups: dict[tuple[Hashable, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in group_keys)].append(row)
    values: list[float] = []
    for group in groups.values():
        value = auroc(
            [float(row[score_key]) for row in group],
            [int(bool(row[label_key])) for row in group],
        )
        if value is not None:
            values.append(value)
    return (sum(values) / len(values), len(values)) if values else (None, 0)


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("cannot take a percentile of an empty sequence")
    if not 0 <= q <= 1:
        raise ValueError("q must lie in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = q * (len(ordered) - 1)
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return ordered[lo]
    weight = position - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def cluster_bootstrap(
    task_values: Mapping[Hashable, object],
    metric: Callable[[Sequence[object]], float],
    reps: int = 10_000,
    seed: int = 0,
) -> dict[str, float | int]:
    """Bootstrap tasks with replacement and return a percentile interval."""
    keys = sorted(task_values, key=str)
    if not keys:
        raise ValueError("task_values is empty")
    observed = float(metric([task_values[key] for key in keys]))
    rng = random.Random(seed)
    draws = []
    for _ in range(reps):
        sample = [task_values[rng.choice(keys)] for _ in keys]
        draws.append(float(metric(sample)))
    return {
        "estimate": observed,
        "ci_low": percentile(draws, 0.025),
        "ci_high": percentile(draws, 0.975),
        "reps": reps,
        "n_tasks": len(keys),
    }


def mcnemar_exact(a: Sequence[bool], b: Sequence[bool]) -> dict[str, float | int]:
    """Two-sided exact McNemar test for paired binary task outcomes."""
    if len(a) != len(b):
        raise ValueError("paired outcomes must have equal length")
    a_only = sum(bool(x) and not bool(y) for x, y in zip(a, b))
    b_only = sum(bool(y) and not bool(x) for x, y in zip(a, b))
    discordant = a_only + b_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(0, min(a_only, b_only) + 1)) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {"a_only": a_only, "b_only": b_only, "discordant": discordant, "p_two_sided": p_value}
