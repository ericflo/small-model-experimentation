"""Dependency-light statistics for preregistered gates and evaluation."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def roc_auc(labels: Sequence[bool | int], scores: Sequence[float]) -> float | None:
    """Binary AUROC via pairwise wins, with half credit for ties."""
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have equal length")
    positive = [float(score) for label, score in zip(labels, scores) if bool(label)]
    negative = [float(score) for label, score in zip(labels, scores) if not bool(label)]
    if not positive or not negative:
        return None
    wins = 0.0
    for p_score in positive:
        for n_score in negative:
            if p_score > n_score:
                wins += 1.0
            elif p_score == n_score:
                wins += 0.5
    return wins / (len(positive) * len(negative))


def kendall_tau_b(left: Sequence[float], right: Sequence[float]) -> float | None:
    """Kendall tau-b with explicit tie correction."""
    if len(left) != len(right):
        raise ValueError("rank vectors must have equal length")
    concordant = discordant = ties_left = ties_right = 0
    for first in range(len(left)):
        for second in range(first + 1, len(left)):
            delta_left = left[first] - left[second]
            delta_right = right[first] - right[second]
            if delta_left == 0 and delta_right == 0:
                continue
            if delta_left == 0:
                ties_left += 1
            elif delta_right == 0:
                ties_right += 1
            elif delta_left * delta_right > 0:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + ties_left)
        * (concordant + discordant + ties_right)
    )
    if denominator == 0:
        return None
    return (concordant - discordant) / denominator


def paired_bootstrap(
    pairs: Mapping[str, tuple[float, float]],
    *,
    resamples: int,
    seed: int,
) -> dict[str, float | int]:
    task_ids = sorted(pairs)
    if not task_ids:
        raise ValueError("paired bootstrap requires at least one task")
    differences = [float(pairs[task][0]) - float(pairs[task][1]) for task in task_ids]
    rng = random.Random(seed)
    bootstraps = []
    for _ in range(resamples):
        bootstraps.append(
            sum(differences[rng.randrange(len(differences))] for _ in differences)
            / len(differences)
        )
    return {
        "n_tasks": len(task_ids),
        "mean_delta": mean(differences),
        "ci95_low": quantile(bootstraps, 0.025),
        "ci95_high": quantile(bootstraps, 0.975),
        "resamples": resamples,
        "seed": seed,
    }


def accuracy_interval(scores: Sequence[float], *, z: float = 1.96) -> dict[str, float | int]:
    """Wilson interval for binary scores."""
    n = len(scores)
    if n == 0:
        return {"n": 0, "accuracy": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    p = mean([float(value) for value in scores])
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denominator
    return {"n": n, "accuracy": p, "ci95_low": center - radius, "ci95_high": center + radius}


def macro_by(rows: Iterable[Mapping[str, Any]], *, group: str, value: str) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group])].append(float(row[value]))
    return mean([mean(values) for values in grouped.values()])
