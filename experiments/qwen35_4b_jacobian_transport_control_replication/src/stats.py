"""Frozen small-sample statistics for the confirmation gate."""

from __future__ import annotations

import random
from typing import Any


def paired_bootstrap_mean_ci(
    differences: list[float],
    *,
    resamples: int,
    seed: int,
    alpha: float = 0.05,
) -> dict[str, Any]:
    if not differences:
        raise ValueError("paired bootstrap requires at least one difference")
    if resamples < 100 or not 0.0 < alpha < 1.0:
        raise ValueError("invalid bootstrap configuration")
    rng = random.Random(seed)
    n = len(differences)
    values = []
    for _ in range(resamples):
        values.append(sum(differences[rng.randrange(n)] for _ in range(n)) / n)
    values.sort()
    lower_index = max(0, int((alpha / 2) * resamples))
    upper_index = min(resamples - 1, int((1 - alpha / 2) * resamples) - 1)
    return {
        "n": n,
        "mean": sum(differences) / n,
        "lower": values[lower_index],
        "upper": values[upper_index],
        "confidence": 1.0 - alpha,
        "resamples": resamples,
        "seed": seed,
    }
