"""Statistics used by the frozen causal gates."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterable


def binary_auc(labels: list[int], scores: list[float]) -> float:
    """Tie-aware Mann-Whitney AUROC without an sklearn dependency."""
    if len(labels) != len(scores) or not labels:
        raise ValueError("labels and scores must be equally sized and nonempty")
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("AUROC requires both classes")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    rank_sum = 0.0
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and scores[order[end]] == scores[order[cursor]]:
            end += 1
        average_rank = ((cursor + 1) + end) / 2.0
        rank_sum += average_rank * sum(labels[order[i]] for i in range(cursor, end))
        cursor = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def task_macro_auc(rows: Iterable[dict], *, label_key: str, score_key: str, task_key: str = "task_id") -> tuple[float, int]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row[task_key])].append(row)
    values = []
    for task_rows in grouped.values():
        labels = [int(row[label_key]) for row in task_rows]
        if len(set(labels)) != 2:
            continue
        values.append(binary_auc(labels, [float(row[score_key]) for row in task_rows]))
    if not values:
        raise ValueError("no mixed-label tasks")
    return sum(values) / len(values), len(values)


def paired_bootstrap_difference(
    left: list[float],
    right: list[float],
    *,
    resamples: int,
    seed: int,
) -> dict[str, float]:
    if len(left) != len(right) or not left:
        raise ValueError("paired samples must be equally sized and nonempty")
    differences = [a - b for a, b in zip(left, right, strict=True)]
    rng = random.Random(seed)
    boot = []
    for _ in range(resamples):
        boot.append(sum(rng.choice(differences) for _ in differences) / len(differences))
    boot.sort()

    def quantile(q: float) -> float:
        index = min(len(boot) - 1, max(0, math.floor(q * (len(boot) - 1))))
        return boot[index]

    return {
        "mean": sum(differences) / len(differences),
        "ci_low": quantile(0.025),
        "ci_high": quantile(0.975),
        "n": float(len(differences)),
    }
