"""Backend-free reference mechanics and statistical utilities."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence, TypeVar


T = TypeVar("T")


def carry_unroll(initial: T, transition: Callable[[T, int], T], k: int) -> list[T]:
    """Apply a transition serially; call t consumes the state from call t-1."""
    if k < 1:
        raise ValueError("k must be positive")
    states = [initial]
    for step in range(2, k + 1):
        states.append(transition(states[-1], step))
    return states


def bag_unroll(initial: T, transition: Callable[[T, int], T], k: int) -> list[T]:
    """Apply the same transition K-1 times from the reset state."""
    if k < 1:
        raise ValueError("k must be positive")
    return [initial, *(transition(initial, step) for step in range(2, k + 1))]


def last_mean_aggregate(states: Sequence[float], last_weight: float) -> float:
    if not states:
        raise ValueError("cannot aggregate an empty state collection")
    if not 0.0 <= last_weight <= 1.0:
        raise ValueError("last_weight must be in [0, 1]")
    mean = sum(states) / len(states)
    return last_weight * states[-1] + (1.0 - last_weight) * mean


@dataclass(frozen=True)
class ComputeReceipt:
    sequence_tokens: int
    total_layers: int
    loop_layers: int
    k: int
    recurrent_layer_token_applications: int
    base_layer_token_applications: int
    total_layer_token_applications: int


def recurrent_compute_receipt(
    *, sequence_tokens: int, total_layers: int, loop_layers: int, k: int
) -> ComputeReceipt:
    if min(sequence_tokens, total_layers, loop_layers, k) < 1:
        raise ValueError("all compute geometry values must be positive")
    if loop_layers >= total_layers:
        raise ValueError("loop block must be a strict subset of model layers")
    base = sequence_tokens * total_layers
    extra = sequence_tokens * loop_layers * (k - 1)
    return ComputeReceipt(sequence_tokens, total_layers, loop_layers, k, extra, base, base + extra)


def paired_bootstrap_interval(
    differences: Sequence[float], *, resamples: int, seed: int, alpha: float = 0.05
) -> tuple[float, float, float]:
    if not differences:
        raise ValueError("paired bootstrap needs at least one difference")
    if resamples < 1000:
        raise ValueError("paired bootstrap needs at least 1,000 resamples")
    rng = random.Random(seed)
    values = list(map(float, differences))
    n = len(values)
    draws = []
    for _ in range(resamples):
        draws.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    draws.sort()
    lower_index = max(0, math.floor((alpha / 2) * resamples))
    upper_index = min(resamples - 1, math.ceil((1 - alpha / 2) * resamples) - 1)
    return sum(values) / n, draws[lower_index], draws[upper_index]


def hierarchical_paired_bootstrap_interval(
    groups: Mapping[str | int, Sequence[float]],
    *,
    resamples: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Resample training seeds, then paired task differences within each seed."""
    if not groups or any(not values for values in groups.values()):
        raise ValueError("hierarchical bootstrap needs non-empty groups")
    if resamples < 1000:
        raise ValueError("hierarchical bootstrap needs at least 1,000 resamples")
    normalized = {str(key): list(map(float, values)) for key, values in groups.items()}
    keys = sorted(normalized)
    observed = sum(sum(normalized[key]) / len(normalized[key]) for key in keys) / len(keys)
    rng = random.Random(seed)
    draws = []
    for _ in range(resamples):
        sampled_group_means = []
        for _ in keys:
            key = keys[rng.randrange(len(keys))]
            values = normalized[key]
            sampled_group_means.append(
                sum(values[rng.randrange(len(values))] for _ in values) / len(values)
            )
        draws.append(sum(sampled_group_means) / len(sampled_group_means))
    draws.sort()
    lower_index = max(0, math.floor((alpha / 2) * resamples))
    upper_index = min(resamples - 1, math.ceil((1 - alpha / 2) * resamples) - 1)
    return observed, draws[lower_index], draws[upper_index]


def gate_reachability(baseline: float, required_gain: float, upper_bound: float = 1.0) -> dict[str, float | bool]:
    if not 0 <= baseline <= upper_bound:
        raise ValueError("baseline must lie inside the metric range")
    reachable = baseline + required_gain <= upper_bound
    return {
        "baseline": baseline,
        "required_gain": required_gain,
        "upper_bound": upper_bound,
        "required_score": baseline + required_gain,
        "reachable": reachable,
    }
