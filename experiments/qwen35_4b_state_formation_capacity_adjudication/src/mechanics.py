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


def crossed_paired_bootstrap_interval(
    groups: Mapping[str | int, Mapping[str, float]],
    *,
    resamples: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Bootstrap a crossed model-seed x task paired-difference matrix.

    The procedural tasks are shared by every trained model seed.  Consequently,
    a task draw must be shared by all seed draws; independently resampling tasks
    inside each seed would incorrectly turn repeated model evaluations into new
    task observations.  This routine also rejects ragged matrices rather than
    silently analyzing intersections.
    """
    if not groups:
        raise ValueError("crossed bootstrap needs at least one model seed")
    if resamples < 1000:
        raise ValueError("crossed bootstrap needs at least 1,000 resamples")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")

    normalized: dict[str, dict[str, float]] = {}
    for seed_key, task_values in groups.items():
        normalized_seed = str(seed_key)
        if normalized_seed in normalized:
            raise ValueError(f"duplicate normalized model seed: {normalized_seed}")
        if not task_values:
            raise ValueError("crossed bootstrap needs at least one task per seed")
        values: dict[str, float] = {}
        for task_key, value in task_values.items():
            normalized_task = str(task_key)
            if normalized_task in values:
                raise ValueError(
                    f"duplicate normalized task id for seed {normalized_seed}: "
                    f"{normalized_task}"
                )
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("crossed bootstrap values must be finite")
            values[normalized_task] = number
        normalized[normalized_seed] = values

    seed_keys = sorted(normalized)
    task_ids = set(normalized[seed_keys[0]])
    for seed_key in seed_keys[1:]:
        current = set(normalized[seed_key])
        if current != task_ids:
            missing = sorted(task_ids - current)
            extra = sorted(current - task_ids)
            raise ValueError(
                "crossed bootstrap requires identical task ids for every model "
                f"seed; seed {seed_key} missing={missing[:5]} extra={extra[:5]}"
            )
    ordered_tasks = sorted(task_ids)
    observed = sum(
        normalized[seed_key][task_id]
        for seed_key in seed_keys
        for task_id in ordered_tasks
    ) / (len(seed_keys) * len(ordered_tasks))

    rng = random.Random(seed)
    draws = []
    for _ in range(resamples):
        sampled_seeds = [
            seed_keys[rng.randrange(len(seed_keys))] for _ in seed_keys
        ]
        # One shared task resample is crossed with every sampled model seed.
        sampled_tasks = [
            ordered_tasks[rng.randrange(len(ordered_tasks))] for _ in ordered_tasks
        ]
        draws.append(
            sum(
                normalized[seed_key][task_id]
                for seed_key in sampled_seeds
                for task_id in sampled_tasks
            )
            / (len(sampled_seeds) * len(sampled_tasks))
        )
    draws.sort()
    lower_index = max(0, math.floor((alpha / 2) * resamples))
    upper_index = min(
        resamples - 1, math.ceil((1 - alpha / 2) * resamples) - 1
    )
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
