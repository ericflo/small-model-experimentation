"""Exact paired inference and model-free compound-gate power simulation."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def one_sided_mcnemar(treatment: Sequence[int], control: Sequence[int]) -> dict[str, Any]:
    if len(treatment) != len(control) or not treatment:
        raise ValueError("paired outcomes must be nonempty and equal length")
    if any(value not in {0, 1} for value in [*treatment, *control]):
        raise ValueError("paired outcomes must be binary")
    b = sum(left == 1 and right == 0 for left, right in zip(treatment, control, strict=True))
    c = sum(left == 0 and right == 1 for left, right in zip(treatment, control, strict=True))
    discordant = b + c
    p_value = (
        1.0
        if discordant == 0
        else sum(math.comb(discordant, value) for value in range(b, discordant + 1))
        / (2**discordant)
    )
    return {"b": b, "c": c, "discordant": discordant, "p_value": p_value}


def holm(p_values: Mapping[str, float], *, alpha: float = 0.05) -> dict[str, Any]:
    if not p_values or not 0 < alpha < 1:
        raise ValueError("Holm requires p-values and alpha in (0,1)")
    ordered = sorted(p_values.items(), key=lambda row: (row[1], row[0]))
    decisions: dict[str, bool] = {name: False for name in p_values}
    rows: list[dict[str, Any]] = []
    stopped = False
    total = len(ordered)
    for index, (name, p_value) in enumerate(ordered):
        if not 0 <= p_value <= 1:
            raise ValueError("p-values must lie in [0,1]")
        threshold = alpha / (total - index)
        rejected = not stopped and p_value <= threshold
        if not rejected:
            stopped = True
        decisions[name] = rejected
        rows.append(
            {
                "comparator": name,
                "p_value": p_value,
                "threshold": threshold,
                "rejected": rejected,
            }
        )
    return {"alpha": alpha, "ordered": rows, "decisions": decisions}


def stratified_bootstrap_lower(
    treatment: Sequence[int],
    control: Sequence[int],
    blocks: Sequence[int],
    *,
    seed: int,
    resamples: int,
) -> float:
    if len(treatment) != len(control) or len(treatment) != len(blocks):
        raise ValueError("paired outcomes and blocks must align")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    differences = np.asarray(treatment, dtype=np.float64) - np.asarray(control, dtype=np.float64)
    rng = np.random.default_rng(seed)
    values = np.zeros(resamples, dtype=np.float64)
    for block in sorted(set(blocks)):
        indices = np.flatnonzero(np.asarray(blocks) == block)
        sampled = rng.choice(indices, size=(resamples, len(indices)), replace=True)
        values += differences[sampled].sum(axis=1)
    values /= len(differences)
    values.sort()
    return float(values[math.floor(0.025 * resamples)])


def compound_confirmation_pass(
    treatment: Sequence[int],
    controls: Mapping[str, Sequence[int]],
    blocks: Sequence[int],
    *,
    point_margin: float,
    alpha: float,
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> bool:
    effects = {
        name: (sum(treatment) - sum(control)) / len(treatment)
        for name, control in controls.items()
    }
    tests = {name: one_sided_mcnemar(treatment, control) for name, control in controls.items()}
    adjusted = holm({name: value["p_value"] for name, value in tests.items()}, alpha=alpha)
    lowers = {
        name: stratified_bootstrap_lower(
            treatment,
            control,
            blocks,
            seed=bootstrap_seed + index,
            resamples=bootstrap_resamples,
        )
        for index, (name, control) in enumerate(sorted(controls.items()))
    }
    return all(
        effects[name] >= point_margin
        and adjusted["decisions"][name]
        and lowers[name] > 0
        for name in controls
    )


def simulate_compound_power(config: dict[str, Any], *, seed: int) -> dict[str, Any]:
    trials = int(config["trials"])
    resamples = int(config["bootstrap_resamples_per_trial"])
    treatment_rate = float(config["treatment_accuracy"])
    control_rate = float(config["comparator_accuracy"])
    comparator_count = int(config["comparator_count"])
    rng = random.Random(seed)
    blocks = [index // 24 for index in range(192)]
    passes = 0
    for trial in range(trials):
        treatment = [int(rng.random() < treatment_rate) for _ in range(192)]
        controls = {
            f"control-{index}": [int(rng.random() < control_rate) for _ in range(192)]
            for index in range(comparator_count)
        }
        passes += compound_confirmation_pass(
            treatment,
            controls,
            blocks,
            point_margin=0.10,
            alpha=0.05,
            bootstrap_seed=seed + trial * comparator_count,
            bootstrap_resamples=resamples,
        )
    rate = passes / trials
    standard_error = math.sqrt(rate * (1 - rate) / trials)
    return {
        "trials": trials,
        "bootstrap_resamples_per_trial": resamples,
        "treatment_accuracy": treatment_rate,
        "comparator_accuracy": control_rate,
        "comparators": comparator_count,
        "compound_passes": passes,
        "compound_pass_rate": rate,
        "monte_carlo_standard_error": standard_error,
        "assumption": "shared treatment Bernoulli; independent comparator Bernoulli outcomes",
    }
