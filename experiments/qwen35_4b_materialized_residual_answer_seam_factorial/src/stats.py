"""Deterministic report-only paired inference for the mechanics pilot."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import Any


def _paired_values(
    treatment: Sequence[bool], control: Sequence[bool]
) -> tuple[tuple[int, int], ...]:
    if not treatment or len(treatment) != len(control):
        raise ValueError("paired inference requires equal nonempty arms")
    if any(not isinstance(value, bool) for value in (*treatment, *control)):
        raise ValueError("paired inference requires boolean outcomes")
    return tuple(zip(map(int, treatment), map(int, control), strict=True))


def one_sided_mcnemar(treatment_only: int, control_only: int) -> float:
    """Exact one-sided P(T-only >= observed | discordance, p=.5)."""
    if (
        not isinstance(treatment_only, int)
        or isinstance(treatment_only, bool)
        or not isinstance(control_only, int)
        or isinstance(control_only, bool)
        or min(treatment_only, control_only) < 0
    ):
        raise ValueError("McNemar discordant counts must be nonnegative integers")
    discordant = treatment_only + control_only
    if discordant == 0:
        return 1.0
    numerator = sum(
        math.comb(discordant, successes)
        for successes in range(treatment_only, discordant + 1)
    )
    return numerator / (2**discordant)


def paired_bootstrap_interval(
    treatment: Sequence[bool],
    control: Sequence[bool],
    *,
    seed: int,
    resamples: int = 10_000,
) -> list[float]:
    pairs = _paired_values(treatment, control)
    if not isinstance(resamples, int) or isinstance(resamples, bool) or resamples < 100:
        raise ValueError("paired bootstrap requires at least 100 resamples")
    generator = random.Random(seed)
    size = len(pairs)
    effects = []
    for _ in range(resamples):
        difference = 0
        for _ in range(size):
            treated, controlled = pairs[generator.randrange(size)]
            difference += treated - controlled
        effects.append(difference / size)
    effects.sort()
    lower = effects[math.floor(0.025 * (resamples - 1))]
    upper = effects[math.ceil(0.975 * (resamples - 1))]
    return [lower, upper]


def paired_report(
    treatment: Sequence[bool],
    control: Sequence[bool],
    *,
    seed: int,
    resamples: int = 10_000,
) -> dict[str, Any]:
    pairs = _paired_values(treatment, control)
    treatment_only = sum(treated == 1 and controlled == 0 for treated, controlled in pairs)
    control_only = sum(treated == 0 and controlled == 1 for treated, controlled in pairs)
    return {
        "rows": len(pairs),
        "treatment_successes": sum(treated for treated, _ in pairs),
        "control_successes": sum(controlled for _, controlled in pairs),
        "effect": sum(treated - controlled for treated, controlled in pairs)
        / len(pairs),
        "treatment_only": treatment_only,
        "control_only": control_only,
        "discordant": treatment_only + control_only,
        "one_sided_exact_mcnemar_p": one_sided_mcnemar(
            treatment_only, control_only
        ),
        "paired_bootstrap_95_interval": paired_bootstrap_interval(
            treatment,
            control,
            seed=seed,
            resamples=resamples,
        ),
        "bootstrap_seed": seed,
        "bootstrap_resamples": resamples,
        "report_only": True,
    }
