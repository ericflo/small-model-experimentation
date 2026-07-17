#!/usr/bin/env python3
"""Exact power/false-positive arithmetic for the frozen replication rule.

Model-free, analytic, frozen at design time and quoted verbatim in
``reports/preregistration.md``. Everything is computed from binomial
closed forms — no simulation, no randomness — so ``--check`` can verify
the preregistered numbers byte-exactly forever.

NOISE MODEL (the null): every arm-event independently draws >= 1 FULL
menders episode with probability p. The program's observed arm-event
rate over all 29 medium/tb1024 sealed arm-events on record at design
time is 3/29 = 0.1034 for full-episode draws (round(10*score) >= 1);
every such draw in program history was exactly ONE episode (score 0.1),
so the null models each hitting arm-event as contributing exactly one
episode. Two further arm-events drew partial credit (score 0.0167 =
1/60), which counts as a raw hit but contributes ZERO episodes under
the frozen round(10*score) conversion; the raw >0 arm-event rate 5/29 =
0.1724 is therefore carried as a conservative sensitivity bound in
which every raw hit is (pessimistically) promoted to a full episode.

FALSE REPLICATED under the null at p: the candidate's episode total
E_c ~ Binomial(4, p) and each of the three control arms' totals
E_j ~ Binomial(4, p), all independent;

    P(false REPLICATED) = sum_{k>=2} P(E_c = k) * P(E_j < k)^3.

POWER under a real effect: the candidate's per-event hit rate is q
(one episode per hit), the controls stay at the null p;

    P(hits_c >= 2)    = 1 - (1-q)^4 - 4 q (1-q)^3
    P(REPLICATED)     = sum_{k>=2} P(Bin(4,q) = k) * P(Bin(4,p) < k)^3.
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from math import comb


EVENTS = 4  # the four fresh sealed seeds; 78163 is prior evidence, never pooled
CONTROL_ARMS = 3  # base, zero_root_parent, replay_ctl7

# Frozen design-time audit of every medium/tb1024 sealed arm-event on
# record (8 events, 29 arm-events): full-episode menders draws at
# 78157 (hygiene_explore 0.1), 78162 (replay_ctl6 0.1), 78163
# (count_walk 0.1); partial-credit draws at 78154
# (hygiene_explore_parent 0.0167) and 78160 (statechain_clean 0.0167).
OBSERVED_ARM_EVENTS = 29
OBSERVED_FULL_EPISODE_DRAWS = 3
OBSERVED_RAW_POSITIVE_DRAWS = 5

NULL_P = Fraction(1, 10)  # frozen headline null: p = 0.10 ~= 3/29
SENSITIVITY_P = Fraction(OBSERVED_RAW_POSITIVE_DRAWS, OBSERVED_ARM_EVENTS)
EFFECT_RATES = (Fraction(2, 5), Fraction(1, 2), Fraction(13, 20))

# The numbers quoted in reports/preregistration.md (rounded to 4 dp).
PREREGISTERED = {
    "p_hits_ge2_null": 0.0523,
    "p_false_replicated_null": 0.0450,
    "p_false_replicated_sensitivity": 0.0947,
    "power_hits_ge2": {"0.4": 0.5248, "0.5": 0.6875, "0.65": 0.8735},
    "power_replicated": {"0.4": 0.4717, "0.5": 0.6289, "0.65": 0.8230},
}


def binom_pmf(n: int, k: int, p: Fraction) -> Fraction:
    return comb(n, k) * p**k * (1 - p) ** (n - k)


def binom_cdf_below(n: int, k: int, p: Fraction) -> Fraction:
    """P(X < k) for X ~ Binomial(n, p)."""
    return sum(binom_pmf(n, i, p) for i in range(0, k))


def p_hits_ge2(q: Fraction) -> Fraction:
    """P(at least two of the four candidate events hit)."""
    return 1 - binom_pmf(EVENTS, 0, q) - binom_pmf(EVENTS, 1, q)


def p_replicated(q: Fraction, p_control: Fraction) -> Fraction:
    """P(E_c >= 2 AND E_c strictly exceeds every control total).

    One episode per hitting event on both sides, so hits_c = E_c; the
    dominance clause requires E_c > E_j for EVERY control j.
    """
    return sum(
        binom_pmf(EVENTS, k, q)
        * binom_cdf_below(EVENTS, k, p_control) ** CONTROL_ARMS
        for k in range(2, EVENTS + 1)
    )


def computed() -> dict:
    return {
        "p_hits_ge2_null": round(float(p_hits_ge2(NULL_P)), 4),
        "p_false_replicated_null": round(float(p_replicated(NULL_P, NULL_P)), 4),
        "p_false_replicated_sensitivity": round(
            float(p_replicated(SENSITIVITY_P, SENSITIVITY_P)), 4
        ),
        "power_hits_ge2": {
            str(float(q)).rstrip("0").rstrip("."): round(float(p_hits_ge2(q)), 4)
            for q in EFFECT_RATES
        },
        "power_replicated": {
            str(float(q)).rstrip("0").rstrip("."): round(
                float(p_replicated(q, NULL_P)), 4
            )
            for q in EFFECT_RATES
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the preregistered numbers against the exact recomputation",
    )
    args = parser.parse_args()
    values = computed()
    if args.check and values != PREREGISTERED:
        parser.error(
            "preregistered power numbers do not match the exact recomputation: "
            f"computed={json.dumps(values, sort_keys=True)} "
            f"preregistered={json.dumps(PREREGISTERED, sort_keys=True)}"
        )
    print(
        json.dumps(
            {
                "events": EVENTS,
                "control_arms": CONTROL_ARMS,
                "null_p": float(NULL_P),
                "sensitivity_p": round(float(SENSITIVITY_P), 6),
                "observed": {
                    "arm_events": OBSERVED_ARM_EVENTS,
                    "full_episode_draws": OBSERVED_FULL_EPISODE_DRAWS,
                    "raw_positive_draws": OBSERVED_RAW_POSITIVE_DRAWS,
                },
                "values": values,
                "matches_preregistration": values == PREREGISTERED,
            },
            indent=1,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
