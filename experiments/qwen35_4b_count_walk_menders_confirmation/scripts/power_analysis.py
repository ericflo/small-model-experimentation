#!/usr/bin/env python3
"""Exact power/false-positive arithmetic for the frozen replication rule.

Model-free, analytic, frozen at design time (amended pre-event by the
adversarial review — see reports/preregistration.md, Review amendments)
and quoted verbatim in ``reports/preregistration.md``. Everything is
computed from binomial closed forms — no simulation, no randomness — so
``--check`` can verify the preregistered numbers byte-exactly forever.

NOISE MODEL (the null) — the FULL-EPISODE process: every arm-event
independently draws >= 1 FULL menders episode with probability p. The
frozen rule counts an event as a hit ONLY if it contains at least one
full episode (score contributes int(10*s + 1e-9) episodes, floor
semantics), so the null IS the priced process — the rule and the
pricing coincide exactly. The program's design-time audit covers all 9
recorded medium/tb1024 sealed events (seeds 78,150 / 78,154 / 78,155 /
78,156 / 78,157 / 78,159 / 78,160 / 78,162 / 78,163; 29 arm-events):
3 arm-events drew a full episode (3/29 = 0.1034), each exactly ONE
episode (score 0.1), so the null models each hitting arm-event as
contributing exactly one episode. Two further arm-events drew partial
credit (score 0.0167 = 1/60); under the frozen floor conversion these
are RULE-INVISIBLE — recorded-only raw positives, neither hits nor
episodes. The headline null is p = 0.10; the exact observed rate
p = 3/29 is also priced below. The raw >0 rate 5/29 = 0.1724 is carried
ONLY as an explicit COUNTERFACTUAL sensitivity ceiling — the alpha the
rule WOULD have if every partial draw were (counterfactually) promoted
to a full episode, which the frozen conversion forbids.

FALSE REPLICATED under the null at p: the candidate's episode total
E_c ~ Binomial(4, p) and each of the three control arms' totals
E_j ~ Binomial(4, p), all independent;

    P(false REPLICATED) = sum_{k>=2} P(E_c = k) * P(E_j < k)^3.

POWER under a real effect: the candidate's per-event FULL-EPISODE hit
rate is q (one episode per hit), the controls stay at the headline
null p = 0.10;

    P(hits_c >= 2)    = 1 - (1-q)^4 - 4 q (1-q)^3
    P(REPLICATED)     = sum_{k>=2} P(Bin(4,q) = k) * P(Bin(4,p) < k)^3.

Also priced: P(NOT_REPLICATED) = (1-q)^4 at q = 0.3 — the probability
the four-seed test closes the line even though a modest real effect
exists (0.2401), quoted in the frozen NOT_REPLICATED consequence.
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from math import comb


EVENTS = 4  # the four fresh sealed seeds; 78163 is prior evidence, never pooled
CONTROL_ARMS = 3  # base, zero_root_parent, replay_ctl7

# Frozen design-time audit of every medium/tb1024 sealed arm-event on
# record (9 events — seeds 78150/78154/78155/78156/78157/78159/78160/
# 78162/78163 — totalling 29 arm-events): full-episode menders draws at
# 78157 (hygiene_explore 0.1), 78162 (replay_ctl6 0.1), 78163
# (count_walk 0.1); partial-credit draws at 78154
# (hygiene_explore_parent 0.0167) and 78160 (statechain_clean 0.0167),
# both RULE-INVISIBLE under the frozen floor conversion (recorded-only).
OBSERVED_SEALED_EVENTS = 9
OBSERVED_ARM_EVENTS = 29
OBSERVED_FULL_EPISODE_DRAWS = 3
OBSERVED_RAW_POSITIVE_DRAWS = 5  # 3 full + 2 partial (recorded-only)

NULL_P = Fraction(1, 10)  # frozen headline null: p = 0.10 ~= 3/29
EXACT_NULL_P = Fraction(OBSERVED_FULL_EPISODE_DRAWS, OBSERVED_ARM_EVENTS)  # 3/29
COUNTERFACTUAL_P = Fraction(OBSERVED_RAW_POSITIVE_DRAWS, OBSERVED_ARM_EVENTS)  # 5/29
EFFECT_RATES = (Fraction(2, 5), Fraction(1, 2), Fraction(13, 20))
NOT_REPLICATED_EFFECT = Fraction(3, 10)  # the q = 0.3 closure-risk quote

# The numbers quoted in reports/preregistration.md (rounded to 4 dp).
PREREGISTERED = {
    "p_hits_ge2_null": 0.0523,
    "p_hits_ge2_null_exact": 0.0557,
    "p_false_replicated_null": 0.0450,
    "p_false_replicated_null_exact": 0.0475,
    "p_false_replicated_counterfactual": 0.0947,
    "p_not_replicated_q30": 0.2401,
    "power_hits_ge2": {"0.4": 0.5248, "0.5": 0.6875, "0.65": 0.8735},
    "power_replicated": {"0.4": 0.4717, "0.5": 0.6289, "0.65": 0.8230},
}
# The exact fraction-derived alpha at the exact observed null p = 3/29,
# printed in full and enforced by --check (the 4 dp headline above rounds
# from exactly this value).
PREREGISTERED_EXACT = {
    "p": "3/29",
    "p_false_replicated_fraction": (
        "11885589964581732052992/250246473680347348787521"
    ),
    "p_false_replicated_float": 0.04749553426180864,
}


def binom_pmf(n: int, k: int, p: Fraction) -> Fraction:
    return comb(n, k) * p**k * (1 - p) ** (n - k)


def binom_cdf_below(n: int, k: int, p: Fraction) -> Fraction:
    """P(X < k) for X ~ Binomial(n, p)."""
    return sum(binom_pmf(n, i, p) for i in range(0, k))


def p_hits_ge2(q: Fraction) -> Fraction:
    """P(at least two of the four candidate events hit a full episode)."""
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


def p_not_replicated(q: Fraction) -> Fraction:
    """P(hits_c == 0) under a real per-event full-episode hit rate q."""
    return binom_pmf(EVENTS, 0, q)


def computed() -> dict:
    return {
        "p_hits_ge2_null": round(float(p_hits_ge2(NULL_P)), 4),
        "p_hits_ge2_null_exact": round(float(p_hits_ge2(EXACT_NULL_P)), 4),
        "p_false_replicated_null": round(float(p_replicated(NULL_P, NULL_P)), 4),
        "p_false_replicated_null_exact": round(
            float(p_replicated(EXACT_NULL_P, EXACT_NULL_P)), 4
        ),
        "p_false_replicated_counterfactual": round(
            float(p_replicated(COUNTERFACTUAL_P, COUNTERFACTUAL_P)), 4
        ),
        "p_not_replicated_q30": round(
            float(p_not_replicated(NOT_REPLICATED_EFFECT)), 4
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


def computed_exact() -> dict:
    """The exact fraction-derived alpha at the exact observed null 3/29."""
    exact = p_replicated(EXACT_NULL_P, EXACT_NULL_P)
    return {
        "p": f"{EXACT_NULL_P.numerator}/{EXACT_NULL_P.denominator}",
        "p_false_replicated_fraction": (
            f"{exact.numerator}/{exact.denominator}"
        ),
        "p_false_replicated_float": float(exact),
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
    exact = computed_exact()
    if args.check and (values != PREREGISTERED or exact != PREREGISTERED_EXACT):
        parser.error(
            "preregistered power numbers do not match the exact recomputation: "
            f"computed={json.dumps(values, sort_keys=True)} "
            f"preregistered={json.dumps(PREREGISTERED, sort_keys=True)} "
            f"computed_exact={json.dumps(exact, sort_keys=True)} "
            f"preregistered_exact={json.dumps(PREREGISTERED_EXACT, sort_keys=True)}"
        )
    print(
        json.dumps(
            {
                "events": EVENTS,
                "control_arms": CONTROL_ARMS,
                "null_p": float(NULL_P),
                "exact_null_p": round(float(EXACT_NULL_P), 6),
                "counterfactual_p": round(float(COUNTERFACTUAL_P), 6),
                "observed": {
                    "sealed_events": OBSERVED_SEALED_EVENTS,
                    "arm_events": OBSERVED_ARM_EVENTS,
                    "full_episode_draws": OBSERVED_FULL_EPISODE_DRAWS,
                    "raw_positive_draws": OBSERVED_RAW_POSITIVE_DRAWS,
                    "partial_draws_rule_invisible": True,
                },
                "values": values,
                "exact_null": exact,
                "matches_preregistration": (
                    values == PREREGISTERED and exact == PREREGISTERED_EXACT
                ),
            },
            indent=1,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
