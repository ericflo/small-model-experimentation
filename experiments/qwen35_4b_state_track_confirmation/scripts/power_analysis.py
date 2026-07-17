#!/usr/bin/env python3
"""Exact/quadrature power and false-positive arithmetic for the paired rule.

Frozen at design time and quoted verbatim in
``reports/preregistration.md``. ``--check`` recomputes every printed
number and enforces it (exact for the closed-form pieces; a tight
tolerance for the deterministic-quadrature joints).

NOISE MODEL. The six per-event paired deltas
``d_i = state_track_aggregate - count_walk_aggregate`` are modelled as
i.i.d. ``Normal(mu, sigma_d)``. Pairing on the SAME seed cancels the
large common per-seed benchmark-difficulty variance, so ``sigma_d`` is
much smaller than the marginal across-seed aggregate SD.

  sigma_d ARITHMETIC. The marginal single-arm aggregate SD across sealed
  seeds is ~0.03 (the parent drew 0.3004 at 78169 and 0.3626 at 78168).
  For two arms measured on the SAME seed with correlation ``rho``,
  Var(d) = 2 * sigma_arm^2 * (1 - rho), i.e.
  sigma_d = sigma_arm * sqrt(2 * (1 - rho)). A moderate-to-high positive
  correlation (the two arms share the seed's item difficulty) gives:
    rho = 0.778 -> sigma_d = 0.020
    rho = 0.653 -> sigma_d = 0.025  (HEADLINE)
    rho = 0.500 -> sigma_d = 0.030
  The headline is sigma_d = 0.025; the range 0.02-0.03 is priced too.

THE RULE (over the six new events; 78169 is prior evidence, never
pooled). wins = #{i: d_i > 0}; mean_d = mean of the six d_i. CONFIRMED
iff mean_d > 0 AND wins >= 4; NOT_CONFIRMED iff mean_d <= 0 (dominates);
AMBIGUOUS iff mean_d > 0 AND wins < 4.

(a) FALSE CONFIRMED under the null mu = 0. Because the deltas are
symmetric about 0, wins ~ Binomial(6, 0.5) so
P(wins >= 4) = (15 + 6 + 1)/64 = 22/64 = 0.34375 exactly, and
P(mean_d > 0) = 0.5 exactly. wins and mean_d are POSITIVELY correlated
(more positive deltas -> larger mean), so the joint
P(mean_d > 0 AND wins >= 4) lies strictly between the independence
product 0.17188 and the marginal 0.34375. Under mu = 0 the joint is
SCALE-FREE (it does not depend on sigma_d): both "mean > 0" and each
"d_i > 0" are scale-invariant events of six i.i.d. symmetric normals. It
is computed by deterministic numerical convolution of the sign-split
sub-densities of the six deltas (no simulation, no RNG) — the frozen
false-positive rate the paired rule carries. This is a fairly LIBERAL
directional replication check, NOT a stringent test; the preregistration
states this plainly.

(b) POWER under the observed effect mu = +0.0256, at each sigma_d.
wins ~ Binomial(6, p) with p = Phi(mu/sigma_d); mean_d ~
Normal(mu, sigma_d^2/6) so P(mean_d > 0) = Phi(sqrt(6)*mu/sigma_d).
P(CONFIRMED) = P(mean_d > 0 AND wins >= 4) is the same convolution joint
evaluated at the shifted mean. P(NOT_CONFIRMED) = P(mean_d <= 0) =
Phi(-sqrt(6)*mu/sigma_d) is the closure risk quoted in the frozen
NOT_CONFIRMED consequence.
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from math import comb, erf, sqrt


EVENTS = 6  # the six fresh sealed seeds; 78169 is prior evidence, never pooled
WINS_THRESHOLD = 4  # ceil(2 * EVENTS / 3)

MU_EFFECT = 0.0256  # the observed 78169 paired lift (state_track - count_walk)
SIGMA_ARM = 0.03  # marginal single-arm across-seed aggregate SD
SIGMA_D_HEADLINE = 0.025
SIGMA_D_RANGE = (0.020, 0.025, 0.030)
# rho implied by sigma_d = sigma_arm * sqrt(2*(1-rho)); printed for the record.
RHO_BY_SIGMA_D = {
    0.020: round(1 - (0.020 / SIGMA_ARM) ** 2 / 2, 4),
    0.025: round(1 - (0.025 / SIGMA_ARM) ** 2 / 2, 4),
    0.030: round(1 - (0.030 / SIGMA_ARM) ** 2 / 2, 4),
}

# Exact null marginals.
P_WINS_GE4_NULL = Fraction(sum(comb(EVENTS, k) for k in range(WINS_THRESHOLD, EVENTS + 1)), 2 ** EVENTS)
P_MEAN_GT0_NULL = Fraction(1, 2)
# Independence-product lower bound and marginal upper bound on the joint.
JOINT_INDEP_BOUND = float(P_WINS_GE4_NULL * P_MEAN_GT0_NULL)
JOINT_MARGINAL_BOUND = float(P_WINS_GE4_NULL)

# Deterministic-quadrature tolerance for --check (grid + libm ulp slack).
CHECK_TOL = 5e-4

# The numbers quoted in reports/preregistration.md (rounded to 4 dp).
PREREGISTERED = {
    "p_wins_ge4_null": 0.3438,
    "p_mean_gt0_null": 0.5,
    "p_false_confirmed_null": 0.311,
    "power_confirmed": {"0.02": 0.9839, "0.025": 0.9494, "0.03": 0.9028},
    "power_wins_ge4": {"0.02": 0.984, "0.025": 0.9502, "0.03": 0.9051},
    "power_mean_gt0": {"0.02": 0.9991, "0.025": 0.9939, "0.03": 0.9817},
    "p_not_confirmed_effect": {"0.02": 0.0009, "0.025": 0.0061, "0.03": 0.0183},
    "p_ambiguous_effect": {"0.02": 0.0152, "0.025": 0.0445, "0.03": 0.0789},
}


def phi(z: float) -> float:
    """Standard normal CDF via the error function."""
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def binom_ge(n: int, k: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p)."""
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def _sign_split_masses(mu: float, sigma: float):
    """Grid, positive-part and negative-part probability masses of one delta.

    Returns (h, x0_index_of_zero, pos_mass, neg_mass): pos_mass and
    neg_mass are numpy arrays of bin masses (pdf * h) with the bin centred
    on 0 split evenly. The grid is symmetric about mu with 0 on it.
    """
    import numpy as np

    half_width = abs(mu) + 12.0 * sigma
    h = sigma / 120.0
    n_side = int(round(half_width / h))
    idx = np.arange(-n_side, n_side + 1)
    x = idx * h  # 0 is exactly grid point idx == 0
    pdf = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * sqrt(2.0 * np.pi))
    mass = pdf * h
    pos = np.where(x > 0, mass, 0.0)
    neg = np.where(x < 0, mass, 0.0)
    zero = idx.tolist().index(0)
    pos[zero] = mass[zero] / 2.0
    neg[zero] = mass[zero] / 2.0
    return h, n_side, pos, neg


def joint_confirmed(mu: float, sigma: float) -> float:
    """P(mean_d > 0 AND wins >= 4) by deterministic convolution quadrature.

    For each event count k in {4, 5, 6}, the sum of the six deltas with
    exactly k positive and (6-k) negative is a convolution of k
    positive-part sub-densities and (6-k) negative-part sub-densities;
    integrating that sum-mass over s > 0 and weighting by C(6, k) gives
    the joint (mean > 0 is exactly sum > 0).
    """
    import numpy as np

    h, n_side, pos, neg = _sign_split_masses(mu, sigma)
    # Single-delta grid runs from -n_side*h; a k-pos/(6-k)-neg convolution
    # of six arrays that each start at -n_side*h starts at -6*n_side*h, so
    # s == 0 is at offset 6*n_side.
    zero_offset = 6 * n_side
    total = 0.0
    for k in range(WINS_THRESHOLD, EVENTS + 1):
        arrays = [pos] * k + [neg] * (EVENTS - k)
        acc = arrays[0]
        for arr in arrays[1:]:
            acc = np.convolve(acc, arr)
        s_index = np.arange(acc.shape[0])
        strictly_positive = float(acc[s_index > zero_offset].sum())
        boundary = float(acc[zero_offset]) * 0.5 if 0 <= zero_offset < acc.shape[0] else 0.0
        total += comb(EVENTS, k) * (strictly_positive + boundary)
    return total


def computed() -> dict:
    p_conf = {}
    p_wins = {}
    p_mean = {}
    p_notconf = {}
    p_amb = {}
    for sigma in SIGMA_D_RANGE:
        key = str(sigma).rstrip("0").rstrip(".")
        p = phi(MU_EFFECT / sigma)
        mean_gt0 = phi(sqrt(EVENTS) * MU_EFFECT / sigma)
        conf = joint_confirmed(MU_EFFECT, sigma)
        p_wins[key] = round(binom_ge(EVENTS, WINS_THRESHOLD, p), 4)
        p_mean[key] = round(mean_gt0, 4)
        p_conf[key] = round(conf, 4)
        p_notconf[key] = round(1.0 - mean_gt0, 4)
        p_amb[key] = round(mean_gt0 - conf, 4)
    return {
        "p_wins_ge4_null": round(float(P_WINS_GE4_NULL), 4),
        "p_mean_gt0_null": round(float(P_MEAN_GT0_NULL), 4),
        "p_false_confirmed_null": round(joint_confirmed(0.0, SIGMA_D_HEADLINE), 4),
        "power_confirmed": p_conf,
        "power_wins_ge4": p_wins,
        "power_mean_gt0": p_mean,
        "p_not_confirmed_effect": p_notconf,
        "p_ambiguous_effect": p_amb,
    }


def _flatten(payload: dict) -> dict:
    flat = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            for inner, number in value.items():
                flat[f"{key}.{inner}"] = number
        else:
            flat[key] = value
    return flat


def _matches(values: dict) -> bool:
    flat_v = _flatten(values)
    flat_p = _flatten(PREREGISTERED)
    if set(flat_v) != set(flat_p):
        return False
    return all(abs(flat_v[key] - flat_p[key]) <= CHECK_TOL for key in flat_p)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the preregistered numbers against the recomputation",
    )
    args = parser.parse_args()
    values = computed()
    if args.check and not _matches(values):
        parser.error(
            "preregistered power numbers do not match the recomputation: "
            f"computed={json.dumps(values, sort_keys=True)} "
            f"preregistered={json.dumps(PREREGISTERED, sort_keys=True)}"
        )
    print(
        json.dumps(
            {
                "events": EVENTS,
                "wins_threshold": WINS_THRESHOLD,
                "mu_effect": MU_EFFECT,
                "sigma_arm": SIGMA_ARM,
                "sigma_d_headline": SIGMA_D_HEADLINE,
                "sigma_d_range": list(SIGMA_D_RANGE),
                "rho_by_sigma_d": {str(k): v for k, v in RHO_BY_SIGMA_D.items()},
                "null_joint_bounds": {
                    "independence_product": round(JOINT_INDEP_BOUND, 5),
                    "marginal_upper": round(JOINT_MARGINAL_BOUND, 5),
                },
                "check_tolerance": CHECK_TOL,
                "values": values,
                "matches_preregistration": _matches(values),
            },
            indent=1,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
