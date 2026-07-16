#!/usr/bin/env python3
"""Consolidate the honest all-events sweep rate from the committed table.

Consumes `runs/readings_table.json` (from collect_readings.py) and writes
`runs/sweep_rate_analysis.json`:

1. Sweep rate over ALL SIX all-time goal-gate readings (not the favorable
   four-seed window), with an exact Clopper-Pearson 95% CI and a
   Beta(1,1)-prior posterior summary (mean + equal-tailed 95% credible
   interval). All beta quantiles are computed exactly for integer shape
   parameters by inverting the binomial-tail identity
   I_p(a, b) = P(Binomial(a+b-1, p) >= a) with deterministic bisection —
   stdlib only, byte-identical across runs.
2. Blocker table: which families blocked each missed gate (ties + losses),
   frequency per family, and the zero-strict-losses fact.
3. Base per-family draw distribution across the six seeds (min / median /
   max, and on how many seeds base drew above zero) — computed, not
   assumed.
4. An ERRATUM block: the informal "~50% sweep rate" / "two of four sealed
   seeds" figure was computed over a favorable window (78154-78157) that
   omitted the 78150 miss; the zero-root docs then extended it to a "fifth
   data point" (2/5) while still calling it ~50%. The honest all-events
   figure is 2/6. The block lists every document that carried the informal
   figure.

`--verify` re-derives the analysis from the committed table and requires
it to be byte-identical to the committed `runs/sweep_rate_analysis.json`.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
TABLE = EXP / "runs" / "readings_table.json"
OUT = EXP / "runs" / "sweep_rate_analysis.json"

FAMILIES = (
    "chronicle",
    "lockpick",
    "menders",
    "mirage",
    "rites",
    "siftstack",
    "sirens",
    "stockade",
    "toolsmith",
    "warren",
)

ROUND = 6

# Every document that carried the informal favorable-window figure, with
# the phrasing it used. The correction target list for the erratum.
INFORMAL_FIGURE_CARRIERS = (
    {
        "document": "experiments/qwen35_4b_goal_gate_confirmation/README.md",
        "phrasing": "two full sweeps across four independent sealed seeds (status line and interpretation)",
    },
    {
        "document": "experiments/qwen35_4b_goal_gate_confirmation/reports/report.md",
        "phrasing": "two full sweeps across four independent sealed seeds",
    },
    {
        "document": "knowledge/synthesis.md",
        "phrasing": (
            "confirmation entry: 'two full 10/10s across four independent sealed seeds'; "
            "dose-scale entry: 'two 10/10 sweeps across four independent sealed seeds'; "
            "zero-root entry: 'the fifth consistent data point on the ~50% sweep rate'"
        ),
    },
    {
        "document": "knowledge/experiment_brief.json",
        "phrasing": (
            "qwen35_4b_menders_dose_scale and qwen35_4b_repair_verifier_signal_probe briefs: "
            "'the full ten-family sweep demonstrated on two of four sealed seeds'"
        ),
    },
    {
        "document": "knowledge/experiment_viz.json",
        "phrasing": "zero-root note: 'the fifth data point on the ~50% sweep rate'",
    },
    {
        "document": "experiments/qwen35_4b_zero_root_lineage_rebuild/experiment_log.md",
        "phrasing": "the fifth data point on the ~50% sweep rate",
    },
    {
        "document": "experiments/qwen35_4b_zero_root_lineage_rebuild/README.md",
        "phrasing": "the original read 9/10 on this seed (menders tie — the sweep rate holding)",
    },
    {
        "document": "experiments/qwen35_4b_zero_root_lineage_rebuild/reports/preregistration.md",
        "phrasing": "The original's two recorded sweeps came from four-seed evidence",
    },
)

INFORMAL_WINDOW_SEEDS = (78154, 78155, 78156, 78157)
FIFTH_POINT_SEEDS = (78154, 78155, 78156, 78157, 78159)


def binomial_tail(n: int, k: int, p: float) -> float:
    """P(Binomial(n, p) >= k), exact."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    return math.fsum(
        math.comb(n, j) * p**j * (1.0 - p) ** (n - j) for j in range(k, n + 1)
    )


def beta_cdf(x: float, a: int, b: int) -> float:
    """Regularized incomplete beta I_x(a, b) for positive INTEGER a, b."""
    if a < 1 or b < 1:
        raise ValueError("integer shape parameters >= 1 required")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return binomial_tail(a + b - 1, a, x)


def beta_inv(q: float, a: int, b: int, iterations: int = 200) -> float:
    """Beta quantile for integer shapes via deterministic bisection."""
    if not 0.0 < q < 1.0:
        raise ValueError("q must be strictly inside (0, 1)")
    lo, hi = 0.0, 1.0
    for _ in range(iterations):
        mid = (lo + hi) / 2.0
        if beta_cdf(mid, a, b) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> dict:
    """Exact two-sided (1 - alpha) binomial CI for k successes of n."""
    if not 0 <= k <= n or n <= 0:
        raise ValueError(f"invalid counts: k={k}, n={n}")
    low = 0.0 if k == 0 else beta_inv(alpha / 2.0, k, n - k + 1)
    high = 1.0 if k == n else beta_inv(1.0 - alpha / 2.0, k + 1, n - k)
    return {"low": round(low, ROUND), "high": round(high, ROUND), "alpha": alpha}


def beta_posterior(k: int, n: int, alpha: float = 0.05) -> dict:
    """Beta(1,1)-prior posterior over the sweep rate after k of n passes."""
    a, b = 1 + k, 1 + (n - k)
    return {
        "prior": "Beta(1,1)",
        "alpha": a,
        "beta": b,
        "mean": round(a / (a + b), ROUND),
        "credible_95": {
            "low": round(beta_inv(alpha / 2.0, a, b), ROUND),
            "high": round(beta_inv(1.0 - alpha / 2.0, a, b), ROUND),
        },
    }


def build_analysis(table: dict) -> dict:
    readings = table["readings"]
    if tuple(table.get("families", ())) != FAMILIES:
        raise SystemExit(
            f"table families {table.get('families')!r} do not match the pinned FAMILIES"
        )
    seeds = [r["seed"] for r in readings]
    if len(set(seeds)) != len(readings):
        raise SystemExit(f"duplicate seeds in the readings table: {seeds}")

    passes = [r for r in readings if r["goal_gate_pass"]]
    misses = [r for r in readings if not r["goal_gate_pass"]]
    k, n = len(passes), len(readings)

    sweep_rate = {
        "events": n,
        "passes": k,
        "rate": round(k / n, ROUND),
        "pass_seeds": sorted(r["seed"] for r in passes),
        "miss_seeds": sorted(r["seed"] for r in misses),
        "clopper_pearson_95": clopper_pearson(k, n),
        "beta_posterior": beta_posterior(k, n),
        "method": (
            "exact Clopper-Pearson via integer-shape beta quantiles "
            "(I_p(a,b) = P(Binomial(a+b-1, p) >= a), deterministic bisection)"
        ),
    }

    per_family_miss_counts = {
        family: sum(1 for r in misses if family in r["blockers"]) for family in FAMILIES
    }
    per_family_miss_counts = {
        family: count for family, count in per_family_miss_counts.items() if count > 0
    }
    total_strict_losses = sum(len(r["losses"]) for r in readings)
    blockers = {
        "miss_events": len(misses),
        "per_family_miss_counts": per_family_miss_counts,
        "per_miss_detail": [
            {
                "seed": r["seed"],
                "strict_wins": r["strict_wins"],
                "blockers": r["blockers"],
                "losses": r["losses"],
            }
            for r in sorted(misses, key=lambda r: r["seed"])
        ],
        "total_strict_losses_across_all_events": total_strict_losses,
        "zero_strict_losses_across_all_events": total_strict_losses == 0,
        "every_miss_includes_a_menders_draw": all(
            "menders" in r["blockers"] for r in misses
        ),
    }

    base_draws = {}
    for family in FAMILIES:
        values_by_seed = {
            str(r["seed"]): r["base_per_family"][family]
            for r in sorted(readings, key=lambda r: r["seed"])
        }
        values = list(values_by_seed.values())
        base_draws[family] = {
            "values_by_seed": values_by_seed,
            "min": round(min(values), ROUND),
            "median": round(statistics.median(values), ROUND),
            "max": round(max(values), ROUND),
            "seeds_gt_zero": sum(1 for v in values if v > 0.0),
        }
    base_draws_note = (
        "computed, not assumed: base drew ZERO on all six seeds for "
        + "/".join(f for f in FAMILIES if base_draws[f]["seeds_gt_zero"] == 0)
        + "; the intake note's guess 'base rites>0 on 2 seeds' is wrong — rites "
        "is 0.0 on all six; CHRONICLE is the family above zero on exactly 2 seeds"
    )

    window_rows = [r for r in readings if r["seed"] in INFORMAL_WINDOW_SEEDS]
    window_passes = sum(1 for r in window_rows if r["goal_gate_pass"])
    fifth_rows = [r for r in readings if r["seed"] in FIFTH_POINT_SEEDS]
    fifth_passes = sum(1 for r in fifth_rows if r["goal_gate_pass"])
    erratum = {
        "informal_claim": (
            "'~50% sweep rate' / 'two full sweeps across four independent sealed "
            "seeds' / 'demonstrated on two of four sealed seeds'"
        ),
        "informal_window": {
            "seeds": sorted(INFORMAL_WINDOW_SEEDS),
            "events": len(window_rows),
            "passes": window_passes,
            "rate": round(window_passes / len(window_rows), ROUND),
        },
        "fifth_data_point_extension": {
            "seeds": sorted(FIFTH_POINT_SEEDS),
            "events": len(fifth_rows),
            "passes": fifth_passes,
            "rate": round(fifth_passes / len(fifth_rows), ROUND),
            "note": (
                "the zero-root docs called seed 78159 'the fifth data point on "
                "the ~50% sweep rate'; even that five-event count reads 2/5 = "
                "0.4, and it still omits the 78150 miss"
            ),
        },
        "omitted_by_window": [
            {
                "seed": r["seed"],
                "outcome": f"miss ({r['strict_wins']}/10; ties {'+'.join(r['ties'])})",
                "source_experiment": r["source_experiment"],
            }
            for r in sorted(readings, key=lambda r: r["seed"])
            if r["seed"] not in INFORMAL_WINDOW_SEEDS
        ],
        "corrected": {
            "seeds": sorted(seeds),
            "events": n,
            "passes": k,
            "rate": round(k / n, ROUND),
            "statement": (
                "the full all-time record holds SIX goal-gate readings of the "
                "hygiene_explore composite vs base at sealed medium/tb1024 "
                "seeds; the honest sweep rate is 2/6 (~0.333), not ~50% — the "
                "informal figure was computed over a favorable four-seed window "
                "that started at the first pass (78154) and omitted the earlier "
                "78150 miss (8/10) and, later, undercounted the 78159 miss"
            ),
        },
        "documents_carrying_informal_figure": [dict(c) for c in INFORMAL_FIGURE_CARRIERS],
        "unaffected_facts": (
            "zero strict losses in all six events; every miss includes a "
            "menders 0-margin draw; the aggregate win is 6/6 all-time"
        ),
    }
    aggregate_wins = sum(1 for r in readings if r["aggregate_delta"] > 0.0)
    if aggregate_wins != n:
        erratum["unaffected_facts"] = (
            f"zero-strict-losses and menders-draw facts as computed; NOTE: the "
            f"aggregate win is {aggregate_wins}/{n}, not {n}/{n}"
        )

    return {
        "schema_version": 1,
        "benchmark_data_read": False,
        "source_table": "runs/readings_table.json",
        "sweep_rate": sweep_rate,
        "blockers": blockers,
        "base_draws": {"families": base_draws, "note": base_draws_note},
        "erratum": erratum,
    }


def serialize(analysis: dict) -> str:
    return json.dumps(analysis, indent=1, ensure_ascii=False, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="re-derive and require byte-identity with the committed analysis",
    )
    args = parser.parse_args()

    table = json.loads(TABLE.read_text(encoding="utf-8"))
    analysis = build_analysis(table)
    payload = serialize(analysis)

    if args.verify:
        if not OUT.exists():
            raise SystemExit(f"--verify: missing committed analysis {OUT}")
        committed = OUT.read_text(encoding="utf-8")
        if committed != payload:
            raise SystemExit(
                f"--verify: regenerated analysis is not byte-identical to {OUT}"
            )
        print(f"VERIFIED: {OUT} re-derives byte-identically from {TABLE}")
        return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(payload, encoding="utf-8")
    rate = analysis["sweep_rate"]
    ci = rate["clopper_pearson_95"]
    post = rate["beta_posterior"]
    print(
        f"sweep rate {rate['passes']}/{rate['events']} = {rate['rate']} "
        f"(CP95 [{ci['low']}, {ci['high']}]; posterior mean {post['mean']}, "
        f"CrI95 [{post['credible_95']['low']}, {post['credible_95']['high']}]) -> {OUT}"
    )
    print(
        f"blockers: {analysis['blockers']['per_family_miss_counts']}; "
        f"strict losses {analysis['blockers']['total_strict_losses_across_all_events']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
