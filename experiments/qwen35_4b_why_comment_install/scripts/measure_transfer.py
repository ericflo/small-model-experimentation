#!/usr/bin/env python3
"""Transfer measurement for the WHY-comment installation, via the SHARED harness.

Runs the shared coding-fitness harness
(``experiments/qwen35_4b_coding_fitness_harness/scripts/eval_pass1.py`` —
referenced, NOT copied) for BOTH arms (``base`` and the ``why_comment``
composite) on BOTH target datasets (HumanEval 164 + MBPP 200), greedy pass@1,
identical path, then reads the FROZEN, TIGHTENED two-directional consequence.

Clean-test property (the reason the inline-comment variant went first): the
harness's execution grader IGNORES comments, so training the model to emit
richly-``#WHY:``-commented code is INERT to grading. If pass@1 rises, the CODE
got better, unconfounded by any grader reading the rationale.

Why the rule is tightened (learning from bet #1): the exec-trace bet fired
INSTALLED_CODING by the LETTER on a single-problem HumanEval bump that was
noise near ceiling, while the real agentic target stayed exactly flat. So a
1-problem move must NOT count as installed capability. INSTALLED_CODING now
requires a MEANINGFUL improvement: a pass@1 gain of >= 3 PROBLEMS on HumanEval
OR MBPP, with retention on the other.

Frozen consequence (also unit-tested over its truth table incl. the >=3-problem
threshold and the retention_fail branch):

- RETENTION_FAIL (priority) iff EITHER dataset regresses past the 0.02 retention
  tolerance. A regression past tolerance can never be an install.
- INSTALLED_CODING iff (no regression past tolerance) AND the why_comment
  composite beats base by >= 3 problems on at least one target dataset
  (HumanEval OR MBPP), with the other retained (>= base - 0.02).
- NULL otherwise (no >= 3-problem gain and no regression past tolerance).

All four numbers (base/treatment x HE/MBPP, counts + fractions), the per-problem
paired deltas (McNemar b/c: only-base-passes / only-treatment-passes), and the
per-clause booleans are recorded. The agentic duet-eval (base 8/35) is the
PRIMARY real target but is a MANUAL follow-on confirm (documented in the report),
NOT gated here.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
HARNESS = ROOT / "experiments" / "qwen35_4b_coding_fitness_harness" / "scripts" / "eval_pass1.py"
VLLM_PYTHON = ROOT / ".venv-vllm" / "bin" / "python"
MERGED = ROOT / "large_artifacts" / EXP.name / "merged" / "why_comment"

# Frozen measurement grid.
DATASETS = (("humaneval", 164), ("mbpp", 200))
RETENTION_TOL = 0.02
MIN_GAIN_PROBLEMS = 3
IMPROVE_EPS = 1e-9

INSTALLED_CLAIM = (
    "teaching WHY installs real coding capability: the why_comment composite "
    "beats base by at least 3 problems on a held-out target dataset without "
    "regressing the other past tolerance; annotating each line with its causal "
    "reason (inert to the grader) improved the CODE, and why_comment becomes the "
    "program reference for the agentic confirm and a comment-strip anneal."
)
RETENTION_FAIL_CLAIM = (
    "the forgetting risk is realized: WHY-comment SFT regressed a target dataset "
    "past 0.02. The longer commented targets may have biased generation; "
    "reconsider dose/anneal before any agentic confirm."
)
NULL_CLAIM = (
    "the WHY-comment dose neither meaningfully installs (no >= 3-problem pass@1 "
    "gain) nor forgets (no regression past tolerance) at this dose; teaching WHY "
    "inline reshuffles-without-raising like the passive-skill bets — a preserved "
    "boundary finding that funds the think-block WHY variant (bet #3), not a re-roll."
)


def regressed(frac_delta: float, tol: float = RETENTION_TOL) -> bool:
    """The pass@1 drop STRICTLY exceeds the tolerance (>= base - tol holds at equality)."""
    return (-frac_delta) > tol + IMPROVE_EPS


def meaningful_gain(problem_delta: int, min_gain: int = MIN_GAIN_PROBLEMS) -> bool:
    """A gain of at least ``min_gain`` problems (the tightened install bar)."""
    return problem_delta >= min_gain


def consequence_reading(
    base_he_pass: int, base_he_total: int, base_mbpp_pass: int, base_mbpp_total: int,
    treat_he_pass: int, treat_mbpp_pass: int,
    *, tol: float = RETENTION_TOL, min_gain: int = MIN_GAIN_PROBLEMS,
) -> dict:
    base_he_frac = base_he_pass / base_he_total
    base_mbpp_frac = base_mbpp_pass / base_mbpp_total
    treat_he_frac = treat_he_pass / base_he_total
    treat_mbpp_frac = treat_mbpp_pass / base_mbpp_total
    he_frac_delta = treat_he_frac - base_he_frac
    mbpp_frac_delta = treat_mbpp_frac - base_mbpp_frac
    he_gain = treat_he_pass - base_he_pass
    mbpp_gain = treat_mbpp_pass - base_mbpp_pass

    reg_he = regressed(he_frac_delta, tol)
    reg_mbpp = regressed(mbpp_frac_delta, tol)
    gain_he = meaningful_gain(he_gain, min_gain)
    gain_mbpp = meaningful_gain(mbpp_gain, min_gain)
    any_regressed = reg_he or reg_mbpp
    any_meaningful_gain = gain_he or gain_mbpp

    if any_regressed:
        verdict, claim = "RETENTION_FAIL", RETENTION_FAIL_CLAIM
    elif any_meaningful_gain:
        verdict, claim = "INSTALLED_CODING", INSTALLED_CLAIM
    else:
        verdict, claim = "NULL", NULL_CLAIM
    return {
        "verdict": verdict,
        "frozen_claim": claim,
        "retention_tolerance": tol,
        "min_gain_problems": min_gain,
        "base": {
            "humaneval": {"passed": base_he_pass, "total": base_he_total, "pass_at_1": base_he_frac},
            "mbpp": {"passed": base_mbpp_pass, "total": base_mbpp_total, "pass_at_1": base_mbpp_frac},
        },
        "treatment": {
            "humaneval": {"passed": treat_he_pass, "total": base_he_total, "pass_at_1": treat_he_frac},
            "mbpp": {"passed": treat_mbpp_pass, "total": base_mbpp_total, "pass_at_1": treat_mbpp_frac},
        },
        "problem_delta": {"humaneval": he_gain, "mbpp": mbpp_gain},
        "frac_delta": {"humaneval": he_frac_delta, "mbpp": mbpp_frac_delta},
        "humaneval_meaningful_gain": gain_he,
        "mbpp_meaningful_gain": gain_mbpp,
        "humaneval_regressed_past_tol": reg_he,
        "mbpp_regressed_past_tol": reg_mbpp,
        "humaneval_retention_holds": not reg_he,
        "mbpp_retention_holds": not reg_mbpp,
        "any_meaningful_gain": any_meaningful_gain,
        "any_regressed_past_tol": any_regressed,
        "no_third_state": verdict in ("INSTALLED_CODING", "RETENTION_FAIL", "NULL"),
    }


def paired_deltas(base_per_problem: list[dict], treat_per_problem: list[dict]) -> dict:
    """McNemar b/c counts on the paired per-problem pass booleans."""
    base_by = {str(row["task_id"]): bool(row["passed"]) for row in base_per_problem}
    treat_by = {str(row["task_id"]): bool(row["passed"]) for row in treat_per_problem}
    ids = sorted(set(base_by) & set(treat_by))
    both = only_base = only_treat = neither = 0
    for tid in ids:
        b, t = base_by[tid], treat_by[tid]
        both += int(b and t)
        only_base += int(b and not t)  # McNemar b: base pass, treatment fail (a regression)
        only_treat += int(t and not b)  # McNemar c: treatment pass, base fail (a gain)
        neither += int(not b and not t)
    return {
        "n_paired": len(ids),
        "both_pass": both,
        "only_base_passes": only_base,
        "only_treatment_passes": only_treat,
        "neither_passes": neither,
        "net_treatment_minus_base": only_treat - only_base,
    }


def run_eval(dataset: str, n: int, model_override: str | None, out: Path) -> dict:
    cmd = [
        str(VLLM_PYTHON), "-B", str(HARNESS),
        "--dataset", dataset, "--n", str(n), "--out", str(out),
    ]
    if model_override:
        cmd += ["--model-override", str(model_override)]
    print(f"[measure_transfer] {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True)
    if proc.returncode != 0:
        raise SystemExit(f"[measure_transfer] harness failed (exit {proc.returncode})")
    return json.loads(out.read_text(encoding="utf-8"))


def authenticate_composite() -> None:
    receipt = MERGED / "merge_receipt.json"
    if not (MERGED / "model.safetensors").is_file() or not receipt.is_file():
        raise SystemExit(f"why_comment composite is incomplete: {MERGED}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--run", action="store_true", help="run both arms x both datasets through the shared harness")
    parser.add_argument("--out", type=Path, default=EXP / "runs" / "measure" / "transfer_summary.json")
    args = parser.parse_args()
    if not args.run:
        parser.error("measurement is a GPU stage; pass --run to execute (see run.py --stage measure)")

    authenticate_composite()
    raw_dir = args.out.parent
    raw_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    results: dict[str, dict[str, dict]] = {"base": {}, "why_comment": {}}
    for arm, override in (("base", None), ("why_comment", str(MERGED.resolve()))):
        for dataset, n in DATASETS:
            out = raw_dir / f"{arm}_{dataset}.json"
            results[arm][dataset] = run_eval(dataset, n, override, out)

    base_he = results["base"]["humaneval"]
    base_mbpp = results["base"]["mbpp"]
    treat_he = results["why_comment"]["humaneval"]
    treat_mbpp = results["why_comment"]["mbpp"]
    consequence = consequence_reading(
        base_he["passed"], base_he["total"], base_mbpp["passed"], base_mbpp["total"],
        treat_he["passed"], treat_mbpp["passed"],
    )
    summary = {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "harness": str(HARNESS.resolve()),
        "composite": str(MERGED.resolve()),
        "datasets": {name: n for name, n in DATASETS},
        "pass_at_1": {
            "base": {"humaneval": base_he["pass_at_1"], "mbpp": base_mbpp["pass_at_1"]},
            "why_comment": {"humaneval": treat_he["pass_at_1"], "mbpp": treat_mbpp["pass_at_1"]},
        },
        "passed": {
            "base": {"humaneval": base_he["passed"], "mbpp": base_mbpp["passed"]},
            "why_comment": {"humaneval": treat_he["passed"], "mbpp": treat_mbpp["passed"]},
        },
        "paired": {
            "humaneval": paired_deltas(base_he["per_problem"], treat_he["per_problem"]),
            "mbpp": paired_deltas(base_mbpp["per_problem"], treat_mbpp["per_problem"]),
        },
        "consequence": consequence,
        "wall_seconds": time.perf_counter() - started,
    }
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[measure_transfer] verdict={consequence['verdict']} -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
