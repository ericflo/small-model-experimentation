#!/usr/bin/env python3
"""Transfer measurement for the exec-trace installation, via the SHARED harness.

Runs the shared coding-fitness harness
(``experiments/qwen35_4b_coding_fitness_harness/scripts/eval_pass1.py`` —
referenced, NOT copied) for BOTH arms (``base`` and the ``exec_trace``
composite) on BOTH target datasets (HumanEval 164 + MBPP 200), greedy pass@1,
identical path, then reads the FROZEN two-directional consequence.

Frozen consequence (also unit-tested over its truth table incl. the
retention_fail branch):

- INSTALLED_CODING iff the exec_trace composite's pass@1 STRICTLY exceeds base
  on at least one target dataset (HumanEval OR MBPP) AND does NOT regress the
  other by more than the retention tolerance (>= base - 0.02).
- RETENTION_FAIL iff it regresses EITHER dataset by more than 0.02 (the
  forgetting risk realized). This takes priority: a regression past tolerance
  can never be an install.
- NULL iff no strict improvement and no regression past tolerance.

All four numbers (base/treatment x HE/MBPP), the per-problem paired deltas
(McNemar b/c: only-base-passes / only-treatment-passes), and the per-clause
booleans are recorded. The agentic duet-eval is a follow-on confirm, not gated
here.
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
MERGED = ROOT / "large_artifacts" / EXP.name / "merged" / "exec_trace"

# Frozen measurement grid.
DATASETS = (("humaneval", 164), ("mbpp", 200))
RETENTION_TOL = 0.02
IMPROVE_EPS = 1e-9

INSTALLED_CLAIM = (
    "exec-trace tracing installs real coding capability: the exec_trace "
    "composite strictly beats base on at least one held-out target dataset "
    "without regressing the other past tolerance; the mental-interpreter "
    "curriculum transfers, and exec_trace becomes the program reference for "
    "the agentic confirm."
)
RETENTION_FAIL_CLAIM = (
    "the forgetting risk is realized: trace-only SFT regressed a target "
    "dataset past 0.02; the model shifted toward tracing over generating. "
    "Re-run with --mix-retention before any agentic confirm."
)
NULL_CLAIM = (
    "the exec-trace dose neither installs (no strict pass@1 gain) nor forgets "
    "(no regression past tolerance) at this dose; a preserved boundary finding "
    "that funds a larger/redesigned dose, not a re-roll."
)


def regressed(delta: float, tol: float = RETENTION_TOL) -> bool:
    """The drop STRICTLY exceeds the tolerance (>= base - tol passes at equality)."""
    return (-delta) > tol + IMPROVE_EPS


def improved(delta: float) -> bool:
    return delta > IMPROVE_EPS


def consequence_reading(
    base_he: float, base_mbpp: float, treat_he: float, treat_mbpp: float, *, tol: float = RETENTION_TOL
) -> dict:
    he_delta = treat_he - base_he
    mbpp_delta = treat_mbpp - base_mbpp
    reg_he = regressed(he_delta, tol)
    reg_mbpp = regressed(mbpp_delta, tol)
    imp_he = improved(he_delta)
    imp_mbpp = improved(mbpp_delta)
    any_regressed = reg_he or reg_mbpp
    any_improved = imp_he or imp_mbpp
    if any_regressed:
        verdict, claim = "RETENTION_FAIL", RETENTION_FAIL_CLAIM
    elif any_improved:
        verdict, claim = "INSTALLED_CODING", INSTALLED_CLAIM
    else:
        verdict, claim = "NULL", NULL_CLAIM
    return {
        "verdict": verdict,
        "frozen_claim": claim,
        "retention_tolerance": tol,
        "base": {"humaneval": base_he, "mbpp": base_mbpp},
        "treatment": {"humaneval": treat_he, "mbpp": treat_mbpp},
        "delta": {"humaneval": he_delta, "mbpp": mbpp_delta},
        "humaneval_improved": imp_he,
        "mbpp_improved": imp_mbpp,
        "humaneval_regressed_past_tol": reg_he,
        "mbpp_regressed_past_tol": reg_mbpp,
        "humaneval_retention_holds": not reg_he,
        "mbpp_retention_holds": not reg_mbpp,
        "any_improved": any_improved,
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
        raise SystemExit(f"exec_trace composite is incomplete: {MERGED}")


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
    results: dict[str, dict[str, dict]] = {"base": {}, "exec_trace": {}}
    for arm, override in (("base", None), ("exec_trace", str(MERGED.resolve()))):
        for dataset, n in DATASETS:
            out = raw_dir / f"{arm}_{dataset}.json"
            results[arm][dataset] = run_eval(dataset, n, override, out)

    base_he = results["base"]["humaneval"]["pass_at_1"]
    base_mbpp = results["base"]["mbpp"]["pass_at_1"]
    treat_he = results["exec_trace"]["humaneval"]["pass_at_1"]
    treat_mbpp = results["exec_trace"]["mbpp"]["pass_at_1"]
    consequence = consequence_reading(base_he, base_mbpp, treat_he, treat_mbpp)
    summary = {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "harness": str(HARNESS.resolve()),
        "composite": str(MERGED.resolve()),
        "datasets": {name: n for name, n in DATASETS},
        "pass_at_1": {
            "base": {"humaneval": base_he, "mbpp": base_mbpp},
            "exec_trace": {"humaneval": treat_he, "mbpp": treat_mbpp},
        },
        "paired": {
            "humaneval": paired_deltas(
                results["base"]["humaneval"]["per_problem"], results["exec_trace"]["humaneval"]["per_problem"]
            ),
            "mbpp": paired_deltas(
                results["base"]["mbpp"]["per_problem"], results["exec_trace"]["mbpp"]["per_problem"]
            ),
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
