#!/usr/bin/env python3
"""Per-rung transfer measurement for the WHY scale ladder, via the SHARED harness.

Runs the shared coding-fitness harness
(``experiments/qwen35_4b_coding_fitness_harness/scripts/eval_pass1.py`` —
referenced, NOT copied) for the BASE and ONE merged rung composite
(``merged/why_scale_<rows>``) on BOTH target datasets (HumanEval 164 + MBPP 200),
greedy pass@1, identical vLLM path, and records the rung's pass@1.

This cell is a SWEEP, not a single-shot install: there is NO frozen
INSTALLED/NULL verdict here. The orchestrator runs this per rung and assembles
the scaling curve (pass@1 vs rows) to locate the PEAK before overfit/collapse. For
each rung it records all four numbers (base/rung x HE/MBPP, counts + fractions),
the per-problem paired McNemar b/c deltas per dataset, and the rung-vs-base deltas
in problems. The clean-test property holds exactly as for the why_comment cell:
the harness's execution grader IGNORES comments, so a pass@1 gain is an
unconfounded CODE improvement. The base numbers are re-measured each rung on the
same path so every rung's delta is against a co-measured base.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
HARNESS = ROOT / "experiments" / "qwen35_4b_coding_fitness_harness" / "scripts" / "eval_pass1.py"
VLLM_PYTHON = ROOT / ".venv-vllm" / "bin" / "python"
MERGED_ROOT = ROOT / "large_artifacts" / EXP.name / "merged"

LADDER_SIZES = (2000, 5000, 10000, 20000, 40000)
DATASETS = (("humaneval", 164), ("mbpp", 200))
# Program anchor (co-measured base baseline; the 504-row why_comment bet moved
# HumanEval +5 -> 0.7927 on this exact harness, the fast gain the ladder scales).
BASE_HUMANEVAL_ANCHOR = 0.7622
BASE_MBPP_ANCHOR = 0.565


def merged_dir(rows: int) -> Path:
    return MERGED_ROOT / f"why_scale_{rows}"


def paired_deltas(base_per_problem: list[dict], treat_per_problem: list[dict]) -> dict:
    base_by = {str(row["task_id"]): bool(row["passed"]) for row in base_per_problem}
    treat_by = {str(row["task_id"]): bool(row["passed"]) for row in treat_per_problem}
    ids = sorted(set(base_by) & set(treat_by))
    both = only_base = only_treat = neither = 0
    for tid in ids:
        b, t = base_by[tid], treat_by[tid]
        both += int(b and t)
        only_base += int(b and not t)   # McNemar b: base pass, rung fail (a regression)
        only_treat += int(t and not b)  # McNemar c: rung pass, base fail (a gain)
        neither += int(not b and not t)
    return {
        "n_paired": len(ids),
        "both_pass": both,
        "only_base_passes": only_base,
        "only_rung_passes": only_treat,
        "neither_passes": neither,
        "net_rung_minus_base": only_treat - only_base,
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


def authenticate_composite(rows: int) -> Path:
    merged = merged_dir(rows)
    receipt = merged / "merge_receipt.json"
    if not (merged / "model.safetensors").is_file() or not receipt.is_file():
        raise SystemExit(f"rung composite is incomplete: {merged}")
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--rows", type=int, choices=LADDER_SIZES, required=True)
    parser.add_argument("--run", action="store_true", help="run base + this rung x both datasets through the shared harness")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if not args.run:
        parser.error("measurement is a GPU stage; pass --run to execute (see run.py --stage measure --rows N)")

    rows = args.rows
    merged = authenticate_composite(rows)
    out_path = args.out or (EXP / "runs" / "measure" / f"rung_{rows}.json")
    raw_dir = out_path.parent
    raw_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    results: dict[str, dict[str, dict]] = {"base": {}, "rung": {}}
    for arm, override in (("base", None), ("rung", str(merged.resolve()))):
        for dataset, n in DATASETS:
            out = raw_dir / f"{arm}_{rows}_{dataset}.json"
            results[arm][dataset] = run_eval(dataset, n, override, out)

    base_he, base_mbpp = results["base"]["humaneval"], results["base"]["mbpp"]
    rung_he, rung_mbpp = results["rung"]["humaneval"], results["rung"]["mbpp"]
    summary = {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "rows": rows,
        "harness": str(HARNESS.resolve()),
        "composite": str(merged.resolve()),
        "datasets": {name: n for name, n in DATASETS},
        "pass_at_1": {
            "base": {"humaneval": base_he["pass_at_1"], "mbpp": base_mbpp["pass_at_1"]},
            "rung": {"humaneval": rung_he["pass_at_1"], "mbpp": rung_mbpp["pass_at_1"]},
        },
        "passed": {
            "base": {"humaneval": base_he["passed"], "mbpp": base_mbpp["passed"]},
            "rung": {"humaneval": rung_he["passed"], "mbpp": rung_mbpp["passed"]},
        },
        "problem_delta": {
            "humaneval": rung_he["passed"] - base_he["passed"],
            "mbpp": rung_mbpp["passed"] - base_mbpp["passed"],
        },
        "paired": {
            "humaneval": paired_deltas(base_he["per_problem"], rung_he["per_problem"]),
            "mbpp": paired_deltas(base_mbpp["per_problem"], rung_mbpp["per_problem"]),
        },
        "wall_seconds": time.perf_counter() - started,
    }
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[measure_transfer] rung {rows}: HE {base_he['passed']}->{rung_he['passed']} "
          f"MBPP {base_mbpp['passed']}->{rung_mbpp['passed']} -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
