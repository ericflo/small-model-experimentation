#!/usr/bin/env python3
"""Smoke checks and optional staged full pipeline orchestration."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
TRAIN_PY = REPO / ".venv" / "bin" / "python"
VLLM_PY = REPO / ".venv-vllm" / "bin" / "python"

FROZEN = {
    ("source_rows", "harvest_temperature"): 0.6,
    ("geometry", "min_rejected_minus_chosen_logits"): 0.5,
    ("geometry", "min_rejected_probability"): 0.50,
    ("geometry", "max_entropy"): 1.50,
    ("geometry", "min_varentropy"): 0.10,
    ("geometry", "min_rows_gate"): 128,
    ("geometry", "max_rows_per_arm"): 256,
    ("train", "lora_r"): 256,
    ("train", "lora_alpha"): 128,
    ("train", "learning_rate"): 1.0e-5,
    ("train", "num_epochs"): 2,
    ("train", "demote_margin_logits"): 2.0,
    ("train", "uplift_gain_logits"): 0.5,
    ("train", "lambda_mse"): 0.4,
    ("train", "lambda_mse_target"): 0.05,
    ("train", "tau_mse_target"): 0.5,
    ("eval", "whitebox_n_prompts"): 400,
    ("repo_agent", "deep_turns"): 8,
    ("repo_agent", "sample_more_trajectories"): 2,
    ("repo_agent", "sample_more_turns_each"): 4,
    ("repo_agent", "max_sampled_tokens_per_task"): 6144,
}


def run(cmd: list[str]) -> None:
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=EXP, check=True)


def check_config() -> int:
    cfg = yaml.safe_load((EXP / "configs" / "default.yaml").read_text())
    errors = 0
    for (section, key), expected in FROZEN.items():
        actual = cfg.get(section, {}).get(key)
        if actual != expected:
            print(f"FROZEN-MISMATCH {section}.{key}: {actual!r} != {expected!r}")
            errors += 1
    acfg = cfg["repo_agent"]
    deep = int(acfg["deep_turns"])
    sample = int(acfg["sample_more_trajectories"]) * int(acfg["sample_more_turns_each"])
    if deep != sample:
        print(f"FROZEN-MISMATCH model-call budgets: deep={deep}, sample_more={sample}")
        errors += 1
    print(f"config freeze: {'OK' if not errors else f'{errors} errors'}")
    return errors


def smoke(gpu: bool) -> int:
    failures = check_config()
    for test in ("test_loopdetect.py", "test_repo_agent.py", "test_vllm_runner.py"):
        result = subprocess.run([sys.executable, str(EXP / "tests" / test)], cwd=EXP)
        failures += int(result.returncode != 0)
    result = subprocess.run([str(TRAIN_PY), str(EXP / "tests" / "test_sparse_objective.py")],
                            cwd=EXP)
    failures += int(result.returncode != 0)
    if gpu and failures == 0:
        run([str(TRAIN_PY), "scripts/score_rows.py", "--smoke", "2"])
        run([str(VLLM_PY), "scripts/eval_repo_agent.py", "--arm", "base",
             "--mode", "deep", "--tasks-per-family", "1"])
    print(f"SMOKE {'PASS' if failures == 0 else 'FAIL'}")
    return int(failures != 0)


def full(artifact_root: Path) -> int:
    run([str(TRAIN_PY), "scripts/score_rows.py"])
    adapters = artifact_root / "adapters"
    merged = artifact_root / "merged"
    for arm in ("demote", "uplift", "uplift_shuffled"):
        run([str(TRAIN_PY), "scripts/train_sparse.py", "--arm", arm,
             "--out", str(adapters / arm)])
        run([str(TRAIN_PY), "scripts/audit_logits.py", "--arm", arm,
             "--adapter", str(adapters / arm)])
        run([str(TRAIN_PY), "scripts/merge_ftpo.py", "--adapter", str(adapters / arm),
             "--out", str(merged / arm)])
    run([str(VLLM_PY), "scripts/gate_c49.py", "--model", "base",
         "--out", "runs/gate_base.json"])
    for arm in ("demote", "uplift", "uplift_shuffled"):
        run([str(VLLM_PY), "scripts/gate_c49.py", "--model", str(merged / arm),
             "--out", f"runs/gate_{arm}.json"])
        run([sys.executable, "scripts/gate_c49.py", "--compare", "runs/gate_base.json",
             f"runs/gate_{arm}.json"])
    arms = [("base", None)] + [(arm, merged / arm)
                                for arm in ("demote", "uplift", "uplift_shuffled")]
    for arm, model in arms:
        model_args = ["--model", str(model)] if model else []
        for stage in ("main", "collapse", "nothink"):
            run([str(VLLM_PY), "scripts/eval_whitebox.py", "--arm", arm,
                 "--stage", stage, *model_args])
        run([str(VLLM_PY), "scripts/eval_gym.py", "--arm", arm, *model_args])
        run([str(VLLM_PY), "scripts/eval_repo_agent.py", "--arm", arm,
             "--mode", "deep", *model_args])
    run([str(VLLM_PY), "scripts/eval_repo_agent.py", "--arm", "base",
         "--mode", "sample_more"])
    run([sys.executable, "scripts/analyze.py"])
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--gpu-smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--artifact-root", type=Path,
                        default=REPO / "large_artifacts" / "qwen35_4b_think_ftpo_round2")
    args = parser.parse_args()
    if args.smoke or args.gpu_smoke:
        return smoke(gpu=args.gpu_smoke)
    if args.full:
        if check_config():
            return 1
        return full(args.artifact_root)
    parser.error("choose --smoke, --gpu-smoke, or --full")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
