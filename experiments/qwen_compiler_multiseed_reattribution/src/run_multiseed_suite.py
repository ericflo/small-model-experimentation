#!/usr/bin/env python3
"""Run the Qwen compiler multi-seed reattribution suite."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List


ROOT = Path("/workspace/experiments/qwen_compiler_multiseed_reattribution")
TRAINER = ROOT / "src" / "qwen_compiler_multiseed_reattribution.py"
ANALYZER = ROOT / "src" / "analyze_qwen_compiler_multiseed_reattribution.py"
SUITE_LOGS = ROOT / "suite_logs"


ARM_CONFIGS: Dict[str, Dict[str, str]] = {
    "max24_curriculum": {
        "stage_max_steps": "24,24,24",
        "stage_min_lengths": "1,1,8",
        "stage_train_max_lengths": "8,16,24",
        "stage_steps": "300,150,300",
        "expansion_mode": "copy_last",
    },
    "expand_copy": {
        "stage_max_steps": "8,16,24",
        "stage_min_lengths": "1,1,8",
        "stage_train_max_lengths": "8,16,24",
        "stage_steps": "300,150,300",
        "expansion_mode": "copy_last",
    },
    "max24_no_curriculum": {
        "stage_max_steps": "24",
        "stage_min_lengths": "1",
        "stage_train_max_lengths": "24",
        "stage_steps": "750",
        "expansion_mode": "copy_last",
    },
}


BASE_MAIN: Dict[str, str] = {
    "model_id": "Qwen/Qwen3-4B",
    "modulus": "97",
    "train_examples": "512",
    "eval_examples": "64",
    "paired_eval_pairs": "32",
    "batch_size": "8",
    "grad_accum": "1",
    "eval_batch_size": "8",
    "max_length": "2048",
    "head_width": "512",
    "compiler_layers": "1",
    "compiler_heads": "4",
    "compiler_dropout": "0.05",
    "expansion_noise": "0.005",
    "trace_loss_weight": "1.0",
    "executor_loss_weight": "1.0",
    "state_loss_weight": "1.0",
    "init_trace_loss_weight": "4.0",
    "op_trace_loss_weight": "1.0",
    "arg_trace_loss_weight": "4.0",
    "direct_head_weight": "0.0",
    "lr": "2e-4",
    "weight_decay": "0.0",
    "optimizer": "paged_adamw_8bit",
    "max_grad_norm": "1.0",
    "torch_dtype": "bf16",
    "load_in_4bit": "1",
    "device_map": "auto",
    "use_lora": "1",
    "gradient_checkpointing": "1",
    "lora_r": "8",
    "lora_alpha": "16",
    "lora_dropout": "0.05",
    "lora_target_modules": "all-linear",
    "log_interval": "50",
    "save_checkpoints": "1",
    "selection_split": "paired_L24",
    "selection_metric": "executor_pair_both_correct",
    "eval_lengths": "8,16,24",
    "train_template_mode": "mixed",
}


def args_to_cli(values: Dict[str, str]) -> List[str]:
    cli: List[str] = []
    for key, value in values.items():
        cli.extend([f"--{key}", str(value)])
    return cli


def run_command(name: str, cmd: List[str]) -> int:
    SUITE_LOGS.mkdir(parents=True, exist_ok=True)
    log_path = SUITE_LOGS / f"{name}.log"
    print(f"[suite] start {name}", flush=True)
    print("[suite] command: " + " ".join(cmd), flush=True)
    with log_path.open("w") as log:
        log.write("[command] " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        code = proc.wait()
        log.write(f"\n[exit] {code}\n")
    print(f"[suite] finish {name} code={code} log={log_path}", flush=True)
    return int(code)


def build_run_command(phase: str, arm: str, seed: int, overrides: Dict[str, str]) -> List[str]:
    values: Dict[str, str] = dict(BASE_MAIN)
    values.update(ARM_CONFIGS[arm])
    values.update(overrides)
    if phase == "smoke" and "," not in ARM_CONFIGS[arm]["stage_max_steps"]:
        values["stage_steps"] = "3"
    if phase == "pilot" and "," not in ARM_CONFIGS[arm]["stage_max_steps"]:
        values["stage_steps"] = "10"
    values["run_name"] = f"{phase}_{arm}_seed{seed}"
    values["arm"] = arm
    values["seed"] = str(seed)
    return [sys.executable, str(TRAINER)] + args_to_cli(values)


def phase_grid(phase: str, seeds: List[int], arms: List[str]) -> tuple[List[int], List[str], Dict[str, str]]:
    if phase == "smoke":
        return [seeds[0]], [arms[0]], {
            "stage_steps": "1,1,1",
            "train_examples": "12",
            "eval_examples": "4",
            "paired_eval_pairs": "2",
            "batch_size": "1",
            "eval_batch_size": "2",
            "head_width": "128",
            "compiler_layers": "1",
            "lora_r": "4",
            "lora_alpha": "8",
            "log_interval": "1",
            "save_checkpoints": "0",
        }
    if phase == "pilot":
        return seeds[:2], arms, {
            "stage_steps": "4,2,4",
            "train_examples": "32",
            "eval_examples": "8",
            "paired_eval_pairs": "4",
            "batch_size": "2",
            "eval_batch_size": "4",
            "head_width": "192",
            "compiler_layers": "1",
            "lora_r": "4",
            "lora_alpha": "8",
            "log_interval": "2",
            "save_checkpoints": "0",
        }
    if phase == "main":
        return seeds, arms, {}
    raise ValueError(f"unknown phase {phase!r}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["smoke", "pilot", "main"], required=True)
    p.add_argument("--seeds", default="123,456,789")
    p.add_argument("--arms", default="max24_curriculum,expand_copy,max24_no_curriculum")
    p.add_argument("--stop_on_failure", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(part.strip()) for part in args.seeds.split(",") if part.strip()]
    arms = [part.strip() for part in args.arms.split(",") if part.strip()]
    for arm in arms:
        if arm not in ARM_CONFIGS:
            raise SystemExit(f"unknown arm {arm!r}; known arms: {sorted(ARM_CONFIGS)}")
    phase_seeds, phase_arms, overrides = phase_grid(args.phase, seeds, arms)
    manifest = {
        "phase": args.phase,
        "seeds": phase_seeds,
        "arms": phase_arms,
        "overrides": overrides,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "runs": [],
    }
    manifest_path = ROOT / f"{args.phase}_suite_manifest.json"
    for seed in phase_seeds:
        for arm in phase_arms:
            run_name = f"{args.phase}_{arm}_seed{seed}"
            cmd = build_run_command(args.phase, arm, seed, overrides)
            t0 = time.time()
            code = run_command(run_name, cmd)
            manifest["runs"].append({"run": run_name, "arm": arm, "seed": seed, "exit_code": code, "elapsed_sec": round(time.time() - t0, 3)})
            manifest_path.write_text(json.dumps(manifest, indent=2))
            if code != 0 and args.stop_on_failure:
                raise SystemExit(code)
    analyze_code = run_command(f"{args.phase}_analysis", [sys.executable, str(ANALYZER)])
    manifest["analysis_exit_code"] = analyze_code
    manifest["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    if analyze_code != 0:
        raise SystemExit(analyze_code)


if __name__ == "__main__":
    main()
