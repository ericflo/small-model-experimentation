#!/usr/bin/env python3
"""Authenticate the static parent, then enter one fixed experiment stage."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path


if not (
    sys.flags.isolated == 1
    and sys.flags.ignore_environment == 1
    and sys.flags.safe_path
    and sys.dont_write_bytecode
    and sys.flags.no_site == 1
):
    raise RuntimeError("runtime dispatcher requires static-launcher -I -B -S entry")

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "src"))

from runtime_contract import authenticate_static_launcher  # noqa: E402


STAGES = {
    "training": {
        "adapter_behavior_gate": EXP / "scripts" / "adapter_behavior_gate.py",
        "analyze": EXP / "scripts" / "analyze.py",
        "authorize_stage": EXP / "scripts" / "authorize_stage.py",
        "build_eval_inputs": EXP / "scripts" / "build_eval_inputs.py",
        "build_literal_action_inputs": EXP / "scripts" / "build_literal_action_inputs.py",
        "build_literal_reflection_inputs": EXP / "scripts" / "build_literal_reflection_inputs.py",
        "calibration_gate": EXP / "scripts" / "calibration_gate.py",
        "matched_compute_gate": EXP / "scripts" / "matched_compute_gate.py",
        "merge_adapter": EXP / "scripts" / "merge_adapter.py",
        "retention_gate": EXP / "scripts" / "retention_gate.py",
        "score": EXP / "scripts" / "score.py",
        "score_literal": EXP / "scripts" / "score_literal.py",
        "tokenizer_receipt": EXP / "scripts" / "tokenizer_receipt.py",
        "train": EXP / "scripts" / "train.py",
    },
    "vllm": {
        "run_frozen_reservoir": EXP / "scripts" / "run_frozen_reservoir.py",
        "vllm_runner": EXP / "src" / "vllm_runner.py",
    },
}


def main() -> int:
    if len(sys.argv) < 3:
        raise RuntimeError("runtime launcher requires a backend and fixed stage name")
    backend = sys.argv[1]
    if backend not in STAGES:
        raise RuntimeError("runtime launcher supplied an unknown backend")
    authenticate_static_launcher(ROOT, backend)
    arguments = list(sys.argv[2:])
    if arguments and arguments[0].startswith("--cuda-visible-devices="):
        selector = arguments.pop(0).split("=", 1)[1]
        if re.fullmatch(r"GPU-[A-Za-z0-9-]+", selector) is None:
            raise RuntimeError("runtime launcher received an invalid physical GPU UUID")
        os.environ["CUDA_VISIBLE_DEVICES"] = selector
    if not arguments:
        raise RuntimeError("runtime launcher omitted its fixed stage name")
    stage_name = arguments.pop(0)
    stage = STAGES[backend].get(stage_name)
    if stage is None or not stage.is_file() or stage.is_symlink():
        raise RuntimeError("runtime launcher stage is absent or not allowlisted")
    interpreter = Path(sys.executable)
    os.execve(
        interpreter,
        [str(interpreter), "-I", "-B", "-S", str(stage), *arguments],
        dict(os.environ),
    )
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
