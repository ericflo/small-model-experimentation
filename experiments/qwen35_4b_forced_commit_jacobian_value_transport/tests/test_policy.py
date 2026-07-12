from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import yaml


EXP = Path(__file__).resolve().parents[1]
RUN_SPEC = importlib.util.spec_from_file_location("forced_commit_run", EXP / "scripts" / "run.py")
assert RUN_SPEC and RUN_SPEC.loader
run = importlib.util.module_from_spec(RUN_SPEC)
sys.modules[RUN_SPEC.name] = run
RUN_SPEC.loader.exec_module(run)

MODEL_SPEC = importlib.util.spec_from_file_location("forced_commit_model", EXP / "src" / "model_ops.py")
assert MODEL_SPEC and MODEL_SPEC.loader
model_ops = importlib.util.module_from_spec(MODEL_SPEC)
sys.modules[MODEL_SPEC.name] = model_ops
MODEL_SPEC.loader.exec_module(model_ops)


def config() -> dict:
    return yaml.safe_load((EXP / "configs" / "default.yaml").read_text())


def _row(task: str, cap: int, forced: bool, parsed: bool, correct: bool, answer_cap: bool = False) -> dict:
    return {
        "task_id": task,
        "cap": cap,
        "commit_mode": "forced" if forced else "natural",
        "parseable": parsed,
        "correct": correct,
        "answer_stopped_by": "answer_cap" if answer_cap else "eos",
    }


def test_policy_metrics_separate_forced_and_natural_denominators() -> None:
    rows = [
        _row("a", 256, True, True, True),
        _row("a", 256, True, True, False),
        _row("b", 256, False, True, True),
        _row("b", 512, True, False, False),
    ]
    metrics = run.policy_metrics(rows, ["a", "b"], 256)
    assert metrics["traces"] == 3
    assert metrics["forced_commits"] == 2
    assert metrics["natural_commits"] == 1
    assert metrics["policy_parse_rate"] == 1.0
    assert metrics["forced_parse_rate"] == 1.0
    assert metrics["mixed_policy_tasks"] == 1


def test_frozen_seam_gate_and_reachability() -> None:
    value = config()
    receipt = run.gate_reachability(value)
    assert receipt["passed"] is True
    assert receipt["receipts"]["seam_selection"]["minimum_forced_commits"] == 24
    passing = {
        "policy_parse_rate": 0.95,
        "forced_parse_rate": 0.95,
        "forced_commit_rate": 1.0,
        "policy_success_rate": 0.5,
        "mixed_policy_tasks": 7,
        "answer_cap_rate": 0.0,
    }
    assert run.seam_gate(passing, value["gates"]["seam_selection"])
    assert not run.seam_gate(
        {**passing, "forced_parse_rate": 0.89}, value["gates"]["seam_selection"]
    )


def test_trace_stopper_distinguishes_cap_from_natural_close() -> None:
    stopper = model_ops._TraceStopper(
        prompt_tokens=2, close_id=99, eos_id=100, thought_cap=3, answer_cap=2
    )
    scores = torch.empty(1)
    assert not bool(stopper(torch.tensor([[1, 2, 7, 8]]), scores)[0])
    assert bool(stopper(torch.tensor([[1, 2, 7, 8, 9]]), scores)[0])
    assert not bool(stopper(torch.tensor([[1, 2, 7, 99, 8]]), scores)[0])
    assert bool(stopper(torch.tensor([[1, 2, 7, 99, 8, 9]]), scores)[0])
