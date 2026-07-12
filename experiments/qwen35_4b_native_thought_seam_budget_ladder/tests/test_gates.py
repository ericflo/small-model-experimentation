from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import yaml


EXP = Path(__file__).resolve().parents[1]
RUN_PATH = EXP / "scripts" / "run.py"
SPEC = importlib.util.spec_from_file_location("seam_budget_run", RUN_PATH)
assert SPEC and SPEC.loader
run = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = run
SPEC.loader.exec_module(run)

MODEL_OPS_PATH = EXP / "src" / "model_ops.py"
MODEL_SPEC = importlib.util.spec_from_file_location("seam_budget_model_ops", MODEL_OPS_PATH)
assert MODEL_SPEC and MODEL_SPEC.loader
model_ops = importlib.util.module_from_spec(MODEL_SPEC)
sys.modules[MODEL_SPEC.name] = model_ops
MODEL_SPEC.loader.exec_module(model_ops)


def config() -> dict:
    return yaml.safe_load((EXP / "configs" / "default.yaml").read_text())


def _row(task: str, trace: int, close: int | None, parsed: str | None, correct: bool) -> dict:
    return {
        "task_id": task,
        "trace_index": trace,
        "close_step": close,
        "think_tokens": (close - 1) if close is not None else 1024,
        "parsed_alias": parsed,
        "correct": correct,
    }


def test_cap_classification_is_one_indexed_and_right_censored() -> None:
    rows = [
        _row("a", 0, 256, "cat", True),
        _row("a", 1, 257, "dog", False),
        _row("b", 0, None, None, False),
    ]
    at_256 = run.metrics_at_cap(rows, ["a", "b"], cap=256, minimum_think_tokens=16)
    at_512 = run.metrics_at_cap(rows, ["a", "b"], cap=512, minimum_think_tokens=16)
    assert at_256["natural_closes"] == 1
    assert at_256["usable_traces"] == 1
    assert at_512["natural_closes"] == 2
    assert at_512["mixed_usable_tasks"] == 1


def test_smallest_passing_cap_rule_is_unambiguous() -> None:
    gates = config()["gates"]["selection"]
    base = {
        "natural_close_rate": 0.9,
        "conditional_parse_rate": 0.95,
        "usable_traces": 40,
        "usable_success_rate": 0.5,
        "mixed_usable_tasks": 7,
    }
    metrics = [
        {**base, "cap": 256, "natural_close_rate": 0.7},
        {**base, "cap": 512},
        {**base, "cap": 1024},
    ]
    passing = [value["cap"] for value in metrics if run.gate_pass(value, gates, confirmation=False)]
    assert passing == [512, 1024]
    assert min(passing) == 512


def test_gate_reachability_and_wilson_contract() -> None:
    receipt = run.gate_reachability(config())
    assert receipt["passed"] is True
    assert receipt["receipts"]["selection"]["minimum_close_count"] == 39
    assert receipt["receipts"]["confirmation"]["minimum_close_count"] == 58
    assert 0.70 < run.wilson_lower(72, 72) < 1.0


def test_natural_stopper_obeys_close_and_answer_allowances() -> None:
    stopper = model_ops._NaturalThinkStopper(
        prompt_tokens=2,
        think_close_id=99,
        eos_id=100,
        max_think_steps=3,
        answer_max_tokens=2,
    )
    scores = torch.empty(1)
    assert not bool(stopper(torch.tensor([[1, 2, 7, 8]]), scores)[0])
    assert bool(stopper(torch.tensor([[1, 2, 7, 8, 9]]), scores)[0])
    assert not bool(stopper(torch.tensor([[1, 2, 7, 99, 8]]), scores)[0])
    assert bool(stopper(torch.tensor([[1, 2, 7, 99, 8, 9]]), scores)[0])
    assert bool(stopper(torch.tensor([[1, 2, 7, 99, 100]]), scores)[0])
