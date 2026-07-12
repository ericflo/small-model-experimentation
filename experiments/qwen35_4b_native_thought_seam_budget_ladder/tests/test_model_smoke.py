from __future__ import annotations

import json
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]


def test_outcome_blind_model_smoke_contract() -> None:
    result = json.loads((EXP / "runs" / "model_smoke" / "result.json").read_text())
    assert result["passed"] is True
    assert result["scientific_result"] is False
    assert result["outcomes_recorded"] is False
    assert result["correctness_recorded"] is False
    assert result["model"]["id"] == "Qwen/Qwen3.5-4B"
    assert result["model"]["revision"] == "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    assert result["model"]["layers"] == 32
    assert result["model"]["hidden_size"] == 2560
    assert result["token_ids"]["think_open"] == 248068
    assert result["token_ids"]["think_close"] == 248069
    assert result["cache_contract_pass"] is True
    lengths = result["forward_input_lengths"]
    assert lengths[0] == result["prompt_tokens"] == 472
    assert lengths[1:] == [1] * 7


def test_design_boundary_receipt_passed_before_model_smoke() -> None:
    receipt = json.loads((EXP / "runs" / "design_boundary_receipt.json").read_text())
    assert receipt["passed"] is True
    assert receipt["design_commit"] == "1140df65832fdd9e861bcd38cb33dc2ed0fb7ebf"
    assert receipt["observed_sha256"] == receipt["expected_sha256"]
