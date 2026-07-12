from __future__ import annotations

import json
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]


def test_outcome_blind_forced_replay_smoke() -> None:
    result = json.loads((EXP / "runs" / "model_smoke" / "result.json").read_text())
    assert result["passed"] is True
    assert result["scientific_result"] is False
    assert result["outcomes_recorded"] is False
    assert result["correctness_recorded"] is False
    assert result["counterfactual_policy_acknowledged"] is True
    assert result["forced_close_injected"] is True
    assert result["model"]["id"] == "Qwen/Qwen3.5-4B"
    assert result["model"]["revision"] == "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    assert result["token_ids"]["think_open"] == 248068
    assert result["token_ids"]["think_close"] == 248069
    assert set(result["lens_ranks"].values()) == {24}
    assert result["trace_cache_contract_pass"] is True
    assert result["forced_cache_contract_pass"] is True
    assert result["trace_forward_input_lengths"] == [375] + [1] * 7
    assert result["forced_forward_input_lengths"] == [384, 1]


def test_design_receipt_is_hash_and_lens_bound() -> None:
    receipt = json.loads((EXP / "runs" / "design_boundary_receipt.json").read_text())
    assert receipt["passed"] is True
    assert receipt["design_commit"] == "25a93b5cce1344b3e1dc6a3da378ae422b82a84d"
    assert receipt["observed_sha256"] == receipt["expected_sha256"]
    assert receipt["lens_observed_sha256"] == receipt["lens_expected_sha256"]
