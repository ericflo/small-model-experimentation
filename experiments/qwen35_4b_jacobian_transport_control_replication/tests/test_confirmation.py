from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest
import yaml


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from confirmation import (  # noqa: E402
    CONDITIONS,
    evaluate_confirmation,
    validate_calibration_contract,
)


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_committed_calibration_contract_is_complete_and_outcome_blind() -> None:
    config = yaml.safe_load((EXP / "configs" / "default.yaml").read_text())
    summary = json.loads((EXP / "runs" / "control_calibration.json").read_text())
    rows = _jsonl(EXP / "runs" / "control_calibration_rows.jsonl")
    audit = validate_calibration_contract(summary, rows, config)
    assert audit["passed"]
    assert audit["numeric_rows"] == 480
    assert hashlib.sha256(
        (EXP / "runs" / "control_calibration.json").read_bytes()
    ).hexdigest() == "58ac9086b86efcae78180476e5631024f1e6c3afe08cc3f562ba5d35ea7ba22b"
    assert hashlib.sha256(
        (EXP / "runs" / "control_calibration_rows.jsonl").read_bytes()
    ).hexdigest() == "be7c73edfff88b79bd9f4f115c5086cf56bcf5927f77ee95a92135244f3ea27d"
    tampered = copy.deepcopy(rows)
    tampered[0]["realized_span_projection_fraction"] = 0.02
    with pytest.raises(RuntimeError, match="geometry"):
        validate_calibration_contract(summary, tampered, config)


def _synthetic_config() -> dict:
    return {
        "data": {"confirmation_items": 2},
        "intervention": {
            "band": [4, 5, 6, 7, 8],
            "random_arms": ["random_a", "random_b"],
            "norm_relative_tolerance": 1e-5,
            "realized_span_projection_max": 1e-2,
            "causal_activation_atol": 1e-3,
        },
        "seeds": {"bootstrap": 99},
        "gates": {
            "clean_accuracy_min": 0.8,
            "clean_parse_rate_min": 0.95,
            "donor_direct_target_rate_min": 0.6,
            "donor_consequence_target_rate_min": 0.5,
            "j_direct_shift_min": 0.2,
            "j_consequence_shift_min": 0.15,
            "j_minus_random_min": 0.1,
            "j_minus_wrong_target_min": 0.1,
            "wrong_own_digit_shift_min": 0.1,
            "max_parse_rate_drop": 0.05,
            "bootstrap_resamples": 1000,
            "bootstrap_lower_bound_min": 0.0,
        },
    }


def _synthetic_rows() -> tuple[list[dict], list[dict]]:
    rows = []
    items = ("confirm-a", "confirm-b")
    for item_id in items:
        for kind in ("direct", "consequence"):
            for condition in CONDITIONS:
                target = condition in {"full_target_donor", "j_all24", "j_pair"}
                wrong = condition == "j_wrong_donor"
                rows.append({
                    "item_id": item_id,
                    "prompt_kind": kind,
                    "condition": condition,
                    "source_correct": not target and not wrong,
                    "target_selected": target,
                    "wrong_selected": wrong,
                    "parsed": True,
                    "target_minus_source_logit": 1.0 if target else -1.0,
                    "total_delta_norm": 0.0 if condition == "baseline" else 1.0,
                })
    controls = [
        {
            "item_id": item_id,
            "prompt_kind": kind,
            "arm": arm,
            "layer": layer,
            "passed": True,
            "norm_relative_error": 0.0,
            "realized_span_projection_fraction": 0.0,
            "lattice_pair_steps": 0,
        }
        for item_id in items
        for kind in ("direct", "consequence")
        for arm in ("random_a", "random_b")
        for layer in (4, 5, 6, 7, 8)
    ]
    return rows, controls


def test_frozen_confirmation_requires_both_random_arms_and_wrong_specificity() -> None:
    config = _synthetic_config()
    rows, controls = _synthetic_rows()
    result = evaluate_confirmation(
        rows,
        controls,
        config,
        design_pass=True,
        calibration_pass=True,
        causal_max_abs=0.0,
    )
    assert result["passed"]
    assert result["decision"] == "REPLICATED_J_TRANSPORT"
    assert set(
        result["gate_metrics"]["paired_bootstrap_j_minus_random"]
    ) == {"random_a", "random_b"}

    incomplete = [row for row in rows if not (
        row["item_id"] == "confirm-a"
        and row["prompt_kind"] == "direct"
        and row["condition"] == "random_b"
    )]
    with pytest.raises(RuntimeError, match="arms"):
        evaluate_confirmation(
            incomplete,
            controls,
            config,
            design_pass=True,
            calibration_pass=True,
            causal_max_abs=0.0,
        )

    nonspecific = copy.deepcopy(rows)
    for row in nonspecific:
        if row["condition"] == "j_wrong_donor":
            row["wrong_selected"] = False
            row["target_selected"] = True
    failed = evaluate_confirmation(
        nonspecific,
        controls,
        config,
        design_pass=True,
        calibration_pass=True,
        causal_max_abs=0.0,
    )
    assert not failed["passed"]
    assert failed["gate_metrics"]["consequence_j_minus_wrong_target"] == 0.0
    assert failed["gate_metrics"]["wrong_donor_own_digit_shift"] == 0.0


def test_saved_confirmation_recomputes_to_frozen_terminal_decision() -> None:
    config = yaml.safe_load((EXP / "configs" / "default.yaml").read_text())
    saved = json.loads((EXP / "runs" / "confirmation.json").read_text())
    rows = _jsonl(EXP / "runs" / "confirmation_rows.jsonl")
    controls = _jsonl(EXP / "runs" / "confirmation_control_rows.jsonl")
    recomputed = evaluate_confirmation(
        rows,
        controls,
        config,
        design_pass=True,
        calibration_pass=True,
        causal_max_abs=float(saved["control_audit"]["causal_activation_max_abs"]),
    )
    assert len(rows) == 768
    assert len(controls) == 960
    assert recomputed["decision"] == "REPLICATED_J_TRANSPORT"
    assert recomputed["summaries"] == saved["summaries"]
    assert recomputed["gate_metrics"] == saved["gate_metrics"]
    assert recomputed["control_audit"] == {
        key: saved["control_audit"][key] for key in recomputed["control_audit"]
    }
