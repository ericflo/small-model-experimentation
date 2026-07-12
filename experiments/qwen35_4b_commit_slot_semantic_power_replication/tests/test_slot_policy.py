from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml


EXP = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("commit_slot_run", EXP / "scripts" / "run.py")
assert SPEC and SPEC.loader
run = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = run
SPEC.loader.exec_module(run)


def config() -> dict:
    return yaml.safe_load((EXP / "configs" / "default.yaml").read_text())


def test_gate_reachability_and_frozen_gate() -> None:
    value = config()
    receipt = run.gate_reachability(value)
    assert receipt["passed"] is True
    assert receipt["receipts"]["seam_selection"]["minimum_successes"] == 68
    passing = {
        "slot_success_rate": 0.25,
        "mixed_slot_tasks": 30,
        "real_minus_no_thought_accuracy": 0.08,
        "real_minus_shuffled_thought_accuracy": 0.06,
        "task_real_minus_shuffled_one_sided_95_lower": 0.01,
        "correct_alias_success_support": 9,
        "chosen_alias_support": 9,
        "full_vocab_top_is_alias_rate": 0.80,
        "mean_full_vocab_alias_probability_mass": 0.60,
        "finite_slot_rows_rate": 1.0,
    }
    assert run.seam_gate(passing, value["gates"]["seam_selection"])
    assert not run.seam_gate(
        {**passing, "real_minus_no_thought_accuracy": 0.02},
        value["gates"]["seam_selection"],
    )
    assert not run.seam_gate(
        {**passing, "task_real_minus_shuffled_one_sided_95_lower": 0.0},
        value["gates"]["seam_selection"],
    )
    assert not run.seam_gate(
        {**passing, "correct_alias_success_support": 7},
        value["gates"]["seam_selection"],
    )


def test_thought_prefix_obeys_close_and_cap_boundaries() -> None:
    model = SimpleNamespace(eos_id=100, think_close_id=99)
    trace = {"close_step": None, "generated_token_ids": list(range(20))}
    prefix, mode = run._thought_prefix(model, trace, 8)
    assert prefix == list(range(8)) and mode == "forced_at_cap"
    natural = {"close_step": 5, "generated_token_ids": [7, 8, 9, 10, 99, 3]}
    prefix, mode = run._thought_prefix(model, natural, 8)
    assert prefix == [7, 8, 9, 10] and mode == "natural_prefix_replayed"
    malformed = {"close_step": None, "generated_token_ids": [7, 100, 9]}
    assert run._thought_prefix(model, malformed, 3) == (None, "malformed_pre_cap")


def test_slot_metrics_keep_close_only_and_no_thought_controls_separate() -> None:
    full_probs = {"cat": 0.01}
    slot_rows = [
        {"task_id": "a", "cap": 256, "commit_mode": "forced_at_cap",
         "correct": True, "finite": True,
         "correct_alias_probability": 0.7, "constrained_margin": 1.0,
         "correct_alias": "cat", "chosen_alias": "cat",
         "alias_full_vocab_probabilities": full_probs,
         "full_vocab_alias_probability_mass": 0.1, "full_vocab_top_is_alias": True},
        {"task_id": "a", "cap": 256, "commit_mode": "forced_at_cap",
         "correct": False, "finite": True,
         "correct_alias_probability": 0.1, "constrained_margin": 0.4,
         "correct_alias": "cat", "chosen_alias": "dog",
         "alias_full_vocab_probabilities": full_probs,
         "full_vocab_alias_probability_mass": 0.1, "full_vocab_top_is_alias": False},
        {"task_id": "b", "cap": 256, "commit_mode": "forced_at_cap",
         "correct": True, "finite": True,
         "correct_alias_probability": 0.6, "constrained_margin": 0.8,
         "correct_alias": "cat", "chosen_alias": "cat",
         "alias_full_vocab_probabilities": full_probs,
         "full_vocab_alias_probability_mass": 0.1, "full_vocab_top_is_alias": True},
    ]
    shuffled = [
        {
            "task_id": row["task_id"], "cap": 256, "correct": False,
            "finite": True, "correct_alias_probability": 0.1,
        }
        for row in slot_rows
    ]
    no_thought = [
        {
            "task_id": "a", "correct": False, "finite": True,
            "correct_alias_probability": 0.1,
        },
        {
            "task_id": "b", "correct": False, "finite": True,
            "correct_alias_probability": 0.1,
        },
    ]
    freeform = [
        {"cap": 256, "parseable": False, "correct": False, "answer_stopped_by": "answer_cap"}
        for _ in slot_rows
    ]
    metrics = run.slot_metrics(
        slot_rows, shuffled, no_thought, freeform, ["a", "b"], 256
    )
    assert metrics["slot_success_rate"] == 2 / 3
    assert metrics["mixed_slot_tasks"] == 1
    assert metrics["no_thought_slot_accuracy"] == 0.0
    assert metrics["shuffled_thought_slot_accuracy"] == 0.0
    assert metrics["close_only_answer_cap_rate"] == 1.0
    assert run.observed_gate_reachability(
        metrics, config()["gates"]["seam_selection"]
    )["all_observed_gain_gates_reachable"]


def test_shuffled_thought_is_deterministic_exact_length_multiset() -> None:
    thought = list(range(64))
    first, moved = run._shuffled_thought(thought, 123)
    second, moved_again = run._shuffled_thought(thought, 123)
    other, _ = run._shuffled_thought(thought, 124)
    assert first == second
    assert sorted(first) == thought
    assert len(first) == len(thought)
    assert moved == moved_again and moved > 0.90
    assert first != other


def test_config_payload_hash_excludes_only_self_referential_boundary() -> None:
    value = config()
    original = run._config_payload_sha256(value)
    value["design_boundary"]["commit"] = "different"
    assert run._config_payload_sha256(value) == original
    value["gates"]["seam_selection"]["slot_success_rate_min"] = 0.21
    assert run._config_payload_sha256(value) != original


def test_power_receipt_matches_parent_and_config() -> None:
    receipt = run.require_power_receipt(config())
    assert receipt["passed"] is True
    assert receipt["required_tasks_per_seam_stage"] == 113
    assert receipt["planned_approximate_power"] >= 0.80
