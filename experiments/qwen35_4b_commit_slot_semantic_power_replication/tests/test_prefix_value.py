from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

SPEC = importlib.util.spec_from_file_location(
    "commit_slot_value_run", EXP / "scripts" / "run.py"
)
assert SPEC and SPEC.loader
run = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = run
SPEC.loader.exec_module(run)

from value_probe import (  # noqa: E402
    assign_alias_stratified_folds,
    evaluate_prefix_value,
    validate_rows,
)
from coordinates import non_j_random_dictionary  # noqa: E402


def base_config() -> dict:
    return yaml.safe_load((EXP / "configs" / "default.yaml").read_text())


def value_config() -> dict:
    return yaml.safe_load((EXP / "configs" / "prefix_value.yaml").read_text())


def synthetic_config() -> dict:
    return {
        "outcome": {
            "prospective_fraction": 0.5,
            "endpoint_fraction": 1.0,
            "minimum_pair_label_gap": 0.01,
        },
        "features": {
            "layers": [4],
            "coordinates_per_layer": 2,
            "standardize_from_training_tasks_only": True,
        },
        "cross_validation": {"folds": 4, "seed": 17, "l2": 1.0},
        "null": {"repeats": 8, "seed": 23},
        "uncertainty": {
            "bootstrap_resamples": 1000,
            "one_sided_tail_probability": 0.05,
            "seed": 29,
        },
        "gates": {
            "mixed_tasks_min": 24,
            "scored_prefixes_min": 192,
            "task_macro_pairwise_auc_min": 0.90,
            "prospective_half_pairwise_auc_min": 0.90,
            "j_minus_correct_alias_activity_min": 0.40,
            "j_minus_slot_margin_min": 0.40,
            "j_minus_alias_identity_min": 0.40,
            "j_minus_non_j_random_min": 0.40,
            "task_bootstrap_auc_lower_min": 0.50,
            "task_bootstrap_j_minus_correct_alias_lower_min": 0.0,
            "task_bootstrap_j_minus_slot_margin_lower_min": 0.0,
            "task_bootstrap_j_minus_non_j_random_lower_min": 0.0,
            "shuffled_auc_abs_from_chance_max": 0.20,
            "finite_feature_rows_rate_min": 1.0,
        },
        "decision_labels": {
            "pass": "PREFIX_J_VALUE_PASS",
            "no_value": "NO_PREFIX_J_VALUE",
        },
    }


def synthetic_rows() -> list[dict]:
    aliases = ("cat", "dog", "horse", "tiger")
    rows = []
    labels = (0.1, 0.5, 0.9)
    for task_index in range(32):
        alias_index = task_index % len(aliases)
        for trace_index, terminal in enumerate(labels):
            for fraction in (0.5, 1.0):
                strength = terminal * (0.8 if fraction == 0.5 else 1.0)
                identity = [float(index == alias_index) for index in range(4)]
                rows.append({
                    "task_id": f"task-{task_index:03d}",
                    "trace_index": trace_index,
                    "fraction": fraction,
                    "correct_alias": aliases[alias_index],
                    "alias_count": 4,
                    "j_features": [strength, -strength],
                    "non_j_random_features": [0.0, 0.0],
                    "correct_alias_activity_features": [0.0],
                    "slot_margin_features": [0.0],
                    "alias_identity_features": identity,
                    "terminal_value": terminal,
                    "finite": True,
                })
    return rows


def test_value_config_preserves_inherited_frozen_choices() -> None:
    loaded = value_config()
    assert "null" in loaded and None not in loaded
    run.validate_value_config(base_config(), loaded)
    changed = value_config()
    changed["gates"]["task_macro_pairwise_auc_min"] = 0.64
    with pytest.raises(RuntimeError, match="inherited prefix-value gate changed"):
        run.validate_value_config(base_config(), changed)


def test_prefix_count_uses_available_thought_and_minimum() -> None:
    assert run._value_prefix_token_count(1024, 0.5, 16) == 512
    assert run._value_prefix_token_count(1024, 1.0, 16) == 1024
    assert run._value_prefix_token_count(17, 0.5, 16) == 16
    assert run._value_prefix_token_count(15, 0.5, 16) is None


def test_task_folds_keep_siblings_together_and_stratify_aliases() -> None:
    rows = synthetic_rows()
    folds = assign_alias_stratified_folds(rows, folds=4, seed=17)
    assert set(folds.values()) == {0, 1, 2, 3}
    for alias in {row["correct_alias"] for row in rows}:
        task_ids = {
            row["task_id"] for row in rows if row["correct_alias"] == alias
        }
        assert {folds[task_id] for task_id in task_ids} == {0, 1, 2, 3}


def test_synthetic_prospective_signal_beats_direct_baselines() -> None:
    result = evaluate_prefix_value(synthetic_rows(), synthetic_config())
    metrics = result["metrics"]
    assert result["passed"] is True
    assert result["decision"] == "PREFIX_J_VALUE_PASS"
    assert metrics["task_macro_pairwise_auc"] == 1.0
    assert metrics["prospective_half_pairwise_auc"] == 1.0
    assert metrics["correct_alias_activity_pairwise_auc"] == 0.5
    assert metrics["slot_margin_pairwise_auc"] == 0.5
    assert metrics["alias_identity_pairwise_auc"] == 0.5
    assert metrics["non_j_random_pairwise_auc"] == 0.5
    assert abs(metrics["shuffled_null_mean_pairwise_auc"] - 0.5) <= 0.20


def test_fraction_labels_must_share_one_later_terminal_outcome() -> None:
    rows = synthetic_rows()
    rows[0]["terminal_value"] = 0.2
    with pytest.raises(ValueError, match="different terminal labels"):
        validate_rows(rows, synthetic_config())


def test_each_task_fraction_requires_all_three_paths() -> None:
    rows = synthetic_rows()
    rows = [
        row
        for row in rows
        if not (
            row["task_id"] == "task-000"
            and row["trace_index"] == 2
        )
    ]
    with pytest.raises(ValueError, match="does not contain three paths"):
        validate_rows(rows, synthetic_config())


def test_prefix_value_loader_opens_only_value_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    opened = []

    def fake_reader(path: Path) -> list[dict]:
        opened.append(path)
        return [{} for _ in range(48)]

    monkeypatch.setattr(run, "read_jsonl", fake_reader)
    rows = run.load_value_tasks(base_config(), value_config())
    assert len(rows) == 48
    assert [path.name for path in opened] == ["value_fit.jsonl"]


def test_unanchored_value_boundary_fails_before_model_or_data() -> None:
    with pytest.raises(RuntimeError, match="not anchored"):
        run.value_implementation_boundary_receipt(base_config(), value_config())


def test_frozen_non_j_dictionaries_are_deterministic_and_orthogonal() -> None:
    import torch

    config = value_config()
    state = torch.load(
        EXP / "assets" / "context_lens.pt", map_location="cpu", weights_only=True
    )
    for layer in config["features"]["layers"]:
        directions = state["directions"][layer].float()
        kwargs = {
            "width": config["features"]["non_j_random_coordinates_per_layer"],
            "seed": run._stable_seed(
                config["features"]["non_j_random_seed"], str(layer)
            ),
            "rtol": config["features"]["pseudoinverse_rtol"],
            "max_span_projection": config["features"][
                "non_j_max_span_projection"
            ],
        }
        first, projection = non_j_random_dictionary(directions, **kwargs)
        second, projection_again = non_j_random_dictionary(directions, **kwargs)
        assert first.shape == (2560, 24)
        assert torch.equal(first, second)
        assert projection == projection_again <= 1.0e-5
