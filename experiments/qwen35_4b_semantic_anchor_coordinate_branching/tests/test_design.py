from __future__ import annotations

import importlib.util
import copy
import sys
from types import SimpleNamespace
from pathlib import Path

import yaml
import torch


EXP = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("anchor_run", EXP / "scripts" / "run.py")
assert SPEC and SPEC.loader
run = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = run
SPEC.loader.exec_module(run)
sys.path.insert(0, str(EXP / "src"))
from task_data import build_splits, public_mechanics, task_prompt  # noqa: E402
from branch_geometry import balanced_j_branches  # noqa: E402
from mechanics import ARMS, evaluate, expected_token, wrong_derangement  # noqa: E402
from model_ops import (  # noqa: E402
    CoordinateClampPatcher,
    FullActivationPatcher,
    QuantizationAwareOrthogonalPatcher,
    QwenClampModel,
)


def config() -> dict:
    return yaml.safe_load((EXP / "configs" / "default.yaml").read_text())


def test_only_permitted_model_and_frozen_band():
    value = config()
    assert value["model"]["id"] == "Qwen/Qwen3.5-4B"
    assert value["model"]["revision"] == "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    assert value["lens"]["band"] == [4, 5, 6, 7, 8]


def test_aliases_and_result_labels_partition_frozen_lens():
    validated = run.validate_config(config())
    assert len(validated["aliases"]) == 12
    assert len(validated["result_labels"]) == 12
    assert set(validated["aliases"]).isdisjoint(validated["result_labels"])


def test_all_diagnostic_results_are_distinct():
    results = run.diagnostic_results([3, -1, 2, 0], 2)
    assert len(results) == 12
    assert len({tuple(value) for value in results.values()}) == 12
    assert results["running_sum"] == [3, 2, 4, 4]
    assert results["rotate_k"] == [2, 0, 3, -1]


def test_scientific_stages_remain_fail_closed():
    value = config()
    assert value["design_boundary"]["status"] == "anchored"
    assert value["design_boundary"]["commit"] == "9437bdc2664772f4ad2c50e8403740f11c28688c"
    assert value["implementation_boundary"] == {"status": "pending"}
    assert value["mechanics_boundary"] == {"status": "pending"}


def test_scientific_config_hash_excludes_only_boundary_receipts():
    value = config()
    boundary_changed = copy.deepcopy(value)
    boundary_changed["implementation_boundary"] = {"status": "anchored", "noise": 1}
    boundary_changed["mechanics_boundary"] = {"status": "anchored", "noise": 2}
    assert run.scientific_config_sha256(value) == run.scientific_config_sha256(
        boundary_changed
    )
    gate_changed = copy.deepcopy(value)
    gate_changed["gates"]["mechanics"]["coordinate_consequence_rate_min"] += 0.01
    assert run.scientific_config_sha256(value) != run.scientific_config_sha256(
        gate_changed
    )


def test_fresh_splits_are_deterministic_and_balanced():
    first = build_splits(config())
    second = build_splits(config())
    assert first == second
    assert {name: len(rows) for name, rows in first.items()} == {
        "mechanics": 4,
        "qualification": 24,
        "confirmation": 48,
    }
    aliases = config()["data"]["alias_tokens"]
    operations = config()["data"]["operation_names"]
    for split, expected in (("qualification", 2), ("confirmation", 4)):
        rows = first[split]
        for alias in aliases:
            for operation in operations:
                assert sum(
                    task["alias_to_operation"][alias] == operation for task in rows
                ) == expected


def test_public_mechanics_seals_correctness_and_prompt_ignores_mutation():
    task = build_splits(config())["mechanics"][0]
    public = public_mechanics(task)
    assert set(public) == {
        "task_id", "visible", "alias_to_operation", "source_alias",
        "result_label_by_operation",
    }
    original = task_prompt(public)
    mutated = copy.deepcopy(public)
    mutated["first_op"] = "forbidden"
    mutated["correct_alias"] = "forbidden"
    mutated["hidden"] = [{"input": [999], "output": [888]}]
    mutated["target_pipeline"] = [{"name": "forbidden", "parameter": 999}]
    assert task_prompt(mutated) == original


def test_label_map_seed_changes_only_diagnostic_labels():
    original_config = config()
    changed_config = copy.deepcopy(original_config)
    changed_config["seeds"]["label_map"] += 1
    original = build_splits(original_config)
    changed = build_splits(changed_config)
    for split in original:
        for left, right in zip(original[split], changed[split], strict=True):
            for key in (
                "task_id", "visible", "hidden", "first_op", "correct_alias",
                "target_pipeline", "alias_to_operation", "source_alias",
            ):
                assert left[key] == right[key]
    assert any(
        left["result_label_by_operation"] != right["result_label_by_operation"]
        for left, right in zip(original["mechanics"], changed["mechanics"], strict=True)
    )


def test_full_and_coordinate_patchers_use_clean_requested_state():
    layer = torch.nn.Identity()
    hidden = torch.zeros(1, 3, 4, dtype=torch.bfloat16)
    full = FullActivationPatcher([layer], 1, {0: torch.tensor([1.0, 2.0, 3.0, 4.0])})
    with full:
        output = layer(hidden)
    assert torch.equal(output[0, 1].float(), torch.tensor([1.0, 2.0, 3.0, 4.0]))
    directions = {0: torch.eye(4)[:, :2]}
    coordinate = CoordinateClampPatcher(
        [layer], 1, directions, {0: torch.tensor([3.0, -2.0])}, rtol=1e-5
    )
    with coordinate:
        output = layer(hidden)
    assert torch.equal(output[0, 1, :2].float(), torch.tensor([3.0, -2.0]))
    assert torch.equal(output[0, 1, 2:].float(), torch.zeros(2))


def test_exact_orthogonal_live_control_passes():
    layer = torch.nn.Identity()
    hidden = torch.zeros(1, 3, 4, dtype=torch.bfloat16)
    patcher = QuantizationAwareOrthogonalPatcher(
        [layer],
        1,
        {0: torch.tensor([[0.0, 1.0, 0.0, 0.0]])},
        {0: torch.tensor([[1.0], [0.0], [0.0], [0.0]])},
        {0: 1.0},
        rtol=1e-5,
        norm_tolerance=1e-5,
        projection_tolerance=0.01,
        correction_iterations=8,
        correction_damping=0.5,
        binary_search_steps=16,
        lattice_pair_steps=4,
        repair_safety_margin=0.95,
    )
    with patcher:
        output = layer(hidden)
    assert patcher.passed_by_layer == {0: True}
    assert torch.equal(output[0, 1].float(), torch.tensor([0.0, 1.0, 0.0, 0.0]))


class _FakeTokenizer:
    def __call__(self, text, *, add_special_tokens=False):
        del add_special_tokens
        values = {
            "\n\nCandidate first-operation alias:": [71, 72, 73],
            " cat": [7993],
            " suffix": [81, 101, 82],
            "\n\nCandidate first-operation alias: cat suffix": [
                71, 72, 73, 7993, 81, 101, 82,
            ],
        }
        return SimpleNamespace(input_ids=values[text])

    def decode(self, values, *, skip_special_tokens=False):
        del skip_special_tokens
        assert values == [71, 72, 73, 7993, 81, 101, 82]
        return "\n\nCandidate first-operation alias: cat suffix"


def test_anchor_position_is_constructed_not_searched():
    model = QwenClampModel.__new__(QwenClampModel)
    model.tokenizer = _FakeTokenizer()
    model.device = torch.device("cpu")
    model.model = SimpleNamespace(
        config=SimpleNamespace(
            get_text_config=lambda: SimpleNamespace(eos_token_id=102)
        )
    )
    native = {
        "input_ids": torch.tensor([[100, 7993, 2]]),
        "think_open_id": 100,
        "think_close_id": 101,
    }
    prepared = model.prepare_anchor_context(
        native,
        [4] * 512,
        prefix_text="\n\nCandidate first-operation alias:",
        anchor_alias="cat",
        suffix_text=" suffix",
        max_length=1024,
    )
    assert prepared["position"] == 3 + 512 + 3
    assert int(prepared["input_ids"][0, prepared["position"]]) == 7993
    assert prepared["anchor_occurrences"] == [1, prepared["position"]]


def test_patchers_reject_repeat_application():
    layer = torch.nn.Identity()
    hidden = torch.zeros(1, 2, 4, dtype=torch.bfloat16)
    patcher = FullActivationPatcher([layer], 1, {0: torch.ones(4)})
    try:
        with patcher:
            layer(hidden)
            layer(hidden)
    except RuntimeError as error:
        assert "repeated" in str(error)
    else:
        raise AssertionError("repeated patch did not fail")


def test_additive_bank_is_zero_sum_and_norm_anchored():
    value = config()
    lens = torch.load(EXP / value["lens"]["path"], map_location="cpu", weights_only=True)
    for layer in value["lens"]["band"]:
        branches = balanced_j_branches(
            lens["directions"][layer],
            public_concepts=12,
            target_rms_norm=value["lens"]["replicated_median_delta_norms"][layer],
        )
        assert float(branches.sum(dim=1).abs().max()) < 1e-6
        observed = float(torch.sqrt(torch.mean(branches.norm(dim=0).square())))
        assert abs(observed - value["lens"]["replicated_median_delta_norms"][layer]) < 1e-6


def test_wrong_donor_is_bijective_non_source_derangement():
    aliases = config()["data"]["alias_tokens"]
    mapping = wrong_derangement("cat", aliases)
    assert set(mapping) == set(aliases) - {"cat"}
    assert set(mapping.values()) == set(aliases) - {"cat"}
    assert all(target != wrong and wrong != "cat" for target, wrong in mapping.items())


def _synthetic_positive_mechanics():
    value = config()
    tasks = [public_mechanics(task) for task in build_splits(value)["mechanics"]]
    aliases = value["data"]["alias_tokens"]
    rows = []
    for task in tasks:
        source = task["source_alias"]
        wrongs = wrong_derangement(source, aliases)
        for target in aliases:
            if target == source:
                continue
            wrong = wrongs[target]
            for probe in ("direct", "consequence"):
                target_token = expected_token(task, target, probe)
                wrong_token = expected_token(task, wrong, probe)
                for arm in ARMS:
                    target_selected = arm in {
                        "text_target", "full_donor", "donor_j", "additive_j"
                    }
                    wrong_selected = arm == "wrong_donor_j"
                    rows.append({
                        "task_id": task["task_id"],
                        "target_alias": target,
                        "probe": probe,
                        "arm": arm,
                        "target_selected": target_selected,
                        "wrong_own_selected": wrong_selected,
                        "target_probability": 0.9 if target_selected else 0.05,
                        "target_token": target_token,
                        "wrong_token": wrong_token,
                        "parsed": True,
                        "finite": True,
                    })
    numeric = [{"passed": True} for _ in range(880)]
    interventions = [{"passed": True}]
    return value, rows, numeric, interventions


def test_mechanics_decision_requires_computed_consequence():
    value, rows, numeric, interventions = _synthetic_positive_mechanics()
    positive = evaluate(rows, numeric, interventions, value)
    assert positive["decision"] == "NATIVE_ANCHOR_J_CONSEQUENCE_TRANSPORT"
    assert positive["additive_decision"] == "ADDITIVE_ANCHOR_TRANSPORT"
    for row in rows:
        if row["arm"] == "donor_j" and row["probe"] == "consequence":
            row["target_selected"] = False
            row["target_probability"] = 0.05
    negative = evaluate(rows, numeric, interventions, value)
    assert negative["decision"] == "DIRECT_ONLY_NATIVE_ANCHOR_J"
