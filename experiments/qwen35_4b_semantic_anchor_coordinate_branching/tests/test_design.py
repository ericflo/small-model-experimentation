from __future__ import annotations

import importlib.util
import copy
import sys
from pathlib import Path

import yaml


EXP = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("anchor_run", EXP / "scripts" / "run.py")
assert SPEC and SPEC.loader
run = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = run
SPEC.loader.exec_module(run)
sys.path.insert(0, str(EXP / "src"))
from task_data import build_splits, public_mechanics, task_prompt  # noqa: E402


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
    assert value["design_boundary"] == {"status": "pending"}
    assert value["implementation_boundary"] == {"status": "pending"}


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
