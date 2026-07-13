from __future__ import annotations

import copy
import json
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

import yaml


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from protocol import (  # noqa: E402
    candidate_injection,
    parse_program,
    parse_result,
    program_source,
    select_visible,
    task_prompt,
)
from task_data import (  # noqa: E402
    CONCRETE_OPERATIONS,
    DEPTH_TWO_PROGRAMS,
    IDENTIFIABLE_FIRST_OPERATIONS,
    build_splits,
    canonical_operation,
    diagnostic_apply,
    operation_from_record,
    public_task,
)


def config():
    return yaml.safe_load((EXP / "configs" / "default.yaml").read_text())


@lru_cache(maxsize=1)
def splits():
    return build_splits(config())


def test_only_permitted_model_backend_and_24_operation_inventory_are_frozen():
    value = config()
    assert value["model"]["id"] == "Qwen/Qwen3.5-4B"
    assert value["model"]["revision"] == "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    assert value["model"]["backend"] == "vllm"
    assert len(CONCRETE_OPERATIONS) == 24
    assert len(IDENTIFIABLE_FIRST_OPERATIONS) == 23
    assert len(DEPTH_TWO_PROGRAMS) == 576
    assert ("negate", None) in CONCRETE_OPERATIONS
    assert ("negate", None) not in IDENTIFIABLE_FIRST_OPERATIONS


def test_splits_are_deterministic_unique_and_bound_operation_balanced():
    first = splits()
    second = build_splits(config())
    assert first == second
    assert {name: len(rows) for name, rows in first.items()} == {
        "qualification": 48,
        "confirmation": 96,
    }
    assert len({row["task_id"] for rows in first.values() for row in rows}) == 144
    for rows in first.values():
        counts = Counter(operation_from_record(row["first_op"]) for row in rows)
        assert set(counts) == set(IDENTIFIABLE_FIRST_OPERATIONS)
        assert max(counts.values()) - min(counts.values()) <= 1
        assert all(row["visible_consistent_program_count"] >= 1 for row in rows)
        assert all(row["visible_consistent_hidden_behavior_count"] == 1 for row in rows)
        assert all(row["visible_consistent_probe_behavior_count"] == 1 for row in rows)


def test_public_schema_excludes_every_gold_field_and_prompt_refuses_mutation():
    task = splits()["qualification"][0]
    public = public_task(task)
    assert set(public) == {"task_id", "depth", "visible", "unlabeled_probe_inputs"}
    assert not ({"hidden", "first_op", "target_pipeline"} & set(public))
    prompt = task_prompt(public)
    mutation = copy.deepcopy(public)
    mutation["hidden"] = [{"input": [999], "output": [0]}]
    try:
        task_prompt(mutation)
    except ValueError:
        pass
    else:
        raise AssertionError("prompt accepted a hidden-bearing task schema")
    assert task_prompt(public) == prompt


def test_strict_python_parser_roundtrips_and_ignores_reasoning():
    pipeline = [("reverse", None), ("add_k", -2)]
    source = program_source(pipeline)
    parsed = parse_program(source)
    assert parsed["parsed"]
    assert parsed["pipeline"] == pipeline
    assert parsed["canonical"] == "reverse | add_k(-2)"
    assert parse_program(f"scratch def fake(): pass</think>\n{source}")["parsed"]
    assert parse_program(f"```python\n{source}\n```<|im_end|>")["parsed"]


def test_strict_python_parser_rejects_code_and_wrapper_attacks():
    rejected = [
        "import os\n" + program_source([("reverse", None), ("reverse", None)]),
        "def transform(xs):\n    xs = xs[::-1]\n    xs = reverse(xs)\n    return xs",
        "def transform(xs):\n    xs = reverse(obj.xs)\n    xs = reverse(xs)\n    return xs",
        "def transform(xs):\n    if xs:\n        xs = reverse(xs)\n    xs = reverse(xs)\n    return xs",
        "def transform(xs):\n    xs = add_k(xs, 999)\n    xs = reverse(xs)\n    return xs",
        "prose\n" + program_source([("reverse", None), ("reverse", None)]),
        "```python\n"
        + program_source([("reverse", None), ("reverse", None)])
        + "\n```\n```python\npass\n```",
    ]
    for text in rejected:
        assert not parse_program(text)["parsed"], text


def test_candidate_injection_is_bound_and_exactly_provisional():
    negative = candidate_injection(("add_k", -2))
    positive = candidate_injection(("add_k", 2))
    assert "Hypothesis fork — provisional" in negative
    assert "Concrete first operation: add_k(-2)" in negative
    assert negative != positive
    assert "<think>" not in negative and "</think>" not in negative


def test_result_parser_is_unrestricted_line_contract():
    assert parse_result("Some reasoning</think>\nRESULT: [3, -1, 2]")["parsed"]
    assert parse_result("Some reasoning</think>\nRESULT: [3]<|im_end|>")["parsed"]
    assert not parse_result("RESULT: label")["parsed"]
    assert not parse_result("RESULT: [1]\nRESULT: [2]")["parsed"]
    assert not parse_result("prose after thinking\nRESULT: [1]")["parsed"]


def test_selector_is_order_invariant_hidden_blind_and_canonical_deduplicated():
    task = splits()["qualification"][0]
    public = public_task(task)
    target_pipeline = [operation_from_record(value) for value in task["target_pipeline"]]
    target_text = program_source(target_pipeline)
    candidates = [
        {"candidate_id": "wrong", "text": program_source([("reverse", None)] * 2)},
        {"candidate_id": "target-a", "text": target_text},
        {"candidate_id": "target-b", "text": target_text},
    ]
    left = select_visible(public, candidates)
    right = select_visible(public, list(reversed(candidates)))
    assert left["selected"]["canonical"] == right["selected"]["canonical"]
    assert left["eligible_rows"] == 2
    assert left["eligible_unique_programs"] == 1
    assert left["selected_cluster_size"] == 1
    mutation = copy.deepcopy(public)
    mutation["hidden"] = [{"input": [999], "output": [0]}]
    try:
        select_visible(mutation, candidates)
    except ValueError:
        pass
    else:
        raise AssertionError("selector accepted a hidden-bearing task schema")


def test_cpu_exhaustive_control_reaches_a_visible_passer():
    task = splits()["qualification"][0]
    candidates = [
        {"candidate_id": f"cpu-{index:03d}", "text": program_source(program)}
        for index, program in enumerate(DEPTH_TWO_PROGRAMS)
    ]
    selected = select_visible(public_task(task), candidates)
    assert not selected["abstained"]
    assert selected["eligible_unique_programs"] == task[
        "visible_consistent_program_count"
    ]


def test_mechanics_composition_has_24_distinct_bound_results_per_context():
    value = config()
    maps = []
    for values in value["mechanics"]["diagnostic_inputs"]:
        mapping = {
            canonical_operation(operation): diagnostic_apply(operation, values)
            for operation in CONCRETE_OPERATIONS
        }
        assert len({tuple(result) for result in mapping.values()}) == 24
        maps.append(json.dumps(mapping, sort_keys=True))
    assert len(set(maps)) == 4


def test_model_stages_remain_fail_closed():
    assert config()["boundaries"]["implementation"]["status"] == "locked"
    assert config()["boundaries"]["mechanics"]["status"] == "pending"
