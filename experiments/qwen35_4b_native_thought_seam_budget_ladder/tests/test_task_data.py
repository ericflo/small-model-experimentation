from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path

import yaml


EXP = Path(__file__).resolve().parents[1]
SRC = EXP / "src"
sys.path.insert(0, str(SRC))

from task_data import (  # noqa: E402
    IDENTIFIABLE_FIRST_OPERATIONS,
    _matching_first_types,
    build_splits,
    task_fingerprint,
    task_prompt,
)


def config() -> dict:
    return yaml.safe_load((EXP / "configs" / "default.yaml").read_text())


def test_fresh_split_contract_and_visible_identifiability() -> None:
    value = config()
    splits = build_splits(value)
    assert {name: len(rows) for name, rows in splits.items()} == {
        "budget_selection": 16,
        "seam_confirmation": 24,
    }
    fingerprints = [task_fingerprint(row) for rows in splits.values() for row in rows]
    assert len(fingerprints) == len(set(fingerprints)) == 40
    for rows in splits.values():
        counts = Counter(row["first_op"] for row in rows)
        support = [counts[name] for name in IDENTIFIABLE_FIRST_OPERATIONS]
        assert max(support) - min(support) <= 1
        for row in rows:
            outputs = tuple(tuple(example["output"]) for example in row["visible"])
            inputs = [example["input"] for example in row["visible"]]
            assert _matching_first_types(inputs, outputs, depth=2) == {row["first_op"]}


def test_prompt_inherits_parent_grammar_without_literal_special_tokens() -> None:
    value = config()
    task = build_splits(value)["budget_selection"][0]
    prompt = task_prompt(task, value["data"]["operation_aliases"])
    assert prompt.startswith(
        "Infer the hidden sequence of exactly two list operations from these examples:\n"
    )
    assert prompt.endswith(
        "Reason naturally in the private reasoning section. Then answer with exactly "
        "`First: <alias>` and no other final text."
    )
    assert "<think>" not in prompt and "</think>" not in prompt
