from __future__ import annotations

import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from task_data import (  # noqa: E402
    CONCEPTS,
    consequence_prompt,
    digit_for,
    direct_prompt,
    fingerprint,
    generate_splits,
    shared_prefix,
)


def config() -> dict:
    return yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))


def test_splits_are_deterministic_balanced_and_disjoint() -> None:
    first = generate_splits(config())
    second = generate_splits(config())
    assert first == second
    expected = {"lens_fit": 48, "band_selection": 24, "confirmation": 48}
    assert {name: len(rows) for name, rows in first.items()} == expected
    seen = set()
    for rows in first.values():
        counts = {concept: sum(row["source"] == concept for row in rows) for concept in CONCEPTS}
        assert len(set(counts.values())) == 1
        for row in rows:
            assert fingerprint(row) not in seen
            seen.add(fingerprint(row))


def test_prompt_contract_and_counterfactuals() -> None:
    row = generate_splits(config())["confirmation"][0]
    prefix = shared_prefix(row)
    assert direct_prompt(row).startswith(prefix)
    assert consequence_prompt(row).startswith(prefix)
    assert direct_prompt(row, selected=row["target"]).count(f"Selected key: {row['target']}") == 1
    assert digit_for(row, row["source"]) == row["source_digit"]
    assert digit_for(row, row["target"]) == row["target_digit"]
    assert len({row["source_digit"], row["target_digit"], row["wrong_digit"]}) == 3
