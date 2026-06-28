from __future__ import annotations

import random
from typing import Any

from .operator_library import OperatorSpec


def inventory_text(operators: list[OperatorSpec], *, shuffled_descriptions: bool = False, seed: int = 0) -> str:
    descriptions = [operator.description for operator in operators]
    if shuffled_descriptions:
        rng = random.Random(seed)
        descriptions = list(descriptions)
        rng.shuffle(descriptions)
    lines = []
    for operator, description in zip(operators, descriptions):
        lines.append(f"{operator.alias.removeprefix('op_')}: {description}")
    return "\n".join(lines)


def _case_text(case: dict[str, Any]) -> str:
    env = case["input"]
    fields = [f"xs={env['xs']}"]
    if "m" in env:
        fields.append(f"m={env['m']}")
    if "threshold" in env:
        fields.append(f"threshold={env['threshold']}")
    return "; ".join(fields) + f" => {case['expected']}"


def slot_prompt(
    record: dict[str, Any],
    operators: list[OperatorSpec],
    slot: str,
    *,
    shuffled_inventory: bool = False,
    shuffle_seed: int = 0,
) -> str:
    cases = "\n".join(f"{index + 1}. {_case_text(case)}" for index, case in enumerate(record["visible"]))
    inventory = inventory_text(operators, shuffled_descriptions=shuffled_inventory, seed=shuffle_seed)
    return (
        "You are selecting one operator code from an inventory.\n"
        "Every operator has type list[int] -> int. Codes are arbitrary; use the inventory descriptions and examples.\n"
        "Return exactly the three digits after op_.\n\n"
        f"TEMPLATE:\n{record['template_text']}\n\n"
        f"VISIBLE CASES:\n{cases}\n\n"
        f"INVENTORY:\n{inventory}\n\n"
        f"QUESTION:\nWhich alias fills the {slot} slot?\nANSWER=op_"
    )
