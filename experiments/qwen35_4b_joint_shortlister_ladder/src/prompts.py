from __future__ import annotations

import random
from typing import Any

from .operator_library import OperatorSpec


def inventory_text(
    record: dict[str, Any],
    operators: list[OperatorSpec],
    *,
    shuffled_descriptions: bool = False,
    seed: int = 0,
) -> str:
    descriptions = [operators[operator_index].description for operator_index in record["inventory_permutation"]]
    if shuffled_descriptions:
        rng = random.Random(seed)
        descriptions = list(descriptions)
        rng.shuffle(descriptions)
    return "\n".join(f"{code:03d}: {description}" for code, description in enumerate(descriptions))


def _case_text(case: dict[str, Any]) -> str:
    env = case["input"]
    fields = [f"xs={env['xs']}"]
    if "m" in env:
        fields.append(f"m={env['m']}")
    if "threshold" in env:
        fields.append(f"threshold={env['threshold']}")
    return "; ".join(fields) + f" => {case['expected']}"


def pair_prompt(
    record: dict[str, Any],
    operators: list[OperatorSpec],
    *,
    cases: list[dict[str, Any]] | None = None,
    shuffled_inventory: bool = False,
    shuffle_seed: int = 0,
) -> str:
    prompt_cases = cases if cases is not None else record["visible"]
    case_text = "\n".join(f"{index + 1}. {_case_text(case)}" for index, case in enumerate(prompt_cases))
    inventory = inventory_text(record, operators, shuffled_descriptions=shuffled_inventory, seed=shuffle_seed)
    max_code = int(record["library_size"]) - 1
    return (
        "Select the ordered pair of operator codes that fills LEFT and RIGHT.\n"
        "Every inventory entry has type list[int] -> int. Codes are arbitrary in each record.\n"
        f"Valid codes are 000 through {max_code:03d}. Use the inventory meanings and cases.\n"
        "Return exactly LLL,RRR with three digits per side.\n\n"
        f"TEMPLATE:\n{record['template_text']}\n\n"
        f"CASES:\n{case_text}\n\n"
        f"INVENTORY:\n{inventory}\n\n"
        "PAIR="
    )

