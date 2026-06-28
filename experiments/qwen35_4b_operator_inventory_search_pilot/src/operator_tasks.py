from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from .dsl import execute


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    status: str
    signature: str = "list[int] -> int"


@dataclass(frozen=True)
class TemplateSpec:
    name: str
    depth: int
    prefix: str


OPERATORS = [
    OperatorSpec("sum", "in_bank"),
    OperatorSpec("first", "in_bank"),
    OperatorSpec("last", "in_bank"),
    OperatorSpec("max", "held_out"),
    OperatorSpec("min", "held_out"),
    OperatorSpec("prod", "held_out"),
    OperatorSpec("gcd", "held_out"),
]

CLOSED_OPERATOR_NAMES = ["sum", "first", "last"]
FULL_OPERATOR_NAMES = [op.name for op in OPERATORS]

TEMPLATES = [
    TemplateSpec("mod_format", depth=2, prefix="M"),
    TemplateSpec("offset_format", depth=2, prefix="A"),
    TemplateSpec("threshold_gate", depth=2, prefix="G"),
]


def operator_status(name: str) -> str:
    for op in OPERATORS:
        if op.name == name:
            return op.status
    raise KeyError(name)


def _values(rng: random.Random, *, min_len: int = 3, max_len: int = 6) -> list[int]:
    # Positive values keep prod and gcd numerically stable while preserving
    # ambiguity among list[int] -> int aggregates.
    return [rng.randint(1, 9) for _ in range(rng.randint(min_len, max_len))]


def _near(value: int, rng: random.Random, radius: int = 4) -> int:
    return value + rng.choice([delta for delta in range(-radius, radius + 1) if delta != 0])


def program_for(template: str, operator: str) -> str:
    if template == "mod_format":
        return f'(format "M{{}}" (mod ({operator} xs) m))'
    if template == "offset_format":
        return f'(format "A{{}}" (add ({operator} xs) offset))'
    if template == "threshold_gate":
        return f"(if (gt ({operator} xs) threshold) hi lo)"
    raise KeyError(template)


def _make_input(template: str, operator: str, rng: random.Random) -> dict[str, Any]:
    xs = _values(rng)
    env: dict[str, Any] = {"xs": xs}
    if template == "mod_format":
        env["m"] = rng.randint(2, 13)
    elif template == "offset_format":
        env["offset"] = rng.randint(-12, 12)
    elif template == "threshold_gate":
        op_value = execute(f"({operator} xs)", env)
        env["threshold"] = max(0, _near(int(op_value), rng))
        env["hi"] = rng.choice(["HIGH", "YES", "KEEP", "PASS"])
        env["lo"] = rng.choice(["LOW", "NO", "DROP", "FAIL"])
    else:
        raise KeyError(template)
    return env


def _cases(program: str, template: str, operator: str, rng: random.Random, count: int) -> list[dict[str, Any]]:
    rows = []
    attempts = 0
    while len(rows) < count:
        attempts += 1
        if attempts > count * 200:
            raise RuntimeError(f"could not generate cases for {template}/{operator}")
        env = _make_input(template, operator, rng)
        try:
            expected = execute(program, env)
        except Exception:
            continue
        rows.append({"input": env, "expected": expected})
    return rows


def operator_hole_sketch(template: str) -> str:
    if template == "mod_format":
        return '(format "M{}" (mod (?OP_LIST_INT0 ?SEQ0) ?NUM0))'
    if template == "offset_format":
        return '(format "A{}" (add (?OP_LIST_INT0 ?SEQ0) ?NUM0))'
    if template == "threshold_gate":
        return "(if (gt (?OP_LIST_INT0 ?SEQ0) ?NUM0) hi lo)"
    raise KeyError(template)


def closed_vocab_sketch(template: str) -> str:
    # Current closed-vocabulary behavior: same skeleton, but the operator
    # search space is restricted to known aggregate names.
    return operator_hole_sketch(template)


def build_records(
    *,
    records_per_family: int,
    visible_count: int = 6,
    hidden_count: int = 18,
    query_count: int = 48,
    seed: int = 20260624,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    records = []
    for template in TEMPLATES:
        for operator in OPERATORS:
            program = program_for(template.name, operator.name)
            for index in range(records_per_family):
                cases = _cases(program, template.name, operator.name, rng, visible_count + hidden_count + query_count)
                records.append(
                    {
                        "id": f"{operator.status}_{operator.name}_{template.name}_{index:04d}",
                        "family": f"{operator.name}_{template.name}",
                        "operator": operator.name,
                        "operator_status": operator.status,
                        "operator_signature": operator.signature,
                        "template": template.name,
                        "composition_depth": template.depth,
                        "target_program": program,
                        "operator_hole_sketch": operator_hole_sketch(template.name),
                        "closed_vocab_sketch": closed_vocab_sketch(template.name),
                        "visible": cases[:visible_count],
                        "hidden": cases[visible_count : visible_count + hidden_count],
                        "query_pool": cases[visible_count + hidden_count :],
                    }
                )
    rng.shuffle(records)
    return records

