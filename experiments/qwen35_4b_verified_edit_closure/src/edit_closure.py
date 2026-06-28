from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from typing import Any

from .dsl import DslError, Symbol, parse_expr, program_is_valid, program_pass_count


BOOL_OPS = {"contains", "gt", "ge", "lt", "eq", "and", "or", "not"}
COMPARATOR_SWAPS = {
    "gt": ("ge", "lt", "eq"),
    "ge": ("gt", "lt", "eq"),
    "lt": ("gt", "ge", "eq"),
    "eq": ("gt", "lt", "ge"),
}


@dataclass(frozen=True)
class ClosureConfig:
    max_variants_per_seed: int = 900
    max_total_variants: int = 2400
    rounds: int = 2


def serialize_expr(expr: Any) -> str:
    if isinstance(expr, Symbol):
        return expr.name
    if isinstance(expr, str):
        return json.dumps(expr)
    if isinstance(expr, bool):
        return "true" if expr else "false"
    if isinstance(expr, list):
        return "(" + " ".join(serialize_expr(item) for item in expr) + ")"
    return str(expr)


def op_name(expr: Any) -> str | None:
    if isinstance(expr, list) and expr and isinstance(expr[0], Symbol):
        return expr[0].name
    return None


def sym(name: str) -> Symbol:
    return Symbol(name)


def stable_key(expr: Any) -> str:
    try:
        return serialize_expr(expr)
    except Exception:
        return repr(expr)


def is_symbol(expr: Any, name: str | None = None) -> bool:
    return isinstance(expr, Symbol) and (name is None or expr.name == name)


def contains_symbol(expr: Any, name: str) -> bool:
    if is_symbol(expr, name):
        return True
    if isinstance(expr, list):
        return any(contains_symbol(item, name) for item in expr)
    return False


def first_input(cases: list[dict[str, Any]]) -> dict[str, Any]:
    return cases[0]["input"] if cases else {}


def expression_bank(env: dict[str, Any]) -> dict[str, list[Any]]:
    numeric: list[Any] = []
    sequence: list[Any] = []
    textlike: list[Any] = []
    predicates: list[Any] = []
    constants: list[Any] = [0, 1, 2]

    for name, value in env.items():
        node = sym(name)
        if isinstance(value, int):
            numeric.append(node)
            constants.append(node)
        elif isinstance(value, str):
            textlike.append(node)
            sequence.append(node)
            numeric.append([sym("len"), node])
        elif isinstance(value, list):
            sequence.append(node)
            numeric.append([sym("len"), node])
            if value and all(isinstance(item, int) for item in value):
                numeric.extend([[sym("sum"), node], [sym("first"), node], [sym("last"), node]])
                sequence.append([sym("sort"), node])
            if value and all(isinstance(item, str) for item in value):
                textlike.append([sym("join"), "", node])

    for seq in sequence:
        if "needle" in env:
            predicates.append([sym("contains"), seq, sym("needle")])
            numeric.append([sym("count_eq"), seq, sym("needle")])
        if "index" in env:
            numeric.append([sym("tuple_get"), seq, sym("index")])
            if op_name(seq) != "sort":
                numeric.append([sym("tuple_get"), [sym("sort"), seq], sym("index")])

    for left in list(numeric):
        for right_name in ("threshold", "target", "min_len", "sum_threshold", "offset"):
            if right_name in env:
                predicates.extend(
                    [
                        [sym("gt"), left, sym(right_name)],
                        [sym("ge"), left, sym(right_name)],
                        [sym("lt"), left, sym(right_name)],
                        [sym("eq"), left, sym(right_name)],
                    ]
                )
        if "modulus" in env:
            modded = [sym("mod"), left, sym("modulus")]
            numeric.append(modded)
            if "target" in env:
                predicates.append([sym("eq"), modded, sym("target")])

    for pred in list(predicates):
        predicates.append([sym("not"), pred])

    dedup = lambda items: list({stable_key(item): item for item in items}.values())
    return {
        "numeric": dedup(numeric + constants),
        "sequence": dedup(sequence),
        "textlike": dedup(textlike),
        "predicate": dedup(predicates),
    }


def local_mutations(expr: Any, bank: dict[str, list[Any]]) -> list[Any]:
    mutations: list[Any] = []
    op = op_name(expr)
    if op is None:
        if isinstance(expr, Symbol):
            substitutions = {
                "values": ["item", "tokens"],
                "item": ["values", "tokens"],
                "tokens": ["text", "values"],
                "text": ["tokens"],
                "threshold": ["target", "min_len", "sum_threshold"],
                "target": ["threshold", "min_len"],
                "min_len": ["threshold", "target"],
                "sum_threshold": ["threshold"],
                "high_label": ["low_label"],
                "low_label": ["high_label"],
            }
            for name in substitutions.get(expr.name, []):
                mutations.append(sym(name))
        return mutations

    args = expr[1:]
    if op in COMPARATOR_SWAPS and len(args) == 2:
        for next_op in COMPARATOR_SWAPS[op]:
            mutations.append([sym(next_op), copy.deepcopy(args[0]), copy.deepcopy(args[1])])
        mutations.append([sym(op), copy.deepcopy(args[1]), copy.deepcopy(args[0])])

    if op == "contains" and len(args) == 2:
        haystack, needle = args
        mutations.append([sym("contains"), copy.deepcopy(needle), copy.deepcopy(haystack)])
        if is_symbol(haystack, "tokens"):
            mutations.append([sym("contains"), [sym("join"), "", sym("tokens")], copy.deepcopy(needle)])
        if op_name(haystack) == "join" and len(haystack) == 3:
            mutations.append([sym("contains"), copy.deepcopy(haystack[2]), copy.deepcopy(needle)])
        mutations.append([sym("not"), copy.deepcopy(expr)])

    if op == "not" and len(args) == 1:
        mutations.append(copy.deepcopy(args[0]))
    elif op in BOOL_OPS:
        mutations.append([sym("not"), copy.deepcopy(expr)])

    if op == "and" and len(args) >= 2:
        for index in range(len(args)):
            kept = [copy.deepcopy(arg) for i, arg in enumerate(args) if i != index]
            mutations.append([sym("and"), *kept] if len(kept) >= 2 else kept[0])
        for pred in bank["predicate"][:24]:
            if not any(stable_key(pred) == stable_key(arg) for arg in args):
                mutations.append([sym("and"), *copy.deepcopy(args), copy.deepcopy(pred)])
        mutations.append([sym("or"), *copy.deepcopy(args)])

    if op == "or" and len(args) >= 2:
        mutations.append([sym("and"), *copy.deepcopy(args)])

    if op == "len" and len(args) == 1:
        target = args[0]
        if contains_symbol(target, "text") or is_symbol(target, "tokens"):
            mutations.append([sym("count_eq"), copy.deepcopy(target), sym("needle")])
        if is_symbol(target, "values") or is_symbol(target, "item"):
            mutations.append([sym("sum"), copy.deepcopy(target)])
        for numeric in bank["numeric"][:18]:
            mutations.append(copy.deepcopy(numeric))

    if op == "count_eq" and len(args) == 2:
        mutations.append([sym("len"), copy.deepcopy(args[0])])
        mutations.append([sym("contains"), copy.deepcopy(args[0]), copy.deepcopy(args[1])])

    if op == "sum" and len(args) == 1:
        mutations.extend(
            [
                [sym("len"), copy.deepcopy(args[0])],
                [sym("first"), copy.deepcopy(args[0])],
                [sym("last"), copy.deepcopy(args[0])],
            ]
        )

    if op == "tuple_get" and len(args) == 2:
        source, index = args
        if op_name(source) == "sort" and len(source) == 2:
            mutations.append([sym("tuple_get"), copy.deepcopy(source[1]), copy.deepcopy(index)])
        else:
            mutations.append([sym("tuple_get"), [sym("sort"), copy.deepcopy(source)], copy.deepcopy(index)])
        mutations.append([sym("sum"), copy.deepcopy(source)])
        mutations.append([sym("first"), copy.deepcopy(source)])

    if op == "sort" and len(args) == 1:
        mutations.append(copy.deepcopy(args[0]))

    if op == "mod" and len(args) == 2:
        value, modulus = args
        mutations.append(copy.deepcopy(value))
        if op_name(value) == "add" and len(value) == 3:
            mutations.append([sym("add"), [sym("mod"), copy.deepcopy(value[1]), copy.deepcopy(modulus)], copy.deepcopy(value[2])])
            mutations.append([sym("add"), copy.deepcopy(value[1]), [sym("mod"), copy.deepcopy(value[2]), copy.deepcopy(modulus)]])
        for numeric in bank["numeric"][:24]:
            mutations.append([sym("mod"), copy.deepcopy(numeric), copy.deepcopy(modulus)])

    if op == "add" and len(args) == 2:
        left, right = args
        mutations.append([sym("sub"), copy.deepcopy(left), copy.deepcopy(right)])
        if "modulus" in [item.name for item in bank["numeric"] if isinstance(item, Symbol)]:
            pass
        if op_name(left) == "mod" and len(left) == 3:
            mutations.append([sym("mod"), [sym("add"), copy.deepcopy(left[1]), copy.deepcopy(right)], copy.deepcopy(left[2])])
        if op_name(right) == "mod" and len(right) == 3:
            mutations.append([sym("mod"), [sym("add"), copy.deepcopy(left), copy.deepcopy(right[1])], copy.deepcopy(right[2])])

    if op == "sub" and len(args) == 2:
        mutations.append([sym("add"), copy.deepcopy(args[0]), copy.deepcopy(args[1])])

    if op == "if" and len(args) == 3:
        cond, yes, no = args
        mutations.append([sym("if"), [sym("not"), copy.deepcopy(cond)], copy.deepcopy(no), copy.deepcopy(yes)])
        mutations.append([sym("if"), copy.deepcopy(cond), copy.deepcopy(no), copy.deepcopy(yes)])
        for pred in bank["predicate"][:24]:
            mutations.append([sym("if"), copy.deepcopy(pred), copy.deepcopy(yes), copy.deepcopy(no)])

    return list({stable_key(item): item for item in mutations}.values())


def replace_each(expr: Any, bank: dict[str, list[Any]]) -> list[Any]:
    variants = []
    for mutation in local_mutations(expr, bank):
        variants.append(mutation)
    if isinstance(expr, list):
        for index, child in enumerate(expr):
            for child_variant in replace_each(child, bank):
                next_expr = copy.deepcopy(expr)
                next_expr[index] = child_variant
                variants.append(next_expr)
    return list({stable_key(item): item for item in variants}.values())


def expression_is_viable(program: str) -> bool:
    if len(program) > 512:
        return False
    if not program.startswith("("):
        return False
    return program_is_valid(program)


def closure_programs(seed_program: str, visible: list[dict[str, Any]], config: ClosureConfig) -> list[str]:
    try:
        root = parse_expr(seed_program)
    except Exception as exc:
        raise DslError(str(exc)) from exc

    bank = expression_bank(first_input(visible))
    seen: set[str] = {serialize_expr(root)}
    frontier = [root]
    programs = [serialize_expr(root)]

    for _ in range(config.rounds):
        next_frontier = []
        for expr in frontier:
            for variant in replace_each(expr, bank):
                text = serialize_expr(variant)
                if text in seen or not expression_is_viable(text):
                    continue
                seen.add(text)
                programs.append(text)
                next_frontier.append(variant)
                if len(next_frontier) >= config.max_variants_per_seed:
                    break
            if len(next_frontier) >= config.max_variants_per_seed:
                break
        frontier = next_frontier
        if not frontier or len(programs) >= config.max_total_variants:
            break
    return programs[: config.max_total_variants]


def select_by_visible(programs: list[str], visible: list[dict[str, Any]]) -> dict[str, Any]:
    best: tuple[tuple[int, int], dict[str, Any]] | None = None
    for program in programs:
        valid = program_is_valid(program)
        visible_passes = program_pass_count(program, visible) if valid else 0
        score = (visible_passes, int(valid))
        item = {
            "program": program,
            "valid": valid,
            "visible_passes": visible_passes,
            "visible_total": len(visible),
        }
        if best is None or score > best[0]:
            best = (score, item)
    assert best is not None
    return best[1]
