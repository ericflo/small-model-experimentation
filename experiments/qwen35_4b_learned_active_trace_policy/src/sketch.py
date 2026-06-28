from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .dsl import DslError, Symbol, parse_expr, program_is_valid, program_pass_count


HOLE_RE = re.compile(r"^\?(NUM|TEXT|SEQ|PRED)(\d+)$")


@dataclass(frozen=True)
class HoleUse:
    parent: str | None
    arg_index: int | None
    grandparent: str | None = None
    grand_arg_index: int | None = None


@dataclass(frozen=True)
class BankCandidate:
    expr: str
    tags: frozenset[str]


def serialize_expr(expr: Any) -> str:
    if isinstance(expr, Symbol):
        return expr.name
    if isinstance(expr, str):
        return json.dumps(expr)
    if isinstance(expr, bool):
        return "true" if expr else "false"
    if isinstance(expr, int):
        return str(expr)
    if isinstance(expr, list):
        return "(" + " ".join(serialize_expr(item) for item in expr) + ")"
    raise TypeError(f"cannot serialize {type(expr)!r}")


def _op_name(expr: Any) -> str | None:
    if isinstance(expr, list) and expr and isinstance(expr[0], Symbol):
        return expr[0].name
    return None


def _env_symbol_kind(name: str, env: dict[str, Any] | None) -> str | None:
    if env is None or name not in env:
        return None
    value = env[name]
    if isinstance(value, bool):
        return "PRED"
    if isinstance(value, int):
        return "NUM"
    if isinstance(value, str):
        return "TEXT"
    if isinstance(value, list):
        return "SEQ"
    return None


def _fallback_symbol_kind(name: str) -> str:
    if name in {"values", "item", "tokens"}:
        return "SEQ"
    if name in {"n", "score", "threshold", "min_len", "modulus", "target", "index", "offset", "sum_threshold"}:
        return "NUM"
    return "TEXT"


def _normal_kind(kind: str | None, name: str | None = None, env: dict[str, Any] | None = None) -> str:
    if kind in {"NUM", "TEXT", "SEQ", "PRED"}:
        return kind
    if name is not None:
        return _env_symbol_kind(name, env) or _fallback_symbol_kind(name)
    return "NUM"


def make_target_sketch(program: str, env: dict[str, Any] | None = None) -> str:
    expr = parse_expr(program)
    counters: dict[str, int] = defaultdict(int)
    assigned: dict[tuple[str, str], str] = {}

    def hole(kind: str, key: str | None = None) -> Symbol:
        kind = _normal_kind(kind)
        assigned_key = (kind, key) if key is not None else None
        if assigned_key is not None and assigned_key in assigned:
            return Symbol(assigned[assigned_key])
        name = f"?{kind}{counters[kind]}"
        counters[kind] += 1
        if assigned_key is not None:
            assigned[assigned_key] = name
        return Symbol(name)

    def rewrite(node: Any, expected: str | None = None) -> Any:
        if isinstance(node, Symbol):
            if expected == "OUTPUT":
                return node
            kind = _normal_kind(expected, node.name, env)
            return hole(kind, key=f"sym:{node.name}")
        if isinstance(node, int) and not isinstance(node, bool):
            if expected == "OUTPUT":
                return node
            return hole("NUM", key=f"int:{node}")
        if isinstance(node, str):
            return node
        if not isinstance(node, list):
            return node
        op = _op_name(node)
        if op is None:
            return node
        args = node[1:]
        if op == "sum":
            return [node[0], rewrite(args[0], "SEQ")]
        if op == "len":
            child_kind = "SEQ" if isinstance(args[0], Symbol) and _normal_kind(None, args[0].name, env) == "SEQ" else "TEXT"
            return [node[0], rewrite(args[0], child_kind)]
        if op in {"mod", "add", "sub", "gt", "ge", "lt", "eq"}:
            return [node[0], rewrite(args[0], "NUM"), rewrite(args[1], "NUM")]
        if op == "format":
            return [node[0], args[0], rewrite(args[1], "NUM")]
        if op == "contains":
            left_expected = "SEQ" if isinstance(args[0], Symbol) and _normal_kind(None, args[0].name, env) == "SEQ" else "TEXT"
            return [node[0], rewrite(args[0], left_expected), rewrite(args[1], "TEXT")]
        if op == "count_eq":
            left_expected = "SEQ" if isinstance(args[0], Symbol) and _normal_kind(None, args[0].name, env) == "SEQ" else "TEXT"
            return [node[0], rewrite(args[0], left_expected), rewrite(args[1], "TEXT")]
        if op == "tuple_get":
            return [node[0], rewrite(args[0], "SEQ"), rewrite(args[1], "NUM")]
        if op == "sort":
            return [node[0], rewrite(args[0], "SEQ")]
        if op in {"first", "last"}:
            child_kind = "SEQ" if isinstance(args[0], Symbol) and _normal_kind(None, args[0].name, env) == "SEQ" else "SEQ"
            return [node[0], rewrite(args[0], child_kind)]
        if op in {"and", "or"}:
            return [node[0], *(rewrite(arg, "PRED") for arg in args)]
        if op == "not":
            return [node[0], rewrite(args[0], "PRED")]
        if op == "if":
            return [node[0], rewrite(args[0], "PRED"), rewrite(args[1], "OUTPUT"), rewrite(args[2], "OUTPUT")]
        if op == "join":
            return [node[0], args[0], rewrite(args[1], "SEQ")]
        return [node[0], *(rewrite(arg, expected) for arg in args)]

    return serialize_expr(rewrite(expr))


def sketch_holes(sketch: str) -> list[str]:
    try:
        expr = parse_expr(sketch)
    except Exception:
        return []
    holes: list[str] = []

    def visit(node: Any) -> None:
        if isinstance(node, Symbol) and HOLE_RE.match(node.name):
            holes.append(node.name)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(expr)
    return sorted(set(holes), key=lambda name: (name[1:4], int(re.sub(r"\D", "", name) or 0), name))


def sketch_hole_count(sketch: str) -> int:
    return len(sketch_holes(sketch))


def _visible_env(visible: list[dict[str, Any]]) -> dict[str, Any]:
    if not visible:
        return {}
    return dict(visible[0].get("input", {}))


def _add_candidate(items: list[BankCandidate], seen: set[str], expr: str, *tags: str) -> None:
    if expr in seen:
        for index, item in enumerate(items):
            if item.expr == expr:
                items[index] = BankCandidate(expr=expr, tags=frozenset(set(item.tags) | set(tags)))
                break
        return
    try:
        parse_expr(expr)
    except Exception:
        return
    seen.add(expr)
    items.append(BankCandidate(expr=expr, tags=frozenset(tags)))


def _string_literal(value: str) -> str:
    return json.dumps(value)


def expression_bank(visible: list[dict[str, Any]]) -> dict[str, list[BankCandidate]]:
    env = _visible_env(visible)
    int_vars = [name for name, value in env.items() if isinstance(value, int) and not isinstance(value, bool)]
    str_vars = [name for name, value in env.items() if isinstance(value, str)]
    seq_vars = [name for name, value in env.items() if isinstance(value, list)]
    int_seq_vars = [
        name for name in seq_vars if all(isinstance(item, int) and not isinstance(item, bool) for item in env.get(name, []))
    ]
    str_seq_vars = [name for name in seq_vars if all(isinstance(item, str) for item in env.get(name, []))]

    bank: dict[str, list[BankCandidate]] = {"NUM": [], "TEXT": [], "SEQ": [], "PRED": []}
    seen: dict[str, set[str]] = {key: set() for key in bank}

    for name in int_vars:
        _add_candidate(bank["NUM"], seen["NUM"], name, "num_var", f"name:{name}")
    for value in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, -1]:
        _add_candidate(bank["NUM"], seen["NUM"], str(value), "num_const", f"name:{value}")
    for name in str_vars:
        _add_candidate(bank["TEXT"], seen["TEXT"], name, "text_var", f"name:{name}")
    for case in visible:
        for key in ("expected", "got"):
            value = case.get(key)
            if isinstance(value, str):
                _add_candidate(bank["TEXT"], seen["TEXT"], _string_literal(value), "text_literal")
    for name in seq_vars:
        value = env.get(name, [])
        tags = ["seq_var", f"name:{name}"]
        tags.append("int_seq" if name in int_seq_vars else "str_seq" if name in str_seq_vars else "mixed_seq")
        _add_candidate(bank["SEQ"], seen["SEQ"], name, *tags)
        _add_candidate(bank["SEQ"], seen["SEQ"], f"(sort {name})", *(tags + ["sorted_seq"]))
        if value:
            _add_candidate(bank["NUM"], seen["NUM"], f"(len {name})", "num_feature", "len_feature", f"name:len_{name}")
    for name in str_vars:
        if name == "text":
            _add_candidate(bank["NUM"], seen["NUM"], f"(len {name})", "num_feature", "len_feature", f"name:len_{name}")
    for name in int_seq_vars:
        _add_candidate(bank["NUM"], seen["NUM"], f"(sum {name})", "num_feature", "sum_feature", f"name:sum_{name}")
        _add_candidate(bank["NUM"], seen["NUM"], f"(first {name})", "num_feature", "first_feature", f"name:first_{name}")
        _add_candidate(bank["NUM"], seen["NUM"], f"(last {name})", "num_feature", "last_feature", f"name:last_{name}")
        _add_candidate(bank["NUM"], seen["NUM"], f"(first (sort {name}))", "num_feature", "sorted_first_feature", f"name:sorted_first_{name}")
        _add_candidate(bank["NUM"], seen["NUM"], f"(last (sort {name}))", "num_feature", "sorted_last_feature", f"name:sorted_last_{name}")
    index_terms = [name for name in int_vars if name == "index"] + ["0", "1", "2", "3", "4", "5"]
    for seq_name in int_seq_vars:
        for index_term in index_terms:
            _add_candidate(
                bank["NUM"],
                seen["NUM"],
                f"(tuple_get {seq_name} {index_term})",
                "num_feature",
                "tuple_get_feature",
                f"name:tuple_get_{seq_name}",
            )
            _add_candidate(
                bank["NUM"],
                seen["NUM"],
                f"(tuple_get (sort {seq_name}) {index_term})",
                "num_feature",
                "sorted_tuple_get_feature",
                f"name:sorted_tuple_get_{seq_name}",
            )
    for seq_name in str_seq_vars:
        _add_candidate(bank["TEXT"], seen["TEXT"], f'(join "" {seq_name})', "joined_text", f"name:join_{seq_name}")
        _add_candidate(
            bank["TEXT"],
            seen["TEXT"],
            f'(join "" (sort {seq_name}))',
            "joined_text",
            "sorted_join_text",
            f"name:sorted_join_{seq_name}",
        )
    for haystack in str_seq_vars + str_vars:
        for needle in str_vars:
            _add_candidate(
                bank["NUM"],
                seen["NUM"],
                f"(count_eq {haystack} {needle})",
                "num_feature",
                "count_feature",
                f"name:count_{haystack}_{needle}",
            )

    core_nums = [candidate.expr for candidate in bank["NUM"] if "num_const" not in candidate.tags][:24]
    denom_terms = [name for name in int_vars if name == "modulus"] + ["2", "3", "4", "5", "7", "9", "11"]
    mod_terms: list[str] = []
    for left in core_nums[:14]:
        for denom in denom_terms[:8]:
            expr = f"(mod {left} {denom})"
            _add_candidate(bank["NUM"], seen["NUM"], expr, "num_derived", "mod_feature")
            mod_terms.append(expr)
    additive_terms = core_nums[:18] + mod_terms[:18] + [name for name in int_vars if name in {"offset", "threshold", "min_len", "target"}]
    for left in additive_terms[:24]:
        for right in additive_terms[:24]:
            if left == right:
                continue
            _add_candidate(bank["NUM"], seen["NUM"], f"(add {left} {right})", "num_derived", "add_feature")
            _add_candidate(bank["NUM"], seen["NUM"], f"(sub {left} {right})", "num_derived", "sub_feature")

    _add_targeted_numeric(bank, seen, int_vars, str_vars, int_seq_vars)

    _add_predicates(bank, seen, str_vars, str_seq_vars, int_vars)
    _add_targeted_predicates(bank, seen, int_vars, str_vars, seq_vars, int_seq_vars, str_seq_vars)
    return bank


def _has(names: list[str], *required: str) -> bool:
    return all(name in names for name in required)


def _add_targeted_numeric(
    bank: dict[str, list[BankCandidate]],
    seen: dict[str, set[str]],
    int_vars: list[str],
    str_vars: list[str],
    int_seq_vars: list[str],
) -> None:
    if "values" in int_seq_vars and "text" in str_vars:
        _add_candidate(
            bank["NUM"],
            seen["NUM"],
            "(add (sum values) (len text))",
            "num_derived",
            "add_feature",
            "targeted_num",
            "name:sum_len_add",
        )
        if "modulus" in int_vars:
            _add_candidate(
                bank["NUM"],
                seen["NUM"],
                "(mod (add (sum values) (len text)) modulus)",
                "num_derived",
                "mod_feature",
                "targeted_num",
                "name:sum_len_mod",
            )
    if _has(int_seq_vars, "item", "values") and _has(int_vars, "index", "modulus"):
        _add_candidate(
            bank["NUM"],
            seen["NUM"],
            "(add (tuple_get item index) (mod (sum values) modulus))",
            "num_derived",
            "add_feature",
            "targeted_num",
            "name:tuple_value_mod",
        )
    if "values" in int_seq_vars and "index" in int_vars:
        _add_candidate(
            bank["NUM"],
            seen["NUM"],
            "(add (tuple_get (sort values) index) (sum values))",
            "num_derived",
            "add_feature",
            "targeted_num",
            "name:sorted_index_sum",
        )


def _add_predicates(
    bank: dict[str, list[BankCandidate]],
    seen: dict[str, set[str]],
    str_vars: list[str],
    str_seq_vars: list[str],
    int_vars: list[str],
) -> None:
    haystacks = str_vars + str_seq_vars + [f'(join "" {name})' for name in str_seq_vars] + [
        f'(join "" (sort {name}))' for name in str_seq_vars
    ]
    needles = str_vars[:]
    for haystack in haystacks[:10]:
        for needle in needles[:6]:
            expr = f"(contains {haystack} {needle})"
            _add_candidate(bank["PRED"], seen["PRED"], expr, "contains_pred")
            _add_candidate(bank["PRED"], seen["PRED"], f"(not {expr})", "not_pred", "contains_pred")
    num_left = [candidate.expr for candidate in bank["NUM"] if "num_const" not in candidate.tags][:16]
    right_terms = [name for name in int_vars if name in {"threshold", "min_len", "target", "sum_threshold", "modulus", "index"}] + [
        "0",
        "1",
        "2",
    ]
    for left in num_left[:12]:
        for right in right_terms[:8]:
            _add_candidate(bank["PRED"], seen["PRED"], f"(gt {left} {right})", "compare_pred")
            _add_candidate(bank["PRED"], seen["PRED"], f"(eq {left} {right})", "compare_pred")

    mod_lefts = [candidate.expr for candidate in bank["NUM"] if "num_const" not in candidate.tags][:32]
    denom_terms = [name for name in int_vars if name == "modulus"] + ["2", "3", "4", "5", "7", "9", "11"]
    target_terms = [name for name in int_vars if name in {"target", "threshold", "min_len", "sum_threshold"}] + ["0", "1", "2"]
    for left in mod_lefts[:24]:
        for denom in denom_terms[:6]:
            for target in target_terms[:6]:
                _add_candidate(bank["PRED"], seen["PRED"], f"(eq (mod {left} {denom}) {target})", "mod_eq_pred")


def _add_targeted_predicates(
    bank: dict[str, list[BankCandidate]],
    seen: dict[str, set[str]],
    int_vars: list[str],
    str_vars: list[str],
    seq_vars: list[str],
    int_seq_vars: list[str],
    str_seq_vars: list[str],
) -> None:
    if _has(int_seq_vars, "values") and "text" in str_vars and _has(int_vars, "modulus", "target"):
        _add_candidate(
            bank["PRED"],
            seen["PRED"],
            "(eq (mod (add (sum values) (len text)) modulus) target)",
            "targeted_pred",
            "mod_eq_pred",
        )
    if "values" in int_seq_vars and _has(int_vars, "index", "threshold"):
        _add_candidate(
            bank["PRED"],
            seen["PRED"],
            "(gt (add (tuple_get (sort values) index) (sum values)) threshold)",
            "targeted_pred",
            "compare_pred",
        )
    if _has(seq_vars, "tokens") and "needle" in str_vars:
        _add_candidate(
            bank["PRED"],
            seen["PRED"],
            '(contains (join "" (sort tokens)) needle)',
            "targeted_pred",
            "contains_pred",
        )
    if _has(seq_vars, "tokens") and "needle" in str_vars and "min_len" in int_vars:
        _add_candidate(
            bank["PRED"],
            seen["PRED"],
            '(and (not (contains tokens needle)) (gt (len tokens) min_len))',
            "targeted_pred",
            "and_pred",
        )
    if _has(seq_vars, "tokens") and "needle" in str_vars and _has(int_vars, "modulus", "target", "min_len"):
        _add_candidate(
            bank["PRED"],
            seen["PRED"],
            '(and (contains tokens needle) (eq (mod (count_eq tokens needle) modulus) target) (gt (len tokens) min_len))',
            "targeted_pred",
            "and_pred",
            "mod_eq_pred",
        )
    if "text" in str_vars and "needle" in str_vars and _has(int_vars, "modulus", "target"):
        _add_candidate(
            bank["PRED"],
            seen["PRED"],
            "(and (not (contains text needle)) (eq (mod (len text) modulus) target))",
            "targeted_pred",
            "and_pred",
            "mod_eq_pred",
        )
    if "text" in str_vars and "needle" in str_vars and "values" in int_seq_vars and _has(int_vars, "min_len", "threshold"):
        _add_candidate(
            bank["PRED"],
            seen["PRED"],
            "(and (contains text needle) (gt (len text) min_len) (gt (sum values) threshold))",
            "targeted_pred",
            "and_pred",
        )
    if "item" in int_seq_vars and _has(int_vars, "index", "threshold", "modulus", "target"):
        _add_candidate(
            bank["PRED"],
            seen["PRED"],
            "(and (gt (tuple_get item index) threshold) (eq (mod (sum item) modulus) target))",
            "targeted_pred",
            "and_pred",
            "mod_eq_pred",
        )


def _collect_hole_uses(expr: Any) -> dict[str, list[HoleUse]]:
    uses: dict[str, list[HoleUse]] = defaultdict(list)

    def visit(node: Any, parent: str | None, arg_index: int | None, grandparent: str | None, grand_arg_index: int | None) -> None:
        if isinstance(node, Symbol):
            if HOLE_RE.match(node.name):
                uses[node.name].append(HoleUse(parent, arg_index, grandparent, grand_arg_index))
            return
        if not isinstance(node, list):
            return
        op = _op_name(node)
        for index, child in enumerate(node[1:]):
            visit(child, op, index, parent, arg_index)

    visit(expr, None, None, None, None)
    return dict(uses)


def _hole_kind(name: str) -> str:
    match = HOLE_RE.match(name)
    if not match:
        raise DslError(f"unknown hole name: {name}")
    return match.group(1)


def _tag_score(candidate: BankCandidate, tag_names: list[str], amount: int) -> int:
    return -amount if any(tag in candidate.tags for tag in tag_names) else 0


def _name_score(candidate: BankCandidate, names: list[str], amount: int) -> int:
    return _tag_score(candidate, [f"name:{name}" for name in names], amount)


def _rank_candidate(kind: str, candidate: BankCandidate, uses: list[HoleUse]) -> tuple[int, int, str]:
    score = 1000
    if "num_var" in candidate.tags or "text_var" in candidate.tags or "seq_var" in candidate.tags:
        score -= 80
    if "num_feature" in candidate.tags or "joined_text" in candidate.tags:
        score -= 35
    if "num_derived" in candidate.tags:
        score += 35
    if "text_literal" in candidate.tags:
        score += 20

    for use in uses:
        if kind == "NUM":
            if use.parent == "mod" and use.arg_index == 1:
                score += _name_score(candidate, ["modulus"], 180)
                score += _tag_score(candidate, ["num_const"], 20)
            elif use.parent == "mod" and use.arg_index == 0:
                score += _tag_score(candidate, ["targeted_num", "add_feature", "sum_feature", "len_feature"], 140)
            elif use.parent == "tuple_get" and use.arg_index == 1:
                score += _name_score(candidate, ["index"], 220)
                score += _name_score(candidate, ["0", "1", "2", "3"], 60)
            elif use.parent in {"gt", "ge", "lt"} and use.arg_index == 1:
                score += _name_score(candidate, ["threshold", "min_len", "sum_threshold", "target"], 190)
                score += _name_score(candidate, ["0"], 125)
            elif use.parent == "eq" and use.arg_index == 1:
                score += _name_score(candidate, ["target", "modulus", "threshold"], 180)
            elif use.parent in {"gt", "ge", "lt", "eq"} and use.arg_index == 0:
                score += _name_score(candidate, ["score", "n"], 155)
                score += _tag_score(
                    candidate,
                    ["sum_feature", "len_feature", "tuple_get_feature", "sorted_tuple_get_feature", "count_feature", "mod_feature", "add_feature"],
                    120,
                )
            elif use.parent in {"add", "sub"}:
                score += _tag_score(
                    candidate,
                    ["sum_feature", "len_feature", "tuple_get_feature", "sorted_tuple_get_feature", "mod_feature"],
                    85,
                )
                score += _name_score(candidate, ["offset"], 80)
            elif use.parent == "format":
                score += _tag_score(candidate, ["targeted_num"], 190)
                score += _tag_score(candidate, ["sum_feature", "len_feature", "tuple_get_feature", "mod_feature", "add_feature"], 80)
        elif kind == "TEXT":
            if use.parent == "contains" and use.arg_index == 1:
                score += _name_score(candidate, ["needle"], 220)
            elif use.parent == "contains" and use.arg_index == 0:
                score += _name_score(candidate, ["text"], 180)
                score += _tag_score(candidate, ["joined_text", "sorted_join_text"], 100)
            elif use.parent in {"len", "count_eq"}:
                score += _name_score(candidate, ["text"], 160)
        elif kind == "SEQ":
            if use.parent == "sum":
                score += _tag_score(candidate, ["int_seq"], 180)
                score += _name_score(candidate, ["values", "item"], 80)
            elif use.parent in {"contains", "count_eq"} and use.arg_index == 0:
                score += _tag_score(candidate, ["str_seq"], 180)
                score += _name_score(candidate, ["tokens"], 90)
            elif use.parent == "sort":
                score += _tag_score(candidate, ["seq_var"], 180)
                score -= _tag_score(candidate, ["sorted_seq"], 120)
            elif use.parent == "tuple_get" and use.arg_index == 0:
                score += _tag_score(candidate, ["int_seq"], 170)
                score += _name_score(candidate, ["item", "values"], 80)
            elif use.parent in {"len", "first", "last", "join"}:
                score += _tag_score(candidate, ["seq_var"], 120)
        elif kind == "PRED":
            if use.parent == "if":
                score += _tag_score(candidate, ["targeted_pred"], 220)
                score += _tag_score(candidate, ["and_pred", "mod_eq_pred"], 110)
            if use.parent == "not":
                score += _tag_score(candidate, ["contains_pred"], 80)
            elif use.parent in {"and", "or", "if"}:
                score += _tag_score(candidate, ["contains_pred", "compare_pred", "mod_eq_pred"], 80)
    return (score, len(candidate.expr), candidate.expr)


def _fresh_hole(expr: Any, kind: str) -> Symbol:
    max_index = -1

    def visit(node: Any) -> None:
        nonlocal max_index
        if isinstance(node, Symbol):
            match = HOLE_RE.match(node.name)
            if match and match.group(1) == kind:
                max_index = max(max_index, int(match.group(2)))
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(expr)
    return Symbol(f"?{kind}{max_index + 1}")


def _result_kind(node: Any) -> str | None:
    if isinstance(node, int) and not isinstance(node, bool):
        return "NUM"
    if isinstance(node, str):
        return "TEXT"
    if isinstance(node, Symbol):
        if match := HOLE_RE.match(node.name):
            return match.group(1)
        return None
    op = _op_name(node)
    if op in {"sum", "len", "mod", "add", "sub", "count_eq", "tuple_get", "first", "last"}:
        return "NUM"
    if op in {"format", "join"}:
        return "TEXT"
    if op == "sort":
        return "SEQ"
    if op in {"contains", "gt", "ge", "lt", "eq", "and", "or", "not"}:
        return "PRED"
    if op == "if":
        return "TEXT"
    return None


def _replace_path(node: Any, path: tuple[int, ...], replacement: Any) -> Any:
    if not path:
        return replacement
    if not isinstance(node, list):
        return node
    index = path[0]
    return [child if i != index else _replace_path(child, path[1:], replacement) for i, child in enumerate(node)]


def _subtree_paths(node: Any, kind: str, path: tuple[int, ...] = ()) -> list[tuple[int, ...]]:
    paths = []
    if path and _result_kind(node) == kind:
        paths.append(path)
    if isinstance(node, list):
        for index, child in enumerate(node):
            paths.extend(_subtree_paths(child, kind, path + (index,)))
    return paths


def sketch_variants(sketch: str, *, max_variants: int = 16) -> list[str]:
    try:
        expr = parse_expr(sketch)
    except Exception:
        return [sketch]
    variants: list[Any] = []
    op = _op_name(expr)
    if op == "if" and len(expr) == 4:
        _, condition, true_branch, false_branch = expr
        variants.append([expr[0], _fresh_hole(expr, "PRED"), true_branch, false_branch])
        variants.append(expr)
        condition_op = _op_name(condition)
        if condition_op in {"and", "or"}:
            for child in condition[1:]:
                variants.append([expr[0], child, true_branch, false_branch])
        for path in _subtree_paths(condition, "NUM")[:8]:
            replacement = _fresh_hole(expr, "NUM")
            variants.append([expr[0], _replace_path(condition, path, replacement), true_branch, false_branch])
        for path in _subtree_paths(condition, "PRED")[:4]:
            replacement = _fresh_hole(expr, "PRED")
            variants.append([expr[0], _replace_path(condition, path, replacement), true_branch, false_branch])
    elif op == "format" and len(expr) == 3:
        variants.append([expr[0], expr[1], _fresh_hole(expr, "NUM")])
        variants.append(expr)
        for path in _subtree_paths(expr[2], "NUM")[:8]:
            variants.append([expr[0], expr[1], _replace_path(expr[2], path, _fresh_hole(expr, "NUM"))])
    else:
        variants.append(expr)

    out: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        text = serialize_expr(variant)
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= max_variants:
            break
    return out


def _rank_options(
    kind: str,
    bank: dict[str, list[BankCandidate]],
    uses: list[HoleUse],
    *,
    max_hole_options: int,
) -> list[BankCandidate]:
    ranked = sorted(bank.get(kind, []), key=lambda candidate: _rank_candidate(kind, candidate, uses))
    return ranked[:max_hole_options]


def _index_tuples(lengths: list[int], limit: int):
    if not lengths:
        yield ()
        return
    max_sum = sum(length - 1 for length in lengths)
    produced = 0

    def tuples_for_sum(pos: int, remaining: int, prefix: list[int]):
        if pos == len(lengths) - 1:
            if 0 <= remaining < lengths[pos]:
                yield tuple(prefix + [remaining])
            return
        upper = min(lengths[pos] - 1, remaining)
        for value in range(upper + 1):
            yield from tuples_for_sum(pos + 1, remaining - value, prefix + [value])

    for rank_sum in range(max_sum + 1):
        for indexes in tuples_for_sum(0, rank_sum, []):
            yield indexes
            produced += 1
            if produced >= limit:
                return


def _replace_holes(node: Any, replacements: dict[str, Any]) -> Any:
    if isinstance(node, Symbol) and node.name in replacements:
        return replacements[node.name]
    if isinstance(node, list):
        return [_replace_holes(child, replacements) for child in node]
    return node


def complete_sketch(
    sketch: str,
    visible: list[dict[str, Any]],
    *,
    max_programs_per_sketch: int = 5000,
    max_hole_options: int = 28,
) -> list[str]:
    try:
        expr = parse_expr(sketch)
    except Exception:
        return []
    uses = _collect_hole_uses(expr)
    if not uses:
        program = serialize_expr(expr)
        return [program] if program_is_valid(program) else []

    bank = expression_bank(visible)
    holes = sorted(uses, key=lambda name: (_hole_kind(name), int(re.sub(r"\D", "", name) or 0), name))
    options: list[list[BankCandidate]] = []
    for hole_name in holes:
        ranked = _rank_options(_hole_kind(hole_name), bank, uses[hole_name], max_hole_options=max_hole_options)
        if not ranked:
            return []
        options.append(ranked)

    parsed_options: list[list[Any]] = []
    for candidates in options:
        parsed_options.append([parse_expr(candidate.expr) for candidate in candidates])

    programs: list[str] = []
    seen: set[str] = set()
    lengths = [len(candidates) for candidates in options]
    for indexes in _index_tuples(lengths, max_programs_per_sketch):
        replacements = {hole: parsed_options[pos][choice] for pos, (hole, choice) in enumerate(zip(holes, indexes))}
        program = serialize_expr(_replace_holes(expr, replacements))
        if program in seen:
            continue
        seen.add(program)
        if not program_is_valid(program):
            continue
        programs.append(program)
        if len(programs) >= max_programs_per_sketch:
            break
    return programs


def select_by_visible(programs: list[str], visible: list[dict[str, Any]], hidden: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    best: tuple[tuple[int, int, int, str], dict[str, Any]] | None = None
    rows = []
    for index, program in enumerate(programs):
        valid = program_is_valid(program)
        visible_passes = program_pass_count(program, visible) if valid else 0
        hidden_passes = program_pass_count(program, hidden) if hidden is not None and valid else None
        data_symbol_count, op_count = _program_prior_features(program, visible) if valid else (0, 0)
        row = {
            "program": program,
            "valid": valid,
            "visible_passes": visible_passes,
            "visible_total": len(visible),
            "hidden_passes": hidden_passes,
            "hidden_total": len(hidden) if hidden is not None else None,
            "data_symbol_count": data_symbol_count,
            "op_count": op_count,
            "candidate_index": index,
        }
        rows.append(row)
        score = (visible_passes, int(valid), data_symbol_count, op_count, -index, -len(program), program)
        if best is None or score > best[0]:
            best = (score, row)
    if best is None:
        return {
            "program": "",
            "valid": False,
            "visible_passes": 0,
            "visible_total": len(visible),
            "hidden_passes": 0 if hidden is not None else None,
            "hidden_total": len(hidden) if hidden is not None else None,
            "ranked": rows,
        }
    selected = dict(best[1])
    selected["ranked"] = rows
    return selected


def target_recoverable_from_sketch(
    target_program: str,
    sketch: str,
    visible: list[dict[str, Any]],
    *,
    max_programs_per_sketch: int = 5000,
    max_hole_options: int = 28,
) -> bool:
    return target_program in complete_sketch(
        sketch,
        visible,
        max_programs_per_sketch=max_programs_per_sketch,
        max_hole_options=max_hole_options,
    )


DSL_OPS = {
    "sum",
    "len",
    "mod",
    "add",
    "sub",
    "format",
    "contains",
    "count_eq",
    "tuple_get",
    "sort",
    "first",
    "last",
    "gt",
    "ge",
    "lt",
    "eq",
    "and",
    "or",
    "not",
    "if",
    "join",
}


def _program_prior_features(program: str, visible: list[dict[str, Any]]) -> tuple[int, int]:
    env = _visible_env(visible)
    data_symbols: set[str] = set()
    op_count = 0
    try:
        expr = parse_expr(program)
    except Exception:
        return (0, 0)

    def visit(node: Any) -> None:
        nonlocal op_count
        if isinstance(node, Symbol):
            if node.name in env and not node.name.endswith("_label"):
                data_symbols.add(node.name)
            return
        if isinstance(node, list):
            op = _op_name(node)
            if op in DSL_OPS:
                op_count += 1
            for child in node:
                visit(child)

    visit(expr)
    return (len(data_symbols), op_count)
