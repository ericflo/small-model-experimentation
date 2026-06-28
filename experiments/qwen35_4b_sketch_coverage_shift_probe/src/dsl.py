from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


class DslError(ValueError):
    pass


@dataclass(frozen=True)
class Symbol:
    name: str


TOKEN_RE = re.compile(r'"(?:\\.|[^"\\])*"|[()]|[^\s()]+')


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def parse_atom(token: str) -> Any:
    if token.startswith('"') and token.endswith('"'):
        return json.loads(token)
    if re.fullmatch(r"-?\d+", token):
        return int(token)
    return Symbol(token)


def parse_expr(text: str) -> Any:
    tokens = tokenize(text)
    if not tokens:
        raise DslError("empty program")

    def parse_at(index: int) -> tuple[Any, int]:
        if index >= len(tokens):
            raise DslError("unexpected end")
        token = tokens[index]
        if token == "(":
            values = []
            index += 1
            while index < len(tokens) and tokens[index] != ")":
                value, index = parse_at(index)
                values.append(value)
            if index >= len(tokens):
                raise DslError("missing close paren")
            return values, index + 1
        if token == ")":
            raise DslError("unexpected close paren")
        return parse_atom(token), index + 1

    expr, next_index = parse_at(0)
    if next_index != len(tokens):
        raise DslError("trailing tokens")
    return expr


def eval_expr(expr: Any, env: dict[str, Any]) -> Any:
    if isinstance(expr, Symbol):
        if expr.name in env:
            return env[expr.name]
        raise DslError(f"unknown symbol: {expr.name}")
    if not isinstance(expr, list):
        return expr
    if not expr:
        raise DslError("empty expression")
    op_node = expr[0]
    if not isinstance(op_node, Symbol):
        raise DslError("operator must be symbol")
    op = op_node.name
    args = expr[1:]

    def need(n: int) -> None:
        if len(args) != n:
            raise DslError(f"{op} expects {n}, got {len(args)}")

    if op == "sum":
        need(1)
        return sum(eval_expr(args[0], env))
    if op == "len":
        need(1)
        return len(eval_expr(args[0], env))
    if op == "mod":
        need(2)
        denom = eval_expr(args[1], env)
        if denom == 0:
            raise DslError("mod by zero")
        return eval_expr(args[0], env) % denom
    if op == "add":
        need(2)
        return eval_expr(args[0], env) + eval_expr(args[1], env)
    if op == "sub":
        need(2)
        return eval_expr(args[0], env) - eval_expr(args[1], env)
    if op == "mul":
        need(2)
        return eval_expr(args[0], env) * eval_expr(args[1], env)
    if op == "abs":
        need(1)
        return abs(eval_expr(args[0], env))
    if op == "max":
        need(1)
        values = eval_expr(args[0], env)
        if not values:
            raise DslError("max of empty")
        return max(values)
    if op == "min":
        need(1)
        values = eval_expr(args[0], env)
        if not values:
            raise DslError("min of empty")
        return min(values)
    if op == "prod":
        need(1)
        out = 1
        for value in eval_expr(args[0], env):
            out *= value
        return out
    if op == "format":
        need(2)
        return str(eval_expr(args[0], env)).format(eval_expr(args[1], env))
    if op == "contains":
        need(2)
        return eval_expr(args[1], env) in eval_expr(args[0], env)
    if op == "startswith":
        need(2)
        return str(eval_expr(args[0], env)).startswith(str(eval_expr(args[1], env)))
    if op == "count_eq":
        need(2)
        values = eval_expr(args[0], env)
        needle = eval_expr(args[1], env)
        return sum(1 for value in values if value == needle)
    if op == "tuple_get":
        need(2)
        return eval_expr(args[0], env)[eval_expr(args[1], env)]
    if op == "sort":
        need(1)
        return sorted(eval_expr(args[0], env))
    if op == "first":
        need(1)
        values = eval_expr(args[0], env)
        if not values:
            raise DslError("first of empty")
        return values[0]
    if op == "last":
        need(1)
        values = eval_expr(args[0], env)
        if not values:
            raise DslError("last of empty")
        return values[-1]
    if op == "gt":
        need(2)
        return eval_expr(args[0], env) > eval_expr(args[1], env)
    if op == "ge":
        need(2)
        return eval_expr(args[0], env) >= eval_expr(args[1], env)
    if op == "lt":
        need(2)
        return eval_expr(args[0], env) < eval_expr(args[1], env)
    if op == "eq":
        need(2)
        return eval_expr(args[0], env) == eval_expr(args[1], env)
    if op == "and":
        if len(args) < 2:
            raise DslError("and expects at least two args")
        return all(bool(eval_expr(arg, env)) for arg in args)
    if op == "or":
        if len(args) < 2:
            raise DslError("or expects at least two args")
        return any(bool(eval_expr(arg, env)) for arg in args)
    if op == "not":
        need(1)
        return not bool(eval_expr(args[0], env))
    if op == "if":
        need(3)
        return eval_expr(args[1], env) if bool(eval_expr(args[0], env)) else eval_expr(args[2], env)
    if op == "join":
        need(2)
        sep = eval_expr(args[0], env)
        values = eval_expr(args[1], env)
        return str(sep).join(str(value) for value in values)
    raise DslError(f"unknown op: {op}")


def execute(program: str, env: dict[str, Any]) -> Any:
    return eval_expr(parse_expr(program), env)


def normalize_program(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:dsl)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    start = text.find("(")
    if start < 0:
        return text.splitlines()[0].strip()
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[start : index + 1].strip()
    return text[start:].splitlines()[0].strip()


def program_is_valid(program: str) -> bool:
    try:
        parse_expr(program)
        return True
    except Exception:
        return False


def program_case_passes(program: str, case: dict[str, Any]) -> bool:
    try:
        return execute(program, case["input"]) == case["expected"]
    except Exception:
        return False


def program_pass_count(program: str, cases: list[dict[str, Any]]) -> int:
    return sum(1 for case in cases if program_case_passes(program, case))
