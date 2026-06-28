from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass
from typing import Any

from .dsl import DslError, Symbol, parse_expr


class GraphIrError(ValueError):
    pass


@dataclass(frozen=True)
class Ref:
    name: str


@dataclass(frozen=True)
class Instruction:
    lhs: str
    op: str
    args: tuple[Any, ...]


OP_MAP = {
    "sum": "SUM",
    "len": "LEN",
    "mod": "MOD",
    "add": "ADD",
    "sub": "SUB",
    "format": "FORMAT",
    "contains": "CONTAINS",
    "count_eq": "COUNT_EQ",
    "tuple_get": "GET",
    "sort": "SORT",
    "first": "FIRST",
    "last": "LAST",
    "gt": "GT",
    "ge": "GE",
    "lt": "LT",
    "eq": "EQ",
    "and": "AND",
    "or": "OR",
    "not": "NOT",
    "if": "IF",
    "join": "JOIN",
}


ASSIGN_RE = re.compile(r"^\s*(?:\d+\.\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$")


def _format_arg(value: Any) -> str:
    if isinstance(value, Ref):
        return value.name
    if isinstance(value, Symbol):
        return value.name
    if isinstance(value, str):
        return json.dumps(value)
    return str(value)


def dsl_to_graph(program: str) -> str:
    expr = parse_expr(program)
    lines: list[str] = []
    next_reg = 0

    def compile_node(node: Any) -> Any:
        nonlocal next_reg
        if isinstance(node, list):
            if not node or not isinstance(node[0], Symbol):
                raise GraphIrError("invalid DSL expression")
            op = OP_MAP.get(node[0].name)
            if op is None:
                raise GraphIrError(f"unsupported DSL op: {node[0].name}")
            args = [compile_node(arg) for arg in node[1:]]
            reg = f"r{next_reg}"
            next_reg += 1
            lines.append(f"{reg} = {op} " + " ".join(_format_arg(arg) for arg in args))
            return Ref(reg)
        return node

    out = compile_node(expr)
    lines.append(f"out = {_format_arg(out)}")
    return "\n".join(lines)


def normalize_graph(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:graphir|text)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"^[-*]\s+", "", line)
        match = ASSIGN_RE.match(line)
        if not match:
            continue
        lhs, rhs = match.groups()
        lines.append(f"{lhs} = {rhs.strip()}")
        if lhs == "out":
            break
    if lines:
        return "\n".join(lines)
    return text.splitlines()[0].strip() if text else ""


def _split_rhs(rhs: str) -> list[str]:
    lexer = shlex.shlex(rhs, posix=False)
    lexer.whitespace_split = True
    lexer.commenters = ""
    return list(lexer)


def _parse_arg(token: str) -> Any:
    token = token.strip()
    if token.startswith('"') and token.endswith('"'):
        return json.loads(token)
    if re.fullmatch(r"-?\d+", token):
        return int(token)
    return Ref(token)


def parse_graph(text: str) -> list[Instruction]:
    graph = normalize_graph(text)
    instructions: list[Instruction] = []
    seen: set[str] = set()
    for raw in graph.splitlines():
        match = ASSIGN_RE.match(raw)
        if not match:
            raise GraphIrError(f"bad assignment: {raw}")
        lhs, rhs = match.groups()
        if lhs in seen:
            raise GraphIrError(f"duplicate assignment: {lhs}")
        seen.add(lhs)
        parts = _split_rhs(rhs)
        if not parts:
            raise GraphIrError(f"empty rhs: {raw}")
        if len(parts) == 1:
            instructions.append(Instruction(lhs, "MOV", (_parse_arg(parts[0]),)))
        else:
            op = parts[0].upper()
            instructions.append(Instruction(lhs, op, tuple(_parse_arg(part) for part in parts[1:])))
    if not instructions or instructions[-1].lhs != "out":
        raise GraphIrError("missing out assignment")
    return instructions


def _resolve(arg: Any, env: dict[str, Any]) -> Any:
    if isinstance(arg, Ref):
        if arg.name in env:
            return env[arg.name]
        raise GraphIrError(f"unknown ref: {arg.name}")
    return arg


def _apply(op: str, args: tuple[Any, ...], env: dict[str, Any]) -> Any:
    def values(n: int | None = None) -> list[Any]:
        if n is not None and len(args) != n:
            raise GraphIrError(f"{op} expects {n}, got {len(args)}")
        return [_resolve(arg, env) for arg in args]

    if op == "MOV":
        return values(1)[0]
    if op == "SUM":
        return sum(values(1)[0])
    if op == "LEN":
        return len(values(1)[0])
    if op == "MOD":
        a, b = values(2)
        if b == 0:
            raise GraphIrError("mod by zero")
        return a % b
    if op == "ADD":
        a, b = values(2)
        return a + b
    if op == "SUB":
        a, b = values(2)
        return a - b
    if op == "FORMAT":
        fmt, value = values(2)
        return str(fmt).format(value)
    if op == "CONTAINS":
        container, needle = values(2)
        return needle in container
    if op == "COUNT_EQ":
        container, needle = values(2)
        return sum(1 for value in container if value == needle)
    if op == "GET":
        container, index = values(2)
        return container[index]
    if op == "SORT":
        return sorted(values(1)[0])
    if op == "FIRST":
        container = values(1)[0]
        if not container:
            raise GraphIrError("first of empty")
        return container[0]
    if op == "LAST":
        container = values(1)[0]
        if not container:
            raise GraphIrError("last of empty")
        return container[-1]
    if op == "GT":
        a, b = values(2)
        return a > b
    if op == "GE":
        a, b = values(2)
        return a >= b
    if op == "LT":
        a, b = values(2)
        return a < b
    if op == "EQ":
        a, b = values(2)
        return a == b
    if op == "AND":
        if len(args) < 2:
            raise GraphIrError("AND expects at least two args")
        return all(bool(_resolve(arg, env)) for arg in args)
    if op == "OR":
        if len(args) < 2:
            raise GraphIrError("OR expects at least two args")
        return any(bool(_resolve(arg, env)) for arg in args)
    if op == "NOT":
        return not bool(values(1)[0])
    if op == "IF":
        cond, yes, no = values(3)
        return yes if bool(cond) else no
    if op == "JOIN":
        sep, container = values(2)
        return str(sep).join(str(value) for value in container)
    raise GraphIrError(f"unknown op: {op}")


def execute_graph(graph: str, input_env: dict[str, Any]) -> Any:
    env = dict(input_env)
    for instruction in parse_graph(graph):
        env[instruction.lhs] = _apply(instruction.op, instruction.args, env)
    return env["out"]


def graph_is_valid(graph: str) -> bool:
    try:
        parse_graph(graph)
        return True
    except Exception:
        return False


def graph_case_passes(graph: str, case: dict[str, Any]) -> bool:
    try:
        return execute_graph(graph, case["input"]) == case["expected"]
    except Exception:
        return False


def graph_pass_count(graph: str, cases: list[dict[str, Any]]) -> int:
    return sum(1 for case in cases if graph_case_passes(graph, case))


def safe_execute_graph(graph: str, env: dict[str, Any]) -> Any:
    try:
        return execute_graph(graph, env)
    except Exception as exc:
        return f"<error:{exc}>"
