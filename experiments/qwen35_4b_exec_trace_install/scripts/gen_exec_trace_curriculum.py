#!/usr/bin/env python3
"""Truth-audited synthetic curriculum: EXECUTION TRACING of random Python.

The one designed curriculum of lifecycle 32 — the FIRST curriculum bet of the
cognitive-core coding program. The task installs a UNIVERSAL, transferable
EXECUTION skill: be an accurate "mental interpreter". Each row shows a random
terminating Python program with LITERAL inputs; the think target is the
step-by-step running-state trace (one line per executed statement, showing the
resulting values of the variables it touched); the answer is the program's
final printed output on a clean single line (``FINAL: <value>``).

Why this shape installs the agentic gap and respects provenance:

- The base 4B is already a strong single-FUNCTION coder (HumanEval 76%) but a
  weak multi-step agent (duet-eval 23%). The gap is state-tracking across
  steps. Tracing concrete execution drills exactly that.
- The signal is SELF-GENERATED and EXECUTION-VERIFIED (no larger teacher): the
  programs are constructed here and every trace is confirmed by REAL CPython
  execution, so the provenance constraint holds.
- It looks like NOTHING in HumanEval / MBPP (those are docstring->function;
  this is trace-this-concrete-program), enforced by ``scripts/contamination``.

Fail-closed triple truth audit for every row (never ship an unverified trace):

1. PRIMARY interpreter (``trace_program``): a hand-written tree-walking,
   diff-emitting interpreter produces the trace, the printed output, and the
   final variable state.
2. INDEPENDENT re-execution (``verify_by_execution``): the program is rendered
   to real Python source and executed by REAL CPython under ``sys.settrace``
   in a restricted namespace with a step cap; the tracer reconstructs the
   per-step trace, the output, and the final state directly from actual
   execution. The primary trace, output, AND final state are byte-compared
   against this second code path — a mismatch aborts construction.
3. SAFETY / TERMINATION: no imports, no I/O, no attribute access beyond
   ``str.upper``/``str.lower`` and ``list.append``; bounded loops and bounded
   recursion; a per-program step cap ABORTS and discards the program.

Determinism: construction seed (default 90210), no randomness inside the
programs, deterministic execution. A MIXED, frozen difficulty schedule
(short/medium/long, biased toward medium/long because the base already handles
trivial code) drives multi-step state evolution. Contamination: whole-word
banned-function-name audit (zero hits) plus a present-only benchmark n-gram
overlap aid (zero overlap). Row-level uniqueness within the corpus.

Forgetting guard (documented design decision): pure trace-only SFT risks
shifting the model to ALWAYS trace instead of generate code (cf. answer-only
SFT 0.72->0.09). Mitigated BY DESIGN: (a) an explicit distinct instruction
("Trace the following program's execution") that does not collide with the
code-completion prompt format the eval uses; (b) a moderate dose (~400 rows, 1
epoch); (c) a mixed-difficulty corpus. A ``--mix-retention R`` switch (default
OFF) additionally blends in R self-generated code-COMPLETION rows from the same
generator (the model must WRITE code, not trace it) for re-running if the
pure-trace probe regresses HumanEval.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contamination as contam  # noqa: E402

EXP = Path(__file__).resolve().parents[1]

# Construction seed (verify grep-fresh). No randomness lives inside the
# generated programs; only the generator draws from this seed.
CONSTRUCTION_SEED = 90210

STEP_CAP = 4000  # line events per program; exceeding ABORTS and discards.
RECURSION_CAP = 64  # backstop; real recursion is bounded by decreasing args.

# Frozen mixed difficulty schedule (biased to medium/long). Total 400.
TIER_SCHEDULE = (("short", 80), ("medium", 160), ("long", 160))
DEFAULT_ROWS = sum(count for _, count in TIER_SCHEDULE)

INSTRUCTION = "Trace the following program's execution."

# Curated, contamination-clean identifier pools (asserted against the banned
# set by the unit tests). No single-letter ``f`` (a benchmark def name); no
# ``sum``/``add``/``aux`` etc.
SCALAR_NAMES = (
    "acc", "tally", "total", "prod", "best", "hi", "lo", "cur", "nxt", "tmp",
    "res", "val", "num", "cnt", "delta", "gain", "span", "amt", "score", "carry",
)
LIST_NAMES = ("arr", "seq", "vec", "buf", "bag", "row", "col", "tbl")
DICT_NAMES = ("book", "reg", "map_", "store", "slots", "grid")
STR_NAMES = ("word", "name_", "tag", "token", "label", "note", "text_")
FUNC_NAMES = ("compute", "combine", "twice", "recur", "fold", "grow", "mix", "scale_up")
LOOP_VARS = ("i", "j", "k", "t", "u")
# Neutral string literals (contamination-clean).
WORD_POOL = (
    "alpha", "beta", "gamma", "delta", "node", "unit", "cell", "atom", "zeta",
    "kilo", "mega", "nano", "orb", "quill", "vane", "flux", "prism", "ember",
)


# ------------------------------------------------------------------- program IR
@dataclass
class S:
    """A statement node with a rendered-line number + stripped text."""

    kind: str
    args: tuple = ()
    body: list = field(default_factory=list)
    branches: list = field(default_factory=list)  # for-if: list of (cond, body)
    orelse: list | None = None
    lineno: int = 0
    text: str = ""


# Expression nodes are plain tuples tagged by args[0].
def E(*node) -> tuple:
    return node


# ----------------------------------------------------------------- rendering
def render_expr(node) -> str:
    tag = node[0]
    if tag == "lit":
        return repr(node[1])
    if tag == "name":
        return node[1]
    if tag == "list":
        return "[" + ", ".join(render_expr(e) for e in node[1]) + "]"
    if tag == "dict":
        return "{" + ", ".join(f"{k!r}: {render_expr(v)}" for k, v in node[1]) + "}"
    if tag == "bin":
        return f"({render_expr(node[2])} {node[1]} {render_expr(node[3])})"
    if tag == "neg":
        return f"(-{render_expr(node[1])})"
    if tag == "cmp":
        return f"{render_expr(node[2])} {node[1]} {render_expr(node[3])}"
    if tag == "index":
        return f"{render_expr(node[1])}[{render_expr(node[2])}]"
    if tag == "len":
        return f"len({render_expr(node[1])})"
    if tag == "abs":
        return f"abs({render_expr(node[1])})"
    if tag == "min":
        return f"min({render_expr(node[1])}, {render_expr(node[2])})"
    if tag == "max":
        return f"max({render_expr(node[1])}, {render_expr(node[2])})"
    if tag == "sum":
        return f"sum({render_expr(node[1])})"
    if tag == "str":
        return f"str({render_expr(node[1])})"
    if tag == "int":
        return f"int({render_expr(node[1])})"
    if tag == "call":
        return f"{node[1]}(" + ", ".join(render_expr(a) for a in node[2]) + ")"
    if tag == "meth":
        return f"{render_expr(node[1])}.{node[2]}()"
    raise ValueError(f"cannot render expr: {node!r}")


def render_stmt_line(node: S) -> str:
    kind = node.kind
    if kind == "assign":
        return f"{node.args[0]} = {render_expr(node.args[1])}"
    if kind == "aug":
        return f"{node.args[0]} {node.args[1]}= {render_expr(node.args[2])}"
    if kind == "append":
        return f"{node.args[0]}.append({render_expr(node.args[1])})"
    if kind == "idxset":
        return f"{node.args[0]}[{render_expr(node.args[1])}] = {render_expr(node.args[2])}"
    if kind == "print":
        return f"print({render_expr(node.args[0])})"
    if kind == "for":
        return f"for {node.args[0]} in range({render_expr(node.args[1])}, {render_expr(node.args[2])}):"
    if kind == "while":
        return f"while {render_expr(node.args[0])}:"
    if kind == "if":
        return f"if {render_expr(node.args[0])}:"
    if kind == "elif":
        return f"elif {render_expr(node.args[0])}:"
    if kind == "else":
        return "else:"
    if kind == "func":
        return f"def {node.args[0]}(" + ", ".join(node.args[1]) + "):"
    if kind == "return":
        return f"return {render_expr(node.args[0])}"
    raise ValueError(f"cannot render stmt: {kind}")


def render_program(funcs: list[S], body: list[S]) -> tuple[list[str], dict[int, str]]:
    """Render to real Python source, assigning line numbers + stripped text."""
    lines: list[str] = []
    lineno_to_text: dict[int, str] = {}

    def emit(node: S, indent: int) -> None:
        node.lineno = len(lines) + 1
        node.text = render_stmt_line(node)
        lineno_to_text[node.lineno] = node.text
        lines.append("    " * indent + node.text)
        if node.kind == "func":
            for child in node.body:
                emit(child, indent + 1)
        elif node.kind in ("for", "while"):
            for child in node.body:
                emit(child, indent + 1)
        elif node.kind == "if":
            for child in node.body:
                emit(child, indent + 1)
            for cond, block in node.branches:
                head = S("elif", (cond,))
                emit(head, indent)
                for child in block:
                    emit(child, indent + 1)
            if node.orelse is not None:
                head = S("else")
                emit(head, indent)
                for child in node.orelse:
                    emit(child, indent + 1)

    for fn in funcs:
        emit(fn, 0)
    for stmt in body:
        emit(stmt, 0)
    return lines, lineno_to_text


# ----------------------------------------------------------- value formatting
def repr_val(value) -> str:
    return repr(value)


def print_str(value) -> str:
    return str(value)


JSONABLE = (int, float, str, bool, list, dict)


def jsonable_filter(namespace: dict) -> dict:
    out = {}
    for name, value in namespace.items():
        if name.startswith("__"):
            continue
        if isinstance(value, bool):
            continue  # programs never store bare bools
        if isinstance(value, JSONABLE):
            out[name] = value
    return out


def canonical_updates(before: dict, after: dict) -> list[list[str]]:
    changed = []
    for name in sorted(after):
        if name not in before or before[name] != after[name]:
            changed.append([name, repr_val(after[name])])
    return changed


# ------------------------------------------------------- PRIMARY interpreter
class StepCapExceeded(RuntimeError):
    pass


class Interp:
    """Primary hand-written, diff-emitting tree-walking interpreter."""

    def __init__(self) -> None:
        self.env: dict = {}
        self.funcs: dict[str, S] = {}
        self.steps: list[dict] = []
        self.output: list[str] = []
        self.count = 0

    # -- expression evaluation (raises on any anomaly -> row discarded)
    def eval(self, node, scope: dict):
        tag = node[0]
        if tag == "lit":
            return node[1]
        if tag == "name":
            return scope[node[1]]
        if tag == "list":
            return [self.eval(e, scope) for e in node[1]]
        if tag == "dict":
            return {k: self.eval(v, scope) for k, v in node[1]}
        if tag == "bin":
            left, right = self.eval(node[2], scope), self.eval(node[3], scope)
            op = node[1]
            if op == "+":
                return left + right
            if op == "-":
                return left - right
            if op == "*":
                return left * right
            if op == "//":
                return left // right
            if op == "%":
                return left % right
            raise ValueError(op)
        if tag == "neg":
            return -self.eval(node[1], scope)
        if tag == "cmp":
            left, right = self.eval(node[2], scope), self.eval(node[3], scope)
            op = node[1]
            return {
                "<": left < right,
                "<=": left <= right,
                ">": left > right,
                ">=": left >= right,
                "==": left == right,
                "!=": left != right,
            }[op]
        if tag == "index":
            return self.eval(node[1], scope)[self.eval(node[2], scope)]
        if tag == "len":
            return len(self.eval(node[1], scope))
        if tag == "abs":
            return abs(self.eval(node[1], scope))
        if tag == "min":
            return min(self.eval(node[1], scope), self.eval(node[2], scope))
        if tag == "max":
            return max(self.eval(node[1], scope), self.eval(node[2], scope))
        if tag == "sum":
            return sum(self.eval(node[1], scope))
        if tag == "str":
            return str(self.eval(node[1], scope))
        if tag == "int":
            return int(self.eval(node[1], scope))
        if tag == "call":
            return self.call(node[1], [self.eval(a, scope) for a in node[2]], depth=0)
        if tag == "meth":
            obj = self.eval(node[1], scope)
            return {"upper": obj.upper, "lower": obj.lower}[node[2]]()
        raise ValueError(f"cannot eval expr: {node!r}")

    def call(self, name: str, values: list, depth: int):
        if depth > RECURSION_CAP:
            raise StepCapExceeded("recursion cap")
        fn = self.funcs[name]
        params = fn.args[1]
        local = dict(zip(params, values, strict=True))
        for stmt in fn.body:
            done, value = self.exec_func_stmt(stmt, local, depth)
            if done:
                return value
        return None

    def exec_func_stmt(self, stmt: S, local: dict, depth: int):
        self.tick()
        if stmt.kind == "assign":
            local[stmt.args[0]] = self.eval_call_aware(stmt.args[1], local, depth)
            return False, None
        if stmt.kind == "aug":
            local[stmt.args[0]] = self._augment(
                local[stmt.args[0]], stmt.args[1], self.eval_call_aware(stmt.args[2], local, depth)
            )
            return False, None
        if stmt.kind == "return":
            return True, self.eval_call_aware(stmt.args[0], local, depth)
        if stmt.kind == "if":
            if self.eval_call_aware(stmt.args[0], local, depth):
                for child in stmt.body:
                    done, value = self.exec_func_stmt(child, local, depth)
                    if done:
                        return True, value
            elif stmt.orelse is not None:
                for child in stmt.orelse:
                    done, value = self.exec_func_stmt(child, local, depth)
                    if done:
                        return True, value
            return False, None
        raise ValueError(f"unsupported function-body stmt: {stmt.kind}")

    def eval_call_aware(self, node, scope: dict, depth: int):
        # Same as eval but threads recursion depth through nested user calls.
        if node[0] == "call":
            return self.call(node[1], [self.eval_call_aware(a, scope, depth) for a in node[2]], depth + 1)
        if node[0] in ("lit", "name"):
            return self.eval(node, scope)
        if node[0] == "bin":
            left = self.eval_call_aware(node[2], scope, depth)
            right = self.eval_call_aware(node[3], scope, depth)
            op = node[1]
            return {"+": left + right, "-": left - right, "*": left * right,
                    "//": left // right if right else None, "%": left % right if right else None}[op]
        if node[0] == "neg":
            return -self.eval_call_aware(node[1], scope, depth)
        if node[0] == "cmp":
            left = self.eval_call_aware(node[2], scope, depth)
            right = self.eval_call_aware(node[3], scope, depth)
            op = node[1]
            return {"<": left < right, "<=": left <= right, ">": left > right,
                    ">=": left >= right, "==": left == right, "!=": left != right}[op]
        # fall back for the rest (no nested user calls expected there)
        return self.eval(node, scope)

    @staticmethod
    def _augment(current, op, delta):
        if op == "+":
            return current + delta
        if op == "-":
            return current - delta
        if op == "*":
            return current * delta
        if op == "//":
            return current // delta
        if op == "%":
            return current % delta
        raise ValueError(op)

    def tick(self) -> None:
        self.count += 1
        if self.count > STEP_CAP:
            raise StepCapExceeded("step cap")

    # -- statement execution (module scope; diff-emitting)
    def snapshot(self) -> dict:
        # Deep-copy so in-place mutations (append / idxset) are visible as a
        # diff, exactly as the real-CPython oracle deep-copies frame locals.
        return copy.deepcopy(jsonable_filter(self.env))

    def emit_diff(self, node: S, before: dict) -> None:
        after = jsonable_filter(self.env)
        updates = canonical_updates(before, after)
        if updates:
            self.steps.append({"stmt": node.text, "kind": "set", "updates": updates})

    def exec_stmt(self, node: S) -> None:
        self.tick()
        kind = node.kind
        if kind == "func":
            self.funcs[node.args[0]] = node
            return
        if kind == "assign":
            before = self.snapshot()
            self.env[node.args[0]] = self.eval(node.args[1], self.env)
            self.emit_diff(node, before)
            return
        if kind == "aug":
            before = self.snapshot()
            self.env[node.args[0]] = self._augment(
                self.env[node.args[0]], node.args[1], self.eval(node.args[2], self.env)
            )
            self.emit_diff(node, before)
            return
        if kind == "append":
            before = self.snapshot()
            self.env[node.args[0]].append(self.eval(node.args[1], self.env))
            self.emit_diff(node, before)
            return
        if kind == "idxset":
            before = self.snapshot()
            self.env[node.args[0]][self.eval(node.args[1], self.env)] = self.eval(node.args[2], self.env)
            self.emit_diff(node, before)
            return
        if kind == "print":
            text = print_str(self.eval(node.args[0], self.env))
            self.output.append(text)
            self.steps.append({"stmt": node.text, "kind": "out", "output": text})
            return
        if kind == "for":
            start = self.eval(node.args[1], self.env)
            stop = self.eval(node.args[2], self.env)
            for value in range(start, stop):
                before = self.snapshot()
                self.env[node.args[0]] = value
                self.emit_diff(node, before)
                for child in node.body:
                    self.exec_stmt(child)
            return
        if kind == "while":
            guard = 0
            while self.eval(node.args[0], self.env):
                guard += 1
                if guard > 64:
                    raise StepCapExceeded("while guard")
                for child in node.body:
                    self.exec_stmt(child)
            return
        if kind == "if":
            if self.eval(node.args[0], self.env):
                for child in node.body:
                    self.exec_stmt(child)
                return
            for cond, block in node.branches:
                if self.eval(cond, self.env):
                    for child in block:
                        self.exec_stmt(child)
                    return
            if node.orelse is not None:
                for child in node.orelse:
                    self.exec_stmt(child)
            return
        raise ValueError(f"cannot exec stmt: {kind}")


def trace_program(funcs: list[S], body: list[S]) -> tuple[list[dict], str, dict]:
    interp = Interp()
    for fn in funcs:
        interp.exec_stmt(fn)
    for stmt in body:
        interp.exec_stmt(stmt)
    return interp.steps, "\n".join(interp.output), jsonable_filter(interp.env)


# --------------------------------------------- INDEPENDENT re-execution oracle
def verify_by_execution(
    source_lines: list[str], lineno_to_text: dict[int, str]
) -> tuple[list[dict], str, dict]:
    """Run the rendered source with REAL CPython under sys.settrace.

    Reconstructs the per-step diff trace, the printed output, and the final
    variable state directly from actual execution — the independent second code
    path. Restricted builtins + a step cap enforce safety/termination.
    """
    source = "\n".join(source_lines) + "\n"
    output_log: list[tuple[int, str]] = []

    def traced_print(*values, sep=" ", end="\n"):  # noqa: ARG001 (end ignored)
        lineno = sys._getframe(1).f_lineno
        output_log.append((lineno, sep.join(print_str(v) for v in values)))

    safe_builtins = {
        "range": range, "len": len, "abs": abs, "min": min, "max": max,
        "sum": sum, "str": str, "int": int, "float": float, "bool": bool,
        "list": list, "dict": dict, "print": traced_print,
    }
    globals_ns: dict = {"__builtins__": safe_builtins}

    steps: list[dict] = []
    state = {"module_frame": None, "prev": None, "count": 0}

    def finalize(lineno: int, snapshot: dict, current: dict) -> None:
        updates = canonical_updates(snapshot, current)
        text = lineno_to_text.get(lineno)
        if text is None:
            return
        printed = [t for (ln, t) in output_log if ln == lineno]
        if printed:
            steps.append({"stmt": text, "kind": "out", "output": printed[-1]})
        elif updates:
            steps.append({"stmt": text, "kind": "set", "updates": updates})

    def local_tracer(frame, event, arg):  # noqa: ARG001
        if event == "line":
            state["count"] += 1
            if state["count"] > STEP_CAP:
                raise StepCapExceeded("step cap (exec)")
            current = copy.deepcopy(jsonable_filter(frame.f_locals))
            if state["prev"] is not None:
                pl, ps = state["prev"]
                finalize(pl, ps, current)
            state["prev"] = (frame.f_lineno, current)
        elif event == "return":
            if state["prev"] is not None:
                current = copy.deepcopy(jsonable_filter(frame.f_locals))
                pl, ps = state["prev"]
                finalize(pl, ps, current)
                state["prev"] = None
        return local_tracer

    def global_tracer(frame, event, arg):  # noqa: ARG001
        if event == "call":
            if state["module_frame"] is None:
                state["module_frame"] = frame
                return local_tracer
            return None  # never trace inside user functions
        return None

    code = compile(source, "<exec_trace_program>", "exec")
    old = sys.gettrace()
    sys.settrace(global_tracer)
    try:
        exec(code, globals_ns)  # noqa: S102 (sandboxed: restricted builtins, no imports)
    finally:
        sys.settrace(old)

    final_env = jsonable_filter(globals_ns)
    output = "\n".join(t for _, t in output_log)
    return steps, output, final_env


# --------------------------------------------------------------- generators
def _int_lit(rng, lo=0, hi=9):
    return E("lit", rng.randint(lo, hi))


def _quarter(rng):
    return E("lit", rng.choice((0.25, 0.5, 0.75, 1.5, 2.5, 3.5)))


def _int_expr(rng, ints, depth=0):
    choices = ["var", "lit"]
    if depth < 2 and len(ints) >= 1:
        choices += ["bin", "bin"]
    which = rng.choice(choices) if ints else "lit"
    if which == "var":
        return E("name", rng.choice(ints))
    if which == "lit":
        return _int_lit(rng, 1, 9)
    op = rng.choice(("+", "-", "*", "//", "%"))
    left = E("name", rng.choice(ints))
    if op in ("//", "%"):
        right = E("lit", rng.randint(2, 5))
    else:
        right = _int_expr(rng, ints, depth + 1)
    return E("bin", op, left, right)


class Builder:
    """Assembles a random terminating program for a difficulty tier."""

    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.ints: list[str] = []
        self.floats: list[str] = []
        self.strs: list[str] = []
        self.lists: dict[str, int] = {}  # name -> guaranteed length
        self.dicts: dict[str, list[str]] = {}  # name -> keys
        self.funcs: list[S] = []
        self.body: list[S] = []
        self._pool = {
            "scalar": list(SCALAR_NAMES),
            "list": list(LIST_NAMES),
            "dict": list(DICT_NAMES),
            "str": list(STR_NAMES),
            "loop": list(LOOP_VARS),
            "func": list(FUNC_NAMES),
        }
        for values in self._pool.values():
            rng.shuffle(values)

    def fresh(self, kind: str) -> str:
        return self._pool[kind].pop()

    # -- building blocks
    def init_scalars(self) -> None:
        for _ in range(self.rng.randint(2, 4)):
            name = self.fresh("scalar")
            self.body.append(S("assign", (name, _int_lit(self.rng, 0, 9))))
            self.ints.append(name)
        if self.rng.random() < 0.5:
            name = self.fresh("scalar")
            self.body.append(S("assign", (name, _quarter(self.rng))))
            self.floats.append(name)

    def arith_block(self) -> None:
        for _ in range(self.rng.randint(2, 4)):
            if not self.ints:
                return
            target = self.rng.choice(self.ints)
            if self.rng.random() < 0.5:
                op = self.rng.choice(("+", "-", "*"))
                self.body.append(S("aug", (target, op, _int_expr(self.rng, self.ints))))
            else:
                self.body.append(S("assign", (target, _int_expr(self.rng, self.ints))))

    def float_block(self) -> None:
        if not self.floats:
            name = self.fresh("scalar")
            self.body.append(S("assign", (name, _quarter(self.rng))))
            self.floats.append(name)
        target = self.rng.choice(self.floats)
        self.body.append(S("aug", (target, self.rng.choice(("+", "-")), _quarter(self.rng))))
        if self.ints and self.rng.random() < 0.5:
            self.body.append(S("aug", (target, "+", E("name", self.rng.choice(self.ints)))))

    def string_block(self) -> None:
        name = self.fresh("str")
        a, b = self.rng.sample(WORD_POOL, 2)
        self.body.append(S("assign", (name, E("lit", a))))
        self.strs.append(name)
        choice = self.rng.random()
        if choice < 0.5:
            self.body.append(S("aug", (name, "+", E("lit", b))))
        else:
            self.body.append(S("assign", (name, E("bin", "*", E("name", name), E("lit", self.rng.randint(2, 3))))))
        if self.rng.random() < 0.5:
            self.body.append(S("assign", (name, E("meth", E("name", name), self.rng.choice(("upper", "lower"))))))

    def list_block(self) -> None:
        name = self.fresh("list")
        length = self.rng.randint(2, 4)
        elems = [_int_lit(self.rng, 0, 9) for _ in range(length)]
        self.body.append(S("assign", (name, E("list", elems))))
        self.lists[name] = length
        for _ in range(self.rng.randint(1, 2)):
            if self.ints and self.rng.random() < 0.6:
                self.body.append(S("append", (name, _int_expr(self.rng, self.ints))))
            else:
                self.body.append(S("append", (name, _int_lit(self.rng, 0, 9))))
        if self.rng.random() < 0.6:
            idx = self.rng.randint(0, length - 1)
            self.body.append(S("idxset", (name, E("lit", idx), _int_lit(self.rng, 0, 9))))
        if self.ints and self.rng.random() < 0.6:
            target = self.rng.choice(self.ints)
            idx = self.rng.randint(0, length - 1)
            self.body.append(S("assign", (target, E("index", E("name", name), E("lit", idx)))))

    def dict_block(self) -> None:
        name = self.fresh("dict")
        keys = self.rng.sample(WORD_POOL, self.rng.randint(2, 3))
        pairs = [(k, _int_lit(self.rng, 0, 9)) for k in keys]
        self.body.append(S("assign", (name, E("dict", pairs))))
        self.dicts[name] = list(keys)
        newkey = self.rng.choice([w for w in WORD_POOL if w not in keys])
        self.body.append(S("idxset", (name, E("lit", newkey), _int_lit(self.rng, 0, 9))))
        self.dicts[name].append(newkey)
        upd = self.rng.choice(self.dicts[name])
        self.body.append(S("idxset", (name, E("lit", upd), _int_expr(self.rng, self.ints or [name and 0]) if self.ints else _int_lit(self.rng))))
        if self.ints and self.rng.random() < 0.6:
            target = self.rng.choice(self.ints)
            self.body.append(S("assign", (target, E("index", E("name", name), E("lit", self.rng.choice(self.dicts[name]))))))

    def for_block(self) -> None:
        if not self.ints:
            self.init_scalars()
        var = self.fresh("loop")
        stop = self.rng.randint(3, 6)
        target = self.rng.choice(self.ints)
        inner: list[S] = []
        pick = self.rng.random()
        if pick < 0.45:
            inner.append(S("aug", (target, "+", E("name", var))))
        elif pick < 0.75:
            inner.append(S("aug", (target, "+", _int_expr(self.rng, self.ints + [var]))))
        else:
            # append into a list accumulator
            if self.lists:
                lname = self.rng.choice(list(self.lists))
            else:
                lname = self.fresh("list")
                self.body.append(S("assign", (lname, E("list", []))))
                self.lists[lname] = 0
            inner.append(S("append", (lname, E("bin", "*", E("name", var), E("lit", 2)))))
        self.body.append(S("for", (var, E("lit", 0), E("lit", stop)), body=inner))

    def nested_for_block(self) -> None:
        if not self.ints:
            self.init_scalars()
        outer = self.fresh("loop")
        inner_var = self.fresh("loop")
        target = self.rng.choice(self.ints)
        inner = [S("aug", (target, "+", E("bin", "+", E("name", outer), E("name", inner_var))))]
        loop = S("for", (inner_var, E("lit", 0), E("lit", self.rng.randint(2, 3))), body=inner)
        self.body.append(S("for", (outer, E("lit", 0), E("lit", self.rng.randint(2, 3))), body=[loop]))

    def while_block(self) -> None:
        counter = self.fresh("scalar")
        bound = self.rng.randint(3, 6)
        self.body.append(S("assign", (counter, E("lit", 0))))
        self.ints.append(counter)
        target = self.rng.choice(self.ints)
        inner = [
            S("aug", (target, "+", E("bin", "+", E("name", counter), E("lit", 1)))),
            S("aug", (counter, "+", E("lit", 1))),
        ]
        self.body.append(S("while", (E("cmp", "<", E("name", counter), E("lit", bound)),), body=inner))

    def if_block(self) -> None:
        if len(self.ints) < 1:
            self.init_scalars()
        target = self.rng.choice(self.ints)
        pivot = self.rng.randint(3, 7)
        cond = E("cmp", self.rng.choice((">", "<", ">=", "<=")), E("name", target), E("lit", pivot))
        then = [S("aug", (target, "+", _int_lit(self.rng, 1, 5)))]
        orelse = [S("aug", (target, "-", _int_lit(self.rng, 1, 5)))]
        node = S("if", (cond,), body=then, orelse=orelse)
        if self.rng.random() < 0.4:
            other = self.rng.choice(self.ints)
            node.branches = [(E("cmp", "==", E("name", target), E("lit", pivot)),
                              [S("assign", (other, E("bin", "*", E("name", other), E("lit", 2))))])]
        self.body.append(node)

    def func_block(self, recursive: bool) -> None:
        name = self.fresh("func")
        param = "n"
        if recursive:
            body = [
                S("if", (E("cmp", "<=", E("name", param), E("lit", 1)),),
                  body=[S("return", (E("name", param),))]),
                S("return", (E("bin", "+", E("name", param),
                                 E("call", name, [E("bin", "-", E("name", param), E("lit", 1))])),)),
            ]
        else:
            tmp = "m"
            body = [
                S("assign", (tmp, E("bin", "*", E("name", param), E("lit", self.rng.randint(2, 3))))),
                S("return", (E("bin", "+", E("name", tmp), E("lit", self.rng.randint(1, 4))),)),
            ]
        self.funcs.append(S("func", (name, (param,)), body=body))
        if not self.ints:
            self.init_scalars()
        target = self.fresh("scalar")
        arg = self.rng.randint(2, 5)
        self.body.append(S("assign", (target, E("call", name, [E("lit", arg)]))))
        self.ints.append(target)

    def terminal(self) -> None:
        candidates = self.ints + self.floats + list(self.lists) + list(self.dicts) + self.strs
        name = self.rng.choice(candidates)
        self.body.append(S("print", (E("name", name),)))

    def build(self, tier: str) -> tuple[list[S], list[S]]:
        rng = self.rng
        self.init_scalars()
        if tier == "short":
            blocks = [self.arith_block]
            extra = rng.sample([self.if_block, self.list_block, self.string_block, self.float_block], 1)
            blocks += extra
        elif tier == "medium":
            blocks = [self.arith_block, rng.choice([self.for_block, self.while_block])]
            blocks += rng.sample(
                [self.if_block, self.list_block, self.dict_block, self.string_block,
                 self.float_block, lambda: self.func_block(False)],
                2,
            )
        else:  # long
            blocks = [
                self.arith_block,
                rng.choice([self.for_block, self.nested_for_block]),
                rng.choice([self.while_block, self.for_block]),
                rng.choice([lambda: self.func_block(True), lambda: self.func_block(False)]),
            ]
            blocks += rng.sample(
                [self.if_block, self.list_block, self.dict_block, self.string_block, self.float_block],
                2,
            )
            rng.shuffle(blocks)
        for block in blocks:
            block()
        self.terminal()
        return self.funcs, self.body


# ----------------------------------------------------------------- row build
def build_row(rng: random.Random, tier: str, index: int) -> dict | None:
    builder = Builder(rng)
    funcs, body = builder.build(tier)
    source_lines, lineno_to_text = render_program(funcs, body)
    try:
        steps, output, final_env = trace_program(funcs, body)
    except (StepCapExceeded, ZeroDivisionError, KeyError, IndexError, TypeError, ValueError, RecursionError):
        return None
    # An INDEPENDENT re-execution by real CPython must byte-match trace/out/state.
    try:
        v_steps, v_output, v_env = verify_by_execution(source_lines, lineno_to_text)
    except (StepCapExceeded, Exception):  # noqa: BLE001 (discard any anomaly)
        return None
    if json.dumps(steps, sort_keys=True) != json.dumps(v_steps, sort_keys=True):
        return None
    if output != v_output or json.dumps(final_env, sort_keys=True) != json.dumps(v_env, sort_keys=True):
        return None
    if not output or "\n" in output:
        return None  # exactly one terminal print => single-line answer
    n_set = sum(1 for s in steps if s["kind"] == "set")
    n_out = sum(1 for s in steps if s["kind"] == "out")
    if n_out != 1 or n_set < 2:
        return None
    code = "\n".join(source_lines)
    think = render_think(steps)
    answer = f"FINAL: {output}"
    prompt = (
        f"{INSTRUCTION} Work through it one statement at a time, tracking each "
        "variable's value as it changes. Then give the program's final printed "
        f"output.\n\n```python\n{code}\n```\n\n"
        "End with exactly one line:\nFINAL: <printed output>"
    )
    return {
        "messages": [{"role": "user", "content": prompt}],
        "think": think,
        "answer": answer,
        "kind": "exec_trace",
        "family": "cognitive_core",
        "tier": tier,
        "n_steps": len(steps),
        "row_weight": 1.0,
        "task_id": f"exec_trace_{index:05d}",
        "_audit": {
            "truth_valid": True,
            "reexec_byte_match": True,
            "code": code,
            "final_output": output,
            "final_state": final_env,
            "n_set_steps": n_set,
            "n_out_steps": n_out,
        },
    }


def render_think(steps: list[dict]) -> str:
    lines = ["I run the program one statement at a time, noting each variable as its value changes."]
    for i, step in enumerate(steps, 1):
        if step["kind"] == "set":
            updates = ", ".join(f"{name}={value}" for name, value in step["updates"])
            lines.append(f"Step {i}: {step['stmt']}  ->  {updates}")
        else:
            lines.append(f"Step {i}: {step['stmt']}  ->  output: {step['output']}")
    lines.append(f"The program prints {steps[-1]['output']}.")
    return "\n".join(lines)


# --------------------------------------------------- retention (forgetting guard)
def build_retention_row(rng: random.Random, index: int) -> dict | None:
    """A self-generated code-COMPLETION row: complete a random program's tail.

    Keeps the model in GENERATE-code mode (not trace mode). The model is shown
    a valid program's opening lines and must WRITE the remaining lines. No
    docstring/spec (spec-free), from the SAME random-program generator.
    """
    tier = rng.choice(("short", "medium"))
    builder = Builder(rng)
    funcs, body = builder.build(tier)
    source_lines, _ = render_program(funcs, body)
    try:
        trace_program(funcs, body)
    except Exception:  # noqa: BLE001
        return None
    if len(source_lines) < 4:
        return None
    cut = max(2, len(source_lines) // 2)
    head = "\n".join(source_lines[:cut])
    tail = "\n".join(source_lines[cut:])
    prompt = (
        "Complete this Python program. Write only the remaining lines so that "
        "it runs to completion and prints its result.\n\n```python\n"
        f"{head}\n```"
    )
    return {
        "messages": [{"role": "user", "content": prompt}],
        "think": "I continue the program, keeping the existing variables consistent.",
        "answer": tail,
        "kind": "code_retention",
        "family": "cognitive_core",
        "tier": tier,
        "n_steps": 0,
        "row_weight": 1.0,
        "task_id": f"code_retention_{index:05d}",
        "_audit": {"truth_valid": True, "retention": True, "code": "\n".join(source_lines)},
    }


# --------------------------------------------------------------- curriculum
def generate_curriculum(seed: int, tier_counts, mix_retention: int = 0, smoke: bool = False):
    rng = random.Random(seed)
    rows: list[dict] = []
    index = 0
    for tier, count in tier_counts:
        made = 0
        attempts = 0
        while made < count:
            attempts += 1
            if attempts > count * 200 + 500:
                raise RuntimeError(f"could not synthesize enough {tier} rows")
            row = build_row(rng, tier, index)
            if row is None:
                continue
            rows.append(row)
            index += 1
            made += 1
    retention_rows: list[dict] = []
    if mix_retention > 0:
        ridx = 0
        attempts = 0
        while len(retention_rows) < mix_retention:
            attempts += 1
            if attempts > mix_retention * 200 + 500:
                raise RuntimeError("could not synthesize enough retention rows")
            row = build_retention_row(rng, ridx)
            if row is None:
                continue
            retention_rows.append(row)
            ridx += 1
    rng.shuffle(rows)
    all_rows = rows + retention_rows  # retention appended after the shuffle block
    rng.shuffle(all_rows)
    return all_rows


def public_row(row: dict) -> dict:
    return {key: value for key, value in row.items() if not key.startswith("_")}


# ----------------------------------------------------------------- validation
def validate_generated(rows: list[dict], *, expected_rows: int | None = None) -> dict:
    if not rows:
        raise ValueError("curriculum is empty")
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, got {len(rows)}")
    required = {
        "messages", "think", "answer", "kind", "family", "tier", "n_steps",
        "row_weight", "task_id", "_audit",
    }
    banned = contam.banned_names()
    serialized: set[str] = set()
    prompts: set[str] = set()
    task_ids: set[str] = set()
    codes: set[str] = set()
    tiers: Counter = Counter()
    kinds: Counter = Counter()
    step_counts: list[int] = []
    for i, row in enumerate(rows):
        if set(row) != required:
            raise ValueError(f"row {i} schema mismatch: {sorted(row)}")
        if row["family"] != "cognitive_core" or row["row_weight"] != 1.0:
            raise ValueError(f"row {i} family/weight mismatch")
        if len(row["messages"]) != 1 or row["messages"][0].get("role") != "user":
            raise ValueError(f"row {i} message schema mismatch")
        if not row["think"].strip() or not row["answer"].strip():
            raise ValueError(f"row {i} empty target")
        audit = row["_audit"]
        if audit.get("truth_valid") is not True:
            raise ValueError(f"row {i} lacks truth audit")
        kinds[row["kind"]] += 1
        if row["kind"] == "exec_trace":
            if "\n" in row["answer"] or not row["answer"].startswith("FINAL: "):
                raise ValueError(f"row {i} answer is not a single FINAL line")
            # RE-DERIVE the trace + answer independently from the audited code.
            reverify_row(row, i)
            tiers[row["tier"]] += 1
            step_counts.append(row["n_steps"])
        # contamination: whole-word banned-name audit over prompt+think+answer
        blob = row["messages"][0]["content"] + "\n" + row["think"] + "\n" + row["answer"]
        hits = contam.whole_word_hits(blob, banned)
        if hits:
            raise ValueError(f"row {i} uses banned benchmark vocabulary: {sorted(hits)}")
        canonical = json.dumps(public_row(row), sort_keys=True, ensure_ascii=False)
        prompt = row["messages"][0]["content"]
        if canonical in serialized or prompt in prompts or row["task_id"] in task_ids:
            raise ValueError(f"row {i} duplicate")
        serialized.add(canonical)
        prompts.add(prompt)
        task_ids.add(row["task_id"])
        codes.add(audit["code"])
    return {
        "rows": len(rows),
        "kinds": dict(sorted(kinds.items())),
        "tiers": dict(sorted(tiers.items())),
        "unique_codes": len(codes),
        "min_steps": min(step_counts) if step_counts else 0,
        "max_steps": max(step_counts) if step_counts else 0,
        "mean_steps": round(sum(step_counts) / len(step_counts), 2) if step_counts else 0,
    }


def reverify_row(row: dict, index: int) -> None:
    """Independently re-execute the audited code and byte-match the shipped trace."""
    code = row["_audit"]["code"]
    source_lines = code.split("\n")
    lineno_to_text = {i + 1: line.strip() for i, line in enumerate(source_lines)}
    steps, output, final_env = verify_by_execution(source_lines, lineno_to_text)
    # rebuild the think + answer from the independently derived steps
    if render_think(steps) != row["think"]:
        raise ValueError(f"row {index} think does not match independent re-execution")
    if f"FINAL: {output}" != row["answer"]:
        raise ValueError(f"row {index} answer does not match independent re-execution")
    if output != row["_audit"]["final_output"]:
        raise ValueError(f"row {index} output drifted from audit")
    if json.dumps(final_env, sort_keys=True) != json.dumps(row["_audit"]["final_state"], sort_keys=True):
        raise ValueError(f"row {index} final state drifted from audit")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_code(prompt: str) -> str:
    start = prompt.index("```python\n") + len("```python\n")
    end = prompt.index("\n```", start)
    return prompt[start:end]


def verify_public_corpus(path: Path, receipt_path: Path | None = None) -> dict:
    """Independently re-execute EVERY shipped row and byte-match its think+answer.

    Works on the PUBLIC corpus (no ``_audit``): the program code is read back
    out of the prompt, executed by real CPython under ``verify_by_execution``,
    and the trace/answer are rebuilt and byte-compared. Also re-runs the
    whole-word banned-vocabulary audit and row uniqueness. Standalone: needs no
    HF cache.
    """
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("committed corpus is empty")
    banned = contam.banned_names()
    prompts: set[str] = set()
    task_ids: set[str] = set()
    tiers: Counter = Counter()
    kinds: Counter = Counter()
    for i, row in enumerate(rows):
        kinds[row["kind"]] += 1
        prompt = row["messages"][0]["content"]
        if prompt in prompts or row["task_id"] in task_ids:
            raise ValueError(f"row {i} duplicate in committed corpus")
        prompts.add(prompt)
        task_ids.add(row["task_id"])
        blob = prompt + "\n" + row["think"] + "\n" + row["answer"]
        hits = contam.whole_word_hits(blob, banned)
        if hits:
            raise ValueError(f"row {i} banned vocabulary in committed corpus: {sorted(hits)}")
        if row["kind"] != "exec_trace":
            continue
        tiers[row["tier"]] += 1
        code = extract_code(prompt)
        source_lines = code.split("\n")
        lineno_to_text = {j + 1: line.strip() for j, line in enumerate(source_lines)}
        steps, output, _ = verify_by_execution(source_lines, lineno_to_text)
        if render_think(steps) != row["think"]:
            raise ValueError(f"row {i} think does not match independent re-execution")
        if f"FINAL: {output}" != row["answer"]:
            raise ValueError(f"row {i} answer does not match independent re-execution")
    if receipt_path is not None and receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("corpus_sha256") != sha256_text(path.read_text(encoding="utf-8")):
            raise ValueError("committed corpus sha256 disagrees with the receipt")
    return {"rows": len(rows), "kinds": dict(sorted(kinds.items())), "tiers": dict(sorted(tiers.items()))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=CONSTRUCTION_SEED)
    parser.add_argument("--out", type=Path, default=EXP / "data" / "sft_exec_trace.jsonl")
    parser.add_argument("--receipt", type=Path, default=EXP / "data" / "curriculum_receipt.json")
    parser.add_argument("--mix-retention", type=int, default=0,
                        help="blend in R self-generated code-completion rows (forgetting guard; default OFF)")
    parser.add_argument("--smoke", action="store_true", help="tiny build (a few rows per tier)")
    parser.add_argument("--examples", type=int, default=0, help="print N example rows and exit (no write)")
    parser.add_argument("--verify-corpus", action="store_true",
                        help="independently re-execute the committed corpus (no write, no GPU)")
    args = parser.parse_args()

    if args.verify_corpus:
        summary = verify_public_corpus(args.out, args.receipt)
        print(json.dumps({"verify_corpus": str(args.out), **summary}, indent=2, sort_keys=True))
        return 0

    if args.smoke:
        tier_counts = (("short", 3), ("medium", 4), ("long", 3))
    else:
        tier_counts = TIER_SCHEDULE
    rows = generate_curriculum(args.seed, tier_counts, mix_retention=args.mix_retention, smoke=args.smoke)
    expected = None if args.smoke else (DEFAULT_ROWS + args.mix_retention)
    summary = validate_generated(rows, expected_rows=expected)

    if args.examples:
        shown = 0
        for tier in ("short", "medium", "long"):
            for row in rows:
                if row.get("tier") == tier and row["kind"] == "exec_trace":
                    print("=" * 78)
                    print(f"[{tier}] {row['task_id']} (n_steps={row['n_steps']})")
                    print("--- PROMPT ---")
                    print(row["messages"][0]["content"])
                    print("--- THINK ---")
                    print(row["think"])
                    print("--- ANSWER ---")
                    print(row["answer"])
                    shown += 1
                    break
            if shown >= args.examples:
                break
        return 0

    payload = "".join(json.dumps(public_row(row), ensure_ascii=False) + "\n" for row in rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(payload, encoding="utf-8")

    # contamination n-gram overlap aid (present-only)
    overlap = {"status": "skipped_cache_absent"}
    try:
        streams = contam.build_code_tokens_from_cache()
        bench_grams = contam.benchmark_ngrams(streams)
        corpus_grams: set = set()
        for row in rows:
            corpus_grams |= contam.code_ngrams(row["_audit"]["code"])
        shared = corpus_grams & bench_grams
        distinctive = contam.distinctive_overlap(corpus_grams, bench_grams)
        overlap = {
            "status": "checked",
            "ngram_n": contam.NGRAM_N,
            "benchmark_ngrams": len(bench_grams),
            "corpus_ngrams": len(corpus_grams),
            "shared_ngrams_structural_idiom": len(shared),
            "shared_ngrams_distinctive": len(distinctive),
        }
        if distinctive:
            raise SystemExit(
                f"benchmark n-gram overlap detected: {len(distinctive)} shared "
                f"distinctive {contam.NGRAM_N}-grams"
            )
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 (datasets/cache unavailable -> skip aid)
        overlap = {"status": "skipped", "reason": type(exc).__name__}

    receipt = {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "seed": args.seed,
        "out": str(args.out),
        "corpus_sha256": sha256_text(payload),
        "mix_retention": args.mix_retention,
        "tier_schedule": [list(item) for item in tier_counts],
        "contamination": {
            "banned_function_names": len(banned := contam.banned_names()),
            "banned_hits": 0,
            "ngram_overlap": overlap,
        },
        **summary,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
