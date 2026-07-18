#!/usr/bin/env python3
"""Truth-audited synthetic curriculum: SELF-REPAIR of buggy Python functions.

The SECOND designed curriculum bet of the cognitive-core coding program
(lifecycle 33). Bet #1 (execution-tracing) came back NULL: it reshuffled which
coding tasks the 4B solves without raising the count on HumanEval, MBPP, or the
agentic duet-eval (8/35 -> 8/35). The observed agentic failure mode is a LOOP
failure: the model one-shots a multi-step task, a check fails, and it STOPS
instead of verifying and repairing. This curriculum installs the
CHECK-AND-REPAIR loop directly.

Each row is a self-contained debugging episode:

  (a) a CORRECT small synthetic Python function with a docstring spec and a set
      of concrete ``assert`` tests, all of which PASS (verified by execution);
  (b) a BUG injected by AST mutation (a flipped comparison, a wrong arithmetic
      operator, swapped operands, an off-by-one loop bound, an extra ``+ 1`` on
      the return, an off-by-one constant, or a shifted index) such that AT LEAST
      ONE test now FAILS with a WRONG VALUE (verified by real execution; crashes
      and no-op mutations are rejected);
  (c) the CONCRETE failure (the failing assertion, expected vs got);
  (d) prompt = the buggy code + the tests + the concrete failure output +
      "Diagnose the bug and give the corrected code."; think = a short localized
      diagnosis grounded in the failure (which line/op is wrong and why); answer
      = the CORRECTED function (which passes ALL tests).

Why this shape installs the agentic gap and respects provenance:

- The gap is not writing one function (HumanEval 76%) but persisting through a
  failed check (duet-eval 23%). Self-repair drills exactly the plan-fails ->
  diagnose -> fix loop.
- The signal is SELF-GENERATED and EXECUTION-VERIFIED (no larger teacher): we
  INJECTED the bug, so we KNOW the fix, and every buggy/corrected pair is
  confirmed by REAL CPython execution against the concrete tests.
- It looks like NOTHING in HumanEval / MBPP (those are ``spec -> function``;
  this is ``buggy-function + failing-test -> corrected-function``), enforced by
  ``scripts/contamination``.

Triple truth audit for every row (never ship an unverified pair):

1. The CORRECT function passes ALL of its concrete tests (real execution).
2. The BUGGY function fails AT LEAST ONE test with a wrong value and RAISES on
   NONE of them (real execution; a mutation that crashes or preserves behavior
   is rejected).
3. The corrected code DIFFERS from the buggy code, and the shipped failure
   output matches the actual first failing test.

Safety/termination: no imports, no I/O, restricted builtins, only bounded
for-loops (never ``while``), a per-call step cap that aborts and discards.
Determinism: construction seed (default 91330); the corpus is a pure function
of the seed. A MIXED, frozen difficulty schedule (short/medium/long, biased to
medium/long because the base already handles trivial bugs) drives the loop.

Forgetting guard (documented design decision): unlike pure execution-TRACING,
the self-repair task ENDS by emitting CODE (the correction) under a distinct
instruction ("Diagnose the bug and give the corrected code"), so it does not
bias the model away from code generation. The dose is still kept moderate
(~500 rows, 1 epoch) and mixed-difficulty. A ``--mix-retention R`` switch
(default OFF) additionally blends in R self-generated plain
``spec -> function`` code rows for re-running if a probe regresses HumanEval.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contamination as contam  # noqa: E402

EXP = Path(__file__).resolve().parents[1]

# Construction seed (verify grep-fresh). The corpus is a pure function of it.
CONSTRUCTION_SEED = 91330

STEP_CAP = 4000  # line events per function call; exceeding ABORTS and discards.

# Frozen mixed difficulty schedule (biased to medium/long). Total 504 (divisible
# by the batch*grad-accum of 8 -> 63 whole optimizer steps).
TIER_SCHEDULE = (("short", 120), ("medium", 192), ("long", 192))
DEFAULT_ROWS = sum(count for _, count in TIER_SCHEDULE)

INSTRUCTION = "The code above fails its tests. Diagnose the bug and give the corrected code."

# Contamination-clean identifier pools (asserted against the banned set by the
# unit tests). Reused from the proven exec-trace pools (no ``count``/``find``/
# ``sort``/``check``/``maximum``/``minimum`` — all benchmark def names).
FUNC_NAMES = (
    "compute", "combine", "fold", "grow", "mix", "accumulate", "aggregate",
    "transform", "evaluate", "process", "derive", "resolve", "gather", "blend",
    "condense", "tabulate", "scan", "distill", "fuse", "morph", "crunch",
    "tally_up", "roll_up", "scale_up",
)
LIST_PARAMS = ("seq", "vec", "buf", "bag")
INT_PARAMS = ("n", "k", "m", "p")
# Accumulator / scratch names (contamination-clean AND absent from benchmark
# code idioms: ``total``/``res``/``prod`` collide with universal accumulator
# n-grams in the benchmarks, so they are deliberately excluded here).
ACC_NAMES = ("acc", "tally", "best", "cur", "cnt", "carry", "amt", "gain")


# --------------------------------------------------------------- safe execution
class StepCapExceeded(RuntimeError):
    pass


SAFE_BUILTINS = {
    "range": range, "len": len, "abs": abs, "min": min, "max": max,
    "sum": sum, "str": str, "int": int, "bool": bool, "list": list,
}


def _define_function(source: str, name: str):
    namespace: dict = {"__builtins__": SAFE_BUILTINS}
    code = compile(source, "<self_repair_fn>", "exec")
    exec(code, namespace)  # noqa: S102 (sandboxed: restricted builtins, no imports)
    fn = namespace.get(name)
    if fn is None or not callable(fn):
        raise ValueError(f"source did not define callable {name!r}")
    return fn, namespace


def _run_capped(callable_zero_arg):
    """Run a zero-arg callable under a real-CPython step cap (bounded loops)."""
    state = {"count": 0}

    def tracer(frame, event, arg):  # noqa: ARG001
        if event == "line":
            state["count"] += 1
            if state["count"] > STEP_CAP:
                raise StepCapExceeded("step cap")
        return tracer

    old = sys.gettrace()
    sys.settrace(tracer)
    try:
        return callable_zero_arg()
    finally:
        sys.settrace(old)


JSONABLE_SCALAR = (int, str)


def _jsonable_value(value) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return -10**6 < value < 10**6
    if isinstance(value, str):
        return len(value) <= 64
    if isinstance(value, list):
        return all(_jsonable_value(v) for v in value) and len(value) <= 64
    return False


def call_function(source: str, name: str, args: tuple):
    """Call ``name(*args)`` defined by ``source`` under safe builtins + step cap.

    Returns the value on success; raises on any exception (including StepCap).
    """
    fn, _ = _define_function(source, name)
    return _run_capped(lambda: fn(*args))


# --------------------------------------------------------------- rendering
def render_call(name: str, args: tuple) -> str:
    return f"{name}(" + ", ".join(repr(a) for a in args) + ")"


def render_assert(name: str, args: tuple, expected) -> str:
    return f"assert {render_call(name, args)} == {expected!r}"


def insert_docstring(source: str, doc: str) -> str:
    """Insert a one-line docstring as the first body statement of the def."""
    lines = source.split("\n")
    out: list[str] = []
    inserted = False
    for line in lines:
        out.append(line)
        if not inserted and line.startswith("def "):
            out.append(f'    """{doc}"""')
            inserted = True
    if not inserted:
        raise ValueError("no def line to attach a docstring to")
    return "\n".join(out)


def normalize_source(template_source: str) -> tuple[str, ast.Module]:
    """Canonicalize the template to ``ast.unparse`` form and return its tree.

    Canonicalizing both the correct and the mutated tree through ``ast.unparse``
    makes their rendered text differ on EXACTLY the mutated line, so the line
    diff isolates the bug cleanly.
    """
    tree = ast.parse(template_source)
    return ast.unparse(tree), tree


# --------------------------------------------------------------- AST mutation
COMPARE_VARIANTS = {
    ast.Lt: (ast.LtE, ast.Gt),
    ast.LtE: (ast.Lt, ast.Gt),
    ast.Gt: (ast.GtE, ast.Lt),
    ast.GtE: (ast.Gt, ast.Lt),
    ast.Eq: (ast.NotEq,),
    ast.NotEq: (ast.Eq,),
}
ARITH_VARIANTS = {
    ast.Add: (ast.Sub, ast.Mult),
    ast.Sub: (ast.Add, ast.Mult),
    ast.Mult: (ast.Add, ast.Sub),
    ast.FloorDiv: (ast.Mult, ast.Add),
    ast.Mod: (ast.FloorDiv, ast.Add),
}


def _compares(tree) -> list:
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.Compare) and len(n.ops) == 1 and type(n.ops[0]) in COMPARE_VARIANTS]


def _binops(tree) -> list:
    return [n for n in ast.walk(tree) if isinstance(n, ast.BinOp) and type(n.op) in ARITH_VARIANTS]


def _subs(tree) -> list:
    return [n for n in ast.walk(tree) if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Sub)]


def _ranges(tree) -> list:
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "range" and n.args]


def _returns(tree) -> list:
    return [n for n in ast.walk(tree) if isinstance(n, ast.Return) and n.value is not None]


def _int_consts(tree) -> list:
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, int) and not isinstance(n.value, bool)]


def _name_subscripts(tree) -> list:
    return [n for n in ast.walk(tree) if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Name)]


def enumerate_candidates(tree) -> list[tuple]:
    """All (kind, category_index, new_op-or-None) single-node mutation sites."""
    cands: list[tuple] = []
    for ci, node in enumerate(_compares(tree)):
        for new in COMPARE_VARIANTS[type(node.ops[0])]:
            cands.append(("compare_op", ci, new))
    for bi, node in enumerate(_binops(tree)):
        for new in ARITH_VARIANTS[type(node.op)]:
            cands.append(("arith_op", bi, new))
    for si in range(len(_subs(tree))):
        cands.append(("operand_swap", si, None))
    for ri in range(len(_ranges(tree))):
        cands.append(("range_bound", ri, None))
    for reti in range(len(_returns(tree))):
        cands.append(("return_offset", reti, None))
    for coi in range(len(_int_consts(tree))):
        cands.append(("const_offset", coi, None))
    for sui in range(len(_name_subscripts(tree))):
        cands.append(("index_shift", sui, None))
    return cands


def apply_candidate(tree, cand: tuple):
    kind, idx, new = cand
    t = copy.deepcopy(tree)
    if kind == "compare_op":
        _compares(t)[idx].ops[0] = new()
    elif kind == "arith_op":
        _binops(t)[idx].op = new()
    elif kind == "operand_swap":
        node = _subs(t)[idx]
        node.left, node.right = node.right, node.left
    elif kind == "range_bound":
        node = _ranges(t)[idx]
        last = node.args[-1]
        if (isinstance(last, ast.BinOp) and isinstance(last.op, ast.Add)
                and isinstance(last.right, ast.Constant) and last.right.value == 1):
            node.args[-1] = last.left  # X + 1 -> X (drops last iteration, clean)
        else:
            node.args[-1] = ast.BinOp(left=last, op=ast.Sub(), right=ast.Constant(value=1))
    elif kind == "return_offset":
        node = _returns(t)[idx]
        node.value = ast.BinOp(left=node.value, op=ast.Add(), right=ast.Constant(value=1))
    elif kind == "const_offset":
        node = _int_consts(t)[idx]
        node.value = node.value + 1
    elif kind == "index_shift":
        node = _name_subscripts(t)[idx]
        node.slice = ast.BinOp(left=node.slice, op=ast.Sub(), right=ast.Constant(value=1))
    else:  # pragma: no cover - defensive
        raise ValueError(f"unknown mutation kind: {kind}")
    ast.fix_missing_locations(t)
    return t


MECHANISM = {
    "compare_op": "The comparison operator on that line is wrong, so the condition takes the wrong branch.",
    "arith_op": "That line uses the wrong arithmetic operator, so it computes the wrong value.",
    "operand_swap": "The two operands of the subtraction are in the wrong order.",
    "range_bound": "The loop bound is off by one, so the loop runs one iteration too few.",
    "return_offset": "The returned expression adds an extra 1 that the specification does not call for.",
    "const_offset": "A numeric constant on that line is off by one.",
    "index_shift": "The index into the list is shifted by one position.",
}


# --------------------------------------------------------------- function families
def _distinct_reprs(values) -> int:
    return len({repr(v) for v in values})


def build_family(rng: random.Random, tier: str) -> dict | None:
    """Emit one correct function template + spec + concrete input tuples.

    Returns dict(name, params, source, spec, inputs) or None on a bad draw.
    """
    name = rng.choice(FUNC_NAMES)
    families = FAMILIES[tier]
    return rng.choice(families)(rng, name)


# -- short families -----------------------------------------------------------
def fam_clamp(rng: random.Random, name: str) -> dict:
    lo = rng.randint(0, 4)
    hi = lo + rng.randint(3, 8)
    src = (
        f"def {name}(x, low, high):\n"
        f"    if x < low:\n"
        f"        return low\n"
        f"    if x > high:\n"
        f"        return high\n"
        f"    return x\n"
    )
    spec = f"Return x limited to the band [low, high]: low if x is under low, high if x is over high, otherwise x."
    xs = [lo - rng.randint(1, 3), hi + rng.randint(1, 3), rng.randint(lo, hi), rng.randint(lo, hi)]
    inputs = [(x, lo, hi) for x in xs]
    return {"name": name, "source": src, "spec": spec, "inputs": inputs}


def fam_over_k(rng: random.Random, name: str) -> dict:
    lp = rng.choice(LIST_PARAMS)
    acc = rng.choice(("cnt", "tally"))
    src = (
        f"def {name}({lp}, k):\n"
        f"    {acc} = 0\n"
        f"    for v in {lp}:\n"
        f"        if v > k:\n"
        f"            {acc} = {acc} + 1\n"
        f"    return {acc}\n"
    )
    spec = f"Return how many values in {lp} are strictly greater than k."
    inputs = []
    for _ in range(4):
        length = rng.randint(4, 6)
        seq = [rng.randint(0, 9) for _ in range(length)]
        inputs.append((seq, rng.randint(2, 6)))
    return {"name": name, "source": src, "spec": spec, "inputs": inputs}


def fam_gap(rng: random.Random, name: str) -> dict:
    src = (
        f"def {name}(a, b):\n"
        f"    if a > b:\n"
        f"        return a - b\n"
        f"    return b - a\n"
    )
    spec = "Return the absolute gap between a and b (the larger minus the smaller)."
    inputs = []
    for _ in range(4):
        inputs.append((rng.randint(0, 15), rng.randint(0, 15)))
    return {"name": name, "source": src, "spec": spec, "inputs": inputs}


def fam_sum_scaled(rng: random.Random, name: str) -> dict:
    c = rng.randint(2, 4)
    ip = rng.choice(INT_PARAMS)
    acc = rng.choice(("acc", "tally"))
    src = (
        f"def {name}({ip}):\n"
        f"    {acc} = 0\n"
        f"    for i in range(1, {ip} + 1):\n"
        f"        {acc} = {acc} + i * {c}\n"
        f"    return {acc}\n"
    )
    spec = f"Return the sum of i * {c} for i from 1 to {ip} inclusive."
    inputs = [(v,) for v in sorted({rng.randint(2, 8) for _ in range(6)})][:4]
    return {"name": name, "source": src, "spec": spec, "inputs": inputs}


# -- medium families ----------------------------------------------------------
def fam_largest(rng: random.Random, name: str) -> dict:
    lp = rng.choice(LIST_PARAMS)
    src = (
        f"def {name}({lp}):\n"
        f"    best = {lp}[0]\n"
        f"    for i in range(1, len({lp})):\n"
        f"        if {lp}[i] > best:\n"
        f"            best = {lp}[i]\n"
        f"    return best\n"
    )
    spec = f"Return the largest value in {lp} (assume {lp} is non-empty)."
    inputs = []
    for _ in range(4):
        length = rng.randint(4, 6)
        inputs.append(([rng.randint(0, 20) for _ in range(length)],))
    return {"name": name, "source": src, "spec": spec, "inputs": inputs}


def fam_even_idx(rng: random.Random, name: str) -> dict:
    lp = rng.choice(LIST_PARAMS)
    acc = rng.choice(("acc", "tally"))
    src = (
        f"def {name}({lp}):\n"
        f"    {acc} = 0\n"
        f"    for i in range(len({lp})):\n"
        f"        if i % 2 == 0:\n"
        f"            {acc} = {acc} + {lp}[i]\n"
        f"    return {acc}\n"
    )
    spec = f"Return the sum of the values at even positions (0, 2, 4, ...) of {lp}."
    inputs = []
    for _ in range(4):
        length = rng.randint(4, 7)
        inputs.append(([rng.randint(0, 9) for _ in range(length)],))
    return {"name": name, "source": src, "spec": spec, "inputs": inputs}


def fam_product(rng: random.Random, name: str) -> dict:
    ip = rng.choice(INT_PARAMS)
    acc = rng.choice(("acc", "tally"))
    src = (
        f"def {name}({ip}):\n"
        f"    {acc} = 1\n"
        f"    for i in range(1, {ip} + 1):\n"
        f"        {acc} = {acc} * i\n"
        f"    return {acc}\n"
    )
    spec = f"Return the product 1 * 2 * ... * {ip} (the factorial of {ip})."
    inputs = [(v,) for v in sorted({rng.randint(2, 6) for _ in range(6)})][:4]
    return {"name": name, "source": src, "spec": spec, "inputs": inputs}


def fam_branch_sum(rng: random.Random, name: str) -> dict:
    acc = rng.choice(("acc", "tally"))
    src = (
        f"def {name}(n, k):\n"
        f"    {acc} = 0\n"
        f"    for i in range(n):\n"
        f"        if i < k:\n"
        f"            {acc} = {acc} + i\n"
        f"        else:\n"
        f"            {acc} = {acc} + 1\n"
        f"    return {acc}\n"
    )
    spec = "For i from 0 to n-1, increase a running sum by i while i is below k and by 1 otherwise; return the total."
    inputs = []
    for _ in range(4):
        n = rng.randint(4, 7)
        inputs.append((n, rng.randint(1, n)))
    return {"name": name, "source": src, "spec": spec, "inputs": inputs}


def fam_scale_list(rng: random.Random, name: str) -> dict:
    lp = rng.choice(LIST_PARAMS)
    c = rng.randint(2, 4)
    d = rng.randint(1, 5)
    src = (
        f"def {name}({lp}):\n"
        f"    out = []\n"
        f"    for v in {lp}:\n"
        f"        out.append(v * {c} + {d})\n"
        f"    return out\n"
    )
    spec = f"Return a new list where each value v of {lp} is replaced by v * {c} + {d}."
    inputs = []
    for _ in range(4):
        length = rng.randint(3, 5)
        inputs.append(([rng.randint(0, 9) for _ in range(length)],))
    return {"name": name, "source": src, "spec": spec, "inputs": inputs}


# -- long families ------------------------------------------------------------
def fam_nested_tri(rng: random.Random, name: str) -> dict:
    ip = rng.choice(INT_PARAMS)
    acc = rng.choice(("acc", "tally"))
    src = (
        f"def {name}({ip}):\n"
        f"    {acc} = 0\n"
        f"    for i in range({ip}):\n"
        f"        for j in range(i + 1):\n"
        f"            {acc} = {acc} + j\n"
        f"    return {acc}\n"
    )
    spec = f"For i from 0 to {ip}-1, take every j from 0 to i and increase a running sum by j; return the sum."
    inputs = [(v,) for v in sorted({rng.randint(3, 7) for _ in range(6)})][:4]
    return {"name": name, "source": src, "spec": spec, "inputs": inputs}


def fam_above_avg(rng: random.Random, name: str) -> dict:
    lp = rng.choice(LIST_PARAMS)
    acc = rng.choice(("acc", "tally"))
    src = (
        f"def {name}({lp}):\n"
        f"    {acc} = 0\n"
        f"    for v in {lp}:\n"
        f"        {acc} = {acc} + v\n"
        f"    avg = {acc} // len({lp})\n"
        f"    cnt = 0\n"
        f"    for v in {lp}:\n"
        f"        if v > avg:\n"
        f"            cnt = cnt + 1\n"
        f"    return cnt\n"
    )
    spec = f"Let avg be the integer average (floor) of {lp}; return how many values of {lp} exceed avg."
    inputs = []
    for _ in range(4):
        length = rng.randint(4, 6)
        inputs.append(([rng.randint(0, 12) for _ in range(length)],))
    return {"name": name, "source": src, "spec": spec, "inputs": inputs}


def fam_weighted(rng: random.Random, name: str) -> dict:
    lp = rng.choice(LIST_PARAMS)
    acc = rng.choice(("acc", "tally"))
    src = (
        f"def {name}({lp}):\n"
        f"    {acc} = 0\n"
        f"    for i in range(len({lp})):\n"
        f"        {acc} = {acc} + {lp}[i] * i\n"
        f"    return {acc}\n"
    )
    spec = f"Return the sum of {lp}[i] * i over every position i of {lp}."
    inputs = []
    for _ in range(4):
        length = rng.randint(4, 6)
        inputs.append(([rng.randint(0, 9) for _ in range(length)],))
    return {"name": name, "source": src, "spec": spec, "inputs": inputs}


def fam_running_cap(rng: random.Random, name: str) -> dict:
    lp = rng.choice(LIST_PARAMS)
    cap = rng.randint(8, 15)
    acc = rng.choice(("acc", "tally"))
    src = (
        f"def {name}({lp}):\n"
        f"    {acc} = 0\n"
        f"    out = []\n"
        f"    for v in {lp}:\n"
        f"        {acc} = {acc} + v\n"
        f"        if {acc} > {cap}:\n"
        f"            {acc} = {cap}\n"
        f"        out.append({acc})\n"
        f"    return out\n"
    )
    spec = f"Walk {lp} keeping a running sum that is held down to at most {cap}; return the list of running values."
    inputs = []
    for _ in range(4):
        length = rng.randint(4, 6)
        inputs.append(([rng.randint(0, 6) for _ in range(length)],))
    return {"name": name, "source": src, "spec": spec, "inputs": inputs}


def fam_spread(rng: random.Random, name: str) -> dict:
    lp = rng.choice(LIST_PARAMS)
    src = (
        f"def {name}({lp}):\n"
        f"    hi = {lp}[0]\n"
        f"    lo = {lp}[0]\n"
        f"    for v in {lp}:\n"
        f"        if v > hi:\n"
        f"            hi = v\n"
        f"        if v < lo:\n"
        f"            lo = v\n"
        f"    return hi - lo\n"
    )
    spec = f"Return the spread of {lp}: its largest value minus its smallest value (assume {lp} is non-empty)."
    inputs = []
    for _ in range(4):
        length = rng.randint(4, 6)
        inputs.append(([rng.randint(0, 20) for _ in range(length)],))
    return {"name": name, "source": src, "spec": spec, "inputs": inputs}


def fam_countdown(rng: random.Random, name: str) -> dict:
    lp = rng.choice(LIST_PARAMS)
    budget = rng.randint(15, 25)
    acc = rng.choice(("left", "rem"))
    src = (
        f"def {name}({lp}):\n"
        f"    {acc} = {budget}\n"
        f"    out = []\n"
        f"    for v in {lp}:\n"
        f"        {acc} = {acc} - v\n"
        f"        out.append({acc})\n"
        f"    return out\n"
    )
    spec = f"Start from {budget} and, for each value v of {lp}, subtract v; return the list of running remainders."
    inputs = []
    for _ in range(4):
        length = rng.randint(3, 5)
        inputs.append(([rng.randint(0, 6) for _ in range(length)],))
    return {"name": name, "source": src, "spec": spec, "inputs": inputs}


FAMILIES = {
    "short": (fam_clamp, fam_over_k, fam_sum_scaled, fam_gap),
    "medium": (fam_largest, fam_even_idx, fam_product, fam_branch_sum, fam_scale_list, fam_spread),
    "long": (fam_nested_tri, fam_above_avg, fam_weighted, fam_running_cap, fam_countdown),
}
ALL_MUTATION_KINDS = (
    "compare_op", "arith_op", "operand_swap", "range_bound",
    "return_offset", "const_offset", "index_shift",
)


# --------------------------------------------------------------- row build
def _select_mutation(rng: random.Random, tree, correct_src: str, name: str,
                     inputs: list[tuple], expected: list, doc: str, row_index: int):
    """Pick the first candidate that yields a clean, verified wrong-value bug."""
    candidates = enumerate_candidates(tree)
    if not candidates:
        return None
    # Rotate the preferred mutation KIND by row so the corpus spreads kinds.
    order = ALL_MUTATION_KINDS[row_index % len(ALL_MUTATION_KINDS):] + \
        ALL_MUTATION_KINDS[: row_index % len(ALL_MUTATION_KINDS)]
    rank = {kind: i for i, kind in enumerate(order)}
    grouped: dict[str, list[tuple]] = {}
    for cand in candidates:
        grouped.setdefault(cand[0], []).append(cand)
    for group in grouped.values():
        rng.shuffle(group)
    ordered: list[tuple] = []
    for kind in sorted(grouped, key=lambda kind: rank[kind]):
        ordered.extend(grouped[kind])

    for cand in ordered:
        try:
            mutated = apply_candidate(tree, cand)
            buggy_body = ast.unparse(mutated)
        except (ValueError, TypeError):
            continue
        buggy_src = insert_docstring(buggy_body, doc)
        if buggy_src == correct_src:
            continue
        # Exactly one changed line makes the diagnosis clean.
        cl, bl = correct_src.split("\n"), buggy_src.split("\n")
        if len(cl) != len(bl):
            continue
        changed = [i for i in range(len(cl)) if cl[i] != bl[i]]
        if len(changed) != 1:
            continue
        # Run the buggy function on every test: no crash, >= 1 wrong value.
        got: list = []
        crashed = False
        for args in inputs:
            try:
                got.append(call_function(buggy_src, name, args))
            except Exception:  # noqa: BLE001 (a crash disqualifies the mutation)
                crashed = True
                break
        if crashed:
            continue
        fail_idx = next((i for i in range(len(inputs)) if got[i] != expected[i]), None)
        if fail_idx is None:
            continue  # behavior-preserving mutation -> reject
        if not all(_jsonable_value(v) for v in got):
            continue
        line_i = changed[0]
        return {
            "kind": cand[0],
            "buggy_src": buggy_src,
            "changed_line": line_i,
            "buggy_line": bl[line_i].strip(),
            "correct_line": cl[line_i].strip(),
            "fail_idx": fail_idx,
            "got": got,
        }
    return None


def build_row(rng: random.Random, tier: str, index: int) -> dict | None:
    family = build_family(rng, tier)
    if family is None:
        return None
    name = family["name"]
    doc = family["spec"]
    inputs = [tuple(a) for a in family["inputs"]]
    if len(inputs) < 3:
        return None
    try:
        correct_body, tree = normalize_source(family["source"])
    except SyntaxError:
        return None
    correct_src = insert_docstring(correct_body, doc)
    # The correct function must pass ALL tests (real execution) and discriminate.
    try:
        expected = [call_function(correct_src, name, args) for args in inputs]
    except Exception:  # noqa: BLE001
        return None
    if not all(_jsonable_value(v) for v in expected):
        return None
    if _distinct_reprs(expected) < 2:
        return None  # a constant-output function makes a weak test suite

    mutation = _select_mutation(rng, tree, correct_src, name, inputs, expected, doc, index)
    if mutation is None:
        return None

    got = mutation["got"]
    fail_idx = mutation["fail_idx"]
    fail_args = inputs[fail_idx]
    fail_call = render_call(name, fail_args)
    exp_repr = repr(expected[fail_idx])
    got_repr = repr(got[fail_idx])

    tests = [render_assert(name, args, expected[i]) for i, args in enumerate(inputs)]
    asserts_block = "\n".join(tests)
    failure_block = (
        f"    {tests[fail_idx]}\n"
        f"AssertionError: {fail_call} returned {got_repr}, expected {exp_repr}"
    )
    buggy_src = mutation["buggy_src"]

    prompt = (
        "Here is a Python function with a docstring stating what it should do, "
        "followed by the tests it must satisfy.\n\n"
        f"```python\n{buggy_src}\n```\n\n"
        "Tests:\n\n"
        f"```python\n{asserts_block}\n```\n\n"
        "Running the tests, one of them fails:\n\n"
        f"{failure_block}\n\n"
        f"{INSTRUCTION}"
    )
    think = (
        f"The failing case is `{fail_call} == {exp_repr}`, but the code returns `{got_repr}`.\n"
        f"Reading the implementation against the docstring, the bug is on this line:\n"
        f"    {mutation['buggy_line']}\n"
        f"{MECHANISM[mutation['kind']]}\n"
        f"The corrected line is:\n"
        f"    {mutation['correct_line']}\n"
        f"With that one change the function returns `{exp_repr}` for {fail_call} and all {len(tests)} tests pass."
    )
    answer = f"```python\n{correct_src}\n```"

    return {
        "messages": [{"role": "user", "content": prompt}],
        "think": think,
        "answer": answer,
        "kind": "self_repair",
        "family": "cognitive_core",
        "tier": tier,
        "mutation_kind": mutation["kind"],
        "n_tests": len(tests),
        "row_weight": 1.0,
        "task_id": f"self_repair_{index:05d}",
        "_audit": {
            "truth_valid": True,
            "reexec_verified": True,
            "func_name": name,
            "buggy_code": buggy_src,
            "corrected_code": correct_src,
            "tests": tests,
            "fail_index": fail_idx,
            "fail_expected": exp_repr,
            "fail_got": got_repr,
        },
    }


# --------------------------------------------------- retention (forgetting guard)
def build_retention_row(rng: random.Random, index: int) -> dict | None:
    """A self-generated plain ``spec -> function`` row (keeps GENERATE-code mode)."""
    tier = rng.choice(("short", "medium"))
    family = build_family(rng, tier)
    if family is None:
        return None
    name = family["name"]
    doc = family["spec"]
    inputs = [tuple(a) for a in family["inputs"]]
    try:
        correct_body, _ = normalize_source(family["source"])
    except SyntaxError:
        return None
    correct_src = insert_docstring(correct_body, doc)
    try:
        expected = [call_function(correct_src, name, args) for args in inputs]
    except Exception:  # noqa: BLE001
        return None
    if not all(_jsonable_value(v) for v in expected) or _distinct_reprs(expected) < 2:
        return None
    signature = correct_src.split("\n", 1)[0]
    tests = [render_assert(name, args, expected[i]) for i, args in enumerate(inputs)]
    prompt = (
        "Write a Python function that satisfies this specification and passes "
        "the tests. Return only the function.\n\n"
        f"Specification: {doc}\n\n"
        f"Signature:\n\n```python\n{signature}\n    ...\n```\n\n"
        "Tests:\n\n"
        f"```python\n{chr(10).join(tests)}\n```"
    )
    return {
        "messages": [{"role": "user", "content": prompt}],
        "think": "I write the function directly from the specification, keeping it consistent with the tests.",
        "answer": f"```python\n{correct_src}\n```",
        "kind": "code_retention",
        "family": "cognitive_core",
        "tier": tier,
        "mutation_kind": "none",
        "n_tests": len(tests),
        "row_weight": 1.0,
        "task_id": f"code_retention_{index:05d}",
        "_audit": {"truth_valid": True, "retention": True, "corrected_code": correct_src, "tests": tests},
    }


# --------------------------------------------------------------- curriculum
def generate_curriculum(seed: int, tier_counts, mix_retention: int = 0):
    rng = random.Random(seed)
    rows: list[dict] = []
    index = 0
    seen_codes: set[tuple[str, str]] = set()
    for tier, count in tier_counts:
        made = 0
        attempts = 0
        while made < count:
            attempts += 1
            if attempts > count * 400 + 1000:
                raise RuntimeError(f"could not synthesize enough {tier} rows")
            row = build_row(rng, tier, index)
            if row is None:
                continue
            key = (row["_audit"]["buggy_code"], row["_audit"]["corrected_code"])
            if key in seen_codes:
                continue
            seen_codes.add(key)
            rows.append(row)
            index += 1
            made += 1
    retention_rows: list[dict] = []
    if mix_retention > 0:
        ridx = 0
        attempts = 0
        while len(retention_rows) < mix_retention:
            attempts += 1
            if attempts > mix_retention * 400 + 1000:
                raise RuntimeError("could not synthesize enough retention rows")
            row = build_retention_row(rng, ridx)
            if row is None:
                continue
            retention_rows.append(row)
            ridx += 1
    rng.shuffle(rows)
    all_rows = rows + retention_rows
    rng.shuffle(all_rows)
    return all_rows


def public_row(row: dict) -> dict:
    return {key: value for key, value in row.items() if not key.startswith("_")}


# ------------------------------------------------------- independent grading
def _split_assert(line: str) -> tuple[str, str]:
    body = line.strip()
    if not body.startswith("assert "):
        raise ValueError(f"not an assert line: {line!r}")
    body = body[len("assert "):]
    call_src, expected_src = body.rsplit(" == ", 1)
    return call_src.strip(), expected_src.strip()


def grade_by_asserts(code: str, name: str, tests: list[str]) -> dict:
    """Independently execute the code + evaluate each rendered assert.

    Different code path from generation: the shipped assert STRINGS are parsed
    and each call/expected expression is ``eval``'d, catching any tampering of
    the code, tests, or values. Returns per-test outcomes and the first failure.
    """
    fn, namespace = _define_function(code, name)
    outcomes: list[dict] = []
    first_fail: dict | None = None
    for line in tests:
        call_src, expected_src = _split_assert(line)
        expected = eval(compile(expected_src, "<expected>", "eval"), {"__builtins__": SAFE_BUILTINS})  # noqa: S307
        raised = None
        got = None
        try:
            got = _run_capped(lambda: eval(  # noqa: S307
                compile(call_src, "<call>", "eval"), namespace))
        except Exception as exc:  # noqa: BLE001
            raised = type(exc).__name__
        passed = raised is None and got == expected
        outcome = {"line": line, "passed": passed, "raised": raised, "got": got, "expected": expected}
        outcomes.append(outcome)
        if not passed and first_fail is None:
            first_fail = outcome
    n_fail = sum(1 for o in outcomes if not o["passed"])
    return {"outcomes": outcomes, "n_fail": n_fail, "first_fail": first_fail}


def extract_python_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    marker = "```python\n"
    idx = 0
    while True:
        start = text.find(marker, idx)
        if start < 0:
            break
        start += len(marker)
        end = text.find("\n```", start)
        if end < 0:
            break
        blocks.append(text[start:end])
        idx = end + len("\n```")
    return blocks


def _func_name_from_code(code: str) -> str:
    for line in code.split("\n"):
        stripped = line.strip()
        if stripped.startswith("def "):
            return stripped[len("def "):].split("(", 1)[0]
    raise ValueError("no def in code block")


def verify_row_reexecution(row: dict, index: int) -> None:
    """Independently re-execute a shipped row: buggy fails >= 1, corrected passes all, differ."""
    prompt = row["messages"][0]["content"]
    prompt_blocks = extract_python_blocks(prompt)
    answer_blocks = extract_python_blocks(row["answer"])
    if len(prompt_blocks) != 2 or len(answer_blocks) != 1:
        raise ValueError(f"row {index} unexpected code-block layout")
    buggy_code, tests_block = prompt_blocks
    corrected_code = answer_blocks[0]
    tests = [line for line in tests_block.split("\n") if line.strip()]
    if not tests:
        raise ValueError(f"row {index} has no tests")
    name = _func_name_from_code(corrected_code)
    if _func_name_from_code(buggy_code) != name:
        raise ValueError(f"row {index} buggy/corrected function names differ")
    if buggy_code == corrected_code:
        raise ValueError(f"row {index} correction does not differ from the buggy code")

    buggy_grade = grade_by_asserts(buggy_code, name, tests)
    if buggy_grade["n_fail"] < 1:
        raise ValueError(f"row {index} buggy code passes all tests (no real bug)")
    if any(o["raised"] for o in buggy_grade["outcomes"]):
        raise ValueError(f"row {index} buggy code raises (want a clean wrong-value failure)")
    corrected_grade = grade_by_asserts(corrected_code, name, tests)
    if corrected_grade["n_fail"] != 0:
        raise ValueError(f"row {index} corrected code does not pass all tests")

    # The shipped failure output must match the actual first failing test.
    fail = buggy_grade["first_fail"]
    fail_line = fail["line"].strip()
    if fail_line not in prompt:
        raise ValueError(f"row {index} failure line not shown in prompt")
    if f"expected {fail['expected']!r}" not in prompt or f"returned {fail['got']!r}" not in prompt:
        raise ValueError(f"row {index} shown failure value disagrees with re-execution")


# ----------------------------------------------------------------- validation
def validate_generated(rows: list[dict], *, expected_rows: int | None = None) -> dict:
    if not rows:
        raise ValueError("curriculum is empty")
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, got {len(rows)}")
    required = {
        "messages", "think", "answer", "kind", "family", "tier", "mutation_kind",
        "n_tests", "row_weight", "task_id", "_audit",
    }
    banned = contam.banned_names()
    prompts: set[str] = set()
    task_ids: set[str] = set()
    code_pairs: set[tuple[str, str]] = set()
    tiers: Counter = Counter()
    kinds: Counter = Counter()
    mutations: Counter = Counter()
    for i, row in enumerate(rows):
        if set(row) != required:
            raise ValueError(f"row {i} schema mismatch: {sorted(row)}")
        if row["family"] != "cognitive_core" or row["row_weight"] != 1.0:
            raise ValueError(f"row {i} family/weight mismatch")
        if len(row["messages"]) != 1 or row["messages"][0].get("role") != "user":
            raise ValueError(f"row {i} message schema mismatch")
        if not row["think"].strip() or not row["answer"].strip():
            raise ValueError(f"row {i} empty target")
        if row["_audit"].get("truth_valid") is not True:
            raise ValueError(f"row {i} lacks truth audit")
        kinds[row["kind"]] += 1
        mutations[row["mutation_kind"]] += 1
        if row["kind"] == "self_repair":
            tiers[row["tier"]] += 1
            verify_row_reexecution(row, i)  # independent re-execution
        blob = row["messages"][0]["content"] + "\n" + row["think"] + "\n" + row["answer"]
        hits = contam.whole_word_hits(blob, banned)
        if hits:
            raise ValueError(f"row {i} uses banned benchmark vocabulary: {sorted(hits)}")
        prompt = row["messages"][0]["content"]
        if prompt in prompts or row["task_id"] in task_ids:
            raise ValueError(f"row {i} duplicate prompt/task_id")
        prompts.add(prompt)
        task_ids.add(row["task_id"])
        code_pairs.add((row["_audit"].get("buggy_code", ""), row["_audit"]["corrected_code"]))
    return {
        "rows": len(rows),
        "kinds": dict(sorted(kinds.items())),
        "tiers": dict(sorted(tiers.items())),
        "mutation_kinds": dict(sorted(mutations.items())),
        "unique_code_pairs": len(code_pairs),
    }


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def code_grams_no_docstring(code: str) -> set[tuple[str, ...]]:
    """Code n-grams with the docstring SPEC prose excluded.

    The code-overlap aid measures reuse of executable CODE, not of spec prose
    (spec prose is governed by the whole-word banned-name audit). The docstring
    is the single first body line; we segment the code AT it so no false
    cross-line n-gram is created by its removal, then n-gram each segment.
    """
    segments: list[str] = []
    current: list[str] = []
    for line in code.split("\n"):
        if line.strip().startswith('"""'):
            if current:
                segments.append("\n".join(current))
                current = []
            continue
        current.append(line)
    if current:
        segments.append("\n".join(current))
    grams: set[tuple[str, ...]] = set()
    for segment in segments:
        grams |= contam.code_ngrams(segment)
    return grams


def verify_public_corpus(path: Path, receipt_path: Path | None = None) -> dict:
    """Independently re-execute EVERY shipped row (no ``_audit``). Standalone."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("committed corpus is empty")
    banned = contam.banned_names()
    prompts: set[str] = set()
    task_ids: set[str] = set()
    tiers: Counter = Counter()
    kinds: Counter = Counter()
    mutations: Counter = Counter()
    for i, row in enumerate(rows):
        kinds[row["kind"]] += 1
        mutations[row["mutation_kind"]] += 1
        prompt = row["messages"][0]["content"]
        if prompt in prompts or row["task_id"] in task_ids:
            raise ValueError(f"row {i} duplicate in committed corpus")
        prompts.add(prompt)
        task_ids.add(row["task_id"])
        blob = prompt + "\n" + row["think"] + "\n" + row["answer"]
        hits = contam.whole_word_hits(blob, banned)
        if hits:
            raise ValueError(f"row {i} banned vocabulary in committed corpus: {sorted(hits)}")
        if row["kind"] == "self_repair":
            tiers[row["tier"]] += 1
            verify_row_reexecution(row, i)
    if receipt_path is not None and receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("corpus_sha256") != sha256_text(path.read_text(encoding="utf-8")):
            raise ValueError("committed corpus sha256 disagrees with the receipt")
    return {
        "rows": len(rows),
        "kinds": dict(sorted(kinds.items())),
        "tiers": dict(sorted(tiers.items())),
        "mutation_kinds": dict(sorted(mutations.items())),
    }


def _print_example(row: dict) -> None:
    print("=" * 78)
    print(f"[{row['tier']}] {row['task_id']} (mutation={row['mutation_kind']}, n_tests={row['n_tests']})")
    print("--- PROMPT ---")
    print(row["messages"][0]["content"])
    print("--- THINK (diagnosis) ---")
    print(row["think"])
    print("--- ANSWER (corrected code) ---")
    print(row["answer"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=CONSTRUCTION_SEED)
    parser.add_argument("--out", type=Path, default=EXP / "data" / "sft_self_repair.jsonl")
    parser.add_argument("--receipt", type=Path, default=EXP / "data" / "curriculum_receipt.json")
    parser.add_argument("--mix-retention", type=int, default=0,
                        help="blend in R self-generated spec->code rows (forgetting guard; default OFF)")
    parser.add_argument("--smoke", action="store_true", help="tiny build (a few rows per tier)")
    parser.add_argument("--examples", type=int, default=0, help="print N example rows across tiers and exit")
    parser.add_argument("--verify-corpus", action="store_true",
                        help="independently re-execute the committed corpus (no write, no GPU)")
    args = parser.parse_args()

    if args.verify_corpus:
        summary = verify_public_corpus(args.out, args.receipt)
        print(json.dumps({"verify_corpus": str(args.out), **summary}, indent=2, sort_keys=True))
        return 0

    if args.smoke:
        tier_counts = (("short", 4), ("medium", 5), ("long", 4))
    else:
        tier_counts = TIER_SCHEDULE
    rows = generate_curriculum(args.seed, tier_counts, mix_retention=args.mix_retention)
    expected = None if args.smoke else (DEFAULT_ROWS + args.mix_retention)
    summary = validate_generated(rows, expected_rows=expected)

    if args.examples:
        shown = 0
        for tier in ("short", "medium", "long"):
            for row in rows:
                if row.get("tier") == tier and row["kind"] == "self_repair":
                    _print_example(row)
                    shown += 1
                    break
            if shown >= args.examples:
                break
        return 0

    payload = "".join(json.dumps(public_row(row), ensure_ascii=False) + "\n" for row in rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(payload, encoding="utf-8")

    overlap = {"status": "skipped_cache_absent"}
    try:
        streams = contam.build_code_tokens_from_cache()
        bench_grams = contam.benchmark_ngrams(streams)
        corpus_grams: set = set()
        for row in rows:
            corpus_grams |= code_grams_no_docstring(row["_audit"].get("buggy_code", ""))
            corpus_grams |= code_grams_no_docstring(row["_audit"]["corrected_code"])
        shared = corpus_grams & bench_grams
        distinctive = contam.distinctive_overlap(corpus_grams, bench_grams)
        overlap = {
            "status": "checked",
            "ngram_n": contam.NGRAM_N,
            "note": "code-only n-grams; docstring spec prose excluded (audited by the banned-name gate)",
            "benchmark_ngrams": len(bench_grams),
            "corpus_ngrams": len(corpus_grams),
            "shared_ngrams_structural_idiom": len(shared),
            "shared_ngrams_distinctive": len(distinctive),
        }
        if distinctive:
            raise SystemExit(
                f"benchmark n-gram overlap detected: {len(distinctive)} shared "
                f"distinctive {contam.NGRAM_N}-grams: {sorted(list(distinctive))[:3]}"
            )
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 (datasets/cache unavailable -> skip aid)
        overlap = {"status": "skipped", "reason": type(exc).__name__}

    banned = contam.banned_names()
    receipt = {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "seed": args.seed,
        "out": str(args.out),
        "corpus_sha256": sha256_text(payload),
        "mix_retention": args.mix_retention,
        "tier_schedule": [list(item) for item in tier_counts],
        "contamination": {
            "banned_function_names": len(banned),
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
