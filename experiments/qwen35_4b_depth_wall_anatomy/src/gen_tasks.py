"""Procedural, contamination-free program-synthesis substrate.

Each task is a random composition (depth D) of total primitive ops over list[int]. We execute the
composed target on random inputs to get I/O examples, then ask the model to synthesize a Python
`transform(xs)` matching held-out examples. Novel by construction (never in any pretraining corpus);
difficulty = composition depth. Graded functionally by held-out execution, so ANY program reproducing
the behaviour counts — this is inductive program synthesis, not spec-following.
"""
from __future__ import annotations

import random
from typing import Callable

# -- primitive library: each is total on any list[int] and returns list[int] -------------------
def _p(name, fn, arity=0):
    return {"name": name, "fn": fn, "arity": arity}


def _rotate(xs, k):
    if not xs:
        return []
    k %= len(xs)
    return xs[k:] + xs[:k]


def _chunk_sum(xs, k):
    return [sum(xs[i:i + k]) for i in range(0, len(xs), k)]


PRIMS = [
    _p("reverse", lambda xs: xs[::-1]),
    _p("sort_asc", lambda xs: sorted(xs)),
    _p("sort_desc", lambda xs: sorted(xs, reverse=True)),
    _p("unique_stable", lambda xs: list(dict.fromkeys(xs))),
    _p("dedup_adjacent", lambda xs: [x for i, x in enumerate(xs) if i == 0 or x != xs[i - 1]]),
    _p("abs_all", lambda xs: [abs(x) for x in xs]),
    _p("square", lambda xs: [x * x for x in xs]),
    _p("negate", lambda xs: [-x for x in xs]),
    _p("filter_even", lambda xs: [x for x in xs if x % 2 == 0]),
    _p("filter_odd", lambda xs: [x for x in xs if x % 2 != 0]),
    _p("keep_positive", lambda xs: [x for x in xs if x > 0]),
    _p("running_sum", lambda xs: [sum(xs[:i + 1]) for i in range(len(xs))]),
    _p("running_max", lambda xs: [max(xs[:i + 1]) for i in range(len(xs))]),
    _p("adjacent_diff", lambda xs: [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]),
    # parameterized (arity 1)
    _p("add_k", lambda xs, k: [x + k for x in xs], 1),
    _p("mul_k", lambda xs, k: [x * k for x in xs], 1),
    _p("mod_k", lambda xs, k: [x % k for x in xs], 1),
    _p("filter_gt_k", lambda xs, k: [x for x in xs if x > k], 1),
    _p("filter_lt_k", lambda xs, k: [x for x in xs if x < k], 1),
    _p("take_k", lambda xs, k: xs[:k], 1),
    _p("drop_k", lambda xs, k: xs[k:], 1),
    _p("rotate_k", _rotate, 1),
    _p("chunk_sum_k", _chunk_sum, 1),
]
PRIM_BY_NAME = {p["name"]: p for p in PRIMS}


def _sample_param(name: str, rng: random.Random) -> int:
    if name in ("mod_k",):
        return rng.randint(2, 5)
    if name in ("take_k", "drop_k", "chunk_sum_k"):
        return rng.randint(1, 4)
    if name in ("rotate_k",):
        return rng.randint(1, 4)
    if name in ("mul_k",):
        return rng.choice([-2, -1, 2, 3])
    if name in ("add_k",):
        return rng.choice([-3, -2, -1, 1, 2, 3, 5])
    if name in ("filter_gt_k", "filter_lt_k"):
        return rng.randint(-4, 4)
    return 0


def _apply(step, xs):
    p = PRIM_BY_NAME[step["op"]]
    return p["fn"](xs, step["k"]) if p["arity"] else p["fn"](xs)


def _compose(depth: int, rng: random.Random):
    steps = []
    for _ in range(depth):
        p = rng.choice(PRIMS)
        steps.append({"op": p["name"], "k": _sample_param(p["name"], rng) if p["arity"] else None})

    def target(xs):
        for s in steps:
            xs = _apply(s, xs)
        return xs
    return steps, target


def _rand_input(rng: random.Random):
    return [rng.randint(-9, 9) for _ in range(rng.randint(4, 9))]


def step_repr(s):
    return f"{s['op']}({s['k']})" if s["k"] is not None else s["op"]


def make_task(task_id: int, depth: int, rng: random.Random, k_visible=6, m_hidden=8, tries=200):
    """Return one non-degenerate task dict, or None if generation failed within `tries`."""
    for _ in range(tries):
        steps, target = _compose(depth, rng)
        inputs, outputs, ok = [], [], True
        seen = set()
        for _ in range((k_visible + m_hidden) * 3):
            xs = _rand_input(rng)
            key = tuple(xs)
            if key in seen:
                continue
            seen.add(key)
            try:
                ys = target(list(xs))
            except Exception:
                ok = False
                break
            if not all(isinstance(v, int) and -10**6 < v < 10**6 for v in ys):
                ok = False
                break
            inputs.append(xs)
            outputs.append(ys)
            if len(inputs) >= k_visible + m_hidden:
                break
        if not ok or len(inputs) < k_visible + m_hidden:
            continue
        # non-degeneracy: not identity, outputs vary, not all empty
        if all(o == i for i, o in zip(inputs, outputs)):
            continue
        if len({tuple(o) for o in outputs}) < max(3, (k_visible + m_hidden) // 2):
            continue
        if all(len(o) == 0 for o in outputs):
            continue
        ex = [{"input": i, "output": o} for i, o in zip(inputs, outputs)]
        return {
            "task_id": task_id, "depth": depth,
            "target_ops": [step_repr(s) for s in steps],  # reference only; NOT shown to model
            "visible": ex[:k_visible], "hidden": ex[k_visible:],
        }
    return None


def build_dataset(depths, n_per_depth, seed=0):
    rng = random.Random(seed)
    tasks, tid = [], 0
    for d in depths:
        made = 0
        while made < n_per_depth:
            t = make_task(tid, d, rng)
            if t is not None:
                tasks.append(t)
                tid += 1
                made += 1
    return tasks


# -- adapters to the reused sandbox + the model prompt -----------------------------------------
def to_public_cases(task):
    return [{"call_expr": f"transform({ex['input']!r})", "expected_expr": f"{ex['output']!r}"} for ex in task["visible"]]


def to_hidden_asserts(task):
    return [f"assert transform({ex['input']!r}) == {ex['output']!r}" for ex in task["hidden"]]


# inline Python source per primitive (operates on/returns list `xs`) -> reference oracle solver
PYSRC = {
    "reverse": "xs = xs[::-1]",
    "sort_asc": "xs = sorted(xs)",
    "sort_desc": "xs = sorted(xs, reverse=True)",
    "unique_stable": "xs = list(dict.fromkeys(xs))",
    "dedup_adjacent": "xs = [x for i, x in enumerate(xs) if i == 0 or x != xs[i - 1]]",
    "abs_all": "xs = [abs(x) for x in xs]",
    "square": "xs = [x * x for x in xs]",
    "negate": "xs = [-x for x in xs]",
    "filter_even": "xs = [x for x in xs if x % 2 == 0]",
    "filter_odd": "xs = [x for x in xs if x % 2 != 0]",
    "keep_positive": "xs = [x for x in xs if x > 0]",
    "running_sum": "xs = [sum(xs[:i + 1]) for i in range(len(xs))]",
    "running_max": "xs = [max(xs[:i + 1]) for i in range(len(xs))]",
    "adjacent_diff": "xs = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]",
    "add_k": "xs = [x + {k} for x in xs]",
    "mul_k": "xs = [x * {k} for x in xs]",
    "mod_k": "xs = [x % {k} for x in xs]",
    "filter_gt_k": "xs = [x for x in xs if x > {k}]",
    "filter_lt_k": "xs = [x for x in xs if x < {k}]",
    "take_k": "xs = xs[:{k}]",
    "drop_k": "xs = xs[{k}:]",
    "rotate_k": "xs = (xs[{k} % len(xs):] + xs[:{k} % len(xs)]) if xs else []",
    "chunk_sum_k": "xs = [sum(xs[i:i + {k}]) for i in range(0, len(xs), {k})]",
}


def reference_code(task) -> str:
    """A guaranteed-correct solver for the task (oracle ceiling + solvability check)."""
    body = ["    xs = list(xs)"]
    for s in task["target_ops"]:
        op = s[:s.index("(")] if "(" in s else s
        k = s[s.index("(") + 1:-1] if "(" in s else None
        line = PYSRC[op].format(k=k) if k is not None else PYSRC[op]
        body.append("    " + line)
    body.append("    return xs")
    return "def transform(xs):\n" + "\n".join(body)


def prompt_for(task) -> str:
    lines = [f"transform({ex['input']!r}) == {ex['output']!r}" for ex in task["visible"]]
    ex = "\n".join(lines)
    return (
        "You are given input/output examples of a Python function `transform` that maps a list of "
        "integers to a list of integers. Infer the rule and implement it.\n\n"
        f"Examples:\n{ex}\n\n"
        "Write a single Python function `def transform(xs):` that reproduces this behaviour for all "
        "such inputs. Respond with only the function in one ```python code block."
    )
