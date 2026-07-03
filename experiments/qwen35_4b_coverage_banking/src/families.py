"""Fresh state-tracking families for the cross-family generality test. Each family defines primitives as
Python SNIPPETS operating on a single state variable (so reference code composes trivially and the sandbox
grades identically to the list substrate) plus a state type, input generator, and signature.

Families: list (anchor, matches gen_tasks), string (char edits), register (3-register machine).
"""
from __future__ import annotations

import random

# ---- LIST (anchor: same 23 primitives as gen_tasks.PYSRC) --------------------------------------
LIST_PRIMS = {
    "reverse": ("xs = xs[::-1]", 0, None), "sort_asc": ("xs = sorted(xs)", 0, None),
    "sort_desc": ("xs = sorted(xs, reverse=True)", 0, None),
    "unique_stable": ("xs = list(dict.fromkeys(xs))", 0, None),
    "dedup_adjacent": ("xs = [x for i, x in enumerate(xs) if i == 0 or x != xs[i - 1]]", 0, None),
    "abs_all": ("xs = [abs(x) for x in xs]", 0, None), "square": ("xs = [x * x for x in xs]", 0, None),
    "negate": ("xs = [-x for x in xs]", 0, None),
    "running_sum": ("xs = [sum(xs[:i + 1]) for i in range(len(xs))]", 0, None),
    "adjacent_diff": ("xs = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]", 0, None),
    "add_k": ("xs = [x + {k} for x in xs]", 1, [-3, -2, -1, 1, 2, 3]),
    "mul_k": ("xs = [x * {k} for x in xs]", 1, [-2, 2, 3]),
    "mod_k": ("xs = [x % {k} for x in xs]", 1, [2, 3, 4]),
    "take_k": ("xs = xs[:{k}]", 1, [1, 2, 3, 4]), "drop_k": ("xs = xs[{k}:]", 1, [1, 2, 3]),
    "rotate_k": ("xs = (xs[{k} % len(xs):] + xs[:{k} % len(xs)]) if xs else []", 1, [1, 2, 3]),
}

# ---- STRING (state = lowercase string) ---------------------------------------------------------
STRING_PRIMS = {
    "reverse": ("s = s[::-1]", 0, None), "sort_chars": ("s = ''.join(sorted(s))", 0, None),
    "dedup_adjacent": ("s = ''.join(c for i, c in enumerate(s) if i == 0 or c != s[i - 1])", 0, None),
    "remove_vowels": ("s = ''.join(c for c in s if c not in 'aeiou')", 0, None),
    "double": ("s = ''.join(c * 2 for c in s)", 0, None),
    "dedup_all": ("s = ''.join(dict.fromkeys(s))", 0, None),
    "swap_pairs": ("s = ''.join([s[i:i+2][::-1] for i in range(0, len(s), 2)])", 0, None),
    "shift_k": ("s = ''.join(chr((ord(c) - 97 + {k}) % 26 + 97) for c in s)", 1, [1, 2, 3, 13]),
    "take_k": ("s = s[:{k}]", 1, [1, 2, 3, 4]), "drop_k": ("s = s[{k}:]", 1, [1, 2, 3]),
    "rotate_k": ("s = (s[{k} % len(s):] + s[:{k} % len(s)]) if s else s", 1, [1, 2, 3]),
    "repeat_k": ("s = s * {k}", 1, [2, 3]),
    "keep_every_k": ("s = s[::{k}]", 1, [2, 3]),
}

# ---- REGISTER (state = 3-int list [a, b, c]) ----------------------------------------------------
REG_PRIMS = {
    "a_plus_b": ("r = [r[0] + r[1], r[1], r[2]]", 0, None),
    "b_plus_c": ("r = [r[0], r[1] + r[2], r[2]]", 0, None),
    "c_plus_a": ("r = [r[0], r[1], r[2] + r[0]]", 0, None),
    "neg_a": ("r = [-r[0], r[1], r[2]]", 0, None),
    "double_a": ("r = [r[0] * 2, r[1], r[2]]", 0, None),
    "swap_ab": ("r = [r[1], r[0], r[2]]", 0, None),
    "swap_bc": ("r = [r[0], r[2], r[1]]", 0, None),
    "rotate": ("r = [r[2], r[0], r[1]]", 0, None),
    "abs_all": ("r = [abs(x) for x in r]", 0, None),
    "inc_a": ("r = [r[0] + {k}, r[1], r[2]]", 1, [-2, -1, 1, 2, 3]),
    "mod_a": ("r = [r[0] % {k}, r[1], r[2]]", 1, [2, 3, 4]),
    "mul_a": ("r = [r[0] * {k}, r[1], r[2]]", 1, [-1, 2, 3]),
}


def _mk(name, prims, var, sig, mk_input, cap):
    return {"name": name, "prims": prims, "var": var, "sig": sig, "mk_input": mk_input, "cap": cap}


FAMILIES = {
    "list": _mk("list", LIST_PRIMS, "xs", "def transform(xs):",
                lambda rng: [rng.randint(-9, 9) for _ in range(rng.randint(4, 8))], 64),
    "string": _mk("string", STRING_PRIMS, "s", "def transform(s):",
                  lambda rng: "".join(rng.choice("abcdefghijklmnop") for _ in range(rng.randint(4, 8))), 40),
    "register": _mk("register", REG_PRIMS, "r", "def transform(r):",
                    lambda rng: [rng.randint(-9, 9) for _ in range(3)], None),
}


def snippet(fam, op, k):
    src, arity, _ = fam["prims"][op]
    return src.format(k=k) if arity else src


_CODE = {}  # (family_name, op, k) -> compiled code object (hot path for BFS)


def _code(fam, op, k):
    key = (fam["name"], op, k)
    c = _CODE.get(key)
    if c is None:
        c = compile(snippet(fam, op, k), "<op>", "exec")
        _CODE[key] = c
    return c


def apply_op(fam, op, k, state):
    """Apply one op to a state value; return new value or None on error/explosion."""
    ns = {fam["var"]: (list(state) if isinstance(state, (list, tuple)) else state)}
    try:
        exec(_code(fam, op, k), {}, ns)
    except Exception:
        return None
    v = ns[fam["var"]]
    if fam["cap"] is not None and hasattr(v, "__len__") and len(v) > fam["cap"]:
        return None
    if fam["name"] in ("list", "register"):
        if not all(isinstance(x, int) and abs(x) < 10 ** 7 for x in v):
            return None
    return v


def op_repr(op, k):
    return f"{op}({k})" if k is not None else op


def reference_code(fam, ops):
    body = [f"    {fam['var']} = {'list(' + fam['var'] + ')' if fam['name'] != 'string' else fam['var']}"]
    for op, k in ops:
        body.append("    " + snippet(fam, op, k))
    body.append(f"    return {fam['var']}")
    return fam["sig"] + "\n" + "\n".join(body)


def all_ops(fam):
    out = []
    for name, (_, arity, opts) in fam["prims"].items():
        out += [(name, k) for k in opts] if arity else [(name, None)]
    return out


def _key(v):
    return tuple(v) if isinstance(v, list) else v


def min_depth_leq(fam, inputs, target, cap, n_probe=6, seen_cap=60_000):
    """True iff some pipeline of depth <= cap reproduces target for ALL probe inputs (behavioral BFS).
    Uses a fixed subset of n_probe inputs to disambiguate — enough to reject shallow-equivalents while
    keeping the branching tractable for the 64-op list family."""
    idx = list(range(min(n_probe, len(inputs))))
    start = tuple(_key(inputs[i]) for i in idx)
    tgt = tuple(_key(target[i]) for i in idx)
    if start == tgt:
        return True
    ops = all_ops(fam)
    level, seen = {start}, {start}
    for _ in range(cap):
        nxt = set()
        for st in level:
            for op, k in ops:
                new = tuple(_key(apply_op(fam, op, k, list(s) if isinstance(s, tuple) else s)) for s in st)
                if any(x is None for x in new) or new in seen:
                    continue
                if new == tgt:
                    return True
                seen.add(new); nxt.add(new)
        level = nxt
        if not level or len(seen) > seen_cap:
            break
    return False


def make_task(fam, task_id, depth, rng, k_visible=8, m_hidden=6, verify=True, tries=300):
    names = list(fam["prims"])
    for _ in range(tries):
        ops = []
        for _ in range(depth):
            op = rng.choice(names)
            _, arity, opts = fam["prims"][op]
            ops.append((op, rng.choice(opts) if arity else None))
        inputs, outputs, ok, seen = [], [], True, set()
        for _ in range((k_visible + m_hidden) * 3):
            xs = fam["mk_input"](rng)
            key = _key(xs)
            if key in seen:
                continue
            seen.add(key)
            st = xs
            for op, k in ops:
                st = apply_op(fam, op, k, st)
                if st is None:
                    ok = False; break
            if st is None:
                ok = False; break
            inputs.append(xs); outputs.append(st)
            if len(inputs) >= k_visible + m_hidden:
                break
        if not ok or len(inputs) < k_visible + m_hidden:
            continue
        if all(_key(o) == _key(i) for i, o in zip(inputs, outputs)):
            continue
        if len({_key(o) for o in outputs}) < max(3, (k_visible + m_hidden) // 2):
            continue
        if verify and depth > 1 and min_depth_leq(fam, inputs, outputs, min(depth - 1, 3)):
            continue
        ex = [{"input": i, "output": o} for i, o in zip(inputs, outputs)]
        return {"task_id": task_id, "family": fam["name"], "depth": depth,
                "target_ops": [op_repr(op, k) for op, k in ops],
                "visible": ex[:k_visible], "hidden": ex[k_visible:]}
    return None


def to_public_cases(task):
    return [{"call_expr": f"transform({ex['input']!r})", "expected_expr": f"{ex['output']!r}"}
            for ex in task["visible"]]


def to_hidden_asserts(task):
    return [f"assert transform({ex['input']!r}) == {ex['output']!r}" for ex in task["hidden"]]
