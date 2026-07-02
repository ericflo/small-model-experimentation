"""Controlled factorial task generator: compositions with exact (depth d, #destructive k, positions).
Builds on gen_tasks primitives; classification of transparent vs destructive fixed in the pre-registration
BEFORE any data. Import path: expects gen_tasks on sys.path.
"""
from __future__ import annotations

import random
import gen_tasks as G

TRANSPARENT = ["reverse", "sort_asc", "sort_desc", "square", "negate", "add_k", "mul_k", "rotate_k",
               "running_sum", "adjacent_diff"]
DESTRUCTIVE = ["unique_stable", "dedup_adjacent", "abs_all", "filter_even", "filter_odd", "keep_positive",
               "filter_gt_k", "filter_lt_k", "take_k", "drop_k", "chunk_sum_k", "mod_k", "running_max"]
assert set(TRANSPARENT + DESTRUCTIVE) == set(p["name"] for p in G.PRIMS)


def _steps_for(d, k, rng):
    """Choose op names honoring (d, k): k destructive at random positions, rest transparent."""
    pos = sorted(rng.sample(range(d), k))
    names = []
    for i in range(d):
        pool = DESTRUCTIVE if i in pos else TRANSPARENT
        names.append(rng.choice(pool))
    steps = []
    for n in names:
        p = G.PRIM_BY_NAME[n]
        steps.append({"op": n, "k": G._sample_param(n, rng) if p["arity"] else None})
    return steps, pos


def make_controlled_task(task_id, d, k, rng, k_visible=10, m_hidden=8, tries=300):
    for _ in range(tries):
        steps, pos = _steps_for(d, k, rng)

        def target(xs):
            for s in steps:
                xs = G._apply(s, xs)
            return xs

        inputs, outputs, ok, seen = [], [], True, set()
        for _ in range((k_visible + m_hidden) * 3):
            xs = G._rand_input(rng)
            key = tuple(xs)
            if key in seen:
                continue
            seen.add(key)
            try:
                ys = target(list(xs))
            except Exception:
                ok = False; break
            if not all(isinstance(v, int) and -10**6 < v < 10**6 for v in ys):
                ok = False; break
            inputs.append(xs); outputs.append(ys)
            if len(inputs) >= k_visible + m_hidden:
                break
        if not ok or len(inputs) < k_visible + m_hidden:
            continue
        if all(o == i for i, o in zip(inputs, outputs)):
            continue
        if len({tuple(o) for o in outputs}) < max(3, (k_visible + m_hidden) // 2):
            continue
        if all(len(o) == 0 for o in outputs):
            continue
        ex = [{"input": i, "output": o} for i, o in zip(inputs, outputs)]
        return {"task_id": task_id, "depth": d, "n_destr": k, "destr_pos": pos,
                "target_ops": [G.step_repr(s) for s in steps],
                "visible": ex[:k_visible], "hidden": ex[k_visible:]}
    return None


def _min_depth_leq(task, cap):
    """True iff some pipeline of depth <= cap reproduces ALL (visible+hidden) examples. Exact BFS with
    behavioral-state dedup; bounded by seen-cap for tractability."""
    ex = task["visible"] + task["hidden"]
    inp = tuple(tuple(e["input"]) for e in ex)
    target = tuple(tuple(e["output"]) for e in ex)
    if inp == target:
        return True
    all_ops = [(p["name"], k) for p in G.PRIMS
               for k in ([None] if not p["arity"] else _PARAM_OPTS[p["name"]])]
    level, seen = {inp}, {inp}
    import decompose_lib as D
    for _ in range(cap):
        nxt = set()
        for state in level:
            for op, k in all_ops:
                new = D.apply_prim(op, k, state)
                if new is None or new in seen:
                    continue
                if new == target:
                    return True
                seen.add(new)
                nxt.add(new)
        level = nxt
        if not level or len(seen) > 400_000:
            break
    return False


_PARAM_OPTS = {
    "add_k": [-3, -2, -1, 1, 2, 3, 5], "mul_k": [-2, -1, 2, 3], "mod_k": [2, 3, 4, 5],
    "filter_gt_k": list(range(-4, 5)), "filter_lt_k": list(range(-4, 5)),
    "take_k": [1, 2, 3, 4], "drop_k": [1, 2, 3, 4], "chunk_sum_k": [1, 2, 3, 4], "rotate_k": [1, 2, 3, 4],
}


def build_grid(cells, n_per_cell, seed=0, verify_depth=True):
    """cells: list of (d, k). Exact factorial structure; if verify_depth, reject tasks whose behavioral
    min-depth < nominal (verified up to cap min(d-1, 3); d=5 tasks may still have d4-equivalents — noted
    in the prereg)."""
    rng = random.Random(seed)
    tasks, tid = [], 0
    for (d, k) in cells:
        made = 0
        while made < n_per_cell:
            t = make_controlled_task(tid, d, k, rng)
            if t is None:
                continue
            if verify_depth and d > 1 and _min_depth_leq(t, min(d - 1, 3)):
                continue  # collapsed -> reject
            tasks.append(t); tid += 1; made += 1
    return tasks
