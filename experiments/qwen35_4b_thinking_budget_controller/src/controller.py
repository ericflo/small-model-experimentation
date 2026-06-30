"""Offline thinking-budget controller simulation.

Reuses the greedy generations from `qwen35_4b_thinking_budget_scaling` (copied into
`data/`): for each task and each thinking budget we already know the greedy answer's
full-test pass; here we additionally re-verify the *visible* test (the one assert shown
in the prompt) and use it as the only deployable signal a controller may read.

A controller decides, per task, how much to think — instead of one fixed budget. We
simulate visible-test escalation ladders (s1/STOP-MORE style) and compare their
deployable full-test accuracy vs mean thinking-token cost against fixed budgets and a
non-deployable oracle ceiling.
"""
from __future__ import annotations

import ast
import json
import multiprocessing as mp
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
DATA = EXP / "data"

# ascending forced budgets + endpoints (by thinking-token cap)
BUDGET_ORDER = ["no_think", "think_256", "think_512", "think_1024", "think_2048", "think_unbudgeted"]


# --------------------------------------------------------------------------- verify
def _worker(code, imports, tests, q):
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (5, 6))
        resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    except Exception:
        pass
    g = {"__name__": "__candidate__"}
    try:
        for imp in imports:
            exec(imp, g)
        exec(code, g)
        for t in tests:
            exec(t, g)
        q.put(True)
    except Exception:
        q.put(False)


def _verify_once(code, imports, tests, timeout):
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(code, imports, tests, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate(); p.join(); return None  # timeout
    try:
        return q.get_nowait()
    except Exception:
        return False


def verify(code, imports, tests, timeout=10.0):
    if not code:
        return False
    try:
        ast.parse(code)
    except (SyntaxError, ValueError):
        return False
    r = _verify_once(code, imports, tests, timeout)
    if r is None:  # retry once on timeout
        r = _verify_once(code, imports, tests, timeout * 1.5)
    return bool(r)


# --------------------------------------------------------------------------- data
def load_cells(reverify=True):
    """Return cells[task_id][cond] = {visible_pass, full_pass, n_think}."""
    tasks = json.loads((DATA / "tasks.json").read_text())
    cache = DATA / "greedy_with_visible.jsonl"
    if cache.exists() and not reverify:
        recs = [json.loads(l) for l in cache.read_text().splitlines() if l.strip()]
    else:
        recs = [json.loads(l) for l in (DATA / "greedy_records.jsonl").read_text().splitlines() if l.strip()]
        for r in recs:
            tid = str(r["task_id"])
            tl = tasks[tid]["test_list"]
            imports = tasks[tid].get("test_imports", [])
            r["visible_pass"] = verify(r["code"], imports, tl[:1])  # first assert only
        with cache.open("w") as f:
            for r in recs:
                f.write(json.dumps({k: r[k] for k in
                        ["cond", "task_id", "visible_pass", "full_pass", "n_think", "n_gen", "forced"]}) + "\n")
    cells: dict = {}
    for r in recs:
        cells.setdefault(int(r["task_id"]), {})[r["cond"]] = {
            "visible_pass": bool(r["visible_pass"]), "full_pass": bool(r["full_pass"]),
            "n_think": r["n_think"]}
    return cells


# --------------------------------------------------------------------------- strategies
def fixed_budget(cells, cond):
    accs = [c[cond]["full_pass"] for c in cells.values()]
    cost = [c[cond]["n_think"] for c in cells.values()]
    return {"accuracy": _mean(accs), "mean_think": _mean(cost), "commit_dist": {cond: len(accs)},
            "false_visible_commit": _false_vis(cells, [cond])}


def escalation(cells, ladder, cumulative=True):
    """Walk `ladder` ascending; commit at first budget whose greedy passes the VISIBLE
    test; else commit the last rung. cumulative=True charges every attempted rung's
    thinking tokens (regeneration); False charges only the committed rung (continue-think)."""
    accs, costs, commit_at, fvc = [], [], {}, 0
    for c in cells.values():
        attempted = []
        chosen = ladder[-1]
        for b in ladder:
            attempted.append(b)
            if c[b]["visible_pass"]:
                chosen = b
                break
        accs.append(c[chosen]["full_pass"])
        cost = sum(c[b]["n_think"] for b in attempted) if cumulative else c[chosen]["n_think"]
        costs.append(cost)
        commit_at[chosen] = commit_at.get(chosen, 0) + 1
        if c[chosen]["visible_pass"] and not c[chosen]["full_pass"]:
            fvc += 1
    return {"accuracy": _mean(accs), "mean_think": _mean(costs), "commit_dist": commit_at,
            "false_visible_commit": fvc / len(cells)}


def oracle_ceiling(cells, ladder):
    """Non-deployable: per task pick the cheapest rung that FULL-passes (uses hidden tests)."""
    accs, costs = [], []
    for c in cells.values():
        passing = [b for b in ladder if c[b]["full_pass"]]
        if passing:
            best = min(passing, key=lambda b: c[b]["n_think"])
            accs.append(True); costs.append(c[best]["n_think"])
        else:
            accs.append(False); costs.append(c[ladder[-1]]["n_think"])
    return {"accuracy": _mean(accs), "mean_think": _mean(costs)}


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _false_vis(cells, conds):
    n = sum(1 for c in cells.values() for b in conds if c[b]["visible_pass"] and not c[b]["full_pass"])
    return n / (len(cells) * len(conds))
