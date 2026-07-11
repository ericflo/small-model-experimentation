"""runeward — calibrated abstention over rune-charge constraint systems (atoms).

Each item presents a small system of "bindings" over invented rune wards, each
holding an integer charge 1..9. By construction the system either has EXACTLY
ONE satisfying assignment (verified by enumeration at generation time) or is
provably unsatisfiable. The reply is the queried rune's charge, or exactly
IMPOSSIBLE.

Arm mix: 75% solvable / 25% UNSAT (deterministically, index % 4 == 3), so a
constant "ANSWER: IMPOSSIBLE" reply floors at exactly 0.25 by construction —
this family's selftest therefore passes degenerate_max=0.30 instead of the
default 0.15.

UNSAT items are built by altering ONE constraint of a solvable system and are
accepted only when the full system has zero solutions while every single
constraint AND every pair of constraints remains satisfiable (minimal
contradiction core >= 3): detecting unsatisfiability always requires chaining
constraints, never a bare direct pair like x=3 plus x=5. Solvable and UNSAT
items share the same constraint count per level and the same phrasing
templates, so there is no surface cue separating the arms.

Levels scale variables (3/4/4/5) and constraint counts (4/6/7/9); L1/L2 get a
direct-equality anchor (easy chain start), L3/L4 have none.
"""

from __future__ import annotations

from .. import base

FAMILY = "runeward"
LEVELS = (1, 2, 3, 4)
HAS_EPISODES = False

RUNES = (
    "thorn",
    "wex",
    "ando",
    "kelm",
    "brasp",
    "ule",
    "sorv",
    "quen",
    "mott",
    "drell",
    "fyra",
    "osk",
)

_NUM_WORD = {3: "Three", 4: "Four", 5: "Five"}

_LEVEL_SHAPE = {
    # level: (n_vars, n_constraints, n_eq_anchors, proposal kinds (weighted))
    1: (3, 4, 1, ("sum", "sum", "diff")),
    2: (4, 6, 1, ("sum", "sum", "diff", "gt")),
    3: (4, 7, 0, ("sum", "sum", "diff", "diff", "gt", "ne")),
    4: (5, 9, 0, ("sum", "sum", "diff", "diff", "gt", "ne")),
}

_UNSAT_EVERY = 4  # index % 4 == 3 -> exactly 25% UNSAT in any 4k-item batch


# ---------------------------------------------------------------------------
# Constraint core: tuples, evaluation, enumeration with short-circuit
# ---------------------------------------------------------------------------
# ("eq", i, v)       charge[i] == v
# ("sum", i, j, s)   charge[i] + charge[j] == s      (i < j)
# ("diff", i, j, d)  charge[i] - charge[j] == d      (d >= 1)
# ("gt", i, j)       charge[i] > charge[j]
# ("ne", i, j)       charge[i] != charge[j]          (i < j)


def _vars_of(constraint: tuple) -> tuple[int, ...]:
    if constraint[0] == "eq":
        return (constraint[1],)
    return (constraint[1], constraint[2])


def _holds(constraint: tuple, assign: list[int]) -> bool:
    kind = constraint[0]
    if kind == "eq":
        return assign[constraint[1]] == constraint[2]
    if kind == "sum":
        return assign[constraint[1]] + assign[constraint[2]] == constraint[3]
    if kind == "diff":
        return assign[constraint[1]] - assign[constraint[2]] == constraint[3]
    if kind == "gt":
        return assign[constraint[1]] > assign[constraint[2]]
    if kind == "ne":
        return assign[constraint[1]] != assign[constraint[2]]
    raise ValueError(f"unknown constraint kind {kind!r}")


def _count_solutions(n_vars: int, constraints: list[tuple], limit: int = 2) -> int:
    """Count satisfying assignments in {1..9}^n_vars, stopping at `limit`.

    Backtracking with early short-circuit: each constraint is checked as soon
    as its last variable is assigned, so contradictions prune immediately.
    """
    by_last: list[list[tuple]] = [[] for _ in range(n_vars)]
    for constraint in constraints:
        by_last[max(_vars_of(constraint))].append(constraint)
    count = 0
    assign = [0] * n_vars

    def rec(i: int) -> None:
        nonlocal count
        if count >= limit:
            return
        if i == n_vars:
            count += 1
            return
        checks = by_last[i]
        for value in range(1, 10):
            assign[i] = value
            ok = True
            for constraint in checks:
                if not _holds(constraint, assign):
                    ok = False
                    break
            if ok:
                rec(i + 1)
                if count >= limit:
                    return

    rec(0)
    return count


def _pair_satisfiable(c1: tuple, c2: tuple) -> bool:
    """True if the two constraints can hold together (over their own vars)."""
    idxs = sorted({v for c in (c1, c2) for v in _vars_of(c)})
    remap = {orig: k for k, orig in enumerate(idxs)}

    def remapped(constraint: tuple) -> tuple:
        kind = constraint[0]
        if kind == "eq":
            return ("eq", remap[constraint[1]], constraint[2])
        if kind in ("sum", "diff"):
            return (kind, remap[constraint[1]], remap[constraint[2]], constraint[3])
        return (kind, remap[constraint[1]], remap[constraint[2]])

    return _count_solutions(len(idxs), [remapped(c1), remapped(c2)], limit=1) >= 1


# ---------------------------------------------------------------------------
# System construction
# ---------------------------------------------------------------------------


def _propose(rng, values: list[int], kinds: tuple[str, ...], seen: set) -> tuple | None:
    """One new constraint consistent with the hidden assignment, or None."""
    n = len(values)
    for _ in range(40):
        kind = rng.choice(kinds)
        i, j = rng.sample(range(n), 2)
        if kind == "sum":
            a, b = (i, j) if i < j else (j, i)
            constraint = ("sum", a, b, values[a] + values[b])
        elif kind == "diff":
            if values[i] == values[j]:
                continue
            hi, lo = (i, j) if values[i] > values[j] else (j, i)
            constraint = ("diff", hi, lo, values[hi] - values[lo])
        elif kind == "gt":
            if values[i] == values[j]:
                continue
            hi, lo = (i, j) if values[i] > values[j] else (j, i)
            constraint = ("gt", hi, lo)
        else:  # ne
            if values[i] == values[j]:
                continue
            a, b = (i, j) if i < j else (j, i)
            constraint = ("ne", a, b)
        if constraint not in seen:
            return constraint
    return None


def _build_solvable(rng, level: int) -> tuple[list[int], list[tuple]] | None:
    """A system with EXACTLY ONE solution and exactly k_target constraints."""
    n_vars, k_target, n_eq, kinds = _LEVEL_SHAPE[level]
    values = [rng.randint(1, 9) for _ in range(n_vars)]
    constraints: list[tuple] = []
    seen: set = set()
    for i in rng.sample(range(n_vars), n_eq):
        constraint = ("eq", i, values[i])
        constraints.append(constraint)
        seen.add(constraint)
    # Phase 1: add consistent constraints until the solution is unique.
    while len(constraints) < k_target:
        constraint = _propose(rng, values, kinds, seen)
        if constraint is None:
            return None
        constraints.append(constraint)
        seen.add(constraint)
        if _count_solutions(n_vars, constraints, limit=2) == 1:
            break
    if _count_solutions(n_vars, constraints, limit=2) != 1:
        return None
    # Phase 2: pad with consistent (hence redundant) constraints to k_target,
    # so solvable and UNSAT arms always show the same constraint count.
    while len(constraints) < k_target:
        constraint = _propose(rng, values, kinds, seen)
        if constraint is None:
            return None
        constraints.append(constraint)
        seen.add(constraint)
    return values, constraints


def _alterations(rng, constraint: tuple) -> list[tuple]:
    """Candidate single-constraint perturbations (small, plausible edits)."""
    kind = constraint[0]
    out: list[tuple] = []
    if kind == "eq":
        _, i, v = constraint
        for nv in range(max(1, v - 3), min(9, v + 3) + 1):
            if nv != v:
                out.append(("eq", i, nv))
    elif kind == "sum":
        _, i, j, s = constraint
        for ns in range(max(2, s - 4), min(18, s + 4) + 1):
            if ns != s:
                out.append(("sum", i, j, ns))
    elif kind == "diff":
        _, i, j, d = constraint
        for nd in range(1, 9):
            if nd != d:
                out.append(("diff", i, j, nd))
        out.append(("diff", j, i, d))  # reversed orientation
    elif kind == "gt":
        _, i, j = constraint
        out.append(("gt", j, i))
    # "ne" has no UNSAT-inducing single edit in this grammar; skip it.
    rng.shuffle(out)
    return out


def _build_unsat(rng, level: int) -> tuple[list[int], list[tuple]] | None:
    """Alter ONE constraint of a solvable system into a chained contradiction.

    Accept only if the full system is UNSAT while every pair that includes the
    altered constraint is still satisfiable. Pairs of unaltered constraints
    are satisfiable a fortiori (the original solution satisfies them), so the
    minimal contradiction core has size >= 3: no bare direct pair.
    """
    built = _build_solvable(rng, level)
    if built is None:
        return None
    values, constraints = built
    n_vars = len(values)
    order = list(range(len(constraints)))
    rng.shuffle(order)
    for pos in order:
        for candidate in _alterations(rng, constraints[pos]):
            if candidate in constraints:
                continue
            trial = list(constraints)
            trial[pos] = candidate
            if _count_solutions(n_vars, trial, limit=1) != 0:
                continue
            others = [c for k, c in enumerate(trial) if k != pos]
            if all(_pair_satisfiable(candidate, other) for other in others):
                return values, trial
    return None


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_constraint(constraint: tuple, names: list[str], rng) -> str:
    kind = constraint[0]
    if kind == "eq":
        _, i, v = constraint
        return rng.choice(
            [
                f"The {names[i]} ward holds exactly {v}.",
                f"The charge of {names[i]} is exactly {v}.",
            ]
        )
    if kind == "sum":
        _, i, j, s = constraint
        return rng.choice(
            [
                f"The {names[i]} and {names[j]} charges add up to exactly {s}.",
                f"Together, {names[i]} and {names[j]} hold exactly {s}.",
            ]
        )
    if kind == "diff":
        _, i, j, d = constraint
        return rng.choice(
            [
                f"The {names[i]} charge is exactly {d} more than the {names[j]} charge.",
                f"The {names[i]} ward holds exactly {d} above the {names[j]} ward.",
            ]
        )
    if kind == "gt":
        _, i, j = constraint
        return rng.choice(
            [
                f"The {names[i]} charge is strictly greater than the {names[j]} charge.",
                f"The {names[i]} ward holds more than the {names[j]} ward.",
            ]
        )
    if kind == "ne":
        _, i, j = constraint
        return rng.choice(
            [
                f"The {names[i]} and {names[j]} charges are not equal.",
                f"The {names[i]} and {names[j]} wards hold different charges.",
            ]
        )
    raise ValueError(f"unknown constraint kind {kind!r}")


# ---------------------------------------------------------------------------
# Family contract
# ---------------------------------------------------------------------------


def gen_atoms(seed: int, level: int, n: int) -> list[dict]:
    items = []
    for index in range(n):
        for attempt in range(20):
            item = _gen_one(seed, level, index, attempt)
            if len(item["prompt"]) <= base.ATOM_PROMPT_CHAR_LIMIT:
                break
        items.append(item)
    return items


def _gen_one(seed: int, level: int, index: int, attempt: int) -> dict:
    unsat = index % _UNSAT_EVERY == _UNSAT_EVERY - 1
    n_vars = _LEVEL_SHAPE[level][0]
    for trial in range(300):
        rng = base.rng_for(FAMILY, seed, level, index, attempt, trial)
        built = _build_unsat(rng, level) if unsat else _build_solvable(rng, level)
        if built is not None:
            values, constraints = built
            break
    else:  # pragma: no cover - deterministic; selftest exercises all paths
        raise RuntimeError(
            f"{FAMILY}: could not build a system for "
            f"seed={seed} L{level} index={index} attempt={attempt}"
        )

    names = rng.sample(RUNES, n_vars)
    # Query a rune with no direct-equality anchor (same rule in both arms).
    eq_vars = {c[1] for c in constraints if c[0] == "eq"}
    query = rng.choice([i for i in range(n_vars) if i not in eq_vars])
    gold: int | str = "IMPOSSIBLE" if unsat else values[query]

    shown = list(constraints)
    rng.shuffle(shown)
    lines = [
        f"{_NUM_WORD[n_vars]} rune wards seal a vault door: "
        + ", ".join(names)
        + ".",
        "Each ward holds one whole-number charge from 1 to 9. Charges may",
        "repeat unless a binding forbids it. The warding script declares:",
        "",
    ]
    lines += [
        f"{k + 1}. {_render_constraint(constraint, names, rng)}"
        for k, constraint in enumerate(shown)
    ]
    lines += [
        "",
        "Either the bindings force a single possible assignment of charges,",
        "or they contradict one another and cannot all hold at once.",
        f"If they force one assignment, give the charge of the {names[query]}",
        "ward. If they contradict, give exactly the word IMPOSSIBLE.",
        "",
        base.ATOM_ANSWER_INSTRUCTION,
    ]
    prompt = "\n".join(lines)

    return {
        "id": f"{FAMILY}-L{level}-s{seed}-{index:04d}",
        "family": FAMILY,
        "level": level,
        "prompt": prompt,
        "gold": gold,
        # Lucky-guess guard: solvable answers range over charges 1..9; on the
        # UNSAT arm the effective guess is binary (a number vs IMPOSSIBLE).
        "answer_domain": 2 if unsat else 9,
    }


def score_atom(item: dict, reply_text: str) -> float:
    answer = base.extract_answer(reply_text)
    if answer is None:
        return 0.0
    gold = item["gold"]
    if gold == "IMPOSSIBLE":
        return 1.0 if base.canon_word(answer) == "impossible" else 0.0
    return 1.0 if base.canon_int(answer) == gold else 0.0


def oracle_atom(item: dict) -> str:
    return f"ANSWER: {item['gold']}"


def selftest() -> dict:
    module = __import__(__name__, fromlist=["x"])
    # Deviation from the default harness thresholds, by design: 25% of items
    # are UNSAT, so a constant "ANSWER: IMPOSSIBLE" reply scores exactly 0.25.
    # That floor is the point of a calibrated-abstention family; the ceiling
    # is therefore 0.30 here instead of the default 0.15.
    stats = base.selftest_atoms(
        module,
        degenerate_replies=("", "ANSWER: 0", "ANSWER: yes", "ANSWER: IMPOSSIBLE"),
        degenerate_max=0.30,
    )
    # Family-specific extras: exact arm mix and gold-domain sanity.
    for level in LEVELS:
        items = gen_atoms(7, level, 40)
        n_unsat = sum(1 for item in items if item["gold"] == "IMPOSSIBLE")
        if n_unsat != 10:
            raise base.SelftestError(
                f"{FAMILY} L{level}: expected 10/40 UNSAT items, got {n_unsat}"
            )
        for item in items:
            gold = item["gold"]
            if gold != "IMPOSSIBLE" and not (isinstance(gold, int) and 1 <= gold <= 9):
                raise base.SelftestError(
                    f"{FAMILY} L{level} {item['id']}: bad gold {gold!r}"
                )
        stats["levels"][level]["unsat_frac"] = round(n_unsat / len(items), 4)
    return stats
