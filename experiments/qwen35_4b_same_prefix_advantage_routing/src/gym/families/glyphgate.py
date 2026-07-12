"""glyphgate — active rule induction over a hidden glyph-string machine.

A sealed machine maps glyph strings (dash-joined syllables from an invented
8-glyph alphabet) to glyph strings by a hidden rule. Atoms show a short probe
log (input -> output pairs) and ask the solver either to PREDICT the output
for a fresh input or to CONSTRUCT an input that the machine prints as a given
target (any valid preimage is accepted; scoring applies the hidden rule).
Episodes put the same machine behind an interactive gate: PROBE under a
strict budget, induce the rule, then OPEN with a preimage of the TARGET.

Rule kinds (drawn per item): cycle-shift substitution along the published
alphabet order, full reversal, swap of two fixed positions, conditional
rotate (left if the first glyph is in a secret pair, else right), and
overwrite-one-position-with-a-fixed-glyph. L4 and L5 chain two rules; L6
chains three. Frontier levels stretch the same machine to longer strings
(L5: length 7, L6: length 8), deeper composition, and bigger probe budgets.

Fairness: the prompt states the level's exact hypothesis space, and the
generator regenerates until every rule in that space consistent with the
probes (a) agrees on the queried output (predict atoms) or (b) has a
nonempty preimage set contained in the hidden rule's (invert atoms), so any
rational induction is scored correct. Anti-exploit guards reject items where
echoing the shown input or target would win, probe logs that show no change,
and probe logs that directly reveal a preimage of the target.
"""

from __future__ import annotations

import re
from itertools import combinations

from .. import base

FAMILY = "glyphgate"
LEVELS = (1, 2, 3, 4, 5, 6)
HAS_EPISODES = True

GLYPHS = ("za", "ke", "ro", "mu", "ti", "fen", "gor", "lus")
_G_INDEX = {glyph: i for i, glyph in enumerate(GLYPHS)}
N_GLYPHS = len(GLYPHS)

# The glyph alphabet is already abstract pseudo-syllables AND the hidden rule
# is re-drawn every item, so there is no fixed surface->answer mapping to
# memorize (the scorer is glyph-identity-aware, so generic reskinning is not
# applied here); skin-shuffling adds nothing this family does not already have.
SKINNABLE = ()

_LEVEL_SHAPE = {
    # level: string lengths, probe pairs shown (atoms), rule kinds,
    # compose = number of chained rule steps (1 = a single rule).
    1: {"lengths": (3,), "n_probes": 2, "kinds": ("rev", "swap", "setg"), "compose": 1},
    2: {"lengths": (4,), "n_probes": 3, "kinds": ("rev", "swap", "setg", "shift"), "compose": 1},
    3: {"lengths": (5,), "n_probes": 3, "kinds": ("rev", "swap", "setg", "shift", "crot"), "compose": 1},
    4: {"lengths": (5, 6), "n_probes": 4, "kinds": ("rev", "swap", "setg", "shift", "crot"), "compose": 2},
    5: {"lengths": (7,), "n_probes": 4, "kinds": ("rev", "swap", "setg", "shift", "crot"), "compose": 2},
    6: {"lengths": (8,), "n_probes": 5, "kinds": ("rev", "swap", "setg", "shift", "crot"), "compose": 3},
}

_EP_MAX_TURNS = {1: 4, 2: 6, 3: 10, 4: 14, 5: 18, 6: 22}
# L1-L4 keep the legacy level+2 probe budget byte-for-byte; frontier levels
# get a bigger allowance to match the deeper hypothesis space.
_EP_BUDGET = {1: 3, 2: 4, 3: 5, 4: 6, 5: 8, 6: 9}

_RULE_DESC = {
    "rev": "the whole string is reversed",
    "swap": "the glyphs at two secret fixed positions trade places",
    "setg": "one secret fixed position is overwritten with one secret fixed glyph",
    "shift": "every glyph steps forward the same secret count along the alphabet cycle",
    "crot": (
        "if the first glyph is in a secret pair of glyphs the whole string "
        "rotates one step left, otherwise one step right"
    ),
}


# ---------------------------------------------------------------------------
# Glyph-string plumbing
# ---------------------------------------------------------------------------


def _fmt(s: tuple) -> str:
    return "-".join(GLYPHS[g] for g in s)


def _parse_glyphs(text: str, length: int | None = None) -> tuple | None:
    """Parse 'za-ke-ro' (any case; -, space or comma separated) to an
    int-tuple, or None if any token is not a glyph or the length is wrong."""
    tokens = re.findall(r"[a-z]+", (text or "").lower())
    if not tokens:
        return None
    try:
        s = tuple(_G_INDEX[token] for token in tokens)
    except KeyError:
        return None
    if length is not None and len(s) != length:
        return None
    return s


def _rand_string(rng, length: int) -> tuple:
    return tuple(rng.randrange(N_GLYPHS) for _ in range(length))


# ---------------------------------------------------------------------------
# Rules: apply, invert, generate, enumerate
# ---------------------------------------------------------------------------


def _apply(rule, s: tuple) -> tuple:
    op = rule[0]
    if op == "shift":
        k = rule[1]
        return tuple((g + k) % N_GLYPHS for g in s)
    if op == "rev":
        return tuple(reversed(s))
    if op == "swap":
        i, j = rule[1], rule[2]
        out = list(s)
        out[i], out[j] = out[j], out[i]
        return tuple(out)
    if op == "setg":
        k, g = rule[1], rule[2]
        return s[:k] + (g,) + s[k + 1 :]
    if op == "crot":
        if s[0] == rule[1] or s[0] == rule[2]:
            return s[1:] + s[:1]  # rotate left
        return s[-1:] + s[:-1]  # rotate right
    if op == "comp":
        return _apply(rule[2], _apply(rule[1], s))
    raise ValueError(f"unknown rule op {op!r}")  # pragma: no cover


def _preimages(rule, t: tuple) -> set:
    """Exact preimage set of t under rule, among strings of len(t)."""
    op = rule[0]
    if op == "shift":
        k = rule[1]
        return {tuple((g - k) % N_GLYPHS for g in t)}
    if op == "rev":
        return {tuple(reversed(t))}
    if op == "swap":
        return {_apply(rule, t)}  # swap is an involution
    if op == "setg":
        k, g = rule[1], rule[2]
        if t[k] != g:
            return set()
        return {t[:k] + (x,) + t[k + 1 :] for x in range(N_GLYPHS)}
    if op == "crot":
        out = set()
        y = t[-1:] + t[:-1]  # undo a left rotation
        if y[0] == rule[1] or y[0] == rule[2]:
            out.add(y)
        z = t[1:] + t[:1]  # undo a right rotation
        if not (z[0] == rule[1] or z[0] == rule[2]):
            out.add(z)
        return out
    if op == "comp":
        result: set = set()
        for mid in _preimages(rule[2], t):
            result |= _preimages(rule[1], mid)
        return result
    raise ValueError(f"unknown rule op {op!r}")  # pragma: no cover


def _gen_single(rng, length: int, kinds: tuple) -> list:
    kind = rng.choice(kinds)
    if kind == "rev":
        return ["rev"]
    if kind == "shift":
        return ["shift", rng.randint(1, N_GLYPHS - 1)]
    if kind == "swap":
        i, j = sorted(rng.sample(range(length), 2))
        return ["swap", i, j]
    if kind == "setg":
        return ["setg", rng.randrange(length), rng.randrange(N_GLYPHS)]
    if kind == "crot":
        a, b = sorted(rng.sample(range(N_GLYPHS), 2))
        return ["crot", a, b]
    raise ValueError(f"unknown rule kind {kind!r}")  # pragma: no cover


def _gen_rule(rng, length: int, shape: dict) -> list:
    depth = shape["compose"]
    if depth <= 1:
        return _gen_single(rng, length, shape["kinds"])
    for _ in range(12):
        # RNG order (singles left-to-right, then 6 test strings) matches the
        # historical depth-2 path byte-for-byte.
        rule = _gen_single(rng, length, shape["kinds"])
        for _ in range(depth - 1):
            rule = ["comp", rule, _gen_single(rng, length, shape["kinds"])]
        tests = [_rand_string(rng, length) for _ in range(6)]
        if any(_apply(rule, t) != t for t in tests):
            return rule
    rule = ["comp", ["rev"], ["shift", 1]]  # unreachable in practice; never identity
    for _ in range(depth - 2):
        rule = ["comp", rule, ["shift", 1]]
    return rule


_SINGLES_CACHE: dict = {}


def _singles_pool(length: int, kinds: tuple) -> list:
    key = (length, kinds)
    if key not in _SINGLES_CACHE:
        rules: list = []
        for kind in kinds:
            if kind == "rev":
                rules.append(("rev",))
            elif kind == "shift":
                rules.extend(("shift", k) for k in range(1, N_GLYPHS))
            elif kind == "swap":
                rules.extend(("swap", i, j) for i, j in combinations(range(length), 2))
            elif kind == "setg":
                rules.extend(("setg", k, g) for k in range(length) for g in range(N_GLYPHS))
            elif kind == "crot":
                rules.extend(("crot", a, b) for a, b in combinations(range(N_GLYPHS), 2))
        _SINGLES_CACHE[key] = rules
    return _SINGLES_CACHE[key]


def _iter_consistent(length: int, kinds: tuple, depth: int, probes: list):
    """Yield every rule in the level's declared hypothesis space that maps
    each probe input to its shown output."""
    singles = _singles_pool(length, kinds)
    ins = [i for i, _ in probes]
    outs = [o for _, o in probes]
    if depth <= 1:
        for rule in singles:
            if all(_apply(rule, i) == o for i, o in probes):
                yield rule
        return
    if depth == 2:
        for r1 in singles:
            mids = [_apply(r1, i) for i in ins]
            for r2 in singles:
                ok = True
                for mid, out in zip(mids, outs):
                    if _apply(r2, mid) != out:
                        ok = False
                        break
                if ok:
                    yield ("comp", r1, r2)
        return
    # depth == 3: prune the cubic space with an index of last-step candidates
    # keyed by the value the second intermediate must take on probe 0 (the
    # exact preimages of out[0] under each candidate last step).
    last_by_mid0: dict = {}
    for r3 in singles:
        for pre in _preimages(r3, outs[0]):
            last_by_mid0.setdefault(pre, []).append(r3)
    for r1 in singles:
        mids1 = [_apply(r1, i) for i in ins]
        for r2 in singles:
            m0 = _apply(r2, mids1[0])
            candidates = last_by_mid0.get(m0)
            if not candidates:
                continue
            rest = [_apply(r2, mid) for mid in mids1[1:]]
            for r3 in candidates:
                if all(_apply(r3, mid) == out for mid, out in zip(rest, outs[1:])):
                    yield ("comp", ("comp", r1, r2), r3)


def _predict_wellposed(length, kinds, depth, probes, query, gold_out) -> bool:
    for rule in _iter_consistent(length, kinds, depth, probes):
        if _apply(rule, query) != gold_out:
            return False
    return True


def _invert_wellposed(length, kinds, depth, probes, target, hidden_pre) -> bool:
    for rule in _iter_consistent(length, kinds, depth, probes):
        pre = _preimages(rule, target)
        if not pre or not pre <= hidden_pre:
            return False
    return True


def _spec(rule) -> list:
    """JSON-safe (nested-list) copy of a rule spec."""
    return [_spec(part) if isinstance(part, (list, tuple)) else part for part in rule]


# ---------------------------------------------------------------------------
# Shared prompt text
# ---------------------------------------------------------------------------

_ALPHABET_LINE = (
    "The glyph alphabet, in cycle order: za, ke, ro, mu, ti, fen, gor, lus "
    "(after lus the cycle wraps back to za)."
)


def _menu(shape: dict) -> str:
    letters = "abcde"
    listing = "; ".join(
        f"({letters[i]}) {_RULE_DESC[kind]}" for i, kind in enumerate(shape["kinds"])
    )
    if shape["compose"] >= 3:
        return (
            "The hidden rule chains THREE steps in order (each step reads the "
            "result of the step before it); each step is one of: " + listing + "."
        )
    if shape["compose"] == 2:
        return (
            "The hidden rule chains TWO steps in order (the second step reads "
            "the result of the first); each step is one of: " + listing + "."
        )
    return "The hidden rule is ONE of: " + listing + "."


# ---------------------------------------------------------------------------
# Atoms
# ---------------------------------------------------------------------------


def gen_atoms(seed: int, level: int, n: int) -> list[dict]:
    items = []
    for index in range(n):
        item = None
        for attempt in range(40):
            item = _gen_one(seed, level, index, attempt, strict=attempt < 30)
            if item is not None and len(item["prompt"]) <= base.atom_prompt_limit(level):
                break
        if item is None:  # pragma: no cover - relaxed attempts cannot fail
            raise RuntimeError(f"{FAMILY}: generation failed for L{level} index {index}")
        items.append(item)
    return items


def _gen_one(seed: int, level: int, index: int, attempt: int, strict: bool) -> dict | None:
    rng = base.rng_for(FAMILY, seed, level, index, attempt)
    shape = _LEVEL_SHAPE[level]
    length = rng.choice(shape["lengths"])
    rule = _gen_rule(rng, length, shape)

    probe_ins: list = []
    seen: set = set()
    guard = 0
    while len(probe_ins) < shape["n_probes"] and guard < 100:
        guard += 1
        cand = _rand_string(rng, length)
        if cand not in seen:
            seen.add(cand)
            probe_ins.append(cand)
    probes = [(p, _apply(rule, p)) for p in probe_ins]
    if strict and all(i == o for i, o in probes):
        return None  # a probe log that shows no change is uninformative

    kinds, compose = shape["kinds"], shape["compose"]
    # Mode by index parity: exact 50/50 mix that survives strict-mode
    # rejection-resampling (invert redraws whenever the target would be a
    # fixed point, e.g. under overwrite rules).
    if index % 2 == 0:
        query = None
        for _ in range(50):
            cand = _rand_string(rng, length)
            if cand not in seen:
                query = cand
                break
        if query is None:
            return None
        gold_out = _apply(rule, query)
        if strict and gold_out == query:
            return None  # echoing the shown input must never win
        if strict and not _predict_wellposed(length, kinds, compose, probes, query, gold_out):
            return None
        question = f"The machine now reads: {_fmt(query)}\nWhat does it print?"
        gold = {"mode": "predict", "rule": _spec(rule), "output": _fmt(gold_out), "length": length}
    else:
        witness = _rand_string(rng, length)
        target = _apply(rule, witness)
        if strict and _apply(rule, target) == target:
            return None  # echoing the target must never win
        if strict and any(o == target for _, o in probes):
            return None  # the probe log must not hand over a preimage
        hidden_pre = _preimages(rule, target)
        if strict and not _invert_wellposed(length, kinds, compose, probes, target, hidden_pre):
            return None
        question = (
            f"Give one input that the machine prints as {_fmt(target)}.\n"
            f"Any input of exactly {length} glyphs that works is accepted."
        )
        gold = {
            "mode": "invert",
            "rule": _spec(rule),
            "target": _fmt(target),
            "witness": _fmt(witness),
            "length": length,
        }

    lines = [
        f"A sealed glyph machine transforms any string of exactly {length} glyphs",
        "by one fixed hidden rule.",
        _ALPHABET_LINE,
        _menu(shape),
        "",
        "Probe log (input -> output):",
    ]
    lines += [f"  {_fmt(i)} -> {_fmt(o)}" for i, o in probes]
    lines += [
        "",
        question,
        "Write glyph strings dash-joined, e.g. za-ke-ro.",
        "",
        base.ATOM_ANSWER_INSTRUCTION,
    ]
    return {
        "id": f"{FAMILY}-L{level}-s{seed}-{index:04d}",
        "family": FAMILY,
        "level": level,
        "prompt": "\n".join(lines),
        "gold": gold,
    }


def score_atom(item: dict, reply_text: str) -> float:
    gold = item["gold"]
    answer = base.extract_answer(reply_text)
    if answer is None:
        return 0.0
    s = _parse_glyphs(answer, gold["length"])
    if s is None:
        return 0.0
    if gold["mode"] == "predict":
        return 1.0 if s == _parse_glyphs(gold["output"]) else 0.0
    return 1.0 if _apply(gold["rule"], s) == _parse_glyphs(gold["target"]) else 0.0


def oracle_atom(item: dict) -> str:
    gold = item["gold"]
    value = gold["output"] if gold["mode"] == "predict" else gold["witness"]
    return f"ANSWER: {value}"


# ---------------------------------------------------------------------------
# Oracle traces: narrate the hand-coded hypothesize-and-verify procedure as
# first-person think-channel text (truth-blind: every stated value is derived
# by actually running the procedure, never by citing the stored answer).
# ---------------------------------------------------------------------------

_CYCLE_LINE = "Cycle order: za, ke, ro, mu, ti, fen, gor, lus, then back around to za."


def _flat_steps(rule) -> list:
    if rule[0] == "comp":
        return _flat_steps(rule[1]) + _flat_steps(rule[2])
    return [tuple(rule)]


def _step_desc(step) -> str:
    op = step[0]
    if op == "rev":
        return "reverse the whole string"
    if op == "shift":
        return f"step every glyph {step[1]} forward on the cycle"
    if op == "swap":
        return f"swap slots {step[1] + 1} and {step[2] + 1}"
    if op == "setg":
        return f"overwrite slot {step[1] + 1} with {GLYPHS[step[2]]}"
    return (
        f"rotate one step (left when the first glyph is {GLYPHS[step[1]]} or "
        f"{GLYPHS[step[2]]}, else right)"
    )


def _chain_desc(steps: list) -> str:
    if len(steps) == 1:
        return _step_desc(steps[0])
    return "first " + ", then ".join(_step_desc(s) for s in steps)


def _diff_slots(a: tuple, b: tuple) -> list:
    return [p for p in range(len(a)) if a[p] != b[p]]


def _trace_probes(item: dict, length: int) -> list:
    pairs = re.findall(r"^  ([a-z-]+) -> ([a-z-]+)$", item["prompt"], re.MULTILINE)
    probes = [(_parse_glyphs(a, length), _parse_glyphs(b, length)) for a, b in pairs]
    if not probes or any(i is None or o is None for i, o in probes):  # pragma: no cover
        raise ValueError(f"{FAMILY}: could not recover the probe log for {item['id']}")
    return probes


def _trace_query(item: dict, length: int) -> tuple:
    match = re.search(r"The machine now reads: ([a-z-]+)", item["prompt"])
    query = _parse_glyphs(match.group(1), length) if match else None
    if query is None:  # pragma: no cover
        raise ValueError(f"{FAMILY}: could not recover the query for {item['id']}")
    return query


def _dead_lines(kind: str, probes: list) -> list:
    """Narrate the genuine elimination of a rule family; only called when NO
    parameterization of `kind` is consistent with the probe log, so each
    recipe below is guaranteed to surface a concrete mismatch."""
    if kind == "rev":
        for j, (i, o) in enumerate(probes, start=1):
            r = tuple(reversed(i))
            if r != o:
                p = _diff_slots(r, o)[0]
                return [
                    f"Reversal? Reversed, probe {j}'s input would read {_fmt(r)}, but the log "
                    f"shows {_fmt(o)} -- slot {p + 1} is {GLYPHS[o[p]]}, not {GLYPHS[r[p]]}. "
                    "Not reversal."
                ]
    if kind == "shift":
        i0, o0 = probes[0]
        deltas = [(o0[p] - i0[p]) % N_GLYPHS for p in range(len(i0))]
        bad = [p for p in range(1, len(deltas)) if deltas[p] != deltas[0]]
        if bad:
            p = bad[0]
            return [
                f"A cycle-shift? On probe 1, {GLYPHS[i0[0]]} to {GLYPHS[o0[0]]} is +{deltas[0]}, "
                f"but slot {p + 1} goes {GLYPHS[i0[p]]} to {GLYPHS[o0[p]]}, which is "
                f"+{deltas[p]}. A shift moves every glyph by the same count, so no."
            ]
        if deltas[0] == 0:
            return [
                "A cycle-shift? Probe 1 comes back unchanged, and a shift moves every glyph. No."
            ]
        k = deltas[0]
        for j, (i, o) in enumerate(probes[1:], start=2):
            exp = tuple((g + k) % N_GLYPHS for g in i)
            if exp != o:
                p = _diff_slots(exp, o)[0]
                return [
                    f"A cycle-shift? Probe 1 fits +{k}, but then probe {j} would print "
                    f"{_fmt(exp)}; the log shows {_fmt(o)} (slot {p + 1} is off). No shift works."
                ]
    if kind == "swap":
        first = next(
            ((j, i, o) for j, (i, o) in enumerate(probes, start=1) if i != o), None
        )
        if first is None:  # pragma: no cover - strict generation rejects no-change logs
            return ["A swap? No pair of slots holds matching glyphs in every probe. No."]
        j, i, o = first
        d = _diff_slots(i, o)
        if len(d) != 2:
            n_slots = "exactly one slot" if len(d) == 1 else f"{len(d)} slots"
            return [
                f"A swap? Probe {j} changes {n_slots}, and a single swap changes exactly "
                "two slots or none. No."
            ]
        p, q = d
        if not (i[p] == o[q] and i[q] == o[p]):
            return [
                f"A swap? Probe {j} changes slots {p + 1} and {q + 1}, but {GLYPHS[i[p]]} and "
                f"{GLYPHS[i[q]]} do not simply trade places there. No."
            ]
        for m, (i2, o2) in enumerate(probes, start=1):
            out = list(i2)
            out[p], out[q] = out[q], out[p]
            if tuple(out) != o2:
                return [
                    f"A swap? Probe {j} pins it to slots {p + 1} and {q + 1}, but that swap "
                    f"turns probe {m}'s input into {_fmt(tuple(out))}, not the printed "
                    f"{_fmt(o2)}. No."
                ]
    if kind == "setg":
        for j, (i, o) in enumerate(probes, start=1):
            d = _diff_slots(i, o)
            if len(d) >= 2:
                return [
                    f"An overwrite? Probe {j} changes slots {d[0] + 1} and {d[1] + 1} at once, "
                    "and an overwrite touches exactly one slot. No."
                ]
        pins = [
            (j, _diff_slots(i, o)[0], o[_diff_slots(i, o)[0]])
            for j, (i, o) in enumerate(probes, start=1)
            if i != o
        ]
        if not pins:  # pragma: no cover - strict generation rejects no-change logs
            return ["An overwrite? No slot prints one fixed glyph across the whole log. No."]
        j0, k, g = pins[0]
        for j2, k2, g2 in pins[1:]:
            if k2 != k:
                return [
                    f"An overwrite? Probe {j0} changes only slot {k + 1} while probe {j2} "
                    f"changes only slot {k2 + 1} -- one fixed overwrite cannot do both. No."
                ]
            if g2 != g:
                return [
                    f"An overwrite? Probe {j0} forces slot {k + 1} to {GLYPHS[g]}, but probe "
                    f"{j2} forces it to {GLYPHS[g2]}. No."
                ]
        for m, (i2, o2) in enumerate(probes, start=1):
            exp = i2[:k] + (g,) + i2[k + 1 :]
            if exp != o2:
                return [
                    f"An overwrite? The changes point at slot {k + 1} always becoming "
                    f"{GLYPHS[g]}, but probe {m} prints {GLYPHS[o2[k]]} there. No."
                ]
    if kind == "crot":
        need_in: dict = {}
        need_out: dict = {}
        for j, (i, o) in enumerate(probes, start=1):
            rot_l = i[1:] + i[:1]
            rot_r = i[-1:] + i[:-1]
            if o != rot_l and o != rot_r:
                return [
                    f"The conditional rotation? Probe {j}'s output is not its input rotated one "
                    f"step either way (left gives {_fmt(rot_l)}, right gives {_fmt(rot_r)}). No."
                ]
            if o == rot_l and o != rot_r:
                need_in.setdefault(i[0], j)
            if o == rot_r and o != rot_l:
                need_out.setdefault(i[0], j)
        for g, j in need_in.items():
            if g in need_out:
                return [
                    f"The conditional rotation? Probe {j} rotates left with first glyph "
                    f"{GLYPHS[g]}, yet probe {need_out[g]} rotates right with the same first "
                    "glyph. Impossible. No."
                ]
        if len(need_in) > 2:
            names = ", ".join(GLYPHS[g] for g in sorted(need_in))
            return [
                f"The conditional rotation? First glyphs {names} all rotate left, and the "
                "secret pair holds only two glyphs. No."
            ]
    raise AssertionError(f"{FAMILY}: no elimination narration for {kind}")  # pragma: no cover


def _gold_fit_lines(step: tuple, probes: list) -> list:
    """Narrate reading the surviving rule's parameters off the probe log and
    verifying it against every probe pair (all values computed)."""
    op = step[0]
    lines: list = []
    if op == "rev":
        i0, _ = probes[0]
        lines.append(
            f"Reversal: probe 1 reversed reads {_fmt(tuple(reversed(i0)))} -- exactly what "
            "the machine printed."
        )
        for j, (i, _) in enumerate(probes[1:], start=2):
            lines.append(f"Probe {j} reversed gives {_fmt(tuple(reversed(i)))} -- matches.")
    elif op == "shift":
        k = step[1]
        i0, o0 = probes[0]
        lines.append(
            f"A cycle-shift: on probe 1, {GLYPHS[i0[0]]} lands on {GLYPHS[o0[0]]}, which is "
            f"{k} steps forward, and every other slot steps +{k} as well."
        )
        for j, (i, _) in enumerate(probes[1:], start=2):
            exp = tuple((g + k) % N_GLYPHS for g in i)
            lines.append(f"Probe {j}: +{k} gives {_fmt(exp)} -- matches.")
    elif op == "swap":
        p, q = step[1], step[2]
        vis = next((j for j, (i, _) in enumerate(probes, start=1) if i[p] != i[q]), None)
        if vis is None:  # pragma: no cover - strict generation rejects no-change logs
            lines.append(
                f"A swap of slots {p + 1} and {q + 1}: both slots match in every probe, so "
                "nothing visibly changes -- consistent with the whole log."
            )
            return lines
        iv, _ = probes[vis - 1]
        lines.append(
            f"A swap: probe {vis} changes exactly slots {p + 1} and {q + 1}, trading "
            f"{GLYPHS[iv[p]]} and {GLYPHS[iv[q]]}."
        )
        for j, (i, _) in enumerate(probes, start=1):
            if j == vis:
                continue
            if i[p] == i[q]:
                lines.append(
                    f"Probe {j} holds {GLYPHS[i[p]]} in both slots, so it rightly comes back "
                    "unchanged."
                )
            else:
                out = list(i)
                out[p], out[q] = out[q], out[p]
                lines.append(
                    f"Probe {j}: swapping those slots gives {_fmt(tuple(out))} -- matches."
                )
    elif op == "setg":
        k, g = step[1], step[2]
        vis = next((j for j, (i, _) in enumerate(probes, start=1) if i[k] != g), None)
        if vis is None:  # pragma: no cover - strict generation rejects no-change logs
            lines.append(
                f"An overwrite of slot {k + 1} with {GLYPHS[g]}: every probe already holds "
                f"{GLYPHS[g]} there, so nothing visibly changes -- consistent."
            )
            return lines
        lines.append(
            f"An overwrite: probe {vis} changes only slot {k + 1}, which ends as {GLYPHS[g]}, "
            "with everything else untouched."
        )
        for j, (i, _) in enumerate(probes, start=1):
            if j == vis:
                continue
            if i[k] == g:
                lines.append(
                    f"Probe {j} already holds {GLYPHS[g]} at slot {k + 1}, so no visible "
                    "change -- consistent."
                )
            else:
                exp = i[:k] + (g,) + i[k + 1 :]
                lines.append(
                    f"Probe {j}: slot {k + 1} becomes {GLYPHS[g]}, rest kept: {_fmt(exp)} "
                    "-- matches."
                )
    else:  # crot
        a, b = step[1], step[2]
        for j, (i, _) in enumerate(probes, start=1):
            direction = "left" if i[0] in (a, b) else "right"
            lines.append(
                f"Probe {j} is its input rotated one step {direction} (first glyph "
                f"{GLYPHS[i[0]]})."
            )
        lefts = sorted({i[0] for i, _ in probes if i[0] in (a, b)})
        rights = sorted({i[0] for i, _ in probes if i[0] not in (a, b)})
        left_names = " and ".join(GLYPHS[g] for g in lefts) if lefts else "no observed glyph"
        right_names = " and ".join(GLYPHS[g] for g in rights) if rights else "none"
        lines.append(
            f"A secret pair of {GLYPHS[a]} and {GLYPHS[b]} makes every probe come out right: "
            f"it holds {left_names} and leaves out {right_names}."
        )
    return lines


def _sweep_lines(
    kinds: tuple,
    gold_step: tuple,
    probes: list,
    length: int,
    query: tuple | None,
    final: tuple | None,
    witness: tuple | None,
    target: tuple | None,
) -> tuple:
    """Walk the level's rule menu in order: full fit narration for the kept
    rule, a concrete elimination for each dead family, and an honest note for
    any coincident family that also survives the log."""
    lines: list = []
    extras: list = []
    for kind in kinds:
        if kind == gold_step[0]:
            lines += _gold_fit_lines(gold_step, probes)
            continue
        cands = [
            r
            for r in _singles_pool(length, (kind,))
            if all(_apply(r, i) == o for i, o in probes)
        ]
        if not cands:
            lines += _dead_lines(kind, probes)
            continue
        alt = cands[0]
        if query is not None and _apply(alt, query) == final:
            lines.append(
                f"Curious: '{_step_desc(alt)}' also reproduces every probe pair. Noted -- "
                "I'll check whether it changes the final print."
            )
            extras.append(alt)
        elif witness is not None and _apply(alt, witness) == target:
            lines.append(
                f"Curious: '{_step_desc(alt)}' also reproduces every probe pair; whatever "
                "input I build should satisfy it too."
            )
            extras.append(alt)
        # A coincident rule the derivation cannot reconcile is simply not
        # narrated (well-posed generation makes this branch near-unreachable).
    return lines, extras


def _mutate_step(step: tuple, length: int):
    """A deterministic near-miss variant of a step, for a genuine dead end."""
    op = step[0]
    if op == "shift":
        return ("shift", (step[1] % (N_GLYPHS - 1)) + 1)
    if op == "swap":
        i, j = step[1], step[2]
        j2 = (j + 1) % length
        if j2 == i:
            j2 = (j2 + 1) % length
        a, b = sorted((i, j2))
        return None if (a, b) == (i, j) else ("swap", a, b)
    if op == "setg":
        return ("setg", step[1], (step[2] + 1) % N_GLYPHS)
    if op == "crot":
        a, b = step[1], step[2]
        b2 = (b + 1) % N_GLYPHS
        if b2 == a:
            b2 = (b2 + 1) % N_GLYPHS
        x, y = sorted((a, b2))
        return None if (x, y) == (a, b) else ("crot", x, y)
    return ("shift", 1)  # rev has no parameter to bend; guess a small shift


def _composed_dead_end(steps: list, probes: list, length: int) -> list:
    """Show one chain hypothesis that the probe log genuinely eliminates."""
    candidates = []
    mutated = _mutate_step(steps[-1], length)
    if mutated is not None:
        candidates.append(steps[:-1] + [mutated])
    candidates.append(list(reversed(steps)))
    for cand in candidates:
        if [tuple(s) for s in cand] == [tuple(s) for s in steps]:
            continue  # pragma: no cover - mutations always differ from the chain
        for j, (i, o) in enumerate(probes, start=1):
            mids = [i]
            for s in cand:
                mids.append(_apply(s, mids[-1]))
            if mids[-1] != o:
                seg = " -> ".join(_fmt(m) for m in mids)
                return [
                    f"First guess: {_chain_desc(cand)}. Probe {j}: {seg}, but the log prints "
                    f"{_fmt(o)}. Dead end -- adjust the chain."
                ]
    return []  # pragma: no cover - some candidate always misses a probe


def _composed_obs_lines(probes: list, length: int) -> list:
    lines = [
        f"Probe 1 changes {len(_diff_slots(*probes[0]))} of its {length} slots."
    ]
    if all(sorted(i) == sorted(o) for i, o in probes):
        lines.append(
            "No probe changes which glyphs it holds, only where they sit, so I lean toward "
            "order-moving steps."
        )
    else:
        j = next(j for j, (i, o) in enumerate(probes, start=1) if sorted(i) != sorted(o))
        lines.append(
            f"Probe {j} ends with glyphs it did not start with, so at least one step changes "
            "glyph identity -- a cycle step or an overwrite."
        )
    return lines


def _verify_chain_lines(steps: list, probes: list) -> list:
    lines = [f"Try: {_chain_desc(steps)}."]
    for j, (i, o) in enumerate(probes, start=1):
        mids = [i]
        for s in steps:
            mids.append(_apply(s, mids[-1]))
        seg = " -> ".join(_fmt(m) for m in mids)
        lines.append(f"Probe {j}: {seg} -- matches the log.")
    return lines


def _apply_lines(steps: list, query: tuple) -> list:
    if len(steps) > 1:
        mids = [query]
        parts = []
        for idx, s in enumerate(steps, start=1):
            mids.append(_apply(s, mids[-1]))
            parts.append(f"step {idx} gives {_fmt(mids[-1])}")
        return [f"Now the query {_fmt(query)}: " + "; ".join(parts) + "."]
    step = steps[0]
    out = _apply(step, query)
    op = step[0]
    if op == "rev":
        return [f"Now the query: reversing {_fmt(query)} gives {_fmt(out)}."]
    if op == "shift":
        return [
            f"Now the query: stepping each glyph of {_fmt(query)} forward {step[1]} gives "
            f"{_fmt(out)}."
        ]
    if op == "swap":
        return [
            f"Now the query: swapping slots {step[1] + 1} and {step[2] + 1} of {_fmt(query)} "
            f"gives {_fmt(out)}."
        ]
    if op == "setg":
        return [
            f"Now the query: {_fmt(query)} with slot {step[1] + 1} set to "
            f"{GLYPHS[step[2]]} is {_fmt(out)}."
        ]
    membership = "is" if query[0] in (step[1], step[2]) else "is not"
    direction = "left" if query[0] in (step[1], step[2]) else "right"
    return [
        f"Now the query {_fmt(query)}: its first glyph {GLYPHS[query[0]]} {membership} in "
        f"the pair, so it rotates {direction}, giving {_fmt(out)}."
    ]


def _undo_lines(step: tuple, before: tuple, after: tuple) -> list:
    op = step[0]
    if op == "rev":
        return [f"Undo the reversal: {_fmt(after)} un-reversed is {_fmt(before)}."]
    if op == "shift":
        return [
            f"Undo the +{step[1]} shift: stepping every glyph of {_fmt(after)} back "
            f"{step[1]} gives {_fmt(before)}."
        ]
    if op == "swap":
        return [
            f"Undo the swap: trading slots {step[1] + 1} and {step[2] + 1} of {_fmt(after)} "
            f"back gives {_fmt(before)}."
        ]
    if op == "setg":
        k, g = step[1], step[2]
        return [
            f"The overwrite forces slot {k + 1} to {GLYPHS[g]}, and {_fmt(after)} indeed "
            f"shows {GLYPHS[g]} there. Before the overwrite that slot could have held "
            f"anything; I take {GLYPHS[before[k]]} and keep the rest: {_fmt(before)}."
        ]
    if before[0] in (step[1], step[2]):
        return [
            f"Undo the rotation: {_fmt(after)} turned back one step right is {_fmt(before)}, "
            f"whose first glyph {GLYPHS[before[0]]} is in the pair -- so it rotated left onto "
            f"{_fmt(after)}, consistent."
        ]
    return [
        f"Undo the rotation: {_fmt(after)} turned back one step left is {_fmt(before)}, "
        f"whose first glyph {GLYPHS[before[0]]} is not in the pair -- so it rotated right "
        f"onto {_fmt(after)}, consistent."
    ]


def oracle_trace(item: dict) -> str:
    """A first-person reasoning trace that solves the atom the way the
    generator's hypothesize-and-verify procedure does: test rule families
    against every probe pair, eliminate on concrete mismatches, keep the
    consistent rule, then apply (predict) or invert (construct) it."""
    gold = item["gold"]
    shape = _LEVEL_SHAPE[item["level"]]
    length = gold["length"]
    rng = base.rng_for(FAMILY, "trace", item["id"])
    probes = _trace_probes(item, length)
    steps = _flat_steps(gold["rule"])
    depth = len(steps)
    kinds = shape["kinds"]
    predict = gold["mode"] == "predict"

    query = final = witness = target = None
    if predict:
        query = _trace_query(item, length)
        final = _parse_glyphs(gold["output"], length)
    else:
        target = _parse_glyphs(gold["target"], length)
        witness = _parse_glyphs(gold["witness"], length)

    lines: list = []
    if predict:
        lines.append(rng.choice([
            f"I need to predict what the machine prints for {_fmt(query)}.",
            f"Goal: work out what the machine prints when it reads {_fmt(query)}.",
            f"Let me pin down the hidden rule from the probe log, then run {_fmt(query)} "
            "through it.",
        ]))
    else:
        article = "an" if length == 8 else "a"
        lines.append(rng.choice([
            f"I need an input that the machine prints as {_fmt(target)}.",
            f"Goal: build {article} {length}-glyph input that the machine turns into "
            f"{_fmt(target)}.",
            f"Let me find the hidden rule from the probe log, then run the target "
            f"{_fmt(target)} backward through it.",
        ]))
    if depth == 1:
        lines.append(rng.choice([
            f"The log gives {len(probes)} input/output pairs over {length}-glyph strings, "
            f"and the rule is one of {len(kinds)} families -- small enough to test each "
            "family against every probe pair and drop whatever misses.",
            "I'll walk the rule menu in order: hypothesize a family, verify it on every "
            "probe pair, and eliminate on the first mismatch.",
            "Plan: try each family on the menu against the whole probe log and keep the "
            "one that survives.",
        ]))
    else:
        word = "two" if depth == 2 else "three"
        lines.append(rng.choice([
            f"The rule chains {word} steps from the menu, so I'll read what the log forces, "
            "guess a chain, and verify it on every probe pair.",
            f"Here the machine applies {word} menu steps in order; I'll hypothesize a chain "
            "and check it against each probe, adjusting on any mismatch.",
            f"With {word} chained steps the space is bigger, so: look for structural hints, "
            "propose a chain, verify probe by probe.",
        ]))
    if "shift" in kinds or "crot" in kinds:
        lines.append(_CYCLE_LINE)

    extras: list = []
    if depth == 1:
        swept, extras = _sweep_lines(
            kinds, steps[0], probes, length, query, final, witness, target
        )
        lines += swept
    else:
        lines += _composed_obs_lines(probes, length)
        lines += _composed_dead_end(steps, probes, length)
        lines += _verify_chain_lines(steps, probes)
    lines.append(rng.choice([
        f"So the rule: {_chain_desc(steps)}.",
        f"That settles the rule: {_chain_desc(steps)}.",
        f"Everything in the log checks out, so I'll commit: {_chain_desc(steps)}.",
    ]))

    if predict:
        lines += _apply_lines(steps, query)
        if extras:
            lines.append(
                "The other surviving reading sends the query to the same string, so the "
                "print is forced either way."
            )
        lines.append(rng.choice([
            f"So the machine prints {_fmt(final)}.",
            f"So the output is {_fmt(final)}.",
            f"The machine will print {_fmt(final)}.",
        ]))
    else:
        mids = [witness]
        for s in steps:
            mids.append(_apply(s, mids[-1]))
        lines.append(f"Now work backward from the target {_fmt(target)}.")
        for idx in range(depth - 1, -1, -1):
            lines += _undo_lines(steps[idx], mids[idx], mids[idx + 1])
        seg = " -> ".join(_fmt(m) for m in mids)
        lines.append(f"Check forward: {seg} -- that lands exactly on the target. Good.")
        if extras:
            lines.append("And it satisfies the other surviving reading as well.")
        lines.append(rng.choice([
            f"So an input that the machine prints as {_fmt(target)} is {_fmt(witness)}.",
            f"So the input I give is {_fmt(witness)}.",
            f"So the answer is {_fmt(witness)}.",
        ]))
    return "\n".join(lines)


def _selftest_traces(n_per_level: int = 10) -> dict:
    stats: dict = {"levels": {}}
    for level in LEVELS:
        items = gen_atoms(7, level, n_per_level)
        scores: list = []
        max_words = 0
        for item in items:
            trace = oracle_trace(item)
            base._check(
                trace == oracle_trace(item),
                f"{FAMILY} L{level} {item['id']}: trace is not deterministic",
            )
            words = len(trace.split())
            max_words = max(max_words, words)
            base._check(
                words <= 800,
                f"{FAMILY} L{level} {item['id']}: trace has {words} words (> 800)",
            )
            lowered = trace.lower()
            for word in base.FORBIDDEN_WORDS:
                base._check(
                    word not in lowered,
                    f"{FAMILY} L{level} {item['id']}: forbidden word {word!r} in trace",
                )
            gold = item["gold"]
            value = gold["output"] if gold["mode"] == "predict" else gold["witness"]
            base._check(
                value in trace.rstrip().splitlines()[-1],
                f"{FAMILY} L{level} {item['id']}: trace does not end by stating the answer",
            )
            reply = trace + "\n\n" + oracle_atom(item)
            scores.append(score_atom(item, reply))
        mean = sum(scores) / len(scores)
        base._check(
            mean >= 0.95,
            f"{FAMILY} L{level}: trace+answer replies score {mean:.3f} < 0.95",
        )
        stats["levels"][level] = {"trace_score": round(mean, 4), "max_words": max_words}
    return stats


# ---------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------


class Episode:
    def __init__(self, seed: int, level: int):
        rng = base.rng_for(FAMILY, "episode", seed, level)
        shape = _LEVEL_SHAPE[level]
        self.level = level
        self._length = rng.choice(shape["lengths"])
        rule = witness = target = None
        for _ in range(80):
            rule = _gen_rule(rng, self._length, shape)
            witness = _rand_string(rng, self._length)
            target = _apply(rule, witness)
            if _apply(rule, target) != target:
                break  # OPENing the target itself must never win
        self._rule = rule
        self._target = target
        self._budget = _EP_BUDGET[level]
        self.max_turns = _EP_MAX_TURNS[level]
        self._probes_left = self._budget
        self._turns_left = self.max_turns
        self._done = False
        self._won = False
        self.last_action_ok = True
        self.spec = {
            "family": FAMILY,
            "level": level,
            "seed": seed,
            "length": self._length,
            "rule": _spec(rule),
            "target": _fmt(target),
            "witness": _fmt(witness),
            "budget": self._budget,
        }

    def system_prompt(self) -> str:
        shape = _LEVEL_SHAPE[self.level]
        return (
            "You stand before a sealed glyph gate. It transforms any string of "
            f"exactly {self._length} glyphs by one fixed hidden rule. "
            + _ALPHABET_LINE + " " + _menu(shape) + "\n"
            "Goal: feed the gate an input that it transforms into the TARGET string.\n"
            "Actions (one per turn):\n"
            "  PROBE <glyph-string>  - spend one probe; the gate prints the transformed string\n"
            "  OPEN <glyph-string>   - final commit; you succeed only if the gate's output equals the TARGET\n"
            "Glyph strings are dash-joined, e.g. za-ke-ro. Probes are limited; OPEN ends the episode.\n"
            + base.EPISODE_ACTION_INSTRUCTION
        )

    def initial_observation(self) -> str:
        return (
            f"TARGET: {_fmt(self._target)}. The gate accepts strings of exactly "
            f"{self._length} glyphs. Probes left: {self._probes_left}. "
            f"Turns left: {self._turns_left}."
        )

    def step(self, action_line: str) -> tuple[str, bool]:
        if self._done:
            self.last_action_ok = False
            return ("The gate has already resolved.", True)
        self._turns_left -= 1
        self.last_action_ok = False
        line = base.extract_action(action_line or "")
        match = re.match(r"^(probe|open)\b[:\s]*(.*)$", line, re.IGNORECASE)
        if not match:
            obs = "Bad action. Use: PROBE <glyph-string> or OPEN <glyph-string>."
        else:
            verb = match.group(1).lower()
            s = _parse_glyphs(match.group(2), self._length)
            if s is None:
                obs = (
                    f"Unreadable glyph string. Give exactly {self._length} glyphs from "
                    "za, ke, ro, mu, ti, fen, gor, lus, dash-joined (e.g. za-ke-ro)."
                )
            elif verb == "probe":
                if self._probes_left <= 0:
                    obs = "No probes remain. Commit with: OPEN <glyph-string>."
                else:
                    self.last_action_ok = True
                    self._probes_left -= 1
                    out = _apply(self._rule, s)
                    obs = f"The gate prints: {_fmt(out)}. Probes left: {self._probes_left}."
            else:  # OPEN
                self.last_action_ok = True
                out = _apply(self._rule, s)
                self._done = True
                self._won = out == self._target
                if self._won:
                    return (
                        f"The gate transforms your string into {_fmt(out)} — it matches "
                        "the TARGET. The gate swings open.",
                        True,
                    )
                return (
                    f"The gate transforms your string into {_fmt(out)} — not the TARGET. "
                    "The gate stays sealed.",
                    True,
                )
        if self._turns_left <= 0:
            self._done = True
            return (obs + " Turn limit reached; the gate seals shut.", True)
        return (obs + f" Turns left: {self._turns_left}.", False)

    def score(self) -> float:
        return 1.0 if self._won else 0.0


class OraclePolicy:
    """Probes distinctively (two all-distinct-glyph strings), then OPENs the
    generator's witness — a guaranteed preimage of the TARGET."""

    def __init__(self, episode: Episode):
        length = episode.spec["length"]
        self._open_action = f"OPEN {episode.spec['witness']}"
        self._plan = []
        for k in range(min(2, episode.spec["budget"], episode.max_turns - 1)):
            probe = tuple((k + i) % N_GLYPHS for i in range(length))
            self._plan.append(f"PROBE {_fmt(probe)}")
        self._plan.append(self._open_action)
        self._i = 0

    def act(self, observation_history: list[str]) -> str:
        if self._i < len(self._plan):
            action = self._plan[self._i]
            self._i += 1
            return action
        return self._open_action  # pragma: no cover - plan always fits budget


def selftest() -> dict:
    module = __import__(__name__, fromlist=["x"])
    return {
        "family": FAMILY,
        "atoms": base.selftest_atoms(module),
        "episodes": base.selftest_episodes(module),
        "traces": _selftest_traces(),
    }
