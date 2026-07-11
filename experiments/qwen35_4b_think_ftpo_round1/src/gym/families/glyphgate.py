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
overwrite-one-position-with-a-fixed-glyph. L4 chains two rules.

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
LEVELS = (1, 2, 3, 4)
HAS_EPISODES = True

GLYPHS = ("za", "ke", "ro", "mu", "ti", "fen", "gor", "lus")
_G_INDEX = {glyph: i for i, glyph in enumerate(GLYPHS)}
N_GLYPHS = len(GLYPHS)

_LEVEL_SHAPE = {
    # level: string lengths, probe pairs shown (atoms), rule kinds, compose?
    1: {"lengths": (3,), "n_probes": 2, "kinds": ("rev", "swap", "setg"), "compose": False},
    2: {"lengths": (4,), "n_probes": 3, "kinds": ("rev", "swap", "setg", "shift"), "compose": False},
    3: {"lengths": (5,), "n_probes": 3, "kinds": ("rev", "swap", "setg", "shift", "crot"), "compose": False},
    4: {"lengths": (5, 6), "n_probes": 4, "kinds": ("rev", "swap", "setg", "shift", "crot"), "compose": True},
}

_EP_MAX_TURNS = {1: 4, 2: 6, 3: 10, 4: 14}

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
    if not shape["compose"]:
        return _gen_single(rng, length, shape["kinds"])
    for _ in range(12):
        rule = ["comp", _gen_single(rng, length, shape["kinds"]), _gen_single(rng, length, shape["kinds"])]
        tests = [_rand_string(rng, length) for _ in range(6)]
        if any(_apply(rule, t) != t for t in tests):
            return rule
    return ["comp", ["rev"], ["shift", 1]]  # unreachable in practice; never identity


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


def _iter_consistent(length: int, kinds: tuple, compose: bool, probes: list):
    """Yield every rule in the level's declared hypothesis space that maps
    each probe input to its shown output."""
    singles = _singles_pool(length, kinds)
    if not compose:
        for rule in singles:
            if all(_apply(rule, i) == o for i, o in probes):
                yield rule
        return
    ins = [i for i, _ in probes]
    outs = [o for _, o in probes]
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


def _predict_wellposed(length, kinds, compose, probes, query, gold_out) -> bool:
    for rule in _iter_consistent(length, kinds, compose, probes):
        if _apply(rule, query) != gold_out:
            return False
    return True


def _invert_wellposed(length, kinds, compose, probes, target, hidden_pre) -> bool:
    for rule in _iter_consistent(length, kinds, compose, probes):
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
    if shape["compose"]:
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
            if item is not None and len(item["prompt"]) <= base.ATOM_PROMPT_CHAR_LIMIT:
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
        self._budget = level + 2
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
    }
