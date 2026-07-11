"""spindle — execute a freshly stated symbolic procedure on a token tape.

HELD-OUT family (never trained on): probes transfer of formal execution and
protocol compliance. A spindle-loom reworks a tape of 4-7 invented tokens
through a numbered procedure of 3-6 passes (swap / rotate / replace /
conditional drop / duplicate). Atoms ask for the final tape (L1: the tape
after just pass 1-2); episodes walk the passes one at a time and demand the
intermediate tape after each pass via ``TAPE <dash-joined tape>`` actions.

Frontier levels escalate execution depth on an extended token pool (the new
tokens are appended so L1-L4 sampling is untouched): L5 runs 7 passes on
8-token tapes (episode horizon 18); L6 runs 8 passes on 9-token tapes
(episode horizon 22).

Every generated pass strictly changes the tape (no no-op passes), so each
intermediate state differs from its predecessor and echo strategies score 0.
"""

from __future__ import annotations

import re
import sys

from .. import base

FAMILY = "spindle"
LEVELS = (1, 2, 3, 4, 5, 6)
HAS_EPISODES = True

TOKENS = ("bram", "cusk", "dell", "fith", "gorse", "hew", "ilk")
# Frontier pool: new invented tokens are APPENDED so the L1-L4 pool (and every
# RNG draw on those paths) is byte-identical to the pre-frontier gym.
TOKENS_FRONTIER = TOKENS + ("jarn", "kest", "lorm")

_LEVEL_SHAPE = {
    # level: (tape_len, n_passes)
    1: (4, 3),
    2: (5, 4),
    3: (6, 5),
    4: (7, 6),
    5: (8, 7),
    6: (9, 8),
}

_TURN_CAPS = {1: 4, 2: 6, 3: 10, 4: 14, 5: 18, 6: 22}


def _pool_for(level: int) -> tuple[str, ...]:
    return TOKENS if level <= 4 else TOKENS_FRONTIER

_MAX_TAPE_LEN = 9  # duplication cap so tapes stay short

_RULES = (
    "A spindle-loom reworks a tape of tokens through numbered passes. A tape "
    "is written dash-joined, like bram-cusk-dell. Positions count from 1, "
    "starting at the left. Rotating left moves the first token to the end; "
    "rotating right moves the last token to the front. Duplicating the token "
    "at position k inserts a copy of it immediately after position k. Each "
    "pass acts on the tape left by the pass before it."
)


def _dash(tape: list[str]) -> str:
    return "-".join(tape)


def _canon_tape(value: str) -> str:
    """Canonicalize a tape answer: lowercase, normalize dashes/spaces/commas."""
    return "-".join(re.findall(r"[a-z]+", value.lower()))


# ---------------------------------------------------------------------------
# Pass mechanics
# ---------------------------------------------------------------------------


def _apply_pass(tape: list[str], p: dict) -> list[str]:
    kind = p["kind"]
    tape = list(tape)
    if kind == "swap":
        i, j = p["i"] - 1, p["j"] - 1
        tape[i], tape[j] = tape[j], tape[i]
        return tape
    if kind == "rotate":
        if p["dir"] == "left":
            return tape[1:] + tape[:1]
        return tape[-1:] + tape[:-1]
    if kind == "replace":
        return [p["dst"] if tok == p["src"] else tok for tok in tape]
    if kind == "drop":
        if tape[0] == p["token"]:
            return tape[:-1]
        return tape[1:]
    if kind == "dup":
        k = p["k"] - 1
        return tape[: k + 1] + [tape[k]] + tape[k + 1 :]
    raise ValueError(f"unknown pass kind {kind!r}")  # pragma: no cover


def _render_pass(p: dict) -> str:
    kind = p["kind"]
    if kind == "swap":
        return f"swap positions {p['i']} and {p['j']}"
    if kind == "rotate":
        return f"rotate the tape {p['dir']} by one"
    if kind == "replace":
        return f"replace every {p['src']} with {p['dst']}"
    if kind == "drop":
        return (
            f"if the tape begins with {p['token']}, drop the last token, "
            "otherwise drop the first"
        )
    if kind == "dup":
        return f"duplicate the token at position {p['k']}"
    raise ValueError(f"unknown pass kind {kind!r}")  # pragma: no cover


def _gen_pass(rng, tape: list[str], pool: tuple[str, ...]) -> dict:
    """One random pass that is guaranteed to change the given tape."""
    n = len(tape)
    kinds = ["replace"]
    if n >= 2 and len(set(tape)) >= 2:
        kinds += ["swap", "rotate"]
    if n >= 3:
        kinds.append("drop")
    if n < _MAX_TAPE_LEN:
        kinds.append("dup")
    kind = rng.choice(kinds)
    if kind == "swap":
        pairs = [
            (i, j)
            for i in range(n)
            for j in range(i + 1, n)
            if tape[i] != tape[j]
        ]
        i, j = rng.choice(pairs)
        return {"kind": "swap", "i": i + 1, "j": j + 1}
    if kind == "rotate":
        return {"kind": "rotate", "dir": rng.choice(["left", "right"])}
    if kind == "replace":
        src = rng.choice(sorted(set(tape)))
        dst = rng.choice([tok for tok in pool if tok != src])
        return {"kind": "replace", "src": src, "dst": dst}
    if kind == "drop":
        if rng.random() < 0.5:
            token = tape[0]  # condition holds -> drop the last
        else:
            token = rng.choice([tok for tok in pool if tok != tape[0]])
        return {"kind": "drop", "token": token}
    return {"kind": "dup", "k": rng.randint(1, n)}


def _gen_procedure(rng, tape_len: int, n_passes: int, pool: tuple[str, ...]):
    """Returns (initial_tape, passes, states) with states[k] = tape after pass k+1."""
    initial = list(rng.sample(pool, tape_len))
    passes: list[dict] = []
    states: list[list[str]] = []
    tape = initial
    for _ in range(n_passes):
        p = _gen_pass(rng, tape, pool)
        tape = _apply_pass(tape, p)
        passes.append(p)
        states.append(tape)
    return initial, passes, states


# ---------------------------------------------------------------------------
# Atoms
# ---------------------------------------------------------------------------


def gen_atoms(seed: int, level: int, n: int) -> list[dict]:
    items = []
    for index in range(n):
        for attempt in range(20):
            item, distinct = _gen_one(seed, level, index, attempt)
            if distinct and len(item["prompt"]) <= base.atom_prompt_limit(level):
                break
        items.append(item)
    return items


def _gen_one(seed: int, level: int, index: int, attempt: int):
    rng = base.rng_for(FAMILY, seed, level, index, attempt)
    tape_len, n_passes = _LEVEL_SHAPE[level]
    initial, passes, states = _gen_procedure(rng, tape_len, n_passes, _pool_for(level))

    if level == 1:
        upto = rng.choice([1, 2])
        gold_tape = states[upto - 1]
        scope = "pass 1" if upto == 1 else "passes 1 and 2"
        question = (
            f"What is the tape after pass {upto}? Apply only {scope}, in "
            "order, then stop."
        )
    else:
        gold_tape = states[-1]
        question = (
            f"What is the final tape after all {n_passes} passes are applied "
            "in order?"
        )

    lines = [_RULES, "", f"Initial tape: {_dash(initial)}", "", "Procedure:"]
    lines += [f"{k + 1}. {_render_pass(p)}" for k, p in enumerate(passes)]
    lines += ["", question, "Give the tape dash-joined.", "", base.ATOM_ANSWER_INSTRUCTION]
    prompt = "\n".join(lines)

    item = {
        "id": f"{FAMILY}-L{level}-s{seed}-{index:04d}",
        "family": FAMILY,
        "level": level,
        "prompt": prompt,
        "gold": _dash(gold_tape),
    }
    return item, _dash(gold_tape) != _dash(initial)


def score_atom(item: dict, reply_text: str) -> float:
    answer = base.extract_answer(reply_text)
    if answer is None:
        return 0.0
    return 1.0 if _canon_tape(answer) == _canon_tape(item["gold"]) else 0.0


def oracle_atom(item: dict) -> str:
    return f"ANSWER: {item['gold']}"


# ---------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------

_ACTION_RE = re.compile(r"^\s*tape\b[:\s]*(.+?)\s*$", re.IGNORECASE)


def _parse_tape_action(action_line: str) -> str | None:
    match = _ACTION_RE.match(action_line)
    if not match:
        return None
    canon = _canon_tape(match.group(1))
    return canon or None


class Episode:
    def __init__(self, seed: int, level: int):
        rng = base.rng_for(FAMILY, "episode", seed, level)
        tape_len, n_passes = _LEVEL_SHAPE[level]
        initial, passes, states = _gen_procedure(rng, tape_len, n_passes, _pool_for(level))
        self.spec = {
            "family": FAMILY,
            "level": level,
            "seed": seed,
            "initial": initial,
            "passes": passes,
            "states": states,
        }
        if level <= 4:
            self.max_turns = min(n_passes + 2, _TURN_CAPS[level])
        else:
            # Frontier horizons are fixed by spec: 18 (L5) / 22 (L6).
            self.max_turns = _TURN_CAPS[level]
        self._n_passes = n_passes
        self._pass_index = 0  # passes completed so far
        self._first_try: list[bool | None] = [None] * n_passes
        self._turns = 0
        self._done = False
        self.last_action_ok = True

    def system_prompt(self) -> str:
        return (
            _RULES
            + " The loom announces each pass in turn; after each announced "
            "pass, state the tape that results from it. A wrong or malformed "
            "tape does not advance the pass; you may state it again.\n"
            "Action grammar: TAPE <dash-joined tape>   "
            "(example: TAPE bram-cusk-dell)\n"
            + base.EPISODE_ACTION_INSTRUCTION
        )

    def initial_observation(self) -> str:
        return (
            f"Loom tape: {_dash(self.spec['initial'])}\n"
            f"Pass 1 of {self._n_passes}: {_render_pass(self.spec['passes'][0])}.\n"
            "State the tape after pass 1."
        )

    def step(self, action_line: str) -> tuple[str, bool]:
        if self._done:
            self.last_action_ok = False
            return ("The loom is idle; the run is over.", True)
        self._turns += 1
        self.last_action_ok = False
        idx = self._pass_index
        reply = _parse_tape_action(
            base.extract_action(action_line if isinstance(action_line, str) else "")
        )

        if reply is None:
            if self._first_try[idx] is None:
                self._first_try[idx] = False
            obs = "Malformed action. Use exactly: TAPE <dash-joined tape>"
        elif reply == _dash(self.spec["states"][idx]):
            self.last_action_ok = True
            if self._first_try[idx] is None:
                self._first_try[idx] = True
            self._pass_index += 1
            if self._pass_index == self._n_passes:
                self._done = True
                return (f"Correct. All {self._n_passes} passes complete.", True)
            nxt = self._pass_index
            obs = (
                f"Correct. Pass {nxt + 1} of {self._n_passes}: "
                f"{_render_pass(self.spec['passes'][nxt])}.\n"
                f"State the tape after pass {nxt + 1}."
            )
        else:
            if self._first_try[idx] is None:
                self._first_try[idx] = False
            obs = f"That tape is not right. State the tape after pass {idx + 1} again."

        if self._turns >= self.max_turns:
            self._done = True
            return (obs + "\nThe loom stops: turn budget spent.", True)
        return (obs, False)

    def score(self) -> float:
        credited = sum(1 for flag in self._first_try if flag is True)
        return credited / self._n_passes


class OraclePolicy:
    """Computes every intermediate tape and reports the one currently asked."""

    def __init__(self, episode: Episode):
        self._episode = episode

    def act(self, observation_history: list[str]) -> str:
        states = self._episode.spec["states"]
        idx = min(self._episode._pass_index, len(states) - 1)
        return f"TAPE {_dash(states[idx])}"


# ---------------------------------------------------------------------------
# Selftest
# ---------------------------------------------------------------------------


def selftest() -> dict:
    module = sys.modules[__name__]
    atom_stats = base.selftest_atoms(module)
    episode_stats = base.selftest_episodes(module)
    return {
        "family": FAMILY,
        "atoms": atom_stats["levels"],
        "episodes": episode_stats["levels"],
    }
