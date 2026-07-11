"""kilnrite — protocol/state-machine compliance over a documented firing procedure.

Each item invents a small kiln-firing procedure: 4-8 named steps with
documented needs (steps that must already be done, flag values that must hold)
and effects (flag assignments), plus 2-4 named boolean flags with invented
value words. Flag values are shown only at the start; the solver must track
the hidden state mentally.

Atoms: given the procedure doc plus a log of steps already done (always a
legal execution prefix), either report a flag's current value word, or name
the unique step that is legal next (instances are constructed so exactly one
step is legal).

Episodes: perform the whole procedure with ``DO <StepName>``. Legal steps
apply their effects; illegal, repeated, unknown, or malformed actions change
nothing and cost the turn. Score = steps completed / total steps (1.0 when
the whole procedure resolves).

The generator walks a random step order and derives each step's needs from
the live state at that moment, so that order is always a witness that a full
legal ordering exists within the turn budget. Levels scale step count, flag
count, and the tightness of the needs.
"""

from __future__ import annotations

import re

from .. import base

FAMILY = "kilnrite"
LEVELS = (1, 2, 3, 4)
HAS_EPISODES = True

STEP_NAMES = (
    "Raking",
    "Wadding",
    "Sealing",
    "Emberfast",
    "Quenching",
    "Scrying",
    "Tamping",
    "Coiling",
    "Ashdraw",
    "Glossing",
    "Kindlecall",
    "Veiling",
    "First Draw",
    "Cold Turning",
    "Slagturn",
    "Charmark",
)

# (flag name, (word when True, word when False)); value words are unique
# across the pool so a flag question has exactly one right word.
FLAG_POOL = (
    ("damper", ("OPEN", "SHUT")),
    ("glaze-mark", ("SET", "UNSET")),
    ("ash-vent", ("FREE", "BOUND")),
    ("fire-core", ("LIT", "COLD")),
    ("soot-gate", ("RAISED", "LOWERED")),
    ("slip-ring", ("WHOLE", "CRACKED")),
)

_LEVEL_SHAPE = {
    # level: (n_steps, n_flags, max_turns, (flag_pre lo,hi), (step_pre lo,hi), eff_hi)
    1: (4, 2, 4, (0, 1), (0, 1), 1),
    2: (5, 3, 6, (0, 1), (0, 1), 2),
    3: (7, 3, 10, (1, 2), (0, 2), 2),
    4: (8, 4, 14, (1, 2), (1, 2), 2),
}

_RULES = (
    "- Steps may be done in any order, each at most once.",
    "- A step is legal only if every 'needs' item holds right then: a bare "
    "name is a step that must already be done; flag=WORD means that flag "
    "currently reads WORD.",
    "- Doing a step immediately sets each flag listed after 'then'.",
)

_ACTION_RE = re.compile(r"^do\s+(.+?)\s*$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Procedure generation + state machine
# ---------------------------------------------------------------------------


def _gen_procedure(rng, level: int) -> dict:
    """Build a procedure whose canonical `order` is a legal full execution."""
    n_steps, n_flags, _, flag_pre, step_pre, eff_hi = _LEVEL_SHAPE[level]
    flags = [
        {"name": name, "words": list(words), "init": rng.random() < 0.5}
        for name, words in rng.sample(FLAG_POOL, n_flags)
    ]
    flag_names = [flag["name"] for flag in flags]
    order = rng.sample(STEP_NAMES, n_steps)

    state = {flag["name"]: flag["init"] for flag in flags}
    steps = []
    for i, name in enumerate(order):
        n_step_pre = min(i, rng.randint(*step_pre))
        need_steps = sorted(rng.sample(order[:i], n_step_pre))
        n_flag_pre = min(n_flags, rng.randint(*flag_pre))
        need_flags = [
            [fn, state[fn]] for fn in sorted(rng.sample(flag_names, n_flag_pre))
        ]
        effects = []
        for fn in rng.sample(flag_names, rng.randint(1, eff_hi)):
            value = (not state[fn]) if rng.random() < 0.8 else rng.random() < 0.5
            effects.append([fn, value])
            state[fn] = value
        steps.append(
            {
                "name": name,
                "need_steps": need_steps,
                "need_flags": need_flags,
                "effects": effects,
            }
        )

    proc = {
        "flags": flags,
        "steps": steps,
        "order": list(order),
        # Presentation order for the doc: an independent random permutation,
        # so the doc never leaks the execution order.
        "doc_order": rng.sample(order, len(order)),
    }
    _assert_order_legal(proc)  # generator invariant, cheap
    return proc


def _is_legal(step: dict, done: set, state: dict) -> bool:
    if step["name"] in done:
        return False
    if any(name not in done for name in step["need_steps"]):
        return False
    return all(state[fn] == want for fn, want in step["need_flags"])


def _prefix_state(proc: dict, k: int) -> tuple[dict, list]:
    """State and done-list after legally executing the first k canonical steps."""
    state = {flag["name"]: flag["init"] for flag in proc["flags"]}
    done: list = []
    for step in proc["steps"][:k]:
        for fn, val in step["effects"]:
            state[fn] = val
        done.append(step["name"])
    return state, done


def _assert_order_legal(proc: dict) -> None:
    state = {flag["name"]: flag["init"] for flag in proc["flags"]}
    done: set = set()
    for step in proc["steps"]:
        if not _is_legal(step, done, state):  # pragma: no cover - generator bug
            raise ValueError(f"generated order is not legal at {step['name']!r}")
        for fn, val in step["effects"]:
            state[fn] = val
        done.add(step["name"])


def _word(proc: dict, flag_name: str, value: bool) -> str:
    for flag in proc["flags"]:
        if flag["name"] == flag_name:
            return flag["words"][0] if value else flag["words"][1]
    raise KeyError(flag_name)  # pragma: no cover - generator bug


def _doc_lines(proc: dict) -> list:
    lines = ["Flags at the start:"]
    for flag in proc["flags"]:
        cur = flag["words"][0] if flag["init"] else flag["words"][1]
        lines.append(
            f"- {flag['name']}: {cur} (values {flag['words'][0]}/{flag['words'][1]})"
        )
    lines.append("")
    lines.append(f"The {len(proc['steps'])} steps:")
    by_name = {step["name"]: step for step in proc["steps"]}
    for name in proc["doc_order"]:
        step = by_name[name]
        needs = list(step["need_steps"]) + [
            f"{fn}={_word(proc, fn, want)}" for fn, want in step["need_flags"]
        ]
        need_txt = ", ".join(needs) if needs else "nothing"
        eff_txt = ", ".join(
            f"{fn}={_word(proc, fn, val)}" for fn, val in step["effects"]
        )
        lines.append(f"- {name}: needs {need_txt}; then {eff_txt}.")
    return lines


# ---------------------------------------------------------------------------
# Atoms
# ---------------------------------------------------------------------------


def gen_atoms(seed: int, level: int, n: int) -> list:
    items = []
    for index in range(n):
        for attempt in range(40):
            item = _gen_one(seed, level, index, attempt)
            if len(item["prompt"]) <= base.ATOM_PROMPT_CHAR_LIMIT:
                break
        items.append(item)
    return items


def _gen_one(seed: int, level: int, index: int, attempt: int) -> dict:
    rng = base.rng_for(FAMILY, seed, level, index, attempt)
    proc = _gen_procedure(rng, level)
    n_steps = len(proc["steps"])

    kind = rng.choice(["flag", "next"])
    if kind == "flag":
        k = rng.randint(1, n_steps)
        state, done = _prefix_state(proc, k)
        touched = {fn for step in proc["steps"][:k] for fn, _ in step["effects"]}
        pool = [flag for flag in proc["flags"] if flag["name"] in touched]
        if level >= 2 and pool and rng.random() < 0.8:
            flag = rng.choice(pool)
        else:
            flag = rng.choice(proc["flags"])
        gold = flag["words"][0] if state[flag["name"]] else flag["words"][1]
        question = (
            f"After the logged steps, what does {flag['name']} read now? "
            "Answer with the value word only."
        )
    else:
        candidates = []
        for k in range(n_steps):
            state, done = _prefix_state(proc, k)
            done_set = set(done)
            legal = [s for s in proc["steps"] if _is_legal(s, done_set, state)]
            if len(legal) == 1:
                candidates.append((k, legal[0]["name"]))
        k, gold = candidates[rng.randrange(len(candidates))]
        _, done = _prefix_state(proc, k)
        question = (
            "Exactly one step is legal to do next. Which one? "
            "Answer with the step name only."
        )

    if done:
        log_line = "Steps done so far, in order: " + ", ".join(done) + "."
    else:
        log_line = "No steps have been done yet."

    lines = ["A kiln-firing procedure is documented below."]
    lines += list(_RULES)
    lines.append("")
    lines += _doc_lines(proc)
    lines += ["", log_line, "", question, "", base.ATOM_ANSWER_INSTRUCTION]
    prompt = "\n".join(lines)

    return {
        "id": f"{FAMILY}-L{level}-s{seed}-{index:04d}",
        "family": FAMILY,
        "level": level,
        "prompt": prompt,
        "gold": gold,
        # Lucky-guess guard: size of the plausible answer space for this item
        # (flag questions are binary; next-step questions range over the steps).
        "answer_domain": 2 if kind == "flag" else n_steps,
    }


def score_atom(item: dict, reply_text: str) -> float:
    return base.score_exact_word(item["gold"], reply_text)


def oracle_atom(item: dict) -> str:
    return f"ANSWER: {item['gold']}"


# ---------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------


class Episode:
    def __init__(self, seed: int, level: int):
        rng = base.rng_for(FAMILY, "episode", seed, level)
        self._proc = _gen_procedure(rng, level)
        self.max_turns = _LEVEL_SHAPE[level][2]
        self.spec = {
            "family": FAMILY,
            "level": level,
            "seed": seed,
            "max_turns": self.max_turns,
            "procedure": self._proc,
        }
        self._state = {flag["name"]: flag["init"] for flag in self._proc["flags"]}
        self._done: list = []
        self._turns = 0
        self._finished = False
        self.last_action_ok = True
        self._by_norm = {
            " ".join(step["name"].split()).lower(): step
            for step in self._proc["steps"]
        }

    def system_prompt(self) -> str:
        total = len(self._proc["steps"])
        lines = ["You are performing a kiln-firing procedure, one action per turn."]
        lines += list(_RULES)
        lines += [
            "- An illegal, repeated, unknown, or malformed action changes "
            "nothing and still costs the turn.",
            "- Flag values are never shown after the start; track them yourself.",
            f"- Goal: complete all {total} steps within {self.max_turns} turns.",
            "",
        ]
        lines += _doc_lines(self._proc)
        lines += [
            "",
            "Action grammar: DO <StepName>",
            base.EPISODE_ACTION_INSTRUCTION,
        ]
        return "\n".join(lines)

    def initial_observation(self) -> str:
        total = len(self._proc["steps"])
        return (
            f"The kiln is prepared. 0 of {total} steps done. "
            f"Turn budget: {self.max_turns}."
        )

    def step(self, action_line: str) -> tuple:
        if self._finished:
            return ("The procedure has already ended.", True)
        self._turns += 1
        total = len(self._proc["steps"])
        self.last_action_ok = False

        match = _ACTION_RE.match(str(action_line or "").strip())
        if not match:
            obs = "Bad action. Use exactly: DO <StepName>"
        else:
            name_norm = " ".join(match.group(1).split()).lower()
            step = self._by_norm.get(name_norm)
            if step is None:
                obs = "No step by that name is in this procedure. Nothing changes."
            elif step["name"] in self._done:
                obs = f"{step['name']} was already done. Nothing changes."
            elif not _is_legal(step, set(self._done), self._state):
                obs = f"Refused: the needs of {step['name']} do not hold. Nothing changes."
            else:
                self.last_action_ok = True
                for fn, val in step["effects"]:
                    self._state[fn] = val
                self._done.append(step["name"])
                obs = (
                    f"{step['name']} is done. "
                    f"({len(self._done)} of {total} steps complete.)"
                )

        if len(self._done) == total:
            self._finished = True
            return (obs + " The firing is complete.", True)
        if self._turns >= self.max_turns:
            self._finished = True
            return (obs + " The turn budget is spent.", True)
        return (obs, False)

    def score(self) -> float:
        return len(self._done) / len(self._proc["steps"])


class OraclePolicy:
    """Replays the generator's canonical legal order, one step per turn."""

    def __init__(self, episode: Episode):
        self._order = list(episode.spec["procedure"]["order"])
        self._index = 0

    def act(self, observation_history: list) -> str:
        name = self._order[min(self._index, len(self._order) - 1)]
        self._index += 1
        return f"DO {name}"


# ---------------------------------------------------------------------------
# Selftest
# ---------------------------------------------------------------------------


def selftest() -> dict:
    module = __import__(__name__, fromlist=["x"])
    stats = {
        "atoms": base.selftest_atoms(module),
        "episodes": base.selftest_episodes(module),
    }
    # Family-specific: golds must vary across items (no constant-answer
    # shortcut) and both question kinds must appear.
    for level in LEVELS:
        sample = gen_atoms(11, level, 30)
        golds = {item["gold"] for item in sample}
        if len(golds) < 4:
            raise base.SelftestError(
                f"{FAMILY} L{level}: golds nearly constant ({sorted(golds)})"
            )
        kinds = {
            "flag" if "value word only" in item["prompt"] else "next"
            for item in sample
        }
        if kinds != {"flag", "next"}:
            raise base.SelftestError(
                f"{FAMILY} L{level}: question kinds not mixed ({sorted(kinds)})"
            )
    return stats
