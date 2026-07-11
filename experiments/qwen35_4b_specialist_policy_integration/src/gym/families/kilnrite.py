"""kilnrite — protocol/state-machine compliance over a documented firing procedure.

Each item invents a small kiln-firing procedure: 4-12 named steps with
documented needs (steps that must already be done, flag values that must hold)
and effects (flag assignments), plus 2-6 named boolean flags with invented
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

Frontier levels (L5: 10 steps / 5 flags, horizon 18; L6: 12 steps / 6 flags,
horizon 22) additionally force LONG performed-step logs on atoms: the logged
legal prefix is drawn from the deep end of the procedure, so many effect
applications must be tracked before the question can be answered.

``oracle_trace`` distills the hand-coded solving procedure into think-channel
training text: a first-person, truth-blind replay of the log with the running
flag values shown, ending on the answer value.
"""

from __future__ import annotations

import re

from .. import base

FAMILY = "kilnrite"
LEVELS = (1, 2, 3, 4, 5, 6)
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
    5: (10, 5, 18, (1, 2), (1, 2), 2),
    6: (12, 6, 22, (1, 3), (1, 3), 2),
}

# Frontier atoms question a DEEP legal prefix: at least this many steps have
# already been performed, so the log to replay mentally is long.
_FRONTIER_MIN_LOG = {5: 6, 6: 8}

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
            if len(item["prompt"]) <= base.atom_prompt_limit(level):
                break
        items.append(item)
    return items


def _gen_one(seed: int, level: int, index: int, attempt: int) -> dict:
    item, _meta = _gen_one_with_meta(seed, level, index, attempt)
    return item


def _gen_one_with_meta(seed: int, level: int, index: int, attempt: int) -> tuple:
    """Generate one atom plus the internals ``oracle_trace`` narrates from.

    RNG consumption is byte-identical to the historical ``_gen_one``, so item
    prompts/golds are unchanged; the meta is a pure side-channel.
    """
    rng = base.rng_for(FAMILY, seed, level, index, attempt)
    proc = _gen_procedure(rng, level)
    n_steps = len(proc["steps"])

    # Frontier levels question deep prefixes only: the performed-step log is
    # long, so many effects must be replayed before answering. The branch is
    # level-gated so L1-L4 consume RNG exactly as before.
    min_k = _FRONTIER_MIN_LOG.get(level, 1)

    flag_name = None
    kind = rng.choice(["flag", "next"])
    if kind == "flag":
        k = rng.randint(min_k, n_steps)
        state, done = _prefix_state(proc, k)
        touched = {fn for step in proc["steps"][:k] for fn, _ in step["effects"]}
        pool = [flag for flag in proc["flags"] if flag["name"] in touched]
        if level >= 2 and pool and rng.random() < 0.8:
            flag = rng.choice(pool)
        else:
            flag = rng.choice(proc["flags"])
        flag_name = flag["name"]
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
        if level >= 5:
            # Keep only deep prefixes when any exist; k = n_steps - 1 is
            # always a unique-next candidate, so this never empties the pool.
            deep = [cand for cand in candidates if cand[0] >= min_k]
            if deep:
                candidates = deep
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

    item = {
        "id": f"{FAMILY}-L{level}-s{seed}-{index:04d}",
        "family": FAMILY,
        "level": level,
        "prompt": prompt,
        "gold": gold,
        # Lucky-guess guard: size of the plausible answer space for this item
        # (flag questions are binary; next-step questions range over the steps).
        "answer_domain": 2 if kind == "flag" else n_steps,
    }
    meta = {"proc": proc, "kind": kind, "k": k, "flag_name": flag_name}
    return item, meta


def score_atom(item: dict, reply_text: str) -> float:
    return base.score_exact_word(item["gold"], reply_text)


def oracle_atom(item: dict) -> str:
    return f"ANSWER: {item['gold']}"


# ---------------------------------------------------------------------------
# Oracle trace: think-channel narration of the solving procedure
# ---------------------------------------------------------------------------

_TRACE_WORD_CAP = 800
_ID_RE = re.compile(rf"^{FAMILY}-L(\d+)-s(\d+)-(\d+)$")


def _replay_meta(item: dict) -> dict:
    """Recover the generation internals for an emitted atom from its id.

    Mirrors ``gen_atoms``'s retry rule exactly (first attempt whose prompt
    fits the level's limit), then verifies the regenerated prompt matches.
    """
    match = _ID_RE.match(str(item.get("id", "")))
    if not match:
        raise ValueError(f"unrecognized {FAMILY} item id: {item.get('id')!r}")
    level, seed, index = (int(group) for group in match.groups())
    candidate: dict = {}
    meta: dict = {}
    for attempt in range(40):
        candidate, meta = _gen_one_with_meta(seed, level, index, attempt)
        if len(candidate["prompt"]) <= base.atom_prompt_limit(level):
            break
    if candidate["prompt"] != item["prompt"]:
        raise ValueError(
            f"{item['id']}: item does not match its regenerated form"
        )
    return meta


def _state_words(proc: dict, state: dict) -> str:
    return ", ".join(
        f"{flag['name']} {_word(proc, flag['name'], state[flag['name']])}"
        for flag in proc["flags"]
    )


def oracle_trace(item: dict) -> str:
    """First-person, truth-blind reasoning trace that solves the atom.

    Narrates the hand-coded procedure: initialize flags, replay the logged
    steps applying each documented effect with the running flag values shown,
    then read off the asked flag or eliminate next-step candidates by their
    needs. Deterministic per item; phrasing varies via the item-keyed RNG.
    """
    return _trace_and_answer(item)[0]


def _trace_and_answer(item: dict) -> tuple:
    meta = _replay_meta(item)
    text, answer = _render_trace(item, meta, state_every=1)
    if len(text.split()) > _TRACE_WORD_CAP:  # defensive: thin the state lines
        text, answer = _render_trace(item, meta, state_every=3)
    return text, answer


def _render_trace(item: dict, meta: dict, state_every: int) -> tuple:
    proc = meta["proc"]
    kind = meta["kind"]
    k = meta["k"]
    steps = proc["steps"]
    rng = base.rng_for(FAMILY, "trace", item["id"])
    state = {flag["name"]: flag["init"] for flag in proc["flags"]}
    lines: list = []

    if kind == "flag":
        fname = meta["flag_name"]
        lines.append(rng.choice((
            f"I need to work out what {fname} reads after the logged steps. "
            "A flag only changes through a step's 'then' clause, so I can "
            "replay the log in order and track every flag as I go.",
            f"The question is the current value of {fname}. Each logged step "
            "applied its documented 'then' effects the moment it was done, "
            "so I start from the initial flags and apply the log step by step.",
            "To answer this I need the hidden state right now, which means "
            "replaying the log. The flags begin at their listed values, and "
            "each done step resets exactly the flags named after 'then'.",
        )))
    else:
        lines.append(rng.choice((
            "Exactly one step is supposed to be legal next, so I need the "
            "current state first; then I can test each remaining step's "
            "needs against it.",
            "I have to find the unique step whose needs all hold right now. "
            "First I replay the log to get the current flags, then I check "
            "every step that has not been done yet.",
            "To find the legal next step I need two things: which steps are "
            "already done, and what the flags read now. Both come from "
            "replaying the log.",
        )))

    lines.append(rng.choice((
        f"Starting flags: {_state_words(proc, state)}.",
        f"At the start the flags read {_state_words(proc, state)}.",
        f"Initial state: {_state_words(proc, state)}.",
    )))

    # Small procedures (L1-L4) get a needs-held check per replayed step:
    # genuine simulation detail with the actual values. Frontier replays are
    # long, so there the effect+state lines alone keep the token budget sane.
    verbose = len(steps) <= 8

    done: list = []
    if k == 0:
        lines.append(
            "No steps have been done yet, so that starting state is also "
            "the current state."
        )
    for pos, step in enumerate(steps[:k], start=1):
        changes = []
        for fn, val in step["effects"]:
            old = state[fn]
            state[fn] = val
            new_word = _word(proc, fn, val)
            if old == val:
                changes.append(f"{fn} is set to {new_word}, which it already read")
            else:
                changes.append(f"{fn} goes from {_word(proc, fn, old)} to {new_word}")
        eff_doc = ", ".join(
            f"{fn}={_word(proc, fn, val)}" for fn, val in step["effects"]
        )
        change_txt = " and ".join(changes)
        lines.append(rng.choice((
            f"Log step {pos} is {step['name']}: the doc says then {eff_doc}, "
            f"so {change_txt}.",
            f"Step {pos} of the log, {step['name']}, sets {eff_doc}: {change_txt}.",
            f"Next the log has {step['name']}, whose effect is {eff_doc}, "
            f"so {change_txt}.",
        )))
        if verbose:
            checks = [f"{s} was already done" for s in step["need_steps"]]
            checks += [
                f"{fn} read {_word(proc, fn, want)}"
                for fn, want in step["need_flags"]
            ]
            if checks:
                clause = " and ".join(checks)
                lines.append(rng.choice((
                    f"Its needs held at that point: {clause}.",
                    f"And it was allowed then, since {clause}.",
                )))
        done.append(step["name"])
        if pos % state_every == 0 or pos == k:
            lines.append(rng.choice((
                f"Flags now: {_state_words(proc, state)}.",
                f"So the state reads {_state_words(proc, state)}.",
                f"Running state: {_state_words(proc, state)}.",
            )))

    if kind == "flag":
        fname = meta["flag_name"]
        answer = _word(proc, fname, state[fname])
        setters = [
            step["name"]
            for step in steps[:k]
            if any(fn == fname for fn, _ in step["effects"])
        ]
        if setters:
            names = ", ".join(setters)
            lines.append(rng.choice((
                f"As a check: the only logged steps that touch {fname} are "
                f"{names}, and the last of those, {setters[-1]}, left it {answer}.",
                f"Double-checking {fname} itself: it was set by {names}; the "
                f"most recent of those, {setters[-1]}, left it at {answer}.",
            )))
        else:
            lines.append(
                f"Notably, none of the logged steps ever touches {fname}, so "
                f"it still holds its starting value of {answer}."
            )
        lines.append(rng.choice((
            f"So after the logged steps, {fname} reads {answer}.",
            f"That settles it: {fname} currently reads {answer}.",
            f"So the value of {fname} right now is {answer}.",
        )))
    else:
        done_set = set(done)
        if done:
            lines.append(rng.choice((
                "Steps already done are out, since each step runs at most "
                f"once: that rules out {', '.join(done)}.",
                f"Each step runs at most once, so {', '.join(done)} "
                "are no longer available.",
            )))
        lines.append(rng.choice((
            "Now I test each remaining step against this state.",
            "With that state in hand, I check the needs of every step not "
            "yet done.",
        )))
        answer = None
        by_name = {step["name"]: step for step in steps}
        for name in proc["doc_order"]:
            if name in done_set:
                continue
            step = by_name[name]
            missing = [s for s in step["need_steps"] if s not in done_set]
            bad = [
                (fn, want)
                for fn, want in step["need_flags"]
                if state[fn] != want
            ]
            if missing:
                missing_txt = " and ".join(missing)
                lines.append(rng.choice((
                    f"{name} is out: it needs {missing_txt} done first, and "
                    "that has not happened.",
                    f"{name} needs {missing_txt} before it, and the log shows "
                    "no such step — ruled out.",
                )))
            elif bad:
                fn, want = bad[0]
                want_word = _word(proc, fn, want)
                cur_word = _word(proc, fn, state[fn])
                lines.append(rng.choice((
                    f"{name} wants {fn}={want_word}, but {fn} reads "
                    f"{cur_word} right now — ruled out.",
                    f"{name} is blocked: it needs {fn}={want_word} and "
                    f"{fn} is {cur_word}.",
                )))
            else:
                answer = name
                needs = list(step["need_steps"]) + [
                    f"{fn}={_word(proc, fn, want)}"
                    for fn, want in step["need_flags"]
                ]
                if needs:
                    lines.append(
                        f"{name} needs {', '.join(needs)} — every one of "
                        f"those holds right now, so {name} is legal."
                    )
                else:
                    lines.append(
                        f"{name} needs nothing at all, so nothing blocks it."
                    )
        lines.append(rng.choice((
            f"So the unique legal next step is {answer}.",
            f"That identifies the one legal move: the next step is {answer}.",
            "Everything else is unavailable or blocked, so the legal next "
            f"step is {answer}.",
        )))

    return "\n".join(lines), answer


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
    stats["traces"] = _selftest_traces()
    return stats


def _selftest_traces() -> dict:
    """Oracle-trace checks per level: trace + terse ANSWER must solve the
    atom, fit the deploy think budget, avoid firewall vocabulary, be
    deterministic, and derive the same answer the generator emitted."""
    per_level: dict = {}
    for level in LEVELS:
        sample = gen_atoms(11, level, 12)
        solved = 0
        words_min, words_max, words_sum = 10**9, 0, 0
        for item in sample:
            trace, derived = _trace_and_answer(item)
            if trace != oracle_trace(item):
                raise base.SelftestError(
                    f"{FAMILY} L{level} {item['id']}: trace not deterministic"
                )
            words = len(trace.split())
            if words > _TRACE_WORD_CAP:
                raise base.SelftestError(
                    f"{FAMILY} L{level} {item['id']}: trace has {words} words "
                    f"> {_TRACE_WORD_CAP}"
                )
            lowered = trace.lower()
            for word in base.FORBIDDEN_WORDS:
                if word in lowered:
                    raise base.SelftestError(
                        f"{FAMILY} L{level} {item['id']}: forbidden word "
                        f"{word!r} in trace"
                    )
            if derived != item["gold"]:
                raise base.SelftestError(
                    f"{FAMILY} L{level} {item['id']}: trace derives "
                    f"{derived!r}, generator emitted {item['gold']!r}"
                )
            reply = trace + "\n\n" + oracle_atom(item)
            if score_atom(item, reply) == 1.0:
                solved += 1
            words_min = min(words_min, words)
            words_max = max(words_max, words)
            words_sum += words
        if solved / len(sample) < 0.95:
            raise base.SelftestError(
                f"{FAMILY} L{level}: trace+answer scores 1.0 on only "
                f"{solved}/{len(sample)} items"
            )
        per_level[level] = {
            "n": len(sample),
            "solve": round(solved / len(sample), 4),
            "words_min": words_min,
            "words_max": words_max,
            "words_mean": round(words_sum / len(sample), 1),
        }
    return per_level
