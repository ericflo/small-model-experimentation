"""patchwheel — repair one corrupted rule in an ordered token-rewrite wheel.

A patchwheel is an ordered list of 3-8 rewrite rules (global substitution,
first/last swap, reverse, rotations, positional drops, duplication) applied
in order to short tapes of invented tokens. Exactly one rule is corrupted;
evidence shows 2-4 example tapes with the ORIGINAL wheel's outputs (at least
one example mismatches under the shown wheel).

Atoms (two kinds, alternating):
  loc — name the corrupted rule number (any rule number whose single-rule
        replacement can make every evidence tape match is accepted; the gold
        set is precomputed by exhaustive search over the rule grammar).
  out — apply the CORRECTED (original) wheel to a fresh input tape and give
        the exact output tape (instances are rejected unless every
        evidence-consistent single-rule repair agrees on that output).

Episodes: observe the shown wheel and evidence; replace rules via
``RULE <k>: <rule text>`` and ``RUN`` to recheck all evidence. Done at
all-match (score 1.0), else the match fraction at the last RUN (0 if never
run). Horizons 4/6/10/14/18/22 across levels L1-L6; levels scale rule count
3..8, tape length 4..9, evidence count 2..4, and the rule-kind pool.
"""

from __future__ import annotations

import re

from .. import base

FAMILY = "patchwheel"
LEVELS = (1, 2, 3, 4, 5, 6)
HAS_EPISODES = True

# Invented token vocabulary, disjoint from every other family's surface.
TOKENS = (
    "varn",
    "ruv",
    "tosk",
    "ulm",
    "yeg",
    "gav",
    "hib",
    "zim",
    "dree",
    "fole",
    "quom",
    "nack",
    "sabb",
    "plim",
)

MAX_TURNS = {1: 4, 2: 6, 3: 10, 4: 14, 5: 18, 6: 22}

_LEVEL_SHAPE = {
    # level: (n_rules, tape_len, n_examples, n_vocab)
    1: (3, 4, 2, 4),
    2: (4, 5, 2, 5),
    3: (5, 6, 3, 5),
    4: (6, 7, 3, 6),
    5: (7, 8, 4, 6),
    6: (8, 9, 4, 7),
}

# Rule-kind pool per level (duplicates weight the draw toward substitutions,
# the workhorse rule).
_LEVEL_KINDS = {
    1: ("becomes", "becomes", "swapfl", "dropafter"),
    2: ("becomes", "becomes", "swapfl", "dropafter", "reverse"),
    3: ("becomes", "becomes", "swapfl", "dropafter", "reverse", "dropbefore"),
    4: (
        "becomes",
        "becomes",
        "swapfl",
        "dropafter",
        "reverse",
        "dropbefore",
        "rotleft",
    ),
    5: (
        "becomes",
        "becomes",
        "swapfl",
        "dropafter",
        "reverse",
        "dropbefore",
        "rotleft",
        "rotright",
        "double",
    ),
    6: (
        "becomes",
        "becomes",
        "swapfl",
        "dropafter",
        "reverse",
        "dropbefore",
        "rotleft",
        "rotright",
        "double",
    ),
}

# Reject loc atoms whose repairable-rule set is larger than this (keeps the
# localization question sharp); after enough attempts the first fitting
# candidate is accepted anyway, so generation always terminates.
_MAX_FIXABLE = {1: 2, 2: 2, 3: 2, 4: 2, 5: 3, 6: 3}

_WANT_LEN_MIN = 2
_WANT_LEN_MAX = 10  # correct outputs stay renderable inside every budget
_GOT_LEN_MAX = 12  # corrupted outputs stay renderable in atom evidence
_SHOW_TOKENS = 10  # got-tape render truncation inside episodes
_ACTION_TOKEN_MAX = 8  # cap on token length accepted from RULE actions

_PARAMLESS = ("swapfl", "reverse", "rotleft", "rotright")

_SEMANTICS = (
    "A patchwheel is an ordered list of rewrite rules run on a token tape:\n"
    "rule 1 first on the whole tape, each later rule on the result. Forms:\n"
    "every A becomes B = every token A turns into B\n"
    "swap first last = exchange the first and last tokens\n"
    "reverse all = reverse the tape\n"
    "rotate left = the first token moves to the end\n"
    "rotate right = the last token moves to the front\n"
    "drop after A = left to right, delete the token right after each A\n"
    "  (a deleted token is not scanned)\n"
    "drop before A = left to right, delete the kept token right before\n"
    "  each A (an A can delete an earlier A)\n"
    "double A = every A turns into A A"
)


# ---------------------------------------------------------------------------
# Wheel mechanics
# ---------------------------------------------------------------------------


def _apply_rule(rule: list, tape: list[str]) -> list[str]:
    kind = rule[0]
    if kind == "becomes":
        return [rule[2] if tok == rule[1] else tok for tok in tape]
    if kind == "swapfl":
        if len(tape) >= 2:
            return [tape[-1]] + tape[1:-1] + [tape[0]]
        return list(tape)
    if kind == "reverse":
        return list(reversed(tape))
    if kind == "rotleft":
        return tape[1:] + tape[:1] if len(tape) >= 2 else list(tape)
    if kind == "rotright":
        return tape[-1:] + tape[:-1] if len(tape) >= 2 else list(tape)
    if kind == "dropafter":
        out: list[str] = []
        i = 0
        while i < len(tape):
            out.append(tape[i])
            i += 2 if (tape[i] == rule[1] and i + 1 < len(tape)) else 1
        return out
    if kind == "dropbefore":
        out = []
        for tok in tape:
            if tok == rule[1] and out:
                out.pop()
            out.append(tok)
        return out
    if kind == "double":
        out = []
        for tok in tape:
            out.append(tok)
            if tok == rule[1]:
                out.append(tok)
        return out
    raise ValueError(f"unknown rule kind {kind!r}")  # pragma: no cover


def _apply_wheel(wheel: list[list], tape: list[str]) -> list[str]:
    for rule in wheel:
        tape = _apply_rule(rule, tape)
    return tape


def _render_rule(rule: list) -> str:
    kind = rule[0]
    if kind == "becomes":
        return f"every {rule[1]} becomes {rule[2]}"
    if kind == "swapfl":
        return "swap first last"
    if kind == "reverse":
        return "reverse all"
    if kind == "rotleft":
        return "rotate left"
    if kind == "rotright":
        return "rotate right"
    if kind == "dropafter":
        return f"drop after {rule[1]}"
    if kind == "dropbefore":
        return f"drop before {rule[1]}"
    if kind == "double":
        return f"double {rule[1]}"
    raise ValueError(f"unknown rule kind {kind!r}")  # pragma: no cover


def _render_wheel(wheel: list[list]) -> list[str]:
    return [f"{i + 1}. {_render_rule(rule)}" for i, rule in enumerate(wheel)]


def _parse_rule_text(text: str) -> list | None:
    """Parse a rule in the family grammar; None if malformed."""
    words = re.findall(r"[a-z]+", text.lower())
    if any(len(word) > _ACTION_TOKEN_MAX for word in words):
        return None
    if len(words) == 4 and words[0] == "every" and words[2] == "becomes":
        return ["becomes", words[1], words[3]]
    if words == ["swap", "first", "last"]:
        return ["swapfl"]
    if words == ["reverse", "all"]:
        return ["reverse"]
    if words == ["rotate", "left"]:
        return ["rotleft"]
    if words == ["rotate", "right"]:
        return ["rotright"]
    if len(words) == 3 and words[0] == "drop" and words[1] == "after":
        return ["dropafter", words[2]]
    if len(words) == 3 and words[0] == "drop" and words[1] == "before":
        return ["dropbefore", words[2]]
    if len(words) == 2 and words[0] == "double":
        return ["double", words[1]]
    return None


# ---------------------------------------------------------------------------
# Repair search (fixable set / answer uniqueness)
# ---------------------------------------------------------------------------


def _candidate_rules(vocab: list[str]) -> list[list]:
    candidates: list[list] = []
    for a in vocab:
        for b in vocab:
            if a != b:
                candidates.append(["becomes", a, b])
    for a in vocab:
        candidates.append(["dropafter", a])
        candidates.append(["dropbefore", a])
        candidates.append(["double", a])
    candidates += [["swapfl"], ["reverse"], ["rotleft"], ["rotright"]]
    return candidates


def _passes_all(wheel: list[list], tapes: list[list[str]], wants: list[list[str]]) -> bool:
    return all(_apply_wheel(wheel, tape) == want for tape, want in zip(tapes, wants))


def _fixable_rules(inst: dict) -> list[int]:
    """1-based rule numbers whose single-rule replacement matches all evidence."""
    fixable = []
    candidates = _candidate_rules(inst["vocab"])
    for j in range(len(inst["shown"])):
        trial = [list(rule) for rule in inst["shown"]]
        for candidate in candidates:
            trial[j] = candidate
            if _passes_all(trial, inst["tapes"], inst["wants"]):
                fixable.append(j + 1)
                break
    return fixable


def _query_determined(inst: dict, query: list[str], gold: list[str]) -> bool:
    """True iff every evidence-consistent single-rule repair agrees on query."""
    candidates = _candidate_rules(inst["vocab"])
    for j in range(len(inst["shown"])):
        trial = [list(rule) for rule in inst["shown"]]
        for candidate in candidates:
            trial[j] = candidate
            if _passes_all(trial, inst["tapes"], inst["wants"]):
                if _apply_wheel(trial, query) != gold:
                    return False
    return True


# ---------------------------------------------------------------------------
# Instance generation (shared by atoms and episodes)
# ---------------------------------------------------------------------------


def _gen_rule(rng, kinds: tuple[str, ...], vocab: list[str]) -> list:
    kind = rng.choice(kinds)
    if kind == "becomes":
        a, b = rng.sample(vocab, 2)
        return ["becomes", a, b]
    if kind in ("dropafter", "dropbefore", "double"):
        return [kind, rng.choice(vocab)]
    return [kind]


def _mutate_rule(rng, rule: list, vocab: list[str]) -> list:
    """A same-grammar corruption that always differs from the original."""
    kind = rule[0]
    if kind == "becomes":
        others = [tok for tok in vocab if tok not in (rule[1], rule[2])]
        if rng.random() < 0.5:
            return ["becomes", rng.choice(others), rule[2]]
        return ["becomes", rule[1], rng.choice(others)]
    if kind == "dropafter":
        if rng.random() < 0.5:
            return ["dropbefore", rule[1]]
        return ["dropafter", rng.choice([tok for tok in vocab if tok != rule[1]])]
    if kind == "dropbefore":
        if rng.random() < 0.5:
            return ["dropafter", rule[1]]
        return ["dropbefore", rng.choice([tok for tok in vocab if tok != rule[1]])]
    if kind == "double":
        return ["double", rng.choice([tok for tok in vocab if tok != rule[1]])]
    return [rng.choice([k for k in _PARAMLESS if k != kind])]


def _gen_instance(rng, level: int) -> dict:
    """An original wheel, a one-rule corruption, and mismatching evidence."""
    n_rules, tape_len, n_examples, n_vocab = _LEVEL_SHAPE[level]
    kinds = _LEVEL_KINDS[level]
    inst = None
    for _ in range(200):
        vocab = list(rng.sample(TOKENS, n_vocab))
        wheel = [_gen_rule(rng, kinds, vocab) for _ in range(n_rules)]
        k = rng.randrange(n_rules)
        shown = [list(rule) for rule in wheel]
        shown[k] = _mutate_rule(rng, wheel[k], vocab)
        tapes = [
            [rng.choice(vocab) for _ in range(tape_len)] for _ in range(n_examples)
        ]
        wants = [_apply_wheel(wheel, tape) for tape in tapes]
        gots = [_apply_wheel(shown, tape) for tape in tapes]
        inst = {
            "vocab": vocab,
            "wheel": wheel,
            "shown": shown,
            "k": k,
            "tapes": tapes,
            "wants": wants,
            "gots": gots,
        }
        if not all(
            _WANT_LEN_MIN <= len(want) <= _WANT_LEN_MAX for want in wants
        ):
            continue
        if not all(len(got) <= _GOT_LEN_MAX for got in gots):
            continue
        if any(got != want for got, want in zip(gots, wants)):
            return inst
    return inst  # pragma: no cover - 200 rejections is practically unreachable


# ---------------------------------------------------------------------------
# Atoms
# ---------------------------------------------------------------------------


def gen_atoms(seed: int, level: int, n: int) -> list[dict]:
    items = []
    for index in range(n):
        picked = None
        fallback = None
        item = None
        for attempt in range(40):
            item = _gen_one(seed, level, index, attempt)
            if len(item["prompt"]) > base.atom_prompt_limit(level):
                continue
            if fallback is None:
                fallback = item
            gold = item["gold"]
            if gold["kind"] == "loc" and len(gold["fixable"]) > _MAX_FIXABLE[level]:
                continue
            if gold["kind"] == "out" and not gold["determined"]:
                continue
            picked = item
            break
        items.append(picked or fallback or item)
    return items


def _gen_one(seed: int, level: int, index: int, attempt: int) -> dict:
    rng = base.rng_for(FAMILY, seed, level, index, attempt)
    inst = _gen_instance(rng, level)
    kind = "loc" if index % 2 == 0 else "out"
    if kind == "loc":
        prompt, gold = _loc_atom(inst)
        answer_domain = len(inst["shown"])
    else:
        prompt, gold = _out_atom(rng, inst)
        answer_domain = 50  # exact token tapes: a wide answer space
    return {
        "id": f"{FAMILY}-L{level}-s{seed}-{index:04d}",
        "family": FAMILY,
        "level": level,
        "prompt": prompt,
        "gold": gold,
        "answer_domain": answer_domain,
    }


def _evidence_lines(inst: dict) -> list[str]:
    lines = []
    for i, (tape, want, got) in enumerate(
        zip(inst["tapes"], inst["wants"], inst["gots"])
    ):
        head = f"E{i + 1}: {' '.join(tape)} => {' '.join(want)}"
        if got == want:
            lines.append(head + " | shown wheel: match")
        else:
            lines.append(head + f" | shown wheel gives {' '.join(got)}")
    return lines


def _loc_atom(inst: dict) -> tuple[str, dict]:
    fixable = _fixable_rules(inst)
    lines = [
        _SEMANTICS,
        "",
        "Exactly ONE rule of this patchwheel was corrupted:",
        "",
        *_render_wheel(inst["shown"]),
        "",
        "Evidence (input => ORIGINAL wheel output):",
        *_evidence_lines(inst),
        "",
        "Which rule number was corrupted? (If replacing a different single",
        "rule could also make every evidence tape match, that number is",
        "accepted too.)",
        "",
        base.ATOM_ANSWER_INSTRUCTION,
    ]
    gold = {"kind": "loc", "fixable": fixable, "planted": inst["k"] + 1}
    return "\n".join(lines), gold


def _out_atom(rng, inst: dict) -> tuple[str, dict]:
    tape_len = len(inst["tapes"][0])
    query = None
    gold_tokens = None
    fallback = None
    for _ in range(30):
        probe = [rng.choice(inst["vocab"]) for _ in range(tape_len)]
        out = _apply_wheel(inst["wheel"], probe)
        if not (_WANT_LEN_MIN <= len(out) <= _GOT_LEN_MAX):
            continue
        if probe in inst["tapes"]:
            continue
        if fallback is None:
            fallback = (probe, out)
        # Prefer queries on which the corruption actually shows.
        if _apply_wheel(inst["shown"], probe) != out:
            query, gold_tokens = probe, out
            break
    if query is None:
        query, gold_tokens = fallback if fallback else (
            list(inst["tapes"][0]),
            list(inst["wants"][0]),
        )
    determined = _query_determined(inst, query, gold_tokens)
    lines = [
        _SEMANTICS,
        "",
        "Exactly ONE rule of this patchwheel was corrupted:",
        "",
        *_render_wheel(inst["shown"]),
        "",
        "Evidence (input => ORIGINAL wheel output):",
        *_evidence_lines(inst),
        "",
        "Work out the corrupted rule from the evidence, repair it, then",
        f"apply the CORRECTED wheel to this fresh input: {' '.join(query)}",
        "Give the output tape as tokens separated by single spaces.",
        "",
        base.ATOM_ANSWER_INSTRUCTION,
    ]
    gold = {"kind": "out", "tokens": gold_tokens, "determined": determined}
    return "\n".join(lines), gold


def score_atom(item: dict, reply_text: str) -> float:
    gold = item["gold"]
    answer = base.extract_answer(reply_text)
    if answer is None:
        return 0.0
    if gold["kind"] == "loc":
        value = base.canon_int(answer)
        return 1.0 if value is not None and value in gold["fixable"] else 0.0
    return 1.0 if base.canon_list(answer) == gold["tokens"] else 0.0


def oracle_atom(item: dict) -> str:
    gold = item["gold"]
    if gold["kind"] == "loc":
        return f"ANSWER: {gold['planted']}"
    return "ANSWER: " + " ".join(gold["tokens"])


# ---------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------

_GRAMMAR_HINT = "Use: RULE <k>: <rule text> | RUN"
_RULE_ACTION_RE = re.compile(r"^\s*rule\s+(\d+)\s*:\s*(.+)$", re.IGNORECASE)


class Episode:
    def __init__(self, seed: int, level: int):
        for attempt in range(40):
            rng = base.rng_for(FAMILY, "episode", seed, level, attempt)
            inst = _gen_instance(rng, level)
            self._install(inst, seed, level)
            if len(self.initial_observation()) <= base.EPISODE_OBS_CHAR_LIMIT:
                break

    def _install(self, inst: dict, seed: int, level: int) -> None:
        self.level = level
        self.max_turns = MAX_TURNS[level]
        self._original = [list(rule) for rule in inst["wheel"]]
        self._wheel = [list(rule) for rule in inst["shown"]]
        self._planted = inst["k"]
        self._tapes = inst["tapes"]
        self._wants = inst["wants"]
        self._turns = 0
        self._solved = False
        self._last_run: tuple[int, int] | None = None
        self.last_action_ok = True
        self.spec = {
            "family": FAMILY,
            "level": level,
            "seed": seed,
            "shown_wheel": [_render_rule(rule) for rule in inst["shown"]],
            "original_wheel": [_render_rule(rule) for rule in inst["wheel"]],
            "planted_rule": inst["k"] + 1,
            "tapes": inst["tapes"],
            "wants": inst["wants"],
            "max_turns": self.max_turns,
        }

    def system_prompt(self) -> str:
        return (
            "You are repairing a patchwheel at the tapeworks bench.\n"
            + _SEMANTICS
            + "\n\nExactly one rule of the shown wheel was corrupted. Evidence"
            " lists each input tape and the output the ORIGINAL wheel gave."
            " Repair the wheel so every evidence tape matches, then RUN to"
            " confirm.\n"
            "Actions (one per turn):\n"
            "RULE <k>: <rule text> — replace rule k"
            " (e.g. RULE 2: every varn becomes ruv)\n"
            "RUN — recheck all evidence tapes and report match/mismatch\n"
            + base.EPISODE_ACTION_INSTRUCTION
        )

    def initial_observation(self) -> str:
        verdicts = ", ".join(
            f"E{i + 1} {'match' if got == want else 'MISMATCH'}"
            for i, (got, want) in enumerate(zip(self._gots(), self._wants))
        )
        lines = [
            "Shown wheel:",
            *_render_wheel(self._wheel),
            "Evidence (input => ORIGINAL output):",
            *(
                f"E{i + 1}: {' '.join(tape)} => {' '.join(want)}"
                for i, (tape, want) in enumerate(zip(self._tapes, self._wants))
            ),
            "Shown wheel now gives: " + verdicts,
        ]
        return "\n".join(lines)

    def _gots(self) -> list[list[str]]:
        return [_apply_wheel(self._wheel, tape) for tape in self._tapes]

    def step(self, action_line: str) -> tuple[str, bool]:
        if self._solved or self._turns >= self.max_turns:
            return "Episode over.", True
        self._turns += 1
        self.last_action_ok = False
        observation = self._apply(action_line)
        done = self._solved or self._turns >= self.max_turns
        return observation, done

    def _apply(self, action_line: str) -> str:
        line = (action_line or "").strip()
        if not line:
            return "Empty action. " + _GRAMMAR_HINT
        match = _RULE_ACTION_RE.match(line)
        if match:
            return self._do_rule(int(match.group(1)), match.group(2))
        upper = line.upper()
        if upper == "RUN":
            self.last_action_ok = True
            return self._do_run()
        if upper.startswith("RUN"):
            return "RUN takes no arguments. " + _GRAMMAR_HINT
        if upper.startswith("RULE"):
            return (
                "RULE needs a number, a colon, and a rule,"
                " e.g. RULE 2: swap first last."
            )
        return "Unknown action. " + _GRAMMAR_HINT

    def _do_rule(self, number: int, text: str) -> str:
        if not 1 <= number <= len(self._wheel):
            return f"No rule {number}. Rules are 1..{len(self._wheel)}."
        rule = _parse_rule_text(text)
        if rule is None:
            return (
                "Bad rule text. Forms: every A becomes B | swap first last |"
                " reverse all | rotate left | rotate right | drop after A |"
                " drop before A | double A."
            )
        self.last_action_ok = True
        self._wheel[number - 1] = rule
        return (
            f"Rule {number} replaced. Current wheel:\n"
            + "\n".join(_render_wheel(self._wheel))
            + "\n(Use RUN to check.)"
        )

    def _do_run(self) -> str:
        gots = self._gots()
        verdicts = []
        first_miss = None
        n_match = 0
        for i, (got, want) in enumerate(zip(gots, self._wants)):
            if got == want:
                n_match += 1
                verdicts.append(f"E{i + 1} match")
            else:
                verdicts.append(f"E{i + 1} MISMATCH")
                if first_miss is None:
                    first_miss = i
        self._last_run = (n_match, len(self._wants))
        lines = ["Run results: " + ", ".join(verdicts)]
        if first_miss is None:
            self._solved = True
            lines.append("All evidence tapes match. Wheel repaired.")
            return "\n".join(lines)
        got = gots[first_miss]
        shown = " ".join(got[:_SHOW_TOKENS])
        if len(got) > _SHOW_TOKENS:
            shown += f" (+{len(got) - _SHOW_TOKENS} more)"
        lines.append(f"E{first_miss + 1} got: {shown}")
        lines.append(f"E{first_miss + 1} want: {' '.join(self._wants[first_miss])}")
        lines.append(f"{n_match}/{len(self._wants)} evidence tapes match.")
        return "\n".join(lines)

    def score(self) -> float:
        if self._solved:
            return 1.0
        if self._last_run is None:
            return 0.0
        return self._last_run[0] / self._last_run[1]


class OraclePolicy:
    """Replace the planted rule with its original form, then RUN."""

    def __init__(self, episode: Episode):
        self._episode = episode
        self._patched = False

    def act(self, observation_history: list[str]) -> str:
        if not self._patched:
            self._patched = True
            k = self._episode._planted
            fix = _render_rule(self._episode._original[k])
            return f"RULE {k + 1}: {fix}"
        return "RUN"


def selftest() -> dict:
    module = __import__(__name__, fromlist=["x"])
    return {
        "atoms": base.selftest_atoms(module),
        "episodes": base.selftest_episodes(module),
    }
