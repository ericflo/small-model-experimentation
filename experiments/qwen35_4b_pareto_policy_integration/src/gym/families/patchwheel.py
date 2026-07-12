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

``oracle_trace`` renders the evidence-elimination solve of an atom as
first-person think-channel text (truth-blind: derived by re-running the
procedure, never by citing the gold).
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

# Skin-shuffling: the invented tape-token vocabulary can be consistently
# renamed without changing mechanics (scoring compares tokens as exact
# strings). EXCLUDED: the rule-grammar words (every/becomes/swap/first/last/
# reverse/all/rotate/left/right/drop/after/before/double), the RULE / RUN
# action grammar, evidence labels (E1...), and the ANSWER protocol word.
SKINNABLE: tuple[str, ...] = TOKENS

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
    return [_pick_full(seed, level, index)["item"] for index in range(n)]


def _pick_full(seed: int, level: int, index: int) -> dict:
    """The generation pick for one atom slot: item plus solving internals."""
    picked = None
    fallback = None
    full = None
    for attempt in range(40):
        full = _gen_full(seed, level, index, attempt)
        if len(full["item"]["prompt"]) > base.atom_prompt_limit(level):
            continue
        if fallback is None:
            fallback = full
        gold = full["item"]["gold"]
        if gold["kind"] == "loc" and len(gold["fixable"]) > _MAX_FIXABLE[level]:
            continue
        if gold["kind"] == "out" and not gold["determined"]:
            continue
        picked = full
        break
    return picked or fallback or full


def _gen_full(seed: int, level: int, index: int, attempt: int) -> dict:
    rng = base.rng_for(FAMILY, seed, level, index, attempt)
    inst = _gen_instance(rng, level)
    kind = "loc" if index % 2 == 0 else "out"
    query = None
    if kind == "loc":
        prompt, gold = _loc_atom(inst)
        answer_domain = len(inst["shown"])
    else:
        query, gold_tokens, determined = _pick_query(rng, inst)
        prompt, gold = _out_atom(inst, query, gold_tokens, determined)
        answer_domain = 50  # exact token tapes: a wide answer space
    item = {
        "id": f"{FAMILY}-L{level}-s{seed}-{index:04d}",
        "family": FAMILY,
        "level": level,
        "prompt": prompt,
        "gold": gold,
        "answer_domain": answer_domain,
    }
    return {"item": item, "inst": inst, "query": query}


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


def _pick_query(rng, inst: dict) -> tuple[list[str], list[str], bool]:
    """Draw the fresh input tape for an out atom (rng order is load-bearing)."""
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
    return query, gold_tokens, determined


def _out_atom(
    inst: dict, query: list[str], gold_tokens: list[str], determined: bool
) -> tuple[str, dict]:
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
# Oracle traces (think-channel distillation text)
# ---------------------------------------------------------------------------

_ID_RE = re.compile(rf"^{FAMILY}-L(\d+)-s(-?\d+)-(\d+)$")
_TRACE_HARD_CAP = 800  # words; deploy think budget is 1024 tokens
_TRACE_SOFT_CAP = 740  # above this the condensed rendering is used
_RECONSTRUCT_CACHE: dict[tuple[int, int, int], dict] = {}


def _reconstruct(item: dict) -> dict:
    """Regenerate the instance behind an item id (generation is deterministic)."""
    match = _ID_RE.match(item["id"])
    if match is None:
        raise ValueError(f"unrecognized {FAMILY} item id: {item['id']!r}")
    level, seed, index = (
        int(match.group(1)),
        int(match.group(2)),
        int(match.group(3)),
    )
    key = (seed, level, index)
    full = _RECONSTRUCT_CACHE.get(key)
    if full is None:
        full = _pick_full(seed, level, index)
        if len(_RECONSTRUCT_CACHE) < 4096:
            _RECONSTRUCT_CACHE[key] = full
    if full["item"]["prompt"] != item["prompt"]:
        raise ValueError(f"cannot reconstruct the instance behind {item['id']}")
    return full


def _tape(tokens: list[str]) -> str:
    return " ".join(tokens)


def _names(indices: list[int]) -> str:
    names = [f"E{i + 1}" for i in indices]
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def _walk(wheel: list[list], tape: list[str]) -> list[tuple[list, list[str]]]:
    steps = []
    current = list(tape)
    for rule in wheel:
        current = _apply_rule(rule, current)
        steps.append((rule, current))
    return steps


def _step_line(variant: int, j: int, text: str, after: str) -> str:
    return (
        f"Rule {j}, {text}: {after}.",
        f"Apply rule {j} ({text}): {after}.",
        f"Rule {j} is {text}, giving {after}.",
    )[variant]


def _divergence_lines(got: list[str], want: list[str]) -> list[str]:
    lines = []
    if len(got) != len(want):
        lines.append(
            f"My tape has {len(got)} tokens; the original's has {len(want)}."
        )
    p = next(
        (i for i in range(min(len(got), len(want))) if got[i] != want[i]), None
    )
    if p is not None:
        lines.append(
            f"The first difference is at slot {p + 1}:"
            f" I have {got[p]} where it should be {want[p]}."
        )
    # A one-line reading of what the corruption did on this input.
    if len(got) != len(want):
        lines.append("So the corruption changes how many tokens survive.")
    elif sorted(got) == sorted(want):
        lines.append("Same tokens, different order, so positions got scrambled.")
    else:
        lines.append("Same length, but the token identities are off.")
    return lines


def oracle_trace(item: dict) -> str:
    full = _reconstruct(item)
    text = _build_trace(item, full, condensed=False)
    if len(text.split()) > _TRACE_SOFT_CAP:
        text = _build_trace(item, full, condensed=True)
    return text


def _build_trace(item: dict, full: dict, condensed: bool) -> str:
    inst = full["inst"]
    gold = item["gold"]
    rng = base.rng_for(FAMILY, "trace", item["id"])
    v = rng.randrange(3)
    sv = 0 if condensed else v
    n = len(inst["shown"])
    m = len(inst["tapes"])
    kp = inst["k"] + 1
    shown_txt = _render_rule(inst["shown"][inst["k"]])
    orig_txt = _render_rule(inst["wheel"][inst["k"]])
    mism = [i for i in range(m) if inst["gots"][i] != inst["wants"][i]]
    star = mism[0]
    tape0 = inst["tapes"][star]
    want = inst["wants"][star]
    shown_steps = _walk(inst["shown"], tape0)
    got = shown_steps[-1][1]
    paragraphs: list[str] = []

    # 1. Goal + evidence scan.
    if gold["kind"] == "loc":
        opener = (
            f"I need to work out which rule of this {n}-rule wheel was"
            " corrupted.",
            f"One of the {n} rules in this wheel is wrong, and I have to name"
            " its number.",
            f"Let me find the corrupted rule in this {n}-rule wheel.",
        )[v]
    else:
        qs = _tape(full["query"])
        opener = (
            f"I need to find the corrupted rule in this {n}-rule wheel, repair"
            f" it, and run the corrected wheel on the fresh tape: {qs}.",
            "One rule in this wheel is wrong. I have to fix it, then push the"
            f" fresh tape {qs} through the corrected wheel.",
            f"The plan: locate the bad rule among the {n} shown, restore it,"
            f" then apply the corrected wheel to {qs}.",
        )[v]
    matched = [i for i in range(m) if i not in mism]
    if matched:
        scan = (
            f"The evidence says {_names(matched)}"
            f" {'matches' if len(matched) == 1 else 'match'} under the shown"
            f" wheel, while {_names(mism)}"
            f" {'comes' if len(mism) == 1 else 'come'} out wrong."
        )
    else:
        scan = "Under the shown wheel, every example comes out wrong."
    lines = [opener, scan]
    if matched and not condensed:
        lines.append(
            f"Whatever repair I settle on has to keep {_names(matched)}"
            f" matching too."
        )
    paragraphs.append("\n".join(lines))

    # 2. Rule-by-rule walk of the first mismatching example.
    lines = [
        (
            f"I'll trace E{star + 1} through the shown wheel rule by rule.",
            f"Let me push E{star + 1}'s tape through the shown wheel and watch"
            " each step.",
            f"First, E{star + 1} under the shown wheel, one rule at a time.",
        )[v],
        f"Start: {_tape(tape0)}.",
    ]
    for j, (rule, after) in enumerate(shown_steps, start=1):
        lines.append(_step_line(sv, j, _render_rule(rule), _tape(after)))
    lines.append(
        f"That leaves {_tape(got)}, but the original wheel gave {_tape(want)}."
    )
    lines.extend(_divergence_lines(got, want))
    paragraphs.append("\n".join(lines))

    # 3. Localize the fault and re-run the example with the repair in place.
    lines = [
        (
            "Exactly one rule was changed, so I need a slot where a single"
            " repair makes every example come out right.",
            "Only one rule is corrupted, so I look for a slot where one"
            " replacement squares all the evidence.",
            "Since just one rule was tampered with, one well-chosen repair has"
            " to fix every example at once.",
        )[v]
    ]
    if not condensed and (gold["kind"] == "loc" or n <= 5):
        # Genuine dead ends: slots the exhaustive repair search ruled out.
        if gold["kind"] == "loc":
            fixable = set(gold["fixable"])
        else:
            fixable = set(_fixable_rules(inst))
        nonfix = [j for j in range(1, n + 1) if j not in fixable]
        ruled = ([j for j in nonfix if j < kp] + [j for j in nonfix if j > kp])[:2]
        if ruled:
            entry = tape0 if ruled[0] == 1 else shown_steps[ruled[0] - 2][1]
            lines.append(
                f"Rule {ruled[0]} is a dead end: the tape reaching it in"
                f" E{star + 1} is {_tape(entry)}, and no single replacement"
                f" there makes all {m} examples match."
            )
        if len(ruled) > 1:
            lines.append(f"Rule {ruled[1]} fails the same way.")
    lines.append(
        (
            f"Try rule {kp}. The shown wheel says '{shown_txt}'. Suppose it"
            f" originally read '{orig_txt}'.",
            f"My suspicion is rule {kp}, shown as '{shown_txt}'. What if the"
            f" original was '{orig_txt}'?",
            f"Consider rule {kp}, currently '{shown_txt}'. Test it as"
            f" '{orig_txt}'.",
        )[v]
    )
    entry = tape0 if kp == 1 else shown_steps[kp - 2][1]
    if kp == 1:
        lines.append(
            f"Rule {kp} acts first, straight on the input {_tape(tape0)}."
        )
    else:
        span = (
            "Rule 1 is"
            if kp == 2
            else "Rules 1 and 2 are"
            if kp == 3
            else f"Rules 1 through {kp - 1} are"
        )
        lines.append(
            f"{span} untouched, so the tape entering slot {kp} is still"
            f" {_tape(entry)}."
        )
    if gold["kind"] == "out" and condensed:
        redo = _apply_wheel(inst["wheel"], tape0)
        lines.append(
            f"With rule {kp} read as '{orig_txt}', E{star + 1} comes out"
            f" {_tape(redo)} - match."
        )
    else:
        current = list(entry)
        for j in range(kp, n + 1):
            rule = inst["wheel"][j - 1]
            current = _apply_rule(rule, current)
            lines.append(_step_line(sv, j, _render_rule(rule), _tape(current)))
        lines.append(
            f"Final: {_tape(current)} - exactly what the original wheel gave"
            f" for E{star + 1}."
        )
    paragraphs.append("\n".join(lines))

    # 4. Cross-check the remaining examples under the repaired wheel.
    others = [i for i in range(m) if i != star]
    if others:
        lines = [
            (
                f"Now check the other example{'s' if len(others) != 1 else ''}"
                f" with rule {kp} repaired.",
                f"Cross-check the rest with rule {kp} repaired.",
                "Does that repair hold everywhere?",
            )[v]
        ]
        for i in others:
            out = _apply_wheel(inst["wheel"], inst["tapes"][i])
            if condensed:
                lines.append(f"E{i + 1} comes out {_tape(out)} - match.")
            else:
                lines.append(
                    f"E{i + 1}: {_tape(inst['tapes'][i])} comes out"
                    f" {_tape(out)}; the original gave"
                    f" {_tape(inst['wants'][i])} - match."
                )
        paragraphs.append("\n".join(lines))

    # 5. Conclude (loc) or run the fresh tape through the corrected wheel.
    if gold["kind"] == "loc":
        paragraphs.append(
            (
                f"Every example lines up once rule {kp} reads '{orig_txt}', so"
                f" that is where the corruption sits. The corrupted rule is"
                f" number {kp}.",
                f"That single repair squares all {m} examples. So the"
                f" corrupted rule is rule {kp}.",
                f"With rule {kp} restored, everything matches. The corrupted"
                f" rule is number {kp}.",
            )[v]
        )
    else:
        query = full["query"]
        lines = [
            (
                "Now run the fresh tape through the corrected wheel.",
                "Time to push the fresh tape through the corrected wheel.",
                "With the wheel fixed, I apply it to the fresh tape.",
            )[v],
            f"Start: {_tape(query)}.",
        ]
        current = list(query)
        for j, rule in enumerate(inst["wheel"], start=1):
            current = _apply_rule(rule, current)
            lines.append(_step_line(sv, j, _render_rule(rule), _tape(current)))
        final = _tape(current)
        lines.append(
            (
                f"The final output tape is {final}.",
                f"So the final output tape is {final}.",
                f"That gives the final output tape: {final}.",
            )[v]
        )
        paragraphs.append("\n".join(lines))

    return "\n\n".join(paragraphs)


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


def _selftest_traces(n_per_level: int = 12) -> dict:
    stats: dict = {"family": FAMILY, "levels": {}}
    for level in LEVELS:
        items = gen_atoms(7, level, n_per_level)
        n_exact = 0
        max_words = 0
        for item in items:
            trace = oracle_trace(item)
            if trace != oracle_trace(item):
                raise base.SelftestError(
                    f"{FAMILY} L{level} {item['id']}: trace not deterministic"
                )
            words = len(trace.split())
            max_words = max(max_words, words)
            if words > _TRACE_HARD_CAP:
                raise base.SelftestError(
                    f"{FAMILY} L{level} {item['id']}: trace has {words} words"
                    f" > {_TRACE_HARD_CAP}"
                )
            lowered = trace.lower()
            for word in base.FORBIDDEN_WORDS:
                if word in lowered:
                    raise base.SelftestError(
                        f"{FAMILY}: forbidden word {word!r} in trace"
                    )
            reply = trace + "\n\n" + oracle_atom(item)
            if score_atom(item, reply) == 1.0:
                n_exact += 1
        exact = n_exact / len(items)
        if exact < 0.95:
            raise base.SelftestError(
                f"{FAMILY} L{level}: trace+answer replies score 1.0 on only"
                f" {exact:.3f} of items (< 0.95)"
            )
        stats["levels"][level] = {
            "trace_exact": round(exact, 4),
            "max_words": max_words,
        }
    return stats


def selftest() -> dict:
    module = __import__(__name__, fromlist=["x"])
    return {
        "atoms": base.selftest_atoms(module),
        "episodes": base.selftest_episodes(module),
        "traces": _selftest_traces(),
    }
