#!/usr/bin/env python3
"""Designed feedloop-only dose-scale curriculum: eight episode formalisms.

Lifecycle 15 split the episode verdict: the narrated hidden-state chain
lesson installed while the feedback-loop lesson (``u_feedloop``) died at
0/20 on fresh instances at an 80-row dose. Three small-dose pedagogies have
now failed at the same blocking family; the one permitted mechanism class is
DOSE SCALE (C43: partial installs were data-limited). This cell doses ONLY
the ``u_feedloop`` lesson at 800 rows — ten times the failed dose — on
EIGHT invented legality-bounded machine formalisms:

- ``troughline``, ``trinketcord``, ``crankwheel``, ``sigilslate``: the
  reference cell's four formalisms, machinery reused verbatim, FRESH
  instances at this cell's own construction seed (zero row overlap with the
  reference corpus is enforced by the corpus builder's overlap receipt);
- ``barrowyoke``, ``balesled``, ``millround``, ``skeinreel``: four NEW
  formalisms in the same spirit with fresh invented vocabulary. Every
  parameterized operation is explicitly BOUNDED in the rendered spec text
  (heave sizes 1-4 sacks; exactly four bale brands; windlass settings 1-4
  vanes; reel counts 1-5 coils), the bounding clause is verified verbatim in
  the prompt, and the extended-grammar audit (below) probes past every
  documented bound.

Every row is one complete two-round eliminative episode rendered as plain
text inside ONE user message: a written step sequence carries exactly one
wrong step; trial one shows failing evidence (wanted vs finished); an
earlier WRONG fix attempt — plausible because at least two single-step
changes square with trial one alone — was tried, and a second trial from a
fresh start shows it failing too. The assistant narrates what the
second-trial evidence eliminates, then answers the unique correct change.
Repairs are easy by design; the lesson is the loop. All reviewed invariants
are kept: >=2 legal fix candidates after round-1 evidence (the wrong attempt
among them), exactly 1 after rounds 1+2, and a second enumeration over an
EXTENDED grammar verifies that every extra survivor is excluded by the
documented legality clause alone. The extended probe's scope is stated
per formalism in ``EXTENDED_PROBE_SCOPE`` and recorded row-by-row in the
audit: numeric parameters are probed up to ``EXTENDED_AMOUNT_BOUND``; item
parameters (knots, etches, lashes/shoves) are probed over the FULL module
pools; and for the two named-container machines (``troughline``,
``barrowyoke``) the CONTAINER dimension of every operation is additionally
probed over the full module pool via a tolerant probe apply in which
phantom containers start empty (an op touching a phantom container leaves
an extra key in the state and therefore can never reproduce the wanted
outcome — verified by the same survivor enumeration). Rows that fail
bounded uniqueness are rejected and redrawn from the same deterministic
rng stream (attempt cap 5000).

Every answer is computed by executing the specification; a
banned-vocabulary scan rejects any leak of benchmark family names, gym
flavor nouns, prior surface pools (INCLUDING the statechain formalisms'
noun pools — only the four reused feedloop formalisms' own nouns are
retained), or the public blocker-family description nouns.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ANSWER_LINE = "End with exactly one line:\nANSWER: <answer>"

# Retained surfaces inherited from the reference cell's FEEDLOOP lessons.
# These are deliberately NOT fresh: they appear in the reference corpus and
# are excluded from this cell's fresh-surface audit (documented inheritance).
TROUGHS = ("brann", "clave", "dorst", "grelm", "murd", "prell")
TRINKETS = ("plome", "quenn", "trell", "vosk", "yorv", "zilk")
WHEELFACES = ("drasp", "fyne", "golve", "hilm", "juss", "krev")
SIGILS = ("morv", "nilt", "pryn", "quav", "rulf", "senn")
INHERITED_SURFACE_TOKENS = (
    "troughline", "trinketcord", "crankwheel", "sigilslate",
    *TROUGHS, *TRINKETS, *WHEELFACES, *SIGILS,
    "trough", "troughs", "trinket", "trinkets", "sigil", "sigils",
)

# Fresh invented vocabulary for the four NEW formalisms. Every token here is
# grep-verified against every predecessor corpus, stream, and frozen gate by
# build_corpus.py (zero case-insensitive word-boundary hits allowed).
BARROWS = ("askel", "brumm", "dimber", "fluss", "grond", "torvic")
BALES = ("gorbel", "harnik", "jelve", "kimm", "lorsk", "pindle")
VANES = ("ambel", "cresk", "durnic", "hulvet", "ostrel", "twemb")
LAYS = ("murled", "ferren")
FRESH_SURFACE_TOKENS = (
    # machine surface names
    "barrowyoke", "balesled", "millround", "skeinreel",
    # item pools
    *BARROWS, *BALES, *VANES, *LAYS,
    # distinctive machine nouns and readout words
    "barrow", "barrows", "sack", "sacks", "heave", "heaved", "yoke",
    "bale", "bales", "sled", "lash", "lashed", "uncouple",
    "brand", "brands", "vane", "vanes", "windlass", "creel", "shoe",
    "reel", "coil", "coils",
)

BANNED_PROMPT_TOKENS = (
    # public benchmark family names
    "chronicle", "lockpick", "menders", "mirage", "rites", "siftstack", "sirens",
    "stockade", "toolsmith", "warren",
    # gym replay family names
    "kilnrite", "ferrier", "glyphgate", "burrowmaze", "loomfix", "gatepost",
    "stallwright", "patchwheel", "packhouse", "foundry", "caravan", "runeward",
    # gym-flavored nouns that would echo the replay surfaces
    "kiln", "loom", "burrow", "chamber",
    # predecessor universal-line surface pools (multi-character alphabetic)
    "amber", "cobalt", "fawn", "indigo", "jade", "lilac", "ochre", "pearl", "rust",
    "teal", "umber", "violet",
    "bex", "cor", "dun", "fal", "gim", "hup", "jor", "kav", "lem", "nix", "pov", "ruz",
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "theta", "iota", "kappa",
    "lambda", "sigma",
    "argon", "boron", "radon", "xenon", "krypton", "helium", "lithium", "sulfur",
    "iodine", "nickel",
    "heron", "otter", "lynx", "ibex", "crane", "vole", "skink", "tapir", "zebu",
    "okapi", "bison", "finch",
    "beryl", "coral", "flint", "garnet", "jasper", "onyx", "opal", "quartz", "topaz",
    "zircon", "agate", "pyrite",
    # predecessor attributes and capabilities
    "mass", "glow", "span", "charge", "heft", "shine", "reach", "pulse",
    "text", "image", "exact", "stream", "audio", "ledger", "cipher", "relay", "vault",
    # goal-gap / de-stack axis surface pools (trees, rivers, winds, moons)
    "alder", "aspen", "birch", "cedar", "elm", "fir", "hazel", "larch", "maple",
    "oak", "rowan", "willow",
    "arno", "danube", "ebro", "indus", "loire", "niger", "oder", "rhone", "tagus",
    "volga", "yukon", "zambezi",
    "bora", "chinook", "foehn", "gale", "mistral", "monsoon", "sirocco", "squall",
    "typhoon", "zephyr",
    "callisto", "europa", "ganymede", "io", "titan", "triton", "phobos", "deimos",
    "rhea", "dione", "mimas", "oberon",
    # goal-gap / de-stack axis scene nouns and attributes
    "berth", "dock", "quay", "pier", "tally", "surge", "surges", "grove",
    "station", "stations", "walkway", "walkways", "harbor", "gauge", "gauges",
    "dial", "dials", "marker", "markers",
    # public blocker-family description nouns (menders / rites)
    "program", "programs", "debug", "debugged", "debugging", "bug", "bugs",
    "test", "tests", "tested", "testing", "repair", "repairs", "repaired",
    "rerun", "reruns", "protocol", "protocols", "state-machine", "flag", "flags",
    "flagged",
    # the statechain cells' formalisms: machine names, item-noun pools, and
    # distinctive nouns/readout words. The retained FEEDLOOP surfaces
    # (troughline/trinketcord/crankwheel/sigilslate and their pools) are
    # deliberately NOT banned — they are this dose's inherited vocabulary.
    "brewvat", "courierloft", "peatstove", "muletrack",
    "orvan", "welk", "xemb", "yurr", "zeff", "obbin",
    "darvel", "hosk", "immel", "jarn", "kolvet", "ubble",
    "dram", "drams", "perch", "perches", "satchel", "cork", "corked",
    "laden", "scant", "stout", "faint", "looped", "onward", "brisk",
    "peat", "ember", "embers", "clod", "clods", "rake", "flue",
    "mellow", "keen", "stoke", "grate", "crackles", "crackled",
    "searing", "tepid", "quench", "stoker",
    "mule", "plod", "post", "posts", "pannier", "heap", "halter",
    "pace", "paces", "unhitch", "drover", "lapped", "midway",
    "brimming", "hollow",
)


def make_row(
    *, prompt: str, think: str, answer: str, kind: str, surface: str, level: int, audit: dict
) -> dict:
    assert answer and "\n" not in answer
    assert audit.get("truth_valid") is True
    return {
        "messages": [{"role": "user", "content": prompt}],
        "think": think,
        "answer": f"ANSWER: {answer}",
        "kind": f"u_{kind}",
        "family": "universal",
        "surface": surface,
        "level": level,
        "n_think_tokens": max(1, len(think) // 4),
        "row_weight": 1.0,
        "_audit": audit,
    }


# ---------------------------------------------------------------------------
# reused formalisms (machinery reused verbatim from the reference cell)
# ---------------------------------------------------------------------------

def _trough_grammar(names: list[str]) -> list[tuple]:
    ops: list[tuple] = []
    for target in names:
        for amount in (1, 2, 3):
            ops.append(("ladle", amount, target))
    for source in names:
        for target in names:
            if source != target:
                ops.append(("tipover", source, target))
    for target in names:
        ops.append(("scoop", target))
    return ops


def _trough_describe(op: tuple) -> str:
    if op[0] == "ladle":
        return f"ladle {op[1]} into {op[2]}"
    if op[0] == "tipover":
        return f"tip {op[1]} into {op[2]}"
    return f"scoop half out of {op[1]}"


def _trough_apply(op: tuple, state: dict) -> dict:
    state = dict(state)
    if op[0] == "ladle":
        state[op[2]] += op[1]
    elif op[0] == "tipover":
        state[op[2]] += state[op[1]]
        state[op[1]] = 0
    else:
        state[op[1]] //= 2
    return state


def _cord_grammar(items: list[str]) -> list[tuple]:
    ops: list[tuple] = []
    for item in items:
        ops.append(("knot_tail", item))
    for item in items:
        ops.append(("knot_head", item))
    ops.extend((("shear_tail",), ("shear_head",), ("mirror",)))
    return ops


def _cord_describe(op: tuple) -> str:
    if op[0] == "knot_tail":
        return f"knot {op[1]} at the tail"
    if op[0] == "knot_head":
        return f"knot {op[1]} at the head"
    if op[0] == "shear_tail":
        return "shear off the tail trinket"
    if op[0] == "shear_head":
        return "shear off the head trinket"
    return "mirror the cord"


def _cord_apply(op: tuple, state: tuple) -> tuple:
    if op[0] == "knot_tail":
        return state + (op[1],)
    if op[0] == "knot_head":
        return (op[1],) + state
    if op[0] == "shear_tail":
        return state[:-1]
    if op[0] == "shear_head":
        return state[1:]
    return tuple(reversed(state))


def _wheel_grammar(faces: list[str]) -> list[tuple]:
    ops: list[tuple] = [("crank", 1), ("crank", 2), ("crank", 3), ("harvest",), ("rewind",)]
    del faces
    return ops


def _wheel_describe(op: tuple) -> str:
    if op[0] == "crank":
        return f"crank {op[1]} notches"
    if op[0] == "harvest":
        return "harvest the current face"
    return "rewind to the first face"


def _wheel_apply(op: tuple, state: tuple) -> tuple:
    position, hopper = state
    if op[0] == "crank":
        return ((position + op[1]) % 5, hopper)
    if op[0] == "harvest":
        return (position, hopper + position + 1)
    return (0, hopper)


def _slate_grammar(sigils: list[str]) -> list[tuple]:
    ops: list[tuple] = []
    for sigil in sigils:
        for slot in range(4):
            ops.append(("etch", sigil, slot))
    for slot in range(4):
        ops.append(("smudge", slot))
    for first in range(4):
        for second in range(first + 1, 4):
            ops.append(("trade", first, second))
    return ops


def _slate_describe(op: tuple) -> str:
    if op[0] == "etch":
        return f"etch {op[1]} into slot {op[2] + 1}"
    if op[0] == "smudge":
        return f"smudge slot {op[1] + 1}"
    return f"trade slots {op[1] + 1} and {op[2] + 1}"


def _slate_apply(op: tuple, state: tuple) -> tuple:
    slots = list(state)
    if op[0] == "etch":
        slots[op[2]] = op[1]
    elif op[0] == "smudge":
        slots[op[1]] = None
    else:
        slots[op[1]], slots[op[2]] = slots[op[2]], slots[op[1]]
    return tuple(slots)


EXTENDED_AMOUNT_BOUND = 12


def _trough_grammar_extended(names: list[str]) -> list[tuple]:
    """Legality probe: ladle amounts up to EXTENDED_AMOUNT_BOUND AND every
    op's container dimension over the FULL module trough pool."""
    del names
    pool = list(TROUGHS)
    ops: list[tuple] = []
    for target in pool:
        for amount in range(1, EXTENDED_AMOUNT_BOUND + 1):
            ops.append(("ladle", amount, target))
    for source in pool:
        for target in pool:
            if source != target:
                ops.append(("tipover", source, target))
    for target in pool:
        ops.append(("scoop", target))
    return ops


def _trough_apply_probe(op: tuple, state: dict) -> dict:
    """Tolerant probe apply for the extended container enumeration.

    Identical to the (verbatim-reused) ``_trough_apply`` on every in-instance
    container; phantom containers start empty and any op touching one leaves
    its key behind, so a phantom-container candidate can never land on the
    wanted state. The bounded enumeration never uses this function.
    """
    state = dict(state)
    if op[0] == "ladle":
        state[op[2]] = state.get(op[2], 0) + op[1]
    elif op[0] == "tipover":
        state[op[2]] = state.get(op[2], 0) + state.get(op[1], 0)
        state[op[1]] = 0
    else:
        state[op[1]] = state.get(op[1], 0) // 2
    return state


def _cord_grammar_extended(_items: list[str]) -> list[tuple]:
    """Legality probe: knots over the FULL module trinket pool."""
    return _cord_grammar(list(TRINKETS))


def _wheel_grammar_extended(_faces: list[str]) -> list[tuple]:
    """Legality probe: crank settings up to EXTENDED_AMOUNT_BOUND."""
    ops: list[tuple] = [
        ("crank", amount) for amount in range(1, EXTENDED_AMOUNT_BOUND + 1)
    ]
    ops.extend((("harvest",), ("rewind",)))
    return ops


def _slate_grammar_extended(_sigils: list[str]) -> list[tuple]:
    """Legality probe: etches over the FULL module sigil pool."""
    return _slate_grammar(list(SIGILS))


# ---------------------------------------------------------------------------
# barrowyoke (NEW: three named barrows; heave sizes bounded in the spec)
# ---------------------------------------------------------------------------

YOKE_LEGALITY_CLAUSE = (
    "exactly four sizes, 1, 2, 3, or 4 sacks, and no other heave size exists"
)


def _yoke_grammar(names: list[str]) -> list[tuple]:
    ops: list[tuple] = []
    for target in names:
        for amount in (1, 2, 3, 4):
            ops.append(("heave", amount, target))
    for first in range(len(names)):
        for second in range(first + 1, len(names)):
            ops.append(("swaploads", names[first], names[second]))
    for target in names:
        ops.append(("dumpout", target))
    return ops


def _yoke_grammar_extended(names: list[str]) -> list[tuple]:
    """Legality probe: heave sizes up to EXTENDED_AMOUNT_BOUND AND every
    op's container dimension over the FULL module barrow pool.

    Swap pairs are unordered: the in-instance pairs are emitted exactly as
    the bounded grammar writes them (the machine's own sample order, so the
    bounded grammar stays a subset), and every pair involving at least one
    phantom barrow is added once in pool order.
    """
    pool = list(BARROWS)
    instance = set(names)
    ops: list[tuple] = []
    for target in pool:
        for amount in range(1, EXTENDED_AMOUNT_BOUND + 1):
            ops.append(("heave", amount, target))
    for first in range(len(names)):
        for second in range(first + 1, len(names)):
            ops.append(("swaploads", names[first], names[second]))
    for first in range(len(pool)):
        for second in range(first + 1, len(pool)):
            left, right = pool[first], pool[second]
            if left in instance and right in instance:
                continue  # covered above in the machine's own order
            ops.append(("swaploads", left, right))
    for target in pool:
        ops.append(("dumpout", target))
    return ops


def _yoke_apply_probe(op: tuple, state: dict) -> dict:
    """Tolerant probe apply for the extended container enumeration.

    Identical to ``_yoke_apply`` on every in-instance container; phantom
    containers start empty and any op touching one leaves its key behind, so
    a phantom-container candidate can never land on the wanted state. The
    bounded enumeration never uses this function.
    """
    state = dict(state)
    if op[0] == "heave":
        state[op[2]] = state.get(op[2], 0) + op[1]
    elif op[0] == "swaploads":
        state[op[1]], state[op[2]] = state.get(op[2], 0), state.get(op[1], 0)
    else:
        state[op[1]] = 0
    return state


def _yoke_describe(op: tuple) -> str:
    if op[0] == "heave":
        return f"heave {op[1]} sacks into {op[2]}"
    if op[0] == "swaploads":
        return f"swap the loads of {op[1]} and {op[2]}"
    return f"dump every sack out of {op[1]}"


def _yoke_apply(op: tuple, state: dict) -> dict:
    state = dict(state)
    if op[0] == "heave":
        state[op[2]] += op[1]
    elif op[0] == "swaploads":
        state[op[1]], state[op[2]] = state[op[2]], state[op[1]]
    else:
        state[op[1]] = 0
    return state


# ---------------------------------------------------------------------------
# balesled (NEW: a sled of branded bales; the brand pool bounded in the spec)
# ---------------------------------------------------------------------------

def _sled_legality_clause(items: list[str]) -> str:
    return (
        "exactly four bale brands exist, "
        + ", ".join(items)
        + ", and no other brand exists"
    )


def _sled_grammar(items: list[str]) -> list[tuple]:
    ops: list[tuple] = []
    for item in items:
        ops.append(("lash", item))
    for item in items:
        ops.append(("shove", item))
    ops.extend((("uncouple",), ("walkround",)))
    return ops


def _sled_grammar_extended(_items: list[str]) -> list[tuple]:
    """Legality probe: lashes and shoves over the FULL module bale pool."""
    return _sled_grammar(list(BALES))


def _sled_describe(op: tuple) -> str:
    if op[0] == "lash":
        return f"lash a {op[1]} bale at the back"
    if op[0] == "shove":
        return f"shove a {op[1]} bale in at the front"
    if op[0] == "uncouple":
        return "uncouple the front bale"
    return "walk the front bale around to the back"


def _sled_apply(op: tuple, state: tuple) -> tuple:
    if op[0] == "lash":
        return state + (op[1],)
    if op[0] == "shove":
        return (op[1],) + state
    if op[0] == "uncouple":
        return state[1:]
    return state[1:] + state[:1]


# ---------------------------------------------------------------------------
# millround (NEW: six-vane windlass + creel; settings bounded in the spec)
# ---------------------------------------------------------------------------

ROUND_LEGALITY_CLAUSE = (
    "exactly four settings, 1, 2, 3, or 4 vanes, and no other setting exists"
)


def _round_grammar(faces: list[str]) -> list[tuple]:
    del faces
    return [("turnround", 1), ("turnround", 2), ("turnround", 3), ("turnround", 4),
            ("emptyvane",), ("unwind",)]


def _round_grammar_extended(_faces: list[str]) -> list[tuple]:
    """Legality probe: windlass settings up to EXTENDED_AMOUNT_BOUND."""
    ops: list[tuple] = [
        ("turnround", amount) for amount in range(1, EXTENDED_AMOUNT_BOUND + 1)
    ]
    ops.extend((("emptyvane",), ("unwind",)))
    return ops


def _round_describe(op: tuple) -> str:
    if op[0] == "turnround":
        return f"turn the windlass {op[1]} vanes"
    if op[0] == "emptyvane":
        return "empty the current vane into the creel"
    return "unwind to the first vane"


def _round_apply(op: tuple, state: tuple) -> tuple:
    position, creel = state
    if op[0] == "turnround":
        return ((position + op[1]) % 6, creel)
    if op[0] == "emptyvane":
        return (position, creel + position + 1)
    return (0, creel)


# ---------------------------------------------------------------------------
# skeinreel (NEW: coil count + two-way lay; reel counts bounded in the spec)
# ---------------------------------------------------------------------------

REEL_LEGALITY_CLAUSE = (
    "exactly five counts, 1, 2, 3, 4, or 5 coils, and no other count exists"
)


def _reel_grammar(lays: list[str]) -> list[tuple]:
    del lays
    return [("windon", 1), ("windon", 2), ("windon", 3), ("windon", 4), ("windon", 5),
            ("letout",), ("crosslay",)]


def _reel_grammar_extended(_lays: list[str]) -> list[tuple]:
    """Legality probe: reel counts up to EXTENDED_AMOUNT_BOUND."""
    ops: list[tuple] = [
        ("windon", amount) for amount in range(1, EXTENDED_AMOUNT_BOUND + 1)
    ]
    ops.extend((("letout",), ("crosslay",)))
    return ops


def _reel_describe(op: tuple) -> str:
    if op[0] == "windon":
        return f"wind on {op[1]} coils"
    if op[0] == "letout":
        return "let half the coils out"
    return "cross the lay"


def _reel_apply(op: tuple, state: tuple) -> tuple:
    coils, lay = state
    if op[0] == "windon":
        return (coils + op[1], lay)
    if op[0] == "letout":
        return (coils // 2, lay)
    return (coils, LAYS[1] if lay == LAYS[0] else LAYS[0])


FEEDLOOP_FORMALISMS = (
    "troughline", "trinketcord", "crankwheel", "sigilslate",
    "barrowyoke", "balesled", "millround", "skeinreel",
)
REUSED_FORMALISMS = ("troughline", "trinketcord", "crankwheel", "sigilslate")
NEW_FORMALISMS = ("barrowyoke", "balesled", "millround", "skeinreel")

# Exact per-formalism scope of the extended legality probe, recorded
# row-by-row in the audit. Numeric parameters are probed up to
# EXTENDED_AMOUNT_BOUND; item parameters over the FULL module pools; and for
# the two named-container machines the container dimension of every op is
# additionally probed over the full pool (phantom containers start empty in
# a tolerant probe apply, so an op touching one can never reproduce the
# wanted outcome). Dimensions NOT probed are named explicitly.
EXTENDED_PROBE_SCOPE = {
    "troughline": (
        "ladle amounts to 12 AND every container dimension (ladle/tip/scoop "
        "targets) over the full trough pool; phantom troughs start empty in "
        "the probe apply"
    ),
    "trinketcord": (
        "knot items over the full trinket pool; shear/mirror carry no "
        "parameter"
    ),
    "crankwheel": (
        "crank settings to 12; harvest/rewind carry no parameter; face "
        "names never parameterize an op"
    ),
    "sigilslate": (
        "etch sigils over the full sigil pool; slot indices are the slate's "
        "four physical slots (structural, not a documented-pool parameter) "
        "and are NOT probed past 4"
    ),
    "barrowyoke": (
        "heave sizes to 12 AND every container dimension (heave/swap/dump "
        "targets) over the full barrow pool; phantom barrows start empty in "
        "the probe apply"
    ),
    "balesled": (
        "lash/shove brands over the full bale pool; uncouple/walk carry no "
        "parameter"
    ),
    "millround": (
        "windlass settings to 12; empty/unwind carry no parameter; vane "
        "names never parameterize an op"
    ),
    "skeinreel": "reel counts to 12; letout/crosslay carry no parameter",
}


def _feedloop_machine(rng: random.Random, formalism: str) -> dict:
    """Build one concrete machine: grammar, semantics, scene, two starts."""
    if formalism == "troughline":
        names = rng.sample(TROUGHS, 3)
        start_a = {name: rng.randint(1, 9) for name in names}
        start_b = {name: rng.randint(1, 9) for name in names}
        return {
            "grammar": _trough_grammar(names),
            "describe": _trough_describe,
            "apply": _trough_apply,
            "render": lambda s: ", ".join(f"{n}={s[n]}" for n in names),
            "rules": (
                "How steps act: 'ladle n into X' adds n pails to trough X — the "
                "ladle comes in exactly three sizes, 1, 2, or 3 pails, and no "
                "other ladle size exists; 'tip X into Y' moves every pail from "
                "X into Y (X ends at 0); 'scoop half out of X' halves X, "
                "rounding down."
            ),
            "legality_clauses": [
                "exactly three sizes, 1, 2, or 3 pails, and no other ladle size exists",
            ],
            "extended_grammar": _trough_grammar_extended(names),
            "extended_apply": _trough_apply_probe,
            "scene": "A row of three water troughs runs its steps strictly in order.",
            "vocabulary": list(names),
            "start_a": start_a,
            "start_b": start_b,
        }
    if formalism == "trinketcord":
        items = rng.sample(TRINKETS, 4)
        start_a = tuple(rng.choice(items) for _ in range(rng.randint(3, 4)))
        start_b = tuple(rng.choice(items) for _ in range(rng.randint(3, 4)))
        return {
            "grammar": _cord_grammar(items),
            "describe": _cord_describe,
            "apply": _cord_apply,
            "render": lambda s: " ".join(s) if s else "(no trinkets)",
            "rules": (
                "How steps act: 'knot t at the tail/head' adds trinket t at that "
                "end — exactly four trinket patterns exist, "
                + ", ".join(items)
                + ", and no other pattern exists; 'shear off the tail/head "
                "trinket' removes that end's trinket; 'mirror the cord' "
                "reverses the whole cord. The cord is always read head first."
            ),
            "legality_clauses": [
                "exactly four trinket patterns exist, "
                + ", ".join(items)
                + ", and no other pattern exists",
            ],
            "extended_grammar": _cord_grammar_extended(items),
            "scene": "A cord of trinkets runs its steps strictly in order.",
            "vocabulary": list(items),
            "start_a": start_a,
            "start_b": start_b,
        }
    if formalism == "crankwheel":
        faces = rng.sample(WHEELFACES, 5)
        start_a = (rng.randrange(5), rng.randint(0, 3))
        start_b = (rng.randrange(5), rng.randint(0, 3))
        return {
            "grammar": _wheel_grammar(faces),
            "describe": _wheel_describe,
            "apply": _wheel_apply,
            "render": lambda s: f"needle on {faces[s[0]]}, hopper {s[1]}",
            "rules": (
                "The wheel's faces sit in ring order "
                + " ".join(faces)
                + " (after the last face the needle comes back to the first). "
                "How steps act: 'crank n notches' moves the needle n faces "
                "forward — the crank has exactly three settings, 1, 2, or 3 "
                "notches, and no other setting exists; 'harvest the current "
                "face' adds the needle face's ring position (1 for the first "
                "listed face, 5 for the last) to the hopper; 'rewind to the "
                "first face' puts the needle on the first listed face."
            ),
            "legality_clauses": [
                "exactly three settings, 1, 2, or 3 notches, and no other setting exists",
            ],
            "extended_grammar": _wheel_grammar_extended(faces),
            "scene": "A crank wheel with a grain hopper runs its steps strictly in order.",
            "vocabulary": list(faces),
            "start_a": start_a,
            "start_b": start_b,
        }
    if formalism == "sigilslate":
        sigils = rng.sample(SIGILS, 4)
        start_a = tuple(rng.choice((*sigils, None)) for _ in range(4))
        start_b = tuple(rng.choice((*sigils, None)) for _ in range(4))
        return {
            "grammar": _slate_grammar(sigils),
            "describe": _slate_describe,
            "apply": _slate_apply,
            "render": lambda s: " ".join(
                f"{index + 1}:{value if value is not None else '~'}"
                for index, value in enumerate(s)
            ),
            "rules": (
                "How steps act: 'etch s into slot i' writes sigil s there, over "
                "whatever was there — exactly four sigils exist, "
                + ", ".join(sigils)
                + ", and no other sigil exists; 'smudge slot i' wipes it to ~; "
                "'trade slots i and j' swaps their contents."
            ),
            "legality_clauses": [
                "exactly four sigils exist, "
                + ", ".join(sigils)
                + ", and no other sigil exists",
            ],
            "extended_grammar": _slate_grammar_extended(sigils),
            "scene": "A four-slot sigil slate runs its steps strictly in order.",
            "vocabulary": list(sigils),
            "start_a": start_a,
            "start_b": start_b,
        }
    if formalism == "barrowyoke":
        names = rng.sample(BARROWS, 3)
        start_a = {name: rng.randint(0, 9) for name in names}
        start_b = {name: rng.randint(0, 9) for name in names}
        return {
            "grammar": _yoke_grammar(names),
            "describe": _yoke_describe,
            "apply": _yoke_apply,
            "render": lambda s: ", ".join(f"{n}={s[n]}" for n in names),
            "rules": (
                "How steps act: 'heave n sacks into X' adds n sacks to barrow "
                f"X — the heave comes in {YOKE_LEGALITY_CLAUSE}; 'swap the "
                "loads of X and Y' trades the two barrows' sack counts; 'dump "
                "every sack out of X' leaves barrow X at 0."
            ),
            "legality_clauses": [YOKE_LEGALITY_CLAUSE],
            "extended_grammar": _yoke_grammar_extended(names),
            "extended_apply": _yoke_apply_probe,
            "scene": "A yoke of three barrows runs its steps strictly in order.",
            "vocabulary": list(names),
            "start_a": start_a,
            "start_b": start_b,
        }
    if formalism == "balesled":
        items = rng.sample(BALES, 4)
        start_a = tuple(rng.choice(items) for _ in range(rng.randint(3, 4)))
        start_b = tuple(rng.choice(items) for _ in range(rng.randint(3, 4)))
        return {
            "grammar": _sled_grammar(items),
            "describe": _sled_describe,
            "apply": _sled_apply,
            "render": lambda s: " ".join(s) if s else "(no bales)",
            "rules": (
                "How steps act: 'lash a b bale at the back' adds a bale of "
                "brand b at the back — "
                + _sled_legality_clause(items)
                + "; 'shove a b bale in at the front' adds it at the front; "
                "'uncouple the front bale' removes the front bale; 'walk the "
                "front bale around to the back' moves the front bale to the "
                "back. The sled is always read front first."
            ),
            "legality_clauses": [_sled_legality_clause(items)],
            "extended_grammar": _sled_grammar_extended(items),
            "scene": "A sled of branded bales runs its steps strictly in order.",
            "vocabulary": list(items),
            "start_a": start_a,
            "start_b": start_b,
        }
    if formalism == "millround":
        vanes = list(rng.sample(VANES, 6))
        start_a = (rng.randrange(6), rng.randint(0, 3))
        start_b = (rng.randrange(6), rng.randint(0, 3))
        return {
            "grammar": _round_grammar(vanes),
            "describe": _round_describe,
            "apply": _round_apply,
            "render": lambda s: f"shoe on {vanes[s[0]]}, creel {s[1]}",
            "rules": (
                "The windlass vanes sit in ring order "
                + " ".join(vanes)
                + " (after the last vane the shoe comes back to the first). "
                "How steps act: 'turn the windlass n vanes' moves the shoe n "
                f"vanes forward — the windlass has {ROUND_LEGALITY_CLAUSE}; "
                "'empty the current vane into the creel' adds the shoe vane's "
                "ring position (1 for the first listed vane, 6 for the last) "
                "to the creel; 'unwind to the first vane' puts the shoe on "
                "the first listed vane."
            ),
            "legality_clauses": [ROUND_LEGALITY_CLAUSE],
            "extended_grammar": _round_grammar_extended(vanes),
            "scene": "A six-vane windlass with a creel runs its steps strictly in order.",
            "vocabulary": list(vanes),
            "start_a": start_a,
            "start_b": start_b,
        }
    if formalism == "skeinreel":
        start_a = (rng.randint(0, 9), rng.choice(LAYS))
        start_b = (rng.randint(0, 9), rng.choice(LAYS))
        return {
            "grammar": _reel_grammar(list(LAYS)),
            "describe": _reel_describe,
            "apply": _reel_apply,
            "render": lambda s: f"{s[0]} coils, lay {s[1]}",
            "rules": (
                "How steps act: 'wind on n coils' adds n coils to the reel — "
                f"the reel winds {REEL_LEGALITY_CLAUSE}; 'let half the coils "
                "out' halves the coil count, rounding down; 'cross the lay' "
                f"trades the lay between {LAYS[0]} and {LAYS[1]} and leaves "
                "the coil count alone."
            ),
            "legality_clauses": [REEL_LEGALITY_CLAUSE],
            "extended_grammar": _reel_grammar_extended(list(LAYS)),
            "scene": "A winding reel with a two-way lay runs its steps strictly in order.",
            "vocabulary": list(LAYS),
            "start_a": start_a,
            "start_b": start_b,
        }
    raise ValueError(f"unknown feedloop formalism: {formalism}")


def _run_steps(apply_fn, steps: list[tuple], start):
    current = start
    for op in steps:
        current = apply_fn(op, current)
    return current


def _consistent_fixes(
    apply_fn, grammar: list[tuple], written: list[tuple], start, wanted
) -> list[tuple[int, tuple]]:
    """Every single-step substitution under which the run lands on ``wanted``."""
    fixes: list[tuple[int, tuple]] = []
    for index in range(len(written)):
        for candidate in grammar:
            if candidate == written[index]:
                continue
            patched = written[:index] + [candidate] + written[index + 1 :]
            if _run_steps(apply_fn, patched, start) == wanted:
                fixes.append((index, candidate))
    return fixes


def feedloop_lesson(rng: random.Random, formalism: str) -> dict:
    for _ in range(5000):
        machine = _feedloop_machine(rng, formalism)
        grammar = machine["grammar"]
        describe = machine["describe"]
        apply_fn = machine["apply"]
        render = machine["render"]
        start_a, start_b = machine["start_a"], machine["start_b"]
        if start_a == start_b:
            continue
        length = rng.randint(4, 5)
        intended = [rng.choice(grammar) for _ in range(length)]
        bug_at = rng.randrange(length)
        wrong_op = rng.choice([op for op in grammar if op != intended[bug_at]])
        written = list(intended)
        written[bug_at] = wrong_op

        wanted_a = _run_steps(apply_fn, intended, start_a)
        finished_a = _run_steps(apply_fn, written, start_a)
        if wanted_a == finished_a:
            continue
        fixes_round1 = _consistent_fixes(apply_fn, grammar, written, start_a, wanted_a)
        true_fix = (bug_at, intended[bug_at])
        if true_fix not in fixes_round1 or len(fixes_round1) < 2:
            continue
        wanted_b = _run_steps(apply_fn, intended, start_b)
        fixes_round2 = [
            (index, candidate)
            for index, candidate in fixes_round1
            if _run_steps(
                apply_fn, written[:index] + [candidate] + written[index + 1 :], start_b
            )
            == wanted_b
        ]
        if fixes_round2 != [true_fix]:
            continue
        attempt_index, attempt_op = rng.choice(
            [fix for fix in fixes_round1 if fix != true_fix]
        )
        attempted = written[:attempt_index] + [attempt_op] + written[attempt_index + 1 :]
        finished_b = _run_steps(apply_fn, attempted, start_b)
        if finished_b == wanted_b:
            raise RuntimeError("round-two uniqueness bookkeeping is inconsistent")

        # Legality audit: the documented spec must bound every parameterized
        # operation, and re-enumerating fixes over the EXTENDED grammar (see
        # EXTENDED_PROBE_SCOPE for the exact per-formalism probe dimensions)
        # must surface no within-bound alternative — every extra round-1+2
        # survivor must be excluded by the documented legality clause alone
        # (i.e. sit outside the bounded grammar). The named-container
        # machines probe with a tolerant apply in which phantom containers
        # start empty; it agrees with the bounded apply on every bounded op.
        legality_clauses = machine["legality_clauses"]
        if not legality_clauses or any(
            clause not in machine["rules"] for clause in legality_clauses
        ):
            raise RuntimeError("a parameterized operation is not bounded in the spec")
        extended_grammar = machine["extended_grammar"]
        if any(op not in extended_grammar for op in grammar):
            raise RuntimeError("extended legality grammar does not cover the bounded grammar")
        extended_apply = machine.get("extended_apply", apply_fn)
        if any(
            extended_apply(op, start_a) != apply_fn(op, start_a) for op in grammar
        ):
            raise RuntimeError("probe apply disagrees with the bounded apply on a bounded op")
        bounded_ops = set(grammar)
        extended_round2 = [
            (index, candidate)
            for index, candidate in _consistent_fixes(
                extended_apply, extended_grammar, written, start_a, wanted_a
            )
            if _run_steps(
                extended_apply,
                written[:index] + [candidate] + written[index + 1 :],
                start_b,
            )
            == wanted_b
        ]
        out_of_bound_alternatives = [
            fix for fix in extended_round2 if fix != true_fix
        ]
        if true_fix not in extended_round2 or any(
            candidate in bounded_ops for _, candidate in out_of_bound_alternatives
        ):
            raise RuntimeError("bounded round-two uniqueness failed under the extended grammar")

        listing = "\n".join(
            f"  {index + 1}. {describe(op)}" for index, op in enumerate(written)
        )
        prompt = (
            f"{machine['scene']}\n{machine['rules']}\n"
            f"Steps as written:\n{listing}\n"
            f"First trial, starting from {render(start_a)}: the crew wanted "
            f"{render(wanted_a)}, but the run finished at {render(finished_a)}.\n"
            f"Exactly one written step is off. An earlier attempt changed step "
            f"{attempt_index + 1} to '{describe(attempt_op)}'. With that change, a "
            f"second trial ran the steps from {render(start_b)}: the crew wanted "
            f"{render(wanted_b)}, but it finished at {render(finished_b)}.\n"
            "Using the evidence from both trials, name the one change that makes "
            "both trials come out as wanted.\n"
            f"{ANSWER_LINE} (format: STEP <k>: <corrected step>)"
        )
        candidate_text = "; ".join(
            f"step {index + 1} -> '{describe(candidate)}'"
            for index, candidate in fixes_round1
        )
        reasoning = [
            f"More than one legal single-step change squares with the first trial "
            f"alone: {candidate_text}.",
            f"The earlier attempt picked step {attempt_index + 1} -> "
            f"'{describe(attempt_op)}', and the second trial is what rules it out: "
            f"from {render(start_b)} that change finishes at {render(finished_b)}, "
            f"not the wanted {render(wanted_b)}.",
            f"Of the legal steps, only step {bug_at + 1} -> "
            f"'{describe(intended[bug_at])}' lands on the wanted outcome in the "
            "first trial and the second trial at once.",
        ]
        answer = f"STEP {bug_at + 1}: {describe(intended[bug_at])}"
        return make_row(
            prompt=prompt,
            think=" ".join(reasoning),
            answer=answer,
            kind="feedloop",
            surface=formalism,
            level=length,
            audit={
                "truth_valid": True,
                "formalism": formalism,
                "steps": length,
                "bug_step": bug_at + 1,
                "candidates_after_round1": len(fixes_round1),
                "unique_after_round2": True,
                "wrong_in_round1": True,
                "wrong_attempt_step": attempt_index + 1,
                "wrong_attempt": describe(attempt_op),
                "extended_uniqueness_audit": {
                    "amounts_probed_to": EXTENDED_AMOUNT_BOUND,
                    "probe_scope": EXTENDED_PROBE_SCOPE[formalism],
                    "container_names_probed_over_full_pool": (
                        formalism in ("troughline", "barrowyoke")
                    ),
                    "round2_survivors_extended": len(extended_round2),
                    "out_of_bound_alternatives": len(out_of_bound_alternatives),
                    "out_of_bound_only": True,
                    "legality_clauses_documented": True,
                },
                "legality_clauses": list(legality_clauses),
                "spec": {
                    "vocabulary": machine["vocabulary"],
                    "written": [list(op) for op in written],
                    "true_fix": [bug_at, list(intended[bug_at])],
                    "wrong_fix": [attempt_index, list(attempt_op)],
                    "start_a": start_a,
                    "start_b": start_b,
                    "wanted_a": wanted_a,
                    "wanted_b": wanted_b,
                    "finished_a": finished_a,
                    "finished_b_after_wrong": finished_b,
                    "grammar_size": len(grammar),
                },
            },
        )
    raise RuntimeError(f"could not construct a {formalism} feedloop lesson")


# ---------------------------------------------------------------------------
# corpus assembly
# ---------------------------------------------------------------------------

def _feedloop_at(rng: random.Random, index: int) -> dict:
    return feedloop_lesson(rng, FEEDLOOP_FORMALISMS[index % len(FEEDLOOP_FORMALISMS)])


SKILLS = {
    "feedloop": _feedloop_at,
}
ARM_MIX = "feedloop=800"
HOLDOUT_MIX = "feedloop=40"
SMOKE_MIX = "feedloop=16"


def parse_mix(specification: str) -> list[tuple[str, int]]:
    result: list[tuple[str, int]] = []
    seen: set[str] = set()
    for part in specification.split(","):
        if not part.strip():
            continue
        name, separator, raw_count = part.partition("=")
        name = name.strip()
        if not separator or name not in SKILLS or name in seen:
            raise ValueError(f"invalid mix entry {part!r}")
        count = int(raw_count)
        if count <= 0:
            raise ValueError(f"count must be positive: {part}")
        result.append((name, count))
        seen.add(name)
    if not result:
        raise ValueError("empty mix")
    return result


def public_row(row: dict) -> dict:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def generate_curriculum(specification: str, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    for skill, count in parse_mix(specification):
        for index in range(count):
            row = SKILLS[skill](rng, index)
            row["task_id"] = f"mds_{skill}_{index:05d}"
            rows.append(row)
    rng.shuffle(rows)
    return rows


def validate_generated(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("curriculum is empty")
    required = {
        "messages", "think", "answer", "kind", "family", "surface", "level",
        "n_think_tokens", "row_weight", "task_id", "_audit",
    }
    serialized: set[str] = set()
    prompts: set[str] = set()
    task_ids: set[str] = set()
    for index, row in enumerate(rows):
        if set(row) != required:
            raise ValueError(f"row {index} schema mismatch: {sorted(row)}")
        if row["family"] != "universal" or row["row_weight"] != 1.0:
            raise ValueError(f"row {index} family/weight mismatch")
        if len(row["messages"]) != 1 or row["messages"][0].get("role") != "user":
            raise ValueError(f"row {index} message schema mismatch")
        if not row["think"].strip() or "\n" in row["think"]:
            raise ValueError(f"row {index} malformed think target")
        if not row["answer"].startswith("ANSWER: ") or "\n" in row["answer"]:
            raise ValueError(f"row {index} malformed answer")
        if row["kind"] != "u_feedloop":
            raise ValueError(f"row {index} has an unknown kind: {row['kind']}")
        if row["surface"] not in FEEDLOOP_FORMALISMS:
            raise ValueError(f"row {index} has an unknown surface: {row['surface']}")
        audit = row["_audit"]
        if audit.get("truth_valid") is not True:
            raise ValueError(f"row {index} lacks truth audit")
        if audit.get("formalism") != row["surface"]:
            raise ValueError(f"row {index} formalism/surface mismatch")
        extended = audit.get("extended_uniqueness_audit") or {}
        prompt_text = row["messages"][0]["content"]
        if (
            audit.get("unique_after_round2") is not True
            or not isinstance(audit.get("candidates_after_round1"), int)
            or audit["candidates_after_round1"] < 2
            or audit.get("wrong_in_round1") is not True
            or extended.get("out_of_bound_only") is not True
            or extended.get("legality_clauses_documented") is not True
            or extended.get("amounts_probed_to") != EXTENDED_AMOUNT_BOUND
            or extended.get("probe_scope") != EXTENDED_PROBE_SCOPE.get(row["surface"])
            or extended.get("container_names_probed_over_full_pool")
            is not (row["surface"] in ("troughline", "barrowyoke"))
            or not isinstance(extended.get("round2_survivors_extended"), int)
            or extended["round2_survivors_extended"] < 1
            or not audit.get("legality_clauses")
            or any(
                clause not in prompt_text
                for clause in audit["legality_clauses"]
            )
        ):
            raise ValueError(f"row {index} feedloop evidence audit failed")
        canonical = json.dumps(public_row(row), sort_keys=True, ensure_ascii=False)
        prompt = row["messages"][0]["content"]
        if canonical in serialized or prompt in prompts or row["task_id"] in task_ids:
            raise ValueError(f"row {index} duplicate")
        serialized.add(canonical)
        prompts.add(prompt)
        task_ids.add(row["task_id"])
    return {
        "rows": len(rows),
        "kinds": dict(sorted(Counter(row["kind"] for row in rows).items())),
        "surfaces": dict(sorted(Counter(row["surface"] for row in rows).items())),
        "max_estimated_think_tokens": max(row["n_think_tokens"] for row in rows),
    }


def check_banned_vocabulary(rows: list[dict]) -> None:
    patterns = [
        (re.compile(rf"\b{re.escape(token)}\b"), token) for token in BANNED_PROMPT_TOKENS
    ]
    for index, row in enumerate(rows):
        haystack = "\n".join((row["messages"][0]["content"], row["think"], row["answer"]))
        for pattern, token in patterns:
            if pattern.search(haystack):
                raise ValueError(f"row {index} leaks banned vocabulary: {token!r}")


def check_corpus_balance(rows: list[dict]) -> dict:
    """Corpus-level formalism balance for the frozen 800-row arm."""
    feedloop = [row for row in rows if row["kind"] == "u_feedloop"]
    feedloop_formalisms = Counter(row["_audit"]["formalism"] for row in feedloop)
    ambiguity = Counter(
        row["_audit"]["candidates_after_round1"] for row in feedloop
    )
    out_of_bound = Counter(
        row["_audit"]["extended_uniqueness_audit"]["out_of_bound_alternatives"]
        for row in feedloop
    )
    balance = {
        "feedloop_formalisms": dict(sorted(feedloop_formalisms.items())),
        "feedloop_round1_candidate_counts": {
            str(key): value for key, value in sorted(ambiguity.items())
        },
        "feedloop_out_of_bound_alternative_counts": {
            str(key): value for key, value in sorted(out_of_bound.items())
        },
        "feedloop_rows_with_out_of_bound_alternatives": sum(
            count for key, count in out_of_bound.items() if key > 0
        ),
    }
    if feedloop and (
        set(feedloop_formalisms) != set(FEEDLOOP_FORMALISMS)
        or len(set(feedloop_formalisms.values())) != 1
    ):
        raise ValueError(f"feedloop formalism balance out of range: {balance}")
    return balance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mix", default=None)
    parser.add_argument("--seed", type=int, default=77150)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if sum(bool(value) for value in (args.mix, args.smoke)) != 1:
        parser.error("choose exactly one of --mix, --smoke")
    mix = SMOKE_MIX if args.smoke else args.mix
    rows = generate_curriculum(mix, args.seed)
    summary = validate_generated(rows)
    check_banned_vocabulary(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(public_row(row), ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({"mix": parse_mix(mix), **summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
