#!/usr/bin/env python3
"""Designed statechain-only curriculum: four procedure-machine formalisms.

Lifecycle 15 split the episode verdict: the narrated hidden-state chain
lesson (``u_statechain``) INSTALLED (11/20, strict over both controls) while
the feedback-loop lesson died at 0/20. This cell doses ONLY the proven
statechain skill. Every row is one ``u_statechain`` lesson rendered as plain
text inside ONE user message, on an invented procedure machine with hidden
state that only lossy per-step readouts ever surface:

- ``brewvat`` and ``courierloft``: the reference cell's two installed
  formalisms, machinery reused verbatim, FRESH instances at this cell's own
  construction seed (zero row overlap with the reference corpus is enforced
  by the corpus builder's overlap receipt).
- ``peatstove`` and ``muletrack``: two NEW formalisms in the same spirit with
  fresh invented vocabulary. Per the reference cell's post-review contract,
  every parameterized operation is explicitly BOUNDED in the rendered spec
  text (rake sizes 1-5 clods; halter paces 1-4 posts), the bounding clause is
  verified verbatim in the prompt, and an extended probe (amounts up to 12)
  verifies that no rendered step ever uses a parameter outside the
  documented bound.

Every machine requires at least three hidden-state updates; the generator
simulates the machine and verifies that a stateless reader and a
last-step-only reader both produce a different answer. Think targets narrate
the state chain compactly, step by step, matching the visible readouts.
Answers are computed by executing the specification; a banned-vocabulary
scan rejects any leak of benchmark family names, gym flavor nouns, prior
surface pools (INCLUDING the retired feedloop formalisms' noun pools), or
the public blocker-family description nouns.
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

# Retained surfaces inherited from the reference cell's statechain lessons.
# These are deliberately NOT fresh: they appear in the reference corpus and
# are excluded from this cell's fresh-surface audit (documented inheritance).
PERCHES = ("orvan", "welk", "xemb", "yurr", "zeff", "obbin")
INHERITED_SURFACE_TOKENS = (
    "brewvat", "courierloft",
    *PERCHES,
    "dram", "drams", "perch", "perches", "satchel", "cork", "corked",
    "laden", "scant", "stout", "faint", "looped", "onward", "brisk",
)

# Fresh invented vocabulary for the two NEW formalisms. Every token here is
# grep-verified against every predecessor corpus, stream, and frozen gate by
# build_corpus.py (zero case-insensitive word-boundary hits allowed).
MULEPOSTS = ("darvel", "hosk", "immel", "jarn", "kolvet", "ubble")
FRESH_SURFACE_TOKENS = (
    # machine surface names
    "peatstove", "muletrack",
    # item pool
    *MULEPOSTS,
    # distinctive machine nouns and readout words
    "peat", "ember", "embers", "clod", "clods", "rake", "flue",
    "mellow", "keen", "stoke", "grate", "crackles", "crackled",
    "searing", "tepid", "quench", "stoker",
    "mule", "plod", "post", "posts", "pannier", "heap", "halter",
    "pace", "paces", "unhitch", "drover", "lapped", "midway",
    "brimming", "hollow",
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
    # the reference cell's retired FEEDLOOP formalisms: machine names,
    # item-noun pools, and distinctive nouns. The retained statechain
    # surfaces (brewvat/courierloft and their pools) are deliberately NOT
    # banned — they are this dose's inherited vocabulary.
    "troughline", "trinketcord", "crankwheel", "sigilslate",
    "brann", "clave", "dorst", "grelm", "murd", "prell",
    "plome", "quenn", "trell", "vosk", "yorv", "zilk",
    "drasp", "fyne", "golve", "hilm", "juss", "krev",
    "morv", "nilt", "pryn", "quav", "rulf", "senn",
    "trough", "troughs", "trinket", "trinkets", "sigil", "sigils",
)

# Extended probe bound for the parameterized-operation legality audit of the
# two NEW formalisms (mirrors the reference cell's EXTENDED_AMOUNT_BOUND).
EXTENDED_AMOUNT_BOUND = 12
NEW_FORMALISMS = ("peatstove", "muletrack")
STATECHAIN_FORMALISMS = ("brewvat", "courierloft", "peatstove", "muletrack")


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


def audit_bounded_parameters(
    steps: list[tuple], parameterized_op: str, low: int, high: int
) -> dict:
    """Legality audit for a NEW formalism's parameterized operation.

    The documented spec bounds the parameter to [low, high]; probing the
    extended domain (up to EXTENDED_AMOUNT_BOUND) must surface zero rendered
    steps whose parameter escapes the documented bound.
    """
    parameters = [step[1] for step in steps if step[0] == parameterized_op]
    out_of_bound = [
        value
        for value in parameters
        if not (low <= value <= high) and value <= EXTENDED_AMOUNT_BOUND
    ]
    beyond_probe = [value for value in parameters if value > EXTENDED_AMOUNT_BOUND]
    if out_of_bound or beyond_probe:
        raise RuntimeError(
            f"a rendered {parameterized_op} step escaped the documented bound"
        )
    return {
        "documented_bounds": {parameterized_op: [low, high]},
        "amounts_probed_to": EXTENDED_AMOUNT_BOUND,
        "parameterized_steps": len(parameters),
        "out_of_bound_steps": 0,
        "all_parameters_documented_bounded": True,
    }


# ---------------------------------------------------------------------------
# brewvat (retained; machinery reused verbatim from the reference cell)
# ---------------------------------------------------------------------------

def _brewvat_readout(strength: int, threshold: int) -> str:
    return "STOUT" if strength >= threshold else "FAINT"


def _brewvat_output(state: tuple[int, str]) -> str:
    strength, scent = state
    return f"{scent.upper()}-{strength}"


def _brewvat_apply(step: tuple, state: tuple[int, str]) -> tuple[int, str]:
    strength, scent = state
    if step[0] == "pour":
        return (strength + step[1], scent)
    return (strength, "brisk" if scent == "mild" else "mild")


def _brewvat_describe(step: tuple) -> str:
    if step[0] == "pour":
        return f"tip in {step[1]} drams"
    return "swirl the vat"


def brewvat_lesson(rng: random.Random) -> dict:
    for _ in range(5000):
        threshold = rng.randint(8, 14)
        count = rng.randint(5, 7)
        steps: list[tuple] = []
        for _ in range(count):
            if rng.random() < 0.7:
                steps.append(("pour", rng.randint(1, 5)))
            else:
                steps.append(("swirl",))
        pours = sum(step[0] == "pour" for step in steps)
        swirls = count - pours
        if pours < 3 or swirls < 1:
            continue
        initial = (0, "mild")
        transcript: list[tuple[tuple, str]] = []
        state = initial
        for step in steps:
            state = _brewvat_apply(step, state)
            transcript.append((step, _brewvat_readout(state[0], threshold)))
        answer = _brewvat_output(state)
        distractor_stateless = _brewvat_output(initial)
        distractor_lastonly = _brewvat_output(_brewvat_apply(steps[-1], initial))
        if answer in (distractor_stateless, distractor_lastonly):
            continue
        step_lines = "\n".join(
            f"  {index + 1}. {_brewvat_describe(step)} -> the vat murmured {readout}"
            for index, (step, readout) in enumerate(transcript)
        )
        prompt = (
            "A shut brewing vat keeps two covered values: a strength count "
            "(starts at 0) and a scent (starts mild).\n"
            "Documented workings: 'tip in n drams' raises the strength by n; "
            "'swirl the vat' trades the scent between mild and brisk and leaves "
            "the strength alone. After every step the vat murmurs STOUT when the "
            f"strength is at least {threshold}, otherwise FAINT — the murmur is "
            "all it ever shows.\n"
            "When told 'cork the vat', the keeper must chalk up "
            "<SCENT>-<strength>, with the scent word in capitals.\n"
            f"Steps already done:\n{step_lines}\n"
            "Next order: cork the vat. What must the keeper chalk up?\n"
            f"{ANSWER_LINE}"
        )
        reasoning = [
            f"I follow the covered values step by step; start: strength 0, scent "
            f"mild, murmur threshold {threshold}."
        ]
        state = initial
        for index, (step, readout) in enumerate(transcript, 1):
            state = _brewvat_apply(step, state)
            reasoning.append(
                f"Step {index} ({_brewvat_describe(step)}): strength {state[0]}, "
                f"scent {state[1]} — murmurs {readout}, which matches."
            )
        reasoning.append(f"Corking chalks up {answer}.")
        return make_row(
            prompt=prompt,
            think=" ".join(reasoning),
            answer=answer,
            kind="statechain",
            surface="brewvat",
            level=count,
            audit={
                "truth_valid": True,
                "formalism": "brewvat",
                "hidden_updates": count,
                "distractor_stateless": distractor_stateless,
                "distractor_lastonly": distractor_lastonly,
                "distractors_differ": True,
                "spec": {
                    "threshold": threshold,
                    "steps": [list(step) for step in steps],
                    "readouts": [readout for _, readout in transcript],
                    "final": list(state),
                },
            },
        )
    raise RuntimeError("could not construct a brewvat statechain lesson")


# ---------------------------------------------------------------------------
# courierloft (retained; machinery reused verbatim from the reference cell)
# ---------------------------------------------------------------------------

def _loft_output(state: tuple[int, int], perches: list[str]) -> str:
    position, satchel = state
    return f"{perches[position]}-{satchel}"


def _loft_apply(step: tuple, state: tuple[int, int]) -> tuple[int, int]:
    position, satchel = state
    if step[0] == "flit":
        return ((position + step[1]) % 5, satchel)
    return (position, satchel + position + 1)


def _loft_readout(step: tuple, before: tuple[int, int], threshold: int) -> str:
    if step[0] == "flit":
        return "LOOPED" if before[0] + step[1] >= 5 else "ONWARD"
    after = _loft_apply(step, before)
    return "LADEN" if after[1] >= threshold else "SCANT"


def _loft_describe(step: tuple) -> str:
    if step[0] == "flit":
        return f"flit {step[1]} perches ahead"
    return "stash at the current perch"


def courierloft_lesson(rng: random.Random) -> dict:
    for _ in range(5000):
        perches = rng.sample(PERCHES, 5)
        threshold = rng.randint(5, 10)
        count = rng.randint(5, 7)
        steps: list[tuple] = []
        for _ in range(count):
            if rng.random() < 0.6:
                steps.append(("flit", rng.randint(1, 4)))
            else:
                steps.append(("stash",))
        flits = sum(step[0] == "flit" for step in steps)
        stashes = count - flits
        if flits < 2 or stashes < 2:
            continue
        initial = (0, 0)
        transcript: list[tuple[tuple, str]] = []
        state = initial
        for step in steps:
            readout = _loft_readout(step, state, threshold)
            state = _loft_apply(step, state)
            transcript.append((step, readout))
        answer = _loft_output(state, perches)
        distractor_stateless = _loft_output(initial, perches)
        distractor_lastonly = _loft_output(_loft_apply(steps[-1], initial), perches)
        if answer in (distractor_stateless, distractor_lastonly):
            continue
        step_lines = "\n".join(
            f"  {index + 1}. {_loft_describe(step)} -> the loft chimed {readout}"
            for index, (step, readout) in enumerate(transcript)
        )
        prompt = (
            "A courier loft keeps two covered values: which perch the courier "
            "sits on and a satchel count (starts at 0). The perches sit in ring "
            "order " + " ".join(perches) + " (after the last perch the courier "
            "comes back to the first), and the courier starts on the first "
            "listed perch.\n"
            "Documented workings: 'flit n perches ahead' moves the courier n "
            "perches around the ring — the loft chimes LOOPED when the move "
            "passes the end of the ring, otherwise ONWARD; 'stash at the "
            "current perch' adds the perch's ring position (1 for the first "
            "listed perch, 5 for the last) to the satchel — the loft chimes "
            f"LADEN when the satchel is at least {threshold}, otherwise SCANT. "
            "The chimes are all it ever shows.\n"
            "When told 'deliver', the courier must call out <perch>-<satchel>.\n"
            f"Steps already done:\n{step_lines}\n"
            "Next order: deliver. What must the courier call out?\n"
            f"{ANSWER_LINE}"
        )
        reasoning = [
            "I follow the covered values step by step; start: courier on "
            f"{perches[0]} (ring position 1), satchel 0, chime threshold {threshold}."
        ]
        state = initial
        for index, (step, readout) in enumerate(transcript, 1):
            state = _loft_apply(step, state)
            reasoning.append(
                f"Step {index} ({_loft_describe(step)}): courier on "
                f"{perches[state[0]]} (ring position {state[0] + 1}), satchel "
                f"{state[1]} — chimes {readout}, which matches."
            )
        reasoning.append(f"Delivering calls out {answer}.")
        return make_row(
            prompt=prompt,
            think=" ".join(reasoning),
            answer=answer,
            kind="statechain",
            surface="courierloft",
            level=count,
            audit={
                "truth_valid": True,
                "formalism": "courierloft",
                "hidden_updates": count,
                "distractor_stateless": distractor_stateless,
                "distractor_lastonly": distractor_lastonly,
                "distractors_differ": True,
                "spec": {
                    "perches": perches,
                    "threshold": threshold,
                    "steps": [list(step) for step in steps],
                    "readouts": [readout for _, readout in transcript],
                    "final": list(state),
                },
            },
        )
    raise RuntimeError("could not construct a courierloft statechain lesson")


# ---------------------------------------------------------------------------
# peatstove (NEW: ember count + flue toggle; rake sizes bounded in the spec)
# ---------------------------------------------------------------------------

STOVE_RAKE_LOW = 1
STOVE_RAKE_HIGH = 5
STOVE_LEGALITY_CLAUSE = (
    "exactly five sizes, 1, 2, 3, 4, or 5 clods, and no other rake size exists"
)


def _stove_readout(embers: int, threshold: int) -> str:
    return "SEARING" if embers >= threshold else "TEPID"


def _stove_output(state: tuple[int, str]) -> str:
    embers, flue = state
    return f"{flue.upper()}-{embers}"


def _stove_apply(step: tuple, state: tuple[int, str]) -> tuple[int, str]:
    embers, flue = state
    if step[0] == "rake":
        return (embers + step[1], flue)
    return (embers, "keen" if flue == "mellow" else "mellow")


def _stove_describe(step: tuple) -> str:
    if step[0] == "rake":
        return f"rake in {step[1]} clods"
    return "stoke the grate"


def peatstove_lesson(rng: random.Random) -> dict:
    for _ in range(5000):
        threshold = rng.randint(9, 15)
        count = rng.randint(5, 7)
        steps: list[tuple] = []
        for _ in range(count):
            if rng.random() < 0.7:
                steps.append(("rake", rng.randint(STOVE_RAKE_LOW, STOVE_RAKE_HIGH)))
            else:
                steps.append(("stoke",))
        rakes = sum(step[0] == "rake" for step in steps)
        stokes = count - rakes
        if rakes < 3 or stokes < 1:
            continue
        initial = (0, "mellow")
        transcript: list[tuple[tuple, str]] = []
        state = initial
        for step in steps:
            state = _stove_apply(step, state)
            transcript.append((step, _stove_readout(state[0], threshold)))
        answer = _stove_output(state)
        distractor_stateless = _stove_output(initial)
        distractor_lastonly = _stove_output(_stove_apply(steps[-1], initial))
        if answer in (distractor_stateless, distractor_lastonly):
            continue
        bounded_audit = audit_bounded_parameters(
            steps, "rake", STOVE_RAKE_LOW, STOVE_RAKE_HIGH
        )
        step_lines = "\n".join(
            f"  {index + 1}. {_stove_describe(step)} -> the stove crackled {readout}"
            for index, (step, readout) in enumerate(transcript)
        )
        prompt = (
            "A shut peat stove keeps two covered values: an ember count "
            "(starts at 0) and a flue set (starts mellow).\n"
            "Documented workings: 'rake in n clods' raises the ember count by "
            f"n — the rake comes in {STOVE_LEGALITY_CLAUSE}; 'stoke the grate' "
            "trades the flue between mellow and keen and leaves the ember "
            "count alone. After every step the stove crackles SEARING when "
            f"the ember count is at least {threshold}, otherwise TEPID — the "
            "crackle is all it ever shows.\n"
            "When told 'quench the stove', the stoker must chalk up "
            "<FLUE>-<ember count>, with the flue word in capitals.\n"
            f"Steps already done:\n{step_lines}\n"
            "Next order: quench the stove. What must the stoker chalk up?\n"
            f"{ANSWER_LINE}"
        )
        reasoning = [
            "I follow the covered values step by step; start: ember count 0, "
            f"flue mellow, crackle threshold {threshold}."
        ]
        state = initial
        for index, (step, readout) in enumerate(transcript, 1):
            state = _stove_apply(step, state)
            reasoning.append(
                f"Step {index} ({_stove_describe(step)}): ember count {state[0]}, "
                f"flue {state[1]} — crackles {readout}, which matches."
            )
        reasoning.append(f"The quench order chalks up {answer}.")
        return make_row(
            prompt=prompt,
            think=" ".join(reasoning),
            answer=answer,
            kind="statechain",
            surface="peatstove",
            level=count,
            audit={
                "truth_valid": True,
                "formalism": "peatstove",
                "hidden_updates": count,
                "distractor_stateless": distractor_stateless,
                "distractor_lastonly": distractor_lastonly,
                "distractors_differ": True,
                "legality_clauses": [STOVE_LEGALITY_CLAUSE],
                "bounded_parameter_audit": bounded_audit,
                "spec": {
                    "threshold": threshold,
                    "steps": [list(step) for step in steps],
                    "readouts": [readout for _, readout in transcript],
                    "final": list(state),
                },
            },
        )
    raise RuntimeError("could not construct a peatstove statechain lesson")


# ---------------------------------------------------------------------------
# muletrack (NEW: ring position + pannier count; halter paces bounded)
# ---------------------------------------------------------------------------

TRACK_PACE_LOW = 1
TRACK_PACE_HIGH = 4
TRACK_LEGALITY_CLAUSE = (
    "exactly four paces, 1, 2, 3, or 4 posts, and no other pace exists"
)


def _track_output(state: tuple[int, int], posts: list[str]) -> str:
    position, pannier = state
    return f"{posts[position]}-{pannier}"


def _track_apply(step: tuple, state: tuple[int, int]) -> tuple[int, int]:
    position, pannier = state
    if step[0] == "plod":
        return ((position + step[1]) % 5, pannier)
    return (position, pannier + position + 1)


def _track_readout(step: tuple, before: tuple[int, int], threshold: int) -> str:
    if step[0] == "plod":
        return "LAPPED" if before[0] + step[1] >= 5 else "MIDWAY"
    after = _track_apply(step, before)
    return "BRIMMING" if after[1] >= threshold else "HOLLOW"


def _track_describe(step: tuple) -> str:
    if step[0] == "plod":
        return f"plod {step[1]} posts ahead"
    return "heap at the current post"


def muletrack_lesson(rng: random.Random) -> dict:
    for _ in range(5000):
        posts = rng.sample(MULEPOSTS, 5)
        threshold = rng.randint(6, 11)
        count = rng.randint(5, 7)
        steps: list[tuple] = []
        for _ in range(count):
            if rng.random() < 0.6:
                steps.append(("plod", rng.randint(TRACK_PACE_LOW, TRACK_PACE_HIGH)))
            else:
                steps.append(("heap",))
        plods = sum(step[0] == "plod" for step in steps)
        heaps = count - plods
        if plods < 2 or heaps < 2:
            continue
        initial = (0, 0)
        transcript: list[tuple[tuple, str]] = []
        state = initial
        for step in steps:
            readout = _track_readout(step, state, threshold)
            state = _track_apply(step, state)
            transcript.append((step, readout))
        answer = _track_output(state, posts)
        distractor_stateless = _track_output(initial, posts)
        distractor_lastonly = _track_output(_track_apply(steps[-1], initial), posts)
        if answer in (distractor_stateless, distractor_lastonly):
            continue
        bounded_audit = audit_bounded_parameters(
            steps, "plod", TRACK_PACE_LOW, TRACK_PACE_HIGH
        )
        step_lines = "\n".join(
            f"  {index + 1}. {_track_describe(step)} -> the track called {readout}"
            for index, (step, readout) in enumerate(transcript)
        )
        prompt = (
            "A pack mule track keeps two covered values: which post the mule "
            "stands at and a pannier count (starts at 0). The posts sit in "
            "ring order " + " ".join(posts) + " (after the last post the mule "
            "comes back to the first), and the mule starts at the first "
            "listed post.\n"
            "Documented workings: 'plod n posts ahead' moves the mule n posts "
            f"around the ring — the halter allows {TRACK_LEGALITY_CLAUSE}; "
            "the track calls LAPPED when the move passes the end of the ring, "
            "otherwise MIDWAY; 'heap at the current post' adds the post's "
            "ring position (1 for the first listed post, 5 for the last) to "
            f"the pannier — the track calls BRIMMING when the pannier is at "
            f"least {threshold}, otherwise HOLLOW. The calls are all it ever "
            "shows.\n"
            "When told 'unhitch', the drover must call out <post>-<pannier>.\n"
            f"Steps already done:\n{step_lines}\n"
            "Next order: unhitch. What must the drover call out?\n"
            f"{ANSWER_LINE}"
        )
        reasoning = [
            "I follow the covered values step by step; start: mule at "
            f"{posts[0]} (ring position 1), pannier 0, call threshold {threshold}."
        ]
        state = initial
        for index, (step, readout) in enumerate(transcript, 1):
            state = _track_apply(step, state)
            reasoning.append(
                f"Step {index} ({_track_describe(step)}): mule at "
                f"{posts[state[0]]} (ring position {state[0] + 1}), pannier "
                f"{state[1]} — calls {readout}, which matches."
            )
        reasoning.append(f"The unhitch order calls out {answer}.")
        return make_row(
            prompt=prompt,
            think=" ".join(reasoning),
            answer=answer,
            kind="statechain",
            surface="muletrack",
            level=count,
            audit={
                "truth_valid": True,
                "formalism": "muletrack",
                "hidden_updates": count,
                "distractor_stateless": distractor_stateless,
                "distractor_lastonly": distractor_lastonly,
                "distractors_differ": True,
                "legality_clauses": [TRACK_LEGALITY_CLAUSE],
                "bounded_parameter_audit": bounded_audit,
                "spec": {
                    "posts": posts,
                    "threshold": threshold,
                    "steps": [list(step) for step in steps],
                    "readouts": [readout for _, readout in transcript],
                    "final": list(state),
                },
            },
        )
    raise RuntimeError("could not construct a muletrack statechain lesson")


def statechain_lesson(rng: random.Random, formalism: str) -> dict:
    if formalism == "brewvat":
        return brewvat_lesson(rng)
    if formalism == "courierloft":
        return courierloft_lesson(rng)
    if formalism == "peatstove":
        return peatstove_lesson(rng)
    if formalism == "muletrack":
        return muletrack_lesson(rng)
    raise ValueError(f"unknown statechain formalism: {formalism}")


# ---------------------------------------------------------------------------
# corpus assembly
# ---------------------------------------------------------------------------

def _statechain_at(rng: random.Random, index: int) -> dict:
    return statechain_lesson(
        rng, STATECHAIN_FORMALISMS[index % len(STATECHAIN_FORMALISMS)]
    )


SKILLS = {
    "statechain": _statechain_at,
}
ARM_MIX = "statechain=160"
HOLDOUT_MIX = "statechain=40"
SMOKE_MIX = "statechain=8"


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
            row["task_id"] = f"sod_{skill}_{index:05d}"
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
        if row["kind"] != "u_statechain":
            raise ValueError(f"row {index} has an unknown kind: {row['kind']}")
        audit = row["_audit"]
        if audit.get("truth_valid") is not True:
            raise ValueError(f"row {index} lacks truth audit")
        if audit.get("formalism") not in STATECHAIN_FORMALISMS:
            raise ValueError(f"row {index} has an unknown formalism")
        if (
            not isinstance(audit.get("hidden_updates"), int)
            or audit["hidden_updates"] < 3
            or audit.get("distractors_differ") is not True
            or row["answer"].removeprefix("ANSWER: ")
            in (audit.get("distractor_stateless"), audit.get("distractor_lastonly"))
        ):
            raise ValueError(f"row {index} statechain distractor audit failed")
        if audit["formalism"] in NEW_FORMALISMS:
            bounded = audit.get("bounded_parameter_audit") or {}
            prompt_text = row["messages"][0]["content"]
            if (
                not audit.get("legality_clauses")
                or any(
                    clause not in prompt_text
                    for clause in audit["legality_clauses"]
                )
                or bounded.get("all_parameters_documented_bounded") is not True
                or bounded.get("out_of_bound_steps") != 0
                or bounded.get("amounts_probed_to") != EXTENDED_AMOUNT_BOUND
                or not isinstance(bounded.get("parameterized_steps"), int)
                or not bounded.get("documented_bounds")
            ):
                raise ValueError(f"row {index} legality-bounding audit failed")
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
    """Corpus-level formalism balance for the frozen 160-row arm."""
    statechain = [row for row in rows if row["kind"] == "u_statechain"]
    statechain_formalisms = Counter(row["_audit"]["formalism"] for row in statechain)
    new_formalism_rows = [
        row for row in statechain if row["_audit"]["formalism"] in NEW_FORMALISMS
    ]
    balance = {
        "statechain_formalisms": dict(sorted(statechain_formalisms.items())),
        "statechain_hidden_updates_min": (
            min(row["_audit"]["hidden_updates"] for row in statechain)
            if statechain
            else None
        ),
        "new_formalism_rows_with_out_of_bound_parameters": sum(
            row["_audit"]["bounded_parameter_audit"]["out_of_bound_steps"] > 0
            for row in new_formalism_rows
        ),
    }
    if statechain and (
        set(statechain_formalisms) != set(STATECHAIN_FORMALISMS)
        or len(set(statechain_formalisms.values())) != 1
    ):
        raise ValueError(f"statechain formalism balance out of range: {balance}")
    if balance["new_formalism_rows_with_out_of_bound_parameters"]:
        raise ValueError(f"legality-bounding balance out of range: {balance}")
    return balance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mix", default=None)
    parser.add_argument("--seed", type=int, default=77140)
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
