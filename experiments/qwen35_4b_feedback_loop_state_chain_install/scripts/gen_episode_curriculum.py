#!/usr/bin/env python3
"""Designed episode curriculum: feedback-loop and state-chain lessons.

Every row in every prior universal curriculum is a single-turn atom; the two
blocking public families are bounded multi-step EPISODES. Each lesson below
renders one complete episode as plain text inside ONE user message, on
invented executable machine formalisms with fresh vocabulary:

- ``u_feedloop`` (act -> observe -> revise): four invented machines
  (troughline, trinketcord, crankwheel, sigilslate). A written step sequence
  carries exactly one wrong step. Trial one shows failing evidence (wanted vs
  finished). An earlier WRONG fix attempt — plausible because at least two
  single-step changes square with trial one alone — was tried, and a second
  trial from a fresh start shows it failing too. The assistant narrates what
  the second-trial evidence eliminates, then answers the unique correct
  change. Every parameterized operation is explicitly BOUNDED in the
  rendered spec text (ladle sizes, crank settings, trinket and sigil pools),
  so the fix grammar is finite BY DOCUMENTATION, not by convention.
  Uniqueness after both trials, and >=2 candidates after trial one, are
  verified by exhaustive enumeration over that bounded grammar; a second
  enumeration over an EXTENDED grammar (amounts up to 12, items over the
  full module pools) verifies that every extra survivor is excluded by the
  documented legality clause alone. Rows that fail bounded uniqueness are
  rejected and redrawn from the same deterministic rng stream (attempt cap
  5000).
- ``u_statechain`` (hidden-but-documented state tracking): two invented
  procedure machines (brewvat, courierloft) with hidden counters and a hidden
  two-way scent/position that only lossy per-step readouts ever surface. The
  transcript lists already-executed steps with their visible readouts plus a
  closing order whose required output depends on the ACCUMULATED hidden
  state. The generator simulates the machine, requires at least three hidden
  updates, and verifies that a stateless reader and a last-step-only reader
  both produce a different answer.

Every answer is computed by executing the specification; a banned-vocabulary
scan rejects any leak of benchmark family names, gym flavor nouns, prior
surface pools, or the public blocker-family description nouns.
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

# Fresh invented vocabulary. Every token here is grep-verified against every
# predecessor corpus by build_corpus.py (zero word-boundary hits allowed).
TROUGHS = ("brann", "clave", "dorst", "grelm", "murd", "prell")
TRINKETS = ("plome", "quenn", "trell", "vosk", "yorv", "zilk")
WHEELFACES = ("drasp", "fyne", "golve", "hilm", "juss", "krev")
SIGILS = ("morv", "nilt", "pryn", "quav", "rulf", "senn")
PERCHES = ("orvan", "welk", "xemb", "yurr", "zeff", "obbin")
FRESH_SURFACE_TOKENS = (
    # machine surface names
    "troughline", "trinketcord", "crankwheel", "sigilslate", "brewvat",
    "courierloft",
    # item pools
    *TROUGHS, *TRINKETS, *WHEELFACES, *SIGILS, *PERCHES,
    # distinctive machine nouns and readout words
    "trough", "troughs", "trinket", "trinkets", "sigil", "sigils",
    "dram", "drams", "perch", "perches", "satchel", "cork", "corked",
    "laden", "scant", "stout", "faint", "looped", "onward", "brisk",
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
# u_feedloop machine formalisms (executable; exhaustive fix enumeration)
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
    """Legality probe: ladle amounts up to EXTENDED_AMOUNT_BOUND."""
    ops: list[tuple] = []
    for target in names:
        for amount in range(1, EXTENDED_AMOUNT_BOUND + 1):
            ops.append(("ladle", amount, target))
    for source in names:
        for target in names:
            if source != target:
                ops.append(("tipover", source, target))
    for target in names:
        ops.append(("scoop", target))
    return ops


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


FEEDLOOP_FORMALISMS = ("troughline", "trinketcord", "crankwheel", "sigilslate")


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
        # operation, and re-enumerating fixes over the EXTENDED grammar
        # (amounts up to EXTENDED_AMOUNT_BOUND; items over the full module
        # pool) must surface no within-bound alternative — every extra
        # round-1+2 survivor must be excluded by the documented legality
        # clause alone (i.e. sit outside the bounded grammar).
        legality_clauses = machine["legality_clauses"]
        if not legality_clauses or any(
            clause not in machine["rules"] for clause in legality_clauses
        ):
            raise RuntimeError("a parameterized operation is not bounded in the spec")
        extended_grammar = machine["extended_grammar"]
        if any(op not in extended_grammar for op in grammar):
            raise RuntimeError("extended legality grammar does not cover the bounded grammar")
        bounded_ops = set(grammar)
        extended_round2 = [
            (index, candidate)
            for index, candidate in _consistent_fixes(
                apply_fn, extended_grammar, written, start_a, wanted_a
            )
            if _run_steps(
                apply_fn, written[:index] + [candidate] + written[index + 1 :], start_b
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
                    "items_probed": "full module pool",
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
# u_statechain procedure machines (hidden-but-documented state)
# ---------------------------------------------------------------------------

STATECHAIN_FORMALISMS = ("brewvat", "courierloft")


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


def statechain_lesson(rng: random.Random, formalism: str) -> dict:
    if formalism == "brewvat":
        return brewvat_lesson(rng)
    if formalism == "courierloft":
        return courierloft_lesson(rng)
    raise ValueError(f"unknown statechain formalism: {formalism}")


# ---------------------------------------------------------------------------
# corpus assembly
# ---------------------------------------------------------------------------

def _feedloop_at(rng: random.Random, index: int) -> dict:
    return feedloop_lesson(rng, FEEDLOOP_FORMALISMS[index % len(FEEDLOOP_FORMALISMS)])


def _statechain_at(rng: random.Random, index: int) -> dict:
    return statechain_lesson(
        rng, STATECHAIN_FORMALISMS[index % len(STATECHAIN_FORMALISMS)]
    )


SKILLS = {
    "feedloop": _feedloop_at,
    "statechain": _statechain_at,
}
ARM_MIX = "feedloop=80,statechain=80"
HOLDOUT_MIX = "feedloop=20,statechain=20"
SMOKE_MIX = ",".join(f"{name}=2" for name in SKILLS)


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
            row["task_id"] = f"fls_{skill}_{index:05d}"
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
        audit = row["_audit"]
        if audit.get("truth_valid") is not True:
            raise ValueError(f"row {index} lacks truth audit")
        if row["kind"] == "u_feedloop":
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
                or not isinstance(extended.get("round2_survivors_extended"), int)
                or extended["round2_survivors_extended"] < 1
                or not audit.get("legality_clauses")
                or any(
                    clause not in prompt_text
                    for clause in audit["legality_clauses"]
                )
            ):
                raise ValueError(f"row {index} feedloop evidence audit failed")
        elif row["kind"] == "u_statechain":
            if (
                not isinstance(audit.get("hidden_updates"), int)
                or audit["hidden_updates"] < 3
                or audit.get("distractors_differ") is not True
                or row["answer"].removeprefix("ANSWER: ")
                in (audit.get("distractor_stateless"), audit.get("distractor_lastonly"))
            ):
                raise ValueError(f"row {index} statechain distractor audit failed")
        else:
            raise ValueError(f"row {index} has an unknown kind: {row['kind']}")
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
    feedloop = [row for row in rows if row["kind"] == "u_feedloop"]
    statechain = [row for row in rows if row["kind"] == "u_statechain"]
    feedloop_formalisms = Counter(row["_audit"]["formalism"] for row in feedloop)
    statechain_formalisms = Counter(row["_audit"]["formalism"] for row in statechain)
    ambiguity = Counter(
        row["_audit"]["candidates_after_round1"] for row in feedloop
    )
    out_of_bound = Counter(
        row["_audit"]["extended_uniqueness_audit"]["out_of_bound_alternatives"]
        for row in feedloop
    )
    balance = {
        "feedloop_formalisms": dict(sorted(feedloop_formalisms.items())),
        "statechain_formalisms": dict(sorted(statechain_formalisms.items())),
        "feedloop_round1_candidate_counts": {
            str(key): value for key, value in sorted(ambiguity.items())
        },
        "feedloop_out_of_bound_alternative_counts": {
            str(key): value for key, value in sorted(out_of_bound.items())
        },
        "feedloop_rows_with_out_of_bound_alternatives": sum(
            count for key, count in out_of_bound.items() if key > 0
        ),
        "statechain_hidden_updates_min": (
            min(row["_audit"]["hidden_updates"] for row in statechain)
            if statechain
            else None
        ),
    }
    if feedloop and (
        set(feedloop_formalisms) != set(FEEDLOOP_FORMALISMS)
        or len(set(feedloop_formalisms.values())) != 1
    ):
        raise ValueError(f"feedloop formalism balance out of range: {balance}")
    if statechain and (
        set(statechain_formalisms) != set(STATECHAIN_FORMALISMS)
        or len(set(statechain_formalisms.values())) != 1
    ):
        raise ValueError(f"statechain formalism balance out of range: {balance}")
    return balance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mix", default=None)
    parser.add_argument("--seed", type=int, default=77130)
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
