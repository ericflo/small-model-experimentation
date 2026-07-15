#!/usr/bin/env python3
"""Fresh-surface truth-audited curriculum plus the bounded-check budget lesson.

This generator keeps the proven thirteen lesson constructors byte-compatible with
`gen_curriculum.py` but renders every lesson over six fresh surface pools, fresh
separators, fresh record attributes, and fresh routing capabilities that are all
disjoint from the six predecessor surfaces, from every gym replay family, and from
the public benchmark family names.  It adds one new lesson kind, `u_budget`: a
resource-bounded ordered scan with a hard check allowance, an immediate stop on the
first hit, and a mandatory `BUDGET` commit on exhaustion.  Every exhaustion task
plants a satisfier immediately past the cutoff, so violating the allowance yields a
parseable wrong answer; respecting the contract is the only way to be correct.
Every target is computed from a small executable specification and audited."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ANSWER_LINE = "End with exactly one line:\nANSWER: <answer>"


@dataclass(frozen=True)
class Surface:
    name: str
    items: tuple[str, ...]
    sep: str


SURFACE_POOLS = {
    "greek": ("alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda", "sigma"),
    "elements": ("argon", "boron", "radon", "xenon", "neon", "krypton", "helium", "lithium", "carbon", "sulfur", "iodine", "nickel"),
    "animals": ("heron", "otter", "lynx", "ibex", "crane", "vole", "skink", "tapir", "zebu", "okapi", "bison", "finch"),
    "ordinals": ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth", "eleventh", "twelfth"),
    "gems": ("beryl", "coral", "flint", "garnet", "jasper", "onyx", "opal", "quartz", "topaz", "zircon", "agate", "pyrite"),
    "digraphs": ("BQ", "DZ", "FX", "GK", "HV", "JN", "LW", "MY", "PC", "RT", "SD", "XZ"),
}
SEPARATORS = (" | ", " :: ", " .. ", " * ")
OP_KINDS = ("shift", "reverse", "swap", "overwrite", "conditional_rotate")


def make_surface(rng: random.Random, min_n: int = 6, max_n: int = 10) -> Surface:
    name = rng.choice(tuple(SURFACE_POOLS))
    n = rng.randint(min_n, max_n)
    items = rng.sample(list(SURFACE_POOLS[name]), n)
    return Surface(name, tuple(items), rng.choice(SEPARATORS))


def render(sequence: tuple[int, ...], surface: Surface) -> str:
    return surface.sep.join(surface.items[index] for index in sequence)


def cycle_header(surface: Surface) -> str:
    return (
        "Cycle order: " + " ".join(surface.items)
        + " (after the last item, wrap to the first)."
    )


def random_sequence(rng: random.Random, length: int, size: int) -> tuple[int, ...]:
    return tuple(rng.randrange(size) for _ in range(length))


def apply_op(op: tuple, sequence: tuple[int, ...], size: int) -> tuple[int, ...]:
    kind = op[0]
    if kind == "shift":
        return tuple((value + op[1]) % size for value in sequence)
    if kind == "reverse":
        return tuple(reversed(sequence))
    if kind == "swap":
        result = list(sequence)
        result[op[1]], result[op[2]] = result[op[2]], result[op[1]]
        return tuple(result)
    if kind == "overwrite":
        return sequence[: op[1]] + (op[2],) + sequence[op[1] + 1 :]
    if kind == "conditional_rotate":
        if sequence[0] in (op[1], op[2]):
            return sequence[1:] + sequence[:1]
        return sequence[-1:] + sequence[:-1]
    raise ValueError(f"unknown operation: {op}")


def random_op(rng: random.Random, length: int, size: int) -> tuple:
    kind = rng.choice(OP_KINDS)
    if kind == "shift":
        return (kind, rng.randint(1, size - 1))
    if kind == "reverse":
        return (kind,)
    if kind == "swap":
        first, second = sorted(rng.sample(range(length), 2))
        return (kind, first, second)
    if kind == "overwrite":
        return (kind, rng.randrange(length), rng.randrange(size))
    return (kind, *sorted(rng.sample(range(size), 2)))


@lru_cache(maxsize=None)
def all_ops(length: int, size: int) -> tuple[tuple, ...]:
    result: list[tuple] = [("reverse",)]
    result.extend(("shift", amount) for amount in range(1, size))
    result.extend(("swap", first, second) for first, second in combinations(range(length), 2))
    result.extend(("overwrite", position, value) for position in range(length) for value in range(size))
    result.extend(("conditional_rotate", first, second) for first, second in combinations(range(size), 2))
    return tuple(result)


def describe_op(op: tuple, surface: Surface) -> str:
    kind = op[0]
    if kind == "shift":
        return f"advance every item {op[1]} step(s) in the cycle"
    if kind == "reverse":
        return "reverse the entire sequence"
    if kind == "swap":
        return f"swap positions {op[1] + 1} and {op[2] + 1}"
    if kind == "overwrite":
        return f"overwrite position {op[1] + 1} with {surface.items[op[2]]}"
    return (
        f"if the first item is {surface.items[op[1]]} or {surface.items[op[2]]}, "
        "rotate left once; otherwise rotate right once"
    )


@lru_cache(maxsize=None)
def depth_witnesses(length: int, size: int) -> tuple[tuple[int, ...], ...]:
    rows: list[tuple[int, ...]] = [(value,) * length for value in range(size)]
    for position in range(length):
        for value in range(size):
            row = [0] * length
            row[position] = value
            rows.append(tuple(row))
    rng = random.Random(991_000 + 100 * length + size)
    rows.extend(random_sequence(rng, length, size) for _ in range(160))
    return tuple(dict.fromkeys(rows))


def composition_is_primitive(op1: tuple, op2: tuple, length: int, size: int) -> bool:
    witnesses = depth_witnesses(length, size)
    outputs = tuple(apply_op(op2, apply_op(op1, row, size), size) for row in witnesses)
    return any(
        all(apply_op(candidate, row, size) == output for row, output in zip(witnesses, outputs))
        for candidate in all_ops(length, size)
    )


def fitting_seconds(
    middles: list[tuple[int, ...]], outputs: list[tuple[int, ...]], length: int, size: int
) -> list[tuple]:
    return [
        op
        for op in all_ops(length, size)
        if all(apply_op(op, middle, size) == output for middle, output in zip(middles, outputs))
    ]


def make_row(
    *,
    prompt: str,
    think: str,
    answer: str,
    kind: str,
    surface: str,
    level: int,
    audit: dict,
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


def induct_lesson(rng: random.Random) -> dict:
    for attempt in range(160):
        surface = make_surface(rng)
        size = len(surface.items)
        length = rng.choice((5, 6, 7))
        op1 = random_op(rng, length, size)
        op2 = random_op(rng, length, size)
        if composition_is_primitive(op1, op2, length, size):
            continue
        probes: list[tuple[int, ...]] = []
        while len(probes) < 6:
            candidate = random_sequence(rng, length, size)
            if candidate not in probes:
                probes.append(candidate)
        outputs = [apply_op(op2, apply_op(op1, row, size), size) for row in probes]
        query = random_sequence(rng, length, size)
        if query in probes:
            continue
        answer = apply_op(op2, apply_op(op1, query, size), size)
        consistent = 0
        predictions: set[tuple[int, ...]] = set()
        dead: tuple[tuple, list[tuple[int, ...]]] | None = None
        for candidate1 in all_ops(length, size):
            middles = [apply_op(candidate1, row, size) for row in probes]
            seconds = fitting_seconds(middles, outputs, length, size)
            if not seconds and candidate1 != op1 and dead is None:
                dead = (candidate1, middles)
            for candidate2 in seconds:
                consistent += 1
                predictions.add(
                    apply_op(candidate2, apply_op(candidate1, query, size), size)
                )
        if dead is None or predictions != {answer}:
            continue
        true_middles = [apply_op(op1, row, size) for row in probes]
        assert op2 in fitting_seconds(true_middles, outputs, length, size)
        break
    else:
        raise RuntimeError("could not synthesize an identifiable depth-2 induction lesson")

    probe_text = "\n".join(
        f"  {render(source, surface)} -> {render(target, surface)}"
        for source, target in zip(probes, outputs)
    )
    prompt = (
        f"A hidden rule transforms sequences of {length} items by chaining exactly two "
        "operations. Allowed operations are cycle-advance, full reversal, a fixed-position "
        "swap, a fixed-position overwrite, and a first-item-conditional one-step rotation. "
        "The examples determine the requested output.\n"
        f"{cycle_header(surface)}\nProbe log:\n{probe_text}\n\n"
        f"Find the output for {render(query, surface)}.\n{ANSWER_LINE}"
    )
    dead_op, dead_middles = dead
    dead_examples = ", ".join(
        f"{render(source, surface)}->{render(middle, surface)}"
        for source, middle in zip(probes, dead_middles)
    )
    true_examples = ", ".join(
        f"{render(source, surface)}->{render(middle, surface)}"
        for source, middle in zip(probes, true_middles)
    )
    second_examples = ", ".join(
        f"{render(middle, surface)}->{render(target, surface)}"
        for middle, target in zip(true_middles, outputs)
    )
    query_middle = apply_op(op1, query, size)
    think = " ".join(
        (
            "I search for output = step2(step1(input)). Fix a candidate first step, compute "
            "all intermediate rows, then ask whether one allowed second step explains every probe.",
            f"Candidate first step {describe_op(dead_op, surface)} gives {dead_examples}. "
            "No allowed second step maps all of those intermediates to the logged outputs, so "
            "that branch is a dead end.",
            f"Try first step {describe_op(op1, surface)}. Its intermediates are {true_examples}.",
            f"Second step {describe_op(op2, surface)} matches every probe: {second_examples}.",
            f"Apply that decomposition to the query: {render(query, surface)} -> "
            f"{render(query_middle, surface)} -> {render(answer, surface)}.",
        )
    )
    return make_row(
        prompt=prompt,
        think=think,
        answer=render(answer, surface),
        kind="induct",
        surface=surface.name,
        level=5,
        audit={
            "truth_valid": True,
            "behavioral_min_depth": 2,
            "query_identifiable": True,
            "consistent_compositions": consistent,
            "has_dead_end": True,
            "attempt": attempt + 1,
        },
    )


def execute_lesson(rng: random.Random) -> dict:
    surface = make_surface(rng)
    size = len(surface.items)
    length = rng.choice((5, 6, 7))
    steps = [random_op(rng, length, size) for _ in range(rng.choice((2, 3, 4)))]
    source = random_sequence(rng, length, size)
    states = [source]
    for op in steps:
        states.append(apply_op(op, states[-1], size))
    prompt = (
        f"Apply this procedure in order to {render(source, surface)}:\n  "
        + "; then ".join(describe_op(op, surface) for op in steps)
        + f".\n{cycle_header(surface)}\n{ANSWER_LINE}"
    )
    trace = ["I carry the current state forward instead of restarting from the input."]
    for index, (op, before, after) in enumerate(zip(steps, states[:-1], states[1:]), 1):
        trace.append(
            f"Step {index}, {describe_op(op, surface)}: {render(before, surface)} -> "
            f"{render(after, surface)}."
        )
    trace.append(f"The final state is {render(states[-1], surface)}.")
    return make_row(
        prompt=prompt,
        think=" ".join(trace),
        answer=render(states[-1], surface),
        kind="execute",
        surface=surface.name,
        level=len(steps),
        audit={"truth_valid": True, "steps": len(steps)},
    )


ATTRIBUTES = ("heft", "shine", "reach", "pulse")


def make_candidates(rng: random.Random, surface: Surface, count: int) -> list[dict]:
    return [
        {"id": surface.items[index], **{attribute: rng.randint(1, 9) for attribute in ATTRIBUTES}}
        for index in range(count)
    ]


def constraint_passes(candidate: dict, constraint: tuple[str, str, int]) -> bool:
    attribute, relation, value = constraint
    return candidate[attribute] >= value if relation == ">=" else candidate[attribute] <= value


def candidate_table(candidates: list[dict]) -> str:
    return "\n".join(
        "  - " + candidate["id"] + ": "
        + ", ".join(f"{attribute} {candidate[attribute]}" for attribute in ATTRIBUTES)
        for candidate in candidates
    )


def sample_constraints(
    rng: random.Random, candidates: list[dict], desired: int | None
) -> tuple[list[tuple[str, str, int]], list[dict]]:
    for _ in range(300):
        constraints = [
            (attribute, rng.choice((">=", "<=")), rng.randint(2, 8))
            for attribute in rng.sample(list(ATTRIBUTES), rng.choice((2, 3)))
        ]
        winners = [
            candidate
            for candidate in candidates
            if all(constraint_passes(candidate, constraint) for constraint in constraints)
        ]
        if desired is None and len(winners) == 1:
            return constraints, winners
        if desired == 0 and not winners:
            return constraints, winners
        if desired == 1 and len(winners) == 1:
            return constraints, winners
        if desired == 2 and len(winners) >= 2:
            return constraints, winners
    raise RuntimeError("could not create requested constraint support")


def constraint_reasoning(candidates: list[dict], constraints: list[tuple[str, str, int]]) -> list[str]:
    lines: list[str] = []
    for candidate in candidates:
        checks = []
        for constraint in constraints:
            attribute, relation, value = constraint
            passed = constraint_passes(candidate, constraint)
            checks.append(
                f"{attribute} {candidate[attribute]} {relation} {value}: {'yes' if passed else 'NO'}"
            )
        accepted = all(constraint_passes(candidate, constraint) for constraint in constraints)
        lines.append(f"{candidate['id']}: " + ", ".join(checks) + f" -> {'keep' if accepted else 'reject'}.")
    return lines


def select_lesson(rng: random.Random) -> dict:
    surface = make_surface(rng, 8, 10)
    candidates = make_candidates(rng, surface, rng.choice((6, 7, 8)))
    constraints, winners = sample_constraints(rng, candidates, None)
    constraint_text = " and ".join(f"{a} {op} {value}" for a, op, value in constraints)
    prompt = (
        "Exactly one record satisfies every requirement. Return its id.\nRecords:\n"
        f"{candidate_table(candidates)}\nRequirements: {constraint_text}.\n{ANSWER_LINE}"
    )
    reasoning = ["I apply every requirement conjunctively; one failure rejects a record."]
    reasoning.extend(constraint_reasoning(candidates, constraints))
    reasoning.append(f"Only {winners[0]['id']} survives all checks.")
    return make_row(
        prompt=prompt,
        think=" ".join(reasoning),
        answer=winners[0]["id"],
        kind="select",
        surface=surface.name,
        level=len(constraints),
        audit={"truth_valid": True, "winner_count": 1},
    )


def count_lesson(rng: random.Random) -> dict:
    surface = make_surface(rng, 8, 10)
    candidates = make_candidates(rng, surface, rng.choice((6, 7, 8)))
    constraint = (rng.choice(ATTRIBUTES), rng.choice((">=", "<=")), rng.randint(3, 7))
    passed = [candidate for candidate in candidates if constraint_passes(candidate, constraint)]
    attribute, relation, value = constraint
    prompt = (
        f"Count the records satisfying {attribute} {relation} {value}.\nRecords:\n"
        f"{candidate_table(candidates)}\n{ANSWER_LINE}"
    )
    total = 0
    reasoning = ["I test the predicate once per record and maintain a running total."]
    for candidate in candidates:
        ok = constraint_passes(candidate, constraint)
        total += int(ok)
        reasoning.append(
            f"{candidate['id']}: {candidate[attribute]} {relation} {value} is "
            f"{'true' if ok else 'false'}; total {total}."
        )
    return make_row(
        prompt=prompt,
        think=" ".join(reasoning),
        answer=str(len(passed)),
        kind="count",
        surface=surface.name,
        level=2,
        audit={"truth_valid": True, "count": len(passed)},
    )


def trace_lesson(rng: random.Random) -> dict:
    surface = make_surface(rng, 8, 10)
    count = rng.choice((6, 7, 8))
    nodes = list(range(count))
    while True:
        targets = rng.sample(nodes, len(nodes))
        if all(node != target for node, target in zip(nodes, targets)):
            break
    links = dict(zip(nodes, targets))
    start = rng.choice(nodes)
    hops = rng.choice((3, 4, 5, 6))
    path = [start]
    for _ in range(hops):
        path.append(links[path[-1]])
    link_text = "\n".join(
        f"  - {surface.items[node]} points to {surface.items[links[node]]}" for node in nodes
    )
    prompt = (
        f"Follow exactly {hops} pointers, starting at {surface.items[start]}.\n{link_text}\n"
        f"{ANSWER_LINE}"
    )
    reasoning = [f"Start at {surface.items[start]} and count one hop per pointer."]
    reasoning.extend(
        f"Hop {index}: {surface.items[before]} -> {surface.items[after]}."
        for index, (before, after) in enumerate(zip(path[:-1], path[1:]), 1)
    )
    return make_row(
        prompt=prompt,
        think=" ".join(reasoning),
        answer=surface.items[path[-1]],
        kind="trace",
        surface=surface.name,
        level=hops,
        audit={"truth_valid": True, "hops": hops},
    )


def verify_lesson(rng: random.Random) -> dict:
    surface = make_surface(rng)
    size = len(surface.items)
    length = rng.choice((5, 6, 7))
    steps = [random_op(rng, length, size) for _ in range(rng.choice((2, 3, 4)))]
    source = random_sequence(rng, length, size)
    states = [source]
    for op in steps:
        states.append(apply_op(op, states[-1], size))
    correct = rng.random() < 0.5
    claimed = states[-1]
    if not correct:
        wrong = list(claimed)
        position = rng.randrange(length)
        wrong[position] = (wrong[position] + rng.randint(1, size - 1)) % size
        claimed = tuple(wrong)
    prompt = (
        f"A worker starts with {render(source, surface)} and applies: "
        + "; then ".join(describe_op(op, surface) for op in steps)
        + f". They claim the result is {render(claimed, surface)}. Verify the claim.\n"
        + cycle_header(surface)
        + "\nEnd with exactly one line: ANSWER: YES or ANSWER: NO"
    )
    reasoning = ["I independently execute the procedure before comparing with the claim."]
    for index, (op, before, after) in enumerate(zip(steps, states[:-1], states[1:]), 1):
        reasoning.append(
            f"Step {index}, {describe_op(op, surface)}: {render(before, surface)} -> "
            f"{render(after, surface)}."
        )
    reasoning.append(
        f"Computed {render(states[-1], surface)}; it {'matches' if correct else 'differs from'} "
        f"{render(claimed, surface)}."
    )
    return make_row(
        prompt=prompt,
        think=" ".join(reasoning),
        answer="YES" if correct else "NO",
        kind="verify",
        surface=surface.name,
        level=len(steps),
        audit={"truth_valid": True, "claim_correct": correct},
    )


def repair_lesson(rng: random.Random) -> dict:
    for _ in range(120):
        surface = make_surface(rng)
        size = len(surface.items)
        length = rng.choice((5, 6))
        steps = [random_op(rng, length, size) for _ in range(rng.choice((3, 4)))]
        source = random_sequence(rng, length, size)
        correct_states = [source]
        for op in steps:
            correct_states.append(apply_op(op, correct_states[-1], size))
        bad_step = rng.randrange(len(steps))
        reported_states = [source]
        for index, op in enumerate(steps):
            next_state = apply_op(op, reported_states[-1], size)
            if index == bad_step:
                changed = list(next_state)
                position = rng.randrange(length)
                changed[position] = (changed[position] + rng.randint(1, size - 1)) % size
                next_state = tuple(changed)
            reported_states.append(next_state)
        if reported_states[-1] != correct_states[-1]:
            break
    else:
        raise RuntimeError("could not create a consequential repair example")
    trace_text = "\n".join(
        f"  step {index} ({describe_op(op, surface)}): {render(before, surface)} -> "
        f"{render(after, surface)}"
        for index, (op, before, after) in enumerate(
            zip(steps, reported_states[:-1], reported_states[1:]), 1
        )
    )
    prompt = (
        f"Audit this worked procedure. It starts at {render(source, surface)}, contains exactly "
        "one first error, and later lines continue from the erroneous state. Locate the first "
        f"error, recompute from there, and return the correct final state.\n{trace_text}\n"
        f"{cycle_header(surface)}\n{ANSWER_LINE}"
    )
    reasoning = ["I replay from the start and compare each claimed state with the required operation."]
    for index, (op, before, after) in enumerate(zip(steps, correct_states[:-1], correct_states[1:]), 1):
        marker = " This is the first mismatch." if index - 1 == bad_step else ""
        reasoning.append(
            f"Correct step {index}: {render(before, surface)} -> {render(after, surface)}.{marker}"
        )
    reasoning.append(f"After repairing and continuing, the final state is {render(correct_states[-1], surface)}.")
    return make_row(
        prompt=prompt,
        think=" ".join(reasoning),
        answer=render(correct_states[-1], surface),
        kind="repair",
        surface=surface.name,
        level=len(steps),
        audit={"truth_valid": True, "first_error": bad_step + 1},
    )


def optimize_lesson(rng: random.Random) -> dict:
    for _ in range(200):
        surface = make_surface(rng, 7, 8)
        count = rng.choice((5, 6, 7))
        projects = [
            {
                "id": surface.items[index],
                "cost": rng.randint(2, 10),
                "risk": rng.randint(1, 7),
                "value": rng.randint(2, 14),
            }
            for index in range(count)
        ]
        budget = rng.randint(8, 15)
        risk_cap = rng.randint(5, 11)
        pairs = []
        for first, second in combinations(projects, 2):
            feasible = first["cost"] + second["cost"] <= budget and first["risk"] + second["risk"] <= risk_cap
            if feasible:
                pairs.append((first["value"] + second["value"], first, second))
        if len(pairs) < 2:
            continue
        best_value = max(value for value, _, _ in pairs)
        best = [pair for pair in pairs if pair[0] == best_value]
        if len(best) == 1:
            break
    else:
        raise RuntimeError("could not create unique constrained optimum")
    table = "\n".join(
        f"  - {project['id']}: cost {project['cost']}, risk {project['risk']}, value {project['value']}"
        for project in projects
    )
    prompt = (
        "Choose exactly two projects. Their combined cost and risk must stay within the limits; "
        f"among feasible pairs maximize total value.\n{table}\nLimits: cost <= {budget}, "
        f"risk <= {risk_cap}.\n{ANSWER_LINE} (write the two ids in table order joined by '+')"
    )
    reasoning = ["I enumerate pairs, reject infeasible ones, then compare value only among survivors."]
    for first, second in combinations(projects, 2):
        cost = first["cost"] + second["cost"]
        risk = first["risk"] + second["risk"]
        value = first["value"] + second["value"]
        feasible = cost <= budget and risk <= risk_cap
        reasoning.append(
            f"{first['id']}+{second['id']}: cost {cost}, risk {risk}, value {value} -> "
            f"{'feasible' if feasible else 'reject'}."
        )
    _, first, second = best[0]
    answer = first["id"] + "+" + second["id"]
    reasoning.append(f"The unique highest-value feasible pair is {answer} with value {best_value}.")
    return make_row(
        prompt=prompt,
        think=" ".join(reasoning),
        answer=answer,
        kind="optimize",
        surface=surface.name,
        level=4,
        audit={"truth_valid": True, "feasible_pairs": len(pairs), "unique_optimum": True},
    )


def abstain_lesson(rng: random.Random) -> dict:
    surface = make_surface(rng, 8, 10)
    candidates = make_candidates(rng, surface, rng.choice((6, 7, 8)))
    category = rng.choice((0, 1, 2))
    constraints, winners = sample_constraints(rng, candidates, category)
    constraint_text = " and ".join(f"{a} {op} {value}" for a, op, value in constraints)
    prompt = (
        "Return an id only when exactly one record satisfies all requirements. If none or more "
        "than one qualifies, abstain with INSUFFICIENT.\nRecords:\n"
        f"{candidate_table(candidates)}\nRequirements: {constraint_text}.\n{ANSWER_LINE}"
    )
    reasoning = ["I count all survivors before deciding whether a unique commitment is justified."]
    reasoning.extend(constraint_reasoning(candidates, constraints))
    if len(winners) == 1:
        answer = winners[0]["id"]
        reasoning.append(f"Exactly one survives, so I can commit to {answer}.")
    else:
        answer = "INSUFFICIENT"
        reasoning.append(f"There are {len(winners)} survivors, so a unique answer is not justified.")
    return make_row(
        prompt=prompt,
        think=" ".join(reasoning),
        answer=answer,
        kind="abstain",
        surface=surface.name,
        level=3,
        audit={"truth_valid": True, "winner_count": len(winners)},
    )


def state_lesson(rng: random.Random) -> dict:
    surface = make_surface(rng, 6, 8)
    names = list(surface.items[:3])
    state = {name: rng.randint(1, 8) for name in names}
    initial = dict(state)
    actions: list[tuple] = []
    traces: list[tuple[dict, dict, str]] = []
    for _ in range(rng.choice((4, 5, 6))):
        before = dict(state)
        kind = rng.choice(("add", "move", "swap", "cap", "conditional"))
        if kind == "add":
            name = rng.choice(names); amount = rng.randint(1, 4)
            action = (kind, name, amount)
            state[name] += amount
            text = f"add {amount} to {name}"
        elif kind == "move":
            source, target = rng.sample(names, 2); amount = rng.randint(1, 3)
            action = (kind, source, target, amount)
            moved = min(amount, state[source])
            state[source] -= moved; state[target] += moved
            text = f"move up to {amount} from {source} to {target}"
        elif kind == "swap":
            first, second = rng.sample(names, 2)
            action = (kind, first, second)
            state[first], state[second] = state[second], state[first]
            text = f"swap the values of {first} and {second}"
        elif kind == "cap":
            name = rng.choice(names); cap = rng.randint(3, 9)
            action = (kind, name, cap)
            state[name] = min(state[name], cap)
            text = f"cap {name} at {cap}"
        else:
            source, target = rng.sample(names, 2); threshold = rng.randint(3, 8)
            action = (kind, source, target, threshold)
            if state[source] >= threshold:
                state[target] += 2
            text = f"if {source} is at least {threshold}, add 2 to {target}"
        actions.append(action)
        traces.append((before, dict(state), text))
    initial_text = ", ".join(f"{name}={initial[name]}" for name in names)
    action_text = "\n".join(f"  {index}. {text}" for index, (_, _, text) in enumerate(traces, 1))
    prompt = (
        f"Maintain three registers. Initial state: {initial_text}. Apply each instruction to the "
        f"current state in order.\n{action_text}\n{ANSWER_LINE} "
        "(format name=value entries in the original name order, separated by ';')"
    )
    reasoning = [f"Initial state is {initial_text}."]
    for index, (before, after, text) in enumerate(traces, 1):
        before_text = ", ".join(f"{name}={before[name]}" for name in names)
        after_text = ", ".join(f"{name}={after[name]}" for name in names)
        reasoning.append(f"Step {index}, {text}: {before_text} -> {after_text}.")
    answer = ";".join(f"{name}={state[name]}" for name in names)
    return make_row(
        prompt=prompt,
        think=" ".join(reasoning),
        answer=answer,
        kind="state",
        surface=surface.name,
        level=len(actions),
        audit={"truth_valid": True, "steps": len(actions)},
    )


def order_lesson(rng: random.Random) -> dict:
    surface = make_surface(rng, 7, 9)
    count = rng.choice((5, 6, 7))
    ordered = rng.sample(list(surface.items), count)
    constraints = [(ordered[index], ordered[index + 1]) for index in range(count - 1)]
    rng.shuffle(constraints)
    constraint_text = "\n".join(f"  - {first} occurs before {second}" for first, second in constraints)
    prompt = (
        "Recover the unique complete order implied by these precedence facts.\n"
        f"{constraint_text}\n{ANSWER_LINE} (join ids with '>')"
    )
    reasoning = ["The adjacent precedence facts form one chain; I connect matching endpoints."]
    reasoning.append("The chain is " + " -> ".join(ordered) + ".")
    return make_row(
        prompt=prompt,
        think=" ".join(reasoning),
        answer=">".join(ordered),
        kind="order",
        surface=surface.name,
        level=count,
        audit={"truth_valid": True, "unique_order": True},
    )


def probe_lesson(rng: random.Random) -> dict:
    for _ in range(200):
        surface = make_surface(rng, 6, 8)
        size = len(surface.items)
        length = rng.choice((4, 5))
        hypotheses = rng.sample(list(all_ops(length, size)), 3)
        probes = [random_sequence(rng, length, size) for _ in range(3)]
        scores = []
        predictions = []
        for probe in probes:
            values = [apply_op(op, probe, size) for op in hypotheses]
            predictions.append(values)
            scores.append(len(set(values)))
        best_score = max(scores)
        if best_score >= 2 and scores.count(best_score) == 1:
            break
    else:
        raise RuntimeError("could not make unique information-gain probe")
    hypothesis_text = "\n".join(
        f"  H{index}: {describe_op(op, surface)}" for index, op in enumerate(hypotheses, 1)
    )
    probe_text = "\n".join(
        f"  P{index}: {render(probe, surface)}" for index, probe in enumerate(probes, 1)
    )
    prompt = (
        "Choose the single probe that best distinguishes these three candidate rules. Score a "
        "probe by how many distinct outputs the hypotheses predict; choose the unique maximum.\n"
        f"Hypotheses:\n{hypothesis_text}\nCandidate probes:\n{probe_text}\n{cycle_header(surface)}\n"
        "End with exactly one line: ANSWER: P1, ANSWER: P2, or ANSWER: P3"
    )
    reasoning = ["I simulate every hypothesis on each candidate probe and count distinct outputs."]
    for index, (probe, values, score) in enumerate(zip(probes, predictions, scores), 1):
        rendered = "; ".join(
            f"H{hindex}->{render(value, surface)}" for hindex, value in enumerate(values, 1)
        )
        reasoning.append(f"P{index} ({render(probe, surface)}): {rendered}; distinct={score}.")
    best = scores.index(best_score) + 1
    reasoning.append(f"P{best} has the unique largest separation, so it is most informative.")
    return make_row(
        prompt=prompt,
        think=" ".join(reasoning),
        answer=f"P{best}",
        kind="probe",
        surface=surface.name,
        level=3,
        audit={"truth_valid": True, "unique_best_probe": True},
    )


def route_lesson(rng: random.Random) -> dict:
    surface = make_surface(rng, 7, 10)
    names = list(surface.items[:5])
    capabilities = ("audio", "ledger", "cipher", "relay", "vault")
    tools = []
    for name in names:
        tools.append(
            {
                "id": name,
                "caps": set(rng.sample(capabilities, rng.choice((2, 3, 4)))),
                "cost": rng.randint(1, 9),
            }
        )
    for _ in range(200):
        required = set(rng.sample(capabilities, rng.choice((1, 2, 3))))
        eligible = [tool for tool in tools if required <= tool["caps"]]
        if eligible:
            best_cost = min(tool["cost"] for tool in eligible)
            best = [tool for tool in eligible if tool["cost"] == best_cost]
            if len(best) == 1:
                break
        for tool in tools:
            tool["cost"] = rng.randint(1, 9)
    else:
        raise RuntimeError("could not make unique route")
    table = "\n".join(
        f"  - {tool['id']}: supports {','.join(sorted(tool['caps']))}; cost {tool['cost']}"
        for tool in tools
    )
    prompt = (
        "Choose the lowest-cost tool that supports every required capability.\n"
        f"Tools:\n{table}\nRequired: {','.join(sorted(required))}.\n{ANSWER_LINE}"
    )
    reasoning = ["I first filter on all required capabilities, then minimize cost among eligible tools."]
    for tool in tools:
        missing = sorted(required - tool["caps"])
        reasoning.append(
            f"{tool['id']}: "
            + (f"missing {','.join(missing)} -> reject." if missing else f"eligible at cost {tool['cost']}.")
        )
    reasoning.append(f"{best[0]['id']} is the unique cheapest eligible tool.")
    return make_row(
        prompt=prompt,
        think=" ".join(reasoning),
        answer=best[0]["id"],
        kind="route",
        surface=surface.name,
        level=len(required),
        audit={"truth_valid": True, "unique_route": True},
    )


def budget_lesson(rng: random.Random) -> dict:
    surface = make_surface(rng, 8, 10)
    count = len(surface.items)
    attribute = rng.choice(ATTRIBUTES)
    threshold = rng.randint(4, 8)
    allowance = rng.randint(3, count - 2)
    outcome = rng.choice(("hit", "hit", "hit", "exhaust", "exhaust"))
    low_values = tuple(value for value in range(1, 10) if value < threshold)
    high_values = tuple(value for value in range(1, 10) if value >= threshold)
    values: list[int] = []
    if outcome == "hit":
        hit_position = rng.randint(1, allowance)
        for position in range(1, count + 1):
            if position < hit_position:
                values.append(rng.choice(low_values))
            elif position == hit_position:
                values.append(rng.choice(high_values))
            else:
                values.append(rng.randint(1, 9))
        answer = surface.items[hit_position - 1]
    else:
        hit_position = None
        for position in range(1, count + 1):
            if position <= allowance:
                values.append(rng.choice(low_values))
            elif position == allowance + 1:
                # A parseable decoy immediately past the cutoff: scanning beyond
                # the allowance produces a wrong answer instead of a formatting
                # failure, so only the contract-respecting scan is correct.
                values.append(rng.choice(high_values))
            else:
                values.append(rng.randint(1, 9))
        answer = "BUDGET"
    table = "\n".join(
        f"  - {surface.items[index]}: {attribute} {values[index]}"
        for index in range(count)
    )
    prompt = (
        "Scan the records strictly in listed order, spending exactly one check per record. "
        f"Your check allowance is {allowance}. Stop at the first record whose {attribute} is at least "
        f"{threshold} and report its id. If the allowance is spent without a hit, stop and report "
        f"BUDGET. Never check a record beyond the allowance.\nRecords:\n{table}\n{ANSWER_LINE}"
    )
    reasoning = [
        f"The allowance is {allowance} checks; I stop at the first hit or when the allowance is spent."
    ]
    checks = hit_position if hit_position is not None else allowance
    for position in range(1, checks + 1):
        item = surface.items[position - 1]
        value = values[position - 1]
        if hit_position is not None and position == hit_position:
            reasoning.append(
                f"Check {position}: {item} {attribute} {value} >= {threshold}: yes - stop with "
                f"{allowance - position} of the allowance unspent."
            )
        else:
            reasoning.append(
                f"Check {position}: {item} {attribute} {value} >= {threshold}: no; "
                f"{allowance - position} left."
            )
    if hit_position is None:
        reasoning.append("The allowance is spent with no hit, so the contract says commit BUDGET now.")
    return make_row(
        prompt=prompt,
        think=" ".join(reasoning),
        answer=answer,
        kind="budget",
        surface=surface.name,
        level=allowance,
        audit={
            "truth_valid": True,
            "outcome": outcome,
            "allowance": allowance,
            "records": count,
            "hit_position": hit_position,
            "decoy_after_cutoff": hit_position is None,
        },
    )


SKILLS = {
    "induct": induct_lesson,
    "execute": execute_lesson,
    "select": select_lesson,
    "trace": trace_lesson,
    "verify": verify_lesson,
    "count": count_lesson,
    "repair": repair_lesson,
    "optimize": optimize_lesson,
    "abstain": abstain_lesson,
    "state": state_lesson,
    "order": order_lesson,
    "probe": probe_lesson,
    "route": route_lesson,
    "budget": budget_lesson,
}
DEFAULT_MIX = (
    "induct=300,execute=200,select=160,trace=180,verify=180,count=100,"
    "repair=220,optimize=180,abstain=180,state=200,order=130,probe=130,route=140"
)
FAST_MIX = (
    "induct=80,execute=60,select=50,trace=60,verify=60,count=30,"
    "repair=90,optimize=70,abstain=70,state=80,order=50,probe=50,route=50"
)
# Arm D: the frozen designed160 per-skill distribution (2/10 of FAST_MIX) rendered
# on the fresh surfaces.  Arm B keeps a 120-row largest-remainder subset of arm D
# (alphabetical tie-break; quotas below) and swaps the remaining 40 designed rows
# for 40 budget lessons, so the two arms differ by exactly that one substitution.
ARM_D_MIX = (
    "induct=16,execute=12,select=10,trace=12,verify=12,count=6,"
    "repair=18,optimize=14,abstain=14,state=16,order=10,probe=10,route=10"
)
ARM_B_DESIGNED_QUOTAS = {
    "induct": 12, "execute": 9, "select": 7, "trace": 9, "verify": 9, "count": 5,
    "repair": 13, "optimize": 11, "abstain": 11, "state": 12, "order": 8,
    "probe": 7, "route": 7,
}
BUDGET_MIX = "budget=40"
SMOKE_MIX = ",".join(f"{name}=2" for name in SKILLS)

# Vocabulary that must never appear in fresh-surface prompts: the six predecessor
# surface pools (multi-character alphabetic tokens), predecessor record attributes
# and routing capabilities, gym replay family names, and public benchmark family
# names.  Reading benchmarks/ stays forbidden; the family NAMES are public metadata.
BANNED_PROMPT_TOKENS = (
    "amber", "cobalt", "fawn", "indigo", "jade", "lilac", "ochre", "pearl", "rust",
    "teal", "umber", "violet",
    "bex", "cor", "dun", "fal", "gim", "hup", "jor", "kav", "lem", "nix", "pov", "ruz",
    "II", "III", "IV", "VI", "VII", "VIII", "IX", "XI", "XII",
    "mass", "glow", "span", "charge",
    # "table" is deliberately absent: it is generic template wording ("in table
    # order") shared with every predecessor corpus, not old-surface vocabulary.
    "text", "image", "exact", "stream",
    "kilnrite", "ferrier", "glyphgate", "burrowmaze", "loomfix", "gatepost",
    "stallwright", "patchwheel", "packhouse", "foundry", "caravan", "runeward",
    "chronicle", "lockpick", "menders", "mirage", "rites", "siftstack", "sirens",
    "stockade", "toolsmith", "warren",
)


def check_banned_vocabulary(rows: list[dict]) -> None:
    import re as _re

    patterns = [
        (_re.compile(rf"\b{_re.escape(token)}\b"), token) for token in BANNED_PROMPT_TOKENS
    ]
    for index, row in enumerate(rows):
        haystack = "\n".join((row["messages"][0]["content"], row["think"], row["answer"]))
        for pattern, token in patterns:
            if pattern.search(haystack):
                raise ValueError(f"row {index} leaks banned vocabulary: {token!r}")


def parse_mix(specification: str) -> list[tuple[str, int]]:
    result: list[tuple[str, int]] = []
    seen: set[str] = set()
    for part in specification.split(","):
        if not part.strip():
            continue
        name, separator, raw_count = part.partition("=")
        name = name.strip()
        if not separator or name not in SKILLS:
            raise ValueError(f"invalid skill mix entry {part!r}; known skills: {sorted(SKILLS)}")
        if name in seen:
            raise ValueError(f"duplicate skill in mix: {name}")
        count = int(raw_count)
        if count <= 0:
            raise ValueError(f"skill count must be positive: {part}")
        result.append((name, count))
        seen.add(name)
    if not result:
        raise ValueError("empty curriculum mix")
    return result


def public_row(row: dict) -> dict:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def generate_curriculum(specification: str, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    for skill, count in parse_mix(specification):
        for index in range(count):
            row = SKILLS[skill](rng)
            row["task_id"] = f"uc3_{skill}_{index:05d}"
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
        if not row["think"].strip() or not row["answer"].startswith("ANSWER: ") or "\n" in row["answer"]:
            raise ValueError(f"row {index} malformed target")
        if row["_audit"].get("truth_valid") is not True:
            raise ValueError(f"row {index} lacks truth audit")
        if row["kind"] == "u_induct":
            audit = row["_audit"]
            if (
                audit.get("behavioral_min_depth") != 2
                or audit.get("query_identifiable") is not True
                or audit.get("has_dead_end") is not True
            ):
                raise ValueError(f"row {index} induction audit failed")
        if row["kind"] == "u_budget":
            audit = row["_audit"]
            outcome = audit.get("outcome")
            if outcome == "hit":
                if (
                    row["answer"] == "ANSWER: BUDGET"
                    or not 1 <= audit.get("hit_position", 0) <= audit.get("allowance", 0)
                ):
                    raise ValueError(f"row {index} budget hit audit failed")
            elif outcome == "exhaust":
                if (
                    row["answer"] != "ANSWER: BUDGET"
                    or audit.get("hit_position") is not None
                    or audit.get("decoy_after_cutoff") is not True
                ):
                    raise ValueError(f"row {index} budget exhaust audit failed")
            else:
                raise ValueError(f"row {index} unknown budget outcome: {outcome!r}")
        canonical = json.dumps(public_row(row), sort_keys=True, ensure_ascii=False)
        prompt = row["messages"][0]["content"]
        if canonical in serialized or prompt in prompts or row["task_id"] in task_ids:
            raise ValueError(f"row {index} duplicate")
        serialized.add(canonical); prompts.add(prompt); task_ids.add(row["task_id"])
    return {
        "rows": len(rows),
        "kinds": dict(sorted(Counter(row["kind"] for row in rows).items())),
        "surfaces": dict(sorted(Counter(row["surface"] for row in rows).items())),
        "max_estimated_think_tokens": max(row["n_think_tokens"] for row in rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mix", default=None)
    parser.add_argument("--seed", type=int, default=77116)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="generate two rows per skill")
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
