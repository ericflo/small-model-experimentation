#!/usr/bin/env python3
"""Generate truth-audited staged lessons for bounded two-operation search."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
DEFAULT_MIX = "apply=16,fit=16,reject=16,execute=16,search=16"
DEFAULT_SEED = 77111
STAGES = ("apply", "fit", "reject", "execute", "search")
OP_KINDS = ("shift", "reverse", "swap", "overwrite", "conditional_rotate")
SEPARATORS = ("-", " / ", " ~ ", " · ")
SURFACE_POOLS = {
    "digits": ("02", "13", "24", "35", "46", "57", "68", "79", "80", "91", "12", "34"),
    "letters": tuple("ACDFHJKMPRTW"),
    "romans": ("I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"),
    "colors": ("amber", "cobalt", "fawn", "indigo", "jade", "lilac", "ochre", "pearl", "rust", "teal", "umber", "violet"),
    "syllables": ("bex", "cor", "dun", "fal", "gim", "hup", "jor", "kav", "lem", "nix", "pov", "ruz"),
}


@dataclass(frozen=True)
class Surface:
    name: str
    items: tuple[str, ...]
    separator: str


@dataclass(frozen=True)
class SearchCase:
    surface: Surface
    length: int
    first: tuple
    second: tuple
    probes: tuple[tuple[int, ...], ...]
    outputs: tuple[tuple[int, ...], ...]
    query: tuple[int, ...]
    query_middle: tuple[int, ...]
    answer: tuple[int, ...]
    dead_first: tuple


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def parse_mix(value: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for component in value.split(","):
        name, separator, raw_count = component.strip().partition("=")
        if not separator or name not in STAGES or name in result:
            raise ValueError(f"invalid stage mix component: {component!r}")
        count = int(raw_count)
        if count < 0:
            raise ValueError("stage counts must be nonnegative")
        result[name] = count
    if set(result) != set(STAGES):
        raise ValueError(f"stage mix must name exactly {STAGES}")
    return result


def make_surface(rng: random.Random) -> Surface:
    name = rng.choice((*SURFACE_POOLS, "nonce"))
    size = rng.randint(6, 9)
    if name == "nonce":
        consonants = "bdgklmnprstvz"
        vowels = "aeiou"
        endings = "kmnrtx"
        values: set[str] = set()
        while len(values) < size:
            values.add(rng.choice(consonants) + rng.choice(vowels) + rng.choice(endings))
        items = sorted(values)
        rng.shuffle(items)
    else:
        items = rng.sample(list(SURFACE_POOLS[name]), size)
    return Surface(name=name, items=tuple(items), separator=rng.choice(SEPARATORS))


def random_sequence(rng: random.Random, length: int, size: int) -> tuple[int, ...]:
    return tuple(rng.randrange(size) for _ in range(length))


def render(sequence: tuple[int, ...], surface: Surface) -> str:
    return surface.separator.join(surface.items[index] for index in sequence)


def apply_op(operation: tuple, sequence: tuple[int, ...], size: int) -> tuple[int, ...]:
    kind = operation[0]
    if kind == "shift":
        return tuple((value + operation[1]) % size for value in sequence)
    if kind == "reverse":
        return tuple(reversed(sequence))
    if kind == "swap":
        result = list(sequence)
        result[operation[1]], result[operation[2]] = result[operation[2]], result[operation[1]]
        return tuple(result)
    if kind == "overwrite":
        return sequence[: operation[1]] + (operation[2],) + sequence[operation[1] + 1 :]
    if kind == "conditional_rotate":
        if sequence[0] in (operation[1], operation[2]):
            return sequence[1:] + sequence[:1]
        return sequence[-1:] + sequence[:-1]
    raise ValueError(f"unknown operation: {operation}")


def all_ops(length: int, size: int) -> tuple[tuple, ...]:
    result: list[tuple] = [("reverse",)]
    result.extend(("shift", amount) for amount in range(1, size))
    result.extend(("swap", first, second) for first, second in combinations(range(length), 2))
    result.extend(("overwrite", position, value) for position in range(length) for value in range(size))
    result.extend(
        ("conditional_rotate", first, second)
        for first, second in combinations(range(size), 2)
    )
    return tuple(result)


def random_op(rng: random.Random, length: int, size: int, *, allow_overwrite: bool = True) -> tuple:
    kinds = OP_KINDS if allow_overwrite else tuple(kind for kind in OP_KINDS if kind != "overwrite")
    kind = rng.choice(kinds)
    if kind == "shift":
        return (kind, rng.randint(1, size - 1))
    if kind == "reverse":
        return (kind,)
    if kind == "swap":
        first, second = sorted(rng.sample(range(length), 2))
        return (kind, first, second)
    if kind == "overwrite":
        return (kind, rng.randrange(length), rng.randrange(size))
    first, second = sorted(rng.sample(range(size), 2))
    return (kind, first, second)


def op_code(operation: tuple, surface: Surface) -> str:
    kind = operation[0]
    if kind == "shift":
        return f"ADVANCE_{operation[1]}"
    if kind == "reverse":
        return "REVERSE"
    if kind == "swap":
        return f"SWAP_{operation[1] + 1}_{operation[2] + 1}"
    if kind == "overwrite":
        return f"SET_{operation[1] + 1}_{surface.items[operation[2]]}"
    return f"ROTATE_IF_{surface.items[operation[1]]}_{surface.items[operation[2]]}"


def describe_op(operation: tuple, surface: Surface) -> str:
    kind = operation[0]
    if kind == "shift":
        return f"advance every item {operation[1]} step(s) in the cycle"
    if kind == "reverse":
        return "reverse the entire sequence"
    if kind == "swap":
        return f"swap positions {operation[1] + 1} and {operation[2] + 1}"
    if kind == "overwrite":
        return f"overwrite position {operation[1] + 1} with {surface.items[operation[2]]}"
    return (
        f"if the first item is {surface.items[operation[1]]} or "
        f"{surface.items[operation[2]]}, rotate left once; otherwise rotate right once"
    )


def operation_legend(surface: Surface) -> str:
    return (
        "Allowed operations use these canonical codes: ADVANCE_k advances every item k "
        "cycle steps; REVERSE reverses the sequence; SWAP_i_j swaps one-based positions; "
        "SET_i_value overwrites one position; ROTATE_IF_a_b rotates left when the first "
        "item is a or b and right otherwise. Cycle order: "
        + " ".join(surface.items)
        + "."
    )


def fitting_seconds(
    first: tuple,
    probes: tuple[tuple[int, ...], ...],
    outputs: tuple[tuple[int, ...], ...],
    length: int,
    size: int,
) -> tuple[tuple, ...]:
    middles = tuple(apply_op(first, probe, size) for probe in probes)
    return tuple(
        second
        for second in all_ops(length, size)
        if all(apply_op(second, middle, size) == output for middle, output in zip(middles, outputs))
    )


def fitting_pairs(
    probes: tuple[tuple[int, ...], ...],
    outputs: tuple[tuple[int, ...], ...],
    length: int,
    size: int,
) -> tuple[tuple[tuple, tuple], ...]:
    result = []
    for first in all_ops(length, size):
        result.extend((first, second) for second in fitting_seconds(first, probes, outputs, length, size))
    return tuple(result)


def make_case(rng: random.Random) -> SearchCase:
    for _ in range(600):
        surface = make_surface(rng)
        size = len(surface.items)
        length = rng.choice((4, 5, 6))
        first = random_op(rng, length, size, allow_overwrite=False)
        second = random_op(rng, length, size, allow_overwrite=True)
        probes: list[tuple[int, ...]] = []
        while len(probes) < 7:
            candidate = random_sequence(rng, length, size)
            if candidate not in probes:
                probes.append(candidate)
        probe_tuple = tuple(probes)
        outputs = tuple(
            apply_op(second, apply_op(first, probe, size), size) for probe in probe_tuple
        )
        pairs = fitting_pairs(probe_tuple, outputs, length, size)
        if pairs != ((first, second),):
            continue
        dead = next(
            (
                candidate
                for candidate in all_ops(length, size)
                if candidate != first
                and not fitting_seconds(candidate, probe_tuple, outputs, length, size)
            ),
            None,
        )
        if dead is None:
            continue
        query = random_sequence(rng, length, size)
        if query in probe_tuple:
            continue
        query_middle = apply_op(first, query, size)
        answer = apply_op(second, query_middle, size)
        return SearchCase(
            surface=surface,
            length=length,
            first=first,
            second=second,
            probes=probe_tuple,
            outputs=outputs,
            query=query,
            query_middle=query_middle,
            answer=answer,
            dead_first=dead,
        )
    raise RuntimeError("could not synthesize a uniquely identifiable search case")


def pair_lines(case: SearchCase) -> str:
    return "\n".join(
        f"  {render(source, case.surface)} -> {render(target, case.surface)}"
        for source, target in zip(case.probes, case.outputs)
    )


def middle_lines(case: SearchCase) -> str:
    size = len(case.surface.items)
    return "\n".join(
        f"  {render(apply_op(case.first, source, size), case.surface)} -> "
        f"{render(target, case.surface)}"
        for source, target in zip(case.probes, case.outputs)
    )


def compact_transitions(
    sources: tuple[tuple[int, ...], ...],
    operation: tuple,
    surface: Surface,
) -> str:
    size = len(surface.items)
    return " ; ".join(
        f"{render(source, surface)}->{render(apply_op(operation, source, size), surface)}"
        for source in sources
    )


def audit(case: SearchCase, stage: str, expected: str, candidate_first: tuple | None = None) -> dict:
    return {
        "truth_valid": True,
        "stage": stage,
        "surface_items": list(case.surface.items),
        "separator": case.surface.separator,
        "length": case.length,
        "first": list(case.first),
        "second": list(case.second),
        "dead_first": list(case.dead_first),
        "candidate_first": list(candidate_first) if candidate_first is not None else None,
        "probes": [list(value) for value in case.probes],
        "outputs": [list(value) for value in case.outputs],
        "query": list(case.query),
        "query_middle": list(case.query_middle),
        "answer": list(case.answer),
        "unique_fitting_pairs": 1,
        "expected": expected,
    }


def make_row(
    *,
    prompt: str,
    think: str,
    answer: str,
    stage: str,
    case: SearchCase,
    row_index: int,
    candidate_first: tuple | None = None,
) -> dict:
    if not think.endswith("COMMIT.") or "\n" in answer or not answer:
        raise ValueError("rows require a bounded COMMIT thought and one-line answer")
    return {
        "messages": [{"role": "user", "content": prompt}],
        "think": think,
        "answer": f"ANSWER: {answer}",
        "kind": f"u_scaffold_{stage}",
        "family": "universal_search_scaffold",
        "surface": case.surface.name,
        "level": STAGES.index(stage) + 1,
        "n_think_tokens": max(1, len(think) // 4),
        "row_weight": 1.0,
        "task_id": f"uss_{stage}_{row_index:05d}",
        "_audit": audit(case, stage, answer, candidate_first),
    }


def apply_lesson(case: SearchCase, row_index: int) -> dict:
    sources = case.probes[:3]
    size = len(case.surface.items)
    answers = tuple(apply_op(case.first, source, size) for source in sources)
    answer = " || ".join(render(value, case.surface) for value in answers)
    inputs = "\n".join(f"  {render(value, case.surface)}" for value in sources)
    prompt = (
        f"Apply one proposed first operation to each input.\n{operation_legend(case.surface)}\n"
        f"A = {op_code(case.first, case.surface)} ({describe_op(case.first, case.surface)}).\n"
        f"Inputs:\n{inputs}\nEnd with exactly one line: ANSWER: output1 || output2 || output3"
    )
    think = (
        f"APPLY_FIRST A={op_code(case.first, case.surface)}. "
        f"{compact_transitions(sources, case.first, case.surface)}. CHECK 3/3. COMMIT."
    )
    return make_row(
        prompt=prompt, think=think, answer=answer, stage="apply", case=case,
        row_index=row_index, candidate_first=case.first,
    )


def fit_lesson(case: SearchCase, row_index: int) -> dict:
    operations = all_ops(case.length, len(case.surface.items))
    bad = next(operation for operation in operations if operation != case.second)
    middles = tuple(
        apply_op(case.first, source, len(case.surface.items)) for source in case.probes
    )
    witness = next(
        index
        for index, (middle, output) in enumerate(zip(middles, case.outputs))
        if apply_op(bad, middle, len(case.surface.items)) != output
    )
    answer = op_code(case.second, case.surface)
    prompt = (
        f"Find the single second operation B that maps every intermediate row to its output.\n"
        f"{operation_legend(case.surface)}\nIntermediate log:\n{middle_lines(case)}\n"
        "End with exactly one line: ANSWER: <canonical operation code>"
    )
    think = (
        f"FIT_SECOND try B={op_code(bad, case.surface)}; probe {witness + 1} contradicts, "
        f"so REJECT. Try B={answer}; CHECK 7/7. UNIQUE. COMMIT."
    )
    return make_row(
        prompt=prompt, think=think, answer=answer, stage="fit", case=case,
        row_index=row_index, candidate_first=case.first,
    )


def reject_lesson(case: SearchCase, row_index: int) -> dict:
    candidate = case.first if row_index % 2 == 0 else case.dead_first
    seconds = fitting_seconds(
        candidate, case.probes, case.outputs, case.length, len(case.surface.items)
    )
    if candidate == case.first:
        if seconds != (case.second,):
            raise ValueError("fit lesson lost its unique second operation")
        answer = f"FIT | {op_code(case.second, case.surface)}"
        decision = f"B={op_code(case.second, case.surface)} CHECK 7/7; FIT"
    else:
        if seconds:
            raise ValueError("dead first operation unexpectedly fits")
        answer = "NO_FIT"
        decision = "all allowed B candidates contradict at least one probe; NO_FIT"
    prompt = (
        "Decide whether any allowed second operation B can complete the proposed first "
        f"operation A on every probe.\n{operation_legend(case.surface)}\n"
        f"A = {op_code(candidate, case.surface)} ({describe_op(candidate, case.surface)}).\n"
        f"Probe log:\n{pair_lines(case)}\nEnd with exactly one line: "
        "ANSWER: NO_FIT or ANSWER: FIT | <canonical B code>"
    )
    think = (
        f"REJECT_FIRST A={op_code(candidate, case.surface)}. MID "
        f"{compact_transitions(case.probes, candidate, case.surface)}. {decision}. COMMIT."
    )
    return make_row(
        prompt=prompt, think=think, answer=answer, stage="reject", case=case,
        row_index=row_index, candidate_first=candidate,
    )


def execute_lesson(case: SearchCase, row_index: int) -> dict:
    answer = render(case.answer, case.surface)
    prompt = (
        "Execute a known two-operation rule on the query. Carry the intermediate state "
        f"forward.\n{operation_legend(case.surface)}\n"
        f"A = {op_code(case.first, case.surface)}. B = {op_code(case.second, case.surface)}.\n"
        f"Query: {render(case.query, case.surface)}\n"
        "End with exactly one line: ANSWER: <final sequence>"
    )
    think = (
        f"EXECUTE_PAIR {render(case.query, case.surface)} --A={op_code(case.first, case.surface)}--> "
        f"{render(case.query_middle, case.surface)} --B={op_code(case.second, case.surface)}--> "
        f"{answer}. CHECK. COMMIT."
    )
    return make_row(
        prompt=prompt, think=think, answer=answer, stage="execute", case=case,
        row_index=row_index,
    )


def search_lesson(case: SearchCase, row_index: int) -> dict:
    size = len(case.surface.items)
    dead_mids = tuple(apply_op(case.dead_first, source, size) for source in case.probes)
    true_mids = tuple(apply_op(case.first, source, size) for source in case.probes)
    answer = render(case.answer, case.surface)
    prompt = (
        "A hidden rule chains exactly two allowed operations, output = B(A(input)). Find "
        "the unique pair by applying a candidate A to every probe, fitting or rejecting B, "
        f"then execute the pair on the query.\n{operation_legend(case.surface)}\n"
        f"Probe log:\n{pair_lines(case)}\nQuery: {render(case.query, case.surface)}\n"
        "Use a bounded candidate ledger. End with exactly one line: ANSWER: <final sequence>"
    )
    dead_preview = " ; ".join(
        f"{render(source, case.surface)}->{render(middle, case.surface)}"
        for source, middle in zip(case.probes[:3], dead_mids[:3])
    )
    true_preview = " ; ".join(
        f"{render(source, case.surface)}->{render(middle, case.surface)}"
        for source, middle in zip(case.probes[:3], true_mids[:3])
    )
    think = (
        f"LEDGER 1/2 A={op_code(case.dead_first, case.surface)} MID[{dead_preview}; ...]. "
        "No B fits all 7 probes: REJECT. "
        f"LEDGER 2/2 A={op_code(case.first, case.surface)} MID[{true_preview}; ...]. "
        f"B={op_code(case.second, case.surface)} CHECK 7/7 UNIQUE. QUERY "
        f"{render(case.query, case.surface)}->{render(case.query_middle, case.surface)}->{answer}. "
        "COMMIT."
    )
    return make_row(
        prompt=prompt, think=think, answer=answer, stage="search", case=case,
        row_index=row_index,
    )


BUILDERS = {
    "apply": apply_lesson,
    "fit": fit_lesson,
    "reject": reject_lesson,
    "execute": execute_lesson,
    "search": search_lesson,
}


def generate(mix: str = DEFAULT_MIX, seed: int = DEFAULT_SEED) -> list[dict]:
    counts = parse_mix(mix)
    rng = random.Random(seed)
    rows = []
    for stage in STAGES:
        for row_index in range(counts[stage]):
            rows.append(BUILDERS[stage](make_case(rng), row_index))
    order = list(range(len(rows)))
    random.Random(seed + 1).shuffle(order)
    rows = [rows[index] for index in order]
    if len({row["task_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate scaffold task ids")
    observed = Counter(row["kind"].removeprefix("u_scaffold_") for row in rows)
    if observed != Counter(counts):
        raise ValueError(f"stage mix changed: {observed}")
    if len({row["surface"] for row in rows}) < 5:
        raise ValueError("surface diversity gate failed")
    return rows


def render_rows(rows: list[dict]) -> bytes:
    return ("\n".join(json.dumps(row, sort_keys=True, ensure_ascii=False) for row in rows) + "\n").encode()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mix", default=DEFAULT_MIX)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=EXP / "data" / "search_scaffold_source.jsonl")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rows = generate(args.mix, args.seed)
    value = render_rows(rows)
    if args.check:
        if not args.out.is_file() or args.out.read_bytes() != value:
            parser.error("scaffold source is absent or changed")
    else:
        if args.out.exists():
            parser.error("refusing to overwrite scaffold source")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(value)
    print(json.dumps({
        "out": str(args.out),
        "rows": len(rows),
        "sha256": sha256_bytes(value),
        "kinds": dict(sorted(Counter(row["kind"] for row in rows).items())),
        "surfaces": dict(sorted(Counter(row["surface"] for row in rows).items())),
    }, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
