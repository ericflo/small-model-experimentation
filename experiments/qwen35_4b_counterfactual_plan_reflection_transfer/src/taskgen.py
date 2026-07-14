"""Fresh procedural tasks for a reflection-only capability-transfer experiment.

Every task is a three-primitive machine-induction problem.  The treatment target
names the ordered primitive plan but never contains the query's final answer.  The
actual deployment branch asks for the answer, so the target behavior is never an
SFT target in the reflection-only arms.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from dataclasses import dataclass
from typing import Any, Callable


State = list[int] | str


@dataclass(frozen=True)
class Primitive:
    name: str
    description: str
    apply: Callable[[State], State]


@dataclass(frozen=True)
class Family:
    name: str
    state_description: str
    primitives: tuple[Primitive, ...]
    sample: Callable[[random.Random], State]
    probes: tuple[State, ...]


def _stable_unique(xs: list[int]) -> list[int]:
    return list(dict.fromkeys(xs))


def _running_sum(xs: list[int]) -> list[int]:
    return [sum(xs[: index + 1]) for index in range(len(xs))]


def _adjacent_diff(xs: list[int]) -> list[int]:
    return [xs[index + 1] - xs[index] for index in range(len(xs) - 1)]


LIST = Family(
    name="list",
    state_description="a list of small integers",
    primitives=(
        Primitive("reverse", "reverse element order", lambda x: list(reversed(x))),
        Primitive("sort", "sort ascending", lambda x: sorted(x)),
        Primitive("unique", "keep first occurrences", lambda x: _stable_unique(x)),
        Primitive("absolute", "replace values by absolute values", lambda x: [abs(v) for v in x]),
        Primitive("square", "square every value", lambda x: [v * v for v in x]),
        Primitive("running", "replace by running sums", lambda x: _running_sum(x)),
        Primitive("difference", "take adjacent forward differences", lambda x: _adjacent_diff(x)),
        Primitive("rotate", "move the first element to the end", lambda x: x[1:] + x[:1] if x else x),
        Primitive("evens", "keep values at even zero-based positions", lambda x: x[::2]),
        Primitive("odds", "keep values at odd zero-based positions", lambda x: x[1::2]),
    ),
    sample=lambda rng: [rng.randint(-5, 5) for _ in range(rng.randint(4, 7))],
    probes=(
        [3, -1, 3, 0, -2], [-4, 2, 1, 2], [0, 1, -1, 4, -3, 2],
        [5, 4, 3, 2, 1], [-2, -2, 0, 3], [1, 3, 6, 10],
    ),
)


STRING = Family(
    name="string",
    state_description="a lowercase string",
    primitives=(
        Primitive("reverse", "reverse characters", lambda x: x[::-1]),
        Primitive("sort", "sort characters", lambda x: "".join(sorted(x))),
        Primitive("deduplicate", "remove repeated characters after their first occurrence", lambda x: "".join(dict.fromkeys(x))),
        Primitive("adjacent", "collapse adjacent repeated characters", lambda x: "".join(c for i, c in enumerate(x) if i == 0 or c != x[i - 1])),
        Primitive("vowels", "remove vowels", lambda x: "".join(c for c in x if c not in "aeiou")),
        Primitive("pairs", "reverse each consecutive character pair", lambda x: "".join(x[i:i + 2][::-1] for i in range(0, len(x), 2))),
        Primitive("evens", "keep characters at even zero-based positions", lambda x: x[::2]),
        Primitive("odds", "keep characters at odd zero-based positions", lambda x: x[1::2]),
        Primitive("rotate", "move the first character to the end", lambda x: x[1:] + x[:1] if x else x),
        Primitive("partition", "move vowels before consonants while preserving order", lambda x: "".join(c for c in x if c in "aeiou") + "".join(c for c in x if c not in "aeiou")),
    ),
    # Lengths through 13 expose distinctions among nested stride operations that
    # are behaviorally invisible when every sampled string has length at most 9.
    sample=lambda rng: "".join(rng.choice("aabbccddeeffgghhiijjklmnop") for _ in range(rng.randint(5, 13))),
    probes=("abcaef", "hheelloo", "qwerty", "mississippi", "aeioubc", "abcdefg"),
)


def _register(fn: Callable[[int, int, int], tuple[int, int, int]]) -> Callable[[State], State]:
    return lambda x: list(fn(x[0], x[1], x[2]))


REGISTER = Family(
    name="register",
    state_description="three integer registers [a, b, c]",
    primitives=(
        Primitive("swapAB", "swap registers a and b", _register(lambda a, b, c: (b, a, c))),
        Primitive("swapBC", "swap registers b and c", _register(lambda a, b, c: (a, c, b))),
        Primitive("rotate", "rotate [a,b,c] to [c,a,b]", _register(lambda a, b, c: (c, a, b))),
        Primitive("negate", "negate register a", _register(lambda a, b, c: (-a, b, c))),
        Primitive("double", "double register a", _register(lambda a, b, c: (2 * a, b, c))),
        Primitive("sumAB", "replace a by a+b", _register(lambda a, b, c: (a + b, b, c))),
        Primitive("sumBC", "replace b by b+c", _register(lambda a, b, c: (a, b + c, c))),
        Primitive("sumCA", "replace c by c+a", _register(lambda a, b, c: (a, b, c + a))),
        Primitive("absolute", "take absolute values", _register(lambda a, b, c: (abs(a), abs(b), abs(c)))),
    ),
    sample=lambda rng: [rng.randint(-6, 6) for _ in range(3)],
    probes=([-3, 2, 5], [1, -4, 2], [0, 3, -2], [5, 1, -6], [-1, -2, -3], [2, 4, 7]),
)


FAMILIES = (LIST, STRING, REGISTER)
_SIGNATURE_MULTIPLICITY: dict[str, dict[tuple[str, ...], int]] = {}


def _copy(state: State) -> State:
    return list(state) if isinstance(state, list) else state


def execute(family: Family, ops: tuple[str, ...], state: State) -> State:
    table = {primitive.name: primitive for primitive in family.primitives}
    out = _copy(state)
    for name in ops:
        out = table[name].apply(out)
        if isinstance(out, list) and (len(out) > 32 or any(abs(value) > 10**6 for value in out)):
            raise ValueError("state explosion")
        if isinstance(out, str) and len(out) > 64:
            raise ValueError("state explosion")
    return out


def _signature(family: Family, ops: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(json.dumps(execute(family, ops, probe), sort_keys=True) for probe in family.probes)


def _shallow_signatures(family: Family) -> set[tuple[str, ...]]:
    names = tuple(primitive.name for primitive in family.primitives)
    sequences = [()] + [(a,) for a in names] + [(a, b) for a in names for b in names]
    return {_signature(family, sequence) for sequence in sequences}


def _depth_three_signature_multiplicity(family: Family) -> dict[tuple[str, ...], int]:
    cached = _SIGNATURE_MULTIPLICITY.get(family.name)
    if cached is not None:
        return cached
    names = tuple(primitive.name for primitive in family.primitives)
    shallow = _shallow_signatures(family)
    counts: dict[tuple[str, ...], int] = {}
    for sequence in itertools.product(names, repeat=3):
        try:
            signature = _signature(family, sequence)
        except ValueError:
            continue
        if signature in shallow:
            continue
        counts[signature] = counts.get(signature, 0) + 1
    _SIGNATURE_MULTIPLICITY[family.name] = counts
    return counts


def _matches(
    family: Family, sequence: tuple[str, ...], inputs: list[State], outputs: list[State]
) -> bool:
    try:
        return all(execute(family, sequence, state) == expected for state, expected in zip(inputs, outputs))
    except ValueError:
        return False


def _visible_identifies_exact_plan(
    family: Family, target: tuple[str, ...], inputs: list[State], outputs: list[State]
) -> bool:
    names = tuple(primitive.name for primitive in family.primitives)
    for depth in (0, 1, 2):
        for sequence in itertools.product(names, repeat=depth):
            if _matches(family, sequence, inputs, outputs):
                return False
    matches = []
    for sequence in itertools.product(names, repeat=3):
        if _matches(family, sequence, inputs, outputs):
            matches.append(sequence)
            if len(matches) > 1:
                return False
    return matches == [target]


def _state_text(state: State) -> str:
    return json.dumps(state, separators=(",", ":"))


def _task_prompt(family: Family, examples: list[dict[str, State]], queries: list[State]) -> str:
    library = "\n".join(
        f"- {primitive.name}: {primitive.description}" for primitive in family.primitives
    )
    demonstrations = "\n".join(
        f"{index}. {_state_text(row['input'])} -> {_state_text(row['output'])}"
        for index, row in enumerate(examples, 1)
    )
    query_text = ", ".join(_state_text(query) for query in queries)
    return (
        "Study this deterministic three-step machine. Do not solve it yet; reply only READY.\n"
        f"State type: {family.state_description}.\nPrimitive library:\n{library}\n"
        f"Observed examples:\n{demonstrations}\nQuery inputs: {query_text}\n"
        "The machine applies exactly three listed primitives in a fixed order."
    )


REFLECTION_QUESTION = (
    "Pause before solving. Without calculating or stating the query outputs, identify the exact "
    "three primitives in order. Reply only `PLAN: first -> second -> third`."
)


ACTION_QUESTION = (
    "Now solve the query inputs. Reply only `ANSWER: <compact JSON array of outputs>` and no plan."
)


def _hash_signature(signature: tuple[str, ...]) -> str:
    return hashlib.sha256("\0".join(signature).encode()).hexdigest()


def _eligible_programs(family: Family) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    """Enumerate exact-depth programs with a unique global behavior spelling."""
    names = tuple(primitive.name for primitive in family.primitives)
    shallow = _shallow_signatures(family)
    multiplicity = _depth_three_signature_multiplicity(family)
    eligible = []
    for ops in itertools.product(names, repeat=3):
        try:
            signature = _signature(family, ops)
        except ValueError:
            continue
        if signature not in shallow and multiplicity.get(signature) == 1:
            eligible.append((ops, signature))
    return eligible


def _allocate_programs(
    family: Family, counts: dict[str, int], seed: int
) -> dict[str, list[tuple[tuple[str, ...], tuple[str, ...]]]]:
    """Partition the finite catalog while covering every operation in every slot.

    Coverage rows for all splits are reserved before any split is filled, preventing
    a large early split from consuming a rare operation/position combination.
    """
    catalog = _eligible_programs(family)
    required = sum(counts.values())
    if len(catalog) < required:
        raise ValueError(
            f"{family.name} has {len(catalog)} eligible programs but {required} are required"
        )
    names = tuple(primitive.name for primitive in family.primitives)
    universe = {(position, name) for position in range(3) for name in names}
    # Multiple deterministic restarts make this fail closed even if a future
    # primitive library introduces unusually rare eligible slot placements.
    for attempt in range(256):
        rng = random.Random(seed + 104_729 * attempt)
        remaining = catalog[:]
        rng.shuffle(remaining)
        selected: dict[str, list[tuple[tuple[str, ...], tuple[str, ...]]]] = {
            split: [] for split in counts
        }
        feasible = True
        for split in sorted(counts, key=lambda key: (counts[key], key)):
            # Tiny smoke splits exercise construction but cannot contain every
            # operation in every slot. Full proposed splits can and must.
            uncovered = set(universe) if counts[split] >= len(names) else set()
            while uncovered:
                scores = [
                    sum((position, name) in uncovered for position, name in enumerate(ops))
                    for ops, _ in remaining
                ]
                best = max(scores, default=0)
                if best == 0:
                    feasible = False
                    break
                choices = [index for index, score in enumerate(scores) if score == best]
                index = rng.choice(choices)
                ops, signature = remaining.pop(index)
                selected[split].append((ops, signature))
                uncovered -= {(position, name) for position, name in enumerate(ops)}
            if not feasible or len(selected[split]) > counts[split]:
                feasible = False
                break
        if not feasible:
            continue
        rng.shuffle(remaining)
        for split in counts:
            needed = counts[split] - len(selected[split])
            selected[split].extend(remaining[:needed])
            del remaining[:needed]
        return selected
    raise RuntimeError(f"unable to allocate a position-complete {family.name} catalog")


def _make_task_for_program(
    family: Family,
    ops: tuple[str, ...],
    signature: tuple[str, ...],
    split: str,
    ordinal: int,
    rng: random.Random,
) -> dict[str, Any]:
    for _ in range(2_000):
        states: list[State] = []
        seen: set[str] = set()
        for _ in range(300):
            state = family.sample(rng)
            encoded = _state_text(state)
            if encoded in seen:
                continue
            seen.add(encoded)
            states.append(state)
            if len(states) == 10:
                break
        if len(states) != 10:
            continue
        try:
            outputs = [execute(family, ops, state) for state in states]
        except ValueError:
            continue
        if len({_state_text(output) for output in outputs}) < 5:
            continue
        examples = [{"input": states[i], "output": outputs[i]} for i in range(7)]
        if not _visible_identifies_exact_plan(family, ops, states[:7], outputs[:7]):
            continue
        queries = states[7:10]
        answers = outputs[7:10]
        plan = "PLAN: " + " -> ".join(ops)
        answer = "ANSWER: " + _state_text(answers)
        if answer in plan or _state_text(answers) in plan:
            continue
        task_id = f"cprt-{split}-{family.name}-{ordinal:04d}"
        return {
            "task_id": task_id,
            "split": split,
            "family": family.name,
            "depth": 3,
            "target_ops": list(ops),
            "behavior_signature_sha256": _hash_signature(signature),
            "examples": examples,
            "queries": queries,
            "answers": answers,
            "common_messages": [
                {"role": "user", "content": _task_prompt(family, examples, queries)},
                {"role": "assistant", "content": "READY"},
            ],
            "reflection_question": REFLECTION_QUESTION,
            "action_question": ACTION_QUESTION,
            "target_plan": plan,
            "target_answer": answer,
            "visible_shallow_candidate_count": 0,
            "visible_depth3_candidate_count": 1,
        }
    raise RuntimeError(
        f"unable to materialize visible-identifiable {family.name}/{split} program {ops}"
    )


def build_corpus(counts: dict[str, int], seed: int) -> dict[str, list[dict[str, Any]]]:
    """Build disjoint splits; each count is per family."""
    corpus: dict[str, list[dict[str, Any]]] = {split: [] for split in counts}
    for family_index, family in enumerate(FAMILIES):
        allocated = _allocate_programs(family, counts, seed + 1_009 * family_index)
        for split_index, (split, programs) in enumerate(allocated.items()):
            rng = random.Random(seed + 10_007 * split_index + 1_009 * family_index)
            for ordinal, (ops, signature) in enumerate(programs):
                corpus[split].append(
                    _make_task_for_program(
                        family,
                        ops,
                        signature,
                        split,
                        ordinal,
                        rng,
                    )
                )
    return corpus


def build_reflection_arms(tasks: list[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    """Return correct and within-family deranged reflection targets."""
    correct: list[dict[str, Any]] = []
    shuffled: list[dict[str, Any]] = []
    rng = random.Random(seed)
    by_family: dict[str, list[int]] = {}
    for index, task in enumerate(tasks):
        by_family.setdefault(task["family"], []).append(index)
        row = dict(task)
        row["supervision_plan"] = task["target_plan"]
        row["supervision_ops"] = list(task["target_ops"])
        row["reflection_donor_task_id"] = task["task_id"]
        correct.append(row)
    donor: dict[int, int] = {}
    for indices in by_family.values():
        adjacency: dict[int, list[int]] = {}
        family = next(family for family in FAMILIES if family.name == tasks[indices[0]]["family"])
        for recipient in indices:
            task = tasks[recipient]
            states = [row["input"] for row in task["examples"]] + task["queries"]
            outputs = [row["output"] for row in task["examples"]] + task["answers"]
            allowed = [
                source
                for source in indices
                if source != recipient
                and tasks[source]["target_ops"] != task["target_ops"]
                and not _matches(family, tuple(tasks[source]["target_ops"]), states, outputs)
            ]
            rng.shuffle(allowed)
            adjacency[recipient] = allowed
        source_to_recipient: dict[int, int] = {}

        def augment(recipient: int, seen: set[int]) -> bool:
            for source in adjacency[recipient]:
                if source in seen:
                    continue
                seen.add(source)
                if source not in source_to_recipient or augment(source_to_recipient[source], seen):
                    source_to_recipient[source] = recipient
                    return True
            return False

        for recipient in indices:
            if not augment(recipient, set()):
                raise RuntimeError("could not construct behaviorally wrong reflection derangement")
        donor.update({recipient: source for source, recipient in source_to_recipient.items()})
    for index, task in enumerate(tasks):
        row = dict(task)
        source = tasks[donor[index]]
        row["supervision_plan"] = source["target_plan"]
        row["supervision_ops"] = list(source["target_ops"])
        row["reflection_donor_task_id"] = source["task_id"]
        shuffled.append(row)
    return {"reflection_correct": correct, "reflection_shuffled": shuffled}


def validate_corpus(
    corpus: dict[str, list[dict[str, Any]]], counts: dict[str, int]
) -> dict[str, Any]:
    rows = [row for split in counts for row in corpus[split]]
    ids = [row["task_id"] for row in rows]
    compositions = [(row["family"], tuple(row["target_ops"])) for row in rows]
    signatures = [(row["family"], row["behavior_signature_sha256"]) for row in rows]
    split_compositions = {
        split: {(row["family"], tuple(row["target_ops"])) for row in corpus[split]}
        for split in counts
    }
    split_signatures = {
        split: {(row["family"], row["behavior_signature_sha256"]) for row in corpus[split]}
        for split in counts
    }
    composition_collisions = 0
    signature_collisions = 0
    splits = list(counts)
    for left_index, left in enumerate(splits):
        for right in splits[left_index + 1:]:
            composition_collisions += len(split_compositions[left] & split_compositions[right])
            signature_collisions += len(split_signatures[left] & split_signatures[right])
    exact_answer_in_reflection = sum(
        row["target_answer"] in row["target_plan"]
        or _state_text(row["answers"]) in row["target_plan"]
        for row in rows
    )
    expected = sum(counts.values()) * len(FAMILIES)
    if len(rows) != expected:
        raise ValueError(f"expected {expected} tasks, found {len(rows)}")
    if len(set(ids)) != len(ids):
        raise ValueError("task IDs are not unique")
    if len(set(compositions)) != len(compositions):
        raise ValueError("program compositions are not unique")
    if len(set(signatures)) != len(signatures):
        raise ValueError("behavior signatures are not unique")
    if composition_collisions or signature_collisions:
        raise ValueError("cross-split program or behavior collision")
    if exact_answer_in_reflection:
        raise ValueError("a reflection target contains its exact query answer")
    for split, count in counts.items():
        for family in FAMILIES:
            family_rows = [row for row in corpus[split] if row["family"] == family.name]
            if len(family_rows) != count:
                raise ValueError(f"expected {count} {family.name}/{split} rows")
            expected_names = {primitive.name for primitive in family.primitives}
            if count >= len(expected_names):
                for position in range(3):
                    observed = {row["target_ops"][position] for row in family_rows}
                    if observed != expected_names:
                        raise ValueError(f"incomplete {family.name}/{split} slot {position} coverage")
    return {
        "families": [family.name for family in FAMILIES],
        "unique_task_ids": len(set(ids)),
        "unique_compositions": len(set(compositions)),
        "unique_behavior_signatures": len(set(signatures)),
        "cross_split_composition_collisions": composition_collisions,
        "cross_split_behavior_collisions": signature_collisions,
        "exact_answer_in_reflection_targets": exact_answer_in_reflection,
    }
