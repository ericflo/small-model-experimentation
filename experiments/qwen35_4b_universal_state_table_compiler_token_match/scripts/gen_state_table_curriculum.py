#!/usr/bin/env python3
"""Generate truth-audited natural-language state-table curriculum rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
DEFAULT_MIX = "execute=20,score=20,repair=20,commit=20"
DEFAULT_SEED = 77112
STAGES = ("execute", "score", "repair", "commit")
OP_KINDS = (
    "reverse",
    "rotate_left",
    "rotate_right",
    "swap",
    "overwrite",
    "cycle_forward",
    "move_to_end",
    "conditional_reverse",
)
SEPARATORS = (", ", " / ", " ~ ", " | ")
SURFACE_POOLS = {
    "digits": (
        "03", "14", "25", "36", "47", "58", "69", "70", "81", "92", "17", "28",
    ),
    "letters": tuple("BEGILNOQSVXZ"),
    "romans": (
        "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII",
    ),
    "colors": (
        "azure", "bronze", "coral", "denim", "ebony", "gold", "ivory", "mauve",
        "navy", "plum", "sienna", "white",
    ),
    "syllables": (
        "dap", "fen", "gur", "hix", "jal", "keb", "lom", "miv", "nuz", "peq",
        "sor", "tav",
    ),
}


@dataclass(frozen=True)
class Surface:
    name: str
    items: tuple[str, ...]
    separator: str


@dataclass(frozen=True)
class ExecutionCase:
    surface: Surface
    initial: tuple[int, ...]
    procedure: tuple[tuple, ...]
    states: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class ScoreCase:
    surface: Surface
    hypotheses: tuple[tuple[tuple, ...], ...]
    correct_index: int
    probes: tuple[tuple[int, ...], ...]
    outputs: tuple[tuple[int, ...], ...]
    predictions: tuple[tuple[tuple[int, ...], ...], ...]
    scores: tuple[int, ...]
    query: tuple[int, ...]
    query_states: tuple[tuple[int, ...], ...]


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
    size = rng.randint(7, 10)
    if name == "nonce":
        consonants = "bcdfgklmnprstvz"
        vowels = "aeiou"
        endings = "lmnrstx"
        values: set[str] = set()
        while len(values) < size:
            values.add(rng.choice(consonants) + rng.choice(vowels) + rng.choice(endings))
        items = list(sorted(values))
        rng.shuffle(items)
    else:
        items = rng.sample(list(SURFACE_POOLS[name]), size)
    return Surface(name=name, items=tuple(items), separator=rng.choice(SEPARATORS))


def random_sequence(rng: random.Random, length: int, size: int) -> tuple[int, ...]:
    return tuple(rng.randrange(size) for _ in range(length))


def render(sequence: tuple[int, ...], surface: Surface) -> str:
    return surface.separator.join(surface.items[index] for index in sequence)


def random_op(rng: random.Random, length: int, size: int) -> tuple:
    kind = rng.choice(OP_KINDS)
    if kind == "reverse":
        return (kind,)
    if kind in {"rotate_left", "rotate_right"}:
        return (kind, rng.randint(1, min(2, length - 1)))
    if kind == "swap":
        first, second = sorted(rng.sample(range(length), 2))
        return (kind, first, second)
    if kind == "overwrite":
        return (kind, rng.randrange(length), rng.randrange(size))
    if kind == "cycle_forward":
        return (kind, rng.randint(1, min(2, size - 1)))
    if kind == "move_to_end":
        return (kind, rng.randrange(length - 1))
    first, second = sorted(rng.sample(range(size), 2))
    return (kind, first, second)


def apply_op(operation: tuple, sequence: tuple[int, ...], size: int) -> tuple[int, ...]:
    kind = operation[0]
    if kind == "reverse":
        return tuple(reversed(sequence))
    if kind == "rotate_left":
        amount = operation[1] % len(sequence)
        return sequence[amount:] + sequence[:amount]
    if kind == "rotate_right":
        amount = operation[1] % len(sequence)
        return sequence[-amount:] + sequence[:-amount]
    if kind == "swap":
        result = list(sequence)
        result[operation[1]], result[operation[2]] = result[operation[2]], result[operation[1]]
        return tuple(result)
    if kind == "overwrite":
        return sequence[: operation[1]] + (operation[2],) + sequence[operation[1] + 1 :]
    if kind == "cycle_forward":
        return tuple((value + operation[1]) % size for value in sequence)
    if kind == "move_to_end":
        result = list(sequence)
        result.append(result.pop(operation[1]))
        return tuple(result)
    if kind == "conditional_reverse":
        if sequence[0] in operation[1:]:
            return tuple(reversed(sequence))
        return sequence[1:] + sequence[:1]
    raise ValueError(f"unknown operation: {operation}")


def apply_procedure(
    procedure: tuple[tuple, ...], initial: tuple[int, ...], size: int
) -> tuple[tuple[int, ...], ...]:
    states = [initial]
    for operation in procedure:
        states.append(apply_op(operation, states[-1], size))
    return tuple(states)


def describe_op(operation: tuple, surface: Surface, variant: int) -> str:
    kind = operation[0]
    alternate = variant % 2
    if kind == "reverse":
        return (
            "write the sequence in the opposite order"
            if alternate else "reverse the complete sequence"
        )
    if kind == "rotate_left":
        amount = operation[1]
        return (
            f"move the first {amount} item(s), in order, to the end"
            if alternate else f"rotate the sequence left by {amount} place(s)"
        )
    if kind == "rotate_right":
        amount = operation[1]
        return (
            f"move the last {amount} item(s), in order, to the front"
            if alternate else f"rotate the sequence right by {amount} place(s)"
        )
    if kind == "swap":
        first, second = operation[1] + 1, operation[2] + 1
        return (
            f"exchange the items in positions {first} and {second}"
            if alternate else f"swap position {first} with position {second}"
        )
    if kind == "overwrite":
        position, value = operation[1] + 1, surface.items[operation[2]]
        return (
            f"put {value} into position {position}, replacing what was there"
            if alternate else f"overwrite the item at position {position} with {value}"
        )
    if kind == "cycle_forward":
        amount = operation[1]
        cycle = " → ".join(surface.items) + f" → {surface.items[0]}"
        return (
            f"replace every item by the item {amount} place(s) later in this cycle: {cycle}"
            if alternate
            else f"advance every item {amount} step(s) around the cycle {cycle}"
        )
    if kind == "move_to_end":
        position = operation[1] + 1
        return (
            f"remove the item in position {position} and append it at the end"
            if alternate else f"move position {position} to the end, closing the gap"
        )
    first, second = surface.items[operation[1]], surface.items[operation[2]]
    return (
        f"when the first item is {first} or {second}, reverse everything; otherwise move the first item to the end"
        if alternate
        else f"if the sequence starts with {first} or {second}, reverse it; if not, rotate left once"
    )


def make_changing_procedure(
    rng: random.Random,
    initial: tuple[int, ...],
    size: int,
    depth: int,
) -> tuple[tuple[tuple, ...], tuple[tuple[int, ...], ...]]:
    for _ in range(300):
        procedure: list[tuple] = []
        states = [initial]
        for _step in range(depth):
            operation = next(
                (
                    candidate
                    for candidate in (random_op(rng, len(initial), size) for _ in range(100))
                    if candidate not in procedure
                    and apply_op(candidate, states[-1], size) != states[-1]
                ),
                None,
            )
            if operation is None:
                break
            procedure.append(operation)
            states.append(apply_op(operation, states[-1], size))
        if len(procedure) == depth and len(set(states)) >= min(depth + 1, 4):
            return tuple(procedure), tuple(states)
    raise RuntimeError("could not build a changing variable-depth procedure")


def make_execution_case(rng: random.Random, depth: int) -> ExecutionCase:
    for _ in range(300):
        surface = make_surface(rng)
        initial = random_sequence(rng, rng.randint(5, 7), len(surface.items))
        procedure, states = make_changing_procedure(rng, initial, len(surface.items), depth)
        if states[-1] != initial:
            return ExecutionCase(surface, initial, procedure, states)
    raise RuntimeError("could not build execution case")


def mutate_procedure(
    rng: random.Random,
    procedure: tuple[tuple, ...],
    length: int,
    size: int,
) -> tuple[tuple, ...]:
    for _ in range(200):
        index = rng.randrange(len(procedure))
        replacement = random_op(rng, length, size)
        candidate = procedure[:index] + (replacement,) + procedure[index + 1 :]
        if candidate != procedure:
            return candidate
    raise RuntimeError("could not mutate procedure")


def make_score_case(rng: random.Random, depth: int, correct_index: int) -> ScoreCase:
    for _ in range(800):
        surface = make_surface(rng)
        size = len(surface.items)
        length = rng.randint(5, 7)
        seed_input = random_sequence(rng, length, size)
        true_procedure, _ = make_changing_procedure(rng, seed_input, size, depth)
        probes: list[tuple[int, ...]] = []
        while len(probes) < 5:
            candidate = random_sequence(rng, length, size)
            if candidate not in probes:
                probes.append(candidate)
        probe_tuple = tuple(probes)
        outputs = tuple(
            apply_procedure(true_procedure, probe, size)[-1] for probe in probe_tuple
        )
        false: list[tuple[tuple, ...]] = []
        false_predictions: list[tuple[tuple[int, ...], ...]] = []
        for _candidate_attempt in range(600):
            candidate = mutate_procedure(rng, true_procedure, length, size)
            predictions = tuple(
                apply_procedure(candidate, probe, size)[-1] for probe in probe_tuple
            )
            score = sum(left == right for left, right in zip(predictions, outputs))
            if (
                0 < score < len(probe_tuple)
                and candidate not in false
                and predictions not in false_predictions
            ):
                false.append(candidate)
                false_predictions.append(predictions)
            if len(false) == 2:
                break
        if len(false) != 2:
            continue
        hypotheses = list(false)
        hypotheses.insert(correct_index, true_procedure)
        prediction_rows = tuple(
            tuple(apply_procedure(hypothesis, probe, size)[-1] for probe in probe_tuple)
            for hypothesis in hypotheses
        )
        scores = tuple(
            sum(left == right for left, right in zip(predictions, outputs))
            for predictions in prediction_rows
        )
        if scores[correct_index] != 5 or sum(score == 5 for score in scores) != 1:
            continue
        query = random_sequence(rng, length, size)
        if query in probe_tuple:
            continue
        query_states = apply_procedure(true_procedure, query, size)
        if query_states[-1] == query:
            continue
        return ScoreCase(
            surface=surface,
            hypotheses=tuple(hypotheses),
            correct_index=correct_index,
            probes=probe_tuple,
            outputs=outputs,
            predictions=prediction_rows,
            scores=scores,
            query=query,
            query_states=query_states,
        )
    raise RuntimeError("could not build uniquely scored hypotheses")


def procedure_text(procedure: tuple[tuple, ...], surface: Surface, variant: int) -> str:
    return "\n".join(
        f"  {index}. {describe_op(operation, surface, variant + index)}"
        for index, operation in enumerate(procedure, 1)
    )


def state_table(
    states: tuple[tuple[int, ...], ...],
    procedure: tuple[tuple, ...],
    surface: Surface,
    variant: int,
    *,
    include_start: bool = True,
) -> str:
    lines = []
    if include_start:
        lines.append(f"  state 0 (start): {render(states[0], surface)}")
    for index, operation in enumerate(procedure, 1):
        lines.append(
            f"  state {index} after '{describe_op(operation, surface, variant + index)}': "
            f"{render(states[index], surface)}"
        )
    return "\n".join(lines)


def serialized_procedure(procedure: tuple[tuple, ...]) -> list[list[object]]:
    return [list(operation) for operation in procedure]


def serialized_states(states: tuple[tuple[int, ...], ...]) -> list[list[int]]:
    return [list(state) for state in states]


def common_row(
    *,
    stage: str,
    row_index: int,
    surface: Surface,
    depth: int,
    prompt: str,
    think: str,
    answer: str,
    audit: dict,
) -> dict:
    if "\n" in answer or not answer or not think.endswith("COMMIT ANSWER ONLY."):
        raise ValueError("every row needs one answer line and an explicit bounded commit")
    if len(think.split()) > 900:
        raise ValueError("thought exceeds the prospective bounded-state budget")
    return {
        "messages": [{"role": "user", "content": prompt}],
        "think": think,
        "answer": f"ANSWER: {answer}",
        "kind": f"u_state_table_{stage}",
        "family": "universal_state_table_compiler",
        "surface": surface.name,
        "depth": depth,
        "level": STAGES.index(stage) + 1,
        "n_think_tokens": max(1, len(think.split())),
        "row_weight": 1.0,
        "task_id": f"ustc_{stage}_{row_index:05d}",
        "_audit": {"schema_version": 1, "truth_valid": True, "stage": stage, **audit},
    }


def execute_lesson(rng: random.Random, row_index: int) -> dict:
    depth = 2 + row_index % 4
    case = make_execution_case(rng, depth)
    answer = render(case.states[-1], case.surface)
    prompt = (
        "Carry out the following natural-language procedure in order. Keep the full "
        "sequence after every instruction so that no state is lost.\n"
        f"Procedure:\n{procedure_text(case.procedure, case.surface, row_index)}\n"
        f"Starting sequence: {render(case.initial, case.surface)}\n"
        "Return only one final line in the form ANSWER: <final sequence>."
    )
    lines = [f"STATE TABLE\nstate 0 | {render(case.states[0], case.surface)} | GIVEN"]
    for index, operation in enumerate(case.procedure, 1):
        lines.append(
            f"state {index} | {describe_op(operation, case.surface, row_index + index)} | "
            f"{render(case.states[index], case.surface)} | VERIFIED"
        )
    lines.append(f"FINAL STATE VERIFIED: {answer}. COMMIT ANSWER ONLY.")
    return common_row(
        stage="execute",
        row_index=row_index,
        surface=case.surface,
        depth=depth,
        prompt=prompt,
        think="\n".join(lines),
        answer=answer,
        audit={
            "surface_items": list(case.surface.items),
            "separator": case.surface.separator,
            "initial": list(case.initial),
            "procedure": serialized_procedure(case.procedure),
            "states": serialized_states(case.states),
            "expected": answer,
        },
    )


def score_lesson(rng: random.Random, row_index: int) -> dict:
    depth = 2 + row_index % 3
    case = make_score_case(rng, depth, row_index % 3)
    hypothesis_blocks = "\n".join(
        f"Hypothesis H{index}:\n{procedure_text(procedure, case.surface, row_index + index)}"
        for index, procedure in enumerate(case.hypotheses, 1)
    )
    probe_lines = "\n".join(
        f"  probe {index}: {render(source, case.surface)} -> {render(target, case.surface)}"
        for index, (source, target) in enumerate(zip(case.probes, case.outputs), 1)
    )
    prompt = (
        "Exactly one hypothesis matches every observed input/output probe. Simulate each "
        "hypothesis independently on all five probes, count exact matches, choose the unique "
        "five-of-five hypothesis, and apply it to the query.\n"
        f"{hypothesis_blocks}\nObserved probes:\n{probe_lines}\n"
        f"Query: {render(case.query, case.surface)}\n"
        "Return only one final line in the form ANSWER: <query output>."
    )
    thoughts = ["INDEPENDENT HYPOTHESIS SCORE TABLE"]
    for hypothesis_index, predictions in enumerate(case.predictions, 1):
        thoughts.append(f"H{hypothesis_index}:")
        for probe_index, (prediction, expected) in enumerate(
            zip(predictions, case.outputs), 1
        ):
            verdict = "MATCH" if prediction == expected else "MISS"
            thoughts.append(
                f"  probe {probe_index} | predicted {render(prediction, case.surface)} | "
                f"expected {render(expected, case.surface)} | {verdict}"
            )
        thoughts.append(f"  SCORE {case.scores[hypothesis_index - 1]}/5")
    selected = case.correct_index + 1
    thoughts.append(f"UNIQUE BEST: H{selected} with 5/5.")
    thoughts.append("QUERY STATE TABLE")
    thoughts.append(f"state 0 | {render(case.query_states[0], case.surface)} | GIVEN")
    for index, operation in enumerate(case.hypotheses[case.correct_index], 1):
        thoughts.append(
            f"state {index} | {describe_op(operation, case.surface, row_index + index)} | "
            f"{render(case.query_states[index], case.surface)} | VERIFIED"
        )
    answer = render(case.query_states[-1], case.surface)
    thoughts.append(f"FINAL STATE VERIFIED: {answer}. COMMIT ANSWER ONLY.")
    return common_row(
        stage="score",
        row_index=row_index,
        surface=case.surface,
        depth=depth,
        prompt=prompt,
        think="\n".join(thoughts),
        answer=answer,
        audit={
            "surface_items": list(case.surface.items),
            "separator": case.surface.separator,
            "hypotheses": [serialized_procedure(value) for value in case.hypotheses],
            "correct_index": case.correct_index,
            "probes": serialized_states(case.probes),
            "outputs": serialized_states(case.outputs),
            "predictions": [serialized_states(value) for value in case.predictions],
            "scores": list(case.scores),
            "query": list(case.query),
            "query_states": serialized_states(case.query_states),
            "expected": answer,
        },
    )


def make_repair_draft(
    rng: random.Random, case: ExecutionCase, error_step: int
) -> tuple[tuple[int, ...], ...]:
    size = len(case.surface.items)
    for _ in range(300):
        wrong = list(case.states[error_step])
        position = rng.randrange(len(wrong))
        replacement = rng.randrange(size)
        if replacement == wrong[position]:
            continue
        wrong[position] = replacement
        draft = list(case.states[:error_step]) + [tuple(wrong)]
        for operation in case.procedure[error_step:]:
            draft.append(apply_op(operation, draft[-1], size))
        if draft[-1] != case.states[-1]:
            return tuple(draft)
    raise RuntimeError("could not make a consequential repair error")


def repair_lesson(rng: random.Random, row_index: int) -> dict:
    depth = 2 + row_index % 4
    case = make_execution_case(rng, depth)
    error_step = 1 + (row_index // 4) % depth
    draft = make_repair_draft(rng, case, error_step)
    prompt = (
        "Audit this attempted state table against the numbered natural-language procedure. "
        "The start is correct, but one transition is the first error and later rows may inherit "
        "it. Recompute from the start and return the corrected final sequence.\n"
        f"Procedure:\n{procedure_text(case.procedure, case.surface, row_index)}\n"
        f"Attempted table:\n{state_table(draft, case.procedure, case.surface, row_index)}\n"
        "Return only one final line in the form ANSWER: <corrected final sequence>."
    )
    thoughts = ["RECOMPUTED STATE TABLE"]
    thoughts.append(f"state 0 | {render(case.states[0], case.surface)} | GIVEN")
    for index, operation in enumerate(case.procedure, 1):
        verdict = "FIRST MISMATCH; REPAIR" if index == error_step else "VERIFIED"
        thoughts.append(
            f"state {index} | {describe_op(operation, case.surface, row_index + index)} | "
            f"draft {render(draft[index], case.surface)} | recomputed "
            f"{render(case.states[index], case.surface)} | {verdict}"
        )
    answer = render(case.states[-1], case.surface)
    thoughts.append(f"FINAL STATE VERIFIED: {answer}. COMMIT ANSWER ONLY.")
    return common_row(
        stage="repair",
        row_index=row_index,
        surface=case.surface,
        depth=depth,
        prompt=prompt,
        think="\n".join(thoughts),
        answer=answer,
        audit={
            "surface_items": list(case.surface.items),
            "separator": case.surface.separator,
            "initial": list(case.initial),
            "procedure": serialized_procedure(case.procedure),
            "states": serialized_states(case.states),
            "draft_states": serialized_states(draft),
            "first_error_step": error_step,
            "expected": answer,
        },
    )


def commit_lesson(rng: random.Random, row_index: int) -> dict:
    depth = 2 + row_index % 4
    case = make_execution_case(rng, depth)
    answer = render(case.states[-1], case.surface)
    prompt = (
        "The procedure has already been executed and every row below was independently "
        "verified. Read the final verified state and answer immediately; do not redo or narrate "
        "the work.\n"
        f"Verified table:\n{state_table(case.states, case.procedure, case.surface, row_index)}\n"
        "Return only one final line in the form ANSWER: <final verified sequence>."
    )
    think = f"The last verified row is {answer}; no operation remains. COMMIT ANSWER ONLY."
    return common_row(
        stage="commit",
        row_index=row_index,
        surface=case.surface,
        depth=depth,
        prompt=prompt,
        think=think,
        answer=answer,
        audit={
            "surface_items": list(case.surface.items),
            "separator": case.surface.separator,
            "initial": list(case.initial),
            "procedure": serialized_procedure(case.procedure),
            "states": serialized_states(case.states),
            "expected": answer,
        },
    )


BUILDERS = {
    "execute": execute_lesson,
    "score": score_lesson,
    "repair": repair_lesson,
    "commit": commit_lesson,
}


def operation_from_json(value: list[object]) -> tuple:
    return tuple(value)


def surface_from_audit(audit: dict) -> Surface:
    return Surface("audit", tuple(audit["surface_items"]), audit["separator"])


def validate_row(row: dict) -> None:
    audit = row["_audit"]
    if audit.get("schema_version") != 1 or audit.get("truth_valid") is not True:
        raise ValueError("row lacks a valid truth-audit identity")
    stage = audit["stage"]
    if row.get("kind") != f"u_state_table_{stage}":
        raise ValueError("stage/kind mismatch")
    surface = surface_from_audit(audit)
    expected = audit["expected"]
    if row.get("answer") != f"ANSWER: {expected}" or "\n" in row["answer"]:
        raise ValueError("answer does not match audited expected value")
    if not row.get("think", "").endswith("COMMIT ANSWER ONLY."):
        raise ValueError("missing bounded answer commit")
    if stage == "score":
        hypotheses = tuple(
            tuple(operation_from_json(operation) for operation in procedure)
            for procedure in audit["hypotheses"]
        )
        probes = tuple(tuple(value) for value in audit["probes"])
        outputs = tuple(tuple(value) for value in audit["outputs"])
        predictions = tuple(
            tuple(
                apply_procedure(hypothesis, probe, len(surface.items))[-1]
                for probe in probes
            )
            for hypothesis in hypotheses
        )
        scores = tuple(
            sum(left == right for left, right in zip(values, outputs))
            for values in predictions
        )
        correct_index = audit["correct_index"]
        query = tuple(audit["query"])
        query_states = apply_procedure(
            hypotheses[correct_index], query, len(surface.items)
        )
        if (
            [serialized_states(value) for value in predictions] != audit["predictions"]
            or list(scores) != audit["scores"]
            or scores[correct_index] != len(probes)
            or sum(score == len(probes) for score in scores) != 1
            or serialized_states(query_states) != audit["query_states"]
            or render(query_states[-1], surface) != expected
        ):
            raise ValueError("hypothesis score audit failed recomputation")
        return
    initial = tuple(audit["initial"])
    procedure = tuple(operation_from_json(value) for value in audit["procedure"])
    states = apply_procedure(procedure, initial, len(surface.items))
    if serialized_states(states) != audit["states"] or render(states[-1], surface) != expected:
        raise ValueError("state-table audit failed recomputation")
    if stage == "repair":
        draft = tuple(tuple(value) for value in audit["draft_states"])
        first_error = next(
            (index for index, (left, right) in enumerate(zip(states, draft)) if left != right),
            None,
        )
        if first_error != audit["first_error_step"] or draft[-1] == states[-1]:
            raise ValueError("repair audit lost its first consequential error")


def generate(mix: str = DEFAULT_MIX, seed: int = DEFAULT_SEED) -> list[dict]:
    counts = parse_mix(mix)
    rng = random.Random(seed)
    rows = []
    for stage in STAGES:
        for row_index in range(counts[stage]):
            row = BUILDERS[stage](rng, row_index)
            validate_row(row)
            rows.append(row)
    order = list(range(len(rows)))
    random.Random(seed + 1).shuffle(order)
    rows = [rows[index] for index in order]
    if len({row["task_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate state-table task ids")
    observed = Counter(row["kind"].removeprefix("u_state_table_") for row in rows)
    if observed != Counter(counts):
        raise ValueError(f"stage mix changed: {observed}")
    if len({row["surface"] for row in rows}) < 5:
        raise ValueError("surface diversity gate failed")
    banned = (
        "LEDGER", "FIT_SECOND", "REJECT_FIRST", "APPLY_FIRST", "EXECUTE_PAIR",
        "ADVANCE_", "SWAP_", "SET_", "ROTATE_IF_",
    )
    visible = "\n".join(
        row["messages"][0]["content"] + "\n" + row["think"] for row in rows
    )
    if any(token in visible for token in banned):
        raise ValueError("predecessor canonical scaffold vocabulary leaked")
    return rows


def render_rows(rows: list[dict]) -> bytes:
    return (
        "\n".join(json.dumps(row, sort_keys=True, ensure_ascii=False) for row in rows) + "\n"
    ).encode()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--mix", default=DEFAULT_MIX)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--out", type=Path, default=EXP / "data" / "state_table_curriculum_source.jsonl"
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rows = generate(args.mix, args.seed)
    value = render_rows(rows)
    if args.check:
        if not args.out.is_file() or args.out.read_bytes() != value:
            parser.error("state-table source is absent or changed")
    else:
        if args.out.exists():
            parser.error("refusing to overwrite state-table source")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(value)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "rows": len(rows),
                "sha256": sha256_bytes(value),
                "kinds": dict(sorted(Counter(row["kind"] for row in rows).items())),
                "depths": dict(sorted(Counter(row["depth"] for row in rows).items())),
                "surfaces": dict(sorted(Counter(row["surface"] for row in rows).items())),
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
