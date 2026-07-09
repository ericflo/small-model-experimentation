#!/usr/bin/env python3
"""Exact procedural DSL and verified-macro utilities.

The module is deliberately model-free and standard-library-only.  All programs are
straight-line lists of uppercase primitive tokens.  A macro is never executable code:
its semantics are its literal, recursively forbidden sequence of two or three base
primitives.  This makes verification exact rather than probe-based.
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import math
import random
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Callable


MODULUS = 17
LIST_LENGTH = 5
TOKEN_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


def _normalize(xs: Sequence[int]) -> list[int]:
    if isinstance(xs, (str, bytes)) or not isinstance(xs, Sequence):
        raise TypeError("DSL input must be a sequence of integers")
    if any(not isinstance(value, int) or isinstance(value, bool) for value in xs):
        raise TypeError("DSL input must contain only integers")
    return [int(value) % MODULUS for value in xs]


def _rotl(xs: list[int]) -> list[int]:
    return xs[1:] + xs[:1] if xs else []


def _swap(xs: list[int]) -> list[int]:
    result = list(xs)
    if len(result) >= 2:
        result[0], result[1] = result[1], result[0]
    return result


def _prefix(xs: list[int]) -> list[int]:
    total = 0
    result: list[int] = []
    for value in xs:
        total = (total + value) % MODULUS
        result.append(total)
    return result


def _diff(xs: list[int]) -> list[int]:
    if not xs:
        return []
    return [xs[0]] + [(xs[index] - xs[index - 1]) % MODULUS for index in range(1, len(xs))]


def _zigzag(xs: list[int]) -> list[int]:
    return xs[::2] + xs[1::2]


@dataclasses.dataclass(frozen=True)
class PrimitiveSpec:
    token: str
    description: str
    function: Callable[[list[int]], list[int]] = dataclasses.field(repr=False, compare=False)

    def apply(self, xs: Sequence[int]) -> list[int]:
        return _normalize(self.function(_normalize(xs)))


PRIMITIVES: dict[str, PrimitiveSpec] = {
    "ADD1": PrimitiveSpec("ADD1", "add 1 modulo 17 to every value", lambda xs: [(x + 1) % MODULUS for x in xs]),
    "MUL2": PrimitiveSpec("MUL2", "multiply every value by 2 modulo 17", lambda xs: [(2 * x) % MODULUS for x in xs]),
    "NEG": PrimitiveSpec("NEG", "replace every value x with -x modulo 17", lambda xs: [(-x) % MODULUS for x in xs]),
    "REV": PrimitiveSpec("REV", "reverse the list", lambda xs: list(reversed(xs))),
    "ROTL": PrimitiveSpec("ROTL", "rotate the list left by one position", _rotl),
    "SWAP": PrimitiveSpec("SWAP", "swap the first two positions", _swap),
    "SORT": PrimitiveSpec("SORT", "sort values in ascending numeric order", lambda xs: sorted(xs)),
    "PREFIX": PrimitiveSpec("PREFIX", "replace values by prefix sums modulo 17", _prefix),
    "DIFF": PrimitiveSpec("DIFF", "replace values by first differences modulo 17", _diff),
    "ZIGZAG": PrimitiveSpec("ZIGZAG", "take even-indexed positions then odd-indexed positions", _zigzag),
}


def primitive_descriptions() -> dict[str, str]:
    return {token: spec.description for token, spec in PRIMITIVES.items()}


def _validate_token(token: Any, *, allowed: set[str] | None = None) -> str:
    if not isinstance(token, str) or TOKEN_RE.fullmatch(token) is None:
        raise ValueError(f"invalid DSL token: {token!r}")
    if allowed is not None and token not in allowed:
        raise ValueError(f"token is outside the allowed inventory: {token}")
    return token


def parse_program(text: str, *, allowed_tokens: Iterable[str] | None = None) -> tuple[str, ...]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("program text must be non-empty")
    allowed = set(allowed_tokens) if allowed_tokens is not None else set(PRIMITIVES)
    parts = tuple(part.strip() for part in text.strip().split("|"))
    if not parts:
        raise ValueError("program is empty")
    return tuple(_validate_token(token, allowed=allowed) for token in parts)


def format_program(program: Sequence[str]) -> str:
    return " | ".join(_validate_token(token) for token in program)


def execute_program(program: Sequence[str], input_value: Sequence[int]) -> list[int]:
    if isinstance(program, (str, bytes)) or not isinstance(program, Sequence):
        raise TypeError("program must be a sequence of base primitive tokens")
    state = _normalize(input_value)
    for token in program:
        token = _validate_token(token, allowed=set(PRIMITIVES))
        state = PRIMITIVES[token].apply(state)
    return state


def _make_signature_probes() -> tuple[tuple[int, ...], ...]:
    structured = [
        (0, 1, 2, 3, 4),
        (4, 3, 2, 1, 0),
        (0, 0, 0, 0, 0),
        (1, 1, 2, 2, 3),
        (16, 0, 8, 4, 2),
        (3, 14, 1, 15, 9),
        (5, 4, 3, 2, 1),
        (2, 7, 1, 8, 2),
    ]
    rng = random.Random(7331)
    seen = set(structured)
    while len(structured) < 48:
        candidate = tuple(rng.randrange(MODULUS) for _ in range(LIST_LENGTH))
        if candidate not in seen:
            seen.add(candidate)
            structured.append(candidate)
    return tuple(structured)


FROZEN_SIGNATURE_PROBES = _make_signature_probes()


def behavior_vector(
    program: Sequence[str], probes: Sequence[Sequence[int]] = FROZEN_SIGNATURE_PROBES
) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(execute_program(program, probe)) for probe in probes)


def _signature_hash(vector: Sequence[Sequence[int]]) -> str:
    payload = json.dumps(vector, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def program_signature(
    program: Sequence[str], probes: Sequence[Sequence[int]] = FROZEN_SIGNATURE_PROBES
) -> str:
    return _signature_hash(behavior_vector(program, probes))


class BehavioralDepthIndex:
    """Exhaustive behavioral minimum-depth index on a frozen probe bank."""

    def __init__(
        self,
        max_depth: int = 4,
        probes: Sequence[Sequence[int]] = FROZEN_SIGNATURE_PROBES,
    ) -> None:
        if max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        self.max_depth = int(max_depth)
        self.probes = tuple(tuple(_normalize(probe)) for probe in probes)
        identity = tuple(tuple(probe) for probe in self.probes)
        self.min_depth_by_vector: dict[tuple[tuple[int, ...], ...], int] = {identity: 0}
        self.representative_by_vector: dict[tuple[tuple[int, ...], ...], tuple[str, ...]] = {
            identity: ()
        }
        frontier = [identity]
        for depth in range(1, self.max_depth + 1):
            next_frontier: list[tuple[tuple[int, ...], ...]] = []
            for vector in frontier:
                prefix = self.representative_by_vector[vector]
                for token, primitive in PRIMITIVES.items():
                    new_vector = tuple(tuple(primitive.apply(row)) for row in vector)
                    if new_vector in self.min_depth_by_vector:
                        continue
                    self.min_depth_by_vector[new_vector] = depth
                    self.representative_by_vector[new_vector] = prefix + (token,)
                    next_frontier.append(new_vector)
            frontier = next_frontier

    def minimum_depth(self, program: Sequence[str]) -> int | None:
        vector = behavior_vector(program, self.probes)
        known = self.min_depth_by_vector.get(vector)
        if known is not None:
            return known
        if len(program) == self.max_depth + 1:
            return len(program)
        return None

    def verify_exact_depth(self, program: Sequence[str], expected_depth: int) -> bool:
        if expected_depth < 1 or len(program) != expected_depth:
            return False
        if self.max_depth < expected_depth - 1:
            raise ValueError("depth index does not exhaustively cover every shorter program")
        return self.minimum_depth(program) == expected_depth


def behavioral_min_depth(
    program: Sequence[str], index: BehavioralDepthIndex | None = None
) -> int | None:
    index = index or BehavioralDepthIndex(max_depth=max(0, len(program) - 1))
    return index.minimum_depth(program)


@dataclasses.dataclass(frozen=True)
class MotifSpec:
    name: str
    expansion: tuple[str, ...]
    role: str = "reusable"


REUSABLE_MOTIFS: tuple[MotifSpec, ...] = (
    MotifSpec("REV_ADD", ("REV", "ADD1")),
    MotifSpec("ROTL_PREFIX", ("ROTL", "PREFIX")),
    MotifSpec("SWAP_MUL", ("SWAP", "MUL2")),
)


_DECOY_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("ADD1", "SORT"), ("ADD1", "PREFIX"), ("ADD1", "DIFF"),
    ("MUL2", "SORT"), ("MUL2", "DIFF"), ("NEG", "SORT"),
    ("NEG", "PREFIX"), ("NEG", "DIFF"), ("REV", "ROTL"),
    ("REV", "SWAP"), ("REV", "PREFIX"), ("REV", "DIFF"),
    ("REV", "ZIGZAG"), ("ROTL", "REV"), ("ROTL", "SWAP"),
    ("ROTL", "DIFF"), ("ROTL", "ZIGZAG"), ("SWAP", "REV"),
    ("SWAP", "PREFIX"), ("SWAP", "DIFF"), ("SWAP", "ZIGZAG"),
    ("SORT", "REV"), ("SORT", "ROTL"), ("SORT", "SWAP"),
    ("SORT", "PREFIX"), ("SORT", "DIFF"), ("SORT", "ZIGZAG"),
    ("PREFIX", "REV"), ("PREFIX", "ROTL"), ("PREFIX", "SWAP"),
    ("PREFIX", "SORT"), ("PREFIX", "ZIGZAG"), ("DIFF", "REV"),
    ("DIFF", "ROTL"), ("DIFF", "SWAP"), ("DIFF", "SORT"),
    ("DIFF", "ZIGZAG"), ("ZIGZAG", "REV"), ("ZIGZAG", "ROTL"),
    ("ZIGZAG", "SWAP"), ("ZIGZAG", "PREFIX"), ("ZIGZAG", "DIFF"),
)


def _select_decoy_motifs(count: int = 20) -> tuple[MotifSpec, ...]:
    index = BehavioralDepthIndex(max_depth=1)
    seen_vectors = {behavior_vector(motif.expansion) for motif in REUSABLE_MOTIFS}
    selected: list[MotifSpec] = []
    for expansion in _DECOY_CANDIDATES:
        vector = behavior_vector(expansion)
        if vector in seen_vectors or not index.verify_exact_depth(expansion, 2):
            continue
        seen_vectors.add(vector)
        selected.append(MotifSpec(f"DECOY_{len(selected):02d}", expansion, "decoy"))
        if len(selected) == count:
            return tuple(selected)
    raise RuntimeError(f"only found {len(selected)} nondegenerate decoy motifs")


DECOY_MOTIFS = _select_decoy_motifs()
ALL_RECURRENT_MOTIFS = REUSABLE_MOTIFS + DECOY_MOTIFS


def motif_occurrences(program: Sequence[str], motifs: Sequence[MotifSpec] = ALL_RECURRENT_MOTIFS) -> list[tuple[int, MotifSpec]]:
    program = tuple(program)
    found: list[tuple[int, MotifSpec]] = []
    for motif in motifs:
        width = len(motif.expansion)
        for start in range(len(program) - width + 1):
            if program[start : start + width] == motif.expansion:
                found.append((start, motif))
    return sorted(found, key=lambda item: (item[0], item[1].name))


@dataclasses.dataclass(frozen=True)
class TaskExample:
    input: tuple[int, ...]
    output: tuple[int, ...]

    def to_dict(self) -> dict[str, list[int]]:
        return {"input": list(self.input), "output": list(self.output)}


@dataclasses.dataclass(frozen=True)
class ProgramTask:
    id: str
    split: str
    program: tuple[str, ...]
    min_depth: int
    visible: tuple[TaskExample, ...]
    hidden: tuple[TaskExample, ...]
    probe: tuple[TaskExample, ...]
    paired_task_id: str | None
    motif_names: tuple[str, ...]
    signature: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "split": self.split,
            "program": list(self.program),
            "min_depth": self.min_depth,
            "visible": [example.to_dict() for example in self.visible],
            "hidden": [example.to_dict() for example in self.hidden],
            "probe": [example.to_dict() for example in self.probe],
            "paired_task_id": self.paired_task_id,
            "motif_names": list(self.motif_names),
            "program_signature": self.signature,
        }


@dataclasses.dataclass(frozen=True)
class TaskDataset:
    tasks: tuple[ProgramTask, ...]
    signature_probes: tuple[tuple[int, ...], ...]
    reusable_motifs: tuple[MotifSpec, ...]
    decoy_motifs: tuple[MotifSpec, ...]
    seed: int

    def by_split(self, split: str) -> tuple[ProgramTask, ...]:
        return tuple(task for task in self.tasks if task.split == split)

    def to_dict(self, dataset_manifest: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "dataset_manifest": dict(dataset_manifest or {}),
            "tasks": [task.to_dict() for task in self.tasks],
        }


def _draw_inputs(rng: random.Random, count: int) -> tuple[tuple[int, ...], ...]:
    values: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    while len(values) < count:
        candidate = tuple(rng.randrange(MODULUS) for _ in range(LIST_LENGTH))
        if candidate not in seen:
            seen.add(candidate)
            values.append(candidate)
    return tuple(values)


def _make_task(
    *,
    task_id: str,
    split: str,
    program: tuple[str, ...],
    min_depth: int,
    rng: random.Random,
    visible_count: int,
    hidden_count: int,
    probe_count: int,
    motif_names: Sequence[str],
    paired_task_id: str | None = None,
    inputs: Sequence[Sequence[int]] | None = None,
) -> ProgramTask:
    total = visible_count + hidden_count + probe_count
    input_rows = tuple(tuple(row) for row in inputs) if inputs is not None else _draw_inputs(rng, total)
    if len(input_rows) != total or len(set(input_rows)) != total:
        raise ValueError("task inputs must be unique and match the configured counts")
    examples = tuple(
        TaskExample(tuple(row), tuple(execute_program(program, row))) for row in input_rows
    )
    return ProgramTask(
        id=task_id,
        split=split,
        program=program,
        min_depth=min_depth,
        visible=examples[:visible_count],
        hidden=examples[visible_count : visible_count + hidden_count],
        probe=examples[visible_count + hidden_count :],
        paired_task_id=paired_task_id,
        motif_names=tuple(motif_names),
        signature=program_signature(program),
    )


def _candidate_reuse_programs(index: BehavioralDepthIndex, seed: int) -> list[tuple[tuple[str, ...], tuple[str, str]]]:
    candidates: list[tuple[tuple[str, ...], tuple[str, str]]] = []
    for first in REUSABLE_MOTIFS:
        for second in REUSABLE_MOTIFS:
            for filler in PRIMITIVES:
                for side in ("left", "right"):
                    block = first.expansion + second.expansion
                    program = ((filler,) + block) if side == "left" else (block + (filler,))
                    reusable_hits = motif_occurrences(program, REUSABLE_MOTIFS)
                    decoy_hits = motif_occurrences(program, DECOY_MOTIFS)
                    if len(reusable_hits) != 2 or decoy_hits:
                        continue
                    if not index.verify_exact_depth(program, 5):
                        continue
                    try:
                        _paired_no_reuse_program(
                            program,
                            index=index,
                            rng=random.Random(
                                int(hashlib.sha256(format_program(program).encode()).hexdigest()[:16], 16)
                            ),
                            forbidden_signatures=set(),
                            forbidden_programs=set(),
                        )
                    except RuntimeError:
                        continue
                    candidates.append((program, (first.name, second.name)))
    random.Random(seed).shuffle(candidates)
    # One representative per behavior; the split-level disjointness check is behavioral.
    unique: dict[str, tuple[tuple[str, ...], tuple[str, str]]] = {}
    for item in candidates:
        unique.setdefault(program_signature(item[0]), item)
    return list(unique.values())


def _paired_no_reuse_program(
    source: tuple[str, ...],
    *,
    index: BehavioralDepthIndex,
    rng: random.Random,
    forbidden_signatures: set[str],
    forbidden_programs: set[tuple[str, ...]],
) -> tuple[str, ...]:
    # Sort before seeded shuffling: set iteration is hash-seed dependent and would
    # otherwise change the entire later training corpus across Python processes.
    candidates = sorted(set(itertools.permutations(source)))
    rng.shuffle(candidates)
    for program in candidates:
        # The control excludes the three motifs designated to recur at evaluation.
        # Train-only decoy motifs may occur; this is conservative because it gives
        # composite libraries some utility on the control split too.
        if program == source or motif_occurrences(program, REUSABLE_MOTIFS):
            continue
        signature = program_signature(program)
        if signature in forbidden_signatures or program in forbidden_programs:
            continue
        if index.verify_exact_depth(program, 5):
            return program
    raise RuntimeError(f"could not make a paired no-reuse permutation for {source}")


def generate_task_dataset(
    *,
    seed: int = 20260709,
    train_programs: int = 800,
    smoke_tasks_per_split: int = 6,
    full_reuse_tasks: int = 80,
    full_no_reuse_tasks: int = 40,
    visible_examples: int = 8,
    hidden_examples: int = 8,
    probe_inputs: int = 8,
) -> TaskDataset:
    """Generate and validate the full frozen corpus before macro construction."""

    if full_no_reuse_tasks > full_reuse_tasks:
        raise ValueError("no-reuse tasks need paired reuse sources")
    if min(train_programs, smoke_tasks_per_split, full_reuse_tasks, full_no_reuse_tasks) < 1:
        raise ValueError("all dataset counts must be positive")
    rng = random.Random(seed)
    index = BehavioralDepthIndex(max_depth=4)
    reuse_candidates = _candidate_reuse_programs(index, seed + 1)
    needed_reuse = smoke_tasks_per_split + full_reuse_tasks
    paired_needed = smoke_tasks_per_split + full_no_reuse_tasks
    selected_reuse: list[tuple[tuple[str, ...], tuple[str, str]]] = []
    paired_program_by_source: dict[tuple[str, ...], tuple[str, ...]] = {}
    preused_signatures: set[str] = set()
    preused_programs: set[tuple[str, ...]] = set()
    pair_rng = random.Random(seed + 2)
    # Select paired sources and controls jointly so a control cannot later collide
    # behaviorally with another already-selected reuse task.
    for candidate in reuse_candidates:
        if len(selected_reuse) >= paired_needed:
            break
        program, _motif_names = candidate
        signature = program_signature(program)
        if program in preused_programs or signature in preused_signatures:
            continue
        try:
            control = _paired_no_reuse_program(
                program,
                index=index,
                rng=pair_rng,
                forbidden_signatures=preused_signatures | {signature},
                forbidden_programs=preused_programs | {program},
            )
        except RuntimeError:
            continue
        selected_reuse.append(candidate)
        paired_program_by_source[program] = control
        preused_programs.update((program, control))
        preused_signatures.update((signature, program_signature(control)))
    if len(selected_reuse) < paired_needed:
        raise RuntimeError(
            f"only {len(selected_reuse)} jointly fresh reuse/control pairs, need {paired_needed}"
        )
    for candidate in reuse_candidates:
        if len(selected_reuse) >= needed_reuse:
            break
        program, _motif_names = candidate
        signature = program_signature(program)
        if program in preused_programs or signature in preused_signatures:
            continue
        selected_reuse.append(candidate)
        preused_programs.add(program)
        preused_signatures.add(signature)
    if len(selected_reuse) < needed_reuse:
        raise RuntimeError(f"only {len(selected_reuse)} fresh reuse programs, need {needed_reuse}")

    used_signatures: set[str] = set()
    used_programs: set[tuple[str, ...]] = set()
    tasks: list[ProgramTask] = []

    def reserve(program: tuple[str, ...]) -> None:
        signature = program_signature(program)
        if program in used_programs or signature in used_signatures:
            raise RuntimeError("attempted to reuse a concrete or behavioral program")
        used_programs.add(program)
        used_signatures.add(signature)

    smoke_sources = selected_reuse[:smoke_tasks_per_split]
    eval_sources = selected_reuse[smoke_tasks_per_split:needed_reuse]

    for split, sources, prefix in (
        ("smoke_reuse", smoke_sources, "smoke-reuse"),
        ("reuse", eval_sources, "eval-reuse"),
    ):
        for index_in_split, (program, motif_names) in enumerate(sources):
            reserve(program)
            paired = None
            if split == "smoke_reuse":
                paired = f"smoke-no-reuse-{index_in_split:03d}"
            elif index_in_split < full_no_reuse_tasks:
                paired = f"eval-no-reuse-{index_in_split:03d}"
            task = _make_task(
                task_id=f"{prefix}-{index_in_split:03d}",
                split=split,
                program=program,
                min_depth=5,
                rng=rng,
                visible_count=visible_examples,
                hidden_count=hidden_examples,
                probe_count=probe_inputs,
                motif_names=motif_names,
                paired_task_id=paired,
            )
            tasks.append(task)

    source_lookup = {task.id: task for task in tasks}
    for split, count, source_prefix, target_prefix in (
        ("smoke_no_reuse", smoke_tasks_per_split, "smoke-reuse", "smoke-no-reuse"),
        ("no_reuse", full_no_reuse_tasks, "eval-reuse", "eval-no-reuse"),
    ):
        for index_in_split in range(count):
            source_id = f"{source_prefix}-{index_in_split:03d}"
            source_task = source_lookup[source_id]
            program = paired_program_by_source[source_task.program]
            reserve(program)
            inputs = [
                example.input
                for example in source_task.visible + source_task.hidden + source_task.probe
            ]
            tasks.append(
                _make_task(
                    task_id=f"{target_prefix}-{index_in_split:03d}",
                    split=split,
                    program=program,
                    min_depth=5,
                    rng=rng,
                    visible_count=visible_examples,
                    hidden_count=hidden_examples,
                    probe_count=probe_inputs,
                    motif_names=(),
                    paired_task_id=source_id,
                    inputs=inputs,
                )
            )

    # Training programs contain exactly one frozen recurrent motif. Reusable motifs occur
    # about twice as often individually as decoys, while remaining in the same coarse
    # support bucket used by the placebo matcher.
    max_attempts = max(100_000, train_programs * 500)
    attempts = 0
    while sum(task.split == "train" for task in tasks) < train_programs:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError("training-program rejection sampler exhausted")
        train_index = sum(task.split == "train" for task in tasks)
        if train_index % 13 < 3:
            motif = REUSABLE_MOTIFS[train_index % len(REUSABLE_MOTIFS)]
        else:
            motif = DECOY_MOTIFS[(train_index + attempts) % len(DECOY_MOTIFS)]
        depth = 4 + (train_index % 2)
        insertion = rng.randrange(depth - 1)
        fillers = [rng.choice(tuple(PRIMITIVES)) for _ in range(depth - 2)]
        program_list = fillers[:insertion] + list(motif.expansion) + fillers[insertion:]
        program = tuple(program_list)
        hits = motif_occurrences(program, ALL_RECURRENT_MOTIFS)
        if len(hits) != 1 or hits[0][1].name != motif.name:
            continue
        signature = program_signature(program)
        if program in used_programs or signature in used_signatures:
            continue
        if not index.verify_exact_depth(program, depth):
            continue
        reserve(program)
        tasks.append(
            _make_task(
                task_id=f"train-{train_index:04d}",
                split="train",
                program=program,
                min_depth=depth,
                rng=rng,
                visible_count=visible_examples,
                hidden_count=hidden_examples,
                probe_count=probe_inputs,
                motif_names=(motif.name,),
            )
        )

    dataset = TaskDataset(
        tasks=tuple(sorted(tasks, key=lambda task: task.id)),
        signature_probes=FROZEN_SIGNATURE_PROBES,
        reusable_motifs=REUSABLE_MOTIFS,
        decoy_motifs=DECOY_MOTIFS,
        seed=seed,
    )
    validate_task_dataset(dataset, depth_index=index)
    return dataset


def generate_fresh_smoke_tasks(
    *,
    exclude_tasks: Sequence[ProgramTask],
    seed: int,
    tasks_per_split: int,
    visible_examples: int,
    hidden_examples: int,
    probe_inputs: int,
    id_prefix: str = "smoke-v2",
) -> tuple[ProgramTask, ...]:
    """Construct fresh paired smoke tasks without changing a frozen dataset.

    A smoke-interface repair must not silently regenerate the still-unseen full
    evaluation.  This helper searches the same preregistered true-depth-5
    substrate while excluding every concrete program and behavioral signature
    already reserved by the frozen construction, v1 smoke, and full splits.
    """

    if tasks_per_split < 1:
        raise ValueError("tasks_per_split must be positive")
    if not id_prefix or any(character.isspace() for character in id_prefix):
        raise ValueError("id_prefix must be a non-empty token without whitespace")
    index = BehavioralDepthIndex(max_depth=4)
    forbidden_programs = {task.program for task in exclude_tasks}
    forbidden_signatures = {task.signature for task in exclude_tasks}
    pair_rng = random.Random(seed + 2)
    input_rng = random.Random(seed)
    selected: list[tuple[tuple[str, ...], tuple[str, ...], tuple[str, str]]] = []
    for program, motif_names in _candidate_reuse_programs(index, seed + 1):
        signature = program_signature(program)
        if program in forbidden_programs or signature in forbidden_signatures:
            continue
        try:
            control = _paired_no_reuse_program(
                program,
                index=index,
                rng=pair_rng,
                forbidden_signatures=forbidden_signatures | {signature},
                forbidden_programs=forbidden_programs | {program},
            )
        except RuntimeError:
            continue
        selected.append((program, control, motif_names))
        forbidden_programs.update((program, control))
        forbidden_signatures.update((signature, program_signature(control)))
        if len(selected) == tasks_per_split:
            break
    if len(selected) != tasks_per_split:
        raise RuntimeError(
            f"only {len(selected)} fresh smoke pairs remain after exclusions, "
            f"need {tasks_per_split}"
        )

    tasks: list[ProgramTask] = []
    total_examples = visible_examples + hidden_examples + probe_inputs
    for pair_index, (program, control, motif_names) in enumerate(selected):
        reuse_id = f"{id_prefix}-reuse-{pair_index:03d}"
        control_id = f"{id_prefix}-no-reuse-{pair_index:03d}"
        inputs = _draw_inputs(input_rng, total_examples)
        tasks.append(
            _make_task(
                task_id=reuse_id,
                split="smoke_reuse",
                program=program,
                min_depth=5,
                rng=input_rng,
                visible_count=visible_examples,
                hidden_count=hidden_examples,
                probe_count=probe_inputs,
                motif_names=motif_names,
                paired_task_id=control_id,
                inputs=inputs,
            )
        )
        tasks.append(
            _make_task(
                task_id=control_id,
                split="smoke_no_reuse",
                program=control,
                min_depth=5,
                rng=input_rng,
                visible_count=visible_examples,
                hidden_count=hidden_examples,
                probe_count=probe_inputs,
                motif_names=(),
                paired_task_id=reuse_id,
                inputs=inputs,
            )
        )
    return tuple(sorted(tasks, key=lambda task: task.id))


def validate_task_dataset(
    dataset: TaskDataset, *, depth_index: BehavioralDepthIndex | None = None
) -> dict[str, Any]:
    index = depth_index or BehavioralDepthIndex(max_depth=4, probes=dataset.signature_probes)
    ids: set[str] = set()
    programs: set[tuple[str, ...]] = set()
    signatures: set[str] = set()
    for task in dataset.tasks:
        if task.id in ids or task.program in programs or task.signature in signatures:
            raise ValueError("dataset contains duplicate ids, programs, or behavioral signatures")
        ids.add(task.id)
        programs.add(task.program)
        signatures.add(task.signature)
        if task.signature != program_signature(task.program, dataset.signature_probes):
            raise ValueError(f"task signature drift: {task.id}")
        if not index.verify_exact_depth(task.program, task.min_depth):
            raise ValueError(f"task is not behaviorally min-depth {task.min_depth}: {task.id}")
        examples = task.visible + task.hidden + task.probe
        if len({example.input for example in examples}) != len(examples):
            raise ValueError(f"task reuses an input across evidence partitions: {task.id}")
        for example in examples:
            if tuple(execute_program(task.program, example.input)) != example.output:
                raise ValueError(f"task output does not match its program: {task.id}")
        if task.paired_task_id is not None and task.paired_task_id not in ids and not any(
            other.id == task.paired_task_id for other in dataset.tasks
        ):
            raise ValueError(f"unknown paired task id: {task.paired_task_id}")
        if task.split in {"no_reuse", "smoke_no_reuse"} and motif_occurrences(
            task.program, REUSABLE_MOTIFS
        ):
            raise ValueError(f"no-reuse task contains a recurrent motif: {task.id}")
    return {
        "n_tasks": len(dataset.tasks),
        "split_counts": dict(sorted(Counter(task.split for task in dataset.tasks).items())),
        "unique_programs": len(programs),
        "unique_behavioral_signatures": len(signatures),
        "depth_index_max": index.max_depth,
        "signature_probe_count": len(dataset.signature_probes),
    }


def assert_signature_disjoint(*groups: Sequence[ProgramTask]) -> None:
    seen: dict[str, int] = {}
    for group_index, group in enumerate(groups):
        for task in group:
            previous = seen.get(task.signature)
            if previous is not None and previous != group_index:
                raise ValueError(
                    f"behavioral signature occurs in groups {previous} and {group_index}: {task.id}"
                )
            seen[task.signature] = group_index


@dataclasses.dataclass(frozen=True)
class MacroStats:
    expansion: tuple[str, ...]
    program_support: int
    occurrences: int

    @property
    def support(self) -> int:
        return self.program_support


@dataclasses.dataclass(frozen=True)
class Macro:
    token: str
    expansion: tuple[str, ...]
    support: int
    source_name: str | None = None

    @property
    def length(self) -> int:
        return len(self.expansion)

    def to_dict(self) -> dict[str, Any]:
        return {
            "token": self.token,
            "expansion": list(self.expansion),
            "support": self.support,
            "length": self.length,
            "source_name": self.source_name,
        }


@dataclasses.dataclass(frozen=True)
class MacroVerification:
    valid: bool
    exact: bool
    nondegenerate: bool
    min_depth: int | None
    signature: str | None
    reason: str | None = None


def collect_subsequence_stats(
    programs: Sequence[Sequence[str]], *, min_length: int = 2, max_length: int = 3
) -> dict[tuple[str, ...], MacroStats]:
    if min_length < 1 or max_length < min_length:
        raise ValueError("invalid subsequence lengths")
    occurrences: Counter[tuple[str, ...]] = Counter()
    supports: Counter[tuple[str, ...]] = Counter()
    for program_value in programs:
        program = tuple(_validate_token(token, allowed=set(PRIMITIVES)) for token in program_value)
        seen: set[tuple[str, ...]] = set()
        for width in range(min_length, max_length + 1):
            for start in range(len(program) - width + 1):
                expansion = program[start : start + width]
                occurrences[expansion] += 1
                seen.add(expansion)
        supports.update(seen)
    return {
        expansion: MacroStats(expansion, supports[expansion], count)
        for expansion, count in occurrences.items()
    }


def verify_macro(
    expansion: Sequence[str], *, depth_index: BehavioralDepthIndex | None = None
) -> MacroVerification:
    try:
        normalized = tuple(_validate_token(token, allowed=set(PRIMITIVES)) for token in expansion)
    except (TypeError, ValueError) as exc:
        return MacroVerification(False, False, False, None, None, str(exc))
    if len(normalized) not in {2, 3}:
        return MacroVerification(False, False, False, None, None, "macro must contain 2-3 base primitives")
    index = depth_index or BehavioralDepthIndex(max_depth=len(normalized) - 1)
    minimum = index.minimum_depth(normalized)
    nondegenerate = minimum == len(normalized)
    return MacroVerification(
        valid=True,
        exact=True,
        nondegenerate=nondegenerate,
        min_depth=minimum,
        signature=program_signature(normalized),
        reason=None if nondegenerate else f"behavior is representable at depth {minimum}",
    )


def _ranked_macro_stats(
    programs: Sequence[Sequence[str]], *, min_support: int, max_length: int
) -> list[MacroStats]:
    index = BehavioralDepthIndex(max_depth=max_length - 1)
    stats = collect_subsequence_stats(programs, min_length=2, max_length=max_length)
    candidates = [
        stat
        for stat in stats.values()
        if stat.support >= min_support and verify_macro(stat.expansion, depth_index=index).nondegenerate
    ]
    return sorted(
        candidates,
        key=lambda stat: (-stat.support, -stat.occurrences, len(stat.expansion), stat.expansion),
    )


def mine_frequent_macros(
    programs: Sequence[Sequence[str]], *, count: int = 8, min_support: int = 3, max_length: int = 3
) -> tuple[Macro, ...]:
    ranked = _ranked_macro_stats(programs, min_support=min_support, max_length=max_length)
    selected: list[Macro] = []
    seen_signatures: set[str] = set()
    for stat in ranked:
        signature = program_signature(stat.expansion)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        selected.append(
            Macro(f"M{len(selected)}", stat.expansion, stat.support, "DETERMINISTIC_MINER")
        )
        if len(selected) == count:
            return tuple(selected)
    raise RuntimeError(f"only {len(selected)} verified macros available, need {count}")


def _support_bucket(value: int) -> int:
    return int(math.log2(max(1, value)))


def make_frequency_matched_random_macros(
    programs: Sequence[Sequence[str]],
    target_macros: Sequence[Macro],
    *,
    seed: int,
    exclude_expansions: Iterable[Sequence[str]] = (),
    min_support: int = 3,
    max_length: int = 3,
) -> tuple[Macro, ...]:
    excluded = {tuple(expansion) for expansion in exclude_expansions}
    excluded.update(macro.expansion for macro in target_macros)
    excluded_signatures = {program_signature(expansion) for expansion in excluded}
    candidates = [
        stat
        for stat in _ranked_macro_stats(programs, min_support=min_support, max_length=max_length)
        if stat.expansion not in excluded
        and program_signature(stat.expansion) not in excluded_signatures
    ]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    chosen: list[Macro] = []
    chosen_expansions: set[tuple[str, ...]] = set()
    chosen_signatures: set[str] = set()
    for target in target_macros:
        eligible = [
            stat
            for stat in candidates
            if len(stat.expansion) == len(target.expansion)
            and stat.expansion not in chosen_expansions
            and program_signature(stat.expansion) not in chosen_signatures
        ]
        if not eligible:
            raise RuntimeError("not enough nondegenerate placebo macro candidates")
        bucket_distance = min(
            abs(_support_bucket(stat.support) - _support_bucket(target.support))
            for stat in eligible
        )
        # Draw uniformly across the nearest support bucket instead of always taking
        # the nearest exact count. The exact-count rule made nominally independent
        # placebo libraries share 7/8 entries and failed to estimate library-draw
        # variance even though dozens of preregistered bin-matched candidates exist.
        matched_bucket = [
            stat
            for stat in eligible
            if abs(_support_bucket(stat.support) - _support_bucket(target.support))
            == bucket_distance
        ]
        stat = rng.choice(matched_bucket)
        chosen_expansions.add(stat.expansion)
        chosen_signatures.add(program_signature(stat.expansion))
        chosen.append(Macro(f"M{len(chosen)}", stat.expansion, stat.support, f"PLACEBO_SEED_{seed}"))
    result = tuple(chosen)
    assert_frequency_matched(target_macros, result)
    return result


def assert_frequency_matched(target: Sequence[Macro], control: Sequence[Macro]) -> None:
    if len(target) != len(control):
        raise ValueError("macro libraries differ in entry count")
    if sorted(m.length for m in target) != sorted(m.length for m in control):
        raise ValueError("macro libraries differ in expansion-length multiset")
    distances = [
        abs(_support_bucket(left.support) - _support_bucket(right.support))
        for left, right in zip(target, control)
    ]
    if any(distance > 1 for distance in distances):
        raise ValueError(f"macro support buckets are not matched: {distances}")


def parse_macro_candidate(text: str) -> tuple[str, ...]:
    return parse_program(text, allowed_tokens=PRIMITIVES)


def parse_macro_candidates(values: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    return tuple(parse_macro_candidate(value) for value in values)


def _macro_map(macros: Mapping[str, Sequence[str]] | Sequence[Macro]) -> dict[str, tuple[str, ...]]:
    if isinstance(macros, Mapping):
        values = macros.items()
    else:
        values = ((macro.token, macro.expansion) for macro in macros)
    result: dict[str, tuple[str, ...]] = {}
    for token, expansion in values:
        token = _validate_token(token)
        if token in PRIMITIVES or token in result:
            raise ValueError(f"macro token collides: {token}")
        normalized = tuple(_validate_token(item, allowed=set(PRIMITIVES)) for item in expansion)
        if len(normalized) not in {2, 3}:
            raise ValueError("macro expansions must contain 2-3 base primitives")
        result[token] = normalized
    return result


def expand_program(
    program: Sequence[str], macros: Mapping[str, Sequence[str]] | Sequence[Macro]
) -> tuple[str, ...]:
    mapping = _macro_map(macros)
    allowed = set(PRIMITIVES) | set(mapping)
    expanded: list[str] = []
    for token in program:
        token = _validate_token(token, allowed=allowed)
        expanded.extend(mapping.get(token, (token,)))
    return tuple(expanded)


def compress_program(
    program: Sequence[str], macros: Mapping[str, Sequence[str]] | Sequence[Macro]
) -> tuple[str, ...]:
    base = tuple(_validate_token(token, allowed=set(PRIMITIVES)) for token in program)
    mapping = _macro_map(macros)
    best: list[tuple[str, ...] | None] = [None] * (len(base) + 1)
    best[0] = ()
    for end in range(1, len(base) + 1):
        candidates: list[tuple[str, ...]] = []
        if best[end - 1] is not None:
            candidates.append(best[end - 1] + (base[end - 1],))
        for token, expansion in mapping.items():
            start = end - len(expansion)
            if start >= 0 and best[start] is not None and base[start:end] == expansion:
                candidates.append(best[start] + (token,))
        best[end] = min(candidates, key=lambda item: (len(item), item))
    if best[-1] is None:
        raise RuntimeError("compression dynamic program failed")
    return best[-1]


def task_distribution_diagnostics(tasks: Sequence[ProgramTask]) -> dict[str, Any]:
    primitive_counts = Counter(token for task in tasks for token in task.program)
    distinct_counts = [len(set(task.program)) for task in tasks]
    output_unique_values = [
        len({value for example in task.visible for value in example.output}) for task in tasks
    ]
    return {
        "n_tasks": len(tasks),
        "primitive_counts": dict(sorted(primitive_counts.items())),
        "mean_distinct_ops": sum(distinct_counts) / len(distinct_counts) if tasks else None,
        "mean_visible_output_unique_values": (
            sum(output_unique_values) / len(output_unique_values) if tasks else None
        ),
        "depth_counts": dict(sorted(Counter(task.min_depth for task in tasks).items())),
    }
