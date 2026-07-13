"""Frozen prompts, exact program grammar, execution scoring, and visible selector."""

from __future__ import annotations

import hashlib
import itertools
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from task_data import (
    ALIASES,
    ALIAS_TO_OPERATION,
    CONCRETE_OPERATIONS,
    INVALID,
    BoundOperation,
    Program,
    alias_program,
    apply_pipeline,
    canonical_operation,
    canonical_program,
    operation_from_record,
)


SUFFIX_RE = re.compile(r"\APROGRAM: ([A-X]) \| ([A-X])\Z")
DIRECT_RE = re.compile(r"\APROGRAM: ([A-X]) \| ([A-X]) \| ([A-X])\Z")
TERMINALS = ("<|endoftext|>", "<|im_end|>")
SELECTOR_POLICY = "visible-probe-consensus-v1"


def _digest(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


def _validate_public(task: dict[str, Any]) -> None:
    if set(task) != {"task_id", "depth", "visible", "unlabeled_probe_inputs"}:
        raise ValueError("public task schema changed")
    if task["depth"] != 3 or not task["visible"]:
        raise ValueError("public task depth/visible rows changed")


def operation_menu(task_id: str) -> str:
    rows = [
        f"{alias} = {canonical_operation(operation)}"
        for alias, operation in zip(ALIASES, CONCRETE_OPERATIONS, strict=True)
    ]
    rows.sort(key=lambda row: _digest(task_id, "menu-v1", row))
    return "\n".join(rows)


def _visible_rows(task: dict[str, Any]) -> str:
    return "\n".join(
        f"{row['input']!r} -> {row['output']!r}" for row in task["visible"]
    )


def target_derangement(outputs: Sequence[Sequence[int]], *, salt: str) -> tuple[int, ...]:
    values = [tuple(row) for row in outputs]
    for shift in sorted(range(1, len(values)), key=lambda value: _digest(salt, str(value))):
        permutation = tuple((index + shift) % len(values) for index in range(len(values)))
        if all(values[index] != values[source] for index, source in enumerate(permutation)):
            return permutation
    for permutation in itertools.permutations(range(len(values))):
        if all(
            index != source and values[index] != values[source]
            for index, source in enumerate(permutation)
        ):
            return permutation
    raise ValueError("no target derangement exists")


def _candidate_relation(
    task: dict[str, Any], candidate: BoundOperation, *, shuffled: bool
) -> str:
    targets = [row["output"] for row in task["visible"]]
    if shuffled:
        permutation = target_derangement(
            targets,
            salt=f"{task['task_id']}\0{canonical_operation(candidate)}\0shuffled-v1",
        )
        targets = [targets[index] for index in permutation]
    rows: list[str] = []
    for row, target in zip(task["visible"], targets, strict=True):
        state = apply_pipeline(row["input"], (candidate,))
        if state is INVALID:
            raise ValueError("one-step candidate unexpectedly invalid")
        rows.append(f"{row['input']!r} -> {state!r} -> {target!r}")
    return "\n".join(rows)


def suffix_prompt(
    task: dict[str, Any], *, candidate: BoundOperation, representation: str
) -> str:
    _validate_public(task)
    if representation not in {"materialized", "name_only", "shuffled"}:
        raise ValueError("unknown residual representation")
    evidence = (
        "Original visible relation:\n" + _visible_rows(task)
        if representation == "name_only"
        else "Candidate-state-to-target relations:\n"
        + _candidate_relation(task, candidate, shuffled=representation == "shuffled")
    )
    return (
        "Infer exactly two remaining list operations after the supplied first "
        "operation. Use only aliases from the menu.\n\n"
        f"Alias menu:\n{operation_menu(task['task_id'])}\n\n"
        f"Supplied first operation: {canonical_operation(candidate)}\n"
        f"Its fixed alias: {ALIASES[CONCRETE_OPERATIONS.index(candidate)]}\n\n"
        f"{evidence}\n\n"
        "Reason privately. Return exactly this single-line shape:\n"
        "PROGRAM: <ALIAS> | <ALIAS>"
    )


def direct_prompt(task: dict[str, Any]) -> str:
    _validate_public(task)
    return (
        "Infer exactly three list operations that explain every visible row. "
        "Use only aliases from the menu.\n\n"
        f"Alias menu:\n{operation_menu(task['task_id'])}\n\n"
        f"Visible relation:\n{_visible_rows(task)}\n\n"
        "Reason privately. Return exactly this single-line shape:\n"
        "PROGRAM: <ALIAS> | <ALIAS> | <ALIAS>"
    )


def calibration_spec(task: dict[str, Any], index: int) -> dict[str, Any]:
    _validate_public(task)
    if not 0 <= index < 48:
        raise ValueError("calibration index is out of range")
    if index < 24:
        aliases = (ALIASES[index], ALIASES[(7 * index + 5) % 24])
        candidate = CONCRETE_OPERATIONS[(11 * index + 3) % 24]
        base = suffix_prompt(task, candidate=candidate, representation="materialized")
        arity = 2
    else:
        offset = index - 24
        aliases = (
            ALIASES[offset],
            ALIASES[(7 * offset + 5) % 24],
            ALIASES[(11 * offset + 3) % 24],
        )
        base = direct_prompt(task)
        arity = 3
    expected = "PROGRAM: " + " | ".join(aliases)
    prompt = (
        base
        + "\n\nInterface calibration overrides inference for this row. Echo this known "
        "answer byte-for-byte and emit nothing else:\n"
        + expected
    )
    return {
        "arity": arity,
        "aliases": list(aliases),
        "expected": expected,
        "prompt": prompt,
    }


def transport_spec(
    public: dict[str, Any], audit: dict[str, Any], index: int
) -> dict[str, Any]:
    _validate_public(public)
    if public["task_id"] != audit.get("task_id") or not audit.get("public_live"):
        raise ValueError("transport audit is missing public viability")
    live = audit["public_live"][0]
    candidate = operation_from_record(live["operation"])
    suffix = tuple(
        operation_from_record(value) for value in live["first_fitting_suffix"]
    )
    if index % 2 == 0:
        expected = "PROGRAM: " + alias_program(suffix)
        base = suffix_prompt(public, candidate=candidate, representation="materialized")
        arity = 2
    else:
        full: Program = (candidate, *suffix)
        expected = "PROGRAM: " + alias_program(full)
        base = direct_prompt(public)
        arity = 3
    return {
        "arity": arity,
        "expected": expected,
        "prompt": (
            base
            + "\n\nTransport check only: a public exhaustive visible check supplies this "
            "answer. Echo it byte-for-byte and emit nothing else:\n"
            + expected
        ),
    }


def answer_body(text: str, *, thinking_expected: bool | None = None) -> str:
    if not isinstance(text, str):
        raise ValueError("model output is not text")
    close_count = text.count("</think>")
    if close_count > 1:
        raise ValueError("multiple thinking answer boundaries")
    if thinking_expected is True and close_count != 1:
        raise ValueError("missing thinking answer boundary")
    if thinking_expected is False and close_count != 0:
        raise ValueError("unexpected thinking answer boundary")
    if close_count == 1:
        body = text.split("</think>", 1)[1]
        if not body.startswith("\n\n"):
            raise ValueError("thinking answer boundary changed")
        body = body[2:]
    else:
        body = text
    for terminal in TERMINALS:
        if body.endswith(terminal):
            body = body[: -len(terminal)]
            break
    return body


def parse_program(
    text: str, *, arity: int, thinking_expected: bool | None = None
) -> dict[str, Any]:
    if arity not in {2, 3}:
        raise ValueError("program arity must be two or three")
    try:
        body = answer_body(text, thinking_expected=thinking_expected)
    except ValueError as error:
        return {
            "parsed": False,
            "error": str(error),
            "answer_body": None,
            "program": None,
            "canonical": None,
        }
    match = (SUFFIX_RE if arity == 2 else DIRECT_RE).fullmatch(body)
    if match is None:
        return {
            "parsed": False,
            "error": "program_shape",
            "answer_body": body,
            "program": None,
            "canonical": None,
        }
    program: Program = tuple(ALIAS_TO_OPERATION[alias] for alias in match.groups())
    return {
        "parsed": True,
        "error": None,
        "answer_body": body,
        "program": program,
        "canonical": canonical_program(program),
    }


def score_echo(
    text: str,
    *,
    expected: str,
    arity: int,
    thinking_expected: bool | None = None,
) -> dict[str, Any]:
    parsed = parse_program(
        text, arity=arity, thinking_expected=thinking_expected
    )
    return {**parsed, "exact_echo": parsed["answer_body"] == expected}


def score_visible(
    task: dict[str, Any],
    *,
    text: str,
    candidate: BoundOperation | None,
    thinking_expected: bool | None = None,
) -> dict[str, Any]:
    _validate_public(task)
    parsed = parse_program(
        text,
        arity=2 if candidate is not None else 3,
        thinking_expected=thinking_expected,
    )
    if not parsed["parsed"]:
        return {
            **parsed,
            "full_program": None,
            "full_canonical": None,
            "visible_pass": False,
            "probe_vector": None,
        }
    full: Program = (
        (candidate, *parsed["program"])
        if candidate is not None
        else tuple(parsed["program"])
    )
    visible_pass = all(
        apply_pipeline(row["input"], full) == row["output"]
        for row in task["visible"]
    )
    probes: list[list[int]] | None = []
    if visible_pass:
        for values in task["unlabeled_probe_inputs"]:
            output = apply_pipeline(values, full)
            if output is INVALID:
                visible_pass = False
                probes = None
                break
            probes.append(output)
    else:
        probes = None
    return {
        **parsed,
        "full_program": full,
        "full_canonical": canonical_program(full),
        "visible_pass": visible_pass,
        "probe_vector": probes,
    }


def select_visible(
    task: dict[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    *,
    thinking_expected: bool | None = None,
) -> dict[str, Any]:
    _validate_public(task)
    ids = [row.get("candidate_id") for row in candidates]
    duplicate_ids = {
        value for value, count in Counter(ids).items() if value is not None and count > 1
    }
    scored: list[dict[str, Any]] = []
    for row in candidates:
        candidate_id = row.get("candidate_id")
        if (
            not isinstance(candidate_id, str)
            or not candidate_id
            or candidate_id in duplicate_ids
        ):
            result = {
                "parsed": False,
                "error": "candidate_id",
                "visible_pass": False,
                "probe_vector": None,
                "full_program": None,
                "full_canonical": None,
            }
        else:
            candidate = row.get("candidate")
            candidate = None if candidate is None else tuple(candidate)
            result = score_visible(
                task,
                text=row.get("text"),
                candidate=candidate,
                thinking_expected=thinking_expected,
            )
        scored.append({**dict(row), **result})
    eligible = [
        row
        for row in scored
        if row.get("visible_pass") and row.get("probe_vector") is not None
    ]
    if not eligible:
        return {
            "selector_policy": SELECTOR_POLICY,
            "abstained": True,
            "selected_candidate_id": None,
            "selected_full_canonical": None,
            "eligible_rows": 0,
            "eligible_unique_programs": 0,
            "scored": scored,
        }
    by_program: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        by_program[row["full_canonical"]].append(row)
    representatives = {
        program: min(rows, key=lambda row: _digest(task["task_id"], row["candidate_id"]))
        for program, rows in by_program.items()
    }
    clusters: dict[str, list[str]] = defaultdict(list)
    for program, row in representatives.items():
        clusters[repr(row["probe_vector"])].append(program)
    chosen_cluster = min(
        clusters,
        key=lambda key: (-len(clusters[key]), _digest(task["task_id"], "cluster", key)),
    )
    chosen_program = min(
        clusters[chosen_cluster],
        key=lambda program: (
            -len(by_program[program]),
            _digest(task["task_id"], "program", program),
        ),
    )
    chosen = representatives[chosen_program]
    return {
        "selector_policy": SELECTOR_POLICY,
        "abstained": False,
        "selected_candidate_id": chosen["candidate_id"],
        "selected_full_canonical": chosen_program,
        "eligible_rows": len(eligible),
        "eligible_unique_programs": len(by_program),
        "consensus_unique_programs": len(clusters[chosen_cluster]),
        "scored": scored,
    }


def hidden_correct(gold: dict[str, Any], program: Program | None) -> bool:
    if program is None:
        return False
    return all(
        apply_pipeline(row["input"], program) == row["output"]
        for row in gold["hidden"]
    )
