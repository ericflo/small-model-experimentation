"""Public prompts, strict alias parsers, and hidden-blind selection."""

from __future__ import annotations

import hashlib
import itertools
import json
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
)


SUFFIX_RE = re.compile(r"\APROGRAM:\s*([A-X])\s*\|\s*([A-X])\s*\Z")
DIRECT_RE = re.compile(
    r"\APROGRAM:\s*([A-X])\s*\|\s*([A-X])\s*\|\s*([A-X])\s*\Z"
)
TERMINALS = ("<|endoftext|>", "<|im_end|>")
SELECTOR_POLICY = "visible-probe-cluster-invalid-v2"


def _stable_digest(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()


def _validate_public_task(task: dict[str, Any]) -> None:
    expected = {
        "task_id",
        "depth",
        "viability_live_alias",
        "visible",
        "unlabeled_probe_inputs",
    }
    if not isinstance(task, dict) or set(task) != expected:
        raise ValueError("public task has the wrong schema")
    if not isinstance(task["task_id"], str) or not task["task_id"]:
        raise ValueError("task_id must be nonempty")
    if task["depth"] != 3:
        raise ValueError("task depth must be three")
    if task["viability_live_alias"] not in {"A", "B"}:
        raise ValueError("viability_live_alias must be A or B")
    if not isinstance(task["visible"], list) or not task["visible"]:
        raise ValueError("visible rows must be nonempty")
    for row in task["visible"]:
        if not isinstance(row, dict) or set(row) != {"input", "output"}:
            raise ValueError("visible rows require input/output")
        _validate_int_list(row["input"])
        _validate_int_list(row["output"])
    if not isinstance(task["unlabeled_probe_inputs"], list):
        raise ValueError("probe inputs must be a list")
    for values in task["unlabeled_probe_inputs"]:
        _validate_int_list(values)


def _validate_int_list(values: Any) -> None:
    if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(value, int) or isinstance(value, bool) for value in values)
    ):
        raise ValueError("expected a nonempty integer list")


def operation_menu(task_id: str) -> str:
    """Fixed alias meanings with task-permuted display order."""

    rows = [
        f"{alias} = {canonical_operation(operation)}"
        for alias, operation in zip(ALIASES, CONCRETE_OPERATIONS, strict=True)
    ]
    rows.sort(key=lambda row: _stable_digest(task_id, "menu-order-v1", row))
    return "\n".join(rows)


def _visible_rows(task: dict[str, Any]) -> str:
    return "\n".join(
        f"{row['input']!r} -> {row['output']!r}" for row in task["visible"]
    )


def target_derangement(outputs: Sequence[Sequence[int]], *, salt: str) -> tuple[int, ...]:
    """Return a deterministic index/value derangement or fail closed."""

    values = [tuple(row) for row in outputs]
    shifts = sorted(
        range(1, len(values)), key=lambda shift: _stable_digest(salt, str(shift))
    )
    for shift in shifts:
        permutation = tuple((index + shift) % len(values) for index in range(len(values)))
        if all(values[index] != values[source] for index, source in enumerate(permutation)):
            return permutation
    # Construction requires distinct targets, so this exhaustive fallback is
    # an integrity aid for hand-built test fixtures rather than a live path.
    for permutation in itertools.permutations(range(len(values))):
        if all(
            index != source and values[index] != values[source]
            for index, source in enumerate(permutation)
        ):
            return permutation
    raise ValueError("no index-and-value target derangement exists")


def _candidate_relation(
    task: dict[str, Any], candidate: BoundOperation, *, deranged: bool
) -> str:
    outputs = [row["output"] for row in task["visible"]]
    if deranged:
        permutation = target_derangement(
            outputs, salt=f"{task['task_id']}\0{canonical_operation(candidate)}\0derange-v1"
        )
        targets = [outputs[index] for index in permutation]
    else:
        targets = outputs
    rows: list[str] = []
    for row, target in zip(task["visible"], targets, strict=True):
        state = apply_pipeline(row["input"], (candidate,))
        if state is INVALID:
            raise ValueError("one-step candidate unexpectedly invalid")
        rows.append(f"{row['input']!r} -> {state!r} -> {target!r}")
    return "\n".join(rows)


def suffix_prompt(
    task: dict[str, Any],
    *,
    candidate: BoundOperation,
    representation: str,
    supplied_suffix: Sequence[BoundOperation] | None = None,
) -> str:
    _validate_public_task(task)
    if candidate not in CONCRETE_OPERATIONS:
        raise ValueError("candidate is outside the frozen bank")
    if representation not in {"materialized", "name_only", "shuffled", "echo"}:
        raise ValueError("unknown suffix representation")
    if representation == "echo":
        if supplied_suffix is None or len(tuple(supplied_suffix)) != 2:
            raise ValueError("echo requires exactly two supplied operations")
        evidence = (
            "A public exhaustive check supplies this fitting suffix. Echo it exactly:\n"
            f"PROGRAM: {alias_program(tuple(supplied_suffix))}"
        )
    elif supplied_suffix is not None:
        raise ValueError("only echo may receive a supplied suffix")
    elif representation == "name_only":
        evidence = "Original visible relation:\n" + _visible_rows(task)
    else:
        evidence = (
            "Candidate-state-to-target relations:\n"
            + _candidate_relation(
                task, candidate, deranged=representation == "shuffled"
            )
        )
    return (
        "Infer exactly two remaining list operations after the supplied first "
        "operation. Use only the aliases in the frozen menu.\n\n"
        f"Alias menu:\n{operation_menu(task['task_id'])}\n\n"
        f"Supplied first operation: {canonical_operation(candidate)}\n"
        f"Its fixed alias: {ALIASES[CONCRETE_OPERATIONS.index(candidate)]}\n\n"
        f"{evidence}\n\n"
        "Reason privately. Return exactly one final line and nothing else:\n"
        "PROGRAM: <ALIAS> | <ALIAS>"
    )


def direct_prompt(task: dict[str, Any]) -> str:
    _validate_public_task(task)
    return (
        "Infer exactly three list operations that explain every visible row. "
        "Use only the aliases in the frozen menu.\n\n"
        f"Alias menu:\n{operation_menu(task['task_id'])}\n\n"
        f"Visible relation:\n{_visible_rows(task)}\n\n"
        "Reason privately. Return exactly one final line and nothing else:\n"
        "PROGRAM: <ALIAS> | <ALIAS> | <ALIAS>"
    )


def viability_mapping(task: dict[str, Any]) -> dict[str, str]:
    live = task["viability_live_alias"]
    return {live: "LIVE", "B" if live == "A" else "A": "DEAD"}


def viability_prompt(
    task: dict[str, Any], *, candidate: BoundOperation, representation: str
) -> str:
    _validate_public_task(task)
    mapping = viability_mapping(task)
    if representation == "materialized":
        evidence = _candidate_relation(task, candidate, deranged=False)
    elif representation == "shuffled":
        evidence = _candidate_relation(task, candidate, deranged=True)
    elif representation == "name_only":
        evidence = _visible_rows(task)
    else:
        raise ValueError("unknown viability representation")
    return (
        "Classify whether the supplied first operation is LIVE: whether some "
        "exactly-two-operation suffix from the menu fits every visible target.\n\n"
        f"Alias menu:\n{operation_menu(task['task_id'])}\n\n"
        f"Candidate: {canonical_operation(candidate)}\n"
        f"Evidence:\n{evidence}\n\n"
        f"A means {mapping['A']}; B means {mapping['B']}. Return exactly A or B."
    )


def listwise_prompt(task: dict[str, Any]) -> str:
    _validate_public_task(task)
    return (
        "Choose a first-operation alias that can be followed by exactly two "
        "more allowed operations to fit every visible row.\n\n"
        f"Alias menu:\n{operation_menu(task['task_id'])}\n\n"
        f"Visible relation:\n{_visible_rows(task)}\n\n"
        "Return exactly one alias from A through X."
    )


def _answer_text(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("model output is not a string")
    answer = text.rsplit("</think>", 1)[-1].strip()
    for terminal in TERMINALS:
        if answer.endswith(terminal):
            answer = answer[: -len(terminal)].rstrip()
    return answer


def parse_program(text: str, *, arity: int) -> dict[str, Any]:
    if arity not in {2, 3}:
        raise ValueError("parser arity must be two or three")
    try:
        answer = _answer_text(text)
    except ValueError as error:
        return {"parsed": False, "error": str(error), "program": None, "canonical": None}
    match = (SUFFIX_RE if arity == 2 else DIRECT_RE).fullmatch(answer)
    if match is None:
        return {"parsed": False, "error": "program_shape", "program": None, "canonical": None}
    program: Program = tuple(ALIAS_TO_OPERATION[value] for value in match.groups())
    return {
        "parsed": True,
        "error": None,
        "program": program,
        "canonical": canonical_program(program),
    }


def assemble_suffix(candidate: BoundOperation, text: str) -> dict[str, Any]:
    parsed = parse_program(text, arity=2)
    if not parsed["parsed"]:
        return {**parsed, "full_program": None, "full_canonical": None}
    full: Program = (candidate, *parsed["program"])
    return {
        **parsed,
        "full_program": full,
        "full_canonical": canonical_program(full),
    }


def score_candidate(
    task: dict[str, Any], *, text: str, candidate: BoundOperation | None
) -> dict[str, Any]:
    _validate_public_task(task)
    if candidate is not None:
        parsed = assemble_suffix(candidate, text)
    else:
        direct = parse_program(text, arity=3)
        parsed = {
            **direct,
            "full_program": direct["program"],
            "full_canonical": direct["canonical"],
        }
    if not parsed["parsed"]:
        return {**parsed, "visible_pass": False, "probe_vector": None}
    program = parsed["full_program"]
    visible_pass = True
    for row in task["visible"]:
        state = apply_pipeline(row["input"], program)
        if state is INVALID or state != row["output"]:
            visible_pass = False
            break
    probe_vector: list[list[int]] | None = []
    if visible_pass:
        for values in task["unlabeled_probe_inputs"]:
            state = apply_pipeline(values, program)
            if state is INVALID:
                probe_vector = None
                visible_pass = False
                break
            probe_vector.append(state)
    else:
        probe_vector = None
    return {**parsed, "visible_pass": visible_pass, "probe_vector": probe_vector}


def select_visible(task: dict[str, Any], candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Freeze a candidate using no hidden outputs."""

    _validate_public_task(task)
    ids = [row.get("candidate_id") for row in candidates]
    duplicate_ids = {value for value, count in Counter(ids).items() if value is not None and count > 1}
    scored: list[dict[str, Any]] = []
    for row in candidates:
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id or candidate_id in duplicate_ids:
            result = {"parsed": False, "error": "candidate_id", "visible_pass": False, "probe_vector": None}
        else:
            candidate = row.get("candidate")
            if candidate is not None:
                candidate = tuple(candidate)
            result = score_candidate(task, text=row.get("text"), candidate=candidate)
        scored.append({**dict(row), **result})
    eligible = [row for row in scored if row.get("visible_pass") and row.get("probe_vector") is not None]
    if not eligible:
        return {
            "selector_policy": SELECTOR_POLICY,
            "abstained": True,
            "selected_candidate_id": None,
            "eligible_unique_programs": 0,
            "scored": scored,
        }
    by_program: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        by_program[row["full_canonical"]].append(row)
    representatives = [
        min(rows, key=lambda row: _stable_digest(task["task_id"], "row-v2", row["candidate_id"]))
        for rows in by_program.values()
    ]
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in representatives:
        clusters[json.dumps(row["probe_vector"], separators=(",", ":"))].append(row)
    cluster_key = min(
        clusters,
        key=lambda key: (-len(clusters[key]), _stable_digest(task["task_id"], "cluster-v2", key)),
    )
    selected = min(
        clusters[cluster_key],
        key=lambda row: _stable_digest(task["task_id"], "program-v2", row["full_canonical"]),
    )
    return {
        "selector_policy": SELECTOR_POLICY,
        "abstained": False,
        "selected_candidate_id": selected["candidate_id"],
        "selected_full_canonical": selected["full_canonical"],
        "eligible_unique_programs": len(representatives),
        "selected_cluster_size": len(clusters[cluster_key]),
        "scored": scored,
    }
