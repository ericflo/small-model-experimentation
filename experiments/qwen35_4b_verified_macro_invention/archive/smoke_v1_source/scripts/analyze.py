#!/usr/bin/env python3
"""Analyze verified-macro solver runs without loading a model.

The analyzer deliberately treats hidden examples as grading data only.  Candidate
selection is a pure function of parse/validity, visible-example score, and sample
order; hidden results are attached only after the selected sample is frozen.

Only the Python standard library is used.  DSL semantics and macro expansion come
from the experiment-local :mod:`macro_domain`, and strict answer parsing/token
accounting come from :mod:`model_harness`.
"""

from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import json
import math
import random
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean
from typing import Any


EXP = Path(__file__).resolve().parents[1]
SRC = EXP / "src"
sys.path.insert(0, str(SRC))

import macro_domain as domain  # noqa: E402
import model_harness as harness  # noqa: E402
import vllm_runner as local_vllm  # noqa: E402


PRIMARY_ARMS = ("base", "mined", "mined_hint", "designed_ceiling")
TOKEN_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _integer(value: Any, *, where: str, minimum: int = 0) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value >= minimum,
        f"{where} must be an integer >= {minimum}",
    )
    return value


def _number(value: Any, *, where: str) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{where} must be a finite number",
    )
    return float(value)


def _string(value: Any, *, where: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{where} must be a non-empty string")
    return value


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"missing required artifact: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise ValueError(f"missing required artifact: {path}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {path}:{line_number}: {exc}") from exc
        _require(isinstance(row, dict), f"{path}:{line_number} must contain a JSON object")
        rows.append(row)
    _require(bool(rows), f"{path} contains no JSONL rows")
    return rows


def _parse_yaml_scalar(text: str) -> Any:
    text = text.strip()
    if text in {"true", "True"}:
        return True
    if text in {"false", "False"}:
        return False
    if text in {"null", "None", "~"}:
        return None
    if text.startswith("[") and text.endswith("]"):
        inside = text[1:-1].strip()
        if not inside:
            return []
        return [_parse_yaml_scalar(part) for part in inside.split(",")]
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text


def load_config(path: Path) -> dict[str, Any]:
    """Parse the small mapping/inline-list YAML subset used by default.yaml."""

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        _require(indent % 2 == 0, f"{path}:{line_number} uses non-two-space indentation")
        line = raw.strip()
        _require(":" in line, f"{path}:{line_number} is not a key/value mapping")
        key, value_text = line.split(":", 1)
        _require(bool(key) and " " not in key, f"{path}:{line_number} has an invalid key")
        while stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        _require(key not in parent, f"duplicate config key at {path}:{line_number}")
        if not value_text.strip():
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_yaml_scalar(value_text)
    return root


def _primitive_tokens() -> tuple[str, ...]:
    """Return the committed primitive inventory, accepting its documented shape."""

    primitives = getattr(domain, "PRIMITIVES", None)
    if isinstance(primitives, Mapping):
        tokens = tuple(str(token) for token in primitives)
    elif isinstance(primitives, Sequence) and not isinstance(primitives, (str, bytes)):
        tokens = tuple(str(token) for token in primitives)
    else:
        raise ValueError("macro_domain.PRIMITIVES must be a mapping or sequence")
    _require(bool(tokens), "macro_domain.PRIMITIVES is empty")
    _require(len(set(tokens)) == len(tokens), "macro_domain.PRIMITIVES contains duplicates")
    _require(all(TOKEN_RE.fullmatch(token) for token in tokens), "invalid primitive token")
    return tokens


def _io_pairs(value: Any, *, where: str) -> list[dict[str, Any]]:
    _require(isinstance(value, list) and bool(value), f"{where} must be a non-empty list")
    pairs: list[dict[str, Any]] = []
    for index, pair in enumerate(value):
        _require(isinstance(pair, dict), f"{where}[{index}] must be an object")
        _require(set(("input", "output")).issubset(pair), f"{where}[{index}] needs input/output")
        # Ensure values are real JSON data, not custom Python objects.
        json.dumps(pair["input"], allow_nan=False)
        json.dumps(pair["output"], allow_nan=False)
        pairs.append({"input": pair["input"], "output": pair["output"]})
    return pairs


def _execute(program: Sequence[str], input_value: Any) -> Any:
    """Execute through macro_domain; the adapter is intentionally tiny and explicit."""

    execute = getattr(domain, "execute_program", None)
    if execute is None:
        raise ValueError("macro_domain must expose execute_program(program, input_value)")
    return execute(tuple(program), input_value)


def _expand(program: Sequence[str], macros: Mapping[str, Sequence[str]]) -> tuple[str, ...]:
    """Expand through macro_domain rather than duplicating DSL logic here."""

    expand = getattr(domain, "expand_program", None)
    if expand is None:
        raise ValueError("macro_domain must expose expand_program(program, macros)")
    expanded = expand(tuple(program), {key: tuple(value) for key, value in macros.items()})
    _require(
        isinstance(expanded, Sequence) and not isinstance(expanded, (str, bytes)),
        "macro_domain.expand_program returned a non-sequence",
    )
    return tuple(str(token) for token in expanded)


def load_tasks(path: Path) -> dict[str, dict[str, Any]]:
    payload = _read_json(path)
    _require(isinstance(payload, dict), "tasks.json top level must be an object")
    _require("schema_version" in payload, "tasks.json lacks schema_version")
    _require(isinstance(payload.get("dataset_manifest"), dict), "tasks.json lacks dataset_manifest")
    raw_tasks = payload.get("tasks")
    _require(isinstance(raw_tasks, list) and bool(raw_tasks), "tasks.json tasks must be non-empty")
    primitives = set(_primitive_tokens())
    tasks: dict[str, dict[str, Any]] = {}
    required = {
        "id",
        "split",
        "program",
        "min_depth",
        "visible",
        "hidden",
        "probe",
        "paired_task_id",
        "motif_names",
        "program_signature",
    }
    for index, raw in enumerate(raw_tasks):
        where = f"tasks[{index}]"
        _require(isinstance(raw, dict), f"{where} must be an object")
        missing = sorted(required - set(raw))
        _require(not missing, f"{where} missing fields: {missing}")
        task_id = _string(raw["id"], where=f"{where}.id")
        _require("::" not in task_id, f"{where}.id may not contain '::'")
        _require(task_id not in tasks, f"duplicate task id: {task_id}")
        split = _string(raw["split"], where=f"{where}.split")
        program = raw["program"]
        _require(isinstance(program, list) and bool(program), f"{where}.program must be non-empty")
        _require(all(token in primitives for token in program), f"{where}.program uses unknown primitive")
        min_depth = _integer(raw["min_depth"], where=f"{where}.min_depth", minimum=1)
        visible = _io_pairs(raw["visible"], where=f"{where}.visible")
        hidden = _io_pairs(raw["hidden"], where=f"{where}.hidden")
        probe = _io_pairs(raw["probe"], where=f"{where}.probe")
        paired = raw["paired_task_id"]
        _require(paired is None or isinstance(paired, str), f"{where}.paired_task_id invalid")
        motifs = raw["motif_names"]
        _require(
            isinstance(motifs, list) and all(isinstance(item, str) for item in motifs),
            f"{where}.motif_names must be strings",
        )
        signature = _string(raw["program_signature"], where=f"{where}.program_signature")
        # Freeze trust in the procedural artifact before scoring model output.
        for group_name, pairs in (("visible", visible), ("hidden", hidden), ("probe", probe)):
            for pair_index, pair in enumerate(pairs):
                actual = _execute(program, pair["input"])
                _require(
                    actual == pair["output"],
                    f"{where}.{group_name}[{pair_index}] disagrees with target program",
                )
        tasks[task_id] = {
            "id": task_id,
            "split": split,
            "program": tuple(program),
            "min_depth": min_depth,
            "visible": visible,
            "hidden": hidden,
            "probe": probe,
            "paired_task_id": paired,
            "motif_names": list(motifs),
            "program_signature": signature,
        }
    for task in tasks.values():
        paired = task["paired_task_id"]
        _require(paired is None or paired in tasks, f"task {task['id']} has unknown paired_task_id")
    return tasks


def load_libraries(path: Path) -> dict[str, dict[str, Any]]:
    payload = _read_json(path)
    _require(isinstance(payload, dict), "libraries.json top level must be an object")
    _require("schema_version" in payload, "libraries.json lacks schema_version")
    raw_libraries = payload.get("libraries")
    _require(isinstance(raw_libraries, dict) and bool(raw_libraries), "libraries must be an object")
    primitives = set(_primitive_tokens())
    libraries: dict[str, dict[str, Any]] = {}
    seen_library_ids: set[str] = set()
    for arm, raw in raw_libraries.items():
        where = f"libraries.{arm}"
        _string(arm, where="library arm")
        _require(isinstance(raw, dict), f"{where} must be an object")
        for key in ("id", "provenance", "macros"):
            _require(key in raw, f"{where} lacks {key}")
        library_id = _string(raw["id"], where=f"{where}.id")
        _require(library_id not in seen_library_ids, f"duplicate library id: {library_id}")
        seen_library_ids.add(library_id)
        provenance = _string(raw["provenance"], where=f"{where}.provenance")
        raw_macros = raw["macros"]
        _require(isinstance(raw_macros, list), f"{where}.macros must be a list")
        macros: list[dict[str, Any]] = []
        seen_tokens: set[str] = set()
        for index, macro in enumerate(raw_macros):
            mwhere = f"{where}.macros[{index}]"
            _require(isinstance(macro, dict), f"{mwhere} must be an object")
            for key in ("token", "expansion", "support", "length"):
                _require(key in macro, f"{mwhere} lacks {key}")
            token = _string(macro["token"], where=f"{mwhere}.token")
            _require(TOKEN_RE.fullmatch(token) is not None, f"{mwhere}.token is invalid")
            _require(token not in primitives and token not in seen_tokens, f"{mwhere}.token collides")
            seen_tokens.add(token)
            expansion = macro["expansion"]
            _require(
                isinstance(expansion, list)
                and len(expansion) in {2, 3}
                and all(item in primitives for item in expansion),
                f"{mwhere}.expansion must contain 2-3 base primitives",
            )
            support = _integer(macro["support"], where=f"{mwhere}.support")
            length = _integer(macro["length"], where=f"{mwhere}.length", minimum=2)
            _require(length == len(expansion), f"{mwhere}.length disagrees with expansion")
            source_name = macro.get("source_name")
            _require(source_name is None or isinstance(source_name, str), f"{mwhere}.source_name invalid")
            macros.append(
                {
                    "token": token,
                    "expansion": tuple(expansion),
                    "support": support,
                    "length": length,
                    "source_name": source_name,
                }
            )
        if arm == "base":
            _require(not macros, "base library must not contain macros")
        draw_seed = raw.get("draw_seed")
        _require(draw_seed is None or isinstance(draw_seed, int), f"{where}.draw_seed invalid")
        libraries[arm] = {
            "id": library_id,
            "provenance": provenance,
            "macros": macros,
            "draw_seed": draw_seed,
        }
    return libraries


def _percentile(values: Sequence[float], probability: float) -> float:
    _require(bool(values), "cannot take percentile of an empty sequence")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def paired_bootstrap(
    treatment: Sequence[float],
    control: Sequence[float],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    """Two-sided paired task bootstrap for a mean treatment-control delta."""

    _require(len(treatment) == len(control) and bool(treatment), "paired vectors must align")
    _integer(repetitions, where="bootstrap repetitions", minimum=1)
    differences = [float(a) - float(b) for a, b in zip(treatment, control)]
    rng = random.Random(seed)
    boot: list[float] = []
    for _ in range(repetitions):
        boot.append(fmean(differences[rng.randrange(len(differences))] for _ in differences))
    return {
        "n_tasks": len(differences),
        "repetitions": repetitions,
        "point_delta": fmean(differences),
        "ci95": [_percentile(boot, 0.025), _percentile(boot, 0.975)],
        "probability_delta_le_zero": sum(value <= 0.0 for value in boot) / repetitions,
    }


def hierarchical_random_bootstrap(
    treatment_by_task: Mapping[str, float],
    random_by_draw: Mapping[str, Mapping[str, float]],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    """Resample paired tasks and frozen placebo-library draws in each replicate."""

    task_ids = sorted(treatment_by_task)
    draw_ids = sorted(random_by_draw)
    _require(bool(task_ids) and bool(draw_ids), "hierarchical bootstrap needs tasks and draws")
    for draw, values in random_by_draw.items():
        _require(set(values) == set(task_ids), f"random draw {draw} has unpaired tasks")
    random_mean_by_task = {
        task_id: fmean(random_by_draw[draw][task_id] for draw in draw_ids)
        for task_id in task_ids
    }
    draw_averaged = paired_bootstrap(
        [treatment_by_task[task_id] for task_id in task_ids],
        [random_mean_by_task[task_id] for task_id in task_ids],
        repetitions=repetitions,
        seed=seed,
    )
    rng = random.Random(seed + 1)
    boot: list[float] = []
    for _ in range(repetitions):
        sampled_tasks = [task_ids[rng.randrange(len(task_ids))] for _ in task_ids]
        sampled_draws = [draw_ids[rng.randrange(len(draw_ids))] for _ in draw_ids]
        treatment_mean = fmean(treatment_by_task[task_id] for task_id in sampled_tasks)
        random_mean = fmean(
            random_by_draw[draw][task_id]
            for draw in sampled_draws
            for task_id in sampled_tasks
        )
        boot.append(treatment_mean - random_mean)
    return {
        "n_tasks": len(task_ids),
        "n_library_draws": len(draw_ids),
        "point_delta": fmean(
            treatment_by_task[task_id] - random_mean_by_task[task_id]
            for task_id in task_ids
        ),
        "ci95": [_percentile(boot, 0.025), _percentile(boot, 0.975)],
        "probability_delta_le_zero": sum(value <= 0.0 for value in boot) / repetitions,
        "draw_averaged_paired_task_bootstrap": draw_averaged,
        "method": "paired-task plus frozen-library-draw hierarchical bootstrap",
    }


def _score_pairs(program: Sequence[str], pairs: Sequence[Mapping[str, Any]]) -> tuple[int, bool]:
    correct = 0
    execution_failed = False
    for pair in pairs:
        try:
            actual = _execute(program, copy.deepcopy(pair["input"]))
        except (ValueError, TypeError, IndexError, KeyError, OverflowError):
            execution_failed = True
            continue
        correct += int(actual == pair["output"])
    return correct, execution_failed


def _parse_visible_candidate(
    *,
    output: Mapping[str, Any],
    task: Mapping[str, Any],
    allowed_tokens: set[str],
    macro_map: Mapping[str, Sequence[str]],
    max_surface_calls: int,
    max_expanded_depth: int,
) -> dict[str, Any]:
    """Parse and visible-score one completion without receiving hidden labels."""

    sample_index = _integer(output.get("sample_index"), where="output.sample_index")
    text = output.get("text")
    token_ids = output.get("token_ids")
    _require(isinstance(text, str), f"sample {sample_index} text must be a string")
    _require(
        isinstance(token_ids, list)
        and all(isinstance(token, int) and not isinstance(token, bool) for token in token_ids),
        f"sample {sample_index} token_ids must be integer IDs",
    )
    count_fields = (
        "n_stage1_prompt_tokens",
        "n_stage2_prompt_tokens",
        "n_sampled_tokens",
        "n_injected_tokens",
        "n_completion_tokens",
        "n_thinking_tokens",
        "n_answer_tokens",
        "n_terminal_tokens_trimmed",
    )
    counts = {
        field: _integer(output.get(field), where=f"output {sample_index}.{field}")
        for field in count_fields
    }
    _require(
        counts["n_completion_tokens"] == len(token_ids),
        f"output {sample_index} completion count disagrees with token_ids",
    )
    forced_close = output.get("forced_close")
    truncated = output.get("truncated")
    _require(isinstance(forced_close, bool), f"output {sample_index}.forced_close must be bool")
    _require(isinstance(truncated, bool), f"output {sample_index}.truncated must be bool")
    finish_reason = output.get("finish_reason")
    _require(isinstance(finish_reason, str), f"output {sample_index}.finish_reason must be string")
    _require(
        truncated == (finish_reason == "length"),
        f"output {sample_index} truncation disagrees with finish_reason",
    )

    candidate: dict[str, Any] = {
        "sample_index": sample_index,
        "raw_text": text,
        "parsed": False,
        "valid": False,
        "parse_error": None,
        "validation_error": None,
        "program": None,
        "expanded_program": None,
        "surface_depth": None,
        "expanded_depth": None,
        "visible_correct": 0,
        "visible_total": len(task["visible"]),
        "visible_score": None,
        "visible_pass": False,
        "macro_used": False,
        "macro_tokens": [],
        "forced_close": forced_close,
        "truncated": truncated,
        "finish_reason": finish_reason,
        "sampled_tokens": counts["n_sampled_tokens"],
        "injected_tokens": counts["n_injected_tokens"],
        "completion_tokens": counts["n_completion_tokens"],
        "thinking_tokens": counts["n_thinking_tokens"],
        "answer_tokens": counts["n_answer_tokens"],
        "stage1_logical_prompt_tokens": counts["n_stage1_prompt_tokens"],
        "stage2_logical_prompt_tokens": counts["n_stage2_prompt_tokens"],
        "selected": False,
    }
    try:
        program = harness.parse_program(
            text,
            allowed_tokens=allowed_tokens,
            max_surface_calls=max_surface_calls,
        )
    except ValueError as exc:
        candidate["parse_error"] = str(exc)
        return candidate
    candidate["parsed"] = True
    candidate["program"] = list(program)
    candidate["surface_depth"] = len(program)
    candidate["macro_tokens"] = [token for token in program if token in macro_map]
    candidate["macro_used"] = bool(candidate["macro_tokens"])
    try:
        expanded = _expand(program, macro_map)
        if len(expanded) > max_expanded_depth:
            raise ValueError(
                f"expanded depth {len(expanded)} exceeds limit {max_expanded_depth}"
            )
        visible_correct, visible_execution_failed = _score_pairs(expanded, task["visible"])
        if visible_execution_failed:
            raise ValueError("program failed to execute on at least one visible example")
    except (ValueError, TypeError, IndexError, KeyError, OverflowError) as exc:
        candidate["validation_error"] = str(exc)
        return candidate
    candidate["valid"] = True
    candidate["expanded_program"] = list(expanded)
    candidate["expanded_depth"] = len(expanded)
    candidate["visible_correct"] = visible_correct
    candidate["visible_score"] = visible_correct / len(task["visible"])
    candidate["visible_pass"] = visible_correct == len(task["visible"])
    return candidate


def select_visible_only(candidates: Sequence[Mapping[str, Any]]) -> int | None:
    """Return the earliest sample among valid candidates at maximal visible score."""

    valid = [candidate for candidate in candidates if candidate["valid"]]
    if not valid:
        return None
    best_score = max(float(candidate["visible_score"]) for candidate in valid)
    return min(
        int(candidate["sample_index"])
        for candidate in valid
        if float(candidate["visible_score"]) == best_score
    )


def _attach_hidden_grades(
    candidates: list[dict[str, Any]], task: Mapping[str, Any], selected_index: int | None
) -> None:
    """Attach held-out grades only after visible-only selection is frozen."""

    for candidate in candidates:
        candidate["selected"] = candidate["sample_index"] == selected_index
        candidate["hidden_total"] = len(task["hidden"])
        candidate["hidden_correct"] = 0
        candidate["hidden_score"] = None
        candidate["hidden_pass"] = False
        candidate["probe_total"] = len(task["probe"])
        candidate["probe_correct"] = 0
        candidate["probe_pass"] = False
        candidate["hidden_execution_failed"] = False
        candidate["probe_execution_failed"] = False
        if not candidate["valid"]:
            continue
        expanded = candidate["expanded_program"]
        hidden_correct, hidden_failed = _score_pairs(expanded, task["hidden"])
        probe_correct, probe_failed = _score_pairs(expanded, task["probe"])
        candidate["hidden_correct"] = hidden_correct
        candidate["hidden_score"] = hidden_correct / len(task["hidden"])
        candidate["hidden_pass"] = hidden_correct == len(task["hidden"])
        candidate["probe_correct"] = probe_correct
        candidate["probe_pass"] = probe_correct == len(task["probe"])
        candidate["hidden_execution_failed"] = hidden_failed
        candidate["probe_execution_failed"] = probe_failed


def analyze_arm_rows(
    *,
    arm: str,
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    tasks: Mapping[str, Mapping[str, Any]],
    library: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Validate and score one runner-native arm artifact."""

    harness.extract_token_accounting(rows, summary)
    primitives = set(_primitive_tokens())
    callable_macros = arm not in {"base", "mined_hint"}
    macro_map = (
        {macro["token"]: macro["expansion"] for macro in library["macros"]}
        if callable_macros
        else {}
    )
    allowed_tokens = primitives | set(macro_map)
    by_task: dict[str, dict[str, Any]] = {}
    for row_index, row in enumerate(rows):
        where = f"{arm} row {row_index}"
        meta = row.get("meta")
        _require(isinstance(meta, dict), f"{where} lacks meta")
        for key in (
            "task_id",
            "split",
            "arm",
            "library_id",
            "max_surface_calls",
            "max_expanded_primitive_depth",
        ):
            _require(key in meta, f"{where}.meta lacks {key}")
        task_id = _string(meta["task_id"], where=f"{where}.meta.task_id")
        _require(task_id in tasks, f"{where} references unknown task {task_id}")
        _require(task_id not in by_task, f"duplicate {arm} row for task {task_id}")
        task = tasks[task_id]
        _require(row.get("id") == f"{task_id}::{arm}", f"{where} id disagrees with schema")
        _require(meta["arm"] == arm, f"{where}.meta.arm disagrees")
        _require(meta["split"] == task["split"], f"{where}.meta.split disagrees")
        _require(meta["library_id"] == library["id"], f"{where}.meta.library_id disagrees")
        max_surface = _integer(
            meta["max_surface_calls"], where=f"{where}.meta.max_surface_calls", minimum=1
        )
        max_expanded = _integer(
            meta["max_expanded_primitive_depth"],
            where=f"{where}.meta.max_expanded_primitive_depth",
            minimum=1,
        )
        prompt_tokens = _integer(row.get("n_prompt_tokens"), where=f"{where}.n_prompt_tokens")
        outputs = row.get("outputs")
        _require(isinstance(outputs, list) and bool(outputs), f"{where}.outputs must be non-empty")
        indices = [output.get("sample_index") if isinstance(output, dict) else None for output in outputs]
        _require(indices == list(range(len(outputs))), f"{where} sample indices must be contiguous/orderly")
        candidates = [
            _parse_visible_candidate(
                output=output,
                task=task,
                allowed_tokens=allowed_tokens,
                macro_map=macro_map,
                max_surface_calls=max_surface,
                max_expanded_depth=max_expanded,
            )
            for output in outputs
        ]
        selected_index = select_visible_only(candidates)
        _attach_hidden_grades(candidates, task, selected_index)
        selected = next(
            (candidate for candidate in candidates if candidate["selected"]), None
        )
        by_task[task_id] = {
            "task_id": task_id,
            "split": task["split"],
            "arm": arm,
            "library_id": library["id"],
            "target_program": list(task["program"]),
            "target_min_depth": task["min_depth"],
            "motif_names": list(task["motif_names"]),
            "unique_prompt_tokens": prompt_tokens,
            "n_samples": len(candidates),
            "candidates": candidates,
            "selected_sample_index": selected_index,
            "abstained": selected is None,
            "selected_hidden_pass": bool(selected and selected["hidden_pass"]),
            "selected_visible_pass": bool(selected and selected["visible_pass"]),
            "selected_macro_used": bool(selected and selected["macro_used"]),
            "selected_surface_depth": selected["surface_depth"] if selected else None,
            "selected_expanded_depth": selected["expanded_depth"] if selected else None,
            "oracle_hidden_pass": any(candidate["hidden_pass"] for candidate in candidates),
        }
    return by_task


def _mean(values: Sequence[float]) -> float | None:
    return fmean(values) if values else None


def summarize_task_rows(task_rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    tasks = list(task_rows.values())
    candidates = [candidate for task in tasks for candidate in task["candidates"]]
    valid = [candidate for candidate in candidates if candidate["valid"]]
    selected = [
        candidate
        for task in tasks
        for candidate in task["candidates"]
        if candidate["selected"]
    ]
    visible_pass = [candidate for candidate in valid if candidate["visible_pass"]]
    correct_candidates = [candidate for candidate in valid if candidate["hidden_pass"]]
    false_visible = [
        candidate for candidate in visible_pass if not candidate["hidden_pass"]
    ]
    selected_false_visible = [
        candidate
        for candidate in selected
        if candidate["visible_pass"] and not candidate["hidden_pass"]
    ]
    total_sampled_tokens = sum(candidate["sampled_tokens"] for candidate in candidates)
    total_unique_prompt_tokens = sum(task["unique_prompt_tokens"] for task in tasks)
    return {
        "n_tasks": len(tasks),
        "n_samples": len(candidates),
        "parse_rate": sum(candidate["parsed"] for candidate in candidates) / len(candidates),
        "valid_program_rate": len(valid) / len(candidates),
        "visible_pass_rate": len(visible_pass) / len(candidates),
        "hidden_pass_sample_rate": len(correct_candidates) / len(candidates),
        "oracle_hidden_coverage": sum(task["oracle_hidden_pass"] for task in tasks) / len(tasks),
        "selected_hidden_accuracy": sum(task["selected_hidden_pass"] for task in tasks) / len(tasks),
        "selected_visible_pass_rate": sum(task["selected_visible_pass"] for task in tasks)
        / len(tasks),
        "abstention_rate": sum(task["abstained"] for task in tasks) / len(tasks),
        "false_visible_pass_count": len(false_visible),
        "false_visible_pass_rate_all_samples": len(false_visible) / len(candidates),
        "false_visible_pass_rate_given_visible_pass": (
            len(false_visible) / len(visible_pass) if visible_pass else None
        ),
        "selected_false_visible_pass_rate": len(selected_false_visible) / len(tasks),
        "macro_use_rate_valid_candidates": (
            sum(candidate["macro_used"] for candidate in valid) / len(valid) if valid else None
        ),
        "macro_use_rate_correct_candidates": (
            sum(candidate["macro_used"] for candidate in correct_candidates)
            / len(correct_candidates)
            if correct_candidates
            else None
        ),
        "macro_use_rate_selected": (
            sum(candidate["macro_used"] for candidate in selected) / len(selected)
            if selected
            else None
        ),
        "mean_surface_depth_valid": _mean(
            [float(candidate["surface_depth"]) for candidate in valid]
        ),
        "mean_expanded_depth_valid": _mean(
            [float(candidate["expanded_depth"]) for candidate in valid]
        ),
        "mean_surface_depth_selected": _mean(
            [float(candidate["surface_depth"]) for candidate in selected]
        ),
        "mean_expanded_depth_selected": _mean(
            [float(candidate["expanded_depth"]) for candidate in selected]
        ),
        "total_unique_prompt_tokens": total_unique_prompt_tokens,
        "total_sampled_tokens": total_sampled_tokens,
        "sampled_plus_unique_prompt_tokens": total_sampled_tokens + total_unique_prompt_tokens,
        "logical_model_input_tokens": sum(
            candidate["stage1_logical_prompt_tokens"]
            + candidate["stage2_logical_prompt_tokens"]
            for candidate in candidates
        ),
        "injected_tokens": sum(candidate["injected_tokens"] for candidate in candidates),
        "thinking_tokens": sum(candidate["thinking_tokens"] for candidate in candidates),
        "answer_tokens": sum(candidate["answer_tokens"] for candidate in candidates),
        "mean_sampled_tokens_per_completion": total_sampled_tokens / len(candidates),
        "mean_sampled_tokens_per_task": total_sampled_tokens / len(tasks),
        "forced_close_rate": sum(candidate["forced_close"] for candidate in candidates)
        / len(candidates),
        "truncation_rate": sum(candidate["truncated"] for candidate in candidates)
        / len(candidates),
    }


def summarize_arm(task_rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    split_names = sorted({str(task["split"]) for task in task_rows.values()})
    summaries = {"all": summarize_task_rows(task_rows)}
    for split in split_names:
        subset = {task_id: task for task_id, task in task_rows.items() if task["split"] == split}
        summaries[split] = summarize_task_rows(subset)
    return summaries


def _contrast_seed(base_seed: int, label: str) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return base_seed + int.from_bytes(digest[:4], "big")


def _task_ids_for_split(
    arm_tasks: Mapping[str, Mapping[str, Any]], split: str
) -> list[str]:
    task_ids = sorted(
        task_id for task_id, task in arm_tasks.items() if task["split"] == split
    )
    _require(bool(task_ids), f"no tasks found for required split {split!r}")
    return task_ids


def selected_contrast(
    arms: Mapping[str, Mapping[str, Mapping[str, Any]]],
    treatment: str,
    control: str,
    split: str,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    _require(treatment in arms and control in arms, f"missing contrast arm: {treatment}/{control}")
    task_ids = _task_ids_for_split(arms[treatment], split)
    _require(
        all(task_id in arms[control] for task_id in task_ids),
        f"{control} is missing paired {split} tasks",
    )
    treatment_values = [float(arms[treatment][task_id]["selected_hidden_pass"]) for task_id in task_ids]
    control_values = [float(arms[control][task_id]["selected_hidden_pass"]) for task_id in task_ids]
    result = paired_bootstrap(
        treatment_values,
        control_values,
        repetitions=repetitions,
        seed=_contrast_seed(seed, f"{treatment}-{control}-{split}"),
    )
    result.update(
        {
            "treatment": treatment,
            "control": control,
            "split": split,
            "treatment_accuracy": fmean(treatment_values),
            "control_accuracy": fmean(control_values),
        }
    )
    return result


def token_matched_base_prefix(
    base_task: Mapping[str, Any], treatment_task: Mapping[str, Any]
) -> dict[str, Any]:
    """Select the shortest base prefix with no-smaller prompt+sampled tokens."""

    treatment_budget = treatment_task["unique_prompt_tokens"] + sum(
        candidate["sampled_tokens"] for candidate in treatment_task["candidates"]
    )
    base_candidates = sorted(base_task["candidates"], key=lambda row: row["sample_index"])
    running = int(base_task["unique_prompt_tokens"])
    prefix_k: int | None = None
    for index, candidate in enumerate(base_candidates, start=1):
        running += int(candidate["sampled_tokens"])
        if running >= treatment_budget:
            prefix_k = index
            break
    matched = prefix_k is not None
    if prefix_k is None:
        prefix_k = len(base_candidates)
    prefix = base_candidates[:prefix_k]
    selected_index = select_visible_only(prefix)
    selected = next(
        (candidate for candidate in prefix if candidate["sample_index"] == selected_index),
        None,
    )
    return {
        "task_id": base_task["task_id"],
        "split": base_task["split"],
        "matched": matched,
        "prefix_k": prefix_k,
        "available_base_k": len(base_candidates),
        "treatment_sampled_plus_unique_prompt_tokens": treatment_budget,
        "base_prefix_sampled_plus_unique_prompt_tokens": running,
        "base_prefix_selected_sample_index": selected_index,
        "base_prefix_selected_hidden_pass": bool(selected and selected["hidden_pass"]),
    }


def token_prefix_contrast(
    arms: Mapping[str, Mapping[str, Mapping[str, Any]]],
    treatment: str,
    split: str,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    task_ids = _task_ids_for_split(arms[treatment], split)
    rows = [token_matched_base_prefix(arms["base"][task_id], arms[treatment][task_id]) for task_id in task_ids]
    treatment_values = [float(arms[treatment][task_id]["selected_hidden_pass"]) for task_id in task_ids]
    base_values = [float(row["base_prefix_selected_hidden_pass"]) for row in rows]
    contrast = paired_bootstrap(
        treatment_values,
        base_values,
        repetitions=repetitions,
        seed=_contrast_seed(seed, f"token-prefix-{treatment}-{split}"),
    )
    contrast.update(
        {
            "treatment": treatment,
            "split": split,
            "all_tasks_have_no_smaller_base_budget": all(row["matched"] for row in rows),
            "matched_task_rate": sum(row["matched"] for row in rows) / len(rows),
            "mean_prefix_k": fmean(row["prefix_k"] for row in rows),
            "max_prefix_k": max(row["prefix_k"] for row in rows),
            "treatment_accuracy": fmean(treatment_values),
            "base_prefix_accuracy": fmean(base_values),
            "tasks": rows,
        }
    )
    return contrast


def _support_bin(value: int) -> int:
    for boundary in (4, 7, 15, 31, 63, 127, 255, 511):
        if value <= boundary:
            return boundary
    return 1023


def _library_profile(library: Mapping[str, Any]) -> list[tuple[int, int]]:
    """Count/length/training-support-bin profile frozen by the design review."""

    return sorted(
        (int(macro["length"]), _support_bin(int(macro["support"])))
        for macro in library["macros"]
    )


def _selected_candidate(task: Mapping[str, Any]) -> Mapping[str, Any] | None:
    return next((candidate for candidate in task["candidates"] if candidate["selected"]), None)


def random_ensemble_contrast(
    arms: Mapping[str, Mapping[str, Mapping[str, Any]]],
    treatment: str,
    random_arms: Sequence[str],
    split: str,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    task_ids = _task_ids_for_split(arms[treatment], split)
    treatment_by_task = {
        task_id: float(arms[treatment][task_id]["selected_hidden_pass"])
        for task_id in task_ids
    }
    random_by_draw: dict[str, dict[str, float]] = {}
    for arm in random_arms:
        _require(arm in arms, f"missing random ensemble arm {arm}")
        _require(all(task_id in arms[arm] for task_id in task_ids), f"{arm} tasks do not pair")
        random_by_draw[arm] = {
            task_id: float(arms[arm][task_id]["selected_hidden_pass"])
            for task_id in task_ids
        }
    result = hierarchical_random_bootstrap(
        treatment_by_task,
        random_by_draw,
        repetitions=repetitions,
        seed=_contrast_seed(seed, f"hierarchical-{treatment}-{split}"),
    )
    result.update({"treatment": treatment, "controls": list(random_arms), "split": split})
    return result


def _contingency_status(
    exp: Path,
    run: str,
    offending_arms: Sequence[str],
    *,
    required_budget: int,
    required_tasks: int,
) -> dict[str, Any]:
    if not offending_arms:
        return {"required": False, "complete": True, "offending_arms": []}
    directory = exp / "runs" / f"{run}_contingency"
    details: dict[str, Any] = {}
    task_sets: list[set[str]] = []
    complete = True
    for arm in offending_arms:
        row_path = directory / f"{arm}.jsonl"
        meta_path = directory / f"{arm}.meta.json"
        if not row_path.exists() or not meta_path.exists():
            details[arm] = {"present": False}
            complete = False
            continue
        rows = _read_jsonl(row_path)
        meta = _read_json(meta_path)
        sampling = meta.get("sampling") if isinstance(meta, dict) else None
        budget = sampling.get("thinking_budget") if isinstance(sampling, dict) else None
        task_ids = {
            row.get("meta", {}).get("task_id")
            for row in rows
            if isinstance(row.get("meta"), dict)
        }
        arm_ok = budget == required_budget and len(task_ids) == required_tasks and None not in task_ids
        details[arm] = {
            "present": True,
            "thinking_budget": budget,
            "n_tasks": len(task_ids),
            "valid_contract": arm_ok,
        }
        complete &= arm_ok
        task_sets.append(set(task_ids))
    if task_sets and any(task_set != task_sets[0] for task_set in task_sets[1:]):
        complete = False
    return {
        "required": True,
        "complete": complete,
        "offending_arms": list(offending_arms),
        "directory": str(directory),
        "details": details,
        "paired_frozen_subset": bool(task_sets) and all(
            task_set == task_sets[0] for task_set in task_sets
        ),
    }


def build_smoke_verdict(
    arms: Mapping[str, Mapping[str, Mapping[str, Any]]],
    summaries: Mapping[str, Mapping[str, Mapping[str, Any]]],
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    required_present = all(arm in arms for arm in ("base", "designed_ceiling"))
    if not required_present:
        return {
            "pass": False,
            "required_arms_present": False,
            "reasons": ["base and designed_ceiling smoke artifacts are required"],
        }
    base = summaries["base"]["all"]
    designed = summaries["designed_ceiling"]["all"]
    total_samples = base["n_samples"] + designed["n_samples"]
    overall_parse = (
        base["parse_rate"] * base["n_samples"]
        + designed["parse_rate"] * designed["n_samples"]
    ) / total_samples
    macro_candidates = sum(
        candidate["valid"] and candidate["macro_used"]
        for task in arms["designed_ceiling"].values()
        for candidate in task["candidates"]
    )
    overall_truncation = (
        base["truncation_rate"] * base["n_samples"]
        + designed["truncation_rate"] * designed["n_samples"]
    ) / total_samples
    min_parse = _number(decision["smoke_min_parse_rate"], where="smoke_min_parse_rate")
    min_macro = _integer(
        decision["smoke_min_macro_candidates"], where="smoke_min_macro_candidates"
    )
    max_trunc = _number(
        decision["smoke_max_answer_truncation"], where="smoke_max_answer_truncation"
    )
    gates = {
        "overall_parse": overall_parse >= min_parse,
        "base_parse": base["parse_rate"] >= min_parse,
        "designed_parse": designed["parse_rate"] >= min_parse,
        "designed_valid_macro_candidates": macro_candidates >= min_macro,
        "answer_truncation": overall_truncation < max_trunc,
        "designed_oracle_not_below_base": (
            designed["oracle_hidden_coverage"] >= base["oracle_hidden_coverage"]
        ),
    }
    return {
        "pass": all(gates.values()),
        "required_arms_present": True,
        "gates": gates,
        "metrics": {
            "overall_parse_rate": overall_parse,
            "base_parse_rate": base["parse_rate"],
            "designed_parse_rate": designed["parse_rate"],
            "designed_valid_macro_using_candidates": macro_candidates,
            "overall_truncation_rate": overall_truncation,
            "base_oracle_hidden_coverage": base["oracle_hidden_coverage"],
            "designed_oracle_hidden_coverage": designed["oracle_hidden_coverage"],
        },
        "reasons": [name for name, passed in gates.items() if not passed],
    }


def build_full_verdict(
    *,
    exp: Path,
    arms: Mapping[str, Mapping[str, Mapping[str, Any]]],
    summaries: Mapping[str, Mapping[str, Mapping[str, Any]]],
    libraries: Mapping[str, Mapping[str, Any]],
    decision: Mapping[str, Any],
    expected_arms: Sequence[str],
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    present = set(arms)
    missing_expected = sorted(set(expected_arms) - present)
    required_present = all(arm in present for arm in PRIMARY_ARMS)
    random_arms = sorted(
        arm for arm in arms if re.fullmatch(r"random_[0-9]+", arm)
    )
    complete_primary_inputs = required_present and len(random_arms) >= 5
    verdict: dict[str, Any] = {
        "expected_arms": list(expected_arms),
        "missing_expected_arms": missing_expected,
        "primary_inputs_complete": complete_primary_inputs,
        "random_arms": random_arms,
    }
    if not complete_primary_inputs:
        verdict.update(
            {
                "complete_callable_abstraction": False,
                "claimable_complete_callable_abstraction": False,
                "status": "incomplete_artifacts",
            }
        )
        return verdict

    contrasts = {
        "mined_minus_base_reuse": selected_contrast(
            arms, "mined", "base", "reuse", repetitions=repetitions, seed=seed
        ),
        "mined_minus_base_no_reuse": selected_contrast(
            arms, "mined", "base", "no_reuse", repetitions=repetitions, seed=seed
        ),
        "mined_minus_hint_reuse": selected_contrast(
            arms, "mined", "mined_hint", "reuse", repetitions=repetitions, seed=seed
        ),
        "hint_minus_base_reuse": selected_contrast(
            arms, "mined_hint", "base", "reuse", repetitions=repetitions, seed=seed
        ),
    }
    prefix = token_prefix_contrast(
        arms, "mined", "reuse", repetitions=repetitions, seed=seed
    )
    random_profiles_match = all(
        _library_profile(libraries[arm]) == _library_profile(libraries["mined"])
        for arm in random_arms
    )
    random_contrast = random_ensemble_contrast(
        arms,
        "mined",
        random_arms,
        "reuse",
        repetitions=repetitions,
        seed=seed,
    )
    contrasts["mined_minus_random_reuse"] = random_contrast
    verdict["contrasts"] = contrasts
    verdict["token_matched_base_prefix"] = prefix
    verdict["random_profiles_match_count_length_support_bins"] = random_profiles_match

    reuse = contrasts["mined_minus_base_reuse"]
    no_reuse = contrasts["mined_minus_base_no_reuse"]
    hint = contrasts["mined_minus_hint_reuse"]
    primary_delta_min = _number(decision["primary_min_delta"], where="primary_min_delta")
    callable_min = _number(
        decision["callable_vs_hint_min_delta"], where="callable_vs_hint_min_delta"
    )
    random_min = _number(
        decision["mined_vs_random_min_delta"], where="mined_vs_random_min_delta"
    )
    treatment_only = [
        task_id
        for task_id, task in arms["mined"].items()
        if task["split"] == "reuse"
        and task["selected_hidden_pass"]
        and not arms["base"][task_id]["selected_hidden_pass"]
    ]
    treatment_only_macro_count = sum(
        arms["mined"][task_id]["selected_macro_used"] for task_id in treatment_only
    )
    treatment_only_macro_rate = (
        treatment_only_macro_count / len(treatment_only) if treatment_only else 0.0
    )
    designed_oracle_lift = (
        summaries["designed_ceiling"]["reuse"]["oracle_hidden_coverage"]
        - summaries["base"]["reuse"]["oracle_hidden_coverage"]
    )
    system_gates = {
        "delta_at_least_0_10": reuse["point_delta"] >= primary_delta_min,
        "paired_ci_lower_above_zero": reuse["ci95"][0] > 0.0,
        "positive_vs_no_smaller_token_base_prefix": (
            prefix["all_tasks_have_no_smaller_base_budget"] and prefix["point_delta"] > 0.0
        ),
        "half_treatment_only_successes_use_macro": (
            bool(treatment_only) and treatment_only_macro_rate >= 0.5
        ),
        "no_reuse_lift_at_most_half_reuse": (
            no_reuse["point_delta"] <= 0.5 * reuse["point_delta"]
        ),
    }
    callable_gates = {
        "delta_at_least_0_05": hint["point_delta"] >= callable_min,
        "paired_ci_lower_above_zero": hint["ci95"][0] > 0.0,
    }
    recurrence_gates = {
        "random_profiles_match": random_profiles_match,
        "delta_at_least_0_05": random_contrast["point_delta"] >= random_min,
        "hierarchical_ci_lower_above_zero": random_contrast["ci95"][0] > 0.0,
    }
    verdict.update(
        {
            "designed_interface_oracle_lift": designed_oracle_lift,
            "designed_interface_usable": designed_oracle_lift > 0.0,
            "treatment_only_correct_tasks": treatment_only,
            "treatment_only_correct_macro_count": treatment_only_macro_count,
            "treatment_only_correct_macro_rate": treatment_only_macro_rate,
            "system_benefit_gates": system_gates,
            "callable_chunking_gates": callable_gates,
            "learned_recurrence_gates": recurrence_gates,
            "system_benefit": all(system_gates.values()),
            "callable_chunking": all(callable_gates.values()),
            "learned_recurrence": all(recurrence_gates.values()),
        }
    )

    relevant = ["base", "mined", "mined_hint", *random_arms]
    forced_threshold = _number(
        decision["forced_close_contingency"], where="forced_close_contingency"
    )
    offending = [
        arm for arm in relevant if summaries[arm]["all"]["forced_close_rate"] > forced_threshold
    ]
    contingency = _contingency_status(
        exp,
        "full",
        offending,
        required_budget=_integer(
            decision["contingency_thinking_budget"], where="contingency_thinking_budget"
        ),
        required_tasks=_integer(decision["contingency_tasks"], where="contingency_tasks"),
    )
    verdict["forced_close_contingency"] = contingency
    budget_confounded = contingency["required"] and not contingency["complete"]
    complete = (
        verdict["designed_interface_usable"]
        and verdict["system_benefit"]
        and verdict["callable_chunking"]
        and verdict["learned_recurrence"]
    )
    verdict["complete_callable_abstraction"] = complete
    verdict["budget_confounded"] = budget_confounded
    verdict["claimable_complete_callable_abstraction"] = complete and not budget_confounded

    # Secondary Qwen-specific value.  Prefer a separately named matched ensemble;
    # reuse the standard ensemble only if its exact (length, support) profile matches.
    if "qwen_ranked" in arms:
        qwen_random = sorted(
            arm for arm in arms if re.fullmatch(r"qwen_random_[0-9]+", arm)
        )
        if not qwen_random and all(
            _library_profile(libraries[arm]) == _library_profile(libraries["qwen_ranked"])
            for arm in random_arms
        ):
            qwen_random = random_arms
        qwen_profile_match = bool(qwen_random) and all(
            _library_profile(libraries[arm]) == _library_profile(libraries["qwen_ranked"])
            for arm in qwen_random
        )
        qwen_result: dict[str, Any] = {
            "matched_random_arms": qwen_random,
            "profiles_match": qwen_profile_match,
        }
        if qwen_profile_match:
            qwen_contrast = random_ensemble_contrast(
                arms,
                "qwen_ranked",
                qwen_random,
                "reuse",
                repetitions=repetitions,
                seed=seed,
            )
            qwen_result["contrast"] = qwen_contrast
            mined_accuracy = summaries["mined"]["reuse"]["selected_hidden_accuracy"]
            qwen_accuracy = summaries["qwen_ranked"]["reuse"]["selected_hidden_accuracy"]
            mined_expansions = {
                tuple(macro["expansion"]) for macro in libraries["mined"]["macros"]
            }
            qwen_expansion = {
                macro["token"]: tuple(macro["expansion"])
                for macro in libraries["qwen_ranked"]["macros"]
            }
            exclusive_correct: list[str] = []
            for task_id, task in arms["qwen_ranked"].items():
                if task["split"] != "reuse" or not task["selected_hidden_pass"]:
                    continue
                selected = _selected_candidate(task)
                uses_exclusive = bool(
                    selected
                    and any(
                        qwen_expansion[token] not in mined_expansions
                        for token in selected["macro_tokens"]
                    )
                )
                if uses_exclusive and not arms["mined"][task_id]["selected_hidden_pass"]:
                    exclusive_correct.append(task_id)
            qwen_gates = {
                "exact_verified_macro_count": len(libraries["qwen_ranked"]["macros"])
                == _integer(
                    decision["qwen_required_verified_macros"],
                    where="qwen_required_verified_macros",
                ),
                "delta_vs_random_at_least_0_05": qwen_contrast["point_delta"]
                >= _number(decision["qwen_vs_random_min_delta"], where="qwen_vs_random_min_delta"),
                "hierarchical_ci_lower_above_zero": qwen_contrast["ci95"][0] > 0.0,
                "within_0_05_of_mined": qwen_accuracy
                >= mined_accuracy
                - _number(decision["qwen_within_mined"], where="qwen_within_mined"),
                "two_exclusive_correct": len(exclusive_correct)
                >= _integer(
                    decision["qwen_min_exclusive_correct"], where="qwen_min_exclusive_correct"
                ),
            }
            qwen_result.update(
                {
                    "qwen_accuracy": qwen_accuracy,
                    "mined_accuracy": mined_accuracy,
                    "exclusive_correct_tasks": exclusive_correct,
                    "gates": qwen_gates,
                    "qwen_specific_value": all(qwen_gates.values()),
                }
            )
        else:
            qwen_result.update(
                {
                    "gates": {"matched_random_ensemble_available": False},
                    "qwen_specific_value": False,
                }
            )
        verdict["qwen_specific"] = qwen_result

    if budget_confounded:
        verdict["status"] = "budget_confounded"
    elif verdict["claimable_complete_callable_abstraction"]:
        verdict["status"] = "complete_callable_abstraction_supported"
    elif not verdict["designed_interface_usable"]:
        verdict["status"] = "macro_interface_failure"
    elif verdict["system_benefit"] and verdict["learned_recurrence"] and not verdict["callable_chunking"]:
        verdict["status"] = "highlighted_prior_sufficient_callable_chunking_not_established"
    elif verdict["system_benefit"] and not verdict["learned_recurrence"]:
        verdict["status"] = "generic_extra_inventory_or_shorter_syntax_not_recurrence"
    else:
        verdict["status"] = "complete_callable_abstraction_not_supported"
    return verdict


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_per_task_csv(
    path: Path, arms: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> None:
    fields = [
        "task_id",
        "split",
        "arm",
        "library_id",
        "target_min_depth",
        "n_samples",
        "unique_prompt_tokens",
        "sampled_tokens",
        "parsed_samples",
        "valid_samples",
        "visible_pass_samples",
        "hidden_pass_samples",
        "oracle_hidden_pass",
        "abstained",
        "selected_sample_index",
        "selected_visible_pass",
        "selected_hidden_pass",
        "selected_false_visible_pass",
        "selected_macro_used",
        "selected_surface_depth",
        "selected_expanded_depth",
        "forced_close_samples",
        "truncated_samples",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for arm in sorted(arms):
            for task_id in sorted(arms[arm]):
                task = arms[arm][task_id]
                candidates = task["candidates"]
                writer.writerow(
                    {
                        "task_id": task_id,
                        "split": task["split"],
                        "arm": arm,
                        "library_id": task["library_id"],
                        "target_min_depth": task["target_min_depth"],
                        "n_samples": task["n_samples"],
                        "unique_prompt_tokens": task["unique_prompt_tokens"],
                        "sampled_tokens": sum(row["sampled_tokens"] for row in candidates),
                        "parsed_samples": sum(row["parsed"] for row in candidates),
                        "valid_samples": sum(row["valid"] for row in candidates),
                        "visible_pass_samples": sum(row["visible_pass"] for row in candidates),
                        "hidden_pass_samples": sum(row["hidden_pass"] for row in candidates),
                        "oracle_hidden_pass": task["oracle_hidden_pass"],
                        "abstained": task["abstained"],
                        "selected_sample_index": task["selected_sample_index"],
                        "selected_visible_pass": task["selected_visible_pass"],
                        "selected_hidden_pass": task["selected_hidden_pass"],
                        "selected_false_visible_pass": task["selected_visible_pass"]
                        and not task["selected_hidden_pass"],
                        "selected_macro_used": task["selected_macro_used"],
                        "selected_surface_depth": task["selected_surface_depth"],
                        "selected_expanded_depth": task["selected_expanded_depth"],
                        "forced_close_samples": sum(row["forced_close"] for row in candidates),
                        "truncated_samples": sum(row["truncated"] for row in candidates),
                    }
                )


def analyze_experiment(
    *,
    exp: Path = EXP,
    run: str,
    bootstrap_repetitions: int | None = None,
    bootstrap_seed: int | None = None,
    write: bool = True,
) -> dict[str, Any]:
    _require(run in {"smoke", "full"}, "run must be 'smoke' or 'full'")
    config = load_config(exp / "configs" / "default.yaml")
    decision = config.get("decision")
    inference = config.get("inference")
    _require(isinstance(decision, dict), "config decision section missing")
    _require(isinstance(inference, dict), "config inference section missing")
    repetitions = (
        _integer(bootstrap_repetitions, where="bootstrap_repetitions", minimum=1)
        if bootstrap_repetitions is not None
        else _integer(
            decision.get("bootstrap_repetitions"),
            where="decision.bootstrap_repetitions",
            minimum=1,
        )
    )
    seed = (
        _integer(bootstrap_seed, where="bootstrap_seed")
        if bootstrap_seed is not None
        else _integer(decision.get("bootstrap_seed"), where="decision.bootstrap_seed")
    )
    tasks = load_tasks(exp / "data" / "tasks.json")
    libraries = load_libraries(exp / "data" / "libraries.json")
    runner_path = exp / "src" / "vllm_runner.py"
    _require(runner_path.is_file(), f"missing experiment-local vLLM runner: {runner_path}")
    expected_runner_sha256 = hashlib.sha256(runner_path.read_bytes()).hexdigest()
    run_dir = exp / "runs" / run
    _require(run_dir.is_dir(), f"missing run directory: {run_dir}")
    paths = sorted(run_dir.glob("*.jsonl"))
    arm_paths = [path for path in paths if path.stem in libraries]
    _require(bool(arm_paths), f"no library-arm JSONL artifacts in {run_dir}")
    unknown = [path.name for path in paths if path.stem not in libraries]
    _require(not unknown, f"unknown JSONL artifacts in {run_dir}: {unknown}")

    arms: dict[str, dict[str, dict[str, Any]]] = {}
    runner_metadata: dict[str, Any] = {}
    for path in arm_paths:
        arm = path.stem
        rows = _read_jsonl(path)
        summary_path = run_dir / f"{arm}.meta.json"
        summary = _read_json(summary_path)
        _require(isinstance(summary, dict), f"{summary_path} must be an object")
        _require(
            summary.get("schema_version") == local_vllm.RUNNER_SCHEMA_VERSION,
            f"{summary_path} is not the pinned vLLM runner schema",
        )
        _require(
            summary.get("runner_sha256") == expected_runner_sha256,
            f"{summary_path} runner hash disagrees with experiment-local vLLM runner",
        )
        _require(isinstance(summary.get("engine"), dict), f"{summary_path} lacks vLLM engine metadata")
        _require(
            isinstance(summary.get("sampling"), dict),
            f"{summary_path} lacks vLLM sampling metadata",
        )
        arms[arm] = analyze_arm_rows(
            arm=arm,
            rows=rows,
            summary=summary,
            tasks=tasks,
            library=libraries[arm],
        )
        runner_metadata[arm] = {
            "model": summary.get("model"),
            "model_revision": summary.get("model_revision"),
            "schema_version": summary.get("schema_version"),
            "runner_sha256": summary.get("runner_sha256"),
            "sampling": summary.get("sampling"),
            "counts": summary.get("counts"),
            "timing": summary.get("timing"),
            "engine": summary.get("engine"),
        }
    reference_arm = "base" if "base" in arms else sorted(arms)[0]
    reference_tasks = set(arms[reference_arm])
    for arm, task_rows in arms.items():
        _require(
            set(task_rows) == reference_tasks,
            f"arm {arm} task set does not exactly match {reference_arm}",
        )
    summaries = {arm: summarize_arm(task_rows) for arm, task_rows in arms.items()}
    expected_arms = inference.get("arms")
    _require(isinstance(expected_arms, list), "config inference.arms must be a list")
    if run == "smoke":
        verdict = {"smoke_gate": build_smoke_verdict(arms, summaries, decision)}
    else:
        verdict = {
            "verdict": build_full_verdict(
                exp=exp,
                arms=arms,
                summaries=summaries,
                libraries=libraries,
                decision=decision,
                expected_arms=[str(arm) for arm in expected_arms],
                repetitions=repetitions,
                seed=seed,
            )
        }
    per_task = [
        task
        for arm in sorted(arms)
        for _, task in sorted(arms[arm].items())
    ]
    result = {
        "schema_version": 1,
        "run": run,
        "selection_rule": (
            "earliest valid sample among maximal visible-example score; hidden labels grade only"
        ),
        "bootstrap_repetitions": repetitions,
        "bootstrap_seed": seed,
        "arms": summaries,
        "runner_metadata": runner_metadata,
        **verdict,
        "per_task": per_task,
    }
    if write:
        output_dir = exp / "analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_payload = {key: value for key, value in result.items() if key != "per_task"}
        _json_dump(output_dir / f"{run}_summary.json", summary_payload)
        _json_dump(output_dir / f"{run}_per_task.json", per_task)
        _write_per_task_csv(output_dir / f"{run}_per_task.csv", arms)
        _json_dump(output_dir / f"{run}_verdict.json", {"run": run, **verdict})
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", choices=("smoke", "full"), default="full")
    parser.add_argument(
        "--bootstrap-repetitions",
        type=int,
        help="override the preregistered count (intended for analyzer tests only)",
    )
    parser.add_argument("--bootstrap-seed", type=int, help="override the preregistered seed")
    args = parser.parse_args(argv)
    result = analyze_experiment(
        run=args.run,
        bootstrap_repetitions=args.bootstrap_repetitions,
        bootstrap_seed=args.bootstrap_seed,
    )
    if args.run == "smoke":
        gate = result["smoke_gate"]
        print(f"smoke gate: {'PASS' if gate['pass'] else 'FAIL'}")
        print(json.dumps(gate, indent=2, sort_keys=True))
    else:
        verdict = result["verdict"]
        print(f"full verdict: {verdict['status']}")
        print(json.dumps(verdict, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
