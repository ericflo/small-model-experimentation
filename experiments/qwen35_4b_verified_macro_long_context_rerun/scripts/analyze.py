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
import dataclasses
import hashlib
import json
import math
import os
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
import full_artifacts as full_store  # noqa: E402
import scientific_artifacts as scientific_store  # noqa: E402


PRIMARY_ARMS = ("base", "mined", "mined_hint", "designed_ceiling")
TOKEN_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
LOOP_OVERRIDE_MIN_BUDGET = 32768
LOOP_TAIL_TOKENS = 8192
LOOP_MAX_PERIOD_TOKENS = 2048
LOOP_MIN_MATCH_RATE = 0.99


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


def _loop_settings(decision: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return the registered, content-blind periodic-tail classifier settings."""

    source = decision or {}
    minimum_budget = _integer(
        source.get("loop_override_min_budget", LOOP_OVERRIDE_MIN_BUDGET),
        where="decision.loop_override_min_budget",
        minimum=1,
    )
    tail_tokens = _integer(
        source.get("loop_tail_tokens", LOOP_TAIL_TOKENS),
        where="decision.loop_tail_tokens",
        minimum=2,
    )
    max_period = _integer(
        source.get("loop_max_period_tokens", LOOP_MAX_PERIOD_TOKENS),
        where="decision.loop_max_period_tokens",
        minimum=1,
    )
    min_match_rate = _number(
        source.get("loop_min_match_rate", LOOP_MIN_MATCH_RATE),
        where="decision.loop_min_match_rate",
    )
    _require(
        0.0 <= min_match_rate <= 1.0,
        "decision.loop_min_match_rate must be in [0, 1]",
    )
    return {
        "minimum_budget": minimum_budget,
        "tail_tokens": tail_tokens,
        "max_period_tokens": max_period,
        "min_match_rate": min_match_rate,
    }


def _periodic_loop_classification(
    output: Mapping[str, Any],
    *,
    cap_contact: bool,
    thinking_budget: int | None,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify a cap-contact tail by exact token equality at every registered lag."""

    raw_ids = output.get(
        "retained_thinking_token_ids", output.get("stage1_token_ids")
    )
    if raw_ids is None:
        thinking_ids: list[int] = []
        token_source = None
    else:
        _require(
            isinstance(raw_ids, list)
            and all(
                isinstance(token, int) and not isinstance(token, bool)
                for token in raw_ids
            ),
            "thinking token IDs must be an integer list",
        )
        thinking_ids = raw_ids
        token_source = (
            "retained_thinking_token_ids"
            if "retained_thinking_token_ids" in output
            else "stage1_token_ids"
        )

    tail_limit = int(settings["tail_tokens"])
    tail_length = min(len(thinking_ids), tail_limit)
    result: dict[str, Any] = {
        "periodic_loop": False,
        "unresolved_cap_contact": cap_contact,
        "loop_period_tokens": None,
        "loop_tail_tokens": tail_length,
        "loop_match_rate": None,
        "loop_token_source": token_source,
    }
    if (
        not cap_contact
        or thinking_budget is None
        or thinking_budget < int(settings["minimum_budget"])
        or tail_length < tail_limit
    ):
        return result

    tail = thinking_ids[-tail_limit:]
    max_period = min(int(settings["max_period_tokens"]), len(tail) // 2)
    best_rate = -1.0
    best_period: int | None = None
    for period in range(1, max_period + 1):
        comparisons = len(tail) - period
        allowed_mismatches = math.floor(
            comparisons * (1.0 - float(settings["min_match_rate"])) + 1e-12
        )
        mismatches = 0
        for index in range(period, len(tail)):
            if tail[index] != tail[index - period]:
                mismatches += 1
                if mismatches > allowed_mismatches:
                    break
        if mismatches <= allowed_mismatches:
            rate = (comparisons - mismatches) / comparisons
            if rate > best_rate:
                best_rate = rate
                best_period = period
                if rate == 1.0:
                    break

    periodic_loop = (
        best_period is not None
        and best_rate >= float(settings["min_match_rate"])
    )
    result.update(
        {
            "periodic_loop": periodic_loop,
            "unresolved_cap_contact": cap_contact and not periodic_loop,
            "loop_period_tokens": best_period,
            "loop_match_rate": best_rate if best_period is not None else None,
        }
    )
    return result


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
    thinking_mode: str,
    thinking_budget: int | None,
    answer_max_tokens: int,
    loop_settings: Mapping[str, Any],
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
    stage1_finish_reason = output.get("stage1_finish_reason")
    _require(
        isinstance(stage1_finish_reason, str),
        f"output {sample_index}.stage1_finish_reason must be string",
    )
    _require(
        truncated == (finish_reason == "length"),
        f"output {sample_index} truncation disagrees with finish_reason",
    )
    stage1_length_finish = stage1_finish_reason == "length"
    answer_limit_contact = (
        truncated
        or finish_reason == "length"
        or counts["n_answer_tokens"] >= answer_max_tokens
    )
    if thinking_mode == "budget":
        _require(thinking_budget is not None, "budget mode lacks a thinking budget")
        _require(
            counts["n_thinking_tokens"] <= thinking_budget,
            f"output {sample_index} thinking tokens exceed registered budget",
        )
        thinking_tokens_at_budget = counts["n_thinking_tokens"] == thinking_budget
        forced_intervention = forced_close
        reasoning_boundary_contact = (
            counts["n_thinking_tokens"] + 1 >= thinking_budget
        )
        cap_contact = forced_intervention or reasoning_boundary_contact
        answer_restart_after_natural_close = (
            stage1_length_finish
            and not forced_intervention
            and not reasoning_boundary_contact
        )
    else:
        _require(
            not forced_close,
            f"output {sample_index} forced_close is only valid in budget mode",
        )
        thinking_tokens_at_budget = False
        forced_intervention = False
        reasoning_boundary_contact = False
        answer_restart_after_natural_close = False
        cap_contact = False

    loop = _periodic_loop_classification(
        output,
        cap_contact=cap_contact,
        thinking_budget=thinking_budget,
        settings=loop_settings,
    )

    candidate: dict[str, Any] = {
        "sample_index": sample_index,
        # Derived artifacts retain a content hash, not multi-kilobyte/full-
        # reasoning completion text.  Raw text and exact token arrays remain in
        # the checksummed runner shard.
        "completion_sha256": hashlib.sha256(
            json.dumps(
                dict(output),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
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
        # Keep the historical alias while broadening it conservatively to a
        # natural stop exactly at the registered answer ceiling.  ``truncated``
        # remains the runner's unchanged stage-two length signal.
        "answer_truncated": answer_limit_contact,
        "answer_limit_contact": answer_limit_contact,
        "answer_tokens_at_limit": counts["n_answer_tokens"] >= answer_max_tokens,
        "answer_max_tokens": answer_max_tokens,
        "finish_reason": finish_reason,
        "stage1_finish_reason": stage1_finish_reason,
        "stage1_length_finish": stage1_length_finish,
        "forced_intervention": forced_intervention,
        "reasoning_boundary_contact": reasoning_boundary_contact,
        "answer_restart_after_natural_close": answer_restart_after_natural_close,
        "thinking_tokens_at_budget": thinking_tokens_at_budget,
        "cap_contact": cap_contact,
        **loop,
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
    decision: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Validate and score one runner-native arm artifact."""

    harness.extract_token_accounting(rows, summary)
    sampling = summary.get("sampling")
    _require(isinstance(sampling, dict), f"{arm} summary lacks sampling metadata")
    thinking_mode = _string(sampling.get("thinking"), where=f"{arm} sampling.thinking")
    _require(
        thinking_mode in {"off", "natural", "budget"},
        f"{arm} sampling.thinking is unsupported",
    )
    thinking_budget = (
        _integer(
            sampling.get("thinking_budget"),
            where=f"{arm} sampling.thinking_budget",
            minimum=1,
        )
        if thinking_mode == "budget"
        else None
    )
    answer_max_tokens = _integer(
        sampling.get("answer_max_tokens"),
        where=f"{arm} sampling.answer_max_tokens",
        minimum=1,
    )
    loop_settings = _loop_settings(decision)
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
                thinking_mode=thinking_mode,
                thinking_budget=thinking_budget,
                answer_max_tokens=answer_max_tokens,
                loop_settings=loop_settings,
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


def _unresolved_cap_contact(candidate: Mapping[str, Any]) -> bool:
    """Support both newly classified candidates and historical fixtures/artifacts."""

    return bool(
        candidate.get("unresolved_cap_contact", candidate.get("cap_contact"))
    )


def _answer_limit_contact(candidate: Mapping[str, Any]) -> bool:
    """Support conservative new classification and historical artifacts."""

    return any(
        bool(candidate.get(field))
        for field in ("answer_limit_contact", "answer_truncated", "truncated")
    )


def summarize_task_rows(
    task_rows: Mapping[str, Mapping[str, Any]],
    *,
    max_cap_contact: float = 0.05,
    max_answer_truncation: float | None = None,
    max_periodic_loop_rate: float = 0.25,
) -> dict[str, Any]:
    _require(0.0 <= max_cap_contact <= 1.0, "max_cap_contact must be in [0, 1]")
    if max_answer_truncation is None:
        max_answer_truncation = max_cap_contact
    _require(
        0.0 <= max_answer_truncation <= 1.0,
        "max_answer_truncation must be in [0, 1]",
    )
    _require(
        0.0 <= max_periodic_loop_rate <= 1.0,
        "max_periodic_loop_rate must be in [0, 1]",
    )
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
    cap_contact_rate = sum(bool(candidate.get("cap_contact")) for candidate in candidates) / len(
        candidates
    )
    stage1_length_finish_rate = sum(
        bool(candidate.get("stage1_length_finish")) for candidate in candidates
    ) / len(candidates)
    forced_intervention_rate = sum(
        bool(candidate.get("forced_intervention", candidate.get("forced_close")))
        for candidate in candidates
    ) / len(candidates)
    reasoning_boundary_contact_rate = sum(
        bool(candidate.get("reasoning_boundary_contact")) for candidate in candidates
    ) / len(candidates)
    answer_restart_after_natural_close_rate = sum(
        bool(candidate.get("answer_restart_after_natural_close"))
        for candidate in candidates
    ) / len(candidates)
    periodic_loop_rate = sum(
        bool(candidate.get("periodic_loop")) for candidate in candidates
    ) / len(candidates)
    unresolved_cap_contact_rate = sum(
        _unresolved_cap_contact(candidate) for candidate in candidates
    ) / len(candidates)
    answer_truncation_rate = sum(
        _answer_limit_contact(candidate) for candidate in candidates
    ) / len(candidates)
    adequate_completion_rate = sum(
        not _unresolved_cap_contact(candidate)
        and not _answer_limit_contact(candidate)
        for candidate in candidates
    ) / len(candidates)
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
        "forced_intervention_rate": forced_intervention_rate,
        "stage1_length_finish_rate": stage1_length_finish_rate,
        "reasoning_boundary_contact_rate": reasoning_boundary_contact_rate,
        "answer_restart_after_natural_close_rate": answer_restart_after_natural_close_rate,
        "truncation_rate": answer_truncation_rate,
        "answer_truncation_rate": answer_truncation_rate,
        "answer_limit_contact_rate": answer_truncation_rate,
        "cap_contact_rate": cap_contact_rate,
        "periodic_loop_rate": periodic_loop_rate,
        "unresolved_cap_contact_rate": unresolved_cap_contact_rate,
        "adequate_completion_rate": adequate_completion_rate,
        "budget_max_cap_contact": max_cap_contact,
        "budget_max_answer_truncation": max_answer_truncation,
        "budget_max_periodic_loop_rate": max_periodic_loop_rate,
        "budget_adequacy": (
            unresolved_cap_contact_rate < max_cap_contact
            and answer_truncation_rate < max_answer_truncation
            and periodic_loop_rate <= max_periodic_loop_rate
        ),
    }


def summarize_arm(
    task_rows: Mapping[str, Mapping[str, Any]],
    *,
    max_cap_contact: float = 0.05,
    max_answer_truncation: float | None = None,
    max_periodic_loop_rate: float = 0.25,
) -> dict[str, Any]:
    split_names = sorted({str(task["split"]) for task in task_rows.values()})
    summary_kwargs = {
        "max_cap_contact": max_cap_contact,
        "max_answer_truncation": max_answer_truncation,
        "max_periodic_loop_rate": max_periodic_loop_rate,
    }
    summaries = {"all": summarize_task_rows(task_rows, **summary_kwargs)}
    for split in split_names:
        subset = {task_id: task for task_id, task in task_rows.items() if task["split"] == split}
        summaries[split] = summarize_task_rows(subset, **summary_kwargs)
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


def build_smoke_verdict(
    arms: Mapping[str, Mapping[str, Mapping[str, Any]]],
    summaries: Mapping[str, Mapping[str, Mapping[str, Any]]],
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    required_arms = ("base", "designed_ceiling")
    required_present = all(
        arm in arms and arm in summaries and "all" in summaries[arm]
        for arm in required_arms
    )
    if not required_present:
        return {
            "pass": False,
            "required_arms_present": False,
            "reasons": ["base and designed_ceiling smoke artifacts are required"],
        }

    matched_k = _integer(
        decision.get("smoke_matched_k", 12), where="smoke_matched_k", minimum=1
    )
    task_sets_match = set(arms["base"]) == set(arms["designed_ceiling"])
    task_ids = sorted(set(arms["base"]) & set(arms["designed_ceiling"]))
    matched: dict[str, dict[str, list[Mapping[str, Any]]]] = {
        arm: {} for arm in required_arms
    }
    matched_prefix_complete = task_sets_match and bool(task_ids)
    split_by_task: dict[str, str] = {}
    for task_id in task_ids:
        base_task = arms["base"][task_id]
        designed_task = arms["designed_ceiling"][task_id]
        base_split = _string(base_task.get("split"), where=f"base {task_id}.split")
        designed_split = _string(
            designed_task.get("split"), where=f"designed_ceiling {task_id}.split"
        )
        if base_split != designed_split:
            matched_prefix_complete = False
        split_by_task[task_id] = base_split
        for arm, task in (("base", base_task), ("designed_ceiling", designed_task)):
            raw_candidates = task.get("candidates")
            _require(
                isinstance(raw_candidates, list),
                f"{arm} {task_id}.candidates must be a list",
            )
            ordered = sorted(
                raw_candidates,
                key=lambda candidate: _integer(
                    candidate.get("sample_index"),
                    where=f"{arm} {task_id}.candidate.sample_index",
                ),
            )
            indices = [
                _integer(
                    candidate.get("sample_index"),
                    where=f"{arm} {task_id}.candidate.sample_index",
                )
                for candidate in ordered
            ]
            if len(indices) != len(set(indices)) or len(ordered) < matched_k:
                matched_prefix_complete = False
            matched[arm][task_id] = ordered[:matched_k]

    def arm_rate(arm: str, field: str) -> float | None:
        candidates = [
            candidate
            for task_candidates in matched[arm].values()
            for candidate in task_candidates
        ]
        return (
            sum(bool(candidate.get(field)) for candidate in candidates) / len(candidates)
            if candidates
            else None
        )

    def arm_answer_limit_rate(arm: str) -> float | None:
        candidates = [
            candidate
            for task_candidates in matched[arm].values()
            for candidate in task_candidates
        ]
        return (
            sum(_answer_limit_contact(candidate) for candidate in candidates)
            / len(candidates)
            if candidates
            else None
        )

    def oracle_coverage(arm: str, split: str) -> float | None:
        relevant = [task_id for task_id in task_ids if split_by_task[task_id] == split]
        if not relevant:
            return None
        return sum(
            any(bool(candidate.get("hidden_pass")) for candidate in matched[arm][task_id])
            for task_id in relevant
        ) / len(relevant)

    base_parse = arm_rate("base", "parsed")
    designed_parse = arm_rate("designed_ceiling", "parsed")
    base_truncation = arm_answer_limit_rate("base")
    designed_truncation = arm_answer_limit_rate("designed_ceiling")
    matched_candidates = [
        candidate
        for arm in required_arms
        for task_candidates in matched[arm].values()
        for candidate in task_candidates
    ]
    overall_parse = (
        sum(bool(candidate.get("parsed")) for candidate in matched_candidates)
        / len(matched_candidates)
        if matched_candidates
        else None
    )
    overall_truncation = (
        sum(_answer_limit_contact(candidate) for candidate in matched_candidates)
        / len(matched_candidates)
        if matched_candidates
        else None
    )
    overall_cap_contact = (
        sum(bool(candidate.get("cap_contact")) for candidate in matched_candidates)
        / len(matched_candidates)
        if matched_candidates
        else None
    )
    overall_periodic_loop = (
        sum(bool(candidate.get("periodic_loop")) for candidate in matched_candidates)
        / len(matched_candidates)
        if matched_candidates
        else None
    )
    overall_unresolved_cap_contact = (
        sum(_unresolved_cap_contact(candidate) for candidate in matched_candidates)
        / len(matched_candidates)
        if matched_candidates
        else None
    )
    reuse_task_ids = [
        task_id for task_id in task_ids if split_by_task[task_id] == "smoke_reuse"
    ]
    no_reuse_task_ids = [
        task_id for task_id in task_ids if split_by_task[task_id] == "smoke_no_reuse"
    ]
    designed_macro_reuse_task_ids = [
        task_id
        for task_id in reuse_task_ids
        if any(
            bool(candidate.get("valid")) and bool(candidate.get("macro_used"))
            for candidate in matched["designed_ceiling"][task_id]
        )
    ]
    designed_macro_reuse_candidates = sum(
        bool(candidate.get("valid")) and bool(candidate.get("macro_used"))
        for task_id in reuse_task_ids
        for candidate in matched["designed_ceiling"][task_id]
    )
    base_reuse_oracle = oracle_coverage("base", "smoke_reuse")
    designed_reuse_oracle = oracle_coverage("designed_ceiling", "smoke_reuse")
    base_no_reuse_oracle = oracle_coverage("base", "smoke_no_reuse")
    designed_no_reuse_oracle = oracle_coverage("designed_ceiling", "smoke_no_reuse")
    min_parse = _number(decision["smoke_min_parse_rate"], where="smoke_min_parse_rate")
    min_macro_value = decision.get("smoke_min_macro_tasks")
    if min_macro_value is None:
        min_macro_value = decision["smoke_min_macro_candidates"]
    min_macro_tasks = _integer(min_macro_value, where="smoke_min_macro_tasks")
    max_trunc = _number(
        decision["smoke_max_answer_truncation"], where="smoke_max_answer_truncation"
    )
    max_cap_contact = _number(
        decision["scored_max_cap_contact"], where="scored_max_cap_contact"
    )
    max_periodic_loop_rate = _number(
        decision.get("loop_max_rate", 0.25), where="loop_max_rate"
    )
    _require(
        0.0 <= max_cap_contact <= 1.0,
        "scored_max_cap_contact must be in [0, 1]",
    )
    _require(
        0.0 <= max_periodic_loop_rate <= 1.0,
        "loop_max_rate must be in [0, 1]",
    )
    unresolved_cap_gate = (
        overall_unresolved_cap_contact is not None
        and overall_unresolved_cap_contact < max_cap_contact
    )
    gates = {
        "matched_prefix_complete": matched_prefix_complete,
        "overall_parse": overall_parse is not None and overall_parse >= min_parse,
        "base_parse": base_parse is not None and base_parse >= min_parse,
        "designed_parse": designed_parse is not None and designed_parse >= min_parse,
        "designed_valid_macro_reuse_tasks": (
            len(designed_macro_reuse_task_ids) >= min_macro_tasks
        ),
        "answer_truncation": (
            overall_truncation is not None and overall_truncation < max_trunc
        ),
        "scored_cap_contact": (
            unresolved_cap_gate
        ),
        "scored_unresolved_cap_contact": unresolved_cap_gate,
        "periodic_loop_rate": (
            overall_periodic_loop is not None
            and overall_periodic_loop <= max_periodic_loop_rate
        ),
        "scored_answer_truncation": (
            overall_truncation is not None
            and overall_truncation < max_cap_contact
        ),
        "designed_oracle_not_below_base": (
            base_reuse_oracle is not None
            and designed_reuse_oracle is not None
            and designed_reuse_oracle >= base_reuse_oracle
        ),
    }
    return {
        "pass": all(gates.values()),
        "required_arms_present": True,
        "gates": gates,
        "metrics": {
            "matched_k": matched_k,
            "matched_task_count": len(task_ids),
            "matched_prefix_complete": matched_prefix_complete,
            "overall_parse_rate": overall_parse,
            "base_parse_rate": base_parse,
            "designed_parse_rate": designed_parse,
            "base_truncation_rate": base_truncation,
            "designed_truncation_rate": designed_truncation,
            "base_answer_limit_contact_rate": base_truncation,
            "designed_answer_limit_contact_rate": designed_truncation,
            "designed_valid_macro_using_candidates": designed_macro_reuse_candidates,
            "designed_valid_macro_using_reuse_candidates": designed_macro_reuse_candidates,
            "designed_valid_macro_using_reuse_tasks": len(
                designed_macro_reuse_task_ids
            ),
            "designed_valid_macro_using_reuse_task_ids": designed_macro_reuse_task_ids,
            "overall_truncation_rate": overall_truncation,
            "overall_answer_truncation_rate": overall_truncation,
            "overall_answer_limit_contact_rate": overall_truncation,
            "overall_cap_contact_rate": overall_cap_contact,
            "overall_periodic_loop_rate": overall_periodic_loop,
            "overall_unresolved_cap_contact_rate": overall_unresolved_cap_contact,
            "scored_max_cap_contact": max_cap_contact,
            "loop_max_rate": max_periodic_loop_rate,
            "budget_adequacy": (
                overall_unresolved_cap_contact is not None
                and overall_periodic_loop is not None
                and overall_truncation is not None
                and overall_unresolved_cap_contact < max_cap_contact
                and overall_truncation < max_cap_contact
                and overall_periodic_loop <= max_periodic_loop_rate
            ),
            "base_oracle_hidden_coverage": base_reuse_oracle,
            "designed_oracle_hidden_coverage": designed_reuse_oracle,
            "base_oracle_hidden_coverage_reuse": base_reuse_oracle,
            "designed_oracle_hidden_coverage_reuse": designed_reuse_oracle,
            "base_oracle_hidden_coverage_no_reuse": base_no_reuse_oracle,
            "designed_oracle_hidden_coverage_no_reuse": designed_no_reuse_oracle,
            "matched_by_split": {
                "smoke_reuse": {
                    "n_tasks": len(reuse_task_ids),
                    "base_oracle_hidden_coverage": base_reuse_oracle,
                    "designed_oracle_hidden_coverage": designed_reuse_oracle,
                },
                "smoke_no_reuse": {
                    "n_tasks": len(no_reuse_task_ids),
                    "base_oracle_hidden_coverage": base_no_reuse_oracle,
                    "designed_oracle_hidden_coverage": designed_no_reuse_oracle,
                },
            },
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

    max_cap_contact = _number(
        decision["scored_max_cap_contact"], where="scored_max_cap_contact"
    )
    max_periodic_loop_rate = _number(
        decision.get("loop_max_rate", 0.25), where="loop_max_rate"
    )
    _require(
        0.0 <= max_cap_contact <= 1.0,
        "scored_max_cap_contact must be in [0, 1]",
    )
    _require(
        0.0 <= max_periodic_loop_rate <= 1.0,
        "loop_max_rate must be in [0, 1]",
    )
    confirmatory_arms = sorted(arms)
    budget_by_arm = {
        arm: {
            "cap_contact_rate": summaries[arm]["all"]["cap_contact_rate"],
            "periodic_loop_rate": summaries[arm]["all"]["periodic_loop_rate"],
            "unresolved_cap_contact_rate": summaries[arm]["all"][
                "unresolved_cap_contact_rate"
            ],
            "answer_truncation_rate": summaries[arm]["all"]["answer_truncation_rate"],
            "answer_limit_contact_rate": summaries[arm]["all"][
                "answer_limit_contact_rate"
            ],
            "budget_adequacy": (
                summaries[arm]["all"]["unresolved_cap_contact_rate"]
                < max_cap_contact
                and summaries[arm]["all"]["answer_truncation_rate"]
                < max_cap_contact
                and summaries[arm]["all"]["periodic_loop_rate"]
                <= max_periodic_loop_rate
            ),
        }
        for arm in confirmatory_arms
    }
    offending = [
        arm for arm in confirmatory_arms if not budget_by_arm[arm]["budget_adequacy"]
    ]
    budget_unresolved = bool(offending)
    verdict["budget_adequacy"] = {
        "scored_max_cap_contact": max_cap_contact,
        "loop_max_rate": max_periodic_loop_rate,
        "confirmatory_arms": confirmatory_arms,
        "by_arm": budget_by_arm,
        "offending_arms": offending,
        "all_confirmatory_arms_adequate": not budget_unresolved,
    }
    complete = (
        verdict["designed_interface_usable"]
        and verdict["system_benefit"]
        and verdict["callable_chunking"]
        and verdict["learned_recurrence"]
    )
    verdict["complete_callable_abstraction"] = complete
    verdict["budget_unresolved"] = budget_unresolved
    verdict["claimable_complete_callable_abstraction"] = complete and not budget_unresolved

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

    if budget_unresolved:
        verdict["status"] = "budget_unresolved"
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
        "forced_intervention_samples",
        "stage1_length_finish_samples",
        "reasoning_boundary_contact_samples",
        "answer_restart_after_natural_close_samples",
        "cap_contact_samples",
        "periodic_loop_samples",
        "unresolved_cap_contact_samples",
        "truncated_samples",
        "answer_limit_contact_samples",
        "budget_adequate_samples",
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
                        "forced_intervention_samples": sum(
                            bool(row.get("forced_intervention", row.get("forced_close")))
                            for row in candidates
                        ),
                        "stage1_length_finish_samples": sum(
                            bool(row.get("stage1_length_finish")) for row in candidates
                        ),
                        "reasoning_boundary_contact_samples": sum(
                            bool(row.get("reasoning_boundary_contact")) for row in candidates
                        ),
                        "answer_restart_after_natural_close_samples": sum(
                            bool(row.get("answer_restart_after_natural_close"))
                            for row in candidates
                        ),
                        "cap_contact_samples": sum(
                            bool(row.get("cap_contact")) for row in candidates
                        ),
                        "periodic_loop_samples": sum(
                            bool(row.get("periodic_loop")) for row in candidates
                        ),
                        "unresolved_cap_contact_samples": sum(
                            _unresolved_cap_contact(row) for row in candidates
                        ),
                        "truncated_samples": sum(
                            _answer_limit_contact(row) for row in candidates
                        ),
                        "answer_limit_contact_samples": sum(
                            _answer_limit_contact(row) for row in candidates
                        ),
                        "budget_adequate_samples": sum(
                            not _unresolved_cap_contact(row)
                            and not _answer_limit_contact(row)
                            for row in candidates
                        ),
                    }
                )


def _expected_full_arm_order(
    inference: Mapping[str, Any], libraries: Mapping[str, Any]
) -> list[str]:
    configured = inference.get("arms")
    _require(isinstance(configured, list), "config inference.arms must be a list")
    order: list[str] = []
    for raw_arm in configured:
        arm = str(raw_arm)
        if arm in libraries and arm not in order:
            order.append(arm)
        if arm == "qwen_ranked":
            order.extend(
                name
                for name in sorted(libraries)
                if name.startswith("qwen_random_") and name not in order
            )
    order.extend(name for name in sorted(libraries) if name not in order)
    _require(bool(order) and order[0] == "base", "full arm order must begin with base")
    return order


def _scientific_external_root(exp: Path, config: Mapping[str, Any]) -> Path:
    scientific = config.get("scientific_smoke")
    _require(isinstance(scientific, Mapping), "config scientific_smoke section missing")
    configured = str(scientific.get("external_root", ""))
    override = os.environ.get(scientific_store.ARTIFACT_ROOT_ENV)
    return scientific_store.resolve_artifact_root(override or configured)


def _verify_smoke_artifact_catalog(
    *, exp: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify protocol binding and every selected receipt before row parsing."""

    legacy = exp / "runs" / "smoke"
    _require(
        not legacy.exists(),
        f"repository-local selected smoke copy violates logical promotion: {legacy}",
    )
    catalog_path = exp / "analysis" / "scientific_smoke_artifact_catalog.json"
    selection_path = exp / "analysis" / "smoke_budget_selection.json"
    _require(catalog_path.is_file(), f"missing scientific artifact catalog: {catalog_path}")
    _require(selection_path.is_file(), f"missing smoke budget selection: {selection_path}")
    root = _scientific_external_root(exp, config)
    catalog = scientific_store.verify_catalog(
        catalog_path,
        root,
        protocol_binding=scientific_store.build_protocol_binding(exp),
        selection_file=selection_path,
    )
    inference = config.get("inference")
    _require(isinstance(inference, Mapping), "config inference section missing")
    raw_arms = inference.get("smoke_arms")
    _require(isinstance(raw_arms, list), "config smoke_arms must be a list")
    arm_order = [str(arm) for arm in raw_arms]
    budget, prefixes = scientific_store.selected_bundle_prefixes(catalog, arm_order)
    return {
        "root": root,
        "catalog": catalog,
        "catalog_path": catalog_path,
        "selection_path": selection_path,
        "selected_budget": budget,
        "arm_order": arm_order,
        "paths": {
            arm: scientific_store.bundle_paths(root, prefixes[arm]) for arm in arm_order
        },
    }


def _verify_full_artifact_catalog(
    *,
    exp: Path,
    config: Mapping[str, Any],
    tasks: Mapping[str, Mapping[str, Any]],
    libraries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Verify the catalog, plan, selection, and every referenced receipt."""

    inference = config.get("inference")
    full_run = config.get("full_run")
    _require(isinstance(inference, Mapping), "config inference section missing")
    _require(isinstance(full_run, Mapping), "config full_run section missing")
    raw_root = full_run.get("external_root")
    _require(isinstance(raw_root, str) and raw_root.startswith("/"), "full external root invalid")
    root = Path(raw_root)
    _require(root.is_dir(), f"canonical full external root is missing: {root}")
    resolved_root = root.resolve(strict=True)

    # Amendment 8 forbids a repository-local promoted copy.  Diagnostic smoke
    # rows remain local, but full raw rows must exist only under ``root``.
    legacy_run_dir = exp / "runs" / "full"
    if legacy_run_dir.exists():
        _require(
            not any(legacy_run_dir.rglob("*.jsonl")),
            f"repository-local full raw rows violate logical promotion: {legacy_run_dir}",
        )

    catalog_path = exp / "analysis" / "full_artifact_catalog.json"
    catalog = _read_json(catalog_path)
    _require(isinstance(catalog, dict), "full artifact catalog must be an object")
    _require(
        catalog.get("schema_version") == full_store.FULL_ARTIFACT_SCHEMA_VERSION,
        "full artifact catalog schema mismatch",
    )
    _require(catalog.get("experiment_id") == exp.name, "full artifact catalog experiment mismatch")
    _require(
        catalog.get("canonical_external_root") == str(resolved_root),
        "full artifact catalog root mismatch",
    )

    plan_ref = catalog.get("shard_plan")
    selection_ref = catalog.get("budget_selection")
    _require(isinstance(plan_ref, Mapping), "full artifact catalog lacks shard plan reference")
    _require(isinstance(selection_ref, Mapping), "full artifact catalog lacks selection reference")
    _require(plan_ref.get("path") == "analysis/full_shard_plan.json", "catalog shard plan path drifted")
    _require(
        selection_ref.get("path") == "analysis/full_budget_selection.json",
        "catalog budget selection path drifted",
    )
    plan_path = exp / str(plan_ref["path"])
    selection_path = exp / str(selection_ref["path"])
    _require(
        plan_ref.get("sha256") == full_store.file_integrity(plan_path)["sha256"],
        "catalog shard plan file hash mismatch",
    )
    _require(
        selection_ref.get("sha256") == full_store.file_integrity(selection_path)["sha256"],
        "catalog budget selection file hash mismatch",
    )
    plan = _read_json(plan_path)
    selection = _read_json(selection_path)
    _require(isinstance(plan, dict), "full shard plan must be an object")
    _require(isinstance(selection, dict), "full budget selection must be an object")
    plan_hash = full_store.plan_sha256(plan)
    _require(plan_ref.get("content_sha256") == plan_hash, "catalog shard plan content hash mismatch")
    _require(selection.get("pass") is True, "full budget selection did not pass")
    _require(selection.get("shard_plan_sha256") == plan_hash, "selection shard plan hash mismatch")
    _require(
        selection.get("canonical_external_root") == str(resolved_root),
        "selection external root mismatch",
    )
    selected_budget = selection.get("selected_thinking_budget")
    _require(isinstance(selected_budget, int), "full selection lacks selected thinking budget")
    arm_order = _expected_full_arm_order(inference, libraries)
    full_tasks = [
        {"id": task["id"], "split": task["split"]}
        for task in tasks.values()
        if task["split"] in {"reuse", "no_reuse"}
    ]
    expected_plan = full_store.build_shard_plan(
        full_tasks,
        arm_order,
        base_k=int(inference["base_max_k"]),
        macro_k=int(inference["macro_k"]),
    )
    _require(plan == expected_plan, "full shard plan differs from frozen tasks/arms/K")
    _require(catalog.get("arm_order") == arm_order, "full artifact catalog arm order mismatch")
    selected_tier = catalog.get("selected_tier")
    _require(isinstance(selected_tier, Mapping), "full artifact catalog lacks selected tier")
    _require(selected_tier.get("thinking_budget") == selected_budget, "catalog selected budget mismatch")
    _require(
        selected_tier.get("relative_path") == f"think_{selected_budget}",
        "catalog selected tier pointer mismatch",
    )
    _require(selected_tier.get("logical_promotion_only") is True, "full tier was not logically promoted")
    _require(selected_tier.get("repository_raw_copy") is None, "catalog records a forbidden raw copy")

    expected_completed: set[tuple[int, str, int]] = set()
    tiers = selection.get("tiers")
    _require(isinstance(tiers, list), "full selection lacks tiers")
    for tier in tiers:
        _require(isinstance(tier, Mapping), "full selection tier must be an object")
        budget = _integer(tier.get("budget"), where="full tier budget", minimum=1)
        tier_arms = tier.get("arms")
        _require(isinstance(tier_arms, Mapping), "full selection tier lacks arms")
        for arm, arm_record in tier_arms.items():
            _require(str(arm) in arm_order, f"full selection contains unknown arm {arm}")
            _require(isinstance(arm_record, Mapping), f"full selection arm {arm} invalid")
            shard_records = arm_record.get("shards", [])
            _require(isinstance(shard_records, list), f"full selection arm {arm} shards invalid")
            for shard_record in shard_records:
                _require(isinstance(shard_record, Mapping), "full selection shard invalid")
                if shard_record.get("status") == "complete":
                    expected_completed.add((budget, str(arm), int(shard_record["shard_index"])))

    raw_completed = catalog.get("completed_shards")
    raw_selected = catalog.get("selected_shards")
    _require(isinstance(raw_completed, list), "full catalog completed_shards must be a list")
    _require(isinstance(raw_selected, list), "full catalog selected_shards must be a list")
    catalog_keys: set[tuple[int, str, int]] = set()
    rebuilt_entries: list[dict[str, Any]] = []
    identity_by_arm_tier: dict[tuple[int, str], Mapping[str, Any]] = {}
    for entry in raw_completed:
        _require(isinstance(entry, Mapping), "full catalog shard entry must be an object")
        budget = _integer(entry.get("budget"), where="catalog shard budget", minimum=1)
        arm = _string(entry.get("arm"), where="catalog shard arm")
        shard_index = _integer(entry.get("shard_index"), where="catalog shard index")
        key = (budget, arm, shard_index)
        _require(key not in catalog_keys, f"duplicate full catalog shard {key}")
        catalog_keys.add(key)
        _require(key in expected_completed, f"catalog contains unregistered completed shard {key}")
        spec = full_store.shard_spec(plan, arm, shard_index)
        canonical_dir = full_store.shard_directory(
            root, budget=budget, arm=arm, shard_index=shard_index
        )
        relative_path = entry.get("relative_path")
        _require(isinstance(relative_path, str), f"catalog shard {key} lacks relative path")
        supplied_dir = root / relative_path
        _require(
            supplied_dir.resolve(strict=True) == canonical_dir.resolve(strict=True),
            f"catalog shard {key} path is not canonical",
        )
        identity_key = (budget, arm)
        receipt = full_store.validate_shard_directory(
            supplied_dir,
            root=root,
            shard_plan_sha256=plan_hash,
            budget=budget,
            arm=arm,
            shard_index=shard_index,
            task_ids=spec["task_ids"],
            k=int(spec["k"]),
            expected_identity=identity_by_arm_tier.get(identity_key),
        )
        identity = receipt["identity"]
        expected_sampling_object = harness.SamplingConfig(
            thinking="budget",
            thinking_budget=budget,
            n=int(spec["k"]),
            max_tokens=int(inference["answer_max_tokens"]),
            answer_max_tokens=int(inference["answer_max_tokens"]),
            temperature=float(inference["temperature"]),
            top_p=float(inference["top_p"]),
            top_k=int(inference["top_k"]),
            run_seed=int(config["seeds"]["vllm_solver"]),
        )
        expected_sampling = json.loads(
            json.dumps(dataclasses.asdict(expected_sampling_object), sort_keys=True)
        )
        expected_engine = json.loads(
            json.dumps(
                dataclasses.asdict(
                    harness.EngineConfig(
                        max_model_len=int(inference["max_model_len"]),
                        max_num_seqs=int(inference["max_num_seqs"]),
                        max_num_batched_tokens=int(inference["max_num_batched_tokens"]),
                    )
                ),
                sort_keys=True,
                default=str,
            )
        )
        _require(identity.get("model") == harness.REQUIRED_MODEL_ID, f"catalog shard {key} model mismatch")
        _require(identity.get("model_revision") == harness.MODEL_REVISION, f"catalog shard {key} revision mismatch")
        _require(
            identity.get("schema_version") == local_vllm.RUNNER_SCHEMA_VERSION,
            f"catalog shard {key} runner schema mismatch",
        )
        _require(
            identity.get("runner_sha256")
            == hashlib.sha256((exp / "src" / "vllm_runner.py").read_bytes()).hexdigest(),
            f"catalog shard {key} runner hash mismatch",
        )
        _require(identity.get("adapter") is None, f"catalog shard {key} unexpectedly used an adapter")
        _require(identity.get("sampling") == expected_sampling, f"catalog shard {key} sampling mismatch")
        _require(
            identity.get("resolved_sampling") == expected_sampling_object.resolved_sampling(),
            f"catalog shard {key} resolved sampling mismatch",
        )
        _require(identity.get("engine") == expected_engine, f"catalog shard {key} engine mismatch")
        identity_by_arm_tier.setdefault(identity_key, receipt["identity"])
        rebuilt = full_store.catalog_shard_entry(root, supplied_dir, receipt)
        _require(dict(entry) == rebuilt, f"catalog shard integrity entry drifted: {key}")
        rebuilt_entries.append(rebuilt)
    _require(catalog_keys == expected_completed, "catalog omits a completed full receipt")

    expected_selected_keys = {
        (selected_budget, arm, index)
        for arm in arm_order
        for index in range(int(plan["arms"][arm]["shard_count"]))
    }
    selected_entries = [
        entry for entry in rebuilt_entries if int(entry["budget"]) == selected_budget
    ]
    selected_keys = {
        (int(entry["budget"]), str(entry["arm"]), int(entry["shard_index"]))
        for entry in selected_entries
    }
    _require(selected_keys == expected_selected_keys, "selected full tier is incomplete")
    _require(
        [dict(entry) for entry in raw_selected]
        == sorted(
            selected_entries,
            key=lambda row: (arm_order.index(str(row["arm"])), int(row["shard_index"])),
        ),
        "catalog selected_shards is not the exact logical selected tier",
    )
    return {
        "root": root,
        "plan": plan,
        "selection": selection,
        "catalog": catalog,
        "selected_budget": selected_budget,
        "selected_entries": selected_entries,
        "arm_order": arm_order,
    }


def _load_full_arm_artifacts(
    context: Mapping[str, Any], arm: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = context["root"]
    entries = sorted(
        (entry for entry in context["selected_entries"] if entry["arm"] == arm),
        key=lambda row: int(row["shard_index"]),
    )
    expected_count = int(context["plan"]["arms"][arm]["shard_count"])
    _require(len(entries) == expected_count, f"selected full arm {arm} has missing shards")
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for entry in entries:
        shard_dir = root / str(entry["relative_path"])
        rows.extend(_read_jsonl(shard_dir / "rows.jsonl"))
        summary = _read_json(shard_dir / "runner.meta.json")
        _require(isinstance(summary, dict), f"full shard metadata must be an object: {shard_dir}")
        summaries.append(summary)
    return rows, full_store.aggregate_runner_summaries(summaries)


def _compact_full_task(task: Mapping[str, Any]) -> dict[str, Any]:
    """Drop candidate prose while preserving hashes, counts, and the selected audit."""

    candidates = task.get("candidates")
    _require(isinstance(candidates, list) and bool(candidates), "derived task lacks candidates")
    selected = next((candidate for candidate in candidates if candidate.get("selected")), None)
    aggregate_fields = {
        "parsed_samples": sum(bool(row.get("parsed")) for row in candidates),
        "valid_samples": sum(bool(row.get("valid")) for row in candidates),
        "visible_pass_samples": sum(bool(row.get("visible_pass")) for row in candidates),
        "hidden_pass_samples": sum(bool(row.get("hidden_pass")) for row in candidates),
        "forced_close_samples": sum(bool(row.get("forced_close")) for row in candidates),
        "forced_intervention_samples": sum(
            bool(row.get("forced_intervention", row.get("forced_close")))
            for row in candidates
        ),
        "stage1_length_finish_samples": sum(
            bool(row.get("stage1_length_finish")) for row in candidates
        ),
        "reasoning_boundary_contact_samples": sum(
            bool(row.get("reasoning_boundary_contact")) for row in candidates
        ),
        "answer_restart_after_natural_close_samples": sum(
            bool(row.get("answer_restart_after_natural_close")) for row in candidates
        ),
        "unresolved_cap_contact_samples": sum(_unresolved_cap_contact(row) for row in candidates),
        "periodic_loop_samples": sum(bool(row.get("periodic_loop")) for row in candidates),
        "answer_limit_contact_samples": sum(_answer_limit_contact(row) for row in candidates),
        "sampled_tokens": sum(int(row["sampled_tokens"]) for row in candidates),
        "thinking_tokens": sum(int(row["thinking_tokens"]) for row in candidates),
        "answer_tokens": sum(int(row["answer_tokens"]) for row in candidates),
        "injected_tokens": sum(int(row["injected_tokens"]) for row in candidates),
    }
    selected_fields = None
    if selected is not None:
        keep = (
            "sample_index",
            "completion_sha256",
            "program",
            "expanded_program",
            "surface_depth",
            "expanded_depth",
            "visible_correct",
            "visible_total",
            "visible_pass",
            "hidden_correct",
            "hidden_total",
            "hidden_pass",
            "probe_correct",
            "probe_total",
            "probe_pass",
            "macro_used",
            "macro_tokens",
            "sampled_tokens",
            "thinking_tokens",
            "answer_tokens",
            "injected_tokens",
            "forced_close",
            "forced_intervention",
            "stage1_length_finish",
            "reasoning_boundary_contact",
            "answer_restart_after_natural_close",
            "unresolved_cap_contact",
            "periodic_loop",
            "answer_limit_contact",
        )
        selected_fields = {key: copy.deepcopy(selected.get(key)) for key in keep}
    return {
        "task_id": task["task_id"],
        "split": task["split"],
        "arm": task["arm"],
        "library_id": task["library_id"],
        "target_min_depth": task["target_min_depth"],
        "n_samples": task["n_samples"],
        "unique_prompt_tokens": task["unique_prompt_tokens"],
        "completion_sha256_by_sample": [
            {
                "sample_index": row["sample_index"],
                "sha256": row["completion_sha256"],
            }
            for row in candidates
        ],
        "aggregate": aggregate_fields,
        "oracle_hidden_pass": task["oracle_hidden_pass"],
        "abstained": task["abstained"],
        "selected": selected_fields,
    }


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
    smoke_context: dict[str, Any] | None = None
    full_context: dict[str, Any] | None = None
    if run == "smoke":
        smoke_context = _verify_smoke_artifact_catalog(exp=exp, config=config)
        arm_names = list(smoke_context["arm_order"])
    else:
        full_context = _verify_full_artifact_catalog(
            exp=exp,
            config=config,
            tasks=tasks,
            libraries=libraries,
        )
        arm_paths = []
        arm_names = list(full_context["arm_order"])

    arms: dict[str, dict[str, dict[str, Any]]] = {}
    runner_metadata: dict[str, Any] = {}
    for arm in arm_names:
        if run == "smoke":
            _require(smoke_context is not None, "internal smoke catalog context missing")
            path = smoke_context["paths"][arm].rows
            # The catalog and receipt were fully verified before this first row
            # read, including the hidden-label-bearing tasks.json binding.
            rows = _read_jsonl(path)
            summary_path = path.with_suffix(".meta.json")
            summary = _read_json(summary_path)
            where = str(summary_path)
        else:
            _require(full_context is not None, "internal full catalog context missing")
            rows, summary = _load_full_arm_artifacts(full_context, arm)
            where = f"selected external full shards for {arm}"
        _require(isinstance(summary, dict), f"{where} must contain runner metadata")
        _require(
            summary.get("schema_version") == local_vllm.RUNNER_SCHEMA_VERSION,
            f"{where} is not the pinned vLLM runner schema",
        )
        _require(
            summary.get("runner_sha256") == expected_runner_sha256,
            f"{where} runner hash disagrees with experiment-local vLLM runner",
        )
        _require(isinstance(summary.get("engine"), dict), f"{where} lacks vLLM engine metadata")
        _require(
            isinstance(summary.get("sampling"), dict),
            f"{where} lacks vLLM sampling metadata",
        )
        arms[arm] = analyze_arm_rows(
            arm=arm,
            rows=rows,
            summary=summary,
            tasks=tasks,
            library=libraries[arm],
            decision=decision,
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
    scored_max_cap_contact = _number(
        decision.get("scored_max_cap_contact"),
        where="decision.scored_max_cap_contact",
    )
    _require(
        0.0 <= scored_max_cap_contact <= 1.0,
        "decision.scored_max_cap_contact must be in [0, 1]",
    )
    budget_max_cap_contact = _number(
        decision.get("budget_max_cap_contact", scored_max_cap_contact),
        where="decision.budget_max_cap_contact",
    )
    budget_max_answer_truncation = _number(
        decision.get("budget_max_answer_truncation", scored_max_cap_contact),
        where="decision.budget_max_answer_truncation",
    )
    max_periodic_loop_rate = _number(
        decision.get("loop_max_rate", 0.25),
        where="decision.loop_max_rate",
    )
    _require(
        0.0 <= budget_max_cap_contact <= 1.0,
        "decision.budget_max_cap_contact must be in [0, 1]",
    )
    _require(
        0.0 <= budget_max_answer_truncation <= 1.0,
        "decision.budget_max_answer_truncation must be in [0, 1]",
    )
    _require(
        0.0 <= max_periodic_loop_rate <= 1.0,
        "decision.loop_max_rate must be in [0, 1]",
    )
    summaries = {
        arm: summarize_arm(
            task_rows,
            max_cap_contact=budget_max_cap_contact,
            max_answer_truncation=budget_max_answer_truncation,
            max_periodic_loop_rate=max_periodic_loop_rate,
        )
        for arm, task_rows in arms.items()
    }
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
    per_task_raw = [
        task
        for arm in sorted(arms)
        for _, task in sorted(arms[arm].items())
    ]
    # Both smoke and full derived artifacts are compact. Raw completion text and
    # token arrays exist only in their checksummed external runner artifacts.
    per_task = [_compact_full_task(task) for task in per_task_raw]
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
        **(
            {
                "full_artifact_catalog_sha256": full_store.file_integrity(
                    exp / "analysis" / "full_artifact_catalog.json"
                )["sha256"],
                "selected_thinking_budget": full_context["selected_budget"],
                "raw_artifact_location": "external_checksums_only",
            }
            if full_context is not None
            else {}
        ),
        **(
            {
                "scientific_smoke_artifact_catalog_sha256": full_store.file_integrity(
                    smoke_context["catalog_path"]
                )["sha256"],
                "selected_thinking_budget": smoke_context["selected_budget"],
                "raw_artifact_location": "external_checksums_only",
            }
            if smoke_context is not None
            else {}
        ),
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
