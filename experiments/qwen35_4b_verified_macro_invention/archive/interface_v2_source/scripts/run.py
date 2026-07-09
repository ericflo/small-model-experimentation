#!/usr/bin/env python3
"""Prepare, smoke, and run the verified-macro experiment.

Preparation is deliberately CPU-only.  Every model-facing request in smoke and
full execution goes through the experiment-local :mod:`model_harness` and one
shared :class:`VLLMRunner` instance.  Artifacts are written atomically and
frozen inputs are checked before a scored stage is allowed to start.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


EXP = Path(__file__).resolve().parents[1]
SRC = EXP / "src"
CONFIG_PATH = EXP / "configs" / "default.yaml"
DATA = EXP / "data"
RUNS = EXP / "runs"
ANALYSIS = EXP / "analysis"
sys.path.insert(0, str(SRC))

SCHEMA_VERSION = 1
MAX_SURFACE_CALLS = 5
MAX_EXPANDED_PRIMITIVE_DEPTH = 5
QWEN_RANDOM_DRAWS = 5


class FrozenArtifactError(RuntimeError):
    """Raised when an existing frozen artifact differs from regeneration."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _parse_yaml_scalar(text: str) -> Any:
    text = text.strip()
    if text in {"true", "True"}:
        return True
    if text in {"false", "False"}:
        return False
    if text in {"null", "None", "~"}:
        return None
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        return [] if not inner else [_parse_yaml_scalar(item) for item in inner.split(",")]
    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the repository's small mapping/inline-list YAML subset."""

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        _require(indent % 2 == 0, f"{path}:{line_number}: indentation must use pairs of spaces")
        line = raw.strip()
        _require(":" in line, f"{path}:{line_number}: expected a key/value mapping")
        key, value = line.split(":", 1)
        _require(bool(key) and " " not in key, f"{path}:{line_number}: invalid key")
        while stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        _require(key not in parent, f"{path}:{line_number}: duplicate key {key!r}")
        if value.strip():
            parent[key] = _parse_yaml_scalar(value)
        else:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
    return root


def _canonical_bytes(value: Any, *, pretty: bool = False) -> bytes:
    if pretty:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        ) + "\n"
    else:
        text = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    return text.encode("utf-8")


def _sha256_value(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, _canonical_bytes(value, pretty=True))


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _require(bool(rows), f"refusing to write empty JSONL artifact: {path}")
    payload = b"".join(_canonical_bytes(dict(row)) + b"\n" for row in rows)
    _atomic_bytes(path, payload)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"missing required artifact: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON artifact {path}: {exc}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise ValueError(f"missing required artifact: {path}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
        _require(isinstance(row, dict), f"{path}:{line_number} must be an object")
        rows.append(row)
    _require(bool(rows), f"{path} has no rows")
    return rows


def _freeze_json(path: Path, value: Any) -> None:
    """Create an immutable JSON artifact, or verify byte-identical regeneration."""

    payload = _canonical_bytes(value, pretty=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise FrozenArtifactError(
                f"frozen artifact differs from deterministic regeneration: {path}"
            )
        return
    _atomic_bytes(path, payload)


def _library_id(arm: str, macros: Sequence[Mapping[str, Any]]) -> str:
    digest = _sha256_value(
        {"arm": arm, "expansions": [list(macro["expansion"]) for macro in macros]}
    )[:16]
    return f"{arm}-{digest}"


def _macro_rows(
    expansions: Sequence[Sequence[str]],
    support: Mapping[tuple[str, ...], int],
    *,
    source_names: Mapping[tuple[str, ...], str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(expansions):
        expansion = tuple(str(token) for token in raw)
        row: dict[str, Any] = {
            "token": f"M{index}",
            "expansion": list(expansion),
            "support": int(support[expansion]),
            "length": len(expansion),
        }
        if source_names is not None and expansion in source_names:
            row["source_name"] = source_names[expansion]
        rows.append(row)
    return rows


def _library(
    arm: str,
    provenance: str,
    macros: Sequence[Mapping[str, Any]],
    *,
    draw_seed: int | None = None,
) -> dict[str, Any]:
    normalized = [dict(macro) for macro in macros]
    result: dict[str, Any] = {
        "id": _library_id(arm, normalized),
        "provenance": provenance,
        "macros": normalized,
    }
    if draw_seed is not None:
        result["draw_seed"] = int(draw_seed)
    return result


def _programs_from_payload(payload: Mapping[str, Any], key: str) -> list[dict[str, Any]]:
    programs = payload.get(key)
    _require(isinstance(programs, list) and bool(programs), f"artifact needs non-empty {key}")
    _require(all(isinstance(program, dict) for program in programs), f"{key} entries must be objects")
    return [dict(program) for program in programs]


def _window_support(programs: Sequence[Mapping[str, Any]]) -> Counter[tuple[str, ...]]:
    """Count program-level support for contiguous length-2/3 windows."""

    support: Counter[tuple[str, ...]] = Counter()
    for index, item in enumerate(programs):
        raw = item.get("program")
        _require(isinstance(raw, list) and bool(raw), f"programs[{index}].program is invalid")
        program = tuple(str(token) for token in raw)
        present = {
            program[start : start + length]
            for length in (2, 3)
            for start in range(len(program) - length + 1)
        }
        support.update(present)
    return support


def _support_bin(value: int) -> int:
    """Stable coarse bins used by both pre-registered placebo ensembles."""

    for boundary in (4, 7, 15, 31, 63, 127, 255, 511):
        if value <= boundary:
            return boundary
    return 1023


def _draw_matched_expansions(
    target: Sequence[Sequence[str]],
    candidates: Sequence[Sequence[str]],
    support: Mapping[tuple[str, ...], int],
    *,
    seed: int,
) -> list[tuple[str, ...]]:
    """Draw a unique non-selected length/support-bin matched placebo library."""

    target_tuples = [tuple(expansion) for expansion in target]
    excluded = set(target_tuples)
    candidate_tuples = sorted({tuple(expansion) for expansion in candidates} - excluded)
    rng = random.Random(seed)
    pools: dict[int, list[tuple[str, ...]]] = {}
    for slot, expansion in enumerate(target_tuples):
        profile = (len(expansion), _support_bin(int(support[expansion])))
        pool = [
            candidate
            for candidate in candidate_tuples
            if (len(candidate), _support_bin(int(support[candidate]))) == profile
        ]
        rng.shuffle(pool)
        pools[slot] = pool
        _require(pool, f"no placebo candidate for slot {slot} profile {profile}")

    assignment: dict[int, tuple[str, ...]] = {}

    def search(unfilled: set[int], used: set[tuple[str, ...]]) -> bool:
        if not unfilled:
            return True
        slot = min(unfilled, key=lambda item: sum(value not in used for value in pools[item]))
        for candidate in pools[slot]:
            if candidate in used:
                continue
            assignment[slot] = candidate
            if search(unfilled - {slot}, used | {candidate}):
                return True
            assignment.pop(slot, None)
        return False

    _require(search(set(pools), set()), "could not construct a unique matched placebo library")
    return [assignment[index] for index in range(len(target_tuples))]


def _validate_config(config: Mapping[str, Any]) -> None:
    _require(config.get("model") == "Qwen/Qwen3.5-4B", "the only permitted model is Qwen/Qwen3.5-4B")
    inference = config.get("inference")
    macros = config.get("macros")
    data = config.get("data")
    seeds = config.get("seeds")
    _require(isinstance(inference, Mapping), "config.inference missing")
    _require(isinstance(macros, Mapping), "config.macros missing")
    _require(isinstance(data, Mapping), "config.data missing")
    _require(isinstance(seeds, Mapping), "config.seeds missing")
    _require(inference.get("backend") == "vllm", "this experiment requires the vLLM backend")
    _require(inference.get("thinking") == "budget", "this experiment is frozen to budget thinking")
    _require(int(macros.get("count", 0)) == 8, "the frozen macro library size must be eight")
    _require(int(inference.get("base_max_k", 0)) == 24, "base K must remain 24")
    _require(int(inference.get("macro_k", 0)) == 12, "macro-arm K must remain 12")
    _require(
        int(inference.get("smoke_thinking_budget", 0))
        == int(inference.get("full_thinking_budget", -1))
        == 768,
        "v2 smoke must use the already-preregistered 768-token full budget",
    )
    _require(
        inference.get("smoke_arms") == ["base", "designed_ceiling"],
        "v2 smoke is frozen to the two gate arms",
    )
    _require(int(data.get("smoke_attempt", 0)) == 2, "only the registered v2 smoke is runnable")
    _require(int(seeds.get("smoke_v2", 0)) == 20260710, "v2 smoke seed drifted")


# Domain-specific preparation is defined below the generic artifact/model helpers.


def _load_prepared() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    tasks_payload = _read_json(DATA / "tasks.json")
    libraries_payload = _read_json(DATA / "libraries.json")
    proposal_payload = _read_json(DATA / "proposal_view.json")
    demonstrations_payload = _read_json(DATA / "demonstrations.json")
    _require(isinstance(tasks_payload, dict), "tasks.json must be an object")
    _require(isinstance(libraries_payload, dict), "libraries.json must be an object")
    _require(isinstance(proposal_payload, dict), "proposal_view.json must be an object")
    _require(isinstance(demonstrations_payload, dict), "demonstrations.json must be an object")
    proposal_view = _programs_from_payload(proposal_payload, "programs")
    demonstrations = _programs_from_payload(demonstrations_payload, "demonstrations")
    return tasks_payload, libraries_payload, proposal_view, demonstrations


def _verify_frozen_data(config: Mapping[str, Any]) -> None:
    manifest = _read_json(DATA / "dataset_manifest.json")
    _require(isinstance(manifest, dict), "dataset_manifest.json must be an object")
    _require(manifest.get("schema_version") == SCHEMA_VERSION, "unsupported dataset manifest schema")
    _require(
        manifest.get("config_sha256") == _sha256_value(dict(config)),
        "frozen data was prepared with a different experiment config",
    )
    hashes = manifest.get("artifact_sha256")
    _require(isinstance(hashes, dict) and bool(hashes), "dataset manifest lacks artifact hashes")
    for relative, expected in hashes.items():
        path = DATA / str(relative)
        _require(path.is_file(), f"frozen artifact is missing: {path}")
        actual = _sha256_file(path)
        _require(actual == expected, f"frozen artifact hash mismatch: {path}")
    cpu_gate = _read_json(DATA / "cpu_gate.json")
    _require(isinstance(cpu_gate, dict) and cpu_gate.get("pass") is True, "CPU gate did not pass")
    libraries_payload = _read_json(DATA / "libraries.json")
    _require(isinstance(libraries_payload, dict), "libraries.json must be an object")
    libraries = libraries_payload.get("libraries")
    _require(isinstance(libraries, dict), "libraries.json lacks libraries")
    prepared_hashes = manifest.get("prepared_library_sha256_by_arm")
    _require(isinstance(prepared_hashes, dict) and bool(prepared_hashes), "manifest lacks library hashes")
    for arm, expected in prepared_hashes.items():
        _require(arm in libraries, f"prepared library disappeared: {arm}")
        _require(
            _sha256_value(libraries[arm]) == expected,
            f"prepared library drifted after CPU freeze: {arm}",
        )
    allowed = set(prepared_hashes) | {"qwen_ranked"} | {
        f"qwen_random_{draw}" for draw in range(QWEN_RANDOM_DRAWS)
    }
    _require(not (set(libraries) - allowed), "libraries.json contains an unregistered arm")


def _primitive_descriptions(domain: Any) -> dict[str, str]:
    factory = getattr(domain, "primitive_descriptions", None)
    if callable(factory):
        produced = factory()
        _require(isinstance(produced, Mapping), "primitive_descriptions() must return a mapping")
        result = {str(token): str(description) for token, description in produced.items()}
        _require(bool(result), "primitive inventory cannot be empty")
        return result
    descriptions = getattr(domain, "PRIMITIVE_DESCRIPTIONS", None)
    if isinstance(descriptions, Mapping):
        result = {str(token): str(description) for token, description in descriptions.items()}
    else:
        primitives = getattr(domain, "PRIMITIVES", None)
        _require(isinstance(primitives, Mapping), "macro_domain must expose primitive descriptions")
        result = {str(token): str(description) for token, description in primitives.items()}
    _require(bool(result), "primitive inventory cannot be empty")
    return result


def _proposal_sampling(harness: Any, config: Mapping[str, Any]) -> Any:
    inference = config["inference"]
    return harness.SamplingConfig(
        thinking="budget",
        thinking_budget=int(inference["full_thinking_budget"]),
        n=int(inference["proposal_n"]),
        max_tokens=int(inference["answer_max_tokens"]),
        answer_max_tokens=int(inference["answer_max_tokens"]),
        temperature=float(inference["temperature"]),
        top_p=float(inference["top_p"]),
        top_k=int(inference["top_k"]),
        run_seed=int(config["seeds"]["vllm_proposal"]),
    )


def _solver_sampling(harness: Any, config: Mapping[str, Any], *, run: str, n: int) -> Any:
    inference = config["inference"]
    budget_key = "smoke_thinking_budget" if run == "smoke" else "full_thinking_budget"
    return harness.SamplingConfig(
        thinking="budget",
        thinking_budget=int(inference[budget_key]),
        n=n,
        max_tokens=int(inference["answer_max_tokens"]),
        answer_max_tokens=int(inference["answer_max_tokens"]),
        temperature=float(inference["temperature"]),
        top_p=float(inference["top_p"]),
        top_k=int(inference["top_k"]),
        run_seed=int(config["seeds"]["vllm_solver"]),
    )


def _engine_config(harness: Any, config: Mapping[str, Any]) -> Any:
    inference = config["inference"]
    return harness.EngineConfig(
        max_model_len=int(inference["max_model_len"]),
        max_num_seqs=int(inference["max_num_seqs"]),
        max_num_batched_tokens=int(inference["max_num_batched_tokens"]),
    )


def _write_batch(path: Path, batch: Any) -> None:
    """Persist unmodified runner-native rows and its exact native summary."""

    _atomic_jsonl(path, batch.rows)
    _atomic_json(path.with_suffix(".meta.json"), batch.summary)


def _preflight_records(runner: Any, records: Sequence[dict[str, Any]], sampling: Any) -> dict[str, Any]:
    """Tokenize and fail on context overflow before starting any generation."""

    prepared = runner.prepare(records, sampling.thinking, sampling.allow_custom_prompts)
    runner._check_context(prepared, sampling)  # local runner's exact context contract
    if sampling.thinking == "budget":
        reserve = int(sampling.thinking_budget) + len(runner.close_ids) + sampling.answer_max_tokens
    else:
        reserve = sampling.max_tokens
    counts = [len(record.prompt_token_ids) for record in prepared]
    input_by_id = {str(record["id"]): record for record in records}
    return {
        "schema_version": SCHEMA_VERSION,
        "pass": True,
        "max_model_len": runner.config.max_model_len,
        "generation_reserve_tokens": reserve,
        "n_records": len(prepared),
        "min_prompt_tokens": min(counts),
        "max_prompt_tokens": max(counts),
        "max_prompt_plus_reserve_tokens": max(counts) + reserve,
        "records": [
            {
                "id": record.record_id,
                "input_record_sha256": _sha256_value(input_by_id[record.record_id]),
                "rendered_prompt_sha256": hashlib.sha256(
                    record.prompt_text.encode("utf-8")
                ).hexdigest(),
                "prompt_tokens": len(record.prompt_token_ids),
                "prompt_plus_reserve_tokens": len(record.prompt_token_ids) + reserve,
            }
            for record in prepared
        ],
    }


def _proposal_candidates(
    parsed: Sequence[Any],
    programs: Sequence[Mapping[str, Any]],
    *,
    min_support: int,
    verify_expansion: Any,
) -> tuple[list[tuple[str, ...]], dict[tuple[str, ...], str], dict[str, Any]]:
    """Rank unique supported proposals using model vote/order only, never eval data."""

    support = _window_support(programs)
    votes: Counter[tuple[str, ...]] = Counter()
    order_score: Counter[tuple[str, ...]] = Counter()
    names: dict[tuple[str, ...], str] = {}
    first_seen: dict[tuple[str, ...], tuple[int, int]] = {}
    parse_failures = 0
    accepted_occurrences = 0
    sample_audit: list[dict[str, Any]] = []
    for sample_order, completion in enumerate(parsed):
        proposals = completion.proposals
        if proposals is None:
            parse_failures += 1
            sample_audit.append(
                {
                    "record_id": completion.record_id,
                    "sample_index": completion.sample_index,
                    "parse_error": completion.parse_error,
                    "accepted": [],
                }
            )
            continue
        accepted: list[list[str]] = []
        seen_in_sample: set[tuple[str, ...]] = set()
        for position, proposal in enumerate(proposals):
            expansion = tuple(proposal.expansion)
            verification = verify_expansion(expansion)
            if (
                support.get(expansion, 0) < min_support
                or expansion in seen_in_sample
                or not verification.valid
                or not verification.exact
                or not verification.nondegenerate
            ):
                continue
            seen_in_sample.add(expansion)
            votes[expansion] += 1
            order_score[expansion] += max(1, 8 - position)
            names.setdefault(expansion, proposal.name)
            first_seen.setdefault(expansion, (sample_order, position))
            accepted_occurrences += 1
            accepted.append(list(expansion))
        sample_audit.append(
            {
                "record_id": completion.record_id,
                "sample_index": completion.sample_index,
                "parse_error": completion.parse_error,
                "accepted": accepted,
            }
        )
    ranked_exact = sorted(
        votes,
        key=lambda expansion: (
            -votes[expansion],
            -order_score[expansion],
            first_seen[expansion],
            expansion,
        ),
    )
    ranked: list[tuple[str, ...]] = []
    seen_signatures: set[str] = set()
    dropped_behavioral_aliases: list[dict[str, Any]] = []
    for expansion in ranked_exact:
        signature = str(verify_expansion(expansion).signature)
        if signature in seen_signatures:
            dropped_behavioral_aliases.append(
                {"expansion": list(expansion), "behavioral_signature": signature}
            )
            continue
        seen_signatures.add(signature)
        ranked.append(expansion)
    audit = {
        "parse_failures": parse_failures,
        "parsed_samples": len(parsed) - parse_failures,
        "unique_supported_exact_expansions": len(ranked_exact),
        "unique_supported_candidates": len(ranked),
        "dropped_behavioral_aliases": dropped_behavioral_aliases,
        "accepted_occurrences": accepted_occurrences,
        "ranking_rule": "votes, summed inverse output position, first occurrence, lexical",
        "ranked_candidates": [
            {
                "rank": rank,
                "expansion": list(expansion),
                "source_name": names[expansion],
                "votes": votes[expansion],
                "order_score": order_score[expansion],
                "support": support[expansion],
                "first_seen": list(first_seen[expansion]),
            }
            for rank, expansion in enumerate(ranked, 1)
        ],
        "samples": sample_audit,
    }
    return ranked, names, audit


def _ensure_qwen_libraries(
    *,
    runner: Any,
    harness: Any,
    domain: Any,
    config: Mapping[str, Any],
    libraries_payload: dict[str, Any],
    proposal_view: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Generate/freeze Qwen proposals and train-only matched placebo libraries."""

    proposal_path = RUNS / "macro_proposals.jsonl"
    proposal_meta_path = RUNS / "macro_proposals.meta.json"
    proposal_preflight_path = RUNS / "macro_proposal_preflight.json"
    parsed_path = RUNS / "macro_proposal_parsed.json"
    primitives = _primitive_descriptions(domain)
    record = harness.build_macro_proposal_record(
        "macro-proposal-v1",
        primitives=primitives,
        verified_programs=proposal_view,
        max_macros=int(config["macros"]["count"]),
        meta={
            "split": "train",
            "proposal_view_sha256": _sha256_value(list(proposal_view)),
            "eval_data_used": False,
        },
    )
    sampling = _proposal_sampling(harness, config)
    proposal_present = _validate_runner_artifact(
        proposal_path,
        preflight_path=proposal_preflight_path,
        records=[record],
        sampling=sampling,
        harness=harness,
        config=config,
        expected_n=int(config["inference"]["proposal_n"]),
    )
    if not proposal_present:
        preflight = _preflight_records(runner, [record], sampling)
        _freeze_json(proposal_preflight_path, preflight)
        batch = harness.generate_vllm_batch(runner, [record], sampling)
        _write_batch(proposal_path, batch)
    rows = _read_jsonl(proposal_path)
    summary = _read_json(proposal_meta_path)
    harness.extract_token_accounting(rows, summary)
    allowed_primitives = set(_primitive_descriptions(domain))
    parsed: list[Any] = []
    extraction_audit: list[dict[str, Any]] = []
    for completion in harness.extract_completion_texts(rows):
        try:
            extraction = harness.extract_macro_proposal_lines(
                completion.text,
                allowed_primitives=allowed_primitives,
                max_macros=int(config["macros"]["count"]),
            )
            proposals = extraction.proposals
            error = None if proposals else "no independently valid macro line"
            extraction_audit.append(
                {
                    "record_id": completion.record_id,
                    "sample_index": completion.sample_index,
                    "accepted_line_count": len(proposals),
                    "total_valid_line_count": extraction.total_valid_lines,
                    "extra_valid_lines_capped": extraction.extra_valid_lines_capped,
                    "rejected_nonblank_lines": [
                        dataclasses.asdict(item) for item in extraction.rejected_nonblank_lines
                    ],
                }
            )
        except ValueError as exc:
            proposals = None
            error = str(exc)
            extraction_audit.append(
                {
                    "record_id": completion.record_id,
                    "sample_index": completion.sample_index,
                    "accepted_line_count": 0,
                    "total_valid_line_count": 0,
                    "extra_valid_lines_capped": False,
                    "extraction_error": error,
                    "rejected_nonblank_lines": [],
                }
            )
        parsed.append(
            harness.ParsedMacroCompletion(
                completion.record_id,
                completion.sample_index,
                completion.text,
                proposals,
                error,
            )
        )
    ranked, source_names, audit = _proposal_candidates(
        parsed,
        proposal_view,
        min_support=int(config["macros"]["min_train_support"]),
        verify_expansion=domain.verify_macro,
    )
    target_count = int(config["decision"]["qwen_required_verified_macros"])
    proposal_hash = _sha256_file(proposal_path)
    parsed_payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "proposal_jsonl_sha256": proposal_hash,
        "proposal_view_sha256": _sha256_value(list(proposal_view)),
        "target_count": target_count,
        "construction_status": "complete" if len(ranked) >= target_count else "insufficient_candidates",
        "eval_data_used": False,
        "proposal_extraction_policy": "first eight independently valid macro lines per sample",
        "line_extraction": extraction_audit,
        **audit,
    }
    if parsed_path.exists():
        existing = _read_json(parsed_path)
        _require(existing == parsed_payload, "frozen parsed proposal artifact disagrees with raw proposals")
    else:
        _atomic_json(parsed_path, parsed_payload)

    libraries = libraries_payload.get("libraries")
    _require(isinstance(libraries, dict), "libraries.json lacks libraries")
    existing_qwen = "qwen_ranked" in libraries
    if len(ranked) < target_count:
        _require(not existing_qwen, "qwen_ranked exists despite insufficient frozen candidates")
        print(
            f"[run] Qwen construction gate: FAIL ({len(ranked)}/{target_count} supported unique candidates); "
            "continuing without Qwen solver arms",
            flush=True,
        )
        return libraries_payload

    chosen = ranked[:target_count]
    support = _window_support(proposal_view)
    qwen_macros = _macro_rows(chosen, support, source_names=source_names)
    additions: dict[str, Any] = {
        "qwen_ranked": _library(
            "qwen_ranked",
            "Qwen-proposed train-only supported expansions ranked by frozen proposal outputs",
            qwen_macros,
        )
    }
    qwen_draw_seeds = config["seeds"]["qwen_random_draws"]
    _require(
        isinstance(qwen_draw_seeds, list) and len(qwen_draw_seeds) == QWEN_RANDOM_DRAWS,
        "exactly five qwen_random_draws are required",
    )
    proposal_programs = [tuple(program["program"]) for program in proposal_view]
    qwen_target_macros = tuple(
        domain.Macro(f"M{index}", expansion, int(support[expansion]), source_names[expansion])
        for index, expansion in enumerate(chosen)
    )
    for draw, draw_seed in enumerate(qwen_draw_seeds):
        arm = f"qwen_random_{draw}"
        placebo = domain.make_frequency_matched_random_macros(
            proposal_programs,
            qwen_target_macros,
            seed=int(draw_seed),
            exclude_expansions=chosen,
            min_support=int(config["macros"]["min_train_support"]),
            max_length=int(config["macros"]["max_expansion"]),
        )
        target_signatures = {
            domain.program_signature(macro.expansion) for macro in qwen_target_macros
        }
        placebo_signatures = {
            domain.program_signature(macro.expansion) for macro in placebo
        }
        _require(
            target_signatures.isdisjoint(placebo_signatures),
            f"{arm} contains a behavioral duplicate of qwen_ranked",
        )
        additions[arm] = _library(
            arm,
            "train-only non-selected placebo matched to qwen_ranked length/support bins",
            [macro.to_dict() for macro in placebo],
            draw_seed=int(draw_seed),
        )

    qwen_placebo_sets = [
        frozenset(tuple(macro["expansion"]) for macro in additions[f"qwen_random_{draw}"]["macros"])
        for draw in range(QWEN_RANDOM_DRAWS)
    ]
    _require(
        len(set(qwen_placebo_sets)) == QWEN_RANDOM_DRAWS,
        "qwen matched-placebo draws are not five distinct content sets",
    )
    _require(
        len(set().union(*qwen_placebo_sets)) >= 12,
        "qwen matched-placebo union has fewer than twelve unique expansions",
    )

    if existing_qwen:
        for arm, library in additions.items():
            _require(libraries.get(arm) == library, f"frozen {arm} library disagrees with proposals")
    else:
        collisions = sorted(set(additions) & set(libraries))
        _require(not collisions, f"cannot freeze Qwen libraries over existing arms: {collisions}")
        libraries.update(additions)
        _atomic_json(DATA / "libraries.json", libraries_payload)
    print(f"[run] Qwen construction gate: PASS ({target_count}/{target_count})", flush=True)
    return libraries_payload


def _solver_records(
    *,
    harness: Any,
    domain: Any,
    tasks: Sequence[Mapping[str, Any]],
    arm: str,
    library: Mapping[str, Any],
    demonstrations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    macros = [
        harness.MacroDefinition(
            token=str(macro["token"]),
            expansion=tuple(str(token) for token in macro["expansion"]),
        )
        for macro in library["macros"]
    ]
    callable_macros = arm not in {"base", "mined_hint"}
    records: list[dict[str, Any]] = []
    for task in tasks:
        task_id = str(task["id"])
        records.append(
            harness.build_solver_record(
                f"{task_id}::{arm}",
                primitives=_primitive_descriptions(domain),
                macros=macros,
                macros_callable=callable_macros,
                solved_demonstrations=demonstrations,
                io_examples=task["visible"],
                max_surface_calls=MAX_SURFACE_CALLS,
                max_expanded_primitive_depth=MAX_EXPANDED_PRIMITIVE_DEPTH,
                meta={
                    "task_id": task_id,
                    "split": task["split"],
                    "arm": arm,
                    "library_id": library["id"],
                    "max_surface_calls": MAX_SURFACE_CALLS,
                    "max_expanded_primitive_depth": MAX_EXPANDED_PRIMITIVE_DEPTH,
                },
            )
        )
    return records


def _interface_gate_records(
    *, harness: Any, domain: Any, library: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Create task-independent, plan-given alias-transcription probes.

    These probes expose no evaluation I/O or target.  They only test whether
    Qwen can apply the weighted-depth macro syntax before induction is scored.
    """

    raw_macros = library.get("macros")
    _require(isinstance(raw_macros, list) and len(raw_macros) >= 4, "interface gate needs four macros")
    macro_lines = [
        f"{macro['token']} := {' | '.join(str(token) for token in macro['expansion'])}"
        for macro in raw_macros
    ]
    records: list[dict[str, Any]] = []
    suffixes = (
        ("NEG", "SORT", "ZIGZAG"),
        ("MUL2", "REV", "DIFF"),
        ("PREFIX", "NEG", "ROTL"),
        ("SORT", "SWAP", "NEG"),
    )
    macro_map = {
        str(macro["token"]): tuple(str(token) for token in macro["expansion"])
        for macro in raw_macros
    }
    for index, macro in enumerate(raw_macros[:4]):
        expansion = tuple(str(token) for token in macro["expansion"])
        target = expansion + suffixes[index]
        _require(len(target) == MAX_EXPANDED_PRIMITIVE_DEPTH, "interface target depth drifted")
        compressed = domain.compress_program(target, macro_map)
        _require(len(compressed) < len(target), "interface target is not macro-compressible")
        _require(any(token in macro_map for token in compressed), "interface target lacks an alias")
        records.append(
            {
                "id": f"interface-v2-{index:02d}::designed_ceiling",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Rewrite a supplied verified primitive plan as its shortest legal macro "
                            "surface. Expand aliases exactly when checking equivalence. Return exactly "
                            "one line `PROGRAM: TOKEN | TOKEN | ...`; no prose or markdown."
                        ),
                    },
                    {
                        "role": "user",
                        "content": "\n".join(
                            [
                                "BASE TOKENS: " + ", ".join(domain.PRIMITIVES),
                                "VERIFIED MACROS:",
                                *macro_lines,
                                "",
                                "Maximum surface calls: 5",
                                "Maximum expanded primitive depth: 5",
                                "Use at least one alias when that shortens this supplied plan.",
                                "SUPPLIED VERIFIED PRIMITIVE PLAN: " + " | ".join(target),
                                "FINAL ANSWER: one PROGRAM line only.",
                            ]
                        ),
                    },
                ],
                "meta": {
                    "prompt_kind": "macro_interface_transcription",
                    "split": "train_interface",
                    "arm": "designed_ceiling",
                    "library_id": library["id"],
                    "target_program": list(target),
                    "optimal_surface_calls": len(compressed),
                    "eval_data_used": False,
                },
            }
        )
    return records


def _run_interface_gate(
    *, runner: Any, harness: Any, domain: Any, config: Mapping[str, Any], library: Mapping[str, Any]
) -> dict[str, Any]:
    records = _interface_gate_records(harness=harness, domain=domain, library=library)
    n = int(config["inference"]["interface_k"])
    sampling = _solver_sampling(harness, config, run="smoke", n=n)
    run_dir = RUNS / "interface_v2"
    output_path = run_dir / "designed_ceiling.jsonl"
    preflight_path = run_dir / "designed_ceiling.preflight.json"
    present = _validate_runner_artifact(
        output_path,
        preflight_path=preflight_path,
        records=records,
        sampling=sampling,
        harness=harness,
        config=config,
        expected_n=n,
    )
    if not present:
        preflight = _preflight_records(runner, records, sampling)
        _freeze_json(preflight_path, preflight)
        batch = harness.generate_vllm_batch(runner, records, sampling)
        _write_batch(output_path, batch)

    rows = _read_jsonl(output_path)
    macro_map = {
        str(macro["token"]): tuple(str(token) for token in macro["expansion"])
        for macro in library["macros"]
    }
    parsed = harness.parse_program_outputs(
        rows,
        allowed_tokens=set(domain.PRIMITIVES) | set(macro_map),
        max_surface_calls=MAX_SURFACE_CALLS,
    )
    record_by_id = {str(record["id"]): record for record in records}
    successful_records: set[str] = set()
    valid_samples = 0
    macro_samples = 0
    for completion in parsed:
        if completion.program is None:
            continue
        try:
            expanded = domain.expand_program(completion.program, macro_map)
        except ValueError:
            continue
        record = record_by_id[completion.record_id]
        target = tuple(str(token) for token in record["meta"]["target_program"])
        optimal = int(record["meta"]["optimal_surface_calls"])
        valid_samples += 1
        uses_macro = any(token in macro_map for token in completion.program)
        macro_samples += int(uses_macro)
        if expanded == target and uses_macro and len(completion.program) == optimal:
            successful_records.add(completion.record_id)
    outputs = [output for row in rows for output in row["outputs"]]
    truncation_rate = sum(bool(output.get("truncated")) for output in outputs) / len(outputs)
    gate = {
        "schema_version": SCHEMA_VERSION,
        "pass": len(successful_records) >= 3 and truncation_rate < 0.05,
        "requirements": {
            "successful_records_at_least_three": len(successful_records) >= 3,
            "answer_truncation_below_0_05": truncation_rate < 0.05,
        },
        "metrics": {
            "records": len(records),
            "samples": len(outputs),
            "successful_records": len(successful_records),
            "successful_record_ids": sorted(successful_records),
            "valid_samples": valid_samples,
            "macro_using_samples": macro_samples,
            "answer_truncation_rate": truncation_rate,
        },
        "eval_data_used": False,
    }
    _atomic_json(ANALYSIS / "interface_v2_gate.json", gate)
    _require(gate["pass"] is True, f"macro interface transcription gate failed: {gate}")
    return gate


def _stage_tasks(tasks_payload: Mapping[str, Any], run: str, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = tasks_payload.get("tasks")
    _require(isinstance(raw, list) and bool(raw), "tasks.json has no tasks")
    if run == "smoke":
        selected = [dict(task) for task in raw if str(task.get("split", "")).startswith("smoke")]
        expected = 2 * int(config["data"]["smoke_tasks_per_split"])
    else:
        selected = [dict(task) for task in raw if task.get("split") in {"reuse", "no_reuse"}]
        expected = int(config["data"]["full_reuse_tasks"]) + int(config["data"]["full_no_reuse_tasks"])
    _require(len(selected) == expected, f"{run} task count {len(selected)} != frozen expected {expected}")
    return sorted(selected, key=lambda task: str(task["id"]))


def _arm_order(config: Mapping[str, Any], libraries: Mapping[str, Any]) -> list[str]:
    configured = [str(arm) for arm in config["inference"]["arms"]]
    order: list[str] = []
    for arm in configured:
        if arm in libraries and arm not in order:
            order.append(arm)
        if arm == "qwen_ranked":
            order.extend(
                arm_name
                for arm_name in sorted(libraries)
                if arm_name.startswith("qwen_random_") and arm_name not in order
            )
    order.extend(arm for arm in sorted(libraries) if arm not in order)
    return order


def _expected_engine_metadata(harness: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in dataclasses.asdict(_engine_config(harness, config)).items()
    }


def _validate_runner_artifact(
    path: Path,
    *,
    preflight_path: Path,
    records: Sequence[dict[str, Any]],
    sampling: Any,
    harness: Any,
    config: Mapping[str, Any],
    expected_n: int,
) -> bool:
    """Validate a resumable runner artifact against every frozen input/config field."""

    meta_path = path.with_suffix(".meta.json")
    if not path.exists():
        _require(not meta_path.exists(), f"orphaned runner metadata without rows: {meta_path}")
        # A preflight may legitimately survive interruption before generation.
        return False
    _require(meta_path.is_file(), f"runner rows lack metadata sidecar: {path}")
    _require(preflight_path.is_file(), f"runner rows lack frozen prompt preflight: {path}")

    rows = _read_jsonl(path)
    summary = _read_json(meta_path)
    preflight = _read_json(preflight_path)
    _require(isinstance(summary, dict), f"{meta_path} must contain an object")
    _require(isinstance(preflight, dict), f"{preflight_path} must contain an object")
    import vllm_runner as local_vllm  # type: ignore[import-not-found]

    _require(
        summary.get("schema_version") == local_vllm.RUNNER_SCHEMA_VERSION,
        f"{meta_path} runner schema differs from the experiment-local runner",
    )
    _require(summary.get("model") == "Qwen/Qwen3.5-4B", f"{meta_path} model mismatch")
    _require(
        summary.get("model_revision") == harness.MODEL_REVISION,
        f"{meta_path} model revision mismatch",
    )
    _require(
        summary.get("runner_sha256") == _sha256_file(SRC / "vllm_runner.py"),
        f"{meta_path} runner hash mismatch",
    )
    _require(
        summary.get("engine") == _expected_engine_metadata(harness, config),
        f"{meta_path} engine configuration mismatch",
    )
    expected_sampling = json.loads(_canonical_bytes(dataclasses.asdict(sampling)).decode("utf-8"))
    _require(summary.get("sampling") == expected_sampling, f"{meta_path} sampling/seed mismatch")
    _require(
        summary.get("resolved_sampling") == sampling.resolved_sampling(),
        f"{meta_path} resolved sampling mismatch",
    )
    _require(summary.get("adapter") is None, f"{meta_path} unexpectedly used an adapter")
    accounting = harness.extract_token_accounting(rows, summary)
    _require(accounting.requests == len(records), f"{path} request count mismatch")
    _require(
        accounting.completions == len(records) * expected_n,
        f"{path} completion count mismatch",
    )

    _require(preflight.get("schema_version") == SCHEMA_VERSION, f"{preflight_path} schema mismatch")
    _require(preflight.get("pass") is True, f"{preflight_path} did not pass")
    _require(preflight.get("n_records") == len(records), f"{preflight_path} record count mismatch")
    raw_preflight_rows = preflight.get("records")
    _require(isinstance(raw_preflight_rows, list), f"{preflight_path} lacks records")
    preflight_by_id = {
        str(item.get("id")): item for item in raw_preflight_rows if isinstance(item, dict)
    }
    expected_by_id = {str(record["id"]): record for record in records}
    _require(len(expected_by_id) == len(records), "expected runner records have duplicate ids")
    _require(set(preflight_by_id) == set(expected_by_id), f"{preflight_path} id set mismatch")
    _require(len(rows) == len(records), f"{path} row count mismatch")
    row_by_id = {str(row.get("id")): row for row in rows}
    _require(len(row_by_id) == len(rows), f"{path} contains duplicate row ids")
    _require(set(row_by_id) == set(expected_by_id), f"{path} row id set mismatch")

    for record_id, expected_record in expected_by_id.items():
        frozen_prompt = preflight_by_id[record_id]
        row = row_by_id[record_id]
        _require(
            frozen_prompt.get("input_record_sha256") == _sha256_value(expected_record),
            f"{path} input prompt/meta drift for {record_id}",
        )
        _require(row.get("meta") == expected_record.get("meta"), f"{path} row meta drift for {record_id}")
        _require(
            row.get("prompt_sha256") == frozen_prompt.get("rendered_prompt_sha256"),
            f"{path} rendered prompt hash drift for {record_id}",
        )
        _require(
            row.get("n_prompt_tokens") == frozen_prompt.get("prompt_tokens"),
            f"{path} prompt token count drift for {record_id}",
        )
        outputs = row.get("outputs")
        _require(
            isinstance(outputs, list) and len(outputs) == expected_n,
            f"{path} K mismatch for {record_id}",
        )
    return True


def _validate_cached_arm(
    path: Path,
    *,
    arm: str,
    records: Sequence[dict[str, Any]],
    sampling: Any,
    harness: Any,
    config: Mapping[str, Any],
    expected_n: int,
) -> bool:
    present = _validate_runner_artifact(
        path,
        preflight_path=path.with_suffix(".preflight.json"),
        records=records,
        sampling=sampling,
        harness=harness,
        config=config,
        expected_n=expected_n,
    )
    if not present:
        return False
    found: set[str] = set()
    rows = _read_jsonl(path)
    for row in rows:
        meta = row.get("meta")
        _require(isinstance(meta, dict), f"cached {arm} row lacks meta")
        task_id = str(meta.get("task_id", ""))
        _require(row.get("id") == f"{task_id}::{arm}", f"cached {arm} row id mismatch")
        _require(meta.get("arm") == arm, f"cached {arm} arm mismatch")
        found.add(task_id)
    expected_task_ids = {str(record["meta"]["task_id"]) for record in records}
    _require(found == expected_task_ids, f"cached {arm} task set mismatch")
    return True


def _invoke_analyzer(run: str) -> dict[str, Any]:
    subprocess.run(
        [sys.executable, str(EXP / "scripts" / "analyze.py"), "--run", run],
        cwd=EXP,
        check=True,
    )
    verdict = _read_json(ANALYSIS / f"{run}_verdict.json")
    _require(isinstance(verdict, dict), f"{run} verdict must be an object")
    return verdict


def _require_smoke_gate() -> None:
    # Always recompute: this makes the verdict a function of the currently
    # frozen rows, libraries, tasks, analyzer, and thresholds, never a stale cache.
    verdict = _invoke_analyzer("smoke")
    gate = verdict.get("smoke_gate") if isinstance(verdict, dict) else None
    _require(isinstance(gate, dict), "smoke verdict lacks smoke_gate")
    _require(gate.get("pass") is True, "full run forbidden because the frozen smoke gate did not pass")


def run_model_stage(run: str, config: Mapping[str, Any]) -> None:
    _require(run in {"smoke", "full"}, "run must be smoke or full")
    _verify_frozen_data(config)

    # Lazy imports keep --prepare model-free and make the vLLM boundary explicit.
    import macro_domain as domain  # type: ignore[import-not-found]
    import model_harness as harness  # type: ignore[import-not-found]

    tasks_payload, libraries_payload, proposal_view, demonstrations = _load_prepared()
    libraries = libraries_payload.get("libraries")
    _require(isinstance(libraries, dict), "libraries.json lacks libraries")

    class _NoGeneration:
        def generate(self, *_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("unexpected proposal regeneration")

    runner: Any | None = None
    try:
        if run == "smoke":
            if not (RUNS / "interface_v2" / "designed_ceiling.jsonl").exists():
                runner = harness.VLLMRunner(_engine_config(harness, config))
            _run_interface_gate(
                runner=runner if runner is not None else _NoGeneration(),
                harness=harness,
                domain=domain,
                config=config,
                library=libraries["designed_ceiling"],
            )
        else:
            _require(
                (RUNS / "interface_v2" / "designed_ceiling.jsonl").is_file(),
                "full requires the frozen v2 plan-given interface gate",
            )
            _run_interface_gate(
                runner=_NoGeneration(),
                harness=harness,
                domain=domain,
                config=config,
                library=libraries["designed_ceiling"],
            )
            smoke_tasks = _stage_tasks(tasks_payload, "smoke", config)
            smoke_arms = [str(arm) for arm in config["inference"]["smoke_arms"]]
            for arm in smoke_arms:
                library = libraries[arm]
                k = int(config["decision"]["smoke_matched_k"])
                records = _solver_records(
                    harness=harness,
                    domain=domain,
                    tasks=smoke_tasks,
                    arm=arm,
                    library=library,
                    demonstrations=demonstrations,
                )
                _require(
                    _validate_cached_arm(
                        RUNS / "smoke" / f"{arm}.jsonl",
                        arm=arm,
                        records=records,
                        sampling=_solver_sampling(harness, config, run="smoke", n=k),
                        harness=harness,
                        config=config,
                        expected_n=k,
                    ),
                    f"full requires complete frozen smoke arm {arm}",
                )
            _require_smoke_gate()

            # Qwen proposals are secondary and train-only.  Defer their vLLM
            # cost until the primary base/designed smoke has actually passed.
            if not (RUNS / "macro_proposals.jsonl").exists():
                runner = harness.VLLMRunner(_engine_config(harness, config))
            libraries_payload = _ensure_qwen_libraries(
                runner=runner if runner is not None else _NoGeneration(),
                harness=harness,
                domain=domain,
                config=config,
                libraries_payload=libraries_payload,
                proposal_view=proposal_view,
            )
            libraries = libraries_payload["libraries"]

        # Qwen arms are permitted in full only if construction froze all eight.
        if run == "full" and "qwen_ranked" in libraries:
            _require(
                len(libraries["qwen_ranked"]["macros"])
                == int(config["decision"]["qwen_required_verified_macros"]),
                "full requires an exactly-eight-entry qwen_ranked library",
            )

        tasks = _stage_tasks(tasks_payload, run, config)
        run_dir = RUNS / run
        stage_arms = (
            [str(arm) for arm in config["inference"]["smoke_arms"]]
            if run == "smoke"
            else _arm_order(config, libraries)
        )
        arm_inputs: dict[str, tuple[list[dict[str, Any]], Any, int]] = {}
        missing: list[str] = []
        for arm in stage_arms:
            library = libraries[arm]
            k = (
                int(config["decision"]["smoke_matched_k"])
                if run == "smoke"
                else int(
                    config["inference"]["base_max_k"]
                    if arm == "base"
                    else config["inference"]["macro_k"]
                )
            )
            records = _solver_records(
                harness=harness,
                domain=domain,
                tasks=tasks,
                arm=arm,
                library=library,
                demonstrations=demonstrations,
            )
            sampling = _solver_sampling(harness, config, run=run, n=k)
            arm_inputs[arm] = (records, sampling, k)
            output_path = run_dir / f"{arm}.jsonl"
            if _validate_cached_arm(
                output_path,
                arm=arm,
                records=records,
                sampling=sampling,
                harness=harness,
                config=config,
                expected_n=k,
            ):
                print(f"[run] {run}/{arm}: frozen output exists, skip", flush=True)
            else:
                missing.append(arm)
        if missing and runner is None:
            runner = harness.VLLMRunner(_engine_config(harness, config))

        for arm in missing:
            records, sampling, k = arm_inputs[arm]
            output_path = run_dir / f"{arm}.jsonl"
            print(f"[run] {run}/{arm}: {len(records)} prompts x K={k} through vLLM", flush=True)
            _require(runner is not None, f"internal error: {run}/{arm} has no vLLM runner")
            preflight = _preflight_records(runner, records, sampling)
            _freeze_json(run_dir / f"{arm}.preflight.json", preflight)
            batch = harness.generate_vllm_batch(
                runner,
                records,
                sampling,
            )
            _write_batch(output_path, batch)
    finally:
        if runner is not None:
            runner.close()

    verdict = _invoke_analyzer(run)
    if run == "smoke":
        gate = verdict["smoke_gate"]
        if gate.get("pass") is not True:
            raise RuntimeError(f"smoke gate failed closed: {gate.get('reasons', [])}")


def prepare(config: Mapping[str, Any]) -> None:
    """Create and freeze all model-free data, libraries, manifests, and CPU gates."""

    # Filled against macro_domain's public construction API.  Keeping this
    # import inside the CPU stage ensures importing the runner cannot allocate a
    # model or initialize CUDA.
    import macro_domain as domain  # type: ignore[import-not-found]

    data_config = config["data"]
    macro_config = config["macros"]
    seeds = config["seeds"]
    base_dataset = domain.generate_task_dataset(
        seed=int(seeds["data"]),
        train_programs=int(data_config["train_programs"]),
        smoke_tasks_per_split=int(data_config["smoke_tasks_per_split"]),
        full_reuse_tasks=int(data_config["full_reuse_tasks"]),
        full_no_reuse_tasks=int(data_config["full_no_reuse_tasks"]),
        visible_examples=int(data_config["visible_examples"]),
        hidden_examples=int(data_config["hidden_examples"]),
        probe_inputs=int(data_config["probe_inputs"]),
    )
    # Smoke v1 exposed only its own twelve tasks.  The registered repair keeps
    # construction and the still-unseen full evaluation byte-for-byte fixed,
    # while replacing those twelve tasks with a dedicated seed and new ids.
    v1_smoke = tuple(
        task for task in base_dataset.tasks if task.split in {"smoke_reuse", "smoke_no_reuse"}
    )
    fresh_smoke = domain.generate_fresh_smoke_tasks(
        exclude_tasks=base_dataset.tasks,
        seed=int(seeds["smoke_v2"]),
        tasks_per_split=int(data_config["smoke_tasks_per_split"]),
        visible_examples=int(data_config["visible_examples"]),
        hidden_examples=int(data_config["hidden_examples"]),
        probe_inputs=int(data_config["probe_inputs"]),
        id_prefix="smoke-v2",
    )
    retained = tuple(
        task
        for task in base_dataset.tasks
        if task.split not in {"smoke_reuse", "smoke_no_reuse"}
    )
    dataset = domain.TaskDataset(
        tasks=tuple(sorted((*retained, *fresh_smoke), key=lambda task: task.id)),
        signature_probes=base_dataset.signature_probes,
        reusable_motifs=base_dataset.reusable_motifs,
        decoy_motifs=base_dataset.decoy_motifs,
        seed=base_dataset.seed,
    )
    validation = domain.validate_task_dataset(dataset)
    train_tasks = sorted(dataset.by_split("train"), key=lambda task: task.id)
    _require(
        len(train_tasks) == int(data_config["train_programs"]),
        "domain returned the wrong number of construction programs",
    )
    proposal_rng = random.Random(int(seeds["proposal_view"]))
    proposal_order = list(train_tasks)
    proposal_rng.shuffle(proposal_order)
    proposal_count = int(data_config["proposal_view_programs"])
    demo_count = int(data_config["solver_demonstrations"])
    _require(
        proposal_count + demo_count <= len(proposal_order),
        "proposal view and solver demonstrations exceed the train corpus",
    )
    proposal_tasks = proposal_order[:proposal_count]
    demonstration_tasks = proposal_order[proposal_count : proposal_count + demo_count]
    demonstration_io = int(data_config["demonstration_io_examples"])

    def solved_view(task: Any, *, io_count: int) -> dict[str, Any]:
        _require(io_count <= len(task.visible), f"not enough visible examples for {task.id}")
        return {
            "id": task.id,
            "split": "train",
            "verified": True,
            "program": list(task.program),
            "io": [example.to_dict() for example in task.visible[:io_count]],
        }

    # construction_corpus is an auditable program-only source; proposal_view is
    # the strict subset actually exposed to Qwen and the deterministic miner.
    construction = [
        {
            "id": task.id,
            "split": "train",
            "verified": True,
            "program": list(task.program),
        }
        for task in train_tasks
    ]
    proposal_view = [solved_view(task, io_count=demonstration_io) for task in proposal_tasks]
    demonstrations = [
        solved_view(task, io_count=demonstration_io) for task in demonstration_tasks
    ]
    tasks = [task.to_dict() for task in dataset.tasks if task.split != "train"]

    proposal_programs = [tuple(task.program) for task in proposal_tasks]
    mined = domain.mine_frequent_macros(
        proposal_programs,
        count=int(macro_config["count"]),
        min_support=int(macro_config["min_train_support"]),
        max_length=int(macro_config["max_expansion"]),
    )
    random_draws = seeds["macro_random_draws"]
    _require(
        isinstance(random_draws, list) and len(random_draws) == 5,
        "exactly five macro_random_draws are required",
    )
    random_libraries = [
        domain.make_frequency_matched_random_macros(
            proposal_programs,
            mined,
            seed=int(draw_seed),
            exclude_expansions=[macro.expansion for macro in mined],
            min_support=int(macro_config["min_train_support"]),
            max_length=int(macro_config["max_expansion"]),
        )
        for draw_seed in random_draws
    ]

    stats = domain.collect_subsequence_stats(
        proposal_programs,
        min_length=int(macro_config["min_expansion"]),
        max_length=int(macro_config["max_expansion"]),
    )
    designed_specs = list(domain.REUSABLE_MOTIFS)
    decoys = sorted(
        (motif for motif in domain.DECOY_MOTIFS if motif.expansion in stats),
        key=lambda motif: (-stats[motif.expansion].support, motif.name),
    )
    designed_specs.extend(decoys[: int(macro_config["count"]) - len(designed_specs)])
    _require(
        len(designed_specs) == int(macro_config["count"]),
        "designed ceiling could not fill its frozen eight-entry inventory",
    )
    designed = tuple(
        domain.Macro(
            f"M{index}",
            tuple(motif.expansion),
            int(stats[motif.expansion].support),
            motif.name,
        )
        for index, motif in enumerate(designed_specs)
    )

    empty: list[dict[str, Any]] = []
    mined_rows = [macro.to_dict() for macro in mined]
    libraries: dict[str, Any] = {
        "base": _library("base", "base primitives only", empty),
        "mined": _library(
            "mined",
            "deterministic frequency-ranked nondegenerate train-only subsequences",
            mined_rows,
        ),
        "mined_hint": _library(
            "mined_hint",
            "identical mined chunks displayed but forbidden as callable output actions",
            mined_rows,
        ),
        "designed_ceiling": _library(
            "designed_ceiling",
            "generator-known recurrent motifs; non-discovery ceiling",
            [macro.to_dict() for macro in designed],
        ),
    }
    for draw, (draw_seed, macros) in enumerate(zip(random_draws, random_libraries)):
        arm = f"random_{draw}"
        libraries[arm] = _library(
            arm,
            "train-only non-selected placebo matched to mined length/support bins",
            [macro.to_dict() for macro in macros],
            draw_seed=int(draw_seed),
        )

    eval_tasks = tuple(
        task for task in dataset.tasks if task.split in {"reuse", "no_reuse", "smoke_reuse", "smoke_no_reuse"}
    )
    current_smoke = tuple(
        task for task in dataset.tasks if task.split in {"smoke_reuse", "smoke_no_reuse"}
    )
    full_reuse = dataset.by_split("reuse")
    full_no_reuse = dataset.by_split("no_reuse")
    v1_smoke_programs = {task.program for task in v1_smoke}
    v1_smoke_signatures = {task.signature for task in v1_smoke}
    current_smoke_programs = {task.program for task in current_smoke}
    current_smoke_signatures = {task.signature for task in current_smoke}
    v1_tasks_payload = _read_json(DATA / "smoke_v1_frozen" / "tasks.json")
    _require(isinstance(v1_tasks_payload, dict), "archived v1 tasks must be an object")
    v1_task_rows = v1_tasks_payload.get("tasks")
    _require(isinstance(v1_task_rows, list), "archived v1 tasks lack rows")
    v1_full_rows = sorted(
        (
            dict(task)
            for task in v1_task_rows
            if isinstance(task, dict) and task.get("split") in {"reuse", "no_reuse"}
        ),
        key=lambda task: str(task["id"]),
    )
    current_full_rows = sorted(
        (task.to_dict() for task in (*full_reuse, *full_no_reuse)),
        key=lambda task: str(task["id"]),
    )
    designed_map = {macro.token: macro.expansion for macro in designed}
    reuse_reductions = [
        len(task.program) - len(domain.compress_program(task.program, designed_map))
        for task in full_reuse
    ]
    no_reuse_reductions = [
        len(task.program) - len(domain.compress_program(task.program, designed_map))
        for task in full_no_reuse
    ]
    construction_program_set = {tuple(task.program) for task in train_tasks}
    construction_signatures = {task.signature for task in train_tasks}
    eval_program_set = {tuple(task.program) for task in eval_tasks}
    eval_signatures = {task.signature for task in eval_tasks}
    all_initial_macros = [
        tuple(macro["expansion"])
        for library in libraries.values()
        for macro in library["macros"]
    ]
    verification_rows = [domain.verify_macro(expansion) for expansion in all_initial_macros]
    mined_signatures = {domain.program_signature(macro.expansion) for macro in mined}
    placebo_signature_disjoint = all(
        mined_signatures.isdisjoint(
            {domain.program_signature(macro.expansion) for macro in placebo}
        )
        for placebo in random_libraries
    )
    expected_split_counts = {
        "train": int(data_config["train_programs"]),
        "smoke_reuse": int(data_config["smoke_tasks_per_split"]),
        "smoke_no_reuse": int(data_config["smoke_tasks_per_split"]),
        "reuse": int(data_config["full_reuse_tasks"]),
        "no_reuse": int(data_config["full_no_reuse_tasks"]),
    }
    configured_eval_depths = {int(depth) for depth in data_config["eval_base_depths"]}
    placebo_content_sets = [
        frozenset(macro.expansion for macro in placebo) for placebo in random_libraries
    ]
    placebo_union = set().union(*placebo_content_sets)
    placebo_pairwise: list[dict[str, Any]] = []
    for left in range(len(placebo_content_sets)):
        for right in range(left + 1, len(placebo_content_sets)):
            intersection = placebo_content_sets[left] & placebo_content_sets[right]
            union = placebo_content_sets[left] | placebo_content_sets[right]
            placebo_pairwise.append(
                {
                    "left": f"random_{left}",
                    "right": f"random_{right}",
                    "overlap": len(intersection),
                    "jaccard": len(intersection) / len(union),
                }
            )
    gates = {
        "task_dataset_validation": validation["n_tasks"] == len(dataset.tasks),
        "split_counts_match_config": validation["split_counts"] == expected_split_counts,
        "evaluation_depths_match_config": {
            task.min_depth for task in eval_tasks
        } == configured_eval_depths,
        "construction_eval_program_disjoint": not (construction_program_set & eval_program_set),
        "construction_eval_signature_disjoint": not (
            construction_signatures & eval_signatures
        ),
        "v2_smoke_program_disjoint_from_v1": not (
            current_smoke_programs & v1_smoke_programs
        ),
        "v2_smoke_signature_disjoint_from_v1": not (
            current_smoke_signatures & v1_smoke_signatures
        ),
        "full_evaluation_unchanged_from_v1": current_full_rows == v1_full_rows,
        "all_macros_exact_nondegenerate": all(
            result.valid and result.exact and result.nondegenerate
            for result in verification_rows
        ),
        "no_macro_equals_full_eval_program": not (
            set(all_initial_macros) & eval_program_set
        ),
        "designed_reuse_median_reduction": statistics.median(reuse_reductions) >= 2.0,
        "designed_no_reuse_median_reduction": statistics.median(no_reuse_reductions) <= 0.5,
        "five_frequency_matched_placebos": len(random_libraries) == 5,
        "five_distinct_placebo_content_sets": len(set(placebo_content_sets)) == 5,
        "placebo_union_at_least_twelve": len(placebo_union) >= 12,
        "placebos_exclude_mined_behavioral_duplicates": placebo_signature_disjoint,
    }
    cpu_gate = {
        "pass": all(gates.values()),
        "gates": gates,
        "reasons": [name for name, passed in gates.items() if not passed],
        "metrics": {
            "dataset_validation": validation,
            "reuse_median_surface_reduction": statistics.median(reuse_reductions),
            "no_reuse_median_surface_reduction": statistics.median(no_reuse_reductions),
            "reuse_surface_reductions": dict(sorted(Counter(reuse_reductions).items())),
            "no_reuse_surface_reductions": dict(sorted(Counter(no_reuse_reductions).items())),
            "construction_eval_exact_program_overlap": len(
                construction_program_set & eval_program_set
            ),
            "construction_eval_behavioral_signature_overlap": len(
                construction_signatures & eval_signatures
            ),
            "v2_v1_smoke_exact_program_overlap": len(
                current_smoke_programs & v1_smoke_programs
            ),
            "v2_v1_smoke_behavioral_signature_overlap": len(
                current_smoke_signatures & v1_smoke_signatures
            ),
            "full_evaluation_sha256": _sha256_value(current_full_rows),
            "v1_full_evaluation_sha256": _sha256_value(v1_full_rows),
            "placebo_diversity": {
                "distinct_content_sets": len(set(placebo_content_sets)),
                "union_expansions": len(placebo_union),
                "shared_by_all": len(
                    set.intersection(*(set(value) for value in placebo_content_sets))
                ),
                "pairwise": placebo_pairwise,
            },
            "distribution": {
                split: domain.task_distribution_diagnostics(dataset.by_split(split))
                for split in ("train", "smoke_reuse", "smoke_no_reuse", "reuse", "no_reuse")
            },
        },
    }
    _require(cpu_gate.get("pass") is True, f"CPU gate failed closed: {cpu_gate.get('reasons', [])}")

    construction_payload = {"schema_version": SCHEMA_VERSION, "programs": construction}
    proposal_payload = {"schema_version": SCHEMA_VERSION, "programs": proposal_view}
    demonstrations_payload = {"schema_version": SCHEMA_VERSION, "demonstrations": demonstrations}
    cpu_gate_payload = {"schema_version": SCHEMA_VERSION, **cpu_gate}

    _freeze_json(DATA / "construction_corpus.json", construction_payload)
    _freeze_json(DATA / "proposal_view.json", proposal_payload)
    _freeze_json(DATA / "demonstrations.json", demonstrations_payload)
    _freeze_json(DATA / "cpu_gate.json", cpu_gate_payload)

    immutable_hashes = {
        name: _sha256_file(DATA / name)
        for name in (
            "construction_corpus.json",
            "proposal_view.json",
            "demonstrations.json",
            "cpu_gate.json",
        )
    }
    manifest_core = {
        "schema_version": SCHEMA_VERSION,
        "data_seed": int(config["seeds"]["data"]),
        "smoke_attempt": int(config["data"]["smoke_attempt"]),
        "smoke_seed": int(config["seeds"]["smoke_v2"]),
        "v1_full_evaluation_sha256": _sha256_value(v1_full_rows),
        "v2_smoke_sha256": _sha256_value(
            sorted((task.to_dict() for task in current_smoke), key=lambda task: str(task["id"]))
        ),
        "config_sha256": _sha256_value(dict(config)),
        "prepared_library_sha256_by_arm": {
            arm: _sha256_value(library) for arm, library in sorted(libraries.items())
        },
        "counts": {
            "construction_programs": len(construction),
            "proposal_view_programs": len(proposal_view),
            "demonstrations": len(demonstrations),
            "tasks": len(tasks),
        },
        "semantic_sha256": {
            "construction_programs": _sha256_value(construction),
            "proposal_view": _sha256_value(proposal_view),
            "demonstrations": _sha256_value(demonstrations),
            "tasks": _sha256_value(tasks),
        },
        "cpu_gate_sha256": _sha256_value(cpu_gate_payload),
    }
    tasks_payload = {
        "schema_version": SCHEMA_VERSION,
        "dataset_manifest": manifest_core,
        "tasks": tasks,
    }
    _freeze_json(DATA / "tasks.json", tasks_payload)
    immutable_hashes["tasks.json"] = _sha256_file(DATA / "tasks.json")
    manifest_payload = {**manifest_core, "artifact_sha256": immutable_hashes}
    _freeze_json(DATA / "dataset_manifest.json", manifest_payload)

    prepared_libraries = {"schema_version": SCHEMA_VERSION, "libraries": libraries}
    library_path = DATA / "libraries.json"
    if library_path.exists():
        existing = _read_json(library_path)
        _require(isinstance(existing, dict) and isinstance(existing.get("libraries"), dict), "invalid libraries.json")
        for arm, library in libraries.items():
            _require(existing["libraries"].get(arm) == library, f"prepared library {arm} changed")
    else:
        _atomic_json(library_path, prepared_libraries)
    print(
        f"[run] CPU gate PASS; froze {len(tasks)} tasks, {len(proposal_view)} proposal-view programs, "
        f"and {len(libraries)} non-Qwen arms",
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_mutually_exclusive_group(required=True)
    stages.add_argument("--prepare", action="store_true", help="freeze deterministic CPU artifacts and gates")
    stages.add_argument("--smoke", action="store_true", help="run the gated vLLM smoke workload")
    stages.add_argument("--full", action="store_true", help="run the frozen full vLLM workload")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args(argv)
    if args.config.resolve() != CONFIG_PATH.resolve():
        parser.error(
            "alternate configs are forbidden for this preregistered runner because the committed "
            "analyzer is frozen to configs/default.yaml"
        )
    config = load_config(args.config)
    _validate_config(config)
    if args.prepare:
        prepare(config)
    elif args.smoke:
        run_model_stage("smoke", config)
    else:
        run_model_stage("full", config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
