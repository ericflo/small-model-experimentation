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
import math
import os
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
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

import full_artifacts as full_store  # noqa: E402
import scientific_artifacts as scientific_store  # noqa: E402

SCHEMA_VERSION = 1
MAX_SURFACE_CALLS = 5
MAX_EXPANDED_PRIMITIVE_DEPTH = 5
QWEN_RANDOM_DRAWS = 5
PARENT_COMMIT = "1c8c5bbb81d2a67618891597205ceb2f40f498d8"
PARENT_CONFIG_SHA256 = "89cfa5dc882fba4cb87582404f39ac0acf3cbca46401cb8a8bbff728d3ee2b42"
PARENT_ARTIFACT_SHA256 = {
    "construction_corpus.json": "b32a6608923bd6068069432efa3856bd6fb29068681e7cc34b03d4a72e891e94",
    "cpu_gate.json": "4372a1e268ab718f66e8f7dd2fb3946e39ab1bed2e42a7581911c5db180fa08b",
    "demonstrations.json": "1531b2722c5dc64530cbafda3e20a3de8a52ab537e50dc35ee4ec50a9fae06cf",
    "proposal_view.json": "206343ac2d643edc94b237b01a5bf5cb9e134e8b1f56affa672eaedc6fc2e011",
    "tasks.json": "82fbbd57e26fd392aa8f30ec6f26d370dc08dd78b3279bed6ee2e2174aea5073",
}
PARENT_SMOKE_V1_ARTIFACT_SHA256 = {
    "construction_corpus.json": "b32a6608923bd6068069432efa3856bd6fb29068681e7cc34b03d4a72e891e94",
    "cpu_gate.json": "a0f35b2bf5b9df221c0651fa0c7519bb5986f7482deb4a2874f89bb31fb698f1",
    "dataset_manifest.json": "23a41cd96909c7bd01e3692ad2145d088db837505772ca53eb381ffca6065312",
    "demonstrations.json": "1531b2722c5dc64530cbafda3e20a3de8a52ab537e50dc35ee4ec50a9fae06cf",
    "libraries.json": "a2ae3663753a3a0d0c9614a5d7c1d250506c74fd7879e11e99b66f5c1e43f865",
    "proposal_view.json": "206343ac2d643edc94b237b01a5bf5cb9e134e8b1f56affa672eaedc6fc2e011",
    "tasks.json": "85f684a4ce709f10baca39c8a67b4c7b07d42e24fd13990c955a2c769fc4aa75",
}


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
    context_envelope = config.get("context_envelope")
    scientific_smoke = config.get("scientific_smoke")
    full_run = config.get("full_run")
    _require(isinstance(inference, Mapping), "config.inference missing")
    _require(isinstance(macros, Mapping), "config.macros missing")
    _require(isinstance(data, Mapping), "config.data missing")
    _require(isinstance(seeds, Mapping), "config.seeds missing")
    _require(isinstance(context_envelope, Mapping), "config.context_envelope missing")
    _require(isinstance(scientific_smoke, Mapping), "config.scientific_smoke missing")
    _require(isinstance(full_run, Mapping), "config.full_run missing")
    _require(inference.get("backend") == "vllm", "this experiment requires the vLLM backend")
    _require(inference.get("thinking") == "budget", "this experiment is frozen to budget thinking")
    _require(int(macros.get("count", 0)) == 8, "the frozen macro library size must be eight")
    _require(int(inference.get("base_max_k", 0)) == 24, "base K must remain 24")
    _require(int(inference.get("macro_k", 0)) == 12, "macro-arm K must remain 12")
    _require(
        int(inference.get("scientific_probe_k", 0)) == 4,
        "scientific budget probes must remain K=4",
    )
    ladder = inference.get("thinking_budget_ladder")
    _require(
        ladder == [16384, 32768, 49152, 61440],
        "thinking-budget escalation ladder drifted",
    )
    _require(int(inference.get("answer_max_tokens", 0)) == 512, "answer cap must remain 512")
    _require(int(inference.get("max_model_len", 0)) == 65536, "vLLM context must remain 65,536")
    frozen_context_envelope = {
        "proposal_record_sha256": "df4735015e69149acba33eab02156ed56252ddb512a7bc669efc99b3a1c51e7d",
        "proposal_max_prompt_tokens": 3478,
        "forced_close_tokens": 2,
        "minimum_headroom_tokens": 104,
    }
    _require(
        dict(context_envelope) == frozen_context_envelope,
        "Amendment-9 context-envelope constants drifted",
    )
    _require(int(inference.get("calibration_k", 0)) == 16, "calibration must use K=16")
    decision = config.get("decision")
    _require(isinstance(decision, Mapping), "config.decision missing")
    _require(int(inference.get("max_num_seqs", 0)) == 64, "vLLM concurrency must remain 64")
    _require(
        int(inference.get("max_num_batched_tokens", 0)) == 32768,
        "vLLM batch-token ceiling must remain 32,768",
    )
    _require(int(decision.get("smoke_matched_k", 0)) == 12, "smoke matched K must remain 12")
    _require(
        float(decision.get("budget_max_cap_contact", -1.0)) == 0.05,
        "budget cap-contact ceiling must remain 0.05",
    )
    _require(
        float(decision.get("budget_max_answer_truncation", -1.0)) == 0.05,
        "budget answer-truncation ceiling must remain 0.05",
    )
    _require(float(decision.get("budget_p99_thinking_fraction", 0.0)) == 0.80, "thinking headroom drifted")
    _require(int(decision.get("loop_override_min_budget", 0)) == 16384, "loop override budget drifted")
    _require(int(decision.get("loop_tail_tokens", 0)) == 8192, "loop tail length drifted")
    _require(int(decision.get("loop_max_period_tokens", 0)) == 2048, "loop period search drifted")
    _require(float(decision.get("loop_min_match_rate", 0.0)) == 0.99, "loop threshold drifted")
    _require(float(decision.get("loop_max_rate", 0.0)) == 0.25, "loop-rate ceiling drifted")
    _require(
        inference.get("smoke_arms") == ["base", "designed_ceiling"],
        "v2 smoke is frozen to the two gate arms",
    )
    _require(int(inference.get("interface_k", 0)) == 4, "held-out interface gate must use K=4")
    _require(int(data.get("smoke_attempt", 0)) == 2, "only the registered v2 smoke is runnable")
    _require(int(seeds.get("smoke_v2", 0)) == 20260710, "v2 smoke seed drifted")
    _require(
        scientific_smoke.get("external_root") == str(scientific_store.DEFAULT_ARTIFACT_ROOT),
        "canonical scientific-smoke external root drifted",
    )
    _require(
        full_run.get("external_root")
        == "/workspace/large_artifacts/qwen35_4b_verified_macro_long_context_rerun/full",
        "canonical full external root drifted",
    )
    frozen_full = {
        "triplets": 40,
        "base_shards": 20,
        "base_tasks_per_shard": 6,
        "macro_shards": 10,
        "macro_tasks_per_shard": 12,
        "completions_per_shard": 144,
        "base_early_unresolved": 144,
        "base_early_answer_limit": 144,
        "base_early_periodic_loops": 721,
        "macro_early_unresolved": 72,
        "macro_early_answer_limit": 72,
        "macro_early_periodic_loops": 361,
    }
    for key, expected in frozen_full.items():
        _require(
            int(full_run.get(key, -1)) == expected,
            f"Amendment-8 full-run constant drifted: {key}",
        )


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
    # This is a follow-up over byte-identical, still-unseen v2 smoke/full data.
    # Its inference config intentionally differs from the under-budgeted parent,
    # so bind the copied data to the parent's committed config and exact hashes
    # instead of pretending it was regenerated under this follow-up config.
    _require(
        manifest.get("config_sha256") == PARENT_CONFIG_SHA256,
        "copied data no longer identifies the frozen parent config",
    )
    hashes = manifest.get("artifact_sha256")
    _require(isinstance(hashes, dict) and bool(hashes), "dataset manifest lacks artifact hashes")
    for relative, expected in hashes.items():
        path = DATA / str(relative)
        _require(path.is_file(), f"frozen artifact is missing: {path}")
        actual = _sha256_file(path)
        _require(actual == expected, f"frozen artifact hash mismatch: {path}")
    _require(hashes == PARENT_ARTIFACT_SHA256, "parent immutable-artifact manifest drifted")
    for relative, expected in PARENT_SMOKE_V1_ARTIFACT_SHA256.items():
        path = DATA / "smoke_v1_frozen" / relative
        _require(path.is_file(), f"archived smoke-v1 artifact is missing: {path}")
        actual = _sha256_file(path)
        _require(actual == expected, f"archived smoke-v1 artifact hash mismatch: {path}")
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


def _budgeted_sampling(
    harness: Any,
    config: Mapping[str, Any],
    *,
    budget: int,
    n: int,
    run_seed: int,
) -> Any:
    inference = config["inference"]
    return harness.SamplingConfig(
        thinking="budget",
        thinking_budget=int(budget),
        n=int(n),
        max_tokens=int(inference["answer_max_tokens"]),
        answer_max_tokens=int(inference["answer_max_tokens"]),
        temperature=float(inference["temperature"]),
        top_p=float(inference["top_p"]),
        top_k=int(inference["top_k"]),
        run_seed=int(run_seed),
    )


def _proposal_sampling(harness: Any, config: Mapping[str, Any], *, budget: int) -> Any:
    return _budgeted_sampling(
        harness,
        config,
        budget=budget,
        n=int(config["inference"]["proposal_n"]),
        run_seed=int(config["seeds"]["vllm_proposal"]),
    )


def _solver_sampling(
    harness: Any,
    config: Mapping[str, Any],
    *,
    budget: int,
    n: int,
) -> Any:
    return _budgeted_sampling(
        harness,
        config,
        budget=budget,
        n=n,
        run_seed=int(config["seeds"]["vllm_solver"]),
    )


def _interface_sampling(
    harness: Any,
    config: Mapping[str, Any],
    *,
    budget: int,
    n: int,
    calibration: bool = False,
) -> Any:
    return _budgeted_sampling(
        harness,
        config,
        budget=budget,
        n=n,
        run_seed=int(
            config["seeds"]["vllm_calibration" if calibration else "vllm_solver"]
        ),
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


def _percentile(values: Sequence[int], probability: float) -> int:
    _require(bool(values), "cannot compute a percentile of no values")
    ordered = sorted(int(value) for value in values)
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return ordered[index]


def _periodic_loop_audit(
    output: Mapping[str, Any], *, budget: int, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Detect a near-exact periodic tail without decoding or judging content."""

    decision = config["decision"]
    minimum_budget = int(decision["loop_override_min_budget"])
    tail_size = int(decision["loop_tail_tokens"])
    max_period = int(decision["loop_max_period_tokens"])
    threshold = float(decision["loop_min_match_rate"])
    raw_ids = output.get("retained_thinking_token_ids", output.get("stage1_token_ids"))
    if budget < minimum_budget or not isinstance(raw_ids, list) or len(raw_ids) < tail_size:
        return {
            "periodic_loop": False,
            "period_tokens": None,
            "tail_tokens": min(len(raw_ids), tail_size) if isinstance(raw_ids, list) else 0,
            "match_rate": None,
        }
    tail = [int(token) for token in raw_ids[-tail_size:]]
    best_rate = 0.0
    best_period: int | None = None
    for period in range(1, min(max_period, len(tail) // 2) + 1):
        comparisons = len(tail) - period
        allowed_mismatches = math.floor(comparisons * (1.0 - threshold) + 1e-12)
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
    return {
        "periodic_loop": best_period is not None,
        "period_tokens": best_period,
        "tail_tokens": len(tail),
        "match_rate": best_rate if best_period is not None else None,
    }


def _termination_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    budget: int,
    answer_cap: int,
    config: Mapping[str, Any],
    require_headroom: bool,
) -> dict[str, Any]:
    """Summarize censoring only; never inspect answer content or correctness."""

    outputs: list[Mapping[str, Any]] = []
    contacts: list[dict[str, Any]] = []
    unresolved_contacts: list[dict[str, Any]] = []
    periodic_loops: list[dict[str, Any]] = []
    answer_limit_contacts: list[dict[str, Any]] = []
    stage2_truncations: list[dict[str, Any]] = []
    stage1_length_finishes: list[dict[str, Any]] = []
    forced_interventions: list[dict[str, Any]] = []
    reasoning_boundary_contacts: list[dict[str, Any]] = []
    answer_restarts_after_natural_close: list[dict[str, Any]] = []
    thinking_counts: list[int] = []
    naturally_closed_thinking_counts: list[int] = []
    answer_counts: list[int] = []
    for row in rows:
        raw_outputs = row.get("outputs")
        _require(isinstance(raw_outputs, list) and raw_outputs, "runner row lacks outputs")
        for output in raw_outputs:
            _require(isinstance(output, Mapping), "runner output must be an object")
            outputs.append(output)
            thinking = int(output.get("n_thinking_tokens", -1))
            answer = int(output.get("n_answer_tokens", -1))
            _require(thinking >= 0 and answer >= 0, "runner output lacks token counts")
            _require(thinking <= budget, "runner thinking count exceeds the registered budget")
            thinking_counts.append(thinking)
            answer_counts.append(answer)
            forced_intervention = bool(output.get("forced_close"))
            stage1_length_finish = str(output.get("stage1_finish_reason")) == "length"
            reasoning_boundary_contact = thinking + 1 >= budget
            cap_contact = forced_intervention or reasoning_boundary_contact
            answer_restart = (
                stage1_length_finish
                and not forced_intervention
                and not reasoning_boundary_contact
            )
            termination_record = {
                "id": str(row.get("id")),
                "sample_index": int(output.get("sample_index", -1)),
                "forced_close": forced_intervention,
                "forced_intervention": forced_intervention,
                "stage1_finish_reason": output.get("stage1_finish_reason"),
                "stage1_length_finish": stage1_length_finish,
                "reasoning_boundary_contact": reasoning_boundary_contact,
                "answer_restart_after_natural_close": answer_restart,
                "thinking_tokens": thinking,
            }
            if stage1_length_finish:
                stage1_length_finishes.append(dict(termination_record))
            if forced_intervention:
                forced_interventions.append(dict(termination_record))
            if reasoning_boundary_contact:
                reasoning_boundary_contacts.append(dict(termination_record))
            if answer_restart:
                answer_restarts_after_natural_close.append(dict(termination_record))
            if cap_contact:
                contact = dict(termination_record)
                contacts.append(contact)
                loop = _periodic_loop_audit(output, budget=budget, config=config)
                if loop["periodic_loop"]:
                    periodic_loops.append({**contact, **loop})
                else:
                    unresolved_contacts.append({**contact, **loop})
            else:
                naturally_closed_thinking_counts.append(thinking)
            stage2_truncated = bool(output.get("truncated")) or str(
                output.get("finish_reason")
            ) == "length"
            answer_limit_contact = stage2_truncated or answer >= answer_cap
            if stage2_truncated:
                stage2_truncations.append(
                    {
                        "id": str(row.get("id")),
                        "sample_index": int(output.get("sample_index", -1)),
                        "answer_tokens": answer,
                        "truncated": bool(output.get("truncated")),
                        "finish_reason": output.get("finish_reason"),
                    }
                )
            if answer_limit_contact:
                answer_limit_contacts.append(
                    {
                        "id": str(row.get("id")),
                        "sample_index": int(output.get("sample_index", -1)),
                        "answer_tokens": answer,
                        "answer_max_tokens": answer_cap,
                        "answer_limit_contact": True,
                        "truncated": bool(output.get("truncated")),
                        "finish_reason": output.get("finish_reason"),
                    }
                )
    total = len(outputs)
    decision = config["decision"]
    cap_rate = len(contacts) / total
    unresolved_cap_rate = len(unresolved_contacts) / total
    periodic_loop_rate = len(periodic_loops) / total
    answer_limit_contact_rate = len(answer_limit_contacts) / total
    stage2_truncation_rate = len(stage2_truncations) / total
    p99_thinking = _percentile(thinking_counts, 0.99)
    p99_natural_thinking = (
        _percentile(naturally_closed_thinking_counts, 0.99)
        if naturally_closed_thinking_counts
        else None
    )
    p99_answer = _percentile(answer_counts, 0.99)
    cap_ok = unresolved_cap_rate < float(decision["budget_max_cap_contact"])
    truncation_ok = answer_limit_contact_rate < float(
        decision["budget_max_answer_truncation"]
    )
    raw_thinking_headroom = p99_thinking <= float(decision["budget_p99_thinking_fraction"]) * budget
    thinking_headroom = (
        p99_natural_thinking is not None
        and p99_natural_thinking
        <= float(decision["budget_p99_thinking_fraction"]) * budget
    )
    answer_headroom = p99_answer <= float(decision["budget_p99_answer_fraction"]) * answer_cap
    loop_rate_ok = periodic_loop_rate <= float(decision["loop_max_rate"])
    adequate = cap_ok and truncation_ok and (
        (thinking_headroom and answer_headroom) if require_headroom else True
    ) and loop_rate_ok
    return {
        "samples": total,
        "thinking_budget": budget,
        "answer_max_tokens": answer_cap,
        "cap_contacts": len(contacts),
        "cap_contact_rate": cap_rate,
        "stage1_length_finishes": len(stage1_length_finishes),
        "stage1_length_finish_rate": len(stage1_length_finishes) / total,
        "forced_interventions": len(forced_interventions),
        "forced_intervention_rate": len(forced_interventions) / total,
        "reasoning_boundary_contacts": len(reasoning_boundary_contacts),
        "reasoning_boundary_contact_rate": len(reasoning_boundary_contacts) / total,
        "answer_restarts_after_natural_close": len(answer_restarts_after_natural_close),
        "answer_restart_after_natural_close_rate": (
            len(answer_restarts_after_natural_close) / total
        ),
        "unresolved_cap_contacts": len(unresolved_contacts),
        "unresolved_cap_contact_rate": unresolved_cap_rate,
        "periodic_loop_contacts": len(periodic_loops),
        "periodic_loop_rate": periodic_loop_rate,
        "loop_rate_pass": loop_rate_ok,
        # Historical field names are retained for artifact compatibility.  They
        # conservatively include a natural stop exactly at the registered answer
        # ceiling, which cannot demonstrate unused answer-budget headroom.
        "answer_truncations": len(answer_limit_contacts),
        "answer_truncation_rate": answer_limit_contact_rate,
        "answer_limit_contacts": len(answer_limit_contacts),
        "answer_limit_contact_rate": answer_limit_contact_rate,
        "stage2_truncations": len(stage2_truncations),
        "stage2_truncation_rate": stage2_truncation_rate,
        "p99_thinking_tokens": p99_thinking,
        "p99_naturally_closed_thinking_tokens": p99_natural_thinking,
        "p99_answer_tokens": p99_answer,
        "thinking_headroom_pass": thinking_headroom,
        "raw_thinking_headroom_pass": raw_thinking_headroom,
        "answer_headroom_pass": answer_headroom,
        "headroom_required": require_headroom,
        "adequate": adequate,
        "cap_contact_samples": contacts,
        "stage1_length_finish_samples": stage1_length_finishes,
        "forced_intervention_samples": forced_interventions,
        "reasoning_boundary_contact_samples": reasoning_boundary_contacts,
        "answer_restart_after_natural_close_samples": answer_restarts_after_natural_close,
        "unresolved_cap_contact_samples": unresolved_contacts,
        "periodic_loop_samples": periodic_loops,
        "answer_truncated_samples": answer_limit_contacts,
        "answer_limit_contact_samples": answer_limit_contacts,
        "stage2_truncated_samples": stage2_truncations,
        "selection_uses_output_content": False,
    }


def _stage1_prefix_audit(
    lower_rows: Sequence[Mapping[str, Any]], higher_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    def samples(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int], list[int]]:
        result: dict[tuple[str, int], list[int]] = {}
        for row in rows:
            raw_outputs = row.get("outputs")
            _require(isinstance(raw_outputs, list), "prefix-audit row lacks outputs")
            for output in raw_outputs:
                _require(isinstance(output, Mapping), "prefix-audit output must be an object")
                key = (str(row.get("id")), int(output.get("sample_index", -1)))
                token_ids = output.get("stage1_token_ids")
                _require(isinstance(token_ids, list), "prefix-audit output lacks stage1 tokens")
                result[key] = [int(token) for token in token_ids]
        return result

    lower = samples(lower_rows)
    higher = samples(higher_rows)
    _require(set(lower) == set(higher), "budget tiers have different sample identities")
    failures = []
    for key, prefix in lower.items():
        candidate = higher[key]
        if candidate[: len(prefix)] != prefix:
            failures.append({"id": key[0], "sample_index": key[1]})
    return {
        "samples": len(lower),
        "pass": not failures,
        "failures": failures,
    }


def _budget_ladder(config: Mapping[str, Any]) -> list[int]:
    return [int(value) for value in config["inference"]["thinking_budget_ladder"]]


def _artifact_hashes(path: Path) -> dict[str, str]:
    return {
        "rows": _sha256_file(path),
        "meta": _sha256_file(path.with_suffix(".meta.json")),
        "preflight": _sha256_file(path.with_suffix(".preflight.json")),
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


def _macro_proposal_record(
    *,
    harness: Any,
    domain: Any,
    config: Mapping[str, Any],
    proposal_view: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the one frozen train-only proposal request without loading a model."""

    return harness.build_macro_proposal_record(
        "macro-proposal-v1",
        primitives=_primitive_descriptions(domain),
        verified_programs=proposal_view,
        max_macros=int(config["macros"]["count"]),
        meta={
            "split": "train",
            "proposal_view_sha256": _sha256_value(list(proposal_view)),
            "eval_data_used": False,
        },
    )


def _rendered_proposal_prompt_tokens_cpu(
    *, harness: Any, proposal_record: Mapping[str, Any]
) -> int:
    """Tokenize the frozen proposal prompt on CPU before vLLM loads the model."""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        harness.MODEL_ID,
        revision=harness.MODEL_REVISION,
        trust_remote_code=True,
        use_fast=True,
    )
    messages = proposal_record.get("messages")
    _require(isinstance(messages, list) and messages, "proposal record lacks messages")
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    except TypeError as exc:
        raise RuntimeError(
            "the pinned Qwen3.5 chat template rejected enable_thinking during the "
            "pre-model context-envelope guard"
        ) from exc
    _require(isinstance(rendered, str), "proposal chat template did not return text")
    token_ids = tokenizer.encode(rendered, add_special_tokens=False)
    _require(token_ids, "proposal prompt tokenized to an empty sequence")
    return len(token_ids)


def _context_envelope_regression(
    *,
    config: Mapping[str, Any],
    proposal_record: Mapping[str, Any],
    observed_prompt_tokens: int | None = None,
) -> dict[str, Any]:
    """Fail model-free if the preregistered largest-rung context bound drifts.

    The pinned proposal-record hash freezes the prompt builder inputs/messages.
    When ``observed_prompt_tokens`` is supplied, the pinned CPU tokenizer has
    also verified the rendered 3,478-token count before model loading. Runtime
    preflight verifies the same count again before proposal generation.
    """

    envelope = config.get("context_envelope")
    inference = config.get("inference")
    _require(isinstance(envelope, Mapping), "config.context_envelope missing")
    _require(isinstance(inference, Mapping), "config.inference missing")
    actual_record_sha256 = _sha256_value(dict(proposal_record))
    expected_record_sha256 = str(envelope["proposal_record_sha256"])
    _require(
        actual_record_sha256 == expected_record_sha256,
        "frozen macro-proposal record drifted from the context-envelope audit",
    )
    largest_budget = max(_budget_ladder(config))
    proposal_prompt_tokens = int(envelope["proposal_max_prompt_tokens"])
    if observed_prompt_tokens is not None:
        _require(
            int(observed_prompt_tokens) == proposal_prompt_tokens,
            "rendered macro-proposal prompt drifted from the 3,478-token envelope",
        )
    forced_close_tokens = int(envelope["forced_close_tokens"])
    answer_tokens = int(inference["answer_max_tokens"])
    max_model_len = int(inference["max_model_len"])
    reserve = largest_budget + forced_close_tokens + answer_tokens
    total = proposal_prompt_tokens + reserve
    headroom = max_model_len - total
    minimum_headroom = int(envelope["minimum_headroom_tokens"])
    _require(total <= max_model_len, "largest-rung macro proposal exceeds max_model_len")
    _require(
        headroom >= minimum_headroom,
        "largest-rung macro-proposal context headroom fell below the frozen guard",
    )
    return {
        "proposal_record_sha256": actual_record_sha256,
        "proposal_max_prompt_tokens": proposal_prompt_tokens,
        "observed_prompt_tokens": observed_prompt_tokens,
        "prompt_token_count_verified": observed_prompt_tokens is not None,
        "largest_thinking_budget": largest_budget,
        "forced_close_tokens": forced_close_tokens,
        "answer_max_tokens": answer_tokens,
        "generation_reserve_tokens": reserve,
        "max_prompt_plus_reserve_tokens": total,
        "max_model_len": max_model_len,
        "headroom_tokens": headroom,
        "minimum_headroom_tokens": minimum_headroom,
        "pass": True,
    }


def _ensure_qwen_libraries(
    *,
    runner: Any,
    harness: Any,
    domain: Any,
    config: Mapping[str, Any],
    libraries_payload: dict[str, Any],
    proposal_view: Sequence[Mapping[str, Any]],
    starting_budget: int,
) -> dict[str, Any]:
    """Generate/freeze Qwen proposals and train-only matched placebo libraries."""

    parsed_path = ANALYSIS / "macro_proposal_parsed.json"
    record = _macro_proposal_record(
        harness=harness,
        domain=domain,
        config=config,
        proposal_view=proposal_view,
    )
    _context_envelope_regression(config=config, proposal_record=record)
    ladder = _budget_ladder(config)
    _require(starting_budget in ladder, "proposal starting budget is not registered")
    n = int(config["inference"]["proposal_n"])
    answer_cap = int(config["inference"]["answer_max_tokens"])
    tiers: list[dict[str, Any]] = []
    previous_rows: list[dict[str, Any]] | None = None
    proposal_path: Path | None = None
    for budget in ladder[ladder.index(starting_budget) :]:
        candidate_path = RUNS / "macro_proposal" / f"think_{budget}" / "proposals.jsonl"
        sampling = _proposal_sampling(harness, config, budget=budget)
        proposal_present = _validate_runner_artifact(
            candidate_path,
            preflight_path=candidate_path.with_suffix(".preflight.json"),
            records=[record],
            sampling=sampling,
            harness=harness,
            config=config,
            expected_n=n,
        )
        if not proposal_present:
            proposal_preflight = _preflight_records(runner, [record], sampling)
            _require(
                proposal_preflight["max_prompt_tokens"]
                == int(config["context_envelope"]["proposal_max_prompt_tokens"]),
                "rendered macro-proposal prompt drifted from the 3,478-token envelope",
            )
            _freeze_json(
                candidate_path.with_suffix(".preflight.json"),
                proposal_preflight,
            )
            batch = harness.generate_vllm_batch(runner, [record], sampling)
            _write_batch(candidate_path, batch)
        candidate_rows = _read_jsonl(candidate_path)
        prefix = (
            None
            if previous_rows is None
            else _stage1_prefix_audit(previous_rows, candidate_rows)
        )
        termination = _termination_metrics(
            candidate_rows,
            budget=budget,
            answer_cap=answer_cap,
            config=config,
            require_headroom=False,
        )
        tiers.append(
            {
                "budget": budget,
                "termination": termination,
                "prefix_audit_vs_previous_tier": prefix,
                "artifacts": _artifact_hashes(candidate_path),
            }
        )
        previous_rows = candidate_rows
        if termination["adequate"]:
            proposal_path = candidate_path
            break
    proposal_selection = {
        "schema_version": SCHEMA_VERSION,
        "pass": proposal_path is not None,
        "selected_thinking_budget": (
            int(proposal_path.parent.name.removeprefix("think_")) if proposal_path else None
        ),
        "selection_uses_output_content": False,
        "eval_data_used": False,
        "tiers": tiers,
    }
    _atomic_json(ANALYSIS / "proposal_budget_selection.json", proposal_selection)
    _require(proposal_path is not None, "macro proposal remained censoring-bound at every budget")
    proposal_meta_path = proposal_path.with_suffix(".meta.json")
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
        "selected_thinking_budget": proposal_selection["selected_thinking_budget"],
        "proposal_budget_selection_sha256": _sha256_file(
            ANALYSIS / "proposal_budget_selection.json"
        ),
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


def _make_interface_records(
    *,
    domain: Any,
    library: Mapping[str, Any],
    specs: Sequence[tuple[int, tuple[str, str, str]]],
    id_prefix: str,
    prompt_kind: str,
) -> list[dict[str, Any]]:
    """Create task-independent, plan-given probes without evaluation data."""

    raw_macros = library.get("macros")
    _require(isinstance(raw_macros, list) and len(raw_macros) == 8, "interface needs eight macros")
    macro_lines = [
        f"{macro['token']} := {' | '.join(str(token) for token in macro['expansion'])}"
        for macro in raw_macros
    ]
    macro_map = {
        str(macro["token"]): tuple(str(token) for token in macro["expansion"])
        for macro in raw_macros
    }
    records: list[dict[str, Any]] = []
    for index, (macro_index, suffix) in enumerate(specs):
        macro = raw_macros[macro_index]
        expansion = tuple(str(token) for token in macro["expansion"])
        target = expansion + suffix
        _require(len(target) == MAX_EXPANDED_PRIMITIVE_DEPTH, "interface target depth drifted")
        compressed = domain.compress_program(target, macro_map)
        _require(len(compressed) < len(target), "interface target is not macro-compressible")
        _require(any(token in macro_map for token in compressed), "interface target lacks an alias")
        records.append(
            {
                "id": f"{id_prefix}-{index:02d}::designed_ceiling",
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
                    "prompt_kind": prompt_kind,
                    "split": "train_interface",
                    "arm": "designed_ceiling",
                    "library_id": library["id"],
                    "target_program": list(target),
                    "optimal_surface_calls": len(compressed),
                    "eval_data_used": False,
                },
            }
        )
    _require(len({record["id"] for record in records}) == len(records), "duplicate interface ids")
    return records


def _calibration_records(*, domain: Any, library: Mapping[str, Any]) -> list[dict[str, Any]]:
    specs = (
        (0, ("NEG", "SORT", "ZIGZAG")),
        (1, ("MUL2", "REV", "DIFF")),
        (2, ("PREFIX", "NEG", "ROTL")),
        (3, ("SORT", "SWAP", "NEG")),
    )
    return _make_interface_records(
        domain=domain,
        library=library,
        specs=specs,
        id_prefix="budget-calibration",
        prompt_kind="budget_calibration_plan_given",
    )


def _interface_gate_records(
    *, harness: Any, domain: Any, library: Mapping[str, Any]
) -> list[dict[str, Any]]:
    del harness  # Kept in the signature for compatibility with the parent harness tests.
    suffixes = (
        ("ADD1", "SORT", "DIFF"),
        ("NEG", "ROTL", "ZIGZAG"),
        ("MUL2", "REV", "SORT"),
        ("PREFIX", "SWAP", "SORT"),
        ("DIFF", "NEG", "ROTL"),
        ("ZIGZAG", "ADD1", "REV"),
        ("SORT", "MUL2", "ROTL"),
        ("REV", "PREFIX", "NEG"),
        ("DIFF", "SWAP", "ADD1"),
        ("PREFIX", "ROTL", "SORT"),
        ("REV", "DIFF", "MUL2"),
        ("ROTL", "ZIGZAG", "PREFIX"),
        ("NEG", "REV", "SWAP"),
        ("SORT", "DIFF", "REV"),
        ("ADD1", "NEG", "ROTL"),
        ("MUL2", "PREFIX", "ZIGZAG"),
    )
    specs = tuple((index % 8, suffix) for index, suffix in enumerate(suffixes))
    records = _make_interface_records(
        domain=domain,
        library=library,
        specs=specs,
        id_prefix="interface-long-context-heldout",
        prompt_kind="macro_interface_transcription_heldout",
    )
    _require(len(records) == 16, "held-out interface set must contain sixteen records")
    return records


def _run_budget_calibration(
    *, runner: Any, harness: Any, domain: Any, config: Mapping[str, Any], library: Mapping[str, Any]
) -> dict[str, Any]:
    records = _calibration_records(domain=domain, library=library)
    n = int(config["inference"]["calibration_k"])
    answer_cap = int(config["inference"]["answer_max_tokens"])
    tiers: list[dict[str, Any]] = []
    previous_rows: list[dict[str, Any]] | None = None
    selected: int | None = None
    for budget in _budget_ladder(config):
        run_dir = RUNS / "budget_calibration" / f"think_{budget}"
        output_path = run_dir / "calibration.jsonl"
        sampling = _interface_sampling(
            harness, config, budget=budget, n=n, calibration=True
        )
        present = _validate_runner_artifact(
            output_path,
            preflight_path=output_path.with_suffix(".preflight.json"),
            records=records,
            sampling=sampling,
            harness=harness,
            config=config,
            expected_n=n,
        )
        if not present:
            _freeze_json(output_path.with_suffix(".preflight.json"), _preflight_records(runner, records, sampling))
            batch = harness.generate_vllm_batch(runner, records, sampling)
            _write_batch(output_path, batch)
        rows = _read_jsonl(output_path)
        prefix = None if previous_rows is None else _stage1_prefix_audit(previous_rows, rows)
        metrics = _termination_metrics(
            rows,
            budget=budget,
            answer_cap=answer_cap,
            config=config,
            require_headroom=True,
        )
        tiers.append(
            {
                "budget": budget,
                "metrics": metrics,
                "prefix_audit_vs_previous_tier": prefix,
                "artifacts": _artifact_hashes(output_path),
            }
        )
        previous_rows = rows
        if metrics["adequate"]:
            selected = budget
            break
    result = {
        "schema_version": SCHEMA_VERSION,
        "pass": selected is not None,
        "selected_thinking_budget": selected,
        "selection_rule": "smallest rung passing cap, truncation, and token-headroom metadata gates",
        "selection_uses_output_content": False,
        "records": len(records),
        "samples_per_record": n,
        "eval_data_used": False,
        "tiers": tiers,
    }
    _atomic_json(ANALYSIS / "budget_selection.json", result)
    _require(selected is not None, "all registered thinking budgets remained censoring-bound")
    return result


def _run_interface_gate(
    *,
    runner: Any,
    harness: Any,
    domain: Any,
    config: Mapping[str, Any],
    library: Mapping[str, Any],
    starting_budget: int,
) -> dict[str, Any]:
    records = _interface_gate_records(harness=harness, domain=domain, library=library)
    n = int(config["inference"]["interface_k"])
    answer_cap = int(config["inference"]["answer_max_tokens"])
    ladder = _budget_ladder(config)
    _require(starting_budget in ladder, "interface starting budget is not registered")
    tier_audits: list[dict[str, Any]] = []
    previous_rows: list[dict[str, Any]] | None = None
    for budget in ladder[ladder.index(starting_budget) :]:
        run_dir = RUNS / "interface" / f"think_{budget}"
        output_path = run_dir / "designed_ceiling.jsonl"
        sampling = _interface_sampling(harness, config, budget=budget, n=n)
        present = _validate_runner_artifact(
            output_path,
            preflight_path=output_path.with_suffix(".preflight.json"),
            records=records,
            sampling=sampling,
            harness=harness,
            config=config,
            expected_n=n,
        )
        if not present:
            _freeze_json(output_path.with_suffix(".preflight.json"), _preflight_records(runner, records, sampling))
            batch = harness.generate_vllm_batch(runner, records, sampling)
            _write_batch(output_path, batch)
        rows = _read_jsonl(output_path)
        prefix = None if previous_rows is None else _stage1_prefix_audit(previous_rows, rows)
        termination = _termination_metrics(
            rows,
            budget=budget,
            answer_cap=answer_cap,
            config=config,
            require_headroom=False,
        )
        tier_audits.append(
            {
                "budget": budget,
                "termination": termination,
                "prefix_audit_vs_previous_tier": prefix,
                "artifacts": _artifact_hashes(output_path),
            }
        )
        previous_rows = rows
        if not termination["adequate"]:
            continue

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
        required = int(config["decision"]["interface_min_successful_records"])
        gate = {
            "schema_version": SCHEMA_VERSION,
            "pass": len(successful_records) >= required,
            "thinking": "budget",
            "selected_thinking_budget": budget,
            "requirements": {
                "successful_records": required,
                "termination_adequate": True,
            },
            "metrics": {
                "records": len(records),
                "samples": sum(len(row["outputs"]) for row in rows),
                "successful_records": len(successful_records),
                "successful_record_ids": sorted(successful_records),
                "valid_samples": valid_samples,
                "macro_using_samples": macro_samples,
                "termination": termination,
            },
            "budget_tiers": tier_audits,
            "eval_data_used": False,
        }
        _atomic_json(ANALYSIS / "interface_gate.json", gate)
        _require(
            gate["pass"] is True,
            "adequately budgeted held-out macro interface gate failed exact transcription",
        )
        return gate
    result = {
        "schema_version": SCHEMA_VERSION,
        "pass": False,
        "status": "budget_unresolved",
        "selected_thinking_budget": None,
        "budget_tiers": tier_audits,
        "eval_data_used": False,
    }
    _atomic_json(ANALYSIS / "interface_gate.json", result)
    raise RuntimeError("held-out interface remained censoring-bound at every registered budget")


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
    meta_path: Path | None = None,
    records: Sequence[dict[str, Any]],
    sampling: Any,
    harness: Any,
    config: Mapping[str, Any],
    expected_n: int,
) -> bool:
    """Validate a resumable runner artifact against every frozen input/config field."""

    meta_path = path.with_suffix(".meta.json") if meta_path is None else meta_path
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
    expected_engine = _expected_engine_metadata(harness, config)
    _require(
        summary.get("engine") == expected_engine,
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
    expected_ids = [str(record["id"]) for record in records]
    _require(len(set(expected_ids)) == len(records), "expected runner records have duplicate ids")
    if sampling.thinking == "budget":
        think_token_ids = summary.get("think_token_ids")
        _require(isinstance(think_token_ids, Mapping), f"{meta_path} lacks think-token metadata")
        close_ids = think_token_ids.get("forced_close_sequence")
        _require(isinstance(close_ids, list) and bool(close_ids), f"{meta_path} lacks forced-close IDs")
        expected_close_tokens = int(config["context_envelope"]["forced_close_tokens"])
        _require(
            len(close_ids) == expected_close_tokens,
            f"{meta_path} forced-close token count drifted",
        )
        reserve = int(sampling.thinking_budget) + len(close_ids) + int(
            sampling.answer_max_tokens
        )
    else:
        reserve = int(sampling.max_tokens)
    prompt_counts = [
        int(item.get("prompt_tokens", -1)) if isinstance(item, Mapping) else -1
        for item in raw_preflight_rows
    ]
    _require(
        len(prompt_counts) == len(records) and all(value >= 0 for value in prompt_counts),
        f"{preflight_path} has invalid prompt-token counts",
    )
    max_model_len = int(expected_engine["max_model_len"])
    _require(preflight.get("max_model_len") == max_model_len, f"{preflight_path} model length mismatch")
    _require(
        preflight.get("generation_reserve_tokens") == reserve,
        f"{preflight_path} generation reserve mismatch",
    )
    _require(
        preflight.get("min_prompt_tokens") == min(prompt_counts),
        f"{preflight_path} minimum prompt count mismatch",
    )
    _require(
        preflight.get("max_prompt_tokens") == max(prompt_counts),
        f"{preflight_path} maximum prompt count mismatch",
    )
    max_prompt_plus_reserve = max(prompt_counts) + reserve
    _require(
        preflight.get("max_prompt_plus_reserve_tokens") == max_prompt_plus_reserve,
        f"{preflight_path} maximum prompt-plus-reserve mismatch",
    )
    _require(
        max_prompt_plus_reserve <= max_model_len,
        f"{preflight_path} exceeds the registered context envelope",
    )
    if expected_ids == ["macro-proposal-v1"]:
        _require(
            max(prompt_counts)
            == int(config["context_envelope"]["proposal_max_prompt_tokens"]),
            f"{preflight_path} macro-proposal prompt count drifted",
        )
    preflight_ids = [
        str(item.get("id")) if isinstance(item, dict) else None for item in raw_preflight_rows
    ]
    _require(
        preflight_ids == expected_ids,
        f"{preflight_path} record id order mismatch",
    )
    _require(len(rows) == len(records), f"{path} row count mismatch")
    row_ids = [str(row.get("id")) if isinstance(row, dict) else None for row in rows]
    _require(row_ids == expected_ids, f"{path} row id order mismatch")

    for expected_record, frozen_prompt, row in zip(records, raw_preflight_rows, rows, strict=True):
        record_id = str(expected_record["id"])
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
        _require(
            frozen_prompt.get("prompt_plus_reserve_tokens")
            == int(frozen_prompt.get("prompt_tokens")) + reserve,
            f"{path} prompt-plus-reserve drift for {record_id}",
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


def _passed_budget(path: Path) -> int:
    payload = _read_json(path)
    _require(
        isinstance(payload, dict) and payload.get("pass") is True,
        f"required gate did not pass: {path}",
    )
    budget = payload.get("selected_thinking_budget")
    _require(isinstance(budget, int), f"required gate lacks selected budget: {path}")
    return budget


def _scientific_external_root(config: Mapping[str, Any]) -> Path:
    configured = str(config["scientific_smoke"]["external_root"])
    override = os.environ.get(scientific_store.ARTIFACT_ROOT_ENV)
    return scientific_store.resolve_artifact_root(override or configured)


def _scientific_protocol_binding() -> dict[str, Any]:
    return scientific_store.build_protocol_binding(EXP)


def _scientific_catalog_path() -> Path:
    return ANALYSIS / "scientific_smoke_artifact_catalog.json"


def _scientific_prefix(*, probe: bool, budget: int, arm: str) -> str:
    namespace = "smoke_budget_probes" if probe else "smoke_tiers"
    return f"{namespace}/think_{budget}/{arm}"


def _scientific_expected_identity(
    harness: Any, config: Mapping[str, Any], sampling: Any
) -> dict[str, Any]:
    return {
        "model": harness.REQUIRED_MODEL_ID,
        "model_revision": harness.MODEL_REVISION,
        "runner_sha256": _sha256_file(SRC / "vllm_runner.py"),
        "sampling": json.loads(
            _canonical_bytes(dataclasses.asdict(sampling)).decode("utf-8")
        ),
        "engine": _expected_engine_metadata(harness, config),
    }


def _verify_scientific_catalog(
    config: Mapping[str, Any], *, require_selected: bool = False
) -> dict[str, Any]:
    catalog_path = _scientific_catalog_path()
    _require(catalog_path.is_file(), f"missing scientific artifact catalog: {catalog_path}")
    raw_catalog = _read_json(catalog_path)
    _require(isinstance(raw_catalog, dict), "scientific artifact catalog must be an object")
    selection_file = (
        ANALYSIS / "smoke_budget_selection.json"
        if raw_catalog.get("selected") is not None
        else None
    )
    catalog = scientific_store.verify_catalog(
        catalog_path,
        _scientific_external_root(config),
        protocol_binding=_scientific_protocol_binding(),
        selection_file=selection_file,
    )
    if require_selected:
        _require(catalog.get("selected") is not None, "scientific smoke has no selected tier")
    return catalog


def _initialize_scientific_catalog(config: Mapping[str, Any]) -> dict[str, Any]:
    catalog_path = _scientific_catalog_path()
    if catalog_path.exists():
        return _verify_scientific_catalog(config)
    local_scientific = [RUNS / "smoke_tiers", RUNS / "smoke_budget_probes", RUNS / "smoke"]
    _require(
        not any(path.exists() for path in local_scientific),
        "repository-local scientific artifacts require explicit "
        "--migrate-scientific-artifacts before smoke can resume",
    )
    root = _scientific_external_root(config)
    if root.exists():
        _require(
            not any(root.iterdir()),
            "external scientific artifacts exist without a tracked catalog; "
            "use --migrate-scientific-artifacts or quarantine the root",
        )
    catalog = scientific_store.build_catalog(
        root, protocol_binding=_scientific_protocol_binding()
    )
    scientific_store.write_catalog(catalog_path, catalog)
    return _verify_scientific_catalog(config)


def _write_scientific_catalog(
    config: Mapping[str, Any],
    *,
    selected_budget: int | None = None,
    selected_arms: Sequence[str] | None = None,
) -> dict[str, Any]:
    root = _scientific_external_root(config)
    kwargs: dict[str, Any] = {"protocol_binding": _scientific_protocol_binding()}
    if selected_budget is not None:
        arms = [str(arm) for arm in (selected_arms or ())]
        _require(bool(arms), "selected smoke catalog requires arms")
        kwargs.update(
            {
                "selection_file": ANALYSIS / "smoke_budget_selection.json",
                "selected_budget": selected_budget,
                "selected_entries": {
                    arm: f"matrix/think_{selected_budget}/{arm}" for arm in arms
                },
            }
        )
    else:
        _require(selected_arms is None, "selected arms require a selected budget")
    catalog = scientific_store.build_catalog(root, **kwargs)
    scientific_store.write_catalog(_scientific_catalog_path(), catalog)
    return _verify_scientific_catalog(
        config, require_selected=selected_budget is not None
    )


def _selected_scientific_smoke_paths(
    config: Mapping[str, Any], expected_arms: Sequence[str]
) -> tuple[int, dict[str, scientific_store.BundlePaths]]:
    legacy = RUNS / "smoke"
    _require(
        not legacy.exists(),
        f"repository-local selected smoke copy violates logical promotion: {legacy}",
    )
    catalog = _verify_scientific_catalog(config, require_selected=True)
    budget, prefixes = scientific_store.selected_bundle_prefixes(catalog, expected_arms)
    root = _scientific_external_root(config)
    return budget, {
        arm: scientific_store.bundle_paths(root, prefixes[arm])
        for arm in expected_arms
    }


def _validate_cached_scientific_arm(
    *,
    root: Path,
    relative_prefix: str,
    arm: str,
    records: Sequence[dict[str, Any]],
    sampling: Any,
    harness: Any,
    config: Mapping[str, Any],
    expected_n: int,
) -> bool:
    state = scientific_store.bundle_state(root, relative_prefix)
    if state["status"] in {"absent", "preflight_only"}:
        return False
    expected_identity = _scientific_expected_identity(harness, config, sampling)
    scientific_store.verify_receipt(
        root,
        relative_prefix,
        expected={
            "arm": arm,
            "k": expected_n,
            **expected_identity,
        },
    )
    paths = scientific_store.bundle_paths(root, relative_prefix)
    return _validate_cached_arm(
        paths.rows,
        arm=arm,
        records=records,
        sampling=sampling,
        harness=harness,
        config=config,
        expected_n=expected_n,
    )


def _commit_scientific_receipt(
    *,
    root: Path,
    relative_prefix: str,
    probe: bool,
    budget: int,
    arm: str,
    k: int,
    harness: Any,
    config: Mapping[str, Any],
    sampling: Any,
) -> dict[str, Any]:
    return scientific_store.commit_receipt(
        root,
        relative_prefix,
        role="termination_probe" if probe else "complete_matrix_arm",
        tier_mode="termination_probe_only" if probe else "complete_k12_matrix",
        thinking_budget=budget,
        arm=arm,
        k=k,
        expected_identity=_scientific_expected_identity(harness, config, sampling),
    )


def _scientific_artifact_hashes(
    paths: scientific_store.BundlePaths,
) -> dict[str, str]:
    return {
        **_artifact_hashes(paths.rows),
        "receipt": _sha256_file(paths.receipt),
    }


def _validate_migration_preflight(
    path: Path,
    *,
    records: Sequence[dict[str, Any]],
    sampling: Any,
    config: Mapping[str, Any],
) -> None:
    """Validate a genuine interrupted preflight without loading a tokenizer/model."""

    preflight = _read_json(path)
    _require(isinstance(preflight, dict), f"migration preflight must be an object: {path}")
    _require(
        preflight.get("pass") is True,
        f"migration rejects guard or non-passing preflight: {path}",
    )
    raw = preflight.get("records")
    _require(isinstance(raw, list), f"migration preflight lacks records: {path}")
    _require(len(raw) == len(records), f"migration preflight record count mismatch: {path}")
    expected_ids = [str(record["id"]) for record in records]
    actual_ids = [str(item.get("id")) if isinstance(item, Mapping) else None for item in raw]
    _require(actual_ids == expected_ids, f"migration preflight record order mismatch: {path}")
    reserve = int(sampling.thinking_budget) + int(
        config["context_envelope"]["forced_close_tokens"]
    ) + int(sampling.answer_max_tokens)
    prompt_counts: list[int] = []
    for expected, frozen in zip(records, raw, strict=True):
        _require(isinstance(frozen, Mapping), f"migration preflight row is invalid: {path}")
        _require(
            frozen.get("input_record_sha256") == _sha256_value(expected),
            f"migration preflight input identity mismatch: {path}",
        )
        rendered_hash = frozen.get("rendered_prompt_sha256")
        _require(
            isinstance(rendered_hash, str)
            and len(rendered_hash) == 64
            and all(character in "0123456789abcdef" for character in rendered_hash),
            f"migration preflight rendered-prompt hash is invalid: {path}",
        )
        prompt_tokens = frozen.get("prompt_tokens")
        _require(
            isinstance(prompt_tokens, int) and prompt_tokens > 0,
            f"migration preflight prompt count is invalid: {path}",
        )
        _require(
            frozen.get("prompt_plus_reserve_tokens") == prompt_tokens + reserve,
            f"migration preflight reserve arithmetic mismatch: {path}",
        )
        prompt_counts.append(prompt_tokens)
    max_model_len = int(config["inference"]["max_model_len"])
    _require(preflight.get("max_model_len") == max_model_len, f"migration model length mismatch: {path}")
    _require(preflight.get("generation_reserve_tokens") == reserve, f"migration reserve mismatch: {path}")
    _require(preflight.get("min_prompt_tokens") == min(prompt_counts), f"migration minimum mismatch: {path}")
    _require(preflight.get("max_prompt_tokens") == max(prompt_counts), f"migration maximum mismatch: {path}")
    _require(
        preflight.get("max_prompt_plus_reserve_tokens") == max(prompt_counts) + reserve,
        f"migration maximum reserve mismatch: {path}",
    )
    _require(max(prompt_counts) + reserve <= max_model_len, f"migration context overflow: {path}")


def _copy_local_scientific_tree(stage: Path) -> list[Path]:
    copied: list[Path] = []
    for name in ("smoke_tiers", "smoke_budget_probes"):
        source = RUNS / name
        if not source.exists():
            continue
        _require(source.is_dir() and not source.is_symlink(), f"unsafe migration source: {source}")
        for item in source.rglob("*"):
            _require(not item.is_symlink(), f"migration source contains symlink: {item}")
        destination = stage / name
        shutil.copytree(source, destination, symlinks=True)
        copied.append(source)
    _require(copied, "no repository-local scientific tier/probe tree exists to migrate")
    return copied


def _validate_legacy_smoke_copy(config: Mapping[str, Any]) -> Path | None:
    legacy = RUNS / "smoke"
    if not legacy.exists():
        return None
    _require(legacy.is_dir() and not legacy.is_symlink(), f"unsafe legacy smoke copy: {legacy}")
    selection = _read_json(ANALYSIS / "smoke_budget_selection.json")
    _require(
        isinstance(selection, dict)
        and selection.get("pass") is True
        and isinstance(selection.get("selected_thinking_budget"), int),
        "legacy selected smoke copy lacks a passing exact budget selection",
    )
    budget = int(selection["selected_thinking_budget"])
    arms = [str(arm) for arm in config["inference"]["smoke_arms"]]
    expected_names: set[str] = set()
    for arm in arms:
        for suffix in (".jsonl", ".meta.json", ".preflight.json"):
            expected_names.add(f"{arm}{suffix}")
            promoted = legacy / f"{arm}{suffix}"
            source = RUNS / "smoke_tiers" / f"think_{budget}" / f"{arm}{suffix}"
            _require(promoted.is_file() and source.is_file(), "legacy smoke promotion is incomplete")
            _require(
                promoted.stat().st_size == source.stat().st_size
                and _sha256_file(promoted) == _sha256_file(source),
                f"legacy smoke promotion differs from selected tier: {promoted}",
            )
    actual_names = {path.name for path in legacy.iterdir() if path.is_file()}
    _require(actual_names == expected_names, "legacy smoke promotion has unexpected files")
    return legacy


def _migration_bundle_spec(
    *,
    prefix: str,
    harness: Any,
    domain: Any,
    config: Mapping[str, Any],
    tasks: Sequence[Mapping[str, Any]],
    libraries: Mapping[str, Any],
    demonstrations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    parts = prefix.split("/")
    _require(len(parts) == 3, f"invalid migration bundle prefix: {prefix}")
    probe = parts[0] == "smoke_budget_probes"
    _require(parts[0] in {"smoke_tiers", "smoke_budget_probes"}, "invalid migration namespace")
    _require(parts[1].startswith("think_"), "invalid migration budget directory")
    budget = int(parts[1].removeprefix("think_"))
    _require(budget in _budget_ladder(config), f"unregistered migration budget: {budget}")
    arm = parts[2]
    allowed_arms = ["base"] if probe else [str(value) for value in config["inference"]["smoke_arms"]]
    _require(arm in allowed_arms, f"unregistered migration arm {arm} in {prefix}")
    k = int(
        config["inference"]["scientific_probe_k"]
        if probe
        else config["decision"]["smoke_matched_k"]
    )
    records = _solver_records(
        harness=harness,
        domain=domain,
        tasks=tasks,
        arm=arm,
        library=libraries[arm],
        demonstrations=demonstrations,
    )
    sampling = _solver_sampling(harness, config, budget=budget, n=k)
    return {
        "probe": probe,
        "budget": budget,
        "arm": arm,
        "k": k,
        "records": records,
        "sampling": sampling,
    }


def _validate_and_receipt_migration_stage(
    *,
    stage: Path,
    harness: Any,
    domain: Any,
    config: Mapping[str, Any],
    tasks: Sequence[Mapping[str, Any]],
    libraries: Mapping[str, Any],
    demonstrations: Sequence[Mapping[str, Any]],
) -> None:
    prefixes = scientific_store.discover_bundle_prefixes(stage)
    _require(bool(prefixes), "migration staging root has no scientific bundles")
    for prefix in prefixes:
        spec = _migration_bundle_spec(
            prefix=prefix,
            harness=harness,
            domain=domain,
            config=config,
            tasks=tasks,
            libraries=libraries,
            demonstrations=demonstrations,
        )
        paths = scientific_store.bundle_paths(stage, prefix)
        present = {
            "preflight": os.path.lexists(paths.preflight),
            "rows": os.path.lexists(paths.rows),
            "metadata": os.path.lexists(paths.metadata),
            "receipt": os.path.lexists(paths.receipt),
        }
        if present == {"preflight": True, "rows": False, "metadata": False, "receipt": False}:
            _validate_migration_preflight(
                paths.preflight,
                records=spec["records"],
                sampling=spec["sampling"],
                config=config,
            )
            # Re-freezing is byte-exact and classifies this as the one genuine
            # resumable state. A temporary guard fails validation above.
            scientific_store.write_preflight_only(
                stage, prefix, _read_json(paths.preflight)
            )
            continue
        if present == {"preflight": True, "rows": True, "metadata": True, "receipt": False}:
            _require(
                _validate_cached_arm(
                    paths.rows,
                    arm=spec["arm"],
                    records=spec["records"],
                    sampling=spec["sampling"],
                    harness=harness,
                    config=config,
                    expected_n=spec["k"],
                ),
                f"migration cache validation failed: {prefix}",
            )
            _commit_scientific_receipt(
                root=stage,
                relative_prefix=prefix,
                probe=spec["probe"],
                budget=spec["budget"],
                arm=spec["arm"],
                k=spec["k"],
                harness=harness,
                config=config,
                sampling=spec["sampling"],
            )
        elif all(present.values()):
            pass
        else:
            raise ValueError(f"migration rejects partial scientific bundle {prefix}: {present}")
        _require(
            _validate_cached_scientific_arm(
                root=stage,
                relative_prefix=prefix,
                arm=spec["arm"],
                records=spec["records"],
                sampling=spec["sampling"],
                harness=harness,
                config=config,
                expected_n=spec["k"],
            ),
            f"migration receipt validation failed: {prefix}",
        )


def _remove_local_scientific_after_verified_migration(paths: Sequence[Path]) -> None:
    for path in paths:
        _require(path.parent == RUNS, f"refusing to remove noncanonical migration path: {path}")
        _require(path.name in {"smoke_tiers", "smoke_budget_probes", "smoke"}, "unsafe removal")
        if path.exists():
            _require(path.is_dir() and not path.is_symlink(), f"unsafe removal path: {path}")
            shutil.rmtree(path)


def migrate_scientific_artifacts(
    config: Mapping[str, Any], *, remove_local: bool = False
) -> dict[str, Any]:
    """Stage, validate, receipt, and atomically install local scientific caches.

    This is model-free. It imports prompt/domain helpers but never constructs a
    VLLMRunner. By default local canonical directories are preserved and the
    operator is told how to remove them in a separately authorized invocation.
    """

    import macro_domain as domain  # type: ignore[import-not-found]
    import model_harness as harness  # type: ignore[import-not-found]

    tasks_payload, libraries_payload, _, demonstrations = _load_prepared()
    libraries = libraries_payload.get("libraries")
    _require(isinstance(libraries, dict), "libraries.json lacks libraries")
    tasks = _stage_tasks(tasks_payload, "smoke", config)
    root = _scientific_external_root(config)
    catalog_path = _scientific_catalog_path()
    protocol = _scientific_protocol_binding()
    legacy = _validate_legacy_smoke_copy(config)
    local_sources = [
        path
        for path in (RUNS / "smoke_tiers", RUNS / "smoke_budget_probes")
        if path.exists()
    ]

    if root.exists():
        if catalog_path.is_file():
            catalog = _verify_scientific_catalog(config)
        else:
            # Recover only the narrow crash window after the fully receipted
            # staging directory was atomically installed but before its tracked
            # catalog was committed. Every bundle is revalidated first.
            _validate_and_receipt_migration_stage(
                stage=root,
                harness=harness,
                domain=domain,
                config=config,
                tasks=tasks,
                libraries=libraries,
                demonstrations=demonstrations,
            )
            catalog = scientific_store.build_catalog(
                root, protocol_binding=protocol
            )
            scientific_store.write_catalog(catalog_path, catalog)
            catalog = _verify_scientific_catalog(config)
        # Re-stage local bytes when they remain, proving idempotence and exact
        # equivalence rather than trusting path existence.
        if local_sources:
            root.parent.mkdir(parents=True, exist_ok=True)
            stage = Path(tempfile.mkdtemp(prefix=f".{root.name}.verify-", dir=root.parent))
            try:
                _copy_local_scientific_tree(stage)
                _validate_and_receipt_migration_stage(
                    stage=stage,
                    harness=harness,
                    domain=domain,
                    config=config,
                    tasks=tasks,
                    libraries=libraries,
                    demonstrations=demonstrations,
                )
                staged = scientific_store.build_catalog(
                    stage, protocol_binding=protocol
                )
                existing_unselected = scientific_store.build_catalog(
                    root, protocol_binding=protocol
                )
                _require(
                    staged["tree"] == existing_unselected["tree"]
                    and staged["entries"] == existing_unselected["entries"],
                    "existing external root differs from repository-local migration source",
                )
            finally:
                shutil.rmtree(stage, ignore_errors=True)
        removal_paths = [*local_sources, *([legacy] if legacy is not None else [])]
        if remove_local:
            _remove_local_scientific_after_verified_migration(removal_paths)
        return {
            "status": "already_installed_and_verified",
            "external_root": str(root),
            "catalog_sha256": _sha256_file(catalog_path),
            "local_removed": remove_local,
            "local_paths": [str(path) for path in removal_paths],
            "selected": catalog.get("selected"),
        }

    _require(not catalog_path.exists(), "tracked scientific catalog exists without external root")
    root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{root.name}.migrate-", dir=root.parent))
    temporary_catalog = ANALYSIS / f".scientific_smoke_artifact_catalog.migrate-{os.getpid()}.json"
    installed = False
    try:
        copied = _copy_local_scientific_tree(stage)
        _validate_and_receipt_migration_stage(
            stage=stage,
            harness=harness,
            domain=domain,
            config=config,
            tasks=tasks,
            libraries=libraries,
            demonstrations=demonstrations,
        )
        catalog = scientific_store.build_catalog(stage, protocol_binding=protocol)
        scientific_store.write_catalog(temporary_catalog, catalog)
        scientific_store.verify_catalog(
            temporary_catalog, stage, protocol_binding=protocol
        )
        os.replace(stage, root)
        installed = True
        scientific_store.verify_catalog(
            temporary_catalog, root, protocol_binding=protocol
        )
        scientific_store.write_catalog(catalog_path, catalog)
        scientific_store.verify_catalog(
            catalog_path, root, protocol_binding=protocol
        )
        removal_paths = [*copied, *([legacy] if legacy is not None else [])]
        if remove_local:
            _remove_local_scientific_after_verified_migration(removal_paths)
        return {
            "status": "installed_and_verified",
            "external_root": str(root),
            "catalog_sha256": _sha256_file(catalog_path),
            "local_removed": remove_local,
            "local_paths": [str(path) for path in removal_paths],
            "next_step": (
                None
                if remove_local
                else "rerun with --migrate-scientific-artifacts "
                "--remove-local-scientific-artifacts after reviewing this receipt"
            ),
        }
    finally:
        temporary_catalog.unlink(missing_ok=True)
        if not installed:
            shutil.rmtree(stage, ignore_errors=True)


def _full_external_root(config: Mapping[str, Any]) -> Path:
    full_run = config.get("full_run")
    _require(isinstance(full_run, Mapping), "config.full_run missing")
    raw = full_run.get("external_root")
    _require(isinstance(raw, str) and raw.startswith("/"), "full external root must be absolute")
    return Path(raw)


def _full_termination_counts(metrics: Mapping[str, Any]) -> dict[str, int]:
    """Keep only content-blind integer evidence needed by the full short circuit."""

    fields = {
        "samples": "samples",
        "cap_contacts": "cap_contacts",
        "unresolved_cap_contacts": "unresolved_cap_contacts",
        "periodic_loop_contacts": "periodic_loop_contacts",
        "answer_limit_contacts": "answer_limit_contacts",
        "stage2_truncations": "stage2_truncations",
    }
    result: dict[str, int] = {}
    for output_key, metric_key in fields.items():
        value = metrics.get(metric_key)
        _require(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0,
            f"full termination metric lacks non-negative integer {metric_key}",
        )
        result[output_key] = value
    return result


def _merge_full_termination_counts(
    totals: Mapping[str, int], addition: Mapping[str, int]
) -> dict[str, int]:
    _require(set(totals) == set(addition), "full termination count fields drifted")
    return {key: int(totals[key]) + int(addition[key]) for key in totals}


def _summarize_full_termination(
    counts: Mapping[str, int], config: Mapping[str, Any]
) -> dict[str, Any]:
    samples = int(counts["samples"])
    _require(samples > 0, "cannot summarize zero full completions")
    decision = config["decision"]
    unresolved_rate = int(counts["unresolved_cap_contacts"]) / samples
    answer_rate = int(counts["answer_limit_contacts"]) / samples
    loop_rate = int(counts["periodic_loop_contacts"]) / samples
    return {
        **{key: int(value) for key, value in counts.items()},
        "cap_contact_rate": int(counts["cap_contacts"]) / samples,
        "unresolved_cap_contact_rate": unresolved_rate,
        "periodic_loop_rate": loop_rate,
        "answer_limit_contact_rate": answer_rate,
        "stage2_truncation_rate": int(counts["stage2_truncations"]) / samples,
        "adequate": (
            unresolved_rate < float(decision["budget_max_cap_contact"])
            and answer_rate < float(decision["budget_max_answer_truncation"])
            and loop_rate <= float(decision["loop_max_rate"])
        ),
        "selection_uses_output_content": False,
    }


def _full_early_fail_reason(
    counts: Mapping[str, int], *, arm: str, config: Mapping[str, Any]
) -> str | None:
    """Return the first irreversible registered full-arm termination failure."""

    full_run = config["full_run"]
    prefix = "base" if arm == "base" else "macro"
    thresholds = (
        ("unresolved_cap_contacts", int(full_run[f"{prefix}_early_unresolved"])),
        ("answer_limit_contacts", int(full_run[f"{prefix}_early_answer_limit"])),
        ("periodic_loop_contacts", int(full_run[f"{prefix}_early_periodic_loops"])),
    )
    for field, threshold in thresholds:
        if int(counts[field]) >= threshold:
            return f"irreversible_{field}_bound_reached"
    return None


def _validate_cached_full_shard(
    shard_dir: Path,
    *,
    root: Path,
    shard_plan_sha256: str,
    budget: int,
    arm: str,
    shard_index: int,
    task_ids: Sequence[str],
    k: int,
    records: Sequence[dict[str, Any]],
    sampling: Any,
    harness: Any,
    config: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not shard_dir.exists():
        return None
    # A malformed final directory is never overwritten or treated as a cache
    # miss.  The operator must move it to a quarantine path explicitly.
    receipt = full_store.validate_shard_directory(
        shard_dir,
        root=root,
        shard_plan_sha256=shard_plan_sha256,
        budget=budget,
        arm=arm,
        shard_index=shard_index,
        task_ids=task_ids,
        k=k,
    )
    _require(
        _validate_runner_artifact(
            shard_dir / "rows.jsonl",
            preflight_path=shard_dir / "preflight.json",
            meta_path=shard_dir / "runner.meta.json",
            records=records,
            sampling=sampling,
            harness=harness,
            config=config,
            expected_n=k,
        ),
        f"full shard runner artifact failed exact cache validation: {shard_dir}",
    )
    return receipt


def _generate_atomic_full_shard(
    *,
    runner: Any,
    harness: Any,
    config: Mapping[str, Any],
    root: Path,
    shard_dir: Path,
    shard_plan_sha256: str,
    budget: int,
    arm: str,
    shard_index: int,
    task_ids: Sequence[str],
    k: int,
    records: Sequence[dict[str, Any]],
    sampling: Any,
) -> dict[str, Any]:
    """Generate one all-or-nothing external shard and commit its directory."""

    _require(not shard_dir.exists(), f"refusing to overwrite final full shard: {shard_dir}")
    shard_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{shard_dir.name}.tmp-", dir=shard_dir.parent)
    )
    try:
        _atomic_json(temporary / "preflight.json", _preflight_records(runner, records, sampling))
        batch = harness.generate_vllm_batch(runner, records, sampling)
        _atomic_jsonl(temporary / "rows.jsonl", batch.rows)
        _atomic_json(temporary / "runner.meta.json", batch.summary)
        _require(
            _validate_runner_artifact(
                temporary / "rows.jsonl",
                preflight_path=temporary / "preflight.json",
                meta_path=temporary / "runner.meta.json",
                records=records,
                sampling=sampling,
                harness=harness,
                config=config,
                expected_n=k,
            ),
            f"new full shard failed runner validation: {temporary}",
        )
        receipt = full_store.make_receipt(
            temporary,
            shard_plan_sha256=shard_plan_sha256,
            budget=budget,
            arm=arm,
            shard_index=shard_index,
            task_ids=task_ids,
            k=k,
        )
        # Receipt is deliberately last.  Any interruption before this point
        # leaves a non-reusable temporary directory, never a partial final.
        _atomic_json(temporary / full_store.RECEIPT_FILE, receipt)
        full_store.validate_shard_directory(
            temporary,
            root=root,
            shard_plan_sha256=shard_plan_sha256,
            budget=budget,
            arm=arm,
            shard_index=shard_index,
            task_ids=task_ids,
            k=k,
            allow_temporary_name=True,
        )
        full_store.fsync_directory(temporary)
        _require(not shard_dir.exists(), f"final shard appeared during generation: {shard_dir}")
        os.rename(temporary, shard_dir)
        full_store.fsync_directory(shard_dir.parent)
    except BaseException:
        # Never salvage or promote an interrupted temporary shard.  It remains
        # visibly named ``.tmp-*`` for explicit inspection/removal, and a
        # resume starts the whole first incomplete shard again.
        raise
    return full_store.validate_shard_directory(
        shard_dir,
        root=root,
        shard_plan_sha256=shard_plan_sha256,
        budget=budget,
        arm=arm,
        shard_index=shard_index,
        task_ids=task_ids,
        k=k,
    )


def _write_full_artifact_catalog(
    *,
    root: Path,
    plan: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify every completed receipt and atomically write the tracked catalog."""

    selected = selection.get("selected_thinking_budget")
    _require(isinstance(selected, int), "cannot catalog a full run without a selected budget")
    plan_hash = full_store.plan_sha256(plan)
    completed: list[dict[str, Any]] = []
    selected_entries: list[dict[str, Any]] = []
    identity_by_arm_tier: dict[tuple[int, str], Mapping[str, Any]] = {}
    tiers = selection.get("tiers")
    _require(isinstance(tiers, list), "full budget selection lacks tiers")
    for tier in tiers:
        _require(isinstance(tier, Mapping), "full budget tier must be an object")
        budget = int(tier["budget"])
        arms = tier.get("arms")
        _require(isinstance(arms, Mapping), "full budget tier lacks arms")
        for arm, arm_record in arms.items():
            if not isinstance(arm_record, Mapping):
                continue
            shard_records = arm_record.get("shards", [])
            _require(isinstance(shard_records, list), f"full arm {arm} shards must be a list")
            for shard_record in shard_records:
                _require(isinstance(shard_record, Mapping), "full shard audit must be an object")
                if shard_record.get("status") != "complete":
                    continue
                shard_index = int(shard_record["shard_index"])
                spec = full_store.shard_spec(plan, str(arm), shard_index)
                shard_dir = full_store.shard_directory(
                    root, budget=budget, arm=str(arm), shard_index=shard_index
                )
                key = (budget, str(arm))
                receipt = full_store.validate_shard_directory(
                    shard_dir,
                    root=root,
                    shard_plan_sha256=plan_hash,
                    budget=budget,
                    arm=str(arm),
                    shard_index=shard_index,
                    task_ids=spec["task_ids"],
                    k=int(spec["k"]),
                    expected_identity=identity_by_arm_tier.get(key),
                )
                identity_by_arm_tier.setdefault(key, receipt["identity"])
                entry = full_store.catalog_shard_entry(root, shard_dir, receipt)
                completed.append(entry)
                if budget == selected:
                    selected_entries.append(entry)

    arm_order = [str(arm) for arm in plan["arm_order"]]
    expected_selected = sum(int(plan["arms"][arm]["shard_count"]) for arm in arm_order)
    _require(
        len(selected_entries) == expected_selected,
        "selected full tier does not contain every planned shard",
    )
    selected_keys = {(entry["arm"], entry["shard_index"]) for entry in selected_entries}
    expected_keys = {
        (arm, index)
        for arm in arm_order
        for index in range(int(plan["arms"][arm]["shard_count"]))
    }
    _require(selected_keys == expected_keys, "selected full tier shard set drifted")
    selection_path = ANALYSIS / "full_budget_selection.json"
    plan_path = ANALYSIS / "full_shard_plan.json"
    catalog = {
        "schema_version": full_store.FULL_ARTIFACT_SCHEMA_VERSION,
        "experiment_id": EXP.name,
        "canonical_external_root": str(root.resolve(strict=True)),
        "shard_plan": {
            "path": "analysis/full_shard_plan.json",
            "sha256": _sha256_file(plan_path),
            "content_sha256": plan_hash,
        },
        "budget_selection": {
            "path": "analysis/full_budget_selection.json",
            "sha256": _sha256_file(selection_path),
        },
        "selected_tier": {
            "thinking_budget": selected,
            "relative_path": f"think_{selected}",
            "logical_promotion_only": True,
            "repository_raw_copy": None,
        },
        "arm_order": arm_order,
        "completed_shards": sorted(
            completed, key=lambda row: (row["budget"], arm_order.index(row["arm"]), row["shard_index"])
        ),
        "selected_shards": sorted(
            selected_entries, key=lambda row: (arm_order.index(row["arm"]), row["shard_index"])
        ),
    }
    _atomic_json(ANALYSIS / "full_artifact_catalog.json", catalog)
    return catalog


def _run_full_scientific_stage(
    *,
    runner: Any,
    harness: Any,
    domain: Any,
    config: Mapping[str, Any],
    tasks: Sequence[Mapping[str, Any]],
    libraries: Mapping[str, Any],
    demonstrations: Sequence[Mapping[str, Any]],
    starting_budget: int,
) -> int:
    """Run/resume the full matrix under Amendment 8's external shard contract."""

    ladder = _budget_ladder(config)
    _require(starting_budget in ladder, "full starting budget is not registered")
    arms = _arm_order(config, libraries)
    plan = full_store.build_shard_plan(
        tasks,
        arms,
        base_k=int(config["inference"]["base_max_k"]),
        macro_k=int(config["inference"]["macro_k"]),
    )
    plan_hash = full_store.plan_sha256(plan)
    _freeze_json(ANALYSIS / "full_shard_plan.json", plan)
    root = _full_external_root(config)
    root.mkdir(parents=True, exist_ok=True)
    tasks_by_id = {str(task["id"]): task for task in tasks}
    _require(len(tasks_by_id) == len(tasks), "full tasks contain duplicate ids")
    zero_counts = {
        "samples": 0,
        "cap_contacts": 0,
        "unresolved_cap_contacts": 0,
        "periodic_loop_contacts": 0,
        "answer_limit_contacts": 0,
        "stage2_truncations": 0,
    }
    tiers: list[dict[str, Any]] = []
    selected: int | None = None

    for budget in ladder[ladder.index(starting_budget) :]:
        per_arm: dict[str, Any] = {}
        rejecting_arm: str | None = None
        for arm_index, arm in enumerate(arms):
            arm_plan = plan["arms"][arm]
            k = int(arm_plan["k"])
            sampling = _solver_sampling(harness, config, budget=budget, n=k)
            totals = dict(zero_counts)
            shard_audits: list[dict[str, Any]] = []
            early_reason: str | None = None
            for raw_spec in arm_plan["shards"]:
                spec = dict(raw_spec)
                shard_index = int(spec["shard_index"])
                task_ids = [str(task_id) for task_id in spec["task_ids"]]
                shard_tasks = [tasks_by_id[task_id] for task_id in task_ids]
                records = _solver_records(
                    harness=harness,
                    domain=domain,
                    tasks=shard_tasks,
                    arm=arm,
                    library=libraries[arm],
                    demonstrations=demonstrations,
                )
                _require(
                    [record["id"] for record in records] == spec["record_ids"],
                    f"full {arm} shard {shard_index} record order drifted",
                )
                shard_dir = full_store.shard_directory(
                    root, budget=budget, arm=arm, shard_index=shard_index
                )
                receipt = _validate_cached_full_shard(
                    shard_dir,
                    root=root,
                    shard_plan_sha256=plan_hash,
                    budget=budget,
                    arm=arm,
                    shard_index=shard_index,
                    task_ids=task_ids,
                    k=k,
                    records=records,
                    sampling=sampling,
                    harness=harness,
                    config=config,
                )
                if receipt is None:
                    print(
                        f"[run] full/think_{budget}/{arm}/shard_{shard_index:03d}: "
                        f"{len(records)} prompts x K={k} (144 completions) through vLLM",
                        flush=True,
                    )
                    receipt = _generate_atomic_full_shard(
                        runner=runner,
                        harness=harness,
                        config=config,
                        root=root,
                        shard_dir=shard_dir,
                        shard_plan_sha256=plan_hash,
                        budget=budget,
                        arm=arm,
                        shard_index=shard_index,
                        task_ids=task_ids,
                        k=k,
                        records=records,
                        sampling=sampling,
                    )
                else:
                    print(
                        f"[run] full/think_{budget}/{arm}/shard_{shard_index:03d}: "
                        "validated receipt exists, skip",
                        flush=True,
                    )
                rows = _read_jsonl(shard_dir / "rows.jsonl")
                termination = _termination_metrics(
                    rows,
                    budget=budget,
                    answer_cap=int(config["inference"]["answer_max_tokens"]),
                    config=config,
                    require_headroom=False,
                )
                shard_counts = _full_termination_counts(termination)
                _require(shard_counts["samples"] == 144, "full shard completion count drifted")
                totals = _merge_full_termination_counts(totals, shard_counts)
                shard_audits.append(
                    {
                        "shard_index": shard_index,
                        "status": "complete",
                        "task_ids": task_ids,
                        "termination": _summarize_full_termination(shard_counts, config),
                        "artifact": full_store.catalog_shard_entry(root, shard_dir, receipt),
                    }
                )
                early_reason = _full_early_fail_reason(totals, arm=arm, config=config)
                if early_reason is not None:
                    for skipped in arm_plan["shards"][shard_index + 1 :]:
                        shard_audits.append(
                            {
                                "shard_index": int(skipped["shard_index"]),
                                "status": "skipped",
                                "skip_reason": early_reason,
                            }
                        )
                    break

            aggregate = _summarize_full_termination(totals, config)
            if early_reason is not None:
                per_arm[arm] = {
                    "status": "irreversibly_rejected",
                    "complete": False,
                    "adequate": False,
                    "early_fail_reason": early_reason,
                    "termination": aggregate,
                    "shards": shard_audits,
                    "selection_uses_output_content": False,
                }
                rejecting_arm = arm
                for skipped_arm in arms[arm_index + 1 :]:
                    per_arm[skipped_arm] = {
                        "status": "skipped",
                        "complete": False,
                        "adequate": False,
                        "skip_reason": "rung_irreversibly_rejected_by_full_arm",
                        "rejecting_arm": arm,
                        "shards": [],
                        "selection_uses_output_content": False,
                    }
                break
            _require(
                len([row for row in shard_audits if row["status"] == "complete"])
                == int(arm_plan["shard_count"]),
                f"full {arm} arm did not complete every shard",
            )
            _require(
                aggregate["adequate"] is True,
                f"full {arm} completed inadequately without crossing its irreversible bound",
            )
            per_arm[arm] = {
                "status": "complete",
                "complete": True,
                "adequate": True,
                "termination": aggregate,
                "shards": shard_audits,
                "selection_uses_output_content": False,
            }

        tier_complete = all(per_arm.get(arm, {}).get("complete") is True for arm in arms)
        tier_adequate = tier_complete and all(
            per_arm[arm].get("adequate") is True for arm in arms
        )
        tiers.append(
            {
                "budget": budget,
                "status": "selectable" if tier_adequate else "irreversibly_rejected",
                "complete": tier_complete,
                "adequate": tier_adequate,
                "rejecting_arm": rejecting_arm,
                "arms": per_arm,
            }
        )
        if tier_adequate:
            selected = budget
        selection = {
            "schema_version": full_store.FULL_ARTIFACT_SCHEMA_VERSION,
            "run": "full",
            "pass": selected is not None,
            "selected_thinking_budget": selected,
            "shard_plan_sha256": plan_hash,
            "canonical_external_root": str(root.resolve(strict=True)),
            "selection_rule": (
                "smallest complete shared-budget matrix with every arm termination-adequate; "
                "only preregistered irreversible full-arm integer bounds may short-circuit a rung"
            ),
            "selection_uses_output_content": False,
            "lower_tiers_excluded_from_scoring": True,
            "logical_promotion_only": True,
            "tiers": tiers,
        }
        _atomic_json(ANALYSIS / "full_budget_selection.json", selection)
        if selected is not None:
            _write_full_artifact_catalog(root=root, plan=plan, selection=selection)
            return selected

    raise ValueError(
        "full setup-inconclusive: every registered thinking-budget rung crossed a "
        "termination-inadequacy bound"
    )


def _run_smoke_budget_probe(
    *,
    runner: Any,
    harness: Any,
    domain: Any,
    config: Mapping[str, Any],
    tasks: Sequence[Mapping[str, Any]],
    library: Mapping[str, Any],
    demonstrations: Sequence[Mapping[str, Any]],
    budget: int,
) -> dict[str, Any]:
    """Run or exactly validate a termination-only K=4 base-arm probe.

    Probe artifacts live outside the scientific tier tree so they cannot be
    promoted or accidentally pooled with the complete K=12 smoke matrix.  The
    decision returned here is a function only of runner token/finish metadata.
    """

    arm = "base"
    k = int(config["inference"]["scientific_probe_k"])
    _require(k == 4, "scientific budget probe K drifted")
    records = _solver_records(
        harness=harness,
        domain=domain,
        tasks=tasks,
        arm=arm,
        library=library,
        demonstrations=demonstrations,
    )
    expected_records = 2 * int(config["data"]["smoke_tasks_per_split"])
    _require(
        len(records) == expected_records,
        f"scientific budget probe must cover all {expected_records} frozen smoke records",
    )
    sampling = _solver_sampling(harness, config, budget=budget, n=k)
    root = _scientific_external_root(config)
    relative_prefix = _scientific_prefix(probe=True, budget=budget, arm=arm)
    paths = scientific_store.bundle_paths(root, relative_prefix)
    present = _validate_cached_scientific_arm(
        root=root,
        relative_prefix=relative_prefix,
        arm=arm,
        records=records,
        sampling=sampling,
        harness=harness,
        config=config,
        expected_n=k,
    )
    if not present:
        print(
            f"[run] smoke-budget-probe/think_{budget}/base: "
            f"{len(records)} prompts x K={k} through vLLM",
            flush=True,
        )
        scientific_store.write_preflight_only(
            root,
            relative_prefix,
            _preflight_records(runner, records, sampling),
        )
        _write_scientific_catalog(config)
        batch = harness.generate_vllm_batch(runner, records, sampling)
        _write_batch(paths.rows, batch)
        _require(
            _validate_cached_arm(
                paths.rows,
                arm=arm,
                records=records,
                sampling=sampling,
                harness=harness,
                config=config,
                expected_n=k,
            ),
            f"smoke budget probe at think_{budget} did not freeze a complete runner artifact",
        )
        _commit_scientific_receipt(
            root=root,
            relative_prefix=relative_prefix,
            probe=True,
            budget=budget,
            arm=arm,
            k=k,
            harness=harness,
            config=config,
            sampling=sampling,
        )
        _require(
            _validate_cached_scientific_arm(
                root=root,
                relative_prefix=relative_prefix,
                arm=arm,
                records=records,
                sampling=sampling,
                harness=harness,
                config=config,
                expected_n=k,
            ),
            f"smoke budget probe at think_{budget} lacks a valid receipt",
        )
        _write_scientific_catalog(config)
    else:
        print(
            f"[run] smoke-budget-probe/think_{budget}/base: frozen output exists, skip",
            flush=True,
        )

    # Receipt validation above precedes any row read or termination inspection.
    rows = _read_jsonl(paths.rows)
    termination = _termination_metrics(
        rows,
        budget=budget,
        answer_cap=int(config["inference"]["answer_max_tokens"]),
        config=config,
        require_headroom=False,
    )
    return {
        "status": "complete",
        "role": "termination_only_budget_probe",
        "budget": budget,
        "arm": arm,
        "k": k,
        "records": len(records),
        "termination": termination,
        "artifacts": _scientific_artifact_hashes(paths),
        "eligible_for_promotion": False,
        "eligible_for_scoring": False,
        "eligible_for_prefix_pooling": False,
        "selection_uses_output_content": False,
    }


def _run_scientific_stage(
    *,
    run: str,
    runner: Any,
    harness: Any,
    domain: Any,
    config: Mapping[str, Any],
    tasks: Sequence[Mapping[str, Any]],
    libraries: Mapping[str, Any],
    demonstrations: Sequence[Mapping[str, Any]],
    starting_budget: int,
) -> int:
    """Run arm matrices until the first complete, termination-adequate rung.

    Arm order is a termination-only sequential gate within each rung.  Once a
    fully written and validated arm is termination-inadequate, that frozen arm
    irreversibly rejects the rung and the remaining arms are skipped.  Smoke
    always evaluates complete K=12 matrices through 32k.  After a matrix at
    32k or above rejects, later rungs are screened by a separate K=4 base-arm
    termination probe before any new complete matrix is generated.  Probe rows
    are never promoted, scored, or included in cross-tier prefix diagnostics.
    """

    _require(run == "smoke", "unsharded scientific stage is smoke-only")
    ladder = _budget_ladder(config)
    _require(starting_budget in ladder, f"{run} starting budget is not registered")
    stage_arms = (
        [str(arm) for arm in config["inference"]["smoke_arms"]]
        if run == "smoke"
        else _arm_order(config, libraries)
    )
    answer_cap = int(config["inference"]["answer_max_tokens"])
    tiers: list[dict[str, Any]] = []
    previous_by_arm: dict[str, list[dict[str, Any]]] = {}
    selected: int | None = None
    probe_higher_smoke_rungs = False
    scientific_root = _scientific_external_root(config)

    eligible_ladder = ladder[ladder.index(starting_budget) :]
    for budget in eligible_ladder:
        scientific_probe: dict[str, Any] | None = None
        if run == "smoke" and probe_higher_smoke_rungs:
            scientific_probe = _run_smoke_budget_probe(
                runner=runner,
                harness=harness,
                domain=domain,
                config=config,
                tasks=tasks,
                library=libraries["base"],
                demonstrations=demonstrations,
                budget=budget,
            )
            if scientific_probe["termination"]["adequate"] is not True:
                per_arm = {
                    arm: {
                        "status": "skipped",
                        "skip_reason": "termination_inadequate_scientific_budget_probe",
                        "rejecting_probe_arm": "base",
                        "selection_uses_output_content": False,
                    }
                    for arm in stage_arms
                }
                tiers.append(
                    {
                        "budget": budget,
                        "status": "probe_only_rejected",
                        "tier_mode": "termination_probe_only",
                        "complete": False,
                        "adequate": False,
                        "rejecting_arm": None,
                        "rejecting_probe_arm": "base",
                        "scientific_probe": scientific_probe,
                        "arms": per_arm,
                    }
                )
                continue

        per_arm: dict[str, Any] = {}
        current_by_arm: dict[str, list[dict[str, Any]]] = {}
        rejecting_arm: str | None = None
        for arm_index, arm in enumerate(stage_arms):
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
            sampling = _solver_sampling(harness, config, budget=budget, n=k)
            relative_prefix = _scientific_prefix(
                probe=False, budget=budget, arm=arm
            )
            paths = scientific_store.bundle_paths(scientific_root, relative_prefix)
            present = _validate_cached_scientific_arm(
                root=scientific_root,
                relative_prefix=relative_prefix,
                arm=arm,
                records=records,
                sampling=sampling,
                harness=harness,
                config=config,
                expected_n=k,
            )
            if not present:
                print(
                    f"[run] {run}/think_{budget}/{arm}: {len(records)} prompts x K={k} through vLLM",
                    flush=True,
                )
                scientific_store.write_preflight_only(
                    scientific_root,
                    relative_prefix,
                    _preflight_records(runner, records, sampling),
                )
                _write_scientific_catalog(config)
                batch = harness.generate_vllm_batch(runner, records, sampling)
                _write_batch(paths.rows, batch)
                _require(
                    _validate_cached_arm(
                        paths.rows,
                        arm=arm,
                        records=records,
                        sampling=sampling,
                        harness=harness,
                        config=config,
                        expected_n=k,
                    ),
                    f"{run}/think_{budget}/{arm} did not freeze a complete runner artifact",
                )
                _commit_scientific_receipt(
                    root=scientific_root,
                    relative_prefix=relative_prefix,
                    probe=False,
                    budget=budget,
                    arm=arm,
                    k=k,
                    harness=harness,
                    config=config,
                    sampling=sampling,
                )
                _require(
                    _validate_cached_scientific_arm(
                        root=scientific_root,
                        relative_prefix=relative_prefix,
                        arm=arm,
                        records=records,
                        sampling=sampling,
                        harness=harness,
                        config=config,
                        expected_n=k,
                    ),
                    f"{run}/think_{budget}/{arm} lacks a valid receipt",
                )
                _write_scientific_catalog(config)
            else:
                print(f"[run] {run}/think_{budget}/{arm}: frozen output exists, skip", flush=True)
            # Exact receipt/cache validation always precedes row interpretation.
            rows = _read_jsonl(paths.rows)
            current_by_arm[arm] = rows
            prefix = (
                _stage1_prefix_audit(previous_by_arm[arm], rows)
                if arm in previous_by_arm
                else None
            )
            termination = _termination_metrics(
                rows,
                budget=budget,
                answer_cap=answer_cap,
                config=config,
                require_headroom=False,
            )
            per_arm[arm] = {
                "status": "complete",
                "termination": termination,
                "prefix_audit_vs_previous_tier": prefix,
                "artifacts": _scientific_artifact_hashes(paths),
            }
            if not termination["adequate"]:
                rejecting_arm = arm
                for skipped_arm in stage_arms[arm_index + 1 :]:
                    per_arm[skipped_arm] = {
                        "status": "skipped",
                        "skip_reason": "rung_irreversibly_rejected_by_termination_inadequate_arm",
                        "rejecting_arm": arm,
                        "selection_uses_output_content": False,
                    }
                break

        tier_complete = all(
            arm in per_arm and per_arm[arm].get("status") == "complete"
            for arm in stage_arms
        )
        tier_adequate = tier_complete and all(
            per_arm[arm]["termination"]["adequate"] for arm in stage_arms
        )
        tier_record = {
            "budget": budget,
            "status": "selectable" if tier_adequate else "rejected",
            "complete": tier_complete,
            "adequate": tier_adequate,
            "rejecting_arm": rejecting_arm,
            "arms": per_arm,
        }
        if run == "smoke":
            tier_record.update(
                {
                    "tier_mode": "complete_k12_matrix",
                    "scientific_probe": scientific_probe,
                }
            )
        tiers.append(tier_record)
        previous_by_arm = current_by_arm
        if tier_adequate:
            selected = budget
            break
        if run == "smoke" and budget >= 32768 and rejecting_arm is not None:
            probe_higher_smoke_rungs = True

    selection = {
        "schema_version": SCHEMA_VERSION,
        "run": run,
        "pass": selected is not None,
        "selected_thinking_budget": selected,
        "selection_rule": (
            "smallest rung where every arm is complete and termination-adequate; "
            "the first inadequate complete arm irreversibly rejects a rung and skips its remaining arms"
        ),
        "selection_uses_output_content": False,
        "lower_tiers_excluded_from_scoring": True,
        "tiers": tiers,
    }
    if run == "smoke":
        selection.update(
            {
                "scientific_probe_k": int(config["inference"]["scientific_probe_k"]),
                "probe_policy": (
                    "16k and 32k use complete K=12 matrices; after a complete matrix at "
                    "32k or above rejects, each higher rung first uses a separate K=4 "
                    "base-arm termination probe"
                ),
                "probes_excluded_from_promotion_scoring_and_prefix_pooling": True,
            }
        )
    _atomic_json(ANALYSIS / f"{run}_budget_selection.json", selection)
    if selected is None:
        _write_scientific_catalog(config)
    else:
        _write_scientific_catalog(
            config, selected_budget=selected, selected_arms=stage_arms
        )
    _require(
        selected is not None,
        f"{run} setup-inconclusive: every registered thinking-budget rung was termination-inadequate",
    )
    return selected


def run_model_stage(run: str, config: Mapping[str, Any]) -> None:
    _require(run in {"smoke", "full"}, "run must be smoke or full")
    _verify_frozen_data(config)

    # Lazy imports keep data verification model-free and make the vLLM boundary explicit.
    import macro_domain as domain  # type: ignore[import-not-found]
    import model_harness as harness  # type: ignore[import-not-found]

    tasks_payload, libraries_payload, proposal_view, demonstrations = _load_prepared()
    libraries = libraries_payload.get("libraries")
    _require(isinstance(libraries, dict), "libraries.json lacks libraries")

    if run == "smoke":
        # This is deliberately before model allocation.  A missing, corrupt, or
        # un-migrated scientific cache must fail without consuming GPU memory.
        _initialize_scientific_catalog(config)
    proposal_record = _macro_proposal_record(
        harness=harness,
        domain=domain,
        config=config,
        proposal_view=proposal_view,
    )
    proposal_prompt_tokens = _rendered_proposal_prompt_tokens_cpu(
        harness=harness,
        proposal_record=proposal_record,
    )
    _context_envelope_regression(
        config=config,
        proposal_record=proposal_record,
        observed_prompt_tokens=proposal_prompt_tokens,
    )

    runner: Any | None = harness.VLLMRunner(_engine_config(harness, config))
    try:
        if run == "smoke":
            calibration = _run_budget_calibration(
                runner=runner,
                harness=harness,
                domain=domain,
                config=config,
                library=libraries["designed_ceiling"],
            )
            interface = _run_interface_gate(
                runner=runner,
                harness=harness,
                domain=domain,
                config=config,
                library=libraries["designed_ceiling"],
                starting_budget=int(calibration["selected_thinking_budget"]),
            )
            starting_budget = int(interface["selected_thinking_budget"])
        else:
            _passed_budget(ANALYSIS / "interface_gate.json")
            smoke_tasks = _stage_tasks(tasks_payload, "smoke", config)
            smoke_arms = [str(arm) for arm in config["inference"]["smoke_arms"]]
            smoke_budget = _passed_budget(ANALYSIS / "smoke_budget_selection.json")
            catalog_budget, selected_paths = _selected_scientific_smoke_paths(
                config, smoke_arms
            )
            _require(
                catalog_budget == smoke_budget,
                "smoke selection and scientific catalog budget disagree",
            )
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
                        selected_paths[arm].rows,
                        arm=arm,
                        records=records,
                        sampling=_solver_sampling(
                            harness, config, budget=smoke_budget, n=k
                        ),
                        harness=harness,
                        config=config,
                        expected_n=k,
                    ),
                    f"full requires complete frozen smoke arm {arm}",
                )
            _require_smoke_gate()

            # Qwen proposals are secondary and train-only.  Defer their vLLM
            # cost until the primary base/designed smoke has actually passed.
            libraries_payload = _ensure_qwen_libraries(
                runner=runner,
                harness=harness,
                domain=domain,
                config=config,
                libraries_payload=libraries_payload,
                proposal_view=proposal_view,
                starting_budget=smoke_budget,
            )
            libraries = libraries_payload["libraries"]
            starting_budget = smoke_budget

        # Qwen arms are permitted in full only if construction froze all eight.
        if run == "full" and "qwen_ranked" in libraries:
            _require(
                len(libraries["qwen_ranked"]["macros"])
                == int(config["decision"]["qwen_required_verified_macros"]),
                "full requires an exactly-eight-entry qwen_ranked library",
            )

        stage_tasks = _stage_tasks(tasks_payload, run, config)
        if run == "full":
            _run_full_scientific_stage(
                runner=runner,
                harness=harness,
                domain=domain,
                config=config,
                tasks=stage_tasks,
                libraries=libraries,
                demonstrations=demonstrations,
                starting_budget=starting_budget,
            )
        else:
            _run_scientific_stage(
                run=run,
                runner=runner,
                harness=harness,
                domain=domain,
                config=config,
                tasks=stage_tasks,
                libraries=libraries,
                demonstrations=demonstrations,
                starting_budget=starting_budget,
            )
    finally:
        if runner is not None:
            runner.close()

    verdict = _invoke_analyzer(run)
    if run == "smoke":
        gate = verdict["smoke_gate"]
        if gate.get("pass") is not True:
            raise RuntimeError(f"smoke gate failed closed: {gate.get('reasons', [])}")


def prepare(config: Mapping[str, Any]) -> None:
    """Historical parent generator retained for audit; follow-up data must not be regenerated."""

    raise RuntimeError(
        "this follow-up reuses byte-frozen parent data; use --prepare to verify provenance, "
        "not to regenerate it"
    )


def _historical_parent_prepare(config: Mapping[str, Any]) -> None:
    """Parent generator retained verbatim below as an auditable recovery recipe."""

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
    stages.add_argument(
        "--prepare",
        action="store_true",
        help="verify byte-frozen parent artifacts and CPU gate without regenerating them",
    )
    stages.add_argument("--smoke", action="store_true", help="run the gated vLLM smoke workload")
    stages.add_argument("--full", action="store_true", help="run the frozen full vLLM workload")
    stages.add_argument(
        "--migrate-scientific-artifacts",
        action="store_true",
        help="model-free staged migration of local scientific smoke caches to external storage",
    )
    parser.add_argument(
        "--remove-local-scientific-artifacts",
        action="store_true",
        help="after verified migration, remove byte-identical local tier/probe/promotion dirs",
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args(argv)
    if args.config.resolve() != CONFIG_PATH.resolve():
        parser.error(
            "alternate configs are forbidden for this preregistered runner because the committed "
            "analyzer is frozen to configs/default.yaml"
        )
    config = load_config(args.config)
    _validate_config(config)
    if args.remove_local_scientific_artifacts and not args.migrate_scientific_artifacts:
        parser.error("--remove-local-scientific-artifacts requires --migrate-scientific-artifacts")
    if args.prepare:
        _verify_frozen_data(config)
        print(f"[run] copied parent data verified at commit {PARENT_COMMIT}", flush=True)
    elif args.smoke:
        run_model_stage("smoke", config)
    elif args.full:
        run_model_stage("full", config)
    else:
        result = migrate_scientific_artifacts(
            config, remove_local=args.remove_local_scientific_artifacts
        )
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
