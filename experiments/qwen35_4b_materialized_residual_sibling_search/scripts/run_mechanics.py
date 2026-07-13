#!/usr/bin/env python3
"""Prepare, authenticate, run, and score the frozen mechanics-only gate."""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import re
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
EXP_REL = EXP.relative_to(ROOT)
_BOOTSTRAP_DESIGN_COMMIT = "ba08a7320e5fe2d5fb88f2e5b6f3a359ed727cd7"
_BOOTSTRAP_IMPORT_FILES = (
    str(EXP_REL / "scripts/run_mechanics.py"),
    str(EXP_REL / "src/mechanics.py"),
    str(EXP_REL / "src/protocol.py"),
    str(EXP_REL / "src/task_data.py"),
    str(EXP_REL / "src/vllm_runner.py"),
)
_BOOTSTRAP_DESIGN_IMPORT_FILES = _BOOTSTRAP_IMPORT_FILES[2:]


def _bootstrap_cli_options() -> dict[str, str]:
    found: dict[str, str] = {}
    index = 1
    while index < len(sys.argv):
        argument = sys.argv[index]
        matched = None
        value = None
        for name in ("--stage", "--lock"):
            if argument == name:
                matched = name
                if index + 1 >= len(sys.argv) or sys.argv[index + 1].startswith("--"):
                    raise RuntimeError(f"bootstrap option lacks a value: {name}")
                value = sys.argv[index + 1]
                index += 1
                break
            if argument.startswith(name + "="):
                matched = name
                value = argument.split("=", 1)[1]
                if not value:
                    raise RuntimeError(f"bootstrap option lacks a value: {name}")
                break
        if matched is not None:
            if matched in found:
                raise RuntimeError(f"duplicate bootstrap option is forbidden: {matched}")
            found[matched] = str(value)
        index += 1
    return found


def _bootstrap_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bootstrap_git_blob(commit: str, relative: str) -> bytes:
    return subprocess.check_output(
        ["git", "show", f"{commit}:{relative}"], cwd=ROOT
    )


def _install_sealed_path_audit_hook() -> None:
    benchmark_root = (ROOT / "benchmarks").resolve()
    procedural_root = (EXP / "data" / "procedural").resolve()
    allowed_procedural = {
        (procedural_root / "manifest.json").resolve(),
        (procedural_root / "mechanics_public.jsonl").resolve(),
        (procedural_root / "mechanics_audit.jsonl").resolve(),
    }

    def audit(event: str, arguments: tuple[Any, ...]) -> None:
        if event != "open" or not arguments:
            return
        raw_path = arguments[0]
        if not isinstance(raw_path, (str, bytes, os.PathLike)):
            return
        try:
            path = Path(raw_path).resolve()
        except (OSError, TypeError, ValueError):
            return
        if path.is_relative_to(benchmark_root):
            raise PermissionError(f"mechanics process forbids benchmark reads: {path}")
        if path.is_relative_to(procedural_root) and path not in allowed_procedural:
            raise PermissionError(f"mechanics process forbids sealed data reads: {path}")

    sys.addaudithook(audit)


def _bootstrap_normalize_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _bootstrap_validate_environment() -> None:
    executable = Path(sys.executable)
    if (
        executable.parent != ROOT / ".venv-vllm" / "bin"
        or not executable.name.startswith("python")
    ):
        raise RuntimeError("mechanics stages require the pinned .venv-vllm interpreter")
    expected: dict[str, str] = {}
    direct: set[str] = set()
    for raw_line in (ROOT / "requirements-vllm.lock.txt").read_text().splitlines():
        if not raw_line or raw_line[:1].isspace() or raw_line.startswith("#"):
            continue
        line = raw_line.split(";", 1)[0].strip()
        if " @ " in line:
            direct.add(
                _bootstrap_normalize_distribution_name(line.split(" @ ", 1)[0])
            )
            continue
        if "==" not in line:
            raise RuntimeError("bootstrap found a non-exact vLLM requirement")
        name, version = line.split("==", 1)
        expected[_bootstrap_normalize_distribution_name(name)] = version.strip()
    if direct != {"vllm"}:
        raise RuntimeError("bootstrap direct-requirement set changed")
    expected["vllm"] = "0.24.0+cu129"
    mismatches: dict[str, tuple[str, str | None]] = {}
    for name, version in expected.items():
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        if installed != version:
            mismatches[name] = (version, installed)
    if mismatches:
        raise RuntimeError(f"bootstrap environment differs from vLLM lock: {mismatches}")


def _bootstrap_verify_before_local_imports() -> None:
    options = _bootstrap_cli_options()
    stage = options.get("--stage")
    if stage not in {"prepare", "lock", "run", "analyze"}:
        return
    _bootstrap_validate_environment()
    _install_sealed_path_audit_hook()
    if stage not in {"run", "analyze"}:
        return
    lock_value = options.get("--lock")
    lock_path = (
        Path(lock_value).resolve()
        if lock_value is not None
        else EXP / "runs" / "mechanics" / "implementation_lock.json"
    )
    if lock_path.is_symlink() or not lock_path.is_file():
        raise RuntimeError("local imports require a real implementation lock")
    lock = json.loads(lock_path.read_text())
    if not isinstance(lock, dict):
        raise RuntimeError("pre-import implementation lock schema changed")
    critical = lock.get("critical_files")
    implementation = lock.get("implementation_commit")
    if (
        lock.get("design_commit") != _BOOTSTRAP_DESIGN_COMMIT
        or not isinstance(critical, dict)
        or not isinstance(implementation, str)
        or len(implementation) != 40
    ):
        raise RuntimeError("pre-import implementation lock schema changed")
    for relative in _BOOTSTRAP_IMPORT_FILES:
        path = ROOT / relative
        expected = critical.get(relative)
        if (
            not isinstance(expected, str)
            or path.is_symlink()
            or not path.is_file()
            or _bootstrap_sha256(path) != expected
            or hashlib.sha256(_bootstrap_git_blob(implementation, relative)).hexdigest()
            != expected
        ):
            raise RuntimeError(f"pre-import critical file changed: {relative}")
    for relative in _BOOTSTRAP_DESIGN_IMPORT_FILES:
        if (ROOT / relative).read_bytes() != _bootstrap_git_blob(
            _BOOTSTRAP_DESIGN_COMMIT, relative
        ):
            raise RuntimeError(f"pre-import frozen design file changed: {relative}")


_bootstrap_verify_before_local_imports()

import yaml  # noqa: E402


sys.path.insert(0, str(EXP / "src"))

from mechanics import (  # noqa: E402
    SURFACE_FEATURE_NAMES,
    SURFACE_SOLVER,
    binary_rank_score,
    build_random_control,
    build_surface_control,
    canonical_json,
    decide_mechanics_a,
    decide_mechanics_b,
    listwise_rank_scores,
    mechanics_authorization,
    operation_alias,
    public_live_map,
    ranking_metrics,
    record_id,
    score_generation_arm,
)
from protocol import (  # noqa: E402
    direct_prompt,
    listwise_prompt,
    suffix_prompt,
    viability_prompt,
)
from task_data import (  # noqa: E402
    ALIASES,
    CONCRETE_OPERATIONS,
    canonical_operation,
    operation_from_record,
    operation_record,
)
from vllm_runner import (  # noqa: E402
    MAX_LOGPROBS,
    MODEL_ID,
    MODEL_REVISION,
    EngineConfig,
    SamplingConfig,
    VLLMRunner,
    _stable_seed as runner_stable_seed,
)


CONFIG_PATH = EXP / "configs" / "default.yaml"
CONSTRUCTION_MANIFEST = EXP / "data" / "procedural" / "manifest.json"
PUBLIC_PATH = EXP / "data" / "procedural" / "mechanics_public.jsonl"
AUDIT_PATH = EXP / "data" / "procedural" / "mechanics_audit.jsonl"
PREPARED = EXP / "runs" / "mechanics" / "prepared"
RAW = EXP / "runs" / "mechanics" / "raw"
SCORED = EXP / "runs" / "mechanics" / "scored"
SUMMARY = EXP / "runs" / "mechanics" / "summary.json"
IMPLEMENTATION_LOCK = EXP / "runs" / "mechanics" / "implementation_lock.json"
DESIGN_COMMIT = "ba08a7320e5fe2d5fb88f2e5b6f3a359ed727cd7"
DESIGN_MANIFEST_SHA256 = "a566129343a4c09473005b355f2ea7aa35549efe101dc14dc8d73ab5a0323857"
DESIGN_FROZEN_FILES = (
    "requirements-vllm.lock.txt",
    str(EXP_REL / "configs/default.yaml"),
    str(EXP_REL / "data/procedural/manifest.json"),
    str(EXP_REL / "data/procedural/mechanics_public.jsonl"),
    str(EXP_REL / "data/procedural/mechanics_audit.jsonl"),
    str(EXP_REL / "reports/preregistration.md"),
    str(EXP_REL / "reports/design_review.md"),
    str(EXP_REL / "src/protocol.py"),
    str(EXP_REL / "src/task_data.py"),
    str(EXP_REL / "src/vllm_runner.py"),
    str(EXP_REL / "tests/test_protocol.py"),
    str(EXP_REL / "tests/test_task_data.py"),
    str(EXP_REL / "tests/test_vllm_runner.py"),
)

SUFFIX_ARMS = (
    "suffix_materialized",
    "suffix_name_only",
    "suffix_shuffled",
    "suffix_echo",
)
BINARY_ARMS = (
    "viability_materialized",
    "viability_name_only",
    "viability_shuffled",
)
INVOCATIONS = (*SUFFIX_ARMS, "direct", *BINARY_ARMS, "listwise")
EXPECTED_COUNTS = {
    **{name: 52 for name in SUFFIX_ARMS},
    "direct": 24,
    **{name: 576 for name in BINARY_ARMS},
    "listwise": 24,
}
CONTROL_FILES = (
    "surface_scores.jsonl",
    "surface_folds.json",
    "random_scores.jsonl",
)
PREPARED_PAYLOAD_NAMES = tuple(
    f"{name}_requests.jsonl" for name in INVOCATIONS
) + CONTROL_FILES

PREPARE_SOURCE_FILES = (
    "requirements-vllm.lock.txt",
    str(EXP_REL / "configs/default.yaml"),
    str(EXP_REL / "data/procedural/manifest.json"),
    str(EXP_REL / "data/procedural/mechanics_public.jsonl"),
    str(EXP_REL / "data/procedural/mechanics_audit.jsonl"),
    str(EXP_REL / "reports/preregistration.md"),
    str(EXP_REL / "reports/design_review.md"),
    str(EXP_REL / "reports/mechanics_implementation_review.md"),
    str(EXP_REL / "scripts/run_mechanics.py"),
    str(EXP_REL / "src/mechanics.py"),
    str(EXP_REL / "src/protocol.py"),
    str(EXP_REL / "src/task_data.py"),
    str(EXP_REL / "src/vllm_runner.py"),
    str(EXP_REL / "tests/test_mechanics.py"),
    str(EXP_REL / "tests/test_vllm_runner.py"),
)
PREPARED_RELATIVE_FILES = tuple(
    str(EXP_REL / "runs/mechanics/prepared" / f"{name}_requests.jsonl")
    for name in INVOCATIONS
) + tuple(str(EXP_REL / "runs/mechanics/prepared" / name) for name in CONTROL_FILES) + (
    str(EXP_REL / "runs/mechanics/prepared/preoutcome_receipt.json"),
)
IMPLEMENTATION_CRITICAL_FILES = frozenset(
    (*PREPARE_SOURCE_FILES, *PREPARED_RELATIVE_FILES)
)
ALLOWED_LIVE_PREFIX = str(EXP_REL / "runs/mechanics") + "/"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def read_json(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"required JSON file is missing or unsafe: {path}")
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"required JSONL file is missing or unsafe: {path}")
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line:
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise RuntimeError(f"non-object JSONL row at {path}:{number}")
        rows.append(value)
    return rows


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(row, sort_keys=True) + "\n").encode("utf-8") for row in rows
    )


def write_frozen(path: Path, value: Any, *, jsonl: bool = False) -> None:
    data = jsonl_bytes(value) if jsonl else json_bytes(value)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
            raise RuntimeError(f"frozen artifact differs on rebuild: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def write_exclusive(path: Path, value: Any, *, jsonl: bool = False) -> None:
    data = jsonl_bytes(value) if jsonl else json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(data)
    except FileExistsError as exc:
        raise RuntimeError(f"refusing to overwrite live artifact: {path}") from exc


def _prepared_file_table() -> dict[str, dict[str, Any]]:
    expected = set(PREPARED_PAYLOAD_NAMES)
    if not PREPARED.is_dir() or PREPARED.is_symlink():
        raise RuntimeError("prepared directory is missing or unsafe")
    observed = {
        path.name
        for path in PREPARED.iterdir()
        if path.name != "preoutcome_receipt.json"
    }
    if observed != expected:
        raise RuntimeError("prepared payload inventory changed")
    table: dict[str, dict[str, Any]] = {}
    for name in PREPARED_PAYLOAD_NAMES:
        path = PREPARED / name
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"prepared payload is missing or unsafe: {name}")
        value = read_jsonl(path) if path.suffix == ".jsonl" else read_json(path)
        if not isinstance(value, list):
            raise RuntimeError(f"prepared payload is not row-oriented: {name}")
        table[name] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "rows": len(value),
        }
    return table


def _validate_public_inputs(
    public_rows: list[dict[str, Any]], audit_rows: list[dict[str, Any]]
) -> None:
    public_keys = {
        "task_id",
        "depth",
        "viability_live_alias",
        "visible",
        "unlabeled_probe_inputs",
    }
    audit_keys = {
        "task_id",
        "stratum",
        "viability_live_alias",
        "visible_attempt",
        "public_live",
    }
    if len(public_rows) != 24 or len(audit_rows) != 24:
        raise RuntimeError("mechanics input row counts changed")
    if any(set(row) != public_keys for row in public_rows):
        raise RuntimeError("mechanics public schema changed")
    if any(set(row) != audit_keys for row in audit_rows):
        raise RuntimeError("mechanics audit schema changed")
    public_ids = [row["task_id"] for row in public_rows]
    audit_ids = [row["task_id"] for row in audit_rows]
    if public_ids != audit_ids or len(set(public_ids)) != 24:
        raise RuntimeError("mechanics task order or IDs changed")
    if sum(len(row["public_live"]) for row in audit_rows) != 52:
        raise RuntimeError("mechanics live-row count changed")
    if any(
        public["viability_live_alias"] != audit["viability_live_alias"]
        for public, audit in zip(public_rows, audit_rows, strict=True)
    ):
        raise RuntimeError("viability orientation differs across public inputs")


def _request(
    *, request_id: str, prompt: str, meta: dict[str, Any]
) -> dict[str, Any]:
    return {
        "id": request_id,
        "messages": [{"role": "user", "content": prompt}],
        "meta": meta,
    }


def _build_requests(
    public_rows: list[dict[str, Any]], audit_rows: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    audit_by_id = {row["task_id"]: row for row in audit_rows}
    requests = {name: [] for name in INVOCATIONS}
    for public in public_rows:
        task_id = public["task_id"]
        audit = audit_by_id[task_id]
        witnesses = {
            operation_alias(operation_from_record(row["operation"])): row
            for row in audit["public_live"]
        }
        for alias, candidate in zip(ALIASES, CONCRETE_OPERATIONS, strict=True):
            candidate_meta = {
                "task_id": task_id,
                "candidate_alias": alias,
                "candidate_canonical": canonical_operation(candidate),
                "candidate": operation_record(candidate),
            }
            binary_id = record_id("mechanics-b-binary-v1", task_id, candidate)
            binary_seed_key = [
                "mechanics-b-binary-v1",
                task_id,
                canonical_operation(candidate),
            ]
            for arm, representation in (
                ("viability_materialized", "materialized"),
                ("viability_name_only", "name_only"),
                ("viability_shuffled", "shuffled"),
            ):
                requests[arm].append(
                    _request(
                        request_id=binary_id,
                        prompt=viability_prompt(
                            public,
                            candidate=candidate,
                            representation=representation,
                        ),
                        meta={
                            **candidate_meta,
                            "condition": arm,
                            "seed_key": binary_seed_key,
                            "public_live": alias in witnesses,
                            "viability_live_alias": public["viability_live_alias"],
                        },
                    )
                )
            if alias not in witnesses:
                continue
            suffix_id = record_id("mechanics-a-suffix-v1", task_id, candidate)
            suffix_seed_key = [
                "mechanics-a-suffix-v1",
                task_id,
                canonical_operation(candidate),
            ]
            witness = witnesses[alias]
            suffix = tuple(
                operation_from_record(value)
                for value in witness["first_fitting_suffix"]
            )
            for arm, representation in (
                ("suffix_materialized", "materialized"),
                ("suffix_name_only", "name_only"),
                ("suffix_shuffled", "shuffled"),
            ):
                requests[arm].append(
                    _request(
                        request_id=suffix_id,
                        prompt=suffix_prompt(
                            public,
                            candidate=candidate,
                            representation=representation,
                        ),
                        meta={
                            **candidate_meta,
                            "condition": arm,
                            "seed_key": suffix_seed_key,
                            "supplied_suffix": None,
                        },
                    )
                )
            requests["suffix_echo"].append(
                _request(
                    request_id=suffix_id,
                    prompt=suffix_prompt(
                        public,
                        candidate=candidate,
                        representation="echo",
                        supplied_suffix=suffix,
                    ),
                    meta={
                        **candidate_meta,
                        "condition": "suffix_echo",
                        "seed_key": suffix_seed_key,
                        "supplied_suffix": [operation_record(value) for value in suffix],
                    },
                )
            )
        requests["direct"].append(
            _request(
                request_id=record_id("mechanics-a-direct-v1", task_id),
                prompt=direct_prompt(public),
                meta={
                    "task_id": task_id,
                    "condition": "direct",
                    "seed_key": ["mechanics-a-direct-v1", task_id],
                },
            )
        )
        requests["listwise"].append(
            _request(
                request_id=record_id("mechanics-b-listwise-v1", task_id),
                prompt=listwise_prompt(public),
                meta={
                    "task_id": task_id,
                    "condition": "listwise",
                    "seed_key": ["mechanics-b-listwise-v1", task_id],
                },
            )
        )
    for name, expected in EXPECTED_COUNTS.items():
        if len(requests[name]) != expected:
            raise RuntimeError(f"prepared count changed for {name}")
        ids = [row["id"] for row in requests[name]]
        if len(set(ids)) != len(ids):
            raise RuntimeError(f"prepared IDs collide within {name}")
    suffix_ids = [[row["id"] for row in requests[name]] for name in SUFFIX_ARMS]
    binary_ids = [[row["id"] for row in requests[name]] for name in BINARY_ARMS]
    if len({tuple(values) for values in suffix_ids}) != 1:
        raise RuntimeError("suffix causal arms do not share exact record IDs/order")
    if len({tuple(values) for values in binary_ids}) != 1:
        raise RuntimeError("binary causal arms do not share exact record IDs/order")
    return requests


def _tokenizer_receipt(
    config: dict[str, Any], requests: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    from collections import Counter
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["id"],
        revision=config["model"]["revision"],
        trust_remote_code=True,
        use_fast=True,
    )
    plain_ids: dict[str, int] = {}
    leading_ids: dict[str, int] = {}
    for alias in ALIASES:
        plain = tokenizer.encode(alias, add_special_tokens=False)
        leading = tokenizer.encode(" " + alias, add_special_tokens=False)
        if len(plain) != 1 or len(leading) != 1:
            raise RuntimeError("candidate alias is not single-token")
        plain_ids[alias] = int(plain[0])
        leading_ids[alias] = int(leading[0])
    if plain_ids != {
        alias: 32 + index for index, alias in enumerate(ALIASES)
    } or MAX_LOGPROBS != 24:
        raise RuntimeError("24-way targeted-logprob geometry changed")
    close_ids = tokenizer.encode("</think>\n\n", add_special_tokens=False)
    if close_ids != [248069, 271]:
        raise RuntimeError("forced-close tokenization changed")
    think_open_ids = tokenizer.encode("<think>", add_special_tokens=False)
    think_close_ids = tokenizer.encode("</think>", add_special_tokens=False)
    thinking_prompt_suffix_ids = tokenizer.encode(
        "<|im_start|>assistant\n<think>\n", add_special_tokens=False
    )
    no_thinking_prompt_suffix_ids = tokenizer.encode(
        "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        add_special_tokens=False,
    )
    if think_open_ids != [248068] or think_close_ids != [248069]:
        raise RuntimeError("think-boundary tokenization changed")

    receipt: dict[str, Any] = {
        "plain_alias_token_ids": plain_ids,
        "leading_space_alias_token_ids": leading_ids,
        "think_open_token_ids": think_open_ids,
        "think_close_token_ids": think_close_ids,
        "forced_close_token_ids": close_ids,
        "thinking_prompt_suffix_ids": thinking_prompt_suffix_ids,
        "no_thinking_prompt_suffix_ids": no_thinking_prompt_suffix_ids,
        "invocations": {},
    }
    token_rows_by_name: dict[str, list[list[int]]] = {}
    for name in INVOCATIONS:
        enable_thinking = name in (*SUFFIX_ARMS, "direct")
        if name in SUFFIX_ARMS:
            reserve = (
                int(config["generation"]["suffix_thinking_budget"])
                + len(close_ids)
                + int(config["generation"]["suffix_answer_max_tokens"])
            )
        elif name == "direct":
            reserve = (
                int(config["generation"]["direct_thinking_budget"])
                + len(close_ids)
                + int(config["generation"]["direct_answer_max_tokens"])
            )
        else:
            reserve = 1
        user_prompts = [row["messages"][0]["content"] for row in requests[name]]
        rendered_prompts: list[str] = []
        token_rows: list[list[int]] = []
        for prompt in user_prompts:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            if not isinstance(rendered, str):
                raise RuntimeError("chat template returned non-text")
            rendered_prompts.append(rendered)
            token_rows.append(tokenizer.encode(rendered, add_special_tokens=False))
        lengths = [len(row) for row in token_rows]
        maximum_total = max(lengths) + reserve
        if maximum_total > int(config["generation"]["max_model_len"]):
            raise RuntimeError(f"prepared {name} prompt exceeds context")
        token_rows_by_name[name] = token_rows
        receipt["invocations"][name] = {
            "rows": len(user_prompts),
            "enable_thinking": enable_thinking,
            "reserve_tokens": reserve,
            "prompt_tokens_min": min(lengths),
            "prompt_tokens_max": max(lengths),
            "max_prompt_plus_reserve": maximum_total,
            "context_slack_min": int(config["generation"]["max_model_len"])
            - maximum_total,
            "record_ids_sha256": canonical_sha256(
                [row["id"] for row in requests[name]]
            ),
            "seed_keys_sha256": canonical_sha256(
                [row["meta"]["seed_key"] for row in requests[name]]
            ),
            "user_prompt_text_sha256": canonical_sha256(user_prompts),
            "rendered_prompt_text_sha256": canonical_sha256(rendered_prompts),
            "prompt_token_ids_sha256": canonical_sha256(token_rows),
        }
    for clean_name, shuffled_name in (
        ("suffix_materialized", "suffix_shuffled"),
        ("viability_materialized", "viability_shuffled"),
    ):
        clean = token_rows_by_name[clean_name]
        shuffled = token_rows_by_name[shuffled_name]
        if len(clean) != len(shuffled) or any(
            Counter(left) != Counter(right)
            for left, right in zip(clean, shuffled, strict=True)
        ):
            raise RuntimeError("clean/shuffled rendered token multisets differ")
    receipt["materialized_shuffled_token_multisets_exact"] = True
    return receipt


def _source_hashes() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in PREPARE_SOURCE_FILES:
        path = ROOT / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"prepare source is missing or unsafe: {relative}")
        result[relative] = sha256_file(path)
    return result


def build_prepared() -> tuple[
    dict[str, list[dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]
]:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    construction = read_json(CONSTRUCTION_MANIFEST)
    if sha256_file(CONSTRUCTION_MANIFEST) != DESIGN_MANIFEST_SHA256:
        raise RuntimeError("published construction manifest changed")
    if construction.get("model_loaded") is not False or construction.get("model_calls") != 0:
        raise RuntimeError("construction receipt is not model-free")
    for key, path in (("mechanics_public", PUBLIC_PATH), ("mechanics_audit", AUDIT_PATH)):
        expected = construction["paths"][key]["sha256"]
        if sha256_file(path) != expected:
            raise RuntimeError(f"published mechanics input changed: {key}")
    public_rows = read_jsonl(PUBLIC_PATH)
    audit_rows = read_jsonl(AUDIT_PATH)
    _validate_public_inputs(public_rows, audit_rows)
    requests = _build_requests(public_rows, audit_rows)
    surface_scores, surface_folds = build_surface_control(public_rows, audit_rows)
    random_scores = build_random_control(
        public_rows, seed=int(config["seeds"]["branch_permutation"])
    )
    tokenizer = _tokenizer_receipt(config, requests)
    published_tokenizer = construction["tokenizer"]
    published_aliases = published_tokenizer["alias_token_ids"]
    if (
        tokenizer["plain_alias_token_ids"]
        != {alias: published_aliases[alias]["plain"][0] for alias in ALIASES}
        or tokenizer["leading_space_alias_token_ids"]
        != {
            alias: published_aliases[alias]["leading_space"][0]
            for alias in ALIASES
        }
        or tokenizer["think_open_token_ids"]
        != published_tokenizer["think_open_token_ids"]
        or tokenizer["think_close_token_ids"]
        != published_tokenizer["think_close_token_ids"]
        or tokenizer["forced_close_token_ids"]
        != published_tokenizer["forced_close_token_ids"]
        or tokenizer["thinking_prompt_suffix_ids"]
        != published_tokenizer["thinking_prompt_suffix_ids"]
        or tokenizer["no_thinking_prompt_suffix_ids"]
        != published_tokenizer["no_thinking_prompt_suffix_ids"]
    ):
        raise RuntimeError("mechanics tokenizer identity differs from construction lock")
    smoke_tokenizer = published_tokenizer["conditions"]
    echo = tokenizer["invocations"]["suffix_echo"]
    published_echo = smoke_tokenizer["suffix_echo_mechanics_live"]
    echo_keys = {
        "enable_thinking",
        "reserve_tokens",
        "prompt_tokens_min",
        "prompt_tokens_max",
        "max_prompt_plus_reserve",
        "context_slack_min",
        "user_prompt_text_sha256",
        "rendered_prompt_text_sha256",
        "prompt_token_ids_sha256",
    }
    if (
        {key: echo[key] for key in echo_keys}
        != {key: published_echo[key] for key in echo_keys}
        or echo["rows"] != published_echo["prompt_count"]
    ):
        raise RuntimeError("mechanics echo tokenizer receipt differs from construction lock")
    static_fit = all(
        smoke_tokenizer[name]["context_slack_min"] >= 0
        for name in ("suffix_materialized", "viability_materialized", "listwise")
    )
    if not static_fit:
        raise RuntimeError("published top-four context receipt does not fit")
    receipt = {
        "schema_version": 1,
        "stage": "mechanics_prepare",
        "decision": "MECHANICS_PREPARE_PASS",
        "design_commit": DESIGN_COMMIT,
        "construction_manifest_sha256": DESIGN_MANIFEST_SHA256,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "backend": "vllm",
        "run_seed": int(config["seeds"]["mechanics"]),
        "invocation_order": list(INVOCATIONS),
        "expected_counts": EXPECTED_COUNTS,
        "total_model_requests": sum(EXPECTED_COUNTS.values()),
        "targeted_raw_logprobs": {
            "binary_values": sum(EXPECTED_COUNTS[name] * 2 for name in BINARY_ARMS),
            "listwise_values": EXPECTED_COUNTS["listwise"] * 24,
            "total_values": sum(EXPECTED_COUNTS[name] * 2 for name in BINARY_ARMS)
            + EXPECTED_COUNTS["listwise"] * 24,
        },
        "same_ids_order_seed_suffix_arms": True,
        "same_ids_order_seed_binary_arms": True,
        "surface_feature_names": list(SURFACE_FEATURE_NAMES),
        "surface_solver": SURFACE_SOLVER,
        "surface_numpy_version": __import__("numpy").__version__,
        "surface_fold_count": len(surface_folds),
        "surface_scores_sha256": canonical_sha256(surface_scores),
        "random_scores_sha256": canonical_sha256(random_scores),
        "registered_top4_static_context_fit": static_fit,
        "published_tokenizer_identity_match": True,
        "published_echo_receipt_match": True,
        "tokenizer": tokenizer,
        "source_sha256": _source_hashes(),
        "environment_lock": {
            "path": "requirements-vllm.lock.txt",
            "sha256": sha256_file(ROOT / "requirements-vllm.lock.txt"),
            "exact_distribution_count": len(_locked_environment_versions(config)),
            "expected_versions_sha256": canonical_sha256(
                _locked_environment_versions(config)
            ),
        },
        "public_inputs": {
            str(PUBLIC_PATH.relative_to(ROOT)): sha256_file(PUBLIC_PATH),
            str(AUDIT_PATH.relative_to(ROOT)): sha256_file(AUDIT_PATH),
        },
        "hidden_files_read": [],
        "qualification_files_read": [],
        "confirmation_files_read": [],
        "benchmark_files_read": [],
        "model_loaded": False,
        "model_calls": 0,
        "outcomes_loaded": False,
    }
    return requests, surface_scores, surface_folds, random_scores, receipt


def prepare() -> dict[str, Any]:
    _validate_current_scientific_environment(yaml.safe_load(CONFIG_PATH.read_text()))
    requests, surface_scores, surface_folds, random_scores, receipt = build_prepared()
    for name in INVOCATIONS:
        write_frozen(PREPARED / f"{name}_requests.jsonl", requests[name], jsonl=True)
    write_frozen(PREPARED / "surface_scores.jsonl", surface_scores, jsonl=True)
    write_frozen(PREPARED / "surface_folds.json", surface_folds)
    write_frozen(PREPARED / "random_scores.jsonl", random_scores, jsonl=True)
    receipt["files"] = _prepared_file_table()
    write_frozen(PREPARED / "preoutcome_receipt.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _git_bytes(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT)


def _verify_committed_file_bytes(path: Path, relative: str, commit: str) -> bytes:
    blob = _git_bytes("show", f"{commit}:{relative}")
    if path.read_bytes() != blob:
        raise RuntimeError(f"working bytes differ from {commit}: {relative}")
    return blob


def verify_implementation_lock(lock_path: Path) -> dict[str, Any]:
    try:
        relative_lock = lock_path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise RuntimeError("implementation lock escapes the repository") from exc
    if lock_path.is_symlink() or not lock_path.is_file():
        raise RuntimeError("live mechanics requires a committed implementation lock")
    lock = read_json(lock_path)
    required = {
        "schema_version",
        "design_commit",
        "design_manifest_sha256",
        "design_frozen_files",
        "implementation_commit",
        "critical_files",
        "model_calls_before_lock",
        "authorization",
    }
    if set(lock) != required or lock["schema_version"] != 1:
        raise RuntimeError("implementation lock schema changed")
    if lock["design_commit"] != DESIGN_COMMIT or lock["design_manifest_sha256"] != DESIGN_MANIFEST_SHA256:
        raise RuntimeError("implementation lock design boundary changed")
    if lock["model_calls_before_lock"] != 0 or lock["authorization"] != "mechanics_only":
        raise RuntimeError("implementation lock does not certify a pre-model mechanics boundary")
    critical = lock["critical_files"]
    if not isinstance(critical, dict) or set(critical) != IMPLEMENTATION_CRITICAL_FILES:
        raise RuntimeError("implementation critical-file allowlist changed")
    frozen = lock["design_frozen_files"]
    if not isinstance(frozen, dict) or set(frozen) != set(DESIGN_FROZEN_FILES):
        raise RuntimeError("implementation frozen-design allowlist changed")
    for relative in DESIGN_FROZEN_FILES:
        try:
            design_blob = _git_bytes("show", f"{DESIGN_COMMIT}:{relative}")
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"frozen design file absent: {relative}") from exc
        expected = sha256_bytes(design_blob)
        path = ROOT / relative
        if (
            frozen[relative] != expected
            or path.is_symlink()
            or not path.is_file()
            or sha256_file(path) != expected
        ):
            raise RuntimeError(f"frozen design file changed: {relative}")
    if _git("ls-files", "--error-unmatch", "--", relative_lock) != relative_lock:
        raise RuntimeError("implementation lock is not tracked")
    _git("cat-file", "-e", f"HEAD:{relative_lock}")
    _verify_committed_file_bytes(lock_path, relative_lock, "HEAD")

    dirty = _git("status", "--porcelain=v1", "--untracked-files=all")
    for line in dirty.splitlines():
        paths = line[3:].split(" -> ")
        if not all(
            path == str(SUMMARY.relative_to(ROOT))
            or path.startswith(ALLOWED_LIVE_PREFIX + "raw/")
            or path.startswith(ALLOWED_LIVE_PREFIX + "scored/")
            for path in paths
        ):
            raise RuntimeError(f"live mechanics has unrelated worktree change: {line}")

    subprocess.run(["git", "fetch", "--quiet", "origin", "main"], cwd=ROOT, check=True)
    implementation_commit = str(lock["implementation_commit"])
    commits = (DESIGN_COMMIT, implementation_commit, _git("rev-parse", "HEAD"))
    if any(
        len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit)
        for commit in commits
    ):
        raise RuntimeError("implementation boundary contains an invalid commit")
    for ancestor, descendant in (
        (DESIGN_COMMIT, implementation_commit),
        (implementation_commit, "HEAD"),
        ("HEAD", "origin/main"),
    ):
        if subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant], cwd=ROOT
        ).returncode:
            raise RuntimeError(f"unpublished implementation boundary: {ancestor} !<= {descendant}")
    for relative, expected in critical.items():
        path = ROOT / relative
        if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"implementation critical file changed: {relative}")
        try:
            blob = _git_bytes("show", f"{implementation_commit}:{relative}")
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"critical file absent at implementation commit: {relative}") from exc
        if sha256_bytes(blob) != expected:
            raise RuntimeError(f"implementation commit blob changed: {relative}")
    return lock


def publish_implementation_lock() -> dict[str, Any]:
    _validate_current_scientific_environment(yaml.safe_load(CONFIG_PATH.read_text()))
    _load_prepared()
    if RAW.exists() or SCORED.exists() or SUMMARY.exists():
        raise RuntimeError("implementation lock must precede every live mechanics artifact")
    if _git("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("implementation lock requires a clean committed worktree")
    subprocess.run(["git", "fetch", "--quiet", "origin", "main"], cwd=ROOT, check=True)
    implementation_commit = _git("rev-parse", "HEAD")
    for ancestor, descendant in (
        (DESIGN_COMMIT, implementation_commit),
        (implementation_commit, "origin/main"),
    ):
        if subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant], cwd=ROOT
        ).returncode:
            raise RuntimeError(
                f"implementation is not published: {ancestor} !<= {descendant}"
            )
    critical: dict[str, str] = {}
    for relative in sorted(IMPLEMENTATION_CRITICAL_FILES):
        path = ROOT / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"implementation critical file is unsafe: {relative}")
        if _git("ls-files", "--error-unmatch", "--", relative) != relative:
            raise RuntimeError(f"implementation critical file is untracked: {relative}")
        blob = _verify_committed_file_bytes(path, relative, implementation_commit)
        critical[relative] = sha256_bytes(blob)
    frozen: dict[str, str] = {}
    for relative in DESIGN_FROZEN_FILES:
        design_blob = _git_bytes("show", f"{DESIGN_COMMIT}:{relative}")
        if (ROOT / relative).read_bytes() != design_blob:
            raise RuntimeError(f"frozen design file changed: {relative}")
        frozen[relative] = sha256_bytes(design_blob)
    lock = {
        "schema_version": 1,
        "design_commit": DESIGN_COMMIT,
        "design_manifest_sha256": DESIGN_MANIFEST_SHA256,
        "design_frozen_files": dict(sorted(frozen.items())),
        "implementation_commit": implementation_commit,
        "critical_files": critical,
        "model_calls_before_lock": 0,
        "authorization": "mechanics_only",
    }
    write_frozen(IMPLEMENTATION_LOCK, lock)
    print(json.dumps(lock, indent=2, sort_keys=True))
    return lock


def _load_prepared() -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    rebuilt = build_prepared()
    requests, surface_scores, surface_folds, random_scores, base_receipt = rebuilt
    stored: dict[str, list[dict[str, Any]]] = {}
    for name in INVOCATIONS:
        path = PREPARED / f"{name}_requests.jsonl"
        rows = read_jsonl(path)
        if rows != requests[name]:
            raise RuntimeError(f"prepared request file changed: {name}")
        stored[name] = rows
    if read_jsonl(PREPARED / "surface_scores.jsonl") != surface_scores:
        raise RuntimeError("prepared surface scores changed")
    if read_json(PREPARED / "surface_folds.json") != surface_folds:
        raise RuntimeError("prepared surface folds changed")
    if read_jsonl(PREPARED / "random_scores.jsonl") != random_scores:
        raise RuntimeError("prepared random scores changed")
    receipt = read_json(PREPARED / "preoutcome_receipt.json")
    expected = dict(base_receipt)
    expected["files"] = _prepared_file_table()
    if receipt != expected:
        raise RuntimeError("prepared preoutcome receipt changed")
    return receipt, stored


def _sampling(name: str, config: dict[str, Any], token_ids: dict[str, int]) -> SamplingConfig:
    generation = config["generation"]
    common = {
        "n": 1,
        "temperature": float(generation["temperature"]),
        "top_p": float(generation["top_p"]),
        "top_k": int(generation["top_k"]),
        "run_seed": int(config["seeds"]["mechanics"]),
    }
    if name in SUFFIX_ARMS:
        return SamplingConfig(
            thinking="budget",
            thinking_budget=int(generation["suffix_thinking_budget"]),
            max_tokens=int(generation["suffix_thinking_budget"]),
            answer_max_tokens=int(generation["suffix_answer_max_tokens"]),
            **common,
        )
    if name == "direct":
        return SamplingConfig(
            thinking="budget",
            thinking_budget=int(generation["direct_thinking_budget"]),
            max_tokens=int(generation["direct_thinking_budget"]),
            answer_max_tokens=int(generation["direct_answer_max_tokens"]),
            **common,
        )
    requested = ("A", "B") if name in BINARY_ARMS else ALIASES
    return SamplingConfig(
        thinking="off",
        max_tokens=1,
        answer_max_tokens=1,
        logprobs=len(requested),
        logprob_token_ids=tuple(token_ids[alias] for alias in requested),
        **common,
    )


def _engine_config(config: dict[str, Any]) -> EngineConfig:
    generation = config["generation"]
    return EngineConfig(
        max_model_len=int(generation["max_model_len"]),
        gpu_memory_utilization=0.90,
        max_num_seqs=int(generation["max_num_seqs"]),
        max_num_batched_tokens=int(generation["max_num_batched_tokens"]),
        enable_prefix_caching=bool(generation["enable_prefix_caching"]),
        cudagraph_capture_sizes=tuple(generation["cudagraph_capture_sizes"]),
    )


def _normalized(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def _normalize_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _locked_environment_versions(config: dict[str, Any]) -> dict[str, str]:
    versions: dict[str, str] = {}
    direct: set[str] = set()
    for raw_line in (ROOT / "requirements-vllm.lock.txt").read_text().splitlines():
        if not raw_line or raw_line[:1].isspace() or raw_line.startswith("#"):
            continue
        line = raw_line.split(";", 1)[0].strip()
        if " @ " in line:
            name, url = line.split(" @ ", 1)
            normalized = _normalize_distribution_name(name.strip())
            if not normalized or not url.strip() or normalized in versions:
                raise RuntimeError("invalid direct requirement in vLLM lock")
            direct.add(normalized)
            continue
        if "==" not in line:
            raise RuntimeError(f"unrecognized exact requirement in vLLM lock: {line}")
        name, version = line.split("==", 1)
        normalized = _normalize_distribution_name(name.strip())
        version = version.strip()
        if not normalized or not version or normalized in versions:
            raise RuntimeError("invalid or duplicate exact requirement in vLLM lock")
        versions[normalized] = version
    expected_direct = {"vllm"}
    if direct != expected_direct or "vllm" in versions:
        raise RuntimeError(f"unexpected direct requirements in vLLM lock: {sorted(direct)}")
    versions["vllm"] = str(config["model"]["vllm_version"])
    return dict(sorted(versions.items()))


def _validate_package_inventory(
    packages: Any, config: dict[str, Any], *, source: str
) -> dict[str, str]:
    if not isinstance(packages, dict) or any(
        not isinstance(name, str) or not isinstance(version, str)
        for name, version in packages.items()
    ):
        raise RuntimeError(f"{source} package inventory has invalid schema")
    normalized = {
        _normalize_distribution_name(name): version for name, version in packages.items()
    }
    if len(normalized) != len(packages):
        raise RuntimeError(f"{source} package inventory has normalized-name collisions")
    expected = _locked_environment_versions(config)
    mismatches = {
        name: {"expected": version, "observed": normalized.get(name)}
        for name, version in expected.items()
        if normalized.get(name) != version
    }
    if mismatches:
        raise RuntimeError(f"{source} differs from the exact vLLM lock: {mismatches}")
    return normalized


def _installed_environment_versions(config: dict[str, Any]) -> dict[str, str]:
    expected = _locked_environment_versions(config)
    installed: dict[str, str] = {}
    for name in expected:
        try:
            installed[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return installed


def _validate_locked_environment(config: dict[str, Any]) -> dict[str, str]:
    return _validate_package_inventory(
        _installed_environment_versions(config), config, source="live environment"
    )


def _validate_current_scientific_environment(
    config: dict[str, Any]
) -> dict[str, str]:
    executable = Path(sys.executable)
    if (
        executable.parent != ROOT / ".venv-vllm" / "bin"
        or not executable.name.startswith("python")
    ):
        raise RuntimeError("mechanics science requires the pinned .venv-vllm interpreter")
    return _validate_locked_environment(config)


def _validate_runtime_receipt(
    runtime: Any, config: dict[str, Any], *, source: str
) -> dict[str, Any]:
    required = {
        "python",
        "python_executable",
        "platform",
        "packages",
        "environment_lock",
        "uv",
        "cuda_toolkit",
        "gpu",
        "vllm_enable_v1_multiprocessing",
        "git_commit",
        "git_dirty",
    }
    if not isinstance(runtime, dict) or set(runtime) != required:
        raise RuntimeError(f"{source} runtime schema changed")
    _validate_package_inventory(runtime["packages"], config, source=source)
    environment_lock = runtime["environment_lock"]
    executable = Path(str(runtime["python_executable"]))
    if (
        not isinstance(environment_lock, dict)
        or environment_lock.get("sha256")
        != sha256_file(ROOT / "requirements-vllm.lock.txt")
        or runtime["vllm_enable_v1_multiprocessing"] != "0"
        or executable.parent != ROOT / ".venv-vllm" / "bin"
        or not executable.name.startswith("python")
        or not isinstance(runtime["python"], str)
        or not runtime["python"]
        or not isinstance(runtime["platform"], str)
        or not runtime["platform"]
        or not isinstance(runtime["cuda_toolkit"], str)
        or not runtime["cuda_toolkit"]
        or not isinstance(runtime["gpu"], str)
        or not runtime["gpu"]
        or not isinstance(runtime["git_commit"], str)
        or len(runtime["git_commit"]) != 40
        or any(value not in "0123456789abcdef" for value in runtime["git_commit"])
        or not isinstance(runtime["git_dirty"], bool)
    ):
        raise RuntimeError(f"{source} runtime fingerprint changed")
    return runtime


def _runtime_projection(runtime: dict[str, Any]) -> dict[str, Any]:
    return {
        key: runtime[key]
        for key in (
            "python",
            "python_executable",
            "platform",
            "packages",
            "environment_lock",
            "uv",
            "cuda_toolkit",
            "gpu",
            "vllm_enable_v1_multiprocessing",
        )
    }


def _validate_invocation_runtime(
    runtime: Any,
    preflight: dict[str, Any],
    config: dict[str, Any],
    *,
    source: str,
) -> None:
    _validate_runtime_receipt(runtime, config, source=source)
    stored = preflight.get("runtime")
    if not isinstance(stored, dict) or _runtime_projection(runtime) != _runtime_projection(
        stored
    ):
        raise RuntimeError(f"{source} differs from the live preflight runtime")


def _expected_engine_args(config: dict[str, Any]) -> dict[str, Any]:
    engine = _engine_config(config)
    capture_sizes = list(engine.cudagraph_capture_sizes or ())
    return {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "max_model_len": engine.max_model_len,
        "gpu_memory_utilization": engine.gpu_memory_utilization,
        "max_num_seqs": engine.max_num_seqs,
        "max_num_batched_tokens": engine.max_num_batched_tokens,
        "language_model_only": True,
        "enable_prefix_caching": False,
        "mamba_cache_mode": "none",
        "enforce_eager": False,
        "generation_config": "vllm",
        "max_logprobs": 24,
        "logprobs_mode": "raw_logprobs",
        "seed": 0,
        "async_scheduling": False,
        "cudagraph_capture_sizes": capture_sizes,
        "max_cudagraph_capture_size": capture_sizes[-1],
    }


def _validate_recorded_live_preflight(
    preflight: dict[str, Any],
    config: dict[str, Any],
    receipt: dict[str, Any],
    lock_path: Path,
) -> None:
    required = {
        "schema_version",
        "decision",
        "model",
        "revision",
        "implementation_lock_sha256",
        "prepare_receipt_sha256",
        "engine",
        "engine_args",
        "resolved_cudagraph",
        "resolved_logprobs_mode",
        "live_model",
        "live_scheduler",
        "live_parallel",
        "live_cache",
        "runtime",
        "invocations",
    }
    if set(preflight) != required:
        raise RuntimeError("live preflight schema changed")
    if (
        preflight["schema_version"] != 1
        or preflight["decision"] != "LIVE_ENGINE_PREFLIGHT_PASS"
        or preflight["model"] != MODEL_ID
        or preflight["revision"] != MODEL_REVISION
        or preflight["implementation_lock_sha256"] != sha256_file(lock_path)
        or preflight["prepare_receipt_sha256"]
        != sha256_file(PREPARED / "preoutcome_receipt.json")
        or preflight["engine"] != _normalized(dataclasses.asdict(_engine_config(config)))
        or preflight["engine_args"] != _normalized(_expected_engine_args(config))
        or preflight["resolved_logprobs_mode"] != "raw_logprobs"
    ):
        raise RuntimeError("live preflight static authentication failed")
    resolved = preflight["resolved_cudagraph"]
    expected_sizes = list(config["generation"]["cudagraph_capture_sizes"])
    supported_modes = {
        "FULL": ("FULL", "FULL"),
        "FULL_DECODE_ONLY": ("FULL", "NONE"),
        "FULL_AND_PIECEWISE": ("FULL", "PIECEWISE"),
    }
    mode = resolved.get("mode") if isinstance(resolved, dict) else None
    if (
        not isinstance(resolved, dict)
        or resolved.get("source")
        != "llm_engine.vllm_config.compilation_config"
        or resolved.get("cudagraph_capture_sizes") != expected_sizes
        or resolved.get("max_cudagraph_capture_size") != expected_sizes[-1]
        or mode not in supported_modes
        or (resolved.get("decode_mode"), resolved.get("mixed_mode"))
        != supported_modes.get(mode)
        or resolved.get("has_full_cudagraphs") is not True
    ):
        raise RuntimeError("live preflight CUDA-graph authentication failed")
    engine = _engine_config(config)
    if preflight["live_model"] != {
        "max_model_len": engine.max_model_len,
        "dtype": "torch.bfloat16",
    } and preflight["live_model"] != {
        "max_model_len": engine.max_model_len,
        "dtype": "bfloat16",
    }:
        raise RuntimeError("live preflight model geometry changed")
    if preflight["live_scheduler"] != {
        "max_num_seqs": engine.max_num_seqs,
        "max_num_batched_tokens": engine.max_num_batched_tokens,
        "async_scheduling": False,
    } or preflight["live_parallel"] != {
        "world_size": 1,
        "tensor_parallel_size": 1,
        "data_parallel_size": 1,
    }:
        raise RuntimeError("live preflight scheduler/parallel geometry changed")
    cache = preflight["live_cache"]
    if (
        not isinstance(cache, dict)
        or set(cache)
        != {
            "num_gpu_blocks",
            "block_size",
            "kv_cache_size_tokens",
            "kv_cache_max_concurrency",
            "enable_prefix_caching",
            "mamba_cache_mode",
            "mamba_block_size",
        }
        or not isinstance(cache["num_gpu_blocks"], int)
        or cache["num_gpu_blocks"] < engine.max_num_seqs
        or not isinstance(cache["block_size"], int)
        or cache["block_size"] < 1
        or not isinstance(cache["kv_cache_size_tokens"], int)
        or cache["kv_cache_size_tokens"] < 1
        or isinstance(cache["kv_cache_max_concurrency"], bool)
        or not isinstance(cache["kv_cache_max_concurrency"], (int, float))
        or not math.isfinite(float(cache["kv_cache_max_concurrency"]))
        or not math.isclose(
            float(cache["kv_cache_max_concurrency"]),
            cache["kv_cache_size_tokens"] / engine.max_model_len,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or not isinstance(cache["mamba_block_size"], int)
        or isinstance(cache["mamba_block_size"], bool)
        or cache["mamba_block_size"] < 1
        or cache["enable_prefix_caching"] is not False
        or cache["mamba_cache_mode"] != "none"
    ):
        raise RuntimeError("live preflight cache geometry changed")
    _validate_runtime_receipt(
        preflight["runtime"], config, source="live preflight"
    )
    invocations = preflight["invocations"]
    if not isinstance(invocations, dict) or set(invocations) != set(INVOCATIONS):
        raise RuntimeError("live preflight invocation inventory changed")
    block_size = cache["block_size"]
    capacity = cache["kv_cache_size_tokens"]
    for name in INVOCATIONS:
        row = invocations[name]
        token_row = receipt["tokenizer"]["invocations"][name]
        sampling = _sampling(
            name, config, receipt["tokenizer"]["plain_alias_token_ids"]
        )
        reserve = (
            int(sampling.thinking_budget)
            + len(receipt["tokenizer"]["forced_close_token_ids"])
            + sampling.answer_max_tokens
            if sampling.thinking == "budget"
            else sampling.max_tokens
        )
        maximum_total = token_row["prompt_tokens_max"] + reserve
        rounded = math.ceil(maximum_total / block_size) * block_size
        active = min(EXPECTED_COUNTS[name], engine.max_num_seqs)
        expected_row = {
            "requests": EXPECTED_COUNTS[name],
            "prompt_tokens_min": token_row["prompt_tokens_min"],
            "prompt_tokens_max": token_row["prompt_tokens_max"],
            "reserve_tokens": reserve,
            "max_prompt_plus_reserve": maximum_total,
            "active_sequences": active,
            "rounded_tokens_per_sequence": rounded,
            "required_cache_tokens": active * rounded,
            "remaining_cache_tokens": capacity - active * rounded,
        }
        if row != expected_row or expected_row["remaining_cache_tokens"] < 0:
            raise RuntimeError(f"live preflight invocation geometry changed: {name}")


def _load_recorded_live_preflight(
    config: dict[str, Any], receipt: dict[str, Any], lock_path: Path
) -> dict[str, Any] | None:
    path = RAW / "live_preflight.json"
    if not path.exists():
        return None
    preflight = read_json(path)
    _validate_recorded_live_preflight(preflight, config, receipt, lock_path)
    _validate_invocation_runtime(
        VLLMRunner.runtime_metadata(),
        preflight,
        config,
        source="current mechanics process",
    )
    return preflight


def _live_preflight(
    runner: VLLMRunner,
    config: dict[str, Any],
    prepared: dict[str, list[dict[str, Any]]],
    receipt: dict[str, Any],
    lock_path: Path,
) -> dict[str, Any]:
    if (
        _normalized(dataclasses.asdict(runner.config))
        != _normalized(dataclasses.asdict(_engine_config(config)))
        or _normalized(runner.engine_args)
        != _normalized(_expected_engine_args(config))
        or runner.resolved_logprobs_mode != "raw_logprobs"
    ):
        raise RuntimeError("loaded runner differs from frozen engine settings")
    vllm_config = runner.llm.llm_engine.vllm_config
    cache = vllm_config.cache_config
    scheduler = vllm_config.scheduler_config
    model = vllm_config.model_config
    parallel = vllm_config.parallel_config
    capacity = getattr(cache, "kv_cache_size_tokens", None)
    block_size = getattr(cache, "block_size", None)
    num_gpu_blocks = getattr(cache, "num_gpu_blocks", None)
    if not isinstance(capacity, int) or capacity <= 0 or not isinstance(block_size, int) or block_size <= 0:
        raise RuntimeError("live engine did not expose positive KV geometry")
    if not isinstance(num_gpu_blocks, int) or num_gpu_blocks < runner.config.max_num_seqs:
        raise RuntimeError("live engine cannot honor frozen max_num_seqs")
    if (
        int(model.max_model_len) != runner.config.max_model_len
        or str(model.dtype) not in {"bfloat16", "torch.bfloat16"}
        or int(scheduler.max_num_seqs) != runner.config.max_num_seqs
        or int(scheduler.max_num_batched_tokens) != runner.config.max_num_batched_tokens
        or bool(scheduler.async_scheduling)
        or int(parallel.world_size) != 1
        or int(parallel.tensor_parallel_size) != 1
        or int(parallel.data_parallel_size) != 1
        or bool(cache.enable_prefix_caching)
        or str(cache.mamba_cache_mode) != "none"
    ):
        raise RuntimeError("live engine geometry differs from the frozen protocol")
    token_ids = receipt["tokenizer"]["plain_alias_token_ids"]
    invocation_capacity: dict[str, Any] = {}
    for name in INVOCATIONS:
        sampling = _sampling(name, config, token_ids)
        exact = runner.prepare(prepared[name], sampling.thinking, False)
        expected_token_sha = receipt["tokenizer"]["invocations"][name][
            "prompt_token_ids_sha256"
        ]
        if canonical_sha256([row.prompt_token_ids for row in exact]) != expected_token_sha:
            raise RuntimeError(f"live rendered prompt IDs changed: {name}")
        reserve = (
            int(sampling.thinking_budget) + len(runner.close_ids) + sampling.answer_max_tokens
            if sampling.thinking == "budget"
            else sampling.max_tokens
        )
        lengths = [len(row.prompt_token_ids) for row in exact]
        maximum_total = max(lengths) + reserve
        rounded = math.ceil(maximum_total / block_size) * block_size
        active = min(len(lengths), runner.config.max_num_seqs)
        required = active * rounded
        if maximum_total > runner.config.max_model_len or required > capacity:
            raise RuntimeError(f"live KV capacity cannot fit invocation {name}")
        invocation_capacity[name] = {
            "requests": len(lengths),
            "prompt_tokens_min": min(lengths),
            "prompt_tokens_max": max(lengths),
            "reserve_tokens": reserve,
            "max_prompt_plus_reserve": maximum_total,
            "active_sequences": active,
            "rounded_tokens_per_sequence": rounded,
            "required_cache_tokens": required,
            "remaining_cache_tokens": capacity - required,
        }
    preflight = {
        "schema_version": 1,
        "decision": "LIVE_ENGINE_PREFLIGHT_PASS",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "implementation_lock_sha256": sha256_file(lock_path),
        "prepare_receipt_sha256": sha256_file(PREPARED / "preoutcome_receipt.json"),
        "engine": _normalized(dataclasses.asdict(runner.config)),
        "engine_args": _normalized(runner.engine_args),
        "resolved_cudagraph": _normalized(runner.resolved_cudagraph),
        "resolved_logprobs_mode": runner.resolved_logprobs_mode,
        "live_model": {"max_model_len": int(model.max_model_len), "dtype": str(model.dtype)},
        "live_scheduler": {
            "max_num_seqs": int(scheduler.max_num_seqs),
            "max_num_batched_tokens": int(scheduler.max_num_batched_tokens),
            "async_scheduling": bool(scheduler.async_scheduling),
        },
        "live_parallel": {
            "world_size": int(parallel.world_size),
            "tensor_parallel_size": int(parallel.tensor_parallel_size),
            "data_parallel_size": int(parallel.data_parallel_size),
        },
        "live_cache": {
            key: _normalized(getattr(cache, key, None))
            for key in (
                "num_gpu_blocks",
                "block_size",
                "kv_cache_size_tokens",
                "kv_cache_max_concurrency",
                "enable_prefix_caching",
                "mamba_cache_mode",
                "mamba_block_size",
            )
        },
        "runtime": runner.runtime_metadata(),
        "invocations": invocation_capacity,
    }
    path = RAW / "live_preflight.json"
    if path.exists():
        if path.is_symlink() or read_json(path) != preflight:
            raise RuntimeError("live preflight differs from resume receipt")
    else:
        write_exclusive(path, preflight)
    _validate_recorded_live_preflight(preflight, config, receipt, lock_path)
    return preflight


def _artifact_paths(name: str) -> dict[str, Path]:
    return {
        "started": RAW / f"{name}.started.json",
        "raw": RAW / f"{name}.jsonl",
        "metadata": RAW / f"{name}.metadata.json",
        "complete": RAW / f"{name}.complete.json",
    }


def _validate_raw_inventory() -> None:
    if not RAW.exists():
        return
    if RAW.is_symlink() or not RAW.is_dir():
        raise RuntimeError("raw mechanics directory is unsafe")
    allowed = {"run.lock", "live_preflight.json", "authentication_receipt.json"}
    for name in INVOCATIONS:
        allowed.update(path.name for path in _artifact_paths(name).values())
    unknown = sorted(path.name for path in RAW.iterdir() if path.name not in allowed)
    if unknown or any(path.is_dir() or path.is_symlink() for path in RAW.iterdir()):
        raise RuntimeError(f"unsafe or unknown raw mechanics inventory: {unknown}")


@contextmanager
def _run_lock():
    RAW.mkdir(parents=True, exist_ok=True)
    if RAW.is_symlink() or not RAW.is_dir():
        raise RuntimeError("raw mechanics directory is unsafe")
    handle = (RAW / "run.lock").open("a+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another mechanics process holds the run lock") from exc
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _started_receipt(name: str, lock_path: Path) -> dict[str, Any]:
    prepared_path = PREPARED / f"{name}_requests.jsonl"
    return {
        "schema_version": 1,
        "invocation": name,
        "prepared_sha256": sha256_file(prepared_path),
        "prepared_rows": len(read_jsonl(prepared_path)),
        "implementation_lock_sha256": sha256_file(lock_path),
        "live_preflight_sha256": sha256_file(RAW / "live_preflight.json"),
        "status": "STARTED",
    }


def _completion_receipt(name: str, lock_path: Path) -> dict[str, Any]:
    paths = _artifact_paths(name)
    index = INVOCATIONS.index(name)
    previous = (
        sha256_file(_artifact_paths(INVOCATIONS[index - 1])["complete"])
        if index > 0
        else sha256_file(RAW / "live_preflight.json")
    )
    return {
        "schema_version": 1,
        "invocation": name,
        "prepared_sha256": sha256_file(PREPARED / f"{name}_requests.jsonl"),
        "raw_sha256": sha256_file(paths["raw"]),
        "metadata_sha256": sha256_file(paths["metadata"]),
        "started_sha256": sha256_file(paths["started"]),
        "raw_rows": len(read_jsonl(paths["raw"])),
        "implementation_lock_sha256": sha256_file(lock_path),
        "live_preflight_sha256": sha256_file(RAW / "live_preflight.json"),
        "previous_chain_sha256": previous,
        "status": "COMPLETE",
    }


def _budget_branch_contract(
    stage1_trimmed: list[int], stage1_finish_reason: Any
) -> dict[str, Any]:
    close_index = (
        stage1_trimmed.index(248069) if 248069 in stage1_trimmed else None
    )
    natural = close_index is not None and stage1_finish_reason == "stop"
    retained = (
        stage1_trimmed[:close_index]
        if close_index is not None
        else list(stage1_trimmed)
    )
    return {
        "natural": natural,
        "retained_thinking_token_ids": retained,
        "forced_close": close_index is None,
        "close_index": close_index,
    }


def _authenticate_invocation(
    name: str,
    raw_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    prepared_rows: list[dict[str, Any]],
    config: dict[str, Any],
    prepare_receipt: dict[str, Any],
    preflight: dict[str, Any],
) -> None:
    if len(raw_rows) != EXPECTED_COUNTS[name] or len(raw_rows) != len(prepared_rows):
        raise RuntimeError(f"raw row count changed: {name}")
    token_ids = prepare_receipt["tokenizer"]["plain_alias_token_ids"]
    sampling = _sampling(name, config, token_ids)
    expected_sampling = _normalized(dataclasses.asdict(sampling))
    runtime = metadata.get("runtime")
    counts = metadata.get("counts")
    _validate_invocation_runtime(
        runtime, preflight, config, source=f"runner metadata {name}"
    )
    if (
        metadata.get("schema_version") != 4
        or metadata.get("model") != MODEL_ID
        or metadata.get("model_revision") != MODEL_REVISION
        or metadata.get("runner_sha256")
        != sha256_file(EXP / "src" / "vllm_runner.py")
        or _normalized(metadata.get("engine"))
        != _normalized(dataclasses.asdict(_engine_config(config)))
        or _normalized(metadata.get("engine_args"))
        != _normalized(_expected_engine_args(config))
        or metadata.get("resolved_cudagraph") != preflight["resolved_cudagraph"]
        or metadata.get("resolved_logprobs_mode") != "raw_logprobs"
        or _normalized(metadata.get("sampling")) != expected_sampling
        or _normalized(metadata.get("resolved_sampling"))
        != _normalized(sampling.resolved_sampling())
        or metadata.get("adapter") is not None
        or not isinstance(counts, dict)
        or counts.get("requests") != EXPECTED_COUNTS[name]
        or counts.get("completions") != EXPECTED_COUNTS[name]
    ):
        raise RuntimeError(f"runner metadata authentication failed: {name}")
    if metadata.get("think_token_ids") != {
        "open": prepare_receipt["tokenizer"]["think_open_token_ids"][0],
        "close": prepare_receipt["tokenizer"]["think_close_token_ids"][0],
        "forced_close_sequence": prepare_receipt["tokenizer"][
            "forced_close_token_ids"
        ],
        "thinking_prompt_suffix": prepare_receipt["tokenizer"][
            "thinking_prompt_suffix_ids"
        ],
        "no_thinking_prompt_suffix": prepare_receipt["tokenizer"][
            "no_thinking_prompt_suffix_ids"
        ],
    }:
        raise RuntimeError(f"runner thinking-token authentication failed: {name}")
    if metadata.get("termination") != {
        "hf_model_eos_token_id": 248044,
        "vllm_tokenizer_eos_ignored": 248044,
    }:
        raise RuntimeError(f"runner termination authentication failed: {name}")
    if metadata.get("rng_isolation") != {
        "engine_seed": 0,
        "caller_global_rng_state_restored": True,
    }:
        raise RuntimeError(f"runner RNG authentication failed: {name}")
    expected_ids = [row["id"] for row in prepared_rows]
    if [row.get("id") for row in raw_rows] != expected_ids:
        raise RuntimeError(f"raw row IDs/order changed: {name}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True, use_fast=True
    )
    for raw, prepared in zip(raw_rows, prepared_rows, strict=True):
        if raw.get("meta") != prepared["meta"] or len(raw.get("outputs", [])) != 1:
            raise RuntimeError(f"raw row metadata/output shape changed: {name}")
        rendered = tokenizer.apply_chat_template(
            prepared["messages"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=sampling.thinking != "off",
        )
        prompt_ids = tokenizer.encode(rendered, add_special_tokens=False)
        if (
            raw.get("prompt_sha256") != sha256_bytes(rendered.encode("utf-8"))
            or raw.get("n_prompt_tokens") != len(prompt_ids)
            or raw.get("prompt_channel") != ("off" if sampling.thinking == "off" else "thinking")
            or raw.get("prompt_logprobs") is not None
        ):
            raise RuntimeError(f"raw prompt authentication failed: {name}")
        output = raw["outputs"][0]
        expected_stage1 = runner_stable_seed(sampling.run_seed, raw["id"], -1, "stage1")
        if (
            output.get("sample_index") != 0
            or output.get("stage1_parent_seed") != expected_stage1
            or output.get("seed_stage1") != expected_stage1
        ):
            raise RuntimeError(f"raw stage-one seed changed: {name}")
        token_fields = (
            "token_ids",
            "stage1_token_ids",
            "injected_token_ids",
            "stage2_token_ids",
        )
        if any(
            not isinstance(output.get(field), list)
            or any(
                not isinstance(value, int) or isinstance(value, bool)
                for value in output[field]
            )
            for field in token_fields
        ):
            raise RuntimeError(f"raw token schema changed: {name}")
        if output.get("text") != tokenizer.decode(
            output["token_ids"], skip_special_tokens=False
        ):
            raise RuntimeError(f"raw text/token mismatch: {name}")
        stage1_trimmed = list(output["stage1_token_ids"])
        if 248044 in stage1_trimmed:
            stage1_trimmed = stage1_trimmed[: stage1_trimmed.index(248044)]
        stage2_trimmed = list(output["stage2_token_ids"])
        if 248044 in stage2_trimmed:
            stage2_trimmed = stage2_trimmed[: stage2_trimmed.index(248044)]
        expected_terminal_trimmed = (
            len(output["stage1_token_ids"])
            - len(stage1_trimmed)
            + len(output["stage2_token_ids"])
            - len(stage2_trimmed)
        )
        if (
            output.get("n_sampled_tokens")
            != len(output["stage1_token_ids"]) + len(output["stage2_token_ids"])
            or output.get("n_injected_tokens") != len(output["injected_token_ids"])
            or output.get("n_completion_tokens") != len(output["token_ids"])
            or output.get("n_terminal_tokens_trimmed") != expected_terminal_trimmed
            or output.get("n_stage1_prompt_tokens") != len(prompt_ids)
        ):
            raise RuntimeError(f"raw sampled-token accounting changed: {name}")
        if sampling.thinking == "off":
            if (
                output["token_ids"] != stage1_trimmed
                or len(output["stage1_token_ids"]) != 1
                or output["injected_token_ids"]
                or output["stage2_token_ids"]
                or output.get("seed_stage2") is not None
                or output.get("stage2_logprobs") is not None
                or output.get("n_thinking_tokens") != 0
                or output.get("n_answer_tokens") != len(output["token_ids"])
                or output.get("n_stage2_prompt_tokens") != 0
                or output.get("forced_close") is not False
            ):
                raise RuntimeError(f"ranking request unexpectedly used stage two: {name}")
        else:
            expected_stage2 = runner_stable_seed(sampling.run_seed, raw["id"], 0, "stage2")
            if (
                len(output["stage1_token_ids"]) > int(sampling.thinking_budget)
                or len(output["stage2_token_ids"]) > sampling.answer_max_tokens
                or output.get("stage1_logprobs") is not None
                or output.get("stage2_logprobs") is not None
                or output.get("truncated")
                is not (output.get("finish_reason") == "length")
            ):
                raise RuntimeError(f"budgeted-generation accounting changed: {name}")
            if output.get("seed_stage2") not in {None, expected_stage2}:
                raise RuntimeError(f"raw stage-two seed changed: {name}")
            branch = _budget_branch_contract(
                stage1_trimmed, output.get("stage1_finish_reason")
            )
            if (output.get("seed_stage2") is None) is not branch["natural"]:
                raise RuntimeError(f"natural/continuation branch changed: {name}")
            if output.get("seed_stage2") is None:
                if (
                    output["token_ids"] != stage1_trimmed
                    or output["injected_token_ids"]
                    or output["stage2_token_ids"]
                    or output.get("n_stage2_prompt_tokens") != 0
                    or output.get("thinking_closed") is not True
                    or output.get("forced_close") is not False
                    or output.get("stage2_logprobs") is not None
                    or output.get("stage1_finish_reason") != "stop"
                    or output.get("finish_reason") != "stop"
                ):
                    raise RuntimeError(f"natural-close accounting changed: {name}")
                close_index = branch["close_index"]
                if (
                    output.get("n_thinking_tokens") != close_index
                    or output.get("n_answer_tokens")
                    != len(output["token_ids"]) - close_index - 1
                ):
                    raise RuntimeError(f"natural-close token split changed: {name}")
            else:
                retained = output.get("retained_thinking_token_ids")
                close_ids = prepare_receipt["tokenizer"]["forced_close_token_ids"]
                if (
                    not isinstance(retained, list)
                    or any(
                        not isinstance(value, int) or isinstance(value, bool)
                        for value in retained
                    )
                    or output["injected_token_ids"] != close_ids
                    or output["token_ids"] != retained + close_ids + stage2_trimmed
                    or output.get("n_thinking_tokens") != len(retained)
                    or output.get("n_answer_tokens") != len(stage2_trimmed)
                    or output.get("n_stage2_prompt_tokens")
                    != len(prompt_ids) + len(retained) + len(close_ids)
                    or output.get("thinking_closed") is not True
                ):
                    raise RuntimeError(f"forced-continuation accounting changed: {name}")
                if output.get("forced_close") is not branch["forced_close"]:
                    raise RuntimeError(f"forced-close marker changed: {name}")
                if retained != branch["retained_thinking_token_ids"]:
                    raise RuntimeError(f"retained thinking changed: {name}")
    expected_counts = {
        "requests": len(raw_rows),
        "completions": len(raw_rows),
        "unique_input_prompt_tokens": sum(row["n_prompt_tokens"] for row in raw_rows),
        "stage1_logical_prompt_tokens": sum(
            row["outputs"][0]["n_stage1_prompt_tokens"] for row in raw_rows
        ),
        "stage2_logical_prompt_tokens": sum(
            row["outputs"][0]["n_stage2_prompt_tokens"] for row in raw_rows
        ),
        "sampled_tokens": sum(
            row["outputs"][0]["n_sampled_tokens"] for row in raw_rows
        ),
        "injected_tokens": sum(
            row["outputs"][0]["n_injected_tokens"] for row in raw_rows
        ),
    }
    expected_counts["logical_model_input_tokens"] = (
        expected_counts["stage1_logical_prompt_tokens"]
        + expected_counts["stage2_logical_prompt_tokens"]
    )
    if counts != expected_counts:
        raise RuntimeError(f"runner aggregate token accounting changed: {name}")
    if name in BINARY_ARMS:
        for row in raw_rows:
            binary_rank_score(
                row["outputs"][0],
                live_alias=row["meta"]["viability_live_alias"],
                token_ids={alias: token_ids[alias] for alias in ("A", "B")},
            )
    elif name == "listwise":
        for row in raw_rows:
            listwise_rank_scores(row["outputs"][0], token_ids=token_ids)


def _load_completed(
    name: str,
    prepared_rows: list[dict[str, Any]],
    config: dict[str, Any],
    prepare_receipt: dict[str, Any],
    preflight: dict[str, Any] | None,
    lock_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    paths = _artifact_paths(name)
    exists = {key: path.exists() for key, path in paths.items()}
    if not any(exists.values()):
        return None
    recoverable = (
        exists == {"started": True, "raw": True, "metadata": True, "complete": False}
    )
    if not all(exists.values()) and not recoverable:
        raise RuntimeError(
            f"ambiguous STARTED transaction for {name}; it is terminal and must never be deleted or resampled"
        )
    if preflight is None:
        raise RuntimeError(f"completed invocation lacks a live preflight: {name}")
    if read_json(paths["started"]) != _started_receipt(name, lock_path):
        raise RuntimeError(f"started receipt changed: {name}")
    raw_rows = read_jsonl(paths["raw"])
    metadata = read_json(paths["metadata"])
    _authenticate_invocation(
        name,
        raw_rows,
        metadata,
        prepared_rows,
        config,
        prepare_receipt,
        preflight,
    )
    completion = _completion_receipt(name, lock_path)
    if recoverable:
        write_exclusive(paths["complete"], completion)
    elif read_json(paths["complete"]) != completion:
        raise RuntimeError(f"completion receipt changed: {name}")
    return raw_rows, metadata


def _generate(
    name: str,
    runner: VLLMRunner,
    prepared_rows: list[dict[str, Any]],
    config: dict[str, Any],
    prepare_receipt: dict[str, Any],
    preflight: dict[str, Any],
    lock_path: Path,
) -> None:
    paths = _artifact_paths(name)
    if any(path.exists() for path in paths.values()):
        raise RuntimeError(f"invocation is not pending: {name}")
    write_exclusive(paths["started"], _started_receipt(name, lock_path))
    sampling = _sampling(name, config, prepare_receipt["tokenizer"]["plain_alias_token_ids"])
    rows, metadata = runner.generate(prepared_rows, sampling)
    _authenticate_invocation(
        name,
        rows,
        metadata,
        prepared_rows,
        config,
        prepare_receipt,
        preflight,
    )
    write_exclusive(paths["raw"], rows, jsonl=True)
    write_exclusive(paths["metadata"], metadata)
    write_exclusive(paths["complete"], _completion_receipt(name, lock_path))


def _score_ranking_rows(
    name: str,
    raw_rows: list[dict[str, Any]],
    token_ids: dict[str, int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if name in BINARY_ARMS:
        for raw in raw_rows:
            score, values = binary_rank_score(
                raw["outputs"][0],
                live_alias=raw["meta"]["viability_live_alias"],
                token_ids={alias: token_ids[alias] for alias in ("A", "B")},
            )
            rows.append(
                {
                    "task_id": raw["meta"]["task_id"],
                    "candidate_alias": raw["meta"]["candidate_alias"],
                    "score": score,
                    "requested_raw_logprobs": values,
                    "public_live": raw["meta"]["public_live"],
                }
            )
    elif name == "listwise":
        for raw in raw_rows:
            scores, values = listwise_rank_scores(
                raw["outputs"][0], token_ids=token_ids
            )
            rows.extend(
                {
                    "task_id": raw["meta"]["task_id"],
                    "candidate_alias": alias,
                    "score": scores[alias],
                    "requested_raw_logprob": values[alias],
                }
                for alias in ALIASES
            )
    else:
        raise ValueError("not a ranking invocation")
    return rows


def _authenticate_paired_arms(
    authenticated: dict[str, list[dict[str, Any]]]
) -> None:
    for arms in (SUFFIX_ARMS, BINARY_ARMS):
        reference = authenticated[arms[0]]
        reference_ids = [row["id"] for row in reference]
        reference_seed_keys = [row["meta"]["seed_key"] for row in reference]
        reference_stage1 = [
            (
                row["outputs"][0]["stage1_parent_seed"],
                row["outputs"][0]["seed_stage1"],
            )
            for row in reference
        ]
        for name in arms[1:]:
            rows = authenticated[name]
            if (
                [row["id"] for row in rows] != reference_ids
                or [row["meta"]["seed_key"] for row in rows]
                != reference_seed_keys
                or [
                    (
                        row["outputs"][0]["stage1_parent_seed"],
                        row["outputs"][0]["seed_stage1"],
                    )
                    for row in rows
                ]
                != reference_stage1
            ):
                raise RuntimeError(f"paired causal-arm seed contract changed: {name}")


def _analyze_locked(lock_path: Path) -> dict[str, Any]:
    verify_implementation_lock(lock_path)
    config = yaml.safe_load(CONFIG_PATH.read_text())
    _validate_current_scientific_environment(config)
    prepare_receipt, prepared = _load_prepared()
    _validate_raw_inventory()
    preflight = _load_recorded_live_preflight(config, prepare_receipt, lock_path)
    if preflight is None:
        raise RuntimeError("analysis requires an authenticated live preflight")
    authenticated: dict[str, list[dict[str, Any]]] = {}
    completion_hashes: dict[str, str] = {}
    for name in INVOCATIONS:
        completed = _load_completed(
            name, prepared[name], config, prepare_receipt, preflight, lock_path
        )
        if completed is None:
            raise RuntimeError(f"analysis is missing invocation: {name}")
        authenticated[name] = completed[0]
        completion_hashes[name] = sha256_file(_artifact_paths(name)["complete"])
    _authenticate_paired_arms(authenticated)
    authentication = {
        "schema_version": 1,
        "decision": "MECHANICS_AUTHENTICATION_PASS",
        "implementation_lock_sha256": sha256_file(lock_path),
        "prepare_receipt_sha256": sha256_file(PREPARED / "preoutcome_receipt.json"),
        "live_preflight_sha256": sha256_file(RAW / "live_preflight.json"),
        "complete_receipts": completion_hashes,
        "paired_suffix_ids_order_stage1_seeds": True,
        "paired_binary_ids_order_stage1_seeds": True,
        "authenticated_model_ranking_score_rows": 2304,
        "authenticated_requested_raw_logprob_values": 4032,
    }
    authentication_path = RAW / "authentication_receipt.json"
    if authentication_path.exists():
        if read_json(authentication_path) != authentication:
            raise RuntimeError("mechanics authentication receipt changed")
    else:
        write_exclusive(authentication_path, authentication)

    public_rows = read_jsonl(PUBLIC_PATH)
    audit_rows = read_jsonl(AUDIT_PATH)
    public_by_id = {row["task_id"]: row for row in public_rows}
    generation_metrics: dict[str, Any] = {}
    for name in (*SUFFIX_ARMS, "direct"):
        answer_cap = int(
            config["generation"][
                "direct_answer_max_tokens" if name == "direct" else "suffix_answer_max_tokens"
            ]
        )
        scored_rows, metrics = score_generation_arm(
            public_by_id,
            authenticated[name],
            answer_cap=answer_cap,
            direct=name == "direct",
        )
        write_frozen(SCORED / f"{name}.jsonl", scored_rows, jsonl=True)
        generation_metrics[name] = metrics

    token_ids = prepare_receipt["tokenizer"]["plain_alias_token_ids"]
    ranking_score_rows: dict[str, list[dict[str, Any]]] = {}
    ranking_results: dict[str, Any] = {}
    for name in (*BINARY_ARMS, "listwise"):
        rows = _score_ranking_rows(name, authenticated[name], token_ids)
        write_frozen(SCORED / f"{name}.jsonl", rows, jsonl=True)
        ranking_score_rows[name] = rows
        ranking_results[name] = ranking_metrics(rows, audit_rows)
    for name, path in (
        ("surface", PREPARED / "surface_scores.jsonl"),
        ("random", PREPARED / "random_scores.jsonl"),
    ):
        rows = read_jsonl(path)
        ranking_score_rows[name] = rows
        ranking_results[name] = ranking_metrics(rows, audit_rows)

    mechanics_a = decide_mechanics_a(generation_metrics, config["mechanics"])
    mechanics_b = decide_mechanics_b(
        ranking_results,
        config["mechanics"],
        registered_top4_static_context_fit=prepare_receipt[
            "registered_top4_static_context_fit"
        ],
    )
    mechanics_a_decision = mechanics_a["decision"]
    mechanics_b_decision = mechanics_b["decision"]
    authorization = mechanics_authorization(
        mechanics_a_decision, mechanics_b_decision
    )
    summary = {
        "schema_version": 1,
        "stage": "mechanics",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "mechanics_a_decision": mechanics_a_decision,
        "mechanics_b_decision": mechanics_b_decision,
        **authorization,
        "generation_metrics": generation_metrics,
        "ranking_metrics": ranking_results,
        "mechanics_a_gate": mechanics_a,
        "mechanics_b_gate": mechanics_b,
        "ranking_recall_at_4_gains": mechanics_b["recall_at_4_gains"],
        "ranking_gate_pass": mechanics_b["pass"],
        "authentication_receipt_sha256": sha256_file(authentication_path),
        "implementation_lock_sha256": sha256_file(lock_path),
        "prepare_receipt_sha256": sha256_file(PREPARED / "preoutcome_receipt.json"),
        "hidden_files_read": [],
        "qualification_files_read": [],
        "confirmation_files_read": [],
        "benchmark_files_read": [],
    }
    write_frozen(SUMMARY, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def analyze(lock_path: Path) -> dict[str, Any]:
    with _run_lock():
        return _analyze_locked(lock_path)


def run_live(lock_path: Path) -> dict[str, Any]:
    verify_implementation_lock(lock_path)
    config = yaml.safe_load(CONFIG_PATH.read_text())
    _validate_current_scientific_environment(config)
    prepare_receipt, prepared = _load_prepared()
    _validate_raw_inventory()
    with _run_lock():
        preflight = _load_recorded_live_preflight(config, prepare_receipt, lock_path)
        completed: set[str] = set()
        for name in INVOCATIONS:
            loaded = _load_completed(
                name,
                prepared[name],
                config,
                prepare_receipt,
                preflight,
                lock_path,
            )
            if loaded is not None:
                completed.add(name)
        if completed != set(INVOCATIONS):
            with VLLMRunner(_engine_config(config)) as runner:
                preflight = _live_preflight(
                    runner, config, prepared, prepare_receipt, lock_path
                )
                for name in INVOCATIONS:
                    if name not in completed:
                        _generate(
                            name,
                            runner,
                            prepared[name],
                            config,
                            prepare_receipt,
                            preflight,
                            lock_path,
                        )
        return _analyze_locked(lock_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--stage", required=True, choices=("prepare", "lock", "run", "analyze")
    )
    parser.add_argument("--lock", type=Path, default=IMPLEMENTATION_LOCK)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
        return 0
    if args.stage == "lock":
        publish_implementation_lock()
        return 0
    lock_path = args.lock.resolve()
    if args.stage == "run":
        run_live(lock_path)
    else:
        analyze(lock_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
