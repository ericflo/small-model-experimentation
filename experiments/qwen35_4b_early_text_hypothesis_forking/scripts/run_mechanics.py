#!/usr/bin/env python3
"""Prepare, run, and score the frozen early-hypothesis mechanics gate."""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import random
import subprocess
import sys
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from protocol import (  # noqa: E402
    candidate_injection,
    helper_menu,
    parse_program,
    parse_result,
    score_candidate,
    task_prompt,
)
from task_data import (  # noqa: E402
    CONCRETE_OPERATIONS,
    BoundOperation,
    apply_pipeline,
    canonical_operation,
    make_task,
    public_task,
)
from vllm_runner import (  # noqa: E402
    EngineConfig,
    MODEL_ID,
    MODEL_REVISION,
    RUNNER_SCHEMA_VERSION,
    SamplingConfig,
    VLLMRunner,
    _stable_seed as _runner_stable_seed,
    _token_ids_sha256,
)


CONFIG_PATH = EXP / "configs" / "default.yaml"
PREPARED = EXP / "runs" / "mechanics" / "prepared"
RAW = EXP / "runs" / "mechanics" / "raw"
SCORED = EXP / "runs" / "mechanics" / "scored"
SUMMARY = EXP / "runs" / "mechanics" / "summary.json"
ARMS = ("systematic", "deranged", "duplicate", "placebo")
PROGRAM_ARM = "program_ceiling"
INVOCATIONS = (*ARMS, PROGRAM_ARM)
MECHANICS_PROGRAM_FIRST_OPERATIONS: tuple[BoundOperation, ...] = (
    ("reverse", None),
    ("running_sum", None),
    ("adjacent_diff", None),
    ("sort_desc", None),
    ("add_k", -2),
    ("mul_k", 3),
    ("take_k", 3),
    ("rotate_k", 2),
)
EXP_REL = EXP.relative_to(ROOT)
IMPLEMENTATION_CRITICAL_FILES = frozenset(
    {"requirements-vllm.lock.txt"}
    | {
        str(EXP_REL / relative)
        for relative in (
            "configs/default.yaml",
            "scripts/run_mechanics.py",
            "src/protocol.py",
            "src/task_data.py",
            "src/vllm_runner.py",
            "tests/test_mechanics_runner.py",
            "tests/test_protocol.py",
            "tests/test_vllm_runner.py",
            "reports/preregistration.md",
            "reports/preregistration_amendment_1.md",
            "reports/design_review.md",
            "reports/implementation_review.md",
            "runs/mechanics/prepared/systematic_requests.jsonl",
            "runs/mechanics/prepared/deranged_requests.jsonl",
            "runs/mechanics/prepared/duplicate_requests.jsonl",
            "runs/mechanics/prepared/placebo_requests.jsonl",
            "runs/mechanics/prepared/program_ceiling_requests.jsonl",
            "runs/mechanics/prepared/program_ceiling_public.jsonl",
            "runs/mechanics/prepared/preoutcome_receipt.json",
        )
    }
)
ALLOWED_LIVE_DIR = str((EXP_REL / "runs" / "mechanics").as_posix()) + "/"
PLACEBO_26 = (
    "\nHypothesis fork — provisional; test it against every example and revise "
    "if contradicted.\n"
    "Concrete first operation: unknown\n"
)
RAW_ROW_KEYS = frozenset(
    {
        "id",
        "meta",
        "prompt_sha256",
        "prompt_token_ids_sha256",
        "n_prompt_tokens",
        "prompt_channel",
        "prompt_logprobs",
        "outputs",
    }
)
OUTPUT_COMMON_KEYS = frozenset(
    {
        "sample_index",
        "stage1_parent_seed",
        "seed_stage1",
        "seed_stage2",
        "text",
        "token_ids",
        "stage1_token_ids",
        "injected_token_ids",
        "stage2_token_ids",
        "n_thinking_tokens",
        "n_answer_tokens",
        "n_sampled_tokens",
        "n_injected_tokens",
        "n_completion_tokens",
        "n_terminal_tokens_trimmed",
        "n_stage1_prompt_tokens",
        "n_stage2_prompt_tokens",
        "thinking_closed",
        "forced_close",
        "finish_reason",
        "stop_reason",
        "stage1_finish_reason",
        "stage1_stop_reason",
        "truncated",
        "stage1_cumulative_logprob",
        "stage2_cumulative_logprob",
        "sampled_cumulative_logprob",
        "stage1_logprobs",
        "stage2_logprobs",
    }
)
_AUDIT_TOKENIZER: Any | None = None


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_hash(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    )


def _stable_seed(base: int, domain: str, key: str) -> int:
    value = f"{base}\0{domain}\0{key}".encode()
    return int.from_bytes(hashlib.blake2b(value, digest_size=8).digest(), "big")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_bytes(*arguments: str) -> bytes:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout


def _validate_config(
    config: dict[str, Any], *, require_backend_package: bool = False
) -> None:
    if config["model"]["id"] != MODEL_ID or config["model"]["revision"] != MODEL_REVISION:
        raise RuntimeError("the pinned Qwen3.5-4B identity changed")
    if config["model"]["backend"] != "vllm" or config["model"]["dtype"] != "bfloat16":
        raise RuntimeError("mechanics requires the pinned vLLM bf16 backend")
    if len(CONCRETE_OPERATIONS) != 24:
        raise RuntimeError("mechanics requires exactly 24 bound candidates")
    if len(config["mechanics"]["diagnostic_inputs"]) != 4:
        raise RuntimeError("mechanics requires exactly four diagnostic contexts")
    if int(config["data"]["mechanics_tasks"]) != len(
        MECHANICS_PROGRAM_FIRST_OPERATIONS
    ):
        raise RuntimeError("mechanics program ceiling geometry changed")
    if require_backend_package:
        _validate_locked_environment(config)


def mechanics_user_prompt(values: list[int]) -> str:
    """Public prompt with no singled-out operation identity."""

    if not isinstance(values, list) or any(
        not isinstance(value, int) or isinstance(value, bool) for value in values
    ):
        raise ValueError("mechanics input must be an integer list")
    return (
        "At the beginning of private reasoning you will receive one provisional "
        "concrete first-operation hypothesis. Apply exactly that one supplied "
        "operation to the input. Do not apply a second operation.\n\n"
        "Public operation menu:\n"
        f"{helper_menu()}\n\n"
        f"Input xs: {values!r}\n"
        "Reason privately, then give exactly one final line and no other final "
        "text: RESULT: [comma-separated integers]"
    )


def _render_thinking_prompt(tokenizer: Any, user_prompt: str) -> tuple[str, list[int]]:
    messages = [{"role": "user", "content": user_prompt}]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    if not isinstance(rendered, str):
        raise RuntimeError("chat template did not return text")
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    suffix = tokenizer.encode(
        "<|im_start|>assistant\n<think>\n", add_special_tokens=False
    )
    if ids[-len(suffix) :] != suffix:
        raise RuntimeError("base prompt does not end inside the thinking channel")
    return rendered, list(ids)


def _derangement(
    operations: list[BoundOperation], *, seed: int
) -> dict[BoundOperation, BoundOperation]:
    rng = random.Random(seed)
    for _ in range(10_000):
        shuffled = list(operations)
        rng.shuffle(shuffled)
        if all(left != right for left, right in zip(operations, shuffled, strict=True)):
            return dict(zip(operations, shuffled, strict=True))
    raise RuntimeError("failed to construct a frozen derangement")


def _build_injection_controls(
    tokenizer: Any, *, context_id: str, seed: int
) -> dict[str, Any]:
    systematic = {
        operation: tokenizer.encode(candidate_injection(operation), add_special_tokens=False)
        for operation in CONCRETE_OPERATIONS
    }
    lengths = {operation: len(ids) for operation, ids in systematic.items()}
    groups: dict[int, list[BoundOperation]] = defaultdict(list)
    for operation, length in lengths.items():
        groups[length].append(operation)
    if sorted(groups) != [26, 27, 30] or sorted(map(len, groups.values())) != [3, 5, 16]:
        raise RuntimeError(f"candidate token geometry changed: {groups!r}")

    deranged: dict[BoundOperation, BoundOperation] = {}
    for length, operations in sorted(groups.items()):
        deranged.update(
            _derangement(
                operations,
                seed=_stable_seed(seed, f"derangement-{length}", context_id),
            )
        )
    if any(lengths[left] != lengths[right] for left, right in deranged.items()):
        raise RuntimeError("derangement is not injection-length matched")

    filler = tokenizer.encode(" anyway", add_special_tokens=False)
    if len(filler) != 1:
        raise RuntimeError("neutral filler is no longer a single token")
    placebo_base = tokenizer.encode(PLACEBO_26, add_special_tokens=False)
    if len(placebo_base) != 26:
        raise RuntimeError("placebo base is no longer 26 tokens")
    min_candidates = groups[26]
    duplicate_operation = min_candidates[
        _stable_seed(seed, "duplicate-operation", context_id) % len(min_candidates)
    ]
    duplicate_base = systematic[duplicate_operation]
    terminal_ids = {ids[-1] for ids in systematic.values()}
    if len(terminal_ids) != 1 or placebo_base[-1] not in terminal_ids:
        raise RuntimeError("control injection terminal-token geometry changed")

    def pad_before_terminal(base: list[int], target_length: int) -> list[int]:
        if target_length < len(base) or not base:
            raise RuntimeError("invalid control padding geometry")
        return (
            list(base[:-1])
            + filler * (target_length - len(base))
            + list(base[-1:])
        )

    injections: dict[str, dict[BoundOperation, list[int]]] = {
        "systematic": {},
        "deranged": {},
        "duplicate": {},
        "placebo": {},
    }
    supplied: dict[str, dict[BoundOperation, BoundOperation | None]] = {
        arm: {} for arm in ARMS
    }
    for operation in CONCRETE_OPERATIONS:
        target_length = lengths[operation]
        injections["systematic"][operation] = list(systematic[operation])
        supplied["systematic"][operation] = operation
        injections["deranged"][operation] = list(systematic[deranged[operation]])
        supplied["deranged"][operation] = deranged[operation]
        injections["duplicate"][operation] = pad_before_terminal(
            duplicate_base, target_length
        )
        supplied["duplicate"][operation] = duplicate_operation
        injections["placebo"][operation] = pad_before_terminal(
            placebo_base, target_length
        )
        supplied["placebo"][operation] = None
        if any(len(injections[arm][operation]) != target_length for arm in ARMS):
            raise RuntimeError("control injection length mismatch")
        if len({injections[arm][operation][-1] for arm in ARMS}) != 1:
            raise RuntimeError("row-matched control terminal tokens differ")

    return {
        "injections": injections,
        "supplied": supplied,
        "derangement": deranged,
        "duplicate_operation": duplicate_operation,
        "lengths": lengths,
        "filler_token_id": filler[0],
        "terminal_token_id": next(iter(terminal_ids)),
    }


def _mechanics_program_tasks(config: dict[str, Any]) -> list[dict[str, Any]]:
    rng = random.Random(int(config["seeds"]["mechanics"]) + 101)
    tasks = []
    for index, first_operation in enumerate(MECHANICS_PROGRAM_FIRST_OPERATIONS):
        tasks.append(
            make_task(
                rng,
                task_id=f"mechanics-program-{index:05d}",
                first_operation=first_operation,
                visible_count=int(config["data"]["visible_examples"]),
                hidden_count=int(config["data"]["hidden_examples"]),
                probe_count=int(config["data"]["unlabeled_probe_inputs"]),
            )
        )
    return tasks


def prepare_requests() -> dict[str, Any]:
    """Build exact-token requests with AutoTokenizer only; no model engine."""

    from transformers import AutoConfig, AutoTokenizer

    config = yaml.safe_load(CONFIG_PATH.read_text())
    _validate_config(config)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
        use_fast=True,
    )
    model_config = AutoConfig.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True
    )
    eos_id = int(model_config.text_config.eos_token_id)
    close_id = tokenizer.encode("</think>", add_special_tokens=False)
    if eos_id != 248044 or close_id != [248069]:
        raise RuntimeError("Qwen termination token identity changed")

    requests: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARMS}
    context_receipts = []
    base_prompt_hashes: dict[str, str] = {}
    derangement_hashes: set[str] = set()
    seed = int(config["seeds"]["mechanics"])
    for context_index, values in enumerate(config["mechanics"]["diagnostic_inputs"]):
        context_id = f"mechanics-{context_index:05d}"
        user_prompt = mechanics_user_prompt(values)
        rendered, base_ids = _render_thinking_prompt(tokenizer, user_prompt)
        base_prompt_hashes[context_id] = _sha256_bytes(rendered.encode())
        controls = _build_injection_controls(
            tokenizer, context_id=context_id, seed=seed
        )
        serialized_derangement = {
            canonical_operation(left): canonical_operation(right)
            for left, right in controls["derangement"].items()
        }
        derangement_hashes.add(_canonical_hash(serialized_derangement))

        for slot, registered in enumerate(CONCRETE_OPERATIONS):
            expected_registered = apply_pipeline(list(values), [registered])
            record_id = f"{context_id}--slot-{slot:02d}"
            target_length = controls["lengths"][registered]
            for arm in ARMS:
                injection_ids = controls["injections"][arm][registered]
                supplied = controls["supplied"][arm][registered]
                if close_id[0] in injection_ids or eos_id in injection_ids:
                    raise RuntimeError("candidate/control injection contains close or EOS")
                expected_supplied = (
                    apply_pipeline(list(values), [supplied])
                    if supplied is not None
                    else None
                )
                requests[arm].append(
                    {
                        "id": record_id,
                        "prompt_token_ids": base_ids + injection_ids,
                        "meta": {
                            "arm": arm,
                            "context_id": context_id,
                            "slot": slot,
                            "input": values,
                            "registered_operation": canonical_operation(registered),
                            "supplied_operation": (
                                canonical_operation(supplied)
                                if supplied is not None
                                else None
                            ),
                            "expected_registered": expected_registered,
                            "expected_supplied": expected_supplied,
                            "base_prompt_token_count": len(base_ids),
                            "base_prompt_sha256": base_prompt_hashes[context_id],
                            "injection_token_count": len(injection_ids),
                            "injection_token_ids_sha256": _canonical_hash(injection_ids),
                            "systematic_injection_token_count": target_length,
                        },
                    }
                )
        context_receipts.append(
            {
                "context_id": context_id,
                "base_prompt_sha256": base_prompt_hashes[context_id],
                "derangement": serialized_derangement,
                "derangement_sha256": _canonical_hash(serialized_derangement),
                "duplicate_operation": canonical_operation(
                    controls["duplicate_operation"]
                ),
                "injection_lengths": Counter(controls["lengths"].values()),
                "filler_token_id": controls["filler_token_id"],
                "terminal_injection_token_id": controls["terminal_token_id"],
            }
        )
    if len(derangement_hashes) != 4:
        raise RuntimeError("derangement composition does not vary across contexts")

    program_tasks = _mechanics_program_tasks(config)
    program_requests = []
    program_public_rows = []
    for index, task in enumerate(program_tasks):
        public = public_task(task)
        registered = MECHANICS_PROGRAM_FIRST_OPERATIONS[index]
        user_prompt = task_prompt(public)
        rendered, base_ids = _render_thinking_prompt(tokenizer, user_prompt)
        injection_ids = tokenizer.encode(
            candidate_injection(registered), add_special_tokens=False
        )
        if close_id[0] in injection_ids or eos_id in injection_ids:
            raise RuntimeError("program candidate injection contains close or EOS")
        program_requests.append(
            {
                "id": task["task_id"],
                "prompt_token_ids": base_ids + injection_ids,
                "meta": {
                    "arm": "program_ceiling",
                    "task_id": task["task_id"],
                    "registered_operation": canonical_operation(registered),
                    "public_task": public,
                    "base_prompt_sha256": _sha256_bytes(rendered.encode()),
                    "base_prompt_token_count": len(base_ids),
                    "injection_token_count": len(injection_ids),
                    "injection_token_ids_sha256": _canonical_hash(injection_ids),
                },
            }
        )
        program_public_rows.append(public)

    paths: dict[str, Path] = {}
    for arm in ARMS:
        path = PREPARED / f"{arm}_requests.jsonl"
        _write_jsonl(path, requests[arm])
        paths[arm] = path
    program_path = PREPARED / "program_ceiling_requests.jsonl"
    public_path = PREPARED / "program_ceiling_public.jsonl"
    _write_jsonl(program_path, program_requests)
    _write_jsonl(public_path, program_public_rows)
    paths["program_ceiling"] = program_path
    paths["program_ceiling_public"] = public_path

    for arm in ARMS:
        ids = [row["id"] for row in requests[arm]]
        if ids != [row["id"] for row in requests["systematic"]]:
            raise RuntimeError("mechanics arms do not have row-matched seed IDs")
        if len(requests[arm]) != 96 or len(set(ids)) != 96:
            raise RuntimeError("mechanics request geometry changed")
    for row_group in zip(*(requests[arm] for arm in ARMS), strict=True):
        meta = [row["meta"] for row in row_group]
        base_counts = {value["base_prompt_token_count"] for value in meta}
        base_hashes = {value["base_prompt_sha256"] for value in meta}
        injection_counts = {value["injection_token_count"] for value in meta}
        if len(base_counts) != 1 or len(base_hashes) != 1 or len(injection_counts) != 1:
            raise RuntimeError("row-matched prompt/control geometry differs")

    receipt = {
        "schema_version": 1,
        "stage": "mechanics_prepare",
        "decision": "TOKEN_STITCH_PREPARE_PASS",
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "backend": "vllm",
        "tokenizer_only": True,
        "model_loaded": False,
        "outcomes_loaded": False,
        "benchmarks_read": False,
        "think_open_id": 248068,
        "think_close_id": close_id[0],
        "model_eos_id": eos_id,
        "mechanics_rows_per_arm": 96,
        "program_ceiling_rows": len(program_requests),
        "same_ids_and_seeds_across_arms": True,
        "control_injection_lengths_exactly_matched": True,
        "candidate_only_inside_open_think": True,
        "unique_derangement_compositions": len(derangement_hashes),
        "contexts": context_receipts,
        "files": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": _sha256_file(path),
                "rows": len(_read_jsonl(path)),
            }
            for name, path in paths.items()
        },
    }
    receipt_path = PREPARED / "preoutcome_receipt.json"
    _write_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


def verify_implementation_lock(lock_path: Path) -> dict[str, Any]:
    try:
        lock_relative = lock_path.resolve().relative_to(ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError("implementation lock escapes the repository") from exc
    if lock_path.is_symlink() or not lock_path.is_file():
        raise RuntimeError("live mechanics requires a committed implementation lock")
    lock = json.loads(lock_path.read_text())
    required = {
        "schema_version",
        "design_commit",
        "design_amendment_commit",
        "implementation_commit",
        "critical_files",
        "model_calls_before_lock",
    }
    if not isinstance(lock, dict) or set(lock) != required or lock["schema_version"] != 1:
        raise RuntimeError("implementation lock schema is incomplete")
    if lock["model_calls_before_lock"] != 0:
        raise RuntimeError("implementation lock does not certify a pre-model boundary")
    critical = lock["critical_files"]
    if not isinstance(critical, dict) or set(critical) != IMPLEMENTATION_CRITICAL_FILES:
        raise RuntimeError("implementation lock critical-file allowlist is not exact")

    tracked = _git("ls-files", "--error-unmatch", "--", str(lock_relative))
    if tracked != str(lock_relative):
        raise RuntimeError("implementation lock is not tracked at HEAD")
    subprocess.run(
        ["git", "cat-file", "-e", f"HEAD:{lock_relative}"],
        cwd=ROOT,
        check=True,
    )

    dirty = _git("status", "--porcelain=v1", "--untracked-files=all")
    for line in dirty.splitlines():
        path_text = line[3:]
        paths = path_text.split(" -> ")
        if not all(
            path == str(EXP_REL / "runs" / "mechanics" / "summary.json")
            or path.startswith(ALLOWED_LIVE_DIR + "raw/")
            or path.startswith(ALLOWED_LIVE_DIR + "scored/")
            for path in paths
        ):
            raise RuntimeError(f"live mechanics has unrelated worktree change: {line}")

    config = yaml.safe_load(CONFIG_PATH.read_text())
    if config["boundaries"]["design"] != {
        "status": "locked",
        "commit": lock["design_commit"],
    }:
        raise RuntimeError("design boundary differs from the implementation lock")
    if config["boundaries"]["design_amendment"] != {
        "status": "locked",
        "commit": lock["design_amendment_commit"],
    }:
        raise RuntimeError("design amendment differs from the implementation lock")
    if config["boundaries"]["implementation"].get("status") != "locked":
        raise RuntimeError("implementation boundary is not locked in config")

    design_commit = str(lock["design_commit"])
    design_amendment_commit = str(lock["design_amendment_commit"])
    implementation_commit = str(lock["implementation_commit"])
    subprocess.run(
        ["git", "fetch", "--quiet", "origin", "main"],
        cwd=ROOT,
        check=True,
    )
    for commit in (
        design_commit,
        design_amendment_commit,
        implementation_commit,
        _git("rev-parse", "HEAD"),
    ):
        if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
            raise RuntimeError(f"invalid implementation-boundary commit: {commit!r}")
    ancestry = (
        (design_commit, design_amendment_commit),
        (design_amendment_commit, implementation_commit),
        (implementation_commit, "HEAD"),
        ("HEAD", "origin/main"),
    )
    for ancestor, descendant in ancestry:
        completed = subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=ROOT,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"unpublished or inconsistent boundary: {ancestor} !<= {descendant}"
            )

    for relative, expected in critical.items():
        path = ROOT / relative
        try:
            resolved = path.resolve().relative_to(ROOT.resolve())
        except ValueError as exc:
            raise RuntimeError(f"critical path escapes repository: {relative}") from exc
        if str(resolved) != relative or path.is_symlink() or not path.is_file():
            raise RuntimeError(f"invalid implementation critical path: {relative}")
        if _sha256_file(path) != expected:
            raise RuntimeError(f"implementation lock hash mismatch: {relative}")
        try:
            blob = _git_bytes("show", f"{implementation_commit}:{relative}")
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"critical file absent at implementation commit: {relative}"
            ) from exc
        if _sha256_bytes(blob) != expected:
            raise RuntimeError(f"implementation blob hash mismatch: {relative}")
    return lock


def _sampling(config: dict[str, Any], *, program: bool = False) -> SamplingConfig:
    generation = config["generation"]
    return SamplingConfig(
        thinking="budget",
        thinking_budget=int(
            generation["main_thinking_budget"]
            if program
            else generation["mechanics_thinking_budget"]
        ),
        n=1,
        max_tokens=int(generation["main_thinking_budget"]),
        answer_max_tokens=int(generation["answer_max_tokens"]),
        temperature=float(generation["temperature"]),
        top_p=float(generation["top_p"]),
        top_k=int(generation["top_k"]),
        run_seed=int(config["seeds"]["mechanics"]) + (1 if program else 0),
        allow_custom_prompts=True,
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


def _audit_tokenizer() -> Any:
    global _AUDIT_TOKENIZER
    if _AUDIT_TOKENIZER is None:
        from transformers import AutoTokenizer

        _AUDIT_TOKENIZER = AutoTokenizer.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            trust_remote_code=True,
            use_fast=True,
        )
    return _AUDIT_TOKENIZER


def _decode_token_ids(token_ids: list[int]) -> str:
    return str(_audit_tokenizer().decode(token_ids, skip_special_tokens=False))


def _locked_environment_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for line in (ROOT / "requirements-vllm.lock.txt").read_text().splitlines():
        if "==" not in line or line[:1].isspace() or line.startswith("#"):
            continue
        name, version = line.split("==", 1)
        version = version.split(";", 1)[0].strip()
        if name and version and " " not in name:
            versions[name.lower().replace("_", "-")] = version
    return versions


def _installed_environment_versions() -> dict[str, str]:
    installed: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            installed[name.lower().replace("_", "-")] = distribution.version
    return installed


def _validate_locked_environment(config: dict[str, Any]) -> None:
    expected = {
        **_locked_environment_versions(),
        "vllm": str(config["model"]["vllm_version"]),
    }
    installed = _installed_environment_versions()
    mismatches = {
        package: (version, installed.get(package))
        for package, version in expected.items()
        if installed.get(package) != version
    }
    if mismatches:
        raise RuntimeError(
            f"live environment differs from requirements-vllm.lock.txt: {mismatches}"
        )


def _expected_engine_args(engine: EngineConfig) -> dict[str, Any]:
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
        "enable_prefix_caching": engine.enable_prefix_caching,
        "mamba_cache_mode": "align" if engine.enable_prefix_caching else "none",
        "enforce_eager": engine.enforce_eager,
        "generation_config": "vllm",
        "max_logprobs": 20,
        "seed": 0,
        "async_scheduling": False,
        "cudagraph_capture_sizes": list(engine.cudagraph_capture_sizes or ()),
        "max_cudagraph_capture_size": (
            engine.cudagraph_capture_sizes[-1]
            if engine.cudagraph_capture_sizes
            else engine.max_num_seqs
        ),
    }


def _expected_prepared_paths() -> dict[str, Path]:
    return {
        **{arm: PREPARED / f"{arm}_requests.jsonl" for arm in ARMS},
        PROGRAM_ARM: PREPARED / "program_ceiling_requests.jsonl",
        "program_ceiling_public": PREPARED / "program_ceiling_public.jsonl",
    }


def _validate_prepared_geometry(
    config: dict[str, Any], prepared: dict[str, list[dict[str, Any]]]
) -> None:
    expected_contexts = {
        f"mechanics-{index:05d}" for index in range(len(config["mechanics"]["diagnostic_inputs"]))
    }
    expected_operations = {canonical_operation(operation) for operation in CONCRETE_OPERATIONS}
    reference_ids = [row["id"] for row in prepared["systematic"]]
    if len(reference_ids) != 96 or len(set(reference_ids)) != 96:
        raise RuntimeError("systematic prepared geometry is not 96 unique rows")
    for arm in ARMS:
        rows = prepared[arm]
        if [row.get("id") for row in rows] != reference_ids:
            raise RuntimeError(f"{arm} prepared ID order differs from systematic")
        by_context: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if set(row) != {"id", "prompt_token_ids", "meta"}:
                raise RuntimeError(f"{arm} prepared row schema changed")
            ids = row["prompt_token_ids"]
            meta = row["meta"]
            if meta.get("arm") != arm or not isinstance(ids, list) or not ids:
                raise RuntimeError(f"{arm} prepared arm or prompt is invalid")
            base_count = int(meta["base_prompt_token_count"])
            injection = ids[base_count:]
            if base_count <= 0 or len(injection) != int(meta["injection_token_count"]):
                raise RuntimeError(f"{arm} prepared token split is invalid")
            if _canonical_hash(injection) != meta["injection_token_ids_sha256"]:
                raise RuntimeError(f"{arm} prepared injection hash is invalid")
            if injection[-1] != 198 or 248069 in injection or 248044 in injection:
                raise RuntimeError(f"{arm} prepared injection termination is invalid")
            by_context[str(meta["context_id"])].append(row)
        if set(by_context) != expected_contexts:
            raise RuntimeError(f"{arm} prepared context set changed")
        for context_id, rows in by_context.items():
            slots = [int(row["meta"]["slot"]) for row in rows]
            operations = {row["meta"]["registered_operation"] for row in rows}
            if slots != list(range(24)) or operations != expected_operations:
                raise RuntimeError(f"{arm}/{context_id} prepared slot geometry changed")
    for row_group in zip(*(prepared[arm] for arm in ARMS), strict=True):
        base_counts = {int(row["meta"]["base_prompt_token_count"]) for row in row_group}
        if len(base_counts) != 1:
            raise RuntimeError("row-matched base prompt counts changed")
        base_count = next(iter(base_counts))
        if len({tuple(row["prompt_token_ids"][:base_count]) for row in row_group}) != 1:
            raise RuntimeError("row-matched exact base prompt IDs changed")
        if len({len(row["prompt_token_ids"]) for row in row_group}) != 1:
            raise RuntimeError("row-matched exact prompt lengths changed")
        if len({row["prompt_token_ids"][-1] for row in row_group}) != 1:
            raise RuntimeError("row-matched injection terminal IDs changed")

    program_rows = prepared[PROGRAM_ARM]
    public_rows = prepared["program_ceiling_public"]
    if len(program_rows) != len(MECHANICS_PROGRAM_FIRST_OPERATIONS):
        raise RuntimeError("program-ceiling prepared row count changed")
    if len(public_rows) != len(program_rows):
        raise RuntimeError("program-ceiling public row count changed")
    for index, (row, public) in enumerate(zip(program_rows, public_rows, strict=True)):
        expected_id = f"mechanics-program-{index:05d}"
        meta = row.get("meta", {})
        if row.get("id") != expected_id or meta.get("task_id") != expected_id:
            raise RuntimeError("program-ceiling ID order changed")
        if meta.get("arm") != PROGRAM_ARM or meta.get("public_task") != public:
            raise RuntimeError("program-ceiling public metadata changed")
        if meta.get("registered_operation") != canonical_operation(
            MECHANICS_PROGRAM_FIRST_OPERATIONS[index]
        ):
            raise RuntimeError("program-ceiling first-operation schedule changed")
        if any(key in public for key in ("hidden", "pipeline", "gold")):
            raise RuntimeError("program-ceiling public row contains private task fields")
        ids = row["prompt_token_ids"]
        base_count = int(meta["base_prompt_token_count"])
        injection = ids[base_count:]
        if (
            len(injection) != int(meta["injection_token_count"])
            or _canonical_hash(injection) != meta["injection_token_ids_sha256"]
            or injection[-1] != 198
            or 248069 in injection
            or 248044 in injection
        ):
            raise RuntimeError("program-ceiling injection geometry changed")


def _independently_rebuild_prepared(
    config: dict[str, Any], prepared: dict[str, list[dict[str, Any]]]
) -> None:
    tokenizer = _audit_tokenizer()
    seed = int(config["seeds"]["mechanics"])
    lookup = {
        arm: {
            (row["meta"]["context_id"], int(row["meta"]["slot"])): row
            for row in prepared[arm]
        }
        for arm in ARMS
    }
    for context_index, values in enumerate(config["mechanics"]["diagnostic_inputs"]):
        context_id = f"mechanics-{context_index:05d}"
        rendered, base_ids = _render_thinking_prompt(
            tokenizer, mechanics_user_prompt(values)
        )
        base_hash = _sha256_bytes(rendered.encode())
        controls = _build_injection_controls(
            tokenizer, context_id=context_id, seed=seed
        )
        for slot, registered in enumerate(CONCRETE_OPERATIONS):
            for arm in ARMS:
                supplied = controls["supplied"][arm][registered]
                injection = controls["injections"][arm][registered]
                expected = {
                    "id": f"{context_id}--slot-{slot:02d}",
                    "prompt_token_ids": base_ids + injection,
                    "meta": {
                        "arm": arm,
                        "context_id": context_id,
                        "slot": slot,
                        "input": values,
                        "registered_operation": canonical_operation(registered),
                        "supplied_operation": (
                            canonical_operation(supplied)
                            if supplied is not None
                            else None
                        ),
                        "expected_registered": apply_pipeline(
                            list(values), [registered]
                        ),
                        "expected_supplied": (
                            apply_pipeline(list(values), [supplied])
                            if supplied is not None
                            else None
                        ),
                        "base_prompt_token_count": len(base_ids),
                        "base_prompt_sha256": base_hash,
                        "injection_token_count": len(injection),
                        "injection_token_ids_sha256": _canonical_hash(injection),
                        "systematic_injection_token_count": controls["lengths"][
                            registered
                        ],
                    },
                }
                if lookup[arm][(context_id, slot)] != expected:
                    raise RuntimeError(
                        f"{arm}/{context_id}/{slot} differs from independent prompt rebuild"
                    )

    rebuilt_tasks = _mechanics_program_tasks(config)
    rebuilt_program: list[dict[str, Any]] = []
    rebuilt_public: list[dict[str, Any]] = []
    for index, task in enumerate(rebuilt_tasks):
        public = public_task(task)
        registered = MECHANICS_PROGRAM_FIRST_OPERATIONS[index]
        rendered, base_ids = _render_thinking_prompt(tokenizer, task_prompt(public))
        injection = tokenizer.encode(
            candidate_injection(registered), add_special_tokens=False
        )
        rebuilt_program.append(
            {
                "id": task["task_id"],
                "prompt_token_ids": base_ids + injection,
                "meta": {
                    "arm": PROGRAM_ARM,
                    "task_id": task["task_id"],
                    "registered_operation": canonical_operation(registered),
                    "public_task": public,
                    "base_prompt_sha256": _sha256_bytes(rendered.encode()),
                    "base_prompt_token_count": len(base_ids),
                    "injection_token_count": len(injection),
                    "injection_token_ids_sha256": _canonical_hash(injection),
                },
            }
        )
        rebuilt_public.append(public)
    if (
        prepared[PROGRAM_ARM] != rebuilt_program
        or prepared["program_ceiling_public"] != rebuilt_public
    ):
        raise RuntimeError("program ceiling differs from independent deterministic rebuild")


def _load_and_validate_prepared(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    receipt_path = PREPARED / "preoutcome_receipt.json"
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise RuntimeError("prepare-only receipt is missing or unsafe")
    receipt = json.loads(receipt_path.read_text())
    expected_receipt = {
        "decision": "TOKEN_STITCH_PREPARE_PASS",
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "backend": "vllm",
        "tokenizer_only": True,
        "model_loaded": False,
        "outcomes_loaded": False,
        "benchmarks_read": False,
        "think_open_id": 248068,
        "think_close_id": 248069,
        "model_eos_id": 248044,
        "mechanics_rows_per_arm": 96,
        "program_ceiling_rows": len(MECHANICS_PROGRAM_FIRST_OPERATIONS),
        "same_ids_and_seeds_across_arms": True,
        "control_injection_lengths_exactly_matched": True,
        "candidate_only_inside_open_think": True,
        "unique_derangement_compositions": 4,
    }
    if receipt.get("schema_version") != 1 or receipt.get("stage") != "mechanics_prepare":
        raise RuntimeError("prepare-only receipt schema changed")
    for key, expected in expected_receipt.items():
        if receipt.get(key) != expected:
            raise RuntimeError(f"prepare-only receipt mismatch: {key}")
    paths = _expected_prepared_paths()
    if set(receipt.get("files", {})) != set(paths):
        raise RuntimeError("prepare-only receipt file inventory changed")
    prepared: dict[str, list[dict[str, Any]]] = {}
    for name, path in paths.items():
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"prepared file is missing or unsafe: {name}")
        entry = receipt["files"][name]
        if entry.get("path") != str(path.relative_to(ROOT)):
            raise RuntimeError(f"prepared path receipt changed: {name}")
        if entry.get("sha256") != _sha256_file(path):
            raise RuntimeError(f"prepared file hash changed: {name}")
        rows = _read_jsonl(path)
        if entry.get("rows") != len(rows):
            raise RuntimeError(f"prepared row-count receipt changed: {name}")
        prepared[name] = rows
    _validate_prepared_geometry(config, prepared)
    _independently_rebuild_prepared(config, prepared)
    return receipt, prepared


def _metadata_expected_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "requests": len(rows),
        "completions": sum(len(row["outputs"]) for row in rows),
        "unique_input_prompt_tokens": sum(int(row["n_prompt_tokens"]) for row in rows),
        "stage1_logical_prompt_tokens": sum(
            int(output["n_stage1_prompt_tokens"])
            for row in rows
            for output in row["outputs"]
        ),
        "stage2_logical_prompt_tokens": sum(
            int(output["n_stage2_prompt_tokens"])
            for row in rows
            for output in row["outputs"]
        ),
        "sampled_tokens": sum(
            int(output["n_sampled_tokens"])
            for row in rows
            for output in row["outputs"]
        ),
        "injected_tokens": sum(
            int(output["n_injected_tokens"])
            for row in rows
            for output in row["outputs"]
        ),
    }
    counts["logical_model_input_tokens"] = (
        counts["stage1_logical_prompt_tokens"]
        + counts["stage2_logical_prompt_tokens"]
    )
    return counts


def _authenticate_generation(
    name: str,
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    prepared: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    if name not in INVOCATIONS:
        raise RuntimeError(f"unknown mechanics invocation: {name}")
    if len(rows) != len(prepared) or [row.get("id") for row in rows] != [
        row.get("id") for row in prepared
    ]:
        raise RuntimeError(f"{name} raw ID order/count differs from prepared")
    if len({row["id"] for row in rows}) != len(rows):
        raise RuntimeError(f"{name} raw IDs are not unique")
    program = name == PROGRAM_ARM
    sampling = _sampling(config, program=program)
    engine = _engine_config(config)
    expected_sampling = _normalized(dataclasses.asdict(sampling))
    expected_engine = _normalized(dataclasses.asdict(engine))
    runner_hash = _sha256_file(EXP / "src" / "vllm_runner.py")
    if metadata.get("schema_version") != RUNNER_SCHEMA_VERSION:
        raise RuntimeError(f"{name} runner schema changed")
    if (
        metadata.get("model") != MODEL_ID
        or metadata.get("model_revision") != MODEL_REVISION
        or metadata.get("runner_sha256") != runner_hash
        or metadata.get("adapter") is not None
    ):
        raise RuntimeError(f"{name} model/runner identity changed")
    if _normalized(metadata.get("engine")) != expected_engine:
        raise RuntimeError(f"{name} engine metadata differs from config")
    if _normalized(metadata.get("sampling")) != expected_sampling:
        raise RuntimeError(f"{name} sampling metadata differs from config")
    if _normalized(metadata.get("resolved_sampling")) != _normalized(
        sampling.resolved_sampling()
    ):
        raise RuntimeError(f"{name} resolved sampling differs from config")
    if _normalized(metadata.get("engine_args")) != _normalized(
        _expected_engine_args(engine)
    ):
        raise RuntimeError(f"{name} exact engine arguments changed")
    resolved = metadata.get("resolved_cudagraph", {})
    if (
        resolved.get("cudagraph_capture_sizes") != list(engine.cudagraph_capture_sizes or ())
        or resolved.get("max_cudagraph_capture_size") != engine.max_num_seqs
        or not resolved.get("has_full_cudagraphs")
        or resolved.get("decode_mode") != "FULL"
    ):
        raise RuntimeError(f"{name} resolved CUDA-graph geometry changed")
    termination = metadata.get("termination", {})
    think_ids = metadata.get("think_token_ids", {})
    close_ids = think_ids.get("forced_close_sequence")
    if (
        termination.get("hf_model_eos_token_id") != 248044
        or think_ids.get("open") != 248068
        or think_ids.get("close") != 248069
        or close_ids != [248069, 271]
    ):
        raise RuntimeError(f"{name} termination token identity changed")
    runtime = metadata.get("runtime", {})
    packages = runtime.get("packages", {})
    if packages.get("vllm") != str(config["model"]["vllm_version"]):
        raise RuntimeError(f"{name} runtime vLLM version changed")
    locked_versions = _locked_environment_versions()
    mismatched_packages = {
        package: (version, packages.get(package))
        for package, version in locked_versions.items()
        if packages.get(package) != version
    }
    if mismatched_packages:
        raise RuntimeError(
            f"{name} runtime package set differs from lock: {mismatched_packages}"
        )
    environment_lock = runtime.get("environment_lock") or {}
    pinned_lock = ROOT / "requirements-vllm.lock.txt"
    if environment_lock.get("sha256") != _sha256_file(pinned_lock):
        raise RuntimeError(f"{name} environment-lock hash changed")

    run_seed = int(sampling.run_seed)
    for row, request in zip(rows, prepared, strict=True):
        if set(row) != RAW_ROW_KEYS:
            raise RuntimeError(f"{name}/{row.get('id')} raw row schema changed")
        if row.get("meta") != request.get("meta"):
            raise RuntimeError(f"{name}/{row.get('id')} raw metadata changed")
        prompt_ids = request["prompt_token_ids"]
        if (
            row.get("prompt_token_ids_sha256") != _token_ids_sha256(prompt_ids)
            or row.get("n_prompt_tokens") != len(prompt_ids)
            or row.get("prompt_channel") != "custom"
            or row.get("prompt_sha256")
            != _sha256_bytes(_decode_token_ids(prompt_ids).encode())
            or row.get("prompt_logprobs") is not None
        ):
            raise RuntimeError(f"{name}/{row.get('id')} exact prompt receipt changed")
        outputs = row.get("outputs")
        if not isinstance(outputs, list) or len(outputs) != 1:
            raise RuntimeError(f"{name}/{row.get('id')} requires exactly one output")
        output = outputs[0]
        stage2_seed = output.get("seed_stage2")
        expected_output_keys = OUTPUT_COMMON_KEYS | (
            {"retained_thinking_token_ids"} if stage2_seed is not None else set()
        )
        if set(output) != expected_output_keys:
            raise RuntimeError(f"{name}/{row['id']} output schema changed")
        parent_seed = _runner_stable_seed(run_seed, row["id"], -1, "stage1")
        if (
            output.get("sample_index") != 0
            or output.get("stage1_parent_seed") != parent_seed
            or output.get("seed_stage1") != parent_seed
        ):
            raise RuntimeError(f"{name}/{row['id']} stage-one seed changed")
        if stage2_seed is not None and stage2_seed != _runner_stable_seed(
            run_seed, row["id"], 0, "stage2"
        ):
            raise RuntimeError(f"{name}/{row['id']} stage-two seed changed")
        stage1_ids = output.get("stage1_token_ids")
        stage2_ids = output.get("stage2_token_ids")
        final_ids = output.get("token_ids")
        injected_ids = output.get("injected_token_ids")
        if not all(
            isinstance(value, list)
            and all(isinstance(token, int) and not isinstance(token, bool) for token in value)
            for value in (stage1_ids, stage2_ids, final_ids, injected_ids)
        ):
            raise RuntimeError(f"{name}/{row['id']} sampled-token receipts are missing")
        trim_stage1 = (
            stage1_ids[: stage1_ids.index(248044)]
            if 248044 in stage1_ids
            else list(stage1_ids)
        )
        trim_stage2 = (
            stage2_ids[: stage2_ids.index(248044)]
            if 248044 in stage2_ids
            else list(stage2_ids)
        )
        if output.get("n_stage1_prompt_tokens") != len(prompt_ids):
            raise RuntimeError(f"{name}/{row['id']} token accounting changed")
        if stage2_seed is None:
            if 248069 not in trim_stage1:
                raise RuntimeError(f"{name}/{row['id']} natural output lacks think close")
            close_index = trim_stage1.index(248069)
            expected_final = trim_stage1
            expected_thinking = close_index
            expected_answer = len(trim_stage1) - close_index - 1
            expected_terminal_trim = len(stage1_ids) - len(trim_stage1)
            natural_valid = bool(
                not stage2_ids
                and not injected_ids
                and output.get("n_stage2_prompt_tokens") == 0
                and output.get("n_injected_tokens") == 0
                and output.get("forced_close") is False
                and output.get("thinking_closed") is True
                and output.get("finish_reason") == "stop"
                and output.get("stage1_finish_reason") == "stop"
                and output.get("stage2_cumulative_logprob") is None
                and output.get("stage2_logprobs") is None
            )
            if not natural_valid:
                raise RuntimeError(f"{name}/{row['id']} natural output structure changed")
        else:
            retained = output["retained_thinking_token_ids"]
            if not isinstance(retained, list):
                raise RuntimeError(f"{name}/{row['id']} retained thinking is invalid")
            close_index = (
                trim_stage1.index(248069) if 248069 in trim_stage1 else None
            )
            expected_retained = (
                trim_stage1[:close_index]
                if close_index is not None
                else list(trim_stage1)
            )
            expected_final = expected_retained + close_ids + trim_stage2
            expected_thinking = len(expected_retained)
            expected_answer = len(trim_stage2)
            expected_terminal_trim = (
                len(stage1_ids)
                - len(trim_stage1)
                + len(stage2_ids)
                - len(trim_stage2)
            )
            continuation_valid = bool(
                retained == expected_retained
                and injected_ids == close_ids
                and output.get("n_stage2_prompt_tokens")
                == len(prompt_ids) + len(expected_retained) + len(close_ids)
                and output.get("n_injected_tokens") == len(close_ids)
                and output.get("forced_close") is (close_index is None)
                and output.get("thinking_closed") is True
                and 0 <= expected_answer <= int(sampling.answer_max_tokens)
            )
            if not continuation_valid:
                raise RuntimeError(f"{name}/{row['id']} continuation structure changed")
        accounting_valid = bool(
            final_ids == expected_final
            and output.get("text") == _decode_token_ids(final_ids)
            and output.get("n_thinking_tokens") == expected_thinking
            and output.get("n_answer_tokens") == expected_answer
            and output.get("n_sampled_tokens") == len(stage1_ids) + len(stage2_ids)
            and output.get("n_completion_tokens") == len(final_ids)
            and output.get("n_terminal_tokens_trimmed") == expected_terminal_trim
            and output.get("truncated")
            is (output.get("finish_reason") == "length")
            and output.get("stage1_logprobs") is None
            and output.get("stage2_logprobs") is None
        )
        if not accounting_valid:
            raise RuntimeError(f"{name}/{row['id']} exact token/text accounting changed")
    expected_counts = _metadata_expected_counts(rows)
    counts = metadata.get("counts", {})
    for key, expected in expected_counts.items():
        if counts.get(key) != expected:
            raise RuntimeError(f"{name} aggregate count changed: {key}")


def _artifact_paths(name: str) -> dict[str, Path]:
    return {
        "started": RAW / f"{name}.started.json",
        "raw": RAW / f"{name}.jsonl",
        "metadata": RAW / f"{name}.metadata.json",
        "complete": RAW / f"{name}.complete.json",
    }


def _exclusive_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _exclusive_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _completion_receipt(
    name: str,
    paths: dict[str, Path],
    prepared_path: Path,
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    lock_path: Path,
    live_preflight_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "state": "COMPLETE",
        "invocation": name,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "request_path": str(prepared_path.relative_to(ROOT)),
        "request_sha256": _sha256_file(prepared_path),
        "request_count": len(rows),
        "request_order_sha256": _canonical_hash([row["id"] for row in rows]),
        "started_sha256": _sha256_file(paths["started"]),
        "raw_path": str(paths["raw"].relative_to(ROOT)),
        "raw_sha256": _sha256_file(paths["raw"]),
        "raw_count": len(rows),
        "metadata_path": str(paths["metadata"].relative_to(ROOT)),
        "metadata_sha256": _sha256_file(paths["metadata"]),
        "sampling_sha256": _canonical_hash(metadata["sampling"]),
        "engine_sha256": _canonical_hash(metadata["engine"]),
        "runner_sha256": metadata["runner_sha256"],
        "implementation_lock_sha256": _sha256_file(lock_path),
        "live_preflight_sha256": _sha256_file(live_preflight_path),
    }


def _started_receipt(
    name: str,
    prepared: list[dict[str, Any]],
    config: dict[str, Any],
    lock_path: Path,
    live_preflight_path: Path,
) -> dict[str, Any]:
    prepared_path = _expected_prepared_paths()[name]
    return {
        "schema_version": 1,
        "state": "STARTED",
        "invocation": name,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "request_sha256": _sha256_file(prepared_path),
        "request_count": len(prepared),
        "request_order_sha256": _canonical_hash([row["id"] for row in prepared]),
        "sampling_sha256": _canonical_hash(
            dataclasses.asdict(_sampling(config, program=name == PROGRAM_ARM))
        ),
        "engine_sha256": _canonical_hash(dataclasses.asdict(_engine_config(config))),
        "runner_sha256": _sha256_file(EXP / "src" / "vllm_runner.py"),
        "implementation_lock_sha256": _sha256_file(lock_path),
        "live_preflight_sha256": _sha256_file(live_preflight_path),
    }


def _load_completed_invocation(
    name: str,
    prepared: list[dict[str, Any]],
    config: dict[str, Any],
    lock_path: Path,
    live_preflight_path: Path,
    *,
    allow_finalize: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    paths = _artifact_paths(name)
    exists = {key: path.exists() for key, path in paths.items()}
    if exists["complete"] and not all(exists.values()):
        raise RuntimeError(f"{name} complete receipt has missing transaction files")
    if not exists["complete"]:
        if not any(exists.values()):
            return None
        if not exists["started"]:
            raise RuntimeError(f"{name} has orphan output without a started receipt")
        if not exists["raw"] and not exists["metadata"]:
            raise RuntimeError(f"{name} has an ambiguous started model call; refusing resample")
        if exists["raw"] != exists["metadata"]:
            raise RuntimeError(f"{name} has an incomplete raw/metadata pair")
        if not allow_finalize:
            raise RuntimeError(f"{name} transaction requires verification-only finalization")
    for key, path in paths.items():
        if path.is_symlink():
            raise RuntimeError(f"{name} transaction contains unsafe symlink: {key}")
    expected_started = _started_receipt(
        name, prepared, config, lock_path, live_preflight_path
    )
    if json.loads(paths["started"].read_text()) != expected_started:
        raise RuntimeError(f"{name} started receipt authentication failed")
    rows = _read_jsonl(paths["raw"])
    metadata = json.loads(paths["metadata"].read_text())
    _authenticate_generation(name, rows, metadata, prepared, config)
    prepared_path = _expected_prepared_paths()[name]
    expected = _completion_receipt(
        name, paths, prepared_path, rows, metadata, lock_path, live_preflight_path
    )
    if exists["complete"]:
        complete = json.loads(paths["complete"].read_text())
        if complete != expected:
            raise RuntimeError(f"{name} complete receipt authentication failed")
    else:
        _exclusive_json(paths["complete"], expected)
    return rows, metadata


def _live_engine_preflight(
    runner: VLLMRunner,
    config: dict[str, Any],
    prepared: dict[str, list[dict[str, Any]]],
    lock_path: Path,
) -> dict[str, Any]:
    vllm_config = runner.llm.llm_engine.vllm_config
    cache = vllm_config.cache_config
    scheduler = vllm_config.scheduler_config
    model = vllm_config.model_config
    parallel = vllm_config.parallel_config
    capacity = getattr(cache, "kv_cache_size_tokens", None)
    block_size = getattr(cache, "block_size", None)
    num_gpu_blocks = getattr(cache, "num_gpu_blocks", None)
    if not isinstance(capacity, int) or capacity <= 0:
        raise RuntimeError("live vLLM engine did not expose positive KV token capacity")
    if not isinstance(block_size, int) or block_size <= 0:
        raise RuntimeError("live vLLM engine did not expose positive KV block size")
    if not isinstance(num_gpu_blocks, int) or num_gpu_blocks < runner.config.max_num_seqs:
        raise RuntimeError("live Mamba/KV block concurrency is below max_num_seqs")
    if (
        int(model.max_model_len) != runner.config.max_model_len
        or str(model.dtype) not in {"bfloat16", "torch.bfloat16"}
        or int(scheduler.max_num_seqs) != runner.config.max_num_seqs
        or int(scheduler.max_num_batched_tokens) != runner.config.max_num_batched_tokens
        or bool(scheduler.async_scheduling)
    ):
        raise RuntimeError("live scheduler/model geometry differs from frozen engine config")
    if (
        int(parallel.world_size) != 1
        or int(parallel.tensor_parallel_size) != 1
        or int(parallel.data_parallel_size) != 1
        or bool(cache.enable_prefix_caching) is not runner.config.enable_prefix_caching
        or str(cache.mamba_cache_mode) != "none"
        or not isinstance(cache.kv_cache_max_concurrency, (int, float))
        or float(cache.kv_cache_max_concurrency) <= 0
    ):
        raise RuntimeError("live parallel/cache protocol differs from the frozen DP=1 run")

    invocation_capacity: dict[str, Any] = {}
    for name in INVOCATIONS:
        sampling = _sampling(config, program=name == PROGRAM_ARM)
        reserve = (
            int(sampling.thinking_budget)
            + len(runner.close_ids)
            + int(sampling.answer_max_tokens)
        )
        prompt_lengths = [len(row["prompt_token_ids"]) for row in prepared[name]]
        max_total = max(prompt_lengths) + reserve
        rounded = math.ceil(max_total / block_size) * block_size
        logical = len(prompt_lengths) * int(sampling.n)
        active = min(logical, runner.config.max_num_seqs)
        required = active * rounded
        if max_total > runner.config.max_model_len or required > capacity:
            raise RuntimeError(f"{name} does not fit the live context/KV capacity")
        invocation_capacity[name] = {
            "request_count": len(prompt_lengths),
            "min_prompt_tokens": min(prompt_lengths),
            "max_prompt_tokens": max(prompt_lengths),
            "generation_reserve_tokens": reserve,
            "max_prompt_plus_reserve_tokens": max_total,
            "logical_sequences": logical,
            "active_sequences": active,
            "rounded_tokens_per_sequence": rounded,
            "required_cache_tokens": required,
            "remaining_cache_tokens": capacity - required,
            "pass": True,
        }

    cache_fields = (
        "num_gpu_blocks",
        "block_size",
        "kv_cache_size_tokens",
        "kv_cache_max_concurrency",
        "gpu_memory_utilization",
        "cache_dtype",
        "enable_prefix_caching",
        "mamba_cache_mode",
        "mamba_block_size",
        "mamba_page_size_padded",
        "mamba_cache_dtype",
        "mamba_ssm_cache_dtype",
        "hash_block_size",
    )
    preflight = {
        "schema_version": 1,
        "decision": "LIVE_ENGINE_PREFLIGHT_PASS",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "backend": "vllm",
        "runner_sha256": _sha256_file(EXP / "src" / "vllm_runner.py"),
        "implementation_lock_sha256": _sha256_file(lock_path),
        "prepare_receipt_sha256": _sha256_file(PREPARED / "preoutcome_receipt.json"),
        "forced_close_token_count": len(runner.close_ids),
        "engine": _normalized(dataclasses.asdict(runner.config)),
        "engine_args": _normalized(runner.engine_args),
        "resolved_cudagraph": _normalized(runner.resolved_cudagraph),
        "live_model": {
            "max_model_len": int(model.max_model_len),
            "dtype": str(model.dtype),
        },
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
            key: _normalized(getattr(cache, key, None)) for key in cache_fields
        },
        "capacity_source": "vllm_config.cache_config.kv_cache_size_tokens",
        "invocations": invocation_capacity,
    }
    path = RAW / "live_preflight.json"
    if path.exists():
        if path.is_symlink() or json.loads(path.read_text()) != preflight:
            raise RuntimeError("live engine preflight differs from the frozen resume receipt")
    else:
        _exclusive_json(path, preflight)
    return preflight


def _validate_recorded_live_preflight(
    config: dict[str, Any],
    prepared: dict[str, list[dict[str, Any]]],
    lock_path: Path,
) -> dict[str, Any]:
    path = RAW / "live_preflight.json"
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("live engine preflight is missing or unsafe")
    receipt = json.loads(path.read_text())
    expected_identity = {
        "schema_version": 1,
        "decision": "LIVE_ENGINE_PREFLIGHT_PASS",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "backend": "vllm",
        "runner_sha256": _sha256_file(EXP / "src" / "vllm_runner.py"),
        "implementation_lock_sha256": _sha256_file(lock_path),
        "prepare_receipt_sha256": _sha256_file(PREPARED / "preoutcome_receipt.json"),
        "capacity_source": "vllm_config.cache_config.kv_cache_size_tokens",
    }
    for key, expected in expected_identity.items():
        if receipt.get(key) != expected:
            raise RuntimeError(f"recorded live preflight identity changed: {key}")
    engine = _engine_config(config)
    if _normalized(receipt.get("engine")) != _normalized(dataclasses.asdict(engine)):
        raise RuntimeError("recorded live preflight engine config changed")
    scheduler = receipt.get("live_scheduler", {})
    if scheduler != {
        "max_num_seqs": engine.max_num_seqs,
        "max_num_batched_tokens": engine.max_num_batched_tokens,
        "async_scheduling": False,
    }:
        raise RuntimeError("recorded live scheduler geometry changed")
    if receipt.get("live_parallel") != {
        "world_size": 1,
        "tensor_parallel_size": 1,
        "data_parallel_size": 1,
    }:
        raise RuntimeError("recorded live parallel geometry changed")
    graph = receipt.get("resolved_cudagraph", {})
    if (
        graph.get("cudagraph_capture_sizes") != list(engine.cudagraph_capture_sizes or ())
        or graph.get("max_cudagraph_capture_size") != engine.max_num_seqs
        or graph.get("decode_mode") != "FULL"
        or not graph.get("has_full_cudagraphs")
    ):
        raise RuntimeError("recorded live CUDA-graph geometry changed")
    live_model = receipt.get("live_model", {})
    if (
        live_model.get("max_model_len") != engine.max_model_len
        or live_model.get("dtype") not in {"bfloat16", "torch.bfloat16"}
    ):
        raise RuntimeError("recorded live model length changed")
    cache = receipt.get("live_cache", {})
    close_token_count = receipt.get("forced_close_token_count")
    if not isinstance(close_token_count, int) or close_token_count < 1:
        raise RuntimeError("recorded forced-close token geometry is invalid")
    capacity = cache.get("kv_cache_size_tokens")
    block_size = cache.get("block_size")
    num_gpu_blocks = cache.get("num_gpu_blocks")
    max_concurrency = cache.get("kv_cache_max_concurrency")
    if (
        not isinstance(capacity, int)
        or capacity <= 0
        or not isinstance(block_size, int)
        or block_size <= 0
        or not isinstance(num_gpu_blocks, int)
        or num_gpu_blocks < engine.max_num_seqs
        or not isinstance(max_concurrency, (int, float))
        or max_concurrency <= 0
        or cache.get("enable_prefix_caching") is not False
        or cache.get("mamba_cache_mode") != "none"
    ):
        raise RuntimeError("recorded live cache geometry is invalid")
    invocations = receipt.get("invocations", {})
    if set(invocations) != set(INVOCATIONS):
        raise RuntimeError("recorded live invocation inventory changed")
    for name in INVOCATIONS:
        sampling = _sampling(config, program=name == PROGRAM_ARM)
        reserve = (
            int(sampling.thinking_budget)
            + close_token_count
            + int(sampling.answer_max_tokens)
        )
        prompt_lengths = [len(row["prompt_token_ids"]) for row in prepared[name]]
        max_total = max(prompt_lengths) + reserve
        rounded = math.ceil(max_total / block_size) * block_size
        logical = len(prompt_lengths)
        active = min(logical, engine.max_num_seqs)
        required = active * rounded
        expected = {
            "request_count": len(prompt_lengths),
            "min_prompt_tokens": min(prompt_lengths),
            "max_prompt_tokens": max(prompt_lengths),
            "generation_reserve_tokens": reserve,
            "max_prompt_plus_reserve_tokens": max_total,
            "logical_sequences": logical,
            "active_sequences": active,
            "rounded_tokens_per_sequence": rounded,
            "required_cache_tokens": required,
            "remaining_cache_tokens": capacity - required,
            "pass": True,
        }
        if invocations[name] != expected or required > capacity:
            raise RuntimeError(f"recorded live KV fit changed: {name}")
    return receipt


def _validate_raw_inventory() -> None:
    if not RAW.exists():
        return
    allowed = {"run.lock", "live_preflight.json", "authentication_receipt.json"}
    for name in INVOCATIONS:
        allowed.update(path.name for path in _artifact_paths(name).values())
    unknown = sorted(path.name for path in RAW.iterdir() if path.name not in allowed)
    if unknown:
        raise RuntimeError(f"unknown raw mechanics artifacts: {unknown}")
    if any(path.is_dir() or path.is_symlink() for path in RAW.iterdir()):
        raise RuntimeError("raw mechanics transaction inventory contains unsafe entries")


@contextmanager
def _exclusive_run_lock():
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / "run.lock"
    handle = path.open("a+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another mechanics runner holds the process lock") from exc
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _generate_invocation(
    name: str,
    runner: VLLMRunner,
    prepared: list[dict[str, Any]],
    config: dict[str, Any],
    lock_path: Path,
    live_preflight_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = _artifact_paths(name)
    if any(path.exists() for path in paths.values()):
        raise RuntimeError(f"{name} is not pending at generation entry")
    prepared_path = _expected_prepared_paths()[name]
    _exclusive_json(
        paths["started"],
        _started_receipt(name, prepared, config, lock_path, live_preflight_path),
    )
    rows, metadata = runner.generate(
        prepared,
        _sampling(config, program=name == PROGRAM_ARM),
    )
    _authenticate_generation(name, rows, metadata, prepared, config)
    _exclusive_jsonl(paths["raw"], rows)
    _exclusive_json(paths["metadata"], metadata)
    expected = _completion_receipt(
        name, paths, prepared_path, rows, metadata, lock_path, live_preflight_path
    )
    _exclusive_json(paths["complete"], expected)
    return rows, metadata


def _resource_fields(output: dict[str, Any], meta: dict[str, Any]) -> dict[str, int]:
    stage1_sampled = len(output.get("stage1_token_ids") or [])
    stage2_sampled = len(output.get("stage2_token_ids") or [])
    logical = (
        int(output["n_stage1_prompt_tokens"])
        + stage1_sampled
        + int(output["n_stage2_prompt_tokens"])
        + stage2_sampled
    )
    return {
        "candidate_injection_tokens": int(meta["injection_token_count"]),
        "sampled_tokens": int(output["n_sampled_tokens"]),
        "logical_model_tokens": logical,
    }


def _cap_contact(output: dict[str, Any], answer_max: int) -> bool:
    return bool(
        int(output["n_answer_tokens"]) >= answer_max
        or output.get("stage2_finish_reason") == "length"
        or output.get("finish_reason") == "length"
    )


def _score_authenticated(
    config: dict[str, Any], authenticated_rows: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    answer_max = int(config["generation"]["answer_max_tokens"])
    scored_paths: dict[str, Path] = {}
    metrics: dict[str, dict[str, Any]] = {}
    operation_success: Counter[str] = Counter()
    context_success: Counter[str] = Counter()

    for arm in ARMS:
        raw_rows = authenticated_rows[arm]
        if len(raw_rows) != 96:
            raise RuntimeError(f"{arm} raw row count changed")
        scored_rows = []
        parse_count = 0
        registered_count = 0
        supplied_count = 0
        cap_count = 0
        context_counts: dict[str, Counter[str]] = defaultdict(Counter)
        for row in raw_rows:
            if len(row.get("outputs", [])) != 1:
                raise RuntimeError("mechanics requires n=1 outputs")
            output = row["outputs"][0]
            meta = row["meta"]
            parsed = parse_result(str(output["text"]))
            parse_count += int(parsed["parsed"])
            result = parsed["result"] if parsed["parsed"] else None
            registered_ok = result == meta["expected_registered"]
            supplied_ok = (
                meta["expected_supplied"] is not None
                and result == meta["expected_supplied"]
            )
            cap = _cap_contact(output, answer_max)
            registered_count += int(registered_ok)
            supplied_count += int(supplied_ok)
            cap_count += int(cap)
            context_id = str(meta["context_id"])
            context_counts[context_id].update(
                {
                    "rows": 1,
                    "parsed": int(parsed["parsed"]),
                    "registered": int(registered_ok),
                    "supplied": int(supplied_ok),
                    "cap": int(cap),
                }
            )
            if arm == "systematic" and supplied_ok:
                operation_success[meta["registered_operation"]] += 1
                context_success[meta["context_id"]] += 1
            scored_rows.append(
                {
                    "id": row["id"],
                    "arm": arm,
                    "meta": meta,
                    "parse": parsed,
                    "registered_execution_correct": registered_ok,
                    "supplied_execution_correct": supplied_ok,
                    "answer_limit_contact": cap,
                    "completion_think_close_count": list(
                        output.get("token_ids") or []
                    ).count(248069),
                    "resource": _resource_fields(output, meta),
                    "seed_stage1": output["seed_stage1"],
                    "seed_stage2": output["seed_stage2"],
                    "forced_close": output["forced_close"],
                    "text": output["text"],
                }
            )
        path = SCORED / f"{arm}.jsonl"
        _write_jsonl(path, scored_rows)
        scored_paths[arm] = path
        metrics[arm] = {
            "rows": len(scored_rows),
            "parse_rate": parse_count / len(scored_rows),
            "registered_execution_rate": registered_count / len(scored_rows),
            "supplied_execution_rate": supplied_count / len(scored_rows),
            "answer_limit_contact_rate": cap_count / len(scored_rows),
            "sampled_tokens": sum(row["resource"]["sampled_tokens"] for row in scored_rows),
            "logical_model_tokens": sum(
                row["resource"]["logical_model_tokens"] for row in scored_rows
            ),
            "contexts": {
                context_id: {
                    "rows": counts["rows"],
                    "parse_count": counts["parsed"],
                    "parse_rate": counts["parsed"] / counts["rows"],
                    "registered_execution_count": counts["registered"],
                    "registered_execution_rate": counts["registered"] / counts["rows"],
                    "supplied_execution_count": counts["supplied"],
                    "supplied_execution_rate": counts["supplied"] / counts["rows"],
                    "answer_limit_contact_count": counts["cap"],
                    "answer_limit_contact_rate": counts["cap"] / counts["rows"],
                }
                for context_id, counts in sorted(context_counts.items())
            },
        }

    seed_rows = {
        arm: {row["id"]: row for row in _read_jsonl(scored_paths[arm])}
        for arm in ARMS
    }
    systematic_seeds = {
        record_id: row["seed_stage1"]
        for record_id, row in seed_rows["systematic"].items()
    }
    for arm in ARMS[1:]:
        arm_seeds = {
            record_id: row["seed_stage1"]
            for record_id, row in seed_rows[arm].items()
        }
        if arm_seeds != systematic_seeds:
            raise RuntimeError(f"{arm} does not have row-matched stage-one seeds")
    for record_id in systematic_seeds:
        used_stage2_seeds = {
            seed_rows[arm][record_id]["seed_stage2"]
            for arm in ARMS
            if seed_rows[arm][record_id]["seed_stage2"] is not None
        }
        if len(used_stage2_seeds) > 1:
            raise RuntimeError("row-matched stage-two seeds disagree when used")

    program_raw = authenticated_rows[PROGRAM_ARM]
    if len(program_raw) != int(config["data"]["mechanics_tasks"]):
        raise RuntimeError("program-ceiling row count changed")
    program_scored = []
    for row in program_raw:
        if len(row.get("outputs", [])) != 1:
            raise RuntimeError("program ceiling requires exactly one output")
        output = row["outputs"][0]
        meta = row["meta"]
        parsed = parse_program(str(output["text"]))
        scored = score_candidate(meta["public_task"], str(output["text"]))
        program_scored.append(
            {
                "id": row["id"],
                "meta": meta,
                "parsed": parsed["parsed"],
                "parse_error": parsed["error"],
                "visible_pass": scored["visible_pass"],
                "parameterized_first_operation": any(
                    str(meta["registered_operation"]).startswith(f"{family}(")
                    for family in ("add_k", "mul_k", "take_k", "rotate_k")
                ),
                "answer_limit_contact": _cap_contact(output, answer_max),
                "completion_think_close_count": list(
                    output.get("token_ids") or []
                ).count(248069),
                "resource": _resource_fields(output, meta),
                "text": output["text"],
            }
        )
    program_path = SCORED / "program_ceiling.jsonl"
    _write_jsonl(program_path, program_scored)
    program_visible_rate = sum(row["visible_pass"] for row in program_scored) / len(
        program_scored
    )
    program_parse_rate = sum(row["parsed"] for row in program_scored) / len(
        program_scored
    )
    program_cap_rate = sum(row["answer_limit_contact"] for row in program_scored) / len(
        program_scored
    )
    parameterized_program_rows = [
        row for row in program_scored if row["parameterized_first_operation"]
    ]
    if len(parameterized_program_rows) != 4:
        raise RuntimeError("program ceiling parameterized stratum changed")
    parameterized_program_visible_rate = sum(
        row["visible_pass"] for row in parameterized_program_rows
    ) / len(parameterized_program_rows)

    gates = config["mechanics"]
    primary_interface_valid = all(
        metrics[arm]["parse_rate"] >= float(gates["parse_rate_min"])
        and metrics[arm]["answer_limit_contact_rate"]
        <= float(gates["answer_limit_contact_max"])
        for arm in ARMS
    )
    program_interface_valid = bool(
        program_parse_rate >= float(gates["parse_rate_min"])
        and program_cap_rate <= float(gates["answer_limit_contact_max"])
    )
    context_gates: dict[str, Any] = {}
    for context_id in sorted(metrics["systematic"]["contexts"]):
        systematic = metrics["systematic"]["contexts"][context_id]
        deranged = metrics["deranged"]["contexts"][context_id]
        delta = (
            systematic["registered_execution_rate"]
            - deranged["registered_execution_rate"]
        )
        checks = {
            "systematic_execution": systematic["supplied_execution_rate"]
            >= float(gates["execution_rate_min"]),
            "systematic_candidate_adherence": systematic["supplied_execution_rate"]
            >= float(gates["candidate_adherence_min"]),
            "deranged_supplied_adherence": deranged["supplied_execution_rate"]
            >= float(gates["deranged_own_execution_min"]),
            "registered_delta": delta >= float(gates["deranged_adherence_gain_min"]),
        }
        context_gates[context_id] = {
            "rows": systematic["rows"],
            "systematic_registered_count": systematic["registered_execution_count"],
            "systematic_registered_rate": systematic["registered_execution_rate"],
            "systematic_supplied_count": systematic["supplied_execution_count"],
            "systematic_supplied_rate": systematic["supplied_execution_rate"],
            "deranged_registered_count": deranged["registered_execution_count"],
            "deranged_registered_rate": deranged["registered_execution_rate"],
            "deranged_supplied_count": deranged["supplied_execution_count"],
            "deranged_supplied_rate": deranged["supplied_execution_rate"],
            "systematic_registered_minus_deranged_registered": delta,
            "thresholds": {
                "execution_rate_min": float(gates["execution_rate_min"]),
                "candidate_adherence_min": float(gates["candidate_adherence_min"]),
                "deranged_own_execution_min": float(gates["deranged_own_execution_min"]),
                "deranged_adherence_gain_min": float(gates["deranged_adherence_gain_min"]),
            },
            "checks": checks,
            "pass": all(checks.values()),
        }
    context_adherence_valid = bool(
        len(context_gates) == 4 and all(row["pass"] for row in context_gates.values())
    )
    adherence_valid = bool(
        metrics["systematic"]["supplied_execution_rate"]
        >= float(gates["execution_rate_min"])
        and metrics["systematic"]["supplied_execution_rate"]
        >= float(gates["candidate_adherence_min"])
        and metrics["deranged"]["supplied_execution_rate"]
        >= float(gates["deranged_own_execution_min"])
        and metrics["systematic"]["registered_execution_rate"]
        - metrics["deranged"]["registered_execution_rate"]
        >= float(gates["deranged_adherence_gain_min"])
        and len(operation_success) >= int(gates["operation_support_min"])
        and len(context_success) >= int(gates["task_support_min"])
        and metrics["duplicate"]["registered_execution_rate"]
        <= float(gates["duplicate_target_rate_max"])
        and metrics["placebo"]["registered_execution_rate"]
        <= float(gates["placebo_target_rate_max"])
        and context_adherence_valid
    )
    program_valid = bool(
        program_interface_valid
        and program_visible_rate
        >= float(gates["correct_hypothesis_visible_pass_min"])
        and parameterized_program_visible_rate
        >= float(gates["parameterized_hypothesis_visible_pass_min"])
    )
    interface_valid = primary_interface_valid and program_interface_valid
    if not interface_valid:
        decision = "INVALID_INTERFACE_PARSE"
    elif not adherence_valid:
        decision = "NO_HYPOTHESIS_ADHERENCE"
    elif not program_valid:
        decision = "NO_CORRECT_HYPOTHESIS_CEILING"
    else:
        decision = "EARLY_HYPOTHESIS_MECHANICS_PASS"

    summary = {
        "schema_version": 1,
        "stage": "mechanics",
        "decision": decision,
        "passed": decision == "EARLY_HYPOTHESIS_MECHANICS_PASS",
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "backend": "vllm",
        "metrics": metrics,
        "systematic_operation_support": len(operation_success),
        "systematic_context_support": len(context_success),
        "systematic_registered_minus_deranged_registered": (
            metrics["systematic"]["registered_execution_rate"]
            - metrics["deranged"]["registered_execution_rate"]
        ),
        "context_adherence": context_gates,
        "program_ceiling": {
            "scope": "noncausal_reachability_only",
            "rows": len(program_scored),
            "first_operations": [
                canonical_operation(operation)
                for operation in MECHANICS_PROGRAM_FIRST_OPERATIONS
            ],
            "parse_rate": program_parse_rate,
            "visible_pass_rate": program_visible_rate,
            "parameterized_rows": len(parameterized_program_rows),
            "parameterized_visible_pass_rate": parameterized_program_visible_rate,
            "answer_limit_contact_rate": program_cap_rate,
        },
        "gates": {
            "interface_valid": interface_valid,
            "primary_interface_valid": primary_interface_valid,
            "program_interface_valid": program_interface_valid,
            "context_adherence_valid": context_adherence_valid,
            "hypothesis_adherence_valid": adherence_valid,
            "correct_hypothesis_ceiling_valid": program_valid,
        },
        "qualification_authorized": decision == "EARLY_HYPOTHESIS_MECHANICS_PASS",
        "scored_files": {
            **{arm: _sha256_file(path) for arm, path in scored_paths.items()},
            "program_ceiling": _sha256_file(program_path),
        },
    }
    _write_json(SUMMARY, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def analyze(config: dict[str, Any], lock_path: Path) -> dict[str, Any]:
    verify_implementation_lock(lock_path)
    _validate_config(config, require_backend_package=True)
    _validate_raw_inventory()
    _receipt, prepared = _load_and_validate_prepared(config)
    live_preflight_path = RAW / "live_preflight.json"
    if live_preflight_path.is_symlink() or not live_preflight_path.is_file():
        raise RuntimeError("authenticated analysis requires the live engine preflight")
    _validate_recorded_live_preflight(config, prepared, lock_path)
    authenticated: dict[str, list[dict[str, Any]]] = {}
    completion_hashes: dict[str, str] = {}
    for name in INVOCATIONS:
        completed = _load_completed_invocation(
            name,
            prepared[name],
            config,
            lock_path,
            live_preflight_path,
            allow_finalize=False,
        )
        if completed is None:
            raise RuntimeError(f"authenticated analysis is missing invocation: {name}")
        authenticated[name] = completed[0]
        completion_hashes[name] = _sha256_file(_artifact_paths(name)["complete"])
    authentication = {
        "schema_version": 1,
        "decision": "MECHANICS_AUTHENTICATION_PASS",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "implementation_lock_sha256": _sha256_file(lock_path),
        "prepare_receipt_sha256": _sha256_file(PREPARED / "preoutcome_receipt.json"),
        "live_preflight_sha256": _sha256_file(live_preflight_path),
        "complete_receipts": completion_hashes,
    }
    authentication_path = RAW / "authentication_receipt.json"
    if authentication_path.exists():
        if (
            authentication_path.is_symlink()
            or json.loads(authentication_path.read_text()) != authentication
        ):
            raise RuntimeError("mechanics authentication receipt changed")
    else:
        _exclusive_json(authentication_path, authentication)
    summary = _score_authenticated(config, authenticated)
    summary["authentication_receipt_sha256"] = _sha256_file(authentication_path)
    _write_json(SUMMARY, summary)
    return summary


def run_live(lock_path: Path) -> dict[str, Any]:
    verify_implementation_lock(lock_path)
    config = yaml.safe_load(CONFIG_PATH.read_text())
    _validate_config(config, require_backend_package=True)
    _receipt, prepared = _load_and_validate_prepared(config)
    _validate_raw_inventory()
    live_preflight_path = RAW / "live_preflight.json"

    with _exclusive_run_lock():
        completed: set[str] = set()
        if live_preflight_path.exists():
            _validate_recorded_live_preflight(config, prepared, lock_path)
            for name in INVOCATIONS:
                loaded = _load_completed_invocation(
                    name,
                    prepared[name],
                    config,
                    lock_path,
                    live_preflight_path,
                    allow_finalize=True,
                )
                if loaded is not None:
                    completed.add(name)
        elif any(
            any(path.exists() for path in _artifact_paths(name).values())
            for name in INVOCATIONS
        ):
            raise RuntimeError("transaction state exists without a live preflight receipt")

        if completed != set(INVOCATIONS):
            with VLLMRunner(_engine_config(config)) as runner:
                _live_engine_preflight(runner, config, prepared, lock_path)
                for name in INVOCATIONS:
                    if name in completed:
                        continue
                    _generate_invocation(
                        name,
                        runner,
                        prepared[name],
                        config,
                        lock_path,
                        live_preflight_path,
                    )
    return analyze(config, lock_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("prepare", "run", "analyze"))
    parser.add_argument("--lock", type=Path)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare_requests()
        return 0
    if args.stage == "run":
        if args.lock is None:
            raise RuntimeError("--stage run requires --lock")
        run_live(args.lock.resolve())
        return 0
    if args.lock is None:
        raise RuntimeError("--stage analyze requires --lock")
    analyze(yaml.safe_load(CONFIG_PATH.read_text()), args.lock.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
