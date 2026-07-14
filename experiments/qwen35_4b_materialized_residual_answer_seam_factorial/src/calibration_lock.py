"""Published calibration implementation lock and live-engine preflight."""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from calibration_stage import (
    EXP,
    EXPECTED_ROWS,
    INVOCATION_ORDER,
    ROOT,
    CalibrationInputs,
    canonical_sha256,
    engine_config,
    load_calibration_inputs,
    sampling_configs,
)
from transactions import (
    MODEL_ID,
    MODEL_REVISION,
    read_canonical,
    sha256_bytes,
    sha256_file,
    write_exclusive_durable,
)


IMPLEMENTATION_LOCK = EXP / "runs/calibration/implementation_lock.json"
LIVE_PREFLIGHT = EXP / "runs/calibration/live_preflight.json"
RAW_DIR = EXP / "runs/calibration/raw"
DECISION = EXP / "runs/calibration/decision.json"
REQUIRED_WORKFLOWS = ("Validate Repository", "Publish Research Site")
CRITICAL_FILES = (
    "requirements-vllm.lock.txt",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/configs/default.yaml",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/reports/design_review.md",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/reports/calibration_implementation_review.md",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/reports/preregistration.md",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/runs/prepared/calibration_requests.jsonl",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/runs/prepared/preoutcome_receipt.json",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/runs/tokenizer/receipt_v3.json",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/scripts/run_calibration.py",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/scripts/run_mechanics.py",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/src/calibration_lock.py",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/src/calibration_stage.py",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/src/identity.py",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/src/interface_analysis.py",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/src/mechanics_lock.py",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/src/mechanics_stage.py",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/src/plans.py",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/src/protocol.py",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/src/stats.py",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/src/task_data.py",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/src/transactions.py",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/src/vllm_runner.py",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/tests/test_calibration_lock.py",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/tests/test_calibration_stage.py",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/tests/test_calibration_bootstrap.py",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/tests/test_interface_analysis.py",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/tests/test_identity.py",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/tests/test_mechanics_lock.py",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/tests/test_mechanics_stage.py",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/tests/test_plans.py",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/tests/test_protocol.py",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/tests/test_stats.py",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/tests/test_tokenizer_receipt.py",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/tests/test_transactions.py",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/tests/test_vllm_runner.py",
)
CALIBRATION_RUNTIME_FILES = (
    "requirements-vllm.lock.txt",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/configs/default.yaml",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/runs/prepared/calibration_requests.jsonl",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/runs/prepared/preoutcome_receipt.json",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/runs/tokenizer/receipt_v3.json",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/scripts/run_calibration.py",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/src/calibration_lock.py",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/src/calibration_stage.py",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/src/identity.py",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/src/interface_analysis.py",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/src/protocol.py",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/src/stats.py",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/src/task_data.py",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/src/transactions.py",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/src/vllm_runner.py",
)
FROZEN_MECHANICS_FILES = (
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/data/procedural/mechanics_public.jsonl",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/data/procedural/mechanics_audit.jsonl",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/data/procedural/mechanics_gold.jsonl",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/runs/prepared/transport_requests.jsonl",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/runs/prepared/direct_requests.jsonl",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/runs/prepared/suffix_materialized_requests.jsonl",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/runs/prepared/suffix_name_only_requests.jsonl",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/runs/prepared/suffix_shuffled_requests.jsonl",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/scripts/run_mechanics.py",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/src/mechanics_lock.py",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/src/mechanics_stage.py",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/src/plans.py",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/src/identity.py",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/src/protocol.py",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/src/stats.py",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/src/task_data.py",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/src/transactions.py",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/src/vllm_runner.py",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/tests/test_mechanics_lock.py",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/tests/test_mechanics_stage.py",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/tests/test_plans.py",
    "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/tests/test_stats.py",
)
if not set(CALIBRATION_RUNTIME_FILES) <= set(CRITICAL_FILES):
    raise RuntimeError("calibration runtime inventory escapes critical files")


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _git_bytes(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT)


def _ancestor(older: str, newer: str) -> bool:
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", older, newer],
            cwd=ROOT,
            capture_output=True,
        ).returncode
        == 0
    )


def _commit_id(value: Any) -> str:
    commit = str(value)
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise RuntimeError("implementation boundary has an invalid commit ID")
    return commit


def _normalized(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
    )


def _query_ci_rows(commit: str) -> list[dict[str, Any]]:
    commit = _commit_id(commit)
    command = [
        "gh",
        "run",
        "list",
        "--commit",
        commit,
        "--limit",
        "20",
        "--json",
        "databaseId,headSha,status,conclusion,workflowName,url",
    ]
    try:
        rows = json.loads(
            subprocess.check_output(command, cwd=ROOT, text=True)
        )
    except (subprocess.CalledProcessError, json.JSONDecodeError) as error:
        raise RuntimeError("could not authenticate GitHub workflow state") from error
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise RuntimeError("GitHub workflow response changed")
    return rows


def query_green_ci(commit: str) -> dict[str, dict[str, Any]]:
    commit = _commit_id(commit)
    rows = _query_ci_rows(commit)
    evidence: dict[str, dict[str, Any]] = {}
    for workflow in REQUIRED_WORKFLOWS:
        candidates = [
            row
            for row in rows
            if row.get("workflowName") == workflow and row.get("headSha") == commit
        ]
        if not candidates:
            raise RuntimeError(f"required workflow has no run for {commit}: {workflow}")
        row = max(candidates, key=lambda value: int(value["databaseId"]))
        if row.get("status") != "completed" or row.get("conclusion") != "success":
            raise RuntimeError(f"required workflow is not green: {workflow}")
        evidence[workflow] = {
            "database_id": int(row["databaseId"]),
            "head_sha": commit,
            "status": "completed",
            "conclusion": "success",
            "url": str(row["url"]),
        }
    return evidence


def verify_recorded_ci(
    commit: str, evidence: Mapping[str, Mapping[str, Any]]
) -> None:
    commit = _commit_id(commit)
    if set(evidence) != set(REQUIRED_WORKFLOWS):
        raise RuntimeError("recorded workflow inventory changed")
    rows = _query_ci_rows(commit)
    for workflow in REQUIRED_WORKFLOWS:
        recorded = evidence[workflow]
        matches = [
            row
            for row in rows
            if row.get("workflowName") == workflow
            and row.get("headSha") == commit
            and int(row.get("databaseId", -1)) == recorded.get("database_id")
        ]
        if len(matches) != 1:
            raise RuntimeError(f"recorded workflow run disappeared: {workflow}")
        row = matches[0]
        expected = {
            "database_id": int(row["databaseId"]),
            "head_sha": commit,
            "status": str(row["status"]),
            "conclusion": str(row["conclusion"]),
            "url": str(row["url"]),
        }
        if (
            dict(recorded) != expected
            or expected["status"] != "completed"
            or expected["conclusion"] != "success"
        ):
            raise RuntimeError(f"recorded workflow is not authenticated: {workflow}")


def _critical_hashes(commit: str) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in CRITICAL_FILES:
        path = ROOT / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"critical implementation file is unsafe: {relative}")
        if _git("ls-files", "--error-unmatch", "--", relative) != relative:
            raise RuntimeError(f"critical implementation file is untracked: {relative}")
        blob = _git_bytes("show", f"{commit}:{relative}")
        if path.read_bytes() != blob:
            raise RuntimeError(f"critical working bytes differ from {commit}: {relative}")
        hashes[relative] = sha256_bytes(blob)
    return hashes


def _git_blob_inventory(commit: str, files: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in files:
        blob = _git("rev-parse", f"{commit}:{relative}")
        if len(blob) not in {40, 64} or any(
            character not in "0123456789abcdef" for character in blob
        ):
            raise RuntimeError(f"invalid frozen Git blob: {relative}")
        result[relative] = blob
    return result


def build_lock_value(
    *,
    implementation_commit: str,
    critical_files: Mapping[str, str],
    inputs: CalibrationInputs,
    ci_evidence: Mapping[str, Mapping[str, Any]],
    frozen_mechanics_blobs: Mapping[str, str],
) -> dict[str, Any]:
    implementation_commit = _commit_id(implementation_commit)
    if tuple(critical_files) != CRITICAL_FILES:
        raise RuntimeError("critical implementation allowlist changed")
    if tuple(ci_evidence) != REQUIRED_WORKFLOWS:
        raise RuntimeError("implementation CI evidence changed")
    if set(frozen_mechanics_blobs) != set(FROZEN_MECHANICS_FILES):
        raise RuntimeError("frozen mechanics inventory changed")
    return {
        "schema_version": 1,
        "stage": "calibration_implementation_lock",
        "authorization": "interface_calibration_only",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "implementation_commit": implementation_commit,
        "critical_files": dict(critical_files),
        "calibration_runtime_files": list(CALIBRATION_RUNTIME_FILES),
        "frozen_mechanics_blobs": dict(frozen_mechanics_blobs),
        "calibration_inputs": inputs.read_receipt,
        "invocation_order": list(INVOCATION_ORDER),
        "expected_rows_each": EXPECTED_ROWS,
        "engine": _normalized(dataclasses.asdict(engine_config(inputs))),
        "sampling": {
            name: _normalized(dataclasses.asdict(value))
            for name, value in sampling_configs(inputs).items()
        },
        "implementation_ci": {
            name: dict(value) for name, value in ci_evidence.items()
        },
        "experimental_model_requests_before_lock": 0,
        "sampled_model_outputs_before_lock": 0,
        "hidden_files_read": [],
        "qualification_files_read": [],
        "confirmation_files_read": [],
        "benchmark_files_read": [],
    }


def validate_lock_value(lock: Any, *, inputs: CalibrationInputs) -> dict[str, Any]:
    if not isinstance(lock, dict) or set(lock) != {
        "schema_version",
        "stage",
        "authorization",
        "model",
        "revision",
        "implementation_commit",
        "critical_files",
        "calibration_runtime_files",
        "frozen_mechanics_blobs",
        "calibration_inputs",
        "invocation_order",
        "expected_rows_each",
        "engine",
        "sampling",
        "implementation_ci",
        "experimental_model_requests_before_lock",
        "sampled_model_outputs_before_lock",
        "hidden_files_read",
        "qualification_files_read",
        "confirmation_files_read",
        "benchmark_files_read",
    }:
        raise RuntimeError("calibration implementation lock schema changed")
    expected_plan = {
        name: _normalized(dataclasses.asdict(value))
        for name, value in sampling_configs(inputs).items()
    }
    if (
        lock["schema_version"] != 1
        or lock["stage"] != "calibration_implementation_lock"
        or lock["authorization"] != "interface_calibration_only"
        or lock["model"] != MODEL_ID
        or lock["revision"] != MODEL_REVISION
        or lock["calibration_inputs"] != inputs.read_receipt
        or lock["calibration_runtime_files"] != list(CALIBRATION_RUNTIME_FILES)
        or lock["invocation_order"] != list(INVOCATION_ORDER)
        or lock["expected_rows_each"] != EXPECTED_ROWS
        or lock["engine"] != _normalized(dataclasses.asdict(engine_config(inputs)))
        or lock["sampling"] != expected_plan
        or lock["experimental_model_requests_before_lock"] != 0
        or lock["sampled_model_outputs_before_lock"] != 0
        or any(
            lock[field] != []
            for field in (
                "hidden_files_read",
                "qualification_files_read",
                "confirmation_files_read",
                "benchmark_files_read",
            )
        )
    ):
        raise RuntimeError("calibration implementation lock boundary changed")
    _commit_id(lock["implementation_commit"])
    critical = lock["critical_files"]
    if not isinstance(critical, dict) or set(critical) != set(CRITICAL_FILES):
        raise RuntimeError("calibration critical file inventory changed")
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in critical.values()
    ):
        raise RuntimeError("calibration critical file hash changed")
    frozen = lock["frozen_mechanics_blobs"]
    if (
        not isinstance(frozen, dict)
        or set(frozen) != set(FROZEN_MECHANICS_FILES)
        or any(
            not isinstance(value, str)
            or len(value) not in {40, 64}
            or any(character not in "0123456789abcdef" for character in value)
            for value in frozen.values()
        )
    ):
        raise RuntimeError("calibration frozen mechanics blob inventory changed")
    ci = lock["implementation_ci"]
    if not isinstance(ci, dict) or set(ci) != set(REQUIRED_WORKFLOWS):
        raise RuntimeError("calibration implementation CI inventory changed")
    for workflow, row in ci.items():
        if (
            not isinstance(row, dict)
            or set(row)
            != {"database_id", "head_sha", "status", "conclusion", "url"}
            or row["head_sha"] != lock["implementation_commit"]
            or row["status"] != "completed"
            or row["conclusion"] != "success"
            or not isinstance(row["database_id"], int)
            or not isinstance(row["url"], str)
            or not row["url"].startswith("https://github.com/")
        ):
            raise RuntimeError(f"calibration implementation CI evidence changed: {workflow}")
    return lock


def _ensure_clean_for_lock() -> None:
    dirty = _git("status", "--porcelain=v1", "--untracked-files=all")
    if dirty:
        raise RuntimeError("publishing a calibration lock requires a clean worktree")
    if any(
        path.exists() or path.is_symlink()
        for path in (IMPLEMENTATION_LOCK, LIVE_PREFLIGHT, RAW_DIR, DECISION)
    ):
        raise RuntimeError("calibration lock must precede every calibration artifact")


def publish_calibration_lock(path: Path = IMPLEMENTATION_LOCK) -> dict[str, Any]:
    if path.resolve() != IMPLEMENTATION_LOCK.resolve():
        raise RuntimeError("calibration lock path changed")
    _ensure_clean_for_lock()
    subprocess.run(["git", "fetch", "--quiet", "origin", "main"], cwd=ROOT, check=True)
    commit = _commit_id(_git("rev-parse", "HEAD"))
    if not _ancestor(commit, "origin/main"):
        raise RuntimeError("calibration implementation commit is not published on main")
    inputs = load_calibration_inputs()
    ci = query_green_ci(commit)
    value = build_lock_value(
        implementation_commit=commit,
        critical_files=_critical_hashes(commit),
        inputs=inputs,
        ci_evidence=ci,
        frozen_mechanics_blobs=_git_blob_inventory(commit, FROZEN_MECHANICS_FILES),
    )
    validate_lock_value(value, inputs=inputs)
    write_exclusive_durable(path, value)
    return value


def _relative_lock_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError as error:
        raise RuntimeError("calibration lock escapes the repository") from error


def _validate_live_worktree(
    allowed_prefixes: Sequence[str] = ("runs/calibration/",),
) -> None:
    prefixes = tuple(
        "experiments/qwen35_4b_materialized_residual_answer_seam_factorial/" + value
        for value in allowed_prefixes
    )
    dirty = _git("status", "--porcelain=v1", "--untracked-files=all")
    for line in dirty.splitlines():
        paths = line[3:].split(" -> ")
        if not all(path.startswith(prefixes) for path in paths):
            raise RuntimeError(f"live calibration has unrelated worktree change: {line}")


def verify_calibration_lock(
    path: Path = IMPLEMENTATION_LOCK,
    *,
    verify_network: bool = True,
    allowed_live_prefixes: Sequence[str] = ("runs/calibration/",),
) -> dict[str, Any]:
    relative = _relative_lock_path(path)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("live calibration requires a committed implementation lock")
    inputs = load_calibration_inputs()
    lock = validate_lock_value(read_canonical(path), inputs=inputs)
    if _git("ls-files", "--error-unmatch", "--", relative) != relative:
        raise RuntimeError("calibration implementation lock is untracked")
    if _git_bytes("show", f"HEAD:{relative}") != path.read_bytes():
        raise RuntimeError("calibration implementation lock differs from HEAD")
    implementation_commit = lock["implementation_commit"]
    if verify_network:
        subprocess.run(
            ["git", "fetch", "--quiet", "origin", "main"], cwd=ROOT, check=True
        )
    head = _commit_id(_git("rev-parse", "HEAD"))
    if not _ancestor(implementation_commit, head) or not _ancestor(head, "origin/main"):
        raise RuntimeError("calibration implementation/lock is not published on main")
    for relative_file, expected in lock["critical_files"].items():
        if sha256_bytes(_git_bytes("show", f"{implementation_commit}:{relative_file}")) != expected:
            raise RuntimeError(f"calibration critical file changed: {relative_file}")
        if _git("rev-parse", f"HEAD:{relative_file}") != _git(
            "rev-parse", f"{implementation_commit}:{relative_file}"
        ):
            raise RuntimeError(f"critical file changed after calibration freeze: {relative_file}")
    for relative_file in CALIBRATION_RUNTIME_FILES:
        path_file = ROOT / relative_file
        if (
            path_file.is_symlink()
            or not path_file.is_file()
            or sha256_file(path_file) != lock["critical_files"][relative_file]
        ):
            raise RuntimeError(f"calibration runtime file changed: {relative_file}")
    for relative_file, expected_blob in lock["frozen_mechanics_blobs"].items():
        if _git("rev-parse", f"HEAD:{relative_file}") != expected_blob:
            raise RuntimeError(f"mechanics changed after calibration freeze: {relative_file}")
    if verify_network:
        verify_recorded_ci(implementation_commit, lock["implementation_ci"])
        query_green_ci(head)
    _validate_live_worktree(allowed_live_prefixes)
    return lock


def _prompt_receipt(runner: Any, inputs: CalibrationInputs) -> dict[str, Any]:
    plan = sampling_configs(inputs)
    result: dict[str, Any] = {}
    for channel, thinking in (("thinking", "budget"), ("no_thinking", "off")):
        prepared = runner.prepare(inputs.records, thinking, False)
        token_rows = [row.prompt_token_ids for row in prepared]
        lengths = [len(row) for row in token_rows]
        result[channel] = {
            "rows": len(prepared),
            "ids": [row.record_id for row in prepared],
            "prompt_token_ids_sha256": canonical_sha256(token_rows),
            "prompt_tokens_min": min(lengths),
            "prompt_tokens_max": max(lengths),
        }
    think_reserve = (
        int(plan["calibration_thoughts"].thinking_budget)
        + len(runner.close_ids)
        + len(inputs.tokenizer_receipt["answer_prefix"]["token_ids"])
        + plan["think512_program_slot"].answer_max_tokens
    )
    no_think_reserve = (
        len(inputs.tokenizer_receipt["answer_prefix"]["token_ids"])
        + plan["no_think_program_slot"].max_tokens
    )
    result["max_prompt_plus_reserve"] = {
        "think512_program_slot": result["thinking"]["prompt_tokens_max"]
        + think_reserve,
        "no_think_program_slot": result["no_thinking"]["prompt_tokens_max"]
        + no_think_reserve,
    }
    return result


def _validate_loaded_runner(runner: Any, inputs: CalibrationInputs) -> dict[str, Any]:
    expected_engine = _normalized(dataclasses.asdict(engine_config(inputs)))
    if (
        _normalized(dataclasses.asdict(runner.config)) != expected_engine
        or runner.engine_args.get("model") != MODEL_ID
        or runner.engine_args.get("revision") != MODEL_REVISION
        or runner.engine_args.get("tokenizer_revision") != MODEL_REVISION
        or runner.engine_args.get("dtype") != "bfloat16"
        or runner.engine_args.get("async_scheduling") is not False
        or runner.resolved_logprobs_mode != "raw_logprobs"
        or runner.close_ids
        != inputs.tokenizer_receipt["think_token_ids"]["forced_close_sequence"]
        or runner.tokenizer.encode("PROGRAM:", add_special_tokens=False)
        != inputs.tokenizer_receipt["answer_prefix"]["token_ids"]
    ):
        raise RuntimeError("loaded calibration runner differs from frozen settings")
    resolved = runner.resolved_cudagraph
    expected_sizes = list(engine_config(inputs).cudagraph_capture_sizes or ())
    if (
        not isinstance(resolved, dict)
        or resolved.get("cudagraph_capture_sizes") != expected_sizes
        or resolved.get("max_cudagraph_capture_size") != expected_sizes[-1]
        or resolved.get("has_full_cudagraphs") is not True
    ):
        raise RuntimeError("loaded calibration CUDA-graph geometry changed")
    vllm_config = runner.llm.llm_engine.vllm_config
    scheduler = vllm_config.scheduler_config
    model = vllm_config.model_config
    parallel = vllm_config.parallel_config
    cache = vllm_config.cache_config
    if (
        int(model.max_model_len) != engine_config(inputs).max_model_len
        or str(model.dtype) not in {"bfloat16", "torch.bfloat16"}
        or int(scheduler.max_num_seqs) != engine_config(inputs).max_num_seqs
        or int(scheduler.max_num_batched_tokens)
        != engine_config(inputs).max_num_batched_tokens
        or bool(scheduler.async_scheduling)
        or int(parallel.world_size) != 1
        or int(parallel.tensor_parallel_size) != 1
        or int(parallel.data_parallel_size) != 1
        or bool(cache.enable_prefix_caching)
        or str(cache.mamba_cache_mode) != "none"
        or not isinstance(cache.num_gpu_blocks, int)
        or cache.num_gpu_blocks < EXPECTED_ROWS
    ):
        raise RuntimeError("loaded calibration vLLM geometry changed")
    prompts = _prompt_receipt(runner, inputs)
    if (
        prompts["thinking"]["rows"] != EXPECTED_ROWS
        or prompts["no_thinking"]["rows"] != EXPECTED_ROWS
        or max(prompts["max_prompt_plus_reserve"].values())
        > engine_config(inputs).max_model_len
    ):
        raise RuntimeError("loaded calibration prompt/context geometry changed")
    runtime = runner.runtime_metadata()
    packages = runtime.get("packages", {})
    expected_packages = {
        "vllm": inputs.config["model"]["vllm_version"],
        "torch": "2.11.0+cu129",
        "transformers": "5.13.0",
    }
    if (
        not str(runtime.get("python_executable", "")).endswith("/.venv-vllm/bin/python")
        or not runtime.get("gpu")
        or any(packages.get(name) != version for name, version in expected_packages.items())
    ):
        raise RuntimeError("loaded calibration runtime/environment changed")
    return {"runtime": runtime, "prompts": prompts}


def live_preflight_value(
    *, runner: Any, inputs: CalibrationInputs, lock_path: Path = IMPLEMENTATION_LOCK
) -> dict[str, Any]:
    lock = verify_calibration_lock(lock_path)
    loaded = _validate_loaded_runner(runner, inputs)
    head = _commit_id(_git("rev-parse", "HEAD"))
    return {
        "schema_version": 1,
        "decision": "CALIBRATION_LIVE_ENGINE_PREFLIGHT_PASS",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "implementation_lock_sha256": sha256_file(lock_path),
        "implementation_commit": lock["implementation_commit"],
        "live_head": head,
        "live_head_ci": query_green_ci(head),
        "engine": _normalized(dataclasses.asdict(runner.config)),
        "engine_args_sha256": canonical_sha256(_normalized(runner.engine_args)),
        "resolved_cudagraph": _normalized(runner.resolved_cudagraph),
        "resolved_logprobs_mode": runner.resolved_logprobs_mode,
        "prompt_receipt": loaded["prompts"],
        "runtime": loaded["runtime"],
        "invocation_order": list(INVOCATION_ORDER),
        "expected_rows_each": EXPECTED_ROWS,
        "experimental_generation_requests_before_preflight": 0,
        "sampled_model_outputs_before_preflight": 0,
        "hidden_files_read": [],
        "qualification_files_read": [],
        "confirmation_files_read": [],
        "benchmark_files_read": [],
    }


def verify_recorded_live_preflight(
    *,
    inputs: CalibrationInputs,
    lock_path: Path = IMPLEMENTATION_LOCK,
    path: Path = LIVE_PREFLIGHT,
    runner: Any | None = None,
) -> dict[str, Any]:
    lock = verify_calibration_lock(lock_path)
    value = read_canonical(path)
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "decision",
        "model",
        "revision",
        "implementation_lock_sha256",
        "implementation_commit",
        "live_head",
        "live_head_ci",
        "engine",
        "engine_args_sha256",
        "resolved_cudagraph",
        "resolved_logprobs_mode",
        "prompt_receipt",
        "runtime",
        "invocation_order",
        "expected_rows_each",
        "experimental_generation_requests_before_preflight",
        "sampled_model_outputs_before_preflight",
        "hidden_files_read",
        "qualification_files_read",
        "confirmation_files_read",
        "benchmark_files_read",
    }:
        raise RuntimeError("recorded calibration live preflight schema changed")
    live_head = _commit_id(value["live_head"])
    if (
        value["schema_version"] != 1
        or value["decision"] != "CALIBRATION_LIVE_ENGINE_PREFLIGHT_PASS"
        or value["model"] != MODEL_ID
        or value["revision"] != MODEL_REVISION
        or value["implementation_lock_sha256"] != sha256_file(lock_path)
        or value["implementation_commit"] != lock["implementation_commit"]
        or not _ancestor(lock["implementation_commit"], live_head)
        or not _ancestor(live_head, "origin/main")
        or value["engine"] != _normalized(dataclasses.asdict(engine_config(inputs)))
        or value["resolved_logprobs_mode"] != "raw_logprobs"
        or value["invocation_order"] != list(INVOCATION_ORDER)
        or value["expected_rows_each"] != EXPECTED_ROWS
        or value["experimental_generation_requests_before_preflight"] != 0
        or value["sampled_model_outputs_before_preflight"] != 0
        or any(
            value[field] != []
            for field in (
                "hidden_files_read",
                "qualification_files_read",
                "confirmation_files_read",
                "benchmark_files_read",
            )
        )
    ):
        raise RuntimeError("recorded calibration live preflight boundary changed")
    verify_recorded_ci(live_head, value["live_head_ci"])
    if runner is not None:
        loaded = _validate_loaded_runner(runner, inputs)
        if (
            value["engine"] != _normalized(dataclasses.asdict(runner.config))
            or value["engine_args_sha256"]
            != canonical_sha256(_normalized(runner.engine_args))
            or value["resolved_cudagraph"]
            != _normalized(runner.resolved_cudagraph)
            or value["prompt_receipt"] != loaded["prompts"]
        ):
            raise RuntimeError("current runner differs from recorded live preflight")
        recorded_runtime = value["runtime"]
        current_runtime = loaded["runtime"]
        for field in ("python", "python_executable", "packages", "gpu"):
            if current_runtime.get(field) != recorded_runtime.get(field):
                raise RuntimeError(
                    f"current runtime differs from recorded live preflight: {field}"
                )
    return value


def publish_or_verify_live_preflight(
    *,
    runner: Any,
    inputs: CalibrationInputs,
    lock_path: Path = IMPLEMENTATION_LOCK,
    path: Path = LIVE_PREFLIGHT,
) -> dict[str, Any]:
    if path.exists() or path.is_symlink():
        return verify_recorded_live_preflight(
            inputs=inputs,
            lock_path=lock_path,
            path=path,
            runner=runner,
        )
    value = live_preflight_value(runner=runner, inputs=inputs, lock_path=lock_path)
    write_exclusive_durable(path, value)
    return value
