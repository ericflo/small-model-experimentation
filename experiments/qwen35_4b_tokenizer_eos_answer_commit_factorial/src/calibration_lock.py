"""Published calibration implementation lock and zero-generation live preflight."""

from __future__ import annotations

import dataclasses
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from calibration_stage import (
    EXP,
    EXPECTED_ROWS,
    INVOCATION_ORDER,
    ROOT,
    CalibrationInputs,
    engine_config,
    load_calibration_inputs,
    sampling_configs,
)
from transactions import (
    MODEL_ID,
    MODEL_REVISION,
    canonical_sha256,
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
PREFIX = "experiments/qwen35_4b_tokenizer_eos_answer_commit_factorial/"
CRITICAL_FILES = (
    "requirements-vllm.lock.txt",
    PREFIX + "configs/default.yaml",
    PREFIX + "reports/design_review.md",
    PREFIX + "reports/preregistration.md",
    PREFIX + "reports/calibration_implementation_review.md",
    PREFIX + "runs/prepared/calibration_requests.jsonl",
    PREFIX + "runs/prepared/preoutcome_receipt.json",
    PREFIX + "runs/tokenizer/receipt.json",
    PREFIX + "scripts/run_calibration.py",
    PREFIX + "scripts/tokenizer_receipt.py",
    PREFIX + "src/calibration_lock.py",
    PREFIX + "src/calibration_stage.py",
    PREFIX + "src/identity.py",
    PREFIX + "src/interface_analysis.py",
    PREFIX + "src/protocol.py",
    PREFIX + "src/task_data.py",
    PREFIX + "src/transactions.py",
    PREFIX + "src/vllm_runner.py",
    PREFIX + "tests/test_calibration_lock.py",
    PREFIX + "tests/test_calibration_stage.py",
    PREFIX + "tests/test_calibration_bootstrap.py",
    PREFIX + "tests/test_construction.py",
    PREFIX + "tests/test_interface_analysis.py",
    PREFIX + "tests/test_protocol.py",
    PREFIX + "tests/test_tokenizer_receipt.py",
    PREFIX + "tests/test_transactions.py",
    PREFIX + "tests/test_vllm_runner.py",
)
CALIBRATION_RUNTIME_FILES = (
    "requirements-vllm.lock.txt",
    PREFIX + "configs/default.yaml",
    PREFIX + "runs/prepared/calibration_requests.jsonl",
    PREFIX + "runs/prepared/preoutcome_receipt.json",
    PREFIX + "runs/tokenizer/receipt.json",
    PREFIX + "scripts/run_calibration.py",
    PREFIX + "src/calibration_lock.py",
    PREFIX + "src/calibration_stage.py",
    PREFIX + "src/identity.py",
    PREFIX + "src/interface_analysis.py",
    PREFIX + "src/protocol.py",
    PREFIX + "src/task_data.py",
    PREFIX + "src/transactions.py",
    PREFIX + "src/vllm_runner.py",
)
FROZEN_MECHANICS_FILES = (
    PREFIX + "data/procedural/mechanics_public.jsonl",
    PREFIX + "data/procedural/mechanics_audit.jsonl",
    PREFIX + "data/procedural/mechanics_gold.jsonl.aesgcm",
    PREFIX + "runs/construction/summary.json",
    PREFIX + "runs/prepared/transport_requests.jsonl",
    PREFIX + "runs/prepared/direct_requests.jsonl",
    PREFIX + "runs/prepared/suffix_materialized_requests.jsonl",
    PREFIX + "runs/prepared/suffix_name_only_requests.jsonl",
    PREFIX + "runs/prepared/suffix_shuffled_requests.jsonl",
)
if not set(CALIBRATION_RUNTIME_FILES) <= set(CRITICAL_FILES):
    raise RuntimeError("calibration runtime inventory escapes critical files")


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _git_bytes(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT)


def _commit_id(value: Any) -> str:
    commit = str(value)
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise RuntimeError("implementation boundary has an invalid commit ID")
    return commit


def _ancestor(older: str, newer: str) -> bool:
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", older, newer],
            cwd=ROOT,
            capture_output=True,
        ).returncode
        == 0
    )


def _normalized(value: Any) -> Any:
    return json.loads(
        json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False)
    )


def _query_ci_rows(commit: str) -> list[dict[str, Any]]:
    command = [
        "gh",
        "run",
        "list",
        "--commit",
        _commit_id(commit),
        "--limit",
        "20",
        "--json",
        "databaseId,headSha,status,conclusion,workflowName,url",
    ]
    try:
        rows = json.loads(subprocess.check_output(command, cwd=ROOT, text=True))
    except (subprocess.CalledProcessError, json.JSONDecodeError) as error:
        raise RuntimeError("could not authenticate GitHub workflow state") from error
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise RuntimeError("GitHub workflow response changed")
    return rows


def query_green_ci(commit: str) -> dict[str, dict[str, Any]]:
    commit = _commit_id(commit)
    rows = _query_ci_rows(commit)
    result = {}
    for workflow in REQUIRED_WORKFLOWS:
        candidates = [
            row
            for row in rows
            if row.get("workflowName") == workflow and row.get("headSha") == commit
        ]
        if not candidates:
            raise RuntimeError(f"required workflow has no run: {workflow}")
        row = max(candidates, key=lambda value: int(value["databaseId"]))
        if row.get("status") != "completed" or row.get("conclusion") != "success":
            raise RuntimeError(f"required workflow is not green: {workflow}")
        result[workflow] = {
            "database_id": int(row["databaseId"]),
            "head_sha": commit,
            "status": "completed",
            "conclusion": "success",
            "url": str(row["url"]),
        }
    return result


def verify_recorded_ci(commit: str, evidence: Mapping[str, Mapping[str, Any]]) -> None:
    commit = _commit_id(commit)
    if set(evidence) != set(REQUIRED_WORKFLOWS):
        raise RuntimeError("recorded workflow inventory changed")
    rows = _query_ci_rows(commit)
    for workflow, recorded in evidence.items():
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
            raise RuntimeError(f"recorded workflow is not green: {workflow}")


def _critical_hashes(commit: str) -> dict[str, str]:
    result = {}
    for relative in CRITICAL_FILES:
        path = ROOT / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"critical implementation file is unsafe: {relative}")
        if _git("ls-files", "--error-unmatch", "--", relative) != relative:
            raise RuntimeError(f"critical implementation file is untracked: {relative}")
        blob = _git_bytes("show", f"{commit}:{relative}")
        if path.read_bytes() != blob:
            raise RuntimeError(f"critical working bytes differ from commit: {relative}")
        result[relative] = sha256_bytes(blob)
    return result


def _blob_inventory(commit: str, files: Sequence[str]) -> dict[str, str]:
    result = {}
    for relative in files:
        blob = _git("rev-parse", f"{commit}:{relative}")
        if len(blob) not in {40, 64}:
            raise RuntimeError(f"invalid frozen Git blob: {relative}")
        result[relative] = blob
    return result


def build_lock_value(
    *,
    implementation_commit: str,
    critical_files: Mapping[str, str],
    frozen_mechanics_blobs: Mapping[str, str],
    inputs: CalibrationInputs,
    ci_evidence: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    implementation_commit = _commit_id(implementation_commit)
    if tuple(critical_files) != CRITICAL_FILES:
        raise RuntimeError("critical implementation allowlist changed")
    if set(frozen_mechanics_blobs) != set(FROZEN_MECHANICS_FILES):
        raise RuntimeError("frozen mechanics inventory changed")
    if tuple(ci_evidence) != REQUIRED_WORKFLOWS:
        raise RuntimeError("implementation CI inventory changed")
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
        "expected_source_rows": EXPECTED_ROWS,
        "expected_answer_pairs": 192,
        "expected_answer_requests": 384,
        "engine": _normalized(dataclasses.asdict(engine_config(inputs))),
        "sampling": {
            name: _normalized(dataclasses.asdict(value))
            for name, value in sampling_configs(inputs).items()
        },
        "implementation_ci": {
            name: dict(value) for name, value in ci_evidence.items()
        },
        "implementation_review_verdict": "PASS_IMPLEMENTATION",
        "experimental_model_requests_before_lock": 0,
        "sampled_model_outputs_before_lock": 0,
        "hidden_files_read": [],
        "qualification_files_read": [],
        "confirmation_files_read": [],
        "benchmark_files_read": [],
    }


def validate_lock_value(lock: Any, *, inputs: CalibrationInputs) -> dict[str, Any]:
    expected_keys = {
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
        "expected_source_rows",
        "expected_answer_pairs",
        "expected_answer_requests",
        "engine",
        "sampling",
        "implementation_ci",
        "implementation_review_verdict",
        "experimental_model_requests_before_lock",
        "sampled_model_outputs_before_lock",
        "hidden_files_read",
        "qualification_files_read",
        "confirmation_files_read",
        "benchmark_files_read",
    }
    expected_sampling = {
        name: _normalized(dataclasses.asdict(value))
        for name, value in sampling_configs(inputs).items()
    }
    if not isinstance(lock, dict) or set(lock) != expected_keys:
        raise RuntimeError("calibration implementation lock schema changed")
    if (
        lock["schema_version"] != 1
        or lock["stage"] != "calibration_implementation_lock"
        or lock["authorization"] != "interface_calibration_only"
        or lock["model"] != MODEL_ID
        or lock["revision"] != MODEL_REVISION
        or lock["calibration_runtime_files"] != list(CALIBRATION_RUNTIME_FILES)
        or lock["calibration_inputs"] != inputs.read_receipt
        or lock["invocation_order"] != list(INVOCATION_ORDER)
        or lock["expected_source_rows"] != 48
        or lock["expected_answer_pairs"] != 192
        or lock["expected_answer_requests"] != 384
        or lock["engine"] != _normalized(dataclasses.asdict(engine_config(inputs)))
        or lock["sampling"] != expected_sampling
        or lock["implementation_review_verdict"] != "PASS_IMPLEMENTATION"
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
    if not isinstance(lock["critical_files"], dict) or set(
        lock["critical_files"]
    ) != set(CRITICAL_FILES):
        raise RuntimeError("calibration critical file inventory changed")
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in lock["critical_files"].values()
    ):
        raise RuntimeError("calibration critical file hash changed")
    if not isinstance(lock["frozen_mechanics_blobs"], dict) or set(
        lock["frozen_mechanics_blobs"]
    ) != set(FROZEN_MECHANICS_FILES):
        raise RuntimeError("calibration frozen mechanics inventory changed")
    if any(
        not isinstance(value, str)
        or len(value) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in value)
        for value in lock["frozen_mechanics_blobs"].values()
    ):
        raise RuntimeError("calibration frozen mechanics blob changed")
    if not isinstance(lock["implementation_ci"], dict) or set(
        lock["implementation_ci"]
    ) != set(REQUIRED_WORKFLOWS):
        raise RuntimeError("calibration implementation CI inventory changed")
    for workflow, row in lock["implementation_ci"].items():
        if (
            not isinstance(row, dict)
            or set(row)
            != {"database_id", "head_sha", "status", "conclusion", "url"}
            or row["head_sha"] != lock["implementation_commit"]
            or row["status"] != "completed"
            or row["conclusion"] != "success"
            or not isinstance(row["database_id"], int)
            or isinstance(row["database_id"], bool)
            or row["database_id"] < 1
            or not isinstance(row["url"], str)
            or not row["url"].startswith("https://github.com/")
        ):
            raise RuntimeError(f"calibration implementation CI changed: {workflow}")
    return lock


def _review_passes() -> None:
    text = (EXP / "reports/calibration_implementation_review.md").read_text()
    if "`PASS_IMPLEMENTATION`" not in text or "`HOLD_LIVE_CALLS`" in text:
        raise RuntimeError("independent implementation review has not passed")


def publish_calibration_lock(path: Path = IMPLEMENTATION_LOCK) -> dict[str, Any]:
    if path.resolve() != IMPLEMENTATION_LOCK.resolve():
        raise RuntimeError("calibration lock path changed")
    if _git("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("publishing a calibration lock requires a clean worktree")
    if any(
        candidate.exists() or candidate.is_symlink()
        for candidate in (IMPLEMENTATION_LOCK, LIVE_PREFLIGHT, RAW_DIR, DECISION)
    ):
        raise RuntimeError("calibration lock must precede every calibration artifact")
    subprocess.run(["git", "fetch", "--quiet", "origin", "main"], cwd=ROOT, check=True)
    commit = _commit_id(_git("rev-parse", "HEAD"))
    if commit != _commit_id(_git("rev-parse", "origin/main")):
        raise RuntimeError("implementation commit must equal current origin/main")
    _review_passes()
    inputs = load_calibration_inputs()
    value = build_lock_value(
        implementation_commit=commit,
        critical_files=_critical_hashes(commit),
        frozen_mechanics_blobs=_blob_inventory(commit, FROZEN_MECHANICS_FILES),
        inputs=inputs,
        ci_evidence=query_green_ci(commit),
    )
    write_exclusive_durable(path, value)
    return value


def verify_calibration_lock(
    path: Path = IMPLEMENTATION_LOCK, *, verify_network: bool = True
) -> dict[str, Any]:
    inputs = load_calibration_inputs()
    lock = validate_lock_value(read_canonical(path), inputs=inputs)
    commit = _commit_id(lock["implementation_commit"])
    for relative, expected in lock["critical_files"].items():
        current = ROOT / relative
        if sha256_file(current) != expected:
            raise RuntimeError(f"critical runtime file changed after lock: {relative}")
        if sha256_bytes(_git_bytes("show", f"{commit}:{relative}")) != expected:
            raise RuntimeError(f"implementation Git blob changed: {relative}")
    for relative, blob in lock["frozen_mechanics_blobs"].items():
        if _git("rev-parse", f"{commit}:{relative}") != blob:
            raise RuntimeError(f"frozen mechanics Git blob changed: {relative}")
    head = _commit_id(_git("rev-parse", "HEAD"))
    if not _ancestor(commit, head):
        raise RuntimeError("current HEAD does not descend from implementation")
    if verify_network:
        subprocess.run(["git", "fetch", "--quiet", "origin", "main"], cwd=ROOT, check=True)
        if not _ancestor(head, _git("rev-parse", "origin/main")):
            raise RuntimeError("current live HEAD is not published on main")
        verify_recorded_ci(commit, lock["implementation_ci"])
        query_green_ci(head)
    allowed_prefix = PREFIX + "runs/calibration/"
    dirty = _git("status", "--porcelain=v1", "--untracked-files=all").splitlines()
    if any(allowed_prefix not in line for line in dirty):
        raise RuntimeError("live worktree has changes outside calibration artifacts")
    return lock


def _prompt_receipt(runner: Any, inputs: CalibrationInputs) -> dict[str, Any]:
    result = {}
    registry = inputs.tokenizer_receipt["calibration_prompt_token_ids"]
    for policy, thinking in (("think512", "budget"), ("no_think", "off")):
        prepared = runner.prepare(inputs.records, thinking, False)
        observed = [row.prompt_token_ids for row in prepared]
        expected = [registry[row.record_id][policy]["token_ids"] for row in prepared]
        if observed != expected:
            raise RuntimeError("live rendered prompts differ from tokenizer receipt")
        result[policy] = {
            "rows": len(prepared),
            "ids": [row.record_id for row in prepared],
            "prompt_token_ids_sha256": canonical_sha256(observed),
            "prompt_tokens_min": min(map(len, observed)),
            "prompt_tokens_max": max(map(len, observed)),
        }
    return result


def _validate_loaded_runner(runner: Any, inputs: CalibrationInputs) -> dict[str, Any]:
    expected_engine = _normalized(dataclasses.asdict(engine_config(inputs)))
    receipt = inputs.tokenizer_receipt
    if (
        _normalized(dataclasses.asdict(runner.config)) != expected_engine
        or runner.engine_args.get("model") != MODEL_ID
        or runner.engine_args.get("revision") != MODEL_REVISION
        or runner.engine_args.get("tokenizer_revision") != MODEL_REVISION
        or runner.engine_args.get("dtype") != "bfloat16"
        or runner.engine_args.get("async_scheduling") is not False
        or runner.resolved_logprobs_mode != "raw_logprobs"
        or runner.hf_eos_id != 248044
        or runner.tokenizer_eos_id != 248046
        or runner.close_ids != receipt["think_token_ids"]["forced_close_sequence"]
        or runner.tokenizer.encode("PROGRAM:", add_special_tokens=False)
        != receipt["program_slot_prefix_token_ids"]
        or runner.adapter_info is not None
    ):
        raise RuntimeError("loaded calibration runner differs from frozen settings")
    expected_sizes = list(engine_config(inputs).cudagraph_capture_sizes or ())
    resolved = runner.resolved_cudagraph
    if (
        resolved.get("cudagraph_capture_sizes") != expected_sizes
        or resolved.get("max_cudagraph_capture_size") != expected_sizes[-1]
        or resolved.get("has_full_cudagraphs") is not True
    ):
        raise RuntimeError("loaded calibration CUDA-graph geometry changed")
    prompts = _prompt_receipt(runner, inputs)
    runtime = runner.runtime_metadata()
    expected_packages = {
        "vllm": inputs.config["model"]["vllm_version"],
        "torch": "2.11.0+cu129",
        "transformers": "5.13.0",
    }
    if (
        not str(runtime.get("python_executable", "")).endswith("/.venv-vllm/bin/python")
        or not runtime.get("gpu")
        or any(
            runtime.get("packages", {}).get(name) != version
            for name, version in expected_packages.items()
        )
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
        "expected_source_rows": 48,
        "expected_answer_pairs": 192,
        "expected_answer_requests": 384,
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
    if (
        value.get("schema_version") != 1
        or value.get("decision") != "CALIBRATION_LIVE_ENGINE_PREFLIGHT_PASS"
        or value.get("model") != MODEL_ID
        or value.get("revision") != MODEL_REVISION
        or value.get("implementation_lock_sha256") != sha256_file(lock_path)
        or value.get("implementation_commit") != lock["implementation_commit"]
        or value.get("engine")
        != _normalized(dataclasses.asdict(engine_config(inputs)))
        or value.get("resolved_logprobs_mode") != "raw_logprobs"
        or value.get("invocation_order") != list(INVOCATION_ORDER)
        or value.get("expected_source_rows") != 48
        or value.get("expected_answer_pairs") != 192
        or value.get("expected_answer_requests") != 384
        or value.get("experimental_generation_requests_before_preflight") != 0
        or value.get("sampled_model_outputs_before_preflight") != 0
        or any(
            value.get(field) != []
            for field in (
                "hidden_files_read",
                "qualification_files_read",
                "confirmation_files_read",
                "benchmark_files_read",
            )
        )
    ):
        raise RuntimeError("recorded calibration live preflight boundary changed")
    live_head = _commit_id(value["live_head"])
    if not _ancestor(lock["implementation_commit"], live_head):
        raise RuntimeError("live preflight predates implementation")
    verify_recorded_ci(live_head, value["live_head_ci"])
    if runner is not None:
        loaded = _validate_loaded_runner(runner, inputs)
        if (
            value["engine_args_sha256"]
            != canonical_sha256(_normalized(runner.engine_args))
            or value["resolved_cudagraph"] != _normalized(runner.resolved_cudagraph)
            or value["prompt_receipt"] != loaded["prompts"]
        ):
            raise RuntimeError("current runner differs from live preflight")
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
            inputs=inputs, lock_path=lock_path, path=path, runner=runner
        )
    value = live_preflight_value(runner=runner, inputs=inputs, lock_path=lock_path)
    write_exclusive_durable(path, value)
    return value
