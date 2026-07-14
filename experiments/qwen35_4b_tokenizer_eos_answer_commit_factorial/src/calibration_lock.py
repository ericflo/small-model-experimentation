"""Published calibration implementation lock and zero-generation live preflight."""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from calibration_stage import (
    EXP,
    EXPECTED_ROWS,
    INVOCATION_ORDER,
    ROOT,
    RUNTIME_METADATA_KEYS,
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
IMPLEMENTATION_REVIEW = EXP / "reports/calibration_implementation_review.json"
IMPLEMENTATION_REVIEW_REPORT = EXP / "reports/calibration_implementation_review.md"
LIVE_PREFLIGHT = EXP / "runs/calibration/live_preflight.json"
RAW_DIR = EXP / "runs/calibration/raw"
DECISION = EXP / "runs/calibration/decision.json"
REQUIRED_WORKFLOWS = ("Validate Repository", "Publish Research Site")
REVIEW_RECEIPT_KEYS = {
    "schema_version",
    "verdict",
    "reviewed_commit",
    "reviewer",
    "review_report_sha256",
    "reviewed_ci",
    "adversarial_review_rounds",
    "experimental_model_requests_reviewed",
    "sampled_model_outputs_reviewed",
    "hidden_files_read",
    "qualification_files_read",
    "confirmation_files_read",
    "benchmark_files_read",
}
LIVE_PREFLIGHT_KEYS = {
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
    "adapter",
    "rng_isolation",
    "prompt_receipt",
    "runtime",
    "invocation_order",
    "expected_source_rows",
    "expected_answer_pairs",
    "expected_answer_requests",
    "experimental_generation_requests_before_preflight",
    "sampled_model_outputs_before_preflight",
    "hidden_files_read",
    "qualification_files_read",
    "confirmation_files_read",
    "benchmark_files_read",
}
PREFIX = "experiments/qwen35_4b_tokenizer_eos_answer_commit_factorial/"
CANONICAL_REPOSITORY = "ericflo/small-model-experimentation"
CANONICAL_ORIGIN = "https://github.com/ericflo/small-model-experimentation.git"
GIT_EXECUTABLE = "/usr/bin/git"
GH_EXECUTABLE = "/usr/bin/gh"
CRITICAL_FILES = (
    "requirements-vllm.lock.txt",
    PREFIX + "configs/default.yaml",
    PREFIX + "reports/design_review.md",
    PREFIX + "reports/preregistration.md",
    PREFIX + "reports/calibration_implementation_review.md",
    PREFIX + "scripts/calibration_launcher",
    PREFIX + "scripts/calibration_launcher.S",
    PREFIX + "runs/prepared/calibration_requests.jsonl",
    PREFIX + "runs/prepared/preoutcome_receipt.json",
    PREFIX + "runs/tokenizer/receipt.json",
    PREFIX + "scripts/run_calibration.py",
    PREFIX + "scripts/tokenizer_receipt.py",
    PREFIX + "src/calibration_lock.py",
    PREFIX + "src/calibration_stage.py",
    PREFIX + "src/identity.py",
    PREFIX + "src/interface_analysis.py",
    PREFIX + "src/process_lock.py",
    PREFIX + "src/protocol.py",
    PREFIX + "src/task_data.py",
    PREFIX + "src/transactions.py",
    PREFIX + "src/vllm_runner.py",
    PREFIX + "tests/test_calibration_lock.py",
    PREFIX + "tests/test_calibration_process_lock.py",
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
    PREFIX + "scripts/calibration_launcher",
    PREFIX + "scripts/calibration_launcher.S",
    PREFIX + "runs/prepared/calibration_requests.jsonl",
    PREFIX + "runs/prepared/preoutcome_receipt.json",
    PREFIX + "runs/tokenizer/receipt.json",
    PREFIX + "scripts/run_calibration.py",
    PREFIX + "src/calibration_lock.py",
    PREFIX + "src/calibration_stage.py",
    PREFIX + "src/identity.py",
    PREFIX + "src/interface_analysis.py",
    PREFIX + "src/process_lock.py",
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


def _child_environment() -> dict[str, str]:
    result = {
        "HOME": "/root",
        "PATH": f"{ROOT}/.venv-vllm/bin:/usr/local/cuda/bin:/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "GH_CONFIG_DIR": "/root/.config/gh",
        "GH_HOST": "github.com",
        "NO_COLOR": "1",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_TERMINAL_PROMPT": "0",
    }
    if os.environ.get("GITHUB_TOKEN"):
        result["GITHUB_TOKEN"] = os.environ["GITHUB_TOKEN"]
    return result


def _git_command(*args: str) -> list[str]:
    return [
        GIT_EXECUTABLE,
        "--no-replace-objects",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "core.fsmonitor=false",
        *args,
    ]


def _git(*args: str) -> str:
    return subprocess.check_output(
        _git_command(*args), cwd=ROOT, text=True, env=_child_environment()
    ).strip()


def _git_bytes(*args: str) -> bytes:
    return subprocess.check_output(
        _git_command(*args), cwd=ROOT, env=_child_environment()
    )


def _commit_id(value: Any) -> str:
    commit = str(value)
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise RuntimeError("implementation boundary has an invalid commit ID")
    return commit


def _exact_int(value: Any, expected: int) -> bool:
    return type(value) is int and value == expected


def _ancestor(older: str, newer: str) -> bool:
    return (
        subprocess.run(
            _git_command("merge-base", "--is-ancestor", older, newer),
            cwd=ROOT,
            capture_output=True,
            env=_child_environment(),
        ).returncode
        == 0
    )


def _normalized(value: Any) -> Any:
    return json.loads(
        json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False)
    )


def _query_ci_rows(commit: str) -> list[dict[str, Any]]:
    command = [
        GH_EXECUTABLE,
        "run",
        "list",
        "--repo",
        CANONICAL_REPOSITORY,
        "--commit",
        _commit_id(commit),
        "--limit",
        "20",
        "--json",
        "databaseId,headSha,status,conclusion,workflowName,url",
    ]
    try:
        rows = json.loads(
            subprocess.check_output(
                command, cwd=ROOT, text=True, env=_child_environment()
            )
        )
    except (subprocess.CalledProcessError, json.JSONDecodeError) as error:
        raise RuntimeError("could not authenticate GitHub workflow state") from error
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise RuntimeError("GitHub workflow response changed")
    return rows


def _fetch_main() -> None:
    if _git("remote", "get-url", "origin") != CANONICAL_ORIGIN:
        raise RuntimeError("Git origin URL changed")
    subprocess.run(
        _git_command(
            "fetch",
            "--quiet",
            "--no-tags",
            CANONICAL_ORIGIN,
            "+refs/heads/main:refs/remotes/origin/main",
        ),
        cwd=ROOT,
        check=True,
        env=_child_environment(),
    )


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


def _validate_ci_evidence(
    commit: str, evidence: Any, *, label: str
) -> dict[str, Mapping[str, Any]]:
    commit = _commit_id(commit)
    if not isinstance(evidence, dict) or set(evidence) != set(REQUIRED_WORKFLOWS):
        raise RuntimeError(f"{label} workflow inventory changed")
    for workflow, row in evidence.items():
        if (
            not isinstance(row, dict)
            or set(row)
            != {"database_id", "head_sha", "status", "conclusion", "url"}
            or row["head_sha"] != commit
            or row["status"] != "completed"
            or row["conclusion"] != "success"
            or not isinstance(row["database_id"], int)
            or isinstance(row["database_id"], bool)
            or row["database_id"] < 1
            or not isinstance(row["url"], str)
            or not row["url"].startswith("https://github.com/")
        ):
            raise RuntimeError(f"{label} workflow evidence changed: {workflow}")
    return evidence


def _tracked_commit(relative: str) -> str:
    commit = _git("log", "-1", "--format=%H", "--", relative)
    if not commit:
        raise RuntimeError(f"required artifact is not committed: {relative}")
    return _commit_id(commit)


def validate_implementation_review(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != REVIEW_RECEIPT_KEYS:
        raise RuntimeError("implementation review receipt schema changed")
    reviewed_commit = _commit_id(value["reviewed_commit"])
    if (
        not _exact_int(value["schema_version"], 1)
        or value["verdict"] != "PASS_IMPLEMENTATION"
        or not isinstance(value["reviewer"], str)
        or not value["reviewer"].strip()
        or not isinstance(value["review_report_sha256"], str)
        or len(value["review_report_sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in value["review_report_sha256"]
        )
        or not isinstance(value["adversarial_review_rounds"], int)
        or isinstance(value["adversarial_review_rounds"], bool)
        or value["adversarial_review_rounds"] < 3
        or not _exact_int(value["experimental_model_requests_reviewed"], 0)
        or not _exact_int(value["sampled_model_outputs_reviewed"], 0)
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
        raise RuntimeError("implementation review receipt boundary changed")
    _validate_ci_evidence(
        reviewed_commit, value["reviewed_ci"], label="reviewed implementation"
    )
    return value


def authenticate_implementation_review(
    path: Path = IMPLEMENTATION_REVIEW, *, verify_network: bool = True
) -> dict[str, Any]:
    value = validate_implementation_review(read_canonical(path))
    reviewed_commit = _commit_id(value["reviewed_commit"])
    relative = str(path.relative_to(ROOT))
    receipt_commit = _tracked_commit(relative)
    if path.read_bytes() != _git_bytes("show", f"{receipt_commit}:{relative}"):
        raise RuntimeError("implementation review receipt differs from committed bytes")
    report_relative = str(IMPLEMENTATION_REVIEW_REPORT.relative_to(ROOT))
    report_blob = _git_bytes("show", f"{receipt_commit}:{report_relative}")
    if (
        sha256_bytes(report_blob) != value["review_report_sha256"]
        or IMPLEMENTATION_REVIEW_REPORT.read_bytes() != report_blob
    ):
        raise RuntimeError("implementation review report differs from receipt")
    head = _commit_id(_git("rev-parse", "HEAD"))
    if not _ancestor(reviewed_commit, receipt_commit) or not _ancestor(
        receipt_commit, head
    ):
        raise RuntimeError("implementation review commit ancestry changed")
    for relative in CALIBRATION_RUNTIME_FILES:
        current = ROOT / relative
        reviewed_blob = _git_bytes("show", f"{reviewed_commit}:{relative}")
        if current.is_symlink() or current.read_bytes() != reviewed_blob:
            raise RuntimeError(
                f"reviewed calibration runtime changed after review: {relative}"
            )
    if verify_network:
        _fetch_main()
        origin = _commit_id(_git("rev-parse", "origin/main"))
        if not _ancestor(receipt_commit, origin) or not _ancestor(head, origin):
            raise RuntimeError("implementation review is not published on main")
        verify_recorded_ci(reviewed_commit, value["reviewed_ci"])
        query_green_ci(receipt_commit)
        query_green_ci(head)
    return {
        **value,
        "receipt_commit": receipt_commit,
        "receipt_sha256": sha256_file(path),
    }


def _critical_hashes(commit: str) -> dict[str, str]:
    result = {}
    for relative in CRITICAL_FILES:
        path = ROOT / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"critical implementation file is unsafe: {relative}")
        if _git("ls-files", "--error-unmatch", "--", relative) != relative:
            raise RuntimeError(f"critical implementation file is untracked: {relative}")
        blob = _git_bytes("show", f"{commit}:{relative}")
        if relative in CALIBRATION_RUNTIME_FILES and path.read_bytes() != blob:
            raise RuntimeError(
                f"reviewed runtime bytes differ from implementation: {relative}"
            )
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
    review_receipt: Mapping[str, Any],
    review_receipt_sha256: str,
    review_receipt_commit: str,
    release_commit: str,
    release_ci: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    implementation_commit = _commit_id(implementation_commit)
    review = validate_implementation_review(dict(review_receipt))
    review_receipt_commit = _commit_id(review_receipt_commit)
    release_commit = _commit_id(release_commit)
    if tuple(critical_files) != CRITICAL_FILES:
        raise RuntimeError("critical implementation allowlist changed")
    if set(frozen_mechanics_blobs) != set(FROZEN_MECHANICS_FILES):
        raise RuntimeError("frozen mechanics inventory changed")
    if implementation_commit != review["reviewed_commit"]:
        raise RuntimeError("implementation commit differs from review receipt")
    if (
        not isinstance(review_receipt_sha256, str)
        or len(review_receipt_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in review_receipt_sha256
        )
    ):
        raise RuntimeError("implementation review receipt hash changed")
    _validate_ci_evidence(release_commit, release_ci, label="release")
    return {
        "schema_version": 2,
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
            name: dict(value) for name, value in review["reviewed_ci"].items()
        },
        "implementation_review": {
            "verdict": review["verdict"],
            "reviewer": review["reviewer"],
            "reviewed_commit": review["reviewed_commit"],
            "receipt_sha256": review_receipt_sha256,
            "receipt_commit": review_receipt_commit,
        },
        "release_commit": release_commit,
        "release_ci": {name: dict(value) for name, value in release_ci.items()},
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
        "implementation_review",
        "release_commit",
        "release_ci",
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
    expected_engine = _normalized(dataclasses.asdict(engine_config(inputs)))
    if not isinstance(lock, dict) or set(lock) != expected_keys:
        raise RuntimeError("calibration implementation lock schema changed")
    if (
        not _exact_int(lock["schema_version"], 2)
        or lock["stage"] != "calibration_implementation_lock"
        or lock["authorization"] != "interface_calibration_only"
        or lock["model"] != MODEL_ID
        or lock["revision"] != MODEL_REVISION
        or lock["calibration_runtime_files"] != list(CALIBRATION_RUNTIME_FILES)
        or lock["calibration_inputs"] != inputs.read_receipt
        or lock["invocation_order"] != list(INVOCATION_ORDER)
        or not _exact_int(lock["expected_source_rows"], 48)
        or not _exact_int(lock["expected_answer_pairs"], 192)
        or not _exact_int(lock["expected_answer_requests"], 384)
        or canonical_sha256(lock["engine"]) != canonical_sha256(expected_engine)
        or canonical_sha256(lock["sampling"])
        != canonical_sha256(expected_sampling)
        or not _exact_int(lock["experimental_model_requests_before_lock"], 0)
        or not _exact_int(lock["sampled_model_outputs_before_lock"], 0)
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
    implementation_commit = _commit_id(lock["implementation_commit"])
    release_commit = _commit_id(lock["release_commit"])
    review = lock["implementation_review"]
    if (
        not isinstance(review, dict)
        or set(review)
        != {
            "verdict",
            "reviewer",
            "reviewed_commit",
            "receipt_sha256",
            "receipt_commit",
        }
        or review["verdict"] != "PASS_IMPLEMENTATION"
        or not isinstance(review["reviewer"], str)
        or not review["reviewer"].strip()
        or review["reviewed_commit"] != implementation_commit
        or _commit_id(review["receipt_commit"]) != review["receipt_commit"]
        or not isinstance(review["receipt_sha256"], str)
        or len(review["receipt_sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in review["receipt_sha256"]
        )
    ):
        raise RuntimeError("calibration implementation review binding changed")
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
    _validate_ci_evidence(
        implementation_commit,
        lock["implementation_ci"],
        label="calibration implementation",
    )
    _validate_ci_evidence(release_commit, lock["release_ci"], label="calibration release")
    return lock


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
    _fetch_main()
    release_commit = _commit_id(_git("rev-parse", "HEAD"))
    if release_commit != _commit_id(_git("rev-parse", "origin/main")):
        raise RuntimeError("release commit must equal current origin/main")
    review = authenticate_implementation_review()
    implementation_commit = _commit_id(review["reviewed_commit"])
    inputs = load_calibration_inputs()
    value = build_lock_value(
        implementation_commit=implementation_commit,
        critical_files=_critical_hashes(implementation_commit),
        frozen_mechanics_blobs=_blob_inventory(
            release_commit, FROZEN_MECHANICS_FILES
        ),
        inputs=inputs,
        review_receipt={key: review[key] for key in REVIEW_RECEIPT_KEYS},
        review_receipt_sha256=review["receipt_sha256"],
        review_receipt_commit=review["receipt_commit"],
        release_commit=release_commit,
        release_ci=query_green_ci(release_commit),
    )
    write_exclusive_durable(path, value)
    return value


def verify_calibration_lock(
    path: Path = IMPLEMENTATION_LOCK, *, verify_network: bool = True
) -> dict[str, Any]:
    inputs = load_calibration_inputs()
    lock = validate_lock_value(read_canonical(path), inputs=inputs)
    commit = _commit_id(lock["implementation_commit"])
    review = authenticate_implementation_review(verify_network=verify_network)
    expected_review = lock["implementation_review"]
    if (
        review["verdict"] != expected_review["verdict"]
        or review["reviewer"] != expected_review["reviewer"]
        or review["reviewed_commit"] != expected_review["reviewed_commit"]
        or review["receipt_sha256"] != expected_review["receipt_sha256"]
        or review["receipt_commit"] != expected_review["receipt_commit"]
    ):
        raise RuntimeError("implementation review no longer matches lock")
    for relative, expected in lock["critical_files"].items():
        current = ROOT / relative
        if relative in CALIBRATION_RUNTIME_FILES and sha256_file(current) != expected:
            raise RuntimeError(f"critical runtime file changed after lock: {relative}")
        if sha256_bytes(_git_bytes("show", f"{commit}:{relative}")) != expected:
            raise RuntimeError(f"implementation Git blob changed: {relative}")
    head = _commit_id(_git("rev-parse", "HEAD"))
    for relative, blob in lock["frozen_mechanics_blobs"].items():
        if _git("rev-parse", f"{head}:{relative}") != blob:
            raise RuntimeError(f"frozen mechanics Git blob changed: {relative}")
    lock_relative = str(path.relative_to(ROOT))
    lock_commit = _tracked_commit(lock_relative)
    if path.read_bytes() != _git_bytes("show", f"{lock_commit}:{lock_relative}"):
        raise RuntimeError("calibration lock differs from committed bytes")
    release_commit = _commit_id(lock["release_commit"])
    if not _ancestor(commit, release_commit) or not _ancestor(
        release_commit, lock_commit
    ) or not _ancestor(lock_commit, head):
        raise RuntimeError("calibration lock commit ancestry changed")
    if verify_network:
        _fetch_main()
        origin = _commit_id(_git("rev-parse", "origin/main"))
        if not _ancestor(head, origin):
            raise RuntimeError("current live HEAD is not published on main")
        verify_recorded_ci(commit, lock["implementation_ci"])
        verify_recorded_ci(release_commit, lock["release_ci"])
        query_green_ci(lock_commit)
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
    if (
        loaded["runtime"].get("git_commit") != head
        or loaded["runtime"].get("git_dirty") is not False
    ):
        raise RuntimeError("live preflight requires the clean current Git commit")
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
        "adapter": runner.adapter_info,
        "rng_isolation": {
            "engine_seed": runner.engine_args["seed"],
            "caller_global_rng_state_restored": True,
        },
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


def validate_live_preflight_value(
    value: Any,
    *,
    inputs: CalibrationInputs,
    implementation_commit: str,
    implementation_lock_sha256: str,
) -> dict[str, Any]:
    rng = value.get("rng_isolation") if isinstance(value, dict) else None
    valid_rng = (
        isinstance(rng, dict)
        and set(rng) == {"engine_seed", "caller_global_rng_state_restored"}
        and type(rng["engine_seed"]) is int
        and rng["engine_seed"] == 0
        and rng["caller_global_rng_state_restored"] is True
    )
    if (
        not isinstance(value, dict)
        or set(value) != LIVE_PREFLIGHT_KEYS
        or not _exact_int(value.get("schema_version"), 1)
        or value.get("decision") != "CALIBRATION_LIVE_ENGINE_PREFLIGHT_PASS"
        or value.get("model") != MODEL_ID
        or value.get("revision") != MODEL_REVISION
        or value.get("implementation_lock_sha256")
        != implementation_lock_sha256
        or value.get("implementation_commit") != implementation_commit
        or canonical_sha256(value.get("engine"))
        != canonical_sha256(_normalized(dataclasses.asdict(engine_config(inputs))))
        or not isinstance(value.get("engine_args_sha256"), str)
        or len(value["engine_args_sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in value["engine_args_sha256"]
        )
        or not isinstance(value.get("resolved_cudagraph"), dict)
        or value.get("resolved_logprobs_mode") != "raw_logprobs"
        or value.get("adapter") is not None
        or not valid_rng
        or not isinstance(value.get("prompt_receipt"), dict)
        or not isinstance(value.get("runtime"), dict)
        or set(value["runtime"]) != RUNTIME_METADATA_KEYS
        or value["runtime"].get("git_commit") != value.get("live_head")
        or value["runtime"].get("git_dirty") is not False
        or value.get("invocation_order") != list(INVOCATION_ORDER)
        or not _exact_int(value.get("expected_source_rows"), 48)
        or not _exact_int(value.get("expected_answer_pairs"), 192)
        or not _exact_int(value.get("expected_answer_requests"), 384)
        or not _exact_int(
            value.get("experimental_generation_requests_before_preflight"), 0
        )
        or not _exact_int(value.get("sampled_model_outputs_before_preflight"), 0)
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
    _validate_ci_evidence(live_head, value["live_head_ci"], label="live preflight")
    return value


def authenticate_live_preflight_ancestry(
    *, lock_commit: str, live_head: str, current_head: str
) -> None:
    lock_commit = _commit_id(lock_commit)
    live_head = _commit_id(live_head)
    current_head = _commit_id(current_head)
    if not _ancestor(lock_commit, live_head) or not _ancestor(
        live_head, current_head
    ):
        raise RuntimeError("live preflight Git ancestry changed")


def verify_recorded_live_preflight(
    *,
    inputs: CalibrationInputs,
    lock_path: Path = IMPLEMENTATION_LOCK,
    path: Path = LIVE_PREFLIGHT,
    runner: Any | None = None,
) -> dict[str, Any]:
    lock = verify_calibration_lock(lock_path)
    value = validate_live_preflight_value(
        read_canonical(path),
        inputs=inputs,
        implementation_commit=lock["implementation_commit"],
        implementation_lock_sha256=sha256_file(lock_path),
    )
    live_head = _commit_id(value["live_head"])
    lock_commit = _tracked_commit(str(lock_path.relative_to(ROOT)))
    current_head = _commit_id(_git("rev-parse", "HEAD"))
    authenticate_live_preflight_ancestry(
        lock_commit=lock_commit,
        live_head=live_head,
        current_head=current_head,
    )
    _validate_ci_evidence(live_head, value["live_head_ci"], label="live preflight")
    verify_recorded_ci(live_head, value["live_head_ci"])
    if runner is not None:
        loaded = _validate_loaded_runner(runner, inputs)
        stable_runtime_keys = RUNTIME_METADATA_KEYS - {"git_dirty"}
        if (
            value["engine_args_sha256"]
            != canonical_sha256(_normalized(runner.engine_args))
            or value["resolved_cudagraph"] != _normalized(runner.resolved_cudagraph)
            or value["adapter"] != runner.adapter_info
            or value["rng_isolation"]
            != {
                "engine_seed": runner.engine_args["seed"],
                "caller_global_rng_state_restored": True,
            }
            or value["prompt_receipt"] != loaded["prompts"]
            or any(
                value["runtime"].get(key) != loaded["runtime"].get(key)
                for key in stable_runtime_keys
            )
            or value["runtime"].get("git_dirty") is not False
            or loaded["runtime"].get("git_dirty") is not True
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
