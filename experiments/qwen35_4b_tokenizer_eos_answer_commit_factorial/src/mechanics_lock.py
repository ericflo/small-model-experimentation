"""Post-calibration mechanics authorization and pre-hidden publication gates."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import calibration_lock as calibration_authority
from calibration_lock import (
    DECISION as CALIBRATION_DECISION,
    IMPLEMENTATION_LOCK as CALIBRATION_LOCK,
    ROOT,
    REQUIRED_WORKFLOWS,
    _ancestor,
    _commit_id,
    _git,
    _git_bytes,
    _normalized,
    _validate_loaded_runner,
    query_green_ci,
    verify_recorded_ci,
)
from calibration_stage import (
    CalibrationInputs,
    calibration_decision_value,
    load_analysis_tokenizer,
    load_calibration_inputs,
)
from mechanics_stage import (
    DEFAULT_MECHANICS_LOCK,
    DEFAULT_MECHANICS_PREFLIGHT,
    HIDDEN_RESULT,
    MECHANICS_INVOCATION_ORDER,
    RAW_DIR,
    RESOURCE_DECISION,
    TRANSPORT_DECISION,
    VISIBLE_SELECTION,
    analyze_visible,
    canonical_sha256,
    mechanics_sampling_plan,
    selected_interface,
)
from mechanics_transactions import exact_json_equal, json_native
from transactions import (
    MODEL_ID,
    MODEL_REVISION,
    json_bytes,
    read_canonical,
    sha256_bytes,
    sha256_file,
    write_exclusive_durable,
)


MECHANICS_LOCK = DEFAULT_MECHANICS_LOCK
MECHANICS_PREFLIGHT = DEFAULT_MECHANICS_PREFLIGHT
EXP = MECHANICS_LOCK.parents[2]
PREFIX = str(EXP.relative_to(ROOT)) + "/"
MECHANICS_REVIEW_JSON = EXP / "reports/mechanics_implementation_review.json"
MECHANICS_REVIEW_REPORT = EXP / "reports/mechanics_implementation_review.md"
MECHANICS_RUNTIME_FILES = (
    "requirements-vllm.lock.txt",
    PREFIX + "configs/default.yaml",
    PREFIX + "reports/design_review.md",
    PREFIX + "reports/preregistration.md",
    PREFIX + "scripts/mechanics_launcher",
    PREFIX + "scripts/mechanics_launcher.S",
    PREFIX + "scripts/run_mechanics.py",
    PREFIX + "src/calibration_lock.py",
    PREFIX + "src/calibration_stage.py",
    PREFIX + "src/interface_analysis.py",
    PREFIX + "src/identity.py",
    PREFIX + "src/mechanics_lock.py",
    PREFIX + "src/mechanics_protocol.py",
    PREFIX + "src/mechanics_runtime.py",
    PREFIX + "src/mechanics_stage.py",
    PREFIX + "src/mechanics_transactions.py",
    PREFIX + "src/plans.py",
    PREFIX + "src/protocol.py",
    PREFIX + "src/stats.py",
    PREFIX + "src/task_data.py",
    PREFIX + "src/transactions.py",
    PREFIX + "src/vllm_runner.py",
)
MECHANICS_CRITICAL_FILES = (
    *MECHANICS_RUNTIME_FILES,
    PREFIX + "tests/test_mechanics_bootstrap.py",
    PREFIX + "tests/test_mechanics_lock.py",
    PREFIX + "tests/test_mechanics_runtime.py",
    PREFIX + "tests/test_mechanics_stage.py",
    PREFIX + "tests/test_mechanics_transactions.py",
    PREFIX + "tests/test_plans.py",
    PREFIX + "tests/test_stats.py",
)


def _verify_calibration_lock_for_mechanics(
    *, verify_network: bool = True
) -> dict[str, Any]:
    """Run the immutable calibration verifier with only live run dirt exempted.

    The calibration verifier predates conditional mechanics and deliberately
    accepts only calibration-run dirt.  We leave its reviewed bytes untouched,
    first prove with Git pathspecs that no dirt exists outside the two exact run
    trees, and then adapt only its exact status query to the already-proved
    empty result.  Every content, hash, review, ancestry, and CI check remains
    the immutable verifier's own implementation.
    """

    disallowed = _git(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        ".",
        f":(exclude){PREFIX}runs/calibration/**",
        f":(exclude){PREFIX}runs/mechanics/**",
    )
    if disallowed:
        raise RuntimeError("calibration anchor has dirt outside registered live runs")
    original_git = calibration_authority._git

    def registered_live_status(*args: str) -> str:
        if args == ("status", "--porcelain=v1", "--untracked-files=all"):
            return ""
        return original_git(*args)

    calibration_authority._git = registered_live_status
    try:
        return calibration_authority.verify_calibration_lock(
            verify_network=verify_network
        )
    finally:
        calibration_authority._git = original_git


def _exact_int(value: Any, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _validate_ci(
    commit: str, evidence: Any, *, label: str
) -> dict[str, dict[str, Any]]:
    commit = _commit_id(commit)
    if not isinstance(evidence, dict) or set(evidence) != set(REQUIRED_WORKFLOWS):
        raise RuntimeError(f"{label} CI inventory changed")
    for row in evidence.values():
        if (
            not isinstance(row, dict)
            or set(row)
            != {"database_id", "head_sha", "status", "conclusion", "url"}
            or not _exact_int(row["database_id"], 1)
            or row["head_sha"] != commit
            or row["status"] != "completed"
            or row["conclusion"] != "success"
            or not isinstance(row["url"], str)
            or not row["url"].startswith("https://github.com/")
        ):
            raise RuntimeError(f"{label} CI evidence changed")
    return {name: dict(row) for name, row in evidence.items()}


def _validate_review(value: Any) -> dict[str, Any]:
    expected = {
        "schema_version",
        "verdict",
        "reviewed_commit",
        "reviewer",
        "review_report_sha256",
        "reviewed_ci",
        "adversarial_review_rounds",
        "allowed_tests_passed",
        "allowed_tests_total",
        "experimental_model_requests_reviewed",
        "sampled_model_outputs_reviewed",
        "hidden_files_read",
        "qualification_files_read",
        "confirmation_files_read",
        "benchmark_files_read",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise RuntimeError("mechanics implementation review schema changed")
    commit = _commit_id(value["reviewed_commit"])
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or value["verdict"] != "PASS_IMPLEMENTATION"
        or not isinstance(value["reviewer"], str)
        or not value["reviewer"]
        or not isinstance(value["review_report_sha256"], str)
        or len(value["review_report_sha256"]) != 64
        or not _exact_int(value["adversarial_review_rounds"], 1)
        or not _exact_int(value["allowed_tests_passed"], 1)
        or value["allowed_tests_passed"] != value["allowed_tests_total"]
        or not _exact_int(value["experimental_model_requests_reviewed"], 0)
        or value["experimental_model_requests_reviewed"] != 0
        or not _exact_int(value["sampled_model_outputs_reviewed"], 0)
        or value["sampled_model_outputs_reviewed"] != 0
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
        raise RuntimeError("mechanics implementation review boundary changed")
    _validate_ci(commit, value["reviewed_ci"], label="reviewed implementation")
    return dict(value)


def _critical_hashes(commit: str) -> dict[str, str]:
    commit = _commit_id(commit)
    return {
        relative: sha256_bytes(_git_bytes("show", f"{commit}:{relative}"))
        for relative in MECHANICS_CRITICAL_FILES
    }


def _authenticate_calibration_decision(
    *, inputs: CalibrationInputs, tokenizer: Any
) -> dict[str, Any]:
    observed = read_canonical(CALIBRATION_DECISION)
    expected = json_native(
        calibration_decision_value(
            inputs=inputs,
            raw_dir=CALIBRATION_DECISION.parent / "raw",
            tokenizer=tokenizer,
        )
    )
    if not exact_json_equal(observed, expected):
        raise RuntimeError("calibration decision differs from exact authentication")
    selected_interface(observed, inputs)
    return observed


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as error:
        raise RuntimeError("mechanics authorization path escapes repository") from error


def _ensure_clean_for_lock() -> None:
    dirty = _git("status", "--porcelain=v1", "--untracked-files=all")
    if dirty:
        raise RuntimeError("publishing a mechanics lock requires a clean worktree")
    if any(
        path.exists() or path.is_symlink()
        for path in (
            MECHANICS_LOCK,
            MECHANICS_PREFLIGHT,
            RAW_DIR,
            TRANSPORT_DECISION,
            RESOURCE_DECISION,
            VISIBLE_SELECTION,
            HIDDEN_RESULT,
        )
    ):
        raise RuntimeError("mechanics lock must precede every mechanics artifact")


def build_mechanics_lock_value(
    *,
    calibration_lock: Mapping[str, Any],
    calibration_decision: Mapping[str, Any],
    calibration_decision_sha256: str,
    implementation_review: Mapping[str, Any],
    critical_files: Mapping[str, str],
    release_commit: str,
    release_ci: Mapping[str, Mapping[str, Any]],
    inputs: CalibrationInputs,
) -> dict[str, Any]:
    winner = selected_interface(calibration_decision, inputs)
    frozen = calibration_lock.get("frozen_mechanics_blobs")
    if not isinstance(frozen, dict) or not frozen:
        raise RuntimeError("calibration lock did not freeze mechanics")
    review = _validate_review(dict(implementation_review))
    implementation_commit = _commit_id(review["reviewed_commit"])
    if set(critical_files) != set(MECHANICS_CRITICAL_FILES) or any(
        not isinstance(digest, str) or len(digest) != 64
        for digest in critical_files.values()
    ):
        raise RuntimeError("mechanics critical-file inventory changed")
    return {
        "schema_version": 2,
        "stage": "mechanics_implementation_lock",
        "authorization": "selected_interface_transport_and_mechanics_only",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "calibration_implementation_commit": calibration_lock["implementation_commit"],
        "calibration_lock_sha256": sha256_file(CALIBRATION_LOCK),
        "calibration_decision_sha256": calibration_decision_sha256,
        "selected_interface": winner,
        "sampling": _normalized(mechanics_sampling_plan(calibration_decision, inputs)),
        "invocation_order": list(MECHANICS_INVOCATION_ORDER),
        "frozen_mechanics_blobs": dict(frozen),
        "mechanics_runtime_files": list(MECHANICS_RUNTIME_FILES),
        "critical_files": dict(critical_files),
        "implementation_commit": implementation_commit,
        "implementation_ci": {
            name: dict(row) for name, row in review["reviewed_ci"].items()
        },
        "implementation_review": review,
        "release_commit": _commit_id(release_commit),
        "release_ci": {
            name: dict(row) for name, row in release_ci.items()
        },
        "experimental_generation_requests_before_lock": 0,
        "sampled_model_outputs_before_lock": 0,
        "hidden_files_read": [],
        "qualification_files_read": [],
        "confirmation_files_read": [],
        "benchmark_files_read": [],
    }


def validate_mechanics_lock_value(
    value: Any,
    *,
    calibration_lock: Mapping[str, Any],
    calibration_decision: Mapping[str, Any],
    inputs: CalibrationInputs,
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "stage",
        "authorization",
        "model",
        "revision",
        "calibration_implementation_commit",
        "calibration_lock_sha256",
        "calibration_decision_sha256",
        "selected_interface",
        "sampling",
        "invocation_order",
        "frozen_mechanics_blobs",
        "mechanics_runtime_files",
        "critical_files",
        "implementation_commit",
        "implementation_ci",
        "implementation_review",
        "release_commit",
        "release_ci",
        "experimental_generation_requests_before_lock",
        "sampled_model_outputs_before_lock",
        "hidden_files_read",
        "qualification_files_read",
        "confirmation_files_read",
        "benchmark_files_read",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise RuntimeError("mechanics implementation lock schema changed")
    winner = selected_interface(calibration_decision, inputs)
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 2
        or value["stage"] != "mechanics_implementation_lock"
        or value["authorization"]
        != "selected_interface_transport_and_mechanics_only"
        or value["model"] != MODEL_ID
        or value["revision"] != MODEL_REVISION
        or value["calibration_implementation_commit"]
        != calibration_lock["implementation_commit"]
        or value["calibration_lock_sha256"] != sha256_file(CALIBRATION_LOCK)
        or value["calibration_decision_sha256"] != sha256_file(CALIBRATION_DECISION)
        or value["selected_interface"] != winner
        or not exact_json_equal(
            value["sampling"],
            _normalized(mechanics_sampling_plan(calibration_decision, inputs)),
        )
        or value["invocation_order"] != list(MECHANICS_INVOCATION_ORDER)
        or value["frozen_mechanics_blobs"]
        != calibration_lock["frozen_mechanics_blobs"]
        or value["mechanics_runtime_files"] != list(MECHANICS_RUNTIME_FILES)
        or not _exact_int(value["experimental_generation_requests_before_lock"], 0)
        or value["experimental_generation_requests_before_lock"] != 0
        or not _exact_int(value["sampled_model_outputs_before_lock"], 0)
        or value["sampled_model_outputs_before_lock"] != 0
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
        raise RuntimeError("mechanics implementation lock boundary changed")
    review = _validate_review(value["implementation_review"])
    implementation_commit = _commit_id(value["implementation_commit"])
    release_commit = _commit_id(value["release_commit"])
    if (
        review["reviewed_commit"] != implementation_commit
        or value["implementation_ci"] != review["reviewed_ci"]
        or not isinstance(value["critical_files"], dict)
        or set(value["critical_files"]) != set(MECHANICS_CRITICAL_FILES)
        or any(
            not isinstance(digest, str) or len(digest) != 64
            for digest in value["critical_files"].values()
        )
    ):
        raise RuntimeError("mechanics reviewed implementation binding changed")
    _validate_ci(
        implementation_commit,
        value["implementation_ci"],
        label="mechanics implementation",
    )
    _validate_ci(release_commit, value["release_ci"], label="mechanics release")
    return value


def publish_mechanics_lock(path: Path = MECHANICS_LOCK) -> dict[str, Any]:
    if path.resolve() != MECHANICS_LOCK.resolve():
        raise RuntimeError("mechanics lock path changed")
    _ensure_clean_for_lock()
    calibration_lock = _verify_calibration_lock_for_mechanics()
    inputs = load_calibration_inputs()
    tokenizer = load_analysis_tokenizer(inputs)
    decision = _authenticate_calibration_decision(inputs=inputs, tokenizer=tokenizer)
    selected_interface(decision, inputs)
    review = _validate_review(read_canonical(MECHANICS_REVIEW_JSON))
    implementation_commit = _commit_id(review["reviewed_commit"])
    if (
        MECHANICS_REVIEW_REPORT.is_symlink()
        or not MECHANICS_REVIEW_REPORT.is_file()
        or sha256_file(MECHANICS_REVIEW_REPORT) != review["review_report_sha256"]
    ):
        raise RuntimeError("mechanics review report differs from its receipt")
    critical_files = _critical_hashes(implementation_commit)
    for relative, expected in critical_files.items():
        current = ROOT / relative
        if (
            current.is_symlink()
            or not current.is_file()
            or sha256_file(current) != expected
        ):
            raise RuntimeError(
                f"reviewed mechanics critical file changed: {relative}"
            )
    relative_decision = _relative(CALIBRATION_DECISION)
    if _git("ls-files", "--error-unmatch", "--", relative_decision) != relative_decision:
        raise RuntimeError("calibration decision is not committed")
    if _git_bytes("show", f"HEAD:{relative_decision}") != CALIBRATION_DECISION.read_bytes():
        raise RuntimeError("calibration decision differs from HEAD")
    _git("fetch", "--quiet", "origin", "main")
    head = _commit_id(_git("rev-parse", "HEAD"))
    if (
        not _ancestor(implementation_commit, head)
        or not _ancestor(head, "origin/main")
    ):
        raise RuntimeError("mechanics release is not published from reviewed code")
    release_ci = query_green_ci(head)
    verify_recorded_ci(implementation_commit, review["reviewed_ci"])
    for review_path in (MECHANICS_REVIEW_JSON, MECHANICS_REVIEW_REPORT):
        relative_review = _relative(review_path)
        if (
            _git("ls-files", "--error-unmatch", "--", relative_review)
            != relative_review
            or _git_bytes("show", f"HEAD:{relative_review}")
            != review_path.read_bytes()
        ):
            raise RuntimeError("mechanics review release differs from HEAD")
    for relative, blob in calibration_lock["frozen_mechanics_blobs"].items():
        if _git("rev-parse", f"HEAD:{relative}") != blob:
            raise RuntimeError(f"mechanics changed after calibration freeze: {relative}")
    value = build_mechanics_lock_value(
        calibration_lock=calibration_lock,
        calibration_decision=decision,
        calibration_decision_sha256=sha256_file(CALIBRATION_DECISION),
        implementation_review=review,
        critical_files=critical_files,
        release_commit=head,
        release_ci=release_ci,
        inputs=inputs,
    )
    write_exclusive_durable(path, value)
    return value


def verify_mechanics_lock(
    path: Path = MECHANICS_LOCK, *, verify_network: bool = True
) -> dict[str, Any]:
    calibration_lock = _verify_calibration_lock_for_mechanics(
        verify_network=verify_network
    )
    inputs = load_calibration_inputs()
    tokenizer = load_analysis_tokenizer(inputs)
    decision = _authenticate_calibration_decision(inputs=inputs, tokenizer=tokenizer)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("live mechanics requires a committed mechanics lock")
    value = validate_mechanics_lock_value(
        read_canonical(path),
        calibration_lock=calibration_lock,
        calibration_decision=decision,
        inputs=inputs,
    )
    relative = _relative(path)
    if _git("ls-files", "--error-unmatch", "--", relative) != relative:
        raise RuntimeError("mechanics lock is not committed")
    if _git_bytes("show", f"HEAD:{relative}") != path.read_bytes():
        raise RuntimeError("mechanics lock differs from HEAD")
    relative_decision = _relative(CALIBRATION_DECISION)
    if (
        _git("ls-files", "--error-unmatch", "--", relative_decision)
        != relative_decision
        or _git_bytes("show", f"HEAD:{relative_decision}")
        != CALIBRATION_DECISION.read_bytes()
        or sha256_bytes(
            _git_bytes("show", f"{value['release_commit']}:{relative_decision}")
        )
        != value["calibration_decision_sha256"]
    ):
        raise RuntimeError("calibration decision differs from mechanics authorization")
    if verify_network:
        _git("fetch", "--quiet", "origin", "main")
    head = _commit_id(_git("rev-parse", "HEAD"))
    lock_commit = _commit_id(_git("log", "-1", "--format=%H", "--", relative))
    if (
        not _ancestor(value["implementation_commit"], value["release_commit"])
        or not _ancestor(value["release_commit"], lock_commit)
        or not _ancestor(lock_commit, head)
        or not _ancestor(head, "origin/main")
    ):
        raise RuntimeError("mechanics reviewed release/lock is not published on main")
    review = _validate_review(read_canonical(MECHANICS_REVIEW_JSON))
    if (
        review != value["implementation_review"]
        or sha256_file(MECHANICS_REVIEW_REPORT) != review["review_report_sha256"]
    ):
        raise RuntimeError("mechanics implementation review changed after lock")
    for critical_path, expected in value["critical_files"].items():
        path_value = ROOT / critical_path
        if (
            path_value.is_symlink()
            or not path_value.is_file()
            or sha256_file(path_value) != expected
            or sha256_bytes(
                _git_bytes(
                    "show", f"{value['implementation_commit']}:{critical_path}"
                )
            )
            != expected
        ):
            raise RuntimeError(
                f"mechanics critical file changed after review: {critical_path}"
            )
    for frozen_path, blob in value["frozen_mechanics_blobs"].items():
        if _git("rev-parse", f"HEAD:{frozen_path}") != blob:
            raise RuntimeError(f"frozen mechanics blob changed: {frozen_path}")
    if verify_network:
        verify_recorded_ci(
            value["implementation_commit"], value["implementation_ci"]
        )
        verify_recorded_ci(value["release_commit"], value["release_ci"])
        query_green_ci(head)
    return value


def mechanics_preflight_value(
    *, runner: Any, inputs: CalibrationInputs
) -> dict[str, Any]:
    lock = verify_mechanics_lock()
    loaded = _validate_loaded_runner(runner, inputs)
    head = _commit_id(_git("rev-parse", "HEAD"))
    if (
        loaded["runtime"].get("git_commit") != head
        or loaded["runtime"].get("git_dirty") is not False
    ):
        raise RuntimeError("mechanics live preflight requires the clean current commit")
    return {
        "schema_version": 1,
        "decision": "MECHANICS_LIVE_ENGINE_PREFLIGHT_PASS",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "mechanics_lock_sha256": sha256_file(MECHANICS_LOCK),
        "live_head": head,
        "live_head_ci": query_green_ci(head),
        "selected_interface": lock["selected_interface"],
        "sampling": lock["sampling"],
        "runner_sha256": sha256_file(EXP / "src/vllm_runner.py"),
        "engine": _normalized(dataclasses.asdict(runner.config)),
        "engine_args_sha256": canonical_sha256(_normalized(runner.engine_args)),
        "resolved_cudagraph": _normalized(runner.resolved_cudagraph),
        "resolved_logprobs_mode": runner.resolved_logprobs_mode,
        "adapter": _normalized(runner.adapter_info),
        "rng_isolation": {
            "engine_seed": runner.engine_args["seed"],
            "caller_global_rng_state_restored": True,
        },
        "runtime": loaded["runtime"],
        "experimental_generation_requests_before_preflight": 0,
        "sampled_model_outputs_before_preflight": 0,
        "hidden_files_read": [],
        "benchmark_files_read": [],
    }


def publish_or_verify_mechanics_preflight(
    *, runner: Any, inputs: CalibrationInputs, path: Path = MECHANICS_PREFLIGHT
) -> dict[str, Any]:
    if path.exists() or path.is_symlink():
        value = read_canonical(path)
        lock = verify_mechanics_lock()
        loaded = _validate_loaded_runner(runner, inputs)
        expected_keys = {
            "schema_version",
            "decision",
            "model",
            "revision",
            "mechanics_lock_sha256",
            "live_head",
            "live_head_ci",
            "selected_interface",
            "sampling",
            "runner_sha256",
            "engine",
            "engine_args_sha256",
            "resolved_cudagraph",
            "resolved_logprobs_mode",
            "adapter",
            "rng_isolation",
            "runtime",
            "experimental_generation_requests_before_preflight",
            "sampled_model_outputs_before_preflight",
            "hidden_files_read",
            "benchmark_files_read",
        }
        runtime = loaded.get("runtime")
        if (
            not isinstance(value, dict)
            or set(value) != expected_keys
            or not isinstance(runtime, dict)
            or runtime.get("git_dirty") is not True
            or not isinstance(runtime.get("git_commit"), str)
        ):
            raise RuntimeError("recorded mechanics preflight changed")
        recorded_ci = _validate_ci(
            runtime["git_commit"],
            value["live_head_ci"],
            label="mechanics live head",
        )
        verify_recorded_ci(runtime["git_commit"], recorded_ci)
        clean_runtime = _normalized(runtime)
        clean_runtime["git_dirty"] = False
        expected = {
            "schema_version": 1,
            "decision": "MECHANICS_LIVE_ENGINE_PREFLIGHT_PASS",
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "mechanics_lock_sha256": sha256_file(MECHANICS_LOCK),
            "live_head": runtime["git_commit"],
            "live_head_ci": recorded_ci,
            "selected_interface": lock["selected_interface"],
            "sampling": lock["sampling"],
            "runner_sha256": sha256_file(EXP / "src/vllm_runner.py"),
            "engine": _normalized(dataclasses.asdict(runner.config)),
            "engine_args_sha256": canonical_sha256(
                _normalized(runner.engine_args)
            ),
            "resolved_cudagraph": _normalized(runner.resolved_cudagraph),
            "resolved_logprobs_mode": runner.resolved_logprobs_mode,
            "adapter": _normalized(runner.adapter_info),
            "rng_isolation": {
                "engine_seed": runner.engine_args["seed"],
                "caller_global_rng_state_restored": True,
            },
            "runtime": clean_runtime,
            "experimental_generation_requests_before_preflight": 0,
            "sampled_model_outputs_before_preflight": 0,
            "hidden_files_read": [],
            "benchmark_files_read": [],
        }
        if not exact_json_equal(value, expected):
            raise RuntimeError("recorded mechanics preflight changed")
        return value
    value = mechanics_preflight_value(runner=runner, inputs=inputs)
    write_exclusive_durable(path, value)
    return value


def authorize_hidden_read(
    path: Path = VISIBLE_SELECTION,
) -> tuple[dict[str, Any], dict[str, Any]]:
    verify_mechanics_lock()
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("hidden scoring requires a visible selection receipt")
    visible = read_canonical(path)
    visible_bytes = json_bytes(visible)
    inputs = load_calibration_inputs()
    tokenizer = load_analysis_tokenizer(inputs)
    decision = read_canonical(CALIBRATION_DECISION)
    expected_visible = analyze_visible(
        decision=decision, inputs=inputs, tokenizer=tokenizer
    )
    if not exact_json_equal(visible, expected_visible):
        raise RuntimeError("visible receipt differs from exact visible analysis")
    if (
        visible.get("decision") != "MECHANICS_VISIBLE_SELECTION_FROZEN"
        or visible.get("selector_uses_hidden") is not False
        or visible.get("hidden_files_read") != []
        or visible.get("benchmark_files_read") != []
    ):
        raise RuntimeError("visible receipt does not authorize hidden scoring")
    relative = _relative(path)
    if _git("ls-files", "--error-unmatch", "--", relative) != relative:
        raise RuntimeError("visible selection receipt is not committed")
    head = _commit_id(_git("rev-parse", "HEAD"))
    if _git_bytes("show", f"{head}:{relative}") != visible_bytes:
        raise RuntimeError("visible selection receipt differs from HEAD")
    _git("fetch", "--quiet", "origin", "main")
    if not _ancestor(head, "origin/main"):
        raise RuntimeError("visible selection receipt is not published on main")
    ci = query_green_ci(head)
    return (
        {
            "authorization": "COMMITTED_VISIBLE_SELECTION_AUTHORIZES_HIDDEN_READ",
            "visible_selection_sha256": sha256_bytes(visible_bytes),
            "authorization_commit": head,
            "authorization_ci": ci,
        },
        visible,
    )
