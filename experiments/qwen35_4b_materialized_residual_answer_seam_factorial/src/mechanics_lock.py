"""Post-calibration mechanics authorization and pre-hidden publication gates."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import Any

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
    verify_calibration_lock,
    verify_recorded_ci,
)
from calibration_stage import (
    CalibrationInputs,
    authenticate_calibration_decision,
    load_analysis_tokenizer,
    load_calibration_inputs,
)
from mechanics_stage import (
    DEFAULT_MECHANICS_LOCK,
    DEFAULT_MECHANICS_PREFLIGHT,
    MECHANICS_INVOCATION_ORDER,
    RAW_DIR,
    TRANSPORT_DECISION,
    VISIBLE_SELECTION,
    canonical_sha256,
    mechanics_sampling_plan,
    selected_interface,
)
from transactions import (
    MODEL_ID,
    MODEL_REVISION,
    read_canonical,
    sha256_file,
    write_exclusive_durable,
)


MECHANICS_LOCK = DEFAULT_MECHANICS_LOCK
MECHANICS_PREFLIGHT = DEFAULT_MECHANICS_PREFLIGHT


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
        for path in (MECHANICS_LOCK, MECHANICS_PREFLIGHT, RAW_DIR, TRANSPORT_DECISION, VISIBLE_SELECTION)
    ):
        raise RuntimeError("mechanics lock must precede every mechanics artifact")


def build_mechanics_lock_value(
    *,
    calibration_lock: Mapping[str, Any],
    calibration_decision: Mapping[str, Any],
    calibration_decision_sha256: str,
    authorization_commit: str,
    authorization_ci: Mapping[str, Mapping[str, Any]],
    inputs: CalibrationInputs,
) -> dict[str, Any]:
    winner = selected_interface(calibration_decision, inputs)
    frozen = calibration_lock.get("frozen_mechanics_blobs")
    if not isinstance(frozen, dict) or not frozen:
        raise RuntimeError("calibration lock did not freeze mechanics")
    return {
        "schema_version": 1,
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
        "authorization_commit": _commit_id(authorization_commit),
        "authorization_ci": {
            name: dict(value) for name, value in authorization_ci.items()
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
        "authorization_commit",
        "authorization_ci",
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
        value["schema_version"] != 1
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
        or value["sampling"]
        != _normalized(mechanics_sampling_plan(calibration_decision, inputs))
        or value["invocation_order"] != list(MECHANICS_INVOCATION_ORDER)
        or value["frozen_mechanics_blobs"]
        != calibration_lock["frozen_mechanics_blobs"]
        or value["experimental_generation_requests_before_lock"] != 0
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
    commit = _commit_id(value["authorization_commit"])
    ci = value["authorization_ci"]
    if not isinstance(ci, dict) or set(ci) != set(REQUIRED_WORKFLOWS):
        raise RuntimeError("mechanics authorization CI changed")
    for row in ci.values():
        if (
            not isinstance(row, dict)
            or set(row) != {"database_id", "head_sha", "status", "conclusion", "url"}
            or row["head_sha"] != commit
            or row["status"] != "completed"
            or row["conclusion"] != "success"
            or not isinstance(row["database_id"], int)
            or not isinstance(row["url"], str)
        ):
            raise RuntimeError("mechanics authorization CI changed")
    return value


def publish_mechanics_lock(path: Path = MECHANICS_LOCK) -> dict[str, Any]:
    if path.resolve() != MECHANICS_LOCK.resolve():
        raise RuntimeError("mechanics lock path changed")
    _ensure_clean_for_lock()
    calibration_lock = verify_calibration_lock()
    inputs = load_calibration_inputs()
    tokenizer = load_analysis_tokenizer(inputs)
    decision = authenticate_calibration_decision(
        inputs=inputs,
        raw_dir=CALIBRATION_DECISION.parent / "raw",
        tokenizer=tokenizer,
        decision_path=CALIBRATION_DECISION,
    )
    selected_interface(decision, inputs)
    relative_decision = _relative(CALIBRATION_DECISION)
    if _git("ls-files", "--error-unmatch", "--", relative_decision) != relative_decision:
        raise RuntimeError("calibration decision is not committed")
    if _git_bytes("show", f"HEAD:{relative_decision}") != CALIBRATION_DECISION.read_bytes():
        raise RuntimeError("calibration decision differs from HEAD")
    _git("fetch", "--quiet", "origin", "main")
    head = _commit_id(_git("rev-parse", "HEAD"))
    if not _ancestor(head, "origin/main"):
        raise RuntimeError("calibration decision commit is not published on main")
    ci = query_green_ci(head)
    for relative, blob in calibration_lock["frozen_mechanics_blobs"].items():
        if _git("rev-parse", f"HEAD:{relative}") != blob:
            raise RuntimeError(f"mechanics changed after calibration freeze: {relative}")
    value = build_mechanics_lock_value(
        calibration_lock=calibration_lock,
        calibration_decision=decision,
        calibration_decision_sha256=sha256_file(CALIBRATION_DECISION),
        authorization_commit=head,
        authorization_ci=ci,
        inputs=inputs,
    )
    write_exclusive_durable(path, value)
    return value


def verify_mechanics_lock(
    path: Path = MECHANICS_LOCK, *, verify_network: bool = True
) -> dict[str, Any]:
    calibration_lock = verify_calibration_lock(
        verify_network=verify_network,
        allowed_live_prefixes=("runs/calibration/", "runs/mechanics/"),
    )
    inputs = load_calibration_inputs()
    tokenizer = load_analysis_tokenizer(inputs)
    decision = authenticate_calibration_decision(
        inputs=inputs,
        raw_dir=CALIBRATION_DECISION.parent / "raw",
        tokenizer=tokenizer,
        decision_path=CALIBRATION_DECISION,
    )
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
    if verify_network:
        _git("fetch", "--quiet", "origin", "main")
    head = _commit_id(_git("rev-parse", "HEAD"))
    if not _ancestor(value["authorization_commit"], head) or not _ancestor(
        head, "origin/main"
    ):
        raise RuntimeError("mechanics authorization is not published on main")
    for frozen_path, blob in value["frozen_mechanics_blobs"].items():
        if _git("rev-parse", f"HEAD:{frozen_path}") != blob:
            raise RuntimeError(f"frozen mechanics blob changed: {frozen_path}")
    if verify_network:
        verify_recorded_ci(value["authorization_commit"], value["authorization_ci"])
        query_green_ci(head)
    return value


def mechanics_preflight_value(
    *, runner: Any, inputs: CalibrationInputs
) -> dict[str, Any]:
    lock = verify_mechanics_lock()
    loaded = _validate_loaded_runner(runner, inputs)
    return {
        "schema_version": 1,
        "decision": "MECHANICS_LIVE_ENGINE_PREFLIGHT_PASS",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "mechanics_lock_sha256": sha256_file(MECHANICS_LOCK),
        "selected_interface": lock["selected_interface"],
        "sampling": lock["sampling"],
        "engine": _normalized(dataclasses.asdict(runner.config)),
        "engine_args_sha256": canonical_sha256(_normalized(runner.engine_args)),
        "resolved_cudagraph": _normalized(runner.resolved_cudagraph),
        "resolved_logprobs_mode": runner.resolved_logprobs_mode,
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
            "selected_interface",
            "sampling",
            "engine",
            "engine_args_sha256",
            "resolved_cudagraph",
            "resolved_logprobs_mode",
            "runtime",
            "experimental_generation_requests_before_preflight",
            "sampled_model_outputs_before_preflight",
            "hidden_files_read",
            "benchmark_files_read",
        }
        if (
            not isinstance(value, dict)
            or set(value) != expected_keys
            or value["schema_version"] != 1
            or value["decision"] != "MECHANICS_LIVE_ENGINE_PREFLIGHT_PASS"
            or value["model"] != MODEL_ID
            or value["revision"] != MODEL_REVISION
            or value["mechanics_lock_sha256"] != sha256_file(MECHANICS_LOCK)
            or value["selected_interface"] != lock["selected_interface"]
            or value["sampling"] != lock["sampling"]
            or value["engine"] != _normalized(dataclasses.asdict(runner.config))
            or value["engine_args_sha256"]
            != canonical_sha256(_normalized(runner.engine_args))
            or value["resolved_cudagraph"] != _normalized(runner.resolved_cudagraph)
            or value["resolved_logprobs_mode"] != runner.resolved_logprobs_mode
            or value["experimental_generation_requests_before_preflight"] != 0
            or value["sampled_model_outputs_before_preflight"] != 0
            or value["hidden_files_read"] != []
            or value["benchmark_files_read"] != []
            or any(
                value["runtime"].get(field) != loaded["runtime"].get(field)
                for field in ("python", "python_executable", "packages", "gpu")
            )
        ):
            raise RuntimeError("recorded mechanics preflight changed")
        return value
    value = mechanics_preflight_value(runner=runner, inputs=inputs)
    write_exclusive_durable(path, value)
    return value


def authorize_hidden_read(path: Path = VISIBLE_SELECTION) -> dict[str, Any]:
    verify_mechanics_lock()
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("hidden scoring requires a visible selection receipt")
    visible = read_canonical(path)
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
    if _git_bytes("show", f"HEAD:{relative}") != path.read_bytes():
        raise RuntimeError("visible selection receipt differs from HEAD")
    _git("fetch", "--quiet", "origin", "main")
    head = _commit_id(_git("rev-parse", "HEAD"))
    if not _ancestor(head, "origin/main"):
        raise RuntimeError("visible selection receipt is not published on main")
    ci = query_green_ci(head)
    return {
        "authorization": "COMMITTED_VISIBLE_SELECTION_AUTHORIZES_HIDDEN_READ",
        "visible_selection_sha256": sha256_file(path),
        "authorization_commit": head,
        "authorization_ci": ci,
    }
