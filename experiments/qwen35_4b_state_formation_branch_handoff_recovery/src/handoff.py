"""Continue immutable producer stages after a successful recovered G0 occupies the retired slot."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
FIRST_ROOT = REPO_ROOT / "experiments" / "qwen35_4b_state_formation_branch_recovery"
PRODUCER_ROOT = REPO_ROOT / "experiments" / "qwen35_4b_state_formation_capacity_adjudication"
EXPERIMENT_ID = "qwen35_4b_state_formation_branch_handoff_recovery"
PRODUCER_ID = "qwen35_4b_state_formation_capacity_adjudication"
FIRST_ID = "qwen35_4b_state_formation_branch_recovery"

EXPECTED_MODEL_ID = "Qwen/Qwen3.5-4B"
EXPECTED_MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
EXPECTED_PRODUCER_SOURCE = "5a8ed26ddb9446c728191ca8e7849ae44cff92a700e24b237dac522cf4286666"
EXPECTED_PRODUCER_CLI = "32d2b9a5cd6293b7a01b17a10b8307157d068978d5fd2d6b74af58ad7e3be467"
EXPECTED_PRODUCER_ANALYSIS = "876888987d816fe29ae93fde0053fb91ed58301d16bf429d5cabaa809c23a2b0"
EXPECTED_AUTHORIZATION_SHA256 = "cb9fee75368ce3f555bc182742126c49f5312d9ca3ab324baf0c39021a58818a"
EXPECTED_AUTHORIZATION_IDENTITY = "b973bc01e4ec5dba12fb61493f122a657344300d38cf7e4a975e102588dda862"
EXPECTED_FIRST_SOURCE_FILE = "035842cc748f535ce1611ea18e0b80754907f01ceb939b7ad020a5c889c11d01"
EXPECTED_FIRST_SOURCE_CONTRACT = "55d0a4550abfd09da74bc065cfe61f405f288d32344df1607b0b37ed97f956f3"
EXPECTED_FIRST_SMOKE_SHA256 = "8bf5bb36f62504163b62f1962258fad15088ed307be6139068e9b0062b446849"
EXPECTED_FIRST_SMOKE_IDENTITY = "d1135ea21f3f8f3815526ef0e23b52a22cd3baa237480c38897e34fd7cc349b5"
EXPECTED_RETIREMENT_SHA256 = "6e4c8ee379736c977e457f2a507e94e96b16294b60b0352f005329cdecf453ad"
EXPECTED_RETIREMENT_IDENTITY = "c9abdc592669c3a2fb13388acdce9bde326b6960fe16175060eddea50612eae7"
EXPECTED_FAILURE_SHA256 = "47305826eb1f9b7e34a7edac8cb6c7ba0f5e34921037c02c7400921825ca2c71"
EXPECTED_G0_SHA256 = "cdc90cd15748e4f7434598f76fb4c59bf64ef39a29fb729adbdfa8f8e178c68f"
EXPECTED_G0_IDENTITY = "e1f1c9069b9d94224a358e26795e2938e6a9623b7be296eca0f5f55b0e6f89dc"
EXPECTED_G0_STARTED_SHA256 = "f0c7805551d511b71aa48a33391313f93c03ed6b7138c8e790eb07878abb3c87"
EXPECTED_G0_STARTED_IDENTITY = "d65037c34b45e506d05903cf388dec669cdbe2997883d0855392a45ce29cac81"
EXPECTED_G0_COMPLETE_SHA256 = "755ad5615b45fdc14bfaa241e7562eba3064e85b2e3d60e350a6a85354f7f941"
EXPECTED_G0_COMPLETE_IDENTITY = "0e85031d80977ed339126d1a548899dbcd17818a971078ae2aee0571349a4b96"

FIRST_SOURCE = FIRST_ROOT / "src" / "recovery.py"
FIRST_SMOKE = FIRST_ROOT / "runs" / "smoke.json"
RETIREMENT = FIRST_ROOT / "runs" / "failures" / "retirement_receipt.json"
ARCHIVED_FAILURE = FIRST_ROOT / "runs" / "failures" / "g0_fullrank_seed7411_branch_authorization.json"
SUCCESSFUL_G0 = PRODUCER_ROOT / "runs" / "setup" / "g0_fullrank_seed7411.json"
RETIRED_FAILURE_MIRROR = (
    PRODUCER_ROOT / "runs" / "failures" / "g0_fullrank_seed7411_source_5a8ed26ddb94.json"
)
G0_STARTED = FIRST_ROOT / "runs" / "invocations" / "model_smoke_fullrank_joint_seed7411_trigger_started.json"
G0_COMPLETE = FIRST_ROOT / "runs" / "invocations" / "model_smoke_fullrank_joint_seed7411_trigger_complete.json"
AUTHORIZATION = PRODUCER_ROOT / "analysis" / "lora_joint_trigger.json"

CONTRACT_FILES = (
    "configs/default.yaml",
    "reports/design_review.md",
    "scripts/run.py",
    "src/__init__.py",
    "src/handoff.py",
    "tests/test_handoff.py",
)


def _load_first_recovery() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_frozen_state_formation_branch_recovery", FIRST_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load first branch recovery")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module


first = _load_first_recovery()
ALLOWED_STAGES = first.ALLOWED_STAGES


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["receipt_identity_sha256"] = hashlib.sha256(
        _canonical_json(result).encode("utf-8")
    ).hexdigest()
    return result


def _read_config() -> dict[str, Any]:
    path = ROOT / "configs" / "default.yaml"
    try:
        value = yaml.safe_load(first._regular_bytes(path).decode("utf-8"))
    except (UnicodeError, yaml.YAMLError) as exc:
        raise RuntimeError("handoff-recovery config is invalid") from exc
    if not isinstance(value, dict):
        raise RuntimeError("handoff-recovery config must be a mapping")
    expected = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "producer_experiment_id": PRODUCER_ID,
        "first_recovery_experiment_id": FIRST_ID,
        "producer_source_contract_sha256": EXPECTED_PRODUCER_SOURCE,
        "model_id": EXPECTED_MODEL_ID,
        "model_revision": EXPECTED_MODEL_REVISION,
        "backend": "transformers",
        "allowed_producer_stages": list(ALLOWED_STAGES),
        "first_recovery_source_file_sha256": EXPECTED_FIRST_SOURCE_FILE,
        "first_recovery_source_contract_sha256": EXPECTED_FIRST_SOURCE_CONTRACT,
        "first_recovery_smoke_sha256": EXPECTED_FIRST_SMOKE_SHA256,
        "retirement_receipt_sha256": EXPECTED_RETIREMENT_SHA256,
        "archived_failure_sha256": EXPECTED_FAILURE_SHA256,
        "successful_g0_sha256": EXPECTED_G0_SHA256,
        "successful_g0_receipt_identity_sha256": EXPECTED_G0_IDENTITY,
        "successful_g0_started_sha256": EXPECTED_G0_STARTED_SHA256,
        "successful_g0_complete_sha256": EXPECTED_G0_COMPLETE_SHA256,
        "authorization_sha256": EXPECTED_AUTHORIZATION_SHA256,
        "authorization_receipt_identity_sha256": EXPECTED_AUTHORIZATION_IDENTITY,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise RuntimeError(f"handoff-recovery config changed {key}")
    return value


def handoff_source_contract() -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for relative in CONTRACT_FILES:
        path = ROOT / relative
        info = os.lstat(path)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RuntimeError(f"handoff contract file is not one regular inode: {relative}")
        files.append({"path": relative, "sha256": first._sha256(path), "bytes": info.st_size})
    payload = {"schema_version": 1, "files": files}
    return {
        **payload,
        "source_contract_sha256": hashlib.sha256(
            _canonical_json(payload).encode("utf-8")
        ).hexdigest(),
    }


def _exact_receipt(path: Path, sha256: str, identity: str, label: str) -> tuple[dict[str, Any], bytes]:
    receipt, raw = first._strict_json(path)
    first._validate_identity(receipt, label=label)
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise RuntimeError(f"{label} bytes changed")
    if receipt.get("receipt_identity_sha256") != identity:
        raise RuntimeError(f"{label} identity changed")
    return receipt, raw


def validate_handoff() -> dict[str, Any]:
    if os.path.lexists(RETIRED_FAILURE_MIRROR):
        raise RuntimeError("retired failure mirror reappeared")
    retirement, _ = _exact_receipt(
        RETIREMENT, EXPECTED_RETIREMENT_SHA256, EXPECTED_RETIREMENT_IDENTITY, "failure retirement"
    )
    if retirement.get("status") != "BRANCH_AUTHORIZATION_FAILURE_RETIRED":
        raise RuntimeError("failure retirement is not terminal")
    archived_raw = first._regular_bytes(ARCHIVED_FAILURE)
    if hashlib.sha256(archived_raw).hexdigest() != EXPECTED_FAILURE_SHA256:
        raise RuntimeError("archived failure bytes changed")
    g0, g0_raw = first._strict_json(SUCCESSFUL_G0)
    g0_sha256 = hashlib.sha256(g0_raw).hexdigest()
    if g0_sha256 == EXPECTED_FAILURE_SHA256:
        raise RuntimeError("retired failure bytes reoccupied the canonical G0 slot")
    first._validate_identity(g0, label="successful recovered G0")
    checks = {
        "sha256": g0_sha256,
        "receipt_identity_sha256": g0.get("receipt_identity_sha256"),
        "status": g0.get("status"),
        "phase": g0.get("phase"),
        "capacity": g0.get("capacity"),
        "model_seed": g0.get("model_seed"),
        "source_contract_sha256": g0.get("source_contract_sha256"),
        "model_id": g0.get("model_id"),
        "model_revision": g0.get("model_revision"),
        "authorizes_positive_control": g0.get("authorizes_positive_control"),
        "authorizes_training": g0.get("authorizes_training"),
        "training_or_evaluation_started": g0.get("training_or_evaluation_started"),
        "benchmark_files_read": g0.get("benchmark_files_read"),
        "sealed_contrast_payloads_opened": g0.get("sealed_contrast_payloads_opened"),
    }
    expected = {
        "sha256": EXPECTED_G0_SHA256,
        "receipt_identity_sha256": EXPECTED_G0_IDENTITY,
        "status": "MODEL_SMOKE_PASS",
        "phase": "fullrank_g0",
        "capacity": "fullrank",
        "model_seed": 7411,
        "source_contract_sha256": EXPECTED_PRODUCER_SOURCE,
        "model_id": EXPECTED_MODEL_ID,
        "model_revision": EXPECTED_MODEL_REVISION,
        "authorizes_positive_control": True,
        "authorizes_training": False,
        "training_or_evaluation_started": False,
        "benchmark_files_read": 0,
        "sealed_contrast_payloads_opened": [],
    }
    if checks != expected:
        raise RuntimeError("canonical G0 slot is not the exact successful recovered G0")
    started, _ = _exact_receipt(
        G0_STARTED, EXPECTED_G0_STARTED_SHA256, EXPECTED_G0_STARTED_IDENTITY, "first recovery G0 start"
    )
    complete, _ = _exact_receipt(
        G0_COMPLETE, EXPECTED_G0_COMPLETE_SHA256, EXPECTED_G0_COMPLETE_IDENTITY, "first recovery G0 completion"
    )
    if (
        started.get("status") != "RECOVERED_PRODUCER_INVOCATION_STARTED"
        or complete.get("status") != "RECOVERED_PRODUCER_INVOCATION_COMPLETE"
        or complete.get("producer_status") != "MODEL_SMOKE_PASS"
        or complete.get("producer_output_sha256") != EXPECTED_G0_SHA256
        or complete.get("producer_receipt_identity_sha256") != EXPECTED_G0_IDENTITY
        or complete.get("invocation_started_sha256") != EXPECTED_G0_STARTED_SHA256
    ):
        raise RuntimeError("first recovery G0 invocation lineage changed")
    return {
        "retirement_receipt_identity_sha256": retirement["receipt_identity_sha256"],
        "successful_g0_receipt_identity_sha256": g0["receipt_identity_sha256"],
        "first_recovery_started_identity_sha256": started["receipt_identity_sha256"],
        "first_recovery_complete_identity_sha256": complete["receipt_identity_sha256"],
    }


def run_smoke() -> dict[str, Any]:
    _read_config()
    if first._sha256(FIRST_SOURCE) != EXPECTED_FIRST_SOURCE_FILE:
        raise RuntimeError("first recovery source file changed")
    prior_smoke, prior_smoke_sha256 = first._require_smoke()
    if (
        prior_smoke_sha256 != EXPECTED_FIRST_SMOKE_SHA256
        or prior_smoke.get("receipt_identity_sha256") != EXPECTED_FIRST_SMOKE_IDENTITY
        or prior_smoke.get("recovery_source_contract_sha256") != EXPECTED_FIRST_SOURCE_CONTRACT
    ):
        raise RuntimeError("first recovery smoke lineage changed")
    first._validate_authorization_files()
    handoff = validate_handoff()
    false_rejection_arguments = {
        "stage": "positive-control",
        "capacity": "fullrank",
        "objective": "joint",
        "eval_set": "trigger",
        "seed": 7411,
        "checkpoint": None,
        "initialization_bundle": "large_artifacts/qwen35_4b_state_formation_capacity_adjudication/initialization_seed7411.pt",
        "model_smoke_receipt": SUCCESSFUL_G0.relative_to(REPO_ROOT).as_posix(),
        "positive_control_receipt": None,
        "authorization_receipt": AUTHORIZATION.relative_to(REPO_ROOT).as_posix(),
        "output": (PRODUCER_ROOT / "runs" / "setup" / "positive_control_fullrank_seed7411.json").relative_to(REPO_ROOT).as_posix(),
    }
    try:
        first.invoke_producer(false_rejection_arguments, [])
    except RuntimeError as exc:
        original_rejection = str(exc)
    else:
        raise RuntimeError("first recovery unexpectedly accepted the occupied successful G0 slot")
    if original_rejection != "failed G0 pair must be archived and retired before recovered invocation":
        raise RuntimeError("first recovery failed for an unexpected reason")
    contract = handoff_source_contract()
    receipt = _identity({
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "BRANCH_HANDOFF_RECOVERY_SMOKE_PASS",
        "handoff_source_contract_sha256": contract["source_contract_sha256"],
        "handoff_source_files": contract["files"],
        "producer_experiment_id": PRODUCER_ID,
        "producer_source_contract_sha256": EXPECTED_PRODUCER_SOURCE,
        "first_recovery_source_contract_sha256": EXPECTED_FIRST_SOURCE_CONTRACT,
        "first_recovery_smoke_sha256": EXPECTED_FIRST_SMOKE_SHA256,
        "original_false_rejection": original_rejection,
        "lineage": handoff,
        "control_passes": {
            "pathname_only_false_rejection_reproduced": 1,
            "retirement_receipt_exact": 1,
            "archived_failure_exact": 1,
            "retired_mirror_absent": 1,
            "successful_g0_bytes_identity_and_status_exact": 1,
            "successful_g0_invocation_lineage_exact": 1,
        },
        "model_loaded": False,
        "training_or_evaluation_started": False,
        "benchmark_paths_opened": 0,
        "sealed_contrast_rows_opened": 0,
        "scientific_interpretation_changed": False,
    })
    first._publish_json(ROOT / "runs" / "smoke.json", receipt)
    return receipt


def _require_smoke() -> tuple[dict[str, Any], str]:
    smoke, raw = first._strict_json(ROOT / "runs" / "smoke.json")
    first._validate_identity(smoke, label="branch-handoff smoke")
    if smoke.get("status") != "BRANCH_HANDOFF_RECOVERY_SMOKE_PASS":
        raise RuntimeError("branch-handoff smoke did not pass")
    if smoke.get("handoff_source_contract_sha256") != handoff_source_contract()["source_contract_sha256"]:
        raise RuntimeError("branch-handoff source changed after smoke")
    return smoke, hashlib.sha256(raw).hexdigest()


def _completion_from_output(
    arguments: Mapping[str, Any], started_path: Path, smoke: Mapping[str, Any]
) -> dict[str, Any]:
    output = first._producer_output_leaf(arguments)
    receipt, raw = first._strict_json(output)
    first._validate_identity(receipt, label="producer output")
    if receipt.get("source_contract_sha256") != EXPECTED_PRODUCER_SOURCE:
        raise RuntimeError("producer output source changed")
    if receipt.get("model_id") != EXPECTED_MODEL_ID or receipt.get("model_revision") != EXPECTED_MODEL_REVISION:
        raise RuntimeError("producer output model identity changed")
    expected_status = {
        "model-smoke": "MODEL_SMOKE_PASS",
        "positive-control": "POSITIVE_CONTROL_PASS",
        "train": "TRAINING_COMPLETE",
        "evaluate-state": "STATE_EVALUATION_COMPLETE",
    }[str(arguments["stage"])]
    if receipt.get("status") != expected_status:
        raise RuntimeError(f"producer stage did not complete: {receipt.get('status')}: {receipt.get('error')}")
    return _identity({
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "HANDOFF_PRODUCER_INVOCATION_COMPLETE",
        "invocation": dict(arguments),
        "producer_output": output.relative_to(REPO_ROOT).as_posix(),
        "producer_output_sha256": hashlib.sha256(raw).hexdigest(),
        "producer_receipt_identity_sha256": receipt["receipt_identity_sha256"],
        "producer_status": receipt.get("status"),
        "producer_source_contract_sha256": EXPECTED_PRODUCER_SOURCE,
        "handoff_source_contract_sha256": smoke["handoff_source_contract_sha256"],
        "invocation_started_sha256": first._sha256(started_path),
        "changed_runtime_function": "src.analysis._canonical_expected_path only",
    })


def invoke_producer(arguments: Mapping[str, Any], producer_argv: Sequence[str]) -> dict[str, Any]:
    if arguments.get("stage") not in ALLOWED_STAGES:
        raise RuntimeError("producer stage is not allowed through branch handoff")
    if not arguments.get("authorization_receipt"):
        raise RuntimeError("branch handoff requires an authorization receipt")
    if not arguments.get("output"):
        raise RuntimeError("branch handoff requires an explicit canonical producer output")
    first._validate_invocation_shape(arguments)
    validate_handoff()
    smoke, smoke_sha256 = _require_smoke()
    _, prior_smoke_sha256 = first._require_smoke()
    if prior_smoke_sha256 != EXPECTED_FIRST_SMOKE_SHA256:
        raise RuntimeError("first recovery smoke changed")
    first._validate_authorization_files()
    context = first.load_producer_context()
    slug = first._invocation_slug(arguments)
    started_path = ROOT / "runs" / "invocations" / f"{slug}_started.json"
    complete_path = ROOT / "runs" / "invocations" / f"{slug}_complete.json"
    output_leaf = first._producer_output_leaf(arguments)
    started = _identity({
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "HANDOFF_PRODUCER_INVOCATION_STARTED",
        "invocation": dict(arguments),
        "producer_argv": list(producer_argv),
        "producer_source_contract_sha256": EXPECTED_PRODUCER_SOURCE,
        "producer_cli_sha256": EXPECTED_PRODUCER_CLI,
        "producer_analysis_sha256": EXPECTED_PRODUCER_ANALYSIS,
        "authorization_sha256": first._sha256(first._canonical(Path(str(arguments["authorization_receipt"])))),
        "handoff_smoke_sha256": smoke_sha256,
        "handoff_source_contract_sha256": smoke["handoff_source_contract_sha256"],
        "first_recovery_smoke_sha256": prior_smoke_sha256,
        "successful_g0_sha256": EXPECTED_G0_SHA256,
        "changed_runtime_function": "src.analysis._canonical_expected_path only",
    })
    started_preexisting = os.path.lexists(started_path)
    output_preexisting = os.path.lexists(output_leaf)
    if output_preexisting and not started_preexisting:
        raise RuntimeError("producer output predates its handoff invocation receipt")
    first._publish_json(started_path, started)
    if os.path.lexists(complete_path):
        complete, _ = first._strict_json(complete_path)
        first._validate_identity(complete, label="handoff invocation")
        expected = _completion_from_output(arguments, started_path, smoke)
        if complete != expected:
            raise RuntimeError("handoff invocation completion changed")
        return complete
    if not os.path.lexists(output_leaf):
        with first.installed_path_seam(context):
            result = context.cli.main(list(producer_argv))
        if result != 0:
            raise RuntimeError(f"producer CLI returned nonzero status: {result}")
    complete = _completion_from_output(arguments, started_path, smoke)
    first._publish_json(complete_path, complete)
    return complete
