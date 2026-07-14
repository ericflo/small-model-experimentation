"""Recover immutable v11 branch consumption through one exact path seam."""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import importlib.util
import json
import os
import re
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator, Mapping, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
PRODUCER_ROOT = REPO_ROOT / "experiments" / "qwen35_4b_state_formation_capacity_adjudication"
ANALYSIS_RECOVERY_ROOT = REPO_ROOT / "experiments" / "qwen35_4b_state_formation_analysis_recovery"
RECOVERY_ID = "qwen35_4b_state_formation_branch_recovery"
PRODUCER_ID = "qwen35_4b_state_formation_capacity_adjudication"
EXPECTED_SOURCE = "5a8ed26ddb9446c728191ca8e7849ae44cff92a700e24b237dac522cf4286666"
EXPECTED_IMPLEMENTATION = "7d6cd93fead0e524e10e7afe4b60a531ea2d6aa7f3f70778ef962889aaeed278"
EXPECTED_CLI_SHA256 = "32d2b9a5cd6293b7a01b17a10b8307157d068978d5fd2d6b74af58ad7e3be467"
EXPECTED_GPU_RUNNER_SHA256 = "7faae60bba58b2083ed8cc2d958c279b3d70912f496997bdbd46475c320072a6"
EXPECTED_ANALYSIS_SHA256 = "876888987d816fe29ae93fde0053fb91ed58301d16bf429d5cabaa809c23a2b0"
EXPECTED_CONFIG_FILE_SHA256 = "b165537c8c86531ac17aecfa5c65045a708cd4b196b7614c6f98331a9fae1ca8"
EXPECTED_CONFIG_SHA256 = "eeb4e828526f750dce1258bcc91d03114c80688d300112e03d18c9d911489393"
EXPECTED_MODEL_ID = "Qwen/Qwen3.5-4B"
EXPECTED_MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
EXPECTED_PREFIX = "../../large_artifacts/qwen35_4b_state_formation_capacity_adjudication"
EXPECTED_AUTHORIZATION_SHA256 = "cb9fee75368ce3f555bc182742126c49f5312d9ca3ab324baf0c39021a58818a"
EXPECTED_AUTHORIZATION_IDENTITY = "b973bc01e4ec5dba12fb61493f122a657344300d38cf7e4a975e102588dda862"
EXPECTED_ANALYSIS_RECOVERY_SHA256 = "aa43077b60fc98931c88eccf6680f80f0e5f2c36c98f9516926a9d3ea46c6b7e"
EXPECTED_ANALYSIS_RECOVERY_IDENTITY = "d068482a22ca5162efbce16bc563efa57dcb6cbd634ff4f0b95497dbcf60f40e"
EXPECTED_ANALYSIS_RECOVERY_SOURCE = "6ab26016b3de397307c7c8def9c685315b6660370ee98af1a757da11fe1ee94b"
EXPECTED_FAILURE_SHA256 = "47305826eb1f9b7e34a7edac8cb6c7ba0f5e34921037c02c7400921825ca2c71"
EXPECTED_FAILURE_IDENTITY = "070c23aff4c08db60edcf8d548c80466d177fa116391845bab5a32e3f6dcaa24"
ALLOWED_STAGES = ("model-smoke", "positive-control", "train", "evaluate-state")
CONTRACT_FILES = (
    "configs/default.yaml",
    "reports/design_review.md",
    "scripts/run.py",
    "src/__init__.py",
    "src/recovery.py",
    "tests/test_recovery.py",
)
AUTHORIZATION_PATH = PRODUCER_ROOT / "analysis" / "lora_joint_trigger.json"
ANALYSIS_RECOVERY_PATH = ANALYSIS_RECOVERY_ROOT / "analysis" / "lora_joint_recovery.json"
FAILURE_CANONICAL = PRODUCER_ROOT / "runs" / "setup" / "g0_fullrank_seed7411.json"
FAILURE_MIRROR = (
    PRODUCER_ROOT
    / "runs"
    / "failures"
    / "g0_fullrank_seed7411_source_5a8ed26ddb94.json"
)
ARCHIVED_FAILURE = ROOT / "runs" / "failures" / "g0_fullrank_seed7411_branch_authorization.json"
ARCHIVE_RECEIPT = ROOT / "runs" / "failures" / "archive_receipt.json"
RETIREMENT_STARTED = ROOT / "runs" / "failures" / "retirement_started.json"
RETIREMENT_RECEIPT = ROOT / "runs" / "failures" / "retirement_receipt.json"


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _with_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["receipt_identity_sha256"] = hashlib.sha256(
        _canonical_json(result).encode("utf-8")
    ).hexdigest()
    return result


def _canonical(path: Path) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        absolute.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise RuntimeError(f"recovery path escapes repository: {path}") from exc
    return absolute


def _require_no_symlink_ancestors(path: Path) -> None:
    absolute = _canonical(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if os.path.lexists(current) and stat.S_ISLNK(os.lstat(current).st_mode):
            raise RuntimeError(f"recovery path uses a symlink alias: {path}")


def _regular_bytes(path: Path) -> bytes:
    canonical = _canonical(path)
    if canonical != path:
        raise RuntimeError(f"recovery input is not lexical-canonical: {path}")
    _require_no_symlink_ancestors(canonical)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(canonical, flags)
    try:
        opened = os.fstat(descriptor)
        before = os.stat(canonical, follow_symlinks=False)
        identity = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1 or identity(opened) != identity(before):
            raise RuntimeError(f"recovery input is not one stable regular inode: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        rebound = os.stat(canonical, follow_symlinks=False)
        if identity(opened) != identity(after) or identity(after) != identity(rebound):
            raise RuntimeError(f"recovery input changed while read: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _sha256(path: Path) -> str:
    return hashlib.sha256(_regular_bytes(path)).hexdigest()


def _strict_json_bytes(raw: bytes, *, label: str) -> dict[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise RuntimeError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                RuntimeError(f"nonfinite JSON value in {label}: {value}")
            ),
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid JSON in {label}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON is not an object in {label}")
    return value


def _strict_json(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = _regular_bytes(path)
    return _strict_json_bytes(raw, label=str(path)), raw


def _validate_identity(receipt: Mapping[str, Any], *, label: str) -> None:
    claimed = receipt.get("receipt_identity_sha256")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_identity_sha256"}
    expected = hashlib.sha256(_canonical_json(unsigned).encode("utf-8")).hexdigest()
    if claimed != expected:
        raise RuntimeError(f"{label} receipt identity mismatch")


def _ensure_parent(path: Path) -> None:
    canonical = _canonical(path)
    canonical.parent.mkdir(parents=True, exist_ok=True)
    _require_no_symlink_ancestors(canonical.parent)


def _publish_or_verify(path: Path, raw: bytes) -> None:
    canonical = _canonical(path)
    _ensure_parent(canonical)
    if os.path.lexists(canonical):
        if _regular_bytes(canonical) != raw:
            raise RuntimeError(f"immutable recovery output differs: {path}")
        return
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(canonical, flags, 0o644)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise RuntimeError(f"short recovery write: {path}")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if _regular_bytes(canonical) != raw:
        raise RuntimeError(f"recovery output failed reopen: {path}")


def _publish_json(path: Path, payload: Mapping[str, Any]) -> None:
    raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    _publish_or_verify(path, raw)


def _read_config() -> dict[str, Any]:
    path = ROOT / "configs" / "default.yaml"
    try:
        value = yaml.safe_load(_regular_bytes(path).decode("utf-8"))
    except (UnicodeError, yaml.YAMLError) as exc:
        raise RuntimeError("branch-recovery config is invalid") from exc
    if not isinstance(value, dict):
        raise RuntimeError("branch-recovery config must be a mapping")
    expected = {
        "schema_version": 1,
        "experiment_id": RECOVERY_ID,
        "producer_experiment_id": PRODUCER_ID,
        "producer_source_contract_version": 11,
        "producer_source_contract_sha256": EXPECTED_SOURCE,
        "producer_implementation_sha256": EXPECTED_IMPLEMENTATION,
        "producer_cli_sha256": EXPECTED_CLI_SHA256,
        "producer_gpu_runner_sha256": EXPECTED_GPU_RUNNER_SHA256,
        "producer_analysis_sha256": EXPECTED_ANALYSIS_SHA256,
        "producer_config_file_sha256": EXPECTED_CONFIG_FILE_SHA256,
        "producer_config_sha256": EXPECTED_CONFIG_SHA256,
        "registered_external_prefix": EXPECTED_PREFIX,
        "model_id": EXPECTED_MODEL_ID,
        "model_revision": EXPECTED_MODEL_REVISION,
        "backend": "transformers",
        "allowed_producer_stages": list(ALLOWED_STAGES),
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise RuntimeError(f"branch-recovery config changed {key}")
    nested = {
        "authorization": {
            "path": AUTHORIZATION_PATH.relative_to(REPO_ROOT).as_posix(),
            "sha256": EXPECTED_AUTHORIZATION_SHA256,
            "receipt_identity_sha256": EXPECTED_AUTHORIZATION_IDENTITY,
            "status": "LORA_JOINT_MISS_CONTROLS_REQUIRED",
            "next_stage": "run_lora_state_only_and_fullrank_joint",
        },
        "analysis_recovery": {
            "path": ANALYSIS_RECOVERY_PATH.relative_to(REPO_ROOT).as_posix(),
            "sha256": EXPECTED_ANALYSIS_RECOVERY_SHA256,
            "receipt_identity_sha256": EXPECTED_ANALYSIS_RECOVERY_IDENTITY,
            "source_contract_sha256": EXPECTED_ANALYSIS_RECOVERY_SOURCE,
        },
        "failed_setup": {
            "canonical_path": FAILURE_CANONICAL.relative_to(REPO_ROOT).as_posix(),
            "mirror_path": FAILURE_MIRROR.relative_to(REPO_ROOT).as_posix(),
            "sha256": EXPECTED_FAILURE_SHA256,
            "receipt_identity_sha256": EXPECTED_FAILURE_IDENTITY,
            "phase": "fullrank_g0",
            "capacity": "fullrank",
            "model_seed": 7411,
            "failure_stage": "branch_authorization",
        },
    }
    for key, expected_value in nested.items():
        if value.get(key) != expected_value:
            raise RuntimeError(f"branch-recovery config changed {key}")
    return value


def recovery_source_contract() -> dict[str, Any]:
    files = []
    for relative in CONTRACT_FILES:
        path = ROOT / relative
        info = os.lstat(path)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RuntimeError(f"recovery contract file is not one regular inode: {relative}")
        files.append({"path": relative, "sha256": _sha256(path), "bytes": info.st_size})
    payload = {"schema_version": 1, "files": files}
    return {
        **payload,
        "source_contract_sha256": hashlib.sha256(
            _canonical_json(payload).encode("utf-8")
        ).hexdigest(),
    }


@dataclass(frozen=True)
class ProducerContext:
    cli: ModuleType
    analysis: ModuleType
    gpu_runner: ModuleType
    config: dict[str, Any]
    config_receipt: dict[str, Any]


def load_producer_context() -> ProducerContext:
    """Import the exact producer CLI and the module objects it actually uses."""

    _read_config()
    existing = sys.modules.get("src")
    producer_init = PRODUCER_ROOT / "src" / "__init__.py"
    if existing is not None and Path(str(getattr(existing, "__file__", ""))).resolve() != producer_init:
        raise RuntimeError("top-level src package is already bound to non-producer code")
    module_name = "_state_formation_capacity_cli_v11"
    cli = sys.modules.get(module_name)
    if cli is None:
        cli_path = PRODUCER_ROOT / "scripts" / "run.py"
        spec = importlib.util.spec_from_file_location(module_name, cli_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("could not construct producer CLI module")
        cli = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = cli
        try:
            spec.loader.exec_module(cli)
        except BaseException:
            sys.modules.pop(module_name, None)
            raise
    analysis = importlib.import_module("src.analysis")
    gpu_runner = importlib.import_module("src.gpu_runner")
    config_module = importlib.import_module("src.config")
    config = cli.load_config(PRODUCER_ROOT / "configs" / "default.yaml")
    config_receipt = cli.resolved_config_receipt(config)
    checks = {
        "producer source": (cli.source_contract_sha256(PRODUCER_ROOT), EXPECTED_SOURCE),
        "reviewed implementation": (
            config_module.reviewed_implementation_sha256(PRODUCER_ROOT),
            EXPECTED_IMPLEMENTATION,
        ),
        "loaded producer source": (cli.LOADED_SOURCE_CONTRACT_SHA256, EXPECTED_SOURCE),
        "producer CLI": (_sha256(PRODUCER_ROOT / "scripts" / "run.py"), EXPECTED_CLI_SHA256),
        "producer GPU runner": (_sha256(PRODUCER_ROOT / "src" / "gpu_runner.py"), EXPECTED_GPU_RUNNER_SHA256),
        "producer analyzer": (_sha256(PRODUCER_ROOT / "src" / "analysis.py"), EXPECTED_ANALYSIS_SHA256),
        "producer config file": (_sha256(PRODUCER_ROOT / "configs" / "default.yaml"), EXPECTED_CONFIG_FILE_SHA256),
        "producer config": (config_receipt.get("config_sha256"), EXPECTED_CONFIG_SHA256),
        "model id": (config_receipt.get("model_id"), EXPECTED_MODEL_ID),
        "model revision": (config_receipt.get("model_revision"), EXPECTED_MODEL_REVISION),
        "backend": (config_receipt.get("backend"), "transformers"),
        "analysis root": (analysis.ROOT, PRODUCER_ROOT),
        "analysis repository": (analysis.REPO_ROOT, REPO_ROOT),
        "registered external prefix": (config["paths"]["large_artifacts_dir"], EXPECTED_PREFIX),
    }
    for label, (actual, expected) in checks.items():
        if actual != expected:
            raise RuntimeError(f"{label} mismatch: {actual!r} != {expected!r}")
    return ProducerContext(
        cli=cli,
        analysis=analysis,
        gpu_runner=gpu_runner,
        config=config,
        config_receipt=config_receipt,
    )


class ExactRegisteredPrefixSeam:
    def __init__(self, context: ProducerContext):
        self.context = context
        self.original = context.analysis._canonical_expected_path
        self.raw_prefix = os.fspath(PRODUCER_ROOT / EXPECTED_PREFIX)
        self.canonical_prefix = Path(os.path.abspath(self.raw_prefix))
        if self.canonical_prefix != REPO_ROOT / "large_artifacts" / PRODUCER_ID:
            raise RuntimeError("registered external prefix resolves unexpectedly")

    def __call__(self, path: Path, *, require_file: bool = False) -> Path:
        raw = os.fspath(path)
        if raw == self.raw_prefix or raw.startswith(self.raw_prefix + os.sep):
            suffix = raw[len(self.raw_prefix):].lstrip(os.sep)
            if (
                "\x00" in suffix
                or "\\" in suffix
                or (suffix and any(part in {"", ".", ".."} for part in suffix.split("/")))
            ):
                raise RuntimeError("registered external descendant is not lexical-canonical")
            lexical = Path(os.path.abspath(raw))
            try:
                relative = lexical.relative_to(REPO_ROOT).as_posix()
                lexical.relative_to(self.canonical_prefix)
            except ValueError as exc:
                raise RuntimeError("registered external descendant escapes its prefix") from exc
            return self.context.analysis.canonical_repo_path(
                REPO_ROOT, relative, require_file=require_file
            )
        return self.original(path, require_file=require_file)


@contextlib.contextmanager
def installed_path_seam(context: ProducerContext) -> Iterator[ExactRegisteredPrefixSeam]:
    seam = ExactRegisteredPrefixSeam(context)
    if context.analysis._canonical_expected_path is not seam.original:
        raise RuntimeError("producer path helper was already replaced")
    context.analysis._canonical_expected_path = seam
    try:
        yield seam
    finally:
        if context.analysis._canonical_expected_path is not seam:
            raise RuntimeError("producer path helper changed during recovery")
        context.analysis._canonical_expected_path = seam.original


def _validate_authorization_files() -> tuple[dict[str, Any], dict[str, Any]]:
    authorization, _ = _strict_json(AUTHORIZATION_PATH)
    recovery, _ = _strict_json(ANALYSIS_RECOVERY_PATH)
    _validate_identity(authorization, label="producer authorization")
    _validate_identity(recovery, label="analysis recovery")
    expected_authorization = {
        "sha256": _sha256(AUTHORIZATION_PATH),
        "receipt_identity_sha256": authorization.get("receipt_identity_sha256"),
        "status": authorization.get("status"),
        "next_stage": authorization.get("next_stage"),
    }
    if expected_authorization != {
        "sha256": EXPECTED_AUTHORIZATION_SHA256,
        "receipt_identity_sha256": EXPECTED_AUTHORIZATION_IDENTITY,
        "status": "LORA_JOINT_MISS_CONTROLS_REQUIRED",
        "next_stage": "run_lora_state_only_and_fullrank_joint",
    }:
        raise RuntimeError("producer authorization changed")
    expected_recovery = {
        "sha256": _sha256(ANALYSIS_RECOVERY_PATH),
        "receipt_identity_sha256": recovery.get("receipt_identity_sha256"),
        "producer_receipt_identity_sha256": recovery.get("producer_receipt_identity_sha256"),
        "producer_output_sha256": recovery.get("producer_output_sha256"),
        "recovery_source_contract_sha256": recovery.get("recovery_source_contract_sha256"),
    }
    if expected_recovery != {
        "sha256": EXPECTED_ANALYSIS_RECOVERY_SHA256,
        "receipt_identity_sha256": EXPECTED_ANALYSIS_RECOVERY_IDENTITY,
        "producer_receipt_identity_sha256": EXPECTED_AUTHORIZATION_IDENTITY,
        "producer_output_sha256": EXPECTED_AUTHORIZATION_SHA256,
        "recovery_source_contract_sha256": EXPECTED_ANALYSIS_RECOVERY_SOURCE,
    }:
        raise RuntimeError("analysis-recovery lineage changed")
    return authorization, recovery


def _failure_pair() -> tuple[dict[str, Any], bytes]:
    canonical, canonical_raw = _strict_json(FAILURE_CANONICAL)
    mirror, mirror_raw = _strict_json(FAILURE_MIRROR)
    if canonical_raw != mirror_raw or canonical != mirror:
        raise RuntimeError("failed G0 canonical/mirror bytes differ")
    canonical_info = os.stat(FAILURE_CANONICAL, follow_symlinks=False)
    mirror_info = os.stat(FAILURE_MIRROR, follow_symlinks=False)
    if (canonical_info.st_dev, canonical_info.st_ino) == (mirror_info.st_dev, mirror_info.st_ino):
        raise RuntimeError("failed G0 pair shares one inode")
    _validate_identity(canonical, label="failed G0")
    checks = {
        "sha256": hashlib.sha256(canonical_raw).hexdigest(),
        "receipt_identity_sha256": canonical.get("receipt_identity_sha256"),
        "status": canonical.get("status"),
        "phase": canonical.get("phase"),
        "capacity": canonical.get("capacity"),
        "model_seed": canonical.get("model_seed"),
        "failure_stage": canonical.get("failure_stage"),
        "completed_checks": canonical.get("completed_checks"),
        "training_or_evaluation_started": canonical.get("training_or_evaluation_started"),
        "scientific_evidence": canonical.get("scientific_evidence"),
        "benchmark_files_read": canonical.get("benchmark_files_read"),
        "sealed_contrast_payloads_opened": canonical.get("sealed_contrast_payloads_opened"),
    }
    expected = {
        "sha256": EXPECTED_FAILURE_SHA256,
        "receipt_identity_sha256": EXPECTED_FAILURE_IDENTITY,
        "status": "SETUP_CONTROL_FAILED",
        "phase": "fullrank_g0",
        "capacity": "fullrank",
        "model_seed": 7411,
        "failure_stage": "branch_authorization",
        "completed_checks": [],
        "training_or_evaluation_started": False,
        "scientific_evidence": False,
        "benchmark_files_read": 0,
        "sealed_contrast_payloads_opened": [],
    }
    if checks != expected or "expected path is not lexical-canonical" not in str(canonical.get("error")):
        raise RuntimeError("failed G0 is not the exact pre-model branch-authorization failure")
    return canonical, canonical_raw


def run_smoke() -> dict[str, Any]:
    context = load_producer_context()
    _validate_authorization_files()
    failure, _ = _failure_pair()
    seam = ExactRegisteredPrefixSeam(context)
    with context.cli.authorized_source_execution_snapshot() as authorized_source:
        if authorized_source != EXPECTED_SOURCE:
            raise RuntimeError("producer source changed inside smoke snapshot")
        try:
            context.gpu_runner._authorization_for(
                context.config, "fullrank", "joint", AUTHORIZATION_PATH
            )
        except RuntimeError as exc:
            original_rejection = str(exc)
        else:
            raise RuntimeError("original downstream authorization unexpectedly succeeded")
    if "expected path is not lexical-canonical" not in original_rejection:
        raise RuntimeError("original downstream authorization failed for an unexpected reason")
    canonical_equivalence = seam(seam.canonical_prefix) == seam.original(seam.canonical_prefix)
    if not canonical_equivalence:
        raise RuntimeError("seam changed canonical path semantics")
    unrelated = PRODUCER_ROOT / "data" / ".." / "data"
    try:
        seam(unrelated)
    except RuntimeError:
        unrelated_alias_rejected = True
    else:
        raise RuntimeError("seam accepted an unrelated lexical alias")
    traversal = Path(seam.raw_prefix + "/../state_formation_capacity_adjudication")
    try:
        seam(traversal)
    except RuntimeError:
        prefix_traversal_rejected = True
    else:
        raise RuntimeError("seam accepted registered-prefix traversal")
    with context.cli.authorized_source_execution_snapshot() as authorized_source:
        if authorized_source != EXPECTED_SOURCE:
            raise RuntimeError("producer source changed inside recovered smoke snapshot")
        with installed_path_seam(context):
            authorization = context.gpu_runner._authorization_for(
                context.config, "fullrank", "joint", AUTHORIZATION_PATH
            )
    helper_restored = context.analysis._canonical_expected_path is seam.original
    if not helper_restored:
        raise RuntimeError("producer path helper was not restored")
    if (
        authorization.get("receipt_identity_sha256") != EXPECTED_AUTHORIZATION_IDENTITY
        or authorization.get("status") != "LORA_JOINT_MISS_CONTROLS_REQUIRED"
        or authorization.get("next_stage") != "run_lora_state_only_and_fullrank_joint"
    ):
        raise RuntimeError("recovered downstream authorization lineage mismatch")
    contract = recovery_source_contract()
    receipt = _with_identity({
        "schema_version": 1,
        "experiment_id": RECOVERY_ID,
        "status": "BRANCH_RECOVERY_SMOKE_PASS",
        "recovery_source_contract_sha256": contract["source_contract_sha256"],
        "recovery_source_files": contract["files"],
        "producer_experiment_id": PRODUCER_ID,
        "producer_source_contract_sha256": EXPECTED_SOURCE,
        "producer_implementation_sha256": EXPECTED_IMPLEMENTATION,
        "producer_cli_sha256": EXPECTED_CLI_SHA256,
        "producer_gpu_runner_sha256": EXPECTED_GPU_RUNNER_SHA256,
        "producer_analysis_sha256": EXPECTED_ANALYSIS_SHA256,
        "producer_config_sha256": EXPECTED_CONFIG_SHA256,
        "authorization_sha256": EXPECTED_AUTHORIZATION_SHA256,
        "authorization_receipt_identity_sha256": EXPECTED_AUTHORIZATION_IDENTITY,
        "failed_g0_sha256": EXPECTED_FAILURE_SHA256,
        "failed_g0_receipt_identity_sha256": failure["receipt_identity_sha256"],
        "original_rejection": original_rejection,
        "control_passes": {
            "original_downstream_defect_reproduced": 1,
            "recovered_downstream_authorization": 1,
            "canonical_equivalence": 1,
            "unrelated_alias_rejected": 1,
            "prefix_traversal_rejected": 1,
            "helper_restored": 1,
        },
        "model_loaded": False,
        "training_or_evaluation_started": False,
        "benchmark_paths_opened": 0,
        "sealed_contrast_rows_opened": 0,
        "scientific_interpretation_changed": False,
    })
    _publish_json(ROOT / "runs" / "smoke.json", receipt)
    return receipt


def _require_smoke() -> tuple[dict[str, Any], str]:
    path = ROOT / "runs" / "smoke.json"
    smoke, raw = _strict_json(path)
    _validate_identity(smoke, label="branch-recovery smoke")
    if smoke.get("status") != "BRANCH_RECOVERY_SMOKE_PASS":
        raise RuntimeError("branch-recovery smoke did not pass")
    if smoke.get("recovery_source_contract_sha256") != recovery_source_contract()[
        "source_contract_sha256"
    ]:
        raise RuntimeError("branch-recovery source changed after smoke")
    return smoke, hashlib.sha256(raw).hexdigest()


def archive_failure() -> dict[str, Any]:
    smoke, smoke_sha256 = _require_smoke()
    failure, raw = _failure_pair()
    _publish_or_verify(ARCHIVED_FAILURE, raw)
    receipt = _with_identity({
        "schema_version": 1,
        "experiment_id": RECOVERY_ID,
        "status": "BRANCH_AUTHORIZATION_FAILURE_ARCHIVED",
        "source_paths": [
            FAILURE_CANONICAL.relative_to(REPO_ROOT).as_posix(),
            FAILURE_MIRROR.relative_to(REPO_ROOT).as_posix(),
        ],
        "archive_path": ARCHIVED_FAILURE.relative_to(REPO_ROOT).as_posix(),
        "failure_sha256": EXPECTED_FAILURE_SHA256,
        "failure_receipt_identity_sha256": failure["receipt_identity_sha256"],
        "source_pair_byte_identical": True,
        "source_pair_inode_distinct": True,
        "archive_inode_distinct": True,
        "recovery_smoke_sha256": smoke_sha256,
        "recovery_source_contract_sha256": smoke["recovery_source_contract_sha256"],
        "model_loaded": False,
        "training_or_evaluation_started": False,
        "benchmark_paths_opened": 0,
        "sealed_contrast_rows_opened": 0,
        "authorizes_retry_after_archive_publication": False,
    })
    _publish_json(ARCHIVE_RECEIPT, receipt)
    if _regular_bytes(ARCHIVED_FAILURE) != raw:
        raise RuntimeError("archived failure differs after publication")
    archive_info = os.stat(ARCHIVED_FAILURE, follow_symlinks=False)
    source_info = os.stat(FAILURE_CANONICAL, follow_symlinks=False)
    if (archive_info.st_dev, archive_info.st_ino) == (source_info.st_dev, source_info.st_ino):
        raise RuntimeError("archived failure shares a source inode")
    return receipt


def _git_blob(commit: str, path: Path) -> bytes:
    relative = path.relative_to(REPO_ROOT).as_posix()
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"archive commit lacks {relative}")
    return result.stdout


def _unlink_exact(path: Path, expected: bytes) -> None:
    if not os.path.lexists(path):
        return
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        before = os.stat(path, follow_symlinks=False)
        identity = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1 or identity(opened) != identity(before):
            raise RuntimeError(f"retirement input is not one stable inode: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        if b"".join(chunks) != expected:
            raise RuntimeError(f"retirement input changed: {path}")
        rebound = os.stat(path, follow_symlinks=False)
        if identity(opened) != identity(rebound):
            raise RuntimeError(f"retirement input rebound: {path}")
        os.unlink(path)
        if os.path.lexists(path):
            raise RuntimeError(f"retirement unlink failed: {path}")
        if identity(opened) != identity(os.fstat(descriptor)):
            raise RuntimeError(f"retired inode changed while held: {path}")
    finally:
        os.close(descriptor)


def retire_failure(archive_commit: str) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{40}", archive_commit):
        raise RuntimeError("archive commit must be a full lowercase SHA-1")
    smoke, smoke_sha256 = _require_smoke()
    archive, archive_raw = _strict_json(ARCHIVE_RECEIPT)
    _validate_identity(archive, label="failure archive")
    if archive.get("status") != "BRANCH_AUTHORIZATION_FAILURE_ARCHIVED":
        raise RuntimeError("failure archive did not complete")
    archived_raw = _regular_bytes(ARCHIVED_FAILURE)
    if hashlib.sha256(archived_raw).hexdigest() != EXPECTED_FAILURE_SHA256:
        raise RuntimeError("archived failure bytes changed")
    present = [path for path in (FAILURE_CANONICAL, FAILURE_MIRROR) if os.path.lexists(path)]
    if len(present) == 2:
        _, current_raw = _failure_pair()
    elif len(present) == 1:
        current_raw = _regular_bytes(present[0])
        if current_raw != archived_raw:
            raise RuntimeError("stranded retirement input differs from archive")
    else:
        current_raw = archived_raw
    required_git_blobs = {
        ARCHIVED_FAILURE: archived_raw,
        ARCHIVE_RECEIPT: archive_raw,
        FAILURE_CANONICAL: current_raw,
        FAILURE_MIRROR: current_raw,
    }
    for path, expected in required_git_blobs.items():
        if _git_blob(archive_commit, path) != expected:
            raise RuntimeError(f"archive commit bytes differ for {path}")
    started = _with_identity({
        "schema_version": 1,
        "experiment_id": RECOVERY_ID,
        "status": "BRANCH_AUTHORIZATION_FAILURE_RETIREMENT_STARTED",
        "archive_commit": archive_commit,
        "archive_receipt_sha256": hashlib.sha256(archive_raw).hexdigest(),
        "archived_failure_sha256": EXPECTED_FAILURE_SHA256,
        "retiring_paths": [path.relative_to(REPO_ROOT).as_posix() for path in (FAILURE_CANONICAL, FAILURE_MIRROR)],
        "recovery_smoke_sha256": smoke_sha256,
        "recovery_source_contract_sha256": smoke["recovery_source_contract_sha256"],
    })
    _publish_json(RETIREMENT_STARTED, started)
    _unlink_exact(FAILURE_CANONICAL, current_raw)
    _unlink_exact(FAILURE_MIRROR, current_raw)
    if os.path.lexists(FAILURE_CANONICAL) or os.path.lexists(FAILURE_MIRROR):
        raise RuntimeError("failed G0 source pair remains after retirement")
    receipt = _with_identity({
        "schema_version": 1,
        "experiment_id": RECOVERY_ID,
        "status": "BRANCH_AUTHORIZATION_FAILURE_RETIRED",
        "archive_commit": archive_commit,
        "archive_receipt_sha256": hashlib.sha256(archive_raw).hexdigest(),
        "archived_failure_sha256": EXPECTED_FAILURE_SHA256,
        "retired_paths": started["retiring_paths"],
        "retirement_started_sha256": _sha256(RETIREMENT_STARTED),
        "model_loaded": False,
        "training_or_evaluation_started": False,
        "benchmark_paths_opened": 0,
        "sealed_contrast_rows_opened": 0,
        "authorizes_recovered_producer_retry": True,
    })
    _publish_json(RETIREMENT_RECEIPT, receipt)
    return receipt


def _invocation_slug(arguments: Mapping[str, Any]) -> str:
    stage = str(arguments["stage"]).replace("-", "_")
    capacity = str(arguments.get("capacity") or "none")
    objective = str(arguments.get("objective") or "none")
    seed = str(arguments.get("seed") or "none")
    eval_set = str(arguments.get("eval_set") or "none")
    return f"{stage}_{capacity}_{objective}_seed{seed}_{eval_set}"


def _producer_output_leaf(arguments: Mapping[str, Any]) -> Path:
    output = _canonical(Path(str(arguments["output"])))
    stage = arguments["stage"]
    if stage in {"model-smoke", "positive-control"}:
        return output
    if stage == "train":
        return output / "run.json"
    if stage == "evaluate-state":
        return output / "summary.json"
    raise RuntimeError(f"unregistered recovered stage: {stage}")


def _completion_from_output(
    arguments: Mapping[str, Any], started_path: Path, smoke: Mapping[str, Any]
) -> dict[str, Any]:
    output = _producer_output_leaf(arguments)
    receipt, raw = _strict_json(output)
    _validate_identity(receipt, label="producer output")
    if receipt.get("source_contract_sha256") != EXPECTED_SOURCE:
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
        raise RuntimeError(
            f"recovered producer stage did not complete: {receipt.get('status')}: "
            f"{receipt.get('error')}"
        )
    return _with_identity({
        "schema_version": 1,
        "experiment_id": RECOVERY_ID,
        "status": "RECOVERED_PRODUCER_INVOCATION_COMPLETE",
        "invocation": dict(arguments),
        "producer_output": output.relative_to(REPO_ROOT).as_posix(),
        "producer_output_sha256": hashlib.sha256(raw).hexdigest(),
        "producer_receipt_identity_sha256": receipt["receipt_identity_sha256"],
        "producer_status": receipt.get("status"),
        "producer_source_contract_sha256": EXPECTED_SOURCE,
        "recovery_source_contract_sha256": smoke["recovery_source_contract_sha256"],
        "invocation_started_sha256": _sha256(started_path),
        "changed_runtime_function": "src.analysis._canonical_expected_path only",
    })


def _validate_invocation_shape(arguments: Mapping[str, Any]) -> None:
    stage = arguments.get("stage")
    capacity = arguments.get("capacity")
    objective = arguments.get("objective")
    eval_set = arguments.get("eval_set")
    if stage in {"model-smoke", "positive-control"}:
        if (capacity, objective, eval_set) != ("fullrank", "joint", "trigger"):
            raise RuntimeError("recovered setup stages must be fullrank/joint/trigger")
        if not arguments.get("initialization_bundle") or arguments.get("checkpoint"):
            raise RuntimeError("recovered setup stage input shape is invalid")
        if stage == "positive-control" and not arguments.get("model_smoke_receipt"):
            raise RuntimeError("recovered positive control requires its G0 receipt")
    elif stage == "train":
        if (capacity, objective) not in {
            ("lora", "state_only"),
            ("fullrank", "joint"),
            ("fullrank", "state_only"),
        } or eval_set != "trigger":
            raise RuntimeError("recovered training cell is not registered")
        if not all(
            arguments.get(key)
            for key in (
                "initialization_bundle",
                "model_smoke_receipt",
                "positive_control_receipt",
            )
        ) or arguments.get("checkpoint"):
            raise RuntimeError("recovered training input shape is invalid")
    elif stage == "evaluate-state":
        if (capacity, objective) not in {
            ("lora", "state_only"),
            ("fullrank", "joint"),
            ("fullrank", "state_only"),
        } or not arguments.get("checkpoint"):
            raise RuntimeError("recovered evaluation cell is not registered")
        if any(
            arguments.get(key)
            for key in (
                "initialization_bundle",
                "model_smoke_receipt",
                "positive_control_receipt",
            )
        ):
            raise RuntimeError("recovered evaluation input shape is invalid")
        if objective == "state_only" and eval_set != "trigger":
            raise RuntimeError("state-only evaluation is trigger-only")
    else:
        raise RuntimeError("producer stage is not allowed through branch recovery")


def invoke_producer(arguments: Mapping[str, Any], producer_argv: Sequence[str]) -> dict[str, Any]:
    if arguments.get("stage") not in ALLOWED_STAGES:
        raise RuntimeError("producer stage is not allowed through branch recovery")
    if not arguments.get("authorization_receipt"):
        raise RuntimeError("branch recovery requires an authorization receipt")
    if not arguments.get("output"):
        raise RuntimeError("branch recovery requires an explicit canonical producer output")
    _validate_invocation_shape(arguments)
    if os.path.lexists(FAILURE_CANONICAL) or os.path.lexists(FAILURE_MIRROR):
        raise RuntimeError("failed G0 pair must be archived and retired before recovered invocation")
    retirement, _ = _strict_json(RETIREMENT_RECEIPT)
    _validate_identity(retirement, label="failure retirement")
    if retirement.get("status") != "BRANCH_AUTHORIZATION_FAILURE_RETIRED":
        raise RuntimeError("failed G0 retirement did not complete")
    smoke, smoke_sha256 = _require_smoke()
    _validate_authorization_files()
    context = load_producer_context()
    slug = _invocation_slug(arguments)
    started_path = ROOT / "runs" / "invocations" / f"{slug}_started.json"
    complete_path = ROOT / "runs" / "invocations" / f"{slug}_complete.json"
    output_leaf = _producer_output_leaf(arguments)
    started = _with_identity({
        "schema_version": 1,
        "experiment_id": RECOVERY_ID,
        "status": "RECOVERED_PRODUCER_INVOCATION_STARTED",
        "invocation": dict(arguments),
        "producer_argv": list(producer_argv),
        "producer_source_contract_sha256": EXPECTED_SOURCE,
        "producer_cli_sha256": EXPECTED_CLI_SHA256,
        "producer_analysis_sha256": EXPECTED_ANALYSIS_SHA256,
        "authorization_sha256": _sha256(
            _canonical(Path(str(arguments["authorization_receipt"])))
        ),
        "recovery_smoke_sha256": smoke_sha256,
        "recovery_source_contract_sha256": smoke["recovery_source_contract_sha256"],
        "changed_runtime_function": "src.analysis._canonical_expected_path only",
    })
    started_preexisting = os.path.lexists(started_path)
    output_preexisting = os.path.lexists(output_leaf)
    if output_preexisting and not started_preexisting:
        raise RuntimeError("producer output predates its recovery invocation receipt")
    _publish_json(started_path, started)
    if os.path.lexists(complete_path):
        complete, _ = _strict_json(complete_path)
        _validate_identity(complete, label="recovered invocation")
        expected = _completion_from_output(arguments, started_path, smoke)
        if complete != expected:
            raise RuntimeError("recovered invocation completion changed")
        return complete
    if not os.path.lexists(output_leaf):
        with installed_path_seam(context):
            result = context.cli.main(list(producer_argv))
        if result != 0:
            raise RuntimeError(f"producer CLI returned nonzero status: {result}")
    complete = _completion_from_output(arguments, started_path, smoke)
    _publish_json(complete_path, complete)
    return complete
