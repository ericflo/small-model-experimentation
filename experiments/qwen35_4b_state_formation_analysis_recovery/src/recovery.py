"""Run the immutable v11 analyzer through one exact registered-path seam.

This module does not copy or modify the producer's scientific analysis. It
loads that source tree under an isolated package name, verifies its complete
source contract, and temporarily replaces only ``_canonical_expected_path``.
The replacement accepts the producer's exact registered external-artifact
prefix (and lexically clean descendants) and delegates every other path to the
original v11 helper.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import importlib.util
import json
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
PRODUCER_ROOT = REPO_ROOT / "experiments" / "qwen35_4b_state_formation_capacity_adjudication"
PRODUCER_PACKAGE = "_state_formation_capacity_v11"

RECOVERY_ID = "qwen35_4b_state_formation_analysis_recovery"
PRODUCER_ID = "qwen35_4b_state_formation_capacity_adjudication"
EXPECTED_MODEL_ID = "Qwen/Qwen3.5-4B"
EXPECTED_MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
EXPECTED_BACKEND = "transformers"
EXPECTED_SOURCE_CONTRACT_VERSION = 11
EXPECTED_SOURCE_CONTRACT_SHA256 = (
    "5a8ed26ddb9446c728191ca8e7849ae44cff92a700e24b237dac522cf4286666"
)
EXPECTED_IMPLEMENTATION_SHA256 = (
    "7d6cd93fead0e524e10e7afe4b60a531ea2d6aa7f3f70778ef962889aaeed278"
)
EXPECTED_ANALYSIS_SHA256 = (
    "876888987d816fe29ae93fde0053fb91ed58301d16bf429d5cabaa809c23a2b0"
)
EXPECTED_CONFIG_FILE_SHA256 = (
    "b165537c8c86531ac17aecfa5c65045a708cd4b196b7614c6f98331a9fae1ca8"
)
EXPECTED_CONFIG_SHA256 = (
    "eeb4e828526f750dce1258bcc91d03114c80688d300112e03d18c9d911489393"
)
EXPECTED_EXTERNAL_PREFIX = (
    "../../large_artifacts/qwen35_4b_state_formation_capacity_adjudication"
)

RECOVERY_CONTRACT_FILES = (
    "configs/default.yaml",
    "reports/design_review.md",
    "scripts/run.py",
    "src/__init__.py",
    "src/recovery.py",
    "tests/test_recovery.py",
)
PHASE_OUTPUTS = {
    "lora_joint": "lora_joint_trigger.json",
    "lora_control": "lora_control.json",
    "stage_b_seal": "stage_b_seal.json",
    "fullrank_joint": "fullrank_joint.json",
    "fullrank_control": "summary.json",
}


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _with_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["receipt_identity_sha256"] = hashlib.sha256(
        _canonical_json(result).encode("utf-8")
    ).hexdigest()
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(_regular_bytes(path)).hexdigest()


def _regular_bytes(path: Path) -> bytes:
    """Read one inode-distinct canonical regular file without following aliases."""

    canonical = Path(os.path.abspath(os.fspath(path)))
    if canonical != path or canonical.resolve(strict=True) != canonical:
        raise RuntimeError(f"recovery input is not canonical: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(canonical, flags)
    try:
        opened = os.fstat(descriptor)
        before = os.stat(canonical, follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
            != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        ):
            raise RuntimeError(f"recovery input is not one stable regular inode: {path}")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        rebound = os.stat(canonical, follow_symlinks=False)
        identity = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
        if identity(opened) != identity(after) or identity(after) != identity(rebound):
            raise RuntimeError(f"recovery input changed while read: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _strict_json_bytes(raw: bytes, path: Path) -> dict[str, Any]:
    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise RuntimeError(f"nonfinite JSON value in {path}: {value}")

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=reject_duplicate,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid recovery JSON: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"recovery JSON is not an object: {path}")
    return value


def _stable_bytes(context: "ProducerContext", path: Path) -> bytes:
    with context.safe_io.open_stable_regular(REPO_ROOT, path) as handle:
        return handle.read()


def _strict_json(context: "ProducerContext", path: Path) -> dict[str, Any]:
    return _strict_json_bytes(_stable_bytes(context, path), path)


def _validate_receipt_identity(receipt: Mapping[str, Any], *, label: str) -> None:
    claimed = receipt.get("receipt_identity_sha256")
    payload = {key: value for key, value in receipt.items() if key != "receipt_identity_sha256"}
    expected = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    if claimed != expected:
        raise RuntimeError(f"{label} receipt identity mismatch")


def _read_recovery_config() -> dict[str, Any]:
    path = ROOT / "configs" / "default.yaml"
    try:
        config = yaml.safe_load(_regular_bytes(path).decode("utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise RuntimeError("recovery configuration is unavailable") from exc
    if not isinstance(config, dict):
        raise RuntimeError("recovery configuration must be a mapping")
    expected = {
        "schema_version": 1,
        "experiment_id": RECOVERY_ID,
        "producer_experiment_id": PRODUCER_ID,
        "producer_source_contract_version": EXPECTED_SOURCE_CONTRACT_VERSION,
        "producer_source_contract_sha256": EXPECTED_SOURCE_CONTRACT_SHA256,
        "producer_implementation_sha256": EXPECTED_IMPLEMENTATION_SHA256,
        "producer_analysis_sha256": EXPECTED_ANALYSIS_SHA256,
        "producer_config_file_sha256": EXPECTED_CONFIG_FILE_SHA256,
        "producer_config_sha256": EXPECTED_CONFIG_SHA256,
        "registered_external_prefix": EXPECTED_EXTERNAL_PREFIX,
        "model_id": EXPECTED_MODEL_ID,
        "model_revision": EXPECTED_MODEL_REVISION,
        "backend": EXPECTED_BACKEND,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise RuntimeError(f"recovery configuration changed {key}")
    if config.get("allowed_phases") != list(PHASE_OUTPUTS):
        raise RuntimeError("recovery configuration changed the analysis phase order")
    return config


def recovery_source_contract() -> dict[str, Any]:
    """Hash the executable, tests, frozen config, and adversarial review."""

    files = []
    for relative in RECOVERY_CONTRACT_FILES:
        path = ROOT / relative
        try:
            info = os.lstat(path)
        except OSError as exc:
            raise RuntimeError(f"recovery contract file is missing: {relative}") from exc
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RuntimeError(
                f"recovery contract file is not an inode-distinct regular file: {relative}"
            )
        files.append({"path": relative, "sha256": _sha256(path), "bytes": info.st_size})
    payload = {"schema_version": 1, "files": files}
    return {
        **payload,
        "source_contract_sha256": hashlib.sha256(
            _canonical_json(payload).encode("utf-8")
        ).hexdigest(),
    }


def _load_producer_package() -> ModuleType:
    existing = sys.modules.get(PRODUCER_PACKAGE)
    if existing is not None:
        if Path(str(existing.__file__)).resolve() != (PRODUCER_ROOT / "src" / "__init__.py"):
            raise RuntimeError("producer package alias is already bound to different source")
        return existing
    init_path = PRODUCER_ROOT / "src" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        PRODUCER_PACKAGE,
        init_path,
        submodule_search_locations=[str(PRODUCER_ROOT / "src")],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not construct isolated producer package")
    package = importlib.util.module_from_spec(spec)
    sys.modules[PRODUCER_PACKAGE] = package
    try:
        spec.loader.exec_module(package)
    except BaseException:
        sys.modules.pop(PRODUCER_PACKAGE, None)
        raise
    return package


@dataclass(frozen=True)
class ProducerContext:
    config: dict[str, Any]
    config_module: ModuleType
    analysis: ModuleType
    design_boundary: ModuleType
    attempt_receipts: ModuleType
    safe_io: ModuleType
    recovery_config: dict[str, Any]


def load_producer_context() -> ProducerContext:
    """Load and independently pin every producer identity used by recovery."""

    recovery_config = _read_recovery_config()
    _load_producer_package()
    config_module = importlib.import_module(f"{PRODUCER_PACKAGE}.config")
    analysis = importlib.import_module(f"{PRODUCER_PACKAGE}.analysis")
    design_boundary = importlib.import_module(f"{PRODUCER_PACKAGE}.design_boundary")
    attempt_receipts = importlib.import_module(f"{PRODUCER_PACKAGE}.attempt_receipts")
    safe_io = importlib.import_module(f"{PRODUCER_PACKAGE}.safe_io")
    config = config_module.load_config(PRODUCER_ROOT / "configs" / "default.yaml")
    receipt = config_module.resolved_config_receipt(config)
    checks = {
        "producer experiment": (config_module.EXPERIMENT_ID, PRODUCER_ID),
        "source-contract version": (
            config_module.SOURCE_CONTRACT_VERSION,
            EXPECTED_SOURCE_CONTRACT_VERSION,
        ),
        "source contract": (
            config_module.source_contract_sha256(PRODUCER_ROOT),
            EXPECTED_SOURCE_CONTRACT_SHA256,
        ),
        "reviewed implementation": (
            config_module.reviewed_implementation_sha256(PRODUCER_ROOT),
            EXPECTED_IMPLEMENTATION_SHA256,
        ),
        "analysis source": (
            _sha256(PRODUCER_ROOT / "src" / "analysis.py"),
            EXPECTED_ANALYSIS_SHA256,
        ),
        "config file": (
            _sha256(PRODUCER_ROOT / "configs" / "default.yaml"),
            EXPECTED_CONFIG_FILE_SHA256,
        ),
        "resolved config": (receipt.get("config_sha256"), EXPECTED_CONFIG_SHA256),
        "model id": (receipt.get("model_id"), EXPECTED_MODEL_ID),
        "model revision": (receipt.get("model_revision"), EXPECTED_MODEL_REVISION),
        "backend": (receipt.get("backend"), EXPECTED_BACKEND),
        "registered external prefix": (
            config.get("paths", {}).get("large_artifacts_dir"),
            EXPECTED_EXTERNAL_PREFIX,
        ),
        "analysis root": (analysis.ROOT, PRODUCER_ROOT),
        "analysis repository": (analysis.REPO_ROOT, REPO_ROOT),
    }
    for label, (actual, expected) in checks.items():
        if actual != expected:
            raise RuntimeError(f"{label} mismatch: {actual!r} != {expected!r}")
    return ProducerContext(
        config=config,
        config_module=config_module,
        analysis=analysis,
        design_boundary=design_boundary,
        attempt_receipts=attempt_receipts,
        safe_io=safe_io,
        recovery_config=recovery_config,
    )


class ExactRegisteredPrefixSeam:
    """Accept only the frozen producer's one registered nonlexical prefix."""

    def __init__(self, context: ProducerContext):
        self.context = context
        self.original = context.analysis._canonical_expected_path
        self.raw_prefix = os.fspath(
            PRODUCER_ROOT / str(context.config["paths"]["large_artifacts_dir"])
        )
        expected_raw = os.fspath(PRODUCER_ROOT / EXPECTED_EXTERNAL_PREFIX)
        if self.raw_prefix != expected_raw:
            raise RuntimeError("registered external prefix construction changed")
        self.canonical_prefix = Path(os.path.abspath(self.raw_prefix))
        expected_canonical = REPO_ROOT / "large_artifacts" / PRODUCER_ID
        if self.canonical_prefix != expected_canonical:
            raise RuntimeError("registered external prefix resolves somewhere unexpected")

    def __call__(self, path: Path, *, require_file: bool = False) -> Path:
        raw = os.fspath(path)
        if raw == self.raw_prefix or raw.startswith(self.raw_prefix + os.sep):
            suffix = raw[len(self.raw_prefix) :]
            if suffix.startswith(os.sep):
                suffix = suffix[1:]
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


def seam_preflight(context: ProducerContext) -> dict[str, Any]:
    """Prove the observed defect and the exact scope of the recovery seam."""

    seam = ExactRegisteredPrefixSeam(context)
    try:
        seam.original(Path(seam.raw_prefix))
    except RuntimeError as exc:
        original_rejection = str(exc)
    else:
        raise RuntimeError("v11 no longer reproduces the registered-prefix defect")
    if "expected path is not lexical-canonical" not in original_rejection:
        raise RuntimeError("v11 failed for an unexpected reason")
    canonical_original = seam.original(seam.canonical_prefix)
    canonical_recovered = seam(seam.canonical_prefix)
    recovered_prefix = seam(Path(seam.raw_prefix))
    descendant_raw = Path(seam.raw_prefix) / "lora_joint_seed7411"
    recovered_descendant = seam(descendant_raw)
    expected_descendant = seam.canonical_prefix / "lora_joint_seed7411"
    if not (
        canonical_original
        == canonical_recovered
        == recovered_prefix
        == seam.canonical_prefix
        and recovered_descendant == expected_descendant
    ):
        raise RuntimeError("recovery seam changed canonical path semantics")
    unrelated = PRODUCER_ROOT / "data" / ".." / "data"
    try:
        seam(unrelated)
    except RuntimeError:
        unrelated_alias_rejected = True
    else:
        unrelated_alias_rejected = False
    if not unrelated_alias_rejected:
        raise RuntimeError("recovery seam broadened to an unrelated lexical alias")
    unsafe_descendant = Path(seam.raw_prefix + "/../state_formation_capacity_adjudication")
    try:
        seam(unsafe_descendant)
    except RuntimeError:
        unsafe_descendant_rejected = True
    else:
        unsafe_descendant_rejected = False
    if not unsafe_descendant_rejected:
        raise RuntimeError("recovery seam accepted a traversal descendant")
    return {
        "status": "EXACT_REGISTERED_PREFIX_SEAM_READY",
        "producer_source_contract_sha256": EXPECTED_SOURCE_CONTRACT_SHA256,
        "producer_analysis_sha256": EXPECTED_ANALYSIS_SHA256,
        "registered_raw_prefix": seam.raw_prefix,
        "canonical_prefix": seam.canonical_prefix.relative_to(REPO_ROOT).as_posix(),
        "original_rejection": original_rejection,
        "canonical_equivalence": True,
        "registered_descendant_equivalence": True,
        "unrelated_alias_rejected": unrelated_alias_rejected,
        "unsafe_descendant_rejected": unsafe_descendant_rejected,
        "control_passes": {
            "original_defect_reproduced": 1,
            "canonical_equivalence": 1,
            "unrelated_alias_rejected": 1,
            "unsafe_descendant_rejected": 1,
        },
        "result_rows_opened": 0,
        "benchmark_paths_opened": 0,
        "sealed_contrast_rows_opened": 0,
    }


def _publish_or_verify(context: ProducerContext, path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    if path.exists():
        existing = _stable_bytes(context, path)
        if existing != encoded:
            raise RuntimeError(f"immutable recovery output already differs: {path}")
        return
    context.safe_io.publish_new_bytes(REPO_ROOT, path, encoded, mode=0o644)


def run_smoke() -> dict[str, Any]:
    context = load_producer_context()
    with context.design_boundary.authorized_source_execution_snapshot() as authorized:
        if authorized != EXPECTED_SOURCE_CONTRACT_SHA256:
            raise RuntimeError("producer source changed inside the smoke snapshot")
        contract = recovery_source_contract()
        preflight = seam_preflight(context)
    receipt = _with_identity({
        "schema_version": 1,
        "experiment_id": RECOVERY_ID,
        "status": "RECOVERY_SMOKE_PASS",
        "recovery_source_contract_sha256": contract["source_contract_sha256"],
        "recovery_source_files": contract["files"],
        "producer_experiment_id": PRODUCER_ID,
        "producer_source_contract_version": EXPECTED_SOURCE_CONTRACT_VERSION,
        "producer_source_contract_sha256": EXPECTED_SOURCE_CONTRACT_SHA256,
        "producer_implementation_sha256": EXPECTED_IMPLEMENTATION_SHA256,
        "producer_analysis_sha256": EXPECTED_ANALYSIS_SHA256,
        "producer_config_sha256": EXPECTED_CONFIG_SHA256,
        "model_id": EXPECTED_MODEL_ID,
        "model_revision": EXPECTED_MODEL_REVISION,
        "backend": EXPECTED_BACKEND,
        "seam_preflight": preflight,
        "scientific_analysis_executed": False,
        "result_values_inspected": False,
    })
    _publish_or_verify(context, ROOT / "runs" / "smoke.json", receipt)
    return receipt


def _require_frozen_smoke(context: ProducerContext) -> tuple[dict[str, Any], str]:
    path = ROOT / "runs" / "smoke.json"
    raw = _stable_bytes(context, path)
    receipt = _strict_json_bytes(raw, path)
    _validate_receipt_identity(receipt, label="smoke")
    if receipt.get("status") != "RECOVERY_SMOKE_PASS":
        raise RuntimeError("recovery smoke did not pass")
    current = recovery_source_contract()
    if receipt.get("recovery_source_contract_sha256") != current["source_contract_sha256"]:
        raise RuntimeError("recovery source changed after its smoke was frozen")
    if receipt.get("seam_preflight") != seam_preflight(context):
        raise RuntimeError("recovery seam preflight changed after freeze")
    return receipt, hashlib.sha256(raw).hexdigest()


def _phase_output(phase: str) -> Path:
    try:
        filename = PHASE_OUTPUTS[phase]
    except KeyError as exc:
        raise RuntimeError(f"unregistered recovery phase: {phase}") from exc
    return PRODUCER_ROOT / "analysis" / filename


def _analysis_receipt_phase(phase: str) -> str:
    return {
        "lora_joint": "lora_joint_analysis",
        "lora_control": "lora_control_analysis",
        "stage_b_seal": "stage_b_seal_analysis",
        "fullrank_joint": "fullrank_joint_analysis",
        "fullrank_control": "fullrank_control_analysis",
    }[phase]


def _phase_authorization(phase: str) -> Path | None:
    analysis_dir = PRODUCER_ROOT / "analysis"
    if phase == "lora_joint":
        return None
    if phase in {"lora_control", "stage_b_seal"}:
        return analysis_dir / "lora_joint_trigger.json"
    if phase == "fullrank_joint":
        return analysis_dir / "stage_b_seal.json"
    if phase == "fullrank_control":
        postcontrast = analysis_dir / "fullrank_joint.json"
        return postcontrast if postcontrast.exists() else analysis_dir / "stage_b_seal.json"
    raise RuntimeError(f"unregistered recovery phase: {phase}")


def _validate_existing_analysis(
    context: ProducerContext, phase: str, output: Path
) -> tuple[dict[str, Any], str]:
    raw = _stable_bytes(context, output)
    receipt = _strict_json_bytes(raw, output)
    claimed = receipt.get("receipt_identity_sha256")
    payload = {key: value for key, value in receipt.items() if key != "receipt_identity_sha256"}
    if claimed != context.analysis._canonical_sha256(payload):
        raise RuntimeError("producer analysis receipt identity mismatch")
    expected = context.analysis._identity(context.config, _analysis_receipt_phase(phase))
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise RuntimeError(f"producer analysis receipt changed {key}")
    if receipt.get("analysis_phase") != phase:
        raise RuntimeError("producer analysis phase mismatch")
    return receipt, hashlib.sha256(raw).hexdigest()


def run_analysis(phase: str) -> dict[str, Any]:
    """Execute one original v11 analysis phase and bind it in a recovery sidecar."""

    context = load_producer_context()
    smoke, smoke_sha256 = _require_frozen_smoke(context)
    output = _phase_output(phase)
    sidecar = ROOT / "analysis" / f"{phase}_recovery.json"
    attempt_path = ROOT / "runs" / f"{phase}_attempt.json"
    lock = PRODUCER_ROOT / "runs" / "run.lock"
    with context.attempt_receipts.locked_regular(lock):
        # Authorization selection, STARTED publication, producer output, and
        # recovery sidecar are one cooperating-writer critical section.
        authorization = _phase_authorization(phase)
        output_preexisting = os.path.lexists(output)
        attempt_preexisting = os.path.lexists(attempt_path)
        if output_preexisting and not attempt_preexisting:
            raise RuntimeError(
                "producer analysis output predates its recovery STARTED receipt"
            )
        if os.path.lexists(sidecar) and not output_preexisting:
            raise RuntimeError("recovery sidecar exists without its producer output")
        attempt = _with_identity({
            "schema_version": 1,
            "experiment_id": RECOVERY_ID,
            "status": "RECOVERY_ANALYSIS_STARTED",
            "phase": phase,
            "producer_output": output.relative_to(REPO_ROOT).as_posix(),
            "producer_authorization": (
                authorization.relative_to(REPO_ROOT).as_posix() if authorization else None
            ),
            "producer_source_contract_sha256": EXPECTED_SOURCE_CONTRACT_SHA256,
            "producer_analysis_sha256": EXPECTED_ANALYSIS_SHA256,
            "producer_config_sha256": EXPECTED_CONFIG_SHA256,
            "recovery_source_contract_sha256": smoke["recovery_source_contract_sha256"],
            "recovery_smoke_sha256": smoke_sha256,
        })
        _publish_or_verify(context, attempt_path, attempt)

        with context.design_boundary.authorized_source_execution_snapshot() as authorized:
            if authorized != EXPECTED_SOURCE_CONTRACT_SHA256:
                raise RuntimeError("producer source changed inside its execution snapshot")
            if output.exists():
                summary, output_sha256 = _validate_existing_analysis(context, phase, output)
                resumed_existing_output = True
            else:
                with installed_path_seam(context):
                    summary = context.analysis.analyze_phase(
                        context.config,
                        PRODUCER_ROOT / "runs",
                        phase,
                        output,
                        authorization,
                    )
                resumed_existing_output = False
                reopened, output_sha256 = _validate_existing_analysis(context, phase, output)
                if summary != reopened:
                    raise RuntimeError("producer analyzer return differs from published receipt")
        recovery_receipt = _with_identity({
            "schema_version": 1,
            "experiment_id": RECOVERY_ID,
            "status": "RECOVERED_V11_ANALYSIS_COMPLETE",
            "phase": phase,
            "producer_output": output.relative_to(REPO_ROOT).as_posix(),
            "producer_authorization": (
                authorization.relative_to(REPO_ROOT).as_posix() if authorization else None
            ),
            "producer_output_sha256": output_sha256,
            "producer_receipt_identity_sha256": summary["receipt_identity_sha256"],
            "producer_status": summary["status"],
            "producer_verdict": summary["verdict"],
            "producer_next_stage": summary["next_stage"],
            "producer_source_contract_sha256": EXPECTED_SOURCE_CONTRACT_SHA256,
            "producer_implementation_sha256": EXPECTED_IMPLEMENTATION_SHA256,
            "producer_analysis_sha256": EXPECTED_ANALYSIS_SHA256,
            "producer_config_sha256": EXPECTED_CONFIG_SHA256,
            "recovery_source_contract_sha256": smoke["recovery_source_contract_sha256"],
            "recovery_smoke_sha256": smoke_sha256,
            "attempt_receipt_sha256": _sha256(attempt_path),
            "path_seam": smoke["seam_preflight"],
            "resumed_existing_output": resumed_existing_output,
            "scientific_functions": "exact immutable producer v11 analysis.py",
            "changed_runtime_function": "_canonical_expected_path only",
        })
        _publish_or_verify(context, sidecar, recovery_receipt)
    return recovery_receipt
