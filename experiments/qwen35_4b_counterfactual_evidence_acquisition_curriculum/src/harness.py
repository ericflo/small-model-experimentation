"""Experiment-local factory for the pinned vLLM runner."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from vllm_runner import EngineConfig, VLLMRunner


# One canonical lock schema shared by every model-facing entrypoint.  Keep the
# order stable: the design receipt records it verbatim, making omissions and
# locally invented subsets invalid rather than merely unusual.
REQUIRED_FROZEN_FILES = (
    "configs/default.yaml",
    "idea_intake.md",
    "reports/preregistration.md",
    "reports/design_review.md",
    "data/locality_contexts.json",
    "src/bank.py",
    "src/harness.py",
    "src/repo_agent.py",
    "src/repo_tasks.py",
    "src/retention_tasks_legacy.py",
    "src/vllm_runner.py",
    "scripts/build_bank.py",
    "scripts/build_locality_contexts.py",
    "scripts/train.py",
    "scripts/merge_adapter.py",
    "scripts/eval_repo_agent.py",
    "scripts/eval_retention.py",
    "scripts/analyze_interface_preflight.py",
    "scripts/select_interface_band.py",
    "scripts/analyze_qualification.py",
    "scripts/analyze_calibration.py",
    "scripts/analyze_transfer_feasibility.py",
    "scripts/analyze_transfer.py",
    "scripts/analyze_sample_pool.py",
    "scripts/downstream_common.py",
    "scripts/analyze_retention.py",
    "scripts/audit_context_geometry.py",
    "scripts/audit_locality.py",
    "scripts/audit_transition_uncertainty.py",
    "scripts/bench.py",
    "scripts/analyze_menagerie.py",
    "scripts/run.py",
    "tests/test_spec_acquisition.py",
    "tests/test_vllm_runner.py",
    "tests/test_downstream_analyzers.py",
)


# AutoTokenizer may change behavior when any of these top-level artifacts changes
# or appears.  Keep this experiment's checkpoint convention deliberately narrow:
# every checkpoint must contain exactly the same three tokenizer artifacts.
TOKENIZER_FILE_NAMES = (
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
)
TOKENIZER_PROVENANCE_KEYS = (
    "tokenizer_files",
    "tokenizer_manifest_sha256",
    "tokenizer_compatibility_sha256",
)
_OTHER_TOKENIZER_FILE_NAMES = frozenset(
    {
        "added_tokens.json",
        "merges.txt",
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "spiece.model",
        "tokenizer.model",
        "vocab.json",
        "vocab.txt",
    }
)
_TOKENIZER_LOCATION_ONLY_CONFIG_KEYS = frozenset(
    {"is_local", "local_files_only"}
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_tokenizer_artifact_name(name: str) -> bool:
    return (
        name.startswith("tokenizer")
        or name.startswith("chat_template")
        or name in _OTHER_TOKENIZER_FILE_NAMES
    )


def tokenizer_file_manifest(model_path: Path) -> dict[str, str]:
    """Return the exact, deterministic tokenizer-file manifest for a checkpoint.

    Unexpected tokenizer-like files are rejected because their mere presence can
    change ``AutoTokenizer.from_pretrained`` loading precedence even if the three
    registered files remain byte-identical.
    """

    model_path = Path(model_path)
    if not model_path.is_dir():
        raise ValueError(f"tokenizer checkpoint is not a directory: {model_path}")
    discovered = {
        path.name
        for path in model_path.iterdir()
        if path.is_file() and _is_tokenizer_artifact_name(path.name)
    }
    required = set(TOKENIZER_FILE_NAMES)
    if discovered != required:
        missing = sorted(required - discovered)
        unexpected = sorted(discovered - required)
        raise ValueError(
            "tokenizer artifact convention mismatch at "
            f"{model_path}: missing={missing}, unexpected={unexpected}"
        )
    return {
        name: _sha256_file(model_path / name)
        for name in TOKENIZER_FILE_NAMES
    }


def tokenizer_manifest_sha256(manifest: object) -> str:
    """Hash a raw manifest using the experiment's canonical JSON convention."""

    if not isinstance(manifest, dict) or set(manifest) != set(TOKENIZER_FILE_NAMES):
        raise ValueError("tokenizer manifest has the wrong file set")
    return _canonical_json_sha256(manifest)


def tokenizer_compatibility_manifest(model_path: Path) -> dict[str, str]:
    """Return the behavior-relevant tokenizer manifest.

    The two dropped tokenizer-config fields describe only where the tokenizer was
    loaded from.  Every other byte/field remains compatibility-significant.
    """

    model_path = Path(model_path)
    raw = tokenizer_file_manifest(model_path)
    try:
        config = json.loads(
            (model_path / "tokenizer_config.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid tokenizer_config.json at {model_path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"tokenizer_config.json is not an object at {model_path}")
    semantic_config = {
        key: value
        for key, value in config.items()
        if key not in _TOKENIZER_LOCATION_ONLY_CONFIG_KEYS
    }
    return {
        **raw,
        "tokenizer_config.json": _canonical_json_sha256(semantic_config),
    }


def tokenizer_provenance(model_path: Path) -> dict[str, object]:
    """Return raw identity plus the narrow cross-checkpoint compatibility hash."""

    raw = tokenizer_file_manifest(model_path)
    compatibility = tokenizer_compatibility_manifest(model_path)
    return {
        "tokenizer_files": raw,
        "tokenizer_manifest_sha256": tokenizer_manifest_sha256(raw),
        "tokenizer_compatibility_sha256": tokenizer_manifest_sha256(compatibility),
    }


def validate_tokenizer_file_manifest(
    model_path: Path, expected: object
) -> dict[str, str]:
    """Re-hash ``model_path`` and require exact equality with ``expected``."""

    if not isinstance(expected, dict) or set(expected) != set(TOKENIZER_FILE_NAMES):
        raise ValueError("registered tokenizer manifest has the wrong file set")
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in expected.values()
    ):
        raise ValueError("registered tokenizer manifest has an invalid SHA-256")
    observed = tokenizer_file_manifest(model_path)
    if observed != expected:
        raise ValueError(
            f"tokenizer file manifest drift at {Path(model_path)}: "
            f"observed={observed}, expected={expected}"
        )
    return observed


def validate_tokenizer_provenance(
    model_path: Path, expected: object
) -> dict[str, object]:
    """Recompute and require the complete tokenizer provenance object."""

    if not isinstance(expected, dict):
        raise ValueError("registered tokenizer provenance is not an object")
    observed = tokenizer_provenance(model_path)
    if observed != expected:
        raise ValueError(
            f"tokenizer provenance drift at {Path(model_path)}: "
            f"observed={observed}, expected={expected}"
        )
    return observed


def validate_registered_tokenizer_provenance(
    model_path: Path,
    registration: object,
    *,
    allow_absent: bool = False,
) -> dict[str, object]:
    """Validate flat tokenizer fields embedded in an evaluation/merge receipt."""

    if not isinstance(registration, dict):
        raise ValueError("tokenizer registration is not an object")
    present = {key for key in TOKENIZER_PROVENANCE_KEYS if key in registration}
    if not present:
        if allow_absent:
            return tokenizer_provenance(model_path)
        raise ValueError("receipt has no tokenizer provenance")
    if present != set(TOKENIZER_PROVENANCE_KEYS):
        raise ValueError(f"receipt has partial tokenizer provenance: {sorted(present)}")
    expected = {key: registration[key] for key in TOKENIZER_PROVENANCE_KEYS}
    return validate_tokenizer_provenance(model_path, expected)


def validate_model_execution_lock(
    experiment_path: Path,
    lock_path: Path,
    caller_relative_path: str,
) -> dict:
    """Fail closed unless a model-facing entrypoint is frozen and pushed."""
    experiment_path = Path(experiment_path).resolve()
    repository = experiment_path.parents[1]
    lock_path = Path(lock_path).resolve()
    expected_lock = experiment_path / "runs" / "preregistration_receipt.json"
    if lock_path != expected_lock or not lock_path.is_file():
        raise ValueError("model execution requires the registered design-lock path")
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid design-lock receipt: {exc}") from exc
    frozen_order = payload.get("frozen_file_order")
    frozen_files = payload.get("frozen_files")
    design_commit = payload.get("design_commit")
    if (
        payload.get("schema_version") != 1
        or payload.get("status") != "locked"
        or payload.get("experiment_id") != experiment_path.name
        or payload.get("model_output_precedes_lock") is not False
        or not isinstance(frozen_order, list)
        or not isinstance(frozen_files, dict)
        or tuple(frozen_order) != REQUIRED_FROZEN_FILES
        or set(frozen_files) != set(REQUIRED_FROZEN_FILES)
        or not isinstance(design_commit, str)
        or len(design_commit) != 40
        or any(character not in "0123456789abcdef" for character in design_commit)
        or frozen_files.get(caller_relative_path)
        != _sha256_file(experiment_path / caller_relative_path)
    ):
        raise ValueError("model-facing entrypoint is not covered by the immutable lock")
    for relative, expected in frozen_files.items():
        path = experiment_path / relative
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"frozen design changed after lock: {relative}")
    commands = (
        ["git", "merge-base", "--is-ancestor", design_commit, "HEAD"],
        [
            "git", "ls-files", "--error-unmatch",
            str(lock_path.relative_to(repository)),
        ],
    )
    for command in commands:
        if subprocess.run(
            command,
            cwd=repository,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode:
            raise ValueError("design lock is not tracked on the current ancestry")
    dirty = subprocess.run(
        ["git", "status", "--short", "--", str(lock_path)],
        cwd=repository,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository,
        text=True, capture_output=True, check=True,
    ).stdout.strip()
    origin = subprocess.run(
        ["git", "rev-parse", "origin/main"], cwd=repository,
        text=True, capture_output=True, check=True,
    ).stdout.strip()
    if dirty or head != origin:
        raise ValueError("design lock must be clean and pushed to origin/main")
    return payload


def validate_canonical_config_path(
    experiment_path: Path, config_path: Path
) -> Path:
    """Reject alternate model-facing configs before any checkpoint is loaded."""

    expected = Path(experiment_path).resolve() / "configs" / "default.yaml"
    observed = Path(config_path).resolve()
    if observed != expected or not expected.is_file():
        raise ValueError(
            f"model execution requires the frozen config path: {expected}"
        )
    return expected


def _resolve_registered_path(repository: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"missing registered checkpoint path: {label}")
    path = Path(value)
    return (path if path.is_absolute() else repository / path).resolve()


def validate_registered_checkpoint(
    experiment_path: Path,
    model_path: Path,
    cfg: dict,
    design_lock_path: Path,
    role: str,
) -> dict[str, object]:
    """Validate one frozen parent or experiment-produced checkpoint role.

    This is intentionally file-only: callers run it before constructing a model.
    """

    experiment_path = Path(experiment_path).resolve()
    repository = experiment_path.parents[1]
    model_path = Path(model_path).resolve()
    model_cfg = cfg.get("model")
    artifacts_cfg = cfg.get("artifacts")
    if not isinstance(model_cfg, dict) or not isinstance(artifacts_cfg, dict):
        raise ValueError("checkpoint registry is missing from frozen config")
    trained_roles = {
        "evidence_binding",
        "explicit_redundant",
        "shuffled_binding",
    }
    if role == "start":
        expected_path = _resolve_registered_path(
            repository, model_cfg.get("start_checkpoint"), "start_checkpoint"
        )
        expected_weight = model_cfg.get("start_weight_sha256")
        expected_tokenizer = model_cfg.get("start_tokenizer_manifest_sha256")
        expected_config = model_cfg.get("start_config_sha256")
        expected_generation_config = model_cfg.get(
            "start_generation_config_sha256"
        )
        expected_merge_receipt = model_cfg.get("start_merge_receipt_sha256")
    elif role == "anchor":
        expected_path = _resolve_registered_path(
            repository, model_cfg.get("locality_anchor"), "locality_anchor"
        )
        incumbent = _resolve_registered_path(
            repository, model_cfg.get("menagerie_incumbent"), "menagerie_incumbent"
        )
        if incumbent != expected_path:
            raise ValueError("locality anchor and Menagerie incumbent paths differ")
        expected_weight = model_cfg.get("anchor_weight_sha256")
        expected_tokenizer = model_cfg.get("anchor_tokenizer_manifest_sha256")
        expected_config = model_cfg.get("anchor_config_sha256")
        expected_generation_config = model_cfg.get(
            "anchor_generation_config_sha256"
        )
        expected_merge_receipt = model_cfg.get("anchor_merge_receipt_sha256")
    elif role in trained_roles:
        artifact_root = _resolve_registered_path(
            repository, artifacts_cfg.get("root"), "artifacts.root"
        )
        expected_path = (artifact_root / "merged" / role).resolve()
        expected_weight = None
        expected_tokenizer = model_cfg.get("start_tokenizer_manifest_sha256")
        expected_config = None
        expected_generation_config = None
        expected_merge_receipt = None
    else:
        raise ValueError(f"unknown registered checkpoint role: {role}")
    if model_path != expected_path:
        raise ValueError(
            f"checkpoint path is not registered for {role}: {model_path} != {expected_path}"
        )
    required = {
        "config": model_path / "config.json",
        "generation_config": model_path / "generation_config.json",
        "merge_receipt": model_path / "merge_receipt.json",
        "weights": model_path / "model.safetensors",
    }
    if any(not path.is_file() for path in required.values()):
        raise ValueError(f"registered {role} checkpoint is incomplete: {model_path}")
    try:
        model_config = json.loads(required["config"].read_text(encoding="utf-8"))
        merge = json.loads(required["merge_receipt"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {role} checkpoint metadata: {exc}") from exc
    text_config = model_config.get("text_config", {})
    if (
        model_config.get("model_type") != "qwen3_5"
        or not isinstance(text_config, dict)
        or text_config.get("model_type") != "qwen3_5_text"
        or text_config.get("vocab_size") != 248320
        or text_config.get("hidden_size") != 2560
        or text_config.get("num_hidden_layers") != 32
        or text_config.get("eos_token_id") != 248044
    ):
        raise ValueError(f"registered {role} model config fingerprint drifted")
    lineage = merge.get("model_lineage", merge.get("base_model"))
    revision = merge.get("model_revision", merge.get("base_revision"))
    if lineage != model_cfg.get("id") or revision != model_cfg.get("revision"):
        raise ValueError(f"registered {role} lineage/revision drifted")
    observed_weight = _sha256_file(required["weights"])
    observed_config = _sha256_file(required["config"])
    observed_generation_config = _sha256_file(required["generation_config"])
    observed_merge_receipt = _sha256_file(required["merge_receipt"])
    recorded_weights = {
        row.get("name"): row.get("sha256")
        for row in merge.get("weight_files", [])
        if isinstance(row, dict)
    }
    if recorded_weights.get("model.safetensors") != observed_weight:
        raise ValueError(f"registered {role} weight receipt drifted")
    if expected_weight is not None and observed_weight != expected_weight:
        raise ValueError(f"registered {role} weight hash drifted")
    if expected_config is not None and observed_config != expected_config:
        raise ValueError(f"registered {role} config hash drifted")
    if (
        expected_generation_config is not None
        and observed_generation_config != expected_generation_config
    ):
        raise ValueError(f"registered {role} generation-config hash drifted")
    if (
        expected_merge_receipt is not None
        and observed_merge_receipt != expected_merge_receipt
    ):
        raise ValueError(f"registered {role} merge-receipt hash drifted")
    tokenizer = validate_registered_tokenizer_provenance(
        model_path, merge, allow_absent=role in {"start", "anchor"}
    )
    if (
        tokenizer["tokenizer_manifest_sha256"] != expected_tokenizer
        or tokenizer["tokenizer_compatibility_sha256"]
        != model_cfg.get("tokenizer_compatibility_sha256")
    ):
        raise ValueError(f"registered {role} tokenizer drifted")
    if role in trained_roles:
        if (
            merge.get("arm") != role
            or merge.get("base_weight_sha256")
            != model_cfg.get("start_weight_sha256")
            or merge.get("base_config_sha256")
            != model_cfg.get("start_config_sha256")
            or merge.get("base_generation_config_sha256")
            != model_cfg.get("start_generation_config_sha256")
            or merge.get("base_merge_receipt_sha256")
            != model_cfg.get("start_merge_receipt_sha256")
            or merge.get("base_tokenizer_manifest_sha256")
            != model_cfg.get("start_tokenizer_manifest_sha256")
            or merge.get("base_tokenizer_compatibility_sha256")
            != model_cfg.get("tokenizer_compatibility_sha256")
            or merge.get("design_lock_sha256")
            != _sha256_file(Path(design_lock_path))
            or merge.get("output_config_sha256") != observed_config
            or merge.get("output_generation_config_sha256")
            != observed_generation_config
        ):
            raise ValueError(f"registered {role} merge lineage drifted")
    return {
        "role": role,
        "path": str(model_path),
        "model_weight_sha256": observed_weight,
        "model_config_sha256": observed_config,
        "generation_config_sha256": observed_generation_config,
        "merge_receipt_sha256": observed_merge_receipt,
        **tokenizer,
    }


def make_runner(
    engine_cfg: dict,
    *,
    adapter: str | None = None,
    model_override: str | None = None,
) -> VLLMRunner:
    return VLLMRunner(
        EngineConfig(
            max_model_len=int(engine_cfg.get("max_model_len", 16384)),
            gpu_memory_utilization=float(engine_cfg.get("gpu_memory_utilization", 0.85)),
            max_num_seqs=int(engine_cfg.get("max_num_seqs", 32)),
            max_num_batched_tokens=int(engine_cfg.get("max_num_batched_tokens", 16384)),
            adapter=Path(adapter) if adapter else None,
            model_override=Path(model_override) if model_override else None,
        )
    )
