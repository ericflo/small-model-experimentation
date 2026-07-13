"""Fresh deterministic procedural data for the capacity adjudication."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import (
    MODEL_ID, MODEL_REVISION, SOURCE_CONTRACT_VERSION, config_sha256,
    source_contract_sha256,
)
from .design_boundary import design_lineage, validate_design_receipt
from .substrate import generate_example, verify_example


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
SPLITS = {
    "train", "validation", "depth_extrapolation", "joint_holdout",
    "contrast_validation", "contrast_depth", "contrast_joint",
}
SEALED_SPLITS = {"contrast_validation", "contrast_depth", "contrast_joint"}
ACCESS_LEDGER_NAME = "contrast_access_ledger.json"
REQUIREMENTS_LOCK = REPO_ROOT / "requirements-training.lock.txt"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError as exc:
        raise RuntimeError(f"path is outside repository workspace: {resolved}") from exc


def data_contract_sha256(config: Mapping[str, Any], *, source_digest: str | None = None) -> str:
    payload = {
        "experiment_id": config["experiment_id"],
        "substrate": config["substrate"],
        "state_token": config["architecture"]["state_token"],
        "state_slots": config["architecture"]["state_slots"],
        "source_contract_version": SOURCE_CONTRACT_VERSION,
        "source_contract_sha256": source_digest or source_contract_sha256(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _write_jsonl_gz(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as handle:
                for row in rows:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def canonical_rows_receipt(path: str | Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    rows = 0
    for row in read_jsonl(path):
        digest.update(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        )
        rows += 1
    return {"rows": rows, "canonical_rows_sha256": digest.hexdigest()}


def _generate_rows(
    *,
    count: int,
    seed: int,
    split: str,
    families: Sequence[str],
    templates: Sequence[str],
    depths: Sequence[int],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    substrate = config["substrate"]
    architecture = config["architecture"]
    rows = []
    for index in range(count):
        query_kind = ("node", "checksum")[index % 2]
        cell_index = index // 2
        rows.append(
            generate_example(
                seed=seed * 10_000_000 + index,
                split=split,
                family=families[cell_index % len(families)],
                template=templates[(cell_index // len(families)) % len(templates)],
                depth=int(
                    depths[
                        (cell_index // (len(families) * len(templates))) % len(depths)
                    ]
                ),
                node_count=int(substrate["node_count"]),
                checksum_modulus=int(substrate["checksum_modulus"]),
                num_choices=int(substrate["num_choices"]),
                state_token=str(architecture["state_token"]),
                state_slots=int(architecture["state_slots"]),
                max_attempts=int(substrate["max_generation_attempts"]),
                query_kind=query_kind,
            )
        )
    return rows


def _query_grid(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    cells: dict[str, Counter[str]] = {}
    for row in rows:
        key = f"{row['family']}|{row['template']}|depth={int(row['depth'])}"
        cells.setdefault(key, Counter())[str(row["query_kind"])] += 1
    return {key: dict(sorted(value.items())) for key, value in sorted(cells.items())}


def _expected_metadata(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    substrate = config["substrate"]
    train_families = list(substrate["train_families"])
    train_templates = list(substrate["train_templates"])
    train_depths = list(map(int, substrate["train_depths"]))
    deep = list(map(int, substrate["extrapolation_depths"]))
    specs = {
        "train": (int(substrate["train_examples"]), train_families, train_templates, train_depths),
        "validation": (
            int(substrate["validation_examples"]), train_families, train_templates, train_depths,
        ),
        "depth_extrapolation": (
            int(substrate["depth_examples"]), train_families, train_templates, deep,
        ),
        "joint_holdout": (
            int(substrate["joint_examples"]), [substrate["heldout_family"]],
            [substrate["heldout_template"]], deep,
        ),
        "contrast_validation": (
            int(substrate["contrast_validation_examples"]), train_families,
            train_templates, [2, 3, 4],
        ),
        "contrast_depth": (
            int(substrate["contrast_depth_examples"]), train_families, train_templates, deep,
        ),
        "contrast_joint": (
            int(substrate["contrast_joint_examples"]), [substrate["heldout_family"]],
            [substrate["heldout_template"]], deep,
        ),
    }
    expected = {}
    for split, (count, families, templates, depths) in specs.items():
        family_counts: Counter[str] = Counter()
        template_counts: Counter[str] = Counter()
        depth_counts: Counter[str] = Counter()
        query_counts: Counter[str] = Counter()
        grid: dict[str, Counter[str]] = {}
        for index in range(count):
            query = ("node", "checksum")[index % 2]
            cell_index = index // 2
            family = str(families[cell_index % len(families)])
            template = str(templates[(cell_index // len(families)) % len(templates)])
            depth = int(
                depths[(cell_index // (len(families) * len(templates))) % len(depths)]
            )
            family_counts[family] += 1
            template_counts[template] += 1
            depth_counts[str(depth)] += 1
            query_counts[query] += 1
            key = f"{family}|{template}|depth={depth}"
            grid.setdefault(key, Counter())[query] += 1
        expected[split] = {
            "path": f"{split}.jsonl.gz",
            "rows": count,
            "families": dict(family_counts),
            "templates": dict(template_counts),
            "depths": dict(sorted(depth_counts.items())),
            "query_kinds": dict(sorted(query_counts.items())),
            "query_kind_grid": {
                key: dict(sorted(value.items())) for key, value in sorted(grid.items())
            },
        }
    return expected


def validate_data_manifest(
    config: Mapping[str, Any], output_dir: Path, manifest: Mapping[str, Any],
    *, content_splits: set[str] | frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Validate bytes for every split and decompress only explicitly licensed splits.

    In particular, callers before the Stage-B authorization must never put a
    sealed contrast split in ``content_splits``.  A compressed SHA check does
    not expose row contents or labels.
    """

    validate_design_receipt(config)
    content_splits = set(content_splits)
    if not content_splits <= SPLITS:
        raise RuntimeError(f"unknown content-validation splits: {sorted(content_splits - SPLITS)}")
    source_digest = source_contract_sha256()
    expected_header = {
        "experiment_id": config["experiment_id"],
        "data_contract_sha256": data_contract_sha256(config, source_digest=source_digest),
        "source_contract_sha256": source_digest,
        "config_sha256": config_sha256(config),
        "requirements_training_lock_sha256": _sha256(REQUIREMENTS_LOCK),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "design_lineage": design_lineage(config),
        "cross_split_structural_duplicates": 0,
        "benchmark_files_read": 0,
    }
    for key, value in expected_header.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"prepared data manifest {key} mismatch")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != SPLITS:
        raise RuntimeError("prepared data split set is incomplete")
    expected_metadata = _expected_metadata(config)
    fingerprint_owner: dict[str, str] = {}
    for split, metadata in files.items():
        for key, value in expected_metadata[split].items():
            if metadata.get(key) != value:
                raise RuntimeError(f"prepared data {split}/{key} geometry mismatch")
        path = output_dir / str(metadata["path"])
        if not path.is_file() or _sha256(path) != metadata.get("sha256"):
            raise RuntimeError(f"prepared data payload changed: {split}")
        if path.stat().st_size != int(metadata.get("bytes", -1)):
            raise RuntimeError(f"prepared data byte count changed: {split}")
        with path.open("rb") as handle:
            header = handle.read(10)
        if len(header) != 10 or header[:3] != b"\x1f\x8b\x08" or header[4:8] != b"\x00\x00\x00\x00":
            raise RuntimeError(f"prepared data is not a deterministic gzip payload: {split}")
        fingerprints = metadata.get("structural_fingerprints")
        if (
            not isinstance(fingerprints, list)
            or len(fingerprints) != int(metadata.get("rows", -1))
            or len(set(fingerprints)) != len(fingerprints)
            or any(not isinstance(item, str) or len(item) != 64 for item in fingerprints)
        ):
            raise RuntimeError(f"prepared structural-fingerprint index is malformed: {split}")
        if fingerprints != sorted(fingerprints):
            raise RuntimeError(f"prepared structural-fingerprint index is not canonical: {split}")
        for fingerprint in fingerprints:
            if fingerprint in fingerprint_owner:
                raise RuntimeError(
                    f"structural fingerprint crosses {fingerprint_owner[fingerprint]} and {split}"
                )
            fingerprint_owner[fingerprint] = split
        canonical = metadata.get("canonical_rows")
        if (
            not isinstance(canonical, Mapping)
            or canonical.get("rows") != metadata["rows"]
            or not isinstance(canonical.get("canonical_rows_sha256"), str)
            or len(canonical["canonical_rows_sha256"]) != 64
        ):
            raise RuntimeError(f"prepared canonical-row receipt is malformed: {split}")
        if split in content_splits and canonical_rows_receipt(path) != metadata.get("canonical_rows"):
            raise RuntimeError(f"prepared canonical rows changed: {split}")
    return dict(manifest)


def _ledger_identity(payload: Mapping[str, Any]) -> str:
    return _canonical_sha256(
        {key: value for key, value in payload.items() if key != "receipt_identity_sha256"}
    )


def _valid_archived_failed_contrast_attempts(
    config: Mapping[str, Any], evaluation_output: Path, *, require_empty_output: bool
) -> list[dict[str, Any]]:
    """Return content-validated archives for earlier same-cell access attempts."""

    output_relative = _repo_relative(evaluation_output)
    if require_empty_output and (
        not evaluation_output.is_dir() or any(evaluation_output.iterdir())
    ):
        raise RuntimeError("contrast replay requires a newly created empty canonical output")
    label = evaluation_output.name
    failures_dir = ROOT / "runs" / "failures"
    candidates = sorted(failures_dir.glob(f"{label}-*.json")) if failures_dir.is_dir() else []
    valid: list[dict[str, Any]] = []
    expected_design = design_lineage(config)
    large_root = (ROOT / str(config["paths"]["large_artifacts_dir"])).resolve()
    required_keys = {
        "schema_version", "status", "experiment_id", "model_id", "model_revision",
        "backend", "config_sha256", "source_contract_sha256",
        "requirements_training_lock_sha256", "design_lineage",
        "attempt_identity_sha256", "archive_path", "attempts",
        "scientific_evidence", "receipt_identity_sha256",
    }
    for receipt_path in candidates:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if set(receipt) != required_keys:
            raise RuntimeError("failed-attempt archive receipt fields changed")
        identity_payload = {
            key: value for key, value in receipt.items()
            if key != "receipt_identity_sha256"
        }
        if receipt["receipt_identity_sha256"] != _canonical_sha256(identity_payload):
            raise RuntimeError("failed-attempt archive receipt identity mismatch")
        expected_header = {
            "schema_version": 1,
            "status": "FAILED_ATTEMPT_ARCHIVED",
            "experiment_id": config["experiment_id"],
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "backend": "transformers",
            "config_sha256": config_sha256(config),
            "source_contract_sha256": source_contract_sha256(),
            "requirements_training_lock_sha256": _sha256(REQUIREMENTS_LOCK),
            "design_lineage": expected_design,
            "scientific_evidence": False,
        }
        if any(receipt.get(key) != value for key, value in expected_header.items()):
            raise RuntimeError("failed-attempt archive receipt lineage mismatch")
        attempts = receipt.get("attempts")
        if not isinstance(attempts, list) or len(attempts) != 1:
            continue
        attempt = attempts[0]
        if not isinstance(attempt, Mapping) or attempt.get("source_path") != output_relative:
            continue
        files = attempt.get("files")
        if (
            not isinstance(files, list)
            or attempt.get("files_sha256") != _canonical_sha256(files)
            or attempt.get("tree_identity_sha256") != _canonical_sha256(
                {
                    "source_path": output_relative,
                    "files": files,
                    "files_sha256": attempt.get("files_sha256"),
                }
            )
        ):
            raise RuntimeError("failed-attempt archived tree identity mismatch")
        attempt_identity = _canonical_sha256({"attempts": attempts})
        if receipt.get("attempt_identity_sha256") != attempt_identity:
            raise RuntimeError("failed-attempt archive set identity mismatch")
        expected_archive = (
            large_root / "failed_attempts" / f"{label}-{attempt_identity[:16]}"
        ).resolve()
        archive_path = (REPO_ROOT / str(receipt["archive_path"])).resolve()
        if (
            archive_path != expected_archive
            or receipt_path.name != f"{label}-{attempt_identity[:16]}.json"
            or not archive_path.is_dir()
        ):
            raise RuntimeError("failed-attempt archive uses a noncanonical path")
        archive_receipt = archive_path / "archive_receipt.json"
        archived_source = archive_path / f"source_1_{label}"
        if (
            not archive_receipt.is_file()
            or archive_receipt.read_bytes() != receipt_path.read_bytes()
            or not archived_source.is_dir()
            or (archived_source / "summary.json").exists()
        ):
            raise RuntimeError("failed-attempt archive is incomplete or contains a completed result")
        observed_files = []
        for item in sorted(archived_source.rglob("*")):
            if item.is_symlink():
                raise RuntimeError("failed-attempt archive contains a symlink")
            if item.is_file():
                observed_files.append(
                    {
                        "path": item.relative_to(archived_source).as_posix(),
                        "bytes": item.stat().st_size,
                        "sha256": _sha256(item),
                    }
                )
        if observed_files != files:
            raise RuntimeError("failed-attempt archived bytes changed")
        valid.append(
            {
                "path": _repo_relative(receipt_path),
                "sha256": _sha256(receipt_path),
                "receipt_identity_sha256": receipt["receipt_identity_sha256"],
                "attempt_identity_sha256": attempt_identity,
                "archive_path": receipt["archive_path"],
            }
        )
    return valid


def load_contrast_access_ledger(
    config: Mapping[str, Any], output_dir: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    path = output_dir / ACCESS_LEDGER_NAME
    if not path.is_file():
        raise RuntimeError(f"contrast access ledger is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("receipt_identity_sha256") != _ledger_identity(payload):
        raise RuntimeError("contrast access ledger identity mismatch")
    manifest_path = output_dir / "manifest.json"
    expected = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "data_manifest_sha256": _sha256(manifest_path),
        "design_lineage": design_lineage(config),
        "sealed_splits": {
            split: {
                "path": manifest["files"][split]["path"],
                "bytes": manifest["files"][split]["bytes"],
                "sha256": manifest["files"][split]["sha256"],
            }
            for split in sorted(SEALED_SPLITS)
        },
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"contrast access ledger {key} mismatch")
    if not isinstance(payload.get("events"), list):
        raise RuntimeError("contrast access ledger events are malformed")
    event_keys = {
        "event_index", "opened_at_utc", "authorization", "capacity", "objective",
        "model_seed", "evaluation_output", "checkpoint_lineage", "splits",
        "replay_archives", "event_identity_sha256",
    }
    for index, event in enumerate(payload["events"], start=1):
        if not isinstance(event, Mapping) or set(event) != event_keys:
            raise RuntimeError("contrast access event fields are malformed")
        if event.get("event_index") != index or event.get("splits") != sorted(SEALED_SPLITS):
            raise RuntimeError("contrast access event order or split set changed")
        event_payload = {
            key: value for key, value in event.items()
            if key != "event_identity_sha256"
        }
        if event.get("event_identity_sha256") != _canonical_sha256(event_payload):
            raise RuntimeError("contrast access event identity mismatch")
        recorded = event.get("replay_archives")
        if not isinstance(recorded, list) or len(recorded) != len(
            {_canonical_sha256(item) for item in recorded if isinstance(item, Mapping)}
        ):
            raise RuntimeError("contrast replay archive history is malformed")
        if recorded:
            output = (REPO_ROOT / str(event["evaluation_output"])).resolve()
            if not output.is_relative_to(REPO_ROOT):
                raise RuntimeError("contrast evaluation output escapes repository")
            valid = _valid_archived_failed_contrast_attempts(
                config, output, require_empty_output=False
            )
            if any(item not in valid for item in recorded):
                raise RuntimeError("contrast replay archive lineage changed")
    return payload


def record_contrast_access(
    config: Mapping[str, Any], output_dir: Path, manifest: Mapping[str, Any],
    *, authorization: Mapping[str, Any], capacity: str, objective: str,
    model_seed: int, evaluation_output: Path,
    checkpoint_lineage: Mapping[str, Any],
) -> dict[str, Any]:
    """Append an authorization-bound event before any sealed gzip is opened."""

    import fcntl

    path = output_dir / ACCESS_LEDGER_NAME
    required_authorization = {
        "path", "sha256", "receipt_identity_sha256", "status", "phase"
    }
    if set(authorization) != required_authorization:
        raise RuntimeError("contrast authorization lineage has the wrong fields")
    if (
        authorization.get("status") != "STAGE_B_CONTRAST_AUTHORIZED"
        or authorization.get("phase") != "stage_b_seal_analysis"
    ):
        raise RuntimeError("contrast authorization has the wrong status or phase")
    authorization_path = (REPO_ROOT / str(authorization["path"])).resolve()
    if not authorization_path.is_relative_to(REPO_ROOT):
        raise RuntimeError("contrast authorization path escapes repository")
    expected_authorization_path = (ROOT / "analysis" / "stage_b_seal.json").resolve()
    if authorization_path != expected_authorization_path:
        raise RuntimeError("contrast authorization must be the canonical Stage-B seal")
    if not authorization_path.is_file() or _sha256(authorization_path) != authorization["sha256"]:
        raise RuntimeError("contrast authorization file changed")
    authorization_receipt = json.loads(authorization_path.read_text(encoding="utf-8"))
    authorization_identity_payload = {
        key: value
        for key, value in authorization_receipt.items()
        if key != "receipt_identity_sha256"
    }
    if (
        authorization_receipt.get("receipt_identity_sha256")
        != _canonical_sha256(authorization_identity_payload)
    ):
        raise RuntimeError("contrast authorization identity self-check changed")
    if (
        authorization_receipt.get("receipt_identity_sha256")
        != authorization["receipt_identity_sha256"]
    ):
        raise RuntimeError("contrast authorization identity changed")
    if capacity not in {"lora", "fullrank"} or objective != "joint":
        raise RuntimeError("only registered joint capacity cells may open contrast data")
    if int(model_seed) not in set(map(int, config["training"]["train_seeds"])):
        raise RuntimeError("contrast access model seed is not registered")
    if set(checkpoint_lineage) != {
        "path", "metadata_sha256", "checkpoint_identity_sha256"
    }:
        raise RuntimeError("contrast checkpoint lineage has the wrong fields")
    checkpoint_path = (REPO_ROOT / str(checkpoint_lineage["path"])).resolve()
    if not checkpoint_path.is_relative_to(REPO_ROOT):
        raise RuntimeError("contrast checkpoint path escapes repository")
    checkpoint_metadata_path = checkpoint_path / "checkpoint.json"
    if (
        not checkpoint_metadata_path.is_file()
        or _sha256(checkpoint_metadata_path) != checkpoint_lineage["metadata_sha256"]
    ):
        raise RuntimeError("contrast checkpoint metadata changed")
    checkpoint_metadata = json.loads(checkpoint_metadata_path.read_text(encoding="utf-8"))
    if (
        checkpoint_metadata.get("checkpoint_identity_sha256")
        != checkpoint_lineage["checkpoint_identity_sha256"]
    ):
        raise RuntimeError("contrast checkpoint identity changed")
    current_design = design_lineage(config)
    expected_checkpoint_cell = {
        "experiment_id": config["experiment_id"],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "backend": "transformers",
        "capacity": capacity,
        "objective": objective,
        "model_seed": int(model_seed),
        "step": int(config["training"]["train_steps"]),
        "phase": f"{capacity}_{objective}_training",
        "config_sha256": config_sha256(config),
        "source_contract_sha256": source_contract_sha256(),
        "requirements_training_lock_sha256": _sha256(REQUIREMENTS_LOCK),
        "data_manifest_sha256": _sha256(output_dir / "manifest.json"),
        "design_receipt_sha256": current_design["sha256"],
        "design_receipt_identity_sha256": current_design[
            "receipt_identity_sha256"
        ],
    }
    if any(
        checkpoint_metadata.get(key) != value
        for key, value in expected_checkpoint_cell.items()
    ):
        raise RuntimeError("contrast checkpoint does not match the requested cell")
    identity_payload = {
        key: value
        for key, value in checkpoint_metadata.items()
        if key != "checkpoint_identity_sha256"
    }
    if _canonical_sha256(identity_payload) != checkpoint_lineage["checkpoint_identity_sha256"]:
        raise RuntimeError("contrast checkpoint self-identity changed")
    matching = authorization_receipt.get("matching", {}).get("per_seed", {})
    authorized_lineage = (
        matching.get(str(model_seed), {})
        .get("checkpoint_lineages", {})
        .get(f"{capacity}_joint")
    )
    if authorized_lineage != dict(checkpoint_lineage):
        raise RuntimeError("contrast checkpoint is not the exact Stage-B-authorized cell")
    cell = (capacity, objective, int(model_seed))
    expected_evaluation_output = (
        ROOT / "runs" / f"{capacity}_joint_seed{int(model_seed)}_contrast"
    ).resolve()
    if evaluation_output.resolve() != expected_evaluation_output:
        raise RuntimeError(
            "contrast access requires the exact canonical evaluation output path"
        )
    output_relative = _repo_relative(evaluation_output)
    lock_path = path.with_name(f"{path.name}.lock")
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if not path.is_file():
            raise RuntimeError("contrast access ledger disappeared while acquiring its lock")
        payload = json.loads(path.read_text(encoding="utf-8"))
        # The separate lock inode remains stable across atomic ledger replaces.
        if payload.get("receipt_identity_sha256") != _ledger_identity(payload):
            raise RuntimeError("contrast access ledger identity mismatch")
        locked_expected = {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "data_manifest_sha256": _sha256(output_dir / "manifest.json"),
            "design_lineage": design_lineage(config),
            "sealed_splits": {
                split: {
                    "path": manifest["files"][split]["path"],
                    "bytes": manifest["files"][split]["bytes"],
                    "sha256": manifest["files"][split]["sha256"],
                }
                for split in sorted(SEALED_SPLITS)
            },
        }
        for key, value in locked_expected.items():
            if payload.get(key) != value:
                raise RuntimeError(f"contrast access ledger {key} mismatch")
        if not isinstance(payload.get("events"), list):
            raise RuntimeError("contrast access ledger events are malformed")
        for prior in payload["events"]:
            if prior.get("authorization") != dict(authorization):
                raise RuntimeError("contrast access ledger mixes authorization receipts")
            prior_cell = (
                prior.get("capacity"), prior.get("objective"), prior.get("model_seed")
            )
            if prior_cell == cell:
                if (
                    prior.get("evaluation_output") != output_relative
                    or prior.get("checkpoint_lineage") != dict(checkpoint_lineage)
                ):
                    raise RuntimeError("contrast cell already opened under a different output path")
                recorded_archives = prior.get("replay_archives")
                if not isinstance(recorded_archives, list):
                    raise RuntimeError("contrast access event replay archive history is malformed")
                valid_archives = _valid_archived_failed_contrast_attempts(
                    config, evaluation_output, require_empty_output=True
                )
                if any(item not in valid_archives for item in recorded_archives):
                    raise RuntimeError("a previously bound failed-attempt archive changed")
                new_archives = [
                    item for item in valid_archives if item not in recorded_archives
                ]
                if len(new_archives) != 1:
                    raise RuntimeError(
                        "same-cell contrast replay requires exactly one newly preserved "
                        "failed-attempt archive"
                    )
                prior["replay_archives"].append(new_archives[0])
                prior["event_identity_sha256"] = _canonical_sha256(
                    {
                        key: value for key, value in prior.items()
                        if key != "event_identity_sha256"
                    }
                )
                payload["receipt_identity_sha256"] = _ledger_identity(payload)
                _atomic_write_json(path, payload)
                return dict(prior)
        if _valid_archived_failed_contrast_attempts(
            config, evaluation_output, require_empty_output=True
        ):
            raise RuntimeError(
                "a failed-attempt archive predates the first contrast access event"
            )
        event = {
            "event_index": len(payload["events"]) + 1,
            "opened_at_utc": datetime.now(timezone.utc).isoformat(),
            "authorization": dict(authorization),
            "capacity": capacity,
            "objective": objective,
            "model_seed": int(model_seed),
            "evaluation_output": output_relative,
            "checkpoint_lineage": dict(checkpoint_lineage),
            "splits": sorted(SEALED_SPLITS),
            "replay_archives": [],
        }
        event["event_identity_sha256"] = _canonical_sha256(event)
        payload["events"].append(event)
        payload["receipt_identity_sha256"] = _ledger_identity(payload)
        _atomic_write_json(path, payload)
    return event


def build_datasets(config: Mapping[str, Any], output_dir: str | Path) -> dict[str, Any]:
    validate_design_receipt(config)
    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.json"
    ledger_path = output_dir / ACCESS_LEDGER_NAME
    if manifest_path.exists() or ledger_path.exists() or any(output_dir.glob("*.jsonl.gz")):
        raise RuntimeError(f"refusing to overwrite prepared data: {output_dir}")
    source_digest = source_contract_sha256()
    substrate = config["substrate"]
    architecture = config["architecture"]
    seeds = substrate["seeds"]
    train_families = list(substrate["train_families"])
    train_templates = list(substrate["train_templates"])
    train_depths = list(map(int, substrate["train_depths"]))
    extrapolation = list(map(int, substrate["extrapolation_depths"]))
    specs = {
        "train": (
            int(substrate["train_examples"]), int(seeds["train"]), train_families,
            train_templates, train_depths,
        ),
        "validation": (
            int(substrate["validation_examples"]), int(seeds["validation"]), train_families,
            train_templates, train_depths,
        ),
        "depth_extrapolation": (
            int(substrate["depth_examples"]), int(seeds["depth"]), train_families,
            train_templates, extrapolation,
        ),
        "joint_holdout": (
            int(substrate["joint_examples"]), int(seeds["joint"]),
            [substrate["heldout_family"]], [substrate["heldout_template"]], extrapolation,
        ),
        "contrast_validation": (
            int(substrate["contrast_validation_examples"]),
            int(seeds["contrast_validation"]), train_families, train_templates, [2, 3, 4],
        ),
        "contrast_depth": (
            int(substrate["contrast_depth_examples"]), int(seeds["contrast_depth"]),
            train_families, train_templates, extrapolation,
        ),
        "contrast_joint": (
            int(substrate["contrast_joint_examples"]), int(seeds["contrast_joint"]),
            [substrate["heldout_family"]], [substrate["heldout_template"]], extrapolation,
        ),
    }
    all_seen: dict[str, str] = {}
    files: dict[str, Any] = {}
    for split, (count, seed, families, templates, depths) in specs.items():
        rows = _generate_rows(
            count=count, seed=seed, split=split, families=families, templates=templates,
            depths=depths, config=config,
        )
        for row in rows:
            verify_example(row, str(architecture["state_token"]), int(architecture["state_slots"]))
            fingerprint = str(row["structural_fingerprint"])
            if fingerprint in all_seen:
                raise RuntimeError(f"structural duplicate crosses {all_seen[fingerprint]} and {split}")
            all_seen[fingerprint] = split
        grid = _query_grid(rows)
        if any(cell.get("node", 0) != cell.get("checksum", 0) for cell in grid.values()):
            raise RuntimeError(f"query-kind grid is imbalanced in {split}")
        path = output_dir / f"{split}.jsonl.gz"
        _write_jsonl_gz(path, rows)
        files[split] = {
            "path": path.name,
            "rows": len(rows),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "canonical_rows": canonical_rows_receipt(path),
            "structural_fingerprints": sorted(str(row["structural_fingerprint"]) for row in rows),
            "families": dict(Counter(row["family"] for row in rows)),
            "templates": dict(Counter(row["template"] for row in rows)),
            "depths": dict(sorted(Counter(str(row["depth"]) for row in rows).items())),
            "query_kinds": dict(sorted(Counter(row["query_kind"] for row in rows).items())),
            "query_kind_grid": grid,
        }
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "data_contract_sha256": data_contract_sha256(config, source_digest=source_digest),
        "source_contract_version": SOURCE_CONTRACT_VERSION,
        "source_contract_sha256": source_digest,
        "config_sha256": config_sha256(config),
        "requirements_training_lock_sha256": _sha256(REQUIREMENTS_LOCK),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "design_lineage": design_lineage(config),
        "generator": "src.data_pipeline.build_datasets",
        "files": files,
        "cross_split_structural_duplicates": 0,
        "benchmark_files_read": 0,
        "state_token": architecture["state_token"],
        "state_slots": architecture["state_slots"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(manifest_path, manifest)
    ledger = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "data_manifest_sha256": _sha256(manifest_path),
        "design_lineage": design_lineage(config),
        "sealed_splits": {
            split: {
                "path": files[split]["path"],
                "bytes": files[split]["bytes"],
                "sha256": files[split]["sha256"],
            }
            for split in sorted(SEALED_SPLITS)
        },
        "events": [],
    }
    ledger["receipt_identity_sha256"] = _ledger_identity(ledger)
    _atomic_write_json(ledger_path, ledger)
    return manifest
