"""Fresh deterministic procedural data for the capacity adjudication."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import stat
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .attempt_receipts import (
    ATTEMPT_MARKER_NAME,
    AttemptReceiptError,
    archive_lineage,
    atomic_write_json as durable_atomic_write_json,
    build_attempt_authorization,
    canonical_path as canonical_attempt_path,
    find_exact_failed_archive,
    locked_regular,
    read_json as read_attempt_json,
    repo_relative as attempt_repo_relative,
    validate_attempt_authorization,
    validate_attempt_marker,
    validate_failed_archive,
)
from .config import (
    MODEL_ID, MODEL_REVISION, SOURCE_CONTRACT_VERSION, config_sha256,
    requirements_training_lock_bytes,
    source_contract_sha256,
)
from .design_boundary import design_lineage, validate_design_receipt
from .safe_io import (
    open_stable_regular,
    publish_new_bytes,
    publish_new_file,
    read_stable_bytes,
    read_verified_jsonl_gzip,
)
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


def _registered_contrast_cells(config: Mapping[str, Any]) -> tuple[tuple[str, str, int], ...]:
    seeds = tuple(map(int, config["training"]["train_seeds"]))
    return tuple(
        (capacity, "joint", seed)
        for capacity in ("lora", "fullrank")
        for seed in seeds
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    lexical = Path(os.path.abspath(os.fspath(path)))
    trusted_root = REPO_ROOT if lexical.is_relative_to(REPO_ROOT) else lexical.parent
    with open_stable_regular(trusted_root, lexical) as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _requirements_sha256() -> str:
    return hashlib.sha256(requirements_training_lock_bytes()).hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _stable_json_snapshot(
    trusted_root: Path, path: Path, label: str
) -> tuple[dict[str, Any], str]:
    try:
        with open_stable_regular(trusted_root, path) as handle:
            raw = handle.read()
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
        raise RuntimeError(f"{label} is not one stable strict JSON object") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} is not a JSON object")
    return payload, hashlib.sha256(raw).hexdigest()


def _repo_relative(path: Path) -> str:
    try:
        return attempt_repo_relative(REPO_ROOT, path)
    except AttemptReceiptError as exc:
        raise RuntimeError(f"path is not canonical in repository workspace: {path}") from exc


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


def _write_jsonl_gz(
    path: Path, rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    path = Path(os.path.abspath(os.fspath(path)))
    path.parent.mkdir(parents=True, exist_ok=True)
    def write_compressed(handle: Any) -> None:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=handle, mtime=0
        ) as compressed:
            for row in rows:
                compressed.write(
                    (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode(
                        "utf-8"
                    )
                )

    digest = publish_new_file(
        path.parent,
        path,
        write_compressed,
        mode=0o644,
    )
    encoded = read_stable_bytes(path.parent, path)
    if hashlib.sha256(encoded).hexdigest() != digest:
        raise RuntimeError("published prepared-data payload changed after commit")
    return {
        "bytes": len(encoded),
        "sha256": digest,
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(os.path.abspath(os.fspath(path)))
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    publish_new_bytes(path.parent, path, encoded, mode=0o644)


def _data_crash_point(name: str) -> None:
    if os.environ.get("QWEN35_DATA_CRASH_AT") == name:
        raise RuntimeError(f"injected prepared-data crash at {name}")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _rows_from_compressed_snapshot(raw: bytes, split: str) -> list[dict[str, Any]]:
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(raw), mode="rb") as compressed:
            text = compressed.read().decode("utf-8")
        rows = [
            json.loads(
                line,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_json_constant,
            )
            for line in text.splitlines()
            if line.strip()
        ]
    except (OSError, EOFError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"prepared data content is invalid: {split}") from exc
    if any(not isinstance(row, dict) for row in rows):
        raise RuntimeError(f"prepared data rows are not objects: {split}")
    return rows


def _canonical_rows_from_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
            + b"\n"
        )
    return {"rows": len(rows), "canonical_rows_sha256": digest.hexdigest()}


def _compressed_payload_snapshot(
    output_dir: Path,
    path: Path,
    *,
    split: str,
    expected_sha256: Any,
    expected_bytes: Any,
) -> bytes:
    lexical = Path(os.path.abspath(os.fspath(path)))
    lexical_output = Path(os.path.abspath(os.fspath(output_dir)))
    trusted_root = REPO_ROOT if lexical.is_relative_to(REPO_ROOT) else lexical_output
    try:
        with open_stable_regular(trusted_root, lexical) as handle:
            raw = handle.read()
    except RuntimeError as exc:
        raise RuntimeError(f"prepared data payload changed: {split}") from exc
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256 or len(raw) != expected_bytes:
        raise RuntimeError(f"prepared data payload changed: {split}")
    header = raw[:10]
    if (
        len(header) != 10
        or header[:3] != b"\x1f\x8b\x08"
        or header[4:8] != b"\x00\x00\x00\x00"
    ):
        raise RuntimeError(f"prepared data is not a deterministic gzip payload: {split}")
    return raw


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


def validated_data_rows(
    config: Mapping[str, Any],
    data_dir: Path,
    manifest: Mapping[str, Any],
    splits: Sequence[str],
) -> dict[str, list[dict[str, Any]]]:
    """Return exact same-snapshot rows bound by manifest compressed hashes."""

    validate_data_manifest(config, data_dir, manifest, content_splits=set())
    result: dict[str, list[dict[str, Any]]] = {}
    for split in splits:
        if split not in SPLITS:
            raise RuntimeError(f"unknown prepared data split: {split}")
        metadata = manifest["files"][split]
        path = data_dir / f"{split}.jsonl.gz"
        rows = read_verified_jsonl_gzip(
            data_dir, path, str(metadata["sha256"])
        )
        digest = hashlib.sha256()
        for row in rows:
            digest.update(
                json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
                + b"\n"
            )
        observed = {
            "rows": len(rows),
            "canonical_rows_sha256": digest.hexdigest(),
        }
        if observed != metadata.get("canonical_rows"):
            raise RuntimeError(f"prepared canonical rows changed: {split}")
        for row in rows:
            verify_example(
                row,
                str(config["architecture"]["state_token"]),
                int(config["architecture"]["state_slots"]),
            )
        result[split] = rows
    return result


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
        "requirements_training_lock_sha256": _requirements_sha256(),
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
        raw = _compressed_payload_snapshot(
            output_dir,
            path,
            split=split,
            expected_sha256=metadata.get("sha256"),
            expected_bytes=metadata.get("bytes"),
        )
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
        if split in content_splits:
            rows = _rows_from_compressed_snapshot(raw, split)
            if _canonical_rows_from_rows(rows) != metadata.get("canonical_rows"):
                raise RuntimeError(f"prepared canonical rows changed: {split}")
            for row in rows:
                verify_example(
                    row,
                    str(config["architecture"]["state_token"]),
                    int(config["architecture"]["state_slots"]),
                )
    return dict(manifest)


def _ledger_identity(payload: Mapping[str, Any]) -> str:
    return _canonical_sha256(
        {key: value for key, value in payload.items() if key != "receipt_identity_sha256"}
    )


def _failed_archive_header(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "FAILED_ATTEMPT_ARCHIVED",
        "experiment_id": config["experiment_id"],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "backend": "transformers",
        "config_sha256": config_sha256(config),
        "source_contract_sha256": source_contract_sha256(),
        "requirements_training_lock_sha256": _requirements_sha256(),
        "design_lineage": design_lineage(config),
        "scientific_evidence": False,
    }


def _require_no_symlink_components(root: Path, path: Path, label: str) -> Path:
    lexical_root = Path(os.path.abspath(os.fspath(root)))
    lexical_path = Path(os.path.abspath(os.fspath(path)))
    if not lexical_path.is_relative_to(lexical_root):
        raise RuntimeError(f"{label} escapes its canonical root")
    current = lexical_root
    if current.is_symlink():
        raise RuntimeError(f"{label} uses a symlinked root")
    for part in lexical_path.relative_to(lexical_root).parts:
        current /= part
        if os.path.lexists(current) and stat.S_ISLNK(current.lstat().st_mode):
            raise RuntimeError(f"{label} uses a symlinked path component")
    return lexical_path


def _archived_tree_receipt(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reopen a preserved attempt without following or ignoring node types."""

    files: list[dict[str, Any]] = []
    directories: list[dict[str, Any]] = []

    def walk(directory: Path, relative: Path) -> None:
        try:
            root_info = directory.lstat()
        except OSError as exc:
            raise RuntimeError(f"failed-attempt archive directory is unreadable: {directory}") from exc
        if not stat.S_ISDIR(root_info.st_mode) or stat.S_ISLNK(root_info.st_mode):
            raise RuntimeError("failed-attempt archive directory is not canonical")
        with os.scandir(directory) as scanner:
            entries = sorted(scanner, key=lambda entry: entry.name)
        direct: list[dict[str, str]] = []
        children: list[tuple[Path, Path]] = []
        for entry in entries:
            path = directory / entry.name
            info = entry.stat(follow_symlinks=False)
            child_relative = relative / entry.name
            if stat.S_ISDIR(info.st_mode):
                direct.append({"name": entry.name, "type": "directory"})
                children.append((path, child_relative))
            elif stat.S_ISREG(info.st_mode):
                direct.append({"name": entry.name, "type": "regular_file"})
                flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
                if hasattr(os, "O_NOFOLLOW"):
                    flags |= os.O_NOFOLLOW
                descriptor = os.open(path, flags)
                try:
                    opened = os.fstat(descriptor)
                    if not stat.S_ISREG(opened.st_mode):
                        raise RuntimeError(
                            "failed-attempt archive file changed type during validation"
                        )
                    digest = hashlib.sha256()
                    with os.fdopen(descriptor, "rb", closefd=False) as handle:
                        for block in iter(lambda: handle.read(1024 * 1024), b""):
                            digest.update(block)
                    files.append(
                        {
                            "path": child_relative.as_posix(),
                            "bytes": opened.st_size,
                            "sha256": digest.hexdigest(),
                        }
                    )
                finally:
                    os.close(descriptor)
            else:
                raise RuntimeError(
                    "failed-attempt archive contains a symlink or non-regular node"
                )
        directories.append(
            {
                "path": "." if not relative.parts else relative.as_posix(),
                "entries": direct,
            }
        )
        for child, child_relative in children:
            walk(child, child_relative)

    walk(root, Path())
    files.sort(key=lambda item: str(item["path"]))
    directories.sort(key=lambda item: str(item["path"]))
    return files, directories


def _valid_archived_failed_contrast_attempts(
    config: Mapping[str, Any], evaluation_output: Path, *, require_empty_output: bool
) -> list[dict[str, Any]]:
    """Return fully reopened archives for this exact contrast output."""

    output_relative = _repo_relative(evaluation_output)
    if require_empty_output and os.path.lexists(evaluation_output):
        raise RuntimeError("contrast replay requires an absent canonical output")
    failures_dir = ROOT / "runs" / "failures"
    candidates = (
        sorted(failures_dir.glob(f"{evaluation_output.name}-*.json"))
        if failures_dir.is_dir()
        else []
    )
    valid: list[dict[str, Any]] = []
    try:
        for receipt_path in candidates:
            receipt = validate_failed_archive(
                REPO_ROOT,
                receipt_path,
                expected_header=_failed_archive_header(config),
            )
            manifests = receipt["attempts"]
            if len(manifests) != 1 or manifests[0]["source_path"] != output_relative:
                continue
            authority = manifests[0]["archive_authority"]
            if (
                authority.get("attempt_kind") != "contrast"
                or authority.get("canonical_paths") != [output_relative]
                or any(row.get("path") == "summary.json" for row in manifests[0]["files"])
            ):
                raise RuntimeError("contrast archive authority or terminal state changed")
            valid.append(archive_lineage(REPO_ROOT, receipt_path, receipt))
    except (OSError, ValueError, AttemptReceiptError) as exc:
        raise RuntimeError(f"failed-attempt archive validation failed: {exc}") from exc
    return valid


def _contrast_attempt_cell(capacity: str, objective: str, model_seed: int) -> dict[str, Any]:
    return {
        "capacity": capacity,
        "objective": objective,
        "seed": int(model_seed),
        "slug": f"{capacity}_{objective}_seed{int(model_seed)}_contrast",
    }


def _contrast_attempt_context(
    *,
    authorization: Mapping[str, Any],
    checkpoint_lineage: Mapping[str, Any],
    data_manifest_sha256: str,
    data_root: Path,
) -> dict[str, Any]:
    return {
        "authorization": dict(authorization),
        "checkpoint_lineage": dict(checkpoint_lineage),
        "data_manifest_sha256": data_manifest_sha256,
        "data_manifest_path": _repo_relative(data_root / "manifest.json"),
        "contrast_access_ledger_path": _repo_relative(
            data_root / ACCESS_LEDGER_NAME
        ),
        "splits": sorted(SEALED_SPLITS),
    }


def _refresh_event_identity(event: dict[str, Any]) -> None:
    event.pop("event_identity_sha256", None)
    event["event_identity_sha256"] = _canonical_sha256(event)


def _terminal_contrast_summary(event: Mapping[str, Any]) -> dict[str, Any] | None:
    output = canonical_attempt_path(
        REPO_ROOT, str(event["evaluation_output"]), require_exists=False
    )
    summary_path = output / "summary.json"
    if not os.path.lexists(summary_path):
        return None
    try:
        summary = read_attempt_json(summary_path)
    except (OSError, ValueError, AttemptReceiptError) as exc:
        raise RuntimeError(f"terminal contrast summary cannot be reopened: {exc}") from exc
    claimed = summary.get("receipt_identity_sha256")
    if claimed != _canonical_sha256(
        {key: value for key, value in summary.items() if key != "receipt_identity_sha256"}
    ):
        raise RuntimeError("terminal contrast summary identity changed")
    expected = {
        "status": "STATE_EVALUATION_COMPLETE",
        "eval_set": "contrast",
        "capacity": event["capacity"],
        "objective": event["objective"],
        "model_seed": event["model_seed"],
        "contrast_access_event": dict(event),
    }
    if any(summary.get(key) != value for key, value in expected.items()):
        raise RuntimeError("terminal contrast summary/event binding changed")
    return summary


def load_contrast_access_ledger(
    config: Mapping[str, Any],
    output_dir: Path,
    manifest: Mapping[str, Any],
    *,
    payload: Mapping[str, Any] | None = None,
    manifest_sha256: str | None = None,
) -> dict[str, Any]:
    path = output_dir / ACCESS_LEDGER_NAME
    if payload is None:
        if not path.is_file():
            raise RuntimeError(f"contrast access ledger is missing: {path}")
        observed_payload, _ = _stable_json_snapshot(
            output_dir if not path.is_relative_to(REPO_ROOT) else REPO_ROOT,
            path,
            "contrast access ledger",
        )
    elif not isinstance(payload, Mapping):
        raise RuntimeError("contrast access ledger snapshot is not an object")
    else:
        observed_payload = json.loads(json.dumps(dict(payload)))
    if observed_payload.get("receipt_identity_sha256") != _ledger_identity(observed_payload):
        raise RuntimeError("contrast access ledger identity mismatch")
    manifest_path = output_dir / "manifest.json"
    if manifest_sha256 is None:
        manifest_snapshot, observed_manifest_sha256 = _stable_json_snapshot(
            output_dir if not manifest_path.is_relative_to(REPO_ROOT) else REPO_ROOT,
            manifest_path,
            "prepared data manifest",
        )
        if manifest_snapshot != dict(manifest):
            raise RuntimeError("contrast access ledger manifest snapshot changed")
    else:
        observed_manifest_sha256 = manifest_sha256
    if (
        type(observed_manifest_sha256) is not str
        or len(observed_manifest_sha256) != 64
    ):
        raise RuntimeError("contrast access ledger manifest digest is malformed")
    expected = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "data_manifest_sha256": observed_manifest_sha256,
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
        if observed_payload.get(key) != value:
            raise RuntimeError(f"contrast access ledger {key} mismatch")
    if not isinstance(observed_payload.get("events"), list):
        raise RuntimeError("contrast access ledger events are malformed")
    event_keys = {
        "event_index", "opened_at_utc", "authorization", "capacity", "objective",
        "model_seed", "evaluation_output", "checkpoint_lineage", "splits",
        "attempts", "replay_archives", "event_identity_sha256",
    }
    registered_cells = _registered_contrast_cells(config)
    if len(observed_payload["events"]) > len(registered_cells):
        raise RuntimeError("contrast access ledger exceeds the registered six-cell prefix")
    common_authorization = None
    for index, event in enumerate(observed_payload["events"], start=1):
        if not isinstance(event, Mapping) or set(event) != event_keys:
            raise RuntimeError("contrast access event fields are malformed")
        if event.get("event_index") != index or event.get("splits") != sorted(SEALED_SPLITS):
            raise RuntimeError("contrast access event order or split set changed")
        expected_cell = registered_cells[index - 1]
        observed_cell = (
            event.get("capacity"), event.get("objective"), event.get("model_seed")
        )
        if observed_cell != expected_cell:
            raise RuntimeError("contrast access ledger is not the registered canonical prefix")
        expected_output = (
            ROOT
            / "runs"
            / f"{expected_cell[0]}_joint_seed{expected_cell[2]}_contrast"
        )
        if event.get("evaluation_output") != _repo_relative(expected_output):
            raise RuntimeError("contrast access event uses a noncanonical output path")
        if common_authorization is None:
            common_authorization = event.get("authorization")
        elif event.get("authorization") != common_authorization:
            raise RuntimeError("contrast access ledger mixes authorization receipts")
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
            output = canonical_attempt_path(
                REPO_ROOT, str(event["evaluation_output"]), require_exists=False
            )
            valid = _valid_archived_failed_contrast_attempts(
                config, output, require_empty_output=False
            )
            if any(item not in valid for item in recorded):
                raise RuntimeError("contrast replay archive lineage changed")
        attempts = event.get("attempts")
        if not isinstance(attempts, list) or not attempts:
            raise RuntimeError("contrast access event has no attempt journal")
        expected_cell_payload = _contrast_attempt_cell(*expected_cell)
        expected_context = _contrast_attempt_context(
            authorization=event["authorization"],
            checkpoint_lineage=event["checkpoint_lineage"],
            data_manifest_sha256=observed_payload["data_manifest_sha256"],
            data_root=output_dir,
        )
        for attempt_index, attempt in enumerate(attempts, start=1):
            if not isinstance(attempt, Mapping) or set(attempt) != {"authorization", "state"}:
                raise RuntimeError("contrast attempt event fields changed")
            try:
                attempt_auth = validate_attempt_authorization(
                    attempt["authorization"], attempt_kind="contrast"
                )
            except AttemptReceiptError as exc:
                raise RuntimeError(f"contrast attempt authorization failed: {exc}") from exc
            if (
                attempt_auth["attempt_index"] != attempt_index
                or attempt_auth["cell"] != expected_cell_payload
                or attempt_auth["canonical_paths"] != [event["evaluation_output"]]
                or attempt_auth["context"] != expected_context
            ):
                raise RuntimeError("contrast attempt authorization binding changed")
            expected_replay = None if attempt_index == 1 else recorded[attempt_index - 2]
            if attempt_auth["replay_archive"] != expected_replay:
                raise RuntimeError("contrast attempt/replay archive binding changed")
            if attempt.get("state") not in {"PREPARED", "STARTED"}:
                raise RuntimeError("contrast attempt state changed")
            if attempt_index < len(attempts) and attempt["state"] != "STARTED":
                raise RuntimeError("historical contrast attempt was never STARTED")
        if len(recorded) != len(attempts) - 1:
            raise RuntimeError("contrast replay history length changed")
        if index < len(observed_payload["events"]):
            if attempts[-1]["state"] != "STARTED" or _terminal_contrast_summary(event) is None:
                raise RuntimeError("contrast prefix advances past a nonterminal cell")
    return observed_payload


def record_contrast_access(
    config: Mapping[str, Any], output_dir: Path, manifest: Mapping[str, Any],
    *, authorization: Mapping[str, Any], capacity: str, objective: str,
    model_seed: int, evaluation_output: Path,
    checkpoint_lineage: Mapping[str, Any],
) -> dict[str, Any]:
    """Durably PREPARE an authorization-bound access before output creation."""

    path = output_dir / ACCESS_LEDGER_NAME
    manifest_path = output_dir / "manifest.json"
    manifest_snapshot, manifest_sha256 = _stable_json_snapshot(
        output_dir if not manifest_path.is_relative_to(REPO_ROOT) else REPO_ROOT,
        manifest_path,
        "prepared data manifest",
    )
    if manifest_snapshot != dict(manifest):
        raise RuntimeError("contrast access manifest argument differs from durable bytes")
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
    try:
        authorization_path = canonical_attempt_path(
            REPO_ROOT, str(authorization["path"])
        )
    except AttemptReceiptError as exc:
        raise RuntimeError("contrast authorization path is noncanonical") from exc
    expected_authorization_path = ROOT / "analysis" / "stage_b_seal.json"
    if authorization_path != expected_authorization_path:
        raise RuntimeError("contrast authorization must be the canonical Stage-B seal")
    authorization_receipt, authorization_sha256 = _stable_json_snapshot(
        REPO_ROOT, authorization_path, "contrast authorization"
    )
    if authorization_sha256 != authorization["sha256"]:
        raise RuntimeError("contrast authorization file changed")
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
    firewall = authorization_receipt.get("contrast_firewall")
    if (
        not isinstance(firewall, Mapping)
        or firewall.get("status") != "CONTRAST_FIREWALL_UNOPENED"
        or firewall.get("events") != 0
        or firewall.get("ledger_path") != _repo_relative(path)
        or firewall.get("data_manifest_sha256") != manifest_sha256
        or firewall.get("authorization") != authorization_receipt.get("authorization")
    ):
        raise RuntimeError("Stage-B contrast firewall receipt is invalid")
    if capacity not in {"lora", "fullrank"} or objective != "joint":
        raise RuntimeError("only registered joint capacity cells may open contrast data")
    if int(model_seed) not in set(map(int, config["training"]["train_seeds"])):
        raise RuntimeError("contrast access model seed is not registered")
    if set(checkpoint_lineage) != {
        "path", "metadata_sha256", "checkpoint_identity_sha256"
    }:
        raise RuntimeError("contrast checkpoint lineage has the wrong fields")
    try:
        checkpoint_path = canonical_attempt_path(
            REPO_ROOT, str(checkpoint_lineage["path"])
        )
    except AttemptReceiptError as exc:
        raise RuntimeError("contrast checkpoint path is noncanonical") from exc
    checkpoint_metadata_path = checkpoint_path / "checkpoint.json"
    checkpoint_metadata, checkpoint_sha256 = _stable_json_snapshot(
        REPO_ROOT, checkpoint_metadata_path, "contrast checkpoint metadata"
    )
    if checkpoint_sha256 != checkpoint_lineage["metadata_sha256"]:
        raise RuntimeError("contrast checkpoint metadata changed")
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
        "requirements_training_lock_sha256": _requirements_sha256(),
        "data_manifest_sha256": manifest_sha256,
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
    )
    lexical_evaluation_output = _require_no_symlink_components(
        REPO_ROOT,
        evaluation_output,
        "contrast evaluation output",
    )
    if lexical_evaluation_output != expected_evaluation_output:
        raise RuntimeError(
            "contrast access requires the exact canonical evaluation output path"
        )
    output_relative = _repo_relative(evaluation_output)
    lock_path = path.with_name(f"{path.name}.lock")
    with locked_regular(lock_path):
        if not path.is_file():
            raise RuntimeError("contrast access ledger disappeared while acquiring its lock")
        payload, locked_ledger_sha256 = _stable_json_snapshot(
            REPO_ROOT, path, "contrast access ledger"
        )
        # The separate lock inode remains stable across atomic ledger replaces.
        if payload.get("receipt_identity_sha256") != _ledger_identity(payload):
            raise RuntimeError("contrast access ledger identity mismatch")
        locked_manifest, locked_manifest_sha256 = _stable_json_snapshot(
            output_dir if not manifest_path.is_relative_to(REPO_ROOT) else REPO_ROOT,
            manifest_path,
            "prepared data manifest",
        )
        if locked_manifest != manifest_snapshot or locked_manifest_sha256 != manifest_sha256:
            raise RuntimeError("prepared data manifest changed while authorizing contrast")
        locked_expected = {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "data_manifest_sha256": locked_manifest_sha256,
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
        # Reopen the locked bytes through the full prefix validator before
        # deciding whether this is a new access or a same-cell replay.
        validated_locked = load_contrast_access_ledger(
            config,
            output_dir,
            manifest,
            payload=payload,
            manifest_sha256=locked_manifest_sha256,
        )
        if validated_locked != payload:
            raise RuntimeError("locked contrast ledger changed during prefix validation")
        registered_cells = _registered_contrast_cells(config)
        if not payload["events"] and locked_ledger_sha256 != firewall.get("ledger_sha256"):
            raise RuntimeError("first contrast access does not match the Stage-B empty ledger")
        if len(payload["events"]) > len(registered_cells):
            raise RuntimeError("contrast access ledger exceeds the registered six cells")
        for prior in payload["events"]:
            if prior.get("authorization") != dict(authorization):
                raise RuntimeError("contrast access ledger mixes authorization receipts")
            prior_cell = (
                prior.get("capacity"), prior.get("objective"), prior.get("model_seed")
            )
            expected_prior_checkpoint = (
                authorization_receipt.get("matching", {})
                .get("per_seed", {})
                .get(str(prior.get("model_seed")), {})
                .get("checkpoint_lineages", {})
                .get(f"{prior.get('capacity')}_joint")
            )
            if prior.get("checkpoint_lineage") != expected_prior_checkpoint:
                raise RuntimeError(
                    "contrast access prefix contains a checkpoint outside the Stage-B seal"
                )
            if prior_cell == cell:
                if (
                    prior.get("evaluation_output") != output_relative
                    or prior.get("checkpoint_lineage") != dict(checkpoint_lineage)
                ):
                    raise RuntimeError("contrast cell already opened under a different output path")
                if prior is not payload["events"][-1]:
                    raise RuntimeError("completed contrast cells cannot be replayed out of order")
                recorded_archives = prior.get("replay_archives")
                if not isinstance(recorded_archives, list):
                    raise RuntimeError("contrast access event replay archive history is malformed")
                attempts = prior.get("attempts")
                if not isinstance(attempts, list) or not attempts:
                    raise RuntimeError("contrast access event has no attempt journal")
                head = attempts[-1]
                if head.get("state") == "PREPARED":
                    if os.path.lexists(evaluation_output):
                        marker = evaluation_output / ATTEMPT_MARKER_NAME
                        try:
                            if (
                                not evaluation_output.is_dir()
                                or {entry.name for entry in evaluation_output.iterdir()}
                                != {ATTEMPT_MARKER_NAME}
                            ):
                                raise RuntimeError("PREPARED contrast output is not marker-only")
                            validate_attempt_marker(
                                read_attempt_json(marker), head["authorization"]
                            )
                        except (OSError, ValueError, AttemptReceiptError) as exc:
                            raise RuntimeError(f"PREPARED contrast marker changed: {exc}") from exc
                    return dict(prior)
                if head.get("state") != "STARTED":
                    raise RuntimeError("contrast attempt head state changed")
                if os.path.lexists(evaluation_output):
                    raise RuntimeError(
                        "STARTED contrast output must be archived before identical replay"
                    )
                try:
                    replay_archive = find_exact_failed_archive(
                        REPO_ROOT,
                        label=evaluation_output.name,
                        expected_header=_failed_archive_header(config),
                        attempt_kind="contrast",
                        attempt_identity_sha256=head["authorization"][
                            "attempt_identity_sha256"
                        ],
                        canonical_paths=[output_relative],
                    )
                except AttemptReceiptError as exc:
                    raise RuntimeError(f"contrast replay archive is invalid: {exc}") from exc
                if replay_archive in recorded_archives:
                    raise RuntimeError("contrast replay archive was already consumed")
                prior["replay_archives"].append(replay_archive)
                next_auth = build_attempt_authorization(
                    attempt_kind="contrast",
                    attempt_index=len(attempts) + 1,
                    cell=_contrast_attempt_cell(capacity, objective, model_seed),
                    canonical_paths=[output_relative],
                    context=_contrast_attempt_context(
                        authorization=authorization,
                        checkpoint_lineage=checkpoint_lineage,
                        data_manifest_sha256=payload["data_manifest_sha256"],
                        data_root=output_dir,
                    ),
                    replay_archive=replay_archive,
                )
                attempts.append({"authorization": next_auth, "state": "PREPARED"})
                _refresh_event_identity(prior)
                payload["receipt_identity_sha256"] = _ledger_identity(payload)
                durable_atomic_write_json(path, payload, replace=True)
                return dict(prior)
        next_cell = registered_cells[len(payload["events"])] if len(payload["events"]) < len(
            registered_cells
        ) else None
        if cell != next_cell:
            raise RuntimeError("contrast access is out of registered canonical order")
        if os.path.lexists(evaluation_output):
            raise RuntimeError("first contrast access requires an absent canonical output")
        if _valid_archived_failed_contrast_attempts(
            config, evaluation_output, require_empty_output=True
        ):
            raise RuntimeError(
                "a failed-attempt archive predates the first contrast access event"
            )
        attempt_authorization = build_attempt_authorization(
            attempt_kind="contrast",
            attempt_index=1,
            cell=_contrast_attempt_cell(capacity, objective, model_seed),
            canonical_paths=[output_relative],
            context=_contrast_attempt_context(
                authorization=authorization,
                checkpoint_lineage=checkpoint_lineage,
                data_manifest_sha256=payload["data_manifest_sha256"],
                data_root=output_dir,
            ),
            replay_archive=None,
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
            "attempts": [
                {"authorization": attempt_authorization, "state": "PREPARED"}
            ],
            "replay_archives": [],
        }
        _refresh_event_identity(event)
        payload["events"].append(event)
        payload["receipt_identity_sha256"] = _ledger_identity(payload)
        durable_atomic_write_json(path, payload, replace=True)
    return event


def start_contrast_access(
    config: Mapping[str, Any],
    output_dir: Path,
    manifest: Mapping[str, Any],
    *,
    capacity: str,
    objective: str,
    model_seed: int,
    evaluation_output: Path,
) -> dict[str, Any]:
    """Transition the marker-bound ledger head PREPARED -> STARTED durably."""

    path = output_dir / ACCESS_LEDGER_NAME
    lock_path = path.with_name(f"{path.name}.lock")
    with locked_regular(lock_path):
        payload = load_contrast_access_ledger(config, output_dir, manifest)
        if not payload["events"]:
            raise RuntimeError("contrast access has no PREPARED event")
        event = payload["events"][-1]
        expected_cell = (capacity, objective, int(model_seed))
        observed_cell = (
            event.get("capacity"), event.get("objective"), event.get("model_seed")
        )
        if observed_cell != expected_cell:
            raise RuntimeError("contrast STARTED transition targets a different cell")
        output_relative = _repo_relative(evaluation_output)
        if event.get("evaluation_output") != output_relative:
            raise RuntimeError("contrast STARTED transition targets a different output")
        attempts = event.get("attempts")
        if not isinstance(attempts, list) or not attempts:
            raise RuntimeError("contrast event has no attempt journal")
        head = attempts[-1]
        if head.get("state") == "STARTED":
            return dict(event)
        if head.get("state") != "PREPARED":
            raise RuntimeError("contrast attempt cannot transition to STARTED")
        try:
            auth = validate_attempt_authorization(
                head["authorization"], attempt_kind="contrast"
            )
            canonical = canonical_attempt_path(REPO_ROOT, output_relative)
            if (
                canonical != evaluation_output.absolute()
                or not canonical.is_dir()
                or {entry.name for entry in canonical.iterdir()} != {ATTEMPT_MARKER_NAME}
            ):
                raise RuntimeError("contrast output opened before its STARTED transition")
            validate_attempt_marker(
                read_attempt_json(canonical / ATTEMPT_MARKER_NAME), auth
            )
        except (OSError, ValueError, AttemptReceiptError) as exc:
            raise RuntimeError(f"contrast attempt marker is invalid: {exc}") from exc
        head["state"] = "STARTED"
        _refresh_event_identity(event)
        payload["receipt_identity_sha256"] = _ledger_identity(payload)
        durable_atomic_write_json(path, payload, replace=True)
        return dict(event)


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
        compressed_receipt = _write_jsonl_gz(path, rows)
        files[split] = {
            "path": path.name,
            "rows": len(rows),
            "bytes": compressed_receipt["bytes"],
            "sha256": compressed_receipt["sha256"],
            "canonical_rows": _canonical_rows_from_rows(rows),
            "structural_fingerprints": sorted(str(row["structural_fingerprint"]) for row in rows),
            "families": dict(Counter(row["family"] for row in rows)),
            "templates": dict(Counter(row["template"] for row in rows)),
            "depths": dict(sorted(Counter(str(row["depth"]) for row in rows).items())),
            "query_kinds": dict(sorted(Counter(row["query_kind"] for row in rows).items())),
            "query_kind_grid": grid,
        }
        _data_crash_point(f"{split}_published")
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "data_contract_sha256": data_contract_sha256(config, source_digest=source_digest),
        "source_contract_version": SOURCE_CONTRACT_VERSION,
        "source_contract_sha256": source_digest,
        "config_sha256": config_sha256(config),
        "requirements_training_lock_sha256": _requirements_sha256(),
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
    _data_crash_point("manifest_published")
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
    _data_crash_point("ledger_published")
    return manifest
