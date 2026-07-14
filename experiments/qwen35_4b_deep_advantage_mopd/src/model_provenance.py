"""Fail-closed authentication for every deployable checkpoint byte.

The confirmation campaign must not infer trust from ``Path.is_file`` or from a
merge receipt alone.  This module defines the only two checkpoint load
profiles produced by the frozen pipeline and binds the exact seven root files,
the complete weight inventory, and (for new merges) the receipt's exhaustive
inference-file inventory without following symlinks.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path
from typing import Any, Mapping

from io_utils import canonical_hash, sha256_file


EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
SOURCE_RECEIPT = EXP / "runs" / "checkpoint_receipts.json"
SOURCE_RECEIPT_SHA256 = (
    "f8ea66a5482f305092f58067479b5e24b6f2e16850edd3ca9761629bc31a1178"
)
SOURCE_RECEIPT_COMMIT = "37dc74ef74d5a82e014d5496a042223a92f75c69"
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"

MODEL_ROOT_FILES = (
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "merge_receipt.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
)
INFERENCE_FILES = tuple(
    name for name in MODEL_ROOT_FILES if name != "merge_receipt.json"
)
COMMON_LOAD_PROFILE = {
    "chat_template.jinja": (
        "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"
    ),
    "config.json": (
        "f64803e92c6b93971a8e4fa89a811c8607b9b4efd6d0df0e6ca6016404f42630"
    ),
    "generation_config.json": (
        "8e29eeb222454c2a417b17547a84aee206c110a033418b7c0342cd75e75d92a0"
    ),
    "tokenizer.json": (
        "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
    ),
}
LOAD_PROFILES = {
    "source": {
        **COMMON_LOAD_PROFILE,
        "tokenizer_config.json": (
            "9cf04fffe3d8c3b85e439fb35c7acad0761ab51c422a8c4256d9f887c3a0be7d"
        ),
    },
    "local": {
        **COMMON_LOAD_PROFILE,
        "tokenizer_config.json": (
            "bee8eba30f0eb4af73c0fe2cd06d0f89b657d7819941c438157ec42f7c80ea87"
        ),
    },
}
CONFIRMATION_ARM_NAMES = frozenset(
    {
        "quick",
        "deep",
        "soup",
        "primary_seed42",
        "primary_seed43",
        "primary_seed44",
        "non_advantage_route",
        "wrong_teacher",
        "offpolicy_sft",
        "soup25",
        "soup50",
        "soup75",
        "soup_best8",
    }
)
ARM_FIELDS = {
    "model",
    "model_merge_receipt_sha256",
    "model_config_sha256",
    "model_inference_inventory_sha256",
    "decode",
}


def _lexical_absolute(path: Path) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = REPO / value
    return Path(os.path.abspath(os.fspath(value)))


def safe_existing_path(path: Path, *, label: str, directory: bool) -> Path:
    """Require a symlink-free lexical path of the requested filesystem kind."""

    lexical = _lexical_absolute(path)
    current = Path(lexical.anchor)
    parts = lexical.parts[1:] if lexical.anchor else lexical.parts
    for index, part in enumerate(parts):
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError as exc:
            raise ValueError(f"{label} is missing: {lexical}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{label} has a symlinked component: {current}")
        if index < len(parts) - 1 and not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"{label} has a non-directory ancestor: {current}")
    metadata = lexical.lstat()
    expected = stat.S_ISDIR if directory else stat.S_ISREG
    if not expected(metadata.st_mode):
        kind = "directory" if directory else "regular file"
        raise ValueError(f"{label} is not a {kind}: {lexical}")
    return lexical


def _inventory(model: Path) -> list[dict[str, str]]:
    """Hash the exact root inventory; nested, symlink, and special entries fail."""

    model = safe_existing_path(model, label="model directory", directory=True)
    rows: list[dict[str, str]] = []
    observed: set[str] = set()
    with os.scandir(model) as entries:
        for entry in entries:
            path = model / entry.name
            metadata = path.lstat()
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"model root contains an unsafe entry: {path}")
            observed.add(entry.name)
            rows.append({"path": entry.name, "sha256": sha256_file(path)})
    expected = set(MODEL_ROOT_FILES)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            f"model root inventory is not exact (missing={missing}, extra={extra})"
        )
    return sorted(rows, key=lambda row: row["path"])


def inference_file_inventory_for_receipt(model: Path) -> list[dict[str, str]]:
    """Inventory a newly saved model immediately before its receipt is written."""

    model = safe_existing_path(model, label="new model directory", directory=True)
    rows: list[dict[str, str]] = []
    observed: set[str] = set()
    with os.scandir(model) as entries:
        for entry in entries:
            path = model / entry.name
            metadata = path.lstat()
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"new model root contains an unsafe entry: {path}")
            observed.add(entry.name)
            rows.append({"path": entry.name, "sha256": sha256_file(path)})
    expected = set(INFERENCE_FILES)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            "new model inference inventory is not exact "
            f"(missing={missing}, extra={extra})"
        )
    return sorted(rows, key=lambda row: row["path"])


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    path = safe_existing_path(path, label=label, directory=False)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} is not an object")
    return payload


def _validate_rows(
    rows: object, *, fields: set[str], path_key: str, label: str
) -> list[dict[str, str]]:
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{label} is missing or empty")
    normalized: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != fields:
            raise ValueError(f"{label} row is malformed")
        name = row.get(path_key)
        digest = row.get("sha256")
        if (
            not isinstance(name, str)
            or not name
            or "\\" in name
            or Path(name).is_absolute()
            or Path(name).as_posix() != name
            or any(part in {"", ".", ".."} for part in Path(name).parts)
            or not isinstance(digest, str)
            or len(digest) != 64
        ):
            raise ValueError(f"{label} row is invalid")
        normalized.append({path_key: name, "sha256": digest})
    if (
        normalized != sorted(normalized, key=lambda row: row[path_key])
        or len({row[path_key] for row in normalized}) != len(normalized)
    ):
        raise ValueError(f"{label} is not canonical")
    return normalized


def validate_model_checkpoint(
    model: Path,
    *,
    profile: str,
    expected_merge_receipt_sha256: str | None = None,
    expected_weight_sha256: str | None = None,
    require_recorded_inference_inventory: bool = False,
) -> dict[str, Any]:
    """Authenticate one checkpoint and return its canonical confirmation binding."""

    if profile not in LOAD_PROFILES:
        raise ValueError(f"unknown model load profile: {profile}")
    model = safe_existing_path(model, label="model directory", directory=True)
    rows = _inventory(model)
    by_path = {row["path"]: row["sha256"] for row in rows}
    for name, expected in LOAD_PROFILES[profile].items():
        if by_path.get(name) != expected:
            raise ValueError(f"model {profile} load profile changed: {name}")

    receipt_path = model / "merge_receipt.json"
    receipt = _load_json_object(receipt_path, label="model merge receipt")
    receipt_sha256 = by_path["merge_receipt.json"]
    if (
        expected_merge_receipt_sha256 is not None
        and receipt_sha256 != expected_merge_receipt_sha256
    ):
        raise ValueError("model merge receipt hash changed")

    weight_rows = _validate_rows(
        receipt.get("weight_files"),
        fields={"name", "sha256"},
        path_key="name",
        label="model weight inventory",
    )
    if [row["name"] for row in weight_rows] != ["model.safetensors"]:
        raise ValueError("model weight inventory is not the exact load inventory")
    if weight_rows[0]["sha256"] != by_path["model.safetensors"]:
        raise ValueError("model weight inventory is stale")
    if (
        expected_weight_sha256 is not None
        and by_path["model.safetensors"] != expected_weight_sha256
    ):
        raise ValueError("model weight hash changed")

    recorded_inference = receipt.get("inference_files")
    actual_inference = [row for row in rows if row["path"] != "merge_receipt.json"]
    if recorded_inference is None:
        if require_recorded_inference_inventory:
            raise ValueError("model merge receipt lacks its inference inventory")
    else:
        normalized = _validate_rows(
            recorded_inference,
            fields={"path", "sha256"},
            path_key="path",
            label="model inference inventory",
        )
        if normalized != actual_inference:
            raise ValueError("model inference inventory is stale")

    return {
        "model": str(model),
        "profile": profile,
        "model_merge_receipt_sha256": receipt_sha256,
        "model_weight_inventory_sha256": canonical_hash(weight_rows),
        "model_config_sha256": by_path["config.json"],
        "model_inference_inventory_sha256": canonical_hash(rows),
        "model_inference_files_sha256": canonical_hash(actual_inference),
        "inventory": rows,
        "inference_files": actual_inference,
        "receipt": receipt,
    }


def validate_known_model_checkpoint(model: Path) -> dict[str, Any]:
    """Authenticate a checkpoint against either frozen, explicitly known profile."""

    tokenizer_config = safe_existing_path(
        Path(model) / "tokenizer_config.json",
        label="model tokenizer configuration",
        directory=False,
    )
    digest = sha256_file(tokenizer_config)
    matches = [
        name
        for name, rows in LOAD_PROFILES.items()
        if rows["tokenizer_config.json"] == digest
    ]
    if len(matches) != 1:
        raise ValueError("model does not match a frozen load profile")
    return validate_model_checkpoint(model, profile=matches[0])


def validate_source_checkpoint_receipts(
    config: Mapping[str, Any],
    *,
    receipt_path: Path = SOURCE_RECEIPT,
    expected_receipt_sha256: str = SOURCE_RECEIPT_SHA256,
    verify_source_commit: bool = True,
) -> dict[str, dict[str, Any]]:
    """Authenticate the committed quick/deep/soup receipt and current bytes."""

    receipt_path = safe_existing_path(
        receipt_path, label="source checkpoint receipt", directory=False
    )
    if sha256_file(receipt_path) != expected_receipt_sha256:
        raise ValueError("source checkpoint receipt changed")
    if verify_source_commit:
        completed = subprocess.run(
            ["git", "merge-base", "--is-ancestor", SOURCE_RECEIPT_COMMIT, "HEAD"],
            cwd=REPO,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if completed.returncode != 0:
            raise ValueError("source checkpoint receipt commit is not an ancestor")
    payload = _load_json_object(receipt_path, label="source checkpoint receipt")
    if (
        set(payload) != {"schema_version", "model", "revision", "quick", "deep", "soup"}
        or payload.get("schema_version") != 1
        or payload.get("model") != MODEL_ID
        or payload.get("revision") != MODEL_REVISION
        or config.get("model", {}).get("id") != MODEL_ID
        or config.get("model", {}).get("revision") != MODEL_REVISION
    ):
        raise ValueError("source checkpoint receipt identity is stale")

    expected_paths = {
        "quick": _lexical_absolute(Path(str(config["model"]["quick_teacher"]))),
        "deep": _lexical_absolute(Path(str(config["model"]["deep_teacher"]))),
        "soup": _lexical_absolute(Path(str(config["model"]["student_checkpoint"]))),
    }
    result: dict[str, dict[str, Any]] = {}
    for name, expected_path in expected_paths.items():
        row = payload.get(name)
        if (
            not isinstance(row, Mapping)
            or set(row) != {"path", "merge_receipt_sha256", "model_sha256"}
            or row.get("path") != str(expected_path)
            or not isinstance(row.get("merge_receipt_sha256"), str)
            or not isinstance(row.get("model_sha256"), str)
        ):
            raise ValueError(f"source {name} checkpoint receipt is stale")
        result[name] = validate_model_checkpoint(
            expected_path,
            profile="source",
            expected_merge_receipt_sha256=str(row["merge_receipt_sha256"]),
            expected_weight_sha256=str(row["model_sha256"]),
        )
    return result


def _repo_path(value: object) -> Path:
    return _lexical_absolute(Path(str(value)))


def _confirmation_specifications(
    config: Mapping[str, Any],
) -> dict[str, tuple[Path, str, str]]:
    """Construct canonical arm paths and profiles without trusting their bytes."""

    model_cfg = config["model"]
    artifacts_root = _repo_path(model_cfg["artifacts_root"])
    final_round = int(config["mopd"]["rounds"]) - 1
    seeds = [int(value) for value in config["seeds"]["integration_training"]]
    if seeds != [42, 43, 44]:
        raise ValueError("confirmation integration seed inventory changed")
    soup = _repo_path(model_cfg["student_checkpoint"])
    specifications: dict[str, tuple[Path, str, str]] = {
        "quick": (_repo_path(model_cfg["quick_teacher"]), "source", "greedy"),
        "deep": (_repo_path(model_cfg["deep_teacher"]), "source", "greedy"),
        "soup": (soup, "source", "greedy"),
        "soup_best8": (soup, "source", "sample8"),
    }
    for seed in seeds:
        specifications[f"primary_seed{seed}"] = (
            artifacts_root
            / "merged"
            / "primary"
            / f"seed_{seed}"
            / f"round_{final_round}",
            "local",
            "greedy",
        )
    for name in ("non_advantage_route", "wrong_teacher", "offpolicy_sft"):
        specifications[name] = (
            artifacts_root
            / "merged"
            / "controls"
            / name
            / f"round_{final_round}",
            "local",
            "greedy",
        )
    weights = [
        float(value) for value in config["controls"]["parameter_merge_deep_weights"]
    ]
    if weights != [0.25, 0.5, 0.75]:
        raise ValueError("confirmation parameter-control inventory changed")
    for weight in weights:
        name = f"soup{int(round(weight * 100)):02d}"
        specifications[name] = (
            artifacts_root / "merged" / name,
            "source",
            "greedy",
        )
    if set(specifications) != CONFIRMATION_ARM_NAMES:
        raise ValueError("confirmation arm inventory is incomplete")
    return specifications


def canonical_confirmation_arm_map(
    config: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    """Reauthenticate and return the one canonical 13-arm confirmation map."""

    validate_source_checkpoint_receipts(config)
    specifications = _confirmation_specifications(config)
    result: dict[str, dict[str, str]] = {}
    for name, (model, profile, decode) in sorted(specifications.items()):
        provenance = validate_model_checkpoint(
            model,
            profile=profile,
            require_recorded_inference_inventory=name.startswith("soup")
            and name not in {"soup", "soup_best8"},
        )
        result[name] = {
            "model": provenance["model"],
            "model_merge_receipt_sha256": provenance[
                "model_merge_receipt_sha256"
            ],
            "model_config_sha256": provenance["model_config_sha256"],
            "model_inference_inventory_sha256": provenance[
                "model_inference_inventory_sha256"
            ],
            "decode": decode,
        }
    validate_confirmation_arm_map(result)
    return result


def reauthenticate_confirmation_arm(
    config: Mapping[str, Any],
    *,
    name: str,
    expected: Mapping[str, Any],
) -> dict[str, str]:
    """Rehash one authorized arm immediately around its evaluation transaction."""

    specifications = _confirmation_specifications(config)
    if name not in specifications:
        raise ValueError(f"unknown confirmation arm: {name}")
    model, profile, decode = specifications[name]
    require_inventory = name.startswith("soup") and name not in {
        "soup",
        "soup_best8",
    }
    provenance = validate_model_checkpoint(
        model,
        profile=profile,
        expected_merge_receipt_sha256=str(
            expected.get("model_merge_receipt_sha256", "")
        ),
        require_recorded_inference_inventory=require_inventory,
    )
    observed = {
        "model": provenance["model"],
        "model_merge_receipt_sha256": provenance[
            "model_merge_receipt_sha256"
        ],
        "model_config_sha256": provenance["model_config_sha256"],
        "model_inference_inventory_sha256": provenance[
            "model_inference_inventory_sha256"
        ],
        "decode": decode,
    }
    if observed != dict(expected):
        raise ValueError(f"confirmation arm bytes changed: {name}")
    return observed


def validate_confirmation_arm_map(
    arms: object,
) -> dict[str, dict[str, str]]:
    """Validate the sealed arm-map schema without adopting any new model bytes."""

    if not isinstance(arms, Mapping) or set(arms) != CONFIRMATION_ARM_NAMES:
        raise ValueError("confirmation authorization arm inventory is stale")
    normalized: dict[str, dict[str, str]] = {}
    for name in sorted(arms):
        row = arms[name]
        if not isinstance(row, Mapping) or set(row) != ARM_FIELDS:
            raise ValueError(f"confirmation authorization arm is malformed: {name}")
        model = row.get("model")
        decode = row.get("decode")
        if (
            not isinstance(model, str)
            or not Path(model).is_absolute()
            or decode != ("sample8" if name == "soup_best8" else "greedy")
        ):
            raise ValueError(f"confirmation authorization arm is invalid: {name}")
        normalized[name] = {key: str(row[key]) for key in sorted(ARM_FIELDS)}
        for key in ARM_FIELDS - {"model", "decode"}:
            if len(normalized[name][key]) != 64:
                raise ValueError(f"confirmation authorization digest is invalid: {name}")
    soup = dict(normalized["soup"])
    best8 = dict(normalized["soup_best8"])
    soup["decode"] = "sample8"
    if best8 != soup:
        raise ValueError("soup_best8 is not the exact soup checkpoint alias")
    return normalized


def confirmation_arm_map_sha256(arms: object) -> str:
    return canonical_hash(validate_confirmation_arm_map(arms))
