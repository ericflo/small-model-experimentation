"""Atomic, fail-closed storage for sealed confirmation artifacts."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import stat
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping

from confirmation_protocol import (
    PINNED_FORCED_CLOSE,
    _strict_json_equal,
    canonical_backend_protocol,
    canonical_confirmation_sampling_protocol,
    capacity_receipt,
    validate_capacity_receipt,
    validate_live_cache_capacity,
)
from control_code_inventory import control_code_inventory
from gym import base
from gym.families import load as load_family
from io_utils import (
    all_families,
    canonical_hash,
    confirmation_evaluator_source_inventory,
    load_config,
    sha256_file,
)
from model_provenance import (  # noqa: E402
    SOURCE_RECEIPT,
    SOURCE_RECEIPT_COMMIT,
    confirmation_arm_map_sha256,
    validate_confirmation_arm_map,
)


EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
CONFIRMATION_SCORE_ROOT = EXP / "runs" / "confirmation"
CONFIRMATION_RAW_ROOT = REPO / "large_artifacts" / EXP.name / "confirmation"
RAW_FILENAMES = {
    "atom_rows": "atom_rows.jsonl.gz",
    "episode_rows": "episode_rows.jsonl.gz",
}
DESCRIPTOR_FIELDS = {"path", "sha256", "bytes", "rows"}
MARKER_FILENAMES = {
    "started": "STARTED.json",
    "generated": "GENERATED.json",
    "complete": "COMPLETE.json",
    "quarantined": "QUARANTINED.json",
}
BUNDLE_PREFIX = "bundle_"
BUNDLE_SUFFIX = ".jsonl.gz"
POLICY_STARTED_CONTEXT_FIELDS = {
    "stage",
    "tag",
    "scope",
    "block_seed",
    "decode",
    "k",
    "sampling_protocol",
    "config_sha256",
    "evaluator_sha256",
    "evaluator_source_inventory",
    "model",
    "model_merge_receipt_sha256",
    "model_config_sha256",
    "model_inference_inventory_sha256",
    "task_manifest_sha256",
    "ordered_plan_sha256",
    "capacity_preflight",
    "controls_authorization",
    "confirmation_admission",
}
POLICY_STARTED_DIRECT_BINDINGS = POLICY_STARTED_CONTEXT_FIELDS - {
    "evaluator_source_inventory",
    "capacity_preflight",
}
STARTED_MARKER_FIELDS = {
    "schema_version",
    "state",
    "tag",
    "score_path",
    "context",
    "context_sha256",
}
GENERATED_MARKER_FIELDS = {
    "schema_version",
    "state",
    "tag",
    "started_sha256",
    "candidate_payload",
    "candidate_payload_sha256",
    "raw_artifacts",
    "call_bundles",
}


def _lexical_absolute(path: Path, *, base: Path = REPO) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = base / value
    return Path(os.path.abspath(os.fspath(value)))


def _reject_symlink_components(
    path: Path, *, label: str, require_leaf: bool
) -> None:
    """Inspect lexical components with lstat before any path resolution."""

    path = _lexical_absolute(path)
    current = Path(path.anchor)
    parts = path.parts[1:] if path.anchor else path.parts
    for index, part in enumerate(parts):
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            if require_leaf:
                raise ValueError(f"confirmation {label} path component is missing")
            return
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"confirmation {label} path contains a symlink")


def _sealed_canonical_file(
    path: Path, *, expected: Path, containment_root: Path, label: str
) -> Path:
    lexical = _lexical_absolute(path)
    expected_lexical = _lexical_absolute(expected)
    root_lexical = _lexical_absolute(containment_root)
    if lexical != expected_lexical or not lexical.is_relative_to(root_lexical):
        raise ValueError(f"confirmation {label} path is not canonical")
    _reject_symlink_components(lexical, label=label, require_leaf=True)
    metadata = lexical.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"confirmation {label} path is not a regular file")
    resolved = lexical.resolve(strict=True)
    if resolved != expected_lexical or not resolved.is_relative_to(root_lexical):
        raise ValueError(f"confirmation {label} resolved outside its sealed root")
    return resolved


def configured_confirmation_raw_root(config: Mapping) -> Path:
    """Resolve the config-bound external root for sealed confirmation rows."""

    try:
        artifacts_root = config["model"]["artifacts_root"]
    except (KeyError, TypeError) as exc:
        raise ValueError("confirmation config lacks model.artifacts_root") from exc
    configured = _lexical_absolute(Path(str(artifacts_root)))
    expected = _lexical_absolute(REPO / "large_artifacts" / EXP.name)
    if configured != expected:
        raise ValueError("confirmation artifact root is not the frozen experiment root")
    raw_root = expected / "confirmation"
    _reject_symlink_components(
        raw_root, label="raw artifact root", require_leaf=False
    )
    resolved = raw_root.resolve(strict=False)
    repository = _lexical_absolute(REPO)
    if resolved != raw_root or not resolved.is_relative_to(repository):
        raise ValueError("confirmation raw artifact root escaped its canonical path")
    return raw_root


def controls_authorization_binding(
    path: Path, *, expected_config_sha256: str
) -> dict[str, Any]:
    """Authenticate the exact pre-confirmation authorization and code inventory."""

    path = _sealed_canonical_file(
        Path(path),
        expected=EXP / "analysis" / "controls_authorization.json",
        containment_root=EXP,
        label="controls authorization",
    )
    payload = _read_json_object(path, label="controls authorization receipt")
    confirmation_arms = validate_confirmation_arm_map(
        payload.get("confirmation_arms")
    )
    inventory = payload.get("control_code_inventory")
    if not isinstance(inventory, dict) or set(inventory) != {
        "files",
        "file_count",
        "sha256",
    }:
        raise ValueError("confirmation control-code inventory is missing or malformed")
    files = inventory.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("confirmation control-code file inventory is empty")
    normalized = []
    seen = set()
    for row in files:
        if not isinstance(row, Mapping) or set(row) != {"path", "sha256"}:
            raise ValueError("confirmation control-code inventory row is malformed")
        relative = row.get("path")
        digest = row.get("sha256")
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise ValueError("confirmation control-code path is invalid")
        source = (REPO / relative).resolve()
        if (
            relative in seen
            or not source.is_relative_to(REPO)
            or not source.is_file()
            or source.is_symlink()
            or not isinstance(digest, str)
            or digest != sha256_file(source)
        ):
            raise ValueError("confirmation control-code inventory is stale")
        seen.add(relative)
        normalized.append({"path": relative, "sha256": digest})
    if normalized != sorted(normalized, key=lambda row: row["path"]):
        raise ValueError("confirmation control-code inventory is not canonical")
    if (
        inventory.get("file_count") != len(normalized)
        or inventory.get("sha256") != canonical_hash(normalized)
        or inventory != control_code_inventory()
        or payload.get("stage") != "semantic_controls_confirmation_authorization"
        or payload.get("config_sha256") != expected_config_sha256
        or payload.get("control_code_inventory_sha256")
        != canonical_hash(inventory)
        or payload.get("control_code_inventory_before_sha256")
        != inventory["sha256"]
        or payload.get("control_code_inventory_after_sha256")
        != inventory["sha256"]
        or payload.get("source_checkpoint_receipt")
        != {
            "path": str(SOURCE_RECEIPT),
            "sha256": sha256_file(SOURCE_RECEIPT),
            "commit": SOURCE_RECEIPT_COMMIT,
        }
        or payload.get("confirmation_arms") != confirmation_arms
        or payload.get("confirmation_arms_sha256")
        != confirmation_arm_map_sha256(confirmation_arms)
        or payload.get("gate") != {"passed": True}
        or payload.get("downstream_authorization")
        != "sealed_confirmation_evaluation"
    ):
        raise ValueError("confirmation controls authorization receipt is stale")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "payload_sha256": canonical_hash(payload),
        "control_code_inventory": inventory,
        "control_code_inventory_sha256": canonical_hash(inventory),
        "confirmation_arms": confirmation_arms,
        "confirmation_arms_sha256": confirmation_arm_map_sha256(
            confirmation_arms
        ),
    }


def confirmation_admission_binding(
    path: Path,
    *,
    expected_config_sha256: str,
    expected_controls_authorization: Mapping,
    expected_tag: str | None = None,
    expected_block_seed: int | None = None,
    expected_model: Mapping | None = None,
) -> dict[str, Any]:
    """Authenticate the global no-clobber admission that predates every score."""

    path = _sealed_canonical_file(
        Path(path),
        expected=CONFIRMATION_SCORE_ROOT / "ADMISSION.json",
        containment_root=CONFIRMATION_SCORE_ROOT,
        label="global admission",
    )
    payload = _read_json_object(path, label="global confirmation admission")
    authorized_arms = validate_confirmation_arm_map(
        expected_controls_authorization.get("confirmation_arms")
    )
    if (
        payload.get("schema_version") != 1
        or payload.get("stage") != "sealed_confirmation_admission"
        or payload.get("config_sha256") != expected_config_sha256
        or payload.get("controls_authorization")
        != dict(expected_controls_authorization)
        or not isinstance(payload.get("blocks"), list)
        or payload.get("arms") != authorized_arms
        or expected_controls_authorization.get("confirmation_arms_sha256")
        != confirmation_arm_map_sha256(authorized_arms)
        or payload.get("evaluator_sha256")
        != sha256_file(EXP / "scripts" / "eval_policy.py")
        or payload.get("evaluator_source_inventory")
        != confirmation_evaluator_source_inventory()
    ):
        raise ValueError("confirmation global admission is stale")
    if expected_tag is not None:
        try:
            block_name, arm = expected_tag.split("_", 2)[1:]
            block_index = int(block_name)
        except (ValueError, IndexError) as exc:
            raise ValueError("confirmation tag cannot bind global admission") from exc
        blocks = payload["blocks"]
        arms = payload["arms"]
        if (
            block_index < 0
            or block_index >= len(blocks)
            or blocks[block_index] != expected_block_seed
            or arm not in arms
            or expected_model is None
            or arms[arm] != dict(expected_model)
        ):
            raise ValueError("confirmation invocation is absent from global admission")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "payload_sha256": canonical_hash(payload),
        "controls_authorization_sha256": expected_controls_authorization.get("sha256"),
        "control_code_inventory_sha256": expected_controls_authorization.get(
            "control_code_inventory_sha256"
        ),
    }


def _resolved_root(path: Path) -> Path:
    lexical = _lexical_absolute(Path(path))
    _reject_symlink_components(
        lexical, label="artifact root", require_leaf=False
    )
    resolved = lexical.resolve(strict=False)
    if resolved != lexical:
        raise ValueError("confirmation artifact root is not canonical")
    return lexical


def _score_layout(
    score_path: Path,
    *,
    score_root: Path,
    raw_root: Path,
) -> tuple[Path, Path, Path, Path]:
    score_root = _resolved_root(score_root)
    raw_root = _resolved_root(raw_root)
    score_path = _lexical_absolute(Path(score_path))
    if score_path.name != "scores.json":
        raise ValueError("confirmation commit marker must be named scores.json")
    try:
        relative = score_path.relative_to(score_root)
    except ValueError as exc:
        raise ValueError("confirmation score escaped the visible score root") from exc
    if len(relative.parts) != 3:
        raise ValueError("confirmation score must have block/arm/scores.json layout")
    block_name, arm_name, _ = relative.parts
    if (
        not block_name.startswith("block_")
        or not block_name.removeprefix("block_").isdigit()
        or not arm_name
    ):
        raise ValueError("confirmation score path has invalid block or arm")
    _reject_symlink_components(
        score_path, label="score", require_leaf=False
    )
    if score_path.resolve(strict=False) != score_path:
        raise ValueError("confirmation score path contains a symlink")
    raw_dir = raw_root / relative.parent
    _reject_symlink_components(
        raw_dir, label="raw", require_leaf=False
    )
    if raw_dir.resolve(strict=False) != raw_dir:
        raise ValueError("confirmation raw path contains a symlink")
    if not raw_dir.is_relative_to(raw_root):
        raise ValueError("confirmation raw directory escaped the artifact root")
    return score_path, score_root, raw_dir, raw_root


def confirmation_raw_dir(
    score_path: Path,
    *,
    score_root: Path = CONFIRMATION_SCORE_ROOT,
    raw_root: Path = CONFIRMATION_RAW_ROOT,
) -> Path:
    """Return the exact external raw directory mirrored from a score path."""

    return _score_layout(
        score_path, score_root=score_root, raw_root=raw_root
    )[2]


def _safe_entries(directory: Path) -> list[Path]:
    if not directory.exists():
        if directory.is_symlink():
            raise ValueError("confirmation artifact directory is a broken symlink")
        return []
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("confirmation artifact directory is not a real directory")
    return list(directory.iterdir())


def _bundle_index(name: str) -> int | None:
    if not name.startswith(BUNDLE_PREFIX) or not name.endswith(BUNDLE_SUFFIX):
        return None
    value = name[len(BUNDLE_PREFIX) : -len(BUNDLE_SUFFIX)]
    return int(value) if len(value) == 4 and value.isdigit() else None


def _read_json_object(path: Path, *, label: str) -> dict:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"confirmation {label} is missing or unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"confirmation {label} is invalid") from exc
    if not isinstance(value, dict):
        raise ValueError(f"confirmation {label} is not an object")
    return value


def _descriptor(path: Path, *, rows: int) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "rows": rows,
    }


def _validate_quarantine(raw_dir: Path, marker: Path) -> None:
    payload = _read_json_object(marker, label="quarantine marker")
    inventory = payload.get("retained_files")
    if not isinstance(inventory, list):
        raise ValueError("confirmation quarantine inventory is malformed")
    observed = []
    for entry in sorted(
        (value for value in raw_dir.iterdir() if value.name != marker.name),
        key=lambda value: value.name,
    ):
        if not entry.is_file() or entry.is_symlink():
            raise ValueError("confirmation quarantine contains an unsafe entry")
        observed.append(
            {
                "name": entry.name,
                "sha256": sha256_file(entry),
                "bytes": entry.stat().st_size,
            }
        )
    if inventory != observed:
        raise ValueError("confirmation quarantine inventory is stale")


def confirmation_transaction_state(
    score_path: Path,
    *,
    score_root: Path = CONFIRMATION_SCORE_ROOT,
    raw_root: Path = CONFIRMATION_RAW_ROOT,
) -> str:
    """Inspect, but never repair, one sealed confirmation transaction."""

    score_path, _, raw_dir, _ = _score_layout(
        score_path, score_root=score_root, raw_root=raw_root
    )
    visible = _safe_entries(score_path.parent)
    if visible and {entry.name for entry in visible} != {"scores.json"}:
        raise ValueError("unknown partial confirmation artifacts beside score")
    entries = _safe_entries(raw_dir)
    names = {entry.name for entry in entries}
    if any(not entry.is_file() or entry.is_symlink() for entry in entries):
        raise ValueError("confirmation transaction contains an unsafe raw entry")
    bundle_indexes = sorted(
        index
        for entry in entries
        if (index := _bundle_index(entry.name)) is not None
    )
    if bundle_indexes != list(range(len(bundle_indexes))):
        raise ValueError("confirmation call journal is not contiguous")
    quarantine = raw_dir / MARKER_FILENAMES["quarantined"]
    if quarantine.name in names:
        _validate_quarantine(raw_dir, quarantine)
        return "QUARANTINED"
    known = set(RAW_FILENAMES.values()) | set(MARKER_FILENAMES.values())
    unknown = sorted(
        name for name in names if name not in known and _bundle_index(name) is None
    )
    if unknown:
        raise ValueError("unknown partial confirmation artifacts: " + ", ".join(unknown))
    started = MARKER_FILENAMES["started"] in names
    generated = MARKER_FILENAMES["generated"] in names
    complete = MARKER_FILENAMES["complete"] in names
    raw_present = {name for name in RAW_FILENAMES.values() if name in names}
    if score_path.exists() or score_path.is_symlink():
        if score_path.is_symlink() or not score_path.is_file():
            raise ValueError("confirmation visible score is unsafe")
        if not (started and generated and complete) or raw_present != set(
            RAW_FILENAMES.values()
        ):
            raise ValueError("committed confirmation score lacks its transaction")
        return "COMMITTED"
    if complete:
        if not started or not generated or raw_present != set(RAW_FILENAMES.values()):
            raise ValueError("confirmation COMPLETE state is structurally invalid")
        return "COMPLETE"
    if generated:
        if not started or raw_present != set(RAW_FILENAMES.values()):
            raise ValueError("confirmation GENERATED state is structurally invalid")
        return "GENERATED"
    if started:
        return "STARTED_DIRTY" if raw_present else "STARTED"
    if names:
        raise ValueError("unreceipted confirmation bytes are terminal")
    return "EMPTY"


def validate_confirmation_campaign_tree(
    admission_path: Path,
    *,
    raw_root: Path,
    score_root: Path = CONFIRMATION_SCORE_ROOT,
    terminal: bool,
    require_manifest: bool = False,
) -> dict[str, str]:
    """Exhaustively inventory the visible and raw admitted campaign geometry."""

    score_root = _resolved_root(score_root)
    raw_root = _resolved_root(raw_root)
    admission_path = _lexical_absolute(Path(admission_path))
    expected_admission = score_root / "ADMISSION.json"
    if admission_path != expected_admission:
        raise ValueError("confirmation campaign admission path is not canonical")
    _reject_symlink_components(
        admission_path, label="campaign admission", require_leaf=True
    )
    admission = _read_json_object(
        admission_path, label="campaign admission"
    )
    blocks = admission.get("blocks")
    arms = admission.get("arms")
    if (
        admission.get("schema_version") != 1
        or admission.get("stage") != "sealed_confirmation_admission"
        or not isinstance(blocks, list)
        or not blocks
        or any(
            not isinstance(seed, int) or isinstance(seed, bool) for seed in blocks
        )
        or len(blocks) != len(set(blocks))
        or not isinstance(arms, dict)
        or not arms
        or any(
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or name in {".", ".."}
            or not isinstance(value, Mapping)
            for name, value in arms.items()
        )
    ):
        raise ValueError("confirmation campaign admission geometry is malformed")
    arm_names = set(arms)
    block_names = {f"block_{index}" for index in range(len(blocks))}

    visible_entries = _safe_entries(score_root)
    visible_by_name = {entry.name: entry for entry in visible_entries}
    if len(visible_by_name) != len(visible_entries):
        raise ValueError("confirmation visible campaign has duplicate entries")
    allowed_visible = {"ADMISSION.json", "manifest.json", *block_names}
    extras = set(visible_by_name) - allowed_visible
    if extras:
        raise ValueError(
            "confirmation visible campaign contains unregistered entries: "
            + ", ".join(sorted(extras))
        )
    if set(visible_by_name) & {"ADMISSION.json"} != {"ADMISSION.json"}:
        raise ValueError("confirmation visible campaign lacks ADMISSION.json")
    if "manifest.json" in visible_by_name and (
        visible_by_name["manifest.json"].is_symlink()
        or not visible_by_name["manifest.json"].is_file()
    ):
        raise ValueError("confirmation campaign manifest is unsafe")
    if require_manifest and "manifest.json" not in visible_by_name:
        raise ValueError("confirmation terminal campaign lacks manifest.json")
    visible_blocks = set(visible_by_name) & block_names
    if terminal and visible_blocks != block_names:
        raise ValueError("confirmation terminal campaign lacks admitted blocks")

    visible_arms: dict[str, set[str]] = {}
    for block_name in sorted(visible_blocks):
        block_dir = visible_by_name[block_name]
        if block_dir.is_symlink() or not block_dir.is_dir():
            raise ValueError("confirmation visible block is unsafe")
        entries = list(block_dir.iterdir())
        by_name = {entry.name: entry for entry in entries}
        if len(by_name) != len(entries) or not set(by_name).issubset(arm_names):
            raise ValueError(
                "confirmation visible block contains unregistered arms"
            )
        if terminal and set(by_name) != arm_names:
            raise ValueError("confirmation terminal block lacks admitted arms")
        visible_arms[block_name] = set(by_name)
        for arm_name, arm_dir in by_name.items():
            if arm_dir.is_symlink() or not arm_dir.is_dir():
                raise ValueError("confirmation visible arm is unsafe")
            contents = list(arm_dir.iterdir())
            names = {entry.name for entry in contents}
            if len(names) != len(contents) or names not in (
                set(),
                {"scores.json"},
            ):
                raise ValueError(
                    "confirmation visible arm contains unregistered files"
                )
            if names and (
                contents[0].is_symlink() or not contents[0].is_file()
            ):
                raise ValueError("confirmation visible score is unsafe")
            if terminal and names != {"scores.json"}:
                raise ValueError("confirmation terminal arm lacks scores.json")

    raw_entries = _safe_entries(raw_root)
    raw_by_name = {entry.name: entry for entry in raw_entries}
    if (
        len(raw_by_name) != len(raw_entries)
        or not set(raw_by_name).issubset(block_names)
        or set(raw_by_name) != visible_blocks
    ):
        raise ValueError(
            "confirmation raw campaign differs from visible admitted blocks"
        )
    states: dict[str, str] = {}
    for block_name in sorted(visible_blocks):
        raw_block = raw_by_name[block_name]
        if raw_block.is_symlink() or not raw_block.is_dir():
            raise ValueError("confirmation raw block is unsafe")
        entries = list(raw_block.iterdir())
        raw_arms = {entry.name: entry for entry in entries}
        if (
            len(raw_arms) != len(entries)
            or set(raw_arms) != visible_arms[block_name]
        ):
            raise ValueError(
                "confirmation raw arms differ from visible admitted arms"
            )
        for arm_name, raw_arm in raw_arms.items():
            if raw_arm.is_symlink() or not raw_arm.is_dir():
                raise ValueError("confirmation raw arm is unsafe")
            score_path = score_root / block_name / arm_name / "scores.json"
            state = confirmation_transaction_state(
                score_path, score_root=score_root, raw_root=raw_root
            )
            states[f"{block_name}/{arm_name}"] = state
            if terminal and state != "COMMITTED":
                raise ValueError(
                    "confirmation terminal campaign has an incomplete arm"
                )
            if terminal:
                validate_confirmation_score_artifacts(
                    score_path,
                    expected_tag=f"{block_name}_{arm_name}",
                    score_root=score_root,
                    raw_root=raw_root,
                )
    if terminal and len(states) != len(blocks) * len(arms):
        raise ValueError("confirmation terminal campaign geometry is incomplete")
    return states


def prepare_confirmation_output(
    score_path: Path,
    *,
    score_root: Path = CONFIRMATION_SCORE_ROOT,
    raw_root: Path = CONFIRMATION_RAW_ROOT,
) -> Path:
    """Admit an empty or verification-only resumable output; never delete bytes."""

    score_path, _, raw_dir, _ = _score_layout(
        score_path, score_root=score_root, raw_root=raw_root
    )
    score_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    state = confirmation_transaction_state(
        score_path, score_root=score_root, raw_root=raw_root
    )
    if state not in {"EMPTY", "GENERATED", "COMPLETE"}:
        if state == "COMMITTED":
            raise ValueError("confirmation score commit marker already exists")
        raise ValueError(f"confirmation transaction is terminal in state {state}")
    return raw_dir


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_temp(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return tempfile.NamedTemporaryFile(
        mode="w+b",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )


def _publish_no_clobber(temporary_path: Path, path: Path, *, label: str) -> None:
    """Atomically publish a same-directory temp without replacing any writer."""

    try:
        os.link(temporary_path, path)
    except FileExistsError as exc:
        raise ValueError(f"refusing to overwrite confirmation {label}: {path}") from exc
    temporary_path.unlink()
    _fsync_directory(path.parent)


def _write_gzip_jsonl_atomic(path: Path, rows: Iterable[Mapping]) -> dict:
    count = 0
    temporary_path: Path | None = None
    try:
        with _atomic_temp(path) as raw_handle:
            temporary_path = Path(raw_handle.name)
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw_handle, mtime=0
            ) as compressed:
                for row in rows:
                    if not isinstance(row, Mapping):
                        raise ValueError("confirmation raw row is not an object")
                    line = json.dumps(
                        dict(row), sort_keys=True, ensure_ascii=False
                    ).encode("utf-8")
                    compressed.write(line + b"\n")
                    count += 1
            raw_handle.flush()
            os.fsync(raw_handle.fileno())
        _publish_no_clobber(temporary_path, path, label="raw artifact")
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "rows": count,
    }


def _json_bytes(payload: Mapping) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _write_json_atomic(path: Path, payload: Mapping, *, label: str = "score") -> None:
    temporary_path: Path | None = None
    try:
        with _atomic_temp(path) as handle:
            temporary_path = Path(handle.name)
            handle.write(_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        _publish_no_clobber(temporary_path, path, label=label)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def begin_confirmation_transaction(
    score_path: Path,
    context: Mapping,
    *,
    score_root: Path = CONFIRMATION_SCORE_ROOT,
    raw_root: Path = CONFIRMATION_RAW_ROOT,
) -> dict:
    """Reserve one arm with an immutable STARTED marker."""

    if not isinstance(context, Mapping):
        raise ValueError("confirmation STARTED context is malformed")
    score_path, resolved_score_root, raw_dir, _ = _score_layout(
        score_path, score_root=score_root, raw_root=raw_root
    )
    expected_tag = f"{score_path.parent.parent.name}_{score_path.parent.name}"
    if resolved_score_root == _resolved_root(CONFIRMATION_SCORE_ROOT):
        _validate_policy_started_context(context, expected_tag=expected_tag)
    prepare_confirmation_output(
        score_path, score_root=score_root, raw_root=raw_root
    )
    if confirmation_transaction_state(
        score_path, score_root=score_root, raw_root=raw_root
    ) != "EMPTY":
        raise ValueError("confirmation transaction reservation lost a race")
    tag = context.get("tag")
    if not isinstance(tag, str) or not tag:
        raise ValueError("confirmation STARTED context requires a non-empty tag")
    if tag != expected_tag:
        raise ValueError("confirmation STARTED tag does not match its score path")
    marker = {
        "schema_version": 1,
        "state": "STARTED",
        "tag": tag,
        "score_path": str(score_path),
        "context": dict(context),
        "context_sha256": canonical_hash(dict(context)),
    }
    path = raw_dir / MARKER_FILENAMES["started"]
    _write_json_atomic(path, marker, label="STARTED marker")
    return marker


def journal_confirmation_bundle(
    score_path: Path,
    bundle: Mapping,
    *,
    score_root: Path = CONFIRMATION_SCORE_ROOT,
    raw_root: Path = CONFIRMATION_RAW_ROOT,
) -> dict:
    """Durably retain each returned runner call before downstream scoring."""

    score_path, _, raw_dir, _ = _score_layout(
        score_path, score_root=score_root, raw_root=raw_root
    )
    state = confirmation_transaction_state(
        score_path, score_root=score_root, raw_root=raw_root
    )
    if state != "STARTED":
        raise ValueError(f"cannot journal confirmation call in state {state}")
    indexes = sorted(
        index
        for entry in raw_dir.iterdir()
        if (index := _bundle_index(entry.name)) is not None
    )
    if indexes != list(range(len(indexes))):
        raise ValueError("confirmation call journal is not contiguous")
    path = raw_dir / f"{BUNDLE_PREFIX}{len(indexes):04d}{BUNDLE_SUFFIX}"
    return _write_gzip_jsonl_atomic(path, [dict(bundle)])


def _bundle_inventory(raw_dir: Path) -> list[dict]:
    rows = []
    for entry in sorted(raw_dir.iterdir(), key=lambda value: value.name):
        index = _bundle_index(entry.name)
        if index is None:
            continue
        if not entry.is_file() or entry.is_symlink():
            raise ValueError("confirmation call journal contains an unsafe entry")
        loaded = _read_gzip_jsonl(entry)
        if len(loaded) != 1:
            raise ValueError("confirmation call journal bundle is malformed")
        rows.append({"index": index, **_descriptor(entry, rows=1)})
    if [row["index"] for row in rows] != list(range(len(rows))):
        raise ValueError("confirmation call journal is not contiguous")
    return rows


def persist_confirmation_generated(
    score_path: Path,
    payload: Mapping,
    *,
    atom_rows: Iterable[Mapping],
    episode_rows: Iterable[Mapping],
    score_root: Path = CONFIRMATION_SCORE_ROOT,
    raw_root: Path = CONFIRMATION_RAW_ROOT,
) -> dict:
    """Persist raw rows and publish GENERATED before any semantic authentication."""

    if "raw_artifacts" in payload or "raw_artifact_schema_version" in payload:
        raise ValueError("caller must not supply confirmation raw descriptors")
    tag = payload.get("tag")
    if not isinstance(tag, str) or not tag:
        raise ValueError("confirmation score requires a non-empty tag")
    score_path, _, raw_dir, _ = _score_layout(
        score_path, score_root=score_root, raw_root=raw_root
    )
    if confirmation_transaction_state(
        score_path, score_root=score_root, raw_root=raw_root
    ) != "STARTED":
        raise ValueError("confirmation GENERATED requires a clean STARTED state")
    atom_rows = list(atom_rows)
    episode_rows = list(episode_rows)
    descriptors = {
        "atom_rows": _write_gzip_jsonl_atomic(
            raw_dir / RAW_FILENAMES["atom_rows"], atom_rows
        ),
        "episode_rows": _write_gzip_jsonl_atomic(
            raw_dir / RAW_FILENAMES["episode_rows"], episode_rows
        ),
    }
    started_path = raw_dir / MARKER_FILENAMES["started"]
    generated = {
        "schema_version": 1,
        "state": "GENERATED",
        "tag": tag,
        "started_sha256": sha256_file(started_path),
        "candidate_payload": dict(payload),
        "candidate_payload_sha256": canonical_hash(dict(payload)),
        "raw_artifacts": descriptors,
        "call_bundles": _bundle_inventory(raw_dir),
    }
    _write_json_atomic(
        raw_dir / MARKER_FILENAMES["generated"],
        generated,
        label="GENERATED marker",
    )
    return generated


def _load_generated(raw_dir: Path, *, score_path: Path) -> tuple[dict, dict]:
    started_path = raw_dir / MARKER_FILENAMES["started"]
    generated_path = raw_dir / MARKER_FILENAMES["generated"]
    started = _read_json_object(started_path, label="STARTED marker")
    generated = _read_json_object(generated_path, label="GENERATED marker")
    context = started.get("context")
    expected_tag = f"{score_path.parent.parent.name}_{score_path.parent.name}"
    if (
        set(started) != STARTED_MARKER_FIELDS
        or started.get("schema_version") != 1
        or started.get("state") != "STARTED"
        or started.get("tag") != expected_tag
        or started.get("score_path") != str(score_path)
        or not isinstance(context, dict)
        or started.get("context_sha256") != canonical_hash(context)
        or set(generated) != GENERATED_MARKER_FIELDS
        or generated.get("schema_version") != 1
        or generated.get("state") != "GENERATED"
        or generated.get("started_sha256") != sha256_file(started_path)
        or generated.get("tag") != started.get("tag")
    ):
        raise ValueError("confirmation transaction marker binding is stale")
    candidate = generated.get("candidate_payload")
    if (
        not isinstance(candidate, dict)
        or generated.get("candidate_payload_sha256") != canonical_hash(candidate)
        or candidate.get("tag") != generated.get("tag")
    ):
        raise ValueError("confirmation GENERATED candidate is stale")
    if generated.get("call_bundles") != _bundle_inventory(raw_dir):
        raise ValueError("confirmation call journal changed after generation")
    return started, generated


def load_confirmation_generated_payload(
    score_path: Path,
    *,
    score_root: Path = CONFIRMATION_SCORE_ROOT,
    raw_root: Path = CONFIRMATION_RAW_ROOT,
) -> dict:
    """Read the immutable GENERATED candidate for verification-only resume."""

    _, _, raw_dir, _ = _score_layout(
        score_path, score_root=score_root, raw_root=raw_root
    )
    _, generated = _load_generated(raw_dir, score_path=score_path)
    return dict(generated["candidate_payload"])


def finalize_confirmation_score(
    score_path: Path,
    *,
    expected_tag: str,
    payload_validator: Callable[[Mapping], None] | None = None,
    score_root: Path = CONFIRMATION_SCORE_ROOT,
    raw_root: Path = CONFIRMATION_RAW_ROOT,
) -> dict:
    """Authenticate GENERATED bytes, publish COMPLETE, then scores.json last."""

    score_path, _, raw_dir, _ = _score_layout(
        score_path, score_root=score_root, raw_root=raw_root
    )
    state = confirmation_transaction_state(
        score_path, score_root=score_root, raw_root=raw_root
    )
    if state == "COMMITTED":
        return validate_confirmation_score_artifacts(
            score_path,
            expected_tag=expected_tag,
            score_root=score_root,
            raw_root=raw_root,
        )
    if state not in {"GENERATED", "COMPLETE"}:
        raise ValueError(f"cannot finalize confirmation transaction in state {state}")
    started_marker, generated = _load_generated(raw_dir, score_path=score_path)
    generated_candidate = dict(generated["candidate_payload"])
    if generated_candidate.get("tag") != expected_tag:
        raise ValueError("confirmation GENERATED tag does not match requested arm")
    descriptors = generated.get("raw_artifacts")
    loaded = _validate_raw_descriptors(
        descriptors,
        raw_dir=raw_dir,
        raw_root=_resolved_root(raw_root),
        expected_row_counts=_expected_raw_row_counts(generated_candidate),
    )
    if _raw_item_projection(
        loaded["atom_rows"], loaded["episode_rows"]
    ) != _score_item_projection(generated_candidate):
        raise ValueError("confirmation raw semantics do not match scored items")
    candidate = _authenticated_payload(
        generated_candidate,
        atom_rows=loaded["atom_rows"],
        episode_rows=loaded["episode_rows"],
    )
    context = started_marker["context"]
    _validate_started_context(candidate, context)
    _validate_capacity_against_started(candidate, context)
    _validate_raw_item_semantics(
        candidate,
        atom_rows=loaded["atom_rows"],
        episode_rows=loaded["episode_rows"],
    )
    _validate_call_journal(
        candidate,
        generated,
        atom_rows=loaded["atom_rows"],
        episode_rows=loaded["episode_rows"],
    )
    _validate_preregistered_confirmation_tasks(
        candidate,
        atom_rows=loaded["atom_rows"],
        episode_rows=loaded["episode_rows"],
    )
    if payload_validator is not None:
        payload_validator(candidate)
    generated_path = raw_dir / MARKER_FILENAMES["generated"]
    started_path = raw_dir / MARKER_FILENAMES["started"]
    committed = {
        **candidate,
        "raw_artifact_schema_version": 2,
        "raw_artifacts": descriptors,
        "confirmation_transaction": {
            "schema_version": 1,
            "started_sha256": sha256_file(started_path),
            "generated_sha256": sha256_file(generated_path),
            "call_bundle_count": len(generated["call_bundles"]),
        },
    }
    score_sha256 = hashlib.sha256(_json_bytes(committed)).hexdigest()
    complete = {
        "schema_version": 1,
        "state": "COMPLETE",
        "tag": expected_tag,
        "started_sha256": sha256_file(started_path),
        "generated_sha256": sha256_file(generated_path),
        "score_sha256": score_sha256,
        "raw_artifacts": descriptors,
        "task_manifest_sha256": candidate.get("task_manifest_sha256"),
        "ordered_plan_sha256": candidate.get("ordered_plan_sha256"),
    }
    complete_path = raw_dir / MARKER_FILENAMES["complete"]
    if state == "GENERATED":
        _write_json_atomic(complete_path, complete, label="COMPLETE marker")
    elif _read_json_object(complete_path, label="COMPLETE marker") != complete:
        raise ValueError("confirmation COMPLETE marker is stale")
    if score_path.exists() or score_path.is_symlink():
        raise ValueError("refusing to overwrite confirmation score")
    _write_json_atomic(score_path, committed)
    return validate_confirmation_score_artifacts(
        score_path,
        expected_tag=expected_tag,
        score_root=score_root,
        raw_root=raw_root,
    )


def quarantine_confirmation_transaction(
    score_path: Path,
    *,
    reason: str,
    score_root: Path = CONFIRMATION_SCORE_ROOT,
    raw_root: Path = CONFIRMATION_RAW_ROOT,
) -> dict:
    """Seal all retained bytes after a failed started transaction."""

    score_path, _, raw_dir, _ = _score_layout(
        score_path, score_root=score_root, raw_root=raw_root
    )
    marker = raw_dir / MARKER_FILENAMES["quarantined"]
    if marker.exists() or marker.is_symlink():
        _validate_quarantine(raw_dir, marker)
        return _read_json_object(marker, label="quarantine marker")
    if score_path.exists() or (raw_dir / MARKER_FILENAMES["complete"]).exists():
        raise ValueError("completed confirmation transaction cannot be quarantined")
    if not (raw_dir / MARKER_FILENAMES["started"]).is_file():
        raise ValueError("confirmation quarantine requires STARTED provenance")
    retained = []
    for entry in sorted(raw_dir.iterdir(), key=lambda value: value.name):
        if not entry.is_file() or entry.is_symlink():
            raise ValueError("confirmation quarantine found an unsafe entry")
        retained.append(
            {"name": entry.name, "sha256": sha256_file(entry), "bytes": entry.stat().st_size}
        )
    payload = {
        "schema_version": 1,
        "state": "QUARANTINED",
        "reason": str(reason),
        "retained_files": retained,
    }
    _write_json_atomic(marker, payload, label="QUARANTINED marker")
    return payload


def commit_confirmation_score(
    score_path: Path,
    payload: Mapping,
    *,
    atom_rows: Iterable[Mapping],
    episode_rows: Iterable[Mapping],
    score_root: Path = CONFIRMATION_SCORE_ROOT,
    raw_root: Path = CONFIRMATION_RAW_ROOT,
) -> dict:
    """Convenience transaction used by tests; production journals each call too."""

    begin_confirmation_transaction(
        score_path,
        {"tag": payload.get("tag"), "mode": "single_commit"},
        score_root=score_root,
        raw_root=raw_root,
    )
    try:
        persist_confirmation_generated(
            score_path,
            payload,
            atom_rows=atom_rows,
            episode_rows=episode_rows,
            score_root=score_root,
            raw_root=raw_root,
        )
        return finalize_confirmation_score(
            score_path,
            expected_tag=str(payload.get("tag")),
            score_root=score_root,
            raw_root=raw_root,
        )
    except Exception as exc:
        try:
            quarantine_confirmation_transaction(
                score_path,
                reason=f"{type(exc).__name__}: {exc}",
                score_root=score_root,
                raw_root=raw_root,
            )
        except ValueError:
            pass
        raise


def _read_gzip_jsonl(path: Path) -> list[dict]:
    rows = []
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.endswith("\n") or not line.strip():
                    raise ValueError("confirmation raw JSONL has a malformed line")
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("confirmation raw JSONL row is not an object")
                rows.append(row)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid confirmation raw gzip: {path}") from exc
    return rows


def _finite_score(value, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"confirmation {label} is not numeric")
    score = float(value)
    if not math.isfinite(score):
        raise ValueError(f"confirmation {label} is non-finite")
    return score


def _plain_int(value, *, label: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"confirmation {label} is invalid")
    return value


def _score_item_projection(payload: Mapping) -> list[dict]:
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("confirmation score has no item geometry")
    projected = []
    for item in items:
        if not isinstance(item, Mapping):
            raise ValueError("confirmation score item is not an object")
        key = item.get("key")
        family = item.get("family")
        kind = item.get("kind")
        if not isinstance(key, str) or not key:
            raise ValueError("confirmation score item has invalid key")
        if not isinstance(family, str) or not family:
            raise ValueError("confirmation score item has invalid family")
        if kind not in {"atom", "episode"}:
            raise ValueError("confirmation score item has invalid kind")
        projected.append(
            {
                "key": key,
                "family": family,
                "kind": kind,
                "level": _plain_int(item.get("level"), label="item level"),
                "score": _finite_score(item.get("score"), label="item score"),
                "samples": _plain_int(
                    item.get("samples"), label="item sample count", minimum=1
                ),
            }
        )
    projected.sort(key=lambda row: row["key"])
    keys = [row["key"] for row in projected]
    if len(keys) != len(set(keys)):
        raise ValueError("confirmation score contains duplicate item keys")
    return projected


def _json_task_value(value: Any, *, label: str) -> Any:
    try:
        json.dumps(value, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"confirmation {label} is not canonical JSON") from exc
    return value


def confirmation_task_hashes(
    atom_rows: list[dict], episode_rows: list[dict]
) -> dict[str, str]:
    """Hash exact task bytes and their frozen evaluation order."""

    atoms = []
    ordered = []
    atom_ids = set()
    for row in atom_rows:
        key = row.get("id")
        if not isinstance(key, str) or not key or key in atom_ids:
            raise ValueError("confirmation task manifest has duplicate atom ids")
        atom_ids.add(key)
        task = {
            "kind": "atom",
            "id": key,
            "family": row.get("family"),
            "level": _plain_int(row.get("level"), label="raw atom level"),
            "prompt": row.get("prompt"),
            "gold": _json_task_value(row.get("gold"), label="atom gold"),
            "answer_domain": _json_task_value(
                row.get("answer_domain"), label="atom answer domain"
            ),
        }
        if not isinstance(task["family"], str) or not task["family"]:
            raise ValueError("confirmation task manifest has invalid atom family")
        if not isinstance(task["prompt"], str):
            raise ValueError("confirmation task manifest has invalid atom prompt")
        atoms.append(task)
        ordered.append({"kind": "atom", "id": key})

    episodes: dict[tuple[str, int, int], dict] = {}
    episode_order = []
    for row in episode_rows:
        family = row.get("family")
        if not isinstance(family, str) or not family:
            raise ValueError("confirmation task manifest has invalid episode family")
        level = _plain_int(row.get("level"), label="raw episode level")
        seed = _plain_int(row.get("ep_seed"), label="raw episode seed")
        identity = (family, level, seed)
        task = {
            "kind": "episode",
            "family": family,
            "level": level,
            "ep_seed": seed,
            "spec": _json_task_value(row.get("spec"), label="episode spec"),
            "system_prompt": row.get("system_prompt"),
            "initial_observation": row.get("initial_observation"),
        }
        if not isinstance(task["system_prompt"], str) or not isinstance(
            task["initial_observation"], str
        ):
            raise ValueError("confirmation task manifest has invalid episode prompts")
        if identity in episodes and not _strict_json_equal(
            episodes[identity], task
        ):
            raise ValueError("confirmation episode rollouts disagree on exact task bytes")
        if identity not in episodes:
            episodes[identity] = task
            episode_order.append(
                {
                    "kind": "episode",
                    "family": family,
                    "level": level,
                    "ep_seed": seed,
                }
            )
    manifest = {
        "schema_version": 1,
        "atoms": sorted(atoms, key=lambda row: row["id"]),
        "episodes": [episodes[key] for key in sorted(episodes)],
    }
    plan = {"schema_version": 1, "items": [*ordered, *episode_order]}
    return {
        "task_manifest_sha256": canonical_hash(manifest),
        "ordered_plan_sha256": canonical_hash(plan),
    }


def _validate_preregistered_confirmation_tasks(
    payload: Mapping,
    *,
    atom_rows: list[dict],
    episode_rows: list[dict],
) -> None:
    """Regenerate the admitted block and require exact task bytes and order."""

    if payload.get("stage") != "policy_eval":
        return
    config_value = payload.get("config")
    if not isinstance(config_value, str) or not config_value:
        raise ValueError("confirmation score lacks its task-generation config")
    config_path = _lexical_absolute(Path(config_value))
    expected_config = _lexical_absolute(EXP / "configs" / "default.yaml")
    if config_path != expected_config:
        raise ValueError("confirmation task-generation config is not canonical")
    _reject_symlink_components(
        config_path, label="task-generation config", require_leaf=True
    )
    if payload.get("config_sha256") != sha256_file(config_path):
        raise ValueError("confirmation task-generation config changed")
    try:
        config, loaded_path = load_config(config_path)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise ValueError("confirmation task-generation config is invalid") from exc
    if loaded_path != config_path:
        raise ValueError("confirmation task-generation config path drifted")

    block_seed = _plain_int(
        payload.get("block_seed"), label="task-generation block seed"
    )
    block_seeds = [int(value) for value in config["seeds"]["confirmatory_blocks"]]
    tag = payload.get("tag")
    try:
        tag_parts = tag.split("_", 2)
        if len(tag_parts) != 3 or tag_parts[0] != "block" or not tag_parts[2]:
            raise ValueError
        block_index = int(tag_parts[1])
    except (AttributeError, ValueError) as exc:
        raise ValueError("confirmation task tag cannot bind a block seed") from exc
    if (
        block_index < 0
        or block_index >= len(block_seeds)
        or block_seeds[block_index] != block_seed
    ):
        raise ValueError("confirmation task block seed is not preregistered")

    families = all_families(config)
    atom_n = _plain_int(
        config["confirmation"]["atoms_per_family_level"],
        label="preregistered atom count",
        minimum=1,
    )
    episode_n = _plain_int(
        config["confirmation"]["episodes_per_family_level"],
        label="preregistered episode count",
        minimum=1,
    )
    decode = payload.get("decode")
    if decode == "greedy":
        expected_k = 1
    elif decode == "sample8":
        expected_k = int(config["controls"]["sample_more_k"])
    else:
        expected_k = None
    payload_atom_n = _plain_int(
        payload.get("atoms_per_level"), label="task-generation atom count"
    )
    payload_episode_n = _plain_int(
        payload.get("episodes_per_level"), label="task-generation episode count"
    )
    payload_k = _plain_int(payload.get("k"), label="task-generation sample count")
    if (
        payload.get("scope") != "confirmatory"
        or payload.get("families") != families
        or payload_atom_n != atom_n
        or payload_episode_n != episode_n
        or expected_k is None
        or payload_k != expected_k
    ):
        raise ValueError("confirmation task-generation geometry is stale")

    strata = config["strata"]
    atom_levels = [
        *strata["quick_atom_levels"],
        *strata["deep_atom_levels"],
    ]
    expected_atoms = []
    for family_index, family_name in enumerate(families):
        family = load_family(family_name)
        for raw_level in atom_levels:
            level = int(raw_level)
            if level not in family.LEVELS:
                continue
            seed = block_seed + family_index * 100_000 + level * 1_000
            generated = family.gen_atoms(seed, level, atom_n)
            if not isinstance(generated, list) or len(generated) != atom_n:
                raise ValueError("confirmation atom generator geometry drifted")
            for item in generated:
                if not isinstance(item, Mapping) or not {
                    "id",
                    "family",
                    "level",
                    "prompt",
                    "gold",
                }.issubset(item):
                    raise ValueError("confirmation regenerated atom is malformed")
                expected_atoms.append(
                    {
                        "id": item["id"],
                        "family": item["family"],
                        "level": item["level"],
                        "prompt": item["prompt"],
                        "gold": item["gold"],
                        "answer_domain": item.get("answer_domain"),
                    }
                )
    actual_atoms = []
    for row in atom_rows:
        if not isinstance(row, Mapping) or not {
            "id",
            "family",
            "level",
            "prompt",
            "gold",
            "answer_domain",
        }.issubset(row):
            raise ValueError("confirmation raw atom task is malformed")
        actual_atoms.append(
            {
                key: row[key]
                for key in (
                    "id",
                    "family",
                    "level",
                    "prompt",
                    "gold",
                    "answer_domain",
                )
            }
        )
    if not _strict_json_equal(actual_atoms, expected_atoms):
        raise ValueError(
            "confirmation atom tasks differ from the preregistered block draw"
        )

    rollout_count = 1 if decode == "greedy" else expected_k
    expected_episodes = []
    for family_index, family_name in enumerate(families):
        family = load_family(family_name)
        if not getattr(family, "HAS_EPISODES", False):
            continue
        for raw_level in strata["deep_episode_levels"]:
            level = int(raw_level)
            if level not in family.LEVELS:
                continue
            for index in range(episode_n):
                seed = (
                    block_seed
                    + 50_000_000
                    + family_index * 100_000
                    + level * 1_000
                    + index
                )
                for rollout in range(rollout_count):
                    episode = family.Episode(seed, level)
                    rid = f"{family_name}-L{level}-e{seed}-r{rollout}"
                    expected_episodes.append(
                        {
                            "rid": rid,
                            "family": family_name,
                            "level": level,
                            "ep_seed": seed,
                            "rollout": rollout,
                            "spec": episode.spec,
                            "system_prompt": episode.system_prompt(),
                            "initial_observation": episode.initial_observation(),
                        }
                    )
    actual_episodes = []
    episode_fields = (
        "rid",
        "family",
        "level",
        "ep_seed",
        "rollout",
        "spec",
        "system_prompt",
        "initial_observation",
    )
    for row in episode_rows:
        if not isinstance(row, Mapping) or not set(episode_fields).issubset(row):
            raise ValueError("confirmation raw episode task is malformed")
        actual_episodes.append({key: row[key] for key in episode_fields})
    if not _strict_json_equal(actual_episodes, expected_episodes):
        raise ValueError(
            "confirmation episode tasks differ from the preregistered block draw"
        )


def _sampled_token_evidence(row: Mapping, *, label: str) -> int:
    first = row.get("stage1_sampled_token_ids")
    second = row.get("stage2_sampled_token_ids")
    for stage, values in (("stage1", first), ("stage2", second)):
        if not isinstance(values, list) or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in values
        ):
            raise ValueError(f"confirmation {label} {stage} token evidence is invalid")
    recomputed = len(first) + len(second)
    recorded = _plain_int(row.get("n_sampled_tokens"), label=f"{label} sampled tokens")
    if recorded != recomputed:
        raise ValueError(f"confirmation {label} sampled-token count is unauthenticated")
    return recomputed


AUTHENTICATED_FIELDS = {
    "task_manifest_sha256",
    "ordered_plan_sha256",
    "token_ledger",
    "backend_protocol",
    "backend_fingerprint",
    "sampling_protocol",
    "engine_protocol",
}


def _authenticated_payload(
    payload: Mapping, *, atom_rows: list[dict], episode_rows: list[dict]
) -> dict:
    """Derive redundant score fields only after GENERATED bytes are durable."""

    derived: dict[str, Any] = confirmation_task_hashes(atom_rows, episode_rows)
    sampled_tokens = 0
    for row in atom_rows:
        for output in row.get("outputs") or []:
            sampled_tokens += _sampled_token_evidence(output, label="raw atom output")
    for row in episode_rows:
        turns = row.get("turns")
        if not isinstance(turns, list):
            raise ValueError("confirmation raw episode turns are malformed")
        for turn in turns:
            if not isinstance(turn, Mapping):
                raise ValueError("confirmation raw episode turn is malformed")
            sampled_tokens += _sampled_token_evidence(turn, label="raw episode turn")
    derived["token_ledger"] = {"sampled_tokens": sampled_tokens}
    if payload.get("stage") == "policy_eval":
        protocol, fingerprint = canonical_backend_protocol(
            payload.get("runner_summary"), expected_model=str(payload.get("model", ""))
        )
        summaries = payload.get("runner_summary") or []
        sampling_protocol = canonical_confirmation_sampling_protocol(summaries)
        model = payload.get("model")
        model_config = payload.get("model_config_sha256")
        exact_model = bool(summaries) and all(
            summary.get("model") == model
            and summary.get("model_config_sha256") == model_config
            for summary in summaries
        )
        capacity = bool(summaries)
        for summary in summaries:
            try:
                validate_capacity_receipt(summary.get("confirmation_capacity"))
            except ValueError:
                capacity = False
        derived.update(
            {
                "backend_protocol": protocol,
                "backend_fingerprint": fingerprint,
                "sampling_protocol": sampling_protocol,
                "engine_protocol": {
                    "canonical_backend_authenticated": True,
                    "exact_local_model": exact_model,
                    "hybrid_capacity_no_preemption": capacity,
                },
            }
        )
    for key, value in derived.items():
        if key in payload and payload.get(key) != value:
            raise ValueError(f"confirmation authenticated field is stale: {key}")
    return {**dict(payload), **derived}


def _validate_token_and_runner_accounting(
    payload: Mapping, *, atom_rows: list[dict], episode_rows: list[dict]
) -> None:
    sampled_tokens = 0
    raw_requests = len(atom_rows)
    raw_completions = 0
    for row in atom_rows:
        for output in row["outputs"]:
            sampled_tokens += _sampled_token_evidence(output, label="raw atom output")
            raw_completions += 1
    for row in episode_rows:
        turns = row.get("turns")
        if not isinstance(turns, list):
            raise ValueError("confirmation raw episode turns are malformed")
        for turn in turns:
            if not isinstance(turn, Mapping):
                raise ValueError("confirmation raw episode turn is malformed")
            sampled_tokens += _sampled_token_evidence(turn, label="raw episode turn")
            raw_requests += 1
            raw_completions += 1
    ledger = payload.get("token_ledger")
    if not isinstance(ledger, Mapping) or set(ledger) != {"sampled_tokens"}:
        raise ValueError("confirmation sampled-token ledger is missing or malformed")
    if _plain_int(ledger.get("sampled_tokens"), label="sampled-token ledger") != sampled_tokens:
        raise ValueError("confirmation sampled-token ledger disagrees with raw evidence")
    summaries = payload.get("runner_summary")
    if not isinstance(summaries, list) or not summaries:
        raise ValueError("confirmation runner summary ledger is missing")
    summary_tokens = 0
    summary_requests = 0
    summary_completions = 0
    for summary in summaries:
        if not isinstance(summary, Mapping):
            raise ValueError("confirmation runner summary is malformed")
        counts = summary.get("counts")
        if not isinstance(counts, Mapping):
            raise ValueError("confirmation runner count summary is malformed")
        summary_tokens += _plain_int(
            counts.get("sampled_tokens"), label="runner sampled-token count"
        )
        summary_requests += _plain_int(
            counts.get("requests"), label="runner request count"
        )
        summary_completions += _plain_int(
            counts.get("completions"), label="runner completion count"
        )
    if (summary_tokens, summary_requests, summary_completions) != (
        sampled_tokens,
        raw_requests,
        raw_completions,
    ):
        raise ValueError("confirmation runner counts disagree with exact raw evidence")

    if payload.get("stage") == "policy_eval":
        model = payload.get("model")
        model_config = payload.get("model_config_sha256")
        if not isinstance(model, str) or not model or not isinstance(model_config, str):
            raise ValueError("confirmation policy model provenance is missing")
        if any(
            summary.get("model") != model
            or summary.get("model_config_sha256") != model_config
            for summary in summaries
        ):
            raise ValueError("confirmation runner model provenance disagrees")
        protocol, fingerprint = canonical_backend_protocol(
            summaries, expected_model=str(payload.get("model", ""))
        )
        if payload.get("backend_protocol") != protocol:
            raise ValueError("confirmation canonical backend protocol is stale")
        if payload.get("backend_fingerprint") != fingerprint:
            raise ValueError("confirmation canonical backend fingerprint is stale")
        sampling_protocol = canonical_confirmation_sampling_protocol(summaries)
        if payload.get("sampling_protocol") != sampling_protocol:
            raise ValueError("confirmation canonical sampling protocol is stale")
        for summary in summaries:
            validate_capacity_receipt(summary.get("confirmation_capacity"))
        recorded_authorization = payload.get("controls_authorization")
        recorded_admission = payload.get("confirmation_admission")
        config_sha256 = payload.get("config_sha256")
        if (
            not isinstance(recorded_authorization, Mapping)
            or not isinstance(recorded_admission, Mapping)
            or not isinstance(config_sha256, str)
        ):
            raise ValueError("confirmation score lacks pre-admission provenance")
        authorization = controls_authorization_binding(
            Path(str(recorded_authorization.get("path", ""))),
            expected_config_sha256=config_sha256,
        )
        expected_model = {
            "model": model,
            "model_merge_receipt_sha256": payload.get(
                "model_merge_receipt_sha256"
            ),
            "model_config_sha256": model_config,
            "model_inference_inventory_sha256": payload.get(
                "model_inference_inventory_sha256"
            ),
            "decode": payload.get("decode"),
        }
        admission = confirmation_admission_binding(
            Path(str(recorded_admission.get("path", ""))),
            expected_config_sha256=config_sha256,
            expected_controls_authorization=authorization,
            expected_tag=str(payload.get("tag", "")),
            expected_block_seed=_plain_int(
                payload.get("block_seed"), label="confirmation block seed"
            ),
            expected_model=expected_model,
        )
        if dict(recorded_authorization) != authorization or dict(
            recorded_admission
        ) != admission:
            raise ValueError("confirmation score pre-admission provenance is stale")


JOURNAL_TO_RAW_FIELDS = (
    "sample_index",
    "text",
    "n_thinking_tokens",
    "n_answer_tokens",
    "n_sampled_tokens",
    "thinking_closed",
    "forced_close",
    "finish_reason",
    "truncated",
    "token_ids",
    "retained_thinking_token_ids",
    "injected_token_ids",
)


def _journal_to_raw_projection(output: Mapping, *, label: str) -> dict[str, Any]:
    if any(field not in output for field in JOURNAL_TO_RAW_FIELDS):
        raise ValueError(f"confirmation {label} lacks a scored runner field")
    projection = {field: output[field] for field in JOURNAL_TO_RAW_FIELDS}
    _plain_int(projection["sample_index"], label=f"{label} sample index")
    for field in ("n_thinking_tokens", "n_answer_tokens", "n_sampled_tokens"):
        _plain_int(projection[field], label=f"{label} {field}")
    if not isinstance(projection["text"], str):
        raise ValueError(f"confirmation {label} text is malformed")
    for field in (
        "token_ids",
        "retained_thinking_token_ids",
        "injected_token_ids",
    ):
        values = projection[field]
        if not isinstance(values, list) or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in values
        ):
            raise ValueError(f"confirmation {label} {field} is malformed")
    for field in ("thinking_closed", "forced_close", "truncated"):
        if not isinstance(projection[field], bool):
            raise ValueError(f"confirmation {label} {field} is malformed")
    if projection["finish_reason"] is not None and not isinstance(
        projection["finish_reason"], str
    ):
        raise ValueError(f"confirmation {label} finish reason is malformed")
    return projection


def _recompute_atom_semantics(row: Mapping, output: Mapping) -> None:
    family_name = row.get("family")
    if not isinstance(family_name, str) or not family_name:
        raise ValueError("confirmation atom family is malformed")
    task = {
        "id": row.get("id"),
        "family": family_name,
        "level": _plain_int(row.get("level"), label="raw atom level"),
        "prompt": row.get("prompt"),
        "gold": row.get("gold"),
        "answer_domain": row.get("answer_domain"),
    }
    family = load_family(family_name)
    text = output["text"]
    expected_score = _finite_score(
        float(family.score_atom(task, text)), label="recomputed atom score"
    )
    observed_score = _finite_score(output.get("score"), label="raw atom score")
    if observed_score != expected_score:
        raise ValueError("confirmation atom score differs from journaled model text")
    if output.get("answer_value") != base.extract_answer(text):
        raise ValueError("confirmation atom answer extraction differs from journaled text")


def _recompute_episode_semantics(row: Mapping) -> None:
    family_name = row.get("family")
    if not isinstance(family_name, str) or not family_name:
        raise ValueError("confirmation episode family is malformed")
    level = _plain_int(row.get("level"), label="raw episode level")
    seed = _plain_int(row.get("ep_seed"), label="raw episode seed")
    rollout = _plain_int(row.get("rollout"), label="raw episode rollout")
    expected_rid = f"{family_name}-L{level}-e{seed}-r{rollout}"
    if row.get("rid") != expected_rid:
        raise ValueError("confirmation episode rollout identity is malformed")
    family = load_family(family_name)
    episode = family.Episode(seed, level)
    if (
        row.get("spec") != episode.spec
        or row.get("system_prompt") != episode.system_prompt()
        or row.get("initial_observation") != episode.initial_observation()
    ):
        raise ValueError("confirmation episode task differs from simulator replay")
    turns = row.get("turns")
    if not isinstance(turns, list):
        raise ValueError("confirmation episode turn ledger is malformed")
    done = False
    for index, turn in enumerate(turns):
        if not isinstance(turn, Mapping) or done:
            raise ValueError("confirmation episode has a turn after termination")
        if _plain_int(turn.get("turn"), label="raw episode turn") != index:
            raise ValueError("confirmation episode turns are not contiguous")
        action = base.extract_action(turn.get("text"))
        observation, done = episode.step(action)
        if (
            turn.get("action") != action
            or turn.get("action_ok")
            is not bool(getattr(episode, "last_action_ok", True))
            or turn.get("observation") != observation
            or turn.get("context_messages") != 2 + 2 * index
        ):
            raise ValueError(
                "confirmation episode transition differs from journaled model text"
            )
    maximum = _plain_int(episode.max_turns, label="episode maximum turns")
    if not done and len(turns) != maximum:
        raise ValueError("confirmation episode stopped before termination or horizon")
    if (
        row.get("done") is not done
        or row.get("n_turns") != len(turns)
        or row.get("max_turns") != maximum
        or _finite_score(row.get("score"), label="raw episode score")
        != _finite_score(float(episode.score()), label="recomputed episode score")
    ):
        raise ValueError("confirmation episode outcome differs from simulator replay")


REQUEST_EVIDENCE_FIELDS = {
    "schema_version",
    "id",
    "record_sha256",
    "prompt_token_ids_sha256",
    "prompt_sha256",
    "n_prompt_tokens",
    "prompt_channel",
}


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _expected_confirmation_request_calls(
    atom_rows: list[dict], episode_rows: list[dict]
) -> list[list[dict[str, str]]]:
    """Rebuild the exact harness record bytes and per-call request order."""

    atom_call = []
    for row in atom_rows:
        record_id = row.get("id")
        prompt = row.get("prompt")
        if not isinstance(record_id, str) or not isinstance(prompt, str):
            raise ValueError("confirmation atom request bytes are malformed")
        record = {
            "id": record_id,
            "messages": [{"role": "user", "content": prompt}],
        }
        atom_call.append(
            {"id": record_id, "record_sha256": canonical_hash(record)}
        )
    if not atom_call:
        raise ValueError("confirmation request proof has no atom call")

    maximum_turns = max(
        (len(row.get("turns") or []) for row in episode_rows), default=0
    )
    calls = [atom_call]
    for turn_index in range(maximum_turns):
        turn_call = []
        for row in episode_rows:
            rid = row.get("rid")
            system = row.get("system_prompt")
            initial = row.get("initial_observation")
            turns = row.get("turns")
            if (
                not isinstance(rid, str)
                or not isinstance(system, str)
                or not isinstance(initial, str)
                or not isinstance(turns, list)
            ):
                raise ValueError("confirmation episode request bytes are malformed")
            if turn_index >= len(turns):
                continue
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": initial},
            ]
            for prior_index, prior in enumerate(turns[:turn_index]):
                if (
                    not isinstance(prior, Mapping)
                    or prior.get("turn") != prior_index
                    or not isinstance(prior.get("action"), str)
                    or not isinstance(prior.get("observation"), str)
                ):
                    raise ValueError(
                        "confirmation prior episode conversation is malformed"
                    )
                messages.extend(
                    (
                        {"role": "assistant", "content": prior["action"]},
                        {"role": "user", "content": prior["observation"]},
                    )
                )
            record_id = f"{rid}-t{turn_index}"
            record = {"id": record_id, "messages": messages}
            turn_call.append(
                {"id": record_id, "record_sha256": canonical_hash(record)}
            )
        if not turn_call:
            raise ValueError("confirmation episode request call is unexpectedly empty")
        calls.append(turn_call)
    if len(calls) < 2:
        raise ValueError("confirmation request proof has no episode calls")
    return calls


def _validate_journal_call_geometry(
    summary: Any,
    *,
    request_evidence: list[Mapping[str, Any]],
    rows: list[Mapping[str, Any]],
) -> None:
    """Recompute one summary and capacity receipt from journaled requests."""

    if not isinstance(summary, Mapping):
        raise ValueError("confirmation call journal summary is malformed")
    sampling = summary.get("sampling")
    recorded_capacity = summary.get("confirmation_capacity")
    if not isinstance(sampling, Mapping) or not isinstance(
        recorded_capacity, Mapping
    ):
        raise ValueError("confirmation journal capacity evidence is malformed")
    validate_capacity_receipt(recorded_capacity)
    static = {
        "formula": recorded_capacity.get("formula"),
        "max_model_len": recorded_capacity.get("max_model_len"),
        "max_num_seqs": recorded_capacity.get("max_num_seqs"),
        "kv_cache_size_tokens": recorded_capacity.get("available_tokens"),
        "num_gpu_blocks": recorded_capacity.get("available_blocks"),
        "forced_close_tokens": len(PINNED_FORCED_CLOSE),
    }
    try:
        recomputed_capacity = capacity_receipt(
            static,
            prompt_token_lengths=[
                _plain_int(
                    row.get("n_prompt_tokens"),
                    label="journal prepared prompt tokens",
                    minimum=1,
                )
                for row in request_evidence
            ],
            sampling=SimpleNamespace(**dict(sampling)),
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            "confirmation journal capacity cannot be recomputed"
        ) from exc
    if dict(recorded_capacity) != recomputed_capacity:
        raise ValueError(
            "confirmation journal capacity differs from actual request geometry"
        )

    expected_n = _plain_int(
        sampling.get("n"), label="journal sampling multiplicity", minimum=1
    )
    counts = summary.get("counts")
    if not isinstance(counts, Mapping):
        raise ValueError("confirmation journal count summary is malformed")
    expected_counts = {
        "requests": len(rows),
        "completions": 0,
        "unique_input_prompt_tokens": sum(
            int(row["n_prompt_tokens"]) for row in request_evidence
        ),
        "stage1_logical_prompt_tokens": 0,
        "stage2_logical_prompt_tokens": 0,
        "logical_model_input_tokens": 0,
        "sampled_tokens": 0,
        "injected_tokens": 0,
    }
    for row in rows:
        outputs = row.get("outputs")
        if not isinstance(outputs, list) or len(outputs) != expected_n:
            raise ValueError(
                "confirmation journal completions differ from sampling multiplicity"
            )
        expected_counts["completions"] += len(outputs)
        for output in outputs:
            if not isinstance(output, Mapping):
                raise ValueError("confirmation call journal output is malformed")
            expected_counts["stage1_logical_prompt_tokens"] += _plain_int(
                output.get("n_stage1_prompt_tokens"),
                label="journal stage-one prompt tokens",
            )
            expected_counts["stage2_logical_prompt_tokens"] += _plain_int(
                output.get("n_stage2_prompt_tokens"),
                label="journal stage-two prompt tokens",
            )
            expected_counts["sampled_tokens"] += _plain_int(
                output.get("n_sampled_tokens"), label="journal sampled tokens"
            )
            expected_counts["injected_tokens"] += _plain_int(
                output.get("n_injected_tokens"), label="journal injected tokens"
            )
    expected_counts["logical_model_input_tokens"] = (
        expected_counts["stage1_logical_prompt_tokens"]
        + expected_counts["stage2_logical_prompt_tokens"]
    )
    if dict(counts) != expected_counts:
        raise ValueError(
            "confirmation journal summary counts differ from returned outputs"
        )


def _validate_call_journal(
    payload: Mapping,
    generated: Mapping,
    *,
    atom_rows: list[dict],
    episode_rows: list[dict],
) -> None:
    bundles = generated.get("call_bundles")
    if not isinstance(bundles, list):
        raise ValueError("confirmation call journal inventory is malformed")
    if payload.get("stage") != "policy_eval" and not bundles:
        return
    summaries = payload.get("runner_summary")
    if not isinstance(summaries, list) or len(bundles) != len(summaries):
        raise ValueError("confirmation call journal does not cover every generation call")
    expected_calls = _expected_confirmation_request_calls(atom_rows, episode_rows)
    if len(expected_calls) != len(bundles):
        raise ValueError("confirmation call journal request schedule is stale")
    evidence: dict[tuple[str, int], dict[str, Any]] = {}
    journal_summaries = []
    for call_index, descriptor in enumerate(bundles):
        if not isinstance(descriptor, Mapping):
            raise ValueError("confirmation call journal descriptor is malformed")
        rows = _read_gzip_jsonl(Path(str(descriptor.get("path", ""))))
        if len(rows) != 1 or set(rows[0]) != {
            "rows",
            "summary",
            "request_evidence",
        }:
            raise ValueError("confirmation call journal payload is malformed")
        bundle = rows[0]
        journal_summaries.append(bundle["summary"])
        if not isinstance(bundle["rows"], list):
            raise ValueError("confirmation call journal rows are malformed")
        request_evidence = bundle["request_evidence"]
        expected_call = expected_calls[call_index]
        if (
            not isinstance(request_evidence, list)
            or len(request_evidence) != len(expected_call)
            or len(bundle["rows"]) != len(expected_call)
            or any(not isinstance(row, Mapping) for row in bundle["rows"])
            or any(not isinstance(row, Mapping) for row in request_evidence)
        ):
            raise ValueError("confirmation call request evidence geometry is stale")
        expected_ids = [row["id"] for row in expected_call]
        if (
            [row.get("id") for row in bundle["rows"]] != expected_ids
            or [row.get("id") for row in request_evidence] != expected_ids
        ):
            raise ValueError("confirmation call request order differs from exact tasks")
        for row, pre_call, exact in zip(
            bundle["rows"], request_evidence, expected_call
        ):
            if not isinstance(row, Mapping) or not isinstance(row.get("id"), str):
                raise ValueError("confirmation call journal row is malformed")
            if (
                not isinstance(pre_call, Mapping)
                or set(pre_call) != REQUEST_EVIDENCE_FIELDS
                or pre_call.get("schema_version") != 1
                or pre_call.get("id") != row["id"]
                or pre_call.get("record_sha256") != exact["record_sha256"]
                or not _is_sha256(pre_call.get("record_sha256"))
                or not _is_sha256(pre_call.get("prompt_token_ids_sha256"))
                or not _is_sha256(pre_call.get("prompt_sha256"))
                or _plain_int(
                    pre_call.get("n_prompt_tokens"),
                    label="journal prepared prompt tokens",
                    minimum=1,
                )
                != row.get("n_prompt_tokens")
                or not isinstance(pre_call.get("prompt_channel"), str)
                or not pre_call["prompt_channel"]
                or row.get("prompt_sha256") != pre_call["prompt_sha256"]
                or row.get("prompt_channel") != pre_call["prompt_channel"]
            ):
                raise ValueError(
                    "confirmation journal request differs from task/prepared prompt"
                )
            request_sha256 = canonical_hash(
                {key: value for key, value in row.items() if key != "outputs"}
            )
            outputs = row.get("outputs")
            if not isinstance(outputs, list) or not outputs:
                raise ValueError("confirmation call journal row has no outputs")
            for output in outputs:
                if not isinstance(output, Mapping):
                    raise ValueError("confirmation call journal output is malformed")
                key = (
                    row["id"],
                    _plain_int(
                        output.get("sample_index"), label="journal sample index"
                    ),
                )
                first = list(output.get("stage1_token_ids") or [])
                second = list(output.get("stage2_token_ids") or [])
                count = _plain_int(
                    output.get("n_sampled_tokens"), label="journal sampled tokens"
                )
                if count != len(first) + len(second) or key in evidence:
                    raise ValueError("confirmation call journal token evidence is invalid")
                evidence[key] = {
                    "stage1_sampled_token_ids": first,
                    "stage2_sampled_token_ids": second,
                    "n_sampled_tokens": count,
                    "generation_request_sha256": request_sha256,
                    "generation_output_sha256": canonical_hash(output),
                    "generation_record_sha256": pre_call["record_sha256"],
                    "generation_prompt_token_ids_sha256": pre_call[
                        "prompt_token_ids_sha256"
                    ],
                    "scored_output": _journal_to_raw_projection(
                        output, label="call journal output"
                    ),
                }
        _validate_journal_call_geometry(
            bundle["summary"],
            request_evidence=request_evidence,
            rows=bundle["rows"],
        )
    if journal_summaries != summaries:
        raise ValueError("confirmation call journal summaries differ from scored payload")
    consumed: set[tuple[str, int]] = set()
    for row in atom_rows:
        for output in row["outputs"]:
            key = (str(row["id"]), int(output["sample_index"]))
            expected = evidence.get(key)
            observed = {
                "stage1_sampled_token_ids": output.get(
                    "stage1_sampled_token_ids"
                ),
                "stage2_sampled_token_ids": output.get(
                    "stage2_sampled_token_ids"
                ),
                "n_sampled_tokens": output.get("n_sampled_tokens"),
                "generation_request_sha256": output.get(
                    "generation_request_sha256"
                ),
                "generation_output_sha256": output.get("generation_output_sha256"),
                "generation_record_sha256": output.get("generation_record_sha256"),
                "generation_prompt_token_ids_sha256": output.get(
                    "generation_prompt_token_ids_sha256"
                ),
                "scored_output": _journal_to_raw_projection(
                    output, label="raw atom output"
                ),
            }
            if expected != observed:
                raise ValueError("confirmation atom output differs from call journal")
            _recompute_atom_semantics(row, output)
            consumed.add(key)
    for row in episode_rows:
        for turn in row["turns"]:
            key = (f"{row['rid']}-t{int(turn['turn'])}", 0)
            expected = evidence.get(key)
            observed = {
                "stage1_sampled_token_ids": turn.get("stage1_sampled_token_ids"),
                "stage2_sampled_token_ids": turn.get("stage2_sampled_token_ids"),
                "n_sampled_tokens": turn.get("n_sampled_tokens"),
                "generation_request_sha256": turn.get(
                    "generation_request_sha256"
                ),
                "generation_output_sha256": turn.get("generation_output_sha256"),
                "generation_record_sha256": turn.get("generation_record_sha256"),
                "generation_prompt_token_ids_sha256": turn.get(
                    "generation_prompt_token_ids_sha256"
                ),
                "scored_output": _journal_to_raw_projection(
                    turn, label="raw episode turn"
                ),
            }
            if expected != observed:
                raise ValueError("confirmation episode output differs from call journal")
            consumed.add(key)
        _recompute_episode_semantics(row)
    if consumed != set(evidence):
        raise ValueError("confirmation call journal evidence was not consumed exactly")


def _validate_policy_started_context(
    context: Mapping, *, expected_tag: str
) -> None:
    """Authenticate the complete policy admission before STARTED is published."""

    if set(context) != POLICY_STARTED_CONTEXT_FIELDS:
        raise ValueError("confirmation policy STARTED context is incomplete")
    sha_fields = {
        "config_sha256",
        "evaluator_sha256",
        "model_merge_receipt_sha256",
        "model_config_sha256",
        "model_inference_inventory_sha256",
        "task_manifest_sha256",
        "ordered_plan_sha256",
    }
    decode = context.get("decode")
    expected_k = 1 if decode == "greedy" else 8 if decode == "sample8" else None
    source = context.get("evaluator_source_inventory")
    sampling = context.get("sampling_protocol")
    if (
        context.get("stage") != "policy_eval"
        or context.get("tag") != expected_tag
        or context.get("scope") != "confirmatory"
        or not isinstance(context.get("block_seed"), int)
        or isinstance(context.get("block_seed"), bool)
        or expected_k is None
        or context.get("k") != expected_k
        or any(not _is_sha256(context.get(key)) for key in sha_fields)
        or context.get("evaluator_sha256")
        != sha256_file(EXP / "scripts" / "eval_policy.py")
        or not isinstance(context.get("model"), str)
        or not context.get("model")
        or not isinstance(source, Mapping)
        or set(source) != {"sha256", "file_count"}
        or dict(source) != confirmation_evaluator_source_inventory()
        or not isinstance(sampling, Mapping)
        or set(sampling) != {"schema_version", "atom", "episode"}
        or sampling.get("schema_version") != 1
    ):
        raise ValueError("confirmation policy STARTED context is stale")
    validate_live_cache_capacity(context["capacity_preflight"])

    authorization = context.get("controls_authorization")
    admission = context.get("confirmation_admission")
    if not isinstance(authorization, Mapping) or not isinstance(admission, Mapping):
        raise ValueError("confirmation policy STARTED admission is malformed")
    try:
        authorization_now = controls_authorization_binding(
            Path(str(authorization.get("path", ""))),
            expected_config_sha256=str(context["config_sha256"]),
        )
        expected_model = {
            "model": context["model"],
            "model_merge_receipt_sha256": context[
                "model_merge_receipt_sha256"
            ],
            "model_config_sha256": context["model_config_sha256"],
            "model_inference_inventory_sha256": context[
                "model_inference_inventory_sha256"
            ],
            "decode": decode,
        }
        admission_now = confirmation_admission_binding(
            Path(str(admission.get("path", ""))),
            expected_config_sha256=str(context["config_sha256"]),
            expected_controls_authorization=authorization_now,
            expected_tag=expected_tag,
            expected_block_seed=int(context["block_seed"]),
            expected_model=expected_model,
        )
    except (OSError, ValueError) as exc:
        raise ValueError("confirmation policy STARTED admission is stale") from exc
    if dict(authorization) != authorization_now or dict(admission) != admission_now:
        raise ValueError("confirmation policy STARTED admission changed")


def _validate_started_context(candidate: Mapping, context: Any) -> None:
    """Require a complete immutable policy admission before COMPLETE."""

    if not isinstance(context, Mapping):
        raise ValueError("confirmation STARTED context is malformed")
    if candidate.get("stage") == "policy_eval":
        _validate_policy_started_context(
            context, expected_tag=str(candidate.get("tag", ""))
        )
        if any(
            context.get(key) != candidate.get(key)
            for key in POLICY_STARTED_DIRECT_BINDINGS
        ):
            raise ValueError(
                "confirmation GENERATED bytes disagree with STARTED admission"
            )
        source = context["evaluator_source_inventory"]
        if (
            not isinstance(source, Mapping)
            or set(source) != {"sha256", "file_count"}
            or dict(source) != confirmation_evaluator_source_inventory()
            or source.get("sha256")
            != candidate.get("evaluator_source_inventory_sha256")
            or source.get("file_count")
            != candidate.get("evaluator_source_file_count")
        ):
            raise ValueError("confirmation evaluator source changed after STARTED")
        validate_live_cache_capacity(context["capacity_preflight"])
        return

    # Test-only/generic transactions retain the historical optional bindings;
    # production policy transactions never enter this branch.
    for key in POLICY_STARTED_DIRECT_BINDINGS:
        if key in context and context.get(key) != candidate.get(key):
            raise ValueError(
                "confirmation GENERATED bytes disagree with STARTED admission"
            )
    source = context.get("evaluator_source_inventory")
    if source is not None and (
        not isinstance(source, Mapping)
        or source.get("sha256")
        != candidate.get("evaluator_source_inventory_sha256")
        or source.get("file_count") != candidate.get("evaluator_source_file_count")
    ):
        raise ValueError("confirmation evaluator source changed after STARTED")
    if "capacity_preflight" in context:
        validate_live_cache_capacity(context["capacity_preflight"])


def _validate_capacity_against_started(
    payload: Mapping, started_context: Mapping
) -> None:
    if payload.get("stage") != "policy_eval":
        return
    preflight = started_context.get("capacity_preflight")
    validate_live_cache_capacity(preflight)
    for summary in payload.get("runner_summary") or []:
        receipt = summary.get("confirmation_capacity")
        validate_capacity_receipt(receipt)
        if (
            receipt.get("available_tokens")
            != preflight.get("kv_cache_size_tokens")
            or receipt.get("available_blocks") != preflight.get("num_gpu_blocks")
            or receipt.get("max_model_len") != preflight.get("max_model_len")
            or receipt.get("max_num_seqs") != preflight.get("max_num_seqs")
            or receipt.get("formula") != preflight.get("formula")
        ):
            raise ValueError(
                "confirmation generation capacity differs from STARTED preflight"
            )


def _raw_item_projection(
    atom_rows: list[dict], episode_rows: list[dict]
) -> list[dict]:
    projected = []
    for row in atom_rows:
        key = row.get("id")
        family = row.get("family")
        outputs = row.get("outputs")
        if not isinstance(key, str) or not key:
            raise ValueError("confirmation raw atom has invalid id")
        if not isinstance(family, str) or not family:
            raise ValueError("confirmation raw atom has invalid family")
        if not isinstance(outputs, list) or not outputs:
            raise ValueError("confirmation raw atom has no outputs")
        normalized_outputs = []
        for output in outputs:
            if not isinstance(output, Mapping):
                raise ValueError("confirmation raw atom output is not an object")
            normalized_outputs.append(
                (
                    _finite_score(output.get("score"), label="raw atom score"),
                    _plain_int(
                        output.get("sample_index"), label="raw atom sample index"
                    ),
                )
            )
        sample_indexes = [index for _, index in normalized_outputs]
        if sorted(sample_indexes) != list(range(len(outputs))):
            raise ValueError("confirmation raw atom sample indexes are not contiguous")
        best_score, _ = max(
            normalized_outputs, key=lambda value: (value[0], -value[1])
        )
        projected.append(
            {
                "key": key,
                "family": family,
                "kind": "atom",
                "level": _plain_int(row.get("level"), label="raw atom level"),
                "score": best_score,
                "samples": len(outputs),
            }
        )

    grouped: dict[tuple[str, int, int], list[tuple[float, int]]] = defaultdict(list)
    seen_rollouts = set()
    for row in episode_rows:
        family = row.get("family")
        if not isinstance(family, str) or not family:
            raise ValueError("confirmation raw episode has invalid family")
        level = _plain_int(row.get("level"), label="raw episode level")
        seed = _plain_int(row.get("ep_seed"), label="raw episode seed")
        rollout = _plain_int(row.get("rollout"), label="raw episode rollout")
        identity = (family, level, seed, rollout)
        if identity in seen_rollouts:
            raise ValueError("confirmation raw episode has duplicate rollout")
        seen_rollouts.add(identity)
        grouped[(family, level, seed)].append(
            (_finite_score(row.get("score"), label="raw episode score"), rollout)
        )
    for (family, level, seed), outputs in grouped.items():
        if sorted(rollout for _, rollout in outputs) != list(range(len(outputs))):
            raise ValueError("confirmation raw episode rollouts are not contiguous")
        best_score, _ = max(outputs, key=lambda value: (value[0], -value[1]))
        projected.append(
            {
                "key": f"{family}/episode/L{level}/s{seed}",
                "family": family,
                "kind": "episode",
                "level": level,
                "score": best_score,
                "samples": len(outputs),
            }
        )
    projected.sort(key=lambda row: row["key"])
    keys = [row["key"] for row in projected]
    if len(keys) != len(set(keys)):
        raise ValueError("confirmation raw rows produce duplicate item keys")
    return projected


def _validate_raw_item_semantics(
    payload: Mapping, *, atom_rows: list[dict], episode_rows: list[dict]
) -> None:
    if _raw_item_projection(atom_rows, episode_rows) != _score_item_projection(payload):
        raise ValueError("confirmation raw semantics do not match scored items")
    hashes = confirmation_task_hashes(atom_rows, episode_rows)
    if any(payload.get(key) != value for key, value in hashes.items()):
        raise ValueError("confirmation exact task manifest or ordered plan is stale")
    _validate_token_and_runner_accounting(
        payload, atom_rows=atom_rows, episode_rows=episode_rows
    )


def _expected_raw_row_counts(payload: Mapping) -> dict[str, int]:
    atom_rows = 0
    episode_rows = 0
    for item in _score_item_projection(payload):
        if item["kind"] == "atom":
            atom_rows += 1
        else:
            episode_rows += item["samples"]
    return {"atom_rows": atom_rows, "episode_rows": episode_rows}


def validate_confirmation_geometry(payload: Mapping, config: Mapping) -> None:
    """Require the complete frozen family/count/stratum confirmation geometry."""

    families = all_families(config)
    confirmation = config["confirmation"]
    atoms_per_level = int(confirmation["atoms_per_family_level"])
    episodes_per_level = int(confirmation["episodes_per_family_level"])
    decode = payload.get("decode")
    expected_k = (
        1
        if decode == "greedy"
        else int(config["controls"]["sample_more_k"])
        if decode == "sample8"
        else None
    )
    if (
        payload.get("scope") != "confirmatory"
        or payload.get("families") != families
        or payload.get("atoms_per_level") != atoms_per_level
        or payload.get("episodes_per_level") != episodes_per_level
        or expected_k is None
        or payload.get("k") != expected_k
    ):
        raise ValueError("confirmation score does not use the frozen full geometry")

    quick_levels = {int(value) for value in config["strata"]["quick_atom_levels"]}
    deep_levels = {int(value) for value in config["strata"]["deep_atom_levels"]}
    episode_levels = {
        int(value) for value in config["strata"]["deep_episode_levels"]
    }
    expected = Counter()
    for family_name in families:
        family = load_family(family_name)
        for level in sorted(quick_levels | deep_levels):
            if level in family.LEVELS:
                expected[(family_name, "atom", level)] = atoms_per_level
        if getattr(family, "HAS_EPISODES", False):
            for level in sorted(episode_levels):
                if level in family.LEVELS:
                    expected[(family_name, "episode", level)] = episodes_per_level

    _score_item_projection(payload)
    observed = Counter()
    for item in payload["items"]:
        family = item["family"]
        kind = item["kind"]
        level = int(item["level"])
        if item["samples"] != expected_k:
            raise ValueError("confirmation item sample count violates decode geometry")
        expected_stratum = (
            "quick" if kind == "atom" and level in quick_levels else "deep"
        )
        if item.get("stratum") != expected_stratum:
            raise ValueError("confirmation item stratum violates frozen geometry")
        observed[(family, kind, level)] += 1
    if observed != expected:
        raise ValueError("confirmation item cells do not match frozen geometry")


def _validate_raw_descriptors(
    descriptors: Any,
    *,
    raw_dir: Path,
    raw_root: Path,
    expected_row_counts: Mapping[str, int],
) -> dict[str, list[dict]]:
    if not isinstance(descriptors, dict) or set(descriptors) != set(RAW_FILENAMES):
        raise ValueError("confirmation raw artifact inventory is incomplete")
    loaded_raw_rows = {}
    for key, filename in RAW_FILENAMES.items():
        descriptor = descriptors[key]
        if not isinstance(descriptor, dict) or set(descriptor) != DESCRIPTOR_FIELDS:
            raise ValueError(f"confirmation raw descriptor is malformed: {key}")
        expected_path = raw_dir / filename
        recorded_path = descriptor.get("path")
        expected_hash = descriptor.get("sha256")
        expected_bytes = descriptor.get("bytes")
        expected_rows = descriptor.get("rows")
        if (
            not isinstance(recorded_path, str)
            or recorded_path != str(expected_path)
            or not expected_path.is_relative_to(raw_root)
            or not expected_path.is_file()
            or expected_path.is_symlink()
        ):
            raise ValueError(f"confirmation raw path is invalid: {key}")
        if expected_rows != expected_row_counts[key]:
            raise ValueError("confirmation raw rows do not match scored item geometry")
        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or any(char not in "0123456789abcdef" for char in expected_hash)
            or not isinstance(expected_bytes, int)
            or isinstance(expected_bytes, bool)
            or expected_bytes <= 0
            or not isinstance(expected_rows, int)
            or isinstance(expected_rows, bool)
            or expected_rows < 0
            or expected_path.stat().st_size != expected_bytes
            or sha256_file(expected_path) != expected_hash
        ):
            raise ValueError(f"confirmation raw descriptor is stale: {key}")
        loaded_raw_rows[key] = _read_gzip_jsonl(expected_path)
        if len(loaded_raw_rows[key]) != expected_rows:
            raise ValueError(f"confirmation raw descriptor is stale: {key}")
    return loaded_raw_rows


def validate_confirmation_score_artifacts(
    score_path: Path,
    *,
    expected_tag: str,
    score_root: Path = CONFIRMATION_SCORE_ROOT,
    raw_root: Path = CONFIRMATION_RAW_ROOT,
) -> dict:
    """Validate one committed score and every byte of its mirrored raw inputs."""

    score_path, score_root, raw_dir, raw_root = _score_layout(
        score_path, score_root=score_root, raw_root=raw_root
    )
    if not score_path.is_file() or score_path.is_symlink():
        raise ValueError("confirmation score commit marker is missing or unsafe")
    visible_entries = {entry.name for entry in score_path.parent.iterdir()}
    if visible_entries != {"scores.json"}:
        raise ValueError("unknown partials exist beside a committed confirmation score")
    try:
        payload = json.loads(score_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("confirmation score commit marker is invalid") from exc
    if not isinstance(payload, dict):
        raise ValueError("confirmation score commit marker is not an object")
    if payload.get("tag") != expected_tag:
        raise ValueError("confirmation score tag does not match its sealed arm")
    relative = score_path.relative_to(score_root)
    if expected_tag != f"{relative.parts[0]}_{relative.parts[1]}":
        raise ValueError("confirmation score tag does not match its filesystem arm")
    if payload.get("raw_artifact_schema_version") != 2:
        raise ValueError("confirmation raw artifact schema is missing or stale")
    expected_row_counts = _expected_raw_row_counts(payload)
    descriptors = payload.get("raw_artifacts")
    if not raw_dir.is_dir() or raw_dir.is_symlink():
        raise ValueError("confirmation raw artifact directory is missing or unsafe")
    started, generated = _load_generated(raw_dir, score_path=score_path)
    transaction = payload.get("confirmation_transaction")
    expected_transaction = {
        "schema_version": 1,
        "started_sha256": sha256_file(raw_dir / MARKER_FILENAMES["started"]),
        "generated_sha256": sha256_file(raw_dir / MARKER_FILENAMES["generated"]),
        "call_bundle_count": len(generated["call_bundles"]),
    }
    if transaction != expected_transaction or generated.get("raw_artifacts") != descriptors:
        raise ValueError("confirmation transaction provenance is stale")
    complete = _read_json_object(
        raw_dir / MARKER_FILENAMES["complete"], label="COMPLETE marker"
    )
    if complete != {
        "schema_version": 1,
        "state": "COMPLETE",
        "tag": expected_tag,
        "started_sha256": expected_transaction["started_sha256"],
        "generated_sha256": expected_transaction["generated_sha256"],
        "score_sha256": sha256_file(score_path),
        "raw_artifacts": descriptors,
        "task_manifest_sha256": payload.get("task_manifest_sha256"),
        "ordered_plan_sha256": payload.get("ordered_plan_sha256"),
    }:
        raise ValueError("confirmation COMPLETE marker is stale")
    if started.get("tag") != expected_tag:
        raise ValueError("confirmation STARTED marker tag is stale")
    expected_names = {
        *RAW_FILENAMES.values(),
        MARKER_FILENAMES["started"],
        MARKER_FILENAMES["generated"],
        MARKER_FILENAMES["complete"],
        *(f"{BUNDLE_PREFIX}{index:04d}{BUNDLE_SUFFIX}" for index in range(len(generated["call_bundles"]))),
    }
    if {entry.name for entry in raw_dir.iterdir()} != expected_names:
        raise ValueError("unknown or missing committed confirmation transaction artifacts")
    loaded_raw_rows = _validate_raw_descriptors(
        descriptors,
        raw_dir=raw_dir,
        raw_root=raw_root,
        expected_row_counts=expected_row_counts,
    )
    authenticated = _authenticated_payload(
        generated["candidate_payload"],
        atom_rows=loaded_raw_rows["atom_rows"],
        episode_rows=loaded_raw_rows["episode_rows"],
    )
    scored_candidate = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "raw_artifact_schema_version",
            "raw_artifacts",
            "confirmation_transaction",
        }
    }
    if authenticated != scored_candidate:
        raise ValueError("confirmation score differs from authenticated GENERATED bytes")
    _validate_raw_item_semantics(
        payload,
        atom_rows=loaded_raw_rows["atom_rows"],
        episode_rows=loaded_raw_rows["episode_rows"],
    )
    _validate_call_journal(
        payload,
        generated,
        atom_rows=loaded_raw_rows["atom_rows"],
        episode_rows=loaded_raw_rows["episode_rows"],
    )
    _validate_started_context(payload, started["context"])
    _validate_capacity_against_started(payload, started["context"])
    _validate_preregistered_confirmation_tasks(
        payload,
        atom_rows=loaded_raw_rows["atom_rows"],
        episode_rows=loaded_raw_rows["episode_rows"],
    )
    return payload
