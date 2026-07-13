"""Atomic, fail-closed storage for sealed confirmation artifacts."""

from __future__ import annotations

import gzip
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping

from gym.families import load as load_family
from io_utils import all_families, resolve_repo_path, sha256_file


EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
CONFIRMATION_SCORE_ROOT = (EXP / "runs" / "confirmation").resolve()
CONFIRMATION_RAW_ROOT = (
    REPO / "large_artifacts" / EXP.name / "confirmation"
).resolve()
RAW_FILENAMES = {
    "atom_rows": "atom_rows.jsonl.gz",
    "episode_rows": "episode_rows.jsonl.gz",
}
DESCRIPTOR_FIELDS = {"path", "sha256", "bytes", "rows"}


def configured_confirmation_raw_root(config: Mapping) -> Path:
    """Resolve the config-bound external root for sealed confirmation rows."""

    try:
        artifacts_root = config["model"]["artifacts_root"]
    except (KeyError, TypeError) as exc:
        raise ValueError("confirmation config lacks model.artifacts_root") from exc
    resolved = resolve_repo_path(artifacts_root).resolve()
    expected = (REPO / "large_artifacts" / EXP.name).resolve()
    if resolved != expected:
        raise ValueError("confirmation artifact root is not the frozen experiment root")
    return (resolved / "confirmation").resolve()


def _resolved_root(path: Path) -> Path:
    return Path(path).expanduser().resolve()


def _score_layout(
    score_path: Path,
    *,
    score_root: Path,
    raw_root: Path,
) -> tuple[Path, Path, Path, Path]:
    score_root = _resolved_root(score_root)
    raw_root = _resolved_root(raw_root)
    score_path = Path(
        os.path.abspath(os.fspath(Path(score_path).expanduser()))
    )
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
    if score_path.resolve() != score_path:
        raise ValueError("confirmation score path contains a symlink")
    raw_dir = raw_root / relative.parent
    if raw_dir.resolve() != raw_dir:
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


def _is_atomic_temp(name: str, final_name: str) -> bool:
    prefix = f".{final_name}."
    return name.startswith(prefix) and name.endswith(".tmp") and len(name) > len(
        prefix
    ) + len(".tmp")


def _known_score_orphan(name: str) -> bool:
    return (
        _is_atomic_temp(name, "scores.json")
        or name in RAW_FILENAMES.values()
        or any(_is_atomic_temp(name, value) for value in RAW_FILENAMES.values())
    )


def _known_raw_orphan(name: str) -> bool:
    return name in RAW_FILENAMES.values() or any(
        _is_atomic_temp(name, value) for value in RAW_FILENAMES.values()
    )


def _remove_known_orphans(directory: Path, predicate) -> None:
    if not directory.exists():
        if directory.is_symlink():
            raise ValueError("confirmation artifact directory is a broken symlink")
        return
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("confirmation artifact directory is not a real directory")
    entries = list(directory.iterdir())
    unknown = [entry.name for entry in entries if not predicate(entry.name)]
    if unknown:
        raise ValueError(
            "unknown partial confirmation artifacts: " + ", ".join(sorted(unknown))
        )
    for entry in entries:
        if entry.is_dir() and not entry.is_symlink():
            raise ValueError(f"confirmation orphan is unexpectedly a directory: {entry}")
        entry.unlink()


def prepare_confirmation_output(
    score_path: Path,
    *,
    score_root: Path = CONFIRMATION_SCORE_ROOT,
    raw_root: Path = CONFIRMATION_RAW_ROOT,
) -> Path:
    """Clear only recognized crash orphans before a new uncommitted evaluation."""

    score_path, _, raw_dir, _ = _score_layout(
        score_path, score_root=score_root, raw_root=raw_root
    )
    if score_path.exists() or score_path.is_symlink():
        raise ValueError("confirmation score commit marker already exists")
    _remove_known_orphans(score_path.parent, _known_score_orphan)
    _remove_known_orphans(raw_dir, _known_raw_orphan)
    score_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
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


def _write_json_atomic(path: Path, payload: Mapping) -> None:
    temporary_path: Path | None = None
    try:
        with _atomic_temp(path) as handle:
            temporary_path = Path(handle.name)
            encoded = (
                json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
            ).encode("utf-8")
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        _publish_no_clobber(temporary_path, path, label="score")
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def commit_confirmation_score(
    score_path: Path,
    payload: Mapping,
    *,
    atom_rows: Iterable[Mapping],
    episode_rows: Iterable[Mapping],
    score_root: Path = CONFIRMATION_SCORE_ROOT,
    raw_root: Path = CONFIRMATION_RAW_ROOT,
) -> dict:
    """Write both raw gzip artifacts, then atomically publish scores.json last."""

    if "raw_artifacts" in payload or "raw_artifact_schema_version" in payload:
        raise ValueError("caller must not supply confirmation raw descriptors")
    tag = payload.get("tag")
    if not isinstance(tag, str) or not tag:
        raise ValueError("confirmation score requires a non-empty tag")
    expected_rows = _expected_raw_row_counts(payload)
    score_path, _, raw_dir, _ = _score_layout(
        score_path, score_root=score_root, raw_root=raw_root
    )
    prepare_confirmation_output(
        score_path, score_root=score_root, raw_root=raw_root
    )
    descriptors = {
        "atom_rows": _write_gzip_jsonl_atomic(
            raw_dir / RAW_FILENAMES["atom_rows"], atom_rows
        ),
        "episode_rows": _write_gzip_jsonl_atomic(
            raw_dir / RAW_FILENAMES["episode_rows"], episode_rows
        ),
    }
    if any(
        descriptors[key]["rows"] != expected_rows[key]
        for key in RAW_FILENAMES
    ):
        raise ValueError("confirmation raw rows do not match scored item geometry")
    _validate_raw_item_semantics(
        payload,
        atom_rows=_read_gzip_jsonl(Path(descriptors["atom_rows"]["path"])),
        episode_rows=_read_gzip_jsonl(Path(descriptors["episode_rows"]["path"])),
    )
    committed = {
        **dict(payload),
        "raw_artifact_schema_version": 1,
        "raw_artifacts": descriptors,
    }
    _write_json_atomic(score_path, committed)
    return validate_confirmation_score_artifacts(
        score_path,
        expected_tag=tag,
        score_root=score_root,
        raw_root=raw_root,
    )


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
    if payload.get("raw_artifact_schema_version") != 1:
        raise ValueError("confirmation raw artifact schema is missing or stale")
    expected_row_counts = _expected_raw_row_counts(payload)
    descriptors = payload.get("raw_artifacts")
    if not isinstance(descriptors, dict) or set(descriptors) != set(RAW_FILENAMES):
        raise ValueError("confirmation raw artifact inventory is incomplete")
    if not raw_dir.is_dir() or raw_dir.is_symlink():
        raise ValueError("confirmation raw artifact directory is missing or unsafe")
    if {entry.name for entry in raw_dir.iterdir()} != set(RAW_FILENAMES.values()):
        raise ValueError("unknown or missing committed confirmation raw artifacts")

    loaded_raw_rows = {}
    for key, filename in RAW_FILENAMES.items():
        descriptor = descriptors[key]
        if not isinstance(descriptor, dict) or set(descriptor) != DESCRIPTOR_FIELDS:
            raise ValueError(f"confirmation raw descriptor is malformed: {key}")
        expected_path = raw_dir / filename
        recorded_path = descriptor.get("path")
        if (
            not isinstance(recorded_path, str)
            or recorded_path != str(expected_path)
            or not expected_path.is_relative_to(raw_root)
            or not expected_path.is_file()
            or expected_path.is_symlink()
        ):
            raise ValueError(f"confirmation raw path is invalid: {key}")
        expected_hash = descriptor.get("sha256")
        expected_bytes = descriptor.get("bytes")
        expected_rows = descriptor.get("rows")
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
            or expected_rows != expected_row_counts[key]
            or expected_path.stat().st_size != expected_bytes
            or sha256_file(expected_path) != expected_hash
        ):
            raise ValueError(f"confirmation raw descriptor is stale: {key}")
        loaded_raw_rows[key] = _read_gzip_jsonl(expected_path)
        if len(loaded_raw_rows[key]) != expected_rows:
            raise ValueError(f"confirmation raw descriptor is stale: {key}")
    _validate_raw_item_semantics(
        payload,
        atom_rows=loaded_raw_rows["atom_rows"],
        episode_rows=loaded_raw_rows["episode_rows"],
    )
    return payload
