"""Durable attempt identities and filesystem-safe recovery primitives.

The scientific receipts in this experiment are immutable, but launch history is
necessarily append-only mutable state.  This module keeps that state outside a
result directory so deleting or losing an incomplete directory can never turn a
retry back into an indistinguishable first launch.

Only small JSON metadata is handled here.  The failed-attempt archiver owns the
large-tree copy transaction; consumers use :func:`validate_failed_archive` to
reopen the resulting immutable graph.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import stat
import tempfile
from contextlib import ExitStack, contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping, Sequence

from .safe_io import (
    StableArtifactError,
    open_stable_regular,
    publish_new_bytes,
    read_stable_bytes,
)


EXPERIMENT_ID = "qwen35_4b_state_formation_capacity_adjudication"
EXPERIMENT_RELATIVE = Path("experiments") / EXPERIMENT_ID
LARGE_RELATIVE = Path("large_artifacts") / EXPERIMENT_ID
ATTEMPT_MARKER_NAME = "attempt.json"


class AttemptReceiptError(RuntimeError):
    """An attempt journal, marker, or archive violates its closed contract."""


def canonical_sha256(payload: Any) -> str:
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AttemptReceiptError("attempt payload is not finite canonical JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _canonical_relative(value: str) -> PurePosixPath:
    if type(value) is not str or not value or "\\" in value or "\x00" in value:
        raise AttemptReceiptError("attempt path is not a canonical POSIX relative path")
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or pure.as_posix() != value
        or any(part in ("", ".", "..") for part in pure.parts)
    ):
        raise AttemptReceiptError(f"attempt path is not canonical: {value!r}")
    return pure


def canonical_root(repo_root: Path | str) -> Path:
    lexical = Path(os.path.abspath(os.fspath(repo_root)))
    if lexical.resolve(strict=True) != lexical or not lexical.is_dir():
        raise AttemptReceiptError("repository root is missing, aliased, or symlinked")
    return lexical


def canonical_path(
    repo_root: Path | str,
    relative: str,
    *,
    require_exists: bool = True,
) -> Path:
    root = canonical_root(repo_root)
    pure = _canonical_relative(relative)
    current = root
    for part in pure.parts:
        current /= part
        if not os.path.lexists(current):
            if require_exists:
                raise AttemptReceiptError(f"attempt path is missing: {relative}")
            continue
        mode = current.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise AttemptReceiptError(f"attempt path uses a symlink: {relative}")
    if require_exists and current.resolve(strict=True) != current:
        raise AttemptReceiptError(f"attempt path is aliased: {relative}")
    return current


def repo_relative(repo_root: Path | str, path: Path | str) -> str:
    root = canonical_root(repo_root)
    lexical = Path(os.path.abspath(os.fspath(path)))
    try:
        relative = lexical.relative_to(root).as_posix()
    except ValueError as exc:
        raise AttemptReceiptError(f"attempt path escapes repository: {lexical}") from exc
    canonical_path(root, relative, require_exists=False)
    return relative


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_file(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise AttemptReceiptError(f"attempt leaf is not a regular file: {path}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _open_regular(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode):
        os.close(descriptor)
        raise AttemptReceiptError(f"attempt leaf is not a regular file: {path}")
    return descriptor


def sha256_file(path: Path) -> str:
    descriptor = _open_regular(path)
    digest = hashlib.sha256()
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    descriptor = _open_regular(path)
    try:
        with os.fdopen(descriptor, "r", encoding="utf-8", closefd=False) as handle:
            payload = json.load(
                handle,
                object_pairs_hook=_reject_duplicates,
                parse_constant=_reject_constant,
            )
    finally:
        os.close(descriptor)
    if not isinstance(payload, dict):
        raise AttemptReceiptError(f"attempt JSON is not an object: {path}")
    return payload


def _json_object_from_bytes(raw: bytes, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, ValueError) as exc:
        raise AttemptReceiptError(f"{label} is not strict JSON") from exc
    if not isinstance(payload, dict):
        raise AttemptReceiptError(f"{label} is not a JSON object")
    return payload


def _ensure_directory(path: Path) -> None:
    missing: list[Path] = []
    current = path
    while not os.path.lexists(current):
        missing.append(current)
        current = current.parent
    if stat.S_ISLNK(current.lstat().st_mode) or not current.is_dir():
        raise AttemptReceiptError(f"attempt parent is unsafe: {current}")
    for directory in reversed(missing):
        directory.mkdir(exist_ok=False)
        fsync_directory(directory.parent)
    current = path
    while True:
        if stat.S_ISLNK(current.lstat().st_mode) or not current.is_dir():
            raise AttemptReceiptError(f"attempt directory is unsafe: {current}")
        if current.parent == current:
            break
        current = current.parent


def atomic_write_json(path: Path, payload: Mapping[str, Any], *, replace: bool) -> None:
    """Write finite JSON through a private regular staging inode and fsync it."""

    _ensure_directory(path.parent)
    if os.path.lexists(path):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise AttemptReceiptError(f"refusing an unsafe JSON destination: {path}")
        if not replace:
            raise AttemptReceiptError(f"refusing to overwrite immutable JSON: {path}")
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    if not replace:
        try:
            publish_new_bytes(path.parent, path, encoded)
        except StableArtifactError as exc:
            raise AttemptReceiptError(
                f"refusing to overwrite immutable JSON: {path}"
            ) from exc
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        if os.path.lexists(temporary):
            temporary.unlink()
            fsync_directory(temporary.parent)


def _stat_fingerprint(info: os.stat_result) -> tuple[int, ...]:
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_mode),
        int(info.st_nlink),
        int(info.st_size),
        int(info.st_mtime_ns),
        int(info.st_ctime_ns),
    )


def _directory_open_flags() -> int:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
    )
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


def _regular_open_flags(*, writable: bool = False) -> int:
    flags = (os.O_RDWR if writable else os.O_RDONLY) | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


def _open_absolute_directory_chain(path: Path) -> tuple[int, list[int]]:
    """Open an absolute directory one no-follow component at a time."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    descriptors: list[int] = []
    try:
        current = os.open("/", _directory_open_flags())
        descriptors.append(current)
        for component in absolute.parts[1:]:
            current = os.open(component, _directory_open_flags(), dir_fd=current)
            descriptors.append(current)
        return current, descriptors
    except Exception:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise


def _verify_absolute_directory_binding(
    path: Path, expected: os.stat_result
) -> None:
    rebound: list[int] = []
    try:
        descriptor, rebound = _open_absolute_directory_chain(path)
        observed = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(observed.st_mode)
            or (observed.st_dev, observed.st_ino)
            != (expected.st_dev, expected.st_ino)
        ):
            raise AttemptReceiptError(
                "attempt canonical directory binding changed while held"
            )
    except AttemptReceiptError:
        raise
    except OSError as exc:
        raise AttemptReceiptError(
            "attempt canonical directory binding changed while held"
        ) from exc
    finally:
        for held in reversed(rebound):
            os.close(held)


def _verify_lock_binding(
    parent_descriptor: int,
    name: str,
    expected: os.stat_result,
) -> None:
    try:
        observed = os.stat(
            name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
    except OSError as exc:
        raise AttemptReceiptError(
            "attempt lock pathname changed while the lock was held"
        ) from exc
    if (
        not stat.S_ISREG(observed.st_mode)
        or observed.st_nlink != 1
        or (observed.st_dev, observed.st_ino)
        != (expected.st_dev, expected.st_ino)
    ):
        raise AttemptReceiptError(
            "attempt lock pathname changed while the lock was held"
        )


@contextmanager
def locked_regular(path: Path) -> Iterator[None]:
    """Serialize cooperating writers and hold one canonically bound lock inode.

    Locking only the leaf is insufficient: an atomic replacement can leave one
    writer holding the old inode while a second writer locks the new pathname.
    Every cooperating writer therefore locks the held canonical parent first,
    opens the leaf relative to that descriptor, and proves both bindings again
    before reporting success.
    """

    absolute = Path(os.path.abspath(os.fspath(path)))
    _ensure_directory(absolute.parent)
    parent_descriptor: int | None = None
    parent_chain: list[int] = []
    descriptor: int | None = None
    parent_locked = False
    leaf_locked = False
    body_active = False
    try:
        parent_descriptor, parent_chain = _open_absolute_directory_chain(
            absolute.parent
        )
        fcntl.flock(parent_descriptor, fcntl.LOCK_EX)
        parent_locked = True
        _verify_absolute_directory_binding(
            absolute.parent, os.fstat(parent_descriptor)
        )
        try:
            os.stat(
                absolute.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            existed = True
        except FileNotFoundError:
            existed = False
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(
            absolute.name,
            flags,
            0o600,
            dir_fd=parent_descriptor,
        )
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise AttemptReceiptError("attempt lock must be one unaliased regular inode")
        if not existed:
            os.fsync(descriptor)
            os.fsync(parent_descriptor)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        leaf_locked = True
        _verify_lock_binding(parent_descriptor, absolute.name, info)
        body_active = True
        yield
        body_active = False
        _verify_lock_binding(parent_descriptor, absolute.name, info)
        _verify_absolute_directory_binding(absolute.parent, os.fstat(parent_descriptor))
    except AttemptReceiptError:
        raise
    except OSError as exc:
        if body_active:
            raise
        raise AttemptReceiptError(
            "attempt lock could not be opened through its canonical parent"
        ) from exc
    finally:
        if descriptor is not None:
            if leaf_locked:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
        if parent_descriptor is not None and parent_locked:
            fcntl.flock(parent_descriptor, fcntl.LOCK_UN)
        for held in reversed(parent_chain):
            os.close(held)


def _identity_checked(payload: Mapping[str, Any], field: str, label: str) -> None:
    claimed = payload.get(field)
    if type(claimed) is not str or len(claimed) != 64:
        raise AttemptReceiptError(f"{label} identity is malformed")
    expected = canonical_sha256({key: value for key, value in payload.items() if key != field})
    if claimed != expected:
        raise AttemptReceiptError(f"{label} identity mismatch")


def build_attempt_authorization(
    *,
    attempt_kind: str,
    attempt_index: int,
    cell: Mapping[str, Any],
    canonical_paths: Sequence[str],
    context: Mapping[str, Any],
    replay_archive: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if attempt_kind not in {"training", "contrast"}:
        raise AttemptReceiptError("unknown attempt kind")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "ATTEMPT_AUTHORIZED",
        "attempt_kind": attempt_kind,
        "attempt_index": int(attempt_index),
        "cell": copy.deepcopy(dict(cell)),
        "canonical_paths": list(canonical_paths),
        "context": copy.deepcopy(dict(context)),
        "replay_archive": copy.deepcopy(dict(replay_archive)) if replay_archive else None,
    }
    payload["attempt_identity_sha256"] = canonical_sha256(payload)
    return payload


def validate_attempt_authorization(
    value: Any,
    *,
    attempt_kind: str | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AttemptReceiptError("attempt authorization is not an object")
    authorization = copy.deepcopy(dict(value))
    required = {
        "schema_version",
        "status",
        "attempt_kind",
        "attempt_index",
        "cell",
        "canonical_paths",
        "context",
        "replay_archive",
        "attempt_identity_sha256",
    }
    if set(authorization) != required:
        raise AttemptReceiptError("attempt authorization fields changed")
    if authorization["schema_version"] != 1 or authorization["status"] != "ATTEMPT_AUTHORIZED":
        raise AttemptReceiptError("attempt authorization status changed")
    if attempt_kind is not None and authorization["attempt_kind"] != attempt_kind:
        raise AttemptReceiptError("attempt authorization kind changed")
    if type(authorization["attempt_index"]) is not int or authorization["attempt_index"] <= 0:
        raise AttemptReceiptError("attempt index is invalid")
    if not isinstance(authorization["cell"], dict):
        raise AttemptReceiptError("attempt cell is malformed")
    if (
        not isinstance(authorization["canonical_paths"], list)
        or not authorization["canonical_paths"]
        or any(type(item) is not str for item in authorization["canonical_paths"])
    ):
        raise AttemptReceiptError("attempt canonical paths are malformed")
    if not isinstance(authorization["context"], dict):
        raise AttemptReceiptError("attempt context is malformed")
    if authorization["replay_archive"] is not None and not isinstance(
        authorization["replay_archive"], dict
    ):
        raise AttemptReceiptError("attempt replay archive is malformed")
    _identity_checked(authorization, "attempt_identity_sha256", "attempt authorization")
    return authorization


def build_attempt_marker(authorization: Mapping[str, Any]) -> dict[str, Any]:
    auth = validate_attempt_authorization(authorization)
    marker: dict[str, Any] = {
        "schema_version": 1,
        "status": "ATTEMPT_OUTPUT_CREATED",
        "attempt_kind": auth["attempt_kind"],
        "attempt_identity_sha256": auth["attempt_identity_sha256"],
        "attempt_authorization": auth,
    }
    marker["marker_identity_sha256"] = canonical_sha256(marker)
    return marker


def validate_attempt_marker(
    value: Any,
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AttemptReceiptError("attempt marker is not an object")
    marker = copy.deepcopy(dict(value))
    if set(marker) != {
        "schema_version",
        "status",
        "attempt_kind",
        "attempt_identity_sha256",
        "attempt_authorization",
        "marker_identity_sha256",
    }:
        raise AttemptReceiptError("attempt marker fields changed")
    auth = validate_attempt_authorization(authorization)
    if (
        marker["schema_version"] != 1
        or marker["status"] != "ATTEMPT_OUTPUT_CREATED"
        or marker["attempt_kind"] != auth["attempt_kind"]
        or marker["attempt_identity_sha256"] != auth["attempt_identity_sha256"]
        or marker["attempt_authorization"] != auth
    ):
        raise AttemptReceiptError("attempt marker authorization changed")
    _identity_checked(marker, "marker_identity_sha256", "attempt marker")
    return marker


def ensure_attempt_output(output: Path, authorization: Mapping[str, Any]) -> dict[str, Any]:
    """Create or recover the marker-only PREPARED output, durably."""

    auth = validate_attempt_authorization(authorization)
    marker = build_attempt_marker(auth)
    if os.path.lexists(output):
        info = output.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise AttemptReceiptError("attempt output is not a canonical directory")
        members = {entry.name for entry in output.iterdir()}
        if members != {ATTEMPT_MARKER_NAME}:
            raise AttemptReceiptError("PREPARED attempt output is not marker-only")
        observed = read_json(output / ATTEMPT_MARKER_NAME)
        return validate_attempt_marker(observed, auth)
    _ensure_directory(output.parent)
    output.mkdir(exist_ok=False)
    fsync_directory(output.parent)
    atomic_write_json(output / ATTEMPT_MARKER_NAME, marker, replace=False)
    fsync_directory(output)
    return marker


def attempt_marker_lineage(path: Path, marker: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "marker_identity_sha256": marker["marker_identity_sha256"],
        "attempt_identity_sha256": marker["attempt_identity_sha256"],
    }


def training_journal_path(repo_root: Path | str, slug: str) -> Path:
    root = canonical_root(repo_root)
    return root / EXPERIMENT_RELATIVE / "runs" / "attempts" / "training" / f"{slug}.json"


def _journal_lock_path(journal: Path) -> Path:
    return journal.with_name(f"{journal.name}.lock")


def _validate_journal(
    value: Any,
    *,
    header: Mapping[str, Any],
    cell: Mapping[str, Any],
    canonical_paths: Sequence[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AttemptReceiptError("training attempt journal is not an object")
    journal = copy.deepcopy(dict(value))
    required = {
        "schema_version",
        "status",
        "header",
        "cell",
        "canonical_paths",
        "events",
        "receipt_identity_sha256",
    }
    if set(journal) != required:
        raise AttemptReceiptError("training attempt journal fields changed")
    if journal["schema_version"] != 1 or journal["status"] != "TRAINING_ATTEMPT_JOURNAL":
        raise AttemptReceiptError("training attempt journal status changed")
    if journal["header"] != dict(header):
        raise AttemptReceiptError("training attempt journal provenance changed")
    if journal["cell"] != dict(cell) or journal["canonical_paths"] != list(canonical_paths):
        raise AttemptReceiptError("training attempt journal cell changed")
    events = journal["events"]
    if not isinstance(events, list) or not events:
        raise AttemptReceiptError("training attempt journal has no events")
    for index, event in enumerate(events, start=1):
        if not isinstance(event, dict) or set(event) != {
            "authorization",
            "state",
            "terminal_run_lineage",
        }:
            raise AttemptReceiptError("training attempt event fields changed")
        auth = validate_attempt_authorization(event["authorization"], attempt_kind="training")
        if auth["attempt_index"] != index:
            raise AttemptReceiptError("training attempt journal index changed")
        if event["state"] not in {"PREPARED", "STARTED", "COMPLETE"}:
            raise AttemptReceiptError("training attempt state changed")
        if index < len(events):
            # Only a durable STARTED crash can become historical.  PREPARED is
            # marker-only recovery, and COMPLETE is terminal forever; neither
            # can legitimately have a successor even under a fully rehashed
            # journal forgery.
            if event["state"] != "STARTED":
                raise AttemptReceiptError(
                    "historical training attempt was not a STARTED crash"
                )
            if events[index]["authorization"]["replay_archive"] is None:
                raise AttemptReceiptError("training attempt history omits replay archive")
        if event["state"] == "COMPLETE":
            if not isinstance(event["terminal_run_lineage"], dict):
                raise AttemptReceiptError("complete training attempt omits terminal lineage")
        elif event["terminal_run_lineage"] is not None:
            raise AttemptReceiptError("nonterminal training attempt has terminal lineage")
    _identity_checked(journal, "receipt_identity_sha256", "training attempt journal")
    return journal


def load_training_journal(
    repo_root: Path | str,
    slug: str,
    *,
    header: Mapping[str, Any],
    cell: Mapping[str, Any],
    canonical_paths: Sequence[str],
) -> dict[str, Any] | None:
    path = training_journal_path(repo_root, slug)
    if not os.path.lexists(path):
        return None
    canonical_path(repo_root, repo_relative(repo_root, path))
    return _validate_journal(
        read_json(path), header=header, cell=cell, canonical_paths=canonical_paths
    )


def _write_journal(path: Path, journal: dict[str, Any]) -> None:
    journal.pop("receipt_identity_sha256", None)
    journal["receipt_identity_sha256"] = canonical_sha256(journal)
    atomic_write_json(path, journal, replace=os.path.lexists(path))


def prepare_training_attempt(
    repo_root: Path | str,
    *,
    slug: str,
    header: Mapping[str, Any],
    cell: Mapping[str, Any],
    canonical_paths: Sequence[str],
    context: Mapping[str, Any],
    replay_archive: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Create or idempotently recover one PREPARED training authorization.

    The caller, which owns archive discovery, must supply the one exact archive
    for a STARTED predecessor.  Supplying an archive on a first launch or while
    the predecessor/output still exists fails closed.
    """

    root = canonical_root(repo_root)
    journal_path = training_journal_path(root, slug)
    with locked_regular(_journal_lock_path(journal_path)):
        journal = load_training_journal(
            root,
            slug,
            header=header,
            cell=cell,
            canonical_paths=canonical_paths,
        )
        live = [os.path.lexists(canonical_path(root, item, require_exists=False)) for item in canonical_paths]
        if journal is None:
            if any(live):
                raise AttemptReceiptError("first training attempt has preexisting output")
            if replay_archive is not None:
                raise AttemptReceiptError("first training attempt cannot consume an archive")
            events: list[dict[str, Any]] = []
            index = 1
        else:
            events = journal["events"]
            latest = events[-1]
            latest_auth = validate_attempt_authorization(
                latest["authorization"], attempt_kind="training"
            )
            if latest["state"] == "PREPARED":
                if replay_archive is not None:
                    raise AttemptReceiptError("PREPARED attempt cannot consume another archive")
                if latest_auth["context"] != dict(context):
                    raise AttemptReceiptError("PREPARED training attempt context changed")
                return latest_auth
            if latest["state"] == "COMPLETE":
                raise AttemptReceiptError("completed training attempt cannot be relaunched")
            if any(live):
                raise AttemptReceiptError("STARTED training attempt must be archived before replay")
            if not isinstance(replay_archive, Mapping):
                raise AttemptReceiptError("lost STARTED training output has no exact archive")
            if replay_archive.get("attempt_identity_sha256") != latest_auth[
                "attempt_identity_sha256"
            ]:
                raise AttemptReceiptError("training replay archive binds a different attempt")
            index = len(events) + 1
        authorization = build_attempt_authorization(
            attempt_kind="training",
            attempt_index=index,
            cell=cell,
            canonical_paths=canonical_paths,
            context=context,
            replay_archive=replay_archive,
        )
        events.append(
            {
                "authorization": authorization,
                "state": "PREPARED",
                "terminal_run_lineage": None,
            }
        )
        new_journal = {
            "schema_version": 1,
            "status": "TRAINING_ATTEMPT_JOURNAL",
            "header": copy.deepcopy(dict(header)),
            "cell": copy.deepcopy(dict(cell)),
            "canonical_paths": list(canonical_paths),
            "events": events,
        }
        _write_journal(journal_path, new_journal)
        return authorization


def start_training_attempt(
    repo_root: Path | str,
    *,
    slug: str,
    header: Mapping[str, Any],
    cell: Mapping[str, Any],
    canonical_paths: Sequence[str],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    root = canonical_root(repo_root)
    auth = validate_attempt_authorization(authorization, attempt_kind="training")
    journal_path = training_journal_path(root, slug)
    with locked_regular(_journal_lock_path(journal_path)):
        journal = load_training_journal(
            root, slug, header=header, cell=cell, canonical_paths=canonical_paths
        )
        if journal is None or journal["events"][-1]["authorization"] != auth:
            raise AttemptReceiptError("training attempt is not the journal head")
        event = journal["events"][-1]
        if event["state"] == "STARTED":
            return auth
        if event["state"] != "PREPARED":
            raise AttemptReceiptError("training attempt cannot transition to STARTED")
        external = canonical_path(root, canonical_paths[0])
        marker = read_json(external / ATTEMPT_MARKER_NAME)
        validate_attempt_marker(marker, auth)
        if {item.name for item in external.iterdir()} != {ATTEMPT_MARKER_NAME}:
            raise AttemptReceiptError("training attempt opened output before STARTED")
        event["state"] = "STARTED"
        _write_journal(journal_path, journal)
        return auth


def complete_training_attempt(
    repo_root: Path | str,
    *,
    slug: str,
    header: Mapping[str, Any],
    cell: Mapping[str, Any],
    canonical_paths: Sequence[str],
    authorization: Mapping[str, Any],
    terminal_run_lineage: Mapping[str, Any],
) -> None:
    root = canonical_root(repo_root)
    auth = validate_attempt_authorization(authorization, attempt_kind="training")
    journal_path = training_journal_path(root, slug)
    with locked_regular(_journal_lock_path(journal_path)):
        journal = load_training_journal(
            root, slug, header=header, cell=cell, canonical_paths=canonical_paths
        )
        if journal is None or journal["events"][-1]["authorization"] != auth:
            raise AttemptReceiptError("training completion does not bind journal head")
        event = journal["events"][-1]
        if event["state"] == "COMPLETE":
            if event["terminal_run_lineage"] != dict(terminal_run_lineage):
                raise AttemptReceiptError("training completion lineage changed")
            return
        if event["state"] != "STARTED":
            raise AttemptReceiptError("only STARTED training may complete")
        event["state"] = "COMPLETE"
        event["terminal_run_lineage"] = copy.deepcopy(dict(terminal_run_lineage))
        _write_journal(journal_path, journal)


def training_attempt_is_marker_only(
    repo_root: Path | str,
    *,
    slug: str,
    header: Mapping[str, Any],
    cell: Mapping[str, Any],
    canonical_paths: Sequence[str],
) -> bool:
    journal = load_training_journal(
        repo_root, slug, header=header, cell=cell, canonical_paths=canonical_paths
    )
    if journal is None or journal["events"][-1]["state"] != "PREPARED":
        return False
    auth = journal["events"][-1]["authorization"]
    external = canonical_path(repo_root, canonical_paths[0], require_exists=False)
    tracked = canonical_path(repo_root, canonical_paths[1], require_exists=False)
    if os.path.lexists(tracked) or not os.path.lexists(external):
        return False
    if stat.S_ISLNK(external.lstat().st_mode) or not external.is_dir():
        return False
    if {item.name for item in external.iterdir()} != {ATTEMPT_MARKER_NAME}:
        return False
    try:
        validate_attempt_marker(read_json(external / ATTEMPT_MARKER_NAME), auth)
    except (OSError, ValueError, AttemptReceiptError):
        return False
    return True


def validate_training_attempt_history(
    repo_root: Path | str,
    *,
    slug: str,
    header: Mapping[str, Any],
    cell: Mapping[str, Any],
    canonical_paths: Sequence[str],
    current_authorization: Mapping[str, Any],
    expected_archive_header: Mapping[str, Any],
    journal_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Reopen every replay archive and return one immutable history lineage."""

    root = canonical_root(repo_root)
    journal_path = training_journal_path(root, slug)
    if journal_payload is None:
        raw = read_stable_bytes(root, journal_path)
        journal_value: Any = _json_object_from_bytes(
            raw, "training attempt journal"
        )
    else:
        journal_value = copy.deepcopy(dict(journal_payload))
    journal = _validate_journal(
        journal_value,
        header=header,
        cell=cell,
        canonical_paths=canonical_paths,
    )
    current = validate_attempt_authorization(
        current_authorization, attempt_kind="training"
    )
    if journal["events"][-1]["authorization"] != current:
        raise AttemptReceiptError("terminal attempt is not the journal head")
    if journal["events"][-1]["state"] not in {"STARTED", "COMPLETE"}:
        raise AttemptReceiptError("terminal attempt journal head was never STARTED")

    rows: list[dict[str, Any]] = []
    consumed_archive_paths: set[str] = set()
    for index, event in enumerate(journal["events"]):
        auth = validate_attempt_authorization(
            event["authorization"], attempt_kind="training"
        )
        replay = auth["replay_archive"]
        if index == 0:
            if replay is not None:
                raise AttemptReceiptError("first training attempt has replay lineage")
        else:
            if not isinstance(replay, dict):
                raise AttemptReceiptError("successor attempt omits predecessor archive")
            replay_path = replay.get("path")
            if type(replay_path) is not str or replay_path in consumed_archive_paths:
                raise AttemptReceiptError("training replay archive path is malformed or reused")
            consumed_archive_paths.add(replay_path)
            receipt_path = canonical_path(root, replay_path)
            receipt = validate_failed_archive(
                root, receipt_path, expected_header=expected_archive_header
            )
            observed_lineage = archive_lineage(root, receipt_path, receipt)
            if observed_lineage != replay:
                raise AttemptReceiptError("training replay archive lineage changed")
            authority = receipt["attempts"][0]["archive_authority"]
            predecessor = journal["events"][index - 1]["authorization"]
            if (
                authority.get("attempt_kind") != "training"
                or authority.get("attempt_identity_sha256")
                != predecessor["attempt_identity_sha256"]
                or authority.get("canonical_paths") != list(canonical_paths)
            ):
                raise AttemptReceiptError(
                    "training replay archive does not bind the immediate predecessor"
                )
        rows.append(
            {
                "attempt_index": auth["attempt_index"],
                "attempt_identity_sha256": auth["attempt_identity_sha256"],
                "state": (
                    "TERMINAL_STARTED_OR_COMPLETE"
                    if index == len(journal["events"]) - 1
                    else event["state"]
                ),
                "replay_archive": copy.deepcopy(replay),
            }
        )
    immutable_projection = {
        "header": copy.deepcopy(dict(header)),
        "cell": copy.deepcopy(dict(cell)),
        "canonical_paths": list(canonical_paths),
        "attempts": rows,
    }
    lineage: dict[str, Any] = {
        "schema_version": 1,
        "status": "TRAINING_ATTEMPT_HISTORY_VALIDATED",
        "journal_path": repo_relative(root, journal_path),
        "immutable_journal_history_sha256": canonical_sha256(
            immutable_projection
        ),
        "attempts": rows,
    }
    lineage["history_identity_sha256"] = canonical_sha256(lineage)
    return lineage


def validate_training_attempt_for_terminal(
    repo_root: Path | str,
    *,
    slug: str,
    header: Mapping[str, Any],
    cell: Mapping[str, Any],
    canonical_paths: Sequence[str],
    authorization: Mapping[str, Any],
    external_marker: Path,
    tracked_marker: Path,
    run_lineage: Mapping[str, Any],
    expected_archive_header: Mapping[str, Any],
    expected_history: Mapping[str, Any],
    require_complete: bool = True,
    external_marker_snapshot: Mapping[str, Any] | None = None,
    tracked_marker_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    auth = validate_attempt_authorization(authorization, attempt_kind="training")
    if (external_marker_snapshot is None) != (tracked_marker_snapshot is None):
        raise AttemptReceiptError("training attempt marker snapshot pair is incomplete")
    if external_marker_snapshot is None:
        with open_stable_regular(repo_root, external_marker) as external_handle:
            with open_stable_regular(repo_root, tracked_marker) as tracked_handle:
                external_bytes = external_handle.read()
                tracked_bytes = tracked_handle.read()
                left = os.fstat(external_handle.fileno())
                right = os.fstat(tracked_handle.fileno())
        left_identity = (int(left.st_dev), int(left.st_ino))
        right_identity = (int(right.st_dev), int(right.st_ino))
    else:
        required_snapshot_fields = {"raw", "device", "inode"}
        if (
            set(external_marker_snapshot) != required_snapshot_fields
            or set(tracked_marker_snapshot or {}) != required_snapshot_fields
            or not isinstance(external_marker_snapshot["raw"], bytes)
            or not isinstance((tracked_marker_snapshot or {})["raw"], bytes)
        ):
            raise AttemptReceiptError("training attempt marker snapshot is malformed")
        external_bytes = external_marker_snapshot["raw"]
        tracked_bytes = (tracked_marker_snapshot or {})["raw"]
        left_identity = (
            int(external_marker_snapshot["device"]),
            int(external_marker_snapshot["inode"]),
        )
        right_identity = (
            int((tracked_marker_snapshot or {})["device"]),
            int((tracked_marker_snapshot or {})["inode"]),
        )
    validate_attempt_marker(
        _json_object_from_bytes(external_bytes, "external training attempt marker"), auth
    )
    validate_attempt_marker(
        _json_object_from_bytes(tracked_bytes, "tracked training attempt marker"), auth
    )
    if external_bytes != tracked_bytes:
        raise AttemptReceiptError("training attempt marker mirrors differ")
    if left_identity == right_identity:
        raise AttemptReceiptError("training attempt marker mirrors share an inode")
    journal_path = training_journal_path(repo_root, slug)
    journal = _validate_journal(
        _json_object_from_bytes(
            read_stable_bytes(repo_root, journal_path),
            "training attempt journal",
        ),
        header=header,
        cell=cell,
        canonical_paths=canonical_paths,
    )
    matches = [
        event
        for event in journal["events"]
        if event["authorization"]["attempt_identity_sha256"]
        == auth["attempt_identity_sha256"]
    ]
    if len(matches) != 1 or matches[0]["authorization"] != auth:
        raise AttemptReceiptError("terminal training attempt is not unique in its journal")
    event = matches[0]
    if event["state"] == "COMPLETE":
        if event["terminal_run_lineage"] != dict(run_lineage):
            raise AttemptReceiptError("terminal training journal/run lineage differs")
    elif event["state"] == "STARTED":
        if require_complete:
            raise AttemptReceiptError(
                "terminal training receipts were published before journal completion"
            )
    else:
        raise AttemptReceiptError("terminal training attempt was never STARTED")
    history = validate_training_attempt_history(
        repo_root,
        slug=slug,
        header=header,
        cell=cell,
        canonical_paths=canonical_paths,
        current_authorization=auth,
        expected_archive_header=expected_archive_header,
        journal_payload=journal,
    )
    if history != dict(expected_history):
        raise AttemptReceiptError("terminal training history lineage changed")
    return {"authorization": auth, "history": history}


def _directory_entries_snapshot(descriptor: int) -> tuple[tuple[Any, ...], ...]:
    entries: list[tuple[Any, ...]] = []
    for name in sorted(os.listdir(descriptor)):
        info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if stat.S_ISDIR(info.st_mode):
            kind = "directory"
        elif stat.S_ISREG(info.st_mode):
            kind = "regular_file"
        elif stat.S_ISLNK(info.st_mode):
            kind = "symlink"
        else:
            kind = "special"
        entries.append((name, kind, *_stat_fingerprint(info)))
    return tuple(entries)


@contextmanager
def _held_absolute_directory(path: Path) -> Iterator[int]:
    """Hold one absolute directory and prove its membership and final rebind."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    descriptors: list[int] = []
    try:
        descriptor, descriptors = _open_absolute_directory_chain(absolute)
        before = os.fstat(descriptor)
        if not stat.S_ISDIR(before.st_mode):
            raise AttemptReceiptError("attempt tree root is not a directory")
        before_fingerprint = _stat_fingerprint(before)
        before_entries = _directory_entries_snapshot(descriptor)
        yield descriptor
        if (
            _stat_fingerprint(os.fstat(descriptor)) != before_fingerprint
            or _directory_entries_snapshot(descriptor) != before_entries
        ):
            raise AttemptReceiptError(
                "attempt tree membership changed while it was consumed"
            )
        _verify_absolute_directory_binding(absolute, before)
    except AttemptReceiptError:
        raise
    except OSError as exc:
        raise AttemptReceiptError(
            "attempt tree could not be opened without following aliases"
        ) from exc
    finally:
        for held in reversed(descriptors):
            os.close(held)


def _digest_descriptor(descriptor: int) -> str:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            break
        digest.update(block)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest()


@contextmanager
def _held_tree_manifest_from_descriptor(
    root_descriptor: int,
    *,
    source_path: str,
    seen_inodes: set[tuple[int, int]],
) -> Iterator[dict[str, Any]]:
    """Build a manifest while holding every directory and regular leaf open."""

    files: list[dict[str, Any]] = []
    directories: list[dict[str, Any]] = []
    held_descriptors: list[int] = []
    directory_records: list[tuple[int, tuple[int, ...], tuple[tuple[Any, ...], ...]]] = []
    file_records: list[tuple[int, int, str, tuple[int, ...]]] = []
    seen_directories: set[tuple[int, int]] = set()

    def walk(directory_descriptor: int, relative: Path) -> None:
        directory_info = os.fstat(directory_descriptor)
        if not stat.S_ISDIR(directory_info.st_mode):
            raise AttemptReceiptError(
                "attempt tree contains a noncanonical directory"
            )
        directory_inode = (directory_info.st_dev, directory_info.st_ino)
        if directory_inode in seen_directories:
            raise AttemptReceiptError("attempt tree contains a directory inode alias")
        seen_directories.add(directory_inode)
        entry_snapshot = _directory_entries_snapshot(directory_descriptor)
        directory_records.append(
            (
                directory_descriptor,
                _stat_fingerprint(directory_info),
                entry_snapshot,
            )
        )
        direct: list[dict[str, str]] = []
        children: list[tuple[int, Path]] = []
        for name in sorted(os.listdir(directory_descriptor)):
            item_info = os.stat(
                name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            child = relative / name
            if stat.S_ISDIR(item_info.st_mode):
                child_descriptor = os.open(
                    name,
                    _directory_open_flags(),
                    dir_fd=directory_descriptor,
                )
                held_descriptors.append(child_descriptor)
                opened = os.fstat(child_descriptor)
                if (
                    not stat.S_ISDIR(opened.st_mode)
                    or _stat_fingerprint(opened) != _stat_fingerprint(item_info)
                ):
                    raise AttemptReceiptError(
                        "attempt directory entry changed while it was opened"
                    )
                direct.append({"name": name, "type": "directory"})
                children.append((child_descriptor, child))
            elif stat.S_ISREG(item_info.st_mode):
                descriptor = os.open(
                    name,
                    _regular_open_flags(),
                    dir_fd=directory_descriptor,
                )
                held_descriptors.append(descriptor)
                opened = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or _stat_fingerprint(opened) != _stat_fingerprint(item_info)
                ):
                    raise AttemptReceiptError(
                        "attempt regular entry changed while it was opened"
                    )
                inode = (opened.st_dev, opened.st_ino)
                if inode in seen_inodes or opened.st_nlink != 1:
                    raise AttemptReceiptError(
                        "attempt tree contains a cross-tree hardlink alias"
                    )
                seen_inodes.add(inode)
                before = _stat_fingerprint(opened)
                digest = _digest_descriptor(descriptor)
                if _stat_fingerprint(os.fstat(descriptor)) != before:
                    raise AttemptReceiptError(
                        "attempt regular entry changed while it was hashed"
                    )
                file_records.append(
                    (descriptor, directory_descriptor, name, before)
                )
                direct.append({"name": name, "type": "regular_file"})
                files.append(
                    {
                        "path": child.as_posix(),
                        "bytes": opened.st_size,
                        "sha256": digest,
                    }
                )
            else:
                raise AttemptReceiptError(
                    "attempt tree contains a symlink or special node"
                )
        directories.append(
            {
                "path": "." if not relative.parts else relative.as_posix(),
                "entries": direct,
            }
        )
        for child_descriptor, child_relative in children:
            walk(child_descriptor, child_relative)

    try:
        walk(root_descriptor, Path())
        files.sort(key=lambda item: item["path"])
        directories.sort(key=lambda item: item["path"])
        manifest: dict[str, Any] = {
            "source_path": source_path,
            "files": files,
            "files_sha256": canonical_sha256(files),
            "directory_entries": directories,
            "directory_entries_sha256": canonical_sha256(directories),
        }
        manifest["tree_identity_sha256"] = canonical_sha256(
            {
                "source_path": source_path,
                "files": files,
                "files_sha256": manifest["files_sha256"],
            }
        )
        manifest["manifest_identity_sha256"] = canonical_sha256(manifest)
        yield manifest
        for descriptor, parent_descriptor, name, before in file_records:
            rebound = os.stat(
                name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if (
                _stat_fingerprint(os.fstat(descriptor)) != before
                or _stat_fingerprint(rebound) != before
            ):
                raise AttemptReceiptError(
                    "attempt regular entry changed while its tree was consumed"
                )
        for descriptor, before, entries in reversed(directory_records):
            if (
                _stat_fingerprint(os.fstat(descriptor)) != before
                or _directory_entries_snapshot(descriptor) != entries
            ):
                raise AttemptReceiptError(
                    "attempt directory membership changed while its tree was consumed"
                )
    except AttemptReceiptError:
        raise
    except OSError as exc:
        raise AttemptReceiptError(
            "attempt tree changed or could not be traversed descriptor-relatively"
        ) from exc
    finally:
        for descriptor in reversed(held_descriptors):
            os.close(descriptor)


@contextmanager
def _held_child_tree_manifest(
    parent_descriptor: int,
    name: str,
    *,
    source_path: str,
    seen_inodes: set[tuple[int, int]],
) -> Iterator[dict[str, Any]]:
    descriptor: int | None = None
    try:
        entry = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        if not stat.S_ISDIR(entry.st_mode):
            raise AttemptReceiptError("archived attempt source root is unsafe")
        descriptor = os.open(
            name,
            _directory_open_flags(),
            dir_fd=parent_descriptor,
        )
        opened = os.fstat(descriptor)
        if _stat_fingerprint(opened) != _stat_fingerprint(entry):
            raise AttemptReceiptError(
                "archived attempt source root changed while it was opened"
            )
        with _held_tree_manifest_from_descriptor(
            descriptor,
            source_path=source_path,
            seen_inodes=seen_inodes,
        ) as manifest:
            yield manifest
        rebound = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
        if _stat_fingerprint(rebound) != _stat_fingerprint(opened):
            raise AttemptReceiptError(
                "archived attempt source root changed while it was consumed"
            )
    except AttemptReceiptError:
        raise
    except OSError as exc:
        raise AttemptReceiptError(
            "archived attempt source root could not be opened descriptor-relatively"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def tree_manifest(root: Path, *, source_path: str) -> dict[str, Any]:
    """Hash one exact descriptor-stable regular-file/directory tree."""

    with _held_absolute_directory(root) as descriptor:
        with _held_tree_manifest_from_descriptor(
            descriptor,
            source_path=source_path,
            seen_inodes=set(),
        ) as manifest:
            return copy.deepcopy(manifest)


def _compare_tree_manifest(
    observed: Mapping[str, Any], manifest: Mapping[str, Any]
) -> None:
    tree_fields = {
        "source_path",
        "files",
        "files_sha256",
        "directory_entries",
        "directory_entries_sha256",
        "tree_identity_sha256",
    }
    if {key: observed[key] for key in tree_fields} != {
        key: manifest.get(key) for key in tree_fields
    }:
        raise AttemptReceiptError("archived attempt tree differs from its exact manifest")


def validate_tree_manifest(root: Path, manifest: Mapping[str, Any]) -> None:
    with _held_absolute_directory(root) as descriptor:
        with _held_tree_manifest_from_descriptor(
            descriptor,
            source_path=str(manifest.get("source_path")),
            seen_inodes=set(),
        ) as observed:
            _compare_tree_manifest(observed, manifest)


def _validate_failed_archive_snapshot(
    repo_root: Path | str,
    receipt_path: Path,
    *,
    expected_header: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], str, str]:
    """Validate one archive graph from a single held receipt-mirror snapshot."""

    root = canonical_root(repo_root)
    receipt_relative = repo_relative(root, receipt_path)
    canonical_path(root, receipt_relative)
    try:
        with ExitStack() as stack:
            tracked_handle = stack.enter_context(
                open_stable_regular(root, receipt_path)
            )
            tracked_bytes = tracked_handle.read()
            receipt = _json_object_from_bytes(
                tracked_bytes, "tracked failed-attempt archive receipt"
            )
            required = {
                "schema_version",
                "status",
                "experiment_id",
                "model_id",
                "model_revision",
                "backend",
                "config_sha256",
                "source_contract_sha256",
                "requirements_training_lock_sha256",
                "design_lineage",
                "attempt_identity_sha256",
                "archive_path",
                "attempts",
                "scientific_evidence",
                "receipt_identity_sha256",
            }
            if set(receipt) != required:
                raise AttemptReceiptError(
                    "failed-attempt archive receipt fields changed"
                )
            if (
                receipt["schema_version"] != 1
                or receipt["status"] != "FAILED_ATTEMPT_ARCHIVED"
            ):
                raise AttemptReceiptError("failed-attempt archive status changed")
            _identity_checked(
                receipt, "receipt_identity_sha256", "failed-attempt archive"
            )
            if expected_header is not None:
                for key, expected in expected_header.items():
                    if receipt.get(key) != expected:
                        raise AttemptReceiptError(
                            f"failed-attempt archive {key} changed"
                        )
            attempts = receipt.get("attempts")
            if not isinstance(attempts, list) or not attempts:
                raise AttemptReceiptError(
                    "failed-attempt archive has no source manifests"
                )
            if receipt["attempt_identity_sha256"] != canonical_sha256(
                {"attempts": attempts}
            ):
                raise AttemptReceiptError(
                    "failed-attempt archive set identity changed"
                )
            first_source = (
                attempts[0].get("source_path")
                if isinstance(attempts[0], dict)
                else None
            )
            if type(first_source) is not str:
                raise AttemptReceiptError(
                    "failed-attempt archive has no canonical first source"
                )
            _canonical_relative(first_source)
            label = PurePosixPath(first_source).name
            suffix = receipt["attempt_identity_sha256"][:16]
            expected_archive_relative = (
                LARGE_RELATIVE / "failed_attempts" / f"{label}-{suffix}"
            ).as_posix()
            expected_receipt_relative = (
                EXPERIMENT_RELATIVE
                / "runs"
                / "failures"
                / f"{label}-{suffix}.json"
            ).as_posix()
            if receipt.get("archive_path") != expected_archive_relative:
                raise AttemptReceiptError(
                    "failed-attempt archive path is noncanonical"
                )
            if receipt_relative != expected_receipt_relative:
                raise AttemptReceiptError(
                    "failed-attempt tracked receipt path is noncanonical"
                )
            archive = canonical_path(root, str(receipt["archive_path"]))
            archive_descriptor = stack.enter_context(
                _held_absolute_directory(archive)
            )
            archive_receipt = archive / "archive_receipt.json"
            archive_handle = stack.enter_context(
                open_stable_regular(root, archive_receipt)
            )
            archive_bytes = archive_handle.read()
            archive_info = os.fstat(archive_handle.fileno())
            tracked_info = os.fstat(tracked_handle.fileno())
            archive_entry = os.stat(
                "archive_receipt.json",
                dir_fd=archive_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(archive_entry.st_mode)
                or (archive_entry.st_dev, archive_entry.st_ino)
                != (archive_info.st_dev, archive_info.st_ino)
                or archive_bytes != tracked_bytes
                or (archive_info.st_dev, archive_info.st_ino)
                == (tracked_info.st_dev, tracked_info.st_ino)
            ):
                raise AttemptReceiptError(
                    "failed-attempt receipt mirrors are not independent exact copies"
                )

            expected_members = {"archive_receipt.json"}
            common_authority: dict[str, Any] | None = None
            observed_sources: list[str] = []
            seen_inodes: set[tuple[int, int]] = set()
            for index, manifest in enumerate(attempts, start=1):
                if not isinstance(manifest, dict):
                    raise AttemptReceiptError(
                        "failed-attempt source manifest is malformed"
                    )
                required_manifest = {
                    "source_path",
                    "files",
                    "files_sha256",
                    "directory_entries",
                    "directory_entries_sha256",
                    "tree_identity_sha256",
                    "archive_authority",
                    "manifest_identity_sha256",
                }
                if set(manifest) != required_manifest:
                    raise AttemptReceiptError(
                        "failed-attempt source manifest fields changed"
                    )
                payload = {
                    key: value
                    for key, value in manifest.items()
                    if key != "manifest_identity_sha256"
                }
                if manifest["manifest_identity_sha256"] != canonical_sha256(
                    payload
                ):
                    raise AttemptReceiptError(
                        "failed-attempt source manifest identity changed"
                    )
                authority = manifest.get("archive_authority")
                if not isinstance(authority, dict):
                    raise AttemptReceiptError(
                        "failed-attempt archive authority is malformed"
                    )
                _identity_checked(
                    authority, "authority_identity_sha256", "archive authority"
                )
                if common_authority is None:
                    common_authority = copy.deepcopy(authority)
                elif authority != common_authority:
                    raise AttemptReceiptError(
                        "failed-attempt sources have different authority"
                    )
                source_relative = manifest.get("source_path")
                if type(source_relative) is not str:
                    raise AttemptReceiptError(
                        "failed-attempt source path is malformed"
                    )
                _canonical_relative(source_relative)
                observed_sources.append(source_relative)
                source_label = PurePosixPath(source_relative).name
                source_name = f"source_{index}_{source_label}"
                expected_members.add(source_name)
                observed = stack.enter_context(
                    _held_child_tree_manifest(
                        archive_descriptor,
                        source_name,
                        source_path=source_relative,
                        seen_inodes=seen_inodes,
                    )
                )
                _compare_tree_manifest(observed, manifest)
            if common_authority is None:
                raise AttemptReceiptError("failed-attempt archive omits authority")
            authorized_paths = common_authority.get("canonical_paths")
            present_paths = common_authority.get("present_paths")
            if (
                not isinstance(authorized_paths, list)
                or not authorized_paths
                or len(set(authorized_paths)) != len(authorized_paths)
                or any(type(item) is not str for item in authorized_paths)
            ):
                raise AttemptReceiptError(
                    "failed-attempt canonical paths are malformed"
                )
            for item in authorized_paths:
                _canonical_relative(item)
            if present_paths is None:
                if authorized_paths != observed_sources:
                    raise AttemptReceiptError(
                        "failed-attempt authority/source paths differ"
                    )
            elif (
                present_paths != observed_sources
                or not isinstance(present_paths, list)
                or not present_paths
                or len(set(present_paths)) != len(present_paths)
                or present_paths
                != [item for item in authorized_paths if item in set(present_paths)]
            ):
                raise AttemptReceiptError(
                    "failed-attempt present/source paths differ"
                )
            if common_authority.get("attempt_kind") not in {
                "training",
                "contrast",
                "evaluation",
            }:
                raise AttemptReceiptError(
                    "failed-attempt authority kind changed"
                )
            attempt_id = common_authority.get("attempt_identity_sha256")
            if type(attempt_id) is not str or len(attempt_id) != 64:
                raise AttemptReceiptError(
                    "failed-attempt authority attempt identity is malformed"
                )
            if set(os.listdir(archive_descriptor)) != expected_members:
                raise AttemptReceiptError(
                    "failed-attempt archive root members changed"
                )
            digest = hashlib.sha256(tracked_bytes).hexdigest()
            return receipt, digest, receipt_relative
    except StableArtifactError as exc:
        raise AttemptReceiptError(
            "failed-attempt archive snapshot changed while it was consumed"
        ) from exc


def validate_failed_archive(
    repo_root: Path | str,
    receipt_path: Path,
    *,
    expected_header: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Reopen both receipt copies and every member as one stable snapshot."""

    receipt, _, _ = _validate_failed_archive_snapshot(
        repo_root,
        receipt_path,
        expected_header=expected_header,
    )
    return receipt


def _archive_lineage_from_snapshot(
    receipt_relative: str,
    receipt_sha256: str,
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    authority = dict(receipt["attempts"][0]["archive_authority"])
    return {
        "path": receipt_relative,
        "sha256": receipt_sha256,
        "receipt_identity_sha256": receipt["receipt_identity_sha256"],
        "attempt_set_identity_sha256": receipt["attempt_identity_sha256"],
        "attempt_identity_sha256": authority["attempt_identity_sha256"],
        "archive_path": receipt["archive_path"],
    }


def archive_lineage(
    repo_root: Path | str,
    receipt_path: Path,
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Build lineage only from the exact snapshot that passed graph validation."""

    observed, digest, relative = _validate_failed_archive_snapshot(
        repo_root, receipt_path
    )
    if observed != dict(receipt):
        raise AttemptReceiptError(
            "failed-attempt receipt changed between validation and lineage"
        )
    return _archive_lineage_from_snapshot(relative, digest, observed)


def find_exact_failed_archive(
    repo_root: Path | str,
    *,
    label: str,
    expected_header: Mapping[str, Any],
    attempt_kind: str,
    attempt_identity_sha256: str,
    canonical_paths: Sequence[str],
) -> dict[str, Any]:
    """Return the unique fully validated archive for one STARTED attempt.

    Receipt filename globbing is only candidate discovery.  Every candidate is
    reopened through the complete archive graph, and only an exact authority
    binding to the durable attempt authorization is accepted.
    """

    root = canonical_root(repo_root)
    failures = root / EXPERIMENT_RELATIVE / "runs" / "failures"
    candidates = sorted(failures.glob(f"{label}-*.json")) if failures.is_dir() else []
    matches: list[dict[str, Any]] = []
    for candidate in candidates:
        receipt = validate_failed_archive(
            root, candidate, expected_header=expected_header
        )
        attempts = receipt["attempts"]
        authority = attempts[0]["archive_authority"]
        if (
            authority.get("attempt_kind") == attempt_kind
            and authority.get("attempt_identity_sha256") == attempt_identity_sha256
            and authority.get("canonical_paths") == list(canonical_paths)
        ):
            matches.append(archive_lineage(root, candidate, receipt))
    if len(matches) != 1:
        raise AttemptReceiptError(
            "STARTED attempt requires exactly one fully validated bound archive; "
            f"found {len(matches)}"
        )
    return matches[0]


def required_training_replay_archive(
    repo_root: Path | str,
    *,
    slug: str,
    header: Mapping[str, Any],
    cell: Mapping[str, Any],
    canonical_paths: Sequence[str],
    expected_archive_header: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Resolve the exact archive required by the journal head, if any."""

    root = canonical_root(repo_root)
    journal = load_training_journal(
        root, slug, header=header, cell=cell, canonical_paths=canonical_paths
    )
    if journal is None:
        return None
    head = journal["events"][-1]
    if head["state"] in {"PREPARED", "COMPLETE"}:
        return None
    if head["state"] != "STARTED":
        raise AttemptReceiptError("training journal head state changed")
    if any(
        os.path.lexists(canonical_path(root, item, require_exists=False))
        for item in canonical_paths
    ):
        raise AttemptReceiptError("live STARTED training must be archived before replay")
    auth = validate_attempt_authorization(
        head["authorization"], attempt_kind="training"
    )
    return find_exact_failed_archive(
        root,
        label=slug,
        expected_header=expected_archive_header,
        attempt_kind="training",
        attempt_identity_sha256=auth["attempt_identity_sha256"],
        canonical_paths=canonical_paths,
    )
