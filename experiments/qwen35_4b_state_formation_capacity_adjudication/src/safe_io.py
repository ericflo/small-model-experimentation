"""Stable no-follow byte snapshots for scientific artifact consumption.

Hashing a pathname and later reopening that pathname is not a proof that the
consumer saw the hashed bytes.  These helpers walk from a trusted repository
directory with ``openat``-style descriptors, reject symlinks and hardlink
aliases, hash the opened inode, and keep that same descriptor open for the
consumer.  A before/after stat receipt also detects in-place mutation while an
artifact is being parsed or deserialized.
"""

from __future__ import annotations

import contextlib
import ctypes
import errno
import gzip
import hashlib
import io
import json
import os
import secrets
import stat
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Callable, Iterator


_SHA256_HEX = frozenset("0123456789abcdef")
_RENAME_NOREPLACE = 1

try:
    _LIBC = ctypes.CDLL(None, use_errno=True)
    _RENAMEAT2 = _LIBC.renameat2
except (AttributeError, OSError):
    _RENAMEAT2 = None
else:
    _RENAMEAT2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    _RENAMEAT2.restype = ctypes.c_int


class StableArtifactError(RuntimeError):
    """An artifact could not be consumed as one stable canonical inode."""


def _root_binding(info: os.stat_result) -> tuple[int, int, int]:
    return (int(info.st_dev), int(info.st_ino), int(info.st_mode))


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json_loads(value: str, label: str) -> Any:
    try:
        return json.loads(
            value,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise StableArtifactError(f"{label} is not strict UTF-8 JSON") from exc


def _canonical_root(
    root: str | os.PathLike[str],
) -> tuple[Path, int, os.stat_result]:
    raw = os.fspath(root)
    candidate = Path(raw)
    lexical = Path(os.path.abspath(raw))
    if candidate.is_absolute():
        if raw.startswith("//") or raw != lexical.as_posix():
            raise StableArtifactError("artifact root is not a canonical absolute path")
    elif (
        raw != candidate.as_posix()
        or not candidate.parts
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise StableArtifactError("artifact root is not a canonical relative path")
    descriptor: int | None = None
    try:
        before = os.stat(lexical, follow_symlinks=False)
        if not stat.S_ISDIR(before.st_mode):
            raise StableArtifactError(
                "artifact root is a symlink, special node, or not a directory"
            )
        resolved = lexical.resolve(strict=True)
        if resolved != lexical:
            raise StableArtifactError("artifact root is a symlink or path alias")
        descriptor = os.open(lexical, _open_flags(directory=True))
        opened = os.fstat(descriptor)
        rebound = os.stat(lexical, follow_symlinks=False)
    except StableArtifactError:
        if descriptor is not None:
            os.close(descriptor)
        raise
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise StableArtifactError(f"artifact root is unavailable: {lexical}") from exc
    if (
        _root_binding(opened) != _root_binding(before)
        or _root_binding(rebound) != _root_binding(opened)
    ):
        os.close(descriptor)
        raise StableArtifactError("artifact root changed while it was opened")
    return lexical, descriptor, opened


@contextlib.contextmanager
def _held_canonical_root(
    root: str | os.PathLike[str],
) -> Iterator[tuple[Path, int]]:
    trusted, descriptor, opened = _canonical_root(root)
    body_error: BaseException | None = None
    try:
        try:
            yield trusted, descriptor
        except BaseException as exc:
            body_error = exc
            raise
        finally:
            try:
                rebound = os.stat(trusted, follow_symlinks=False)
                if (
                    trusted.resolve(strict=True) != trusted
                    or _root_binding(os.fstat(descriptor)) != _root_binding(opened)
                    or _root_binding(rebound) != _root_binding(opened)
                ):
                    raise StableArtifactError(
                        "artifact root changed while it was consumed"
                    )
            except StableArtifactError as exc:
                if body_error is not None:
                    raise exc from body_error
                raise
            except OSError as exc:
                failure = StableArtifactError(
                    "artifact root changed while it was consumed"
                )
                if body_error is not None:
                    raise failure from body_error
                raise failure from exc
    finally:
        os.close(descriptor)


def _relative_parts(root: Path, path: str | os.PathLike[str]) -> tuple[str, ...]:
    raw = os.fspath(path)
    candidate = Path(raw)
    if candidate.is_absolute():
        lexical = Path(os.path.abspath(raw))
        if raw.startswith("//") or raw != lexical.as_posix():
            raise StableArtifactError("artifact path is not canonical")
        try:
            relative = lexical.relative_to(root)
        except ValueError as exc:
            raise StableArtifactError("artifact path escapes its trusted root") from exc
        raw_relative = relative.as_posix()
    else:
        if raw != candidate.as_posix():
            raise StableArtifactError("artifact path is not canonical")
        raw_relative = raw
    pure = PurePosixPath(raw_relative)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} or "\x00" in part for part in pure.parts)
        or pure.as_posix() != raw_relative
    ):
        raise StableArtifactError("artifact path is not canonical")
    return tuple(pure.parts)


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


def _open_flags(*, directory: bool = False) -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if directory:
        flags |= getattr(os, "O_DIRECTORY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


def _open_leaf(root_descriptor: int, parts: tuple[str, ...]) -> tuple[int, list[int]]:
    """Open ``parts`` below ``root`` without following any path component."""

    descriptors: list[int] = []
    try:
        current = os.dup(root_descriptor)
        descriptors.append(current)
        for component in parts[:-1]:
            current = os.open(
                component,
                _open_flags(directory=True),
                dir_fd=current,
            )
            descriptors.append(current)
        leaf = os.open(parts[-1], _open_flags(), dir_fd=current)
        return leaf, descriptors
    except Exception:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise


def _open_parent(root_descriptor: int, parts: tuple[str, ...]) -> tuple[int, list[int]]:
    """Open the canonical parent of ``parts`` and return its held descriptor."""

    descriptors: list[int] = []
    try:
        current = os.dup(root_descriptor)
        descriptors.append(current)
        for component in parts[:-1]:
            current = os.open(
                component,
                _open_flags(directory=True),
                dir_fd=current,
            )
            descriptors.append(current)
        return current, descriptors
    except Exception:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise


def _rename_noreplace_at(
    source_directory_fd: int,
    source_name: str,
    destination_directory_fd: int,
    destination_name: str,
) -> None:
    """Invoke Linux ``renameat2(RENAME_NOREPLACE)`` without a fallback.

    A link/unlink or check/rename emulation would reopen both alias and
    clobber races.  This experiment therefore fails closed when libc or the
    running kernel cannot provide the atomic primitive.
    """

    if _RENAMEAT2 is None:
        raise StableArtifactError(
            "atomic no-replace publication is unavailable: libc has no renameat2"
        )
    ctypes.set_errno(0)
    result = _RENAMEAT2(
        source_directory_fd,
        os.fsencode(source_name),
        destination_directory_fd,
        os.fsencode(destination_name),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(error_number, os.strerror(error_number), destination_name)
    if error_number in {errno.ENOSYS, errno.EINVAL}:
        raise StableArtifactError(
            "atomic no-replace publication is unsupported by the running kernel"
        )
    raise OSError(error_number, os.strerror(error_number), destination_name)


def _entry_fd(parent_fd: int, name: str, *, directory: bool) -> int:
    return os.open(name, _open_flags(directory=directory), dir_fd=parent_fd)


def _entry_identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_mode),
        int(info.st_nlink),
        int(info.st_size),
    )


def _validate_publishable_entry(descriptor: int) -> tuple[os.stat_result, bool]:
    info = os.fstat(descriptor)
    is_directory = stat.S_ISDIR(info.st_mode)
    if not (is_directory or stat.S_ISREG(info.st_mode)):
        raise StableArtifactError("publication source is not a regular file or directory")
    if not is_directory and info.st_nlink != 1:
        raise StableArtifactError("publication source has a hardlink alias")
    return info, is_directory


def _require_entry_binding(
    parent_fd: int,
    name: str,
    held_info: os.stat_result,
    *,
    directory: bool,
) -> None:
    rebound: int | None = None
    try:
        rebound = _entry_fd(parent_fd, name, directory=directory)
        rebound_info, rebound_is_directory = _validate_publishable_entry(rebound)
        if rebound_is_directory != directory or _entry_identity(rebound_info) != _entry_identity(
            held_info
        ):
            raise StableArtifactError("publication entry changed during commit")
    except OSError as exc:
        raise StableArtifactError("publication entry changed during commit") from exc
    finally:
        if rebound is not None:
            os.close(rebound)


def _require_canonical_entry_binding(
    root_descriptor: int,
    parts: tuple[str, ...],
    held_info: os.stat_result,
    *,
    directory: bool,
) -> None:
    descriptors: list[int] = []
    try:
        parent_fd, descriptors = _open_parent(root_descriptor, parts)
        _require_entry_binding(
            parent_fd,
            parts[-1],
            held_info,
            directory=directory,
        )
    except OSError as exc:
        raise StableArtifactError(
            "publication canonical path changed during commit"
        ) from exc
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _require_canonical_entry_absence(
    root_descriptor: int,
    parts: tuple[str, ...],
) -> None:
    descriptors: list[int] = []
    try:
        parent_fd, descriptors = _open_parent(root_descriptor, parts)
        try:
            os.stat(parts[-1], dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        raise StableArtifactError("publication source survived the final rename")
    except StableArtifactError:
        raise
    except OSError as exc:
        raise StableArtifactError(
            "publication canonical source path changed during commit"
        ) from exc
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _commit_new_entry_at(
    parent_fd: int,
    source_name: str,
    destination_name: str,
    source_descriptor: int,
    *,
    directory: bool,
) -> None:
    """Commit one held same-parent entry and prove the final path binding."""

    source_info, source_is_directory = _validate_publishable_entry(source_descriptor)
    if source_is_directory != directory:
        raise StableArtifactError("publication source type changed during commit")
    _require_entry_binding(
        parent_fd,
        source_name,
        source_info,
        directory=directory,
    )
    _rename_noreplace_at(parent_fd, source_name, parent_fd, destination_name)
    # The rename has committed once this point is reached.  A subsequent
    # fsync failure is deliberately reported, but the destination must not be
    # rolled back: rollback would itself be non-durable and could remove a
    # concurrently observed immutable artifact.
    try:
        os.fsync(parent_fd)
    except OSError as exc:
        raise StableArtifactError(
            "publication rename committed but parent directory fsync failed"
        ) from exc
    _require_entry_binding(
        parent_fd,
        destination_name,
        source_info,
        directory=directory,
    )
    try:
        os.stat(source_name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        raise StableArtifactError("publication staging name survived the final rename")


@contextlib.contextmanager
def _open_stable_regular_at(
    trusted: Path,
    root_descriptor: int,
    path: str | os.PathLike[str],
    *,
    expected_sha256: str | None = None,
    require_single_link: bool = True,
) -> Iterator[BinaryIO]:
    """Consume a regular leaf relative to one already-held trusted root."""
    parts = _relative_parts(trusted, path)
    descriptors: list[int] = []
    handle: BinaryIO | None = None
    leaf: int | None = None
    try:
        try:
            leaf, descriptors = _open_leaf(root_descriptor, parts)
            info = os.fstat(leaf)
            if not stat.S_ISREG(info.st_mode):
                raise StableArtifactError("artifact leaf is not a regular file")
            if require_single_link and info.st_nlink != 1:
                raise StableArtifactError(
                    "artifact leaf has an external hardlink alias"
                )
            before = _stat_fingerprint(info)
            handle = os.fdopen(leaf, "rb", closefd=True)
            leaf = None  # ownership transferred to ``handle``
            if expected_sha256 is not None:
                if (
                    type(expected_sha256) is not str
                    or len(expected_sha256) != 64
                    or any(
                        character not in _SHA256_HEX
                        for character in expected_sha256
                    )
                ):
                    raise StableArtifactError(
                        "expected artifact digest is not canonical SHA-256"
                    )
                digest = hashlib.sha256()
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
                if digest.hexdigest() != expected_sha256:
                    raise StableArtifactError(
                        "artifact bytes do not match their bound SHA-256"
                    )
                handle.seek(0)
        except StableArtifactError:
            raise
        except (OSError, UnicodeError) as exc:
            raise StableArtifactError(
                "artifact could not be opened without following aliases"
            ) from exc

        body_error: BaseException | None = None
        try:
            assert handle is not None
            yield handle
        except BaseException as exc:
            body_error = exc
            raise
        finally:
            try:
                assert handle is not None
                if expected_sha256 is not None:
                    handle.seek(0)
                    final_digest = hashlib.sha256()
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        final_digest.update(block)
                    if final_digest.hexdigest() != expected_sha256:
                        raise StableArtifactError(
                            "artifact inode changed while it was consumed"
                        )
                after = _stat_fingerprint(os.fstat(handle.fileno()))
                # An open descriptor survives rename-away, so fstat alone
                # cannot prove that the canonical pathname still names it.
                rebound: int | None = None
                rebound_descriptors: list[int] = []
                try:
                    rebound, rebound_descriptors = _open_leaf(
                        root_descriptor, parts
                    )
                    rebound_info = os.fstat(rebound)
                    if (
                        not stat.S_ISREG(rebound_info.st_mode)
                        or (require_single_link and rebound_info.st_nlink != 1)
                        or (rebound_info.st_dev, rebound_info.st_ino)
                        != (after[0], after[1])
                    ):
                        raise StableArtifactError(
                            "artifact canonical path changed while it was consumed"
                        )
                except OSError as exc:
                    raise StableArtifactError(
                        "artifact canonical path changed while it was consumed"
                    ) from exc
                finally:
                    if rebound is not None:
                        os.close(rebound)
                    for descriptor in reversed(rebound_descriptors):
                        os.close(descriptor)
                if after != before:
                    raise StableArtifactError(
                        "artifact inode changed while it was consumed"
                    )
            except StableArtifactError as exc:
                if body_error is not None:
                    raise exc from body_error
                raise
            except (OSError, UnicodeError) as exc:
                failure = StableArtifactError(
                    "artifact changed while it was consumed"
                )
                if body_error is not None:
                    raise failure from body_error
                raise failure from exc
    finally:
        if handle is not None:
            handle.close()
        elif leaf is not None:
            os.close(leaf)
        for descriptor in reversed(descriptors):
            os.close(descriptor)


@contextlib.contextmanager
def open_stable_regular(
    root: str | os.PathLike[str],
    path: str | os.PathLike[str],
    *,
    expected_sha256: str | None = None,
    require_single_link: bool = True,
) -> Iterator[BinaryIO]:
    """Yield one stable leaf while its canonical trusted root stays bound."""

    with _held_canonical_root(root) as (trusted, root_descriptor):
        with _open_stable_regular_at(
            trusted,
            root_descriptor,
            path,
            expected_sha256=expected_sha256,
            require_single_link=require_single_link,
        ) as handle:
            yield handle


def _mutable_regular_identity(info: os.stat_result) -> tuple[int, int, int, int]:
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_mode),
        int(info.st_nlink),
    )


def _mutable_directory_identity(info: os.stat_result) -> tuple[int, int, int]:
    # A directory's link count legitimately changes when child directories are
    # added or removed.  Its own device/inode/type+mode binding must not.
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_mode),
    )


def _require_mutable_entry_binding(
    parent_fd: int,
    name: str,
    held_info: os.stat_result,
    *,
    directory: bool,
) -> None:
    rebound: int | None = None
    try:
        rebound = _entry_fd(parent_fd, name, directory=directory)
        info = os.fstat(rebound)
        if (
            (directory and not stat.S_ISDIR(info.st_mode))
            or (not directory and not stat.S_ISREG(info.st_mode))
            or (not directory and info.st_nlink != 1)
            or (
                _mutable_directory_identity(info)
                != _mutable_directory_identity(held_info)
                if directory
                else _mutable_regular_identity(info)
                != _mutable_regular_identity(held_info)
            )
        ):
            raise StableArtifactError(
                "mutable artifact canonical binding changed"
            )
    except OSError as exc:
        raise StableArtifactError(
            "mutable artifact canonical binding changed"
        ) from exc
    finally:
        if rebound is not None:
            os.close(rebound)


@contextlib.contextmanager
def open_stable_regular_for_update(
    root: str | os.PathLike[str],
    path: str | os.PathLike[str],
) -> Iterator[int]:
    """Hold one canonical single-link regular inode open for byte updates.

    Every ancestor is traversed from one held trusted-root descriptor.  The
    caller may change file bytes through the yielded descriptor, but may not
    replace, relink, or change the type of the canonical entry.
    """

    with _held_canonical_root(root) as (trusted, root_descriptor):
        parts = _relative_parts(trusted, path)
        descriptors: list[int] = []
        descriptor: int | None = None
        try:
            try:
                parent_fd, descriptors = _open_parent(root_descriptor, parts)
                entry = os.stat(
                    parts[-1], dir_fd=parent_fd, follow_symlinks=False
                )
                if not stat.S_ISREG(entry.st_mode) or entry.st_nlink != 1:
                    raise StableArtifactError(
                        "mutable artifact is not a single-link regular file"
                    )
                flags = (
                    os.O_RDWR
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                )
                descriptor = os.open(parts[-1], flags, dir_fd=parent_fd)
                opened = os.fstat(descriptor)
                if (
                    _mutable_regular_identity(opened)
                    != _mutable_regular_identity(entry)
                ):
                    raise StableArtifactError(
                        "mutable artifact changed while it was opened"
                    )
            except StableArtifactError:
                raise
            except OSError as exc:
                raise StableArtifactError(
                    "mutable artifact could not be held below its trusted root"
                ) from exc

            body_error: BaseException | None = None
            try:
                try:
                    assert descriptor is not None
                    yield descriptor
                except BaseException as exc:
                    body_error = exc
                    raise
                finally:
                    try:
                        assert descriptor is not None
                        after = os.fstat(descriptor)
                        if (
                            not stat.S_ISREG(after.st_mode)
                            or after.st_nlink != 1
                            or _mutable_regular_identity(after)
                            != _mutable_regular_identity(opened)
                        ):
                            raise StableArtifactError(
                                "mutable artifact inode changed while it was held"
                            )
                        _require_mutable_entry_binding(
                            parent_fd,
                            parts[-1],
                            opened,
                            directory=False,
                        )
                        os.fsync(parent_fd)
                        canonical_parent, canonical_descriptors = _open_parent(
                            root_descriptor, parts
                        )
                        try:
                            _require_mutable_entry_binding(
                                canonical_parent,
                                parts[-1],
                                opened,
                                directory=False,
                            )
                        finally:
                            for item in reversed(canonical_descriptors):
                                os.close(item)
                    except StableArtifactError as exc:
                        if body_error is not None:
                            raise exc from body_error
                        raise
                    except OSError as exc:
                        failure = StableArtifactError(
                            "mutable artifact changed while it was held"
                        )
                        if body_error is not None:
                            raise failure from body_error
                        raise failure from exc
            finally:
                pass
        finally:
            if descriptor is not None:
                os.close(descriptor)
            for item in reversed(descriptors):
                os.close(item)


@contextlib.contextmanager
def open_stable_directory_for_update(
    root: str | os.PathLike[str],
    path: str | os.PathLike[str],
) -> Iterator[int]:
    """Hold one canonical directory below a trusted root for child updates."""

    with _held_canonical_root(root) as (trusted, root_descriptor):
        parts = _relative_parts(trusted, path)
        descriptors: list[int] = []
        descriptor: int | None = None
        try:
            try:
                parent_fd, descriptors = _open_parent(root_descriptor, parts)
                entry = os.stat(
                    parts[-1], dir_fd=parent_fd, follow_symlinks=False
                )
                if not stat.S_ISDIR(entry.st_mode):
                    raise StableArtifactError(
                        "mutable artifact is not a canonical directory"
                    )
                descriptor = _entry_fd(parent_fd, parts[-1], directory=True)
                opened = os.fstat(descriptor)
                if (
                    _mutable_directory_identity(opened)
                    != _mutable_directory_identity(entry)
                ):
                    raise StableArtifactError(
                        "mutable directory changed while it was opened"
                    )
            except StableArtifactError:
                raise
            except OSError as exc:
                raise StableArtifactError(
                    "mutable directory could not be held below its trusted root"
                ) from exc

            body_error: BaseException | None = None
            try:
                try:
                    assert descriptor is not None
                    yield descriptor
                except BaseException as exc:
                    body_error = exc
                    raise
                finally:
                    try:
                        assert descriptor is not None
                        after = os.fstat(descriptor)
                        if (
                            not stat.S_ISDIR(after.st_mode)
                            or _mutable_directory_identity(after)
                            != _mutable_directory_identity(opened)
                        ):
                            raise StableArtifactError(
                                "mutable directory inode changed while it was held"
                            )
                        _require_mutable_entry_binding(
                            parent_fd,
                            parts[-1],
                            opened,
                            directory=True,
                        )
                        os.fsync(descriptor)
                        os.fsync(parent_fd)
                        canonical_parent, canonical_descriptors = _open_parent(
                            root_descriptor, parts
                        )
                        try:
                            _require_mutable_entry_binding(
                                canonical_parent,
                                parts[-1],
                                opened,
                                directory=True,
                            )
                        finally:
                            for item in reversed(canonical_descriptors):
                                os.close(item)
                    except StableArtifactError as exc:
                        if body_error is not None:
                            raise exc from body_error
                        raise
                    except OSError as exc:
                        failure = StableArtifactError(
                            "mutable directory changed while it was held"
                        )
                        if body_error is not None:
                            raise failure from body_error
                        raise failure from exc
            finally:
                pass
        finally:
            if descriptor is not None:
                os.close(descriptor)
            for item in reversed(descriptors):
                os.close(item)


def _require_canonical_directory_chain(
    root_descriptor: int,
    parts: tuple[str, ...],
    expected: tuple[tuple[int, int, int], ...],
) -> None:
    """Re-walk and bind every held directory component from the root."""

    if len(parts) != len(expected):
        raise StableArtifactError("canonical directory chain receipt is malformed")
    descriptors: list[int] = []
    try:
        current = os.dup(root_descriptor)
        descriptors.append(current)
        for component, identity in zip(parts, expected, strict=True):
            current = os.open(
                component,
                _open_flags(directory=True),
                dir_fd=current,
            )
            descriptors.append(current)
            info = os.fstat(current)
            if (
                not stat.S_ISDIR(info.st_mode)
                or _mutable_directory_identity(info) != identity
            ):
                raise StableArtifactError(
                    "canonical directory chain changed while it was created"
                )
    except StableArtifactError:
        raise
    except OSError as exc:
        raise StableArtifactError(
            "canonical directory chain changed while it was created"
        ) from exc
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def ensure_canonical_directory(
    root: str | os.PathLike[str],
    path: str | os.PathLike[str],
    *,
    mode: int = 0o700,
) -> None:
    """Create missing directory components durably below one held root."""

    with _held_canonical_root(root) as (trusted, root_descriptor):
        parts = _relative_parts(trusted, path)
        current = os.dup(root_descriptor)
        descriptors = [current]
        try:
            for component in parts:
                created = False
                try:
                    os.mkdir(component, mode=mode, dir_fd=current)
                    created = True
                except FileExistsError:
                    pass
                if created:
                    os.fsync(current)
                child = os.open(
                    component,
                    _open_flags(directory=True),
                    dir_fd=current,
                )
                info = os.fstat(child)
                if not stat.S_ISDIR(info.st_mode):
                    os.close(child)
                    raise StableArtifactError(
                        "canonical directory component is not a directory"
                    )
                descriptors.append(child)
                current = child
                if created:
                    os.fsync(current)
            expected = tuple(
                _mutable_directory_identity(os.fstat(descriptor))
                for descriptor in descriptors[1:]
            )
            _require_canonical_directory_chain(root_descriptor, parts, expected)
        except StableArtifactError:
            raise
        except OSError as exc:
            raise StableArtifactError(
                "canonical directory tree could not be created durably"
            ) from exc
        finally:
            for descriptor in reversed(descriptors):
                os.close(descriptor)


def fsync_canonical_directory(
    root: str | os.PathLike[str],
    path: str | os.PathLike[str],
) -> None:
    """Fsync one canonical directory and its held parent binding."""

    with open_stable_directory_for_update(root, path) as descriptor:
        os.fsync(descriptor)


def read_verified_bytes(
    root: str | os.PathLike[str],
    path: str | os.PathLike[str],
    expected_sha256: str,
) -> bytes:
    """Return one stable byte snapshot after hashing that same open inode."""

    with open_stable_regular(root, path, expected_sha256=expected_sha256) as handle:
        return handle.read()


def read_stable_bytes(
    root: str | os.PathLike[str],
    path: str | os.PathLike[str],
) -> bytes:
    """Return bytes from one no-follow, single-link, stat-stable inode."""

    with open_stable_regular(root, path) as handle:
        return handle.read()


def _create_private_stage(parent_fd: int, mode: int) -> tuple[int, str]:
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    # A dead process can leave a named staging inode behind.  Each attempt
    # uses an independent 128-bit name and retries collisions, so debris can
    # neither alias the eventual destination nor reserve the next stage.
    for _ in range(128):
        temporary = f".publish-{secrets.token_hex(16)}.tmp"
        try:
            descriptor = os.open(temporary, flags, mode, dir_fd=parent_fd)
        except FileExistsError:
            continue
        try:
            os.fchmod(descriptor, mode)
        except BaseException:
            os.close(descriptor)
            try:
                os.unlink(temporary, dir_fd=parent_fd)
            finally:
                os.fsync(parent_fd)
            raise
        return descriptor, temporary
    raise StableArtifactError("could not allocate a unique private publication stage")


def _sha256_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while True:
        block = os.pread(descriptor, 1024 * 1024, offset)
        if not block:
            return digest.hexdigest()
        digest.update(block)
        offset += len(block)


def _publish_new_file_at(
    trusted: Path,
    root_descriptor: int,
    path: str | os.PathLike[str],
    writer: Callable[[BinaryIO], None],
    *,
    mode: int = 0o600,
) -> str:
    """Stream, fsync, and atomically publish one new immutable regular file.

    Every parent must already exist below the trusted root.  A private
    descriptor-relative named staging inode is fsynced and moved into place
    with Linux ``renameat2(RENAME_NOREPLACE)``.  No link/unlink fallback is
    permitted.  The final canonical pathname is reopened without following
    aliases and verified against the digest of the same held staging inode.

    ``writer`` receives a binary stream and may write incrementally without
    materializing the complete artifact.  The returned SHA-256 identifies the
    exact committed bytes.
    """

    if not callable(writer):
        raise StableArtifactError("publication writer must be callable")
    if type(mode) is not int or not 0 <= mode <= 0o777:
        raise StableArtifactError("published artifact mode must be canonical permission bits")
    parts = _relative_parts(trusted, path)
    parent_fd: int | None = None
    descriptors: list[int] = []
    temporary: str | None = None
    stage_descriptor: int | None = None
    digest: str | None = None
    failed = False
    try:
        parent_fd, descriptors = _open_parent(root_descriptor, parts)
        leaf = parts[-1]
        stage_descriptor, temporary = _create_private_stage(parent_fd, mode)
        with os.fdopen(os.dup(stage_descriptor), "wb", closefd=True) as handle:
            writer(handle)
            if not handle.closed:
                handle.flush()
        os.fsync(stage_descriptor)
        stage_info, is_directory = _validate_publishable_entry(stage_descriptor)
        if is_directory:
            raise StableArtifactError("private publication stage is not a regular file")
        if stat.S_IMODE(stage_info.st_mode) != mode:
            raise StableArtifactError("private publication stage has the wrong mode")
        digest = _sha256_descriptor(stage_descriptor)
        try:
            _commit_new_entry_at(
                parent_fd,
                temporary,
                leaf,
                stage_descriptor,
                directory=False,
            )
        except FileExistsError as exc:
            raise StableArtifactError(
                "refusing to overwrite immutable published artifact"
            ) from exc
        temporary = None
    except StableArtifactError:
        failed = True
        raise
    except OSError as exc:
        failed = True
        raise StableArtifactError(
            "artifact could not be durably published without following aliases"
        ) from exc
    except BaseException:
        failed = True
        raise
    finally:
        if parent_fd is not None and temporary is not None:
            try:
                os.unlink(temporary, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
            except OSError as exc:
                if not failed:
                    raise StableArtifactError(
                        "private publication stage could not be removed"
                    ) from exc
            else:
                try:
                    os.fsync(parent_fd)
                except OSError as exc:
                    if not failed:
                        raise StableArtifactError(
                            "private publication stage cleanup was not durable"
                        ) from exc
        if stage_descriptor is not None:
            os.close(stage_descriptor)
        for descriptor in reversed(descriptors):
            os.close(descriptor)
    assert digest is not None
    with _open_stable_regular_at(
        trusted,
        root_descriptor,
        path,
        expected_sha256=digest,
    ) as handle:
        handle.read()
    return digest


def publish_new_file(
    root: str | os.PathLike[str],
    path: str | os.PathLike[str],
    writer: Callable[[BinaryIO], None],
    *,
    mode: int = 0o600,
) -> str:
    """Publish one new file while holding its canonical trusted root."""

    with _held_canonical_root(root) as (trusted, root_descriptor):
        return _publish_new_file_at(
            trusted,
            root_descriptor,
            path,
            writer,
            mode=mode,
        )


def publish_new_bytes(
    root: str | os.PathLike[str],
    path: str | os.PathLike[str],
    payload: bytes,
    *,
    mode: int = 0o600,
) -> None:
    """Durably install exact bytes without replacing any existing leaf."""

    if type(payload) is not bytes:
        raise StableArtifactError("published artifact payload must be exact bytes")
    digest = publish_new_file(root, path, lambda handle: handle.write(payload), mode=mode)
    if digest != hashlib.sha256(payload).hexdigest():
        raise StableArtifactError("published artifact bytes changed during installation")


def _rename_new_entry_at(
    trusted: Path,
    root_descriptor: int,
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
) -> None:
    """Durably rename one existing file or directory to a new same-parent name.

    The source and destination must share the exact canonical parent below the
    trusted root.  Existing destination leaves of every kind are preserved.
    This is the directory-commit counterpart to :func:`publish_new_file`.
    """

    source_parts = _relative_parts(trusted, source)
    destination_parts = _relative_parts(trusted, destination)
    if source_parts[:-1] != destination_parts[:-1]:
        raise StableArtifactError("entry commit requires one canonical parent directory")
    if source_parts[-1] == destination_parts[-1]:
        raise StableArtifactError("entry commit source and destination must differ")
    parent_fd: int | None = None
    descriptors: list[int] = []
    source_descriptor: int | None = None
    committed_info: os.stat_result | None = None
    committed_is_directory: bool | None = None
    try:
        parent_fd, descriptors = _open_parent(root_descriptor, destination_parts)
        try:
            source_info = os.stat(
                source_parts[-1],
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise StableArtifactError("publication source entry is unavailable") from exc
        if stat.S_ISLNK(source_info.st_mode):
            raise StableArtifactError("publication source entry is a symlink")
        is_directory = stat.S_ISDIR(source_info.st_mode)
        if not (is_directory or stat.S_ISREG(source_info.st_mode)):
            raise StableArtifactError("publication source is not a regular file or directory")
        source_descriptor = _entry_fd(
            parent_fd,
            source_parts[-1],
            directory=is_directory,
        )
        held_info, held_is_directory = _validate_publishable_entry(source_descriptor)
        if held_is_directory != is_directory or _entry_identity(held_info) != _entry_identity(
            source_info
        ):
            raise StableArtifactError("publication source entry changed while it was opened")
        os.fsync(source_descriptor)
        try:
            _commit_new_entry_at(
                parent_fd,
                source_parts[-1],
                destination_parts[-1],
                source_descriptor,
                directory=is_directory,
            )
        except FileExistsError as exc:
            raise StableArtifactError(
                "refusing to overwrite immutable published artifact"
            ) from exc
        committed_info = held_info
        committed_is_directory = is_directory
    except StableArtifactError:
        raise
    except OSError as exc:
        raise StableArtifactError(
            "entry could not be durably committed without following aliases"
        ) from exc
    finally:
        if source_descriptor is not None:
            os.close(source_descriptor)
        for descriptor in reversed(descriptors):
            os.close(descriptor)
    assert committed_info is not None and committed_is_directory is not None
    _require_canonical_entry_binding(
        root_descriptor,
        destination_parts,
        committed_info,
        directory=committed_is_directory,
    )


def rename_new_entry(
    root: str | os.PathLike[str],
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
) -> None:
    """Commit one same-parent entry while holding its canonical trusted root."""

    with _held_canonical_root(root) as (trusted, root_descriptor):
        _rename_new_entry_at(
            trusted,
            root_descriptor,
            source,
            destination,
        )


def _move_new_entry_at(
    trusted: Path,
    root_descriptor: int,
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
) -> None:
    """Durably no-clobber move one held entry between canonical parents."""

    source_parts = _relative_parts(trusted, source)
    destination_parts = _relative_parts(trusted, destination)
    if source_parts == destination_parts:
        raise StableArtifactError("entry move source and destination must differ")
    source_parent: int | None = None
    destination_parent: int | None = None
    source_descriptors: list[int] = []
    destination_descriptors: list[int] = []
    source_descriptor: int | None = None
    committed_info: os.stat_result | None = None
    committed_is_directory: bool | None = None
    try:
        source_parent, source_descriptors = _open_parent(
            root_descriptor, source_parts
        )
        destination_parent, destination_descriptors = _open_parent(
            root_descriptor, destination_parts
        )
        try:
            source_info = os.stat(
                source_parts[-1],
                dir_fd=source_parent,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise StableArtifactError("publication source entry is unavailable") from exc
        if stat.S_ISLNK(source_info.st_mode):
            raise StableArtifactError("publication source entry is a symlink")
        is_directory = stat.S_ISDIR(source_info.st_mode)
        if not (is_directory or stat.S_ISREG(source_info.st_mode)):
            raise StableArtifactError(
                "publication source is not a regular file or directory"
            )
        source_descriptor = _entry_fd(
            source_parent,
            source_parts[-1],
            directory=is_directory,
        )
        held_info, held_is_directory = _validate_publishable_entry(source_descriptor)
        if (
            held_is_directory != is_directory
            or _entry_identity(held_info) != _entry_identity(source_info)
        ):
            raise StableArtifactError(
                "publication source entry changed while it was opened"
            )
        os.fsync(source_descriptor)
        _require_entry_binding(
            source_parent,
            source_parts[-1],
            held_info,
            directory=is_directory,
        )
        try:
            _rename_noreplace_at(
                source_parent,
                source_parts[-1],
                destination_parent,
                destination_parts[-1],
            )
        except FileExistsError as exc:
            raise StableArtifactError(
                "refusing to overwrite immutable published artifact"
            ) from exc
        # A cross-directory rename is durable only after both directory
        # entries have been flushed.  Attempt both even if the first reports an
        # I/O error, and finish every feasible binding check before reporting a
        # committed-but-not-proven-durable outcome.
        committed_failures: list[BaseException] = []
        source_parent_info = os.fstat(source_parent)
        destination_parent_info = os.fstat(destination_parent)
        parent_descriptors = [source_parent]
        if (destination_parent_info.st_dev, destination_parent_info.st_ino) != (
            source_parent_info.st_dev,
            source_parent_info.st_ino,
        ):
            parent_descriptors.append(destination_parent)
        for parent_descriptor in parent_descriptors:
            try:
                os.fsync(parent_descriptor)
            except OSError as exc:
                committed_failures.append(exc)
        try:
            _require_entry_binding(
                destination_parent,
                destination_parts[-1],
                held_info,
                directory=is_directory,
            )
        except StableArtifactError as exc:
            committed_failures.append(exc)
        try:
            try:
                os.stat(
                    source_parts[-1],
                    dir_fd=source_parent,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            else:
                raise StableArtifactError(
                    "publication source survived the held-parent move"
                )
        except (OSError, StableArtifactError) as exc:
            committed_failures.append(
                StableArtifactError(
                    "publication source absence could not be verified"
                )
            )
        committed_info = held_info
        committed_is_directory = is_directory
        try:
            _require_canonical_entry_binding(
                root_descriptor,
                destination_parts,
                committed_info,
                directory=committed_is_directory,
            )
            _require_canonical_entry_absence(root_descriptor, source_parts)
        except StableArtifactError as exc:
            committed_failures.append(exc)
        if committed_failures:
            raise StableArtifactError(
                "publication move committed but durability or binding verification failed"
            ) from committed_failures[0]
    except StableArtifactError:
        raise
    except OSError as exc:
        raise StableArtifactError(
            "entry could not be durably moved without following aliases"
        ) from exc
    finally:
        if source_descriptor is not None:
            os.close(source_descriptor)
        for descriptor in reversed(destination_descriptors):
            os.close(descriptor)
        for descriptor in reversed(source_descriptors):
            os.close(descriptor)
    assert committed_info is not None and committed_is_directory is not None
    _require_canonical_entry_binding(
        root_descriptor,
        destination_parts,
        committed_info,
        directory=committed_is_directory,
    )
    _require_canonical_entry_absence(root_descriptor, source_parts)


def move_new_entry(
    root: str | os.PathLike[str],
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
) -> None:
    """No-clobber move one entry between canonical parents under one root."""

    with _held_canonical_root(root) as (trusted, root_descriptor):
        _move_new_entry_at(
            trusted,
            root_descriptor,
            source,
            destination,
        )


def read_stable_json_object(
    root: str | os.PathLike[str],
    path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Read a strict JSON object from one stat-stable inode."""

    raw = read_stable_bytes(root, path)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise StableArtifactError("stable JSON is not UTF-8") from exc
    value = _strict_json_loads(text, "stable JSON")
    if not isinstance(value, dict):
        raise StableArtifactError("stable JSON root is not an object")
    return value


def read_verified_json_object(
    root: str | os.PathLike[str],
    path: str | os.PathLike[str],
    expected_sha256: str,
) -> dict[str, Any]:
    """Read a strict JSON object from one verified byte snapshot."""

    raw = read_verified_bytes(root, path, expected_sha256)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise StableArtifactError("verified JSON is not UTF-8") from exc
    value = _strict_json_loads(text, "verified JSON")
    if not isinstance(value, dict):
        raise StableArtifactError("verified JSON root is not an object")
    return value


def read_verified_jsonl_gzip(
    root: str | os.PathLike[str],
    path: str | os.PathLike[str],
    expected_sha256: str,
) -> list[dict[str, Any]]:
    """Decompress and parse JSONL from exactly the compressed bytes hashed."""

    raw = read_verified_bytes(root, path, expected_sha256)
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(raw), mode="rb") as archive:
            text = archive.read().decode("utf-8")
    except (OSError, EOFError, UnicodeDecodeError) as exc:
        raise StableArtifactError("verified gzip JSONL payload is invalid") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        value = _strict_json_loads(line, f"verified JSONL line {line_number}")
        if not isinstance(value, dict):
            raise StableArtifactError(
                f"verified JSONL line {line_number} is not an object"
            )
        rows.append(value)
    return rows
