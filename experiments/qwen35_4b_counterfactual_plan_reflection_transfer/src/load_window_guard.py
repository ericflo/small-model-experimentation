"""Fail-closed Linux guard for authenticated files during model/tokenizer loads."""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import json
import os
import signal
from pathlib import Path
from typing import Any, Iterable


IN_MODIFY = 0x00000002
IN_ATTRIB = 0x00000004
IN_CLOSE_WRITE = 0x00000008
IN_MOVED_FROM = 0x00000040
IN_MOVED_TO = 0x00000080
IN_CREATE = 0x00000100
IN_DELETE = 0x00000200
IN_DELETE_SELF = 0x00000400
IN_MOVE_SELF = 0x00000800
IN_UNMOUNT = 0x00002000
IN_Q_OVERFLOW = 0x00004000
IN_IGNORED = 0x00008000
WATCH_MASK = (
    IN_MODIFY
    | IN_ATTRIB
    | IN_CLOSE_WRITE
    | IN_MOVED_FROM
    | IN_MOVED_TO
    | IN_CREATE
    | IN_DELETE
    | IN_DELETE_SELF
    | IN_MOVE_SELF
    | IN_UNMOUNT
    | IN_Q_OVERFLOW
    | IN_IGNORED
)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _entry(path: Path, root: Path) -> dict[str, Any]:
    relative = "." if path == root else path.relative_to(root).as_posix()
    link = path.is_symlink()
    stat = path.lstat()
    value: dict[str, Any] = {
        "path": relative,
        "kind": "symlink" if link else ("directory" if path.is_dir() else "file"),
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "mode": stat.st_mode,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if link:
        target = path.resolve(strict=True)
        target_stat = target.stat()
        value.update(
            target=str(target),
            target_device=target_stat.st_dev,
            target_inode=target_stat.st_ino,
            target_mode=target_stat.st_mode,
            target_size=target_stat.st_size,
            target_mtime_ns=target_stat.st_mtime_ns,
        )
    return value


def root_commitments(roots: Iterable[Path]) -> list[dict[str, Any]]:
    """Commit the exact directory/symlink/inode surface without rereading huge tensors."""
    result: list[dict[str, Any]] = []
    resolved_roots = sorted({Path(root).resolve() for root in roots}, key=str)
    if not resolved_roots:
        raise ValueError("load-window guard requires at least one root")
    for root in resolved_roots:
        if not root.is_dir() or root.is_symlink():
            raise ValueError(f"load-window root is not a regular directory: {root}")
        paths = [root, *sorted(root.rglob("*"), key=lambda item: item.as_posix())]
        entries = [_entry(path, root) for path in paths]
        if not any(entry["kind"] in {"file", "symlink"} for entry in entries):
            raise ValueError(f"load-window root contains no files: {root}")
        result.append(
            {
                "root": str(root),
                "entries": len(entries),
                "surface_sha256": _canonical_sha256(entries),
            }
        )
    return result


def _files_and_watch_directories(roots: Iterable[Path]) -> tuple[list[Path], list[Path]]:
    files: set[Path] = set()
    directories: set[Path] = set()
    for raw_root in roots:
        root = Path(raw_root).resolve()
        directories.add(root)
        directories.add(root.parent)
        for path in root.rglob("*"):
            if path.is_dir() and not path.is_symlink():
                directories.add(path.resolve())
            elif path.is_file():
                files.add(path.resolve(strict=True))
                directories.add(path.parent.resolve())
                directories.add(path.resolve(strict=True).parent)
    return sorted(files, key=str), sorted(directories, key=str)


def _content_hashes(value: dict[str, Any]) -> dict[str, str]:
    if not isinstance(value, dict) or not value or any(
        not isinstance(key, str) or not key for key in value
    ):
        raise ValueError("load-window content commitments must be a non-empty mapping")
    return {key: _canonical_sha256(item) for key, item in sorted(value.items())}


class LoadWindowGuard:
    """Hold read leases and detect any transient namespace/content mutation."""

    def __init__(
        self,
        roots: Iterable[Path],
        *,
        expected_content: dict[str, Any],
    ):
        self.roots = tuple(sorted({Path(root).resolve() for root in roots}, key=str))
        self._expected_content = _content_hashes(expected_content)
        self._authenticated_content: dict[str, str] | None = None
        self._before: list[dict[str, Any]] | None = None
        self._inotify_fd: int | None = None
        self._lease_fds: list[int] = []
        self._watch_count = 0
        self._sigio_previous: Any = None
        self._sigio_count = 0
        self._closed = False
        self.receipt: dict[str, Any] | None = None

    def __enter__(self) -> "LoadWindowGuard":
        if self._before is not None:
            raise RuntimeError("load-window guard cannot be entered twice")
        self._before = root_commitments(self.roots)
        files, directories = _files_and_watch_directories(self.roots)
        if not files:
            raise ValueError("load-window guard found no regular target files")
        libc = ctypes.CDLL(None, use_errno=True)
        init = libc.inotify_init1
        init.argtypes = [ctypes.c_int]
        init.restype = ctypes.c_int
        descriptor = init(os.O_NONBLOCK | os.O_CLOEXEC)
        if descriptor < 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        self._inotify_fd = descriptor
        add_watch = libc.inotify_add_watch
        add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
        add_watch.restype = ctypes.c_int
        try:
            for directory in directories:
                watch = add_watch(descriptor, os.fsencode(directory), WATCH_MASK)
                if watch < 0:
                    error = ctypes.get_errno()
                    raise OSError(error, f"cannot watch load-window directory {directory}")
                self._watch_count += 1
            self._sigio_previous = signal.getsignal(signal.SIGIO)

            def saw_sigio(_signum: int, _frame: Any) -> None:
                self._sigio_count += 1

            signal.signal(signal.SIGIO, saw_sigio)
            for path in files:
                file_descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
                try:
                    fcntl.fcntl(file_descriptor, fcntl.F_SETLEASE, fcntl.F_RDLCK)
                except OSError as error:
                    os.close(file_descriptor)
                    raise RuntimeError(
                        f"mandatory read lease denied for guarded file: {path}"
                    ) from error
                self._lease_fds.append(file_descriptor)
        except Exception:
            self._close()
            raise
        return self

    def bind_authenticated_content(
        self, before_load: dict[str, Any], after_load: dict[str, Any]
    ) -> None:
        """Bind two in-guard content authentications to the pinned expectation."""
        if self._before is None or self._closed:
            raise RuntimeError("load-window guard is not active")
        if self._authenticated_content is not None:
            raise RuntimeError("load-window content can be authenticated only once")
        before = _content_hashes(before_load)
        after = _content_hashes(after_load)
        if before != self._expected_content or after != self._expected_content:
            raise RuntimeError(
                "authenticated content differs before/after load or from expectation"
            )
        self._authenticated_content = before

    def _event_bytes(self) -> bytes:
        if self._inotify_fd is None:
            return b""
        chunks: list[bytes] = []
        while True:
            try:
                chunk = os.read(self._inotify_fd, 1024 * 1024)
            except BlockingIOError:
                break
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)

    def verify(self) -> dict[str, Any]:
        if self._before is None or self._closed:
            raise RuntimeError("load-window guard is not active")
        if self._authenticated_content is None:
            self._close()
            raise RuntimeError("load-window content was not authenticated while guarded")
        events = self._event_bytes()
        after = root_commitments(self.roots)
        receipt = {
            "schema_version": 2,
            "method": "linux_inotify_plus_read_leases_plus_inode_surface",
            "roots": self._before,
            "authenticated_content_sha256": self._authenticated_content,
            "protected_files": len(self._lease_fds),
            "watched_directories": self._watch_count,
            "inotify_event_bytes": len(events),
            "lease_break_signals": self._sigio_count,
            "decision": "LOAD_WINDOW_IMMUTABLE",
        }
        self._close()
        if events or self._sigio_count or after != self._before:
            raise RuntimeError("authenticated files changed during the load window")
        self.receipt = receipt
        return receipt

    def _close(self) -> None:
        if self._closed:
            return
        for descriptor in self._lease_fds:
            try:
                fcntl.fcntl(descriptor, fcntl.F_SETLEASE, fcntl.F_UNLCK)
            except OSError:
                pass
            os.close(descriptor)
        self._lease_fds.clear()
        if self._inotify_fd is not None:
            os.close(self._inotify_fd)
            self._inotify_fd = None
        if self._sigio_previous is not None:
            signal.signal(signal.SIGIO, self._sigio_previous)
            self._sigio_previous = None
        self._closed = True

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if exc_type is None:
            self.verify()
        else:
            self._close()
        return False


def validate_load_window_receipt(
    receipt: Any,
    roots: Iterable[Path],
    *,
    expected_content: dict[str, Any],
) -> None:
    expected_roots = root_commitments(roots)
    expected_content_hashes = _content_hashes(expected_content)
    required = {
        "schema_version",
        "method",
        "roots",
        "authenticated_content_sha256",
        "protected_files",
        "watched_directories",
        "inotify_event_bytes",
        "lease_break_signals",
        "decision",
    }
    files, _directories = _files_and_watch_directories(roots)
    if (
        not isinstance(receipt, dict)
        or set(receipt) != required
        or type(receipt.get("schema_version")) is not int
        or receipt.get("schema_version") != 2
        or receipt.get("method")
        != "linux_inotify_plus_read_leases_plus_inode_surface"
        or receipt.get("roots") != expected_roots
        or receipt.get("authenticated_content_sha256") != expected_content_hashes
        or type(receipt.get("protected_files")) is not int
        or receipt["protected_files"] < 1
        or receipt["protected_files"] != len(files)
        or type(receipt.get("watched_directories")) is not int
        or receipt["watched_directories"] < 1
        or type(receipt.get("inotify_event_bytes")) is not int
        or receipt.get("inotify_event_bytes") != 0
        or type(receipt.get("lease_break_signals")) is not int
        or receipt.get("lease_break_signals") != 0
        or receipt.get("decision") != "LOAD_WINDOW_IMMUTABLE"
    ):
        raise ValueError("load-window guard receipt is invalid or stale")
