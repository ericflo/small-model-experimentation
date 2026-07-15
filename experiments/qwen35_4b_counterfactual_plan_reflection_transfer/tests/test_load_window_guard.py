from __future__ import annotations

import sys
import fcntl
import mmap
import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

import load_window_guard as L  # noqa: E402


def _content(root: Path) -> dict[str, str]:
    return {
        "weights": hashlib.sha256((root / "weights.bin").read_bytes()).hexdigest()
    }


class LoadWindowGuardTests(unittest.TestCase):
    def test_unchanged_read_load_emits_replayable_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "authenticated"
            root.mkdir()
            (root / "weights.bin").write_bytes(b"trusted")
            expected = _content(root)
            guard = L.LoadWindowGuard([root], expected_content=expected)
            guard.__enter__()
            before = _content(root)
            self.assertEqual((root / "weights.bin").read_bytes(), b"trusted")
            guard.bind_authenticated_content(before, _content(root))
            receipt = guard.verify()
            L.validate_load_window_receipt(
                receipt, [root], expected_content=expected
            )

    def test_swap_read_restore_is_detected_even_when_final_hash_matches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            outer = Path(temporary)
            root = outer / "authenticated"
            root.mkdir()
            target = root / "weights.bin"
            target.write_bytes(b"trusted")
            expected = _content(root)
            original_hash = hashlib.sha256(target.read_bytes()).hexdigest()
            backup = outer / "backup.bin"
            evil = outer / "evil.bin"
            evil.write_bytes(b"malicious")
            guard = L.LoadWindowGuard([root], expected_content=expected)
            guard.__enter__()
            before = _content(root)
            target.replace(backup)
            evil.replace(target)
            self.assertEqual(target.read_bytes(), b"malicious")
            target.replace(evil)
            backup.replace(target)
            self.assertEqual(
                hashlib.sha256(target.read_bytes()).hexdigest(),
                original_hash,
            )
            guard.bind_authenticated_content(before, _content(root))
            with self.assertRaisesRegex(RuntimeError, "changed during the load window"):
                guard.verify()

    def test_metadata_preserving_pre_guard_substitution_fails_before_load(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "authenticated"
            root.mkdir()
            target = root / "weights.bin"
            target.write_bytes(b"trusted")
            expected = _content(root)
            original = target.stat()
            target.write_bytes(b"altered")
            os.utime(target, ns=(original.st_atime_ns, original.st_mtime_ns))
            self.assertEqual(target.stat().st_ino, original.st_ino)
            with self.assertRaisesRegex(RuntimeError, "differs before/after load"):
                with L.LoadWindowGuard([root], expected_content=expected) as guard:
                    observed = _content(root)
                    guard.bind_authenticated_content(observed, observed)

    def test_kernel_denied_lease_on_mutable_file_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            outer = Path(temporary)
            environment = outer / "environment"
            system = outer / "system"
            environment.mkdir()
            system.mkdir()
            (environment / "weights.bin").write_bytes(b"environment")
            (system / "library.so").write_bytes(b"system")
            expected = {"runtime": "pinned"}
            original_fcntl = fcntl.fcntl

            def scoped_lease(descriptor: int, command: int, argument: int):
                path = Path(os.readlink(f"/proc/self/fd/{descriptor}"))
                if (
                    command == fcntl.F_SETLEASE
                    and argument == fcntl.F_RDLCK
                    and system in path.parents
                ):
                    raise PermissionError(13, "synthetic kernel denial")
                return original_fcntl(descriptor, command, argument)

            with mock.patch.object(
                L.fcntl, "fcntl", side_effect=scoped_lease
            ), self.assertRaisesRegex(RuntimeError, "mandatory read lease denied"):
                L.LoadWindowGuard(
                    [environment, system],
                    expected_content=expected,
                ).__enter__()

    def test_preexisting_shared_writable_mapping_makes_guard_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "authenticated"
            root.mkdir()
            target = root / "weights.bin"
            target.write_bytes(b"trusted")
            expected = _content(root)
            with target.open("r+b") as handle, mmap.mmap(
                handle.fileno(), 0, access=mmap.ACCESS_WRITE
            ):
                with self.assertRaisesRegex(RuntimeError, "mandatory read lease denied"):
                    L.LoadWindowGuard([root], expected_content=expected).__enter__()

    def test_kernel_denied_lease_has_no_read_only_mount_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            outer = Path(temporary)
            environment = outer / "environment"
            system = outer / "system"
            environment.mkdir()
            system.mkdir()
            (environment / "weights.bin").write_bytes(b"environment")
            mounted = system / "library.so"
            mounted.write_bytes(b"system")
            expected = {"runtime": "pinned"}
            original_fcntl = fcntl.fcntl
            def scoped_lease(descriptor: int, command: int, argument: int):
                path = Path(os.readlink(f"/proc/self/fd/{descriptor}"))
                if (
                    command == fcntl.F_SETLEASE
                    and argument == fcntl.F_RDLCK
                    and path == mounted
                ):
                    raise PermissionError(13, "synthetic kernel denial")
                return original_fcntl(descriptor, command, argument)

            with mock.patch.object(
                L.fcntl, "fcntl", side_effect=scoped_lease
            ), self.assertRaisesRegex(RuntimeError, "mandatory read lease denied"):
                L.LoadWindowGuard(
                    [environment, system],
                    expected_content=expected,
                ).__enter__()


if __name__ == "__main__":
    unittest.main()
