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

            with (system / "library.so").open("r+b") as handle, mmap.mmap(
                handle.fileno(), 0, access=mmap.ACCESS_WRITE
            ) as mapping, mock.patch.object(
                L.fcntl, "fcntl", side_effect=scoped_lease
            ), self.assertRaisesRegex(RuntimeError, "exact read-only file mount"):
                mapping[0:1] = b"S"
                mapping.flush()
                L.LoadWindowGuard(
                    [environment, system],
                    expected_content=expected,
                    unleased_roots=[system],
                ).__enter__()

    def test_kernel_denied_lease_accepts_only_stable_exact_read_only_mount(self) -> None:
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
            mount_identity = {
                "mount_id": "1",
                "parent_id": "0",
                "major_minor": "1:1",
                "root": str(mounted),
                "mount_point": str(mounted),
                "mount_options": "ro",
                "optional_fields": [],
                "filesystem_type": "synthetic",
                "source": "/synthetic",
                "super_options": "ro",
            }

            def scoped_lease(descriptor: int, command: int, argument: int):
                path = Path(os.readlink(f"/proc/self/fd/{descriptor}"))
                if (
                    command == fcntl.F_SETLEASE
                    and argument == fcntl.F_RDLCK
                    and path == mounted
                ):
                    raise PermissionError(13, "synthetic kernel denial")
                return original_fcntl(descriptor, command, argument)

            def mounted_identity(path: Path):
                return mount_identity if Path(path) == mounted else None

            with mock.patch.object(
                L.fcntl, "fcntl", side_effect=scoped_lease
            ), mock.patch.object(
                L, "read_only_file_mount_identity", side_effect=mounted_identity
            ):
                guard = L.LoadWindowGuard(
                    [environment, system],
                    expected_content=expected,
                    unleased_roots=[system],
                )
                guard.__enter__()
                guard.bind_authenticated_content(expected, expected)
                receipt = guard.verify()
                self.assertEqual(receipt["schema_version"], 4)
                self.assertEqual(receipt["unleased_files"], 1)
                self.assertEqual(
                    receipt["read_only_file_mounts"], {str(mounted): mount_identity}
                )
                L.validate_load_window_receipt(
                    receipt,
                    [environment, system],
                    expected_content=expected,
                    unleased_roots=[system],
                )
                with self.assertRaisesRegex(ValueError, "invalid or stale"):
                    L.validate_load_window_receipt(
                        receipt,
                        [environment, system],
                        expected_content=expected,
                    )


if __name__ == "__main__":
    unittest.main()
