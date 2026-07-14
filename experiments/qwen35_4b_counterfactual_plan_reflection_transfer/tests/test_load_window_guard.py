from __future__ import annotations

import sys
import hashlib
import os
import tempfile
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
