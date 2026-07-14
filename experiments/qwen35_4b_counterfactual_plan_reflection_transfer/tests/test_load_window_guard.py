from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

import load_window_guard as L  # noqa: E402


class LoadWindowGuardTests(unittest.TestCase):
    def test_unchanged_read_load_emits_replayable_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "authenticated"
            root.mkdir()
            (root / "weights.bin").write_bytes(b"trusted")
            guard = L.LoadWindowGuard([root])
            guard.__enter__()
            self.assertEqual((root / "weights.bin").read_bytes(), b"trusted")
            receipt = guard.verify()
            L.validate_load_window_receipt(receipt, [root])

    def test_swap_read_restore_is_detected_even_when_final_hash_matches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            outer = Path(temporary)
            root = outer / "authenticated"
            root.mkdir()
            target = root / "weights.bin"
            target.write_bytes(b"trusted")
            original_hash = __import__("hashlib").sha256(target.read_bytes()).hexdigest()
            backup = outer / "backup.bin"
            evil = outer / "evil.bin"
            evil.write_bytes(b"malicious")
            guard = L.LoadWindowGuard([root])
            guard.__enter__()
            target.replace(backup)
            evil.replace(target)
            self.assertEqual(target.read_bytes(), b"malicious")
            target.replace(evil)
            backup.replace(target)
            self.assertEqual(
                __import__("hashlib").sha256(target.read_bytes()).hexdigest(),
                original_hash,
            )
            with self.assertRaisesRegex(RuntimeError, "changed during the load window"):
                guard.verify()


if __name__ == "__main__":
    unittest.main()
