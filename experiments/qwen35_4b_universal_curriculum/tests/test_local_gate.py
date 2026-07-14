from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_local.py"
SPEC = importlib.util.spec_from_file_location("universal_check_local", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class LocalGateTests(unittest.TestCase):
    def test_abstention_spellings_are_detected(self) -> None:
        for value in (None, "None", "INSUFFICIENT", "null", "unknown", "N/A", "abstain"):
            with self.subTest(value=value):
                self.assertTrue(MODULE.is_abstention(value))

    def test_valid_route_ids_are_not_abstentions(self) -> None:
        for value in ("IX", "mer", "tool_3", "0"):
            with self.subTest(value=value):
                self.assertFalse(MODULE.is_abstention(value))


if __name__ == "__main__":
    unittest.main()
