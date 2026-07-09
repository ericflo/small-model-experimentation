"""Regression test for the final task-independent interface taxonomy."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verified_macro_interface_audit", EXP / "scripts" / "analyze_interface_failure.py"
)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


class InterfaceFailureAuditTests(unittest.TestCase):
    def test_final_failure_taxonomy_regenerates_from_raw_vllm_rows(self) -> None:
        metrics = audit.analyze()["metrics"]
        self.assertEqual(metrics["strictly_parsed_samples"], 16)
        self.assertEqual(metrics["exact_expansion_samples"], 3)
        self.assertEqual(metrics["successful_records"], 1)
        self.assertEqual(metrics["failed_samples_with_multiple_macros"], 13)
        self.assertEqual(metrics["failed_samples_over_expanded_depth_five"], 13)
        self.assertEqual(metrics["failed_samples_with_designated_alias"], 10)
        self.assertEqual(metrics["failed_samples_without_designated_alias"], 3)


if __name__ == "__main__":
    unittest.main()
