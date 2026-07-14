from __future__ import annotations

import importlib.util
import json
import unittest
from collections import Counter
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_residual_input.py"
SPEC = importlib.util.spec_from_file_location("residual_sibling_input", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ResidualInputTests(unittest.TestCase):
    def test_derived_input_is_deterministic_and_residual_only(self) -> None:
        outputs = MODULE.build_outputs()
        value = outputs[MODULE.OUT]
        rows = [json.loads(line) for line in value.decode().splitlines()]
        counts = Counter(row["meta"]["skill"] for row in rows)
        self.assertEqual(len(rows), 225)
        self.assertEqual(set(counts), set(MODULE.policy.EXPECTED_SKILLS))
        self.assertTrue(all(counts[skill] >= 4 for skill in MODULE.policy.EXPECTED_SKILLS))
        self.assertFalse(set(counts) & set(MODULE.EXCLUDED_SKILLS))

    def test_model_input_excludes_oracle_fields(self) -> None:
        value = MODULE.build_outputs()[MODULE.OUT]
        for forbidden in (b'"answer"', b'"think"', b'"_audit"', b'"truth_valid"', b'"expected_answer"'):
            self.assertNotIn(forbidden, value)

    def test_manifest_binds_inherited_stop_and_all_skill_retention(self) -> None:
        payload = json.loads(MODULE.build_outputs()[MODULE.MANIFEST])
        self.assertEqual(payload["origin"]["hard_failure_rows"], 227)
        self.assertEqual(payload["residual_policy"]["hard_failure_rows"], 225)
        self.assertEqual(payload["residual_policy"]["excluded_skills"], ["select", "count", "route"])
        self.assertTrue(payload["retention_policy"]["unchanged_all_skill_local_gate_required"])
        self.assertFalse(payload["benchmark_data_read"])


if __name__ == "__main__":
    unittest.main()
