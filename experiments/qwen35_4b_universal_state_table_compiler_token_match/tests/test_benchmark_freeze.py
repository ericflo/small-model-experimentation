from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_benchmark.py"
SPEC = importlib.util.spec_from_file_location("state_table_benchmark", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BenchmarkFreezeTests(unittest.TestCase):
    def test_pilot_identity_and_arms_are_frozen(self) -> None:
        self.assertEqual(MODULE.FROZEN_NAME, "pilot1")
        self.assertEqual(MODULE.FROZEN_TIER, "quick")
        self.assertEqual(MODULE.FROZEN_THINK_BUDGET, 1024)
        self.assertEqual(MODULE.FROZEN_SEED, 78138)
        self.assertEqual(MODULE.FROZEN_CANDIDATE, "state_table_after_close")
        self.assertEqual(MODULE.FROZEN_MODELS, {
            "base", "blend", "replay_refresh", "close_xi_parent",
            "replay_after_close", "state_table_after_close",
        })

    def test_public_family_contract_is_complete_and_aggregate_only(self) -> None:
        self.assertEqual(len(MODULE.PUBLIC_FAMILIES), 10)
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("run_benchmark_aggregate.py", source)
        self.assertNotIn("benchmarks/", source)


if __name__ == "__main__":
    unittest.main()
