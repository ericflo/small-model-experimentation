from __future__ import annotations

import sys
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))

from merge_weighted_adapters import _target_module_set  # noqa: E402


class WeightedMergeConfigTests(unittest.TestCase):
    def test_target_module_order_is_not_semantic(self):
        quick = {"target_modules": ["q_proj", "v_proj", "down_proj"]}
        deep = {"target_modules": ["down_proj", "q_proj", "v_proj"]}
        self.assertEqual(_target_module_set(quick), _target_module_set(deep))

    def test_duplicate_or_malformed_targets_fail(self):
        for config in (
            {"target_modules": []},
            {"target_modules": ["q_proj", "q_proj"]},
            {"target_modules": "q_proj"},
        ):
            with self.assertRaises(ValueError):
                _target_module_set(config)


if __name__ == "__main__":
    unittest.main()
