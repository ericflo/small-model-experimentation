from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "merge_parent.py"
SPEC = importlib.util.spec_from_file_location("on_policy_parent_merge", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MergeParentFreezeTests(unittest.TestCase):
    def test_only_authenticated_parent_is_mergeable(self) -> None:
        self.assertEqual(MODULE.FROZEN_ADAPTERS, {"close_xi_parent": MODULE.PARENT_ADAPTER})

    def test_parent_and_model_identities_are_frozen(self) -> None:
        self.assertEqual(MODULE.MODEL_ID, "Qwen/Qwen3.5-4B")
        self.assertEqual(
            MODULE.MODEL_REVISION,
            "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        )
        self.assertEqual(
            MODULE.PARENT_WEIGHTS_SHA256,
            "16e9dc75a0e33e182e916600ff6e1d75fc46dfa45e870216e2c149a41253c179",
        )
        self.assertEqual(
            MODULE.PARENT_CONFIG_SHA256,
            "de953bd57502ff728a12d1627d5aacab6284b045428ec7b83026388afd8c47ff",
        )


if __name__ == "__main__":
    unittest.main()
