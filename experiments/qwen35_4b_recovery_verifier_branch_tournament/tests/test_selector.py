from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "select_tournament", EXP / "scripts" / "select_tournament.py"
)
assert SPEC and SPEC.loader
selector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(selector)


class SelectorTests(unittest.TestCase):
    def test_candidate_is_default_and_action_requires_unique_public_pass(self):
        for candidate_pass, action_pass, expected in (
            (False, False, "candidate"),
            (True, False, "candidate"),
            (True, True, "candidate"),
            (False, True, "action"),
        ):
            candidate = {"final_visible_pass": candidate_pass}
            action = {"final_visible_pass": action_pass}
            self.assertEqual(selector.choose_public(candidate, action), expected)

    def test_selector_never_reads_hidden_success(self):
        candidate = {"final_visible_pass": False, "workspace_success": True}
        action = {"final_visible_pass": True, "workspace_success": False}
        before = selector.choose_public(candidate, action)
        candidate["workspace_success"] = False
        action["workspace_success"] = True
        self.assertEqual(selector.choose_public(candidate, action), before)

    def test_random_control_is_deterministic(self):
        first = selector.choose_random("case-a", 85603)
        self.assertEqual(first, selector.choose_random("case-a", 85603))
        self.assertIn(first, {"candidate", "action"})


if __name__ == "__main__":
    unittest.main()
