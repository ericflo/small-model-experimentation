from __future__ import annotations

import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "collect_parent_rollouts.py"
)


class CollectionGitReceiptTests(unittest.TestCase):
    def test_wrapper_never_requires_post_open_runner_cleanliness(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn('get("git_dirty") is not False', source)
        self.assertIn('get("git_commit") != git_head', source)
        self.assertIn("fresh collection requires a clean worktree before opening its log", source)

    def test_recovery_path_is_explicit_and_forbids_generation_rerun(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('"--recover-completed"', source)
        self.assertIn('"generation_rerun": False', source)
        self.assertIn("all other frozen checks passed", source)


if __name__ == "__main__":
    unittest.main()
