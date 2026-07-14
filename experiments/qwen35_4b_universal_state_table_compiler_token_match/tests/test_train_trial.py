from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_trial.py"
SPEC = importlib.util.spec_from_file_location("state_table_train_trial", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TrainingWrapperTests(unittest.TestCase):
    def test_frozen_arm_paths_and_parent_identity(self) -> None:
        self.assertEqual(set(MODULE.FROZEN_TRAIN_FILES), {
            "replay_after_close", "state_table_after_close",
        })
        self.assertEqual(
            MODULE.PARENT_WEIGHTS_SHA256,
            "16e9dc75a0e33e182e916600ff6e1d75fc46dfa45e870216e2c149a41253c179",
        )
        self.assertEqual(
            MODULE.PARENT_CONFIG_SHA256,
            "de953bd57502ff728a12d1627d5aacab6284b045428ec7b83026388afd8c47ff",
        )
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("target-close", source)
        self.assertNotIn("target_close", source)

    def test_normalize_log_removes_progress_bar_trailing_blanks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.log"
            path.write_text("step one   \n   \nstep two\t\n", encoding="utf-8")
            MODULE.normalize_log(path)
            self.assertEqual(path.read_text(encoding="utf-8"), "step one\n\nstep two\n")


if __name__ == "__main__":
    unittest.main()
