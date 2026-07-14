from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_trial.py"
SPEC = importlib.util.spec_from_file_location("close_weight_train_trial", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TrainingLogTests(unittest.TestCase):
    def test_normalize_log_removes_progress_bar_trailing_blanks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.log"
            path.write_text("step one   \n   \nstep two\t\n", encoding="utf-8")

            MODULE.normalize_log(path)

            self.assertEqual(path.read_text(encoding="utf-8"), "step one\n\nstep two\n")


if __name__ == "__main__":
    unittest.main()
