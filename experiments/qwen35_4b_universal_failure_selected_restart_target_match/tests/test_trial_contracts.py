from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


EXP = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FrozenTrainingContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.trial = load_module("restart_trial", EXP / "scripts" / "train_trial.py")
        cls.harness = load_module("restart_harness", EXP / "scripts" / "run.py")

    def test_identity_parent_and_exact_compute_are_frozen(self) -> None:
        self.assertEqual(self.trial.MODEL_ID, "Qwen/Qwen3.5-4B")
        self.assertEqual(self.trial.EXPECTED_ROWS, 320)
        self.assertEqual(self.trial.EXPECTED_FORWARD, 297731)
        self.assertEqual(self.trial.EXPECTED_NONZERO, 126796)
        self.assertEqual(self.trial.EXPECTED_MASS_X5, 138164)
        self.assertEqual(
            self.trial.PARENT_WEIGHTS_SHA256,
            "bb59d3bd9273ae3bb3dffe54e983590dada69e6e1bdba571009ffedbba05154d",
        )
        self.assertEqual(
            self.trial.TOKEN_RECEIPT_SHA256,
            "52a761ef8fd37f3eac88abf8f090013f571a47511daeb26820ca030201b1c170",
        )
        self.assertEqual(self.trial.PUBLISHED_ARM_HASHES["replay_control"], {
            "receipt": "3a9cc1ea291e201c742a9f72d428387dbb4d421e46fe1236db0bc016caf56d49",
            "log": "3bedc341a075c6c0ed72204cb64aa919bf9763ebe6656e8c8f92707650a86f25",
            "adapter_config": "dce1095c4a6c49611f51efed2a89177cf26945a8694c5fa0bba33cc069a9f8f6",
            "adapter_weights": "5840757d2e639c224cb1abb43320c0b8581eb9eec453ce613e0279803eab6b1c",
        })
        self.assertEqual(
            self.trial.PUBLISHED_ARM_HASHES["counterfactual_restart_candidate"],
            {
                "receipt": "6aa5c3f10699019cedcfb37d179656d534b8fd80e6b9107b2b4b0574790e9871",
                "log": "c8572c88b6977ecb2666dd5ab471b1079517b75bfedf6387ece98551f9f2202a",
                "adapter_config": "6915787d341bdf3c932401586bd209b507e17a36f045c262084fdea7d97f7f50",
                "adapter_weights": "2072c5c81e0ce35161bfe7b49d9995b2152b8e49aa09db3a2f247c37218639bc",
            },
        )

    def test_only_frozen_hyperparameters_are_registered(self) -> None:
        expected = self.trial.expected_hyperparameters()
        self.assertEqual(expected, {
            "epochs": 1.0,
            "lr": 1e-5,
            "rank": 32,
            "alpha": 64,
            "batch_size": 1,
            "grad_accum": 8,
            "max_length": 4096,
            "w_think": 0.2,
            "w_close": 0.2,
            "seed": 48,
            "optimizer_steps": 40,
        })

    def test_harness_constructs_one_exact_control_command(self) -> None:
        with (
            mock.patch.object(self.harness, "require_pushed_checkpoint"),
            mock.patch.object(self.harness, "run") as invoke,
            mock.patch("sys.argv", ["run.py", "--stage", "train-control"]),
        ):
            self.assertEqual(self.harness.main(), 0)
        command = invoke.call_args.args[0]
        self.assertEqual(command[command.index("--name") + 1], "replay_control")
        self.assertEqual(command[command.index("--seed") + 1], "48")
        self.assertEqual(command[command.index("--max-length") + 1], "4096")
        self.assertEqual(command[command.index("--grad-accum") + 1], "8")

    def test_candidate_requires_committed_control_before_launch(self) -> None:
        source = (EXP / "scripts" / "run.py").read_text(encoding="utf-8")
        self.assertIn("runs/training/replay_control.json", source)
        wrapper = (EXP / "scripts" / "train_trial.py").read_text(encoding="utf-8")
        self.assertIn('args.name == "counterfactual_restart_candidate"', wrapper)
        self.assertIn('validate_published_arm("replay_control")', wrapper)
        self.assertIn("committed_at_head(receipt_path)", wrapper)
        self.assertIn("committed_at_head(log_path)", wrapper)

    def test_failure_receipt_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "failure.json"
            self.trial.preserve_failure(path, {"name": "fixture"}, reason="x", returncode=7)
            self.assertEqual(json.loads(path.read_text())["failure_reason"], "x")
            with self.assertRaises(FileExistsError):
                self.trial.preserve_failure(
                    path, {"name": "fixture"}, reason="overwrite", returncode=8
                )


if __name__ == "__main__":
    unittest.main()
