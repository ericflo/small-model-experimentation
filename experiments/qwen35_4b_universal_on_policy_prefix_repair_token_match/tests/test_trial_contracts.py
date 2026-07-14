from __future__ import annotations

import argparse
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
        cls.trial = load_module("prefix_trial", EXP / "scripts" / "train_trial.py")
        cls.harness = load_module("prefix_run", EXP / "scripts" / "run.py")

    def test_trial_identity_and_exact_compute_are_frozen(self) -> None:
        self.assertEqual(self.trial.MODEL_ID, "Qwen/Qwen3.5-4B")
        self.assertEqual(self.trial.EXPECTED_ROWS, 320)
        self.assertEqual(self.trial.EXPECTED_FORWARD_TOKENS, 304313)
        self.assertEqual(
            self.trial.TOKEN_RECEIPT_SHA256,
            "eb08026ffcf82b8780819a26a522f04d69358ffdfd4797dd4c603dd1fbbe0cfc",
        )
        self.assertEqual(
            set(self.trial.FROZEN_TRAIN_FILES),
            {"replay_after_close", "prefix_repair_after_close"},
        )
        self.assertEqual(
            self.trial.CONTROL_DATA_SHA256,
            "541805df2d817707c1e76213e50c8f08fd9caff10d0a3887e1196424b6820be6",
        )
        self.assertEqual(
            self.trial.CONTROL_RECEIPT_SHA256,
            "f78f2069fd1c7b37bbd0b13b581df0ce7360de92256323fcf5f3c7b0936ed6de",
        )
        self.assertEqual(
            self.trial.CONTROL_WEIGHTS_SHA256,
            "bb59d3bd9273ae3bb3dffe54e983590dada69e6e1bdba571009ffedbba05154d",
        )
        self.assertEqual(
            self.trial.CANDIDATE_RECEIPT_SHA256,
            "846d8107ecadad458c18cd985d54feb42748e87677dd708c14a99e84cf4e7098",
        )
        self.assertEqual(
            self.trial.CANDIDATE_WEIGHTS_SHA256,
            "858111918bd8a0a5bb379d6b9b1b2b600f013bd1da516d2b4e7cdf8ebd510f14",
        )

    def test_only_registered_hyperparameters_pass(self) -> None:
        frozen = argparse.Namespace(
            epochs=1.0,
            lr=1e-5,
            rank=32,
            alpha=64,
            batch_size=1,
            grad_accum=8,
            max_length=4096,
            w_think=0.2,
            w_close=0.2,
            seed=47,
        )
        self.assertTrue(self.trial.frozen_hyperparameters(frozen))
        changed = argparse.Namespace(**vars(frozen))
        changed.seed = 48
        self.assertFalse(self.trial.frozen_hyperparameters(changed))

    def test_harness_constructs_one_exact_arm_command(self) -> None:
        with mock.patch.object(self.harness, "run") as invoke:
            self.harness.train_arm("replay_after_close")
        invoke.assert_called_once()
        command = invoke.call_args.args[0]
        self.assertIn("train_trial.py", Path(command[2]).name)
        self.assertEqual(command[command.index("--name") + 1], "replay_after_close")
        self.assertEqual(command[command.index("--seed") + 1], "47")
        self.assertEqual(command[command.index("--max-length") + 1], "4096")
        self.assertEqual(command[command.index("--grad-accum") + 1], "8")

    def test_candidate_stage_requires_published_control_receipt(self) -> None:
        source = (EXP / "scripts" / "run.py").read_text(encoding="utf-8")
        self.assertIn(
            "(TOKEN_RECEIPT, COMPUTE_REVIEW, CONTROL_RECEIPT)",
            source,
        )
        self.assertLess(
            source.index('elif args.stage == "train-control"'),
            source.index('elif args.stage == "train-candidate"'),
        )
        wrapper = (EXP / "scripts" / "train_trial.py").read_text(encoding="utf-8")
        self.assertIn('args.name == "prefix_repair_after_close"', wrapper)
        self.assertIn("validate_control_prerequisite()", wrapper)
        self.assertIn("committed_at_head(log)", wrapper)
        self.assertIn("validate_control_prerequisite(require_committed=False)", source)
        self.assertIn("validate_candidate_checkpoint(require_committed=False)", source)

    def test_candidate_checkpoint_reauthenticates_published_control(self) -> None:
        source = (EXP / "scripts" / "train_trial.py").read_text(encoding="utf-8")
        candidate = source[source.index("def validate_candidate_checkpoint") :]
        self.assertIn("validate_control_prerequisite(", candidate)
        self.assertIn('payload.get("control_prerequisite")', candidate)
        self.assertIn("committed_at_head(CANDIDATE_RECEIPT)", candidate)
        self.assertIn("committed_at_head(log)", candidate)

    def test_local_design_and_merge_order_are_frozen(self) -> None:
        self.assertEqual(
            self.harness.LOCAL_DESIGN_RECEIPT_SHA256,
            "3982d5b80e17a39c23b2e93d1d57ffd9895067ba08c7b74b39e7b50b04f6e85a",
        )
        source = (EXP / "scripts" / "run.py").read_text(encoding="utf-8")
        self.assertLess(
            source.index('elif args.stage == "merge-control"'),
            source.index('elif args.stage == "merge-candidate"'),
        )
        self.assertLess(
            source.index('elif args.stage == "merge-candidate"'),
            source.index('elif args.stage == "local"'),
        )
        self.assertIn("CONTROL_MERGE_RECEIPT", source)
        self.assertIn("CANDIDATE_MERGE_RECEIPT", source)
        self.assertIn("PASS_CONTROL_MERGE", source)
        self.assertEqual(
            self.harness.CONTROL_MERGE_RECEIPT_SHA256,
            "bc78f33218afb99b4ebd5b173f1f24aa628b20fad82d627b00529cabf911d550",
        )
        self.assertEqual(
            self.harness.CONTROL_MERGED_WEIGHTS_SHA256,
            "7ab4c419f70135d3fe058dba6e79e3a9a61c6661d43e6acb9662f331efe36e2e",
        )
        self.assertEqual(
            self.harness.CONTROL_EXTERNAL_MERGE_RECEIPT_SHA256,
            "aa763255cb3b05599e765948d3a3db1787d5813b1cfafbdc7e1c21653ae745a3",
        )
        self.assertEqual(
            self.harness.CANDIDATE_MERGE_RECEIPT_SHA256,
            "3deff026e85f7f855fa6cc8db2218fa0ed7c7da48b6c992174ec5bc5e38a438d",
        )
        self.assertEqual(
            self.harness.CANDIDATE_MERGED_WEIGHTS_SHA256,
            "376e208298c2a13308c4955c42d87f5ce1464ca7cd46efb35d5c608b3bedb528",
        )
        self.assertEqual(
            self.harness.CANDIDATE_EXTERNAL_MERGE_RECEIPT_SHA256,
            "baa2027e0e2032315913a3e2b41a986f296fedd932f6c71a8f1fa289d0746d5a",
        )
        self.assertEqual(
            self.harness.LOCAL_RECEIPT_SHA256,
            "b4b333ca1095e78b5eb8cc2c3395c99e03efe1b583fda270c3d3dc26c206b8c8",
        )
        self.assertEqual(
            self.harness.PROMOTION_RECEIPT_SHA256,
            "1e048e75ef58eae94f2555f70a16002231f01d70750ff590c06aa553fcad5f5c",
        )

    def test_failure_receipt_is_durable_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "failure.json"
            self.trial.preserve_failure(
                path,
                {"name": "arm"},
                reason="test_failure",
                returncode=7,
                adapter_complete=False,
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["failure_reason"], "test_failure")
            self.assertEqual(payload["returncode"], 7)
            with self.assertRaises(FileExistsError):
                self.trial.preserve_failure(
                    path,
                    {"name": "arm"},
                    reason="overwrite",
                    returncode=8,
                    adapter_complete=False,
                )

    def test_receipt_exposes_target_composition_difference(self) -> None:
        receipt = json.loads(
            (EXP / "data" / "stream_token_receipt.json").read_text(encoding="utf-8")
        )
        self.assertEqual(receipt["forward_token_delta"], 0)
        self.assertEqual(receipt["candidate_minus_control_spans"]["target_span"], -33421)
        self.assertEqual(receipt["candidate_minus_control_spans"]["answer_target"], 528)


if __name__ == "__main__":
    unittest.main()
