"""Stage-recipe fidelity against the copied manifest, zero-root rewiring exact.

Each stage's dataset, seed, hyperparameters, and trainer variant must
equal the manifest's recorded recipe. The ONE designed change is the
warm-start chain: stage 1 must be FRESH (no --warm-start anywhere in its
command; the trainer's default fresh path zero-initializes the LoRA
delta) and stages 2-6 must warm-start from the PREVIOUS zero-root
stage's output. The manifest's recorded per-stage adapter hashes are
CONTRAST fields only and must never act as verification.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
SCRIPTS = EXP / "scripts"
sys.path.insert(0, str(SCRIPTS))

import gen_design_receipt as gd  # noqa: E402
import rebuild_zero_root as rz  # noqa: E402


def command_value(command: list[str], flag: str) -> str:
    index = command.index(flag)
    return command[index + 1]


def command_values(command: list[str], flag: str) -> list[str]:
    return [
        command[index + 1]
        for index, token in enumerate(command)
        if token == flag
    ]


def build_commands() -> list[tuple[dict, list[str], Path | None]]:
    manifest = rz.load_manifest()
    rows = manifest["stages"]
    out = []
    warm: Path | None = None
    for row in rows:
        out_dir = rz.ADAPTER_ROOT / rz.stage_dirname(row)
        dataset = EXP / "data" / "lineage" / Path(row["dataset"]["file"]).name
        command = rz.stage_command(row, dataset, out_dir, warm)
        out.append((row, command, warm))
        warm = out_dir
    return out


class TestStageCommands(unittest.TestCase):
    def setUp(self):
        self.commands = build_commands()

    def test_stage_one_is_fresh_no_warm_start_anywhere(self):
        row, command, warm = self.commands[0]
        self.assertIsNone(warm)
        self.assertNotIn("--warm-start", command)
        # The fresh path builds the adapter from the CLI rank/alpha: they
        # must be the recorded rank-32/alpha-64.
        self.assertEqual(command_value(command, "--rank"), "32")
        self.assertEqual(command_value(command, "--alpha"), "64")

    def test_stage_one_exact_recorded_hyperparameters(self):
        row, command, _ = self.commands[0]
        self.assertEqual(command_value(command, "--lr"), "1e-05")
        self.assertEqual(command_value(command, "--batch-size"), "1")
        self.assertEqual(command_value(command, "--grad-accum"), "8")
        self.assertEqual(command_value(command, "--max-length"), "4096")
        self.assertEqual(command_value(command, "--epochs"), "1.0")
        self.assertEqual(command_value(command, "--w-think"), "0.2")
        self.assertEqual(command_value(command, "--seed"), "42")
        self.assertNotIn("--w-close", command)  # stage 1 has NO w_close
        self.assertTrue(command[2].endswith("train_think_stage12.py"))

    def test_stage_two_has_no_w_close_and_warm_starts_from_stage_one(self):
        row, command, warm = self.commands[1]
        self.assertNotIn("--w-close", command)
        self.assertEqual(
            command_value(command, "--warm-start"),
            str(rz.ADAPTER_ROOT / "stage01_replay_refresh"),
        )
        self.assertEqual(warm, rz.ADAPTER_ROOT / "stage01_replay_refresh")
        self.assertEqual(command_value(command, "--seed"), "43")
        self.assertTrue(command[2].endswith("train_think_stage12.py"))

    def test_stage_three_targeted_close_overrides(self):
        row, command, _ = self.commands[2]
        self.assertTrue(command[2].endswith("train_think_close_stage3.py"))
        self.assertEqual(command_value(command, "--w-close"), "0.2")
        self.assertEqual(
            command_values(command, "--target-close-kind"),
            ["u_execute", "u_induct"],
        )
        self.assertEqual(command_value(command, "--target-w-close"), "1.0")
        self.assertEqual(command_value(command, "--seed"), "44")

    def test_stages_four_to_six_use_the_456_trainer_with_w_close(self):
        for index, seed in ((3, "47"), (4, "51"), (5, "55")):
            row, command, warm = self.commands[index]
            with self.subTest(stage=index + 1):
                self.assertTrue(command[2].endswith("train_think_stage456.py"))
                self.assertEqual(command_value(command, "--w-close"), "0.2")
                self.assertEqual(command_value(command, "--seed"), seed)
                self.assertNotIn("--target-close-kind", command)
                self.assertNotIn("--target-w-close", command)

    def test_warm_start_chain_is_the_previous_zero_root_stage(self):
        expected_names = (
            None,
            "stage01_replay_refresh",
            "stage02_designed160",
            "stage03_close_xi",
            "stage04_replay_after_close",
            "stage05_designed_fresh",
        )
        for (row, command, warm), expected in zip(self.commands, expected_names):
            with self.subTest(stage=row["stage"]):
                if expected is None:
                    self.assertIsNone(warm)
                    self.assertNotIn("--warm-start", command)
                else:
                    self.assertEqual(
                        command_value(command, "--warm-start"),
                        str(rz.ADAPTER_ROOT / expected),
                    )

    def test_every_stage_recipe_matches_the_manifest(self):
        for row, command, _ in self.commands:
            hypers = row["hyperparameters"]
            with self.subTest(stage=row["stage"]):
                self.assertEqual(command_value(command, "--epochs"), str(hypers["epochs"]))
                self.assertEqual(command_value(command, "--lr"), str(hypers["lr"]))
                self.assertEqual(command_value(command, "--rank"), str(hypers["rank"]))
                self.assertEqual(command_value(command, "--alpha"), str(hypers["alpha"]))
                self.assertEqual(
                    command_value(command, "--batch-size"), str(hypers["batch_size"])
                )
                self.assertEqual(
                    command_value(command, "--grad-accum"), str(hypers["grad_accum"])
                )
                self.assertEqual(
                    command_value(command, "--max-length"), str(hypers["max_length"])
                )
                self.assertEqual(
                    command_value(command, "--w-think"), str(hypers["w_think"])
                )
                self.assertEqual(command_value(command, "--seed"), str(row["seed"]))
                self.assertEqual(
                    Path(command_value(command, "--train")).name,
                    Path(row["dataset"]["file"]).name,
                )
                if "w_close" in hypers:
                    self.assertEqual(
                        command_value(command, "--w-close"), str(hypers["w_close"])
                    )
                else:
                    self.assertNotIn("--w-close", command)
                self.assertTrue(command[2].endswith(Path(row["trainer"]).name))


class TestStageReceipts(unittest.TestCase):
    def setUp(self):
        self.manifest = rz.load_manifest()

    def _fake_adapter(self, directory: Path, tag: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "adapter_config.json").write_text(
            json.dumps({"tag": tag}), encoding="utf-8"
        )
        (directory / "adapter_model.safetensors").write_bytes(tag.encode())

    def test_stage_one_receipt_records_fresh_init_and_contrast_only(self):
        row = self.manifest["stages"][0]
        with tempfile.TemporaryDirectory() as scratch:
            out_dir = Path(scratch) / "stage01_replay_refresh"
            self._fake_adapter(out_dir, "stage1")
            dataset = EXP / "data" / "lineage" / Path(row["dataset"]["file"]).name
            command = rz.stage_command(row, dataset, out_dir, None)
            receipt = rz.build_stage_receipt(
                row, dataset, out_dir, None, command, 1.5
            )
        self.assertEqual(receipt["warm_start"], "fresh_zero_init")
        self.assertTrue(receipt["fresh_init"])
        self.assertEqual(receipt["seed"], 42)
        self.assertTrue(receipt["seed_is_inherited_stage_constant"])
        # CONTRAST fields carry the original chain's hashes with the frozen
        # note; the produced hashes must be computed from disk and differ.
        contrast = receipt["original_produced_contrast"]
        self.assertEqual(
            contrast["adapter_weights_sha256"],
            row["produced"]["adapter_weights_sha256"],
        )
        self.assertEqual(contrast["note"], rz.CONTRAST_NOTE)
        self.assertNotEqual(
            receipt["produced"]["adapter_weights_sha256"],
            contrast["adapter_weights_sha256"],
        )

    def test_receipt_round_trips_through_authentication(self):
        rows = self.manifest["stages"]
        with tempfile.TemporaryDirectory() as scratch:
            warm_dir = Path(scratch) / "stage01_replay_refresh"
            out_dir = Path(scratch) / "stage02_designed160"
            self._fake_adapter(warm_dir, "stage1")
            self._fake_adapter(out_dir, "stage2")
            row = rows[1]
            dataset = EXP / "data" / "lineage" / Path(row["dataset"]["file"]).name
            command = rz.stage_command(row, dataset, out_dir, warm_dir)
            receipt = rz.build_stage_receipt(
                row, dataset, out_dir, warm_dir, command, 2.0
            )
            rz.authenticate_stage_receipt(receipt, row, out_dir, warm_dir)
            self.assertTrue(rz.receipt_matches_disk(receipt, out_dir))
            # A drifted seed in the receipt must refuse.
            tampered = dict(receipt)
            tampered["seed"] = 99
            with self.assertRaises(ValueError):
                rz.authenticate_stage_receipt(tampered, row, out_dir, warm_dir)
            # A drifted warm-start chain must refuse.
            self._fake_adapter(warm_dir, "stage1-tampered")
            with self.assertRaises(ValueError):
                rz.authenticate_stage_receipt(receipt, row, out_dir, warm_dir)

    def test_original_produced_hashes_are_never_used_as_verification(self):
        """receipt_matches_disk compares against the RECEIPT's produced
        hashes, never the manifest's original (blend-rooted) hashes."""
        row = self.manifest["stages"][0]
        with tempfile.TemporaryDirectory() as scratch:
            out_dir = Path(scratch) / "stage01_replay_refresh"
            self._fake_adapter(out_dir, "stage1")
            dataset = EXP / "data" / "lineage" / Path(row["dataset"]["file"]).name
            command = rz.stage_command(row, dataset, out_dir, None)
            receipt = rz.build_stage_receipt(row, dataset, out_dir, None, command, 0.1)
            # The receipt verifies its own produced bytes even though they
            # cannot possibly match the original chain's recorded hashes.
            self.assertTrue(rz.receipt_matches_disk(receipt, out_dir))
            forged = dict(receipt)
            forged["produced"] = {
                "adapter_config_sha256": row["produced"]["adapter_config_sha256"],
                "adapter_weights_sha256": row["produced"]["adapter_weights_sha256"],
                "adapter_weights_size": 1,
            }
            self.assertFalse(rz.receipt_matches_disk(forged, out_dir))


class TestDesignReceiptStagePlan(unittest.TestCase):
    def test_stage_plan_rewires_only_the_warm_start(self):
        manifest = rz.load_manifest()
        plan = gd.stage_plan(manifest)
        self.assertEqual(len(plan), 6)
        for entry, row in zip(plan, manifest["stages"]):
            with self.subTest(stage=row["stage"]):
                self.assertEqual(entry["dataset"]["sha256"], row["dataset"]["sha256"])
                self.assertEqual(entry["seed"], row["seed"])
                self.assertEqual(entry["hyperparameters"], row["hyperparameters"])
                self.assertEqual(entry["trainer"]["sha256"], row["trainer_sha256"])
                self.assertEqual(entry["original_warm_start"], row["warm_start"])
                if row["stage"] == 1:
                    self.assertEqual(entry["warm_start"], "fresh_zero_init")
                    self.assertEqual(entry["original_warm_start"], "root_adapter")
                else:
                    self.assertEqual(
                        entry["warm_start"], f"stage {row['stage'] - 1}"
                    )
                self.assertTrue(
                    entry["original_produced_contrast"]["contrast_only"]
                )
        # Stage 3's targeted overrides must survive into the plan.
        self.assertEqual(
            plan[2]["targeted_close_overrides"],
            {"target_close_kinds": ["u_execute", "u_induct"], "target_w_close": 1.0},
        )
        self.assertNotIn("targeted_close_overrides", plan[0])


if __name__ == "__main__":
    unittest.main()
