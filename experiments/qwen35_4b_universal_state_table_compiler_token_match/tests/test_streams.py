from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "materialize_streams.py"
SPEC = importlib.util.spec_from_file_location("state_table_streams", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class StreamConstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.outputs, cls.manifest = MODULE.build_outputs()

    def test_frozen_bytes_rows_and_forward_tokens(self) -> None:
        self.assertEqual(
            MODULE.sha256_bytes(self.outputs["replay_after_close.jsonl"]),
            "2727e29a7c18e551ed9defe21b7f4e4009c7e6399ac1b2376deb7a4c609ba2b5",
        )
        self.assertEqual(
            MODULE.sha256_bytes(self.outputs["state_table_after_close.jsonl"]),
            "8e1b8fdcc349275ad31b2b1af16fa26384cd433de93cbd7a24d0686e07151355",
        )
        self.assertEqual(
            self.manifest["selection"]["estimated_arm_forward_tokens"],
            {"replay_after_close": 286814, "state_table_after_close": 286814},
        )
        self.assertTrue(all(row["rows"] == 320 for row in self.manifest["outputs"].values()))

    def test_state_table_stage_mix_is_exact(self) -> None:
        self.assertEqual(
            self.manifest["selection"]["curriculum_rows_by_stage"],
            {
                "u_state_table_commit": 20,
                "u_state_table_execute": 20,
                "u_state_table_repair": 20,
                "u_state_table_score": 20,
            },
        )
        self.assertEqual(self.manifest["selection"]["block_forward_tokens"]["curriculum"], 48806)
        self.assertEqual(
            self.manifest["selection"]["block_forward_tokens"]["candidate_replay_filler"],
            38648,
        )

    def test_replay_partition_is_disjoint_and_inherited(self) -> None:
        selection = self.manifest["selection"]
        core = set(selection["replay_core_source_indices"])
        filler = set(selection["candidate_replay_filler_source_indices"])
        control = set(selection["replay_control_source_indices"])
        self.assertEqual((len(core), len(filler), len(control)), (200, 40, 120))
        self.assertFalse(core & filler or core & control or filler & control)
        self.assertEqual(
            self.manifest["predecessor_partition"]["sha256"],
            "abf8b5055e68c0fb2bb6e32a29f7be3b3677a0dd179e77397647777a2aa0966f",
        )

    def test_common_replay_occupies_exactly_the_same_200_slots(self) -> None:
        replay_lines = self.outputs["replay_after_close.jsonl"].decode().splitlines()
        target_lines = self.outputs["state_table_after_close.jsonl"].decode().splitlines()
        identical_positions = [
            index
            for index, (left, right) in enumerate(zip(replay_lines, target_lines, strict=True))
            if left == right
        ]
        self.assertEqual(len(identical_positions), 200)

    def test_token_receipt_authenticates_zero_skip_exact_match(self) -> None:
        receipt = json.loads(
            (SCRIPT.parents[1] / "data" / "stream_token_receipt.json").read_text()
        )
        self.assertTrue(receipt["position_aligned_streams"])
        self.assertEqual(receipt["rows_per_arm"], 320)
        self.assertEqual(receipt["forward_tokens_per_arm"], 286814)
        self.assertEqual(receipt["skipped_rows"], 0)
        self.assertNotEqual(
            receipt["arm_data"]["replay_after_close"],
            receipt["arm_data"]["state_table_after_close"],
        )


if __name__ == "__main__":
    unittest.main()
