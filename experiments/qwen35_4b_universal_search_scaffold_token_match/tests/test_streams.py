from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "materialize_streams.py"
SPEC = importlib.util.spec_from_file_location("search_scaffold_streams", SCRIPT)
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
            "c157fb135f0934375de3c36d3258b4d2621a09f9831f4eb9f1a8f5bb959c355d",
        )
        self.assertEqual(
            MODULE.sha256_bytes(self.outputs["scaffold_after_close.jsonl"]),
            "79a8d7c933a220b809447f144f07c2352f89f462198b07b64b30275cf8790b90",
        )
        self.assertEqual(
            self.manifest["selection"]["estimated_arm_forward_tokens"],
            {"replay_after_close": 286814, "scaffold_after_close": 286814},
        )
        self.assertTrue(all(row["rows"] == 320 for row in self.manifest["outputs"].values()))

    def test_scaffold_stage_mix_is_exact(self) -> None:
        self.assertEqual(
            self.manifest["selection"]["scaffold_rows_by_stage"],
            {
                "u_scaffold_apply": 16,
                "u_scaffold_execute": 16,
                "u_scaffold_fit": 16,
                "u_scaffold_reject": 16,
                "u_scaffold_search": 16,
            },
        )
        self.assertEqual(self.manifest["selection"]["block_forward_tokens"]["scaffold"], 33928)
        self.assertEqual(
            self.manifest["selection"]["block_forward_tokens"]["candidate_replay_filler"],
            53526,
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
        target_lines = self.outputs["scaffold_after_close.jsonl"].decode().splitlines()
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
            receipt["arm_data"]["scaffold_after_close"],
        )


if __name__ == "__main__":
    unittest.main()
