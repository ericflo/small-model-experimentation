from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "materialize_streams.py"
SPEC = importlib.util.spec_from_file_location("close_weight_streams", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class StreamConstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.outputs, cls.manifest = MODULE.build_outputs()

    def test_frozen_bytes_rows_and_forward_tokens(self) -> None:
        self.assertEqual(
            MODULE.sha256_bytes(self.outputs["replay_repeat.jsonl"]),
            "6ec82e2989eda5f37f51ba0b13e2c8326c8107110c4b193f3ac65621779e81d4",
        )
        self.assertEqual(
            MODULE.sha256_bytes(self.outputs["targeted_standard.jsonl"]),
            "12fc613bb31a46bcea9acd49b26467656704aa3b3418dab8d920adf057d14f00",
        )
        self.assertEqual(
            self.manifest["selection"]["estimated_arm_forward_tokens"],
            {"replay_repeat": 286814, "targeted_standard": 286814},
        )
        self.assertTrue(all(row["rows"] == 320 for row in self.manifest["outputs"].values()))

    def test_target_rows_are_fresh_and_kind_balanced(self) -> None:
        parent = set(self.manifest["parent_exclusion"]["designed_source_indices"])
        target = set(self.manifest["selection"]["target_designed_source_indices"])
        self.assertEqual(len(parent), 160)
        self.assertEqual(len(target), 80)
        self.assertTrue(parent.isdisjoint(target))
        self.assertEqual(
            self.manifest["selection"]["target_rows_by_kind"],
            {"u_execute": 40, "u_induct": 40},
        )

    def test_common_replay_occupies_the_same_200_slots(self) -> None:
        replay_lines = self.outputs["replay_repeat.jsonl"].decode().splitlines()
        target_lines = self.outputs["targeted_standard.jsonl"].decode().splitlines()
        identical_positions = [
            index for index, (left, right) in enumerate(
                zip(replay_lines, target_lines, strict=True)
            )
            if left == right
        ]
        self.assertEqual(len(identical_positions), 200)

    def test_standard_and_treatment_share_one_authenticated_file(self) -> None:
        receipt = json.loads(
            (SCRIPT.parents[1] / "data" / "stream_token_receipt.json").read_text()
        )
        self.assertTrue(receipt["standard_close_byte_identity"])
        self.assertEqual(receipt["arm_data"]["standard_xi"], receipt["arm_data"]["close_xi"])


if __name__ == "__main__":
    unittest.main()
