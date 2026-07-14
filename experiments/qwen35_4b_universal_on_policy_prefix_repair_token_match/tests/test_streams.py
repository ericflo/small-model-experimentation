from __future__ import annotations

import hashlib
import importlib.util
import json
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
SCRIPT = EXP / "scripts" / "materialize_streams.py"
SPEC = importlib.util.spec_from_file_location("uop_materialize_streams", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ExactStreamFreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.outputs, cls.manifest = MODULE.build_outputs()
        cls.receipt = json.loads((EXP / "data" / "stream_token_receipt.json").read_text())

    def test_derived_bytes_and_manifest_are_frozen(self) -> None:
        for name, expected in self.outputs.items():
            self.assertEqual((EXP / "data" / name).read_bytes(), expected)
        expected_manifest = (
            json.dumps(self.manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode()
        self.assertEqual((EXP / "data" / "stream_manifest.json").read_bytes(), expected_manifest)
        self.assertEqual(hashlib.sha256(expected_manifest).hexdigest(),
                         "f836d0a192adfd1e85e4b3514b4854515b239be5faae0c33cebea46530593cd3")

    def test_partition_is_disjoint_balanced_and_exact_token_matched(self) -> None:
        selection = self.manifest["selection"]
        core = set(selection["replay_core_source_indices"])
        filler = set(selection["candidate_replay_filler_source_indices"])
        control = set(selection["replay_control_source_indices"])
        self.assertEqual((len(core), len(filler), len(control)), (200, 60, 120))
        self.assertFalse(core & filler or core & control or filler & control)
        self.assertEqual(selection["prefix_repair_rows_by_class"], {
            "bounded_induction": 10,
            "commit_serialization": 10,
            "declaration_operation": 10,
            "probe_scoring": 10,
            "repair_propagation": 10,
            "state_transition": 10,
        })
        self.assertEqual(selection["block_forward_tokens"], {
            "candidate_replay_filler": 28000,
            "prefix_repair": 76953,
            "replay_control": 104953,
            "shared_replay": 199360,
        })
        self.assertEqual(set(selection["estimated_arm_forward_tokens"].values()), {304313})

    def test_exact_token_receipt_records_zero_skips_and_masking(self) -> None:
        self.assertEqual(sha256_file(EXP / "data" / "source_token_lengths.json"),
                         "2ae6aded50fb4ad649bf69eea01e03aee58b73e58083276e2ab5f188b3ff654d")
        self.assertEqual(sha256_file(EXP / "data" / "stream_token_receipt.json"),
                         "eb08026ffcf82b8780819a26a522f04d69358ffdfd4797dd4c603dd1fbbe0cfc")
        self.assertEqual(self.receipt["rows_per_arm"], 320)
        self.assertEqual(self.receipt["forward_tokens_per_arm"], 304313)
        self.assertEqual(self.receipt["shared_position_aligned_rows"], 200)
        self.assertEqual(self.receipt["skipped_rows"], 0)
        files = {Path(row["path"]).name: row for row in self.receipt["files"]}
        candidate = files["prefix_repair_after_close.jsonl"]
        control = files["replay_after_close.jsonl"]
        self.assertEqual(candidate["spans_per_epoch"]["parent_prefix"], 47123)
        self.assertEqual(candidate["max_sequence_tokens"], 2991)
        self.assertEqual(control["max_sequence_tokens"], 2991)
        self.assertEqual(self.receipt["candidate_minus_control_spans"], {
            "answer_target": 528,
            "close_target": 0,
            "forward": 0,
            "masked_context": 33421,
            "parent_prefix": 47123,
            "prompt": -13702,
            "target_span": -33421,
            "think_target": -33949,
        })


if __name__ == "__main__":
    unittest.main()
