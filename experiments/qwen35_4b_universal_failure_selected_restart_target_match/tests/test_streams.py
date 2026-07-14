from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]


def load_json(name: str) -> dict:
    return json.loads((EXP / "data" / name).read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ExactExposureFreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = load_json("source_token_lengths.json")
        cls.manifest = load_json("stream_manifest.json")
        cls.receipt = load_json("stream_token_receipt.json")

    def test_source_measurement_uses_exact_trainer_and_clean_restarts(self) -> None:
        self.assertEqual(
            self.source["encoder_sha256"],
            "0cfb126feae6d73238c02362066229fff0b6a846625eed167d7681edde322cc4",
        )
        restart = self.source["sources"]["restart"]
        self.assertEqual(restart["rows"], 52)
        self.assertEqual(restart["totals"]["parent_prefix"], 0)
        self.assertEqual(restart["totals"]["forward"], 21253)
        self.assertEqual(restart["totals"]["nonzero_target"], 12218)
        self.assertEqual(restart["totals"]["absolute_loss_mass_x5"], 14438)

    def test_partition_is_disjoint_and_preserves_aligned_core(self) -> None:
        selection = self.manifest["selection"]
        core = set(selection["replay_core_source_indices"])
        filler = set(selection["candidate_replay_filler_source_indices"])
        control = set(selection["replay_control_source_indices"])
        self.assertEqual((len(core), len(filler), len(control)), (200, 68, 120))
        self.assertFalse(core & filler or core & control or filler & control)
        self.assertEqual(selection["shared_position_aligned_rows"], 200)
        self.assertFalse(self.manifest["targets_modified_for_matching"])
        self.assertFalse(self.manifest["rows_duplicated_for_matching"])
        self.assertFalse(self.manifest["rows_truncated_for_matching"])

    def test_final_streams_match_all_registered_exposure_axes(self) -> None:
        self.assertEqual(
            sha256_file(EXP / "data" / "stream_token_receipt.json"),
            "52a761ef8fd37f3eac88abf8f090013f571a47511daeb26820ca030201b1c170",
        )
        self.assertEqual(self.receipt["rows_per_arm"], 320)
        self.assertEqual(self.receipt["forward_tokens_per_arm"], 297731)
        self.assertEqual(self.receipt["nonzero_target_tokens_per_arm"], 126796)
        self.assertEqual(self.receipt["absolute_loss_mass_x5_per_arm"], 138164)
        self.assertEqual(self.receipt["skipped_rows"], 0)
        deltas = self.receipt["candidate_minus_control_spans"]
        for axis in self.receipt["match_axes"]:
            self.assertEqual(deltas[axis], 0)
        self.assertEqual(deltas["answer_target"], 0)
        self.assertEqual(deltas["close_target"], 0)
        self.assertEqual(deltas["parent_prefix"], 0)
        self.assertEqual(deltas["target_span"], 16414)

    def test_candidate_has_four_clean_restarts_per_skill(self) -> None:
        candidate = self.receipt["files"]["counterfactual_restart_candidate"]
        restart_kinds = {
            key: value
            for key, value in candidate["kinds"].items()
            if key.startswith("u_counterfactual_restart_")
        }
        self.assertEqual(len(restart_kinds), 13)
        self.assertEqual(set(restart_kinds.values()), {4})
        self.assertEqual(candidate["spans_per_epoch"]["parent_prefix"], 0)
        self.assertEqual((candidate["min_sequence_tokens"], candidate["max_sequence_tokens"]),
                         (128, 2991))


if __name__ == "__main__":
    unittest.main()
