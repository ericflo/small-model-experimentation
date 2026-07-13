from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.mechanics import crossed_paired_bootstrap_interval, recurrent_compute_receipt  # noqa: E402


class MechanicsTests(unittest.TestCase):
    def test_recurrent_compute_receipt_counts_extra_loop_applications(self) -> None:
        receipt = recurrent_compute_receipt(
            sequence_tokens=600,
            total_layers=32,
            loop_layers=8,
            k=8,
        )
        self.assertEqual(receipt.base_layer_token_applications, 19_200)
        self.assertEqual(receipt.recurrent_layer_token_applications, 33_600)
        self.assertEqual(receipt.total_layer_token_applications, 52_800)
        with self.assertRaises(ValueError):
            recurrent_compute_receipt(sequence_tokens=1, total_layers=8, loop_layers=8, k=2)

    def test_crossed_bootstrap_uses_shared_task_axis_and_is_deterministic(self) -> None:
        records = {
            7411: {f"task-{item}": 1.0 for item in range(20)},
            7412: {f"task-{item}": 1.0 for item in range(20)},
            7413: {f"task-{item}": 0.5 for item in range(20)},
        }
        first = crossed_paired_bootstrap_interval(records, resamples=2000, seed=75301)
        second = crossed_paired_bootstrap_interval(records, resamples=2000, seed=75301)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first[0], 5 / 6)
        self.assertGreater(first[1], 0.0)

    def test_crossed_bootstrap_rejects_ragged_or_nonfinite_matrices(self) -> None:
        with self.assertRaisesRegex(ValueError, "identical task ids"):
            crossed_paired_bootstrap_interval(
                {7411: {"a": 1.0}, 7412: {"b": 1.0}},
                resamples=1000,
                seed=1,
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            crossed_paired_bootstrap_interval(
                {7411: {"a": float("nan")}},
                resamples=1000,
                seed=1,
            )


if __name__ == "__main__":
    unittest.main()
