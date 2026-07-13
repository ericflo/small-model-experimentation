from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.mechanics import (
    bag_unroll,
    carry_unroll,
    crossed_paired_bootstrap_interval,
    gate_reachability,
    hierarchical_paired_bootstrap_interval,
    last_mean_aggregate,
    paired_bootstrap_interval,
    recurrent_compute_receipt,
)


class MechanicsTests(unittest.TestCase):
    def test_only_carry_composes_prior_state(self) -> None:
        transition = lambda state, step: state * 2 + step
        carry = carry_unroll(1, transition, 5)
        bag = bag_unroll(1, transition, 5)
        self.assertEqual(carry, [1, 4, 11, 26, 57])
        self.assertEqual(bag, [1, 4, 5, 6, 7])

    def test_k1_is_identity_for_both_modes_and_aggregator(self) -> None:
        transition = lambda state, step: state + step
        self.assertEqual(carry_unroll(9, transition, 1), [9])
        self.assertEqual(bag_unroll(9, transition, 1), [9])
        self.assertEqual(last_mean_aggregate([9.0], 0.9), 9.0)

    def test_compute_is_exactly_matched(self) -> None:
        carry = recurrent_compute_receipt(sequence_tokens=600, total_layers=32, loop_layers=8, k=8)
        bag = recurrent_compute_receipt(sequence_tokens=600, total_layers=32, loop_layers=8, k=8)
        self.assertEqual(carry, bag)
        self.assertEqual(carry.total_layer_token_applications, 52_800)

    def test_paired_bootstrap_has_expected_sign(self) -> None:
        mean, lower, upper = paired_bootstrap_interval([1.0] * 8 + [0.0] * 2, resamples=2000, seed=7)
        self.assertAlmostEqual(mean, 0.8)
        self.assertGreater(lower, 0.0)
        self.assertGreaterEqual(upper, mean)

    def test_hierarchical_bootstrap_resamples_seed_and_task(self) -> None:
        mean, lower, upper = hierarchical_paired_bootstrap_interval(
            {1: [1.0] * 20, 2: [1.0] * 20, 3: [0.5] * 20},
            resamples=2000,
            seed=9,
        )
        self.assertAlmostEqual(mean, 5 / 6)
        self.assertGreater(lower, 0)
        self.assertGreaterEqual(upper, mean)

    def test_crossed_bootstrap_uses_one_common_task_axis(self) -> None:
        mean, lower, upper = crossed_paired_bootstrap_interval(
            {
                1: {f"task-{item}": 1.0 for item in range(20)},
                2: {f"task-{item}": 1.0 for item in range(20)},
                3: {f"task-{item}": 0.5 for item in range(20)},
            },
            resamples=2000,
            seed=11,
        )
        self.assertAlmostEqual(mean, 5 / 6)
        self.assertGreater(lower, 0)
        self.assertGreaterEqual(upper, mean)

    def test_crossed_bootstrap_rejects_ragged_task_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "identical task ids"):
            crossed_paired_bootstrap_interval(
                {1: {"a": 1.0}, 2: {"b": 1.0}},
                resamples=1000,
                seed=1,
            )

    def test_gate_reachability_fails_closed(self) -> None:
        self.assertTrue(gate_reachability(0.4, 0.05)["reachable"])
        self.assertFalse(gate_reachability(0.98, 0.05)["reachable"])


if __name__ == "__main__":
    unittest.main()
