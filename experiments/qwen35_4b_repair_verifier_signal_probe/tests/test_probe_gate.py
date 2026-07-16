"""Unit tests for the frozen probe readout: CI math, the ordered two-state
consequence partition at its exact boundaries, the readings schema, and the
fail-closed receipt-layout validation. No model is ever loaded."""

from __future__ import annotations

import copy
import math
import sys
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))

import check_local as cl  # noqa: E402


def synthetic_rows(
    correct_by_arm: dict[str, set[int]],
    cap_by_arm: dict[str, set[int]] | None = None,
) -> list[dict]:
    """Build a full 400-row graded table with the frozen layout: 25 items
    per formalism, per-formalism 13/12 A-splits (100/100 overall), and
    identical task/expected/surface maps across arms."""
    cap_by_arm = cap_by_arm or {arm: set() for arm in cl.ARMS}
    rows = []
    for arm in cl.ARMS:
        for index in range(cl.PROBE_ROWS):
            block = index // cl.PER_FORMALISM
            offset = index % cl.PER_FORMALISM
            n_a = 13 if block % 2 == 0 else 12
            expected = "A" if offset < n_a else "B"
            correct = index in correct_by_arm[arm]
            rows.append(
                {
                    "arm": arm,
                    "task_id": f"probe{cl.CONSTRUCTION_SEED}_{index:03d}",
                    "kind": cl.PROBE_KIND,
                    "surface": cl.FORMALISMS[block],
                    "expected": expected,
                    "parsed": expected if correct else ("B" if expected == "A" else "A"),
                    "correct": correct,
                    "correct_before_normalization": correct,
                    "n_sampled_tokens": 64,
                    "n_thinking_tokens": 32 if arm == "think" else 0,
                    "n_answer_tokens": 4,
                    "cap_contact": index in cap_by_arm[arm],
                    "finish_reason": "stop",
                    "completion_sha256": "0" * 64,
                }
            )
    return rows


def synthetic_receipt(
    think_correct: int,
    nothink_correct: int = 100,
    think_cap_contacts: int = 0,
) -> dict:
    correct_by_arm = {
        "think": set(range(think_correct)),
        "nothink": set(range(nothink_correct)),
    }
    cap_by_arm = {
        "think": set(range(think_cap_contacts)),
        "nothink": set(),
    }
    return {
        "seed": cl.CONSTRUCTION_SEED,
        "rows_per_arm": cl.PROBE_ROWS,
        "labels": list(cl.ARMS),
        "rows": synthetic_rows(correct_by_arm, cap_by_arm),
    }


class TestBinomialMath(unittest.TestCase):
    def test_cdf_endpoints_and_monotonicity(self) -> None:
        self.assertEqual(cl.binomial_cdf(-1, 10, 0.5), 0.0)
        self.assertEqual(cl.binomial_cdf(10, 10, 0.5), 1.0)
        self.assertAlmostEqual(cl.binomial_cdf(5, 10, 0.0), 1.0)
        self.assertAlmostEqual(cl.binomial_cdf(5, 10, 1.0), 0.0)
        values = [cl.binomial_cdf(5, 10, p / 20) for p in range(21)]
        self.assertTrue(all(a >= b - 1e-12 for a, b in zip(values, values[1:])))

    def test_cdf_matches_direct_sum(self) -> None:
        self.assertAlmostEqual(cl.binomial_cdf(1, 2, 0.5), 0.75, places=12)
        self.assertAlmostEqual(cl.binomial_cdf(0, 3, 0.5), 0.125, places=12)

    def test_clopper_pearson_closed_form_edges(self) -> None:
        # k=0: lower is 0 and upper is 1-(alpha/2)^(1/n); k=n mirrors it.
        n, alpha = 200, 0.05
        low, high = cl.clopper_pearson_interval(0, n)
        self.assertEqual(low, 0.0)
        self.assertAlmostEqual(high, 1 - (alpha / 2) ** (1 / n), places=9)
        low, high = cl.clopper_pearson_interval(n, n)
        self.assertAlmostEqual(low, (alpha / 2) ** (1 / n), places=9)
        self.assertEqual(high, 1.0)

    def test_clopper_pearson_symmetry(self) -> None:
        for k in (10, 87, 130):
            low, high = cl.clopper_pearson_interval(k, 200)
            mirrored_low, mirrored_high = cl.clopper_pearson_interval(200 - k, 200)
            self.assertAlmostEqual(low, 1 - mirrored_high, places=9)
            self.assertAlmostEqual(high, 1 - mirrored_low, places=9)

    def test_clopper_pearson_coverage_definition(self) -> None:
        # At the returned bounds the tail probabilities equal alpha/2.
        k, n = 130, 200
        low, high = cl.clopper_pearson_interval(k, n)
        self.assertAlmostEqual(1 - cl.binomial_cdf(k - 1, n, low), 0.025, places=9)
        self.assertAlmostEqual(cl.binomial_cdf(k, n, high), 0.025, places=9)
        self.assertLess(low, k / n)
        self.assertGreater(high, k / n)

    def test_interval_rejects_bad_inputs(self) -> None:
        with self.assertRaises(ValueError):
            cl.clopper_pearson_interval(-1, 200)
        with self.assertRaises(ValueError):
            cl.clopper_pearson_interval(201, 200)
        with self.assertRaises(ValueError):
            cl.binomial_cdf(1, 2, 1.5)


class TestConsequencePartition(unittest.TestCase):
    def test_exact_threshold_boundary_130_of_200(self) -> None:
        low, high = cl.clopper_pearson_interval(130, 200)
        self.assertEqual(
            cl.signal_verdict(130 / 200, low, high), "SIGNAL_PRESENT"
        )
        self.assertGreaterEqual(130 / 200, cl.SIGNAL_MIN_ACCURACY)

    def test_one_below_threshold_129_of_200(self) -> None:
        low, high = cl.clopper_pearson_interval(129, 200)
        # CI excludes 0.5, but the accuracy floor fails: ABSENT.
        self.assertTrue(cl.ci_excludes_chance(low, high))
        self.assertEqual(cl.signal_verdict(129 / 200, low, high), "SIGNAL_ABSENT")

    def test_ci_edge_high_accuracy_wide_interval(self) -> None:
        # A CI that includes 0.5 forces ABSENT even above the accuracy floor.
        self.assertEqual(
            cl.signal_verdict(0.66, 0.49, 0.80), "SIGNAL_ABSENT"
        )
        self.assertEqual(
            cl.signal_verdict(0.66, 0.51, 0.80), "SIGNAL_PRESENT"
        )

    def test_ci_boundary_touching_half_counts_as_included(self) -> None:
        self.assertFalse(cl.ci_excludes_chance(0.5, 0.8))
        self.assertFalse(cl.ci_excludes_chance(0.2, 0.5))
        self.assertTrue(cl.ci_excludes_chance(0.5000001, 0.8))

    def test_partition_is_total_and_two_state(self) -> None:
        for k in range(0, 201, 7):
            low, high = cl.clopper_pearson_interval(k, 200)
            verdict = cl.signal_verdict(k / 200, low, high)
            self.assertIn(verdict, cl.VERDICTS)

    def test_partition_is_ordered_in_correct_count(self) -> None:
        verdicts = []
        for k in range(0, 201):
            low, high = cl.clopper_pearson_interval(k, 200)
            verdicts.append(cl.signal_verdict(k / 200, low, high))
        # ABSENT... then PRESENT...: exactly one switch, at k=130.
        self.assertEqual(verdicts.index("SIGNAL_PRESENT"), 130)
        self.assertTrue(all(v == "SIGNAL_ABSENT" for v in verdicts[:130]))
        self.assertTrue(all(v == "SIGNAL_PRESENT" for v in verdicts[130:]))

    def test_verdict_rejects_non_finite_inputs(self) -> None:
        with self.assertRaises(ValueError):
            cl.signal_verdict(math.nan, 0.5, 0.6)

    def test_frozen_statements(self) -> None:
        self.assertEqual(set(cl.CONSEQUENCES), set(cl.VERDICTS))
        self.assertIn("on-policy episode charter is fundable", cl.CONSEQUENCES["SIGNAL_PRESENT"])
        self.assertIn("C29-class dissociation", cl.CONSEQUENCES["SIGNAL_PRESENT"])
        self.assertIn(
            "execution-based fix-verification", cl.CONSEQUENCES["SIGNAL_PRESENT"]
        )
        self.assertIn(
            "execution-based fix-verification", cl.CONSEQUENCES["SIGNAL_ABSENT"]
        )
        self.assertIn("the on-policy class closes for menders", cl.CONSEQUENCES["SIGNAL_ABSENT"])
        self.assertIn(
            "demonstrated-not-confirmed", cl.CONSEQUENCES["SIGNAL_ABSENT"]
        )

    def test_frozen_cap_scope_constants(self) -> None:
        self.assertEqual(cl.CAP_SCOPE_THRESHOLD, 0.20)
        self.assertIn("possibly budget-limited", cl.CAP_SCOPE_NOTE)
        self.assertIn("1,024-token cap", cl.CAP_SCOPE_NOTE)
        self.assertIn("no budget scoping applies", cl.CAP_NO_SCOPE_NOTE)


class TestEvaluateProbe(unittest.TestCase):
    def test_signal_present_readout(self) -> None:
        result = cl.evaluate_probe(synthetic_receipt(think_correct=150))
        self.assertEqual(result["outcome"], "PROBE_READ_COMPLETE")
        self.assertIsNone(result["promoted"])
        consequence = result["readings"]["consequence"]
        self.assertEqual(consequence["verdict"], "SIGNAL_PRESENT")
        self.assertEqual(consequence["statement"], cl.CONSEQUENCES["SIGNAL_PRESENT"])
        self.assertEqual(consequence["think_correct"], 150)
        think = result["readings"]["per_arm"]["think"]
        self.assertEqual(think["correct"], 150)
        self.assertAlmostEqual(think["2afc_accuracy"], 0.75)
        self.assertTrue(think["ci95_excludes_chance"])

    def test_signal_absent_at_the_guess_floor(self) -> None:
        result = cl.evaluate_probe(synthetic_receipt(think_correct=100))
        consequence = result["readings"]["consequence"]
        self.assertEqual(consequence["verdict"], "SIGNAL_ABSENT")
        think = result["readings"]["per_arm"]["think"]
        self.assertFalse(think["ci95_excludes_chance"])

    def test_boundary_130_promotes_the_signal_verdict(self) -> None:
        result = cl.evaluate_probe(synthetic_receipt(think_correct=130))
        self.assertEqual(
            result["readings"]["consequence"]["verdict"], "SIGNAL_PRESENT"
        )
        result = cl.evaluate_probe(synthetic_receipt(think_correct=129))
        self.assertEqual(
            result["readings"]["consequence"]["verdict"], "SIGNAL_ABSENT"
        )

    def test_nothink_is_descriptive_only(self) -> None:
        # A nothink arm at ceiling never flips an ABSENT think verdict.
        result = cl.evaluate_probe(
            synthetic_receipt(think_correct=100, nothink_correct=200)
        )
        self.assertEqual(
            result["readings"]["consequence"]["verdict"], "SIGNAL_ABSENT"
        )
        descriptive = result["readings"]["nothink_descriptive"]
        self.assertIs(descriptive["gating"], False)
        self.assertAlmostEqual(descriptive["nothink_accuracy"], 1.0)

    def test_cap_scope_annotates_budget_limited_absent_readings(self) -> None:
        # SIGNAL_ABSENT with cap contacts above 20% of items -> scoped.
        result = cl.evaluate_probe(
            synthetic_receipt(think_correct=100, think_cap_contacts=41)
        )
        consequence = result["readings"]["consequence"]
        self.assertEqual(consequence["verdict"], "SIGNAL_ABSENT")
        diagnostic = consequence["cap_contact_diagnostic"]
        self.assertEqual(diagnostic["think_cap_contacts"], 41)
        self.assertAlmostEqual(diagnostic["think_cap_contact_rate"], 0.205)
        self.assertIs(diagnostic["budget_limited_scope_applies"], True)
        self.assertEqual(diagnostic["note"], cl.CAP_SCOPE_NOTE)

    def test_cap_scope_boundary_is_strictly_greater_than_20_percent(self) -> None:
        # Exactly 20% (40/200) does NOT trigger the scope.
        result = cl.evaluate_probe(
            synthetic_receipt(think_correct=100, think_cap_contacts=40)
        )
        diagnostic = result["readings"]["consequence"]["cap_contact_diagnostic"]
        self.assertAlmostEqual(diagnostic["think_cap_contact_rate"], 0.20)
        self.assertIs(diagnostic["budget_limited_scope_applies"], False)
        self.assertEqual(diagnostic["note"], cl.CAP_NO_SCOPE_NOTE)

    def test_cap_scope_never_applies_to_signal_present(self) -> None:
        result = cl.evaluate_probe(
            synthetic_receipt(think_correct=150, think_cap_contacts=60)
        )
        consequence = result["readings"]["consequence"]
        self.assertEqual(consequence["verdict"], "SIGNAL_PRESENT")
        diagnostic = consequence["cap_contact_diagnostic"]
        self.assertIs(diagnostic["budget_limited_scope_applies"], False)
        self.assertEqual(diagnostic["note"], cl.CAP_NO_SCOPE_NOTE)

    def test_cap_scope_never_creates_a_third_verdict_state(self) -> None:
        for think_correct, cap_contacts in ((100, 0), (100, 200), (150, 200)):
            result = cl.evaluate_probe(
                synthetic_receipt(
                    think_correct=think_correct,
                    think_cap_contacts=cap_contacts,
                )
            )
            self.assertIn(
                result["readings"]["consequence"]["verdict"], cl.VERDICTS
            )

    def test_readings_schema(self) -> None:
        result = cl.evaluate_probe(synthetic_receipt(think_correct=140))
        for arm in cl.ARMS:
            reading = result["readings"]["per_arm"][arm]
            self.assertEqual(reading["rows"], cl.PROBE_ROWS)
            self.assertEqual(set(reading["per_formalism"]), set(cl.FORMALISMS))
            for formalism in cl.FORMALISMS:
                self.assertEqual(
                    reading["per_formalism"][formalism]["n"], cl.PER_FORMALISM
                )
            bias = reading["position_bias"]
            self.assertEqual(bias["a_correct_items"]["n"], 100)
            self.assertEqual(bias["b_correct_items"]["n"], 100)
            self.assertAlmostEqual(
                bias["accuracy_gap_a_minus_b"],
                bias["a_correct_items"]["accuracy"]
                - bias["b_correct_items"]["accuracy"],
            )
            low, high = reading["ci95_exact"]
            self.assertLessEqual(low, reading["2afc_accuracy"])
            self.assertGreaterEqual(high, reading["2afc_accuracy"])
            self.assertAlmostEqual(
                reading["cap_contact_rate"],
                reading["cap_contacts"] / reading["rows"],
            )

    def test_position_bias_reading_counts_by_expected_letter(self) -> None:
        # Make exactly the A-correct items correct in the think arm.
        receipt = synthetic_receipt(think_correct=0)
        for row in receipt["rows"]:
            if row["arm"] == "think" and row["expected"] == "A":
                row["correct"] = True
                row["parsed"] = "A"
                row["correct_before_normalization"] = True
        result = cl.evaluate_probe(receipt)
        bias = result["readings"]["per_arm"]["think"]["position_bias"]
        self.assertEqual(bias["a_correct_items"]["correct"], 100)
        self.assertEqual(bias["b_correct_items"]["correct"], 0)
        self.assertAlmostEqual(bias["accuracy_gap_a_minus_b"], 1.0)


class TestReceiptLayoutFailsClosed(unittest.TestCase):
    def check_rejects(self, mutate) -> None:
        receipt = synthetic_receipt(think_correct=140)
        mutate(receipt)
        with self.assertRaises(ValueError):
            cl.evaluate_probe(receipt)

    def test_rejects_wrong_seed(self) -> None:
        self.check_rejects(lambda r: r.update(seed=77150))

    def test_rejects_arm_order_change(self) -> None:
        self.check_rejects(lambda r: r.update(labels=["nothink", "think"]))

    def test_rejects_missing_rows(self) -> None:
        self.check_rejects(lambda r: r["rows"].pop())

    def test_rejects_unbalanced_positions(self) -> None:
        def flip(receipt: dict) -> None:
            for row in receipt["rows"]:
                if row["task_id"].endswith("_000"):
                    row["expected"] = "B"

        self.check_rejects(flip)

    def test_rejects_task_map_disagreement_across_arms(self) -> None:
        def swap(receipt: dict) -> None:
            think_rows = [r for r in receipt["rows"] if r["arm"] == "think"]
            a_row = next(r for r in think_rows if r["expected"] == "A")
            b_row = next(r for r in think_rows if r["expected"] == "B")
            # Swap ids only: per-arm balance holds, but the task->expected
            # map now disagrees with the nothink arm's.
            a_row["task_id"], b_row["task_id"] = b_row["task_id"], a_row["task_id"]

        self.check_rejects(swap)

    def test_rejects_unknown_kind(self) -> None:
        def poison(receipt: dict) -> None:
            receipt["rows"][0]["kind"] = "u_feedloop"

        self.check_rejects(poison)

    def test_rejects_correct_without_parse(self) -> None:
        def poison(receipt: dict) -> None:
            row = receipt["rows"][0]
            row["parsed"] = None
            row["correct"] = True

        self.check_rejects(poison)

    def test_rejects_non_boolean_correct(self) -> None:
        def poison(receipt: dict) -> None:
            receipt["rows"][0]["correct"] = 1

        self.check_rejects(poison)


class TestNormalization(unittest.TestCase):
    def test_frozen_definition(self) -> None:
        self.assertEqual(cl.normalize_answer("  A  "), "A")
        self.assertEqual(cl.normalize_answer("a > b ; c"), "a>b;c")
        self.assertEqual(cl.normalize_answer("B\n"), "B")
        self.assertEqual(
            cl.ANSWER_NORMALIZATION["function"], "check_local.normalize_answer"
        )


class TestFinalizeProbe(unittest.TestCase):
    def test_shared_writer_fields(self) -> None:
        raw = b"{}"
        result = cl.finalize_probe(
            cl.evaluate_probe(synthetic_receipt(think_correct=140)),
            EXP / "runs" / "local" / "probe.json",
            raw,
        )
        self.assertEqual(result["experiment_id"], EXP.name)
        self.assertIsNone(result["aggregate_seed"])
        self.assertIs(result["aggregate_seed_open"], False)
        self.assertIs(result["benchmark_data_read"], False)
        self.assertEqual(result["backend"], "vllm_merged_composite")
        self.assertEqual(len(result["local_receipt_sha256"]), 64)
        self.assertEqual(len(result["design_receipt_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
