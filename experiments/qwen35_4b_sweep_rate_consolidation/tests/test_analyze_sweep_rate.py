import importlib.util
import json
import sys
import unittest
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, EXP / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ANALYZE = load_module("consolidation_analyze", "analyze_sweep_rate.py")


def committed_table() -> dict:
    return json.loads(
        (EXP / "runs" / "readings_table.json").read_text(encoding="utf-8")
    )


class ExactIntervalMathTests(unittest.TestCase):
    def test_beta_inv_round_trips_the_cdf(self) -> None:
        for q, a, b in ((0.025, 2, 5), (0.975, 3, 4), (0.5, 3, 5), (0.1, 1, 6)):
            x = ANALYZE.beta_inv(q, a, b)
            self.assertAlmostEqual(ANALYZE.beta_cdf(x, a, b), q, places=12)

    def test_clopper_pearson_2_of_6(self) -> None:
        ci = ANALYZE.clopper_pearson(2, 6)
        self.assertAlmostEqual(ci["low"], 0.043272, places=6)
        self.assertAlmostEqual(ci["high"], 0.777222, places=6)

    def test_clopper_pearson_boundary_zero_successes(self) -> None:
        ci = ANALYZE.clopper_pearson(0, 6)
        self.assertEqual(ci["low"], 0.0)
        # Exact closed form for k = 0: high = 1 - (alpha/2)^(1/n).
        self.assertAlmostEqual(ci["high"], 1.0 - 0.025 ** (1.0 / 6.0), places=6)

    def test_clopper_pearson_boundary_all_successes(self) -> None:
        ci = ANALYZE.clopper_pearson(6, 6)
        self.assertEqual(ci["high"], 1.0)
        # Exact closed form for k = n: low = (alpha/2)^(1/n).
        self.assertAlmostEqual(ci["low"], 0.025 ** (1.0 / 6.0), places=6)

    def test_clopper_pearson_symmetry(self) -> None:
        ci_two = ANALYZE.clopper_pearson(2, 6)
        ci_four = ANALYZE.clopper_pearson(4, 6)
        self.assertAlmostEqual(ci_two["low"], 1.0 - ci_four["high"], places=6)
        self.assertAlmostEqual(ci_two["high"], 1.0 - ci_four["low"], places=6)

    def test_clopper_pearson_rejects_invalid_counts(self) -> None:
        with self.assertRaises(ValueError):
            ANALYZE.clopper_pearson(7, 6)
        with self.assertRaises(ValueError):
            ANALYZE.clopper_pearson(-1, 6)
        with self.assertRaises(ValueError):
            ANALYZE.clopper_pearson(0, 0)

    def test_beta_posterior_2_of_6(self) -> None:
        post = ANALYZE.beta_posterior(2, 6)
        self.assertEqual((post["alpha"], post["beta"]), (3, 5))
        self.assertAlmostEqual(post["mean"], 0.375, places=6)
        self.assertAlmostEqual(post["credible_95"]["low"], 0.098988, places=6)
        self.assertAlmostEqual(post["credible_95"]["high"], 0.709579, places=6)


class CommittedAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.analysis = ANALYZE.build_analysis(committed_table())

    def test_sweep_rate_is_2_of_6(self) -> None:
        rate = self.analysis["sweep_rate"]
        self.assertEqual(rate["events"], 6)
        self.assertEqual(rate["passes"], 2)
        self.assertAlmostEqual(rate["rate"], 2 / 6, places=6)
        self.assertEqual(rate["pass_seeds"], [78154, 78157])
        self.assertEqual(rate["miss_seeds"], [78150, 78155, 78156, 78159])

    def test_blocker_frequencies(self) -> None:
        blockers = self.analysis["blockers"]
        self.assertEqual(blockers["miss_events"], 4)
        self.assertEqual(
            blockers["per_family_miss_counts"],
            {"menders": 4, "rites": 1, "warren": 1},
        )
        self.assertTrue(blockers["every_miss_includes_a_menders_draw"])

    def test_zero_strict_losses_across_all_events(self) -> None:
        blockers = self.analysis["blockers"]
        self.assertEqual(blockers["total_strict_losses_across_all_events"], 0)
        self.assertTrue(blockers["zero_strict_losses_across_all_events"])

    def test_base_draw_distribution_is_computed_not_assumed(self) -> None:
        families = self.analysis["base_draws"]["families"]
        # The intake note guessed "base rites>0 on 2 seeds"; the record says
        # rites is 0.0 on ALL SIX seeds and chronicle is the >0-on-2 family.
        self.assertEqual(families["rites"]["seeds_gt_zero"], 0)
        self.assertEqual(families["rites"]["max"], 0.0)
        self.assertEqual(families["chronicle"]["seeds_gt_zero"], 2)
        self.assertEqual(families["sirens"]["seeds_gt_zero"], 6)
        self.assertEqual(families["menders"]["max"], 0.0)
        self.assertIn("rites", self.analysis["base_draws"]["note"])

    def test_erratum_block(self) -> None:
        erratum = self.analysis["erratum"]
        self.assertIn("~50%", erratum["informal_claim"])
        self.assertEqual(erratum["informal_window"]["seeds"], [78154, 78155, 78156, 78157])
        self.assertEqual(erratum["informal_window"]["passes"], 2)
        self.assertEqual(erratum["informal_window"]["events"], 4)
        self.assertAlmostEqual(erratum["informal_window"]["rate"], 0.5, places=6)
        self.assertEqual(erratum["fifth_data_point_extension"]["events"], 5)
        self.assertAlmostEqual(
            erratum["fifth_data_point_extension"]["rate"], 0.4, places=6
        )
        self.assertEqual(erratum["corrected"]["passes"], 2)
        self.assertEqual(erratum["corrected"]["events"], 6)
        self.assertAlmostEqual(erratum["corrected"]["rate"], 2 / 6, places=6)
        omitted = {entry["seed"] for entry in erratum["omitted_by_window"]}
        self.assertEqual(omitted, {78150, 78159})
        documents = [
            entry["document"] for entry in erratum["documents_carrying_informal_figure"]
        ]
        self.assertIn("knowledge/synthesis.md", documents)
        self.assertIn("knowledge/experiment_brief.json", documents)
        self.assertIn(
            "experiments/qwen35_4b_goal_gate_confirmation/README.md", documents
        )
        self.assertGreaterEqual(len(documents), 6)

    def test_rederivation_is_byte_identical(self) -> None:
        first = ANALYZE.serialize(ANALYZE.build_analysis(committed_table()))
        second = ANALYZE.serialize(ANALYZE.build_analysis(committed_table()))
        self.assertEqual(first, second)
        committed = (EXP / "runs" / "sweep_rate_analysis.json").read_text(
            encoding="utf-8"
        )
        self.assertEqual(first, committed)

    def test_no_wall_clock_stamps(self) -> None:
        committed = (EXP / "runs" / "sweep_rate_analysis.json").read_text(
            encoding="utf-8"
        )
        for token in ("generated_on", "timestamp", "2026-"):
            self.assertNotIn(token, committed)


class TableGuardTests(unittest.TestCase):
    def test_families_drift_in_table_aborts(self) -> None:
        table = committed_table()
        table["families"] = ["chronicle"]
        with self.assertRaises(SystemExit):
            ANALYZE.build_analysis(table)

    def test_duplicate_seed_in_table_aborts(self) -> None:
        table = committed_table()
        table["readings"].append(dict(table["readings"][0]))
        with self.assertRaises(SystemExit):
            ANALYZE.build_analysis(table)


if __name__ == "__main__":
    unittest.main()
