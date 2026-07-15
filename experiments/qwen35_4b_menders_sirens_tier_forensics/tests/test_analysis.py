import importlib.util
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


ANALYZE = load_module("forensics_analyze", "analyze_constants.py")

FAMILIES = ANALYZE.FAMILIES


def row(arm: str, seed: int, tier: str, receipt: str, **scores) -> dict:
    families = {f: 0.1 for f in FAMILIES}
    families.update(scores)
    return {
        "experiment": "exp",
        "receipt": receipt,
        "receipt_sha256": "0" * 64,
        "tier": tier,
        "think_budget": None,
        "seed": seed,
        "arm": arm,
        "aggregate": 0.5,
        "families": families,
    }


class CleanRowsTests(unittest.TestCase):
    def test_summary_files_are_dropped(self) -> None:
        rows = [
            row("base", 1, "quick", "experiments/e/runs/summary.json"),
            row("base", 1, "quick", "experiments/e/runs/quick_seed1_base.json"),
        ]
        kept = ANALYZE.clean_rows(rows)
        self.assertEqual(len(kept), 1)
        self.assertIn("quick_seed1_base", kept[0]["receipt"])

    def test_delta_blocks_outside_unit_interval_are_dropped(self) -> None:
        rows = [
            row(
                "arm",
                1,
                "quick",
                "experiments/e/runs/quick_seed1_arm.json",
                menders=-0.125,
            )
        ]
        self.assertEqual(ANALYZE.clean_rows(rows), [])

    def test_identical_blocks_deduplicate(self) -> None:
        rows = [
            row("arm", 1, "quick", "experiments/e/runs/quick_seed1_arm.json"),
            row("arm", 1, "quick", "experiments/e/runs/quick_seed1_arm2.json"),
        ]
        self.assertEqual(len(ANALYZE.clean_rows(rows)), 1)

    def test_different_scores_are_both_kept(self) -> None:
        rows = [
            row("a", 1, "quick", "experiments/e/runs/quick_seed1_a.json"),
            row(
                "b",
                1,
                "quick",
                "experiments/e/runs/quick_seed1_b.json",
                menders=0.2,
            ),
        ]
        self.assertEqual(len(ANALYZE.clean_rows(rows)), 2)


class ArmClassTests(unittest.TestCase):
    def test_base_labels(self) -> None:
        for label in ("base", "base0", "base_reserialized"):
            self.assertEqual(
                ANALYZE.arm_class(
                    row(label, 1, "quick", "experiments/e/runs/x.json")
                ),
                "base",
            )
        self.assertEqual(
            ANALYZE.arm_class(
                row("merged", 1, "quick", "experiments/e/runs/x.json")
            ),
            "treated",
        )


class DistTests(unittest.TestCase):
    def test_dist_counts_special_values(self) -> None:
        stats = ANALYZE.dist([0.0, 0.5, 0.5, 1.0])
        self.assertEqual(stats["n"], 4)
        self.assertEqual(stats["zero"], 1)
        self.assertEqual(stats["one"], 1)
        self.assertEqual(stats["exactly_half"], 2)
        self.assertEqual(stats["min"], 0.0)
        self.assertEqual(stats["max"], 1.0)


if __name__ == "__main__":
    unittest.main()
