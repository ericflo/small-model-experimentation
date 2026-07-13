from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.gpu_runner import _positive_control_rows  # noqa: E402


class FreshPositiveControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(ROOT / "configs" / "default.yaml")

    @staticmethod
    def manifest(fingerprints=()) -> dict:
        return {
            "files": {
                "train": {"structural_fingerprints": list(fingerprints)},
                "validation": {"structural_fingerprints": []},
            }
        }

    def test_fresh_seed_73991_builds_exact_balanced_48_row_factorial(self) -> None:
        rows, receipt = _positive_control_rows(self.config, self.manifest())
        repeated, repeated_receipt = _positive_control_rows(self.config, self.manifest())
        self.assertEqual(rows, repeated)
        self.assertEqual(receipt, repeated_receipt)
        self.assertEqual(receipt["seed"], 73991)
        self.assertEqual(receipt["rows"], 48)
        self.assertEqual(receipt["cross_result_structural_overlap"], 0)
        self.assertEqual({row["split"] for row in rows}, {"setup_positive_control"})
        grid = Counter(
            (row["depth"], row["query_kind"], row["family"], row["template"])
            for row in rows
        )
        self.assertEqual(len(grid), 3 * 2 * 2 * 2)
        self.assertEqual(set(grid.values()), {2})
        self.assertEqual({depth for depth, *_ in grid}, {2, 3, 4})
        self.assertEqual({query for _, query, *_ in grid}, {"node", "checksum"})
        self.assertEqual(
            {family for _, _, family, _ in grid},
            set(self.config["substrate"]["train_families"]),
        )
        self.assertEqual(
            {template for _, _, _, template in grid},
            set(self.config["substrate"]["train_templates"]),
        )
        self.assertEqual(
            len({row["structural_fingerprint"] for row in rows}),
            len(rows),
        )

    def test_any_structural_overlap_with_result_data_fails_closed(self) -> None:
        rows, _ = _positive_control_rows(self.config, self.manifest())
        with self.assertRaisesRegex(RuntimeError, "overlap result data"):
            _positive_control_rows(
                self.config,
                self.manifest([rows[0]["structural_fingerprint"]]),
            )


if __name__ == "__main__":
    unittest.main()
