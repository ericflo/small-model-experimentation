"""Diversity-at-scale: the whole point of this cell (vs replaying the 504-row bet).

These assert the generator produces GENUINELY diverse data at scale so the ladder
tests real scaling, not overfit on a saturated generator:

  * >= 50 program families are exercised in a 5000-row sample,
  * >= 300 distinct NORMALIZED WHY reasoning templates in a 10000-row sample,
  * >= 95% unique programs in a 20000-row build (the deduped corpus is 100%, and
    the raw no-dedup draw stays high, proving real capacity, not dedup masking).

They generate real corpora (each row truth-audited inside generate_curriculum),
so they are the slow tests; run under scripts/run.py --smoke.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))

import gen_why_scale_curriculum as gen  # noqa: E402

SEED = gen.CONSTRUCTION_SEED


class TestDiversityTargets(unittest.TestCase):
    def test_at_least_50_families_in_5000_sample(self):
        div = gen.diversity_report(gen.generate_curriculum(SEED, 5000))
        self.assertGreaterEqual(div["distinct_families"], 50, msg=div["families_exercised"])
        # every declared family must actually be reachable at 5000 rows.
        self.assertEqual(div["distinct_families"], len(gen.FAMILIES),
                         msg=sorted(set(gen.FAMILY_NAMES) - set(div["families_exercised"])))

    def test_at_least_300_normalized_why_templates_in_10000(self):
        div = gen.diversity_report(gen.generate_curriculum(SEED, 10000))
        self.assertGreaterEqual(div["distinct_normalized_why_templates"], 300,
                                msg=div["distinct_normalized_why_templates"])

    def test_at_least_95pct_unique_programs_in_20000(self):
        rows = gen.generate_curriculum(SEED, 20000)
        div = gen.diversity_report(rows)
        self.assertEqual(div["rows"], 20000)
        self.assertGreaterEqual(div["unique_program_pct"], 95.0, msg=div["unique_program_pct"])
        # raw (no-dedup) draw capacity: high uniqueness proves the diversity is
        # genuine and not an artifact of the dedup loop.
        raw = gen.sample_raw_programs(SEED, 20000)
        raw_pct = 100.0 * len(set(raw)) / len(raw)
        self.assertGreaterEqual(raw_pct, 75.0, msg=f"raw uniqueness {raw_pct:.2f}%")

    def test_normalization_removes_numbers_and_vars(self):
        # two rationales differing only in var/number collapse to one pattern.
        a = gen.normalize_why("acc starts at 0 so nothing is tallied before the loop begins")
        b = gen.normalize_why("tally starts at 5 so nothing is tallied before the loop begins")
        self.assertEqual(a, b)
        c = gen.normalize_why("acc begins empty at 0 so the very step sets its true starting value")
        self.assertNotEqual(a, c)


if __name__ == "__main__":
    unittest.main()
