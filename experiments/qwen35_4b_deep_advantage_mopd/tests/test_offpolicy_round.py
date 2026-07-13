from __future__ import annotations

import sys
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))

from train_offpolicy_round import _probe_units  # noqa: E402


class OffpolicyRoundTests(unittest.TestCase):
    def test_pressure_probe_has_frozen_six_two_geometry(self):
        units = [
            {"id": f"a-{index:02d}", "role": "anchor"}
            for index in range(20)
        ] + [
            {"id": f"z-{index:02d}", "role": "capability"}
            for index in range(60)
        ]
        probe = _probe_units(units)
        self.assertEqual(sum(row["role"] == "capability" for row in probe), 6)
        self.assertEqual(sum(row["role"] == "anchor" for row in probe), 2)
        self.assertEqual([row["id"] for row in probe[:6]], [f"z-{i:02d}" for i in range(6)])
        self.assertEqual([row["id"] for row in probe[6:]], ["a-00", "a-01"])

    def test_pressure_probe_fails_closed_on_missing_role(self):
        with self.assertRaisesRegex(ValueError, "6 capability and 2 anchor"):
            _probe_units([{"id": f"c-{index}", "role": "capability"} for index in range(8)])


if __name__ == "__main__":
    unittest.main()
