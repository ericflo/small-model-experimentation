from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.receipt_contracts import validate_gate_lineage  # noqa: E402


def record(status: str, phase: str, digit: str) -> dict[str, str]:
    return {
        "path": f"/tmp/{status}.json",
        "sha256": digit * 64,
        "receipt_identity_sha256": str((int(digit) + 1) % 10) * 64,
        "status": status,
        "phase": phase,
    }


class GateLineageTests(unittest.TestCase):
    def test_pilot_and_full_require_exact_licensing_gates(self) -> None:
        smoke = record("MODEL_SMOKE_PASS", "g0", "1")
        promotion = record("PILOT_PROMOTION_READY", "pilot", "3")
        self.assertEqual(
            set(validate_gate_lineage({"model_smoke": smoke}, checkpoint_phase="pilot")),
            {"model_smoke"},
        )
        self.assertEqual(
            set(
                validate_gate_lineage(
                    {"model_smoke": smoke, "pilot_promotion": promotion},
                    checkpoint_phase="full",
                )
            ),
            {"model_smoke", "pilot_promotion"},
        )

    def test_missing_extra_or_malformed_lineage_fails_closed(self) -> None:
        smoke = record("MODEL_SMOKE_PASS", "g0", "1")
        with self.assertRaisesRegex(RuntimeError, "exactly"):
            validate_gate_lineage({}, checkpoint_phase="pilot")
        with self.assertRaisesRegex(RuntimeError, "exactly"):
            validate_gate_lineage({"model_smoke": smoke}, checkpoint_phase="full")
        broken = dict(smoke, sha256="not-a-digest")
        with self.assertRaisesRegex(RuntimeError, "invalid sha256"):
            validate_gate_lineage({"model_smoke": broken}, checkpoint_phase="pilot")


if __name__ == "__main__":
    unittest.main()
