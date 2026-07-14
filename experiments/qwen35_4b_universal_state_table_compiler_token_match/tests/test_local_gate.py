from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_local.py"
SPEC = importlib.util.spec_from_file_location("state_table_check_local", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def payload(*, execute: float = 0.5, induct: float = 0.5, probe: float = 0.5) -> dict:
    return {
        "seed": 88008,
        "mix": "frozen",
        "summaries": {
            "candidate": {
                "accuracy": 0.65,
                "parse_rate": 0.90,
                "cap_contacts": 2,
                "per_kind": {
                    "u_execute": {"accuracy": execute},
                    "u_induct": {"accuracy": induct},
                    "u_probe": {"accuracy": probe},
                },
            }
        },
        "rows": [
            {"adapter": "candidate", "kind": "u_route", "parsed": "R1"},
            {"adapter": "candidate", "kind": "u_route", "parsed": "R2"},
        ],
    }


class LocalGateTests(unittest.TestCase):
    def test_abstention_spellings_are_detected(self) -> None:
        for value in (None, "None", "INSUFFICIENT", "null", "unknown", "N/A", "abstain"):
            with self.subTest(value=value):
                self.assertTrue(MODULE.is_abstention(value))

    def test_valid_route_ids_are_not_abstentions(self) -> None:
        for value in ("IX", "mer", "tool_3", "0"):
            with self.subTest(value=value):
                self.assertFalse(MODULE.is_abstention(value))

    def test_boundary_values_pass_all_frozen_checks(self) -> None:
        result = MODULE.evaluate(payload(), "candidate")
        self.assertTrue(result["passes"])
        self.assertTrue(all(result["checks"].values()))

    def test_execute_induct_and_probe_are_independent_required_checks(self) -> None:
        for field, kwargs in (
            ("execute_accuracy_at_least_0_50", {"execute": 0.0}),
            ("induct_accuracy_at_least_0_50", {"induct": 0.0}),
            ("probe_accuracy_at_least_0_50", {"probe": 0.0}),
        ):
            with self.subTest(field=field):
                result = MODULE.evaluate(payload(**kwargs), "candidate")
                self.assertFalse(result["passes"])
                self.assertFalse(result["checks"][field])

    def test_two_route_abstentions_fail(self) -> None:
        value = payload()
        value["rows"][0]["parsed"] = "INSUFFICIENT"
        value["rows"][1]["parsed"] = None
        result = MODULE.evaluate(value, "candidate")
        self.assertFalse(result["passes"])
        self.assertFalse(result["checks"]["no_repeated_feasible_route_abstention"])


if __name__ == "__main__":
    unittest.main()
