"""The frozen TWO-SIDED pooled_k3 retention bands, exercised on screen sums.

With three screens per arm every pooled-mean band is evaluated in exact
integer arithmetic on the screen SUMS: correct within +-15, cap contacts
within +-9, parsed within +-9. Both boundary directions are exercised —
the bands are a drift screen, so a candidate that IMPROVES pooled correct
by more than the band also fails promotion.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))

import check_local as cl  # noqa: E402


KINDS = sorted(cl.RETENTION_KINDS)


def build_rows(
    label: str,
    screen: int,
    *,
    correct: int,
    unparsed: int = 0,
    cap_contacts: int = 0,
) -> list[dict]:
    """One arm-screen of 104 rows with the requested exact counts."""
    assert 0 <= correct <= cl.RETENTION_ROWS_PER_SCREEN
    assert correct + unparsed <= cl.RETENTION_ROWS_PER_SCREEN
    rows = []
    for index in range(cl.RETENTION_ROWS_PER_SCREEN):
        kind = KINDS[index % len(KINDS)]
        is_correct = index < correct
        is_unparsed = correct <= index < correct + unparsed
        rows.append(
            {
                "adapter": label,
                "screen": screen,
                "task_id": f"ret{screen}_uc2_{index:05d}",
                "kind": kind,
                "surface": "letters",
                "expected": "X",
                "parsed": None if is_unparsed else ("X" if is_correct else "Y"),
                "correct": is_correct,
                "cap_contact": index < cap_contacts,
                "n_sampled_tokens": 10,
                "n_thinking_tokens": 5,
                "n_answer_tokens": 2,
            }
        )
    return rows


def build_payload(
    parent_correct: tuple[int, int, int] = (60, 60, 60),
    candidate_correct: tuple[int, int, int] = (60, 60, 60),
    parent_unparsed: tuple[int, int, int] = (0, 0, 0),
    candidate_unparsed: tuple[int, int, int] = (0, 0, 0),
    parent_caps: tuple[int, int, int] = (0, 0, 0),
    candidate_caps: tuple[int, int, int] = (0, 0, 0),
) -> dict:
    rows: list[dict] = []
    for index, screen in enumerate(cl.SCREEN_SEEDS):
        rows.extend(
            build_rows(
                cl.PARENT,
                screen,
                correct=parent_correct[index],
                unparsed=parent_unparsed[index],
                cap_contacts=parent_caps[index],
            )
        )
        rows.extend(
            build_rows(
                cl.CANDIDATE,
                screen,
                correct=candidate_correct[index],
                unparsed=candidate_unparsed[index],
                cap_contacts=candidate_caps[index],
            )
        )
    return {
        "screen_seeds": list(cl.SCREEN_SEEDS),
        "rows_per_arm": cl.ROWS_PER_ARM,
        "labels": list(cl.ARMS),
        "rows": rows,
    }


class TestRetentionBands(unittest.TestCase):
    def test_identical_arms_promote(self):
        result = cl.evaluate_promotion(build_payload())
        self.assertEqual(result["promoted"], cl.CANDIDATE)
        self.assertTrue(all(result["checks"].values()))

    def test_correct_down_exactly_fifteen_on_sums_still_promotes(self):
        result = cl.evaluate_promotion(
            build_payload(candidate_correct=(55, 55, 55))
        )
        self.assertEqual(result["promoted"], cl.CANDIDATE)

    def test_correct_down_sixteen_on_sums_fails(self):
        result = cl.evaluate_promotion(
            build_payload(candidate_correct=(55, 55, 54))
        )
        self.assertIsNone(result["promoted"])
        self.assertFalse(
            result["checks"]["retention_pooled_correct_within_5_of_parent"]
        )

    def test_correct_up_exactly_fifteen_on_sums_still_promotes(self):
        result = cl.evaluate_promotion(
            build_payload(candidate_correct=(65, 65, 65))
        )
        self.assertEqual(result["promoted"], cl.CANDIDATE)

    def test_correct_up_sixteen_on_sums_fails_the_two_sided_band(self):
        result = cl.evaluate_promotion(
            build_payload(candidate_correct=(65, 65, 66))
        )
        self.assertIsNone(result["promoted"])
        self.assertFalse(
            result["checks"]["retention_pooled_correct_within_5_of_parent"]
        )

    def test_cap_contacts_up_nine_on_sums_promotes_and_ten_fails(self):
        promoted = cl.evaluate_promotion(
            build_payload(candidate_caps=(3, 3, 3))
        )
        self.assertEqual(promoted["promoted"], cl.CANDIDATE)
        failed = cl.evaluate_promotion(
            build_payload(candidate_caps=(3, 3, 4))
        )
        self.assertIsNone(failed["promoted"])
        self.assertFalse(
            failed["checks"]["retention_pooled_cap_contacts_within_3_of_parent"]
        )

    def test_cap_contacts_down_ten_on_sums_fails_the_two_sided_band(self):
        failed = cl.evaluate_promotion(
            build_payload(parent_caps=(4, 3, 3), candidate_caps=(0, 0, 0))
        )
        self.assertIsNone(failed["promoted"])

    def test_parsed_down_nine_on_sums_promotes_and_ten_fails(self):
        promoted = cl.evaluate_promotion(
            build_payload(candidate_unparsed=(3, 3, 3))
        )
        self.assertEqual(promoted["promoted"], cl.CANDIDATE)
        failed = cl.evaluate_promotion(
            build_payload(candidate_unparsed=(3, 3, 4))
        )
        self.assertIsNone(failed["promoted"])
        self.assertFalse(
            failed["checks"]["retention_pooled_parsed_within_3_of_parent"]
        )

    def test_bands_are_declared_two_sided(self):
        result = cl.evaluate_promotion(build_payload())
        self.assertTrue(result["bands_two_sided"])
        self.assertEqual(result["adjudication_protocol"], "pooled_k3")

    def test_wrong_label_order_fails_closed(self):
        payload = build_payload()
        payload["labels"] = list(reversed(payload["labels"]))
        with self.assertRaises(ValueError):
            cl.evaluate_promotion(payload)

    def test_missing_parsed_key_fails_closed(self):
        payload = build_payload()
        del payload["rows"][0]["parsed"]
        with self.assertRaises(ValueError):
            cl.evaluate_promotion(payload)

    def test_task_id_prefix_drift_fails_closed(self):
        payload = build_payload()
        payload["rows"][0]["task_id"] = "axis88060_uc2_00000"
        with self.assertRaises(ValueError):
            cl.evaluate_promotion(payload)

    def test_arms_must_share_task_ids(self):
        payload = build_payload()
        for row in payload["rows"]:
            if row["adapter"] == cl.CANDIDATE and row["screen"] == cl.SCREEN_SEEDS[0]:
                row["task_id"] = row["task_id"] + "x"
                break
        with self.assertRaises(ValueError):
            cl.evaluate_promotion(payload)

    def test_normalize_answer_is_the_frozen_rule(self):
        self.assertEqual(cl.normalize_answer("  a   >  b ; c  "), "a>b;c")
        self.assertEqual(cl.normalize_answer("x \n y"), "x y")

    def test_band_constants_are_frozen(self):
        self.assertEqual(cl.RETENTION_CORRECT_BAND, 5)
        self.assertEqual(cl.RETENTION_CAP_BAND, 3)
        self.assertEqual(cl.RETENTION_PARSED_BAND, 3)
        self.assertEqual(cl.ROWS_PER_ARM, 312)


if __name__ == "__main__":
    unittest.main()
