import importlib.util
import json
import math
import statistics
import sys
import tempfile
import unittest
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, EXP / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CHECK = load_module("episode_check_local", "check_local.py")

AXIS_KINDS = sorted(CHECK.AXIS_KINDS)
RETENTION_KINDS = sorted(CHECK.RETENTION_KINDS)


def uniform_axis(n: int) -> dict[str, int]:
    return {kind: n for kind in AXIS_KINDS}


def synthetic_payload(
    axis_correct: dict[str, dict[str, int]],
    retention_correct: dict[str, list[int]] | None = None,
    retention_parsed: dict[str, list[int]] | None = None,
    cap_contacts: dict[str, list[int]] | None = None,
) -> dict:
    """Build a receipt: per arm, per kind, the first N rows are correct.

    Retention values are given per screen (a list aligned to SCREEN_SEEDS).
    """
    retention_totals = retention_correct or {}
    parsed_totals = retention_parsed or {}
    caps = cap_contacts or {}
    rows = []
    for label in CHECK.ARMS:
        for kind in AXIS_KINDS:
            correct_n = axis_correct[label].get(kind, 10)
            for index in range(CHECK.AXIS_PER_KIND):
                rows.append({
                    "adapter": label,
                    "screen": CHECK.SEED,
                    "task_id": f"axis{CHECK.SEED}_{kind}_{index}",
                    "kind": kind,
                    "parsed": "x",
                    "correct": bool(index < correct_n),
                    "cap_contact": False,
                })
        for screen_index, screen in enumerate(CHECK.SCREEN_SEEDS):
            per_screen_correct = retention_totals.get(
                label, [CHECK.RETENTION_ROWS_PER_SCREEN] * len(CHECK.SCREEN_SEEDS)
            )[screen_index]
            per_screen_parsed = parsed_totals.get(
                label, [CHECK.RETENTION_ROWS_PER_SCREEN] * len(CHECK.SCREEN_SEEDS)
            )[screen_index]
            per_screen_caps = caps.get(label, [0] * len(CHECK.SCREEN_SEEDS))[
                screen_index
            ]
            remaining_correct = per_screen_correct
            remaining_unparsed = CHECK.RETENTION_ROWS_PER_SCREEN - per_screen_parsed
            remaining_caps = per_screen_caps
            retention_rows = []
            for kind in RETENTION_KINDS:
                for index in range(CHECK.RETENTION_PER_KIND):
                    retention_rows.append({
                        "adapter": label,
                        "screen": screen,
                        "task_id": f"ret{screen}_{kind}_{index}",
                        "kind": kind,
                        "parsed": "x",
                        "correct": False,
                        "cap_contact": False,
                    })
            for row in retention_rows:
                if remaining_correct > 0 and row["parsed"] is not None:
                    row["correct"] = True
                    remaining_correct -= 1
            for row in reversed(retention_rows):
                if remaining_unparsed > 0 and not row["correct"]:
                    row["parsed"] = None
                    remaining_unparsed -= 1
            for row in retention_rows:
                if remaining_caps > 0 and row["kind"] == "u_state":
                    row["cap_contact"] = True
                    remaining_caps -= 1
            rows.extend(retention_rows)
    return {
        "seed": CHECK.SEED,
        "screen_seeds": list(CHECK.SCREEN_SEEDS),
        "rows_per_arm": CHECK.ROWS_PER_ARM,
        "labels": list(CHECK.ARMS),
        "rows": rows,
    }


class PromotionGateTests(unittest.TestCase):
    def test_clean_win_promotes(self) -> None:
        payload = synthetic_payload({
            CHECK.PARENT: uniform_axis(10),
            CHECK.CONTROL: uniform_axis(10),
            CHECK.CANDIDATE: uniform_axis(14),
        })
        result = CHECK.evaluate_promotion(payload)
        self.assertEqual(result["promoted"], "feedloop_state")
        self.assertEqual(result["eligible"], ["feedloop_state"])
        self.assertEqual(result["axis_kind_wins"], 2)
        self.assertTrue(all(result["checks"].values()))

    def test_axis_total_tie_with_replay_fails(self) -> None:
        payload = synthetic_payload({
            CHECK.PARENT: uniform_axis(8),
            CHECK.CONTROL: uniform_axis(12),
            CHECK.CANDIDATE: uniform_axis(12),
        })
        result = CHECK.evaluate_promotion(payload)
        self.assertIsNone(result["promoted"])
        self.assertEqual(result["eligible"], [])
        self.assertFalse(result["checks"]["axis_total_strictly_beats_replay"])

    def test_single_kind_win_fails_breadth(self) -> None:
        candidate = {AXIS_KINDS[0]: 18, AXIS_KINDS[1]: 10}
        payload = synthetic_payload({
            CHECK.PARENT: uniform_axis(10),
            CHECK.CONTROL: uniform_axis(10),
            CHECK.CANDIDATE: candidate,
        })
        result = CHECK.evaluate_promotion(payload)
        # Total 28 strictly beats both 20s, but one kind is an exact tie.
        self.assertTrue(result["checks"]["axis_total_strictly_beats_parent"])
        self.assertTrue(result["checks"]["axis_total_strictly_beats_replay"])
        self.assertEqual(result["axis_kind_wins"], 1)
        self.assertFalse(result["checks"]["axis_kind_wins_both_of_2"])
        self.assertIsNone(result["promoted"])

    def test_kind_tie_with_one_control_fails(self) -> None:
        # Candidate beats the parent on both kinds but ties the replay
        # control on one: ties fail, per kind, against BOTH controls.
        candidate = {AXIS_KINDS[0]: 12, AXIS_KINDS[1]: 14}
        payload = synthetic_payload({
            CHECK.PARENT: uniform_axis(8),
            CHECK.CONTROL: {AXIS_KINDS[0]: 12, AXIS_KINDS[1]: 8},
            CHECK.CANDIDATE: candidate,
        })
        result = CHECK.evaluate_promotion(payload)
        self.assertFalse(result["kind_wins"][AXIS_KINDS[0]])
        self.assertTrue(result["kind_wins"][AXIS_KINDS[1]])
        self.assertFalse(result["checks"]["axis_kind_wins_both_of_2"])
        self.assertIsNone(result["promoted"])

    def test_pooled_retention_band_edge_passes(self) -> None:
        # Parent pooled mean 80; candidate pooled mean exactly 75 = 80 - 5.
        payload = synthetic_payload(
            {
                CHECK.PARENT: uniform_axis(10),
                CHECK.CONTROL: uniform_axis(10),
                CHECK.CANDIDATE: uniform_axis(14),
            },
            retention_correct={
                CHECK.PARENT: [80, 80, 80],
                CHECK.CONTROL: [80, 80, 80],
                CHECK.CANDIDATE: [75, 75, 75],
            },
        )
        result = CHECK.evaluate_promotion(payload)
        self.assertTrue(
            result["checks"]["retention_pooled_correct_within_5_of_parent"]
        )
        self.assertEqual(result["promoted"], "feedloop_state")

    def test_pooled_retention_one_beyond_band_fails(self) -> None:
        payload = synthetic_payload(
            {
                CHECK.PARENT: uniform_axis(10),
                CHECK.CONTROL: uniform_axis(10),
                CHECK.CANDIDATE: uniform_axis(14),
            },
            retention_correct={
                CHECK.PARENT: [80, 80, 80],
                CHECK.CONTROL: [80, 80, 80],
                CHECK.CANDIDATE: [75, 75, 74],  # pooled mean 74.67 < 75
            },
        )
        result = CHECK.evaluate_promotion(payload)
        self.assertFalse(
            result["checks"]["retention_pooled_correct_within_5_of_parent"]
        )
        self.assertIsNone(result["promoted"])

    def test_fractional_pooled_mean_boundary_is_exact(self) -> None:
        # Parent screens sum 241 (mean 80.33..). The band edge is mean-5, so
        # a candidate sum of 226 passes and 225 fails — no float ambiguity.
        for candidate_screens, expected in (
            ([76, 75, 75], True),
            ([75, 75, 75], False),
        ):
            payload = synthetic_payload(
                {
                    CHECK.PARENT: uniform_axis(10),
                    CHECK.CONTROL: uniform_axis(10),
                    CHECK.CANDIDATE: uniform_axis(14),
                },
                retention_correct={
                    CHECK.PARENT: [80, 81, 80],
                    CHECK.CONTROL: [80, 80, 80],
                    CHECK.CANDIDATE: candidate_screens,
                },
            )
            result = CHECK.evaluate_promotion(payload)
            self.assertIs(
                result["checks"]["retention_pooled_correct_within_5_of_parent"],
                expected,
                candidate_screens,
            )

    def test_single_bad_screen_can_be_absorbed_by_the_pooled_mean(self) -> None:
        # A -9 single-screen dip absorbed by two flat screens: pooled mean
        # delta is -3, inside the band. Means, not per-screen.
        payload = synthetic_payload(
            {
                CHECK.PARENT: uniform_axis(10),
                CHECK.CONTROL: uniform_axis(10),
                CHECK.CANDIDATE: uniform_axis(14),
            },
            retention_correct={
                CHECK.PARENT: [80, 80, 80],
                CHECK.CONTROL: [80, 80, 80],
                CHECK.CANDIDATE: [71, 80, 80],
            },
        )
        result = CHECK.evaluate_promotion(payload)
        self.assertTrue(
            result["checks"]["retention_pooled_correct_within_5_of_parent"]
        )
        self.assertEqual(result["promoted"], "feedloop_state")

    def test_pooled_cap_contact_regression_beyond_band_fails(self) -> None:
        payload = synthetic_payload(
            {
                CHECK.PARENT: uniform_axis(10),
                CHECK.CONTROL: uniform_axis(10),
                CHECK.CANDIDATE: uniform_axis(14),
            },
            cap_contacts={CHECK.CANDIDATE: [4, 3, 3]},  # pooled mean 3.33 > 3
        )
        result = CHECK.evaluate_promotion(payload)
        self.assertFalse(
            result["checks"]["retention_pooled_cap_contacts_within_3_of_parent"]
        )
        self.assertIsNone(result["promoted"])

    def test_pooled_parsed_regression_beyond_band_fails(self) -> None:
        payload = synthetic_payload(
            {
                CHECK.PARENT: uniform_axis(10),
                CHECK.CONTROL: uniform_axis(10),
                CHECK.CANDIDATE: uniform_axis(14),
            },
            retention_correct={CHECK.CANDIDATE: [80, 80, 80]},
            retention_parsed={CHECK.CANDIDATE: [100, 100, 102]},  # mean 100.67
        )
        result = CHECK.evaluate_promotion(payload)
        self.assertFalse(
            result["checks"]["retention_pooled_parsed_within_3_of_parent"]
        )
        self.assertIsNone(result["promoted"])

    def test_no_absolute_per_kind_floor_exists(self) -> None:
        payload = synthetic_payload(
            {
                CHECK.PARENT: uniform_axis(10),
                CHECK.CONTROL: uniform_axis(10),
                CHECK.CANDIDATE: uniform_axis(14),
            },
            retention_correct={
                CHECK.PARENT: [8, 8, 8],
                CHECK.CONTROL: [8, 8, 8],
                CHECK.CANDIDATE: [8, 8, 8],
            },
        )
        result = CHECK.evaluate_promotion(payload)
        self.assertTrue(result["no_absolute_per_kind_floors"])
        self.assertEqual(result["promoted"], "feedloop_state")

    def test_pooled_sd_machinery_matches_statistics(self) -> None:
        values = {"a": [80, 76, 82], "b": [70, 71, 75]}
        expected = math.sqrt(
            (statistics.variance(values["a"]) + statistics.variance(values["b"])) / 2
        )
        self.assertAlmostEqual(CHECK.pooled_sd(values), expected)
        self.assertAlmostEqual(
            CHECK.sample_sd([80, 76, 82]), statistics.stdev([80, 76, 82])
        )
        with self.assertRaises(ValueError):
            CHECK.pooled_sd({"a": [80]})

    def test_recovery_writer_schema_parity(self) -> None:
        payload = synthetic_payload({
            CHECK.PARENT: uniform_axis(10),
            CHECK.CONTROL: uniform_axis(10),
            CHECK.CANDIDATE: uniform_axis(14),
        })
        base_result = CHECK.evaluate_promotion(payload)
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "local.json"
            raw = (json.dumps(payload, sort_keys=True) + "\n").encode()
            receipt.write_bytes(raw)
            design = Path(directory) / "design.json"
            design.write_bytes(b"{}\n")
            finalized = CHECK.finalize_promotion(
                dict(base_result), receipt, raw, design_receipt=design
            )
        self.assertEqual(
            set(finalized) - set(base_result),
            {
                "experiment_id",
                "local_receipt",
                "local_receipt_sha256",
                "design_receipt_sha256",
                "backend",
                "aggregate_seed",
                "aggregate_seed_open",
                "benchmark_data_read",
            },
        )
        self.assertEqual(finalized["experiment_id"], EXP.name)
        self.assertEqual(finalized["backend"], "vllm_merged_composite")
        self.assertEqual(finalized["aggregate_seed"], 78151)
        self.assertTrue(finalized["aggregate_seed_open"])
        self.assertFalse(finalized["benchmark_data_read"])
        eval_source = (EXP / "scripts" / "eval_local_vllm.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("evaluate_promotion", eval_source)
        self.assertIn("finalize_promotion", eval_source)

    def test_frozen_gate_constants(self) -> None:
        self.assertEqual(CHECK.SEED, 88026)
        self.assertEqual(CHECK.SCREEN_SEEDS, (88027, 88028, 88030))
        self.assertEqual(CHECK.AGGREGATE_SEED, 78151)
        self.assertEqual(CHECK.ROWS_PER_ARM, 352)
        self.assertEqual(CHECK.AXIS_ROWS, 40)
        self.assertEqual(CHECK.AXIS_PER_KIND, 20)
        self.assertEqual(CHECK.RETENTION_ROWS_PER_SCREEN, 104)
        self.assertEqual(
            CHECK.ARMS,
            ("hygiene_explore_parent", "replay_ctl", "feedloop_state"),
        )
        self.assertEqual(CHECK.AXIS_KINDS, frozenset({"u_feedloop", "u_statechain"}))
        self.assertEqual(CHECK.RETENTION_CORRECT_BAND, 5)
        self.assertEqual(CHECK.RETENTION_CAP_BAND, 3)
        self.assertEqual(CHECK.RETENTION_PARSED_BAND, 3)
        self.assertIn("BUDGET", CHECK.ABSTENTION_ANSWERS)
        self.assertIn("INSUFFICIENT", CHECK.ABSTENTION_ANSWERS)


if __name__ == "__main__":
    unittest.main()
