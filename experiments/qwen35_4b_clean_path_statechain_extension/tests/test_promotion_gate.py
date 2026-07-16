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


CHECK = load_module("statechain_check_local", "check_local.py")

AXIS_SURFACES = list(CHECK.AXIS_SURFACES)
RETENTION_KINDS = sorted(CHECK.RETENTION_KINDS)


def synthetic_payload(
    axis_correct: dict[str, int],
    retention_correct: dict[str, list[int]] | None = None,
    retention_parsed: dict[str, list[int]] | None = None,
    cap_contacts: dict[str, list[int]] | None = None,
    axis_surface_order: list[str] | None = None,
) -> dict:
    """Build a receipt: per arm, the first N axis rows are correct.

    Axis rows are all ``u_statechain``, 10 per formalism. Retention values
    are given per screen (a list aligned to SCREEN_SEEDS).
    """
    retention_totals = retention_correct or {}
    parsed_totals = retention_parsed or {}
    caps = cap_contacts or {}
    surfaces = axis_surface_order or AXIS_SURFACES
    rows = []
    for label in CHECK.ARMS:
        correct_n = axis_correct[label]
        axis_index = 0
        for surface in surfaces:
            for index in range(CHECK.AXIS_PER_SURFACE):
                rows.append({
                    "adapter": label,
                    "screen": CHECK.SEED,
                    "task_id": f"axis{CHECK.SEED}_{surface}_{index}",
                    "kind": "u_statechain",
                    "surface": surface,
                    "parsed": "x",
                    "correct": bool(axis_index < correct_n),
                    "cap_contact": False,
                })
                axis_index += 1
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
            CHECK.PARENT: 20,
            CHECK.CONTROL: 20,
            CHECK.CANDIDATE: 28,
        })
        result = CHECK.evaluate_promotion(payload)
        self.assertEqual(result["promoted"], "statechain_clean")
        self.assertEqual(result["eligible"], ["statechain_clean"])
        self.assertTrue(all(result["checks"].values()))
        self.assertTrue(result["single_kind_dose_no_per_kind_split"])

    def test_axis_total_tie_with_replay_fails(self) -> None:
        payload = synthetic_payload({
            CHECK.PARENT: 16,
            CHECK.CONTROL: 24,
            CHECK.CANDIDATE: 24,
        })
        result = CHECK.evaluate_promotion(payload)
        self.assertIsNone(result["promoted"])
        self.assertEqual(result["eligible"], [])
        self.assertFalse(result["checks"]["axis_total_strictly_beats_replay"])
        self.assertTrue(result["checks"]["axis_total_strictly_beats_parent"])

    def test_axis_total_tie_with_parent_fails(self) -> None:
        payload = synthetic_payload({
            CHECK.PARENT: 24,
            CHECK.CONTROL: 16,
            CHECK.CANDIDATE: 24,
        })
        result = CHECK.evaluate_promotion(payload)
        self.assertIsNone(result["promoted"])
        self.assertFalse(result["checks"]["axis_total_strictly_beats_parent"])
        self.assertTrue(result["checks"]["axis_total_strictly_beats_replay"])

    def test_one_over_both_controls_promotes(self) -> None:
        # Single-kind dose: a +1 TOTAL margin is enough; there is no
        # per-kind (and no per-formalism) split to also satisfy.
        payload = synthetic_payload({
            CHECK.PARENT: 22,
            CHECK.CONTROL: 23,
            CHECK.CANDIDATE: 24,
        })
        result = CHECK.evaluate_promotion(payload)
        self.assertEqual(result["promoted"], "statechain_clean")

    def test_no_per_surface_gate_exists(self) -> None:
        # The candidate's 24 correct rows are concentrated in the first
        # surfaces (the last formalism scores 0) — the gate must still
        # promote on the strict TOTAL because per-surface results are
        # reported, never gated.
        payload = synthetic_payload({
            CHECK.PARENT: 20,
            CHECK.CONTROL: 20,
            CHECK.CANDIDATE: 24,
        })
        result = CHECK.evaluate_promotion(payload)
        candidate_axis = result["summaries"][CHECK.CANDIDATE]["axis"]
        last_surface = AXIS_SURFACES[-1]
        self.assertEqual(candidate_axis["per_surface_correct"][last_surface], 0)
        self.assertEqual(result["promoted"], "statechain_clean")
        self.assertTrue(result["per_surface_reported_not_gated"])

    def test_pooled_retention_band_edge_passes(self) -> None:
        # Parent pooled mean 80; candidate pooled mean exactly 75 = 80 - 5.
        payload = synthetic_payload(
            {
                CHECK.PARENT: 20,
                CHECK.CONTROL: 20,
                CHECK.CANDIDATE: 28,
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
        self.assertEqual(result["promoted"], "statechain_clean")

    def test_pooled_retention_one_beyond_band_fails(self) -> None:
        payload = synthetic_payload(
            {
                CHECK.PARENT: 20,
                CHECK.CONTROL: 20,
                CHECK.CANDIDATE: 28,
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

    def test_replay_band_binds_independently_of_parent_band(self) -> None:
        # The lifecycle-15 failure mode: passing the parent band while
        # falling beyond the REPLAY band must still fail.
        payload = synthetic_payload(
            {
                CHECK.PARENT: 20,
                CHECK.CONTROL: 20,
                CHECK.CANDIDATE: 28,
            },
            retention_correct={
                CHECK.PARENT: [78, 78, 78],
                CHECK.CONTROL: [85, 85, 85],
                CHECK.CANDIDATE: [76, 76, 76],
            },
        )
        result = CHECK.evaluate_promotion(payload)
        self.assertTrue(
            result["checks"]["retention_pooled_correct_within_5_of_parent"]
        )
        self.assertFalse(
            result["checks"]["retention_pooled_correct_within_5_of_replay"]
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
                    CHECK.PARENT: 20,
                    CHECK.CONTROL: 20,
                    CHECK.CANDIDATE: 28,
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
                CHECK.PARENT: 20,
                CHECK.CONTROL: 20,
                CHECK.CANDIDATE: 28,
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
        self.assertEqual(result["promoted"], "statechain_clean")

    def test_pooled_cap_contact_regression_beyond_band_fails(self) -> None:
        payload = synthetic_payload(
            {
                CHECK.PARENT: 20,
                CHECK.CONTROL: 20,
                CHECK.CANDIDATE: 28,
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
                CHECK.PARENT: 20,
                CHECK.CONTROL: 20,
                CHECK.CANDIDATE: 28,
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
                CHECK.PARENT: 20,
                CHECK.CONTROL: 20,
                CHECK.CANDIDATE: 28,
            },
            retention_correct={
                CHECK.PARENT: [8, 8, 8],
                CHECK.CONTROL: [8, 8, 8],
                CHECK.CANDIDATE: [8, 8, 8],
            },
        )
        result = CHECK.evaluate_promotion(payload)
        self.assertTrue(result["no_absolute_per_kind_floors"])
        self.assertEqual(result["promoted"], "statechain_clean")

    def test_axis_surface_imbalance_is_rejected(self) -> None:
        # An axis file that lost its 10-per-formalism balance must abort.
        skewed = [AXIS_SURFACES[0]] * len(AXIS_SURFACES)
        payload = synthetic_payload(
            {
                CHECK.PARENT: 20,
                CHECK.CONTROL: 20,
                CHECK.CANDIDATE: 28,
            },
            axis_surface_order=skewed,
        )
        with self.assertRaises(ValueError):
            CHECK.evaluate_promotion(payload)

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
            CHECK.PARENT: 20,
            CHECK.CONTROL: 20,
            CHECK.CANDIDATE: 28,
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
        self.assertEqual(finalized["aggregate_seed"], 78160)
        self.assertTrue(finalized["aggregate_seed_open"])
        self.assertFalse(finalized["benchmark_data_read"])
        eval_source = (EXP / "scripts" / "eval_local_vllm.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("evaluate_promotion", eval_source)
        self.assertIn("finalize_promotion", eval_source)

    def test_frozen_gate_constants(self) -> None:
        self.assertEqual(CHECK.SEED, 88041)
        self.assertEqual(CHECK.SCREEN_SEEDS, (88042, 88044, 88045))
        self.assertEqual(CHECK.AGGREGATE_SEED, 78160)
        self.assertEqual(CHECK.ROWS_PER_ARM, 352)
        self.assertEqual(CHECK.AXIS_ROWS, 40)
        self.assertEqual(CHECK.AXIS_PER_SURFACE, 10)
        self.assertEqual(CHECK.RETENTION_ROWS_PER_SCREEN, 104)
        self.assertEqual(
            CHECK.ARMS,
            ("zero_root_parent", "replay_ctl4", "statechain_clean"),
        )
        self.assertEqual(CHECK.AXIS_KINDS, frozenset({"u_statechain"}))
        self.assertEqual(
            CHECK.AXIS_SURFACES,
            ("brewvat", "courierloft", "muletrack", "peatstove"),
        )
        self.assertEqual(CHECK.RETENTION_CORRECT_BAND, 5)
        self.assertEqual(CHECK.RETENTION_CAP_BAND, 3)
        self.assertEqual(CHECK.RETENTION_PARSED_BAND, 3)
        self.assertIn("BUDGET", CHECK.ABSTENTION_ANSWERS)
        self.assertIn("INSUFFICIENT", CHECK.ABSTENTION_ANSWERS)



class ParsedFieldValidationTests(unittest.TestCase):
    def test_missing_parsed_key_aborts(self) -> None:
        payload = synthetic_payload({label: 20 for label in CHECK.ARMS})
        for row in payload["rows"]:
            if row["adapter"] == CHECK.ARMS[0]:
                del row["parsed"]
                break
        with self.assertRaisesRegex(ValueError, "row schema changed"):
            CHECK.evaluate_promotion(payload)

    def test_non_string_parsed_aborts(self) -> None:
        payload = synthetic_payload({label: 20 for label in CHECK.ARMS})
        for row in payload["rows"]:
            if row["adapter"] == CHECK.ARMS[0]:
                row["parsed"] = 7
                break
        with self.assertRaisesRegex(ValueError, "row schema changed"):
            CHECK.evaluate_promotion(payload)

    def test_none_parsed_is_legal_unparsed(self) -> None:
        payload = synthetic_payload({label: 20 for label in CHECK.ARMS})
        for row in payload["rows"]:
            if row["adapter"] == CHECK.ARMS[0]:
                row["parsed"] = None
                row["correct"] = False
                break
        CHECK.evaluate_promotion(payload)

if __name__ == "__main__":
    unittest.main()
