import importlib.util
import json
import math
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


CHECK = load_module("calibration_check_local", "check_local.py")

KINDS = sorted(CHECK.RETENTION_KINDS)


def synthetic_payload(correct: dict[str, dict[int, int]] | None = None) -> dict:
    """Build a receipt: per arm per screen, the first N rows are correct."""
    totals = correct or {}
    rows = []
    for seed in CHECK.SEEDS:
        for label in CHECK.ARMS:
            remaining = totals.get(label, {}).get(seed, CHECK.ROWS)
            for kind in KINDS:
                for index in range(CHECK.RETENTION_PER_KIND):
                    is_correct = remaining > 0
                    if is_correct:
                        remaining -= 1
                    rows.append({
                        "adapter": label,
                        "screen": seed,
                        "task_id": f"ret{seed}_{kind}_{index}",
                        "kind": kind,
                        "parsed": "x",
                        "correct": is_correct,
                        "cap_contact": False,
                    })
    return {
        "seeds": list(CHECK.SEEDS),
        "rows_per_arm_per_screen": CHECK.ROWS,
        "labels": list(CHECK.ARMS),
        "rows": rows,
    }


def payload_from_table(table: dict[str, tuple[int, int, int, int]]) -> dict:
    return synthetic_payload(
        {
            label: dict(zip(CHECK.SEEDS, values))
            for label, values in table.items()
        }
    )


def constant_table(value_by_arm: dict[str, int]) -> dict:
    return payload_from_table(
        {label: (value,) * 4 for label, value in value_by_arm.items()}
    )


class SdPoolingTests(unittest.TestCase):
    def parent_constant_table(
        self, non_parent: tuple[int, int, int, int]
    ) -> dict:
        table = {label: non_parent for label in CHECK.ARMS}
        table["clean_parent"] = (70, 70, 70, 70)
        return payload_from_table(table)

    def test_zero_variance_reads_zero_pooled_sd(self) -> None:
        result = CHECK.evaluate_calibration(
            constant_table({label: 70 for label in CHECK.ARMS})
        )
        readings = result["readings"]
        self.assertEqual(readings["delta_sd_pooled"], 0.0)
        self.assertEqual(readings["screen_sd_pooled"], 0.0)
        self.assertEqual(readings["recommended_band"], 5)
        self.assertEqual(readings["adjudication_protocol"], "single_screen")

    def test_exact_delta_sd_two_is_still_single_screen(self) -> None:
        # Parent constant 70; non-parent arms (71, 71, 71, 67): delta series
        # (1, 1, 1, -3), mean 0, ss 12, variance 4, SD 2.
        result = CHECK.evaluate_calibration(
            self.parent_constant_table((71, 71, 71, 67))
        )
        readings = result["readings"]
        self.assertEqual(readings["delta_sd_pooled"], 2.0)
        self.assertEqual(readings["adjudication_protocol"], "single_screen")
        # ceil(2 * 2.0) = 4 < 5: the band floor binds.
        self.assertEqual(readings["recommended_band"], 5)

    def test_delta_sd_three_reads_pooled_k2_with_band_six(self) -> None:
        # Parent constant 70; non-parent arms (73, 73, 73, 67): delta series
        # (3, 3, 3, -3), mean 1.5, ss 27, variance 9, SD 3.
        result = CHECK.evaluate_calibration(
            self.parent_constant_table((73, 73, 73, 67))
        )
        readings = result["readings"]
        self.assertEqual(readings["delta_sd_pooled"], 3.0)
        self.assertEqual(readings["adjudication_protocol"], "pooled_k2")
        self.assertEqual(readings["recommended_band"], 6)

    def test_delta_sd_six_reads_pooled_k3(self) -> None:
        # Parent constant 70; non-parent arms (76, 76, 76, 64): delta series
        # (6, 6, 6, -6), mean 3, ss 108, variance 36, SD 6.
        result = CHECK.evaluate_calibration(
            self.parent_constant_table((76, 76, 76, 64))
        )
        readings = result["readings"]
        self.assertEqual(readings["delta_sd_pooled"], 6.0)
        self.assertEqual(readings["adjudication_protocol"], "pooled_k3")
        self.assertEqual(readings["recommended_band"], 12)

    def test_pooling_averages_per_arm_delta_variances(self) -> None:
        # Parent and three non-parent arms constant (delta variance 0); one
        # non-parent arm at delta variance 4: pooled delta variance over the
        # FOUR non-parent arms (0*3 + 4)/4 = 1.0. The level SD pools over all
        # FIVE arms: (0*4 + 4)/5 = 0.8 — reported, never governing.
        table = {label: (70, 70, 70, 70) for label in CHECK.ARMS}
        table["replay_clean"] = (71, 71, 71, 67)
        result = CHECK.evaluate_calibration(payload_from_table(table))
        self.assertAlmostEqual(
            result["readings"]["delta_sd_pooled"], 1.0, places=12
        )
        self.assertAlmostEqual(
            result["readings"]["screen_sd_pooled"], math.sqrt(0.8), places=12
        )

    def test_common_screen_difficulty_cancels_in_deltas(self) -> None:
        # The adversarial-review regression: every arm moves with screen
        # difficulty (levels wobble, SD sqrt(32/3) ~ 3.27) but each arm sits a
        # constant -5 from the parent on every screen. The governing outputs
        # must read the constant delta (SD 0 -> floor band 5, single_screen),
        # not the difficulty-inflated level SD.
        table = {label: (65, 69, 61, 65) for label in CHECK.ARMS}
        table["clean_parent"] = (70, 74, 66, 70)
        result = CHECK.evaluate_calibration(payload_from_table(table))
        readings = result["readings"]
        self.assertEqual(readings["delta_sd_pooled"], 0.0)
        self.assertEqual(readings["band_and_protocol_basis"], "delta_sd_pooled")
        self.assertEqual(readings["recommended_band"], 5)
        self.assertEqual(readings["adjudication_protocol"], "single_screen")
        self.assertAlmostEqual(
            readings["screen_sd_pooled"], math.sqrt(32 / 3), places=12
        )

    def test_across_screen_stats_use_sample_sd(self) -> None:
        result = CHECK.evaluate_calibration(
            payload_from_table({label: (71, 71, 71, 67) for label in CHECK.ARMS})
        )
        across = result["across_screens"]["clean_parent"]
        self.assertEqual(across["correct_by_screen"], [71, 71, 71, 67])
        self.assertEqual(across["mean_correct"], 70.0)
        self.assertEqual(across["sd_correct"], 2.0)


class ProtocolAndBandBoundaryTests(unittest.TestCase):
    def test_protocol_boundaries_partition_the_reals(self) -> None:
        self.assertEqual(CHECK.adjudication_protocol(0.0), "single_screen")
        self.assertEqual(CHECK.adjudication_protocol(2.0), "single_screen")
        self.assertEqual(
            CHECK.adjudication_protocol(math.nextafter(2.0, 3.0)), "pooled_k2"
        )
        self.assertEqual(CHECK.adjudication_protocol(3.5), "pooled_k2")
        self.assertEqual(
            CHECK.adjudication_protocol(math.nextafter(3.5, 4.0)), "pooled_k3"
        )
        self.assertEqual(CHECK.adjudication_protocol(100.0), "pooled_k3")

    def test_protocol_rejects_out_of_range_sd(self) -> None:
        for bad in (-0.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                CHECK.adjudication_protocol(bad)

    def test_band_formula_and_floor(self) -> None:
        self.assertEqual(CHECK.recommended_band(0.0), 5)
        self.assertEqual(CHECK.recommended_band(2.0), 5)
        self.assertEqual(CHECK.recommended_band(2.5), 5)
        self.assertEqual(CHECK.recommended_band(2.51), 6)
        self.assertEqual(CHECK.recommended_band(3.0), 6)
        self.assertEqual(CHECK.recommended_band(3.5), 7)
        self.assertEqual(CHECK.recommended_band(6.0), 12)

    def test_band_rejects_out_of_range_sd(self) -> None:
        for bad in (-0.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                CHECK.recommended_band(bad)

    def test_every_payload_protocol_stays_in_the_frozen_space(self) -> None:
        for non_parent in ((70, 70, 70, 70), (71, 71, 71, 67), (76, 76, 76, 64)):
            table = {label: non_parent for label in CHECK.ARMS}
            table["clean_parent"] = (70, 70, 70, 70)
            result = CHECK.evaluate_calibration(payload_from_table(table))
            self.assertIn(
                result["readings"]["adjudication_protocol"], CHECK.PROTOCOLS
            )
            self.assertEqual(
                result["adjudication_protocol"],
                result["readings"]["adjudication_protocol"],
            )


class DeltaAndStabilityTests(unittest.TestCase):
    def historical_table(self) -> dict[str, tuple[int, int, int, int]]:
        return {
            "clean_parent": (70, 70, 70, 70),
            "axis160_direct": (61, 61, 61, 61),
            "axis160_r64": (63, 63, 63, 63),
            "hygiene_explore_direct": (60, 60, 60, 60),
            "replay_clean": (65, 65, 65, 65),
        }

    def test_deltas_are_computed_against_clean_parent_per_screen(self) -> None:
        result = CHECK.evaluate_calibration(
            payload_from_table(self.historical_table())
        )
        deltas = result["deltas_vs_clean_parent"]
        self.assertEqual(sorted(deltas), sorted(CHECK.DELTA_ARMS))
        self.assertNotIn("clean_parent", deltas)
        self.assertEqual(deltas["axis160_direct"]["by_screen"], [-9, -9, -9, -9])
        self.assertEqual(deltas["axis160_direct"]["pooled_mean"], -9.0)
        self.assertEqual(deltas["axis160_direct"]["sd"], 0.0)

    def test_all_historical_readings_inside_when_replicated_exactly(self) -> None:
        result = CHECK.evaluate_calibration(
            payload_from_table(self.historical_table())
        )
        flags = result["readings"]["stability_flags"]
        self.assertEqual(len(flags), len(CHECK.HISTORICAL_READINGS))
        self.assertEqual(
            [
                (flag["arm"], flag["gate_seed"], flag["historical_delta"])
                for flag in flags
            ],
            list(CHECK.HISTORICAL_READINGS),
        )
        self.assertTrue(all(flag["inside"] for flag in flags))

    def test_historical_reading_outside_a_tight_interval(self) -> None:
        table = self.historical_table()
        # axis160_direct now retains perfectly: delta 0 on every screen; the
        # historical -9 falls outside [0, 0].
        table["axis160_direct"] = (70, 70, 70, 70)
        result = CHECK.evaluate_calibration(payload_from_table(table))
        by_key = {
            (flag["arm"], flag["gate_seed"]): flag
            for flag in result["readings"]["stability_flags"]
        }
        self.assertFalse(by_key[("axis160_direct", 88020)]["inside"])
        self.assertTrue(by_key[("replay_clean", 88020)]["inside"])

    def test_interval_is_pooled_delta_plus_minus_two_sd(self) -> None:
        table = self.historical_table()
        # axis160_direct deltas per screen: -5, -7, -9, -11 -> mean -8,
        # sd sqrt(20/3); the historical -9 sits inside.
        table["axis160_direct"] = (65, 63, 61, 59)
        result = CHECK.evaluate_calibration(payload_from_table(table))
        flag = next(
            entry
            for entry in result["readings"]["stability_flags"]
            if (entry["arm"], entry["gate_seed"]) == ("axis160_direct", 88020)
        )
        sd = math.sqrt(20 / 3)
        self.assertAlmostEqual(flag["interval_low"], -8 - 2 * sd, places=12)
        self.assertAlmostEqual(flag["interval_high"], -8 + 2 * sd, places=12)
        self.assertTrue(flag["inside"])

    def test_vehicle_reading_is_descriptive_only(self) -> None:
        result = CHECK.evaluate_calibration(
            payload_from_table(self.historical_table())
        )
        vehicle = result["readings"]["vehicle_descriptive"]
        self.assertTrue(vehicle["reported_not_gated"])
        self.assertEqual(vehicle["r64_pooled_delta"], -7.0)
        self.assertEqual(vehicle["r32_pooled_delta"], -9.0)
        self.assertEqual(vehicle["r64_minus_r32"], 2.0)

    def test_no_promotion_ever(self) -> None:
        for values in ((70, 70, 70, 70), (40, 50, 60, 70)):
            result = CHECK.evaluate_calibration(
                payload_from_table({label: values for label in CHECK.ARMS})
            )
            self.assertIsNone(result["promoted"])
            self.assertEqual(result["eligible"], [])
            self.assertEqual(result["candidates"], [])
            self.assertEqual(result["outcome"], "CALIBRATION_READ_COMPLETE")
            self.assertTrue(result["no_promotion_in_calibration_cell"])


class LayoutValidationTests(unittest.TestCase):
    def test_wrong_label_order_fails(self) -> None:
        payload = synthetic_payload()
        payload["labels"] = list(reversed(payload["labels"]))
        with self.assertRaisesRegex(ValueError, "label order"):
            CHECK.evaluate_calibration(payload)

    def test_wrong_seed_list_fails(self) -> None:
        payload = synthetic_payload()
        payload["seeds"] = [88021, 88022, 88023, 88024]
        with self.assertRaisesRegex(ValueError, "seeds or row count"):
            CHECK.evaluate_calibration(payload)

    def test_wrong_row_count_fails(self) -> None:
        payload = synthetic_payload()
        payload["rows"] = payload["rows"][:-1]
        with self.assertRaises(ValueError):
            CHECK.evaluate_calibration(payload)

    def test_task_id_mismatch_across_arms_fails(self) -> None:
        payload = synthetic_payload()
        for row in payload["rows"]:
            if (
                row["adapter"] == "replay_clean"
                and row["task_id"] == "ret88022_u_state_0"
            ):
                row["task_id"] = "ret88022_u_state_hijacked"
                break
        with self.assertRaises(ValueError):
            CHECK.evaluate_calibration(payload)

    def test_foreign_screen_prefix_fails(self) -> None:
        payload = synthetic_payload()
        for row in payload["rows"]:
            if row["screen"] == 88023 and row["task_id"] == "ret88023_u_state_0":
                row["task_id"] = "ret88022_u_state_99"
        with self.assertRaises(ValueError):
            CHECK.evaluate_calibration(payload)

    def test_kind_imbalance_fails(self) -> None:
        payload = synthetic_payload()
        for row in payload["rows"]:
            if (
                row["adapter"] == "replay_clean"
                and row["screen"] == 88024
                and row["kind"] == "u_state"
            ):
                row["kind"] = "u_trace"
        with self.assertRaises(ValueError):
            CHECK.evaluate_calibration(payload)

    def test_non_boolean_correct_fails(self) -> None:
        payload = synthetic_payload()
        payload["rows"][0]["correct"] = 1
        with self.assertRaises(ValueError):
            CHECK.evaluate_calibration(payload)


class NormalizationTests(unittest.TestCase):
    def test_whitespace_runs_collapse(self) -> None:
        self.assertEqual(CHECK.normalize_answer("  a   b\tc  "), "a b c")

    def test_route_separator_spaces_removed(self) -> None:
        self.assertEqual(
            CHECK.normalize_answer("alder > birch >cedar"), "alder>birch>cedar"
        )
        self.assertEqual(CHECK.normalize_answer("x ; y;z"), "x;y;z")

    def test_normalization_is_idempotent(self) -> None:
        for value in ("a  b > c ; d", "STEP 3: add 4 to bora", " HOLD-12 "):
            once = CHECK.normalize_answer(value)
            self.assertEqual(CHECK.normalize_answer(once), once)

    def test_plain_answers_unchanged(self) -> None:
        self.assertEqual(CHECK.normalize_answer("HOLD-12"), "HOLD-12")
        self.assertEqual(
            CHECK.normalize_answer("STEP 3: add 4 to bora"), "STEP 3: add 4 to bora"
        )


class FrozenConstantsTests(unittest.TestCase):
    def test_frozen_gate_constants(self) -> None:
        self.assertEqual(CHECK.SEEDS, (88022, 88023, 88024, 88025))
        self.assertEqual(CHECK.ROWS, 104)
        self.assertEqual(CHECK.RETENTION_PER_KIND, 8)
        self.assertEqual(
            CHECK.ARMS,
            (
                "axis160_direct",
                "axis160_r64",
                "clean_parent",
                "hygiene_explore_direct",
                "replay_clean",
            ),
        )
        # The frozen run order is alphabetical within a screen.
        self.assertEqual(CHECK.ARMS, tuple(sorted(CHECK.ARMS)))
        self.assertEqual(CHECK.PARENT, "clean_parent")
        self.assertEqual(CHECK.SINGLE_SCREEN_MAX_SD, 2.0)
        self.assertEqual(CHECK.POOLED_K2_MAX_SD, 3.5)
        self.assertEqual(CHECK.BAND_MINIMUM, 5)
        self.assertEqual(
            CHECK.PROTOCOLS, ("single_screen", "pooled_k2", "pooled_k3")
        )
        self.assertEqual(
            CHECK.HISTORICAL_READINGS,
            (
                ("axis160_direct", 88020, -9),
                ("axis160_r64", 88021, -7),
                ("hygiene_explore_direct", 88018, -10),
                ("hygiene_explore_direct", 88020, -10),
                ("replay_clean", 88020, -5),
            ),
        )
        self.assertIn("BUDGET", CHECK.ABSTENTION_ANSWERS)
        self.assertIn("INSUFFICIENT", CHECK.ABSTENTION_ANSWERS)


class WriterParityTests(unittest.TestCase):
    def test_finalize_writer_schema_parity(self) -> None:
        payload = synthetic_payload()
        base_result = CHECK.evaluate_calibration(payload)
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "local.json"
            raw = (json.dumps(payload, sort_keys=True) + "\n").encode()
            receipt.write_bytes(raw)
            design = Path(directory) / "design.json"
            design.write_bytes(b"{}\n")
            finalized = CHECK.finalize_calibration(
                dict(base_result), receipt, raw, design_receipt=design
            )
        # The recovery writer adds exactly the shared fields on top of
        # evaluate_calibration; eval_local_vllm.py calls the same function,
        # so the two calibration receipts cannot diverge in schema.
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
        self.assertIsNone(finalized["aggregate_seed"])
        self.assertFalse(finalized["aggregate_seed_open"])
        self.assertFalse(finalized["benchmark_data_read"])
        eval_source = (EXP / "scripts" / "eval_local_vllm.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("evaluate_calibration", eval_source)
        self.assertIn("finalize_calibration", eval_source)

    def test_eval_freezes_screen_major_run_order(self) -> None:
        eval_source = (EXP / "scripts" / "eval_local_vllm.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("for seed in SEEDS:\n        for label in LABELS:", eval_source)


if __name__ == "__main__":
    unittest.main()
