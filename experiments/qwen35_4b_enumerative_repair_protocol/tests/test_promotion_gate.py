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


CHECK = load_module("enum_repair_check_local", "check_local.py")

RETENTION_KINDS = sorted(CHECK.RETENTION_KINDS)
# Five axis rows per formalism (mirrors the generator's split).
AXIS_SURFACE_LAYOUT = [
    surface for surface in CHECK.AXIS_SURFACES for _ in range(CHECK.AXIS_PER_SURFACE)
]


def full_fidelity(correct: bool) -> dict:
    """A canonical-next answer scores all four; a wrong one is modeled as
    legal+untried but out of order (the neutral default for synthesis)."""
    return {
        "parseable": True,
        "legal": True,
        "untried": True,
        "canonical_next": bool(correct),
    }


def synthetic_payload(
    axis_correct: dict[str, int],
    retention_correct: dict[str, list[int]] | None = None,
    retention_parsed: dict[str, list[int]] | None = None,
    cap_contacts: dict[str, list[int]] | None = None,
) -> dict:
    """Build a receipt: per arm the first N axis rows are correct.
    Retention values are given per screen (a list aligned to SCREEN_SEEDS)."""
    retention_totals = retention_correct or {}
    parsed_totals = retention_parsed or {}
    caps = cap_contacts or {}
    rows = []
    for label in CHECK.ARMS:
        correct_n = axis_correct[label]
        for index in range(CHECK.AXIS_ROWS):
            correct = index < correct_n
            rows.append({
                "adapter": label,
                "screen": CHECK.SEED,
                "task_id": f"axis{CHECK.SEED}_u_enum_repair_{index}",
                "kind": "u_enum_repair",
                "surface": AXIS_SURFACE_LAYOUT[index],
                "parsed": "x",
                "correct": correct,
                "cap_contact": False,
                "enumeration_fidelity": full_fidelity(correct),
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


BASELINE = {CHECK.PARENT: 10, CHECK.CONTROL: 10}


class PromotionGateTests(unittest.TestCase):
    def test_strict_axis_win_promotes(self) -> None:
        payload = synthetic_payload({**BASELINE, CHECK.CANDIDATE: 15})
        result = CHECK.evaluate_promotion(payload)
        self.assertEqual(result["promoted"], "enum_repair")
        self.assertEqual(result["eligible"], ["enum_repair"])
        self.assertTrue(all(result["checks"].values()))

    def test_one_row_margin_over_both_controls_promotes(self) -> None:
        payload = synthetic_payload(
            {CHECK.PARENT: 10, CHECK.CONTROL: 12, CHECK.CANDIDATE: 13}
        )
        result = CHECK.evaluate_promotion(payload)
        self.assertEqual(result["promoted"], "enum_repair")

    def test_tie_with_parent_fails(self) -> None:
        payload = synthetic_payload(
            {CHECK.PARENT: 12, CHECK.CONTROL: 10, CHECK.CANDIDATE: 12}
        )
        result = CHECK.evaluate_promotion(payload)
        self.assertFalse(result["checks"]["axis_total_strictly_beats_parent"])
        self.assertTrue(result["checks"]["axis_total_strictly_beats_replay"])
        self.assertIsNone(result["promoted"])

    def test_tie_with_replay_fails(self) -> None:
        payload = synthetic_payload(
            {CHECK.PARENT: 8, CHECK.CONTROL: 12, CHECK.CANDIDATE: 12}
        )
        result = CHECK.evaluate_promotion(payload)
        self.assertTrue(result["checks"]["axis_total_strictly_beats_parent"])
        self.assertFalse(result["checks"]["axis_total_strictly_beats_replay"])
        self.assertIsNone(result["promoted"])

    def test_below_parent_fails(self) -> None:
        payload = synthetic_payload(
            {CHECK.PARENT: 20, CHECK.CONTROL: 2, CHECK.CANDIDATE: 15}
        )
        result = CHECK.evaluate_promotion(payload)
        self.assertFalse(result["checks"]["axis_total_strictly_beats_parent"])
        self.assertIsNone(result["promoted"])

    def test_no_per_kind_or_per_surface_gate_exists(self) -> None:
        # The candidate wins the total while scoring zero on the LAST
        # surfaces (rows are front-loaded); a single-kind gate must promote.
        payload = synthetic_payload({**BASELINE, CHECK.CANDIDATE: 15})
        result = CHECK.evaluate_promotion(payload)
        candidate_axis = result["summaries"][CHECK.CANDIDATE]["axis"]
        last_surface = CHECK.AXIS_SURFACES[-1]
        self.assertEqual(candidate_axis["per_surface_correct"][last_surface], 0)
        self.assertEqual(result["promoted"], "enum_repair")
        self.assertTrue(result["per_surface_reported_not_gated"])
        self.assertTrue(result["single_kind_gate"]["no_per_kind_split_exists"])
        self.assertNotIn("kind_gate", result)

    def test_mechanism_reading_is_reported_but_never_gates(self) -> None:
        # The candidate promotes even with a fidelity decomposition full of
        # already-tried proposals (fidelity is a reading, not a gate).
        payload = synthetic_payload({**BASELINE, CHECK.CANDIDATE: 15})
        for row in payload["rows"]:
            if (
                row["adapter"] == CHECK.CANDIDATE
                and row["screen"] == CHECK.SEED
                and not row["correct"]
            ):
                row["enumeration_fidelity"] = {
                    "parseable": True,
                    "legal": True,
                    "untried": False,
                    "canonical_next": False,
                }
        result = CHECK.evaluate_promotion(payload)
        self.assertEqual(result["promoted"], "enum_repair")
        reading = result["mechanism_reading"]
        self.assertTrue(reading["reported_not_gated"])
        candidate = reading["enumeration_fidelity_per_arm"][CHECK.CANDIDATE]
        self.assertEqual(candidate["rows_with_readout"], CHECK.AXIS_ROWS)
        self.assertEqual(candidate["canonical_next"], 15)
        self.assertEqual(candidate["legal_but_already_tried"], 25)

    def test_missing_fidelity_readout_on_an_axis_row_aborts(self) -> None:
        payload = synthetic_payload({**BASELINE, CHECK.CANDIDATE: 15})
        for row in payload["rows"]:
            if row["adapter"] == CHECK.CANDIDATE and row["screen"] == CHECK.SEED:
                del row["enumeration_fidelity"]
                break
        with self.assertRaisesRegex(ValueError, "enumeration-fidelity"):
            CHECK.evaluate_promotion(payload)

    def test_malformed_fidelity_readout_aborts(self) -> None:
        payload = synthetic_payload({**BASELINE, CHECK.CANDIDATE: 15})
        for row in payload["rows"]:
            if row["adapter"] == CHECK.CANDIDATE and row["screen"] == CHECK.SEED:
                row["enumeration_fidelity"] = {"legal": "yes"}
                break
        with self.assertRaisesRegex(ValueError, "enumeration-fidelity"):
            CHECK.evaluate_promotion(payload)

    def test_pooled_retention_band_edge_passes(self) -> None:
        # Parent pooled mean 80; candidate pooled mean exactly 75 = 80 - 5.
        payload = synthetic_payload(
            {**BASELINE, CHECK.CANDIDATE: 15},
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
        self.assertEqual(result["promoted"], "enum_repair")

    def test_pooled_retention_one_beyond_band_fails(self) -> None:
        payload = synthetic_payload(
            {**BASELINE, CHECK.CANDIDATE: 15},
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
        payload = synthetic_payload(
            {**BASELINE, CHECK.CANDIDATE: 15},
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
                {**BASELINE, CHECK.CANDIDATE: 15},
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
        payload = synthetic_payload(
            {**BASELINE, CHECK.CANDIDATE: 15},
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
        self.assertEqual(result["promoted"], "enum_repair")

    def test_pooled_cap_contact_regression_beyond_band_fails(self) -> None:
        payload = synthetic_payload(
            {**BASELINE, CHECK.CANDIDATE: 15},
            cap_contacts={CHECK.CANDIDATE: [4, 3, 3]},  # pooled mean 3.33 > 3
        )
        result = CHECK.evaluate_promotion(payload)
        self.assertFalse(
            result["checks"]["retention_pooled_cap_contacts_within_3_of_parent"]
        )
        self.assertIsNone(result["promoted"])

    def test_pooled_parsed_regression_beyond_band_fails(self) -> None:
        payload = synthetic_payload(
            {**BASELINE, CHECK.CANDIDATE: 15},
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
            {**BASELINE, CHECK.CANDIDATE: 15},
            retention_correct={
                CHECK.PARENT: [8, 8, 8],
                CHECK.CONTROL: [8, 8, 8],
                CHECK.CANDIDATE: [8, 8, 8],
            },
        )
        result = CHECK.evaluate_promotion(payload)
        self.assertTrue(result["no_absolute_per_kind_floors"])
        self.assertEqual(result["promoted"], "enum_repair")

    def test_axis_surface_imbalance_is_rejected(self) -> None:
        payload = synthetic_payload({**BASELINE, CHECK.CANDIDATE: 15})
        for row in payload["rows"]:
            if (
                row["screen"] == CHECK.SEED
                and row["adapter"] == CHECK.ARMS[0]
                and row["surface"] == CHECK.AXIS_SURFACES[0]
            ):
                row["surface"] = CHECK.AXIS_SURFACES[1]
                break
        with self.assertRaisesRegex(ValueError, "surface balance"):
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
        payload = synthetic_payload({**BASELINE, CHECK.CANDIDATE: 15})
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
        self.assertEqual(finalized["aggregate_seed"], 78162)
        self.assertTrue(finalized["aggregate_seed_open"])
        self.assertFalse(finalized["benchmark_data_read"])
        eval_source = (EXP / "scripts" / "eval_local_vllm.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("evaluate_promotion", eval_source)
        self.assertIn("finalize_promotion", eval_source)
        self.assertIn("enumeration_fidelity", eval_source)

    def test_frozen_gate_constants(self) -> None:
        self.assertEqual(CHECK.SEED, 88052)
        self.assertEqual(CHECK.SCREEN_SEEDS, (88053, 88054, 88055))
        self.assertEqual(CHECK.AGGREGATE_SEED, 78162)
        self.assertEqual(CHECK.ROWS_PER_ARM, 352)
        self.assertEqual(CHECK.AXIS_ROWS, 40)
        self.assertEqual(CHECK.AXIS_KIND_COUNTS, {"u_enum_repair": 40})
        self.assertEqual(CHECK.AXIS_PER_SURFACE, 5)
        self.assertEqual(CHECK.RETENTION_ROWS_PER_SCREEN, 104)
        self.assertEqual(
            CHECK.ARMS,
            ("zero_root_parent", "replay_ctl6", "enum_repair"),
        )
        self.assertEqual(CHECK.AXIS_KINDS, frozenset({"u_enum_repair"}))
        self.assertEqual(
            CHECK.AXIS_SURFACES,
            (
                "balesled",
                "barrowyoke",
                "crankwheel",
                "millround",
                "sigilslate",
                "skeinreel",
                "trinketcord",
                "troughline",
            ),
        )
        self.assertEqual(CHECK.RETENTION_CORRECT_BAND, 5)
        self.assertEqual(CHECK.RETENTION_CAP_BAND, 3)
        self.assertEqual(CHECK.RETENTION_PARSED_BAND, 3)
        self.assertEqual(
            CHECK.FIDELITY_FIELDS,
            ("parseable", "legal", "untried", "canonical_next"),
        )
        self.assertIn("BUDGET", CHECK.ABSTENTION_ANSWERS)
        self.assertIn("INSUFFICIENT", CHECK.ABSTENTION_ANSWERS)


class ParsedFieldValidationTests(unittest.TestCase):
    def test_missing_parsed_key_aborts(self) -> None:
        payload = synthetic_payload({label: 10 for label in CHECK.ARMS})
        for row in payload["rows"]:
            if row["adapter"] == CHECK.ARMS[0]:
                del row["parsed"]
                break
        with self.assertRaisesRegex(ValueError, "row schema changed"):
            CHECK.evaluate_promotion(payload)

    def test_non_string_parsed_aborts(self) -> None:
        payload = synthetic_payload({label: 10 for label in CHECK.ARMS})
        for row in payload["rows"]:
            if row["adapter"] == CHECK.ARMS[0]:
                row["parsed"] = 7
                break
        with self.assertRaisesRegex(ValueError, "row schema changed"):
            CHECK.evaluate_promotion(payload)

    def test_none_parsed_is_legal_unparsed(self) -> None:
        payload = synthetic_payload({label: 10 for label in CHECK.ARMS})
        for row in payload["rows"]:
            if row["adapter"] == CHECK.ARMS[0] and row["screen"] != CHECK.SEED:
                row["parsed"] = None
                row["correct"] = False
                break
        CHECK.evaluate_promotion(payload)


if __name__ == "__main__":
    unittest.main()
