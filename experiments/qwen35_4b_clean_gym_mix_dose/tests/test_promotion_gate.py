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


CHECK = load_module("gym_mix_check_local", "check_local.py")

AXIS_KINDS = sorted(CHECK.AXIS_KIND_COUNTS)
RETENTION_KINDS = sorted(CHECK.RETENTION_KINDS)
# One deterministic surface layout per kind (mirrors the generator's split).
KIND_SURFACES = {
    "u_siren_episode": ["stillroom"] * 14,
    "u_statechain": (
        ["brewvat"] * 4 + ["courierloft"] * 3 + ["peatstove"] * 3 + ["muletrack"] * 3
    ),
    "u_mirage_abstain": ["counterhouse"] * 13,
}


def synthetic_payload(
    axis_correct_by_kind: dict[str, dict[str, int]],
    retention_correct: dict[str, list[int]] | None = None,
    retention_parsed: dict[str, list[int]] | None = None,
    cap_contacts: dict[str, list[int]] | None = None,
) -> dict:
    """Build a receipt: per arm and per kind, the first N axis rows are
    correct. Retention values are given per screen (a list aligned to
    SCREEN_SEEDS)."""
    retention_totals = retention_correct or {}
    parsed_totals = retention_parsed or {}
    caps = cap_contacts or {}
    rows = []
    for label in CHECK.ARMS:
        for kind in AXIS_KINDS:
            correct_n = axis_correct_by_kind[label].get(kind, 0)
            for index in range(CHECK.AXIS_KIND_COUNTS[kind]):
                rows.append({
                    "adapter": label,
                    "screen": CHECK.SEED,
                    "task_id": f"axis{CHECK.SEED}_{kind}_{index}",
                    "kind": kind,
                    "surface": KIND_SURFACES[kind][index],
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


def per_kind(siren: int, chain: int, mirage: int) -> dict[str, int]:
    return {
        "u_siren_episode": siren,
        "u_statechain": chain,
        "u_mirage_abstain": mirage,
    }


BASELINE = {
    CHECK.PARENT: per_kind(6, 6, 6),
    CHECK.CONTROL: per_kind(6, 6, 6),
}


class PromotionGateTests(unittest.TestCase):
    def test_all_three_kinds_won_promotes(self) -> None:
        payload = synthetic_payload({
            **BASELINE,
            CHECK.CANDIDATE: per_kind(9, 9, 9),
        })
        result = CHECK.evaluate_promotion(payload)
        self.assertEqual(result["promoted"], "gym_mix")
        self.assertEqual(result["eligible"], ["gym_mix"])
        self.assertTrue(all(result["checks"].values()))
        self.assertEqual(result["kind_gate"]["kinds_won"], 3)

    def test_exactly_two_kinds_won_promotes(self) -> None:
        # The third kind LOSES outright; the total still beats both controls.
        payload = synthetic_payload({
            **BASELINE,
            CHECK.CANDIDATE: per_kind(10, 10, 4),
        })
        result = CHECK.evaluate_promotion(payload)
        self.assertEqual(result["kind_gate"]["kinds_won"], 2)
        self.assertFalse(
            result["kind_gate"]["per_kind"]["u_mirage_abstain"]["won"]
        )
        self.assertEqual(result["promoted"], "gym_mix")

    def test_only_one_kind_won_fails_even_with_total_win(self) -> None:
        # +6 on sirens alone carries the total but not the breadth gate.
        payload = synthetic_payload({
            **BASELINE,
            CHECK.CANDIDATE: per_kind(12, 6, 6),
        })
        result = CHECK.evaluate_promotion(payload)
        self.assertTrue(result["checks"]["axis_total_strictly_beats_parent"])
        self.assertTrue(result["checks"]["axis_total_strictly_beats_replay"])
        self.assertEqual(result["kind_gate"]["kinds_won"], 1)
        self.assertFalse(
            result["checks"][
                "at_least_2_of_3_kinds_strict_over_both_controls"
            ]
        )
        self.assertIsNone(result["promoted"])

    def test_a_tie_on_a_kind_fails_that_kind(self) -> None:
        # Sirens and statechain strictly win; mirage TIES the parent — the
        # tie fails the kind but two wins still satisfy the breadth gate.
        payload = synthetic_payload({
            **BASELINE,
            CHECK.CANDIDATE: per_kind(9, 9, 6),
        })
        result = CHECK.evaluate_promotion(payload)
        mirage = result["kind_gate"]["per_kind"]["u_mirage_abstain"]
        self.assertFalse(mirage["won"])
        self.assertEqual(result["kind_gate"]["kinds_won"], 2)
        self.assertEqual(result["promoted"], "gym_mix")

    def test_two_ties_fail_the_breadth_gate(self) -> None:
        # One strict win + two ties = one kind won; the gate must fail.
        payload = synthetic_payload({
            **BASELINE,
            CHECK.CANDIDATE: per_kind(9, 6, 6),
        })
        result = CHECK.evaluate_promotion(payload)
        self.assertEqual(result["kind_gate"]["kinds_won"], 1)
        self.assertIsNone(result["promoted"])

    def test_kind_win_requires_both_controls(self) -> None:
        # The candidate beats the parent on statechain but ties the REPLAY
        # control there — that kind is not won.
        payload = synthetic_payload({
            CHECK.PARENT: per_kind(6, 5, 6),
            CHECK.CONTROL: per_kind(6, 8, 6),
            CHECK.CANDIDATE: per_kind(9, 8, 9),
        })
        result = CHECK.evaluate_promotion(payload)
        chain = result["kind_gate"]["per_kind"]["u_statechain"]
        self.assertTrue(chain["strictly_beats_parent"])
        self.assertFalse(chain["strictly_beats_replay"])
        self.assertFalse(chain["won"])
        self.assertEqual(result["kind_gate"]["kinds_won"], 2)
        self.assertEqual(result["promoted"], "gym_mix")

    def test_axis_total_tie_with_replay_fails(self) -> None:
        # Two kinds won, but the reshuffled totals tie the replay control.
        payload = synthetic_payload({
            CHECK.PARENT: per_kind(6, 6, 6),
            CHECK.CONTROL: per_kind(12, 6, 6),
            CHECK.CANDIDATE: per_kind(6, 9, 9),
        })
        result = CHECK.evaluate_promotion(payload)
        self.assertEqual(result["kind_gate"]["kinds_won"], 2)
        self.assertFalse(result["checks"]["axis_total_strictly_beats_replay"])
        self.assertTrue(result["checks"]["axis_total_strictly_beats_parent"])
        self.assertIsNone(result["promoted"])

    def test_axis_total_below_parent_fails(self) -> None:
        payload = synthetic_payload({
            CHECK.PARENT: per_kind(10, 10, 10),
            CHECK.CONTROL: per_kind(2, 2, 2),
            CHECK.CANDIDATE: per_kind(9, 9, 9),
        })
        result = CHECK.evaluate_promotion(payload)
        self.assertFalse(result["checks"]["axis_total_strictly_beats_parent"])
        self.assertIsNone(result["promoted"])

    def test_no_per_surface_gate_exists(self) -> None:
        payload = synthetic_payload({
            **BASELINE,
            CHECK.CANDIDATE: per_kind(9, 9, 9),
        })
        result = CHECK.evaluate_promotion(payload)
        candidate_axis = result["summaries"][CHECK.CANDIDATE]["axis"]
        # The statechain correct rows are front-loaded; muletrack scores 0
        # and the gate must still promote (surfaces reported, never gated).
        self.assertEqual(candidate_axis["per_surface_correct"]["muletrack"], 0)
        self.assertEqual(result["promoted"], "gym_mix")
        self.assertTrue(result["per_surface_reported_not_gated"])

    def test_pooled_retention_band_edge_passes(self) -> None:
        # Parent pooled mean 80; candidate pooled mean exactly 75 = 80 - 5.
        payload = synthetic_payload(
            {**BASELINE, CHECK.CANDIDATE: per_kind(9, 9, 9)},
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
        self.assertEqual(result["promoted"], "gym_mix")

    def test_pooled_retention_one_beyond_band_fails(self) -> None:
        payload = synthetic_payload(
            {**BASELINE, CHECK.CANDIDATE: per_kind(9, 9, 9)},
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
            {**BASELINE, CHECK.CANDIDATE: per_kind(9, 9, 9)},
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
                {**BASELINE, CHECK.CANDIDATE: per_kind(9, 9, 9)},
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
            {**BASELINE, CHECK.CANDIDATE: per_kind(9, 9, 9)},
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
        self.assertEqual(result["promoted"], "gym_mix")

    def test_pooled_cap_contact_regression_beyond_band_fails(self) -> None:
        payload = synthetic_payload(
            {**BASELINE, CHECK.CANDIDATE: per_kind(9, 9, 9)},
            cap_contacts={CHECK.CANDIDATE: [4, 3, 3]},  # pooled mean 3.33 > 3
        )
        result = CHECK.evaluate_promotion(payload)
        self.assertFalse(
            result["checks"]["retention_pooled_cap_contacts_within_3_of_parent"]
        )
        self.assertIsNone(result["promoted"])

    def test_pooled_parsed_regression_beyond_band_fails(self) -> None:
        payload = synthetic_payload(
            {**BASELINE, CHECK.CANDIDATE: per_kind(9, 9, 9)},
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
            {**BASELINE, CHECK.CANDIDATE: per_kind(9, 9, 9)},
            retention_correct={
                CHECK.PARENT: [8, 8, 8],
                CHECK.CONTROL: [8, 8, 8],
                CHECK.CANDIDATE: [8, 8, 8],
            },
        )
        result = CHECK.evaluate_promotion(payload)
        self.assertTrue(result["no_absolute_per_kind_floors"])
        self.assertEqual(result["promoted"], "gym_mix")

    def test_axis_kind_imbalance_is_rejected(self) -> None:
        payload = synthetic_payload({
            **BASELINE,
            CHECK.CANDIDATE: per_kind(9, 9, 9),
        })
        for row in payload["rows"]:
            if (
                row["screen"] == CHECK.SEED
                and row["adapter"] == CHECK.ARMS[0]
                and row["kind"] == "u_mirage_abstain"
            ):
                row["kind"] = "u_siren_episode"
                break
        with self.assertRaisesRegex(ValueError, "kind balance"):
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
            **BASELINE,
            CHECK.CANDIDATE: per_kind(9, 9, 9),
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
        self.assertEqual(finalized["aggregate_seed"], 78161)
        self.assertTrue(finalized["aggregate_seed_open"])
        self.assertFalse(finalized["benchmark_data_read"])
        eval_source = (EXP / "scripts" / "eval_local_vllm.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("evaluate_promotion", eval_source)
        self.assertIn("finalize_promotion", eval_source)

    def test_frozen_gate_constants(self) -> None:
        self.assertEqual(CHECK.SEED, 88046)
        self.assertEqual(CHECK.SCREEN_SEEDS, (88048, 88050, 88051))
        self.assertEqual(CHECK.AGGREGATE_SEED, 78161)
        self.assertEqual(CHECK.ROWS_PER_ARM, 352)
        self.assertEqual(CHECK.AXIS_ROWS, 40)
        self.assertEqual(
            CHECK.AXIS_KIND_COUNTS,
            {"u_siren_episode": 14, "u_statechain": 13, "u_mirage_abstain": 13},
        )
        self.assertEqual(CHECK.KINDS_REQUIRED_TO_WIN, 2)
        self.assertEqual(CHECK.RETENTION_ROWS_PER_SCREEN, 104)
        self.assertEqual(
            CHECK.ARMS,
            ("zero_root_parent", "replay_ctl5", "gym_mix"),
        )
        self.assertEqual(
            CHECK.AXIS_KINDS,
            frozenset({"u_siren_episode", "u_statechain", "u_mirage_abstain"}),
        )
        self.assertEqual(
            CHECK.AXIS_SURFACES,
            (
                "stillroom",
                "counterhouse",
                "brewvat",
                "courierloft",
                "muletrack",
                "peatstove",
            ),
        )
        self.assertEqual(CHECK.RETENTION_CORRECT_BAND, 5)
        self.assertEqual(CHECK.RETENTION_CAP_BAND, 3)
        self.assertEqual(CHECK.RETENTION_PARSED_BAND, 3)
        self.assertIn("BUDGET", CHECK.ABSTENTION_ANSWERS)
        self.assertIn("INSUFFICIENT", CHECK.ABSTENTION_ANSWERS)


class ParsedFieldValidationTests(unittest.TestCase):
    def test_missing_parsed_key_aborts(self) -> None:
        payload = synthetic_payload(
            {label: per_kind(6, 6, 6) for label in CHECK.ARMS}
        )
        for row in payload["rows"]:
            if row["adapter"] == CHECK.ARMS[0]:
                del row["parsed"]
                break
        with self.assertRaisesRegex(ValueError, "row schema changed"):
            CHECK.evaluate_promotion(payload)

    def test_non_string_parsed_aborts(self) -> None:
        payload = synthetic_payload(
            {label: per_kind(6, 6, 6) for label in CHECK.ARMS}
        )
        for row in payload["rows"]:
            if row["adapter"] == CHECK.ARMS[0]:
                row["parsed"] = 7
                break
        with self.assertRaisesRegex(ValueError, "row schema changed"):
            CHECK.evaluate_promotion(payload)

    def test_none_parsed_is_legal_unparsed(self) -> None:
        payload = synthetic_payload(
            {label: per_kind(6, 6, 6) for label in CHECK.ARMS}
        )
        for row in payload["rows"]:
            if row["adapter"] == CHECK.ARMS[0]:
                row["parsed"] = None
                row["correct"] = False
                break
        CHECK.evaluate_promotion(payload)


if __name__ == "__main__":
    unittest.main()
