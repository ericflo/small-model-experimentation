import importlib.util
import json
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


CHECK = load_module("axis_v2_check_local", "check_local.py")

AXIS_KINDS = sorted(CHECK.AXIS_KINDS)
RETENTION_KINDS = sorted(CHECK.RETENTION_KINDS)


def uniform_axis(n: int) -> dict[str, int]:
    return {kind: n for kind in AXIS_KINDS}


def synthetic_payload(
    axis_correct: dict[str, dict[str, int]],
    retention_correct: dict[str, int] | None = None,
    retention_parsed: dict[str, int] | None = None,
    cap_contacts: dict[str, int] | None = None,
    route_answer: dict[str, str] | None = None,
) -> dict:
    """Build a receipt: per arm, per kind, the first N rows are correct."""
    retention_totals = retention_correct or {}
    parsed_totals = retention_parsed or {}
    caps = cap_contacts or {}
    route_answers = route_answer or {}
    rows = []
    for label in CHECK.ARMS:
        for kind in AXIS_KINDS:
            correct_n = axis_correct[label].get(kind, 5)
            for index in range(CHECK.AXIS_PER_KIND):
                rows.append({
                    "adapter": label,
                    "task_id": f"axis_{kind}_{index}",
                    "kind": kind,
                    "parsed": "x",
                    "correct": bool(index < correct_n),
                    "cap_contact": False,
                })
        remaining_correct = retention_totals.get(label, CHECK.RETENTION_ROWS)
        remaining_unparsed = CHECK.RETENTION_ROWS - parsed_totals.get(
            label, CHECK.RETENTION_ROWS
        )
        remaining_caps = caps.get(label, 0)
        retention_rows = []
        for kind in RETENTION_KINDS:
            for index in range(CHECK.RETENTION_PER_KIND):
                retention_rows.append({
                    "adapter": label,
                    "task_id": f"ret_{kind}_{index}",
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
        if label in route_answers:
            for row in retention_rows:
                if row["kind"] == "u_route":
                    row["parsed"] = route_answers[label]
                    row["correct"] = False
        rows.extend(retention_rows)
    return {
        "seed": CHECK.SEED,
        "rows_per_arm": CHECK.ROWS,
        "labels": list(CHECK.ARMS),
        "rows": rows,
    }


class AnswerNormalizationTests(unittest.TestCase):
    def test_separator_whitespace_is_normalized_away(self) -> None:
        self.assertEqual(
            CHECK.normalize_answer("a > b"), CHECK.normalize_answer("a>b")
        )
        self.assertEqual(
            CHECK.normalize_answer("x=1; y=2"), CHECK.normalize_answer("x=1;y=2")
        )

    def test_content_differences_survive_normalization(self) -> None:
        self.assertNotEqual(
            CHECK.normalize_answer("STEP 1: append io"),
            CHECK.normalize_answer("STEP 2: append io"),
        )

    def test_exact_frozen_function(self) -> None:
        self.assertEqual(CHECK.normalize_answer("  a   b  "), "a b")
        self.assertEqual(CHECK.normalize_answer("alder > birch>cedar"), "alder>birch>cedar")
        self.assertEqual(CHECK.normalize_answer("STEP 3 ;  io titan"), "STEP 3;io titan")
        # Idempotent, and interior single spaces are preserved.
        value = "STEP 2;io titan rhea"
        self.assertEqual(CHECK.normalize_answer(value), value)
        self.assertEqual(
            CHECK.normalize_answer(CHECK.normalize_answer("a  >  b ; c")),
            CHECK.normalize_answer("a  >  b ; c"),
        )

    def test_documented_in_receipt_constant(self) -> None:
        self.assertTrue(CHECK.ANSWER_NORMALIZATION["prospective"])
        self.assertIn("three_event_failure_forensics", CHECK.ANSWER_NORMALIZATION["rationale"])
        self.assertIn("21 correct-but-rejected", CHECK.ANSWER_NORMALIZATION["rationale"])


class PromotionGateTests(unittest.TestCase):
    def test_clean_win_promotes(self) -> None:
        payload = synthetic_payload({
            CHECK.PARENT: uniform_axis(5),
            CHECK.CONTROL: uniform_axis(5),
            CHECK.CANDIDATE: uniform_axis(7),
        })
        result = CHECK.evaluate_promotion(payload)
        self.assertEqual(result["promoted"], "axis_v2")
        self.assertEqual(result["eligible"], ["axis_v2"])
        self.assertEqual(result["outcome"], "PROMOTED")
        self.assertEqual(result["detectable_kinds"], AXIS_KINDS)
        self.assertEqual(result["not_detectable"], [])
        self.assertEqual(result["axis_kind_wins"], 5)
        # ceil(2/3 * 5) = 4 when every kind is detectable.
        self.assertEqual(result["axis_kind_wins_required"], 4)
        self.assertTrue(all(result["checks"].values()))

    def test_axis_total_tie_with_replay_fails(self) -> None:
        payload = synthetic_payload({
            CHECK.PARENT: uniform_axis(4),
            CHECK.CONTROL: uniform_axis(6),
            CHECK.CANDIDATE: uniform_axis(6),
        })
        result = CHECK.evaluate_promotion(payload)
        self.assertIsNone(result["promoted"])
        self.assertEqual(result["eligible"], [])
        self.assertEqual(result["outcome"], "NOT_PROMOTED")
        self.assertFalse(result["checks"]["axis_total_strictly_beats_replay"])

    def test_only_three_kind_wins_fails_breadth_when_all_detectable(self) -> None:
        candidate = {
            AXIS_KINDS[0]: 9,
            AXIS_KINDS[1]: 9,
            AXIS_KINDS[2]: 9,
            AXIS_KINDS[3]: 5,
            AXIS_KINDS[4]: 5,
        }
        payload = synthetic_payload({
            CHECK.PARENT: uniform_axis(5),
            CHECK.CONTROL: uniform_axis(5),
            CHECK.CANDIDATE: candidate,
        })
        result = CHECK.evaluate_promotion(payload)
        # Total 37 strictly beats both 25s, but breadth is only 3 of the
        # required ceil(2/3 * 5) = 4. The candidate's own >= 9 kinds do NOT
        # shrink detectability; only control ceilings do.
        self.assertTrue(result["checks"]["axis_total_strictly_beats_parent"])
        self.assertTrue(result["checks"]["axis_total_strictly_beats_replay"])
        self.assertEqual(result["detectable_kinds"], AXIS_KINDS)
        self.assertEqual(result["axis_kind_wins"], 3)
        self.assertEqual(result["axis_kind_wins_required"], 4)
        self.assertFalse(result["checks"]["axis_kind_wins_meet_required_breadth"])
        self.assertIsNone(result["promoted"])

    def test_kind_tie_does_not_count_as_a_win(self) -> None:
        candidate = uniform_axis(7)
        candidate[AXIS_KINDS[0]] = 5  # exact tie with both controls on one kind
        payload = synthetic_payload({
            CHECK.PARENT: uniform_axis(5),
            CHECK.CONTROL: uniform_axis(5),
            CHECK.CANDIDATE: candidate,
        })
        result = CHECK.evaluate_promotion(payload)
        self.assertFalse(result["kind_wins"][AXIS_KINDS[0]])
        self.assertEqual(result["axis_kind_wins"], 4)
        self.assertEqual(result["promoted"], "axis_v2")

    def test_control_ceiling_kind_excluded_at_exactly_nine(self) -> None:
        # The prior experiments' failure mode: one control at the per-kind
        # ceiling. At parent = 9 the kind is NOT detectable and is excluded
        # from the breadth requirement instead of silently tightening it.
        parent = uniform_axis(5)
        parent[AXIS_KINDS[0]] = 9
        candidate = {
            AXIS_KINDS[0]: 5,
            AXIS_KINDS[1]: 8,
            AXIS_KINDS[2]: 8,
            AXIS_KINDS[3]: 8,
            AXIS_KINDS[4]: 5,
        }
        payload = synthetic_payload({
            CHECK.PARENT: parent,
            CHECK.CONTROL: uniform_axis(5),
            CHECK.CANDIDATE: candidate,
        })
        result = CHECK.evaluate_promotion(payload)
        self.assertEqual(result["not_detectable"], [AXIS_KINDS[0]])
        self.assertEqual(result["detectable_kinds"], AXIS_KINDS[1:])
        self.assertNotIn(AXIS_KINDS[0], result["kind_wins"])
        # ceil(2/3 * 4) = 3 and the candidate wins exactly 3 detectable kinds.
        self.assertEqual(result["axis_kind_wins_required"], 3)
        self.assertEqual(result["axis_kind_wins"], 3)
        # Total: 34 > parent 29 and > control 25.
        self.assertTrue(result["checks"]["axis_total_strictly_beats_parent"])
        self.assertEqual(result["outcome"], "PROMOTED")
        self.assertEqual(result["promoted"], "axis_v2")

    def test_replay_control_ceiling_also_excludes(self) -> None:
        # Either control at >= 9 removes the kind; a perfect 10 also counts.
        control = uniform_axis(5)
        control[AXIS_KINDS[1]] = 10
        candidate = {
            AXIS_KINDS[0]: 8,
            AXIS_KINDS[1]: 5,
            AXIS_KINDS[2]: 8,
            AXIS_KINDS[3]: 8,
            AXIS_KINDS[4]: 5,
        }
        payload = synthetic_payload({
            CHECK.PARENT: uniform_axis(5),
            CHECK.CONTROL: control,
            CHECK.CANDIDATE: candidate,
        })
        result = CHECK.evaluate_promotion(payload)
        self.assertEqual(result["not_detectable"], [AXIS_KINDS[1]])
        self.assertEqual(result["axis_kind_wins_required"], 3)
        self.assertEqual(result["axis_kind_wins"], 3)
        # Total: 34 > parent 25 and > control 30.
        self.assertEqual(result["promoted"], "axis_v2")

    def test_two_wins_on_four_detectable_fails_breadth(self) -> None:
        parent = uniform_axis(5)
        parent[AXIS_KINDS[0]] = 9
        candidate = {
            AXIS_KINDS[0]: 8,
            AXIS_KINDS[1]: 9,
            AXIS_KINDS[2]: 9,
            AXIS_KINDS[3]: 5,
            AXIS_KINDS[4]: 5,
        }
        payload = synthetic_payload({
            CHECK.PARENT: parent,
            CHECK.CONTROL: uniform_axis(5),
            CHECK.CANDIDATE: candidate,
        })
        result = CHECK.evaluate_promotion(payload)
        # Total 36 > parent 29 and > control 25, but only 2 of the required
        # ceil(2/3 * 4) = 3 detectable kinds are strictly won.
        self.assertEqual(result["detectable_kinds"], AXIS_KINDS[1:])
        self.assertEqual(result["axis_kind_wins"], 2)
        self.assertEqual(result["axis_kind_wins_required"], 3)
        self.assertFalse(result["checks"]["axis_kind_wins_meet_required_breadth"])
        self.assertEqual(result["outcome"], "NOT_PROMOTED")
        self.assertIsNone(result["promoted"])

    def test_zero_detectable_kinds_fails_closed_as_gate_undetectable(self) -> None:
        # Both controls saturate every kind: nothing is measurable and the
        # gate must fail closed even though the candidate sweeps every total.
        payload = synthetic_payload({
            CHECK.PARENT: uniform_axis(9),
            CHECK.CONTROL: uniform_axis(9),
            CHECK.CANDIDATE: uniform_axis(10),
        })
        result = CHECK.evaluate_promotion(payload)
        self.assertEqual(result["detectable_kinds"], [])
        self.assertEqual(result["not_detectable"], AXIS_KINDS)
        self.assertEqual(result["kind_wins"], {})
        self.assertEqual(result["axis_kind_wins_required"], 0)
        self.assertTrue(result["checks"]["axis_total_strictly_beats_parent"])
        self.assertTrue(result["checks"]["axis_total_strictly_beats_replay"])
        self.assertFalse(result["checks"]["at_least_one_axis_kind_detectable"])
        self.assertEqual(result["outcome"], "GATE_UNDETECTABLE")
        self.assertEqual(result["eligible"], [])
        self.assertIsNone(result["promoted"])

    def test_retention_correct_regression_beyond_band_fails(self) -> None:
        payload = synthetic_payload(
            {
                CHECK.PARENT: uniform_axis(5),
                CHECK.CONTROL: uniform_axis(5),
                CHECK.CANDIDATE: uniform_axis(7),
            },
            retention_correct={
                CHECK.PARENT: 80,
                CHECK.CONTROL: 80,
                CHECK.CANDIDATE: 74,  # 74 < 80 - 5
            },
        )
        result = CHECK.evaluate_promotion(payload)
        self.assertFalse(result["checks"]["retention_correct_within_5_of_parent"])
        self.assertIsNone(result["promoted"])

    def test_retention_regression_at_band_edge_passes(self) -> None:
        payload = synthetic_payload(
            {
                CHECK.PARENT: uniform_axis(5),
                CHECK.CONTROL: uniform_axis(5),
                CHECK.CANDIDATE: uniform_axis(7),
            },
            retention_correct={
                CHECK.PARENT: 80,
                CHECK.CONTROL: 80,
                CHECK.CANDIDATE: 75,  # exactly parent - 5
            },
        )
        result = CHECK.evaluate_promotion(payload)
        self.assertTrue(result["checks"]["retention_correct_within_5_of_parent"])
        self.assertEqual(result["promoted"], "axis_v2")

    def test_cap_contact_regression_beyond_band_fails(self) -> None:
        payload = synthetic_payload(
            {
                CHECK.PARENT: uniform_axis(5),
                CHECK.CONTROL: uniform_axis(5),
                CHECK.CANDIDATE: uniform_axis(7),
            },
            cap_contacts={CHECK.CANDIDATE: 4},
        )
        result = CHECK.evaluate_promotion(payload)
        self.assertFalse(result["checks"]["retention_cap_contacts_within_3_of_parent"])
        self.assertIsNone(result["promoted"])

    def test_parsed_regression_beyond_band_fails(self) -> None:
        payload = synthetic_payload(
            {
                CHECK.PARENT: uniform_axis(5),
                CHECK.CONTROL: uniform_axis(5),
                CHECK.CANDIDATE: uniform_axis(7),
            },
            retention_correct={CHECK.CANDIDATE: 80},
            retention_parsed={CHECK.CANDIDATE: 100},  # 100 < 104 - 3
        )
        result = CHECK.evaluate_promotion(payload)
        self.assertFalse(result["checks"]["retention_parsed_within_3_of_parent"])
        self.assertIsNone(result["promoted"])

    def test_budget_answer_on_route_counts_as_abstention(self) -> None:
        payload = synthetic_payload(
            {
                CHECK.PARENT: uniform_axis(5),
                CHECK.CONTROL: uniform_axis(5),
                CHECK.CANDIDATE: uniform_axis(7),
            },
            route_answer={CHECK.CANDIDATE: "BUDGET"},
        )
        result = CHECK.evaluate_promotion(payload)
        summary = result["summaries"][CHECK.CANDIDATE]
        self.assertEqual(summary["retention"]["route_abstentions"], 8)
        self.assertFalse(result["checks"]["route_abstentions_at_most_4_of_8"])
        self.assertIsNone(result["promoted"])

    def test_no_absolute_per_kind_floor_exists(self) -> None:
        # A candidate at zero on one retention kind still promotes when it
        # holds every relative band: floors are relative-only by design.
        payload = synthetic_payload(
            {
                CHECK.PARENT: uniform_axis(5),
                CHECK.CONTROL: uniform_axis(5),
                CHECK.CANDIDATE: uniform_axis(7),
            },
            retention_correct={
                CHECK.PARENT: 8,
                CHECK.CONTROL: 8,
                CHECK.CANDIDATE: 8,
            },
        )
        result = CHECK.evaluate_promotion(payload)
        self.assertTrue(result["no_absolute_per_kind_floors"])
        self.assertEqual(result["promoted"], "axis_v2")

    def test_required_kind_wins_is_exact_ceiling(self) -> None:
        self.assertEqual(CHECK.required_kind_wins(0), 0)
        self.assertEqual(CHECK.required_kind_wins(1), 1)
        self.assertEqual(CHECK.required_kind_wins(2), 2)
        self.assertEqual(CHECK.required_kind_wins(3), 2)
        self.assertEqual(CHECK.required_kind_wins(4), 3)
        self.assertEqual(CHECK.required_kind_wins(5), 4)
        self.assertEqual(CHECK.required_kind_wins(6), 4)

    def test_recovery_writer_schema_parity(self) -> None:
        payload = synthetic_payload({
            CHECK.PARENT: uniform_axis(5),
            CHECK.CONTROL: uniform_axis(5),
            CHECK.CANDIDATE: uniform_axis(7),
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
        # The recovery writer adds exactly the shared fields on top of
        # evaluate_promotion; eval_local_vllm.py calls the same function, so
        # the two promotion receipts cannot diverge in schema.
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
        self.assertEqual(finalized["aggregate_seed"], 78147)
        self.assertTrue(finalized["aggregate_seed_open"])
        self.assertFalse(finalized["benchmark_data_read"])
        eval_source = (EXP / "scripts" / "eval_local_vllm.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("from check_local import (", eval_source)
        for name in ("evaluate_promotion,", "finalize_promotion,", "normalize_answer,"):
            self.assertIn(name, eval_source)

    def test_frozen_gate_constants(self) -> None:
        self.assertEqual(CHECK.SEED, 88017)
        self.assertEqual(CHECK.AGGREGATE_SEED, 78147)
        self.assertEqual(CHECK.ROWS, 154)
        self.assertEqual(CHECK.AXIS_ROWS, 50)
        self.assertEqual(CHECK.RETENTION_ROWS, 104)
        self.assertEqual(CHECK.DETECTABILITY_CEILING, 9)
        self.assertEqual(
            CHECK.ARMS,
            ("axis_parent", "replay_repeat3", "axis_v2"),
        )
        self.assertEqual(
            set(CHECK.AXIS_KINDS),
            {"u_bugfind", "u_bugmend", "u_retrace", "u_explore", "u_hygiene"},
        )
        self.assertIn("BUDGET", CHECK.ABSTENTION_ANSWERS)
        self.assertIn("INSUFFICIENT", CHECK.ABSTENTION_ANSWERS)


if __name__ == "__main__":
    unittest.main()
