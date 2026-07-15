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


CHECK = load_module("readjudication_check_local", "check_local.py")

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


class PromotionGateTests(unittest.TestCase):
    def test_clean_win_promotes(self) -> None:
        payload = synthetic_payload({
            CHECK.PARENT: uniform_axis(5),
            CHECK.CONTROL: uniform_axis(5),
            CHECK.CANDIDATE: uniform_axis(7),
        })
        result = CHECK.evaluate_promotion(payload)
        self.assertEqual(result["promoted"], "axis_on_replay")
        self.assertEqual(result["eligible"], ["axis_on_replay"])
        self.assertEqual(result["outcome"], "PROMOTED")
        self.assertEqual(result["detectable_kinds"], AXIS_KINDS)
        self.assertEqual(result["not_detectable"], [])
        self.assertEqual(result["axis_kind_wins"], 4)
        # ceil(2/3 * 4) = 3: identical to the predecessor bar when every
        # kind is detectable.
        self.assertEqual(result["axis_kind_wins_required"], 3)
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

    def test_only_two_kind_wins_fails_breadth_when_all_detectable(self) -> None:
        candidate = {
            AXIS_KINDS[0]: 9,
            AXIS_KINDS[1]: 9,
            AXIS_KINDS[2]: 5,
            AXIS_KINDS[3]: 5,
        }
        payload = synthetic_payload({
            CHECK.PARENT: uniform_axis(5),
            CHECK.CONTROL: uniform_axis(5),
            CHECK.CANDIDATE: candidate,
        })
        result = CHECK.evaluate_promotion(payload)
        # Total 28 strictly beats both 20s, but breadth is only 2 of the
        # required ceil(2/3 * 4) = 3. The candidate's own >= 9 kinds do NOT
        # shrink detectability; only control ceilings do.
        self.assertTrue(result["checks"]["axis_total_strictly_beats_parent"])
        self.assertTrue(result["checks"]["axis_total_strictly_beats_replay"])
        self.assertEqual(result["detectable_kinds"], AXIS_KINDS)
        self.assertEqual(result["axis_kind_wins"], 2)
        self.assertEqual(result["axis_kind_wins_required"], 3)
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
        self.assertEqual(result["axis_kind_wins"], 3)
        self.assertEqual(result["promoted"], "axis_on_replay")

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
            AXIS_KINDS[3]: 5,
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
        # ceil(2/3 * 3) = 2 and the candidate wins exactly 2 detectable kinds.
        self.assertEqual(result["axis_kind_wins_required"], 2)
        self.assertEqual(result["axis_kind_wins"], 2)
        # Total: 26 > parent 24 and > control 20.
        self.assertTrue(result["checks"]["axis_total_strictly_beats_parent"])
        self.assertEqual(result["outcome"], "PROMOTED")
        self.assertEqual(result["promoted"], "axis_on_replay")

    def test_replay_control_ceiling_also_excludes(self) -> None:
        # Either control at >= 9 removes the kind; a perfect 10 also counts.
        control = uniform_axis(5)
        control[AXIS_KINDS[1]] = 10
        candidate = {
            AXIS_KINDS[0]: 8,
            AXIS_KINDS[1]: 5,
            AXIS_KINDS[2]: 8,
            AXIS_KINDS[3]: 5,
        }
        payload = synthetic_payload({
            CHECK.PARENT: uniform_axis(5),
            CHECK.CONTROL: control,
            CHECK.CANDIDATE: candidate,
        })
        result = CHECK.evaluate_promotion(payload)
        self.assertEqual(result["not_detectable"], [AXIS_KINDS[1]])
        self.assertEqual(result["axis_kind_wins_required"], 2)
        self.assertEqual(result["axis_kind_wins"], 2)
        # Total: 26 > parent 20 and > control 25.
        self.assertEqual(result["promoted"], "axis_on_replay")

    def test_one_win_on_three_detectable_fails_breadth(self) -> None:
        parent = uniform_axis(5)
        parent[AXIS_KINDS[0]] = 9
        candidate = {
            AXIS_KINDS[0]: 8,
            AXIS_KINDS[1]: 9,
            AXIS_KINDS[2]: 5,
            AXIS_KINDS[3]: 5,
        }
        payload = synthetic_payload({
            CHECK.PARENT: parent,
            CHECK.CONTROL: uniform_axis(5),
            CHECK.CANDIDATE: candidate,
        })
        result = CHECK.evaluate_promotion(payload)
        # Total 27 > parent 24 and > control 20, but only 1 of the required
        # ceil(2/3 * 3) = 2 detectable kinds is strictly won.
        self.assertEqual(result["detectable_kinds"], AXIS_KINDS[1:])
        self.assertEqual(result["axis_kind_wins"], 1)
        self.assertEqual(result["axis_kind_wins_required"], 2)
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
        self.assertEqual(result["promoted"], "axis_on_replay")

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
        self.assertEqual(result["promoted"], "axis_on_replay")

    def test_required_kind_wins_is_exact_ceiling(self) -> None:
        self.assertEqual(CHECK.required_kind_wins(0), 0)
        self.assertEqual(CHECK.required_kind_wins(1), 1)
        self.assertEqual(CHECK.required_kind_wins(2), 2)
        self.assertEqual(CHECK.required_kind_wins(3), 2)
        self.assertEqual(CHECK.required_kind_wins(4), 3)

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
        self.assertEqual(finalized["aggregate_seed"], 78146)
        self.assertTrue(finalized["aggregate_seed_open"])
        self.assertFalse(finalized["benchmark_data_read"])
        eval_source = (EXP / "scripts" / "eval_local_vllm.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "from check_local import evaluate_promotion, finalize_promotion",
            eval_source,
        )

    def test_frozen_gate_constants(self) -> None:
        self.assertEqual(CHECK.SEED, 88016)
        self.assertEqual(CHECK.AGGREGATE_SEED, 78146)
        self.assertEqual(CHECK.ROWS, 144)
        self.assertEqual(CHECK.AXIS_ROWS, 40)
        self.assertEqual(CHECK.RETENTION_ROWS, 104)
        self.assertEqual(CHECK.DETECTABILITY_CEILING, 9)
        self.assertEqual(
            CHECK.ARMS,
            ("replay_parent", "replay_squared", "axis_on_replay"),
        )
        self.assertIn("BUDGET", CHECK.ABSTENTION_ANSWERS)
        self.assertIn("INSUFFICIENT", CHECK.ABSTENTION_ANSWERS)


if __name__ == "__main__":
    unittest.main()
