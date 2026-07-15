import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, EXP / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CHECK = load_module("mechanism_check_local", "check_local.py")

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
            correct_n = axis_correct.get(label, {}).get(kind, 5)
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


def payload_with_retention(parent: int, hygexp: int, candidate: int, replay: int = 60) -> dict:
    return synthetic_payload(
        {},
        retention_correct={
            CHECK.PARENT: parent,
            CHECK.HYGEXP: hygexp,
            CHECK.REPLAY: replay,
            CHECK.CANDIDATE: candidate,
        },
    )


class VerdictBoundaryTests(unittest.TestCase):
    def test_supported_at_exact_boundaries(self) -> None:
        # hygexp exactly -6 (forgetting reproduced), candidate exactly -5
        # (retention held): the minimal SUPPORTED cell.
        result = CHECK.evaluate_mechanism(payload_with_retention(68, 62, 63))
        self.assertEqual(result["readings"]["retention_delta_hygexp"], -6)
        self.assertEqual(result["readings"]["retention_delta_axis160"], -5)
        self.assertEqual(result["diversity_mechanism"], "SUPPORTED")
        self.assertEqual(result["readings"]["diversity_mechanism"], "SUPPORTED")

    def test_supported_with_candidate_gain(self) -> None:
        result = CHECK.evaluate_mechanism(payload_with_retention(68, 58, 70))
        self.assertEqual(result["readings"]["retention_delta_hygexp"], -10)
        self.assertEqual(result["readings"]["retention_delta_axis160"], 2)
        self.assertEqual(result["diversity_mechanism"], "SUPPORTED")

    def test_refuted_intrinsic_at_exact_boundary(self) -> None:
        # hygexp -6 and candidate -6: both direct doses forget.
        result = CHECK.evaluate_mechanism(payload_with_retention(68, 62, 62))
        self.assertEqual(result["readings"]["retention_delta_axis160"], -6)
        self.assertEqual(result["diversity_mechanism"], "REFUTED_INTRINSIC")

    def test_refuted_intrinsic_deep_forgetting(self) -> None:
        result = CHECK.evaluate_mechanism(payload_with_retention(68, 55, 50))
        self.assertEqual(result["diversity_mechanism"], "REFUTED_INTRINSIC")

    def test_screen_fortune_suspect_at_exact_boundary(self) -> None:
        # hygexp exactly -5: the known ~-10 does not reproduce, so the screen
        # cannot adjudicate — regardless of what the candidate does.
        result = CHECK.evaluate_mechanism(payload_with_retention(68, 63, 68))
        self.assertEqual(result["readings"]["retention_delta_hygexp"], -5)
        self.assertEqual(result["diversity_mechanism"], "SCREEN_FORTUNE_SUSPECT")

    def test_screen_fortune_suspect_shadows_candidate_forgetting(self) -> None:
        # Even a badly forgetting candidate reads SCREEN_FORTUNE_SUSPECT when
        # the hygexp reference fails to reproduce its forgetting.
        result = CHECK.evaluate_mechanism(payload_with_retention(68, 70, 40))
        self.assertEqual(result["diversity_mechanism"], "SCREEN_FORTUNE_SUSPECT")

    def test_verdict_partition_is_total_over_integers(self) -> None:
        for hygexp_delta in range(-12, 3):
            for axis_delta in range(-12, 3):
                verdict = CHECK.diversity_mechanism_verdict(axis_delta, hygexp_delta)
                self.assertIn(verdict, CHECK.VERDICTS)
                if hygexp_delta >= -5:
                    self.assertEqual(verdict, "SCREEN_FORTUNE_SUSPECT")
                elif axis_delta >= -5:
                    self.assertEqual(verdict, "SUPPORTED")
                else:
                    self.assertEqual(verdict, "REFUTED_INTRINSIC")

    def test_replay_clean_delta_recorded_but_never_adjudicates(self) -> None:
        supported = CHECK.evaluate_mechanism(
            payload_with_retention(68, 58, 68, replay=0)
        )
        self.assertEqual(supported["readings"]["retention_delta_replay_clean"], -68)
        self.assertEqual(supported["diversity_mechanism"], "SUPPORTED")

    def test_no_promotion_ever(self) -> None:
        for payload in (
            payload_with_retention(68, 58, 68),
            payload_with_retention(68, 58, 40),
            payload_with_retention(68, 68, 68),
        ):
            result = CHECK.evaluate_mechanism(payload)
            self.assertIsNone(result["promoted"])
            self.assertEqual(result["eligible"], [])
            self.assertEqual(result["outcome"], "MECHANISM_READ_COMPLETE")


class LayoutValidationTests(unittest.TestCase):
    def test_wrong_label_order_fails(self) -> None:
        payload = payload_with_retention(68, 58, 68)
        payload["labels"] = list(reversed(payload["labels"]))
        with self.assertRaisesRegex(ValueError, "label order"):
            CHECK.evaluate_mechanism(payload)

    def test_wrong_seed_fails(self) -> None:
        payload = payload_with_retention(68, 58, 68)
        payload["seed"] = 88019
        with self.assertRaisesRegex(ValueError, "seed or row count"):
            CHECK.evaluate_mechanism(payload)

    def test_task_id_mismatch_across_arms_fails(self) -> None:
        payload = payload_with_retention(68, 58, 68)
        for row in payload["rows"]:
            if row["adapter"] == CHECK.CANDIDATE and row["task_id"] == "ret_u_state_0":
                row["task_id"] = "ret_u_state_hijacked"
                break
        with self.assertRaises(ValueError):
            CHECK.evaluate_mechanism(payload)

    def test_kind_imbalance_fails(self) -> None:
        payload = payload_with_retention(68, 58, 68)
        for row in payload["rows"]:
            if row["adapter"] == CHECK.CANDIDATE and row["kind"] == "u_state":
                row["kind"] = "u_trace"
        with self.assertRaises(ValueError):
            CHECK.evaluate_mechanism(payload)


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


class CorpusInheritanceTests(unittest.TestCase):
    DONOR = (
        ROOT
        / "experiments"
        / "qwen35_4b_goal_gap_axis_curriculum_target_match"
        / "data"
        / "sft_axis160.jsonl"
    )
    LOCAL = EXP / "data" / "sft_axis160.jsonl"

    def test_corpus_is_byte_identical_to_donor(self) -> None:
        self.assertEqual(self.LOCAL.read_bytes(), self.DONOR.read_bytes())

    def test_corpus_matches_frozen_pin(self) -> None:
        digest = hashlib.sha256(self.LOCAL.read_bytes()).hexdigest()
        self.assertEqual(
            digest,
            "e7a95d73c619e7c4f20f18ae98ac193e2f57373bd49dc9aede11fd548831686e",
        )


class WriterParityTests(unittest.TestCase):
    def test_finalize_writer_schema_parity(self) -> None:
        payload = payload_with_retention(68, 58, 68)
        base_result = CHECK.evaluate_mechanism(payload)
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "local.json"
            raw = (json.dumps(payload, sort_keys=True) + "\n").encode()
            receipt.write_bytes(raw)
            design = Path(directory) / "design.json"
            design.write_bytes(b"{}\n")
            finalized = CHECK.finalize_mechanism(
                dict(base_result), receipt, raw, design_receipt=design
            )
        # The recovery writer adds exactly the shared fields on top of
        # evaluate_mechanism; eval_local_vllm.py calls the same function, so
        # the two mechanism receipts cannot diverge in schema.
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
        self.assertIn("evaluate_mechanism", eval_source)
        self.assertIn("finalize_mechanism", eval_source)


class FrozenConstantsTests(unittest.TestCase):
    def test_frozen_gate_constants(self) -> None:
        self.assertEqual(CHECK.SEED, 88020)
        self.assertEqual(CHECK.ROWS, 144)
        self.assertEqual(CHECK.AXIS_ROWS, 40)
        self.assertEqual(CHECK.RETENTION_ROWS, 104)
        self.assertEqual(
            CHECK.ARMS,
            (
                "clean_parent",
                "hygiene_explore_direct",
                "replay_clean",
                "axis160_direct",
            ),
        )
        self.assertEqual(CHECK.RETAINED_AT_LEAST, -5)
        self.assertEqual(CHECK.FORGOT_AT_MOST, -6)
        self.assertEqual(CHECK.RETAINED_AT_LEAST, CHECK.FORGOT_AT_MOST + 1)
        self.assertIn("BUDGET", CHECK.ABSTENTION_ANSWERS)
        self.assertIn("INSUFFICIENT", CHECK.ABSTENTION_ANSWERS)


if __name__ == "__main__":
    unittest.main()
