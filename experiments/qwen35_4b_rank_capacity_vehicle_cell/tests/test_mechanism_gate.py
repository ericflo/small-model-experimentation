import ast
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
REF = ROOT / "experiments" / "qwen35_4b_dose_diversity_mechanism_cell"


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, EXP / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CHECK = load_module("capacity_check_local", "check_local.py")
TRIAL = load_module("capacity_train_trial", "train_trial.py")

AXIS_KINDS = sorted(CHECK.AXIS_KINDS)
RETENTION_KINDS = sorted(CHECK.RETENTION_KINDS)


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


def payload_with_retention(
    parent: int,
    r32: int,
    candidate: int,
    axis_correct: dict[str, dict[str, int]] | None = None,
) -> dict:
    return synthetic_payload(
        axis_correct or {},
        retention_correct={
            CHECK.PARENT: parent,
            CHECK.R32: r32,
            CHECK.CANDIDATE: candidate,
        },
    )


class VerdictBoundaryTests(unittest.TestCase):
    def test_supported_at_exact_boundaries(self) -> None:
        # r32 exactly -6 (forgetting reproduced), candidate exactly -5
        # (retention held): the minimal CAPACITY_SUPPORTED cell.
        result = CHECK.evaluate_mechanism(payload_with_retention(68, 62, 63))
        self.assertEqual(result["readings"]["retention_delta_r32"], -6)
        self.assertEqual(result["readings"]["retention_delta_r64"], -5)
        self.assertEqual(result["capacity_mechanism"], "CAPACITY_SUPPORTED")
        self.assertEqual(result["readings"]["capacity_mechanism"], "CAPACITY_SUPPORTED")

    def test_supported_with_candidate_gain(self) -> None:
        result = CHECK.evaluate_mechanism(payload_with_retention(68, 59, 70))
        self.assertEqual(result["readings"]["retention_delta_r32"], -9)
        self.assertEqual(result["readings"]["retention_delta_r64"], 2)
        self.assertEqual(result["capacity_mechanism"], "CAPACITY_SUPPORTED")

    def test_refuted_at_exact_boundary(self) -> None:
        # r32 -6 and candidate -6: the rank-64 vehicle forgets too.
        result = CHECK.evaluate_mechanism(payload_with_retention(68, 62, 62))
        self.assertEqual(result["readings"]["retention_delta_r64"], -6)
        self.assertEqual(result["capacity_mechanism"], "CAPACITY_REFUTED")

    def test_refuted_deep_forgetting(self) -> None:
        result = CHECK.evaluate_mechanism(payload_with_retention(68, 55, 50))
        self.assertEqual(result["capacity_mechanism"], "CAPACITY_REFUTED")

    def test_screen_instability_at_exact_boundary(self) -> None:
        # r32 exactly -5: the known -9 does not reproduce, so the screen
        # cannot adjudicate — regardless of what the candidate does.
        result = CHECK.evaluate_mechanism(payload_with_retention(68, 63, 68))
        self.assertEqual(result["readings"]["retention_delta_r32"], -5)
        self.assertEqual(result["capacity_mechanism"], "SCREEN_INSTABILITY")

    def test_screen_instability_shadows_candidate_forgetting(self) -> None:
        # Even a badly forgetting candidate reads SCREEN_INSTABILITY when the
        # r32 reference fails to reproduce its forgetting.
        result = CHECK.evaluate_mechanism(payload_with_retention(68, 70, 40))
        self.assertEqual(result["capacity_mechanism"], "SCREEN_INSTABILITY")

    def test_verdict_partition_is_total_over_integers(self) -> None:
        for r32_delta in range(-12, 3):
            for r64_delta in range(-12, 3):
                verdict = CHECK.capacity_mechanism_verdict(r64_delta, r32_delta)
                self.assertIn(verdict, CHECK.VERDICTS)
                if r32_delta >= -5:
                    self.assertEqual(verdict, "SCREEN_INSTABILITY")
                elif r64_delta >= -5:
                    self.assertEqual(verdict, "CAPACITY_SUPPORTED")
                else:
                    self.assertEqual(verdict, "CAPACITY_REFUTED")

    def test_no_promotion_ever(self) -> None:
        for payload in (
            payload_with_retention(68, 59, 68),
            payload_with_retention(68, 59, 40),
            payload_with_retention(68, 68, 68),
        ):
            result = CHECK.evaluate_mechanism(payload)
            self.assertIsNone(result["promoted"])
            self.assertEqual(result["eligible"], [])
            self.assertEqual(result["outcome"], "MECHANISM_READ_COMPLETE")


class InstallPreservedFlagTests(unittest.TestCase):
    def test_equal_axis_totals_preserve_install(self) -> None:
        result = CHECK.evaluate_mechanism(payload_with_retention(68, 59, 63))
        self.assertEqual(result["readings"]["axis_total_r64"], 20)
        self.assertEqual(result["readings"]["axis_total_r32"], 20)
        self.assertTrue(result["readings"]["install_preserved"])

    def test_candidate_axis_gain_preserves_install(self) -> None:
        axis_correct = {
            CHECK.CANDIDATE: {kind: 7 for kind in AXIS_KINDS},
            CHECK.R32: {kind: 6 for kind in AXIS_KINDS},
        }
        result = CHECK.evaluate_mechanism(
            payload_with_retention(68, 59, 63, axis_correct)
        )
        self.assertEqual(result["readings"]["axis_total_r64"], 28)
        self.assertEqual(result["readings"]["axis_total_r32"], 24)
        self.assertTrue(result["readings"]["install_preserved"])

    def test_candidate_axis_loss_drops_the_flag(self) -> None:
        axis_correct = {
            CHECK.CANDIDATE: {kind: 4 for kind in AXIS_KINDS},
            CHECK.R32: {kind: 6 for kind in AXIS_KINDS},
        }
        result = CHECK.evaluate_mechanism(
            payload_with_retention(68, 59, 68, axis_correct)
        )
        self.assertFalse(result["readings"]["install_preserved"])
        # The flag never alters the retention verdict.
        self.assertEqual(result["capacity_mechanism"], "CAPACITY_SUPPORTED")

    def test_flag_is_independent_of_the_verdict_branch(self) -> None:
        axis_correct = {
            CHECK.CANDIDATE: {kind: 9 for kind in AXIS_KINDS},
            CHECK.R32: {kind: 5 for kind in AXIS_KINDS},
        }
        result = CHECK.evaluate_mechanism(
            payload_with_retention(68, 59, 40, axis_correct)
        )
        self.assertEqual(result["capacity_mechanism"], "CAPACITY_REFUTED")
        self.assertTrue(result["readings"]["install_preserved"])


class LayoutValidationTests(unittest.TestCase):
    def test_wrong_label_order_fails(self) -> None:
        payload = payload_with_retention(68, 59, 68)
        payload["labels"] = list(reversed(payload["labels"]))
        with self.assertRaisesRegex(ValueError, "label order"):
            CHECK.evaluate_mechanism(payload)

    def test_wrong_seed_fails(self) -> None:
        payload = payload_with_retention(68, 59, 68)
        payload["seed"] = 88020
        with self.assertRaisesRegex(ValueError, "seed or row count"):
            CHECK.evaluate_mechanism(payload)

    def test_task_id_mismatch_across_arms_fails(self) -> None:
        payload = payload_with_retention(68, 59, 68)
        for row in payload["rows"]:
            if row["adapter"] == CHECK.CANDIDATE and row["task_id"] == "ret_u_state_0":
                row["task_id"] = "ret_u_state_hijacked"
                break
        with self.assertRaises(ValueError):
            CHECK.evaluate_mechanism(payload)

    def test_kind_imbalance_fails(self) -> None:
        payload = payload_with_retention(68, 59, 68)
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
    DONOR = REF / "data" / "sft_axis160.jsonl"
    ORIGIN = (
        ROOT
        / "experiments"
        / "qwen35_4b_goal_gap_axis_curriculum_target_match"
        / "data"
        / "sft_axis160.jsonl"
    )
    LOCAL = EXP / "data" / "sft_axis160.jsonl"

    def test_corpus_is_byte_identical_to_donor_and_origin(self) -> None:
        self.assertEqual(self.LOCAL.read_bytes(), self.DONOR.read_bytes())
        self.assertEqual(self.LOCAL.read_bytes(), self.ORIGIN.read_bytes())

    def test_corpus_matches_frozen_pin(self) -> None:
        digest = hashlib.sha256(self.LOCAL.read_bytes()).hexdigest()
        self.assertEqual(
            digest,
            "e7a95d73c619e7c4f20f18ae98ac193e2f57373bd49dc9aede11fd548831686e",
        )


class WriterParityTests(unittest.TestCase):
    def test_finalize_writer_schema_parity(self) -> None:
        payload = payload_with_retention(68, 59, 68)
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
        self.assertEqual(CHECK.SEED, 88021)
        self.assertEqual(CHECK.ROWS, 144)
        self.assertEqual(CHECK.AXIS_ROWS, 40)
        self.assertEqual(CHECK.RETENTION_ROWS, 104)
        self.assertEqual(
            CHECK.ARMS,
            (
                "clean_parent",
                "axis160_direct",
                "axis160_r64",
            ),
        )
        self.assertEqual(CHECK.RETAINED_AT_LEAST, -5)
        self.assertEqual(CHECK.FORGOT_AT_MOST, -6)
        self.assertEqual(CHECK.RETAINED_AT_LEAST, CHECK.FORGOT_AT_MOST + 1)
        self.assertIn("BUDGET", CHECK.ABSTENTION_ANSWERS)
        self.assertIn("INSUFFICIENT", CHECK.ABSTENTION_ANSWERS)


class RankAlphaContractTests(unittest.TestCase):
    def test_frozen_hyperparameters_are_rank64_alpha128_seed58(self) -> None:
        hyper = TRIAL.expected_hyperparameters()
        self.assertEqual(hyper["rank"], 64)
        self.assertEqual(hyper["alpha"], 128)
        self.assertEqual(hyper["seed"], 58)
        self.assertEqual(hyper["optimizer_steps"], 190)
        self.assertEqual(hyper["lr"], 1e-5)
        self.assertEqual(hyper["max_length"], 4096)
        # Merge scale = alpha/rank must stay 2.0 (the merger validates 2.0).
        self.assertEqual(hyper["alpha"] / hyper["rank"], 2.0)
        self.assertEqual(TRIAL.LORA_RANK, 64)
        self.assertEqual(TRIAL.LORA_ALPHA, 128)

    def test_adapter_config_validator_enforces_rank_alpha_and_base(self) -> None:
        good = {
            "r": 64,
            "lora_alpha": 128,
            "base_model_name_or_path": str(TRIAL.MODEL_PATH.resolve()),
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        }
        self.assertTrue(TRIAL.validate_adapter_config(good))
        for corrupt in (
            {**good, "r": 32},
            {**good, "lora_alpha": 64},
            {**good, "base_model_name_or_path": "Qwen/Qwen3.5-4B"},
            {**good, "target_modules": ["q_proj"]},
        ):
            self.assertFalse(TRIAL.validate_adapter_config(corrupt))


class ModelPathPinTests(unittest.TestCase):
    def test_model_path_points_at_the_designed_fresh_composite(self) -> None:
        expected = (
            ROOT
            / "large_artifacts"
            / "qwen35_4b_universal_fresh_surface_budget_commit_target_match"
            / "merged"
            / "designed_fresh"
        )
        self.assertEqual(TRIAL.MODEL_PATH.resolve(), expected.resolve())
        self.assertEqual(
            TRIAL.MODEL_PATH_WEIGHTS_SHA256,
            "0a3b89cdf57ed8a73590580489d744319c12b44b60991db55b5baba6f7c27979",
        )
        self.assertEqual(
            TRIAL.MODEL_PATH_TREE_SHA256,
            "93433aa2d5f3f0d6d4540126579c09feee1d8502df702c1563bae28eb7f60255",
        )

    def test_trainer_gains_exactly_the_model_path_argument(self) -> None:
        trainer = (EXP / "scripts" / "train_think.py").read_text(encoding="utf-8")
        self.assertIn('"--model-path"', trainer)
        self.assertIn("default=MODEL_ID", trainer)
        # The tokenizer load stays hub-pinned.
        self.assertIn(
            "AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION",
            trainer,
        )

    def test_encoder_and_loss_are_byte_unchanged_from_the_reference(self) -> None:
        """encode_row, the datasets/collator, and the loss must be identical to
        the dose-diversity cell's trainer — --model-path is the ONLY delta."""
        def segments(path: Path) -> dict[str, str]:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
            wanted = {"encode_row", "ThinkSftData", "Collator"}
            found = {}
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted:
                    found[node.name] = ast.get_source_segment(source, node)
            return found

        ours = segments(EXP / "scripts" / "train_think.py")
        reference = segments(REF / "scripts" / "train_think.py")
        self.assertEqual(set(ours), {"encode_row", "ThinkSftData", "Collator"})
        for name in sorted(ours):
            self.assertEqual(ours[name], reference[name], name)


class FreshAdapterNoWarmStartTests(unittest.TestCase):
    def test_wrapper_and_harness_never_pass_warm_start(self) -> None:
        trial = (EXP / "scripts" / "train_trial.py").read_text(encoding="utf-8")
        harness = (EXP / "scripts" / "run.py").read_text(encoding="utf-8")
        for token in ("--warm-start", "warm_start"):
            self.assertNotIn(token, trial)
            self.assertNotIn(token, harness)
        self.assertIn('"--model-path"', trial)
        self.assertIn('"--model-path",', harness)
        self.assertIn('"fresh_adapter": True,', trial)

    def test_merge_uses_the_existing_base_model_argument(self) -> None:
        merge = (EXP / "scripts" / "merge_trained_arm.py").read_text(encoding="utf-8")
        self.assertIn('"--base-model"', merge)
        self.assertIn('payload.get("base_revision") is not None', merge)
        self.assertIn(
            'payload.get("base_model") != str(BASE_COMPOSITE.resolve())', merge
        )
        # The external merger is pinned by sha and already supports the local
        # base; no per-experiment merger copy exists.
        self.assertIn(
            'MERGER_SHA256 = "cb9af8b45ca1e5754cb36f2213b7e25290f6eb16427d1a8b41f0b12b10396672"',
            merge,
        )
        self.assertFalse((EXP / "scripts" / "merge_adapter_local.py").exists())


if __name__ == "__main__":
    unittest.main()
