import importlib.util
import json
import sys
import unittest
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, EXP / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


MAT = load_module("statechain_materialize_streams", "materialize_streams.py")
VAL = load_module("statechain_validate_streams", "validate_streams.py")
TRIAL = load_module("statechain_train_trial", "train_trial.py")


class AllocationTests(unittest.TestCase):
    def test_largest_remainder_allocation_is_exact_and_bounded(self) -> None:
        sizes = {("a",): 7, ("b",): 5, ("c",): 3}
        quotas = MAT.allocation(sizes, 10)
        self.assertEqual(sum(quotas.values()), 10)
        for key, size in sizes.items():
            self.assertLessEqual(quotas[key], size)
            self.assertGreaterEqual(quotas[key], size * 10 // 15)
        with self.assertRaises(ValueError):
            MAT.allocation(sizes, 16)

    def test_slot_order_is_a_stable_permutation(self) -> None:
        order = MAT.slot_order()
        self.assertEqual(sorted(order), list(range(MAT.ROWS_PER_ARM)))
        self.assertEqual(order, MAT.slot_order())
        self.assertNotEqual(order, list(range(MAT.ROWS_PER_ARM)))

    def test_deterministic_rank_depends_on_namespace_and_content(self) -> None:
        first = MAT.deterministic_rank("replay-core", 0, "line")
        self.assertEqual(first, MAT.deterministic_rank("replay-core", 0, "line"))
        self.assertNotEqual(first, MAT.deterministic_rank("other", 0, "line"))
        self.assertNotEqual(first, MAT.deterministic_rank("replay-core", 1, "line"))


class ExactMatchSolverTests(unittest.TestCase):
    def test_solver_returns_disjoint_exact_blocks(self) -> None:
        # 400 synthetic replay rows with two length classes; the treatment
        # vector is expressible as control(240) - filler(80) exactly.
        lengths = []
        for index in range(400):
            value = 10 if index % 2 == 0 else 14
            lengths.append(
                {
                    "forward": value,
                    "nonzero_target": value // 2,
                    "absolute_loss_mass_x5": value // 2,
                }
            )
        available = list(range(400))
        # 160*10 + 80*14 - (40*10 + 40*14) == 1760; nonzero/mass mirror halves.
        treatment_vector = (1760, 880, 880)
        filler, control, solver = MAT.solve_exact_match(
            available, lengths, treatment_vector
        )
        self.assertTrue(solver["success"])
        self.assertEqual(len(filler), MAT.FILLER_ROWS)
        self.assertEqual(len(control), MAT.CONTROL_ROWS)
        self.assertFalse(set(filler) & set(control))
        deltas = tuple(
            sum(lengths[i][axis] for i in control)
            - sum(lengths[i][axis] for i in filler)
            for axis in MAT.MATCH_AXES
        )
        self.assertEqual(deltas, treatment_vector)

    def test_solver_reports_infeasible_without_raising(self) -> None:
        lengths = [
            {"forward": 10, "nonzero_target": 5, "absolute_loss_mass_x5": 5}
            for _ in range(400)
        ]
        # An odd forward total is unreachable from uniform even rows.
        filler, control, solver = MAT.solve_exact_match(
            list(range(400)), lengths, (1601, 800, 800)
        )
        self.assertFalse(solver["success"])
        self.assertEqual(filler, [])
        self.assertEqual(control, [])


class FrozenExposureContractTests(unittest.TestCase):
    def test_stream_geometry_constants(self) -> None:
        self.assertEqual(MAT.STREAM_ORDER_SEED, 55150)
        self.assertEqual(MAT.CORE_ROWS, 1280)
        self.assertEqual(MAT.TREATMENT_ROWS, 160)
        self.assertEqual(MAT.FILLER_ROWS, 80)
        self.assertEqual(MAT.CONTROL_ROWS, 240)
        self.assertEqual(MAT.ROWS_PER_ARM, 1520)
        self.assertEqual(
            MAT.MATCH_AXES, ("forward", "nonzero_target", "absolute_loss_mass_x5")
        )
        self.assertEqual(MAT.CONTROL_OUT.name, "replay_ctl4.jsonl")
        self.assertEqual(MAT.CANDIDATE_OUT.name, "statechain_clean.jsonl")

    def test_validator_kind_quotas_and_arm_names(self) -> None:
        self.assertEqual(
            VAL.CANDIDATE_TREATMENT_KINDS, {"u_statechain": 160}
        )
        self.assertEqual(set(VAL.STREAMS), {"replay_ctl4", "statechain_clean"})
        self.assertIsNotNone(VAL.MANIFEST_SHA256)
        with self.assertRaises(ValueError):
            VAL.check_arm_kinds(
                {"replay_ctl4": {"kinds": {"atom": 1519, "u_statechain": 1}}}
            )
        with self.assertRaises(ValueError):
            VAL.check_arm_kinds(
                {
                    "statechain_clean": {
                        "kinds": {"atom": 1360, "u_statechain": 80, "u_feedloop": 80}
                    }
                }
            )

    def test_training_wrapper_constants_match_the_committed_receipt(self) -> None:
        receipt = json.loads(
            (EXP / "data" / "stream_token_receipt.json").read_text(encoding="utf-8")
        )
        self.assertEqual(TRIAL.STREAM_TOKEN_RECEIPT_SHA256 is None, False)
        self.assertEqual(TRIAL.EXPECTED_FORWARD, receipt["forward_tokens_per_arm"])
        self.assertEqual(TRIAL.EXPECTED_NONZERO, receipt["nonzero_target_tokens_per_arm"])
        self.assertEqual(
            TRIAL.EXPECTED_MASS_X5, receipt["absolute_loss_mass_x5_per_arm"]
        )
        self.assertEqual(receipt["skipped_rows"], 0)
        self.assertEqual(receipt["rows_per_arm"], 1520)
        for arm in ("replay_ctl4", "statechain_clean"):
            spans = receipt["files"][arm]["spans_per_epoch"]
            self.assertEqual(spans["forward"], TRIAL.EXPECTED_FORWARD)
            self.assertEqual(spans["nonzero_target"], TRIAL.EXPECTED_NONZERO)
            self.assertEqual(spans["absolute_loss_mass_x5"], TRIAL.EXPECTED_MASS_X5)

    def test_training_hyperparameters_are_frozen(self) -> None:
        self.assertEqual(
            TRIAL.expected_hyperparameters(),
            {
                "epochs": 1.0,
                "lr": 1e-5,
                "rank": 32,
                "alpha": 64,
                "batch_size": 1,
                "grad_accum": 8,
                "max_length": 4096,
                "w_think": 0.2,
                "w_close": 0.2,
                "seed": 73,
                "optimizer_steps": 190,
            },
        )
        self.assertEqual(TRIAL.LORA_RANK, 32)
        self.assertEqual(TRIAL.LORA_ALPHA, 64)
        self.assertEqual(
            TRIAL.ARM_PREREQUISITES,
            {"replay_ctl4": (), "statechain_clean": ("replay_ctl4",)},
        )
        self.assertEqual(
            set(TRIAL.PUBLISHED_ARM_HASHES), {"replay_ctl4", "statechain_clean"}
        )

    def test_require_pin_fails_closed_on_none(self) -> None:
        with self.assertRaises(SystemExit):
            TRIAL.require_pin(None, "PUBLISHED_ARM_HASHES['replay_ctl4']")
        self.assertEqual(TRIAL.require_pin("x", "name"), "x")

    def test_no_warm_start_pathway_exists(self) -> None:
        trial_source = (EXP / "scripts" / "train_trial.py").read_text(encoding="utf-8")
        merge_source = (EXP / "scripts" / "merge_trained_arm.py").read_text(
            encoding="utf-8"
        )
        for token in ("--warm" + "-start", "warm" + "_start"):
            self.assertNotIn(token, trial_source)
            self.assertNotIn(token, merge_source)

    def test_adapter_config_contract(self) -> None:
        good = {
            "r": 32,
            "lora_alpha": 64,
            "base_model_name_or_path": str(TRIAL.MODEL_PATH.resolve()),
            "target_modules": list(TRIAL.LORA_TARGET_MODULES),
        }
        self.assertTrue(TRIAL.validate_adapter_config(good))
        self.assertFalse(TRIAL.validate_adapter_config({**good, "r": 64}))
        self.assertFalse(
            TRIAL.validate_adapter_config(
                {**good, "base_model_name_or_path": "Qwen/Qwen3.5-4B"}
            )
        )


if __name__ == "__main__":
    unittest.main()
