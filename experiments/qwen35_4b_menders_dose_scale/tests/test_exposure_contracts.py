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


MAT = load_module("menders_materialize_streams", "materialize_streams.py")
VAL = load_module("menders_validate_streams", "validate_streams.py")
TRIAL = load_module("menders_train_trial", "train_trial.py")


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
    def make_lengths(self, count: int, values: list[int]) -> list[dict]:
        return [
            {
                "forward": values[index % len(values)],
                "nonzero_target": values[index % len(values)] // 2,
                "absolute_loss_mass_x5": values[index % len(values)] // 2,
            }
            for index in range(count)
        ]

    def test_solver_returns_exact_blocks_with_arm_level_multiplicity(self) -> None:
        # 900 available rows alternating 10/14, 200 core rows of 20; the
        # treatment vector is expressible as control(1000) - filler(200).
        lengths = self.make_lengths(900, [10, 14]) + self.make_lengths(200, [20])
        available = list(range(900))
        core = list(range(900, 1100))
        treatment_vector = (9600, 4800, 4800)
        filler, first, second, core_extra, solver = MAT.solve_exact_match(
            available, core, lengths, treatment_vector
        )
        self.assertTrue(solver["success"])
        self.assertEqual(len(filler), MAT.FILLER_ROWS)
        control = first + second + core_extra
        self.assertEqual(len(control), MAT.CONTROL_ROWS)
        self.assertTrue(set(second) <= set(first))
        self.assertTrue(set(core_extra) <= set(core))
        self.assertFalse(set(first) & set(core))
        deltas = tuple(
            sum(lengths[i][axis] for i in control)
            - sum(lengths[i][axis] for i in filler)
            for axis in MAT.MATCH_AXES
        )
        self.assertEqual(deltas, treatment_vector)

    def test_solver_reports_infeasible_without_raising(self) -> None:
        lengths = self.make_lengths(900, [10]) + self.make_lengths(200, [10])
        # An odd forward total is unreachable from uniform even rows.
        filler, first, second, core_extra, solver = MAT.solve_exact_match(
            list(range(900)), list(range(900, 1100)), lengths, (1601, 800, 800)
        )
        self.assertFalse(solver["success"])
        self.assertEqual((filler, first, second, core_extra), ([], [], [], []))


class DeconflictTests(unittest.TestCase):
    def test_colliding_positions_are_swapped_away(self) -> None:
        control = ["r1", "r2", "r3", "r4"]
        candidate = ["t1", "t2", "r3", "r4"]
        fixed, swaps = MAT.deconflict_blocks(control, candidate)
        self.assertEqual(sorted(fixed), sorted(control))
        self.assertEqual(swaps, 2)
        for index in range(len(fixed)):
            self.assertNotEqual(fixed[index], candidate[index])

    def test_clean_blocks_pass_through_unchanged(self) -> None:
        control = ["r1", "r2"]
        candidate = ["t1", "t2"]
        fixed, swaps = MAT.deconflict_blocks(control, candidate)
        self.assertEqual(fixed, control)
        self.assertEqual(swaps, 0)


class FrozenExposureContractTests(unittest.TestCase):
    def test_stream_geometry_constants(self) -> None:
        self.assertEqual(MAT.STREAM_ORDER_SEED, 55140)
        self.assertEqual(MAT.CORE_ROWS, 1280)
        self.assertEqual(MAT.TREATMENT_ROWS, 800)
        self.assertEqual(MAT.FILLER_ROWS, 200)
        self.assertEqual(MAT.CONTROL_ROWS, 1000)
        self.assertEqual(MAT.VARIABLE_BLOCK_ROWS, 1000)
        self.assertEqual(MAT.ROWS_PER_ARM, 2280)
        self.assertEqual(
            MAT.MATCH_AXES, ("forward", "nonzero_target", "absolute_loss_mass_x5")
        )
        self.assertEqual(MAT.CONTROL_OUT.name, "replay_ctl3.jsonl")
        self.assertEqual(MAT.CANDIDATE_OUT.name, "feedloop_scale.jsonl")

    def test_validator_kind_quotas_and_arm_names(self) -> None:
        self.assertEqual(
            VAL.CANDIDATE_TREATMENT_KINDS, {"u_feedloop": 800}
        )
        self.assertEqual(set(VAL.STREAMS), {"replay_ctl3", "feedloop_scale"})
        self.assertIsNotNone(VAL.MANIFEST_SHA256)
        self.assertEqual(VAL.CONTROL_ARM_MAX_MULTIPLICITY, 2)
        with self.assertRaises(ValueError):
            VAL.check_arm_kinds(
                {"replay_ctl3": {"kinds": {"atom": 2279, "u_feedloop": 1}}}
            )
        with self.assertRaises(ValueError):
            VAL.check_arm_kinds(
                {
                    "feedloop_scale": {
                        "kinds": {"atom": 1480, "u_feedloop": 400, "u_statechain": 400}
                    }
                }
            )

    def test_multiplicity_validator_fails_closed(self) -> None:
        manifest = {
            "row_duplication": {
                "candidate_arm_duplicated_rows": 0,
                "control_arm_max_multiplicity": 2,
                "control_arm_repeated_rows": 1,
            }
        }
        streams = {
            "replay_ctl3": ["a", "a", "b"],
            "feedloop_scale": ["c", "d", "e"],
        }
        result = VAL.check_row_multiplicity(streams, manifest)
        self.assertEqual(result["control_arm_repeated_rows"], 1)
        # Candidate duplicates abort.
        with self.assertRaises(ValueError):
            VAL.check_row_multiplicity(
                {"replay_ctl3": ["a", "b"], "feedloop_scale": ["c", "c"]}, manifest
            )
        # A control row over the arm-level cap aborts.
        with self.assertRaises(ValueError):
            VAL.check_row_multiplicity(
                {"replay_ctl3": ["a", "a", "a"], "feedloop_scale": ["c", "d", "e"]},
                manifest,
            )
        # A manifest count that disagrees with the streams aborts.
        with self.assertRaises(ValueError):
            VAL.check_row_multiplicity(
                {"replay_ctl3": ["a", "b", "c"], "feedloop_scale": ["d", "e", "f"]},
                manifest,
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
        self.assertEqual(receipt["rows_per_arm"], 2280)
        self.assertEqual(
            receipt["row_multiplicity"]["candidate_arm_repeated_rows"], 0
        )
        self.assertLessEqual(
            receipt["row_multiplicity"]["control_arm_max_multiplicity"], 2
        )
        for arm in ("replay_ctl3", "feedloop_scale"):
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
                "seed": 71,
                "optimizer_steps": 285,
            },
        )
        self.assertEqual(TRIAL.LORA_RANK, 32)
        self.assertEqual(TRIAL.LORA_ALPHA, 64)
        self.assertEqual(TRIAL.EXPECTED_ROWS, 2280)
        self.assertEqual(
            TRIAL.ARM_PREREQUISITES,
            {"replay_ctl3": (), "feedloop_scale": ("replay_ctl3",)},
        )
        self.assertEqual(
            set(TRIAL.PUBLISHED_ARM_HASHES), {"replay_ctl3", "feedloop_scale"}
        )

    def test_require_pin_fails_closed_on_none(self) -> None:
        with self.assertRaises(SystemExit):
            TRIAL.require_pin(None, "PUBLISHED_ARM_HASHES['replay_ctl3']")
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
