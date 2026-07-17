import importlib.util
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


BENCH = load_module("stc_run_benchmark_constants", "run_benchmark.py")

FAMILIES = sorted(BENCH.PUBLIC_FAMILIES)


class FrozenEventIdentityTests(unittest.TestCase):
    def test_event_identity(self) -> None:
        self.assertEqual(BENCH.FROZEN_NAME, "confirmation")
        self.assertEqual(BENCH.FROZEN_TIER, "medium")
        self.assertEqual(BENCH.FROZEN_THINK_BUDGET, 1024)
        self.assertEqual(
            BENCH.SEED_ORDER, (78170, 78171, 78172, 78173, 78174, 78175)
        )
        self.assertEqual(BENCH.MODEL_ORDER, ("count_walk", "state_track"))
        self.assertEqual(BENCH.FROZEN_PARENT, "count_walk")
        self.assertEqual(BENCH.CANDIDATE, "state_track")
        self.assertEqual(BENCH.AGG_TIE_EPSILON, 1e-12)
        self.assertEqual(BENCH.WINS_THRESHOLD, 4)
        self.assertEqual(BENCH.EVENTS, 6)

    def test_prior_event_is_78169_and_never_counted(self) -> None:
        self.assertEqual(BENCH.PRIOR_EVENT["seed"], 78169)
        self.assertFalse(BENCH.PRIOR_EVENT["counted_in_verdict"])
        self.assertNotIn(BENCH.PRIOR_EVENT["seed"], BENCH.SEED_ORDER)
        self.assertEqual(
            BENCH.PRIOR_EVENT["summary_sha256"],
            "187cc3acfe81016899cb08a8bebf5f6045a6cabba9868edd5379c51708ec1192",
        )


class FrozenPinTests(unittest.TestCase):
    def test_gateway_pin_matches_the_reference_cell(self) -> None:
        self.assertEqual(
            BENCH.GATEWAY_SHA256,
            "53cf6533dbd710eb167503363c39f73dbf7559a0d91f40a00436a3c218a01c17",
        )

    def test_tree_and_weights_pins_are_the_design_time_constants(self) -> None:
        self.assertEqual(
            BENCH.FROZEN_TREE_SHA256,
            {
                "count_walk": "d5fdc55c0238ffbe2465bd73a5f9d63f442ad4083ff9eb477c9887e15e3da6b1",
                "state_track": "45fd2925e417c82e4848b2ca89907934df9e60503b6529af0bddbd8aa359be7e",
            },
        )
        self.assertEqual(
            BENCH.FROZEN_WEIGHTS_SHA256,
            {
                "count_walk": "ddd7bc4b5b8f4f2393996148bcb1b411a8be4d7f03430babe789b3534b9850a3",
                "state_track": "b4bafbb7d3ff8dedd2fa216bc9c62997d960d43a6cac22a88976245bcc35d1c1",
            },
        )
        self.assertEqual(BENCH.WEIGHTS_SIZE_BYTES, 9_078_620_536)

    def test_arm_provenance_pins(self) -> None:
        self.assertEqual(
            BENCH.ARM_PROVENANCE["count_walk"]["merge_receipt_sha256"],
            "840edca0638b9e291bb34fde28b4b530df8743faf9b7b18b7f2358ce55ec4c36",
        )
        self.assertEqual(
            BENCH.ARM_PROVENANCE["count_walk"]["inner_receipt_sha256"],
            "3c432f110fe96a508d6a75ab34e4a649671a3d7b2d942f3346cab609bef437d7",
        )
        self.assertEqual(
            BENCH.ARM_PROVENANCE["state_track"]["merge_receipt_sha256"],
            "089f280eab1b6f4afd53e636a49f1b4fd92efd5fa1ee42a1a07e35e49a98c94e",
        )
        self.assertEqual(
            BENCH.ARM_PROVENANCE["state_track"]["inner_receipt_sha256"],
            "d23862f70cdbb71b2b232bee0501e65f45a432cacd3e37189418194e27493a0d",
        )

    def test_no_todo_pin_placeholder_exists(self) -> None:
        for table in (BENCH.FROZEN_TREE_SHA256, BENCH.FROZEN_WEIGHTS_SHA256):
            for value in table.values():
                self.assertRegex(value, r"^[0-9a-f]{64}$")
        source = (EXP / "scripts" / "run_benchmark.py").read_text(encoding="utf-8")
        self.assertNotIn("TODO-PIN", source.replace("no TODO-PIN slot", ""))

    def test_provenance_copies_exist_with_pinned_shas(self) -> None:
        for label, block in BENCH.ARM_PROVENANCE.items():
            copy_path = block["copy"]
            self.assertTrue(copy_path.is_file(), label)
            self.assertEqual(
                BENCH.sha256_file(copy_path), block["merge_receipt_sha256"], label
            )
        prior_copy = BENCH.PRIOR_EVENT["summary_copy"]
        self.assertTrue(prior_copy.is_file())
        self.assertEqual(
            BENCH.sha256_file(prior_copy), BENCH.PRIOR_EVENT["summary_sha256"]
        )


def synthetic_scores(deltas: dict[int, float], parent: float = 0.30) -> dict:
    scores = {}
    for index, seed in enumerate(BENCH.SEED_ORDER):
        per_family_parent = {family: 0.2 for family in FAMILIES}
        per_family_candidate = {family: 0.2 for family in FAMILIES}
        scores[seed] = {
            "count_walk": {"aggregate": parent, "per_family": per_family_parent},
            "state_track": {
                "aggregate": parent + deltas.get(seed, 0.0),
                "per_family": per_family_candidate,
            },
        }
    return scores


def synthetic_budget(within: bool = True) -> dict:
    return {
        seed: {
            label: {"within_budget": within, "wall_seconds": 100.0}
            for label in BENCH.MODEL_ORDER
        }
        for seed in BENCH.SEED_ORDER
    }


def synthetic_receipts() -> dict:
    return {
        seed: {
            label: {"path": f"receipt/{seed}/{label}.json", "sha256": "0" * 64}
            for label in BENCH.MODEL_ORDER
        }
        for seed in BENCH.SEED_ORDER
    }


class ReadoutSchemaTests(unittest.TestCase):
    def _build(self, deltas: dict[int, float], within: bool = True) -> dict:
        prior = BENCH.load_prior_reference()
        return BENCH.build_readout(
            synthetic_scores(deltas),
            synthetic_budget(within),
            dict(BENCH.PRIOR_IMPLEMENTATION),
            prior,
            synthetic_receipts(),
        )

    def test_readout_schema_and_verdict_wiring(self) -> None:
        readout = self._build({seed: 0.02 for seed in BENCH.SEED_ORDER})
        self.assertEqual(
            set(readout),
            {
                "schema_version", "experiment_id", "stage", "name", "tier",
                "think_budget", "seeds", "benchmark_data_read", "promoted",
                "outcome", "verdict", "frozen_claim", "paired_comparison_valid",
                "provenance", "prior_event", "benchmark_implementation",
                "receipts", "scores", "budget", "readings",
            },
        )
        self.assertEqual(readout["schema_version"], 1)
        self.assertEqual(readout["experiment_id"], EXP.name)
        self.assertEqual(readout["seeds"], list(BENCH.SEED_ORDER))
        self.assertFalse(readout["benchmark_data_read"])
        self.assertIsNone(readout["promoted"])
        self.assertEqual(readout["outcome"], "CONFIRMATION_READ_COMPLETE")
        self.assertEqual(readout["verdict"], "CONFIRMED")
        self.assertEqual(readout["frozen_claim"], BENCH.FROZEN_CLAIMS["CONFIRMED"])
        self.assertEqual(
            set(readout["readings"]),
            {"paired_replication", "per_seed", "budget_integrity"},
        )
        self.assertEqual(readout["prior_event"]["seed"], 78169)
        self.assertFalse(readout["prior_event"]["counted_in_verdict"])
        per_seed = readout["readings"]["per_seed"]
        self.assertEqual(set(per_seed), {str(seed) for seed in BENCH.SEED_ORDER})
        for block in per_seed.values():
            self.assertTrue(block["descriptive_only"])
            self.assertEqual(set(block["aggregates"]), set(BENCH.MODEL_ORDER))
            self.assertIn("candidate_vs_parent_family_gate", block)
            self.assertIn("candidate_minus_parent_per_family", block)

    def test_all_negative_readout_is_not_confirmed(self) -> None:
        readout = self._build({seed: -0.02 for seed in BENCH.SEED_ORDER})
        self.assertEqual(readout["verdict"], "NOT_CONFIRMED")
        self.assertIn("retired as seed noise", readout["frozen_claim"])

    def test_positive_mean_few_wins_readout_is_ambiguous(self) -> None:
        seeds = BENCH.SEED_ORDER
        deltas = {
            seeds[0]: 0.05, seeds[1]: 0.05, seeds[2]: 0.05,
            seeds[3]: -0.01, seeds[4]: -0.01, seeds[5]: -0.01,
        }
        readout = self._build(deltas)
        self.assertEqual(readout["verdict"], "AMBIGUOUS")

    def test_over_budget_arm_scopes_but_never_gates(self) -> None:
        readout = self._build({seed: 0.02 for seed in BENCH.SEED_ORDER}, within=False)
        self.assertEqual(readout["verdict"], "CONFIRMED")
        self.assertFalse(readout["paired_comparison_valid"])
        self.assertIsNotNone(readout["readings"]["budget_integrity"]["reason"])

    def test_implementation_drift_refuses_the_readout(self) -> None:
        prior = BENCH.load_prior_reference()
        drifted = dict(BENCH.PRIOR_IMPLEMENTATION)
        drifted["source_file_count"] = 57
        with self.assertRaisesRegex(ValueError, "not comparable"):
            BENCH.build_readout(
                synthetic_scores({}),
                synthetic_budget(),
                drifted,
                prior,
                synthetic_receipts(),
            )


class VerdictAndReviewMarkerTests(unittest.TestCase):
    def test_verdict_string_is_the_frozen_marker(self) -> None:
        self.assertEqual(BENCH.BENCH_VERDICT, "**Verdict:** `PASS_BENCHMARK_EVENT`.")

    def test_review_gate_refuses_without_the_marker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            review = Path(directory) / "benchmark_design_review.md"
            with self.assertRaisesRegex(ValueError, "not been authorized"):
                BENCH.require_verdict(review, BENCH.BENCH_VERDICT, "benchmark review")
            review.write_text("# Review\n\nLooks fine.\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not been authorized"):
                BENCH.require_verdict(review, BENCH.BENCH_VERDICT, "benchmark review")
            review.write_text(f"# Review\n\n{BENCH.BENCH_VERDICT}\n", encoding="utf-8")
            BENCH.require_verdict(review, BENCH.BENCH_VERDICT, "benchmark review")


class ScoreFinitenessTests(unittest.TestCase):
    def test_valid_scores(self) -> None:
        for value in (0.0, 0.5, 1.0, 1):
            self.assertTrue(BENCH._valid_score(value))

    def test_invalid_scores(self) -> None:
        for value in (float("nan"), float("inf"), -0.01, 1.01, True, None, "0.5"):
            self.assertFalse(BENCH._valid_score(value))

    def test_wall_seconds_guard(self) -> None:
        for value in (0.0, 1, 3600.5):
            self.assertTrue(BENCH._valid_wall_seconds(value))
        for value in (float("nan"), float("inf"), -1.0, True, None, "1"):
            self.assertFalse(BENCH._valid_wall_seconds(value))


class CandidateVsParentGateTests(unittest.TestCase):
    def test_strict_win_partition(self) -> None:
        parent = {family: 0.1 for family in FAMILIES}
        candidate = dict(parent)
        candidate["siftstack"] = 0.3
        candidate["warren"] = 0.0
        row = BENCH.candidate_vs_parent_gate(parent, candidate)
        self.assertEqual(row["strict_wins"], 1)
        self.assertEqual(row["wins"], ["siftstack"])
        self.assertEqual(row["losses"], ["warren"])
        self.assertEqual(len(row["ties"]), 8)
        self.assertFalse(row["candidate_beats_parent_every_family"])

    def test_ten_strict_wins_pass(self) -> None:
        parent = {family: 0.1 for family in FAMILIES}
        candidate = {family: 0.2 for family in FAMILIES}
        self.assertTrue(
            BENCH.candidate_vs_parent_gate(parent, candidate)[
                "candidate_beats_parent_every_family"
            ]
        )


if __name__ == "__main__":
    unittest.main()
