"""Frozen-constant coherence tests: the gate, the generator, the evaluator,
and the harness must agree on every frozen number, pin, and label before a
GPU event can be authorized. No model is ever loaded."""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))

import check_local as cl  # noqa: E402
import eval_local_vllm as ev  # noqa: E402
import gen_feedloop_curriculum as feedloop  # noqa: E402
import gen_local_gate as gg  # noqa: E402


def harness_constants() -> dict:
    """Read run.py's frozen module constants without importing it (its name
    would collide with other experiments' run modules)."""
    tree = ast.parse((EXP / "scripts" / "run.py").read_text(encoding="utf-8"))
    constants: dict = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    constants[target.id] = ast.literal_eval(node.value)
                except ValueError:
                    continue
    return constants


HARNESS = harness_constants()


class TestFrozenDesignNumbers(unittest.TestCase):
    def test_construction_seed_and_rows(self) -> None:
        self.assertEqual(cl.CONSTRUCTION_SEED, 77160)
        self.assertEqual(cl.PROBE_ROWS, 200)
        self.assertEqual(cl.PER_FORMALISM, 25)
        self.assertEqual(cl.PROBE_ROWS, cl.PER_FORMALISM * len(cl.FORMALISMS))
        self.assertEqual(cl.ROWS_PER_POSITION, 100)

    def test_arms_and_decode_configs(self) -> None:
        self.assertEqual(cl.ARMS, ("think", "nothink"))
        self.assertEqual(cl.GATING_ARM, "think")
        self.assertEqual(cl.ARM_THINKING, {"think": "natural", "nothink": "off"})
        self.assertEqual(cl.MAX_TOKENS, 1024)

    def test_thresholds(self) -> None:
        self.assertEqual(cl.SIGNAL_MIN_ACCURACY, 0.65)
        self.assertEqual(cl.CHANCE_FLOOR, 0.5)
        self.assertEqual(cl.CONFIDENCE, 0.95)
        self.assertEqual(cl.CAP_SCOPE_THRESHOLD, 0.20)
        self.assertEqual(cl.VERDICTS, ("SIGNAL_PRESENT", "SIGNAL_ABSENT"))

    def test_formalisms_match_the_reused_generator(self) -> None:
        self.assertEqual(tuple(cl.FORMALISMS), tuple(feedloop.FEEDLOOP_FORMALISMS))
        self.assertEqual(gg.OVERSAMPLE_MIX, "feedloop=320")
        self.assertEqual(gg.POOL_PER_FORMALISM, 40)
        self.assertEqual(gg.POOL_ROWS, 320)

    def test_dose_scale_lineage_seeds(self) -> None:
        self.assertEqual(gg.DOSE_SCALE_CONSTRUCTION_SEED, 77150)
        self.assertEqual(gg.DOSE_SCALE_HOLDOUT_SEED, 88037)
        self.assertEqual(gg.PRIOR_LOCAL_SEEDS, tuple(range(88000, 88041)))


class TestModuleAgreement(unittest.TestCase):
    def test_eval_agrees_with_the_gate(self) -> None:
        self.assertEqual(ev.SEED, cl.CONSTRUCTION_SEED)
        self.assertEqual(ev.ROWS, cl.PROBE_ROWS)
        self.assertEqual(ev.MAX_TOKENS, cl.MAX_TOKENS)
        self.assertEqual(ev.LABELS, cl.ARMS)
        self.assertEqual(ev.ARM_THINKING, cl.ARM_THINKING)

    def test_harness_agrees_with_the_gate(self) -> None:
        self.assertEqual(HARNESS["SEED"], cl.CONSTRUCTION_SEED)
        self.assertEqual(HARNESS["ROWS"], cl.PROBE_ROWS)
        self.assertEqual(tuple(HARNESS["LABELS"]), cl.ARMS)
        self.assertEqual(tuple(HARNESS["VERDICTS"]), cl.VERDICTS)
        self.assertEqual(
            HARNESS["LOCAL_VERDICT"], "**Verdict:** `PASS_LOCAL_EVENT`."
        )

    def test_composite_pins_agree_everywhere(self) -> None:
        self.assertEqual(gg.EXPECTED_RECEIPT_SHA256, ev.EXPECTED_RECEIPT_SHA256)
        self.assertEqual(gg.EXPECTED_TREE_SHA256, ev.EXPECTED_TREE_SHA256)
        self.assertEqual(gg.EXPECTED_WEIGHTS_SHA256, ev.EXPECTED_WEIGHTS_SHA256)
        self.assertEqual(
            gg.EXPECTED_TREE_SHA256,
            "9eb653d78f05546ca594a831c989fa906d12f3eb7a5a8550d1afcd6bfccc4971",
        )
        harness_pins = HARNESS["EXTERNAL_MERGE_RECEIPTS"]
        self.assertEqual(
            harness_pins,
            {
                gg.COMPOSITE_RECEIPT.relative_to(gg.ROOT).as_posix(): (
                    gg.EXPECTED_RECEIPT_SHA256
                )
            },
        )
        self.assertEqual(
            gg.MERGED.resolve(), ev.MERGED.resolve()
        )

    def test_receipt_paths_agree(self) -> None:
        self.assertEqual(ev.LOCAL_RECEIPT.name, "probe.json")
        self.assertEqual(ev.READOUT_RECEIPT.name, "probe_readout.json")
        self.assertEqual(ev.SOURCE, gg.SOURCE)
        self.assertEqual(ev.INPUT, gg.RUNNER_INPUT)
        self.assertEqual(ev.DESIGN_RECEIPT, gg.RECEIPT)

    def test_run_keys_match_the_frozen_run_order(self) -> None:
        self.assertEqual(
            [ev.run_key(label) for label in ev.LABELS],
            ["think_seed77160", "nothink_seed77160"],
        )

    def test_model_identity_is_the_pinned_qwen(self) -> None:
        self.assertEqual(gg.MODEL_ID, "Qwen/Qwen3.5-4B")
        self.assertEqual(
            gg.MODEL_REVISION, "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
        )


class TestFrozenReceiptContracts(unittest.TestCase):
    def test_design_receipt_on_disk_matches_the_generator(self) -> None:
        receipt_path = gg.RECEIPT
        if not receipt_path.exists():
            self.skipTest("design not yet frozen")
        import json

        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        self.assertEqual(receipt["seed"], cl.CONSTRUCTION_SEED)
        self.assertEqual(receipt["rows"], cl.PROBE_ROWS)
        self.assertEqual(receipt["judgments_total"], 400)
        self.assertEqual(receipt["arms"], list(cl.ARMS))
        self.assertEqual(receipt["gating_arm"], cl.GATING_ARM)
        self.assertEqual(receipt["candidates"], [])
        self.assertEqual(
            receipt["backend"]["arm_thinking"], dict(cl.ARM_THINKING)
        )
        self.assertEqual(receipt["backend"]["max_tokens"], cl.MAX_TOKENS)
        self.assertEqual(receipt["backend"]["run_seed"], cl.CONSTRUCTION_SEED)
        self.assertEqual(
            receipt["readings"]["consequence_partition"]["statements"],
            dict(cl.CONSEQUENCES),
        )
        self.assertEqual(
            receipt["readings"]["cap_contact_diagnostic"]["scope_threshold"],
            cl.CAP_SCOPE_THRESHOLD,
        )
        self.assertIs(
            receipt["readings"]["cap_contact_diagnostic"]["preregistered"], True
        )
        self.assertEqual(receipt["code_pins_deferred"]["files"], [])
        self.assertIs(receipt["firewall"]["benchmark_data_read"], False)
        self.assertIs(receipt["firewall"]["no_aggregate_seed_exists"], True)
        self.assertEqual(receipt["next_authorized_stage"], "local")
        probe = receipt["probe"]
        self.assertEqual(probe["instance_pool_mix"], gg.OVERSAMPLE_MIX)
        self.assertEqual(probe["instance_pool_rows"], gg.POOL_ROWS)
        self.assertEqual(probe["selection"]["rule"], gg.SELECTION_RULE)
        self.assertEqual(
            probe["marker_token_audit"]["tokens"], list(gg.MARKER_TOKENS)
        )
        self.assertEqual(probe["marker_token_audit"]["hits"], 0)
        collisions = probe["listing_collision_audit"]
        self.assertLess(
            collisions["collision_heuristic_ceiling"], cl.SIGNAL_MIN_ACCURACY
        )
        freshness = receipt["freshness"]
        self.assertEqual(freshness["unique_probe_messages"], cl.PROBE_ROWS)
        self.assertEqual(freshness["prior_local_overlap"], 0)
        self.assertEqual(
            freshness["regenerated_dose_scale_treatment"]["overlap"], 0
        )
        self.assertEqual(
            freshness["regenerated_dose_scale_holdout"]["overlap"], 0
        )
        for block in ("dose_scale_sources", "predecessor_sources", "predecessor_gates"):
            self.assertTrue(freshness[block])
            for entry in freshness[block].values():
                self.assertEqual(entry["overlap"], 0)


if __name__ == "__main__":
    unittest.main()
