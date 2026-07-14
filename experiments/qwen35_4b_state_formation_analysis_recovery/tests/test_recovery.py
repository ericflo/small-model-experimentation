"""Adversarial tests for the exact registered-prefix recovery seam."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.recovery import (  # noqa: E402
    EXPECTED_ANALYSIS_SHA256,
    EXPECTED_SOURCE_CONTRACT_SHA256,
    ExactRegisteredPrefixSeam,
    PRODUCER_ROOT,
    _phase_authorization,
    installed_path_seam,
    load_producer_context,
    recovery_source_contract,
    seam_preflight,
)


class RecoveryContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.context = load_producer_context()

    def test_exact_producer_contract_is_pinned(self) -> None:
        self.assertEqual(
            self.context.config_module.source_contract_sha256(),
            EXPECTED_SOURCE_CONTRACT_SHA256,
        )
        self.assertEqual(
            self.context.analysis._sha256(Path(self.context.analysis.__file__)),
            EXPECTED_ANALYSIS_SHA256,
        )

    def test_original_helper_reproduces_registered_prefix_defect(self) -> None:
        seam = ExactRegisteredPrefixSeam(self.context)
        with self.assertRaisesRegex(RuntimeError, "expected path is not lexical-canonical"):
            seam.original(Path(seam.raw_prefix))

    def test_seam_is_equivalent_on_canonical_path_and_registered_prefix(self) -> None:
        seam = ExactRegisteredPrefixSeam(self.context)
        expected = seam.original(seam.canonical_prefix)
        self.assertEqual(seam(seam.canonical_prefix), expected)
        self.assertEqual(seam(Path(seam.raw_prefix)), expected)
        descendant = Path(seam.raw_prefix) / "lora_joint_seed7411"
        self.assertEqual(
            seam(descendant), seam.canonical_prefix / "lora_joint_seed7411"
        )

    def test_seam_rejects_traversal_below_registered_prefix(self) -> None:
        seam = ExactRegisteredPrefixSeam(self.context)
        unsafe = Path(seam.raw_prefix + "/../state_formation_capacity_adjudication")
        with self.assertRaisesRegex(RuntimeError, "descendant is not lexical-canonical"):
            seam(unsafe)

    def test_unrelated_alias_still_delegates_to_v11_and_fails(self) -> None:
        seam = ExactRegisteredPrefixSeam(self.context)
        unrelated = self.context.analysis.ROOT / "data" / ".." / "data"
        with self.assertRaisesRegex(RuntimeError, "expected path is not lexical-canonical"):
            seam(unrelated)

    def test_context_restores_exact_original_function(self) -> None:
        original = self.context.analysis._canonical_expected_path
        with installed_path_seam(self.context) as seam:
            self.assertIs(self.context.analysis._canonical_expected_path, seam)
        self.assertIs(self.context.analysis._canonical_expected_path, original)

    def test_context_restores_original_function_after_exception(self) -> None:
        original = self.context.analysis._canonical_expected_path
        with self.assertRaisesRegex(RuntimeError, "synthetic body failure"):
            with installed_path_seam(self.context):
                raise RuntimeError("synthetic body failure")
        self.assertIs(self.context.analysis._canonical_expected_path, original)

    def test_smoke_preflight_opens_no_results(self) -> None:
        receipt = seam_preflight(self.context)
        self.assertEqual(receipt["status"], "EXACT_REGISTERED_PREFIX_SEAM_READY")
        self.assertEqual(receipt["result_rows_opened"], 0)
        self.assertEqual(receipt["benchmark_paths_opened"], 0)
        self.assertEqual(receipt["sealed_contrast_rows_opened"], 0)
        self.assertEqual(set(receipt["control_passes"].values()), {1})

    def test_phase_authorizations_are_only_original_registered_paths(self) -> None:
        analysis = PRODUCER_ROOT / "analysis"
        self.assertIsNone(_phase_authorization("lora_joint"))
        self.assertEqual(
            _phase_authorization("lora_control"), analysis / "lora_joint_trigger.json"
        )
        self.assertEqual(
            _phase_authorization("stage_b_seal"), analysis / "lora_joint_trigger.json"
        )
        self.assertEqual(
            _phase_authorization("fullrank_joint"), analysis / "stage_b_seal.json"
        )
        expected_stage_c = (
            analysis / "fullrank_joint.json"
            if (analysis / "fullrank_joint.json").exists()
            else analysis / "stage_b_seal.json"
        )
        self.assertEqual(_phase_authorization("fullrank_control"), expected_stage_c)

    def test_recovery_contract_contains_only_frozen_design_sources(self) -> None:
        contract = recovery_source_contract()
        paths = {entry["path"] for entry in contract["files"]}
        self.assertNotIn("runs/smoke.json", paths)
        self.assertFalse(any(path.startswith("analysis/") for path in paths))


if __name__ == "__main__":
    unittest.main()
