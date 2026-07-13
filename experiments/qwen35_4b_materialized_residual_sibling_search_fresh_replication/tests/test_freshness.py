from __future__ import annotations

import copy
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


EXP = Path(__file__).resolve().parents[1]
RUN_PATH = EXP / "scripts" / "run.py"
SPEC = importlib.util.spec_from_file_location("fresh_construction_audit", RUN_PATH)
assert SPEC and SPEC.loader
run = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = run
SPEC.loader.exec_module(run)


class FreshnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = yaml.safe_load(run.CONFIG.read_text())
        cls.common = run.build_common_panel(cls.config)
        excluded, cls.prior_receipt = run.prior_function_fingerprints(cls.common)
        cls.splits, cls.construction, _ = run.build_splits(
            cls.config, excluded_fingerprints=excluded
        )
        cls.lineage = run.verify_parent_lineage(run.ROOT)
        cls.freshness = run._parent_freshness_receipt(
            cls.splits, cls.common, cls.lineage
        )

    def test_fresh_construction_fills_frozen_geometry(self) -> None:
        self.assertEqual(
            self.construction["split_rows"],
            {"mechanics": 24, "qualification": 48, "confirmation": 192},
        )
        self.assertEqual(self.construction["eligible_function_fingerprints"], 3526)
        self.assertEqual(self.construction["excluded_prior_fingerprints"], 0)
        self.assertTrue(
            self.prior_receipt[
                "scientific_parent_exempted_from_function_rejection"
            ]
        )

    def test_required_parent_intersections_are_zero(self) -> None:
        self.assertEqual(
            self.freshness["required_zero_intersections"],
            {
                "task_ids": 0,
                "public_instance_payloads": 0,
                "model_facing_mechanics_prompts": 0,
                "terminal_suffix_materialized_prompts": 0,
            },
        )
        self.assertEqual(
            self.freshness["descriptive_finite_dsl_reuse"],
            {
                "parent_functions_re_evaluated_on_fresh_panel": 264,
                "parent_functions_invalid_on_fresh_panel": 0,
                "shared_behavior_functions": 56,
                "shared_concrete_triples": 41,
                "shared_two_operation_suffixes": 181,
                "rejection_policy": "reported_not_rejected",
            },
        )

    def test_parent_task_id_injection_fails_closed(self) -> None:
        mutated = copy.deepcopy(self.splits)
        mutated["mechanics"][0]["task_id"] = "mechanics-00000"
        with self.assertRaisesRegex(RuntimeError, "overlap gate failed"):
            run._parent_freshness_receipt(mutated, self.common, self.lineage)

    def test_parent_public_payload_injection_fails_closed(self) -> None:
        parent_public, _, _, _ = run._authenticated_parent_inputs(self.lineage)
        mutated = copy.deepcopy(self.splits)
        mutated["mechanics"][0]["visible"] = copy.deepcopy(
            parent_public[0]["visible"]
        )
        mutated["mechanics"][0]["unlabeled_probe_inputs"] = copy.deepcopy(
            parent_public[0]["unlabeled_probe_inputs"]
        )
        with self.assertRaisesRegex(RuntimeError, "overlap gate failed"):
            run._parent_freshness_receipt(mutated, self.common, self.lineage)

    def test_nonzero_prompt_or_malformed_receipt_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "overlap gate failed"):
            run._require_zero_freshness_intersections(
                {
                    "task_ids": 0,
                    "public_instance_payloads": 0,
                    "model_facing_mechanics_prompts": 1,
                    "terminal_suffix_materialized_prompts": 1,
                }
            )
        with self.assertRaisesRegex(RuntimeError, "wrong schema"):
            run._require_zero_freshness_intersections({"task_ids": 0})

    def test_cross_request_file_prompt_injection_fails_closed(self) -> None:
        _, _, parent_prompts, _ = run._authenticated_parent_inputs(self.lineage)
        fresh_prompts = run._fresh_mechanics_prompt_hashes(
            self.splits["mechanics"]
        )
        injected = next(iter(parent_prompts["direct_requests.jsonl"]))
        self.assertNotIn(injected, parent_prompts["listwise_requests.jsonl"])
        fresh_prompts["listwise_requests.jsonl"].add(injected)
        overlap = run._prompt_overlap_counts(parent_prompts, fresh_prompts)
        self.assertEqual(
            overlap["by_same_request_file"]["listwise_requests.jsonl"], 0
        )
        self.assertEqual(overlap["all_parent_vs_all_fresh"], 1)
        with self.assertRaisesRegex(RuntimeError, "overlap gate failed"):
            run._require_zero_freshness_intersections(
                {
                    "task_ids": 0,
                    "public_instance_payloads": 0,
                    "model_facing_mechanics_prompts": overlap[
                        "all_parent_vs_all_fresh"
                    ],
                    "terminal_suffix_materialized_prompts": overlap[
                        "terminal_suffix_materialized"
                    ],
                }
            )

    def test_unrelated_prior_scan_rejects_symlink_and_malformed_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.jsonl"
            target.write_text("{}\n")
            link = root / "link.jsonl"
            link.symlink_to(target)
            with self.assertRaisesRegex(RuntimeError, "symlink"):
                run._verified_unrelated_prior_path(link, experiments_root=root)
            malformed = root / "malformed.jsonl"
            malformed.write_text("{}\n\n")
            with self.assertRaisesRegex(RuntimeError, "blank line"):
                run._strict_jsonl(malformed)


if __name__ == "__main__":
    unittest.main()
