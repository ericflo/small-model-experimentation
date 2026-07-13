from __future__ import annotations

import copy
import hashlib
import json
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import oracle_control as oracle_module  # noqa: E402
from src.oracle_control import (  # noqa: E402
    OracleControlError,
    analyze_oracle_control,
    analyze_positive_control_records,
    build_oracle_positive_control_records,
    build_oracle_prediction_table,
    generate_control_rows,
    produce_oracle_analysis_receipt,
    summarize_positive_control_records,
    validate_control_rows,
    validate_oracle_analysis_receipt,
)


SPLITS = (
    "train",
    "validation",
    "depth_extrapolation",
    "joint_holdout",
    "contrast_validation",
    "contrast_depth",
    "contrast_joint",
)


def registered_config() -> dict:
    return {
        "architecture": {"state_token": "<|fim_pad|>", "state_slots": 8},
        "substrate": {
            "node_count": 16,
            "checksum_modulus": 8,
            "num_choices": 4,
            "max_generation_attempts": 500,
            "train_families": ["phase_branch", "checksum_branch"],
            "train_templates": ["ledger", "prose"],
        },
        "training": {
            "positive_control": {
                "rows": 48,
                "seed": 73991,
                "depths": [2, 3, 4],
                "examples_per_cell": 2,
            }
        },
    }


def empty_result_manifest() -> dict:
    return {
        "files": {
            split: {"rows": 0, "structural_fingerprints": []}
            for split in SPLITS
        }
    }


def exact_rows() -> list[dict]:
    rows, _ = generate_control_rows(registered_config(), empty_result_manifest())
    return rows


def reidentify(receipt: dict) -> None:
    payload = {
        key: value
        for key, value in receipt.items()
        if key != "receipt_identity_sha256"
    }
    receipt["receipt_identity_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class ControlRowGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = registered_config()
        cls.manifest = empty_result_manifest()
        cls.rows, cls.receipt = generate_control_rows(cls.config, cls.manifest)

    def test_generation_is_exact_deterministic_and_consumer_validated(self) -> None:
        repeated_rows, repeated_receipt = generate_control_rows(
            self.config, self.manifest
        )
        self.assertEqual(repeated_rows, self.rows)
        self.assertEqual(repeated_receipt, self.receipt)
        self.assertEqual(validate_control_rows(self.rows), self.receipt)
        self.assertEqual(
            self.receipt,
            {
                "seed": 73991,
                "rows": 48,
                "grid": self.receipt["grid"],
                "canonical_rows_sha256": (
                    "581dadcb7bba053d94a849e42e0490127c7e0de199311d053e7585adcc78ef41"
                ),
                "cross_result_structural_overlap": 0,
            },
        )
        self.assertEqual(len(self.receipt["grid"]), 24)
        self.assertEqual(set(self.receipt["grid"].values()), {2})
        expected_order = [
            (depth, family, template, query_kind)
            for _repeat in range(2)
            for depth in (2, 3, 4)
            for family in ("phase_branch", "checksum_branch")
            for template in ("ledger", "prose")
            for query_kind in ("node", "checksum")
        ]
        self.assertEqual(
            [
                (row["depth"], row["family"], row["template"], row["query_kind"])
                for row in self.rows
            ],
            expected_order,
        )

    def test_registered_generation_config_tampering_is_rejected(self) -> None:
        cases = []
        wrong_seed = copy.deepcopy(self.config)
        wrong_seed["training"]["positive_control"]["seed"] = 73992
        cases.append((wrong_seed, "seed=73991"))
        wrong_depth_order = copy.deepcopy(self.config)
        wrong_depth_order["training"]["positive_control"]["depths"] = [3, 2, 4]
        cases.append((wrong_depth_order, "ordered depths"))
        wrong_family_order = copy.deepcopy(self.config)
        wrong_family_order["substrate"]["train_families"].reverse()
        cases.append((wrong_family_order, "ordered train_families"))
        wrong_slots = copy.deepcopy(self.config)
        wrong_slots["architecture"]["state_slots"] = 7
        cases.append((wrong_slots, "state_slots=8"))
        for config, error in cases:
            with self.subTest(error=error):
                with self.assertRaisesRegex(OracleControlError, error):
                    generate_control_rows(config, self.manifest)

    def test_generated_row_tamper_cannot_make_producer_and_consumer_codrift(self) -> None:
        generate_example = oracle_module.generate_example

        def tampered_generate_example(**kwargs: object) -> dict:
            row = generate_example(**kwargs)
            if kwargs["seed"] == 739910000000:
                row["unregistered_tamper"] = True
            return row

        with mock.patch.object(
            oracle_module, "generate_example", side_effect=tampered_generate_example
        ):
            with self.assertRaisesRegex(OracleControlError, "noncanonical schema"):
                generate_control_rows(self.config, self.manifest)

    def test_overlap_with_any_registered_result_split_is_rejected(self) -> None:
        for split in SPLITS:
            manifest = empty_result_manifest()
            manifest["files"][split] = {
                "rows": 1,
                "structural_fingerprints": [
                    self.rows[17]["structural_fingerprint"]
                ],
            }
            with self.subTest(split=split):
                with self.assertRaisesRegex(OracleControlError, "overlap result data"):
                    generate_control_rows(self.config, manifest)

    def test_manifest_requires_exact_seven_splits_and_validated_indices(self) -> None:
        missing = empty_result_manifest()
        del missing["files"]["contrast_joint"]
        with self.assertRaisesRegex(OracleControlError, "exact seven"):
            generate_control_rows(self.config, missing)

        extra = empty_result_manifest()
        extra["files"]["unregistered"] = {
            "rows": 0,
            "structural_fingerprints": [],
        }
        with self.assertRaisesRegex(OracleControlError, "exact seven"):
            generate_control_rows(self.config, extra)

        wrong_count = empty_result_manifest()
        wrong_count["files"]["train"]["rows"] = 1
        with self.assertRaisesRegex(OracleControlError, "malformed fingerprint index"):
            generate_control_rows(self.config, wrong_count)

        unsorted = empty_result_manifest()
        unsorted["files"]["train"] = {
            "rows": 2,
            "structural_fingerprints": ["b" * 64, "a" * 64],
        }
        with self.assertRaisesRegex(OracleControlError, "not canonical"):
            generate_control_rows(self.config, unsorted)

        duplicate = empty_result_manifest()
        duplicate["files"]["train"] = {
            "rows": 1,
            "structural_fingerprints": ["a" * 64],
        }
        duplicate["files"]["validation"] = {
            "rows": 1,
            "structural_fingerprints": ["a" * 64],
        }
        with self.assertRaisesRegex(OracleControlError, "repeats"):
            generate_control_rows(self.config, duplicate)


class SharedPositiveControlAnalyzerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = exact_rows()
        cls.records = build_oracle_positive_control_records(cls.rows)

    def setUp(self) -> None:
        self.live_rows = copy.deepcopy(self.rows)
        self.live_records = copy.deepcopy(self.records)

    def test_oracle_records_use_exact_production_analyzer_and_schema(self) -> None:
        analysis = analyze_positive_control_records(
            self.live_records, expected_rows=self.live_rows
        )
        self.assertEqual(
            summarize_positive_control_records(
                self.live_records, expected_rows=self.live_rows
            ),
            analysis["overall"],
        )
        overall = analysis["overall"]
        self.assertEqual(overall["rows"], 48)
        self.assertEqual(overall["trajectory_steps"], 144)
        self.assertEqual(
            overall["terminal_correct_counts"],
            {"node": 48, "phase": 48, "checksum": 48, "joint": 48},
        )
        self.assertEqual(
            overall["trajectory_correct_counts"],
            {"node": 144, "phase": 144, "checksum": 144, "joint": 144},
        )
        self.assertEqual(
            {depth: value["rows"] for depth, value in analysis["by_depth"].items()},
            {"2": 16, "3": 16, "4": 16},
        )
        self.assertTrue(
            all(
                overall[f"{head}_{kind}_accuracy"] == 1.0
                for head in ("node", "phase", "checksum", "joint")
                for kind in ("final", "trajectory")
            )
        )
        self.assertEqual(
            hashlib.sha256(
                json.dumps(
                    analysis, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest(),
            "493bc4cb6a592564d2f1ad8ab6187d209348011acb0c4f0d9a64b5430b658a9c",
        )

    def test_analyzer_rejects_task_identity_order_and_depth_drift(self) -> None:
        wrong_id = copy.deepcopy(self.live_records)
        wrong_id[0]["id"] = "wrong-task"
        with self.assertRaisesRegex(OracleControlError, "expected id"):
            analyze_positive_control_records(wrong_id, expected_rows=self.live_rows)

        reordered = copy.deepcopy(self.live_records)
        reordered[0], reordered[1] = reordered[1], reordered[0]
        with self.assertRaisesRegex(OracleControlError, "expected id"):
            analyze_positive_control_records(reordered, expected_rows=self.live_rows)

        wrong_depth = copy.deepcopy(self.live_records)
        wrong_depth[0]["depth"] = 3
        with self.assertRaisesRegex(OracleControlError, "expected depth"):
            analyze_positive_control_records(wrong_depth, expected_rows=self.live_rows)

    def test_analyzer_recomputes_counts_histograms_and_exact_targets(self) -> None:
        wrong_count = copy.deepcopy(self.live_records)
        wrong_count[0]["state"]["trajectory"]["joint"] -= 1
        with self.assertRaisesRegex(OracleControlError, "inconsistent trajectory"):
            analyze_positive_control_records(wrong_count, expected_rows=self.live_rows)

        wrong_histogram = copy.deepcopy(self.live_records)
        wrong_histogram[0]["state"]["histograms"]["node"]["prediction"][0] += 1
        with self.assertRaisesRegex(OracleControlError, "inconsistent histograms"):
            analyze_positive_control_records(
                wrong_histogram, expected_rows=self.live_rows
            )

        wrong_target = copy.deepcopy(self.live_records)
        target = wrong_target[0]["state"]["targets"]["node"][0]
        target[0] = (target[0] + 1) % 16
        with self.assertRaisesRegex(OracleControlError, "changed exact node targets"):
            analyze_positive_control_records(wrong_target, expected_rows=self.live_rows)

        self_attested_prediction = copy.deepcopy(self.live_records)
        predictions = self_attested_prediction[0]["state"]["predictions"]["node"][0]
        predictions[0] = (predictions[0] + 1) % 16
        with self.assertRaisesRegex(OracleControlError, "inconsistent"):
            analyze_positive_control_records(
                self_attested_prediction, expected_rows=self.live_rows
            )

    def test_analyzer_rejects_noninteger_counts_nonfinite_losses_and_schema_drift(self) -> None:
        boolean_count = copy.deepcopy(self.live_records)
        boolean_count[0]["state"]["terminal"]["joint"] = True
        with self.assertRaisesRegex(OracleControlError, "counts are invalid"):
            analyze_positive_control_records(
                boolean_count, expected_rows=self.live_rows
            )

        nonfinite = copy.deepcopy(self.live_records)
        nonfinite[0]["objective_loss"] = float("nan")
        with self.assertRaisesRegex(OracleControlError, "nonfinite"):
            analyze_positive_control_records(nonfinite, expected_rows=self.live_rows)

        extra = copy.deepcopy(self.live_records)
        extra[0]["self_attested_accuracy"] = 1.0
        with self.assertRaisesRegex(OracleControlError, "noncanonical schema"):
            analyze_positive_control_records(extra, expected_rows=self.live_rows)

    def test_analyzer_revalidates_full_expected_corpus(self) -> None:
        rows = copy.deepcopy(self.live_rows)
        rows[0]["prompt"] += " "
        with self.assertRaisesRegex(OracleControlError, "canonical corpus"):
            analyze_positive_control_records(self.live_records, expected_rows=rows)


class OracleControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.exact_rows = exact_rows()

    def setUp(self) -> None:
        self.rows = copy.deepcopy(self.exact_rows)
        self.table = build_oracle_prediction_table(self.rows)

    def test_exact_oracle_receipt_and_consumer_recomputation(self) -> None:
        receipt = analyze_oracle_control(self.rows, self.table)
        self.assertEqual(
            set(receipt),
            {
                "schema_version",
                "status",
                "rows",
                "unique_tasks",
                "depth_counts",
                "terminal_joint_correct",
                "terminal_joint_accuracy",
                "threshold",
                "control_seed",
                "canonical_control_rows_sha256",
                "canonical_table_sha256",
                "analyzer_output_sha256",
                "receipt_identity_sha256",
            },
        )
        self.assertEqual(receipt["status"], "ORACLE_ANALYSIS_PASS")
        self.assertEqual(receipt["control_seed"], 73991)
        self.assertEqual(receipt["depth_counts"], {"2": 16, "3": 16, "4": 16})
        self.assertEqual(receipt["terminal_joint_correct"], 48)
        self.assertEqual(receipt["terminal_joint_accuracy"], 1.0)
        self.assertEqual(
            receipt["canonical_control_rows_sha256"],
            "581dadcb7bba053d94a849e42e0490127c7e0de199311d053e7585adcc78ef41",
        )
        self.assertEqual(
            receipt["canonical_table_sha256"],
            "c42a546ca67992dea7e17309c29ccdd4663cd250803e5d488094c2d7ad108f92",
        )
        self.assertEqual(
            receipt["analyzer_output_sha256"],
            "493bc4cb6a592564d2f1ad8ab6187d209348011acb0c4f0d9a64b5430b658a9c",
        )
        self.assertEqual(
            receipt["receipt_identity_sha256"],
            "16f6b7f04e2e39cc04d7537456774bc214186e6b9bc11049408aac3115b8c63d",
        )
        self.assertEqual(
            validate_oracle_analysis_receipt(self.rows, receipt), receipt
        )
        self.assertEqual(produce_oracle_analysis_receipt(self.rows), receipt)

    def test_consumer_rejects_mutated_prompt_row_order_and_schema(self) -> None:
        prompt = copy.deepcopy(self.rows)
        prompt[0]["prompt"] += " "
        with self.assertRaisesRegex(OracleControlError, "canonical corpus"):
            build_oracle_prediction_table(prompt)

        trajectory = copy.deepcopy(self.rows)
        trajectory[0]["trajectory"][-1]["phase"] ^= 1
        with self.assertRaisesRegex(OracleControlError, "full verify_example"):
            build_oracle_prediction_table(trajectory)

        reordered = copy.deepcopy(self.rows)
        reordered[0], reordered[1] = reordered[1], reordered[0]
        with self.assertRaisesRegex(OracleControlError, "seed/order grid"):
            build_oracle_prediction_table(reordered)

        schema = copy.deepcopy(self.rows)
        schema[0]["unregistered"] = True
        with self.assertRaisesRegex(OracleControlError, "noncanonical schema"):
            build_oracle_prediction_table(schema)

    def test_table_digest_schema_ids_and_predictions_are_fail_closed(self) -> None:
        self.assertEqual(
            hashlib.sha256(
                json.dumps(
                    {"rows": self.table}, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest(),
            "c42a546ca67992dea7e17309c29ccdd4663cd250803e5d488094c2d7ad108f92",
        )
        missing = copy.deepcopy(self.table)
        missing.pop()
        with self.assertRaisesRegex(OracleControlError, "exactly 48"):
            analyze_oracle_control(self.rows, missing)

        wrong_id = copy.deepcopy(self.table)
        wrong_id[3]["id"] = "not-a-control-task"
        with self.assertRaisesRegex(OracleControlError, "IDs/order"):
            analyze_oracle_control(self.rows, wrong_id)

        wrong_prediction = copy.deepcopy(self.table)
        wrong_prediction[0]["node_prediction"] = (
            wrong_prediction[0]["node_target"] + 1
        ) % 16
        wrong_prediction[0]["node_final_correct"] = False
        wrong_prediction[0]["joint_final_correct"] = False
        with self.assertRaisesRegex(OracleControlError, "not oracle-perfect"):
            analyze_oracle_control(self.rows, wrong_prediction)

        nonboolean = copy.deepcopy(self.table)
        nonboolean[0]["node_final_correct"] = 1
        with self.assertRaisesRegex(OracleControlError, "non-boolean"):
            analyze_oracle_control(self.rows, nonboolean)

        schema = copy.deepcopy(self.table)
        schema[0]["extra"] = True
        with self.assertRaisesRegex(OracleControlError, "noncanonical fields"):
            analyze_oracle_control(self.rows, schema)

    def test_receipt_digest_control_binding_and_schema_are_recomputed(self) -> None:
        receipt = produce_oracle_analysis_receipt(self.rows)
        internally_valid_but_nonfrozen = copy.deepcopy(self.rows)
        internally_valid_but_nonfrozen[0]["prompt"] += " "
        with self.assertRaisesRegex(OracleControlError, "canonical corpus"):
            validate_oracle_analysis_receipt(
                internally_valid_but_nonfrozen, receipt
            )

        receipt["canonical_control_rows_sha256"] = "0" * 64
        reidentify(receipt)
        with self.assertRaisesRegex(OracleControlError, "exact recomputation"):
            validate_oracle_analysis_receipt(self.rows, receipt)

        receipt = produce_oracle_analysis_receipt(self.rows)
        receipt["canonical_table_sha256"] = "0" * 64
        reidentify(receipt)
        with self.assertRaisesRegex(OracleControlError, "exact recomputation"):
            validate_oracle_analysis_receipt(self.rows, receipt)

        receipt = produce_oracle_analysis_receipt(self.rows)
        receipt["extra"] = "not registered"
        reidentify(receipt)
        with self.assertRaisesRegex(OracleControlError, "noncanonical fields"):
            validate_oracle_analysis_receipt(self.rows, receipt)

        receipt = produce_oracle_analysis_receipt(self.rows)
        receipt["receipt_identity_sha256"] = "0" * 64
        with self.assertRaisesRegex(OracleControlError, "identity mismatch"):
            validate_oracle_analysis_receipt(self.rows, receipt)


if __name__ == "__main__":
    unittest.main()
