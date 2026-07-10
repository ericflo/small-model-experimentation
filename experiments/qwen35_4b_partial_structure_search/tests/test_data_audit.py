from __future__ import annotations

import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


F = _load("data_audit_test_families", EXP / "src" / "families.py")
_prior = {name: sys.modules.get(name) for name in ("families", "experiment_common")}
sys.modules["families"] = F
try:
    C = _load("data_audit_test_common", EXP / "scripts" / "experiment_common.py")
    sys.modules["experiment_common"] = C
    A = _load("data_audit_under_test", EXP / "scripts" / "data_audit.py")
finally:
    for _name, _module in _prior.items():
        if _module is None:
            sys.modules.pop(_name, None)
        else:
            sys.modules[_name] = _module


def _task(split: str, task_id: str, pipeline, offset: int):
    task = F.build_task_from_pipeline(
        task_id=task_id,
        seed=offset,
        pipeline=pipeline,
        visible_inputs=[[1 + offset, -2, 3]],
        label_probe_inputs=[[4, 1 + offset, -3, 2]],
        hidden_inputs=[[-5, 2, 7 + offset]],
    )
    task["split"] = split
    task["behavior_signature_sha256"] = "legacy-bank-signature"
    return task


def _oracle(task):
    pipeline = F.normalize_pipeline(task["target_pipeline"])
    skeleton = [name for name, _parameter in pipeline]
    return {
        "schema_version": 1,
        "task_id": task["task_id"],
        "depth": task["depth"],
        "label_source_splits": ["visible", "label_probe"],
        "hidden_cases_used_for_labels": False,
        "successful_skeleton_count": 1,
        "successful_parameter_fill_count": 1,
        "successful_skeletons": [
            {
                "skeleton": skeleton,
                "parameter_fill_count": 1,
                "representative_pipeline": task["target_pipeline"],
            }
        ],
        "accounting": {
            "successful_type_skeletons": 1,
            "successful_concrete_pipelines": 1,
        },
    }


def _valid_fixture():
    calibration = _task(
        "calibration", "calibration-d1-0000", (("reverse", None),), 0
    )
    development = _task(
        "development", "development-d1-0000", (("abs_all", None),), 5
    )
    primary = _task("primary", "primary-d1-0000", (("negate", None),), 10)
    return (
        [calibration],
        [_oracle(calibration)],
        [development],
        [_oracle(development)],
        [primary],
        [_oracle(primary)],
    )


def _codes(result):
    return {row["code"] for row in result["errors"]}


class FrozenProbeBankTests(unittest.TestCase):
    def test_common_bank_is_frozen_and_json_stable(self) -> None:
        self.assertEqual(len(A.COMMON_PROBE_BANK), 64)
        self.assertEqual(len(set(A.COMMON_PROBE_BANK)), 64)
        self.assertEqual(
            A.COMMON_PROBE_BANK_SHA256,
            "71df5a618e5237c4645ec6749a74c3dccf43985368bb65c6dc5eaa56537b0412",
        )
        json.dumps(A.COMMON_PROBE_BANK)


class DataAuditHappyPathTests(unittest.TestCase):
    def test_valid_task_oracle_pairs_pass_every_gate(self) -> None:
        result = A.audit_dataset(*_valid_fixture())

        self.assertTrue(result["passed"])
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["exact_depth_receipts"]["recomputed"], 3)
        self.assertIsNone(result["exact_depth_receipts"]["seen_cap"])
        self.assertEqual(result["task_oracle_pairs_checked"], 3)
        self.assertEqual(set(result["splits"]), set(A.SPLIT_NAMES))
        self.assertEqual(
            result["behavioral_disjointness"]["cross_split_collision_groups"], []
        )
        self.assertFalse(
            result["hidden_label_boundary"]["hidden_cases_used_for_oracle_labels"]
        )
        self.assertNotIn("behavior_vector", result["task_rows"][0])
        json.dumps(result, sort_keys=True)


class DataAuditFailureTests(unittest.TestCase):
    def test_tampered_or_capped_depth_receipt_fails_exact_recomputation(self) -> None:
        (
            calibration,
            calibration_oracle,
            development,
            development_oracle,
            primary,
            primary_oracle,
        ) = _valid_fixture()
        calibration[0]["min_depth_audit"]["seen_cap"] = 60_000

        result = A.audit_dataset(
            calibration,
            calibration_oracle,
            development,
            development_oracle,
            primary,
            primary_oracle,
        )

        self.assertFalse(result["passed"])
        self.assertIn("depth_receipt_field", _codes(result))
        self.assertIn("depth_receipt_recompute_mismatch", _codes(result))

    def test_task_oracle_id_membership_and_order_are_enforced(self) -> None:
        (
            calibration,
            calibration_oracle,
            development,
            development_oracle,
            primary,
            primary_oracle,
        ) = _valid_fixture()
        primary_oracle[0]["task_id"] = "wrong-primary-id"

        result = A.audit_dataset(
            calibration,
            calibration_oracle,
            development,
            development_oracle,
            primary,
            primary_oracle,
        )

        self.assertFalse(result["passed"])
        self.assertIn("task_oracle_id_order", _codes(result))
        self.assertFalse(result["splits"]["primary"]["task_oracle_ids_equal_and_ordered"])

    def test_hidden_label_flags_and_hidden_named_payloads_are_rejected(self) -> None:
        (
            calibration,
            calibration_oracle,
            development,
            development_oracle,
            primary,
            primary_oracle,
        ) = _valid_fixture()
        primary_oracle[0]["label_source_splits"] = ["visible", "label_probe", "hidden"]
        primary_oracle[0]["hidden_cases_used_for_labels"] = True
        primary_oracle[0]["hidden_scores"] = [1.0]

        result = A.audit_dataset(
            calibration,
            calibration_oracle,
            development,
            development_oracle,
            primary,
            primary_oracle,
        )

        self.assertFalse(result["passed"])
        self.assertTrue(
            {"oracle_label_splits", "oracle_hidden_boundary", "oracle_hidden_fields"}
            <= _codes(result)
        )

    def test_common_probe_behavior_collision_across_splits_is_rejected(self) -> None:
        (
            calibration,
            calibration_oracle,
            development,
            development_oracle,
            _primary,
            _primary_oracle,
        ) = _valid_fixture()
        primary = _task(
            "primary", "primary-d1-0000", (("reverse", None),), 20
        )

        result = A.audit_dataset(
            calibration,
            calibration_oracle,
            development,
            development_oracle,
            [primary],
            [_oracle(primary)],
        )

        self.assertFalse(result["passed"])
        self.assertIn("cross_split_behavior_collision", _codes(result))
        collisions = result["behavioral_disjointness"]["cross_split_collision_groups"]
        self.assertEqual(len(collisions), 1)
        self.assertEqual(
            collisions[0]["task_ids_by_split"],
            {
                "calibration": ["calibration-d1-0000"],
                "primary": ["primary-d1-0000"],
            },
        )

    def test_development_primary_behavior_collision_is_also_rejected(self) -> None:
        (
            calibration,
            calibration_oracle,
            development,
            development_oracle,
            _primary,
            _primary_oracle,
        ) = _valid_fixture()
        primary = _task(
            "primary", "primary-d1-0000", (("abs_all", None),), 20
        )

        result = A.audit_dataset(
            calibration,
            calibration_oracle,
            development,
            development_oracle,
            [primary],
            [_oracle(primary)],
        )

        self.assertFalse(result["passed"])
        collisions = result["behavioral_disjointness"]["cross_split_collision_groups"]
        self.assertEqual(
            collisions[0]["task_ids_by_split"],
            {
                "development": ["development-d1-0000"],
                "primary": ["primary-d1-0000"],
            },
        )

    def test_visible_probe_hidden_input_overlap_is_rejected(self) -> None:
        (
            calibration,
            calibration_oracle,
            development,
            development_oracle,
            primary,
            primary_oracle,
        ) = _valid_fixture()
        duplicated = copy.deepcopy(calibration[0]["visible"][0])
        calibration[0]["hidden"][0] = duplicated

        result = A.audit_dataset(
            calibration,
            calibration_oracle,
            development,
            development_oracle,
            primary,
            primary_oracle,
        )

        self.assertFalse(result["passed"])
        self.assertIn("case_split_overlap", _codes(result))

    def test_oracle_representative_must_solve_nonhidden_label_cases(self) -> None:
        (
            calibration,
            calibration_oracle,
            development,
            development_oracle,
            primary,
            primary_oracle,
        ) = _valid_fixture()
        primary_oracle[0]["successful_skeletons"][0]["representative_pipeline"] = [
            ["reverse", None]
        ]
        primary_oracle[0]["successful_skeletons"][0]["skeleton"] = ["reverse"]

        result = A.audit_dataset(
            calibration,
            calibration_oracle,
            development,
            development_oracle,
            primary,
            primary_oracle,
        )

        self.assertFalse(result["passed"])
        self.assertIn("oracle_semantics", _codes(result))

    def test_malformed_case_bank_and_oracle_depth_fail_closed(self) -> None:
        (
            calibration,
            calibration_oracle,
            development,
            development_oracle,
            primary,
            primary_oracle,
        ) = _valid_fixture()
        primary[0].pop("label_probe")
        primary_oracle[0]["depth"] = "not-an-integer"

        result = A.audit_dataset(
            calibration,
            calibration_oracle,
            development,
            development_oracle,
            primary,
            primary_oracle,
        )

        self.assertFalse(result["passed"])
        self.assertTrue(
            {"missing_case_split", "oracle_depth", "oracle_case_bank"}
            <= _codes(result)
        )


if __name__ == "__main__":
    unittest.main()
