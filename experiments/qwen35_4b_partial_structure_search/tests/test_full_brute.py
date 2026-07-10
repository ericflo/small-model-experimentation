from __future__ import annotations

import copy
import importlib.util
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


# Load the executable scripts against this experiment's local modules without
# leaking their generic import names into other test files.
F = _load("full_brute_test_families", EXP / "src" / "families.py")
_prior = {name: sys.modules.get(name) for name in ("families", "experiment_common", "oracle_data")}
sys.modules["families"] = F
try:
    C = _load("full_brute_test_common", EXP / "scripts" / "experiment_common.py")
    sys.modules["experiment_common"] = C
    O = _load("full_brute_test_oracle", EXP / "scripts" / "oracle_data.py")
    sys.modules["oracle_data"] = O
    B = _load("full_brute_under_test", EXP / "scripts" / "full_brute.py")
finally:
    for _name, _module in _prior.items():
        if _module is None:
            sys.modules.pop(_name, None)
        else:
            sys.modules[_name] = _module


def _tiny_task():
    return F.build_task_from_pipeline(
        task_id="tiny-full-brute",
        seed=17,
        pipeline=(("reverse", None), ("sort_asc", None)),
        visible_inputs=[[3, 1, 2], [7, -1, 4, 2]],
        label_probe_inputs=[[5, 3, 4, 1, 2]],
        hidden_inputs=[[9, 6, 8, 7], [2, -5, 0, 3]],
        require_exact_depth=False,
    )


def _naive_visible_passers(task):
    inputs, outputs = F.task_cases(task, ("visible",))
    passers = []
    for skeleton in F.enumerate_skeletons(int(task["depth"])):
        for pipeline in F.enumerate_parameter_fills(skeleton):
            if F.pipeline_solves(pipeline, inputs, outputs):
                passers.append(pipeline)
    return passers


class FullBruteTaskTests(unittest.TestCase):
    def test_exact_quotient_matches_tiny_naive_exhaustion_and_shared_selector(self) -> None:
        task = _tiny_task()
        expected = _naive_visible_passers(task)
        expected_selected, expected_selector = C.consensus_select(expected, task)

        row = B.run_full_brute_task(task)

        self.assertTrue(row["exact"])
        self.assertEqual(row["label_source_splits"], ["visible"])
        self.assertFalse(row["label_probe_used_for_search_or_selection"])
        self.assertFalse(row["hidden_used_for_search_or_selection"])
        self.assertEqual(row["logical_search_space"]["type_skeleton_leaves"], 16**2)
        self.assertEqual(
            row["logical_search_space"]["concrete_parameterized_leaves"], 32**2
        )
        self.assertEqual(row["visible_successful_pipeline_count"], len(expected))
        self.assertEqual(
            row["visible_successful_type_skeleton_count"],
            len({tuple(name for name, _ in pipeline) for pipeline in expected}),
        )
        serialized_expected = [
            [[name, parameter] for name, parameter in pipeline] for pipeline in expected
        ]
        self.assertCountEqual(row["visible_successful_pipelines"], serialized_expected)
        self.assertEqual(row["selector"], expected_selector)
        self.assertEqual(
            row["selected_pipeline"],
            [[name, parameter] for name, parameter in expected_selected],
        )
        self.assertEqual(
            row["path_coverage_hidden"],
            any(C.hidden_grade(pipeline, task) for pipeline in expected),
        )
        self.assertEqual(
            row["selected_hidden_success"], C.hidden_grade(expected_selected, task)
        )
        self.assertEqual(
            row["accounting"]["successful_concrete_pipelines"], len(expected)
        )
        self.assertGreater(row["accounting"]["transition_requests"], 0)
        self.assertGreater(row["accounting"]["case_operation_applications"], 0)
        self.assertGreaterEqual(row["wall_seconds"], 0.0)

    def test_label_probe_changes_cannot_change_deployment_candidates_or_selection(self) -> None:
        task = _tiny_task()
        corrupted = copy.deepcopy(task)
        for case in corrupted["label_probe"]:
            case["output"] = [999_999]

        clean = B.run_full_brute_task(task)
        changed = B.run_full_brute_task(corrupted)

        stable_fields = (
            "visible_successful_pipeline_count",
            "visible_successful_type_skeleton_count",
            "visible_successful_pipelines",
            "selected_pipeline",
            "selector",
            "path_coverage_hidden",
            "selected_hidden_success",
            "accounting",
        )
        for field in stable_fields:
            self.assertEqual(clean[field], changed[field], field)


class FullBruteAggregateTests(unittest.TestCase):
    def test_aggregate_receipt_sums_logical_and_physical_work(self) -> None:
        task = _tiny_task()
        second = copy.deepcopy(task)
        second["task_id"] = "tiny-full-brute-2"

        result = B.build_result([task, second], workers=1, smoke=True)

        self.assertEqual(result["task_count"], 2)
        self.assertEqual(result["depths"], [2])
        self.assertEqual(result["logical_accounting"]["type_skeleton_leaves"], 2 * 16**2)
        self.assertEqual(
            result["aggregate_accounting"]["successful_concrete_pipelines"],
            sum(row["visible_successful_pipeline_count"] for row in result["rows"]),
        )
        self.assertFalse(result["hidden_used_for_search_or_selection"])
        self.assertGreaterEqual(result["parallel_wall_seconds"], 0.0)


if __name__ == "__main__":
    unittest.main()
