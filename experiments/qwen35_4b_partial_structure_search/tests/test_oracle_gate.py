from __future__ import annotations

import copy
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock


EXP = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


F = _load("oracle_gate_test_families", EXP / "src" / "families.py")
_prior = {name: sys.modules.get(name) for name in ("families", "experiment_common")}
sys.modules["families"] = F
try:
    C = _load("oracle_gate_test_common", EXP / "scripts" / "experiment_common.py")
    sys.modules["experiment_common"] = C
    G = _load("oracle_gate_under_test", EXP / "scripts" / "oracle_gate.py")
finally:
    for _name, _module in _prior.items():
        if _module is None:
            sys.modules.pop(_name, None)
        else:
            sys.modules[_name] = _module


def _task(split: str, task_id: str, operation, offset: int):
    task = F.build_task_from_pipeline(
        task_id=task_id,
        seed=offset,
        pipeline=(operation,),
        visible_inputs=[[1 + offset, -2, 3]],
        label_probe_inputs=[[4, -1, 2 + offset]],
        hidden_inputs=[[-5, 2, 7 + offset]],
    )
    task["split"] = split
    return task


def _oracle(task):
    pipeline = F.normalize_pipeline(task["target_pipeline"])
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
                "skeleton": [name for name, _parameter in pipeline],
                "parameter_fill_count": 1,
                "representative_pipeline": task["target_pipeline"],
            }
        ],
        "wall_seconds": 0.0,
    }


def _config():
    return {
        "judge": {"beam_width": 4},
        "oracle_gate": {
            "min_hidden_path_rate": 1.0,
            "min_completed_skeleton_compression": 4.0,
        },
    }


class OracleDevelopmentGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.calibration = _task(
            "calibration", "calibration-d1-0000", ("reverse", None), 0
        )
        self.development = _task(
            "development", "development-d1-0000", ("negate", None), 10
        )

    def _result(self, calibration=None, development=None):
        calibration = calibration or self.calibration
        development = development or self.development
        return G.build_result(
            [calibration],
            [_oracle(calibration)],
            [development],
            [_oracle(development)],
            _config(),
        )

    def test_gate_schema_is_development_based_and_primary_blind(self) -> None:
        result = self._result()

        self.assertTrue(result["passed"])
        self.assertEqual(result["gate_basis_split"], "development")
        self.assertEqual(result["evaluated_splits"], ["calibration", "development"])
        self.assertEqual(set(result["splits"]), {"calibration", "development"})
        self.assertTrue(result["development_hidden_used_for_gate"])
        self.assertFalse(result["primary_artifacts_loaded"])
        self.assertFalse(result["primary_hidden_used_for_gate"])
        self.assertFalse(result["hidden_used_for_labels"])
        self.assertEqual({row["split"] for row in result["rows"]}, {"calibration", "development"})

    def test_calibration_hidden_diagnostic_cannot_change_gate_verdict(self) -> None:
        corrupted = copy.deepcopy(self.calibration)
        corrupted["hidden"][0]["output"] = [999_999]

        result = self._result(calibration=corrupted)

        self.assertTrue(result["passed"])
        self.assertEqual(result["splits"]["calibration"]["path_rate"], 0.0)
        self.assertEqual(result["splits"]["development"]["path_rate"], 1.0)

    def test_development_hidden_outcome_controls_gate(self) -> None:
        corrupted = copy.deepcopy(self.development)
        corrupted["hidden"][0]["output"] = [999_999]

        result = self._result(development=corrupted)

        self.assertFalse(result["passed"])
        self.assertEqual(result["splits"]["development"]["path_rate"], 0.0)

    def test_loader_never_requests_primary_paths(self) -> None:
        requested: list[str] = []

        def fake_load(path):
            requested.append(str(path))
            return []

        with mock.patch.object(G.C, "load_jsonl", side_effect=fake_load):
            loaded = G.load_gate_data("_smoke")

        self.assertEqual(set(loaded), {"calibration", "development"})
        self.assertEqual(len(requested), 4)
        self.assertTrue(all("primary" not in path for path in requested))

    def test_source_fingerprint_plan_covers_exactly_consumed_files(self) -> None:
        paths = G.gate_source_paths("_smoke")

        self.assertEqual(
            set(paths),
            {
                "calibration_tasks",
                "calibration_oracle",
                "development_tasks",
                "development_oracle",
            },
        )
        self.assertTrue(all(path.exists() for path in paths.values()))
        self.assertTrue(all(len(G._sha256_file(path)) == 64 for path in paths.values()))
        self.assertTrue(all("primary" not in str(path) for path in paths.values()))


if __name__ == "__main__":
    unittest.main()
