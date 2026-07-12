from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = EXP / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"repo_scb_{name}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


train = load_script("train")
analyze_repo = load_script("analyze_repo")


class TrainingAndAnalysisTests(unittest.TestCase):
    def test_batches_keep_apex_and_four_operator_repository_rows_separate(self):
        apex = [
            {"source": "apex", "input_ids": list(range(10 + index)), "source_code": 0}
            for index in range(5)
        ]
        repo = [
            {"source": "repo", "input_ids": list(range(20)), "source_code": 1,
             "task_id": "task-1", "operator": operator}
            for operator in train.OPERATORS
        ]
        ordered, receipt = train.make_batches(apex + repo, batch_size=4, seed=42)
        chunks = [ordered[index:index + 4] for index in range(0, len(ordered), 4)]
        self.assertTrue(all(len({row["source"] for row in chunk}) == 1 for chunk in chunks))
        repo_chunk = next(chunk for chunk in chunks if chunk[0]["source"] == "repo")
        self.assertEqual({row["operator"] for row in repo_chunk}, set(train.OPERATORS))
        self.assertEqual(receipt["apex_padding_duplicates"], 3)

    def test_paired_bootstrap_uses_taskwise_differences(self):
        left = {f"t{i}": {"success": value} for i, value in enumerate([1, 1, 0, 1])}
        right = {f"t{i}": {"success": value} for i, value in enumerate([0, 1, 0, 0])}
        result = analyze_repo.paired_delta(left, right, seed=7)
        self.assertEqual(result["delta"], 0.5)
        self.assertEqual(result["left_only"], 2)
        self.assertEqual(result["right_only"], 0)

    def test_locality_contexts_are_frozen_and_unique(self):
        payload = json.loads((EXP / "data" / "locality_contexts.json").read_text())
        self.assertEqual(payload["count"], 48)
        self.assertEqual(len({row["id"] for row in payload["contexts"]}), 48)
        self.assertEqual(len({row["content_sha256"] for row in payload["contexts"]}), 48)


if __name__ == "__main__":
    unittest.main()
