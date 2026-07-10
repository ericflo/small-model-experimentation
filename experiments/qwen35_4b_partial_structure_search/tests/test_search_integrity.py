from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(EXP / "src"))
sys.path.insert(0, str(EXP / "scripts"))
import analyze_search as A  # noqa: E402
import run_search as R  # noqa: E402


class SearchCacheIntegrityTests(unittest.TestCase):
    def _complete_cache(self, root: Path, inputs: dict[str, str]) -> dict[str, Path]:
        (root / "runs").mkdir(parents=True)
        result_paths = R._result_paths("")
        auxiliary_paths = R._auxiliary_paths("")
        for method, path in result_paths.items():
            path.write_text(f'{{"method":"{method}"}}\n', encoding="utf-8")
        for name, path in auxiliary_paths.items():
            path.write_text(f"{name}\n", encoding="utf-8")
        receipt = {
            "schema_version": 2,
            "input_sha256": inputs,
            "result_sha256": {
                method: R._sha256(path) for method, path in result_paths.items()
            },
            "auxiliary_sha256": {
                name: R._sha256(path) for name, path in auxiliary_paths.items()
            },
        }
        R._atomic_json(R._receipt_path(""), receipt)
        return auxiliary_paths

    def test_current_input_drift_invalidates_otherwise_complete_cache(self) -> None:
        original_exp = R.EXP
        with tempfile.TemporaryDirectory() as directory:
            R.EXP = Path(directory)
            try:
                frozen = {"config": "a" * 64, "tasks": "b" * 64}
                self._complete_cache(R.EXP, frozen)
                self.assertTrue(R._valid_complete_cache("", frozen))
                changed = {**frozen, "tasks": "c" * 64}
                with self.assertRaisesRegex(RuntimeError, "stale or incompatible"):
                    R._valid_complete_cache("", changed)
            finally:
                R.EXP = original_exp

    def test_auxiliary_artifact_drift_invalidates_complete_cache(self) -> None:
        original_exp = R.EXP
        with tempfile.TemporaryDirectory() as directory:
            R.EXP = Path(directory)
            try:
                frozen = {"config": "a" * 64}
                auxiliary = self._complete_cache(R.EXP, frozen)
                self.assertTrue(R._valid_complete_cache("", frozen))
                auxiliary["scores_thinking"].write_text(
                    "corrupted\n", encoding="utf-8"
                )
                with self.assertRaisesRegex(RuntimeError, "auxiliary cache"):
                    R._valid_complete_cache("", frozen)
            finally:
                R.EXP = original_exp


class DirectPoolInvariantTests(unittest.TestCase):
    @staticmethod
    def _tasks() -> list[dict[str, object]]:
        return [
            {"task_id": f"task-{index}", "depth": 5, "visible": []}
            for index in range(4)
        ]

    def test_direct_pool_rejects_changed_shard_row_order(self) -> None:
        class ReversedRunner:
            def generate(self, records, _sampling):
                return (
                    [
                        {"id": record["id"], "outputs": [{}, {}]}
                        for record in reversed(records)
                    ],
                    {},
                )

        original_exp = R.EXP
        with tempfile.TemporaryDirectory() as directory:
            R.EXP = Path(directory)
            try:
                with self.assertRaisesRegex(RuntimeError, "changed frozen task order"):
                    R._generate_direct_pool(
                        ReversedRunner(),
                        self._tasks(),
                        pool_k=2,
                        thinking_budget=1,
                        answer_max=1,
                        run_seed=1,
                        suffix="",
                    )
            finally:
                R.EXP = original_exp

    def test_direct_pool_rejects_incomplete_per_task_pool(self) -> None:
        class ShortRunner:
            def generate(self, records, _sampling):
                return (
                    [{"id": record["id"], "outputs": [{}]} for record in records],
                    {},
                )

        original_exp = R.EXP
        with tempfile.TemporaryDirectory() as directory:
            R.EXP = Path(directory)
            try:
                with self.assertRaisesRegex(RuntimeError, "exactly pool_k outputs"):
                    R._generate_direct_pool(
                        ShortRunner(),
                        self._tasks(),
                        pool_k=2,
                        thinking_budget=1,
                        answer_max=1,
                        run_seed=1,
                        suffix="",
                    )
            finally:
                R.EXP = original_exp


class SearchAnalysisIntegrityTests(unittest.TestCase):
    @staticmethod
    def _direct_result(method: str, basis: str) -> tuple[dict, dict]:
        canonical = ["task-0", "task-1"]
        settings = {
            "beam_width": 4,
            "thinking_budget": 256,
            "parameter_fill_cap_per_task": 4096,
            "run_seed": 1801,
            "model": "Qwen/Qwen3.5-4B",
            "model_revision": "revision",
            "backend": "vllm",
        }
        inputs = {"config": "a" * 64}
        result = {
            "method": method,
            "match_basis": basis,
            "input_sha256": inputs,
            "task_ids": canonical,
            "settings": settings,
            "rows": [
                {
                    "task_id": task_id,
                    "task_index": index,
                    "shard": index % 2,
                    "method": method,
                    "match_basis": basis,
                    "fill_cap": 4096,
                }
                for index, task_id in enumerate(canonical)
            ],
        }
        receipt = {"input_sha256": inputs, "settings": settings}
        return result, receipt

    def test_direct_sampled_arm_rejects_total_token_basis_swap(self) -> None:
        result, receipt = self._direct_result(
            "direct_sample_more", "total_model_tokens"
        )
        with self.assertRaisesRegex(RuntimeError, "match basis must be sampled_tokens"):
            A._validate_arm_integrity(
                "direct_sample_more",
                result,
                canonical_ids=["task-0", "task-1"],
                receipt=receipt,
                expected_beam=4,
                expected_fill_cap=4096,
            )

    def test_budget_summary_prefers_explicit_expanded_leaf_nodes(self) -> None:
        result = {
            "method": "budget_truncated_brute",
            "rows": [
                {
                    "task_id": "task-0",
                    "shard": 0,
                    "layers": [],
                    "expanded_prefix_nodes": 3,
                    "selected_hidden_success": True,
                    "pool_hidden_coverage": True,
                    "completed_skeletons": 3,
                    "fill_cap_used": 7,
                    "visible_passing_concrete_candidates": 1,
                },
                {
                    "task_id": "task-1",
                    "shard": 1,
                    "layers": [],
                    "expanded_prefix_nodes": 5,
                    "selected_hidden_success": False,
                    "pool_hidden_coverage": False,
                    "completed_skeletons": 5,
                    "fill_cap_used": 7,
                    "visible_passing_concrete_candidates": 0,
                },
            ],
        }
        summary = A._arm_summary(result)
        self.assertEqual(summary["mean_expanded_prefix_nodes"], 4.0)


if __name__ == "__main__":
    unittest.main()
