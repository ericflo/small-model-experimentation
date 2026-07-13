from __future__ import annotations

import ast
import inspect
import math
import sys
import textwrap
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.state_loop_model import (  # noqa: E402
    StateLoopModel,
    _aggregate_recurrent_states,
)


class RecurrentStateAggregationTests(unittest.TestCase):
    @staticmethod
    def aggregate_logit() -> torch.Tensor:
        return torch.tensor(math.log(9.0), dtype=torch.float32, requires_grad=True)

    def test_bf16_mean_is_formed_before_the_fp32_scalar_mix(self) -> None:
        # These values distinguish the registered BF16 recurrence-state mean
        # from a broader change that silently moves the mean itself to FP32.
        states = (
            torch.tensor([[[-9.75]]], dtype=torch.bfloat16),
            torch.tensor([[[-7.5625]]], dtype=torch.bfloat16),
        )
        aggregate_logit = self.aggregate_logit()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = _aggregate_recurrent_states(states, aggregate_logit)

        stacked = torch.stack(states, dim=1)
        mean_bf16 = stacked.mean(dim=1)
        weight_fp32 = torch.sigmoid(aggregate_logit)
        registered_reference = (
            weight_fp32 * stacked[:, -1].float()
            + (1.0 - weight_fp32) * mean_bf16.float()
        ).to(dtype=states[0].dtype)
        broadened_fp32_mean = (
            weight_fp32 * stacked[:, -1].float()
            + (1.0 - weight_fp32) * stacked.float().mean(dim=1)
        ).to(dtype=states[0].dtype)

        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(actual, registered_reference))
        self.assertFalse(torch.equal(actual, broadened_fp32_mean))
        self.assertEqual(float(actual.item()), -7.65625)

    def test_fp32_mix_recovers_the_analytic_gradient_lost_by_legacy_bf16_mix(self) -> None:
        # At width 256 the legacy path independently reduces two nearly equal
        # BF16 terms to the same value, so their scalar-gradient difference is
        # exactly zero.  The registered FP32 mix retains the 0.5 difference.
        last = torch.ones((1, 1, 256), dtype=torch.bfloat16)
        first = last.clone()
        first[..., 0] = 0.0
        states = (first, last)

        legacy_logit = self.aggregate_logit()
        stacked = torch.stack(states, dim=1)
        legacy_weight = torch.sigmoid(legacy_logit).to(dtype=torch.bfloat16)
        legacy_aggregate = legacy_weight * stacked[:, -1] + (
            1.0 - legacy_weight
        ) * stacked.mean(dim=1)
        legacy_aggregate.float().sum().backward()
        self.assertEqual(float(legacy_logit.grad), 0.0)

        repaired_logit = self.aggregate_logit()
        repaired_aggregate = _aggregate_recurrent_states(states, repaired_logit)
        repaired_aggregate.float().sum().backward()

        weight = torch.sigmoid(repaired_logit.detach())
        mean_bf16 = stacked.mean(dim=1)
        projection = (stacked[:, -1].float() - mean_bf16.float()).sum()
        analytic = weight * (1.0 - weight) * projection
        self.assertIsNotNone(repaired_logit.grad)
        self.assertTrue(bool(torch.isfinite(repaired_logit.grad)))
        self.assertNotEqual(float(repaired_logit.grad), 0.0)
        torch.testing.assert_close(repaired_logit.grad, analytic, rtol=0.0, atol=1e-8)
        self.assertAlmostEqual(float(analytic), 0.045, places=7)

    def test_helper_has_one_explicit_cast_back_and_k1_bypasses_it(self) -> None:
        helper_source = textwrap.dedent(inspect.getsource(_aggregate_recurrent_states))
        self.assertEqual(helper_source.count(".to("), 1)
        self.assertIn("enabled=False", helper_source)

        forward_source = textwrap.dedent(inspect.getsource(StateLoopModel.forward))
        function = ast.parse(forward_source).body[0]
        self.assertIsInstance(function, ast.FunctionDef)
        k1_branch = next(
            node
            for node in ast.walk(function)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "k"
            and any(
                isinstance(comparator, ast.Constant) and comparator.value == 1
                for comparator in node.test.comparators
            )
        )
        k1_source = ast.get_source_segment(forward_source, k1_branch) or ""
        k1_body = "\n".join(
            ast.get_source_segment(forward_source, statement) or ""
            for statement in k1_branch.body
        )
        recurrent_body = "\n".join(
            ast.get_source_segment(forward_source, statement) or ""
            for statement in k1_branch.orelse
        )
        self.assertIn("aggregated = raw_first_state", k1_body)
        self.assertNotIn("_aggregate_recurrent_states", k1_body)
        self.assertIn("_aggregate_recurrent_states", recurrent_body)

        # Aggregation feeds only the optional answer branch.  Therefore a
        # state-only objective (compute_answer=False) cannot create a gradient
        # or optimizer moment for aggregate_logit.
        parent: dict[ast.AST, ast.AST] = {
            child: node for node in ast.walk(function) for child in ast.iter_child_nodes(node)
        }
        aggregate_loads = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Name)
            and node.id == "aggregated"
            and isinstance(node.ctx, ast.Load)
        ]
        self.assertEqual(len(aggregate_loads), 1, k1_source)
        for load in aggregate_loads:
            ancestors = []
            current: ast.AST | None = load
            while current in parent:
                current = parent[current]
                ancestors.append(current)
            self.assertTrue(
                any(
                    isinstance(node, ast.If)
                    and isinstance(node.test, ast.Name)
                    and node.test.id == "compute_answer"
                    for node in ancestors
                )
            )


if __name__ == "__main__":
    unittest.main()
