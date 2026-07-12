from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from mopd_loss import (  # noqa: E402
    bias_corrected_topk_reverse_kl,
    full_reverse_kl,
    sparse_teacher_topk_reverse_kl,
)


class MopdLossTests(unittest.TestCase):
    def test_cached_teacher_topk_matches_uncached_objective_and_gradient(self):
        torch.manual_seed(29)
        student = torch.randn(2, 3, 19, dtype=torch.float64, requires_grad=True)
        teacher = torch.randn(2, 3, 19, dtype=torch.float64)
        expected = bias_corrected_topk_reverse_kl(
            student, teacher, top_k=7, reduction="sum"
        )
        expected_gradient = torch.autograd.grad(expected, student, retain_graph=True)[0]
        teacher_log = torch.log_softmax(teacher.float(), dim=-1)
        values, indices = torch.topk(teacher_log, k=7, dim=-1)
        actual = sparse_teacher_topk_reverse_kl(
            student, indices, values, reduction="sum"
        )
        actual_gradient = torch.autograd.grad(actual, student)[0]
        self.assertTrue(torch.allclose(expected, actual, atol=2e-7, rtol=2e-7))
        self.assertTrue(
            torch.allclose(expected_gradient, actual_gradient, atol=2e-7, rtol=2e-7)
        )

    def test_full_vocabulary_value_and_gradient_match_reverse_kl(self):
        generator = torch.Generator().manual_seed(61001)
        student = torch.randn(2, 3, 11, generator=generator, dtype=torch.float64)
        teacher = torch.randn(2, 3, 11, generator=generator, dtype=torch.float64)
        left_logits = student.clone().requires_grad_(True)
        right_logits = student.clone().requires_grad_(True)
        corrected = bias_corrected_topk_reverse_kl(left_logits, teacher, 11)
        reference = full_reverse_kl(right_logits, teacher)
        corrected.backward()
        reference.backward()
        self.assertTrue(torch.allclose(corrected, reference, atol=2e-7, rtol=2e-7))
        self.assertTrue(
            torch.allclose(left_logits.grad, right_logits.grad, atol=2e-7, rtol=2e-7)
        )

    def test_corrected_topk_has_zero_loss_and_gradient_at_teacher(self):
        teacher = torch.tensor(
            [[2.0, 1.0, 0.5, -0.5, -1.0, -2.0]], dtype=torch.float64
        )
        student = teacher.clone().requires_grad_(True)
        loss = bias_corrected_topk_reverse_kl(student, teacher, 3)
        loss.backward()
        self.assertAlmostEqual(float(loss.detach()), 0.0, places=7)
        self.assertTrue(torch.allclose(student.grad, torch.zeros_like(student), atol=2e-7))

    def test_rejects_shape_rank_and_reduction_errors(self):
        logits = torch.zeros(2, 5)
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            bias_corrected_topk_reverse_kl(logits, torch.zeros(3, 5), 2)
        for value in (0, 6):
            with self.assertRaisesRegex(ValueError, "top_k"):
                bias_corrected_topk_reverse_kl(logits, logits, value)
        with self.assertRaisesRegex(ValueError, "reduction"):
            bias_corrected_topk_reverse_kl(logits, logits, 2, reduction="median")


if __name__ == "__main__":
    unittest.main()
