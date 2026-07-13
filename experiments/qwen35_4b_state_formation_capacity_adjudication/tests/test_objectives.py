from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.gpu_runner import _objective_loss  # noqa: E402


class ObjectiveCompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(ROOT / "configs" / "default.yaml")

    @staticmethod
    def output(*, include_answer: bool) -> SimpleNamespace:
        return SimpleNamespace(
            answer_loss=torch.tensor(2.0, requires_grad=True) if include_answer else None,
            answer_logits=torch.ones(1, 4) if include_answer else None,
            state_loss=torch.tensor(3.0, requires_grad=True),
            fixed_point_loss=torch.tensor(5.0, requires_grad=True),
        )

    def test_state_only_omits_answer_term_and_answer_gradient(self) -> None:
        output = self.output(include_answer=False)
        answer_probe = torch.tensor(2.0, requires_grad=True)
        loss = _objective_loss(output, self.config, "state_only")
        self.assertAlmostEqual(float(loss.detach()), 0.5 * 3.0 + 0.05 * 5.0)
        loss.backward()
        self.assertIsNone(answer_probe.grad)
        self.assertAlmostEqual(float(output.state_loss.grad), 0.5)
        self.assertAlmostEqual(float(output.fixed_point_loss.grad), 0.05)

        prohibited = self.output(include_answer=True)
        with self.assertRaisesRegex(RuntimeError, "prohibited answer graph"):
            _objective_loss(prohibited, self.config, "state_only")

    def test_joint_includes_answer_term_and_answer_gradient(self) -> None:
        output = self.output(include_answer=True)
        loss = _objective_loss(output, self.config, "joint")
        self.assertAlmostEqual(float(loss.detach()), 2.0 + 0.5 * 3.0 + 0.05 * 5.0)
        loss.backward()
        self.assertAlmostEqual(float(output.answer_loss.grad), 1.0)
        self.assertAlmostEqual(float(output.state_loss.grad), 0.5)
        self.assertAlmostEqual(float(output.fixed_point_loss.grad), 0.05)


if __name__ == "__main__":
    unittest.main()
