#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

RUNNER_PATH = Path(__file__).resolve().parents[1] / "src" / "vllm_runner.py"
SPEC = importlib.util.spec_from_file_location("round2_vllm_runner", RUNNER_PATH)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


class EngineConfigTests(unittest.TestCase):
    def test_base_geometry_validates(self):
        runner.EngineConfig(max_model_len=12288, max_num_seqs=64,
                            max_num_batched_tokens=16384).validate()

    def test_model_override_requires_composite_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                runner.EngineConfig(model_override=Path(tmp)).validate()
            (Path(tmp) / "config.json").write_text("{}")
            runner.EngineConfig(model_override=Path(tmp)).validate()

    def test_adapter_and_override_are_mutually_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "config.json").write_text("{}")
            with self.assertRaises(ValueError):
                runner.EngineConfig(adapter=root, model_override=root).validate()


class SamplingConfigTests(unittest.TestCase):
    def test_budget_requires_budget_and_answer_allowance(self):
        runner.SamplingConfig(thinking="budget", thinking_budget=512,
                              answer_max_tokens=256, greedy=True).validate()
        with self.assertRaises(ValueError):
            runner.SamplingConfig(thinking="budget", answer_max_tokens=256,
                                  greedy=True).validate()

    def test_greedy_resolves_to_zero_temperature(self):
        config = runner.SamplingConfig(thinking="off", max_tokens=32, greedy=True,
                                       temperature=.6)
        config.validate()
        self.assertEqual(config.resolved_sampling()["temperature"], 0.0)


if __name__ == "__main__":
    unittest.main()
