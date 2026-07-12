from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

import harness  # noqa: E402


class HarnessEngineGeometryTests(unittest.TestCase):
    def test_make_runner_forwards_frozen_capture_geometry(self) -> None:
        captured = []

        def fake_runner(config):
            captured.append(config)
            return config

        engine = {
            "max_model_len": 16_384,
            "gpu_memory_utilization": 0.85,
            "max_num_seqs": 48,
            "max_num_batched_tokens": 16_384,
            "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 24, 32, 40, 48],
        }
        with patch.object(harness, "VLLMRunner", side_effect=fake_runner):
            result = harness.make_runner(engine, model_override="/tmp/model")

        self.assertIs(result, captured[0])
        self.assertEqual(result.max_num_seqs, 48)
        self.assertEqual(
            result.cudagraph_capture_sizes,
            (1, 2, 4, 8, 16, 24, 32, 40, 48),
        )
        self.assertEqual(result.model_override, Path("/tmp/model"))


if __name__ == "__main__":
    unittest.main()
