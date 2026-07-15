from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


RUNNER = Path(__file__).resolve().parents[1] / "src" / "vllm_runner.py"
SPEC = importlib.util.spec_from_file_location("restart_vllm_runner", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ModelOverrideTests(unittest.TestCase):
    def test_exact_qwen_composite_fingerprint_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text(json.dumps({
                "model_type": "qwen3_5",
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "text_config": {
                    "model_type": "qwen3_5_text",
                    "vocab_size": 248320,
                    "hidden_size": 2560,
                    "num_hidden_layers": 32,
                },
            }))
            MODULE.EngineConfig(model_override=path).validate()

    def test_override_and_runtime_adapter_are_mutually_exclusive(self) -> None:
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            MODULE.EngineConfig(adapter=Path("a"), model_override=Path("b")).validate()

    def test_wrong_architecture_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text(json.dumps({"model_type": "other"}))
            with self.assertRaisesRegex(ValueError, "not a merged Qwen"):
                MODULE.EngineConfig(model_override=path).validate()


if __name__ == "__main__":
    unittest.main()
