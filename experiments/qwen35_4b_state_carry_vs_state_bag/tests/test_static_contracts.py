from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class StaticContractTests(unittest.TestCase):
    def test_source_never_imports_benchmarks(self) -> None:
        for path in [*ROOT.glob("src/*.py"), *ROOT.glob("scripts/*.py")]:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                self.assertFalse(any(name == "benchmarks" or name.startswith("benchmarks.") for name in names))

    def test_no_result_bearing_vllm_runner_remains(self) -> None:
        self.assertFalse((ROOT / "src" / "vllm_runner.py").exists())
        self.assertFalse((ROOT / "tests" / "test_vllm_runner.py").exists())

    def test_python_sources_compile_without_gpu_dependencies(self) -> None:
        for path in [*ROOT.glob("src/*.py"), *ROOT.glob("scripts/*.py")]:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")

    def test_sample_more_enforces_compute_and_natural_close(self) -> None:
        source = (ROOT / "src" / "gpu_runner.py").read_text(encoding="utf-8")
        self.assertIn("sample_budget > recurrent_budget", source)
        self.assertIn('if "</think>" not in text', source)

    def test_gpu_loader_fails_on_transformers_drift(self) -> None:
        source = (ROOT / "src" / "gpu_runner.py").read_text(encoding="utf-8")
        self.assertIn("Transformers drift", source)
        self.assertIn("loaded model commit", source)


if __name__ == "__main__":
    unittest.main()
