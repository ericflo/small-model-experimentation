from __future__ import annotations

import ast
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]


class CalibrationBootstrapTests(unittest.TestCase):
    def test_firewall_is_installed_before_every_local_runtime_import(self) -> None:
        source = (EXP / "scripts/run_calibration.py").read_text()
        ast.parse(source)
        bootstrap = source.index("_bootstrap_verify_before_local_imports()")
        path_insert = source.index("sys.path.insert(0, str(SRC))")
        local_import = source.index("from calibration_lock import")
        self.assertLess(bootstrap, path_insert)
        self.assertLess(bootstrap, local_import)
        preimport = source[:local_import]
        self.assertIn("sys.addaudithook(audit)", preimport)
        self.assertIn("forbids benchmark access", preimport)
        self.assertIn("forbids unregistered repository access", preimport)
        self.assertIn("pre-import calibration refuses local Python caches", source)
        self.assertIn("requires the pinned .venv-vllm interpreter", source)

    def test_bootstrap_binds_exact_model_and_runtime_import_inventory(self) -> None:
        source = (EXP / "scripts/run_calibration.py").read_text()
        self.assertIn('_MODEL_ID = "Qwen/Qwen3.5-4B"', source)
        self.assertIn(
            '_MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"',
            source,
        )
        for relative in (
            "src/calibration_lock.py",
            "src/calibration_stage.py",
            "src/interface_analysis.py",
            "src/protocol.py",
            "src/transactions.py",
            "src/vllm_runner.py",
        ):
            self.assertIn(relative, source)


if __name__ == "__main__":
    unittest.main()
