from __future__ import annotations

import ast
import importlib.util
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
        self.assertIn("set(critical) != set(_BOOTSTRAP_CRITICAL_FILES)", source)
        self.assertIn("allowed != list(_BOOTSTRAP_RUNTIME_FILES)", source)
        self.assertIn(
            "_install_calibration_path_audit(list(_BOOTSTRAP_AUDIT_FILES))",
            source,
        )
        self.assertIn("pre-import review provenance changed", source)
        self.assertIn("pre-import frozen mechanics changed", source)
        self.assertIn("for published_commit in dict.fromkeys", source)

    def test_compiled_bootstrap_inventories_equal_runtime_lock_inventories(self) -> None:
        script = EXP / "scripts/run_calibration.py"
        spec = importlib.util.spec_from_file_location(
            "run_calibration_bootstrap_test", script
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        import calibration_lock

        self.assertEqual(
            tuple(module._BOOTSTRAP_RUNTIME_FILES),
            calibration_lock.CALIBRATION_RUNTIME_FILES,
        )
        self.assertEqual(
            set(module._BOOTSTRAP_AUDIT_FILES)
            - set(module._BOOTSTRAP_RUNTIME_FILES),
            {
                str(module.EXP_REL / "reports/calibration_implementation_review.md"),
                str(module.EXP_REL / "reports/calibration_implementation_review.json"),
            },
        )
        self.assertEqual(
            set(module._BOOTSTRAP_CRITICAL_FILES),
            set(calibration_lock.CRITICAL_FILES),
        )
        self.assertEqual(
            set(module._BOOTSTRAP_FROZEN_MECHANICS),
            set(calibration_lock.FROZEN_MECHANICS_FILES),
        )


if __name__ == "__main__":
    unittest.main()
