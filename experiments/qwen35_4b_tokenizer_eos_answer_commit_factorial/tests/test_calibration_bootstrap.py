from __future__ import annotations

import ast
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]


class CalibrationBootstrapTests(unittest.TestCase):
    def test_static_launcher_is_reproducible_and_has_no_elf_interpreter(self) -> None:
        source = EXP / "scripts/calibration_launcher.S"
        launcher = EXP / "scripts/calibration_launcher"
        self.assertTrue(os.access(launcher, os.X_OK))
        self.assertEqual(
            hashlib.sha256(launcher.read_bytes()).hexdigest(),
            "5947d78038cb969caaf2df633468eed9075c90e449fbaa1f634981bc252e41c2",
        )
        headers = subprocess.run(
            ["/usr/bin/readelf", "-l", str(launcher)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        self.assertNotIn("INTERP", headers)
        with tempfile.TemporaryDirectory() as temporary:
            rebuilt = Path(temporary) / "calibration_launcher"
            subprocess.run(
                [
                    "/usr/bin/gcc",
                    "-nostdlib",
                    "-static",
                    "-no-pie",
                    "-s",
                    "-Wl,--build-id=none",
                    "-o",
                    str(rebuilt),
                    str(source),
                ],
                check=True,
            )
            self.assertEqual(rebuilt.read_bytes(), launcher.read_bytes())

    def test_static_launcher_scrubs_inherited_environment_before_python(self) -> None:
        launcher = EXP / "scripts/calibration_launcher"
        environment = {
            "LD_PRELOAD": "/definitely/not/a/library.so",
            "LD_AUDIT": "/definitely/not/an/auditor.so",
            "PATH": "/tmp/forged",
            "PYTHONPATH": "/tmp/forged-python",
            "GIT_DIR": "/tmp/forged-git",
            "GH_REPO": "attacker/repository",
        }
        result = subprocess.run(
            [str(launcher), "--stage", "invalid"],
            env=environment,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("pre-import calibration stage is invalid", result.stderr)
        self.assertNotIn("trusted static calibration_launcher", result.stderr)
        self.assertNotIn("cannot be preloaded", result.stderr)

    def test_firewall_is_installed_before_every_local_runtime_import(self) -> None:
        source = (EXP / "scripts/run_calibration.py").read_text()
        ast.parse(source)
        isolated_guard = source.index("sys.flags.isolated == 1")
        first_shadowable_import = source.index("import argparse")
        bootstrap = source.index("_bootstrap_verify_before_local_imports()")
        path_insert = source.index("sys.path.insert(0, str(SRC))")
        local_import = source.index("from calibration_lock import")
        self.assertLess(isolated_guard, first_shadowable_import)
        self.assertLess(bootstrap, path_insert)
        self.assertLess(bootstrap, local_import)
        preimport = source[:local_import]
        self.assertIn("sys.addaudithook(audit)", preimport)
        self.assertIn("forbids benchmark access", preimport)
        self.assertIn("forbids unregistered repository access", preimport)
        self.assertIn("pre-import calibration refuses local Python caches", source)
        self.assertIn("requires the pinned .venv-vllm interpreter", source)
        self.assertIn("review_release = _bootstrap_authenticate_review_release()", source)
        self.assertIn('if stage == "lock":', source)
        self.assertIn('_GIT_EXECUTABLE = "/usr/bin/git"', source)
        self.assertIn('_GH_EXECUTABLE = "/usr/bin/gh"', source)
        self.assertIn('"--repo", _CANONICAL_REPOSITORY', source)
        self.assertIn("env=_bootstrap_child_environment()", source)
        self.assertIn("pre-import Git origin URL changed", source)
        self.assertIn("requires the trusted static calibration_launcher", source)
        self.assertIn("sealed calibration static launcher bytes changed", source)

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
            "src/process_lock.py",
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

    def test_isolated_mode_guard_precedes_shadowable_imports(self) -> None:
        source_script = EXP / "scripts/run_calibration.py"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = root / "workspace/experiment/scripts/run_calibration.py"
            script.parent.mkdir(parents=True)
            sentinel = root / "shadow-executed"
            shutil.copyfile(source_script, script)
            (script.parent / "json.py").write_text(
                f"open({str(sentinel)!r}, 'w').write('executed')\n"
            )
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(script.parent)
            unsafe = subprocess.run(
                [sys.executable, "-B", str(script), "--stage", "invalid"],
                env=environment,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(unsafe.returncode, 0)
            self.assertIn("requires isolated Python", unsafe.stderr)
            self.assertFalse(sentinel.exists())
            isolated = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-I",
                    str(script),
                    "--stage",
                    "invalid",
                ],
                env=environment,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(isolated.returncode, 0)
            self.assertIn("trusted static calibration_launcher", isolated.stderr)
            self.assertFalse(sentinel.exists())


if __name__ == "__main__":
    unittest.main()
