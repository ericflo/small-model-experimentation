from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from typing import Any
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
LAUNCHER = EXP / "scripts/mechanics_launcher"
SOURCE = EXP / "scripts/mechanics_launcher.S"
SCRIPT = EXP / "scripts/run_mechanics.py"
PYTHON = ROOT / ".venv-vllm/bin/python"
SRC = EXP / "src"
EXPECTED_SHA256 = "6fdfb46399c7880da2be42b93b78975cc3354301840dde79de74569e5e4cc4f2"

sys.path.insert(0, str(SRC))

import mechanics_lock  # noqa: E402


class MechanicsBootstrapTests(unittest.TestCase):
    @staticmethod
    def _bootstrap_namespace() -> dict[str, Any]:
        tree = ast.parse(SCRIPT.read_text())
        names = {
            "_BOOTSTRAP_IMPORT_FILES",
            "_BOOTSTRAP_RUNTIME_FILES",
            "_BOOTSTRAP_SUPPORT_FILES",
        }
        assignments = [
            node
            for node in tree.body
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            and any(
                isinstance(target, ast.Name) and target.id in names
                for target in (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
            )
        ]
        namespace = {
            "EXP_REL": EXP.relative_to(ROOT),
            "str": str,
        }
        exec(compile(ast.Module(assignments, []), str(SCRIPT), "exec"), namespace)
        return namespace

    @staticmethod
    def _strict_json_function() -> Any:
        tree = ast.parse(SCRIPT.read_text())
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_strict_json"
        )
        module = ast.Module(
            [ast.ImportFrom("__future__", [ast.alias("annotations")], 0), function],
            [],
        )
        namespace = {"json": json, "Path": Path, "Any": Any}
        exec(compile(ast.fix_missing_locations(module), str(SCRIPT), "exec"), namespace)
        return namespace["_strict_json"]

    def test_preimport_and_imported_runtime_inventories_match(self) -> None:
        namespace = self._bootstrap_namespace()
        self.assertEqual(
            namespace["_BOOTSTRAP_RUNTIME_FILES"],
            mechanics_lock.MECHANICS_RUNTIME_FILES,
        )

    def test_support_inventory_covers_immutable_calibration_and_reviews(self) -> None:
        namespace = self._bootstrap_namespace()
        prefix = str(EXP.relative_to(ROOT)) + "/"
        expected = {
            prefix + "reports/calibration_implementation_review.json",
            prefix + "reports/calibration_implementation_review.md",
            prefix + "reports/mechanics_implementation_review.json",
            prefix + "reports/mechanics_implementation_review.md",
            prefix + "runs/prepared/calibration_requests.jsonl",
            prefix + "runs/prepared/preoutcome_receipt.json",
            prefix + "runs/tokenizer/receipt.json",
            prefix + "scripts/calibration_launcher",
            prefix + "scripts/calibration_launcher.S",
            prefix + "scripts/run_calibration.py",
            prefix + "src/process_lock.py",
        }
        self.assertEqual(set(namespace["_BOOTSTRAP_SUPPORT_FILES"]), expected)
        tree = ast.parse(SCRIPT.read_text())
        audit = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_install_mechanics_path_audit"
        )
        self.assertIn(
            "_BOOTSTRAP_SUPPORT_FILES",
            {node.id for node in ast.walk(audit) if isinstance(node, ast.Name)},
        )

    def test_preimport_json_requires_canonical_finite_unique_object(self) -> None:
        strict_json = self._strict_json_function()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            path.write_text('{\n  "value": 1\n}\n')
            self.assertEqual(strict_json(path, "test"), {"value": 1})
            for invalid in ('{"value":1}\n', '{"value": NaN}\n', '{"a": 1, "a": 2}\n'):
                path.write_text(invalid)
                with self.assertRaises(RuntimeError):
                    strict_json(path, "test")

    def test_static_launcher_is_reproducible_and_has_no_interpreter(self) -> None:
        self.assertEqual(hashlib.sha256(LAUNCHER.read_bytes()).hexdigest(), EXPECTED_SHA256)
        program_headers = subprocess.check_output(
            ["/usr/bin/readelf", "-l", str(LAUNCHER)], text=True
        )
        self.assertNotIn("INTERP", program_headers)
        with tempfile.TemporaryDirectory() as directory:
            rebuilt = Path(directory) / "mechanics_launcher"
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
                    str(SOURCE),
                ],
                check=True,
            )
            self.assertEqual(rebuilt.read_bytes(), LAUNCHER.read_bytes())

    def test_direct_python_cannot_forge_static_parent_provenance(self) -> None:
        result = subprocess.run(
            [str(PYTHON), "-I", "-B", str(SCRIPT), "--stage", "lock"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("launcher proof", result.stderr)


if __name__ == "__main__":
    unittest.main()
