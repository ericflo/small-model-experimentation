from __future__ import annotations

import ast
import contextlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import design_boundary  # noqa: E402
from src.config import SOURCE_CONTRACT_FILES, load_config  # noqa: E402


class DesignBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(ROOT / "configs" / "default.yaml")

    def test_boundary_manifest_is_exactly_the_scientific_design(self) -> None:
        paths = {design_boundary._relative(path) for path in design_boundary._contract_paths()}
        expected_design = {design_boundary._relative(ROOT / item) for item in design_boundary.DESIGN_FILES}
        self.assertEqual(paths, expected_design)
        self.assertNotIn(
            design_boundary._relative(ROOT / "reports" / "implementation_review.md"),
            paths,
        )
        self.assertNotIn(design_boundary._relative(ROOT / "src" / "analysis.py"), paths)
        self.assertNotIn(design_boundary._relative(ROOT / "tests" / "test_analysis.py"), paths)
        self.assertNotIn(design_boundary._relative(design_boundary.REQUIREMENTS_LOCK), paths)
        self.assertNotIn(
            design_boundary._relative(ROOT / "reports" / "report.md"), paths
        )

    def test_freeze_and_reopen_are_content_addressed_and_tamper_evident(self) -> None:
        synthetic_manifest = [
            {"path": "experiment/config.yaml", "bytes": 10, "sha256": "a" * 64},
            {"path": "experiment/test.py", "bytes": 20, "sha256": "b" * 64},
        ]
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            path = Path(directory) / "design_receipt.json"
            patches = (
                mock.patch.object(
                    design_boundary,
                    "canonical_design_receipt_path",
                    return_value=path.resolve(),
                ),
                mock.patch.object(
                    design_boundary, "_manifest", return_value=synthetic_manifest
                ),
                mock.patch.object(design_boundary, "_git_head", return_value="1" * 40),
                mock.patch.object(design_boundary, "_require_tracked_clean_inputs"),
                mock.patch.object(design_boundary, "require_implementation_go"),
                mock.patch.object(
                    design_boundary,
                    "_implementation_go_snapshot",
                    side_effect=lambda: contextlib.nullcontext(),
                ),
            )
            with (
                patches[0], patches[1], patches[2],
                patches[3] as clean_inputs, patches[4] as implementation_go,
                patches[5],
            ):
                receipt = design_boundary.freeze_design(self.config, path)
                self.assertEqual(clean_inputs.call_count, 2)
                implementation_go.assert_called_once_with()
                self.assertEqual(receipt["status"], "DESIGN_FROZEN")
                self.assertEqual(receipt["phase"], "design_boundary")
                self.assertEqual(
                    set(receipt["implementation_provenance_at_freeze"]),
                    {
                        "source_contract_sha256",
                        "requirements_training_lock_sha256",
                        "git_head",
                    },
                )
                self.assertEqual(
                    design_boundary.validate_design_receipt(self.config, path), receipt
                )
                with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
                    design_boundary.freeze_design(self.config, path)

                tampered = dict(receipt)
                tampered["benchmark_files_read"] = 1
                path.write_text(json.dumps(tampered), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                    design_boundary.validate_design_receipt(self.config, path)

    def test_implementation_review_requires_one_exact_go_status(self) -> None:
        digest = "d" * 64
        exact = (
            "# Review\n\n"
            "**Source-contract version:** `7`\n"
            f"**Reviewed implementation SHA-256:** `{digest}`\n"
        )
        cases = (
            (exact + "**Status:** `GO`\n", True),
            (exact + "**Status:** `NO_GO`\n", False),
            (exact.replace("version:** `7`", "version:** `6`") + "**Status:** `GO`\n", False),
            (exact.replace(digest, "e" * 64) + "**Status:** `GO`\n", False),
            ("# Review\n\n**Status:** `GO`\n", False),
            (exact + "**Status:** `GO`\n**Status:** `GO`\n", False),
            ("The implementation is GO.\n", False),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "implementation_review.md"
            with mock.patch.object(
                design_boundary, "IMPLEMENTATION_REVIEW", path
            ), mock.patch.object(
                design_boundary,
                "reviewed_implementation_snapshot",
                side_effect=lambda: contextlib.nullcontext(digest),
            ):
                with self.assertRaisesRegex(RuntimeError, "review is missing"):
                    design_boundary.require_implementation_go()
                for contents, accepted in cases:
                    path.write_text(contents, encoding="utf-8")
                    with self.subTest(contents=contents, accepted=accepted):
                        if accepted:
                            self.assertIsNone(design_boundary.require_implementation_go())
                        else:
                            with self.assertRaisesRegex(RuntimeError, "exact GO"):
                                design_boundary.require_implementation_go()

    def test_implementation_review_aliases_fail_closed(self) -> None:
        digest = "d" * 64
        encoded = (
            "# Review\n\n"
            "**Source-contract version:** `7`\n"
            f"**Reviewed implementation SHA-256:** `{digest}`\n"
            "**Status:** `GO`\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "real.md"
            real.write_text(encoded, encoding="utf-8")
            alias = root / "alias.md"
            alias.symlink_to(real)
            with mock.patch.object(
                design_boundary, "IMPLEMENTATION_REVIEW", alias
            ), mock.patch.object(
                design_boundary,
                "reviewed_implementation_snapshot",
                side_effect=lambda: contextlib.nullcontext(digest),
            ), self.assertRaisesRegex(RuntimeError, "aliased"):
                design_boundary.require_implementation_go()

            alias.unlink()
            os.link(real, alias)
            try:
                with mock.patch.object(
                    design_boundary, "IMPLEMENTATION_REVIEW", alias
                ), mock.patch.object(
                    design_boundary,
                    "reviewed_implementation_snapshot",
                    side_effect=lambda: contextlib.nullcontext(digest),
                ), self.assertRaisesRegex(RuntimeError, "aliased"):
                    design_boundary.require_implementation_go()
            finally:
                alias.unlink()

    def test_design_lineage_never_splits_parsed_identity_and_byte_hash(self) -> None:
        synthetic_manifest = [
            {"path": "experiment/config.yaml", "bytes": 10, "sha256": "a" * 64},
        ]
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as directory:
            path = Path(directory) / "design_receipt.json"
            with mock.patch.object(
                design_boundary, "canonical_design_receipt_path", return_value=path
            ), mock.patch.object(
                design_boundary, "_manifest", return_value=synthetic_manifest
            ), mock.patch.object(
                design_boundary, "_git_head", return_value="1" * 40
            ), mock.patch.object(
                design_boundary, "_require_tracked_clean_inputs"
            ), mock.patch.object(
                design_boundary, "require_implementation_go"
            ), mock.patch.object(
                design_boundary,
                "_implementation_go_snapshot",
                side_effect=lambda: contextlib.nullcontext(),
            ):
                receipt = design_boundary.freeze_design(self.config, path)
                expected_sha256 = design_boundary._sha256(path)
                lineage = design_boundary.design_lineage(self.config)
                self.assertEqual(lineage["sha256"], expected_sha256)
                self.assertEqual(
                    lineage["receipt_identity_sha256"],
                    receipt["receipt_identity_sha256"],
                )

                replacement = path.with_name("replacement.json")
                replacement.write_bytes(path.read_bytes())
                original_validate = design_boundary._validate_design_payload
                injected = False

                def replace_during_validation(*args: object, **kwargs: object):
                    nonlocal injected
                    os.replace(replacement, path)
                    injected = True
                    return original_validate(*args, **kwargs)

                with mock.patch.object(
                    design_boundary,
                    "_validate_design_payload",
                    side_effect=replace_during_validation,
                ), self.assertRaisesRegex(RuntimeError, "aliased"):
                    design_boundary.design_lineage(self.config)
                self.assertTrue(injected)

    def test_freeze_and_reopen_both_fail_closed_without_implementation_go(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "design_receipt.json"
            denied = RuntimeError("implementation review has not authorized exact GO")

            @contextlib.contextmanager
            def denied_snapshot():
                raise denied
                yield

            with mock.patch.object(
                design_boundary, "require_implementation_go", side_effect=denied
            ) as gate, mock.patch.object(
                design_boundary,
                "_implementation_go_snapshot",
                side_effect=denied_snapshot,
            ), mock.patch.object(
                design_boundary, "canonical_design_receipt_path", return_value=path
            ):
                with self.assertRaisesRegex(RuntimeError, "exact GO"):
                    design_boundary.freeze_design(self.config, path)
                with self.assertRaisesRegex(RuntimeError, "exact GO"):
                    design_boundary.validate_design_receipt(self.config, path)
            self.assertEqual(gate.call_count, 1)

    def test_prefreeze_requires_every_registered_input_tracked_and_clean_at_head(self) -> None:
        expected = {
            design_boundary._relative(design_boundary.REQUIREMENTS_LOCK),
            *(
                design_boundary._relative(ROOT / relative)
                for relative in design_boundary.DESIGN_FILES
            ),
            *(
                design_boundary._relative(ROOT / relative)
                for relative in SOURCE_CONTRACT_FILES
            ),
        }
        completed = (
            mock.Mock(returncode=0, stdout=""),
            mock.Mock(returncode=0, stdout=""),
        )
        with mock.patch.object(
            design_boundary.subprocess, "run", side_effect=completed
        ) as run:
            design_boundary._require_tracked_clean_inputs()

        self.assertEqual(run.call_count, 2)
        tracked_command = run.call_args_list[0].args[0]
        dirty_command = run.call_args_list[1].args[0]
        self.assertEqual(tracked_command[:4], ["git", "ls-files", "--error-unmatch", "--"])
        self.assertEqual(dirty_command[:4], ["git", "status", "--porcelain=v1", "--"])
        self.assertEqual(set(tracked_command[4:]), expected)
        self.assertEqual(set(dirty_command[4:]), expected)
        self.assertIn(
            design_boundary._relative(ROOT / "reports" / "implementation_review.md"),
            expected,
        )
        self.assertTrue(
            {
                design_boundary._relative(ROOT / "tests" / "test_analysis.py"),
                design_boundary._relative(ROOT / "tests" / "test_design_boundary.py"),
            }.issubset(expected)
        )
        self.assertNotIn(
            design_boundary._relative(ROOT / "reports" / "design_receipt.json"),
            expected,
        )

    def test_prefreeze_rejects_untracked_or_dirty_registered_inputs(self) -> None:
        cases = (
            (
                (mock.Mock(returncode=1, stdout=""),),
                "tracked at HEAD",
            ),
            (
                (
                    mock.Mock(returncode=0, stdout=""),
                    mock.Mock(returncode=0, stdout=" M registered-input"),
                ),
                "clean at HEAD",
            ),
        )
        for completed, message in cases:
            with self.subTest(message=message), mock.patch.object(
                design_boundary.subprocess, "run", side_effect=completed
            ), self.assertRaisesRegex(RuntimeError, message):
                design_boundary._require_tracked_clean_inputs()

    def test_every_result_or_artifact_stage_directly_reopens_design_boundary(self) -> None:
        required = {
            ROOT / "src" / "data_pipeline.py": (
                ("build_datasets", r"(?:validate_design_receipt|design_lineage)\s*\("),
            ),
            ROOT / "src" / "initialization.py": (
                ("prepare_initialization_bundle", r"(?:validate_design_receipt|design_lineage)\s*\("),
                ("load_initialization_bundle", r"(?:validate_design_receipt|design_lineage)\s*\("),
            ),
            # _load_data_manifest performs byte-only manifest validation, which
            # itself reopens the design boundary before any model is loaded.
            ROOT / "src" / "gpu_runner.py": tuple(
                (name, r"(?:validate_design_receipt|design_lineage|_load_data_manifest)\s*\(")
                for name in ("model_smoke", "positive_control", "train", "evaluate_state")
            ),
            ROOT / "src" / "analysis.py": (
                ("analyze_phase", r"(?:validate_design_receipt|design_lineage)\s*\("),
            ),
        }
        for path, function_contracts in required.items():
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            functions = {
                node.name: ast.get_source_segment(source, node) or ""
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            for function_name, pattern in function_contracts:
                with self.subTest(path=path.name, function=function_name):
                    self.assertIn(function_name, functions)
                    self.assertRegex(
                        functions[function_name],
                        pattern,
                    )


if __name__ == "__main__":
    unittest.main()
