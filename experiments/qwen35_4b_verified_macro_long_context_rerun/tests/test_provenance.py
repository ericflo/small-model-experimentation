"""Focused tests for active and archived parent-data provenance."""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest import mock


EXP = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verified_macro_provenance_run",
    EXP / "scripts" / "run.py",
)
assert SPEC is not None and SPEC.loader is not None
run = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = run
SPEC.loader.exec_module(run)


class CopiedDataProvenanceTests(unittest.TestCase):
    def test_archived_smoke_v1_files_have_exact_parent_paths_and_hashes(self) -> None:
        provenance = json.loads(
            (EXP / "data" / "source_provenance.json").read_text(encoding="utf-8")
        )
        prefix = "data/smoke_v1_frozen/"
        archived = {
            str(entry["path"])[len(prefix) :]: entry
            for entry in provenance["copied_files"]
            if str(entry["path"]).startswith(prefix)
        }

        self.assertEqual(set(archived), set(run.PARENT_SMOKE_V1_ARTIFACT_SHA256))
        for relative, expected_hash in run.PARENT_SMOKE_V1_ARTIFACT_SHA256.items():
            with self.subTest(relative=relative):
                entry = archived[relative]
                self.assertEqual(entry["sha256"], expected_hash)
                self.assertEqual(
                    entry["source_path"],
                    "experiments/qwen35_4b_verified_macro_invention/"
                    f"data/smoke_v1_frozen/{relative}",
                )
                self.assertEqual(
                    run._sha256_file(EXP / "data" / "smoke_v1_frozen" / relative),
                    expected_hash,
                )

    def test_prepare_verification_rejects_archived_smoke_v1_drift(self) -> None:
        real_sha256_file = run._sha256_file

        def drift_archived(path: Path) -> str:
            if path.parent.name == "smoke_v1_frozen" and path.name == "tasks.json":
                return "0" * 64
            return real_sha256_file(path)

        with mock.patch.object(run, "_sha256_file", side_effect=drift_archived):
            with self.assertRaisesRegex(ValueError, "archived smoke-v1 artifact hash mismatch"):
                run._verify_frozen_data(run.load_config())


if __name__ == "__main__":
    unittest.main()
