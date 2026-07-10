from __future__ import annotations

import json
import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
SRC = EXP / "src"
sys.path.insert(0, str(SRC))

import scientific_artifacts as artifacts  # noqa: E402


ENGINE = {
    "max_model_len": 65536,
    "max_num_seqs": 64,
    "max_num_batched_tokens": 32768,
}


class ScientificArtifactTests(unittest.TestCase):
    def _protocol_binding(self) -> dict:
        core = {
            "schema_version": 1,
            "experiment_id": artifacts.EXPERIMENT_ID,
            "files": [],
            "smoke_libraries": {
                "base": {"library_id": "base-fixture", "content_sha256": "b" * 64},
                "designed_ceiling": {
                    "library_id": "designed-fixture",
                    "content_sha256": "d" * 64,
                },
            },
            "library_scope": "fixture",
        }
        payload = json.dumps(
            core, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        return {**core, "binding_sha256": hashlib.sha256(payload).hexdigest()}

    def _sampling(self, *, budget: int, k: int) -> dict:
        return {
            "thinking": "budget",
            "thinking_budget": budget,
            "answer_max_tokens": 512,
            "n": k,
            "run_seed": 2701,
        }

    def _preflight(self, arm: str, *, count: int = 2, budget: int = 32768) -> dict:
        reserve = budget + artifacts.FORCED_CLOSE_TOKENS + 512
        records = []
        for index in range(count):
            records.append(
                {
                    "id": f"task-{index}::{arm}",
                    "input_record_sha256": f"{index + 1:064x}",
                    "rendered_prompt_sha256": f"{index + 101:064x}",
                    "prompt_tokens": 100 + index,
                    "prompt_plus_reserve_tokens": reserve + 100 + index,
                }
            )
        return {
            "schema_version": 1,
            "pass": True,
            "max_model_len": 65536,
            "generation_reserve_tokens": reserve,
            "n_records": count,
            "min_prompt_tokens": 100,
            "max_prompt_tokens": 100 + count - 1,
            "max_prompt_plus_reserve_tokens": reserve + 99 + count,
            "records": records,
        }

    def _write_complete_inputs(
        self,
        root: Path,
        prefix: str,
        *,
        arm: str,
        k: int = 2,
        budget: int = 32768,
        role: str = "complete_matrix_arm",
        tier_mode: str = "complete_k12_matrix",
    ) -> dict:
        preflight = self._preflight(arm, budget=budget)
        artifacts.write_preflight_only(root, prefix, preflight)
        paths = artifacts.bundle_paths(root, prefix)
        rows = []
        for record in preflight["records"]:
            rows.append(
                {
                    "id": record["id"],
                    "meta": {"arm": arm},
                    "prompt_sha256": record["rendered_prompt_sha256"],
                    "n_prompt_tokens": record["prompt_tokens"],
                    "outputs": [
                        {
                            "sample_index": sample_index,
                            "text": "opaque and deliberately uninspected",
                            "token_ids": [sample_index + 1],
                        }
                        for sample_index in range(k)
                    ],
                }
            )
        paths.rows.write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )
        sampling = self._sampling(budget=budget, k=k)
        metadata = {
            "schema_version": 3,
            "model": artifacts.MODEL_ID,
            "model_revision": artifacts.MODEL_REVISION,
            "runner_sha256": artifacts.RUNNER_SHA256,
            "sampling": sampling,
            "engine": ENGINE,
            "counts": {"requests": len(rows), "completions": len(rows) * k},
        }
        paths.metadata.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        expected_identity = {
            "model": artifacts.MODEL_ID,
            "model_revision": artifacts.MODEL_REVISION,
            "runner_sha256": artifacts.RUNNER_SHA256,
            "sampling": sampling,
            "engine": ENGINE,
        }
        receipt = artifacts.commit_receipt(
            root,
            prefix,
            role=role,
            tier_mode=tier_mode,
            thinking_budget=budget,
            arm=arm,
            k=k,
            expected_identity=expected_identity,
        )
        return {"paths": paths, "receipt": receipt, "identity": expected_identity}

    def test_root_precedence_and_absolute_requirement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            explicit = Path(directory) / "explicit"
            environmental = Path(directory) / "environmental"
            env = {artifacts.ARTIFACT_ROOT_ENV: str(environmental)}
            self.assertEqual(artifacts.resolve_artifact_root(explicit, environ=env), explicit)
            self.assertEqual(artifacts.resolve_artifact_root(environ=env), environmental)
            self.assertEqual(
                artifacts.resolve_artifact_root(environ={}), artifacts.DEFAULT_ARTIFACT_ROOT
            )
        with self.assertRaisesRegex(artifacts.ScientificArtifactError, "absolute"):
            artifacts.resolve_artifact_root("relative/artifacts", environ={})

    def test_containment_namespace_and_symlinks_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "root"
            root.mkdir()
            with self.assertRaisesRegex(artifacts.ScientificArtifactError, "traversal"):
                artifacts.safe_path(root, "../escape.json")
            with self.assertRaisesRegex(artifacts.ScientificArtifactError, "must be relative"):
                artifacts.safe_path(root, "/escape.json")
            with self.assertRaisesRegex(artifacts.ScientificArtifactError, "bundle prefix"):
                artifacts.bundle_paths(root, "smoke/think_32768/base")

            target = Path(directory) / "real-root"
            target.mkdir()
            linked_root = Path(directory) / "linked-root"
            linked_root.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(artifacts.ScientificArtifactError, "symlink"):
                artifacts.resolve_artifact_root(linked_root)

            internal_target = Path(directory) / "internal-target"
            internal_target.mkdir()
            (root / "smoke_tiers").symlink_to(internal_target, target_is_directory=True)
            with self.assertRaisesRegex(artifacts.ScientificArtifactError, "symlink"):
                artifacts.bundle_paths(root, "smoke_tiers/think_32768/base")

    def test_preflight_only_is_the_one_valid_incomplete_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "external"
            prefix = "smoke_tiers/think_32768/base"
            self.assertEqual(artifacts.bundle_state(root, prefix)["status"], "absent")
            preflight = self._preflight("base")
            first = artifacts.write_preflight_only(root, prefix, preflight)
            second = artifacts.write_preflight_only(root, prefix, preflight)
            self.assertEqual(first, second)
            state = artifacts.bundle_state(root, prefix)
            self.assertEqual(state["status"], "preflight_only")
            self.assertEqual(state["n_records"], 2)
            self.assertIsNone(state["k"])

            changed = self._preflight("base")
            changed["records"][0]["prompt_tokens"] += 1
            with self.assertRaisesRegex(
                artifacts.ScientificArtifactError, "mismatch|differs"
            ):
                artifacts.write_preflight_only(root, prefix, changed)

            paths = artifacts.bundle_paths(root, prefix)
            paths.rows.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(artifacts.ScientificArtifactError, "partial"):
                artifacts.bundle_state(root, prefix)

    def test_receipt_is_last_written_and_binds_every_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "external"
            prefix = "smoke_tiers/think_32768/base"
            result = self._write_complete_inputs(root, prefix, arm="base")
            paths = result["paths"]
            receipt = result["receipt"]
            self.assertGreaterEqual(
                paths.receipt.stat().st_mtime_ns,
                max(paths.preflight.stat().st_mtime_ns, paths.rows.stat().st_mtime_ns,
                    paths.metadata.stat().st_mtime_ns),
            )
            self.assertEqual(receipt["commit_state"], "complete")
            self.assertEqual(receipt["n_records"], 2)
            self.assertEqual(receipt["n_completions"], 4)
            self.assertEqual(receipt["ordered_records"][0]["id"], "task-0::base")
            self.assertEqual(receipt["identity"]["sampling"], self._sampling(budget=32768, k=2))
            self.assertEqual(receipt["identity"]["engine"], ENGINE)
            self.assertEqual(receipt["identity"]["runner_sha256"], artifacts.RUNNER_SHA256)
            self.assertEqual(receipt["identity"]["model"], artifacts.MODEL_ID)
            verified = artifacts.verify_receipt(
                root, prefix, expected=result["identity"]
            )
            self.assertEqual(verified, receipt)
            self.assertEqual(artifacts.bundle_state(root, prefix)["status"], "complete")
            # Committing an exact completed bundle is idempotent, not a rewrite.
            again = artifacts.commit_receipt(
                root,
                prefix,
                role="complete_matrix_arm",
                tier_mode="complete_k12_matrix",
                thinking_budget=32768,
                arm="base",
                k=2,
                expected_identity=result["identity"],
            )
            self.assertEqual(again, receipt)

    def test_receipt_corruption_missing_files_and_identity_drift_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "external"
            prefix = "smoke_tiers/think_32768/base"
            result = self._write_complete_inputs(root, prefix, arm="base")
            with self.assertRaisesRegex(artifacts.ScientificArtifactError, "expectation"):
                artifacts.verify_receipt(
                    root, prefix, expected={"runner_sha256": "b" * 64}
                )
            result["paths"].rows.write_text(
                result["paths"].rows.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(artifacts.ScientificArtifactError, "size/hash/path"):
                artifacts.verify_receipt(root, prefix)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "external"
            prefix = "smoke_tiers/think_32768/base"
            result = self._write_complete_inputs(root, prefix, arm="base")
            result["paths"].metadata.unlink()
            with self.assertRaisesRegex(artifacts.ScientificArtifactError, "missing"):
                artifacts.verify_receipt(root, prefix)

    def test_probe_receipt_has_distinct_nonselectable_role(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "external"
            prefix = "smoke_budget_probes/think_49152/base"
            result = self._write_complete_inputs(
                root,
                prefix,
                arm="base",
                budget=49152,
                role="termination_probe",
                tier_mode="termination_probe_only",
            )
            self.assertEqual(result["receipt"]["role"], "termination_probe")
            self.assertEqual(result["receipt"]["tier_mode"], "termination_probe_only")

    def test_catalog_is_deterministic_and_selection_is_logical_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            root = workspace / "external"
            self._write_complete_inputs(
                root, "smoke_tiers/think_32768/base", arm="base"
            )
            self._write_complete_inputs(
                root,
                "smoke_tiers/think_32768/designed_ceiling",
                arm="designed_ceiling",
            )
            artifacts.write_preflight_only(
                root,
                "smoke_budget_probes/think_49152/base",
                self._preflight("base", budget=49152),
            )
            selection = workspace / "smoke_budget_selection.json"
            selection.write_text(
                json.dumps({"pass": True, "selected_thinking_budget": 32768}) + "\n",
                encoding="utf-8",
            )
            selected_entries = {
                "designed_ceiling": "matrix/think_32768/designed_ceiling",
                "base": "matrix/think_32768/base",
            }
            first = artifacts.build_catalog(
                root,
                protocol_binding=self._protocol_binding(),
                selection_file=selection,
                selected_budget=32768,
                selected_entries=selected_entries,
            )
            second = artifacts.build_catalog(
                root,
                protocol_binding=self._protocol_binding(),
                selection_file=selection,
                selected_budget=32768,
                selected_entries=dict(reversed(list(selected_entries.items()))),
            )
            self.assertEqual(first, second)
            self.assertEqual(
                [entry["id"] for entry in first["entries"]],
                sorted(entry["id"] for entry in first["entries"]),
            )
            probe = next(
                entry for entry in first["entries"] if entry["id"] == "probe/think_49152/base"
            )
            self.assertEqual(probe["status"], "preflight_only")
            self.assertEqual(
                first["selected"]["selection_path"], artifacts.SELECTION_LOGICAL_PATH
            )
            self.assertEqual(
                first["selected"]["arms"]["base"], "matrix/think_32768/base"
            )
            self.assertFalse((root / "smoke").exists())

            catalog_path = workspace / "scientific_smoke_artifact_catalog.json"
            artifacts.write_catalog(catalog_path, first)
            self.assertEqual(
                artifacts.verify_catalog(
                    catalog_path,
                    root,
                    protocol_binding=self._protocol_binding(),
                    selection_file=selection,
                ),
                first,
            )
            selection.write_text(
                json.dumps({"pass": False, "selected_thinking_budget": None}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(artifacts.ScientificArtifactError, "differs"):
                artifacts.verify_catalog(
                    catalog_path,
                    root,
                    protocol_binding=self._protocol_binding(),
                    selection_file=selection,
                )

    def test_catalog_rejects_probe_selection_missing_external_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            root = workspace / "external"
            probe = self._write_complete_inputs(
                root,
                "smoke_budget_probes/think_49152/base",
                arm="base",
                budget=49152,
                role="termination_probe",
                tier_mode="termination_probe_only",
            )
            selection = workspace / "selection.json"
            selection.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(artifacts.ScientificArtifactError, "cannot be selected"):
                artifacts.build_catalog(
                    root,
                    protocol_binding=self._protocol_binding(),
                    selection_file=selection,
                    selected_budget=49152,
                    selected_entries={"base": "probe/think_49152/base"},
                )

            catalog = artifacts.build_catalog(
                root, protocol_binding=self._protocol_binding()
            )
            catalog_path = workspace / "catalog.json"
            artifacts.write_catalog(catalog_path, catalog)
            probe["paths"].rows.unlink()
            with self.assertRaises(artifacts.ScientificArtifactError):
                artifacts.verify_catalog(
                    catalog_path, root, protocol_binding=self._protocol_binding()
                )

        if hasattr(os, "symlink"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "external"
                root.mkdir()
                target = Path(directory) / "target.json"
                target.write_text("{}\n", encoding="utf-8")
                linked = root / "smoke_tiers"
                linked.symlink_to(target.parent, target_is_directory=True)
                with self.assertRaisesRegex(artifacts.ScientificArtifactError, "symlink"):
                    artifacts.build_catalog(
                        root, protocol_binding=self._protocol_binding()
                    )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "external"
            (root / "smoke_tiers" / "think_32768" / "unexpected").mkdir(
                parents=True
            )
            with self.assertRaisesRegex(
                artifacts.ScientificArtifactError, "unexpected directory"
            ):
                artifacts.build_catalog(root, protocol_binding=self._protocol_binding())


if __name__ == "__main__":
    unittest.main()
