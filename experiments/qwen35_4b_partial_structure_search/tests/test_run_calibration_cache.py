from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_calibration.py"
SPEC = importlib.util.spec_from_file_location(
    "partial_structure_run_calibration", MODULE_PATH
)
assert SPEC and SPEC.loader
calibration = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = calibration
SPEC.loader.exec_module(calibration)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class CacheFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.config = {
            "model": {
                "id": calibration.MODEL_ID,
                "revision": calibration.MODEL_REVISION,
                "backend": "vllm",
            },
            "judge": {
                "thinking_budget": 256,
                "run_seed": 1701,
                "max_model_len": 4096,
                "max_num_seqs": 32,
                "gpu_memory_utilization": 0.9,
            },
            "calibration": {"shuffle_canary_tasks": 2},
            "search": {"direct_sample_pool_k": 320},
        }
        self.config_path = root / "configs" / "default.yaml"
        self.config_path.parent.mkdir(parents=True)
        # JSON is valid YAML. Keeping bytes frozen is what the cache fingerprints.
        self.config_path.write_text(
            json.dumps(self.config, sort_keys=True) + "\n", encoding="utf-8"
        )
        self.candidate_path = root / "data" / "calibration_candidates.jsonl"
        self.candidates = [
            {
                "id": "task-1:child-A",
                "task_text": "visible task",
                "visible_examples": [{"input": [1], "output": [2]}],
                "candidate_prefix": ["reverse"],
                "remaining_steps": 3,
                "parent_group": "task-1:parent",
                "task_id": "task-1",
            },
            {
                "id": "task-2:child-A",
                "task_text": "visible task",
                "visible_examples": [{"input": [2], "output": [4]}],
                "candidate_prefix": ["negate"],
                "remaining_steps": 3,
                "parent_group": "task-2:parent",
                "task_id": "task-2",
            },
        ]
        _write_jsonl(self.candidate_path, self.candidates)
        self.specs = calibration._output_specs(
            self.candidates, self.config, "", root=root
        )
        for spec in self.specs:
            _write_jsonl(
                spec.path,
                [
                    {
                        "id": record_id,
                        "p_viable": 0.5,
                        "accounting": {"requests": 1},
                    }
                    for record_id in spec.expected_ids
                ],
            )
        self.inputs = calibration._input_fingerprints(
            root=root,
            config_path=self.config_path,
            candidate_path=self.candidate_path,
        )
        self.receipt_path = root / "runs" / "calibration_model_receipt.json"

    def base_receipt(self, *, schema_version: int = 2) -> dict[str, object]:
        plan = calibration._scoring_plan(self.config, self.specs)
        summaries = {
            tag: {
                "method": row["method"],
                "run_seed": row["run_seed"],
                "thinking_budget": row["thinking_budget"],
                "accounting": {"requests": row["logical_requests"]},
            }
            for tag, row in plan.items()
        }
        return {
            "schema_version": schema_version,
            "model": calibration.MODEL_ID,
            "model_revision": calibration.MODEL_REVISION,
            "backend": "vllm",
            "engine_config": calibration._expected_engine_config(self.config),
            "scoring_summaries": summaries,
            "candidate_file_sha256": self.inputs["candidates"]["sha256"],
            "wall_seconds": 12.5,
        }

    def seal(self) -> dict[str, object]:
        receipt = calibration._seal_receipt(
            self.base_receipt(),
            root=self.root,
            cfg=self.config,
            specs=self.specs,
            input_fingerprints=self.inputs,
        )
        calibration._atomic_write_json(self.receipt_path, receipt)
        return receipt


class ReceiptIntegrityTests(unittest.TestCase):
    def test_valid_receipt_is_the_completion_sentinel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = CacheFixture(Path(directory))
            fixture.seal()

            state = calibration._accept_or_upgrade_cache(
                fixture.receipt_path,
                root=fixture.root,
                cfg=fixture.config,
                specs=fixture.specs,
                input_fingerprints=fixture.inputs,
            )

            self.assertEqual(state, "valid")
            receipt = json.loads(fixture.receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(receipt["schema_version"], 2)
            self.assertEqual(receipt["completion_status"], "complete")
            self.assertEqual(set(receipt["output_fingerprints"]), {s.tag for s in fixture.specs})
            self.assertFalse(list(fixture.receipt_path.parent.glob("*.tmp")))

    def test_every_input_fingerprint_is_checked(self) -> None:
        for fingerprint_name in (
            "config",
            "candidates",
            "model_scorer",
            "vllm_runner",
        ):
            with self.subTest(fingerprint=fingerprint_name), tempfile.TemporaryDirectory() as directory:
                fixture = CacheFixture(Path(directory))
                receipt = fixture.seal()
                hash_key = (
                    "projection_sha256" if fingerprint_name == "config" else "sha256"
                )
                receipt["input_fingerprints"][fingerprint_name][hash_key] = "0" * 64
                calibration._atomic_write_json(fixture.receipt_path, receipt)

                with self.assertRaisesRegex(
                    calibration.CacheValidationError, "input fingerprints"
                ):
                    calibration._accept_or_upgrade_cache(
                        fixture.receipt_path,
                        root=fixture.root,
                        cfg=fixture.config,
                        specs=fixture.specs,
                        input_fingerprints=fixture.inputs,
                    )

    def test_search_only_config_change_does_not_invalidate_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = CacheFixture(Path(directory))
            sealed = fixture.seal()
            old_projection_hash = sealed["input_fingerprints"]["config"][
                "projection_sha256"
            ]
            old_full_hash = sealed["full_config_context"]["sha256"]

            changed = json.loads(json.dumps(fixture.config))
            changed["search"]["direct_sample_pool_k"] = 512
            fixture.config_path.write_text(
                json.dumps(changed, sort_keys=True) + "\n", encoding="utf-8"
            )
            current_inputs = calibration._input_fingerprints(
                root=fixture.root,
                config_path=fixture.config_path,
                candidate_path=fixture.candidate_path,
            )

            self.assertEqual(
                current_inputs["config"]["projection_sha256"], old_projection_hash
            )
            self.assertNotEqual(
                calibration._sha256_file(fixture.config_path), old_full_hash
            )
            self.assertEqual(
                calibration._accept_or_upgrade_cache(
                    fixture.receipt_path,
                    root=fixture.root,
                    cfg=changed,
                    specs=fixture.specs,
                    input_fingerprints=current_inputs,
                ),
                "valid",
            )
            stored = json.loads(fixture.receipt_path.read_text(encoding="utf-8"))
            self.assertFalse(
                stored["full_config_context"]["validates_calibration_cache"]
            )

    def test_judge_config_change_invalidates_calibration_projection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = CacheFixture(Path(directory))
            fixture.seal()
            changed = json.loads(json.dumps(fixture.config))
            changed["judge"]["thinking_budget"] = 512
            fixture.config_path.write_text(
                json.dumps(changed, sort_keys=True) + "\n", encoding="utf-8"
            )
            current_inputs = calibration._input_fingerprints(
                root=fixture.root,
                config_path=fixture.config_path,
                candidate_path=fixture.candidate_path,
            )

            with self.assertRaisesRegex(
                calibration.CacheValidationError, "input fingerprints"
            ):
                calibration._accept_or_upgrade_cache(
                    fixture.receipt_path,
                    root=fixture.root,
                    cfg=changed,
                    specs=fixture.specs,
                    input_fingerprints=current_inputs,
                )

    def test_output_hash_order_ids_and_count_are_checked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = CacheFixture(Path(directory))
            fixture.seal()
            thinking = next(spec for spec in fixture.specs if spec.tag == "thinking")
            rows = [
                {"id": record_id, "p_viable": 0.9}
                for record_id in reversed(thinking.expected_ids)
            ]
            _write_jsonl(thinking.path, rows)

            with self.assertRaisesRegex(
                calibration.CacheValidationError, "IDs/count"
            ):
                calibration._accept_or_upgrade_cache(
                    fixture.receipt_path,
                    root=fixture.root,
                    cfg=fixture.config,
                    specs=fixture.specs,
                    input_fingerprints=fixture.inputs,
                )

        with tempfile.TemporaryDirectory() as directory:
            fixture = CacheFixture(Path(directory))
            fixture.seal()
            nextop = next(spec for spec in fixture.specs if spec.tag == "nextop")
            original = nextop.path.read_text(encoding="utf-8")
            nextop.path.write_text(
                original.replace('"p_viable": 0.5', '"p_viable": 0.6'),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                calibration.CacheValidationError, "fingerprints changed"
            ):
                calibration._accept_or_upgrade_cache(
                    fixture.receipt_path,
                    root=fixture.root,
                    cfg=fixture.config,
                    specs=fixture.specs,
                    input_fingerprints=fixture.inputs,
                )

    def test_engine_and_scoring_plan_are_checked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = CacheFixture(Path(directory))
            receipt = fixture.seal()
            receipt["engine_config"]["max_num_seqs"] = 31
            calibration._atomic_write_json(fixture.receipt_path, receipt)
            with self.assertRaisesRegex(
                calibration.CacheValidationError, "engine_config mismatch"
            ):
                calibration._accept_or_upgrade_cache(
                    fixture.receipt_path,
                    root=fixture.root,
                    cfg=fixture.config,
                    specs=fixture.specs,
                    input_fingerprints=fixture.inputs,
                )

        with tempfile.TemporaryDirectory() as directory:
            fixture = CacheFixture(Path(directory))
            receipt = fixture.seal()
            receipt["scoring_plan"]["thinking"]["run_seed"] += 1
            calibration._atomic_write_json(fixture.receipt_path, receipt)
            with self.assertRaisesRegex(
                calibration.CacheValidationError, "scoring plan"
            ):
                calibration._accept_or_upgrade_cache(
                    fixture.receipt_path,
                    root=fixture.root,
                    cfg=fixture.config,
                    specs=fixture.specs,
                    input_fingerprints=fixture.inputs,
                )


class LegacyUpgradeTests(unittest.TestCase):
    def test_valid_legacy_receipt_upgrades_without_generation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = CacheFixture(Path(directory))
            legacy = fixture.base_receipt(schema_version=1)
            calibration._atomic_write_json(fixture.receipt_path, legacy)
            old_hash = calibration._sha256_file(fixture.receipt_path)
            # This mirrors the real transition: the legacy process loaded K=320,
            # then only the search namespace changed while it was still running.
            changed = json.loads(json.dumps(fixture.config))
            changed["search"]["direct_sample_pool_k"] = 512
            fixture.config_path.write_text(
                json.dumps(changed, sort_keys=True) + "\n", encoding="utf-8"
            )
            current_inputs = calibration._input_fingerprints(
                root=fixture.root,
                config_path=fixture.config_path,
                candidate_path=fixture.candidate_path,
            )

            state = calibration._accept_or_upgrade_cache(
                fixture.receipt_path,
                root=fixture.root,
                cfg=changed,
                specs=fixture.specs,
                input_fingerprints=current_inputs,
            )

            self.assertEqual(state, "upgraded")
            upgraded = json.loads(fixture.receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(upgraded["schema_version"], 2)
            self.assertEqual(
                upgraded["legacy_upgrade"]["legacy_receipt_sha256"], old_hash
            )
            self.assertEqual(
                calibration._accept_or_upgrade_cache(
                    fixture.receipt_path,
                    root=fixture.root,
                    cfg=changed,
                    specs=fixture.specs,
                    input_fingerprints=current_inputs,
                ),
                "valid",
            )

    def test_legacy_upgrade_fails_closed_on_missing_canary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = CacheFixture(Path(directory))
            legacy = fixture.base_receipt(schema_version=1)
            calibration._atomic_write_json(fixture.receipt_path, legacy)
            canary = next(
                spec for spec in fixture.specs if spec.tag == "task_shuffled_thinking"
            )
            canary.path.unlink()

            with self.assertRaisesRegex(
                calibration.CacheValidationError, "required output is missing"
            ):
                calibration._accept_or_upgrade_cache(
                    fixture.receipt_path,
                    root=fixture.root,
                    cfg=fixture.config,
                    specs=fixture.specs,
                    input_fingerprints=fixture.inputs,
                )
            self.assertEqual(
                json.loads(fixture.receipt_path.read_text(encoding="utf-8"))[
                    "schema_version"
                ],
                1,
            )


class MainCacheFlowTests(unittest.TestCase):
    def test_valid_cache_returns_before_runner_construction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = CacheFixture(Path(directory))
            fixture.seal()

            class ForbiddenRunner:
                def __init__(self, *_args: object, **_kwargs: object) -> None:
                    raise AssertionError("valid cache must not construct VLLMRunner")

            output = io.StringIO()
            with mock.patch.object(calibration, "EXP", fixture.root), mock.patch.object(
                calibration.C, "load_config", return_value=fixture.config
            ), mock.patch.object(
                calibration, "VLLMRunner", ForbiddenRunner
            ), contextlib.redirect_stdout(output):
                result = calibration.main([])

            self.assertEqual(result, 0)
            self.assertIn("cached (valid receipt)", output.getvalue())

    def test_explicit_legacy_upgrade_returns_before_runner_construction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = CacheFixture(Path(directory))
            calibration._atomic_write_json(
                fixture.receipt_path, fixture.base_receipt(schema_version=1)
            )

            class ForbiddenRunner:
                def __init__(self, *_args: object, **_kwargs: object) -> None:
                    raise AssertionError("receipt upgrade must not construct VLLMRunner")

            output = io.StringIO()
            with mock.patch.object(calibration, "EXP", fixture.root), mock.patch.object(
                calibration.C, "load_config", return_value=fixture.config
            ), mock.patch.object(
                calibration, "VLLMRunner", ForbiddenRunner
            ), contextlib.redirect_stdout(output):
                result = calibration.main(["--upgrade-receipt"])

            self.assertEqual(result, 0)
            self.assertIn("cached (upgraded receipt)", output.getvalue())
            self.assertEqual(
                json.loads(fixture.receipt_path.read_text(encoding="utf-8"))[
                    "schema_version"
                ],
                2,
            )

    def test_orphaned_outputs_fail_before_runner_construction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = CacheFixture(Path(directory))

            class ForbiddenRunner:
                def __init__(self, *_args: object, **_kwargs: object) -> None:
                    raise AssertionError("orphan handling must not load the model")

            with mock.patch.object(calibration, "EXP", fixture.root), mock.patch.object(
                calibration.C, "load_config", return_value=fixture.config
            ), mock.patch.object(calibration, "VLLMRunner", ForbiddenRunner):
                with self.assertRaisesRegex(RuntimeError, "without a completion receipt"):
                    calibration.main([])


if __name__ == "__main__":
    unittest.main()
