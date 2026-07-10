"""Adversarial model-free tests for preregistration Amendment 11."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


EXP = Path(__file__).resolve().parents[1]
SRC = EXP / "src"
sys.path.insert(0, str(SRC))

import full_artifacts as full_store  # noqa: E402


RUN_SPEC = importlib.util.spec_from_file_location(
    "verified_macro_amendment11_run_test", EXP / "scripts" / "run.py"
)
assert RUN_SPEC is not None and RUN_SPEC.loader is not None
run = importlib.util.module_from_spec(RUN_SPEC)
sys.modules[RUN_SPEC.name] = run
RUN_SPEC.loader.exec_module(run)

FIXTURE_SPEC = importlib.util.spec_from_file_location(
    "verified_macro_amendment11_fixture_test", Path(__file__).with_name("test_full_artifacts.py")
)
assert FIXTURE_SPEC is not None and FIXTURE_SPEC.loader is not None
fixtures = importlib.util.module_from_spec(FIXTURE_SPEC)
sys.modules[FIXTURE_SPEC.name] = fixtures
FIXTURE_SPEC.loader.exec_module(fixtures)


def full_tasks() -> list[dict[str, str]]:
    return [
        *({"id": f"no_{index:03d}", "split": "no_reuse"} for index in range(40)),
        *({"id": f"reuse_{index:03d}", "split": "reuse"} for index in range(80)),
    ]


def non_qwen_libraries() -> dict[str, dict[str, object]]:
    return {arm: {} for arm in full_store.NON_QWEN_ARMS}


def rejected_tier(plan: dict[str, object], budget: int) -> dict[str, object]:
    arms = list(plan["arm_order"])
    per_arm: dict[str, object] = {
        "base": {
            "status": "irreversibly_rejected",
            "complete": False,
            "adequate": False,
            "termination": {
                "samples": 144,
                "unresolved_cap_contacts": 144,
                "answer_limit_contacts": 0,
                "periodic_loop_contacts": 0,
            },
            "shards": [],
        }
    }
    per_arm.update(
        {
            arm: {
                "status": "skipped",
                "complete": False,
                "adequate": False,
                "shards": [],
            }
            for arm in arms[1:]
        }
    )
    return {
        "budget": budget,
        "status": "irreversibly_rejected",
        "complete": False,
        "adequate": False,
        "rejecting_arm": "base",
        "arms": per_arm,
    }


def adequate_tier(plan: dict[str, object], budget: int) -> dict[str, object]:
    return {
        "budget": budget,
        "status": "selectable",
        "complete": True,
        "adequate": True,
        "rejecting_arm": None,
        "arms": {
            arm: {
                "status": "complete",
                "complete": True,
                "adequate": True,
                "shards": [],
            }
            for arm in plan["arm_order"]
        },
    }


def selection(
    plan: dict[str, object], tiers: list[dict[str, object]], selected: int | None
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "run": "full",
        "pass": selected is not None,
        "selected_thinking_budget": selected,
        "starting_thinking_budget": 16384,
        "passed_smoke_selection": {
            "path": "analysis/smoke_budget_selection.json",
            "sha256": "a" * 64,
        },
        "tiers": tiers,
    }


class BindingAndSelectionTests(unittest.TestCase):
    def test_exact_binding_detects_hidden_label_only_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            exp = Path(directory) / "experiment"
            for relative in full_store.FULL_BINDING_PATHS:
                path = exp / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((relative + "\n").encode())
            frozen = full_store.build_full_binding(exp)
            tasks_path = exp / "data" / "tasks.json"
            tasks_path.write_text('{"hidden":["mutated"]}\n', encoding="utf-8")
            with self.assertRaisesRegex(full_store.FullArtifactError, "binding drift"):
                full_store.require_full_binding(exp, frozen)

    def test_selection_requires_registered_contiguous_prefix_and_first_adequate(self) -> None:
        config = run.load_config()
        plan = full_store.build_shard_plan(
            full_tasks(), list(full_store.NON_QWEN_ARMS)
        )
        valid = selection(
            plan,
            [rejected_tier(plan, 16384), adequate_tier(plan, 32768)],
            32768,
        )
        full_store.validate_budget_selection(
            valid,
            plan=plan,
            ladder=[16384, 32768, 49152, 61440],
            full_run=config["full_run"],
            expected_starting_budget=16384,
            expected_smoke_selection_sha256="a" * 64,
            final=True,
        )
        for budgets in ([32768, 16384], [16384, 49152], [16384, 16384]):
            bad = copy.deepcopy(valid)
            bad["tiers"] = [
                rejected_tier(plan, budget) for budget in budgets[:-1]
            ] + [adequate_tier(plan, budgets[-1])]
            bad["selected_thinking_budget"] = budgets[-1]
            with self.subTest(budgets=budgets), self.assertRaisesRegex(
                full_store.FullArtifactError, "skipped, reordered, duplicated"
            ):
                full_store.validate_budget_selection(
                    bad,
                    plan=plan,
                    ladder=[16384, 32768, 49152, 61440],
                    full_run=config["full_run"],
                    expected_starting_budget=16384,
                    expected_smoke_selection_sha256="a" * 64,
                    final=True,
                )

    def test_all_rejected_selection_is_final_and_selected_null(self) -> None:
        config = run.load_config()
        plan = full_store.build_shard_plan(
            full_tasks(), list(full_store.NON_QWEN_ARMS)
        )
        all_rejected = selection(
            plan,
            [rejected_tier(plan, budget) for budget in (16384, 32768, 49152, 61440)],
            None,
        )
        verified = full_store.validate_budget_selection(
            all_rejected,
            plan=plan,
            ladder=[16384, 32768, 49152, 61440],
            full_run=config["full_run"],
            expected_starting_budget=16384,
            expected_smoke_selection_sha256="a" * 64,
            final=True,
        )
        self.assertIsNone(verified["selected_thinking_budget"])
        self.assertFalse(verified["pass"])
        with tempfile.TemporaryDirectory() as directory:
            exp = Path(directory) / "experiment"
            analysis = exp / "analysis"
            analysis.mkdir(parents=True)
            (analysis / "full_shard_plan.json").write_text(
                json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            (analysis / "full_budget_selection.json").write_text(
                json.dumps(all_rejected, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(full_store, "require_full_binding"):
                catalog = full_store.build_full_catalog(
                    exp=exp,
                    root=Path(directory) / "external",
                    plan=plan,
                    budgets=[16384, 32768, 49152, 61440],
                    protocol_binding={"binding": "fixed"},
                    status="setup_inconclusive",
                    starting_budget=16384,
                    smoke_selection_sha256="a" * 64,
                    selection=all_rejected,
                )
            self.assertEqual(catalog["schema_version"], 2)
            self.assertEqual(catalog["status"], "setup_inconclusive")
            self.assertIsNone(catalog["selected_tier"])
            self.assertEqual(catalog["selected_shards"], [])
            self.assertIsNotNone(catalog["budget_selection"])


class LockInventoryAndArmTests(unittest.TestCase):
    def test_final_rename_without_catalog_is_reconciled_into_inventory(self) -> None:
        plan = {
            "schema_version": 1,
            "protocol": "test-plan",
            "arm_order": ["base"],
            "arms": {
                "base": {
                    "k": 2,
                    "shard_count": 1,
                    "shards": [
                        {
                            "shard_index": 0,
                            "task_ids": ["t0", "t1"],
                            "record_ids": ["t0::base", "t1::base"],
                            "k": 2,
                        }
                    ],
                }
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            exp = Path(directory) / "experiment"
            analysis = exp / "analysis"
            analysis.mkdir(parents=True)
            (analysis / "full_shard_plan.json").write_text(
                json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            root = Path(directory) / "external"
            fixtures.write_complete_shard(
                root,
                plan_hash=full_store.plan_sha256(plan),
                task_ids=("t0", "t1"),
                k=2,
            )
            self.assertFalse((analysis / "full_artifact_catalog.json").exists())
            with mock.patch.object(full_store, "require_full_binding"):
                catalog = full_store.build_full_catalog(
                    exp=exp,
                    root=root,
                    plan=plan,
                    budgets=[32768],
                    protocol_binding={"binding": "fixed"},
                    status="in_progress",
                    starting_budget=32768,
                    smoke_selection_sha256="a" * 64,
                    selection=None,
                )
            self.assertEqual(len(catalog["completed_shards"]), 1)
            self.assertEqual(
                catalog["completed_shards"][0]["relative_path"],
                "think_32768/base/shard_000",
            )

    def test_exact_nine_and_fifteen_arm_full_geometry(self) -> None:
        nine = non_qwen_libraries()
        fifteen = copy.deepcopy(nine)
        for arm in full_store.QWEN_ARMS:
            fifteen[arm] = {"macros": [f"macro-{index}" for index in range(8)]}
        for libraries, expected_arms, expected_shards in (
            (nine, 9, 100),
            (fifteen, 15, 160),
        ):
            with self.subTest(expected_arms=expected_arms):
                order = full_store.validate_full_arm_set(libraries)
                plan = full_store.build_shard_plan(full_tasks(), order)
                self.assertEqual(len(order), expected_arms)
                self.assertEqual(
                    sum(int(plan["arms"][arm]["shard_count"]) for arm in order),
                    expected_shards,
                )
                self.assertEqual(plan["arms"]["base"]["shard_count"], 20)
                self.assertTrue(
                    all(
                        shard["completions"] == 144
                        for arm in order
                        for shard in plan["arms"][arm]["shards"]
                    )
                )

    def test_duplicate_full_invocation_lock_is_nonblocking_and_persistent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "external"
            lock_path = root.parent / f".{root.name}.lock"
            with full_store.full_stage_lock(root):
                self.assertTrue(lock_path.is_file())
                with self.assertRaisesRegex(full_store.FullArtifactError, "already holds"):
                    with full_store.full_stage_lock(root):
                        self.fail("nested invocation acquired the same full lock")
            self.assertTrue(lock_path.is_file())
            with full_store.full_stage_lock(root):
                pass

    def test_shared_lock_blocks_smoke_full_and_migration_before_internal_work(self) -> None:
        config = copy.deepcopy(run.load_config())
        with tempfile.TemporaryDirectory() as directory:
            coordination_root = Path(directory) / "full"
            config["full_run"]["external_root"] = str(coordination_root)
            locked_body = mock.Mock()
            locked_migration = mock.Mock()
            with full_store.experiment_stage_lock(coordination_root):
                with mock.patch.object(run, "_run_model_stage_locked", locked_body):
                    for stage in ("smoke", "full"):
                        with self.subTest(stage=stage):
                            with self.assertRaisesRegex(
                                full_store.FullArtifactError, "already holds"
                            ):
                                run.run_model_stage(stage, config)
                with mock.patch.object(
                    run, "_migrate_scientific_artifacts_locked", locked_migration
                ):
                    with self.assertRaisesRegex(
                        full_store.FullArtifactError, "already holds"
                    ):
                        run.migrate_scientific_artifacts(config)
            locked_body.assert_not_called()
            locked_migration.assert_not_called()

    def test_shared_lock_rejects_symlink_without_internal_work(self) -> None:
        if not hasattr(Path, "symlink_to"):
            self.skipTest("symlinks unavailable")
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "full"
            target = parent / "target"
            target.write_text("not a lock\n", encoding="utf-8")
            (parent / ".full.lock").symlink_to(target)
            with self.assertRaisesRegex(
                full_store.FullArtifactError, "cannot safely open"
            ):
                with full_store.experiment_stage_lock(root):
                    self.fail("symlinked coordination lock was acquired")

    def test_inventory_rejects_unknown_and_symlinked_entries(self) -> None:
        plan = full_store.build_shard_plan(
            full_tasks(), list(full_store.NON_QWEN_ARMS)
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "external"
            root.mkdir()
            (root / "mystery").mkdir()
            with self.assertRaisesRegex(full_store.FullArtifactError, "unknown full root entry"):
                full_store.inventory_full_root(
                    root, plan=plan, budgets=[16384, 32768, 49152, 61440]
                )
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            target = parent / "target"
            target.mkdir()
            root = parent / "external"
            root.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(full_store.FullArtifactError, "symlink"):
                full_store.inventory_full_root(
                    root, plan=plan, budgets=[16384, 32768, 49152, 61440]
                )

    def test_local_raw_and_partial_qwen_arm_sets_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            local = Path(directory) / "runs" / "full"
            local.mkdir(parents=True)
            (local / "stale.jsonl").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(full_store.FullArtifactError, "raw rows"):
                full_store.reject_repository_local_full_raw(local)
        libraries = non_qwen_libraries()
        libraries["qwen_ranked"] = {"macros": [str(index) for index in range(8)]}
        with self.assertRaisesRegex(full_store.FullArtifactError, "exactly the nine"):
            full_store.validate_full_arm_set(libraries)


class TwoPassTests(unittest.TestCase):
    def _exercise_validation_error(self, error: Exception) -> tuple[mock.Mock, mock.Mock]:
        config = copy.deepcopy(run.load_config())
        tasks = full_tasks()

        def records(**kwargs: object) -> list[dict[str, object]]:
            arm = str(kwargs["arm"])
            return [
                {"id": f"{task['id']}::{arm}"}
                for task in kwargs["tasks"]  # type: ignore[union-attr]
            ]

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name) / "external"
        analysis = Path(temporary.name) / "analysis"
        analysis.mkdir()
        (analysis / "smoke_budget_selection.json").write_text("{}\n", encoding="utf-8")
        config["full_run"]["external_root"] = str(root)
        validate = mock.Mock(side_effect=[None, error])
        generate = mock.Mock()
        with (
            mock.patch.object(run, "ANALYSIS", analysis),
            mock.patch.object(run, "_solver_records", side_effect=records),
            mock.patch.object(run, "_solver_sampling", return_value=types.SimpleNamespace()),
            mock.patch.object(run, "_current_full_protocol_identity", return_value={"fixed": True}),
            mock.patch.object(run, "_validate_cached_full_shard", validate),
            mock.patch.object(run, "_generate_atomic_full_shard", generate),
            mock.patch.object(run.full_store, "build_full_binding", return_value={"fixed": True}),
            mock.patch.object(run, "_write_full_artifact_catalog"),
        ):
            with self.assertRaises(type(error)):
                run._run_full_scientific_stage(
                    runner=object(),
                    harness=object(),
                    domain=object(),
                    config=config,
                    tasks=tasks,
                    libraries=non_qwen_libraries(),
                    demonstrations=[],
                    starting_budget=32768,
                )
        return validate, generate

    def test_missing_shard_zero_then_malformed_shard_one_makes_zero_model_calls(self) -> None:
        validate, generate = self._exercise_validation_error(
            full_store.FullArtifactError("malformed downstream shard")
        )
        self.assertEqual(validate.call_count, 2)
        generate.assert_not_called()

    def test_operational_error_does_not_escalate_budget(self) -> None:
        validate, generate = self._exercise_validation_error(
            OSError("external filesystem unavailable")
        )
        self.assertEqual(validate.call_count, 2)
        generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
