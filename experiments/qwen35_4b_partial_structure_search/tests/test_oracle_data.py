from __future__ import annotations

import copy
import importlib.util
import sys
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


F = _load("partial_structure_families", EXP / "src" / "families.py")
# oracle_data is an executable experiment script and intentionally imports the
# local module as `families`.  Bind that name only while loading the script.
_prior_families = sys.modules.get("families")
sys.modules["families"] = F
try:
    O = _load("partial_structure_oracle_data", EXP / "scripts" / "oracle_data.py")
finally:
    if _prior_families is None:
        sys.modules.pop("families", None)
    else:
        sys.modules["families"] = _prior_families


class ListDslTests(unittest.TestCase):
    def test_frozen_inventory_has_16_types_and_32_concrete_operations(self) -> None:
        self.assertEqual(len(F.TYPES), 16)
        self.assertEqual(len(set(F.TYPES)), 16)
        self.assertEqual(len(F.CONCRETE_OPS), 32)
        self.assertEqual(
            F.execute_pipeline(
                (("add_k", 2), ("reverse", None), ("take_k", 2)), [1, -3, 4]
            ),
            [6, -1],
        )

    def test_complete_skeleton_and_parameter_fill_enumeration_is_accounted(self) -> None:
        accounting = O.CpuAccounting()
        skeletons = list(O.enumerate_full_skeletons(2, accounting))
        self.assertEqual(len(skeletons), 16**2)
        self.assertEqual(accounting.full_skeletons_yielded, 16**2)
        self.assertEqual(skeletons[0], (F.TYPES[0], F.TYPES[0]))
        self.assertEqual(skeletons[-1], (F.TYPES[-1], F.TYPES[-1]))

        fills = list(O.enumerate_parameter_fills(("add_k", "mul_k"), accounting))
        self.assertEqual(len(fills), 6 * 3)
        self.assertEqual(accounting.parameter_fills_yielded, 18)
        self.assertEqual(len(set(fills)), 18)


class FreshTaskTests(unittest.TestCase):
    def test_generation_is_deterministic_fresh_split_and_exact_depth(self) -> None:
        kwargs = dict(
            task_id="fresh-d2",
            depth=2,
            seed=102,
            n_visible=2,
            n_label_probe=2,
            n_hidden=2,
            max_attempts=30,
        )
        first = F.generate_task(**kwargs)
        second = F.generate_task(**kwargs)
        self.assertEqual(first, second)
        self.assertEqual(first["depth"], 2)
        self.assertEqual(first["min_depth_audit"]["seen_cap"], None)
        self.assertFalse(first["min_depth_audit"]["within_limit"])
        self.assertTrue(first["min_depth_audit"]["exhaustive_decision"])

        cases = first["visible"] + first["label_probe"] + first["hidden"]
        inputs = [tuple(case["input"]) for case in cases]
        self.assertEqual(len(inputs), len(set(inputs)))
        self.assertEqual(
            F.normalize_pipeline(first["target_pipeline"]),
            tuple(F.normalize_op(operation) for operation in first["target_pipeline"]),
        )

    def test_shallower_equivalent_pipeline_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "not behaviorally exact depth 2"):
            F.build_task_from_pipeline(
                task_id="double-reverse",
                seed=9,
                pipeline=(("reverse", None), ("reverse", None)),
                visible_inputs=[[1, 2, 3]],
                label_probe_inputs=[[4, 1, -2, 7]],
                hidden_inputs=[[9, 0, 2, 5, 3]],
            )


class SemanticOracleTests(unittest.TestCase):
    @staticmethod
    def _alternative_factorization_task():
        # Both REVERSE -> SORT_ASC and SORT_DESC -> REVERSE compute ascending
        # sort, giving genuinely different live first-operation prefixes.
        return F.build_task_from_pipeline(
            task_id="alternative-factorizations",
            seed=1,
            pipeline=(("reverse", None), ("sort_asc", None)),
            visible_inputs=[[3, 1, 2]],
            label_probe_inputs=[[4, 2, 3, 1]],
            hidden_inputs=[[8, 5, 7, 6]],
            require_exact_depth=False,
        )

    def test_all_semantic_factorizations_make_their_prefixes_live(self) -> None:
        task = self._alternative_factorization_task()
        result = O.build_oracle_result(task)
        by_skeleton = {row.skeleton: row for row in result.successful_skeletons}
        self.assertIn(("reverse", "sort_asc"), by_skeleton)
        self.assertIn(("sort_desc", "reverse"), by_skeleton)
        self.assertGreater(len(by_skeleton), 2)  # The oracle is not target-only.

        by_prefix = {tuple(row["prefix"]): row for row in result.prefix_rows}
        self.assertTrue(by_prefix[("reverse",)]["live"])
        self.assertTrue(by_prefix[("sort_desc",)]["live"])
        self.assertGreaterEqual(
            by_prefix[("sort_desc",)]["completion_skeleton_count"], 1
        )
        self.assertGreaterEqual(
            by_prefix[("sort_desc",)]["completion_parameter_fill_count"], 1
        )
        self.assertEqual(len(result.prefix_rows), 1 + 16)
        self.assertEqual(result.accounting["prefix_rows_emitted"], 17)
        self.assertGreater(result.accounting["case_operation_applications"], 0)

    def test_default_label_oracle_does_not_read_hidden_outputs(self) -> None:
        task = self._alternative_factorization_task()
        corrupted_hidden = copy.deepcopy(task)
        corrupted_hidden["hidden"][0]["output"] = [999_999]
        clean = O.build_oracle_result(task)
        corrupted = O.build_oracle_result(corrupted_hidden)
        self.assertEqual(clean.successful_skeletons, corrupted.successful_skeletons)
        self.assertEqual(clean.prefix_rows, corrupted.prefix_rows)
        self.assertNotIn("hidden", clean.label_source_splits)
        self.assertTrue(
            all(not row["hidden_cases_used_for_label"] for row in clean.prefix_rows)
        )


class ExactDepthFiveAuditTests(unittest.TestCase):
    def test_exact_d5_task_rejects_every_pipeline_through_d4_without_cap(self) -> None:
        # This fixed procedural instance was found by seeded DSL generation.  The
        # assertion is not based on that provenance: build_task_from_pipeline
        # independently exhausts every distinct behavior reachable through d4.
        pipeline = (
            ("dedup_adjacent", None),
            ("running_sum", None),
            ("rotate_k", 2),
            ("rotate_k", 3),
            ("mod_k", 2),
        )
        task = F.build_task_from_pipeline(
            task_id="exact-d5-rejection",
            seed=1,
            pipeline=pipeline,
            visible_inputs=[[6, -9, 3, 4, -9]],
            label_probe_inputs=[[-1, -2, 9, -6, 1, -9, -9, -9]],
            hidden_inputs=[[3, -3, 4, -9, 7]],
        )
        audit = task["min_depth_audit"]
        self.assertEqual(audit["algorithm"], "uncapped_behavioral_bfs")
        self.assertIsNone(audit["seen_cap"])
        self.assertIsNone(audit["found_depth"])
        self.assertEqual(audit["levels_fully_exhausted"], [1, 2, 3, 4])
        self.assertTrue(audit["exhaustive_decision"])
        self.assertGreater(audit["transitions_considered"], 16**4)
        inputs, outputs = F.task_cases(task)
        self.assertTrue(F.pipeline_solves(pipeline, inputs, outputs))


if __name__ == "__main__":
    unittest.main()

