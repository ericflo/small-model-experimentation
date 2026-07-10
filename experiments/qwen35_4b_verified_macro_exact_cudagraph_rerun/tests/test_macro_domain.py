from __future__ import annotations

import sys
import unittest
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import macro_domain as domain  # noqa: E402


class ExecutionTests(unittest.TestCase):
    def test_uppercase_inventory_and_exact_execution(self) -> None:
        self.assertTrue(all(token.isupper() for token in domain.PRIMITIVES))
        xs = [1, 2, 3, 4, 5]
        self.assertEqual(
            domain.execute_program(("REV", "ADD1"), xs),
            [6, 5, 4, 3, 2],
        )
        self.assertEqual(
            domain.execute_program(("PREFIX", "DIFF"), xs),
            xs,
        )

    def test_depth_index_exhausts_every_shorter_depth(self) -> None:
        index = domain.BehavioralDepthIndex(max_depth=4)
        program = ("REV", "ADD1", "ROTL", "PREFIX", "SWAP")
        minimum = index.minimum_depth(program)
        self.assertTrue(minimum is None or 0 <= minimum <= 5)
        self.assertFalse(index.verify_exact_depth(("PREFIX", "DIFF"), 2))


class MacroTests(unittest.TestCase):
    def test_expand_compress_and_non_degenerate_verification(self) -> None:
        macros = (domain.Macro("M0", ("REV", "ADD1"), 12),)
        base = ("REV", "ADD1", "SWAP")
        compressed = domain.compress_program(base, macros)
        self.assertEqual(compressed, ("M0", "SWAP"))
        self.assertEqual(domain.expand_program(compressed, macros), base)
        self.assertTrue(domain.verify_macro(("REV", "ADD1")).nondegenerate)
        self.assertFalse(domain.verify_macro(("PREFIX", "DIFF")).nondegenerate)

    def test_mining_and_placebo_are_deterministic(self) -> None:
        programs = [
            ("REV", "ADD1", "SWAP", "MUL2"),
            ("NEG", "REV", "ADD1", "SORT"),
            ("ROTL", "PREFIX", "REV", "ADD1"),
            ("SWAP", "MUL2", "REV", "ADD1"),
            ("DIFF", "REV", "ADD1", "ZIGZAG"),
            ("SORT", "ROTL", "PREFIX", "REV"),
        ] * 3
        mined = domain.mine_frequent_macros(programs, count=3, min_support=2)
        self.assertEqual(mined, domain.mine_frequent_macros(programs, count=3, min_support=2))
        placebo = domain.make_frequency_matched_random_macros(
            programs, mined, seed=91, exclude_expansions=[macro.expansion for macro in mined]
        )
        self.assertEqual(len(placebo), len(mined))
        self.assertEqual(
            placebo,
            domain.make_frequency_matched_random_macros(
                programs, mined, seed=91, exclude_expansions=[macro.expansion for macro in mined]
            ),
        )


class DatasetTests(unittest.TestCase):
    def test_small_dataset_is_fresh_paired_and_min_depth(self) -> None:
        dataset = domain.generate_task_dataset(
            seed=1234,
            train_programs=12,
            smoke_tasks_per_split=2,
            full_reuse_tasks=4,
            full_no_reuse_tasks=2,
            visible_examples=2,
            hidden_examples=2,
            probe_inputs=2,
        )
        verdict = domain.validate_task_dataset(dataset)
        self.assertEqual(verdict["unique_programs"], verdict["n_tasks"])
        reuse = dataset.by_split("reuse")
        no_reuse = dataset.by_split("no_reuse")
        self.assertEqual(len(reuse), 4)
        self.assertEqual(len(no_reuse), 2)
        lookup = {task.id: task for task in dataset.tasks}
        for task in no_reuse:
            source = lookup[task.paired_task_id]
            self.assertEqual(sorted(task.program), sorted(source.program))
            self.assertFalse(domain.motif_occurrences(task.program, domain.REUSABLE_MOTIFS))
            self.assertEqual(task.min_depth, 5)

    def test_repair_smoke_excludes_every_frozen_program_and_signature(self) -> None:
        frozen = domain.generate_task_dataset(
            seed=4321,
            train_programs=12,
            smoke_tasks_per_split=2,
            full_reuse_tasks=4,
            full_no_reuse_tasks=2,
            visible_examples=2,
            hidden_examples=2,
            probe_inputs=2,
        )
        repaired = domain.generate_fresh_smoke_tasks(
            exclude_tasks=frozen.tasks,
            seed=9876,
            tasks_per_split=2,
            visible_examples=2,
            hidden_examples=2,
            probe_inputs=2,
        )
        self.assertEqual(len(repaired), 4)
        self.assertTrue(all(task.id.startswith("smoke-v2-") for task in repaired))
        self.assertFalse({task.program for task in repaired} & {task.program for task in frozen.tasks})
        self.assertFalse({task.signature for task in repaired} & {task.signature for task in frozen.tasks})
        combined = domain.TaskDataset(
            tasks=tuple((*frozen.tasks, *repaired)),
            signature_probes=frozen.signature_probes,
            reusable_motifs=frozen.reusable_motifs,
            decoy_motifs=frozen.decoy_motifs,
            seed=frozen.seed,
        )
        domain.validate_task_dataset(combined)


if __name__ == "__main__":
    unittest.main()
