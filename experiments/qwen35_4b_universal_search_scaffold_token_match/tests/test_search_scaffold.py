from __future__ import annotations

import importlib.util
import sys
import unittest
from collections import Counter
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "gen_search_scaffold.py"
SPEC = importlib.util.spec_from_file_location("search_scaffold_generator", SCRIPT)
assert SPEC and SPEC.loader
generator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = generator
SPEC.loader.exec_module(generator)


class SearchScaffoldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = generator.generate()

    def surface(self, row: dict) -> generator.Surface:
        audit = row["_audit"]
        return generator.Surface(
            row["surface"], tuple(audit["surface_items"]), audit["separator"]
        )

    def operation(self, value: list) -> tuple:
        return tuple(value)

    def sequences(self, values: list[list[int]]) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(value) for value in values)

    def test_default_mix_and_surface_diversity_are_exact(self) -> None:
        self.assertEqual(len(self.rows), 80)
        self.assertEqual(
            Counter(row["kind"] for row in self.rows),
            Counter({f"u_scaffold_{stage}": 16 for stage in generator.STAGES}),
        )
        self.assertGreaterEqual(len({row["surface"] for row in self.rows}), 5)
        self.assertEqual(len({row["task_id"] for row in self.rows}), 80)

    def test_generation_is_byte_deterministic(self) -> None:
        first = generator.render_rows(generator.generate())
        second = generator.render_rows(generator.generate())
        self.assertEqual(first, second)
        self.assertEqual(generator.sha256_bytes(first), "5854c218479a500f969bf2dbcfdbc30cd8a6095fa38aeaa652a220219b50a093")

    def test_every_case_has_one_true_pair_and_one_dead_first(self) -> None:
        for row in self.rows:
            with self.subTest(task_id=row["task_id"]):
                audit = row["_audit"]
                probes = self.sequences(audit["probes"])
                outputs = self.sequences(audit["outputs"])
                first = self.operation(audit["first"])
                second = self.operation(audit["second"])
                dead = self.operation(audit["dead_first"])
                pairs = generator.fitting_pairs(
                    probes, outputs, audit["length"], len(audit["surface_items"])
                )
                self.assertEqual(pairs, ((first, second),))
                self.assertEqual(
                    generator.fitting_seconds(
                        dead, probes, outputs, audit["length"], len(audit["surface_items"])
                    ),
                    (),
                )

    def test_stage_answers_recompute_from_executable_state(self) -> None:
        for row in self.rows:
            with self.subTest(task_id=row["task_id"]):
                audit = row["_audit"]
                surface = self.surface(row)
                size = len(surface.items)
                stage = audit["stage"]
                expected = row["answer"].removeprefix("ANSWER: ")
                first = self.operation(audit["first"])
                second = self.operation(audit["second"])
                probes = self.sequences(audit["probes"])
                outputs = self.sequences(audit["outputs"])
                if stage == "apply":
                    derived = " || ".join(
                        generator.render(generator.apply_op(first, probe, size), surface)
                        for probe in probes[:3]
                    )
                elif stage == "fit":
                    derived = generator.op_code(second, surface)
                elif stage == "reject":
                    candidate = self.operation(audit["candidate_first"])
                    seconds = generator.fitting_seconds(
                        candidate, probes, outputs, audit["length"], size
                    )
                    derived = (
                        f"FIT | {generator.op_code(seconds[0], surface)}"
                        if seconds
                        else "NO_FIT"
                    )
                else:
                    query = tuple(audit["query"])
                    derived = generator.render(
                        generator.apply_op(second, generator.apply_op(first, query, size), size),
                        surface,
                    )
                self.assertEqual(expected, derived)
                self.assertEqual(audit["expected"], derived)

    def test_reject_stage_balances_fit_and_no_fit(self) -> None:
        answers = [
            row["answer"].removeprefix("ANSWER: ")
            for row in self.rows
            if row["kind"] == "u_scaffold_reject"
        ]
        self.assertEqual(sum(answer == "NO_FIT" for answer in answers), 8)
        self.assertEqual(sum(answer.startswith("FIT | ") for answer in answers), 8)

    def test_all_thoughts_are_bounded_and_commit(self) -> None:
        for row in self.rows:
            with self.subTest(task_id=row["task_id"]):
                self.assertTrue(row["think"].endswith("COMMIT."))
                self.assertNotIn("\n", row["answer"])
                self.assertTrue(row["answer"].startswith("ANSWER: "))
                self.assertLess(len(row["think"]), 2500)
                self.assertTrue(row["_audit"]["truth_valid"])
        searches = [row for row in self.rows if row["kind"] == "u_scaffold_search"]
        self.assertTrue(all("LEDGER 1/2" in row["think"] for row in searches))
        self.assertTrue(all("LEDGER 2/2" in row["think"] for row in searches))


if __name__ == "__main__":
    unittest.main()
