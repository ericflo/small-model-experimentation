from __future__ import annotations

import json
import sys
import unittest
from collections import Counter
from pathlib import Path

import yaml


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

import task_data as td  # noqa: E402


class TaskDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = yaml.safe_load((EXP / "configs" / "default.yaml").read_text())

    def test_common_panel_and_exact_function_capacity_are_frozen(self) -> None:
        panel = td.build_common_panel(self.config)
        self.assertEqual(Counter(map(len, panel)), {4: 16, 5: 16, 6: 16, 7: 16, 8: 16})
        groups, shallow, _ = td.enumerate_exact_depth_three_functions(panel)
        self.assertEqual(len(shallow), 354)
        self.assertEqual(len(groups), 3525)
        live_counts = Counter(len({program[0] for program in rows}) for rows in groups.values())
        self.assertEqual(live_counts, {1: 1552, 2: 1601, 3: 323, 4: 49})

    def test_partial_semantics_returns_typed_invalid(self) -> None:
        state = td.apply_pipeline(
            [1, 2, 3, 4],
            (("take_k", 1), ("adjacent_diff", None), ("rotate_k", 1)),
        )
        self.assertIs(state, td.INVALID)
        self.assertIs(td.apply_pipeline([], (("reverse", None),)), td.INVALID)

    def test_frozen_artifacts_have_disjoint_function_triple_and_suffix(self) -> None:
        fingerprints: set[str] = set()
        triples: set[td.Program] = set()
        suffixes: set[td.Program] = set()
        expected = {"mechanics": 24, "qualification": 48, "confirmation": 192}
        for split, count in expected.items():
            rows = [
                json.loads(line)
                for line in (EXP / "data" / "procedural" / f"{split}_gold.jsonl")
                .read_text()
                .splitlines()
            ]
            audits = [
                json.loads(line)
                for line in (EXP / "data" / "procedural" / f"{split}_audit.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(len(rows), count)
            self.assertEqual(len(audits), count)
            blocks = count // 24
            self.assertEqual(
                Counter(row["stratum"] for row in audits),
                {"single": 8 * blocks, "double": 8 * blocks, "triple": 4 * blocks, "quad": 4 * blocks},
            )
            for row in rows:
                fingerprint = row["common_fingerprint"]
                triple = tuple(td.operation_from_record(value) for value in row["target_pipeline"])
                self.assertNotIn(fingerprint, fingerprints)
                self.assertNotIn(triple, triples)
                self.assertNotIn(triple[1:], suffixes)
                fingerprints.add(fingerprint)
                triples.add(triple)
                suffixes.add(triple[1:])
        self.assertEqual(len(fingerprints), 264)


if __name__ == "__main__":
    unittest.main()
