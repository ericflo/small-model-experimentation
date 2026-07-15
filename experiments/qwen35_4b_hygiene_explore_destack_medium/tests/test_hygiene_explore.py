import json
import random
import re
import sys
import unittest
from collections import Counter, defaultdict
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))

import gen_axis_v2 as axis  # noqa: E402
import gen_curriculum as original  # noqa: E402


TREATMENT_MIX = "hygiene=40,explore=40"
CONSTRUCTION_SEED = 77119


def answer_of(row: dict) -> str:
    assert row["answer"].startswith("ANSWER: ")
    return row["answer"].removeprefix("ANSWER: ")


def prompt_of(row: dict) -> str:
    return row["messages"][0]["content"]


class GenerationValidityTests(unittest.TestCase):
    def test_destacked_mix_generation_valid_and_leak_free(self) -> None:
        rows = axis.generate_curriculum(TREATMENT_MIX, 12345)
        summary = axis.validate_generated(rows)
        axis.check_banned_vocabulary(rows)
        self.assertEqual(summary["rows"], 80)
        self.assertEqual(
            set(summary["kinds"]),
            {"u_explore", "u_hygiene"},
        )

    def test_banned_vocabulary_scan_actually_fires(self) -> None:
        rows = axis.generate_curriculum(TREATMENT_MIX, 54321)
        poisoned = json.loads(json.dumps(axis.public_row(rows[0])))
        poisoned["_audit"] = rows[0]["_audit"]
        poisoned["messages"][0]["content"] += " the warren gate"
        with self.assertRaises(ValueError):
            axis.check_banned_vocabulary([poisoned])

    def test_banned_vocabulary_disjoint_from_original_surfaces(self) -> None:
        original_tokens = {
            item for pool in original.SURFACE_POOLS.values() for item in pool
        }
        axis_vocab = set(axis.TREES) | set(axis.RIVERS) | set(axis.MOONS)
        self.assertFalse(axis_vocab & original_tokens)

    def test_frozen_corpus_regenerates_byte_identically(self) -> None:
        rows = axis.generate_curriculum(TREATMENT_MIX, CONSTRUCTION_SEED)
        regenerated = "".join(
            json.dumps(axis.public_row(row), ensure_ascii=False) + "\n" for row in rows
        )
        frozen = (EXP / "data" / "sft_hygiene_explore.jsonl").read_text(encoding="utf-8")
        self.assertEqual(regenerated, frozen)

    def test_corpus_contains_no_repair_lessons(self) -> None:
        # The de-stacking design point: the closed trace-repair kinds must be
        # entirely absent from the frozen treatment corpus.
        rows = axis.generate_curriculum(TREATMENT_MIX, CONSTRUCTION_SEED)
        kinds = Counter(row["kind"] for row in rows)
        self.assertEqual(
            kinds,
            {
                "u_explore": 40,
                "u_hygiene": 40,
            },
        )
        for banned_kind in ("u_bugfind", "u_bugmend", "u_retrace", "u_protocol"):
            self.assertNotIn(banned_kind, kinds)

    def test_corpus_balance_bounds(self) -> None:
        rows = axis.generate_curriculum(TREATMENT_MIX, CONSTRUCTION_SEED)
        balance = axis.check_corpus_balance(rows)
        self.assertEqual(balance["bugfind_rows"], 0)
        self.assertEqual(balance["bugfind_early_bugs"], 0)
        # Forensics-driven floors carried over from the v2 generator: injected
        # and co-located shares stay inside the generator's hard bounds.
        self.assertGreaterEqual(balance["hygiene_injected"], 20)
        self.assertLessEqual(balance["hygiene_injected"], 36)
        self.assertGreaterEqual(
            balance["hygiene_colocated"],
            max(2, balance["hygiene_injected"] * 2 // 5),
        )


class ExploreRederivationTests(unittest.TestCase):
    def test_answer_route_is_valid_shortest_and_unique(self) -> None:
        rng = random.Random(404)
        for _ in range(25):
            row = axis.explore_lesson(rng)
            prompt = prompt_of(row)
            edges: set[tuple[str, str]] = set()
            for line in prompt.splitlines():
                match = re.match(r"\s*- from (\w+) you can step to (.+)$", line)
                if match:
                    source = match.group(1)
                    for target in match.group(2).split(", "):
                        edges.add((source, target.strip()))
            header = re.search(
                r"Find the route from (\w+) to (\w+) that uses exactly (\d+) steps",
                prompt,
            )
            assert header is not None
            start, goal, budget = header.group(1), header.group(2), int(header.group(3))
            path = answer_of(row).split(">")
            # The answered route is valid and consumes exactly the budget.
            self.assertEqual(path[0], start)
            self.assertEqual(path[-1], goal)
            self.assertEqual(len(path) - 1, budget)
            for a, b in zip(path, path[1:]):
                self.assertIn((a, b), edges)
            # BFS: budget is the shortest distance and the path is unique.
            adjacency: dict[str, list[str]] = defaultdict(list)
            for a, b in sorted(edges):
                adjacency[a].append(b)
            level = {start: 1}
            frontier = [start]
            depth = 0
            found_depth = None
            while frontier and found_depth is None:
                depth += 1
                counts: dict[str, int] = defaultdict(int)
                for node in frontier:
                    for successor in adjacency[node]:
                        if successor not in level:
                            counts[successor] += level[node]
                for node, ways in counts.items():
                    level[node] = ways
                frontier = list(counts)
                if goal in counts:
                    found_depth = depth
            self.assertEqual(found_depth, budget)
            self.assertEqual(level[goal], 1)


class HygieneRederivationTests(unittest.TestCase):
    def test_answer_is_queried_record_value_and_never_the_decoy(self) -> None:
        rng = random.Random(505)
        for _ in range(40):
            row = axis.hygiene_lesson(rng)
            prompt = prompt_of(row)
            question = re.search(r"Question: what is the ([a-z ]+) for (\w+)\?", prompt)
            assert question is not None
            attribute, subject = question.group(1), question.group(2)
            values = re.findall(
                rf"the {re.escape(attribute)} for {re.escape(subject)} is (\d+)", prompt
            )
            self.assertEqual(len(values), 1)
            self.assertEqual(answer_of(row), values[0])
            audit = row["_audit"]
            self.assertNotEqual(str(audit["decoy"]), answer_of(row))
            if audit["injections"]:
                self.assertIn(str(audit["decoy"]), prompt)
            else:
                self.assertNotIn(str(audit["decoy"]), prompt)

    def test_colocated_injections_are_generated(self) -> None:
        rng = random.Random(606)
        rows = [axis.hygiene_lesson(rng) for _ in range(60)]
        colocated = sum(
            1 for row in rows if row["_audit"].get("queried_record_injected")
        )
        self.assertGreater(colocated, 0)


if __name__ == "__main__":
    unittest.main()
