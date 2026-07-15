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


def answer_of(row: dict) -> str:
    assert row["answer"].startswith("ANSWER: ")
    return row["answer"].removeprefix("ANSWER: ")


def prompt_of(row: dict) -> str:
    return row["messages"][0]["content"]


def parse_line_state(text: str) -> tuple:
    return () if text == "(empty)" else tuple(text.split(" "))


def parse_line_op(description: str) -> tuple:
    if description == "remove the front item":
        return ("take",)
    if description == "move the front item to the back":
        return ("spin",)
    match = re.fullmatch(r"append (\w+) at the back", description)
    if match:
        return ("put", match.group(1))
    raise AssertionError(f"unparseable line op: {description}")


def parse_line_program(prompt: str) -> list[tuple]:
    listing = re.findall(r"^\s+(\d+)\. (.+)$", prompt, flags=re.MULTILINE)
    program = [parse_line_op(description) for _, description in listing]
    assert [int(number) for number, _ in listing] == list(range(1, len(program) + 1))
    return program


def run_line(program: list[tuple], state: tuple) -> tuple:
    current = state
    for op in program:
        current = axis._queue_apply(op, current)
    return current


def full_grammar() -> list[tuple]:
    """Superset grammar over every surface token.

    The generator enforces a globally unique single-step fix over its hidden
    4-item grammar; by the append-then-removed argument, no fix with a token
    outside that grammar can exist without contradicting that uniqueness, so
    exhaustive search over this superset must recover exactly the unique fix.
    """
    return [("take",), ("spin",)] + [("put", moon) for moon in axis.MOONS]


def exhaustive_single_step_fixes(
    written: list[tuple], start: tuple, intended: tuple
) -> list[tuple[int, tuple]]:
    return [
        (index, candidate)
        for index in range(len(written))
        for candidate in full_grammar()
        if candidate != written[index]
        and run_line(written[:index] + [candidate] + written[index + 1 :], start)
        == intended
    ]


class GenerationValidityTests(unittest.TestCase):
    def test_smoke_generation_valid_and_leak_free(self) -> None:
        rows = axis.generate_curriculum(axis.SMOKE_MIX, 12345)
        summary = axis.validate_generated(rows)
        axis.check_banned_vocabulary(rows)
        self.assertEqual(summary["rows"], 2 * len(axis.SKILLS))
        self.assertEqual(
            set(summary["kinds"]),
            {"u_bugfind", "u_bugmend", "u_retrace", "u_explore", "u_hygiene"},
        )

    def test_banned_vocabulary_scan_actually_fires(self) -> None:
        rows = axis.generate_curriculum(axis.SMOKE_MIX, 54321)
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
        rows = axis.generate_curriculum(axis.ARM_MIX, 77118)
        regenerated = "".join(
            json.dumps(axis.public_row(row), ensure_ascii=False) + "\n" for row in rows
        )
        frozen = (EXP / "data" / "sft_axis_v2.jsonl").read_text(encoding="utf-8")
        self.assertEqual(regenerated, frozen)

    def test_corpus_balance_bounds(self) -> None:
        rows = axis.generate_curriculum(axis.ARM_MIX, 77118)
        balance = axis.check_corpus_balance(rows)
        self.assertEqual(balance["bugfind_rows"], 30)
        # Forensics-driven floors: early bugs and co-located injections
        # oversampled well above the generator's own hard minima.
        self.assertGreaterEqual(balance["bugfind_early_bugs"], 9)
        self.assertGreaterEqual(balance["hygiene_injected"], 20)
        self.assertLessEqual(balance["hygiene_injected"], 36)
        self.assertGreaterEqual(
            balance["hygiene_colocated"],
            max(2, balance["hygiene_injected"] * 2 // 5),
        )
        kinds = Counter(row["kind"] for row in rows)
        self.assertEqual(
            kinds,
            {
                "u_bugfind": 30,
                "u_bugmend": 25,
                "u_retrace": 25,
                "u_explore": 40,
                "u_hygiene": 40,
            },
        )


class BugfindRederivationTests(unittest.TestCase):
    def test_answer_step_is_the_unique_exhaustive_fix(self) -> None:
        rng = random.Random(101)
        for _ in range(30):
            row = axis.bugfind_lesson(rng)
            prompt = prompt_of(row)
            start_match = re.search(r"Start: (.+?)\.\nProgram as written:", prompt)
            observed_match = re.search(
                r"Running the written program produced: (.+?)\.\n", prompt
            )
            intended_match = re.search(r"The intended result is: (.+?)\.\n", prompt)
            assert start_match and observed_match and intended_match
            start = parse_line_state(start_match.group(1))
            observed = parse_line_state(observed_match.group(1))
            intended = parse_line_state(intended_match.group(1))
            written = parse_line_program(prompt)
            answer = re.fullmatch(r"STEP (\d+)", answer_of(row))
            assert answer is not None
            bug_step = int(answer.group(1))
            self.assertEqual(bug_step, row["_audit"]["bug_step"])
            # The written program really produces the observed (wrong) result.
            self.assertEqual(run_line(written, start), observed)
            self.assertNotEqual(observed, intended)
            # Exhaustive grammar search recomputes the unique fix step.
            fixes = exhaustive_single_step_fixes(written, start, intended)
            self.assertEqual({index for index, _ in fixes}, {bug_step - 1})
            self.assertEqual(len({candidate for _, candidate in fixes}), len(fixes))

    def test_early_bias_is_visible_in_the_audit(self) -> None:
        rng = random.Random(191)
        rows = [axis.bugfind_lesson(rng) for _ in range(30)]
        early = sum(
            1
            for row in rows
            if row["_audit"]["bug_step"] <= row["_audit"]["program_length"] // 2
        )
        self.assertGreaterEqual(early, 9)


class BugmendRederivationTests(unittest.TestCase):
    def test_corrected_instruction_uniquely_reproduces_intended(self) -> None:
        rng = random.Random(202)
        for _ in range(30):
            row = axis.bugmend_lesson(rng)
            prompt = prompt_of(row)
            start_match = re.search(r"Start: (.+?)\.\nProgram as written:", prompt)
            intended_match = re.search(r"The intended result is: (.+?)\.\n", prompt)
            step_match = re.search(r"Step (\d+) was written down wrong\.", prompt)
            assert start_match and intended_match and step_match
            start = parse_line_state(start_match.group(1))
            intended = parse_line_state(intended_match.group(1))
            bug_step = int(step_match.group(1))
            self.assertEqual(bug_step, row["_audit"]["bug_step"])
            written = parse_line_program(prompt)
            corrected = parse_line_op(answer_of(row))
            self.assertNotEqual(corrected, written[bug_step - 1])
            # Swapping in exactly the answered instruction repairs the run...
            repaired = list(written)
            repaired[bug_step - 1] = corrected
            self.assertEqual(run_line(repaired, start), intended)
            # ...and it is the globally unique single-step fix.
            fixes = exhaustive_single_step_fixes(written, start, intended)
            self.assertEqual(fixes, [(bug_step - 1, corrected)])


class RetraceRederivationTests(unittest.TestCase):
    def test_first_mismatch_and_final_state_reexecute(self) -> None:
        rng = random.Random(303)
        for _ in range(30):
            row = axis.retrace_lesson(rng)
            prompt = prompt_of(row)
            start_match = re.search(r"Start: (.+?)\.\nProgram:", prompt)
            assert start_match is not None
            start = parse_line_state(start_match.group(1))
            program = parse_line_program(prompt)
            claimed = [
                parse_line_state(value)
                for value in re.findall(
                    r"^\s+after step \d+: (.+)$", prompt, flags=re.MULTILINE
                )
            ]
            self.assertEqual(len(claimed), len(program))
            answer = re.fullmatch(r"STEP (\d+); (.+)", answer_of(row))
            assert answer is not None
            wrong_step = int(answer.group(1))
            final = parse_line_state(answer.group(2))
            # Re-execute and find the first mismatching transition.
            current = start
            first_wrong = None
            true_states = []
            for index, op in enumerate(program, 1):
                current = axis._queue_apply(op, current)
                true_states.append(current)
                if first_wrong is None and current != claimed[index - 1]:
                    first_wrong = index
            self.assertEqual(first_wrong, wrong_step)
            self.assertEqual(wrong_step, row["_audit"]["wrong_step"])
            # The answered final state is the true final state and differs
            # from the claimed one.
            self.assertEqual(final, true_states[-1])
            self.assertNotEqual(claimed[-1], true_states[-1])


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
