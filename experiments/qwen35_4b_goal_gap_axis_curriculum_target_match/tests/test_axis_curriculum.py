import json
import random
import re
import sys
import unittest
from collections import Counter, defaultdict
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))

import gen_axis_curriculum as axis  # noqa: E402
import gen_curriculum as original  # noqa: E402


def answer_of(row: dict) -> str:
    assert row["answer"].startswith("ANSWER: ")
    return row["answer"].removeprefix("ANSWER: ")


def prompt_of(row: dict) -> str:
    return row["messages"][0]["content"]


class GenerationValidityTests(unittest.TestCase):
    def test_smoke_generation_valid_and_leak_free(self) -> None:
        rows = axis.generate_curriculum(axis.SMOKE_MIX, 12345)
        summary = axis.validate_generated(rows)
        axis.check_banned_vocabulary(rows)
        self.assertEqual(summary["rows"], 2 * len(axis.SKILLS))
        self.assertEqual(
            set(summary["kinds"]),
            {"u_tracefix", "u_explore", "u_hygiene", "u_protocol"},
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
        rows = axis.generate_curriculum(axis.ARM_MIX, 77117)
        regenerated = "".join(
            json.dumps(axis.public_row(row), ensure_ascii=False) + "\n" for row in rows
        )
        frozen = (EXP / "data" / "sft_axis160.jsonl").read_text(encoding="utf-8")
        self.assertEqual(regenerated, frozen)

    def test_corpus_balance_bounds(self) -> None:
        rows = axis.generate_curriculum(axis.ARM_MIX, 77117)
        balance = axis.check_corpus_balance(rows)
        self.assertEqual(balance["hygiene_injected"] + balance["hygiene_clean"], 40)
        self.assertGreaterEqual(balance["hygiene_injected"], 20)
        self.assertLessEqual(balance["hygiene_injected"], 36)
        self.assertEqual(sum(balance["protocol_branches"].values()), 40)
        self.assertGreaterEqual(min(balance["protocol_branches"].values()), 2)
        self.assertEqual(len(balance["tracefix_formalisms"]), 4)
        kinds = Counter(row["kind"] for row in rows)
        self.assertEqual(
            kinds,
            {"u_tracefix": 40, "u_explore": 40, "u_hygiene": 40, "u_protocol": 40},
        )


class ExploreRederivationTests(unittest.TestCase):
    def test_answer_route_is_valid_shortest_and_unique(self) -> None:
        rng = random.Random(101)
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
        rng = random.Random(202)
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


class ProtocolRederivationTests(unittest.TestCase):
    def test_answer_reexecutes_from_documented_rules(self) -> None:
        rng = random.Random(303)
        for _ in range(40):
            row = axis.protocol_lesson(rng)
            prompt = prompt_of(row)
            rule_a = re.search(
                r"The (\w+) marker starts unlit and lights permanently the first "
                r"time the tally reaches (\d+) or more",
                prompt,
            )
            rule_b = re.search(
                r"The (\w+) marker starts unlit and flips \(lit to unlit, unlit to "
                r"lit\) on every surge whose amount is exactly (\d+)",
                prompt,
            )
            closing = re.search(
                r"Closing rule: if both markers are lit, report (\w+)-<final tally>; "
                r"if only the (\w+) marker is lit, report (\w+)-<final tally>; "
                r"otherwise report HOLD-<final tally>",
                prompt,
            )
            assert rule_a and rule_b and closing
            flag_a, set_a = rule_a.group(1), int(rule_a.group(2))
            toggle_b = int(rule_b.group(2))
            sealed_word, open_flag, open_word = closing.groups()
            self.assertEqual(open_flag, flag_a)
            amounts = [int(value) for value in re.findall(r"\d+\. surge of (\d+)", prompt)]
            tally = 0
            lit_a = False
            lit_b = False
            for amount in amounts:
                tally += amount
                if not lit_a and tally >= set_a:
                    lit_a = True
                if amount == toggle_b:
                    lit_b = not lit_b
            if lit_a and lit_b:
                expected = f"{sealed_word}-{tally}"
            elif lit_a:
                expected = f"{open_word}-{tally}"
            else:
                expected = f"HOLD-{tally}"
            self.assertEqual(answer_of(row), expected)
            audit = row["_audit"]
            self.assertEqual(audit["final_tally"], tally)
            self.assertEqual(audit["marker_a_lit"], lit_a)
            self.assertEqual(audit["marker_b_lit"], lit_b)


def parse_gauges_state(text: str) -> dict:
    return {
        name: int(value)
        for name, value in (part.split("=") for part in text.split(", "))
    }


def parse_sequence_state(text: str) -> tuple:
    return () if text == "(empty)" else tuple(text.split(" "))


def parse_gauges_op(description: str) -> tuple:
    match = re.fullmatch(r"add (\d+) to (\w+)", description)
    if match:
        return ("add", match.group(2), int(match.group(1)))
    match = re.fullmatch(r"double (\w+)", description)
    if match:
        return ("double", match.group(1))
    match = re.fullmatch(r"copy (\w+) into (\w+)", description)
    if match:
        return ("copy", match.group(1), match.group(2))
    raise AssertionError(f"unparseable gauges op: {description}")


def parse_line_op(description: str) -> tuple:
    if description == "remove the front item":
        return ("take",)
    if description == "move the front item to the back":
        return ("spin",)
    match = re.fullmatch(r"append (\w+) at the back", description)
    if match:
        return ("put", match.group(1))
    raise AssertionError(f"unparseable line op: {description}")


def parse_pile_op(description: str) -> tuple:
    if description == "discard the top item":
        return ("drop",)
    if description == "duplicate the top item":
        return ("dup",)
    if description == "swap the top two items":
        return ("flip",)
    match = re.fullmatch(r"push (\w+)", description)
    if match:
        return ("push", match.group(1))
    raise AssertionError(f"unparseable pile op: {description}")


def parse_chain_op(description: str) -> tuple:
    if description == "delete the last entry":
        return ("chop",)
    match = re.fullmatch(r"replace the first (\w+) with (\w+)", description)
    if match:
        return ("swapfirst", match.group(1), match.group(2))
    match = re.fullmatch(r"append (\w+)", description)
    if match:
        return ("tag", match.group(1))
    raise AssertionError(f"unparseable chain op: {description}")


FORMALISM_TOOLING = {
    "gauges": (parse_gauges_state, parse_gauges_op, axis._regmachine_apply),
    "line": (parse_sequence_state, parse_line_op, axis._queue_apply),
    "pile": (parse_sequence_state, parse_pile_op, axis._stack_apply),
    "chain": (parse_sequence_state, parse_chain_op, axis._chain_apply),
}


class TracefixRederivationTests(unittest.TestCase):
    def test_corrected_step_reproduces_the_intended_result(self) -> None:
        rng = random.Random(404)
        seen_formalisms = set()
        for _ in range(40):
            row = axis.tracefix_lesson(rng)
            formalism = row["_audit"]["formalism"]
            seen_formalisms.add(formalism)
            parse_state, parse_op, apply_fn = FORMALISM_TOOLING[formalism]
            prompt = prompt_of(row)
            start_match = re.search(r"Start: (.+?)\.\nProgram as written:", prompt)
            observed_match = re.search(
                r"Running the written program produced: (.+?)\.\n", prompt
            )
            intended_match = re.search(r"The intended result is: (.+?)\.\n", prompt)
            assert start_match and observed_match and intended_match
            start = parse_state(start_match.group(1))
            observed = parse_state(observed_match.group(1))
            intended = parse_state(intended_match.group(1))
            listing = re.findall(r"^\s+(\d+)\. (.+)$", prompt, flags=re.MULTILINE)
            written = [parse_op(description) for _, description in listing]
            self.assertEqual(
                [int(number) for number, _ in listing],
                list(range(1, len(written) + 1)),
            )
            answer = re.fullmatch(r"STEP (\d+): (.+)", answer_of(row))
            assert answer is not None
            bug_step = int(answer.group(1))
            corrected = parse_op(answer.group(2))
            self.assertEqual(bug_step, row["_audit"]["bug_step"])

            def run(program, state):
                current = state
                for op in program:
                    current = apply_fn(op, current)
                return current

            # The written program really produces the observed (wrong) result...
            self.assertEqual(run(written, start), observed)
            self.assertNotEqual(observed, intended)
            # ...and swapping in exactly the answered instruction repairs it.
            repaired = list(written)
            self.assertNotEqual(repaired[bug_step - 1], corrected)
            repaired[bug_step - 1] = corrected
            self.assertEqual(run(repaired, start), intended)
        self.assertEqual(seen_formalisms, set(axis.FORMALISMS))


if __name__ == "__main__":
    unittest.main()
