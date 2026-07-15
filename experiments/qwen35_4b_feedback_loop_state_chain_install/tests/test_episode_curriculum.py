import json
import random
import re
import sys
import unittest
from collections import Counter
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))

import gen_curriculum as original  # noqa: E402
import gen_episode_curriculum as episode  # noqa: E402


def answer_of(row: dict) -> str:
    assert row["answer"].startswith("ANSWER: ")
    return row["answer"].removeprefix("ANSWER: ")


def prompt_of(row: dict) -> str:
    return row["messages"][0]["content"]


FEEDLOOP_APPLY = {
    "troughline": episode._trough_apply,
    "trinketcord": episode._cord_apply,
    "crankwheel": episode._wheel_apply,
    "sigilslate": episode._slate_apply,
}
FEEDLOOP_GRAMMAR_BUILDERS = {
    "troughline": episode._trough_grammar,
    "trinketcord": episode._cord_grammar,
    "crankwheel": episode._wheel_grammar,
    "sigilslate": episode._slate_grammar,
}
FEEDLOOP_EXTENDED_BUILDERS = {
    "troughline": episode._trough_grammar_extended,
    "trinketcord": episode._cord_grammar_extended,
    "crankwheel": episode._wheel_grammar_extended,
    "sigilslate": episode._slate_grammar_extended,
}


def run_steps(apply_fn, steps, start):
    current = start
    for op in steps:
        current = apply_fn(op, current)
    return current


class GenerationValidityTests(unittest.TestCase):
    def test_smoke_generation_valid_and_leak_free(self) -> None:
        rows = episode.generate_curriculum(episode.SMOKE_MIX, 12345)
        summary = episode.validate_generated(rows)
        episode.check_banned_vocabulary(rows)
        self.assertEqual(summary["rows"], 2 * len(episode.SKILLS))
        self.assertEqual(set(summary["kinds"]), {"u_feedloop", "u_statechain"})

    def test_banned_vocabulary_scan_actually_fires(self) -> None:
        rows = episode.generate_curriculum(episode.SMOKE_MIX, 54321)
        poisoned = json.loads(json.dumps(episode.public_row(rows[0])))
        poisoned["_audit"] = rows[0]["_audit"]
        poisoned["messages"][0]["content"] += " the warren gate"
        with self.assertRaises(ValueError):
            episode.check_banned_vocabulary([poisoned])

    def test_description_noun_ban_actually_fires(self) -> None:
        rows = episode.generate_curriculum(episode.SMOKE_MIX, 54321)
        for leak in ("a rerun of the steps", "the protocol", "one flag"):
            poisoned = json.loads(json.dumps(episode.public_row(rows[0])))
            poisoned["_audit"] = rows[0]["_audit"]
            poisoned["messages"][0]["content"] += f" {leak}"
            with self.assertRaises(ValueError):
                episode.check_banned_vocabulary([poisoned])

    def test_fresh_vocabulary_disjoint_from_predecessor_pools(self) -> None:
        original_tokens = {
            item for pool in original.SURFACE_POOLS.values() for item in pool
        }
        fresh = set(episode.FRESH_SURFACE_TOKENS)
        self.assertFalse(fresh & original_tokens)
        self.assertFalse(fresh & set(episode.BANNED_PROMPT_TOKENS))
        # The disallowed predecessor formalism nouns never appear as surfaces.
        self.assertFalse(
            {"gauges", "line", "pile", "chain"}
            & set(episode.FEEDLOOP_FORMALISMS)
            | {"gauges", "line", "pile", "chain"}
            & set(episode.STATECHAIN_FORMALISMS)
        )

    def test_frozen_corpus_regenerates_byte_identically(self) -> None:
        rows = episode.generate_curriculum(episode.ARM_MIX, 77130)
        regenerated = "".join(
            json.dumps(episode.public_row(row), ensure_ascii=False) + "\n"
            for row in rows
        )
        frozen = (EXP / "data" / "sft_feedloop_state.jsonl").read_text(encoding="utf-8")
        self.assertEqual(regenerated, frozen)

    def test_corpus_balance_bounds(self) -> None:
        rows = episode.generate_curriculum(episode.ARM_MIX, 77130)
        balance = episode.check_corpus_balance(rows)
        self.assertEqual(
            balance["feedloop_formalisms"],
            {formalism: 20 for formalism in episode.FEEDLOOP_FORMALISMS},
        )
        self.assertEqual(
            balance["statechain_formalisms"],
            {formalism: 40 for formalism in episode.STATECHAIN_FORMALISMS},
        )
        self.assertGreaterEqual(balance["statechain_hidden_updates_min"], 3)
        kinds = Counter(row["kind"] for row in rows)
        self.assertEqual(kinds, {"u_feedloop": 80, "u_statechain": 80})


class FeedloopRederivationTests(unittest.TestCase):
    def test_two_round_evidence_isolates_a_unique_easy_fix(self) -> None:
        rng = random.Random(909)
        seen = set()
        for index in range(24):
            formalism = episode.FEEDLOOP_FORMALISMS[
                index % len(episode.FEEDLOOP_FORMALISMS)
            ]
            seen.add(formalism)
            row = episode.feedloop_lesson(rng, formalism)
            audit = row["_audit"]
            spec = audit["spec"]
            apply_fn = FEEDLOOP_APPLY[formalism]
            written = [tuple(op) for op in spec["written"]]
            bug_at, true_op = spec["true_fix"][0], tuple(spec["true_fix"][1])
            wrong_at, wrong_op = spec["wrong_fix"][0], tuple(spec["wrong_fix"][1])
            start_a, start_b = spec["start_a"], spec["start_b"]

            # The fix search space stays small: the lesson is USING the
            # second-trial evidence, not a hard search.
            self.assertLessEqual(spec["grammar_size"], 26)
            # The written steps really produce the failing round-one evidence.
            self.assertEqual(run_steps(apply_fn, written, start_a), spec["finished_a"])
            self.assertNotEqual(spec["finished_a"], spec["wanted_a"])
            # The true fix restores both trials.
            repaired = list(written)
            self.assertNotEqual(repaired[bug_at], true_op)
            repaired[bug_at] = true_op
            self.assertEqual(run_steps(apply_fn, repaired, start_a), spec["wanted_a"])
            self.assertEqual(run_steps(apply_fn, repaired, start_b), spec["wanted_b"])
            # Independent exhaustive enumeration over the full grammar: at
            # least two candidates square with round one; exactly one (the
            # true fix) survives round two; the earlier wrong attempt is a
            # round-one candidate that round two eliminates.
            fixes1 = []
            grammar_ops = self._grammar_for(formalism, spec, written)
            for slot in range(len(written)):
                for candidate in grammar_ops:
                    if candidate == written[slot]:
                        continue
                    patched = written[:slot] + [candidate] + written[slot + 1 :]
                    if run_steps(apply_fn, patched, start_a) == spec["wanted_a"]:
                        fixes1.append((slot, candidate))
            self.assertGreaterEqual(len(fixes1), 2)
            self.assertEqual(len(fixes1), audit["candidates_after_round1"])
            self.assertIn((bug_at, true_op), fixes1)
            self.assertIn((wrong_at, wrong_op), fixes1)
            fixes2 = [
                (slot, candidate)
                for slot, candidate in fixes1
                if run_steps(
                    apply_fn,
                    written[:slot] + [candidate] + written[slot + 1 :],
                    start_b,
                )
                == spec["wanted_b"]
            ]
            self.assertEqual(fixes2, [(bug_at, true_op)])
            # The wrong attempt's round-two evidence really fails.
            attempted = list(written)
            attempted[wrong_at] = wrong_op
            self.assertEqual(
                run_steps(apply_fn, attempted, start_b),
                spec["finished_b_after_wrong"],
            )
            self.assertNotEqual(spec["finished_b_after_wrong"], spec["wanted_b"])
            # Prompt and answer surface the same episode.
            answer = re.fullmatch(r"STEP (\d+): (.+)", answer_of(row))
            assert answer is not None
            self.assertEqual(int(answer.group(1)), bug_at + 1)
            self.assertIn(audit["wrong_attempt"], prompt_of(row))
            self.assertNotEqual(answer_of(row), audit["wrong_attempt"])
            # The documented spec bounds every parameterized operation, and
            # probing beyond the bound surfaces no legal alternative.
            self.assert_legality_bounded(row, grammar_ops)
        self.assertEqual(seen, set(episode.FEEDLOOP_FORMALISMS))

    def assert_legality_bounded(self, row: dict, bounded_grammar: list) -> None:
        audit = row["_audit"]
        spec = audit["spec"]
        formalism = audit["formalism"]
        apply_fn = FEEDLOOP_APPLY[formalism]
        written = [tuple(op) for op in spec["written"]]
        true_fix = (spec["true_fix"][0], tuple(spec["true_fix"][1]))
        prompt = prompt_of(row)
        clauses = audit["legality_clauses"]
        self.assertTrue(clauses)
        for clause in clauses:
            self.assertIn(clause, prompt)
        extended = FEEDLOOP_EXTENDED_BUILDERS[formalism](
            list(spec["vocabulary"])
        )
        bounded = set(bounded_grammar)
        for op in bounded:
            self.assertIn(op, extended)
        survivors = []
        for slot in range(len(written)):
            for candidate in extended:
                if candidate == written[slot]:
                    continue
                patched = written[:slot] + [candidate] + written[slot + 1 :]
                if (
                    run_steps(apply_fn, patched, spec["start_a"]) == spec["wanted_a"]
                    and run_steps(apply_fn, patched, spec["start_b"])
                    == spec["wanted_b"]
                ):
                    survivors.append((slot, candidate))
        self.assertIn(true_fix, survivors)
        out_of_bound = [fix for fix in survivors if fix != true_fix]
        # Every extra survivor is illegal under the documented bound: the
        # bounded grammar retains exactly the graded answer.
        for _, candidate in out_of_bound:
            self.assertNotIn(candidate, bounded)
        self.assertEqual(
            len(survivors),
            audit["extended_uniqueness_audit"]["round2_survivors_extended"],
        )
        self.assertEqual(
            len(out_of_bound),
            audit["extended_uniqueness_audit"]["out_of_bound_alternatives"],
        )

    def test_frozen_corpus_and_holdout_are_genuinely_unique_when_bounded(self) -> None:
        """Every frozen feedloop row (80 training + 20 holdout) re-verified.

        This is the closing audit for the adversarial-review MAJOR: the
        previously-ambiguous rows (13 training + 2 holdout under unbounded
        semantics) must be genuinely unique under the documented bounded
        grammar, with every out-of-bound alternative excluded by the
        rendered legality clause alone.
        """
        expected_ambiguous = {"arm": 13, "holdout": 2}
        for name, mix, seed in (
            ("arm", episode.ARM_MIX, 77130),
            ("holdout", episode.HOLDOUT_MIX, 88026),
        ):
            rows = [
                row
                for row in episode.generate_curriculum(mix, seed)
                if row["kind"] == "u_feedloop"
            ]
            self.assertEqual(len(rows), 80 if name == "arm" else 20)
            previously_ambiguous = 0
            for row in rows:
                audit = row["_audit"]
                spec = audit["spec"]
                formalism = audit["formalism"]
                apply_fn = FEEDLOOP_APPLY[formalism]
                written = [tuple(op) for op in spec["written"]]
                true_fix = (spec["true_fix"][0], tuple(spec["true_fix"][1]))
                bounded = FEEDLOOP_GRAMMAR_BUILDERS[formalism](
                    list(spec["vocabulary"])
                )
                survivors = []
                for slot in range(len(written)):
                    for candidate in bounded:
                        if candidate == written[slot]:
                            continue
                        patched = (
                            written[:slot] + [candidate] + written[slot + 1 :]
                        )
                        if (
                            run_steps(apply_fn, patched, spec["start_a"])
                            == spec["wanted_a"]
                            and run_steps(apply_fn, patched, spec["start_b"])
                            == spec["wanted_b"]
                        ):
                            survivors.append((slot, candidate))
                self.assertEqual(survivors, [true_fix], row["task_id"])
                self.assert_legality_bounded(row, bounded)
                if audit["extended_uniqueness_audit"]["out_of_bound_alternatives"]:
                    previously_ambiguous += 1
            self.assertEqual(previously_ambiguous, expected_ambiguous[name], name)

    def _grammar_for(self, formalism: str, spec: dict, written: list) -> list:
        """Rebuild the exact per-row grammar from the recorded vocabulary."""
        del written
        return FEEDLOOP_GRAMMAR_BUILDERS[formalism](list(spec["vocabulary"]))


class StatechainRederivationTests(unittest.TestCase):
    def test_brewvat_answer_requires_accumulated_hidden_state(self) -> None:
        rng = random.Random(707)
        for _ in range(12):
            row = episode.brewvat_lesson(rng)
            spec = row["_audit"]["spec"]
            steps = [tuple(step) for step in spec["steps"]]
            state = (0, "mild")
            readouts = []
            for step in steps:
                state = episode._brewvat_apply(step, state)
                readouts.append(episode._brewvat_readout(state[0], spec["threshold"]))
            self.assertEqual(readouts, spec["readouts"])
            self.assertEqual(list(state), spec["final"])
            self.assertEqual(answer_of(row), episode._brewvat_output(state))
            # Distractors: a stateless reader and a last-step-only reader
            # both get it wrong.
            stateless = episode._brewvat_output((0, "mild"))
            lastonly = episode._brewvat_output(
                episode._brewvat_apply(steps[-1], (0, "mild"))
            )
            self.assertNotEqual(answer_of(row), stateless)
            self.assertNotEqual(answer_of(row), lastonly)
            self.assertGreaterEqual(row["_audit"]["hidden_updates"], 3)
            # The documented threshold in the prompt matches the simulation.
            threshold = re.search(
                r"strength is at least (\d+)", prompt_of(row)
            )
            assert threshold is not None
            self.assertEqual(int(threshold.group(1)), spec["threshold"])

    def test_courierloft_answer_requires_accumulated_hidden_state(self) -> None:
        rng = random.Random(808)
        for _ in range(12):
            row = episode.courierloft_lesson(rng)
            spec = row["_audit"]["spec"]
            steps = [tuple(step) for step in spec["steps"]]
            perches = spec["perches"]
            state = (0, 0)
            readouts = []
            for step in steps:
                readouts.append(
                    episode._loft_readout(step, state, spec["threshold"])
                )
                state = episode._loft_apply(step, state)
            self.assertEqual(readouts, spec["readouts"])
            self.assertEqual(list(state), spec["final"])
            self.assertEqual(answer_of(row), episode._loft_output(state, perches))
            stateless = episode._loft_output((0, 0), perches)
            lastonly = episode._loft_output(
                episode._loft_apply(steps[-1], (0, 0)), perches
            )
            self.assertNotEqual(answer_of(row), stateless)
            self.assertNotEqual(answer_of(row), lastonly)
            self.assertGreaterEqual(row["_audit"]["hidden_updates"], 3)
            # The prompt's documented ring order matches the simulation's.
            self.assertIn(" ".join(perches), prompt_of(row))

    def test_statechain_validator_rejects_distractor_equal_answers(self) -> None:
        rng = random.Random(606)
        row = episode.brewvat_lesson(rng)
        row["task_id"] = "fls_statechain_00000"
        broken = dict(row)
        broken["_audit"] = dict(row["_audit"])
        broken["_audit"]["distractor_stateless"] = answer_of(row)
        with self.assertRaises(ValueError):
            episode.validate_generated([broken])


if __name__ == "__main__":
    unittest.main()
