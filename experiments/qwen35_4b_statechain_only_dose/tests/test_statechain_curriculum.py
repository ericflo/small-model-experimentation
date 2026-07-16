import json
import random
import re
import sys
import unittest
from collections import Counter
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
REFERENCE_EXP = ROOT / "experiments" / "qwen35_4b_feedback_loop_state_chain_install"
sys.path.insert(0, str(EXP / "scripts"))

import gen_curriculum as original  # noqa: E402
import gen_statechain_curriculum as statechain  # noqa: E402


def answer_of(row: dict) -> str:
    assert row["answer"].startswith("ANSWER: ")
    return row["answer"].removeprefix("ANSWER: ")


def prompt_of(row: dict) -> str:
    return row["messages"][0]["content"]


def message_bytes(row: dict) -> bytes:
    return json.dumps(
        row["messages"], sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


class GenerationValidityTests(unittest.TestCase):
    def test_smoke_generation_valid_and_leak_free(self) -> None:
        rows = statechain.generate_curriculum(statechain.SMOKE_MIX, 12345)
        summary = statechain.validate_generated(rows)
        statechain.check_banned_vocabulary(rows)
        self.assertEqual(summary["rows"], 8)
        self.assertEqual(set(summary["kinds"]), {"u_statechain"})
        self.assertEqual(
            set(summary["surfaces"]), set(statechain.STATECHAIN_FORMALISMS)
        )

    def test_banned_vocabulary_scan_actually_fires(self) -> None:
        rows = statechain.generate_curriculum(statechain.SMOKE_MIX, 54321)
        poisoned = json.loads(json.dumps(statechain.public_row(rows[0])))
        poisoned["_audit"] = rows[0]["_audit"]
        poisoned["messages"][0]["content"] += " the warren gate"
        with self.assertRaises(ValueError):
            statechain.check_banned_vocabulary([poisoned])

    def test_description_noun_ban_actually_fires(self) -> None:
        rows = statechain.generate_curriculum(statechain.SMOKE_MIX, 54321)
        for leak in ("a rerun of the steps", "the protocol", "one flag"):
            poisoned = json.loads(json.dumps(statechain.public_row(rows[0])))
            poisoned["_audit"] = rows[0]["_audit"]
            poisoned["messages"][0]["content"] += f" {leak}"
            with self.assertRaises(ValueError):
                statechain.check_banned_vocabulary([poisoned])

    def test_retired_feedloop_surface_ban_actually_fires(self) -> None:
        """The reference cell's dead feedloop surfaces are banned here."""
        rows = statechain.generate_curriculum(statechain.SMOKE_MIX, 54321)
        for leak in (
            "the troughline hums",
            "a crankwheel turned",
            "one trinket fell",
            "etch morv here",
            "the sigil faded",
        ):
            poisoned = json.loads(json.dumps(statechain.public_row(rows[0])))
            poisoned["_audit"] = rows[0]["_audit"]
            poisoned["messages"][0]["content"] += f" {leak}"
            with self.assertRaises(ValueError):
                statechain.check_banned_vocabulary([poisoned])

    def test_retained_statechain_surfaces_are_not_banned(self) -> None:
        banned = set(statechain.BANNED_PROMPT_TOKENS)
        self.assertFalse(set(statechain.INHERITED_SURFACE_TOKENS) & banned)
        self.assertFalse(set(statechain.FRESH_SURFACE_TOKENS) & banned)
        self.assertFalse(
            set(statechain.FRESH_SURFACE_TOKENS)
            & set(statechain.INHERITED_SURFACE_TOKENS)
        )

    def test_fresh_vocabulary_disjoint_from_predecessor_pools(self) -> None:
        original_tokens = {
            item for pool in original.SURFACE_POOLS.values() for item in pool
        }
        fresh = set(statechain.FRESH_SURFACE_TOKENS)
        self.assertFalse(fresh & original_tokens)
        # The retired feedloop pools stay banned; the new fresh pool avoids
        # them entirely.
        for retired in ("brann", "plome", "drasp", "morv", "trough", "sigil"):
            self.assertIn(retired, statechain.BANNED_PROMPT_TOKENS)
            self.assertNotIn(retired, fresh)

    def test_frozen_corpus_regenerates_byte_identically(self) -> None:
        rows = statechain.generate_curriculum(statechain.ARM_MIX, 77140)
        regenerated = "".join(
            json.dumps(statechain.public_row(row), ensure_ascii=False) + "\n"
            for row in rows
        )
        frozen = (EXP / "data" / "sft_statechain_only.jsonl").read_text(encoding="utf-8")
        self.assertEqual(regenerated, frozen)

    def test_corpus_balance_bounds(self) -> None:
        rows = statechain.generate_curriculum(statechain.ARM_MIX, 77140)
        balance = statechain.check_corpus_balance(rows)
        self.assertEqual(
            balance["statechain_formalisms"],
            {formalism: 40 for formalism in statechain.STATECHAIN_FORMALISMS},
        )
        self.assertGreaterEqual(balance["statechain_hidden_updates_min"], 3)
        self.assertEqual(
            balance["new_formalism_rows_with_out_of_bound_parameters"], 0
        )
        kinds = Counter(row["kind"] for row in rows)
        self.assertEqual(kinds, {"u_statechain": 160})

    def test_holdout_mix_yields_ten_rows_per_formalism(self) -> None:
        rows = statechain.generate_curriculum(statechain.HOLDOUT_MIX, 88033)
        summary = statechain.validate_generated(rows)
        self.assertEqual(summary["rows"], 40)
        self.assertEqual(
            summary["surfaces"],
            {formalism: 10 for formalism in statechain.STATECHAIN_FORMALISMS},
        )

    def test_zero_row_overlap_with_reference_corpus_and_gate(self) -> None:
        """Retained brewvat/courierloft instances are FRESH, not copies."""
        local = {
            message_bytes(row)
            for row in statechain.generate_curriculum(statechain.ARM_MIX, 77140)
        } | {
            message_bytes(row)
            for row in statechain.generate_curriculum(statechain.HOLDOUT_MIX, 88033)
        }
        for name in ("sft_feedloop_state.jsonl", "local_tasks_seed88026.jsonl"):
            reference_rows = [
                json.loads(line)
                for line in (REFERENCE_EXP / "data" / name)
                .read_text(encoding="utf-8")
                .splitlines()
                if line
            ]
            reference = {message_bytes(row) for row in reference_rows}
            self.assertFalse(local & reference, name)


class BrewvatRederivationTests(unittest.TestCase):
    def test_brewvat_answer_requires_accumulated_hidden_state(self) -> None:
        rng = random.Random(707)
        for _ in range(12):
            row = statechain.brewvat_lesson(rng)
            spec = row["_audit"]["spec"]
            steps = [tuple(step) for step in spec["steps"]]
            state = (0, "mild")
            readouts = []
            for step in steps:
                state = statechain._brewvat_apply(step, state)
                readouts.append(
                    statechain._brewvat_readout(state[0], spec["threshold"])
                )
            self.assertEqual(readouts, spec["readouts"])
            self.assertEqual(list(state), spec["final"])
            self.assertEqual(answer_of(row), statechain._brewvat_output(state))
            # Distractors: a stateless reader and a last-step-only reader
            # both get it wrong.
            stateless = statechain._brewvat_output((0, "mild"))
            lastonly = statechain._brewvat_output(
                statechain._brewvat_apply(steps[-1], (0, "mild"))
            )
            self.assertNotEqual(answer_of(row), stateless)
            self.assertNotEqual(answer_of(row), lastonly)
            self.assertGreaterEqual(row["_audit"]["hidden_updates"], 3)
            # The documented threshold in the prompt matches the simulation.
            threshold = re.search(r"strength is at least (\d+)", prompt_of(row))
            assert threshold is not None
            self.assertEqual(int(threshold.group(1)), spec["threshold"])


class CourierloftRederivationTests(unittest.TestCase):
    def test_courierloft_answer_requires_accumulated_hidden_state(self) -> None:
        rng = random.Random(808)
        for _ in range(12):
            row = statechain.courierloft_lesson(rng)
            spec = row["_audit"]["spec"]
            steps = [tuple(step) for step in spec["steps"]]
            perches = spec["perches"]
            state = (0, 0)
            readouts = []
            for step in steps:
                readouts.append(
                    statechain._loft_readout(step, state, spec["threshold"])
                )
                state = statechain._loft_apply(step, state)
            self.assertEqual(readouts, spec["readouts"])
            self.assertEqual(list(state), spec["final"])
            self.assertEqual(answer_of(row), statechain._loft_output(state, perches))
            stateless = statechain._loft_output((0, 0), perches)
            lastonly = statechain._loft_output(
                statechain._loft_apply(steps[-1], (0, 0)), perches
            )
            self.assertNotEqual(answer_of(row), stateless)
            self.assertNotEqual(answer_of(row), lastonly)
            self.assertGreaterEqual(row["_audit"]["hidden_updates"], 3)
            # The prompt's documented ring order matches the simulation's.
            self.assertIn(" ".join(perches), prompt_of(row))


class PeatstoveRederivationTests(unittest.TestCase):
    def test_peatstove_answer_requires_accumulated_hidden_state(self) -> None:
        rng = random.Random(909)
        for _ in range(12):
            row = statechain.peatstove_lesson(rng)
            audit = row["_audit"]
            spec = audit["spec"]
            steps = [tuple(step) for step in spec["steps"]]
            state = (0, "mellow")
            readouts = []
            for step in steps:
                state = statechain._stove_apply(step, state)
                readouts.append(
                    statechain._stove_readout(state[0], spec["threshold"])
                )
            self.assertEqual(readouts, spec["readouts"])
            self.assertEqual(list(state), spec["final"])
            self.assertEqual(answer_of(row), statechain._stove_output(state))
            stateless = statechain._stove_output((0, "mellow"))
            lastonly = statechain._stove_output(
                statechain._stove_apply(steps[-1], (0, "mellow"))
            )
            self.assertNotEqual(answer_of(row), stateless)
            self.assertNotEqual(answer_of(row), lastonly)
            self.assertGreaterEqual(audit["hidden_updates"], 3)
            threshold = re.search(
                r"ember count is at least (\d+)", prompt_of(row)
            )
            assert threshold is not None
            self.assertEqual(int(threshold.group(1)), spec["threshold"])
            self.assert_legality_bounded(row, "rake", 1, 5)

    def assert_legality_bounded(
        self, row: dict, op: str, low: int, high: int
    ) -> None:
        """Per the reference's post-review contract: the documented spec
        bounds every parameterized operation, the clause appears verbatim in
        the rendered prompt, and probing the extended domain surfaces no
        rendered step outside the documented bound."""
        audit = row["_audit"]
        prompt = prompt_of(row)
        clauses = audit["legality_clauses"]
        self.assertTrue(clauses)
        for clause in clauses:
            self.assertIn(clause, prompt)
        bounded = audit["bounded_parameter_audit"]
        self.assertEqual(bounded["documented_bounds"], {op: [low, high]})
        self.assertEqual(
            bounded["amounts_probed_to"], statechain.EXTENDED_AMOUNT_BOUND
        )
        self.assertEqual(bounded["out_of_bound_steps"], 0)
        self.assertTrue(bounded["all_parameters_documented_bounded"])
        parameters = [
            step[1] for step in audit["spec"]["steps"] if step[0] == op
        ]
        self.assertEqual(len(parameters), bounded["parameterized_steps"])
        for value in parameters:
            self.assertGreaterEqual(value, low)
            self.assertLessEqual(value, high)


class MuletrackRederivationTests(unittest.TestCase):
    def test_muletrack_answer_requires_accumulated_hidden_state(self) -> None:
        rng = random.Random(101)
        for _ in range(12):
            row = statechain.muletrack_lesson(rng)
            audit = row["_audit"]
            spec = audit["spec"]
            steps = [tuple(step) for step in spec["steps"]]
            posts = spec["posts"]
            state = (0, 0)
            readouts = []
            for step in steps:
                readouts.append(
                    statechain._track_readout(step, state, spec["threshold"])
                )
                state = statechain._track_apply(step, state)
            self.assertEqual(readouts, spec["readouts"])
            self.assertEqual(list(state), spec["final"])
            self.assertEqual(answer_of(row), statechain._track_output(state, posts))
            stateless = statechain._track_output((0, 0), posts)
            lastonly = statechain._track_output(
                statechain._track_apply(steps[-1], (0, 0)), posts
            )
            self.assertNotEqual(answer_of(row), stateless)
            self.assertNotEqual(answer_of(row), lastonly)
            self.assertGreaterEqual(audit["hidden_updates"], 3)
            self.assertIn(" ".join(posts), prompt_of(row))
            PeatstoveRederivationTests.assert_legality_bounded(
                self, row, "plod", 1, 4
            )


class ValidatorRejectionTests(unittest.TestCase):
    def test_statechain_validator_rejects_distractor_equal_answers(self) -> None:
        rng = random.Random(606)
        row = statechain.brewvat_lesson(rng)
        row["task_id"] = "sod_statechain_00000"
        broken = dict(row)
        broken["_audit"] = dict(row["_audit"])
        broken["_audit"]["distractor_stateless"] = answer_of(row)
        with self.assertRaises(ValueError):
            statechain.validate_generated([broken])

    def test_bounded_parameter_audit_rejects_out_of_bound_steps(self) -> None:
        with self.assertRaises(RuntimeError):
            statechain.audit_bounded_parameters(
                [("rake", 7), ("stoke",)], "rake", 1, 5
            )
        with self.assertRaises(RuntimeError):
            statechain.audit_bounded_parameters(
                [("plod", 99)], "plod", 1, 4
            )
        clean = statechain.audit_bounded_parameters(
            [("rake", 5), ("rake", 1), ("stoke",)], "rake", 1, 5
        )
        self.assertEqual(clean["parameterized_steps"], 2)
        self.assertEqual(clean["out_of_bound_steps"], 0)

    def test_validator_rejects_poisoned_legality_audit(self) -> None:
        rng = random.Random(505)
        row = statechain.peatstove_lesson(rng)
        row["task_id"] = "sod_statechain_00000"
        # A clause missing from the rendered prompt must be rejected.
        broken = json.loads(json.dumps(statechain.public_row(row)))
        broken["_audit"] = json.loads(json.dumps(row["_audit"]))
        broken["_audit"]["legality_clauses"] = ["a clause not in the prompt"]
        with self.assertRaises(ValueError):
            statechain.validate_generated([broken])
        # A nonzero out-of-bound count must be rejected.
        broken = json.loads(json.dumps(statechain.public_row(row)))
        broken["_audit"] = json.loads(json.dumps(row["_audit"]))
        broken["_audit"]["bounded_parameter_audit"]["out_of_bound_steps"] = 1
        with self.assertRaises(ValueError):
            statechain.validate_generated([broken])

    def test_validator_rejects_unknown_kind_and_formalism(self) -> None:
        rng = random.Random(404)
        row = statechain.brewvat_lesson(rng)
        row["task_id"] = "sod_statechain_00000"
        broken = dict(row)
        broken["kind"] = "u_feedloop"
        with self.assertRaises(ValueError):
            statechain.validate_generated([broken])
        broken = dict(row)
        broken["_audit"] = dict(row["_audit"])
        broken["_audit"]["formalism"] = "troughline"
        with self.assertRaises(ValueError):
            statechain.validate_generated([broken])


if __name__ == "__main__":
    unittest.main()
