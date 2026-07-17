import json
import random
import sys
import unittest
from collections import Counter
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
MENDERS_EXP = ROOT / "experiments" / "qwen35_4b_menders_dose_scale"
sys.path.insert(0, str(EXP / "scripts"))

import gen_enum_repair_curriculum as enum_mod  # noqa: E402
import gen_feedloop_curriculum as feedloop  # noqa: E402


def answer_of(row: dict) -> str:
    assert row["answer"].startswith("ANSWER: ")
    return row["answer"].removeprefix("ANSWER: ")


def prompt_of(row: dict) -> str:
    return row["messages"][0]["content"]


def message_bytes(row: dict) -> bytes:
    return json.dumps(
        row["messages"], sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def poisoned_copy(row: dict) -> dict:
    broken = json.loads(json.dumps(enum_mod.public_row(row)))
    broken["_audit"] = json.loads(json.dumps(row["_audit"]))
    return broken


class GenerationValidityTests(unittest.TestCase):
    def test_smoke_generation_valid_and_leak_free(self) -> None:
        rows = enum_mod.generate_curriculum(enum_mod.SMOKE_MIX, 12345)
        summary = enum_mod.validate_generated(rows)
        enum_mod.check_banned_vocabulary(rows)
        enum_mod.check_corpus_balance(rows)
        self.assertEqual(summary["rows"], 16)
        self.assertEqual(set(summary["kinds"]), {"u_enum_repair"})
        self.assertEqual(
            set(summary["surfaces"]), set(enum_mod.ENUM_FORMALISMS)
        )

    def test_banned_vocabulary_scan_actually_fires(self) -> None:
        rows = enum_mod.generate_curriculum(enum_mod.SMOKE_MIX, 54321)
        for leak in (
            "the warren gate",
            "a quick test",
            "one repair left",
            "rerun it",
            "this protocol",
            "a debug pass",
        ):
            poisoned = poisoned_copy(rows[0])
            poisoned["messages"][0]["content"] += f" {leak}"
            with self.assertRaises(ValueError, msg=leak):
                enum_mod.check_banned_vocabulary([poisoned])

    def test_scan_is_case_insensitive(self) -> None:
        rows = enum_mod.generate_curriculum(enum_mod.SMOKE_MIX, 54321)
        poisoned = poisoned_copy(rows[0])
        poisoned["messages"][0]["content"] += " MENDERS"
        with self.assertRaises(ValueError):
            enum_mod.check_banned_vocabulary([poisoned])

    def test_feedloop_machinery_is_byte_identical_to_the_menders_source(self) -> None:
        """The machinery is the menders dose-scale cell's reviewed
        generator, copied by bytes — never forked."""
        copy = EXP / "scripts" / "gen_feedloop_curriculum.py"
        source = MENDERS_EXP / "scripts" / "gen_feedloop_curriculum.py"
        self.assertEqual(copy.read_bytes(), source.read_bytes())

    def test_no_fresh_surface_tokens_and_inheritance_is_complete(self) -> None:
        self.assertEqual(enum_mod.FRESH_SURFACE_TOKENS, ())
        inherited = set(enum_mod.INHERITED_SURFACE_TOKENS)
        self.assertEqual(
            inherited,
            set(feedloop.INHERITED_SURFACE_TOKENS)
            | set(feedloop.FRESH_SURFACE_TOKENS),
        )
        self.assertFalse(inherited & set(enum_mod.BANNED_PROMPT_TOKENS))

    def test_frozen_corpus_regenerates_byte_identically(self) -> None:
        rows = enum_mod.generate_curriculum(enum_mod.ARM_MIX, 77190)
        regenerated = "".join(
            json.dumps(enum_mod.public_row(row), ensure_ascii=False) + "\n"
            for row in rows
        )
        frozen = (EXP / "data" / "sft_enum_repair.jsonl").read_text(encoding="utf-8")
        self.assertEqual(regenerated, frozen)

    def test_corpus_k_distribution_and_formalism_balance(self) -> None:
        rows = enum_mod.generate_curriculum(enum_mod.ARM_MIX, 77190)
        balance = enum_mod.check_corpus_balance(rows)
        self.assertEqual(
            balance["enum_repair_formalisms"],
            {formalism: 20 for formalism in enum_mod.ENUM_FORMALISMS},
        )
        self.assertEqual(
            balance["k_tried_counts"],
            {str(k): 32 for k in enum_mod.K_CYCLE},
        )
        for formalism in enum_mod.ENUM_FORMALISMS:
            self.assertEqual(
                balance["k_tried_by_formalism"][formalism],
                {str(k): 4 for k in enum_mod.K_CYCLE},
            )
        # k=0 first-candidate rows and deep-in-the-list rows both present.
        self.assertIn("0", balance["k_tried_counts"])
        self.assertIn(str(enum_mod.DEEP_K), balance["k_tried_counts"])

    def test_holdout_mix_yields_the_frozen_split(self) -> None:
        rows = enum_mod.generate_curriculum(enum_mod.HOLDOUT_MIX, 88052)
        summary = enum_mod.validate_generated(rows)
        self.assertEqual(summary["rows"], 40)
        self.assertEqual(summary["kinds"], {"u_enum_repair": 40})
        self.assertEqual(
            summary["surfaces"],
            {formalism: 5 for formalism in enum_mod.ENUM_FORMALISMS},
        )
        balance = enum_mod.check_corpus_balance(rows)
        for formalism in enum_mod.ENUM_FORMALISMS:
            self.assertEqual(
                balance["k_tried_by_formalism"][formalism],
                {str(k): 1 for k in enum_mod.K_CYCLE},
            )

    def test_zero_row_overlap_with_the_formalism_sharing_predecessor(self) -> None:
        """The menders cell shares all eight formalisms; row-level
        freshness is the bar."""
        local = {
            message_bytes(row)
            for row in enum_mod.generate_curriculum(enum_mod.ARM_MIX, 77190)
        } | {
            message_bytes(row)
            for row in enum_mod.generate_curriculum(enum_mod.HOLDOUT_MIX, 88052)
        }
        for path in (
            MENDERS_EXP / "data" / "sft_feedloop_scale.jsonl",
            MENDERS_EXP / "data" / "local_tasks_seed88037.jsonl",
        ):
            reference_rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line
            ]
            reference = {message_bytes(row) for row in reference_rows}
            self.assertFalse(local & reference, path.name)

    def test_answer_format_is_exact_match_gradable(self) -> None:
        rows = enum_mod.generate_curriculum(enum_mod.SMOKE_MIX, 999)
        for row in rows:
            self.assertRegex(answer_of(row), r"^STEP \d+: .+$")
            self.assertNotIn("\n", row["answer"])

    def test_every_prompt_documents_the_enumeration(self) -> None:
        rows = enum_mod.generate_curriculum(enum_mod.SMOKE_MIX, 4242)
        for row in rows:
            prompt = prompt_of(row)
            self.assertIn(enum_mod.CANONICAL_ORDER_STATEMENT, prompt)
            self.assertIn("Numbered action list", prompt)
            self.assertIn(enum_mod.ASK_LINE, prompt)
            self.assertIn("First trial, starting from", prompt)
            self.assertIn("Second trial, starting from", prompt)
            if row["_audit"]["k_tried"] == 0:
                self.assertIn(enum_mod.NONE_TRIED_LINE, prompt)
            else:
                self.assertIn(enum_mod.TRIED_HEADER, prompt)


class CanonicalNextRederivationTests(unittest.TestCase):
    def test_target_rederives_across_formalisms_and_depths(self) -> None:
        rng = random.Random(707)
        for formalism in enum_mod.ENUM_FORMALISMS:
            for k_target in (0, 3, 10):
                row = enum_mod.enum_repair_lesson(rng, formalism, k_target)
                audit = row["_audit"]
                self.assertEqual(audit["k_tried"], k_target)
                candidates, machine = enum_mod.rederive_candidates(audit)
                self.assertGreaterEqual(machine["success_index"], k_target)
                target = candidates[k_target]
                self.assertEqual(
                    answer_of(row),
                    f"STEP {target[0] + 1}: {machine['describe'](target[1])}",
                )
                # verify_row_audit runs the full exhaustive re-derivation.
                enum_mod.verify_row_audit(row)

    def test_unique_both_trials_fix_and_written_fails_both(self) -> None:
        rng = random.Random(808)
        for formalism in enum_mod.ENUM_FORMALISMS:
            row = enum_mod.enum_repair_lesson(rng, formalism, 1)
            audit = row["_audit"]
            candidates, machine = enum_mod.rederive_candidates(audit)
            successes = [
                (index, op)
                for index, op in candidates
                if enum_mod.repairs_both(
                    machine["apply"], machine["written"], index, op,
                    machine["start_a"], machine["wanted_a"],
                    machine["start_b"], machine["wanted_b"],
                )
            ]
            self.assertEqual(len(successes), 1)
            true_fix = (audit["true_fix"][0], tuple(audit["true_fix"][1]))
            self.assertEqual(successes[0], true_fix)
            spec = audit["spec"]
            self.assertNotEqual(spec["finished_a"], spec["wanted_a"])
            self.assertNotEqual(spec["finished_b"], spec["wanted_b"])

    def test_validator_rejects_a_non_canonical_target(self) -> None:
        rng = random.Random(909)
        row = enum_mod.enum_repair_lesson(rng, "troughline", 1)
        row["task_id"] = "erp_enum_repair_00000"
        candidates, machine = enum_mod.rederive_candidates(row["_audit"])
        wrong = candidates[row["_audit"]["k_tried"] + 1]
        broken = poisoned_copy(row)
        broken["_audit"]["target"] = [wrong[0], list(wrong[1])]
        broken["answer"] = (
            f"ANSWER: STEP {wrong[0] + 1}: {machine['describe'](wrong[1])}"
        )
        with self.assertRaises(ValueError):
            enum_mod.validate_generated([broken])

    def test_validator_rejects_a_tampered_tried_prefix(self) -> None:
        rng = random.Random(111)
        row = enum_mod.enum_repair_lesson(rng, "barrowyoke", 3)
        row["task_id"] = "erp_enum_repair_00000"
        # Swap the first two tried entries: still legal and failing, but no
        # longer canonically ordered.
        broken = poisoned_copy(row)
        tried = broken["_audit"]["tried"]
        tried[0], tried[1] = tried[1], tried[0]
        with self.assertRaises(ValueError):
            enum_mod.validate_generated([broken])

    def test_validator_rejects_a_dropped_tried_entry(self) -> None:
        rng = random.Random(222)
        row = enum_mod.enum_repair_lesson(rng, "sigilslate", 3)
        row["task_id"] = "erp_enum_repair_00000"
        broken = poisoned_copy(row)
        broken["_audit"]["tried"] = broken["_audit"]["tried"][:-1]
        with self.assertRaises(ValueError):
            enum_mod.validate_generated([broken])

    def test_validator_rejects_a_poisoned_success_index(self) -> None:
        rng = random.Random(333)
        row = enum_mod.enum_repair_lesson(rng, "skeinreel", 0)
        row["task_id"] = "erp_enum_repair_00000"
        broken = poisoned_copy(row)
        broken["_audit"]["success_index"] += 1
        with self.assertRaises(ValueError):
            enum_mod.validate_generated([broken])

    def test_validator_rejects_a_missing_canonical_statement(self) -> None:
        rng = random.Random(444)
        row = enum_mod.enum_repair_lesson(rng, "balesled", 1)
        row["task_id"] = "erp_enum_repair_00000"
        broken = poisoned_copy(row)
        broken["messages"][0]["content"] = broken["messages"][0]["content"].replace(
            enum_mod.CANONICAL_ORDER_STATEMENT, "some other order"
        )
        with self.assertRaises(ValueError):
            enum_mod.validate_generated([broken])

    def test_validator_rejects_wrong_kind(self) -> None:
        rng = random.Random(555)
        row = enum_mod.enum_repair_lesson(rng, "millround", 0)
        row["task_id"] = "erp_enum_repair_00000"
        broken = dict(row)
        broken["kind"] = "u_feedloop"
        with self.assertRaises(ValueError):
            enum_mod.validate_generated([broken])

    def test_lesson_rejects_out_of_range_k(self) -> None:
        rng = random.Random(666)
        with self.assertRaises(ValueError):
            enum_mod.enum_repair_lesson(rng, "troughline", enum_mod.DEEP_K + 1)

    def test_episode_success_turns_recorded_and_rederived(self) -> None:
        rng = random.Random(777)
        row = enum_mod.enum_repair_lesson(rng, "crankwheel", 6)
        audit = row["_audit"]
        self.assertEqual(
            audit["episode_success_turns"], audit["success_index"] + 1
        )
        self.assertEqual(
            audit["remaining_turns_after_tried"],
            audit["success_index"] - audit["k_tried"] + 1,
        )
        turns = enum_mod.episode_success_turns(audit)
        self.assertEqual(turns["from_scratch"], audit["episode_success_turns"])
        self.assertEqual(
            turns["remaining_after_tried"], audit["remaining_turns_after_tried"]
        )


class EnumerationFidelityReadoutTests(unittest.TestCase):
    """The preregistered NON-GATING mechanism readout logic."""

    @classmethod
    def setUpClass(cls):
        rng = random.Random(2026)
        cls.row = enum_mod.enum_repair_lesson(rng, "trinketcord", 3)
        cls.audit = cls.row["_audit"]
        cls.candidates, cls.machine = enum_mod.rederive_candidates(cls.audit)

    def _answer(self, candidate) -> str:
        index, op = candidate
        return f"STEP {index + 1}: {self.machine["describe"](op)}"

    def test_target_scores_all_three_booleans(self) -> None:
        k = self.audit["k_tried"]
        readout = enum_mod.enumeration_fidelity(
            self.audit, self._answer(self.candidates[k])
        )
        self.assertEqual(
            readout,
            {
                "parseable": True,
                "legal": True,
                "untried": True,
                "canonical_next": True,
            },
        )

    def test_already_tried_candidate_is_legal_but_not_untried(self) -> None:
        readout = enum_mod.enumeration_fidelity(
            self.audit, self._answer(self.candidates[0])
        )
        self.assertTrue(readout["legal"])
        self.assertFalse(readout["untried"])
        self.assertFalse(readout["canonical_next"])

    def test_legal_untried_but_out_of_order_candidate(self) -> None:
        k = self.audit["k_tried"]
        readout = enum_mod.enumeration_fidelity(
            self.audit, self._answer(self.candidates[k + 2])
        )
        self.assertTrue(readout["legal"])
        self.assertTrue(readout["untried"])
        self.assertFalse(readout["canonical_next"])

    def test_illegal_proposal_scores_false(self) -> None:
        readout = enum_mod.enumeration_fidelity(
            self.audit, "STEP 99: dance wildly"
        )
        self.assertEqual(
            readout,
            {
                "parseable": True,
                "legal": False,
                "untried": False,
                "canonical_next": False,
            },
        )

    def test_repeating_the_written_step_is_illegal(self) -> None:
        written = self.machine["written"]
        index = 0
        readout = enum_mod.enumeration_fidelity(
            self.audit, f"STEP {index + 1}: {self.machine["describe"](written[index])}"
        )
        self.assertFalse(readout["legal"])

    def test_unparseable_and_none_score_false_everywhere(self) -> None:
        for value in (None, "", "no idea", "STEP x: y"):
            readout = enum_mod.enumeration_fidelity(self.audit, value)
            self.assertEqual(
                readout,
                {
                    "parseable": False,
                    "legal": False,
                    "untried": False,
                    "canonical_next": False,
                },
                value,
            )

    def test_whitespace_tolerant_parse(self) -> None:
        k = self.audit["k_tried"]
        index, op = self.candidates[k]
        loose = f"STEP  {index + 1} :   {self.machine["describe"](op)}"
        readout = enum_mod.enumeration_fidelity(self.audit, loose)
        self.assertTrue(readout["canonical_next"])


class CorpusBalanceGuardTests(unittest.TestCase):
    def test_unbalanced_formalisms_rejected(self) -> None:
        rows = enum_mod.generate_curriculum(enum_mod.SMOKE_MIX, 31415)
        dropped = [
            row for row in rows if row["surface"] != rows[0]["surface"]
        ]
        with self.assertRaises(ValueError):
            enum_mod.check_corpus_balance(dropped)

    def test_k_cycle_uniformity_enforced_on_full_cycles(self) -> None:
        rows = enum_mod.generate_curriculum(enum_mod.HOLDOUT_MIX, 27182)
        # Poison one row's recorded k: uniformity across the cycle breaks.
        poisoned = [dict(row, _audit=dict(row["_audit"])) for row in rows]
        victim = next(
            row for row in poisoned if row["_audit"]["k_tried"] == 0
        )
        victim["_audit"]["k_tried"] = 1
        with self.assertRaises(ValueError):
            enum_mod.check_corpus_balance(poisoned)


if __name__ == "__main__":
    unittest.main()
