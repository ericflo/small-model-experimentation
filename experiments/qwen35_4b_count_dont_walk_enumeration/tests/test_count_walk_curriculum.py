import json
import random
import sys
import unittest
from collections import Counter
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
MENDERS_EXP = ROOT / "experiments" / "qwen35_4b_menders_dose_scale"
ENUM_REPAIR_EXP = ROOT / "experiments" / "qwen35_4b_enumerative_repair_protocol"
sys.path.insert(0, str(EXP / "scripts"))

import gen_count_walk_curriculum as cw_mod  # noqa: E402
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
    broken = json.loads(json.dumps(cw_mod.public_row(row)))
    broken["_audit"] = json.loads(json.dumps(row["_audit"]))
    return broken


class GenerationValidityTests(unittest.TestCase):
    def test_smoke_generation_valid_and_leak_free(self) -> None:
        rows = cw_mod.generate_curriculum(cw_mod.SMOKE_MIX, 12345)
        summary = cw_mod.validate_generated(rows)
        cw_mod.check_banned_vocabulary(rows)
        cw_mod.check_corpus_balance(rows)
        self.assertEqual(summary["rows"], 16)
        self.assertEqual(set(summary["kinds"]), {"u_count_walk"})
        self.assertEqual(
            set(summary["surfaces"]), set(cw_mod.ENUM_FORMALISMS)
        )

    def test_banned_vocabulary_scan_actually_fires(self) -> None:
        rows = cw_mod.generate_curriculum(cw_mod.SMOKE_MIX, 54321)
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
                cw_mod.check_banned_vocabulary([poisoned])

    def test_scan_is_case_insensitive(self) -> None:
        rows = cw_mod.generate_curriculum(cw_mod.SMOKE_MIX, 54321)
        poisoned = poisoned_copy(rows[0])
        poisoned["messages"][0]["content"] += " MENDERS"
        with self.assertRaises(ValueError):
            cw_mod.check_banned_vocabulary([poisoned])

    def test_feedloop_machinery_is_byte_identical_to_the_menders_source(self) -> None:
        """The machinery is the menders dose-scale cell's reviewed
        generator, copied by bytes — never forked."""
        copy = EXP / "scripts" / "gen_feedloop_curriculum.py"
        source = MENDERS_EXP / "scripts" / "gen_feedloop_curriculum.py"
        self.assertEqual(copy.read_bytes(), source.read_bytes())

    def test_no_fresh_surface_tokens_and_inheritance_is_complete(self) -> None:
        self.assertEqual(cw_mod.FRESH_SURFACE_TOKENS, ())
        inherited = set(cw_mod.INHERITED_SURFACE_TOKENS)
        self.assertEqual(
            inherited,
            set(feedloop.INHERITED_SURFACE_TOKENS)
            | set(feedloop.FRESH_SURFACE_TOKENS),
        )
        self.assertFalse(inherited & set(cw_mod.BANNED_PROMPT_TOKENS))

    def test_frozen_corpus_regenerates_byte_identically(self) -> None:
        rows = cw_mod.generate_curriculum(cw_mod.ARM_MIX, 77191)
        regenerated = "".join(
            json.dumps(cw_mod.public_row(row), ensure_ascii=False) + "\n"
            for row in rows
        )
        frozen = (EXP / "data" / "sft_count_walk.jsonl").read_text(encoding="utf-8")
        self.assertEqual(regenerated, frozen)

    def test_corpus_k_distribution_and_formalism_balance(self) -> None:
        rows = cw_mod.generate_curriculum(cw_mod.ARM_MIX, 77191)
        balance = cw_mod.check_corpus_balance(rows)
        self.assertEqual(
            balance["count_walk_formalisms"],
            {formalism: 20 for formalism in cw_mod.ENUM_FORMALISMS},
        )
        self.assertEqual(
            balance["k_tried_counts"],
            {str(k): 32 for k in cw_mod.K_CYCLE},
        )
        for formalism in cw_mod.ENUM_FORMALISMS:
            self.assertEqual(
                balance["k_tried_by_formalism"][formalism],
                {str(k): 4 for k in cw_mod.K_CYCLE},
            )
        # k=0 first-candidate rows and deep-in-the-list rows both present.
        self.assertIn("0", balance["k_tried_counts"])
        self.assertIn(str(cw_mod.DEEP_K), balance["k_tried_counts"])

    def test_holdout_mix_yields_the_frozen_split(self) -> None:
        rows = cw_mod.generate_curriculum(cw_mod.HOLDOUT_MIX, 88056)
        summary = cw_mod.validate_generated(rows)
        self.assertEqual(summary["rows"], 40)
        self.assertEqual(summary["kinds"], {"u_count_walk": 40})
        self.assertEqual(
            summary["surfaces"],
            {formalism: 5 for formalism in cw_mod.ENUM_FORMALISMS},
        )
        balance = cw_mod.check_corpus_balance(rows)
        for formalism in cw_mod.ENUM_FORMALISMS:
            self.assertEqual(
                balance["k_tried_by_formalism"][formalism],
                {str(k): 1 for k in cw_mod.K_CYCLE},
            )

    def test_zero_row_overlap_with_the_formalism_sharing_predecessor(self) -> None:
        """The menders cell shares all eight formalisms; row-level
        freshness is the bar."""
        local = {
            message_bytes(row)
            for row in cw_mod.generate_curriculum(cw_mod.ARM_MIX, 77191)
        } | {
            message_bytes(row)
            for row in cw_mod.generate_curriculum(cw_mod.HOLDOUT_MIX, 88056)
        }
        for path in (
            MENDERS_EXP / "data" / "sft_feedloop_scale.jsonl",
            MENDERS_EXP / "data" / "local_tasks_seed88037.jsonl",
            ENUM_REPAIR_EXP / "data" / "sft_enum_repair.jsonl",
            ENUM_REPAIR_EXP / "data" / "local_tasks_seed88052.jsonl",
        ):
            reference_rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line
            ]
            reference = {message_bytes(row) for row in reference_rows}
            self.assertFalse(local & reference, path.name)

    def test_answer_format_is_exact_match_gradable(self) -> None:
        rows = cw_mod.generate_curriculum(cw_mod.SMOKE_MIX, 999)
        for row in rows:
            self.assertRegex(answer_of(row), r"^STEP \d+: .+$")
            self.assertNotIn("\n", row["answer"])

    def test_every_prompt_documents_the_enumeration(self) -> None:
        rows = cw_mod.generate_curriculum(cw_mod.SMOKE_MIX, 4242)
        for row in rows:
            prompt = prompt_of(row)
            self.assertIn(cw_mod.CANONICAL_ORDER_STATEMENT, prompt)
            self.assertIn("Numbered action list", prompt)
            self.assertIn(cw_mod.ASK_LINE, prompt)
            self.assertIn("First trial, starting from", prompt)
            self.assertIn("Second trial, starting from", prompt)
            # The rendered range statement sits directly after the frozen
            # rule text (the one designed prompt delta of this cell).
            statement = cw_mod.render_range_statement(
                row["_audit"]["per_step_candidate_counts"]
            )
            self.assertIn(
                f"{cw_mod.CANONICAL_ORDER_STATEMENT}\n{statement}\n", prompt
            )
            if row["_audit"]["k_tried"] == 0:
                self.assertIn(cw_mod.NONE_TRIED_LINE, prompt)
            else:
                self.assertIn(cw_mod.TRIED_HEADER, prompt)


class CanonicalNextRederivationTests(unittest.TestCase):
    def test_target_rederives_across_formalisms_and_depths(self) -> None:
        rng = random.Random(707)
        for formalism in cw_mod.ENUM_FORMALISMS:
            for k_target in (0, 3, 10):
                row = cw_mod.count_walk_lesson(rng, formalism, k_target)
                audit = row["_audit"]
                self.assertEqual(audit["k_tried"], k_target)
                candidates, machine = cw_mod.rederive_candidates(audit)
                self.assertGreaterEqual(machine["success_index"], k_target)
                target = candidates[k_target]
                self.assertEqual(
                    answer_of(row),
                    f"STEP {target[0] + 1}: {machine['describe'](target[1])}",
                )
                # verify_row_audit runs the full exhaustive re-derivation.
                cw_mod.verify_row_audit(row)

    def test_unique_both_trials_fix_and_written_fails_both(self) -> None:
        rng = random.Random(808)
        for formalism in cw_mod.ENUM_FORMALISMS:
            row = cw_mod.count_walk_lesson(rng, formalism, 1)
            audit = row["_audit"]
            candidates, machine = cw_mod.rederive_candidates(audit)
            successes = [
                (index, op)
                for index, op in candidates
                if cw_mod.repairs_both(
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
        row = cw_mod.count_walk_lesson(rng, "troughline", 1)
        row["task_id"] = "cdw_count_walk_00000"
        candidates, machine = cw_mod.rederive_candidates(row["_audit"])
        wrong = candidates[row["_audit"]["k_tried"] + 1]
        broken = poisoned_copy(row)
        broken["_audit"]["target"] = [wrong[0], list(wrong[1])]
        broken["answer"] = (
            f"ANSWER: STEP {wrong[0] + 1}: {machine['describe'](wrong[1])}"
        )
        with self.assertRaises(ValueError):
            cw_mod.validate_generated([broken])

    def test_validator_rejects_a_tampered_tried_prefix(self) -> None:
        rng = random.Random(111)
        row = cw_mod.count_walk_lesson(rng, "barrowyoke", 3)
        row["task_id"] = "cdw_count_walk_00000"
        # Swap the first two tried entries: still legal and failing, but no
        # longer canonically ordered.
        broken = poisoned_copy(row)
        tried = broken["_audit"]["tried"]
        tried[0], tried[1] = tried[1], tried[0]
        with self.assertRaises(ValueError):
            cw_mod.validate_generated([broken])

    def test_validator_rejects_a_dropped_tried_entry(self) -> None:
        rng = random.Random(222)
        row = cw_mod.count_walk_lesson(rng, "sigilslate", 3)
        row["task_id"] = "cdw_count_walk_00000"
        broken = poisoned_copy(row)
        broken["_audit"]["tried"] = broken["_audit"]["tried"][:-1]
        with self.assertRaises(ValueError):
            cw_mod.validate_generated([broken])

    def test_validator_rejects_a_poisoned_success_index(self) -> None:
        rng = random.Random(333)
        row = cw_mod.count_walk_lesson(rng, "skeinreel", 0)
        row["task_id"] = "cdw_count_walk_00000"
        broken = poisoned_copy(row)
        broken["_audit"]["success_index"] += 1
        with self.assertRaises(ValueError):
            cw_mod.validate_generated([broken])

    def test_validator_rejects_a_missing_canonical_statement(self) -> None:
        rng = random.Random(444)
        row = cw_mod.count_walk_lesson(rng, "balesled", 1)
        row["task_id"] = "cdw_count_walk_00000"
        broken = poisoned_copy(row)
        broken["messages"][0]["content"] = broken["messages"][0]["content"].replace(
            cw_mod.CANONICAL_ORDER_STATEMENT, "some other order"
        )
        with self.assertRaises(ValueError):
            cw_mod.validate_generated([broken])

    def test_validator_rejects_wrong_kind(self) -> None:
        rng = random.Random(555)
        row = cw_mod.count_walk_lesson(rng, "millround", 0)
        row["task_id"] = "cdw_count_walk_00000"
        broken = dict(row)
        broken["kind"] = "u_feedloop"
        with self.assertRaises(ValueError):
            cw_mod.validate_generated([broken])

    def test_lesson_rejects_out_of_range_k(self) -> None:
        rng = random.Random(666)
        with self.assertRaises(ValueError):
            cw_mod.count_walk_lesson(rng, "troughline", cw_mod.DEEP_K + 1)

    def test_episode_success_turns_recorded_and_rederived(self) -> None:
        rng = random.Random(777)
        row = cw_mod.count_walk_lesson(rng, "crankwheel", 6)
        audit = row["_audit"]
        self.assertEqual(
            audit["episode_success_turns"], audit["success_index"] + 1
        )
        self.assertEqual(
            audit["remaining_turns_after_tried"],
            audit["success_index"] - audit["k_tried"] + 1,
        )
        turns = cw_mod.episode_success_turns(audit)
        self.assertEqual(turns["from_scratch"], audit["episode_success_turns"])
        self.assertEqual(
            turns["remaining_after_tried"], audit["remaining_turns_after_tried"]
        )


class EnumerationFidelityReadoutTests(unittest.TestCase):
    """The preregistered NON-GATING mechanism readout logic."""

    @classmethod
    def setUpClass(cls):
        rng = random.Random(2026)
        cls.row = cw_mod.count_walk_lesson(rng, "trinketcord", 3)
        cls.audit = cls.row["_audit"]
        cls.candidates, cls.machine = cw_mod.rederive_candidates(cls.audit)

    def _answer(self, candidate) -> str:
        index, op = candidate
        return f"STEP {index + 1}: {self.machine["describe"](op)}"

    def test_target_scores_all_three_booleans(self) -> None:
        k = self.audit["k_tried"]
        readout = cw_mod.enumeration_fidelity(
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
        readout = cw_mod.enumeration_fidelity(
            self.audit, self._answer(self.candidates[0])
        )
        self.assertTrue(readout["legal"])
        self.assertFalse(readout["untried"])
        self.assertFalse(readout["canonical_next"])

    def test_legal_untried_but_out_of_order_candidate(self) -> None:
        k = self.audit["k_tried"]
        readout = cw_mod.enumeration_fidelity(
            self.audit, self._answer(self.candidates[k + 2])
        )
        self.assertTrue(readout["legal"])
        self.assertTrue(readout["untried"])
        self.assertFalse(readout["canonical_next"])

    def test_illegal_proposal_scores_false(self) -> None:
        readout = cw_mod.enumeration_fidelity(
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
        readout = cw_mod.enumeration_fidelity(
            self.audit, f"STEP {index + 1}: {self.machine["describe"](written[index])}"
        )
        self.assertFalse(readout["legal"])

    def test_unparseable_and_none_score_false_everywhere(self) -> None:
        for value in (None, "", "no idea", "STEP x: y"):
            readout = cw_mod.enumeration_fidelity(self.audit, value)
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
        readout = cw_mod.enumeration_fidelity(self.audit, loose)
        self.assertTrue(readout["canonical_next"])


class CorpusBalanceGuardTests(unittest.TestCase):
    def test_unbalanced_formalisms_rejected(self) -> None:
        rows = cw_mod.generate_curriculum(cw_mod.SMOKE_MIX, 31415)
        dropped = [
            row for row in rows if row["surface"] != rows[0]["surface"]
        ]
        with self.assertRaises(ValueError):
            cw_mod.check_corpus_balance(dropped)

    def test_k_cycle_uniformity_enforced_on_full_cycles(self) -> None:
        rows = cw_mod.generate_curriculum(cw_mod.HOLDOUT_MIX, 27182)
        # Poison one row's recorded k: uniformity across the cycle breaks.
        poisoned = [dict(row, _audit=dict(row["_audit"])) for row in rows]
        victim = next(
            row for row in poisoned if row["_audit"]["k_tried"] == 0
        )
        victim["_audit"]["k_tried"] = 1
        with self.assertRaises(ValueError):
            cw_mod.check_corpus_balance(poisoned)


class ThinkBudgetAndShapeTests(unittest.TestCase):
    """The count-don't-walk expression contract: every training think
    target sits under the frozen budget, in the frozen five-line shape,
    with token cost constant in k — tested per row."""

    @classmethod
    def setUpClass(cls):
        cls.rows = cw_mod.generate_curriculum(cw_mod.ARM_MIX, 77191)
        cls.holdout = cw_mod.generate_curriculum(cw_mod.HOLDOUT_MIX, 88056)

    def test_every_training_think_target_under_the_frozen_caps(self) -> None:
        for row in self.rows + self.holdout:
            with self.subTest(task=row["task_id"]):
                cw_mod.check_think_budget(row["think"])
                self.assertLessEqual(len(row["think"]), cw_mod.THINK_CHAR_CAP)
                self.assertLessEqual(
                    row["n_think_tokens"], cw_mod.THINK_TOKEN_CAP
                )

    def test_constant_five_line_shape_in_every_row(self) -> None:
        import re

        for row in self.rows + self.holdout:
            lines = row["think"].split("\n")
            self.assertEqual(len(lines), cw_mod.THINK_LINE_COUNT)
            for line, pattern in zip(lines, cw_mod.THINK_LINE_PATTERNS):
                self.assertIsNotNone(re.fullmatch(pattern, line), line)

    def test_think_cost_is_constant_in_k(self) -> None:
        """A walker's cost grows by hundreds of characters between k=0
        and k=10; the compact computation's spread is digits only."""
        by_k = {}
        for row in self.rows:
            by_k.setdefault(row["_audit"]["k_tried"], []).append(
                len(row["think"])
            )
        mean = {k: sum(v) / len(v) for k, v in by_k.items()}
        self.assertLess(abs(mean[cw_mod.DEEP_K] - mean[0]), 20)
        self.assertLess(
            max(max(v) for v in by_k.values())
            - min(min(v) for v in by_k.values()),
            120,
        )

    def test_think_is_the_pure_function_of_machine_and_k(self) -> None:
        for row in self.holdout:
            candidates, machine = cw_mod.rederive_candidates(row["_audit"])
            self.assertEqual(
                row["think"],
                cw_mod.build_think(
                    machine["grammar"],
                    machine["written"],
                    row["_audit"]["k_tried"],
                    machine["describe"],
                ),
            )

    def test_think_ends_by_emitting_the_answer(self) -> None:
        for row in self.rows:
            self.assertTrue(
                row["think"].endswith(row["answer"].removeprefix("ANSWER: "))
            )

    def test_validator_rejects_a_tampered_think(self) -> None:
        row = next(r for r in self.holdout if r["_audit"]["k_tried"] == 3)
        broken = poisoned_copy(row)
        broken["think"] = broken["think"].replace(
            "Tried entries: 3.", "Tried entries: 4."
        )
        with self.assertRaises(ValueError):
            cw_mod.validate_generated([broken])

    def test_validator_rejects_a_walk_shaped_think(self) -> None:
        row = self.holdout[0]
        broken = poisoned_copy(row)
        broken["think"] = (
            "Step 1 is untried. Slot 1 is already what step 1 reads. "
            "Slot 2 is untried, so I will check it next. " * 20
        ).strip()
        with self.assertRaises(ValueError):
            cw_mod.validate_generated([broken])

    def test_budget_guard_fires_on_an_overlong_conforming_think(self) -> None:
        padded = (
            "Tried entries: 1.\n"
            "Target: change number 2 in the frozen order.\n"
            "Number 2 sits in step 1's range 1-17: offset 2 - 1 + 1 = 2.\n"
            "Step 1's written action is list number 3; skipping it, offset "
            "2 is list number 2: " + "x" * 400 + ".\n"
            "STEP 1: x"
        )
        with self.assertRaises(ValueError):
            cw_mod.check_think_budget(padded)

    def test_real_tokenizer_budget_certified_by_the_frozen_receipt(self) -> None:
        """The REAL token bound is enforced by measure_source_tokens.py;
        once its receipt exists, every treatment row's think span must
        sit under the frozen cap."""
        receipt_path = EXP / "data" / "source_token_lengths.json"
        if not receipt_path.is_file():
            self.skipTest("source token receipt not materialized yet")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        budget = receipt["think_token_budget"]
        self.assertEqual(budget["cap"], cw_mod.THINK_TOKEN_CAP)
        self.assertEqual(budget["enforced_on"], "count_walk")
        self.assertTrue(budget["fail_closed_per_row"])
        lengths = [
            item["think_target"]
            for item in receipt["sources"]["count_walk"]["lengths"]
        ]
        self.assertEqual(len(lengths), 160)
        self.assertLessEqual(max(lengths), cw_mod.THINK_TOKEN_CAP)
        self.assertEqual(budget["max_think_target_tokens"], max(lengths))


class RenderedRangeTests(unittest.TestCase):
    """The order statement's rendered per-step candidate counts must
    equal the generator's own exhaustive enumeration exactly."""

    @classmethod
    def setUpClass(cls):
        cls.rows = cw_mod.generate_curriculum(cw_mod.SMOKE_MIX, 6161)

    def test_rendered_ranges_equal_the_enumeration_exactly(self) -> None:
        for row in self.rows:
            audit = row["_audit"]
            candidates, machine = cw_mod.rederive_candidates(audit)
            counts = cw_mod.per_step_candidate_counts(
                machine["grammar"], machine["written"]
            )
            self.assertEqual(audit["per_step_candidate_counts"], counts)
            cw_mod.verify_ranges_against_enumeration(
                counts, candidates, len(machine["written"])
            )
            ranges = cw_mod.step_ranges(counts)
            self.assertEqual(ranges[-1][1], len(candidates))
            for index, (start, end) in enumerate(ranges):
                block = candidates[start - 1 : end]
                self.assertTrue(
                    all(candidate_index == index for candidate_index, _ in block)
                )

    def test_per_step_counts_follow_the_generic_rule(self) -> None:
        for row in self.rows:
            candidates, machine = cw_mod.rederive_candidates(row["_audit"])
            grammar, written = machine["grammar"], machine["written"]
            for index, count in enumerate(
                row["_audit"]["per_step_candidate_counts"]
            ):
                expected = (
                    len(grammar) - 1 if written[index] in grammar else len(grammar)
                )
                self.assertEqual(count, expected)

    def test_range_statement_wording_matches_the_frozen_example(self) -> None:
        self.assertEqual(
            cw_mod.render_range_statement([7, 7, 7]),
            "In that order, step 1 offers 7 changes (numbers 1-7); "
            "step 2 offers 7 (numbers 8-14); step 3 offers 7 "
            "(numbers 15-21) — 21 changes in all.",
        )

    def test_locate_candidate_and_slot_arithmetic(self) -> None:
        counts = [7, 7, 7]
        self.assertEqual(cw_mod.locate_candidate(counts, 1), (0, 1))
        self.assertEqual(cw_mod.locate_candidate(counts, 7), (0, 7))
        self.assertEqual(cw_mod.locate_candidate(counts, 8), (1, 1))
        self.assertEqual(cw_mod.locate_candidate(counts, 21), (2, 7))
        with self.assertRaises(ValueError):
            cw_mod.locate_candidate(counts, 22)
        grammar = [("a",), ("b",), ("c",), ("d",)]
        # offset skips the written action's slot
        self.assertEqual(
            cw_mod.offset_to_list_number(grammar, ("b",), 1), 1
        )
        self.assertEqual(
            cw_mod.offset_to_list_number(grammar, ("b",), 2), 3
        )
        self.assertEqual(
            cw_mod.offset_to_list_number(grammar, ("a",), 1), 2
        )
        with self.assertRaises(ValueError):
            cw_mod.offset_to_list_number(grammar, ("a",), 4)

    def test_validator_rejects_a_tampered_range_statement(self) -> None:
        row = self.rows[0]
        statement = cw_mod.render_range_statement(
            row["_audit"]["per_step_candidate_counts"]
        )
        broken = poisoned_copy(row)
        broken["messages"][0]["content"] = broken["messages"][0][
            "content"
        ].replace(statement, statement.replace("offers", "gives"))
        with self.assertRaises(ValueError):
            cw_mod.validate_generated([broken])

    def test_validator_rejects_tampered_per_step_counts(self) -> None:
        row = self.rows[0]
        broken = poisoned_copy(row)
        broken["_audit"]["per_step_candidate_counts"] = [
            count + 1
            for count in broken["_audit"]["per_step_candidate_counts"]
        ]
        with self.assertRaises(ValueError):
            cw_mod.validate_generated([broken])

    def test_range_verifier_rejects_a_shifted_range(self) -> None:
        row = self.rows[0]
        candidates, machine = cw_mod.rederive_candidates(row["_audit"])
        counts = list(row["_audit"]["per_step_candidate_counts"])
        counts[0] -= 1
        counts[1] += 1
        with self.assertRaises(ValueError):
            cw_mod.verify_ranges_against_enumeration(
                counts, candidates, len(machine["written"])
            )



if __name__ == "__main__":
    unittest.main()
