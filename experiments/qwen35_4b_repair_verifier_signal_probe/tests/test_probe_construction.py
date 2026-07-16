"""Unit tests for the frozen 200-item repair-verifier 2AFC construction.

Covers the fully symmetric item shape (no repair history, no attempt
narration, no provenance markers, no option self-reference), the frozen
deterministic oversample-and-filter selection (pure failure evidence on
both trials), the correct/wrong fix properties re-verified per item from
the generator's audits AND by re-executing the machine semantics, the
exact 100/100 position balance assigned deterministically from the
construction seed, the letter answer format and grading path, the
oracle-free runner input schema, the banned vocabulary scan, and the
frozen listing-collision artifact reading. Fail-closed negatives tamper
with specs, prompts, and position labels. No model is ever loaded.
"""

from __future__ import annotations

import copy
import re
import sys
import unittest
from collections import Counter
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))

import check_local as cl  # noqa: E402
import eval_local_vllm as ev  # noqa: E402
import gen_feedloop_curriculum as feedloop  # noqa: E402
import gen_local_gate as gg  # noqa: E402


# Built once for the whole module: the frozen probe set is a pure function
# of the construction seed.
POOL, SELECTED, SOURCE_ROWS, RUNNER_ROWS, SELECTION_STATS = gg.build_probe_rows()


class TestSelection(unittest.TestCase):
    def test_pool_shape(self) -> None:
        self.assertEqual(len(POOL), gg.POOL_ROWS)
        self.assertEqual(gg.POOL_ROWS, 320)
        self.assertEqual(SELECTION_STATS["pool_mix"], "feedloop=320")
        self.assertEqual(SELECTION_STATS["pool_rows"], 320)

    def test_selection_rule_pure_failure_evidence(self) -> None:
        for row in SELECTED:
            spec = row["_audit"]["spec"]
            self.assertNotEqual(gg.written_trial_two_outcome(row), spec["wanted_b"])

    def test_selection_stats_are_consistent(self) -> None:
        per_formalism = SELECTION_STATS["per_formalism"]
        self.assertEqual(set(per_formalism), set(cl.FORMALISMS))
        excluded_total = 0
        for formalism, entry in per_formalism.items():
            self.assertEqual(entry["pool"], gg.POOL_PER_FORMALISM, formalism)
            self.assertEqual(entry["selected"], cl.PER_FORMALISM, formalism)
            self.assertEqual(
                entry["pool"],
                entry["selected"]
                + entry["excluded_trial_two_success_before_quota"]
                + entry["unused_after_quota"],
                formalism,
            )
            excluded_total += entry["excluded_trial_two_success_before_quota"]
        self.assertEqual(excluded_total, SELECTION_STATS["excluded_total"])

    def test_selection_preserves_stream_order(self) -> None:
        pool_ids = {id(row): index for index, row in enumerate(POOL)}
        indices = [pool_ids[id(row)] for row in SELECTED]
        by_formalism: dict[str, list[int]] = {}
        for row, index in zip(SELECTED, indices):
            by_formalism.setdefault(row["surface"], []).append(index)
        for formalism, formalism_indices in by_formalism.items():
            self.assertEqual(
                formalism_indices, sorted(formalism_indices), formalism
            )

    def test_excluded_instances_fail_reverification(self) -> None:
        excluded = [
            row
            for row in POOL
            if gg.written_trial_two_outcome(row)
            == row["_audit"]["spec"]["wanted_b"]
        ]
        self.assertGreaterEqual(len(excluded), 1)
        for row in excluded:
            with self.assertRaises(ValueError):
                gg.reverify_fixes(row)


class TestProbeSetShape(unittest.TestCase):
    def test_row_counts_and_ids(self) -> None:
        self.assertEqual(len(SOURCE_ROWS), cl.PROBE_ROWS)
        self.assertEqual(len(RUNNER_ROWS), cl.PROBE_ROWS)
        ids = [row["task_id"] for row in SOURCE_ROWS]
        self.assertEqual(len(set(ids)), cl.PROBE_ROWS)
        self.assertEqual(ids, [f"probe77160_{i:03d}" for i in range(cl.PROBE_ROWS)])
        self.assertEqual([row["id"] for row in RUNNER_ROWS], ids)

    def test_per_formalism_balance(self) -> None:
        surfaces = Counter(row["surface"] for row in SOURCE_ROWS)
        self.assertEqual(
            surfaces,
            Counter({formalism: cl.PER_FORMALISM for formalism in cl.FORMALISMS}),
        )

    def test_exact_position_balance(self) -> None:
        positions = Counter(row["correct_position"] for row in SOURCE_ROWS)
        self.assertEqual(positions, Counter({"A": 100, "B": 100}))

    def test_per_formalism_position_split_is_13_12(self) -> None:
        for f_index, formalism in enumerate(cl.FORMALISMS):
            counts = Counter(
                row["correct_position"]
                for row in SOURCE_ROWS
                if row["surface"] == formalism
            )
            expected_a = 13 if f_index % 2 == 0 else 12
            self.assertEqual(counts["A"], expected_a, formalism)
            self.assertEqual(counts["B"], cl.PER_FORMALISM - expected_a, formalism)

    def test_position_assignment_is_deterministic(self) -> None:
        self.assertEqual(
            gg.assign_positions(SELECTED), gg.assign_positions(SELECTED)
        )
        self.assertEqual(
            gg.assign_positions(SELECTED),
            [row["correct_position"] for row in SOURCE_ROWS],
        )

    def test_kind_level_and_answer_format(self) -> None:
        for row in SOURCE_ROWS:
            self.assertEqual(row["kind"], cl.PROBE_KIND)
            self.assertIn(row["level"], (4, 5))
            self.assertIn(row["answer"], ("ANSWER: A", "ANSWER: B"))
            self.assertEqual(
                row["answer"], f"ANSWER: {row['correct_position']}"
            )

    def test_prompts_are_unique(self) -> None:
        prompts = {row["messages"][0]["content"] for row in SOURCE_ROWS}
        self.assertEqual(len(prompts), cl.PROBE_ROWS)


class TestSymmetricChoiceConstruction(unittest.TestCase):
    def test_options_map_to_true_and_wrong_fixes(self) -> None:
        for row in SOURCE_ROWS:
            correct = row["correct_position"]
            other = "B" if correct == "A" else "A"
            self.assertEqual(row["options"][correct], row["true_fix"])
            self.assertEqual(row["options"][other], row["wrong_fix"])
            self.assertNotEqual(row["true_fix"], row["wrong_fix"])

    def test_prompt_shows_both_trials_of_the_broken_run(self) -> None:
        for row in SOURCE_ROWS:
            prompt = row["messages"][0]["content"]
            self.assertIn("Trial one, starting from", prompt)
            self.assertIn("Trial two, starting from", prompt)
            self.assertEqual(prompt.count("but the run finished at"), 2)
            self.assertIn("Steps as written:", prompt)
            self.assertTrue(prompt.endswith("ANSWER: <A or B>"))

    def test_prompt_carries_no_repair_history(self) -> None:
        for row in SOURCE_ROWS:
            prompt = row["messages"][0]["content"]
            self.assertNotIn("An earlier attempt", prompt)
            self.assertNotIn("With that change", prompt)
            self.assertNotIn("a second trial ran the steps", prompt)
            self.assertNotIn(gg.GENERATION_TAIL, prompt)
            self.assertNotIn("(format: STEP <k>: <corrected step>)", prompt)

    def test_marker_token_audit_is_clean(self) -> None:
        patterns = [
            re.compile(rf"\b{re.escape(token)}\b", re.IGNORECASE)
            for token in gg.MARKER_TOKENS
        ]
        for row in SOURCE_ROWS:
            prompt = row["messages"][0]["content"]
            gg.audit_marker_tokens(prompt)
            for pattern in patterns:
                self.assertIsNone(pattern.search(prompt))

    def test_options_share_one_grammatical_form(self) -> None:
        form = re.compile(r"^[AB]\. change step \d+ to '[^']+'$")
        for row in SOURCE_ROWS:
            prompt = row["messages"][0]["content"]
            option_lines = [
                line
                for line in prompt.split("\n")
                if line.startswith("A. ") or line.startswith("B. ")
            ]
            self.assertEqual(len(option_lines), 2)
            for line in option_lines:
                self.assertRegex(line, form)

    def test_prompt_choice_lines_match_options(self) -> None:
        for row in SOURCE_ROWS:
            prompt = row["messages"][0]["content"]
            a, b = row["options"]["A"], row["options"]["B"]
            self.assertIn(
                f"\nA. change step {a['step']} to '{a['change']}'\n", prompt
            )
            self.assertIn(
                f"\nB. change step {b['step']} to '{b['change']}'\n", prompt
            )

    def test_no_option_self_reference(self) -> None:
        for row in SOURCE_ROWS:
            prompt = row["messages"][0]["content"]
            lines = prompt.split("\n")
            outside_rules = "\n".join(lines[:1] + lines[2:])
            a, b = row["options"]["A"], row["options"]["B"]
            line_a = f"A. change step {a['step']} to '{a['change']}'"
            line_b = f"B. change step {b['step']} to '{b['change']}'"
            self.assertEqual(prompt.count(line_a), 1)
            self.assertEqual(prompt.count(line_b), 1)
            for change in {a["change"], b["change"]}:
                expected = [a["change"], b["change"]].count(change)
                self.assertEqual(
                    outside_rules.count(f"'{change}'"), expected, row["task_id"]
                )

    def test_trial_two_shows_failure_evidence(self) -> None:
        for row in SOURCE_ROWS:
            audit = row["_audit"]
            self.assertIs(audit["trial_two_written_fails"], True)
            self.assertNotEqual(
                audit["finished_b_written"], audit["spec"]["wanted_b"]
            )

    def test_legality_clauses_stay_in_the_prompt(self) -> None:
        for row in SOURCE_ROWS:
            prompt = row["messages"][0]["content"]
            for clause in row["_audit"]["legality_clauses"]:
                self.assertIn(clause, prompt)

    def test_true_fix_matches_the_episode_answer(self) -> None:
        for row in SOURCE_ROWS:
            self.assertEqual(
                row["episode_answer"],
                f"ANSWER: STEP {row['true_fix']['step']}: "
                f"{row['true_fix']['change']}",
            )

    def test_generator_audits_reverified_per_item(self) -> None:
        for row in SOURCE_ROWS:
            audit = row["_audit"]
            self.assertIs(audit["truth_valid"], True)
            self.assertIs(audit["unique_after_round2"], True)
            self.assertIs(audit["wrong_in_round1"], True)
            self.assertGreaterEqual(audit["candidates_after_round1"], 2)
            self.assertIs(audit["reexecuted_fix_properties"], True)
            self.assertIs(audit["marker_tokens_clean"], True)
            extended = audit["extended_uniqueness_audit"]
            self.assertIs(extended["out_of_bound_only"], True)
            self.assertIs(extended["legality_clauses_documented"], True)

    def test_banned_vocabulary_scan_passes(self) -> None:
        gg.check_banned_vocabulary_2afc(SOURCE_ROWS)

    def test_choice_section_template_leaks_no_banned_or_marker_token(self) -> None:
        section = gg.CHOICE_SECTION_TEMPLATE.format(
            start_a="s", wanted_a="w", finished_a="f",
            start_b="s2", wanted_b="w2", finished_b="f2",
            a_step=1, a_change="x", b_step=2, b_change="y",
        )
        for token in feedloop.BANNED_PROMPT_TOKENS + gg.MARKER_TOKENS:
            self.assertIsNone(
                re.search(rf"\b{re.escape(token)}\b", section, re.IGNORECASE),
                token,
            )

    def test_listing_collision_audit_matches_flags(self) -> None:
        audit = gg.listing_collision_audit(SOURCE_ROWS)
        total = (
            audit["both_collide"]
            + audit["neither_collides"]
            + audit["true_fix_only_collides"]
            + audit["wrong_fix_only_collides"]
        )
        self.assertEqual(total, cl.PROBE_ROWS)
        expected_ceiling = (
            0.5 * (audit["both_collide"] + audit["neither_collides"])
            + max(
                audit["true_fix_only_collides"],
                audit["wrong_fix_only_collides"],
            )
        ) / cl.PROBE_ROWS
        self.assertAlmostEqual(
            audit["collision_heuristic_ceiling"], expected_ceiling
        )
        # The frozen artifact ceiling must sit clearly below the signal bar.
        self.assertLess(
            audit["collision_heuristic_ceiling"], cl.SIGNAL_MIN_ACCURACY
        )


class TestReexecutionFailsClosed(unittest.TestCase):
    def _instance(self) -> dict:
        return copy.deepcopy(SELECTED[0])

    def test_accepts_the_untampered_instance(self) -> None:
        gg.reverify_fixes(self._instance())

    def test_rejects_tampered_wanted_b(self) -> None:
        row = self._instance()
        spec = row["_audit"]["spec"]
        spec["wanted_b"] = spec["finished_b_after_wrong"]
        with self.assertRaises(ValueError):
            gg.reverify_fixes(row)

    def test_rejects_trial_two_success(self) -> None:
        row = self._instance()
        spec = row["_audit"]["spec"]
        spec["wanted_b"] = gg.written_trial_two_outcome(row)
        with self.assertRaises(ValueError):
            gg.reverify_fixes(row)

    def test_rejects_tampered_written_steps(self) -> None:
        row = self._instance()
        spec = row["_audit"]["spec"]
        spec["written"] = list(reversed(spec["written"]))
        with self.assertRaises(ValueError):
            gg.reverify_fixes(row)

    def test_rejects_coinciding_fixes(self) -> None:
        row = self._instance()
        spec = row["_audit"]["spec"]
        spec["wrong_fix"] = copy.deepcopy(spec["true_fix"])
        with self.assertRaises(ValueError):
            gg.reverify_fixes(row)

    def test_rejects_audit_step_disagreement(self) -> None:
        row = self._instance()
        row["_audit"]["bug_step"] = row["_audit"]["bug_step"] + 1
        with self.assertRaises(ValueError):
            gg.reverify_fixes(row)

    def test_build_choice_item_rejects_bad_position(self) -> None:
        with self.assertRaises(ValueError):
            gg.build_choice_item(self._instance(), "C")

    def test_build_choice_item_rejects_missing_generation_tail(self) -> None:
        row = self._instance()
        row["messages"][0]["content"] = row["messages"][0]["content"].removesuffix(
            gg.GENERATION_TAIL
        )
        with self.assertRaises(ValueError):
            gg.build_choice_item(row, "A")

    def test_parse_episode_rejects_prompt_drift(self) -> None:
        row = self._instance()
        row["messages"][0]["content"] = row["messages"][0]["content"].replace(
            "Steps as written:", "Steps listed:", 1
        )
        with self.assertRaises(ValueError):
            gg.parse_episode(row)

    def test_parse_episode_rejects_renderer_disagreement(self) -> None:
        row = self._instance()
        spec = row["_audit"]["spec"]
        # Tamper a recorded state: the re-implemented renderer's sentence no
        # longer matches the episode's own rendering.
        spec["start_a"], spec["start_b"] = spec["start_b"], spec["start_a"]
        with self.assertRaises(ValueError):
            gg.parse_episode(row)

    def test_audit_marker_tokens_rejects_provenance_words(self) -> None:
        with self.assertRaises(ValueError):
            gg.audit_marker_tokens("this candidate was tried before")
        with self.assertRaises(ValueError):
            gg.audit_marker_tokens("An earlier Attempt changed step 1")
        gg.audit_marker_tokens("a clean symmetric prompt")


class TestRunnerInput(unittest.TestCase):
    def test_schema_is_oracle_free(self) -> None:
        for row in RUNNER_ROWS:
            self.assertEqual(set(row), {"id", "messages", "meta"})
            self.assertEqual(
                set(row["meta"]), {"kind", "surface", "seed", "instrument"}
            )
            self.assertEqual(row["meta"]["seed"], cl.CONSTRUCTION_SEED)
            self.assertEqual(row["meta"]["instrument"], gg.INSTRUMENT)

    def test_messages_match_source_rows(self) -> None:
        for source, runner in zip(SOURCE_ROWS, RUNNER_ROWS):
            self.assertEqual(source["messages"], runner["messages"])


class TestLetterGrading(unittest.TestCase):
    def test_parse_and_grade_letter_answers(self) -> None:
        text = "some thinking\nANSWER: A"
        self.assertEqual(ev.parse_answer(text), "A")
        self.assertTrue(ev.grade("A", "A"))
        self.assertFalse(ev.grade("B", "A"))
        self.assertFalse(ev.grade(None, "A"))

    def test_last_answer_line_wins(self) -> None:
        text = "ANSWER: B\nreconsidering\nANSWER: A"
        self.assertEqual(ev.parse_answer(text), "A")

    def test_whitespace_is_forgiven_but_prose_is_not(self) -> None:
        self.assertTrue(ev.grade(" A ", "A"))
        self.assertFalse(ev.grade("A.", "A"))
        self.assertFalse(ev.grade("option A", "A"))

    def test_unparsed_completion_grades_incorrect(self) -> None:
        self.assertIsNone(ev.parse_answer("no final line here"))


if __name__ == "__main__":
    unittest.main()
