"""The normalized-hash fill-slot pins (lifecycle 22's mechanism).

A raw hash pin on a fill-slot file would break when the orchestrator fills
its TODO-pin value slots post-merge; a plain substring contract pins
declarations but not call sites. The NORMALIZED-HASH pins in
check_design.py freeze every byte of run_benchmark.py (three slots),
train_trial.py (the PUBLISHED_ARM_HASHES value slot), and
eval_local_vllm.py (the EXPECTED_TRAINED_TREE_SHA256 value slot) outside
exactly those slots; these tests re-run the reviewer's mutations from the
lifecycle-22 repair against the normalization and require every one of
them to change the hash — regardless of the file's live fill state (the
``_none_baseline`` fixture canonicalizes the slots back to None first) —
and probe the two new pins symmetrically (one-byte non-slot edit changes
the hash; a legal slot fill does not).
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
SCRIPTS = EXP / "scripts"
sys.path.insert(0, str(SCRIPTS))

import check_design as cd  # noqa: E402


def _delete_line_containing(text: str, needle: str) -> str:
    lines = text.splitlines(keepends=True)
    kept = [line for line in lines if needle not in line]
    assert len(kept) == len(lines) - 1, f"expected exactly one line: {needle!r}"
    return "".join(kept)


def _delete_block(text: str, start_needle: str, end_stripped: str) -> str:
    lines = text.splitlines(keepends=True)
    start = next(i for i, line in enumerate(lines) if start_needle in line)
    end = next(
        i for i in range(start + 1, len(lines)) if lines[i].strip() == end_stripped
    )
    return "".join(lines[:start] + lines[end + 1:])


class TestNormalizedRunnerPin(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = (SCRIPTS / "run_benchmark.py").read_text(encoding="utf-8")

    def test_frozen_normalized_hash_matches_the_file_on_disk(self):
        self.assertEqual(
            cd.normalized_runner_sha256(self.text),
            cd.RUN_BENCHMARK_NORMALIZED_SHA256,
        )

    def test_normalization_canonicalizes_exactly_three_slots(self):
        normalized = cd.normalize_run_benchmark_source(self.text)
        self.assertEqual(normalized.count(cd.PIN_PLACEHOLDER), 3)
        self.assertNotIn('"replay_compound": None,', normalized)
        self.assertNotIn(
            "REPLAY_COMPOUND_MERGE_RECEIPT_SHA256 = None", normalized
        )

    def _none_baseline(self) -> str:
        """Canonicalize the three pin slots back to None regardless of the
        file's live fill state (the runner is legitimately pin-filled after
        the merge; these mutation fixtures need the pre-fill form)."""
        text = self.text
        for _, pattern, count in cd.PIN_SLOT_PATTERNS["run_benchmark.py"]:
            compiled = re.compile(pattern, re.MULTILINE)

            def to_none(match):
                tail = match.group(3) if match.re.groups >= 3 else ""
                return f"{match.group(1)}None{tail}"

            text, n = compiled.subn(to_none, text)
            self.assertEqual(n, count)
        return text

    def _filled_variant(self, values: list[str]) -> str:
        """Fill the three pin slots (in pattern order) with quoted 64-hex."""
        text = self.text
        remaining = list(values)
        for _, pattern, count in cd.PIN_SLOT_PATTERNS["run_benchmark.py"]:
            compiled = re.compile(pattern, re.MULTILINE)

            def fill(match):
                value = remaining.pop(0)
                tail = match.group(3) if match.re.groups >= 3 else ""
                return f'{match.group(1)}"{value}"{tail}'

            text, n = compiled.subn(fill, text)
            self.assertEqual(n, count)
        self.assertFalse(remaining)
        return text

    def test_none_pins_and_filled_pins_normalize_to_the_same_hash(self):
        baseline = self._none_baseline()
        filled = self._filled_variant([c * 64 for c in "abc"])
        self.assertNotEqual(filled, baseline)
        self.assertEqual(
            cd.normalize_run_benchmark_source(filled),
            cd.normalize_run_benchmark_source(baseline),
        )
        self.assertEqual(
            cd.normalized_runner_sha256(filled),
            cd.RUN_BENCHMARK_NORMALIZED_SHA256,
        )
        self.assertEqual(
            cd.normalized_runner_sha256(baseline),
            cd.RUN_BENCHMARK_NORMALIZED_SHA256,
        )

    def test_filling_with_realistic_distinct_hexes_leaves_the_hash_unchanged(self):
        filled = self._filled_variant(
            [
                "9eb653d78f05546ca594a831c989fa906d12f3eb7a5a8550d1afcd6bfccc4971",
                "e21123443a230ada2c73ded411e0b5b7c2b1459856b2c38e4f1beea8958dc02f",
                "22a22a68234de68314064b809352e7449c59ef821235402b66ecb6e5ebcc486a",
            ]
        )
        self.assertEqual(
            cd.normalized_runner_sha256(filled),
            cd.RUN_BENCHMARK_NORMALIZED_SHA256,
        )

    def test_reviewer_mutation_a_deleting_todo_pin_gate_changes_the_hash(self):
        mutated = _delete_line_containing(
            self.text, "        require_todo_pins_filled()"
        )
        self.assertNotEqual(
            cd.normalized_runner_sha256(mutated),
            cd.RUN_BENCHMARK_NORMALIZED_SHA256,
        )

    def test_reviewer_mutation_b_deleting_verdict_gate_changes_the_hash(self):
        mutated = _delete_line_containing(
            self.text,
            'require_verdict(BENCH_REVIEW, BENCH_VERDICT, "benchmark design review")',
        )
        self.assertNotEqual(
            cd.normalized_runner_sha256(mutated),
            cd.RUN_BENCHMARK_NORMALIZED_SHA256,
        )

    def test_reviewer_mutation_c_deleting_clean_main_block_changes_the_hash(self):
        mutated = _delete_block(self.text, "require_clean_pushed_main(", ")")
        self.assertNotEqual(
            cd.normalized_runner_sha256(mutated),
            cd.RUN_BENCHMARK_NORMALIZED_SHA256,
        )

    def test_reviewer_mutation_d_deleting_promotion_gate_changes_the_hash(self):
        mutated = _delete_line_containing(
            self.text,
            "        promotion = authenticate_local_promotion(args.candidate)",
        )
        self.assertNotEqual(
            cd.normalized_runner_sha256(mutated),
            cd.RUN_BENCHMARK_NORMALIZED_SHA256,
        )

    def test_reviewer_mutation_e_deleting_ledger_guard_changes_the_hash(self):
        mutated = _delete_line_containing(
            self.text,
            "        require_unconsumed_ledger(LEDGER, opened_record, args.resume)",
        )
        self.assertNotEqual(
            cd.normalized_runner_sha256(mutated),
            cd.RUN_BENCHMARK_NORMALIZED_SHA256,
        )

    def test_reviewer_mutation_f_neutralizing_provenance_changes_the_hash(self):
        mutated = self.text.replace(
            "        require_count_walk_parent_provenance(model)",
            "        pass",
            1,
        )
        self.assertNotEqual(mutated, self.text)
        self.assertNotEqual(
            cd.normalized_runner_sha256(mutated),
            cd.RUN_BENCHMARK_NORMALIZED_SHA256,
        )

    def test_reviewer_mutation_g_neutralizing_reconciliation_changes_the_hash(self):
        mutated = self.text.replace(
            "        if result.read_bytes() != rendered:",
            "        if False:",
            1,
        )
        self.assertNotEqual(mutated, self.text)
        self.assertNotEqual(
            cd.normalized_runner_sha256(mutated),
            cd.RUN_BENCHMARK_NORMALIZED_SHA256,
        )

    def test_one_byte_drift_anywhere_else_changes_the_hash(self):
        mutated = self.text.replace(
            "CONDITIONAL on local promotion", "CONDITIONAl on local promotion", 1
        )
        self.assertNotEqual(mutated, self.text)
        self.assertNotEqual(
            cd.normalized_runner_sha256(mutated),
            cd.RUN_BENCHMARK_NORMALIZED_SHA256,
        )

    def test_a_drifted_pin_slot_fails_closed_instead_of_hashing(self):
        baseline = self._none_baseline()
        mutated = _delete_line_containing(
            baseline, "REPLAY_COMPOUND_MERGE_RECEIPT_SHA256 = None"
        )
        with self.assertRaises(ValueError):
            cd.normalize_run_benchmark_source(mutated)
        malformed = baseline.replace(
            "REPLAY_COMPOUND_MERGE_RECEIPT_SHA256 = None",
            'REPLAY_COMPOUND_MERGE_RECEIPT_SHA256 = "not-a-sha"',
            1,
        )
        with self.assertRaises(ValueError):
            cd.normalize_run_benchmark_source(malformed)

    def test_call_site_contracts_are_present_as_belt_and_braces(self):
        for contract in cd.RUN_BENCHMARK_CALL_SITE_CONTRACTS:
            with self.subTest(contract=contract.strip()[:50]):
                self.assertIn(contract, self.text)

    def test_no_benchmark_suite_reads_anywhere(self):
        forbidden = "benchmarks" + "/"
        for name in cd.BENCHMARK_READ_FORBIDDEN_SCRIPTS:
            with self.subTest(script=name):
                self.assertNotIn(
                    forbidden,
                    (SCRIPTS / name).read_text(encoding="utf-8"),
                )


class TestDesignReceiptPin(unittest.TestCase):
    def test_slot_patterns_are_the_frozen_set(self):
        self.assertEqual(
            set(cd.PIN_SLOT_PATTERNS),
            {"run_benchmark.py", "train_trial.py", "eval_local_vllm.py"},
        )
        self.assertEqual(set(cd.NORMALIZED_PIN_SHA256), set(cd.PIN_SLOT_PATTERNS))
        counts = {
            name: sum(count for _, _, count in slots)
            for name, slots in cd.PIN_SLOT_PATTERNS.items()
        }
        self.assertEqual(
            counts,
            {
                "run_benchmark.py": 3,
                "train_trial.py": 1,
                "eval_local_vllm.py": 1,
            },
        )
        self.assertEqual(cd.PIN_PLACEHOLDER, "__REPLAY_COMPOUND_TODO_PIN__")
        self.assertEqual(
            cd.RUN_BENCHMARK_NORMALIZED_SHA256,
            cd.NORMALIZED_PIN_SHA256["run_benchmark.py"],
        )

    def test_placeholder_never_appears_in_any_live_pinned_file(self):
        for name in cd.PIN_SLOT_PATTERNS:
            with self.subTest(file=name):
                text = (SCRIPTS / name).read_text(encoding="utf-8")
                self.assertNotIn(cd.PIN_PLACEHOLDER, text)


class TestPinSymmetry(unittest.TestCase):
    """The train_trial / eval_local_vllm normalized pins, probed like the
    runner's: the frozen hash matches disk, a one-byte NON-slot edit changes
    the hash (fails --check), a legal slot fill does not, and a drifted slot
    fails closed instead of hashing."""

    TRAIN = "train_trial.py"
    EVAL = "eval_local_vllm.py"

    @classmethod
    def setUpClass(cls):
        cls.texts = {
            name: (SCRIPTS / name).read_text(encoding="utf-8")
            for name in (cls.TRAIN, cls.EVAL)
        }

    def test_frozen_normalized_hashes_match_the_files_on_disk(self):
        for name, text in self.texts.items():
            with self.subTest(file=name):
                self.assertEqual(
                    cd.normalized_pinned_sha256(name, text),
                    cd.NORMALIZED_PIN_SHA256[name],
                )

    def test_one_byte_non_slot_edit_changes_the_train_trial_hash(self):
        mutated = self.texts[self.TRAIN].replace(
            "TRAINING_SEED = 86", "TRAINING_SEED = 87", 1
        )
        self.assertNotEqual(mutated, self.texts[self.TRAIN])
        self.assertNotEqual(
            cd.normalized_pinned_sha256(self.TRAIN, mutated),
            cd.NORMALIZED_PIN_SHA256[self.TRAIN],
        )

    def test_deleting_the_parent_provenance_guard_changes_the_train_trial_hash(self):
        mutated = self.texts[self.TRAIN].replace(
            "        parent_sibling_note = check_parent_provenance()",
            "        parent_sibling_note = 'skipped'",
            1,
        )
        self.assertNotEqual(mutated, self.texts[self.TRAIN])
        self.assertNotEqual(
            cd.normalized_pinned_sha256(self.TRAIN, mutated),
            cd.NORMALIZED_PIN_SHA256[self.TRAIN],
        )

    def test_one_byte_non_slot_edit_changes_the_eval_hash(self):
        mutated = self.texts[self.EVAL].replace(
            "SCREEN_SEEDS = (88060, 88061, 88062)",
            "SCREEN_SEEDS = (88060, 88061, 88063)",
            1,
        )
        self.assertNotEqual(mutated, self.texts[self.EVAL])
        self.assertNotEqual(
            cd.normalized_pinned_sha256(self.EVAL, mutated),
            cd.NORMALIZED_PIN_SHA256[self.EVAL],
        )

    def test_neutralizing_the_ledger_guard_changes_the_eval_hash(self):
        mutated = self.texts[self.EVAL].replace(
            "        ledger_index = require_local_ledger_reconciled(LOCAL_LEDGER)",
            "        ledger_index = 0",
            1,
        )
        self.assertNotEqual(mutated, self.texts[self.EVAL])
        self.assertNotEqual(
            cd.normalized_pinned_sha256(self.EVAL, mutated),
            cd.NORMALIZED_PIN_SHA256[self.EVAL],
        )

    def test_64_hex_slot_fill_leaves_the_eval_hash_unchanged(self):
        filled = self.texts[self.EVAL].replace(
            '    "replay_compound": None,\n}\nANSWER_RE',
            '    "replay_compound": "' + "d" * 64 + '",\n}\nANSWER_RE',
            1,
        )
        self.assertNotEqual(filled, self.texts[self.EVAL])
        self.assertEqual(
            cd.normalized_pinned_sha256(self.EVAL, filled),
            cd.NORMALIZED_PIN_SHA256[self.EVAL],
        )

    def test_single_line_dict_slot_fill_leaves_the_train_trial_hash_unchanged(self):
        fill = (
            '{"adapter_config": "' + "a" * 64 + '", '
            '"adapter_weights": "' + "b" * 64 + '", '
            '"log": "' + "c" * 64 + '", '
            '"receipt": "' + "d" * 64 + '"}'
        )
        filled = self.texts[self.TRAIN].replace(
            '    "replay_compound": None,\n}\nADAPTER_ROOT',
            f'    "replay_compound": {fill},\n}}\nADAPTER_ROOT',
            1,
        )
        self.assertNotEqual(filled, self.texts[self.TRAIN])
        self.assertEqual(
            cd.normalized_pinned_sha256(self.TRAIN, filled),
            cd.NORMALIZED_PIN_SHA256[self.TRAIN],
        )

    def test_malformed_train_trial_fill_fails_closed(self):
        # A multi-line or unsorted fill is NOT the frozen fill form: the
        # slot no longer matches and normalization must refuse to hash.
        malformed = self.texts[self.TRAIN].replace(
            '    "replay_compound": None,\n}\nADAPTER_ROOT',
            '    "replay_compound": {"receipt": "' + "d" * 64 + '"},\n}\nADAPTER_ROOT',
            1,
        )
        self.assertNotEqual(malformed, self.texts[self.TRAIN])
        with self.assertRaises(ValueError):
            cd.normalize_pinned_source(self.TRAIN, malformed)

    def test_deleted_eval_slot_fails_closed(self):
        mutated = self.texts[self.EVAL].replace(
            '    "replay_compound": None,\n}\nANSWER_RE',
            "}\nANSWER_RE",
            1,
        )
        self.assertNotEqual(mutated, self.texts[self.EVAL])
        with self.assertRaises(ValueError):
            cd.normalize_pinned_source(self.EVAL, mutated)


if __name__ == "__main__":
    unittest.main()
