"""The normalized-hash runner pin (lifecycle 22's mechanism, review-hardened).

A raw hash pin on run_benchmark.py would break when the orchestrator fills
the six trained-arm TODO-pin value slots post-merge; a plain substring
contract pins declarations but not call sites. The NORMALIZED-HASH pin in
check_design.py freezes every byte of run_benchmark.py outside exactly the
six pin value slots; these tests re-run the reviewer's mutations from the
lifecycle-22 repair against the normalization and require every one of
them to change the hash — regardless of the file's live fill state (the
``_none_baseline`` fixture canonicalizes the slots back to None first).
"""

from __future__ import annotations

import importlib.util
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

    def test_normalization_canonicalizes_exactly_six_slots(self):
        normalized = cd.normalize_run_benchmark_source(self.text)
        self.assertEqual(normalized.count(cd.PIN_PLACEHOLDER), 6)
        self.assertNotIn('"replay_ctl6": None,', normalized)
        self.assertNotIn('"enum_repair": None,', normalized)
        self.assertNotIn("REPLAY_CTL6_MERGE_RECEIPT_SHA256 = None", normalized)
        self.assertNotIn("ENUM_REPAIR_MERGE_RECEIPT_SHA256 = None", normalized)

    def _none_baseline(self) -> str:
        """Canonicalize the six pin slots back to None regardless of the
        file's live fill state (the runner is legitimately pin-filled after
        the merges; these mutation fixtures need the pre-fill form)."""
        text = self.text
        for _, pattern, count in cd.PIN_SLOT_PATTERNS:
            compiled = re.compile(pattern, re.MULTILINE)

            def to_none(match):
                tail = match.group(3) if match.re.groups >= 3 else ""
                return f"{match.group(1)}None{tail}"

            text, n = compiled.subn(to_none, text)
            self.assertEqual(n, count)
        return text

    def _filled_variant(self, values: list[str]) -> str:
        """Fill the six pin slots (in pattern order) with quoted 64-hex."""
        text = self.text
        remaining = list(values)
        for _, pattern, count in cd.PIN_SLOT_PATTERNS:
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
        filled = self._filled_variant([c * 64 for c in "abcdef"])
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
                "95d61d021d382d4a3911694fba0acefb948fd529d3a69f920e7ac48d21b21e97",
                "7dd947dfd20447edd9d6e458944d4b48d326bf591a37f309cadc698cc4dee868",
                "5b741f3658ecd5db9ae42b6279b9e0715baddf8050231688c8e48de8ec7e3760",
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
            "        require_zero_root_parent_provenance(model)",
            "        pass",
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
            baseline, "REPLAY_CTL6_MERGE_RECEIPT_SHA256 = None"
        )
        with self.assertRaises(ValueError):
            cd.normalize_run_benchmark_source(mutated)
        malformed = baseline.replace(
            "ENUM_REPAIR_MERGE_RECEIPT_SHA256 = None",
            'ENUM_REPAIR_MERGE_RECEIPT_SHA256 = "not-a-sha"',
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
        for name in (
            "run.py",
            "run_benchmark.py",
            "check_design.py",
            "check_local.py",
            "eval_local_vllm.py",
            "gen_local_gate.py",
            "rebuild_clean_chain.py",
            "train_trial.py",
            "merge_trained_arm.py",
        ):
            with self.subTest(script=name):
                self.assertNotIn(
                    forbidden,
                    (SCRIPTS / name).read_text(encoding="utf-8"),
                )


class TestDesignReceiptPin(unittest.TestCase):
    def test_slot_patterns_are_the_frozen_six(self):
        self.assertEqual(len(cd.PIN_SLOT_PATTERNS), 4)
        self.assertEqual(sum(count for _, _, count in cd.PIN_SLOT_PATTERNS), 6)
        self.assertEqual(cd.PIN_PLACEHOLDER, "__ENUM_REPAIR_TODO_PIN__")

    def test_placeholder_never_appears_in_the_live_runner(self):
        text = (SCRIPTS / "run_benchmark.py").read_text(encoding="utf-8")
        self.assertNotIn(cd.PIN_PLACEHOLDER, text)


if __name__ == "__main__":
    unittest.main()
