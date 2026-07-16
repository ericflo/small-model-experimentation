"""Cross-module frozen-constant contracts for the zero-root measurement.

The seed, tier, budget, arm order, tree hashes, receipt pins, gateway
pin, manifest pin, and the consequence strings are duplicated across the
harness, the event runner, the readout checker, the rebuild script, and
the design-receipt generator (repo convention: each script is
self-contained). These tests hold the copies identical and equal to the
frozen design (seed 78159, tier medium, think budget 1024, arms base
then hygiene_explore_original then zero_root_hygiene_explore), hold the
FAMILIES tuple byte-for-byte equal to the tier forensics' tuple so the
goal-gate statistic stays the one the original sweeps were computed
with, hold the base/original pins equal to the goal-gate confirmation
cell's (the events being contrasted against), and hold the reference
implementation signature equal to the discovery block.
"""

from __future__ import annotations

import importlib.util
import re
import sys
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
SCRIPTS = EXP / "scripts"
sys.path.insert(0, str(SCRIPTS))

import check_benchmark as cb  # noqa: E402
import gen_design_receipt as gd  # noqa: E402
import rebuild_zero_root as rz  # noqa: E402
import run_benchmark as rb  # noqa: E402


def import_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


harness = import_by_path("zero_root_harness", SCRIPTS / "run.py")
forensics = import_by_path(
    "tier_forensics_constants",
    ROOT / "experiments" / "qwen35_4b_menders_sirens_tier_forensics"
    / "scripts" / "analyze_constants.py",
)
goal_gate_rb = import_by_path(
    "goal_gate_run_benchmark",
    ROOT / "experiments" / "qwen35_4b_goal_gate_confirmation"
    / "scripts" / "run_benchmark.py",
)

EXPECTED_SEED = 78159
EXPECTED_ORDER = (
    "base", "hygiene_explore_original", "zero_root_hygiene_explore"
)
SHA_RE = re.compile(r"[0-9a-f]{64}")


class TestEventConstants(unittest.TestCase):
    def test_frozen_event_identity_across_modules(self):
        for module in (rb, cb, gd, harness):
            with self.subTest(module=module.__name__):
                self.assertEqual(module.FROZEN_NAME, "zero_root")
                self.assertEqual(module.FROZEN_TIER, "medium")
                self.assertEqual(module.FROZEN_THINK_BUDGET, 1024)
                self.assertEqual(module.FROZEN_SEED, EXPECTED_SEED)
                self.assertEqual(tuple(module.MODEL_ORDER), EXPECTED_ORDER)

    def test_event_paths_agree(self):
        self.assertEqual(rb.EVENT_DIR, cb.EVENT_DIR)
        self.assertEqual(rb.EVENT_DIR, harness.EVENT_DIR)
        self.assertEqual(rb.READOUT, cb.READOUT)
        self.assertEqual(rb.READOUT, harness.READOUT)
        self.assertEqual(rb.LEDGER, cb.LEDGER)
        self.assertEqual(rb.LEDGER, harness.LEDGER)
        self.assertEqual(
            rb.EVENT_DIR.name, f"medium_tb1024_seed{EXPECTED_SEED}_zero_root"
        )

    def test_model_paths_agree_across_modules(self):
        for label in EXPECTED_ORDER:
            self.assertEqual(
                rb.FROZEN_MODEL_PATHS[label], cb.FROZEN_MODEL_PATHS[label]
            )
            self.assertEqual(
                rb.FROZEN_MODEL_PATHS[label], gd.FROZEN_MODEL_PATHS[label]
            )
        self.assertEqual(
            rb.FROZEN_MODEL_PATHS["zero_root_hygiene_explore"],
            ROOT / "large_artifacts" / EXP.name / "merged"
            / "zero_root_hygiene_explore",
        )

    def test_gateway_pin_matches_the_gateway_on_disk(self):
        gateway = ROOT / "scripts" / "run_benchmark_aggregate.py"
        digest = rb.sha256_file(gateway)
        for module in (rb, cb, gd, harness):
            self.assertEqual(module.GATEWAY_SHA256, digest)


class TestInheritedArmPins(unittest.TestCase):
    """Base and the original composite pins must equal the goal-gate cell's:
    they are the same published arms the original sweeps measured."""

    def test_base_pins_equal_the_goal_gate_confirmation_pins(self):
        self.assertEqual(
            rb.FROZEN_TREE_SHA256["base"],
            goal_gate_rb.FROZEN_TREE_SHA256["base"],
        )
        self.assertEqual(
            rb.FROZEN_WEIGHTS_SHA256["base"],
            goal_gate_rb.FROZEN_WEIGHTS_SHA256["base"],
        )
        self.assertEqual(
            rb.FROZEN_MODEL_PATHS["base"],
            goal_gate_rb.FROZEN_MODEL_PATHS["base"],
        )
        self.assertEqual(
            rb.BASE_MERGE_RECEIPT_SHA256, goal_gate_rb.BASE_MERGE_RECEIPT_SHA256
        )

    def test_original_arm_pins_equal_the_goal_gate_confirmation_pins(self):
        self.assertEqual(
            rb.FROZEN_TREE_SHA256["hygiene_explore_original"],
            goal_gate_rb.FROZEN_TREE_SHA256["hygiene_explore"],
        )
        self.assertEqual(
            rb.FROZEN_WEIGHTS_SHA256["hygiene_explore_original"],
            goal_gate_rb.FROZEN_WEIGHTS_SHA256["hygiene_explore"],
        )
        self.assertEqual(
            rb.FROZEN_MODEL_PATHS["hygiene_explore_original"],
            goal_gate_rb.FROZEN_MODEL_PATHS["hygiene_explore"],
        )
        self.assertEqual(
            rb.COMMITTED_MERGE_RECEIPTS["hygiene_explore_original"][:2],
            goal_gate_rb.COMMITTED_MERGE_RECEIPTS["hygiene_explore"],
        )

    def test_reference_implementation_is_the_discovery_block(self):
        self.assertEqual(
            rb.REFERENCE_IMPLEMENTATION, goal_gate_rb.DISCOVERY_IMPLEMENTATION
        )
        self.assertEqual(rb.REFERENCE_IMPLEMENTATION, cb.REFERENCE_IMPLEMENTATION)
        self.assertEqual(rb.REFERENCE_IMPLEMENTATION, gd.REFERENCE_IMPLEMENTATION)

    def test_weights_size_matches_the_goal_gate_constant(self):
        self.assertEqual(rb.WEIGHTS_SIZE_BYTES, goal_gate_rb.WEIGHTS_SIZE_BYTES)
        self.assertEqual(rb.WEIGHTS_SIZE_BYTES, gd.WEIGHTS_SIZE_BYTES)

    def test_pin_dicts_agree_between_checker_and_designer(self):
        for label in ("base", "hygiene_explore_original"):
            self.assertEqual(
                rb.FROZEN_TREE_SHA256[label], cb.FROZEN_TREE_SHA256[label]
            )
            self.assertEqual(
                rb.FROZEN_TREE_SHA256[label], gd.FROZEN_TREE_SHA256[label]
            )
            self.assertEqual(
                rb.FROZEN_WEIGHTS_SHA256[label], cb.FROZEN_WEIGHTS_SHA256[label]
            )
            self.assertEqual(
                rb.FROZEN_WEIGHTS_SHA256[label], gd.FROZEN_WEIGHTS_SHA256[label]
            )


class TestFamilies(unittest.TestCase):
    def test_families_tuple_is_byte_identical_to_the_tier_forensics(self):
        self.assertEqual(cb.FAMILIES, forensics.FAMILIES)

    def test_public_family_sets_agree(self):
        self.assertEqual(set(cb.FAMILIES), rb.PUBLIC_FAMILIES)
        self.assertEqual(set(cb.FAMILIES), set(gd.PUBLIC_FAMILIES))

    def test_margin_families(self):
        self.assertEqual(cb.MARGIN_FAMILIES, ("menders", "rites", "warren"))
        self.assertEqual(cb.MARGIN_FAMILIES, tuple(gd.MARGIN_FAMILIES))


class TestConsequenceStrings(unittest.TestCase):
    def test_consequence_partition_is_frozen(self):
        self.assertEqual(
            cb.CONSEQUENCES, ("ZERO_ROOT_COMPARABLE", "ZERO_ROOT_DEGRADED")
        )
        self.assertEqual(cb.CONSEQUENCES, gd.CONSEQUENCES)
        self.assertEqual(cb.CONSEQUENCE_RULE, gd.CONSEQUENCE_RULE)
        self.assertEqual(cb.CONSEQUENCE_STATEMENTS, gd.CONSEQUENCE_STATEMENTS)
        self.assertEqual(
            cb.PREFIX_CONTRIBUTION_FRAMING, gd.PREFIX_CONTRIBUTION_FRAMING
        )

    def test_frozen_statement_texts(self):
        self.assertIn(
            "contamination-clean end-to-end",
            cb.CONSEQUENCE_STATEMENTS["ZERO_ROOT_COMPARABLE"],
        )
        self.assertIn(
            "load-bearing at medium",
            cb.CONSEQUENCE_STATEMENTS["ZERO_ROOT_DEGRADED"],
        )
        self.assertEqual(
            cb.PREFIX_CONTRIBUTION_FRAMING,
            "the gym-era root's contribution at medium, one seed, cross-arm "
            "same-seed paired",
        )


class TestLineagePins(unittest.TestCase):
    def test_manifest_pin_agrees_across_modules(self):
        self.assertEqual(rz.MANIFEST_SHA256, gd.LINEAGE_MANIFEST_SHA256)
        self.assertEqual(rz.MANIFEST_SHA256, harness.MANIFEST_SHA256)

    def test_stage_dirnames_agree_with_the_manifest(self):
        manifest = rz.load_manifest()
        names = tuple(rz.stage_dirname(row) for row in manifest["stages"])
        self.assertEqual(names, harness.STAGE_DIRNAMES)
        self.assertEqual(
            tuple(f"runs/lineage/{name}.json" for name in names),
            rb.ZERO_ROOT_STAGE_RECEIPTS,
        )

    def test_merge_receipt_paths_agree(self):
        self.assertEqual(rz.MERGE_RECEIPT, harness.MERGE_RECEIPT)
        self.assertEqual(rz.MERGE_RECEIPT, rb.ZERO_ROOT_MERGE_RECEIPT)
        self.assertEqual(rz.MERGE_RECEIPT, cb.ZERO_ROOT_MERGE_RECEIPT)
        self.assertEqual(rz.MERGED_OUT, harness.MERGED_OUT)
        self.assertEqual(
            rz.MERGED_OUT, rb.FROZEN_MODEL_PATHS["zero_root_hygiene_explore"]
        )


class TestSeedAudit(unittest.TestCase):
    def test_audit_pattern_catches_seed_contexts(self):
        pattern = re.compile(gd.audit_pattern(EXPECTED_SEED))
        self.assertIsNotNone(pattern.search("seed 78159"))
        self.assertIsNotNone(pattern.search('"seed": 78159,'))
        self.assertIsNotNone(pattern.search("--seed=78159"))
        self.assertIsNotNone(pattern.search("78159 seed"))

    def test_audit_pattern_excludes_substring_hits(self):
        pattern = re.compile(gd.audit_pattern(EXPECTED_SEED))
        self.assertIsNone(pattern.search("0.489781595766544"))
        self.assertIsNone(pattern.search("d3ef8...b78159f0 checksum"))
        self.assertIsNone(pattern.search("value=178159 seedless"))
        self.assertIsNone(pattern.search("seed 781590"))

    def test_verdict_strings(self):
        self.assertEqual(harness.REBUILD_VERDICT, "**Verdict:** `PASS_REBUILD`.")
        self.assertEqual(
            harness.BENCH_VERDICT, "**Verdict:** `PASS_BENCHMARK_EVENT`."
        )
        self.assertEqual(rb.BENCH_VERDICT, harness.BENCH_VERDICT)


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
    """The load-bearing runner control (review regression, 2026-07-16).

    The prior substring-contract mechanism pinned declarations but ZERO
    call sites: a runner with require_todo_pins_filled / require_verdict
    / require_clean_pushed_main / ledger_plan / append_ledger(opened) /
    require_zero_root_provenance deleted still passed --check. The
    NORMALIZED-HASH pin freezes every byte of run_benchmark.py outside
    exactly the three TODO-pin value slots; these tests re-run the
    reviewer's mutations against the normalization and require every one
    of them to change the hash.
    """

    @classmethod
    def setUpClass(cls):
        cls.text = (SCRIPTS / "run_benchmark.py").read_text(encoding="utf-8")

    def test_frozen_normalized_hash_matches_the_file_on_disk(self):
        self.assertEqual(
            gd.normalized_runner_sha256(self.text),
            gd.RUN_BENCHMARK_NORMALIZED_SHA256,
        )

    def test_normalization_canonicalizes_exactly_three_slots(self):
        normalized = gd.normalize_run_benchmark_source(self.text)
        self.assertEqual(normalized.count(gd.PIN_PLACEHOLDER), 3)
        self.assertNotIn('"zero_root_hygiene_explore": None,', normalized)
        self.assertNotIn("ZERO_ROOT_MERGE_RECEIPT_SHA256 = None", normalized)

    def _none_baseline(self) -> str:
        """Canonicalize the three pin slots back to None regardless of the
        file's live fill state (the runner is legitimately pin-filled after
        the merge; these mutation fixtures need the pre-fill form)."""
        text = self.text
        for _, pattern, count in gd.PIN_SLOT_PATTERNS:
            compiled = re.compile(pattern, re.MULTILINE)

            def to_none(match):
                tail = match.group(3) if match.re.groups >= 3 else ""
                return f"{match.group(1)}None{tail}"

            text, n = compiled.subn(to_none, text)
            self.assertEqual(n, count)
        return text

    def _filled_variant(self, values: list[str]) -> str:
        """Fill the three pin slots (in file order) with quoted 64-hex."""
        text = self.text
        remaining = list(values)
        for _, pattern, count in gd.PIN_SLOT_PATTERNS:
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
        filled = self._filled_variant(["a" * 64, "b" * 64, "c" * 64])
        self.assertNotEqual(filled, self.text)
        self.assertNotIn("ZERO_ROOT_MERGE_RECEIPT_SHA256 = None", filled)
        self.assertNotIn('"zero_root_hygiene_explore": None,', filled)
        self.assertEqual(
            gd.normalize_run_benchmark_source(filled),
            gd.normalize_run_benchmark_source(self.text),
        )
        self.assertEqual(
            gd.normalized_runner_sha256(filled),
            gd.RUN_BENCHMARK_NORMALIZED_SHA256,
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
            gd.normalized_runner_sha256(filled),
            gd.RUN_BENCHMARK_NORMALIZED_SHA256,
        )

    def test_reviewer_mutation_a_deleting_todo_pin_gate_changes_the_hash(self):
        mutated = _delete_line_containing(
            self.text, "        require_todo_pins_filled()"
        )
        self.assertNotEqual(
            gd.normalized_runner_sha256(mutated),
            gd.RUN_BENCHMARK_NORMALIZED_SHA256,
        )

    def test_reviewer_mutation_b_deleting_verdict_gate_changes_the_hash(self):
        mutated = _delete_line_containing(
            self.text,
            'require_verdict(BENCH_REVIEW, BENCH_VERDICT, "benchmark design review")',
        )
        self.assertNotEqual(
            gd.normalized_runner_sha256(mutated),
            gd.RUN_BENCHMARK_NORMALIZED_SHA256,
        )

    def test_reviewer_mutation_c_deleting_clean_main_block_changes_the_hash(self):
        mutated = _delete_block(
            self.text, "require_clean_pushed_main(", ")"
        )
        self.assertNotEqual(
            gd.normalized_runner_sha256(mutated),
            gd.RUN_BENCHMARK_NORMALIZED_SHA256,
        )

    def test_reviewer_mutation_d_deleting_opened_record_changes_the_hash(self):
        mutated = _delete_line_containing(
            self.text, "        append_ledger(opened_record())"
        )
        self.assertNotEqual(
            gd.normalized_runner_sha256(mutated),
            gd.RUN_BENCHMARK_NORMALIZED_SHA256,
        )

    def test_reviewer_mutation_e_deleting_ledger_plan_changes_the_hash(self):
        mutated = _delete_line_containing(
            self.text, "plan = ledger_plan(ledger_rows(LEDGER), args.resume)"
        )
        self.assertNotEqual(
            gd.normalized_runner_sha256(mutated),
            gd.RUN_BENCHMARK_NORMALIZED_SHA256,
        )

    def test_reviewer_mutation_f_neutralizing_provenance_changes_the_hash(self):
        mutated = self.text.replace(
            "        require_zero_root_provenance()", "        pass", 1
        )
        self.assertNotEqual(mutated, self.text)
        self.assertNotEqual(
            gd.normalized_runner_sha256(mutated),
            gd.RUN_BENCHMARK_NORMALIZED_SHA256,
        )

    def test_one_byte_drift_anywhere_else_changes_the_hash(self):
        mutated = self.text.replace("MEASUREMENT of a provenance", "MEASUREMENt of a provenance", 1)
        self.assertNotEqual(mutated, self.text)
        self.assertNotEqual(
            gd.normalized_runner_sha256(mutated),
            gd.RUN_BENCHMARK_NORMALIZED_SHA256,
        )

    def test_a_drifted_pin_slot_fails_closed_instead_of_hashing(self):
        baseline = self._none_baseline()
        mutated = _delete_line_containing(
            baseline, "ZERO_ROOT_MERGE_RECEIPT_SHA256 = None"
        )
        with self.assertRaises(ValueError):
            gd.normalize_run_benchmark_source(mutated)
        malformed = baseline.replace(
            "ZERO_ROOT_MERGE_RECEIPT_SHA256 = None",
            'ZERO_ROOT_MERGE_RECEIPT_SHA256 = "not-a-sha"',
            1,
        )
        with self.assertRaises(ValueError):
            gd.normalize_run_benchmark_source(malformed)

    def test_call_site_contracts_are_present_as_belt_and_braces(self):
        for contract in gd.RUN_BENCHMARK_CALL_SITE_CONTRACTS:
            with self.subTest(contract=contract.strip()[:50]):
                self.assertIn(contract, self.text)

    def test_no_benchmark_suite_reads_anywhere(self):
        forbidden = "benchmarks" + "/"
        for name in (
            "run.py", "run_benchmark.py", "check_benchmark.py",
            "gen_design_receipt.py", "rebuild_zero_root.py",
        ):
            with self.subTest(script=name):
                self.assertNotIn(
                    forbidden,
                    (SCRIPTS / name).read_text(encoding="utf-8"),
                )


if __name__ == "__main__":
    unittest.main()
