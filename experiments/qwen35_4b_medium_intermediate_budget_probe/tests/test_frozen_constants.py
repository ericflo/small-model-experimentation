"""Cross-module frozen-constant contracts for the one budget-probe event.

The seed, tier, budget, arm order, tree hashes, receipt pins, gateway
pins, and the tb1024 contrast-source pin are duplicated across the
harness, the event runner, the readout checker, and the design-receipt
generator (repo convention: each script is self-contained). These tests
hold the copies identical and equal to the frozen design (seed 78153,
think budget 4096), hold the FAMILIES tuple byte-for-byte equal to the
tier forensics' tuple so the goal-gate statistic stays the venue-moving
one, and hold the budget-probe contracts (word-boundary seed audit, no
local wall-time cap, contrast source pinned by sha256).
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
import run_benchmark as rb  # noqa: E402


def import_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


harness = import_by_path("measurement_harness", SCRIPTS / "run.py")
gateway = import_by_path(
    "trusted_gateway", ROOT / "scripts" / "run_benchmark_aggregate.py"
)

EXPECTED_ORDER = ("base", "designed_fresh", "replay_repeat", "hygiene_explore")
EXPECTED_TREES = {
    "base": "26d8ee48583adb0fb557d0ff668664949adff0068fa5baafe6f0af68e22fb677",
    "designed_fresh": "93433aa2d5f3f0d6d4540126579c09feee1d8502df702c1563bae28eb7f60255",
    "replay_repeat": "4c4f3561efbcafe1b9f777f4bd21bf4949ff89177f77946d0fa0f88cafafacd7",
    "hygiene_explore": "9eb653d78f05546ca594a831c989fa906d12f3eb7a5a8550d1afcd6bfccc4971",
}
TB1024_SUMMARY = (
    "experiments/qwen35_4b_universal_medium_tier_measurement"
    "/runs/benchmark/medium_tb1024_seed78150_measurement/summary.json"
)
TB1024_SUMMARY_SHA256 = (
    "a927fc838ca8b1eaa3083d6034ba09ad0659c21a2a13b22c525487cf95a6fb43"
)
FORENSICS_ANALYZER = (
    ROOT / "experiments" / "qwen35_4b_menders_sirens_tier_forensics"
    / "scripts" / "analyze_constants.py"
)


def families_block(source: str) -> str:
    match = re.search(r"^FAMILIES = \([^)]*\)\n", source, re.MULTILINE)
    if match is None:
        raise AssertionError("FAMILIES tuple not found")
    return match.group(0)


class TestFrozenEvent(unittest.TestCase):
    def test_seed_tier_budget_name(self) -> None:
        for module in (rb, cb, gd, harness):
            self.assertEqual(module.FROZEN_SEED, 78153)
            self.assertEqual(module.FROZEN_TIER, "medium")
            self.assertEqual(module.FROZEN_THINK_BUDGET, 4096)
            self.assertEqual(module.FROZEN_NAME, "measurement")

    def test_model_order(self) -> None:
        for module in (rb, cb, gd, harness):
            self.assertEqual(module.MODEL_ORDER, EXPECTED_ORDER)
        self.assertEqual(cb.TREATED_ARMS, EXPECTED_ORDER[1:])
        self.assertEqual(gd.TREATED_ARMS, EXPECTED_ORDER[1:])

    def test_model_paths_identical(self) -> None:
        self.assertEqual(rb.FROZEN_MODEL_PATHS, cb.FROZEN_MODEL_PATHS)
        self.assertEqual(rb.FROZEN_MODEL_PATHS, gd.FROZEN_MODEL_PATHS)
        self.assertEqual(set(rb.FROZEN_MODEL_PATHS), set(EXPECTED_ORDER))
        base = rb.FROZEN_MODEL_PATHS["base"]
        self.assertEqual(
            base.relative_to(ROOT).as_posix(),
            "large_artifacts/qwen35_4b_universal_curriculum/merged/base_reserialized",
        )

    def test_tree_and_weight_pins(self) -> None:
        self.assertEqual(rb.FROZEN_TREE_SHA256, EXPECTED_TREES)
        self.assertEqual(gd.FROZEN_TREE_SHA256, EXPECTED_TREES)
        self.assertEqual(rb.FROZEN_WEIGHTS_SHA256, gd.FROZEN_WEIGHTS_SHA256)
        self.assertEqual(
            rb.FROZEN_WEIGHTS_SHA256["base"],
            "b654e033d525d87cbbd746bb681d80813c4b00d8e6202cb3edcfb6dfa3b416db",
        )
        self.assertEqual(rb.WEIGHTS_SIZE_BYTES, 9_078_620_536)
        self.assertEqual(gd.WEIGHTS_SIZE_BYTES, 9_078_620_536)

    def test_merge_receipt_pins(self) -> None:
        self.assertEqual(rb.COMMITTED_MERGE_RECEIPTS, gd.COMMITTED_MERGE_RECEIPTS)
        self.assertEqual(rb.BASE_MERGE_RECEIPT_SHA256, gd.BASE_MERGE_RECEIPT_SHA256)
        as_mapping = {
            relative: digest
            for relative, digest in rb.COMMITTED_MERGE_RECEIPTS.values()
        }
        self.assertEqual(as_mapping, harness.EXTERNAL_MERGE_RECEIPTS)
        for label, (relative, _) in rb.COMMITTED_MERGE_RECEIPTS.items():
            self.assertIn(f"/{label}.json", relative)

    def test_gateway_pins(self) -> None:
        self.assertEqual(rb.GATEWAY_SHA256, gd.GATEWAY_SHA256)
        self.assertEqual(rb.GATEWAY_SHA256, harness.GATEWAY_SHA256)
        self.assertEqual(rb.GATEWAY, gd.GATEWAY)

    def test_tb1024_contrast_source_pins(self) -> None:
        # The budget_contrast reading is only trustworthy if every module
        # agrees on the same sha-pinned committed summary.
        self.assertEqual(cb.TB1024_REFERENCE["summary"], TB1024_SUMMARY)
        self.assertEqual(cb.TB1024_REFERENCE["summary_sha256"], TB1024_SUMMARY_SHA256)
        self.assertEqual(gd.TB1024_SUMMARY, TB1024_SUMMARY)
        self.assertEqual(gd.TB1024_SUMMARY_SHA256, TB1024_SUMMARY_SHA256)
        self.assertEqual(harness.TB1024_SUMMARY, TB1024_SUMMARY)
        self.assertEqual(harness.TB1024_SUMMARY_SHA256, TB1024_SUMMARY_SHA256)
        self.assertEqual(rb.TB1024_SUMMARY, TB1024_SUMMARY)
        self.assertEqual(rb.TB1024_SUMMARY_SHA256, TB1024_SUMMARY_SHA256)
        self.assertEqual(cb.TB1024_REFERENCE["seed"], 78150)
        self.assertEqual(cb.TB1024_REFERENCE["think_budget"], 1024)
        self.assertEqual(cb.TB1024_REFERENCE["tier"], "medium")
        self.assertIs(cb.TB1024_REFERENCE["cross_seed_confound"], True)
        self.assertEqual(gd.TB1024_SEED, 78150)
        self.assertEqual(gd.TB1024_THINK_BUDGET, 1024)
        self.assertTrue(
            TB1024_SUMMARY.startswith(
                "experiments/qwen35_4b_universal_medium_tier_measurement/"
            )
        )
        self.assertNotEqual(cb.TB1024_REFERENCE["seed"], cb.FROZEN_SEED)
        self.assertNotEqual(
            cb.TB1024_REFERENCE["think_budget"], cb.FROZEN_THINK_BUDGET
        )

    def test_tb1024_reference_implementation_signature_loads(self) -> None:
        # MAJOR-2 contract: the pinned contrast source carries the
        # benchmark-implementation signature the tb4096 receipts must
        # match fail-closed at read time.
        self.assertEqual(
            cb.IMPLEMENTATION_KEYS,
            ("runner_sha256", "source_inventory_sha256", "source_file_count"),
        )
        reference = cb.load_tb1024_reference()
        self.assertEqual(
            reference["benchmark_implementation"],
            {
                "runner_sha256": (
                    "a3beecd8b5c89ccfd99a172a6d85321d39b9feb6c29d12f10b2f4d7499e273cb"
                ),
                "source_inventory_sha256": (
                    "218b8615a95f24da962c931e9cd2dba58d853a7bdcd2847cd8e2c42fc2c05f42"
                ),
                "source_file_count": 56,
            },
        )
        self.assertEqual(set(reference["scores"]), set(EXPECTED_ORDER))

    def test_gateway_schema_matches_trusted_gateway(self) -> None:
        self.assertEqual(rb.GATEWAY_KEYS, set(gateway.OUTPUT_KEYS))
        self.assertEqual(cb.GATEWAY_KEYS, set(gateway.OUTPUT_KEYS))
        self.assertEqual(rb.PUBLIC_FAMILIES, set(gateway.PUBLIC_FAMILY_KEYS))

    def test_event_directory_agreement(self) -> None:
        self.assertEqual(cb.EVENT_DIR, harness.EVENT_DIR)
        self.assertEqual(
            cb.EVENT_DIR.name, "medium_tb4096_seed78153_measurement"
        )
        self.assertEqual(cb.READOUT, harness.READOUT)


class TestFamiliesTuple(unittest.TestCase):
    def test_families_tuple_values(self) -> None:
        self.assertEqual(
            cb.FAMILIES,
            (
                "chronicle", "lockpick", "menders", "mirage", "rites",
                "siftstack", "sirens", "stockade", "toolsmith", "warren",
            ),
        )
        self.assertEqual(set(cb.FAMILIES), rb.PUBLIC_FAMILIES)
        self.assertEqual(tuple(gd.PUBLIC_FAMILIES), cb.FAMILIES)

    def test_families_tuple_byte_for_byte_from_forensics(self) -> None:
        ours = families_block(
            (SCRIPTS / "check_benchmark.py").read_text(encoding="utf-8")
        )
        theirs = families_block(FORENSICS_ANALYZER.read_text(encoding="utf-8"))
        self.assertEqual(ours, theirs)

    def test_budget_families_are_public_families(self) -> None:
        self.assertEqual(cb.BUDGET_FAMILIES, ("menders", "rites"))
        self.assertEqual(gd.BUDGET_FAMILIES, cb.BUDGET_FAMILIES)
        for family in cb.BUDGET_FAMILIES:
            self.assertIn(family, cb.FAMILIES)


class TestSeedFreshnessAudit(unittest.TestCase):
    def test_pattern_names_the_frozen_seed(self) -> None:
        self.assertIn("78153", gd.AUDIT_PATTERN)
        self.assertEqual(gd.FROZEN_SEED, 78153)

    def test_word_boundary_guards_exclude_substring_hits(self) -> None:
        # Substring hits inside floats and 10-digit seeds are expected
        # across the repo; the seed-context pattern must not flag them.
        pattern = re.compile(gd.AUDIT_PATTERN)
        for benign in (
            '"seed": 4278153331',      # 78153 inside a 10-digit seed
            '"seed": 781539',          # longer integer prefixed by 78153
            '"seed": 178153',          # digit immediately before
            'seed_acc 0.4781530444',   # float tail, seed context too far
        ):
            self.assertIsNone(pattern.search(benign), benign)
        for hit in (
            '"seed": 78153',
            "seed=78153",
            "78153 seed",
            "the seed 78153 is consumed",
        ):
            self.assertIsNotNone(pattern.search(hit), hit)

    def test_float_with_adjacent_seed_context_still_fails_closed(self) -> None:
        # Ambiguous by construction: a bare 0.78153 immediately next to the
        # word seed keeps the seed context and must flag for human audit.
        pattern = re.compile(gd.AUDIT_PATTERN)
        self.assertIsNotNone(pattern.search("0.78153, seed"))

    def test_audit_roots_are_the_knowledge_bearing_three(self) -> None:
        self.assertEqual(
            gd.AUDIT_ROOTS, ("experiments", "knowledge", "research_programs")
        )
        self.assertEqual(gd.AUDIT_SELF_WINDOW_LINES, 3)


class TestMeasurementIntakeContracts(unittest.TestCase):
    def test_harness_is_single_stage_with_verdict_gate(self) -> None:
        self.assertEqual(
            harness.BENCH_VERDICT, "**Verdict:** `PASS_BENCHMARK_EVENT`."
        )
        source = (SCRIPTS / "run.py").read_text(encoding="utf-8")
        self.assertIn('group.add_argument("--stage", choices=("benchmark",))', source)
        self.assertIn('group.add_argument("--smoke", action="store_true")', source)

    def test_seed_consuming_runner_enforces_the_review_and_code_pins(self) -> None:
        # A direct run_benchmark.py invocation must gate on the review
        # verdict and re-verify the design receipt's code pins before any
        # gateway call.
        self.assertEqual(rb.BENCH_VERDICT, harness.BENCH_VERDICT)
        self.assertEqual(rb.BENCH_REVIEW, harness.BENCH_REVIEW)
        self.assertEqual(rb.DESIGN_RECEIPT, harness.DESIGN_RECEIPT)
        source = (SCRIPTS / "run_benchmark.py").read_text(encoding="utf-8")
        self.assertIn(
            'require_verdict(BENCH_REVIEW, BENCH_VERDICT, "benchmark design review")',
            source,
        )
        self.assertIn('"gen_design_receipt.py"), "--check"', source)
        self.assertIn(
            "[DESIGN_RECEIPT, PREREGISTRATION, BENCH_REVIEW, ROOT / TB1024_SUMMARY]",
            source,
        )

    def test_resume_is_forwarded_only_on_explicit_operator_request(self) -> None:
        import inspect  # noqa: PLC0415

        self.assertIn(
            "resume", inspect.signature(harness.benchmark_stage).parameters
        )
        source = (SCRIPTS / "run.py").read_text(encoding="utf-8")
        self.assertIn('if resume:\n        command.append("--resume")', source)
        self.assertNotIn("if EVENT_DIR.exists():\n        command.append", source)

    def test_write_ahead_ledger_contract(self) -> None:
        source = (SCRIPTS / "run_benchmark.py").read_text(encoding="utf-8")
        self.assertIn('"phase": "opened"', source)
        self.assertIn('"phase": "closed"', source)
        self.assertIn("if not ledger_rows(LEDGER):", source)

    def test_no_local_wall_time_cap(self) -> None:
        # The gateway owns budget policy; the seed-consuming runner must
        # never impose its own subprocess timeout on a tb4096 arm.
        source = (SCRIPTS / "run_benchmark.py").read_text(encoding="utf-8")
        self.assertNotIn("timeout", source)

    def test_no_promotion_logic_anywhere(self) -> None:
        for name in ("run_benchmark.py", "check_benchmark.py", "run.py"):
            source = (SCRIPTS / name).read_text(encoding="utf-8")
            self.assertNotIn("promotion_gate", source)
            self.assertNotIn("passes_pilot_gate", source)

    def test_benchmark_directory_never_referenced(self) -> None:
        forbidden = "benchmarks" + "/"
        for path in sorted(SCRIPTS.glob("*.py")):
            self.assertNotIn(
                forbidden, path.read_text(encoding="utf-8"), path.name
            )


if __name__ == "__main__":
    unittest.main()
