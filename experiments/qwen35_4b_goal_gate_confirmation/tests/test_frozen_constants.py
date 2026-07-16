"""Cross-module frozen-constant contracts for the three-seed confirmation.

The seeds, tier, budget, arm order, tree hashes, receipt pins, gateway
pins, and the discovery-source pin are duplicated across the harness,
the event runner, the readout checker, and the design-receipt generator
(repo convention: each script is self-contained). These tests hold the
copies identical and equal to the frozen design (seeds 78155/78156/78157,
tier medium, think budget 1024, arms base then hygiene_explore), hold
the FAMILIES tuple byte-for-byte equal to the tier forensics' tuple so
the goal-gate statistic stays the one the discovery pass was computed
with, hold the discovery seed OUT of the verdict-counting seeds, and
hold the confirmation contracts (per-seed word-boundary audits, no
local wall-time cap, discovery source pinned by sha256, implementation
signature anchored to the discovery block).
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


harness = import_by_path("confirmation_harness", SCRIPTS / "run.py")
gateway = import_by_path(
    "trusted_gateway", ROOT / "scripts" / "run_benchmark_aggregate.py"
)

EXPECTED_SEEDS = (78155, 78156, 78157)
EXPECTED_ORDER = ("base", "hygiene_explore")
EXPECTED_TREES = {
    "base": "26d8ee48583adb0fb557d0ff668664949adff0068fa5baafe6f0af68e22fb677",
    "hygiene_explore": "9eb653d78f05546ca594a831c989fa906d12f3eb7a5a8550d1afcd6bfccc4971",
}
EXPECTED_WEIGHTS = {
    "base": "b654e033d525d87cbbd746bb681d80813c4b00d8e6202cb3edcfb6dfa3b416db",
    "hygiene_explore": "e21123443a230ada2c73ded411e0b5b7c2b1459856b2c38e4f1beea8958dc02f",
}
DISCOVERY_SUMMARY = (
    "experiments/qwen35_4b_statechain_only_dose"
    "/runs/benchmark/medium_tb1024_seed78154_pilot/summary.json"
)
DISCOVERY_SUMMARY_SHA256 = (
    "6b1a43869f013e24a048a45a04e5603b45fe59488912194eb3e76a43679255fa"
)
DISCOVERY_IMPLEMENTATION = {
    "runner_sha256": (
        "a3beecd8b5c89ccfd99a172a6d85321d39b9feb6c29d12f10b2f4d7499e273cb"
    ),
    "source_inventory_sha256": (
        "218b8615a95f24da962c931e9cd2dba58d853a7bdcd2847cd8e2c42fc2c05f42"
    ),
    "source_file_count": 56,
}
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
    def test_seeds_tier_budget_name(self) -> None:
        for module in (rb, cb, gd, harness):
            self.assertEqual(module.SEED_ORDER, EXPECTED_SEEDS)
            self.assertEqual(module.FROZEN_TIER, "medium")
            self.assertEqual(module.FROZEN_THINK_BUDGET, 1024)
            self.assertEqual(module.FROZEN_NAME, "confirmation")

    def test_seeds_are_pairwise_distinct_and_fresh_of_the_discovery(self) -> None:
        self.assertEqual(len(set(EXPECTED_SEEDS)), 3)
        self.assertNotIn(78154, EXPECTED_SEEDS)
        self.assertNotIn(cb.DISCOVERY["seed"], cb.SEED_ORDER)

    def test_model_order(self) -> None:
        for module in (rb, cb, gd, harness):
            self.assertEqual(module.MODEL_ORDER, EXPECTED_ORDER)
        self.assertEqual(rb.TREATED_ARM, "hygiene_explore")
        self.assertEqual(cb.TREATED_ARM, "hygiene_explore")
        self.assertEqual(gd.TREATED_ARM, "hygiene_explore")
        self.assertEqual(EXPECTED_ORDER.index("base"), 0)

    def test_model_paths_identical(self) -> None:
        self.assertEqual(rb.FROZEN_MODEL_PATHS, cb.FROZEN_MODEL_PATHS)
        self.assertEqual(rb.FROZEN_MODEL_PATHS, gd.FROZEN_MODEL_PATHS)
        self.assertEqual(set(rb.FROZEN_MODEL_PATHS), set(EXPECTED_ORDER))
        base = rb.FROZEN_MODEL_PATHS["base"]
        self.assertEqual(
            base.relative_to(ROOT).as_posix(),
            "large_artifacts/qwen35_4b_universal_curriculum/merged/base_reserialized",
        )
        treated = rb.FROZEN_MODEL_PATHS["hygiene_explore"]
        self.assertEqual(
            treated.relative_to(ROOT).as_posix(),
            "large_artifacts/qwen35_4b_hygiene_explore_destack_medium"
            "/merged/hygiene_explore",
        )

    def test_tree_and_weight_pins(self) -> None:
        self.assertEqual(rb.FROZEN_TREE_SHA256, EXPECTED_TREES)
        self.assertEqual(gd.FROZEN_TREE_SHA256, EXPECTED_TREES)
        self.assertEqual(cb.FROZEN_TREE_SHA256, EXPECTED_TREES)
        self.assertEqual(rb.FROZEN_WEIGHTS_SHA256, EXPECTED_WEIGHTS)
        self.assertEqual(gd.FROZEN_WEIGHTS_SHA256, EXPECTED_WEIGHTS)
        self.assertEqual(cb.FROZEN_WEIGHTS_SHA256, EXPECTED_WEIGHTS)
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
        self.assertEqual(set(rb.COMMITTED_MERGE_RECEIPTS), {"hygiene_explore"})
        for label, (relative, _) in rb.COMMITTED_MERGE_RECEIPTS.items():
            self.assertIn(f"/{label}.json", relative)

    def test_gateway_pins(self) -> None:
        self.assertEqual(rb.GATEWAY_SHA256, gd.GATEWAY_SHA256)
        self.assertEqual(rb.GATEWAY_SHA256, harness.GATEWAY_SHA256)
        self.assertEqual(rb.GATEWAY, gd.GATEWAY)

    def test_discovery_source_pins(self) -> None:
        # The confirmation is only meaningful if every module agrees on
        # the same sha-pinned committed discovery summary.
        self.assertEqual(cb.DISCOVERY["summary"], DISCOVERY_SUMMARY)
        self.assertEqual(cb.DISCOVERY["summary_sha256"], DISCOVERY_SUMMARY_SHA256)
        self.assertEqual(rb.DISCOVERY_SUMMARY, DISCOVERY_SUMMARY)
        self.assertEqual(rb.DISCOVERY_SUMMARY_SHA256, DISCOVERY_SUMMARY_SHA256)
        self.assertEqual(gd.DISCOVERY_SUMMARY, DISCOVERY_SUMMARY)
        self.assertEqual(gd.DISCOVERY_SUMMARY_SHA256, DISCOVERY_SUMMARY_SHA256)
        self.assertEqual(harness.DISCOVERY_SUMMARY, DISCOVERY_SUMMARY)
        self.assertEqual(harness.DISCOVERY_SUMMARY_SHA256, DISCOVERY_SUMMARY_SHA256)
        self.assertEqual(cb.DISCOVERY["seed"], 78154)
        self.assertEqual(gd.DISCOVERY_SEED, 78154)
        self.assertEqual(cb.DISCOVERY["tier"], cb.FROZEN_TIER)
        self.assertEqual(cb.DISCOVERY["think_budget"], cb.FROZEN_THINK_BUDGET)
        self.assertEqual(cb.DISCOVERY["treated_arm"], "hygiene_explore_parent")
        self.assertEqual(gd.DISCOVERY_TREATED_ARM, "hygiene_explore_parent")
        self.assertIs(cb.DISCOVERY["counted_in_verdict"], False)
        self.assertEqual(cb.DISCOVERY_MODEL_ORDER, gd.DISCOVERY_MODEL_ORDER)
        self.assertTrue(
            DISCOVERY_SUMMARY.startswith("experiments/qwen35_4b_statechain_only_dose/")
        )

    def test_discovery_implementation_pins(self) -> None:
        self.assertEqual(cb.DISCOVERY_IMPLEMENTATION, DISCOVERY_IMPLEMENTATION)
        self.assertEqual(rb.DISCOVERY_IMPLEMENTATION, DISCOVERY_IMPLEMENTATION)
        self.assertEqual(gd.DISCOVERY_IMPLEMENTATION, DISCOVERY_IMPLEMENTATION)
        self.assertEqual(
            cb.IMPLEMENTATION_KEYS,
            ("runner_sha256", "source_inventory_sha256", "source_file_count"),
        )

    def test_discovery_reference_loads_from_the_committed_summary(self) -> None:
        # Reads the real committed file: the loader must return the pinned
        # implementation signature, the recorded aggregates (0.3663 vs
        # 0.0800), and survive its own recorded-pass recomputation.
        reference = cb.load_discovery_reference()
        self.assertEqual(
            reference["benchmark_implementation"], DISCOVERY_IMPLEMENTATION
        )
        self.assertEqual(
            set(reference["scores"]), {"base", "hygiene_explore_parent"}
        )
        self.assertEqual(reference["scores"]["base"]["aggregate"], 0.08)
        self.assertAlmostEqual(
            reference["scores"]["hygiene_explore_parent"]["aggregate"],
            0.36634408602150537,
        )
        report = cb.discovery_report(reference["scores"])
        self.assertTrue(report["goal_gate"]["goal_gate_pass"])
        self.assertEqual(report["goal_gate"]["strict_wins"], 10)
        self.assertAlmostEqual(
            report["fragility_margins"]["menders"], 0.016666666666666666
        )
        self.assertAlmostEqual(report["fragility_margins"]["warren"], 0.05)
        self.assertIs(report["counted_in_verdict"], False)

    def test_gateway_schema_matches_trusted_gateway(self) -> None:
        self.assertEqual(rb.GATEWAY_KEYS, set(gateway.OUTPUT_KEYS))
        self.assertEqual(cb.GATEWAY_KEYS, set(gateway.OUTPUT_KEYS))
        self.assertEqual(rb.PUBLIC_FAMILIES, set(gateway.PUBLIC_FAMILY_KEYS))

    def test_event_directory_agreement(self) -> None:
        self.assertEqual(cb.EVENT_DIRS, harness.EVENT_DIRS)
        self.assertEqual(cb.EVENT_DIRS, rb.EVENT_DIRS)
        for seed in EXPECTED_SEEDS:
            self.assertEqual(
                cb.EVENT_DIRS[seed].name,
                f"medium_tb1024_seed{seed}_confirmation",
            )
        self.assertEqual(cb.READOUT, harness.READOUT)
        self.assertEqual(cb.READOUT, rb.READOUT)
        self.assertEqual(rb.LEDGER, harness.LEDGER)
        self.assertEqual(rb.LEDGER.name, "benchmark_events.jsonl")

    def test_harness_and_runner_opened_records_agree(self) -> None:
        for seed in EXPECTED_SEEDS:
            self.assertEqual(rb.opened_record(seed), harness.opened_record(seed))
            self.assertEqual(rb.opened_record(seed), cb.opened_record(seed))

    def test_closed_record_machinery_agrees_across_modules(self) -> None:
        # MAJOR-1: the runner writes the receipt-pinning closed records and
        # the checker refuses anything else — both must share one schema.
        self.assertEqual(rb.CLOSED_RECORD_KEYS, cb.CLOSED_RECORD_KEYS)
        self.assertIn("receipts", rb.CLOSED_RECORD_KEYS)
        self.assertEqual(rb.LEDGER, cb.LEDGER)
        self.assertEqual(cb.GATEWAY_SHA256, rb.GATEWAY_SHA256)
        record = {
            "name": "confirmation",
            "phase": "closed",
            "tier": "medium",
            "think_budget": 1024,
            "seed": EXPECTED_SEEDS[0],
            "summary": str(rb.EVENT_DIRS[EXPECTED_SEEDS[0]] / "summary.json"),
            "summary_sha256": "0" * 64,
            "receipts": {"base": "1" * 64, "hygiene_explore": "2" * 64},
        }
        self.assertTrue(rb.is_closed_record(record, EXPECTED_SEEDS[0]))
        self.assertTrue(cb.is_closed_record(record, EXPECTED_SEEDS[0]))
        legacy = {key: value for key, value in record.items() if key != "receipts"}
        self.assertFalse(rb.is_closed_record(legacy, EXPECTED_SEEDS[0]))
        self.assertFalse(cb.is_closed_record(legacy, EXPECTED_SEEDS[0]))

    def test_readout_is_ledger_anchored(self) -> None:
        # check_benchmark must consume the ledger, not bare receipt files.
        source = (SCRIPTS / "check_benchmark.py").read_text(encoding="utf-8")
        self.assertIn("closed_records = authenticate_ledger(ledger_rows)", source)
        self.assertIn('!= record["receipts"][label]', source)
        self.assertIn("require_summary_consistency(seed, summary, events)", source)


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

    def test_fragility_families_are_public_families(self) -> None:
        self.assertEqual(cb.FRAGILITY_FAMILIES, ("menders", "warren"))
        self.assertEqual(gd.FRAGILITY_FAMILIES, cb.FRAGILITY_FAMILIES)
        for family in cb.FRAGILITY_FAMILIES:
            self.assertIn(family, cb.FAMILIES)


class TestSeedFreshnessAudit(unittest.TestCase):
    def test_patterns_name_each_frozen_seed(self) -> None:
        self.assertEqual(gd.SEED_ORDER, EXPECTED_SEEDS)
        for seed in gd.SEED_ORDER:
            self.assertIn(str(seed), gd.audit_pattern(seed))

    def test_word_boundary_guards_exclude_substring_hits(self) -> None:
        # Substring hits inside floats and 10-digit seeds are expected
        # across the repo; each seed-context pattern must not flag them.
        for seed in gd.SEED_ORDER:
            pattern = re.compile(gd.audit_pattern(seed))
            for benign in (
                f'"seed": 42{seed}31',        # inside a longer seed
                f'"seed": {seed}9',           # longer integer, digit after
                f'"seed": 1{seed}',           # digit immediately before
                f"seed_acc 0.4{seed}0444",    # float tail, digits adjacent
            ):
                self.assertIsNone(pattern.search(benign), (seed, benign))
            for hit in (
                f'"seed": {seed}',
                f"seed={seed}",
                f"{seed} seed",
                f"the seed {seed} is consumed",
            ):
                self.assertIsNotNone(pattern.search(hit), (seed, hit))

    def test_float_with_adjacent_seed_context_still_fails_closed(self) -> None:
        # Ambiguous by construction: a bare 0.78155 immediately next to the
        # word seed keeps the seed context and must flag for human audit.
        pattern = re.compile(gd.audit_pattern(78155))
        self.assertIsNotNone(pattern.search("0.78155, seed"))

    def test_sibling_seeds_do_not_cross_match(self) -> None:
        # The 78156 pattern must never fire on a 78155 mention and vice
        # versa: each seed's freshness is audited independently.
        self.assertIsNone(re.compile(gd.audit_pattern(78156)).search('"seed": 78155'))
        self.assertIsNone(re.compile(gd.audit_pattern(78155)).search('"seed": 78156'))

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
            "[DESIGN_RECEIPT, PREREGISTRATION, BENCH_REVIEW, ROOT / DISCOVERY_SUMMARY]",
            source,
        )

    def test_resume_is_forwarded_only_on_explicit_operator_request(self) -> None:
        import inspect  # noqa: PLC0415

        self.assertIn(
            "resume", inspect.signature(harness.benchmark_stage).parameters
        )
        source = (SCRIPTS / "run.py").read_text(encoding="utf-8")
        self.assertIn('if resume:\n        command.append("--resume")', source)
        self.assertNotIn("EVENT_DIRS[seed].exists():\n        command.append", source)

    def test_write_ahead_ledger_contract(self) -> None:
        source = (SCRIPTS / "run_benchmark.py").read_text(encoding="utf-8")
        self.assertIn('"phase": "opened"', source)
        self.assertIn('"phase": "closed"', source)
        self.assertIn('plan[seed]["status"] == "fresh":\n            append_ledger', source)

    def test_no_local_wall_time_cap(self) -> None:
        # The gateway owns budget policy; the seed-consuming runner must
        # never impose its own subprocess timeout on an arm.
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

    def test_verdict_partition_labels_are_frozen(self) -> None:
        self.assertEqual(
            cb.VERDICTS, ("CONFIRMED", "AGGREGATE_ONLY", "NOT_REPLICATED")
        )
        self.assertIn("AT LEAST TWO of three", cb.VERDICT_RULE)
        self.assertIn("ALL THREE seeds", cb.VERDICT_RULE)

    def test_engine_free_source_tree(self) -> None:
        # Gateway-only rule: no experiment-local vLLM/model engine exists;
        # the only model executions go through the trusted gateway.
        self.assertFalse((EXP / "src" / "vllm_runner.py").exists())
        self.assertFalse((EXP / "tests" / "test_vllm_runner.py").exists())
        self.assertTrue((EXP / "src" / "README.md").is_file())


if __name__ == "__main__":
    unittest.main()
