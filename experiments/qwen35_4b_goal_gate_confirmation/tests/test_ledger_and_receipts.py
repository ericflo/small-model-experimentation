"""Unit tests for the K-SEED write-ahead ledger and gateway-receipt gating.

The ledger contract generalizes the reference one-seed pilots to three
sealed seeds: each seed is spent when it OPENS (a write-ahead ``opened``
record lands before its first gateway call) and closes with its own
``closed`` record after its per-seed summary. The closed record sha-pins
the summary AND both per-arm gateway receipts (MAJOR-1: the verdict
inputs are provenance-anchored at close time — a record without those
pins is not a closed record). The only valid history is a prefix of the
canonical seed-major sequence; a closed record refuses its seed forever
(completed seeds are never re-run); when all three seeds are closed the
whole event refuses, resume or not; a trailing opened record is a
crashed seed that may only continue under an explicit ``--resume`` whose
opened record matches the frozen per-seed record exactly; and seeds are
independent — a crash at seed two never re-opens closed seed one and
never touches fresh seed three. Recovery is total (MAJOR-2/MINOR-1): a
crash between the summary write and the closed append recovers only via
byte-identical deterministic regeneration (divergence refuses with both
digests), and an UNOPENED seed refuses to run over pre-existing event
files (clean slate). Receipt authentication is fail-closed and
SEED-SCOPED: a receipt with any
drifted field (seed, tier, budget, backend, model, family set,
merge-receipt hash) or any non-finite / out-of-range score must be
rejected — a NaN compares unequal to everything and would otherwise
silently drop its family from the strict-win partition. As in the
hardened reference: ``within_budget`` must be a strict bool but FALSE IS
ACCEPTED (the budget_integrity reading scopes the paired comparison;
scores are still recorded), and ``wall_seconds`` must be a finite
non-negative number.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))

import check_benchmark as cb  # noqa: E402
import run_benchmark as rb  # noqa: E402


SEED_1, SEED_2, SEED_3 = rb.SEED_ORDER


def opened(seed: int) -> dict:
    return {
        "name": rb.FROZEN_NAME,
        "phase": "opened",
        "seed": seed,
        "think_budget": rb.FROZEN_THINK_BUDGET,
        "tier": rb.FROZEN_TIER,
    }


def receipt_shas() -> dict:
    return {"base": "1" * 64, "hygiene_explore": "2" * 64}


def closed(seed: int, sha: str = "0" * 64) -> dict:
    return {
        "name": rb.FROZEN_NAME,
        "phase": "closed",
        "tier": rb.FROZEN_TIER,
        "think_budget": rb.FROZEN_THINK_BUDGET,
        "seed": seed,
        "summary": str(rb.EVENT_DIRS[seed] / "summary.json"),
        "summary_sha256": sha,
        "receipts": receipt_shas(),
    }


class TestLedgerRecords(unittest.TestCase):
    def test_opened_record_matches_the_frozen_shape(self) -> None:
        for seed in rb.SEED_ORDER:
            self.assertEqual(rb.opened_record(seed), opened(seed))

    def test_closed_record_recognizer(self) -> None:
        for seed in rb.SEED_ORDER:
            self.assertTrue(rb.is_closed_record(closed(seed), seed))

    def test_closed_record_recognizer_fails_closed(self) -> None:
        base = closed(SEED_1)
        for drift in (
            {"seed": SEED_2},
            {"tier": "quick"},
            {"think_budget": 4096},
            {"name": "other"},
            {"phase": "opened"},
            {"summary": str(rb.EVENT_DIRS[SEED_2] / "summary.json")},
            {"summary_sha256": "not-a-hash"},
            {"summary_sha256": "0" * 63},
            {"summary_sha256": None},
        ):
            self.assertFalse(rb.is_closed_record({**base, **drift}, SEED_1), str(drift))
        extra = dict(base)
        extra["surprise"] = True
        self.assertFalse(rb.is_closed_record(extra, SEED_1))
        missing = {key: value for key, value in base.items() if key != "summary"}
        self.assertFalse(rb.is_closed_record(missing, SEED_1))
        self.assertFalse(rb.is_closed_record(None, SEED_1))
        self.assertFalse(rb.is_closed_record(["closed"], SEED_1))
        # A record built for seed two never authenticates as seed one.
        self.assertFalse(rb.is_closed_record(closed(SEED_2), SEED_1))

    def test_closed_record_must_pin_both_receipt_shas(self) -> None:
        # MAJOR-1 contract: the verdict inputs are provenance-anchored at
        # close time; a closed record without well-formed per-arm receipt
        # pins is not a closed record.
        base = closed(SEED_1)
        legacy = {key: value for key, value in base.items() if key != "receipts"}
        self.assertFalse(rb.is_closed_record(legacy, SEED_1))
        for bad_receipts in (
            None,
            "not-a-dict",
            {},
            {"base": "1" * 64},  # missing hygiene_explore
            {"base": "1" * 64, "hygiene_explore": "2" * 64, "extra": "3" * 64},
            {"base": "1" * 64, "other_arm": "2" * 64},
            {"base": "not-hex", "hygiene_explore": "2" * 64},
            {"base": "1" * 63, "hygiene_explore": "2" * 64},
            {"base": None, "hygiene_explore": "2" * 64},
        ):
            row = {**base, "receipts": bad_receipts}
            self.assertFalse(rb.is_closed_record(row, SEED_1), str(bad_receipts))


class TestLedgerPlan(unittest.TestCase):
    def test_empty_ledger_allows_a_fresh_event_without_resume(self) -> None:
        plan = rb.ledger_plan([], resume=False)
        self.assertEqual(set(plan), set(rb.SEED_ORDER))
        for seed in rb.SEED_ORDER:
            self.assertEqual(plan[seed], {"status": "fresh", "closed": None})

    def test_crashed_first_seed_refuses_without_resume(self) -> None:
        with self.assertRaises(ValueError):
            rb.ledger_plan([opened(SEED_1)], resume=False)

    def test_crashed_first_seed_continues_only_under_resume(self) -> None:
        plan = rb.ledger_plan([opened(SEED_1)], resume=True)
        self.assertEqual(plan[SEED_1]["status"], "crashed")
        self.assertEqual(plan[SEED_2]["status"], "fresh")
        self.assertEqual(plan[SEED_3]["status"], "fresh")

    def test_closed_seed_is_skipped_never_rerun(self) -> None:
        rows = [opened(SEED_1), closed(SEED_1)]
        plan = rb.ledger_plan(rows, resume=True)
        self.assertEqual(plan[SEED_1]["status"], "closed")
        self.assertEqual(plan[SEED_1]["closed"], closed(SEED_1))
        self.assertEqual(plan[SEED_2]["status"], "fresh")

    def test_partial_completion_refuses_without_resume(self) -> None:
        with self.assertRaises(ValueError):
            rb.ledger_plan([opened(SEED_1), closed(SEED_1)], resume=False)

    def test_cross_seed_independence_under_mid_event_crash(self) -> None:
        # Seed one closed, seed two crashed: seed one is skipped (never
        # re-run), seed two resumes, seed three stays fresh.
        rows = [opened(SEED_1), closed(SEED_1), opened(SEED_2)]
        plan = rb.ledger_plan(rows, resume=True)
        self.assertEqual(plan[SEED_1]["status"], "closed")
        self.assertEqual(plan[SEED_2]["status"], "crashed")
        self.assertEqual(plan[SEED_3]["status"], "fresh")

    def test_fully_closed_event_refuses_forever_even_with_resume(self) -> None:
        rows = [
            opened(SEED_1), closed(SEED_1),
            opened(SEED_2), closed(SEED_2),
            opened(SEED_3), closed(SEED_3),
        ]
        for resume in (False, True):
            with self.assertRaises(ValueError):
                rb.ledger_plan(rows, resume=resume)

    def test_rows_beyond_the_three_seed_event_refuse(self) -> None:
        rows = [
            opened(SEED_1), closed(SEED_1),
            opened(SEED_2), closed(SEED_2),
            opened(SEED_3), closed(SEED_3),
            opened(SEED_1),
        ]
        with self.assertRaises(ValueError):
            rb.ledger_plan(rows, resume=True)

    def test_mismatched_opened_record_refuses_even_with_resume(self) -> None:
        for drift in (
            {"seed": SEED_1 + 100},
            {"tier": "quick"},
            {"think_budget": 4096},
            {"name": "other"},
            {"phase": "closed"},
        ):
            with self.assertRaises(ValueError, msg=str(drift)):
                rb.ledger_plan([{**opened(SEED_1), **drift}], resume=True)

    def test_out_of_order_seed_refuses(self) -> None:
        # The canonical history is seed-major: seed two may not open first,
        # and may not open while seed one is still crashed.
        with self.assertRaises(ValueError):
            rb.ledger_plan([opened(SEED_2)], resume=True)
        with self.assertRaises(ValueError):
            rb.ledger_plan([opened(SEED_1), opened(SEED_2)], resume=True)

    def test_duplicate_opened_records_refuse(self) -> None:
        with self.assertRaises(ValueError):
            rb.ledger_plan([opened(SEED_1), opened(SEED_1)], resume=True)

    def test_legacy_record_without_phase_refuses(self) -> None:
        legacy = {key: value for key, value in closed(SEED_1).items() if key != "phase"}
        with self.assertRaises(ValueError):
            rb.ledger_plan([legacy], resume=True)

    def test_closed_without_a_preceding_opened_refuses(self) -> None:
        with self.assertRaises(ValueError):
            rb.ledger_plan([closed(SEED_1)], resume=True)

    def test_malformed_closed_record_refuses(self) -> None:
        rows = [opened(SEED_1), {**closed(SEED_1), "summary_sha256": "xyz"}]
        with self.assertRaises(ValueError):
            rb.ledger_plan(rows, resume=True)

    def test_ledger_rows_reader_skips_blank_lines(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            ledger = Path(scratch) / "benchmark_events.jsonl"
            self.assertEqual(rb.ledger_rows(ledger), [])
            ledger.write_text("\n  \n", encoding="utf-8")
            self.assertEqual(rb.ledger_rows(ledger), [])
            ledger.write_text(
                json.dumps(opened(SEED_1), sort_keys=True) + "\n\n",
                encoding="utf-8",
            )
            self.assertEqual(rb.ledger_rows(ledger), [opened(SEED_1)])


class TestCrashRecoveryAndCleanSlate(unittest.TestCase):
    def test_identical_summary_bytes_recover_silently(self) -> None:
        # MAJOR-2 contract: a crash between the summary write and the
        # closed append recovers when the deterministic regeneration is
        # byte-identical to the file on disk.
        rendered = b'{\n "a": 1\n}\n'
        with tempfile.TemporaryDirectory() as scratch:
            summary = Path(scratch) / "summary.json"
            summary.write_bytes(rendered)
            rb.reconcile_crashed_summary(SEED_1, summary, rendered)

    def test_divergent_summary_bytes_refuse_with_both_digests(self) -> None:
        rendered = b'{\n "a": 1\n}\n'
        tampered = b'{\n "a": 2\n}\n'
        with tempfile.TemporaryDirectory() as scratch:
            summary = Path(scratch) / "summary.json"
            summary.write_bytes(tampered)
            with self.assertRaises(ValueError) as caught:
                rb.reconcile_crashed_summary(SEED_1, summary, rendered)
            message = str(caught.exception)
            self.assertIn(hashlib.sha256(tampered).hexdigest(), message)
            self.assertIn(hashlib.sha256(rendered).hexdigest(), message)

    def test_clean_slate_is_empty_for_missing_or_empty_directory(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            missing = Path(scratch) / "never_created"
            self.assertEqual(rb.stale_event_files(missing), [])
            empty = Path(scratch) / "empty"
            empty.mkdir()
            self.assertEqual(rb.stale_event_files(empty), [])

    def test_clean_slate_names_every_stale_event_file(self) -> None:
        # MINOR-1 contract: receipts, failure receipts, and summaries that
        # predate a seed's opened record are all stale; unrelated files are
        # not the clean-slate rule's business.
        with tempfile.TemporaryDirectory() as scratch:
            event_dir = Path(scratch) / "event"
            event_dir.mkdir()
            (event_dir / "base.json").write_text("{}\n", encoding="utf-8")
            (event_dir / "hygiene_explore.failure.json").write_text(
                "{}\n", encoding="utf-8"
            )
            (event_dir / "summary.json").write_text("{}\n", encoding="utf-8")
            (event_dir / "unrelated.txt").write_text("x\n", encoding="utf-8")
            self.assertEqual(
                rb.stale_event_files(event_dir),
                ["base.json", "hygiene_explore.failure.json", "summary.json"],
            )

    def test_runner_refuses_stale_files_for_unopened_seeds(self) -> None:
        # Source-level contract: the clean-slate refusal guards the fresh
        # path unconditionally (before the opened record is appended).
        source = (EXP / "scripts" / "run_benchmark.py").read_text(encoding="utf-8")
        self.assertIn("stale = stale_event_files(output_dir)", source)
        self.assertIn("unopened seeds require a", source)


def receipt_payload(model: Path, seed: int) -> dict:
    merge_receipt = model / "merge_receipt.json"
    return {
        "schema_version": 1,
        "stage": "menagerie_aggregate_gateway",
        "tier": rb.FROZEN_TIER,
        "think_budget": rb.FROZEN_THINK_BUDGET,
        "seed": seed,
        "backend": "qwen_vllm",
        "model": str(model),
        "model_merge_receipt_sha256": hashlib.sha256(
            merge_receipt.read_bytes()
        ).hexdigest(),
        "benchmark_runner_sha256": "a" * 64,
        "benchmark_source_inventory_sha256": "b" * 64,
        "benchmark_source_file_count": 100,
        "aggregate": 0.25,
        "per_family": {family: 0.25 for family in sorted(rb.PUBLIC_FAMILIES)},
        "within_budget": True,
        "wall_seconds": 1.0,
    }


def make_model(scratch: str) -> Path:
    model = Path(scratch) / "merged" / "arm"
    model.mkdir(parents=True)
    (model / "merge_receipt.json").write_text('{"name": "arm"}\n', encoding="utf-8")
    return model


def write_receipt(scratch: str, payload: dict) -> Path:
    receipt = Path(scratch) / "arm.json"
    receipt.write_text(
        json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


class TestReceiptAuthentication(unittest.TestCase):
    def test_valid_receipt_loads_for_each_frozen_seed(self) -> None:
        for module in (rb, cb):
            for seed in rb.SEED_ORDER:
                with tempfile.TemporaryDirectory() as scratch:
                    model = make_model(scratch)
                    receipt = write_receipt(scratch, receipt_payload(model, seed))
                    loaded = module.load_event(receipt, model, seed)
                    self.assertEqual(loaded["seed"], seed, module.__name__)
                    self.assertEqual(loaded["think_budget"], 1024)

    def test_receipt_seed_must_match_the_expected_seed(self) -> None:
        # A seed-78155 receipt can never authenticate as the 78156 event:
        # the loader is seed-scoped, not merely member-of-the-three.
        for module in (rb, cb):
            with tempfile.TemporaryDirectory() as scratch:
                model = make_model(scratch)
                receipt = write_receipt(scratch, receipt_payload(model, SEED_1))
                with self.assertRaises(ValueError, msg=module.__name__):
                    module.load_event(receipt, model, SEED_2)

    def test_expected_seed_outside_the_frozen_three_is_rejected(self) -> None:
        for module in (rb, cb):
            with tempfile.TemporaryDirectory() as scratch:
                model = make_model(scratch)
                payload = receipt_payload(model, 78154)
                receipt = write_receipt(scratch, payload)
                with self.assertRaises(ValueError, msg=module.__name__):
                    module.load_event(receipt, model, 78154)

    def test_drifted_fields_are_rejected(self) -> None:
        drifts = (
            {"seed": SEED_1 + 100},
            {"tier": "quick"},
            {"think_budget": 4096},
            {"backend": "other"},
            {"model_merge_receipt_sha256": "c" * 64},
            {"stage": "something_else"},
        )
        for drift in drifts:
            with tempfile.TemporaryDirectory() as scratch:
                model = make_model(scratch)
                receipt = write_receipt(
                    scratch, {**receipt_payload(model, SEED_1), **drift}
                )
                with self.assertRaises(ValueError, msg=str(drift)):
                    rb.load_event(receipt, model, SEED_1)

    def test_over_budget_receipt_is_accepted_and_recorded(self) -> None:
        # Reference contract: within_budget false never rejects the
        # receipt; the budget_integrity reading scopes the comparison.
        for module in (rb, cb):
            with tempfile.TemporaryDirectory() as scratch:
                model = make_model(scratch)
                payload = {**receipt_payload(model, SEED_1), "within_budget": False}
                receipt = write_receipt(scratch, payload)
                loaded = module.load_event(receipt, model, SEED_1)
                self.assertIs(loaded["within_budget"], False, module.__name__)

    def test_non_bool_within_budget_is_rejected(self) -> None:
        for bad in (None, "true", "yes", 1, 0.0):
            for module in (rb, cb):
                with tempfile.TemporaryDirectory() as scratch:
                    model = make_model(scratch)
                    payload = {**receipt_payload(model, SEED_1), "within_budget": bad}
                    receipt = write_receipt(scratch, payload)
                    with self.assertRaises(
                        ValueError, msg=(module.__name__, bad)
                    ):
                        module.load_event(receipt, model, SEED_1)

    def test_wall_seconds_must_be_finite_and_non_negative(self) -> None:
        for bad in (float("nan"), float("inf"), -1.0, True, "12.5", None):
            for module in (rb, cb):
                with tempfile.TemporaryDirectory() as scratch:
                    model = make_model(scratch)
                    payload = {**receipt_payload(model, SEED_1), "wall_seconds": bad}
                    receipt = write_receipt(scratch, payload)
                    with self.assertRaises(
                        ValueError, msg=(module.__name__, bad)
                    ):
                        module.load_event(receipt, model, SEED_1)
        # No upper cap: the gateway owns budget policy; record what it
        # returns.
        with tempfile.TemporaryDirectory() as scratch:
            model = make_model(scratch)
            payload = {**receipt_payload(model, SEED_1), "wall_seconds": 86_400.0}
            receipt = write_receipt(scratch, payload)
            self.assertEqual(
                rb.load_event(receipt, model, SEED_1)["wall_seconds"], 86_400.0
            )

    def test_missing_family_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            model = make_model(scratch)
            payload = receipt_payload(model, SEED_1)
            payload["per_family"].pop("menders")
            receipt = write_receipt(scratch, payload)
            with self.assertRaises(ValueError):
                rb.load_event(receipt, model, SEED_1)

    def test_extra_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            model = make_model(scratch)
            payload = receipt_payload(model, SEED_1)
            payload["surprise"] = True
            receipt = write_receipt(scratch, payload)
            with self.assertRaises(ValueError):
                rb.load_event(receipt, model, SEED_1)


class TestFinitenessGuards(unittest.TestCase):
    def test_valid_score_predicate(self) -> None:
        for module in (rb, cb):
            for good in (0.0, 0.5, 1.0, 1):
                self.assertTrue(module._valid_score(good), (module.__name__, good))
            for bad in (
                float("nan"), float("inf"), float("-inf"),
                -0.001, 1.001, True, False, None, "0.5",
            ):
                self.assertFalse(module._valid_score(bad), (module.__name__, bad))

    def test_valid_wall_seconds_predicate(self) -> None:
        for module in (rb, cb):
            for good in (0.0, 1, 230.5, 86_400.0):
                self.assertTrue(
                    module._valid_wall_seconds(good), (module.__name__, good)
                )
            for bad in (
                float("nan"), float("inf"), -0.001, True, False, None, "1.0",
            ):
                self.assertFalse(
                    module._valid_wall_seconds(bad), (module.__name__, bad)
                )

    def test_non_finite_or_out_of_range_scores_rejected_by_both_loaders(self) -> None:
        corruptions = (
            ("aggregate_nan", "aggregate", float("nan")),
            ("aggregate_above_one", "aggregate", 1.5),
            ("family_nan", "menders", float("nan")),
            ("family_infinite", "sirens", float("inf")),
            ("family_negative", "warren", -0.1),
            ("family_bool", "mirage", True),
        )
        for module in (rb, cb):
            for name, field, value in corruptions:
                with tempfile.TemporaryDirectory() as scratch:
                    model = make_model(scratch)
                    payload = receipt_payload(model, SEED_1)
                    if field == "aggregate":
                        payload["aggregate"] = value
                    else:
                        payload["per_family"][field] = value
                    receipt = write_receipt(scratch, payload)
                    with self.assertRaises(
                        ValueError, msg=(module.__name__, name)
                    ):
                        module.load_event(receipt, model, SEED_1)


if __name__ == "__main__":
    unittest.main()
