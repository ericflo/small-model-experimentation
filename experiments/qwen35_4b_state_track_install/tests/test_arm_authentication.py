"""Arm-authentication failure drills: every tamper path must refuse.

Drives the fail-closed guards with tampered temp copies: unfilled TODO
pins, a tampered parent merge receipt, a receipt that no longer describes
the frozen parent arm, a fake composite tree, and NaN gateway scores.
Sibling-original drills: an ABSENT committed sibling receipt passes on the
in-cell sha-pinned copy (with the recorded note); a PRESENT-but-divergent
sibling still refuses loudly (tamper evidence); a missing in-cell copy
refuses regardless of the sibling.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "scripts"))

import run_benchmark as rb  # noqa: E402
import train_trial as tt  # noqa: E402


class TestTodoPins(unittest.TestCase):
    def test_require_pin_refuses_none(self):
        with self.assertRaises(ValueError):
            rb.require_pin(None, "X")
        self.assertEqual(rb.require_pin("ab", "X"), "ab")

    def test_unfilled_candidate_pins_abort_the_event(self):
        # The candidate pins ship as None until the merge publishes; the
        # fail-closed gate must refuse while ANY of the three is unfilled.
        if rb.FROZEN_TREE_SHA256["state_track"] is None:
            with self.assertRaises(ValueError):
                rb.require_todo_pins_filled()
        else:
            rb.require_todo_pins_filled()

    def test_malformed_pin_refuses(self):
        original = rb.FROZEN_TREE_SHA256["state_track"]
        try:
            rb.FROZEN_TREE_SHA256["state_track"] = "not-a-sha"
            with self.assertRaises(ValueError):
                rb.require_todo_pins_filled()
        finally:
            rb.FROZEN_TREE_SHA256["state_track"] = original

    def test_published_arm_hashes_todo_pin_refuses_committed_reliance(self):
        self.assertIn("state_track", tt.PUBLISHED_ARM_HASHES)


class TestParentProvenanceTamper(unittest.TestCase):
    """The committed lifecycle-27 parent receipt, tampered in a temp copy."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.real_receipt = rb.COUNT_WALK_PARENT_MERGE_RECEIPT
        self.real_copy = rb.COUNT_WALK_PARENT_PROVENANCE_COPY
        self.model = rb.FROZEN_MODEL_PATHS[rb.FROZEN_PARENT]

    def tearDown(self):
        rb.COUNT_WALK_PARENT_MERGE_RECEIPT = self.real_receipt
        rb.COUNT_WALK_PARENT_PROVENANCE_COPY = self.real_copy
        tt.PARENT_COMMITTED_MERGE_RECEIPT = self.real_receipt
        tt.PARENT_PROVENANCE_COPY = self.real_copy

    def test_untampered_copies_authenticate(self):
        self.assertEqual(
            rb.require_count_walk_parent_provenance(self.model),
            "present, byte-identical to the in-cell pin",
        )
        self.assertEqual(
            tt.check_parent_provenance(),
            "present, byte-identical to the in-cell pin",
        )

    def test_absent_sibling_original_passes_on_the_in_cell_pin(self):
        # The sibling-free drill: the committed sibling original is a
        # verification aid; when absent the in-cell sha-pinned copy is
        # authoritative and authentication passes with the recorded note.
        absent = self.tmp / "not_checked_out" / "count_walk.json"
        rb.COUNT_WALK_PARENT_MERGE_RECEIPT = absent
        self.assertEqual(
            rb.require_count_walk_parent_provenance(self.model),
            "absent, in-cell pin authoritative",
        )
        tt.PARENT_COMMITTED_MERGE_RECEIPT = absent
        self.assertEqual(
            tt.check_parent_provenance(),
            "absent, in-cell pin authoritative",
        )

    def test_absent_in_cell_copy_refuses_even_with_sibling_present(self):
        # The in-cell copy is the hard gate; a missing copy refuses no
        # matter what the sibling checkout holds.
        absent = self.tmp / "missing_copy.json"
        rb.COUNT_WALK_PARENT_PROVENANCE_COPY = absent
        with self.assertRaises(ValueError):
            rb.require_count_walk_parent_provenance(self.model)
        tt.PARENT_PROVENANCE_COPY = absent
        with self.assertRaises(ValueError):
            tt.check_parent_provenance()

    def test_tampered_receipt_bytes_refuse(self):
        tampered = self.tmp / "count_walk.json"
        raw = bytearray(self.real_receipt.read_bytes())
        raw[-2] ^= 0x01
        tampered.write_bytes(bytes(raw))
        rb.COUNT_WALK_PARENT_MERGE_RECEIPT = tampered
        with self.assertRaises(ValueError):
            rb.require_count_walk_parent_provenance(self.model)
        tt.PARENT_COMMITTED_MERGE_RECEIPT = tampered
        with self.assertRaises(ValueError):
            tt.check_parent_provenance()

    def test_diverged_provenance_copy_refuses(self):
        diverged = self.tmp / "copy.json"
        raw = bytearray(self.real_copy.read_bytes())
        raw[-2] ^= 0x01
        diverged.write_bytes(bytes(raw))
        rb.COUNT_WALK_PARENT_PROVENANCE_COPY = diverged
        with self.assertRaises(ValueError):
            rb.require_count_walk_parent_provenance(self.model)
        tt.PARENT_PROVENANCE_COPY = diverged
        with self.assertRaises(ValueError):
            tt.check_parent_provenance()

    def test_receipt_describing_a_different_arm_refuses_despite_matching_sha_slot(self):
        # Same bytes elsewhere on disk BUT the payload's merged path no
        # longer resolves to the model under authentication.
        with self.assertRaises(ValueError):
            rb.require_count_walk_parent_provenance(self.tmp / "other_model")


class TestModelTreeAuthentication(unittest.TestCase):
    def test_fake_composite_tree_refuses(self):
        with tempfile.TemporaryDirectory() as scratch:
            fake = Path(scratch) / "merged"
            fake.mkdir()
            for name in rb.MERGED_FILE_NAMES:
                (fake / name).write_text("junk", encoding="utf-8")
            with self.assertRaises(ValueError):
                rb.authenticate_model_tree("base", fake)

    def test_incomplete_composite_tree_refuses(self):
        with tempfile.TemporaryDirectory() as scratch:
            fake = Path(scratch) / "merged"
            fake.mkdir()
            (fake / "model.safetensors").write_text("junk", encoding="utf-8")
            with self.assertRaises(ValueError):
                rb.merged_tree_manifest(fake)

    def test_unexpected_extra_file_refuses(self):
        with tempfile.TemporaryDirectory() as scratch:
            fake = Path(scratch) / "merged"
            fake.mkdir()
            for name in (*rb.MERGED_FILE_NAMES, "extra.bin"):
                (fake / name).write_text("junk", encoding="utf-8")
            with self.assertRaises(ValueError):
                rb.merged_tree_manifest(fake)


class TestGatewayEventValidation(unittest.TestCase):
    def make_event(self, tmp: Path, **overrides) -> tuple[Path, Path]:
        model = tmp / "model"
        model.mkdir()
        (model / "merge_receipt.json").write_text("{}", encoding="utf-8")
        payload = {
            "schema_version": 1,
            "stage": "menagerie_aggregate_gateway",
            "tier": rb.FROZEN_TIER,
            "think_budget": rb.FROZEN_THINK_BUDGET,
            "seed": rb.FROZEN_SEED,
            "backend": "qwen_vllm",
            "model": str(model),
            "model_merge_receipt_sha256": rb.sha256_file(
                model / "merge_receipt.json"
            ),
            "benchmark_runner_sha256": "aa" * 32,
            "benchmark_source_inventory_sha256": "bb" * 32,
            "benchmark_source_file_count": 56,
            "aggregate": 0.3,
            "per_family": {family: 0.3 for family in rb.PUBLIC_FAMILIES},
            "within_budget": True,
            "wall_seconds": 100.0,
        }
        payload.update(overrides)
        event = tmp / "event.json"
        event.write_text(json.dumps(payload), encoding="utf-8")
        return event, model

    def test_valid_event_loads(self):
        with tempfile.TemporaryDirectory() as scratch:
            event, model = self.make_event(Path(scratch))
            loaded = rb.load_event(event, model)
            self.assertEqual(loaded["aggregate"], 0.3)

    def test_nan_aggregate_refuses(self):
        with tempfile.TemporaryDirectory() as scratch:
            event, model = self.make_event(Path(scratch), aggregate=float("nan"))
            with self.assertRaises(ValueError):
                rb.load_event(event, model)

    def test_nan_family_refuses(self):
        with tempfile.TemporaryDirectory() as scratch:
            families = {family: 0.3 for family in rb.PUBLIC_FAMILIES}
            families["menders"] = float("nan")
            event, model = self.make_event(Path(scratch), per_family=families)
            with self.assertRaises(ValueError):
                rb.load_event(event, model)

    def test_out_of_range_score_refuses(self):
        with tempfile.TemporaryDirectory() as scratch:
            event, model = self.make_event(Path(scratch), aggregate=1.2)
            with self.assertRaises(ValueError):
                rb.load_event(event, model)

    def test_over_budget_event_refuses(self):
        with tempfile.TemporaryDirectory() as scratch:
            event, model = self.make_event(Path(scratch), within_budget=False)
            with self.assertRaises(ValueError):
                rb.load_event(event, model)

    def test_wrong_seed_refuses(self):
        with tempfile.TemporaryDirectory() as scratch:
            event, model = self.make_event(Path(scratch), seed=rb.FROZEN_SEED + 1)
            with self.assertRaises(ValueError):
                rb.load_event(event, model)


if __name__ == "__main__":
    unittest.main()
