"""Standalone-reproduction drills: sibling originals are verification aids.

Owner's standalone directive: cross-experiment files are NEVER the
reproduction path. Every guard that consults a committed sibling original
must (a) pass on the in-cell sha-pinned copy alone when the sibling is
absent, recording a note; (b) still refuse loudly when a present sibling
diverges from the in-cell copy (tamper evidence); (c) hard-require the
in-cell copy itself. Drilled here for gen_local_gate's predecessor gates
and parent receipt, rebuild_lineage's provenance receipts, and
eval_local_vllm's inherited-arm receipt path.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "scripts"))

import eval_local_vllm as ev  # noqa: E402
import gen_local_gate as gg  # noqa: E402
import rebuild_lineage as rl  # noqa: E402


class TestPredecessorGateCopies(unittest.TestCase):
    def test_eleven_predecessor_gates_with_in_cell_copies(self):
        self.assertEqual(len(gg.PREDECESSOR_GATES), 11)
        seeds = sorted(
            int(copy.stem.removeprefix("local_tasks_seed"))
            for _, copy, _, _ in gg.PREDECESSOR_GATES
        )
        self.assertEqual(seeds, list(range(88052, 88063)))
        for sibling, copy, sha, rows in gg.PREDECESSOR_GATES:
            with self.subTest(copy=copy.name):
                self.assertEqual(copy.parent, gg.PREDECESSOR_GATE_COPY_DIR)
                self.assertTrue(copy.is_file())
                self.assertEqual(gg.sha256_file(copy), sha)
                self.assertEqual(
                    len(gg.load_predecessor_gate(sibling, copy, sha, rows)), rows
                )

    def test_in_cell_copies_match_present_siblings_byte_identically(self):
        for sibling, copy, _, _ in gg.PREDECESSOR_GATES:
            if sibling.is_file():
                with self.subTest(copy=copy.name):
                    self.assertEqual(sibling.read_bytes(), copy.read_bytes())

    def test_absent_sibling_passes_on_the_in_cell_copy(self):
        sibling, copy, sha, rows = gg.PREDECESSOR_GATES[0]
        with tempfile.TemporaryDirectory() as scratch:
            absent = Path(scratch) / "not_checked_out.jsonl"
            loaded = gg.load_predecessor_gate(absent, copy, sha, rows)
            self.assertEqual(len(loaded), rows)

    def test_divergent_present_sibling_refuses(self):
        _, copy, sha, rows = gg.PREDECESSOR_GATES[0]
        with tempfile.TemporaryDirectory() as scratch:
            divergent = Path(scratch) / "divergent.jsonl"
            raw = bytearray(copy.read_bytes())
            raw[-2] ^= 0x01
            divergent.write_bytes(bytes(raw))
            with self.assertRaises(ValueError):
                gg.load_predecessor_gate(divergent, copy, sha, rows)

    def test_missing_in_cell_copy_refuses_even_with_sibling_present(self):
        sibling, copy, sha, rows = gg.PREDECESSOR_GATES[0]
        with tempfile.TemporaryDirectory() as scratch:
            absent_copy = Path(scratch) / "missing_copy.jsonl"
            with self.assertRaises((ValueError, OSError)):
                gg.load_predecessor_gate(sibling, absent_copy, sha, rows)


class TestParentReceiptSibling(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.real_sibling = gg.PARENT_MERGE_RECEIPT
        self.addCleanup(setattr, gg, "PARENT_MERGE_RECEIPT", self.real_sibling)

    def test_absent_parent_sibling_passes_build_outputs(self):
        # A never-checked-out sibling path (must stay under ROOT: the
        # receipt documents it ROOT-relative).
        gg.PARENT_MERGE_RECEIPT = (
            self.real_sibling.parent / "count_walk.__absent_drill__.json"
        )
        self.assertFalse(gg.PARENT_MERGE_RECEIPT.exists())
        outputs = gg.build_outputs(authenticate_parent=False)
        self.assertIn(gg.RECEIPT, outputs)

    def test_divergent_parent_sibling_refuses_build_outputs(self):
        divergent = self.tmp / "divergent.json"
        raw = bytearray(gg.PARENT_PROVENANCE_COPY.read_bytes())
        raw[-2] ^= 0x01
        divergent.write_bytes(bytes(raw))
        gg.PARENT_MERGE_RECEIPT = divergent
        with self.assertRaises(ValueError):
            gg.build_outputs(authenticate_parent=False)


class TestProvenanceReceiptSiblings(unittest.TestCase):
    def setUp(self):
        self.real_source = rl.SOURCE_ZERO_ROOT_EXP
        self.addCleanup(setattr, rl, "SOURCE_ZERO_ROOT_EXP", self.real_source)
        self.manifest = rl.load_manifest()

    def test_present_siblings_verify_byte_identically(self):
        result = rl.verify_provenance_receipts(self.manifest)
        self.assertEqual(result["checked"], 7)
        siblings = result["sibling_originals"]
        self.assertEqual(siblings["present"] + siblings["absent"], 7)

    def test_absent_sibling_cell_passes_on_the_in_cell_copies(self):
        with tempfile.TemporaryDirectory() as scratch:
            rl.SOURCE_ZERO_ROOT_EXP = Path(scratch) / "not_checked_out"
            result = rl.verify_provenance_receipts(self.manifest)
            self.assertEqual(result["checked"], 7)
            self.assertEqual(result["sibling_originals"]["absent"], 7)
            self.assertEqual(result["sibling_originals"]["present"], 0)

    def test_divergent_present_sibling_refuses(self):
        with tempfile.TemporaryDirectory() as scratch:
            fake_cell = Path(scratch) / "cell"
            lineage = fake_cell / "runs" / "lineage"
            lineage.mkdir(parents=True)
            name = sorted(
                self.manifest["clean_chain"]["provenance_receipts"]
            )[0]
            raw = bytearray((rl.PROVENANCE_DIR / name).read_bytes())
            raw[-2] ^= 0x01
            (lineage / name).write_bytes(bytes(raw))
            rl.SOURCE_ZERO_ROOT_EXP = fake_cell
            with self.assertRaises(ValueError):
                rl.verify_provenance_receipts(self.manifest)


class TestEvalInheritedArmSibling(unittest.TestCase):
    def test_inherited_receipt_path_is_the_in_cell_copy(self):
        self.assertEqual(
            ev.COMPOSITE_RECEIPTS["count_walk"],
            EXP / "data" / "provenance" / "count_walk_merge.json",
        )
        self.assertEqual(
            ev.SIBLING_ORIGINALS["count_walk"],
            ROOT
            / "experiments"
            / "qwen35_4b_count_dont_walk_enumeration"
            / "runs"
            / "merges"
            / "count_walk.json",
        )

    def test_sibling_status_notes(self):
        real = ev.SIBLING_ORIGINALS["count_walk"]
        try:
            if real.is_file():
                self.assertEqual(
                    ev.sibling_original_status("count_walk"),
                    "present, byte-identical to the in-cell pin",
                )
            ev.SIBLING_ORIGINALS["count_walk"] = real / "nope.json"
            self.assertEqual(
                ev.sibling_original_status("count_walk"),
                "absent, in-cell pin authoritative",
            )
            self.assertEqual(
                ev.sibling_original_status("state_track"),
                "not_applicable (in-cell trained arm)",
            )
        finally:
            ev.SIBLING_ORIGINALS["count_walk"] = real


if __name__ == "__main__":
    unittest.main()
