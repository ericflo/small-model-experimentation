"""The deterministic UNION-build contract for the repair_why_stack corpus.

This cell does NOT generate data; it COMBINES two committed source corpora. The
build must be a pure function of (the two sha-pinned source copies + the fixed
shuffle seed): each source sha is verified fail-closed, the combined corpus is
exactly 504 self_repair + 504 why_comment = 1008 rows, the two kinds INTERLEAVE
(not block-concatenated), and two independent rebuilds are byte-identical and
match the frozen sha pin + the committed corpus + the receipt.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import unittest
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))

import build_corpus as bc  # noqa: E402

CORPUS = EXP / "data" / "sft_repair_why_stack.jsonl"
RECEIPT = EXP / "data" / "stack_corpus_receipt.json"


class TestSourceShaPins(unittest.TestCase):
    def test_each_source_copy_matches_its_pin(self):
        for kind, path, pinned, _origin in bc.SOURCES:
            self.assertTrue(path.is_file(), msg=f"missing source copy for {kind}: {path}")
            self.assertEqual(bc.sha256_file(path), pinned, msg=f"source sha mismatch for {kind}")

    def test_source_pins_are_the_two_committed_corpora(self):
        pins = {kind: pinned for kind, _p, pinned, _o in bc.SOURCES}
        self.assertEqual(pins["self_repair"], "920cb228172677f005bdbc4501f593ce60dc7a9c4f22cbf177f05660ffc392cb")
        self.assertEqual(pins["why_comment"], "040be350678ea0337b8fe0607f783aba9e9071f789471b0ea00f7ce1ebef2962")

    def test_a_tampered_source_aborts_the_build(self):
        original = bc.SOURCES
        try:
            bc.SOURCES = (
                ("self_repair", bc.SOURCES[0][1], "0" * 64, "qwen35_4b_self_repair_install"),
                bc.SOURCES[1],
            )
            with self.assertRaises(SystemExit):
                bc.verify_sources()
        finally:
            bc.SOURCES = original


class TestUnionByKind(unittest.TestCase):
    def test_committed_corpus_is_504_plus_504(self):
        rows = [json.loads(line) for line in CORPUS.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(rows), 1008)
        kinds: dict[str, int] = {}
        for row in rows:
            kinds[row["kind"]] = kinds.get(row["kind"], 0) + 1
        self.assertEqual(kinds, {"self_repair": 504, "why_comment": 504})
        self.assertEqual(kinds, bc.EXPECTED_ROWS_BY_KIND)

    def test_build_row_by_kind_helper_agrees(self):
        counts = bc.kind_counts(bc.build_bytes())
        self.assertEqual(counts, {"self_repair": 504, "why_comment": 504})

    def test_kinds_interleave_not_block_concatenated(self):
        # A seeded shuffle interleaves; a block concatenation would have all of
        # one kind before the other. Assert neither half is single-kind.
        rows = [json.loads(line) for line in CORPUS.read_text(encoding="utf-8").splitlines() if line.strip()]
        first_half = {row["kind"] for row in rows[:504]}
        second_half = {row["kind"] for row in rows[504:]}
        self.assertEqual(first_half, {"self_repair", "why_comment"})
        self.assertEqual(second_half, {"self_repair", "why_comment"})


class TestDeterministicSha(unittest.TestCase):
    def test_two_rebuilds_are_byte_identical(self):
        first = bc.build_bytes()
        second = bc.build_bytes()
        self.assertEqual(first, second)

    def test_build_matches_the_frozen_pin_and_committed_corpus(self):
        payload = bc.build_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        self.assertEqual(digest, bc.COMBINED_SHA256)
        self.assertEqual(digest, "2462c93ea2a8dcfbd9413e1c6115ed1456ad438e5dabfdc01e924be6148ddbe5")
        self.assertEqual(bc.sha256_file(CORPUS), bc.COMBINED_SHA256)

    def test_shuffle_seed_is_load_bearing(self):
        # A different shuffle seed must produce a different corpus sha (proves the
        # seed, not incidental ordering, fixes the layout).
        lines: list[str] = []
        for _kind, path, _sha, _origin in bc.SOURCES:
            lines.extend(bc.source_lines(path))
        random.Random(bc.SHUFFLE_SEED + 1).shuffle(lines)
        other = hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()
        self.assertNotEqual(other, bc.COMBINED_SHA256)


class TestReceipt(unittest.TestCase):
    def test_receipt_pins_the_union(self):
        receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
        self.assertEqual(receipt["corpus_sha256"], bc.COMBINED_SHA256)
        self.assertEqual(receipt["rows"], 1008)
        self.assertEqual(receipt["kinds"], {"self_repair": 504, "why_comment": 504})
        self.assertEqual(receipt["shuffle_seed"], 93570)
        self.assertEqual(receipt["combine_order"], ["self_repair", "why_comment"])
        self.assertEqual(receipt["contamination"]["banned_hits"], 0)
        source_shas = {s["kind"]: s["sha256"] for s in receipt["sources"]}
        self.assertEqual(source_shas["self_repair"], "920cb228172677f005bdbc4501f593ce60dc7a9c4f22cbf177f05660ffc392cb")
        self.assertEqual(source_shas["why_comment"], "040be350678ea0337b8fe0607f783aba9e9071f789471b0ea00f7ce1ebef2962")

    def test_verify_corpus_passes_fail_closed(self):
        # The full fail-closed verification (determinism + pins + committed +
        # receipt + union banned-name audit) must not raise.
        bc.verify_corpus()


if __name__ == "__main__":
    unittest.main()
