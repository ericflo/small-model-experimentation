"""The state_track curriculum generator: truth audit, contamination, overlap.

The one designed delta of this cell. These tests exercise the generator's
fail-closed truth audit end to end: every ledger is re-derived by a second
independent interpreter and byte-compared, every answer is recomputed from
that independent state, the KIND is constant (u_state_track), the banned-
vocabulary audit rejects any benchmark/inventory collision, and the frozen
160-row corpus on disk matches its pin. Contamination is proven by ZERO
canonical-user-message overlap against the stage-8 replay pool, the eleven
predecessor gate files, and the fresh retention screens the local gate
draws from gen_curriculum.py.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
SCRIPTS = EXP / "scripts"
sys.path.insert(0, str(SCRIPTS))

import gen_curriculum as universal  # noqa: E402
import gen_state_track_curriculum as gst  # noqa: E402


CORPUS = EXP / "data" / "sft_state_track.jsonl"
CORPUS_SHA256 = "66a8d5bec184a8a9cba20c2ea088e0216ac4cdbd0820541ee310170eb386e3ab"


def canonical_message(row: dict) -> str:
    return json.dumps(
        row["messages"], sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class TestGeneratorTruthAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = gst.generate_curriculum(87, 160)
        cls.summary = gst.validate_generated(cls.rows, expected_rows=160)

    def test_160_rows_single_kind(self):
        self.assertEqual(self.summary["rows"], 160)
        self.assertEqual(self.summary["kinds"], {"u_state_track": 160})

    def test_constant_kind_on_every_row(self):
        self.assertEqual({row["kind"] for row in self.rows}, {"u_state_track"})

    def test_every_ledger_rederivation_byte_matches(self):
        for index, row in enumerate(self.rows):
            audit = row["_audit"]
            with self.subTest(row=index):
                self.assertEqual(
                    json.dumps(audit["final_state"], sort_keys=True),
                    json.dumps(audit["independent_final_state"], sort_keys=True),
                )
                self.assertTrue(audit["rederivation_byte_match"])

    def test_every_answer_recomputes_from_the_independent_state(self):
        for index, row in enumerate(self.rows):
            audit = row["_audit"]
            recomputed = gst.compute_answer(
                audit["independent_final_state"], tuple(audit["query"])
            )
            recorded = row["answer"].removeprefix("ANSWER: ").strip()
            with self.subTest(row=index):
                self.assertEqual(recomputed, recorded)

    def test_answer_is_a_clean_single_line(self):
        for row in self.rows:
            self.assertTrue(row["answer"].startswith("ANSWER: "))
            self.assertNotIn("\n", row["answer"])

    def test_axes_are_varied(self):
        self.assertEqual(set(self.summary["surfaces"]), set(gst.SURFACES))
        self.assertEqual(set(self.summary["query_kinds"]), set(gst.QUERIES))
        self.assertGreaterEqual(len(self.summary["chain_lengths"]), 3)
        self.assertGreaterEqual(len(self.summary["quantity_counts"]), 3)

    def test_think_token_stats_recorded(self):
        self.assertIn("max_estimated_think_tokens", self.summary)
        self.assertIn("mean_estimated_think_tokens", self.summary)
        self.assertGreater(self.summary["max_estimated_think_tokens"], 0)

    def test_deterministic_regeneration_is_byte_identical(self):
        again = gst.generate_curriculum(87, 160)
        self.assertEqual(
            [gst.public_row(r) for r in self.rows],
            [gst.public_row(r) for r in again],
        )


class TestFrozenCorpus(unittest.TestCase):
    def test_disk_corpus_matches_the_pin(self):
        import hashlib

        self.assertTrue(CORPUS.is_file())
        digest = hashlib.sha256(CORPUS.read_bytes()).hexdigest()
        self.assertEqual(digest, CORPUS_SHA256)
        rows = load_jsonl(CORPUS)
        self.assertEqual(len(rows), 160)
        self.assertEqual({r["kind"] for r in rows}, {"u_state_track"})

    def test_disk_corpus_matches_the_generator(self):
        on_disk = load_jsonl(CORPUS)
        generated = [gst.public_row(r) for r in gst.generate_curriculum(87, 160)]
        self.assertEqual(on_disk, generated)


class TestFailClosedAudit(unittest.TestCase):
    def _one_row(self) -> dict:
        return gst.generate_curriculum(87, 4)[0]

    def test_constant_kind_violation_rejected(self):
        row = self._one_row()
        row["kind"] = "u_count"
        with self.assertRaises(ValueError):
            gst.validate_generated([row])

    def test_rederivation_mismatch_rejected(self):
        row = self._one_row()
        # Corrupt the independent state so it no longer byte-matches the trace.
        first = next(iter(row["_audit"]["independent_final_state"]))
        row["_audit"]["independent_final_state"][first] += 1
        with self.assertRaises(ValueError):
            gst.validate_generated([row])

    def test_banned_vocabulary_rejected(self):
        row = self._one_row()
        row["messages"][0]["content"] += " lockpick"
        with self.assertRaises(ValueError):
            gst.validate_generated([row])

    def test_banned_set_covers_families_and_inventory(self):
        for token in ("chronicle", "warren", "toolsmith", "kilnrite", "cobalt"):
            self.assertIn(token, gst.BANNED_VOCAB)

    def test_register_names_are_not_banned(self):
        for name in gst.REGISTER_POOL:
            self.assertNotIn(name.lower(), gst.BANNED_VOCAB)


class TestZeroOverlap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows = gst.generate_curriculum(87, 160)
        cls.messages = {canonical_message(gst.public_row(r)) for r in rows}
        cls.assertlen = len(cls.messages)

    def test_no_internal_message_collisions(self):
        self.assertEqual(self.assertlen, 160)

    def test_zero_overlap_with_stage8_replay_pool(self):
        pool = {canonical_message(r) for r in load_jsonl(EXP / "data" / "sft_blend.jsonl")}
        self.assertEqual(self.messages & pool, set())

    def test_zero_overlap_with_predecessor_gate_files(self):
        gate_messages: set[str] = set()
        for path in sorted((EXP / "data" / "predecessor_gates").glob("*.jsonl")):
            gate_messages |= {canonical_message(r) for r in load_jsonl(path)}
        self.assertEqual(self.messages & gate_messages, set())

    def test_zero_overlap_with_local_retention_screens(self):
        screen_messages: set[str] = set()
        mix = ",".join(f"{name}=8" for name in universal.SKILLS)
        for seed in (88063, 88064, 88065):
            for row in universal.generate_curriculum(mix, seed):
                screen_messages.add(canonical_message(row))
        self.assertEqual(self.messages & screen_messages, set())

    def test_zero_overlap_with_the_stage9_curriculum_on_disk(self):
        on_disk = {canonical_message(r) for r in load_jsonl(CORPUS)}
        # sanity: the on-disk corpus IS the generated set (not a foreign one)
        self.assertEqual(on_disk, self.messages)


if __name__ == "__main__":
    unittest.main()
