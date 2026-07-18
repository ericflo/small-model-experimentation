"""Contamination firewall at scale: banned-name audit + distinctive n-gram overlap.

The committed banned-function-name fixture is self-consistent (sha) and identical
to the sibling why_comment cell's; the whole-word audit fires on a benchmark name
yet never on a Python builtin; a 10000-row corpus has ZERO banned-vocabulary hits
(code + spec prose + #WHY: prose) and ZERO distinctive shared code 7-grams vs the
benchmark solutions; and the generator's identifier pools are contamination-clean.
When the HF cache is present the aids RUN; absent it they skip with a note.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))

import contamination as contam  # noqa: E402
import gen_why_scale_curriculum as gen  # noqa: E402

SEED = gen.CONSTRUCTION_SEED


class TestFixtureIntegrity(unittest.TestCase):
    def test_fixture_loads_and_is_sha_consistent(self):
        payload = contam.load_fixture()
        self.assertEqual(payload["function_name_count"], len(payload["function_names"]))
        self.assertGreater(payload["function_name_count"], 500)

    def test_language_tokens_are_whitelisted_out_of_the_banned_set(self):
        banned = contam.banned_names()
        for token in contam.LANGUAGE_WHITELIST:
            self.assertNotIn(token.lower(), banned)
        self.assertIn("sum", {n.lower() for n in contam.load_fixture()["function_names"]})
        self.assertNotIn("sum", banned)


class TestWholeWordAudit(unittest.TestCase):
    def test_audit_fires_on_a_benchmark_name(self):
        banned = contam.banned_names()
        sample = sorted(banned)[0]
        self.assertIn(sample, contam.whole_word_hits(f"here we use {sample} in a sentence", banned))

    def test_audit_ignores_python_builtins_and_keywords(self):
        banned = contam.banned_names()
        text = "for i in range(len(seq)): acc = acc + sum(seq) if seq else int(0)"
        self.assertEqual(contam.whole_word_hits(text, banned), set())

    def test_audit_is_whole_word_only(self):
        banned = contam.banned_names()
        sample = sorted(banned)[0]
        self.assertNotIn(sample, contam.whole_word_hits(f"my_{sample}_variable = 1", banned))


class TestDistinctiveOverlapPredicate(unittest.TestCase):
    def test_structural_only_gram_has_no_distinctive_token(self):
        gram = ("for", "i", "in", "range", "(", "0", ",")
        self.assertFalse(contam.gram_has_distinctive(gram))

    def test_gram_with_real_identifier_is_distinctive(self):
        gram = ("acc", "=", "acc", "+", "seq", "[", "i")
        self.assertTrue(contam.gram_has_distinctive(gram))


class TestContaminationInCorpusAtScale(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = gen.generate_curriculum(SEED, 10000)

    def test_banned_vocabulary_is_zero_at_10000(self):
        banned = contam.banned_names()
        for i, row in enumerate(self.rows):
            blob = row["messages"][0]["content"] + "\n" + row["think"] + "\n" + row["answer"]
            self.assertEqual(contam.whole_word_hits(blob, banned), set(), msg=f"row {i} banned hit")

    def test_identifier_pools_are_contamination_clean(self):
        banned = contam.banned_names()
        pools = (gen.FUNC_NAMES + gen.LIST_PARAMS + gen.INT_PARAMS
                 + gen.STR_PARAMS + gen.ACC_NAMES + gen.SCRATCH_NAMES)
        for name in pools:
            self.assertNotIn(name.lower(), banned, msg=f"identifier {name!r} collides with a benchmark name")

    def test_code_ngram_overlap_zero_at_10000_when_cache_present(self):
        try:
            streams = contam.build_code_tokens_from_cache()
        except Exception as exc:  # noqa: BLE001 (present-only aid)
            self.skipTest(f"HF cache unavailable: {type(exc).__name__}")
        bench = contam.benchmark_ngrams(streams)
        corpus: set = set()
        for row in self.rows:
            corpus |= gen.code_grams_no_comments(row["_audit"]["clean_code"])
        self.assertEqual(contam.distinctive_overlap(corpus, bench), set())


class TestFixtureMatchesSiblingAndCache(unittest.TestCase):
    def test_committed_fixture_equals_cache_when_present(self):
        try:
            names = contam.build_names_from_cache()
        except Exception as exc:  # noqa: BLE001 (present-only aid)
            self.skipTest(f"HF cache unavailable: {type(exc).__name__}")
        committed = set(contam.load_fixture()["function_names"])
        self.assertEqual(names, committed)


if __name__ == "__main__":
    unittest.main()
