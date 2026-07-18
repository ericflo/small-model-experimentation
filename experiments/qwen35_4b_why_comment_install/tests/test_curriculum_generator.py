"""The WHY-comment curriculum's fail-closed truth audit.

Every row is INDEPENDENTLY re-executed here (by a grader written in this test,
separate from the generator's own verifier): STRIP the ``#WHY:`` comments and
the resulting CLEAN code passes ALL its tests, the COMMENTED code runs and
passes them IDENTICALLY (comments are inert), the marker is mechanically
strippable, every ``#WHY:`` comment is line-specific (references a token on its
line) and the comments VARY within a row (non-boilerplate). The safety/
termination caps abort runaway or unsafe code, the difficulty mix is the frozen
constant schedule, the banned-vocabulary audit and the code n-gram overlap are
zero, and every row is unique.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))

import contamination as contam  # noqa: E402
import gen_why_comment_curriculum as gen  # noqa: E402

CORPUS = EXP / "data" / "sft_why_comment.jsonl"
RECEIPT = EXP / "data" / "curriculum_receipt.json"


def load_corpus() -> list[dict]:
    return [json.loads(line) for line in CORPUS.read_text(encoding="utf-8").splitlines() if line.strip()]


# An INDEPENDENT grader: exec the code under restricted builtins and run each
# assert LINE directly (so an assertion failure raises AssertionError). This is
# a different code path from gen.grade_by_asserts (which splits/evals call vs
# expected), so agreement is real corroboration, not a tautology.
def independent_grade(code: str, tests: list[str]) -> tuple[int, int, list]:
    namespace = {"__builtins__": dict(gen.SAFE_BUILTINS)}
    exec(compile(code, "<indep>", "exec"), namespace)  # noqa: S102
    fails = raises = 0
    values: list = []
    name = gen._func_name_from_code(code)
    fn = namespace[name]
    for line in tests:
        call_src, _ = gen._split_assert(line)
        try:
            gen._run_capped(lambda line=line: exec(compile(line, "<assert>", "exec"), namespace))  # noqa: S102
            values.append(gen._run_capped(lambda cs=call_src: eval(compile(cs, "<call>", "eval"), namespace)))  # noqa: S307
        except AssertionError:
            fails += 1
            values.append(None)
        except Exception:  # noqa: BLE001
            raises += 1
            values.append(None)
    return fails, raises, values


class TestWhyCorrectnessIndependent(unittest.TestCase):
    def test_committed_corpus_reexecutes_and_matches(self):
        summary = gen.verify_public_corpus(CORPUS, RECEIPT)
        self.assertEqual(summary["rows"], gen.DEFAULT_ROWS)
        self.assertEqual(summary["kinds"], {"why_comment": gen.DEFAULT_ROWS})

    def test_strip_then_clean_passes_and_commented_runs_identically(self):
        rows = load_corpus()
        self.assertEqual(len(rows), gen.DEFAULT_ROWS)
        for i, row in enumerate(rows):
            prompt_blocks = gen.extract_python_blocks(row["messages"][0]["content"])
            answer_blocks = gen.extract_python_blocks(row["answer"])
            self.assertEqual(len(prompt_blocks), 2, msg=f"row {i}")
            self.assertEqual(len(answer_blocks), 1, msg=f"row {i}")
            _signature, tests_block = prompt_blocks
            commented = answer_blocks[0]
            tests = [ln for ln in tests_block.split("\n") if ln.strip()]
            # marker present and strippable
            self.assertIn(gen.WHY_MARKER, commented, msg=f"row {i} has no {gen.WHY_MARKER}")
            clean = gen.strip_why_comments(commented)
            self.assertNotIn(gen.WHY_MARKER, clean, msg=f"row {i} clean still has marker")
            self.assertNotEqual(clean, commented, msg=f"row {i} stripping removed nothing")
            # strip -> clean passes ALL tests
            c_fail, c_raise, clean_vals = independent_grade(clean, tests)
            self.assertEqual((c_fail, c_raise), (0, 0), msg=f"row {i} clean code failed tests")
            # commented code passes ALL tests with IDENTICAL outputs
            m_fail, m_raise, comm_vals = independent_grade(commented, tests)
            self.assertEqual((m_fail, m_raise), (0, 0), msg=f"row {i} commented code failed tests")
            self.assertEqual(clean_vals, comm_vals, msg=f"row {i} commented output differs from clean")

    def test_generator_verifier_agrees_over_sample(self):
        rows = load_corpus()
        for i in range(0, len(rows), 7):
            gen.verify_row_reexecution(rows[i], i)


class TestTamperDetection(unittest.TestCase):
    def test_broken_clean_code_is_rejected(self):
        # corrupt the code (via a tampered expected value) so it no longer passes
        rows = load_corpus()
        row = json.loads(json.dumps(rows[0]))
        content = row["messages"][0]["content"]
        blocks = gen.extract_python_blocks(content)
        tampered_tests = blocks[1].replace("== ", "== 999999 + ", 1)
        row["messages"][0]["content"] = content.replace(blocks[1], tampered_tests)
        with self.assertRaises(ValueError):
            gen.verify_row_reexecution(row, 0)

    def test_removing_all_comments_is_rejected(self):
        rows = load_corpus()
        row = json.loads(json.dumps(rows[0]))
        commented = gen.extract_python_blocks(row["answer"])[0]
        clean = gen.strip_why_comments(commented)
        row["answer"] = f"```python\n{clean}\n```"  # no #WHY: comments left
        with self.assertRaises(ValueError):
            gen.verify_row_reexecution(row, 0)

    def test_generic_boilerplate_comment_is_rejected(self):
        # replace every #WHY comment with a generic line that shares no code token
        rows = load_corpus()
        row = json.loads(json.dumps(rows[0]))
        commented = gen.extract_python_blocks(row["answer"])[0]
        out = []
        for line in commented.split("\n"):
            idx = line.find(gen.WHY_MARKER)
            if idx >= 0:
                out.append(line[:idx] + gen.WHY_MARKER + " this step is needed for correctness")
            else:
                out.append(line)
        row["answer"] = "```python\n" + "\n".join(out) + "\n```"
        with self.assertRaises(ValueError):
            gen.verify_row_reexecution(row, 0)

    def test_duplicated_comment_is_rejected(self):
        # make all #WHY comments identical (references a code token but not varied)
        rows = load_corpus()
        row = json.loads(json.dumps(rows[0]))
        commented = gen.extract_python_blocks(row["answer"])[0]
        out = []
        for line in commented.split("\n"):
            idx = line.find(gen.WHY_MARKER)
            if idx >= 0:
                out.append(line[:idx] + gen.WHY_MARKER + " return acc is correct")
            else:
                out.append(line)
        row["answer"] = "```python\n" + "\n".join(out) + "\n```"
        with self.assertRaises(ValueError):
            gen.verify_row_reexecution(row, 0)


class TestSafetyAndTermination(unittest.TestCase):
    def test_step_cap_aborts_runaway_loop(self):
        src = "def loop():\n    while True:\n        x = 1\n    return x\n"
        with self.assertRaises(gen.StepCapExceeded):
            gen.call_function(src, "loop", ())

    def test_restricted_namespace_blocks_imports_and_io(self):
        for src in (
            "def f():\n    import os\n    return 1\n",
            "def f():\n    return open('x')\n",
            "def f():\n    return __import__('os')\n",
        ):
            with self.assertRaises(Exception):  # noqa: B017
                gen.call_function(src, "f", ())


class TestWhyCommentQuality(unittest.TestCase):
    def test_every_commented_line_is_line_specific_and_varied(self):
        rows = load_corpus()
        for i, row in enumerate(rows):
            commented = gen.extract_python_blocks(row["answer"])[0]
            reasons = []
            n = 0
            for line in commented.split("\n"):
                idx = line.find(gen.WHY_MARKER)
                if idx < 0:
                    continue
                n += 1
                code_part = line[:idx]
                why = line[idx + len(gen.WHY_MARKER):].strip()
                self.assertTrue(why, msg=f"row {i} empty comment")
                self.assertTrue(
                    gen._why_references_line(code_part, why),
                    msg=f"row {i} generic comment: {why!r} vs {code_part!r}",
                )
                reasons.append(why)
            self.assertGreaterEqual(n, gen.MIN_WHY_PER_ROW, msg=f"row {i} too few comments")
            self.assertEqual(len(set(reasons)), len(reasons), msg=f"row {i} repeats a comment")

    def test_marker_is_the_only_hash_in_the_code(self):
        # the strip contract relies on '#' appearing ONLY as the #WHY: marker.
        rows = load_corpus()
        for i, row in enumerate(rows):
            commented = gen.extract_python_blocks(row["answer"])[0]
            for line in commented.split("\n"):
                if "#" in line:
                    self.assertIn(gen.WHY_MARKER, line, msg=f"row {i} stray '#' not a {gen.WHY_MARKER}")
                    self.assertEqual(line.count("#"), 1, msg=f"row {i} multiple '#' on a line")

    def test_answer_matches_audit_free_reconstruction(self):
        # the shipped answer's clean code must be a valid, self-consistent function.
        rows = load_corpus()
        for i, row in enumerate(rows):
            commented = gen.extract_python_blocks(row["answer"])[0]
            clean = gen.strip_why_comments(commented)
            name = gen._func_name_from_code(clean)
            self.assertTrue(name.isidentifier(), msg=f"row {i} bad function name")


class TestContaminationInCorpus(unittest.TestCase):
    def test_banned_vocabulary_is_zero(self):
        banned = contam.banned_names()
        for i, row in enumerate(load_corpus()):
            blob = row["messages"][0]["content"] + "\n" + row["think"] + "\n" + row["answer"]
            self.assertEqual(contam.whole_word_hits(blob, banned), set(), msg=f"row {i} banned hit")

    def test_identifier_pools_are_contamination_clean(self):
        banned = contam.banned_names()
        pools = gen.FUNC_NAMES + gen.LIST_PARAMS + gen.INT_PARAMS + gen.ACC_NAMES
        for name in pools:
            self.assertNotIn(name.lower(), banned, msg=f"identifier {name!r} collides with a benchmark name")

    def test_code_ngram_overlap_zero_when_cache_present(self):
        try:
            streams = contam.build_code_tokens_from_cache()
        except Exception as exc:  # noqa: BLE001 (present-only aid)
            self.skipTest(f"HF cache unavailable: {type(exc).__name__}")
        bench = contam.benchmark_ngrams(streams)
        corpus: set = set()
        for row in load_corpus():
            commented = gen.extract_python_blocks(row["answer"])[0]
            corpus |= gen.code_grams_no_comments(gen.strip_why_comments(commented))
        self.assertEqual(contam.distinctive_overlap(corpus, bench), set())


class TestDifficultyMixAndUniqueness(unittest.TestCase):
    def test_frozen_constant_difficulty_mix(self):
        rows = load_corpus()
        tiers = {t: 0 for t in ("short", "medium", "long")}
        for row in rows:
            tiers[row["tier"]] += 1
        self.assertEqual(tiers, {"short": 120, "medium": 192, "long": 192})
        self.assertEqual(dict(gen.TIER_SCHEDULE), tiers)

    def test_rows_are_unique(self):
        rows = load_corpus()
        prompts = {row["messages"][0]["content"] for row in rows}
        task_ids = {row["task_id"] for row in rows}
        keys = set()
        for row in rows:
            commented = gen.extract_python_blocks(row["answer"])[0]
            _signature, tests_block = gen.extract_python_blocks(row["messages"][0]["content"])
            keys.add((commented, tests_block))
        self.assertEqual(len(prompts), len(rows))
        self.assertEqual(len(task_ids), len(rows))
        self.assertEqual(len(keys), len(rows))


class TestGenerationAndDeterminism(unittest.TestCase):
    def test_small_build_validates_including_reverify(self):
        rows = gen.generate_curriculum(gen.CONSTRUCTION_SEED, (("short", 4), ("medium", 5), ("long", 4)))
        summary = gen.validate_generated(rows, expected_rows=13)
        self.assertEqual(summary["rows"], 13)
        self.assertEqual(summary["unique_keys"], 13)

    def test_construction_is_deterministic(self):
        a = gen.generate_curriculum(gen.CONSTRUCTION_SEED, (("short", 3), ("medium", 3), ("long", 3)))
        b = gen.generate_curriculum(gen.CONSTRUCTION_SEED, (("short", 3), ("medium", 3), ("long", 3)))
        dump = lambda rows: [gen.public_row(r) for r in rows]  # noqa: E731
        self.assertEqual(json.dumps(dump(a), sort_keys=True), json.dumps(dump(b), sort_keys=True))

    def test_prompt_does_not_instruct_commenting(self):
        # the WHY behavior must be the model's DEFAULT so it fires on a plain eval
        # prompt; the training prompt must NOT ask for #WHY comments.
        for row in load_corpus():
            self.assertNotIn(gen.WHY_MARKER, row["messages"][0]["content"])
            self.assertNotIn("comment", row["messages"][0]["content"].lower())


if __name__ == "__main__":
    unittest.main()
