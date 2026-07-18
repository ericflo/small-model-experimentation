"""The self-repair curriculum's fail-closed truth audit.

Every row is INDEPENDENTLY re-executed here (by a grader written in this test,
separate from the generator's own verifier): the corrected code passes ALL its
tests, the buggy code fails AT LEAST ONE with a wrong value and RAISES on none,
and the correction differs from the buggy code. The mutation set is diverse, the
safety/termination caps abort runaway or unsafe code, the difficulty mix is the
frozen constant schedule, the banned-vocabulary audit and the code n-gram
overlap are zero, and every row is unique.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))

import contamination as contam  # noqa: E402
import gen_self_repair_curriculum as gen  # noqa: E402

CORPUS = EXP / "data" / "sft_self_repair.jsonl"
RECEIPT = EXP / "data" / "curriculum_receipt.json"


def load_corpus() -> list[dict]:
    return [json.loads(line) for line in CORPUS.read_text(encoding="utf-8").splitlines() if line.strip()]


# An INDEPENDENT grader: exec the code under restricted builtins and run each
# assert LINE directly (so an assertion failure raises AssertionError). This is
# a different code path from gen.grade_by_asserts (which splits/evals call vs
# expected), so agreement is real corroboration, not a tautology.
def independent_grade(code: str, tests: list[str]) -> tuple[int, int]:
    namespace = {"__builtins__": dict(gen.SAFE_BUILTINS)}
    exec(compile(code, "<indep>", "exec"), namespace)  # noqa: S102
    fails = raises = 0
    for line in tests:
        try:
            gen._run_capped(lambda line=line: exec(compile(line, "<assert>", "exec"), namespace))  # noqa: S102
        except AssertionError:
            fails += 1
        except Exception:  # noqa: BLE001
            raises += 1
    return fails, raises


class TestRepairCorrectnessIndependent(unittest.TestCase):
    def test_committed_corpus_reexecutes_and_matches(self):
        summary = gen.verify_public_corpus(CORPUS, RECEIPT)
        self.assertEqual(summary["rows"], gen.DEFAULT_ROWS)
        self.assertEqual(summary["kinds"], {"self_repair": gen.DEFAULT_ROWS})

    def test_every_row_buggy_fails_and_corrected_passes_independently(self):
        rows = load_corpus()
        self.assertEqual(len(rows), gen.DEFAULT_ROWS)
        for i, row in enumerate(rows):
            prompt_blocks = gen.extract_python_blocks(row["messages"][0]["content"])
            answer_blocks = gen.extract_python_blocks(row["answer"])
            self.assertEqual(len(prompt_blocks), 2, msg=f"row {i}")
            self.assertEqual(len(answer_blocks), 1, msg=f"row {i}")
            buggy, tests_block = prompt_blocks
            corrected = answer_blocks[0]
            tests = [ln for ln in tests_block.split("\n") if ln.strip()]
            self.assertNotEqual(buggy, corrected, msg=f"row {i} correction equals buggy")
            b_fail, b_raise = independent_grade(buggy, tests)
            self.assertGreaterEqual(b_fail, 1, msg=f"row {i} buggy passed all tests")
            self.assertEqual(b_raise, 0, msg=f"row {i} buggy raised (want a wrong value)")
            c_fail, c_raise = independent_grade(corrected, tests)
            self.assertEqual((c_fail, c_raise), (0, 0), msg=f"row {i} corrected did not pass all tests")

    def test_shown_failure_matches_actual_first_failure(self):
        # gen.verify_row_reexecution cross-checks the shown expected/got against
        # a fresh evaluation; run it over a sample as an extra guard.
        rows = load_corpus()
        for i in range(0, len(rows), 7):
            gen.verify_row_reexecution(rows[i], i)


class TestTamperDetection(unittest.TestCase):
    def test_correction_equal_to_buggy_is_rejected(self):
        rows = load_corpus()
        row = json.loads(json.dumps(rows[0]))
        buggy = gen.extract_python_blocks(row["messages"][0]["content"])[0]
        row["answer"] = f"```python\n{buggy}\n```"  # correction == buggy
        with self.assertRaises(ValueError):
            gen.verify_row_reexecution(row, 0)

    def test_tampered_expected_value_is_caught(self):
        rows = load_corpus()
        row = json.loads(json.dumps(rows[0]))
        # corrupt one expected value in the shown tests so corrected no longer passes
        content = row["messages"][0]["content"]
        blocks = gen.extract_python_blocks(content)
        tampered_tests = blocks[1].replace("== ", "== 999999 + ", 1)
        row["messages"][0]["content"] = content.replace(blocks[1], tampered_tests)
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


class TestMutationDiversity(unittest.TestCase):
    def test_all_mutation_kinds_present_with_spread(self):
        rows = load_corpus()
        counts: dict[str, int] = {}
        for row in rows:
            counts[row["mutation_kind"]] = counts.get(row["mutation_kind"], 0) + 1
        self.assertEqual(set(counts), set(gen.ALL_MUTATION_KINDS),
                         msg=f"mutation kinds present: {sorted(counts)}")
        for kind, n in counts.items():
            self.assertGreaterEqual(n, 5, msg=f"mutation kind {kind} underrepresented ({n})")

    def test_diagnosis_names_the_changed_line(self):
        # every diagnosis quotes the buggy line and the corrected line verbatim.
        rows = load_corpus()
        for i, row in enumerate(rows):
            buggy = gen.extract_python_blocks(row["messages"][0]["content"])[0]
            corrected = gen.extract_python_blocks(row["answer"])[0]
            cl, bl = corrected.split("\n"), buggy.split("\n")
            changed = [j for j in range(min(len(cl), len(bl))) if cl[j] != bl[j]]
            self.assertEqual(len(changed), 1, msg=f"row {i} not a single-line bug")
            self.assertIn(bl[changed[0]].strip(), row["think"], msg=f"row {i} buggy line not diagnosed")
            self.assertIn(cl[changed[0]].strip(), row["think"], msg=f"row {i} fix line not diagnosed")


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
            corrected = gen.extract_python_blocks(row["answer"])[0]
            buggy = gen.extract_python_blocks(row["messages"][0]["content"])[0]
            corpus |= gen.code_grams_no_docstring(corrected)
            corpus |= gen.code_grams_no_docstring(buggy)
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
        pairs = set()
        for row in rows:
            buggy = gen.extract_python_blocks(row["messages"][0]["content"])[0]
            corrected = gen.extract_python_blocks(row["answer"])[0]
            pairs.add((buggy, corrected))
        self.assertEqual(len(prompts), len(rows))
        self.assertEqual(len(task_ids), len(rows))
        self.assertEqual(len(pairs), len(rows))


class TestGenerationAndDeterminism(unittest.TestCase):
    def test_small_build_validates_including_reverify(self):
        rows = gen.generate_curriculum(gen.CONSTRUCTION_SEED, (("short", 4), ("medium", 5), ("long", 4)))
        summary = gen.validate_generated(rows, expected_rows=13)
        self.assertEqual(summary["rows"], 13)
        self.assertEqual(summary["unique_code_pairs"], 13)

    def test_construction_is_deterministic(self):
        a = gen.generate_curriculum(gen.CONSTRUCTION_SEED, (("short", 3), ("medium", 3), ("long", 3)))
        b = gen.generate_curriculum(gen.CONSTRUCTION_SEED, (("short", 3), ("medium", 3), ("long", 3)))
        dump = lambda rows: [gen.public_row(r) for r in rows]  # noqa: E731
        self.assertEqual(json.dumps(dump(a), sort_keys=True), json.dumps(dump(b), sort_keys=True))

    def test_retention_flag_blends_spec_to_code_rows(self):
        rows = gen.generate_curriculum(
            gen.CONSTRUCTION_SEED, (("short", 3), ("medium", 3), ("long", 3)), mix_retention=5
        )
        kinds: dict[str, int] = {}
        for row in rows:
            kinds[row["kind"]] = kinds.get(row["kind"], 0) + 1
        self.assertEqual(kinds.get("code_retention"), 5)
        self.assertEqual(kinds.get("self_repair"), 9)
        retention = [r for r in rows if r["kind"] == "code_retention"]
        # retention rows are spec->code: prompt asks to WRITE a function, answer is code.
        self.assertTrue(all("Write a Python function" in r["messages"][0]["content"] for r in retention))
        self.assertTrue(all(r["answer"].startswith("```python") for r in retention))


if __name__ == "__main__":
    unittest.main()
