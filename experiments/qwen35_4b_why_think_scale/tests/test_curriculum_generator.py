"""The WHY scale curriculum's fail-closed truth audit (independent re-execution).

Every row is INDEPENDENTLY re-executed by a grader written in this test (separate
from the generator's own verifier): STRIP the ``#WHY:`` comments and the resulting
CLEAN code passes ALL its tests, the COMMENTED code runs and passes them
IDENTICALLY (comments inert), the marker is mechanically strippable, every
``#WHY:`` comment is line-specific and the comments VARY within a row
(non-boilerplate). Safety/termination caps abort runaway/unsafe code, the build is
deterministic, and rows are unique programs with unique prompts and no comment
instruction in the prompt.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))

import gen_why_think_curriculum as gen  # noqa: E402

SEED = gen.CONSTRUCTION_SEED
SAMPLE_ROWS = 800


def _corpus(rows: int = SAMPLE_ROWS) -> list[dict]:
    return gen.generate_curriculum(SEED, rows)


# An INDEPENDENT grader: exec the code under restricted builtins and run each
# assert LINE directly (a different code path from gen.grade_by_asserts).
def independent_grade(code: str, tests: list[str]) -> tuple[int, int, list]:
    namespace = {"__builtins__": dict(gen.SAFE_BUILTINS)}
    exec(compile(code, "<indep>", "exec"), namespace)  # noqa: S102
    fails = raises = 0
    values: list = []
    name = gen._func_name_from_code(code)
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
    @classmethod
    def setUpClass(cls):
        cls.rows = _corpus()

    def test_strip_then_clean_passes_and_commented_runs_identically(self):
        for i, row in enumerate(self.rows):
            prompt_blocks = gen.extract_python_blocks(row["messages"][0]["content"])
            answer_blocks = gen.extract_python_blocks(row["answer"])
            self.assertEqual(len(prompt_blocks), 2, msg=f"row {i}")
            self.assertEqual(len(answer_blocks), 1, msg=f"row {i}")
            _signature, tests_block = prompt_blocks
            commented = answer_blocks[0]
            tests = [ln for ln in tests_block.split("\n") if ln.strip()]
            self.assertIn(gen.WHY_MARKER, commented, msg=f"row {i} has no {gen.WHY_MARKER}")
            clean = gen.strip_why_comments(commented)
            self.assertNotIn(gen.WHY_MARKER, clean, msg=f"row {i} clean still has marker")
            self.assertNotEqual(clean, commented, msg=f"row {i} stripping removed nothing")
            c_fail, c_raise, clean_vals = independent_grade(clean, tests)
            self.assertEqual((c_fail, c_raise), (0, 0), msg=f"row {i} clean code failed tests")
            m_fail, m_raise, comm_vals = independent_grade(commented, tests)
            self.assertEqual((m_fail, m_raise), (0, 0), msg=f"row {i} commented code failed tests")
            self.assertEqual(clean_vals, comm_vals, msg=f"row {i} commented output differs from clean")

    def test_generator_verifier_agrees_over_sample(self):
        for i in range(0, len(self.rows), 5):
            gen.verify_row_reexecution(self.rows[i], i)

    def test_every_commented_line_is_line_specific_and_varied(self):
        for i, row in enumerate(self.rows):
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
        for i, row in enumerate(self.rows):
            commented = gen.extract_python_blocks(row["answer"])[0]
            for line in commented.split("\n"):
                if "#" in line:
                    self.assertIn(gen.WHY_MARKER, line, msg=f"row {i} stray '#' not a {gen.WHY_MARKER}")
                    self.assertEqual(line.count("#"), 1, msg=f"row {i} multiple '#' on a line")


class TestTamperDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = gen.generate_curriculum(SEED, 40)

    def test_broken_clean_code_is_rejected(self):
        row = json.loads(json.dumps(self.rows[0]))
        content = row["messages"][0]["content"]
        blocks = gen.extract_python_blocks(content)
        tampered_tests = blocks[1].replace("== ", "== 999999 + ", 1)
        row["messages"][0]["content"] = content.replace(blocks[1], tampered_tests)
        with self.assertRaises(ValueError):
            gen.verify_row_reexecution(row, 0)

    def test_removing_all_comments_is_rejected(self):
        row = json.loads(json.dumps(self.rows[0]))
        commented = gen.extract_python_blocks(row["answer"])[0]
        clean = gen.strip_why_comments(commented)
        row["answer"] = f"```python\n{clean}\n```"
        with self.assertRaises(ValueError):
            gen.verify_row_reexecution(row, 0)

    def test_generic_boilerplate_comment_is_rejected(self):
        row = json.loads(json.dumps(self.rows[0]))
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
        row = json.loads(json.dumps(self.rows[0]))
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

    def test_generated_code_uses_no_while_loops(self):
        for i, row in enumerate(gen.generate_curriculum(SEED, 400)):
            clean = row["_audit"]["clean_code"]
            self.assertNotIn("while", clean, msg=f"row {i} clean code uses a while loop")


class TestGenerationAndDeterminism(unittest.TestCase):
    def test_small_build_validates_including_reverify(self):
        rows = gen.generate_curriculum(SEED, 120)
        summary = gen.validate_generated(rows, expected_rows=120)
        self.assertEqual(summary["rows"], 120)
        self.assertEqual(summary["unique_programs"], 120)

    def test_exact_row_count_for_arbitrary_n(self):
        for n in (1, 7, 333, 1001):
            self.assertEqual(len(gen.generate_curriculum(SEED, n)), n)

    def test_construction_is_deterministic(self):
        a = gen.generate_curriculum(SEED, 400)
        b = gen.generate_curriculum(SEED, 400)
        dump = lambda rows: "".join(json.dumps(gen.public_row(r), ensure_ascii=False) + "\n" for r in rows)  # noqa: E731
        self.assertEqual(gen.sha256_text(dump(a)), gen.sha256_text(dump(b)))

    def test_rows_are_unique_programs_and_prompts(self):
        rows = gen.generate_curriculum(SEED, 1500)
        prompts = {row["messages"][0]["content"] for row in rows}
        task_ids = {row["task_id"] for row in rows}
        programs = {row["_audit"]["clean_code"] for row in rows}
        self.assertEqual(len(prompts), len(rows))
        self.assertEqual(len(task_ids), len(rows))
        self.assertEqual(len(programs), len(rows))

    def test_prompt_does_not_instruct_commenting(self):
        for row in gen.generate_curriculum(SEED, 400):
            self.assertNotIn(gen.WHY_MARKER, row["messages"][0]["content"])
            self.assertNotIn("comment", row["messages"][0]["content"].lower())

    def test_token_budget_under_cap(self):
        stats = gen._length_stats(gen.generate_curriculum(SEED, 1000))
        # conservative >=3 chars/token estimate of the full render, well under 4096.
        self.assertLess(stats["est_tokens_max_chars_over_3"], 4096, msg=stats)


if __name__ == "__main__":
    unittest.main()


class TestThinkDerivationChannel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = gen.generate_curriculum(SEED, 600)

    def test_think_has_approach_and_trace_and_is_not_joined_why(self):
        for i, row in enumerate(self.rows):
            think = row["think"]
            self.assertTrue(gen._APPROACH_MARKER.search(think), msg=f"row {i} no approach phrase")
            self.assertIn(gen.TRACE_LABEL, think, msg=f"row {i} no worked-example trace")
            reasons = row["_audit"]["why_reasons"]
            for joined in ("\n".join(reasons), " ".join(reasons), gen.WHY_JOIN.join(reasons)):
                self.assertNotEqual(think.strip(), joined.strip(), msg=f"row {i} think == joined #WHY")
            self.assertNotEqual(gen.normalize_think(think),
                                gen.normalize_think(" ".join(reasons)), msg=f"row {i}")
            # the shipped think never leaks the #WHY marker (channels stay distinct).
            self.assertNotIn(gen.WHY_MARKER, think, msg=f"row {i} think leaks the #WHY marker")

    def test_worked_example_trace_matches_real_execution_independently(self):
        # Independent code path: strip to clean code, exec it in this test's own
        # namespace, re-run the traced call, and byte-verify the final result that
        # the trace claims equals the real output AND the matching assert's expected.
        import re
        for i, row in enumerate(self.rows):
            commented = gen.extract_python_blocks(row["answer"])[0]
            clean = gen.strip_why_comments(commented)
            name = gen._func_name_from_code(clean)
            ns = {"__builtins__": dict(gen.SAFE_BUILTINS)}
            exec(compile(clean, "<indep>", "exec"), ns)  # noqa: S102
            _sig, tests_block = gen.extract_python_blocks(row["messages"][0]["content"])
            expected_by_call = {}
            for line in [l for l in tests_block.split("\n") if l.strip()]:
                call_src, exp_src = gen._split_assert(line)
                expected_by_call[call_src] = gen._run_capped(
                    lambda es=exp_src: eval(compile(es, "<e>", "eval"), dict(ns)))  # noqa: S307
            matches = re.findall(r"Trace (" + re.escape(name) + r"\([^:]*?\)): .*? so it returns ([^\n]+?)\.",
                                 row["think"])
            self.assertGreaterEqual(len(matches), 1, msg=f"row {i} no parseable trace line")
            for call_src, claimed in matches:
                got = gen._run_capped(lambda cs=call_src: eval(compile(cs, "<c>", "eval"), dict(ns)))  # noqa: S307
                self.assertEqual(repr(got), claimed.strip(), msg=f"row {i} trace result != real output")
                self.assertIn(call_src, expected_by_call, msg=f"row {i} traced a non-asserted input")
                self.assertEqual(got, expected_by_call[call_src], msg=f"row {i} trace != assert expected")

    def test_generator_verify_think_agrees(self):
        for i in range(0, len(self.rows), 4):
            row = self.rows[i]
            commented = gen.extract_python_blocks(row["answer"])[0]
            clean = gen.strip_why_comments(commented)
            name = gen._func_name_from_code(clean)
            _sig, tests_block = gen.extract_python_blocks(row["messages"][0]["content"])
            tests = [l for l in tests_block.split("\n") if l.strip()]
            gen.verify_think(row["think"], clean, name, tests, row["_audit"]["why_reasons"], i)


class TestThinkTamperDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = gen.generate_curriculum(SEED, 40)

    def _clean_name_tests(self, row):
        commented = gen.extract_python_blocks(row["answer"])[0]
        clean = gen.strip_why_comments(commented)
        name = gen._func_name_from_code(clean)
        _sig, tests_block = gen.extract_python_blocks(row["messages"][0]["content"])
        tests = [l for l in tests_block.split("\n") if l.strip()]
        return clean, name, tests

    def test_think_equal_to_joined_why_is_rejected(self):
        row = json.loads(json.dumps(self.rows[0]))
        clean, name, tests = self._clean_name_tests(row)
        row["think"] = " ".join(row["_audit"]["why_reasons"])
        with self.assertRaises(ValueError):
            gen.verify_think(row["think"], clean, name, tests, row["_audit"]["why_reasons"], 0)

    def test_tampered_trace_value_is_rejected(self):
        # corrupt a digit inside the worked-example trace: the recomputed core no
        # longer appears verbatim, so byte-verification fails.
        for row0 in self.rows:
            if "0 -> 1" in row0["think"] or " so it returns " in row0["think"]:
                row = json.loads(json.dumps(row0))
                break
        clean, name, tests = self._clean_name_tests(row)
        original = row["think"]
        tampered = original.replace("so it returns ", "so it returns 999", 1)
        self.assertNotEqual(tampered, original)
        row["think"] = tampered
        with self.assertRaises(ValueError):
            gen.verify_think(row["think"], clean, name, tests, row["_audit"]["why_reasons"], 0)

    def test_missing_approach_phrase_is_rejected(self):
        row = json.loads(json.dumps(self.rows[0]))
        clean, name, tests = self._clean_name_tests(row)
        # keep only the trace line (strip the approach), so the approach marker is gone.
        trace_line = [l for l in row["think"].split("\n") if gen.TRACE_LABEL in l][0]
        row["think"] = trace_line.replace("I'll", "it will").replace("My plan", "the plan").replace("I need to", "one must")
        with self.assertRaises(ValueError):
            gen.verify_think(row["think"], clean, name, tests, row["_audit"]["why_reasons"], 0)
