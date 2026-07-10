from __future__ import annotations

import sys
import unittest
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import model_harness as harness  # noqa: E402


PRIMITIVES = {
    "REVERSE": "reverse the current list",
    "DROP_FIRST": "remove the first element",
    "ROTATE_LEFT": "move the first element to the end",
    "SORT_ASC": "sort ascending",
}

VERIFIED_PROGRAMS = [
    {
        "id": "train-001",
        "split": "train",
        "verified": True,
        "program": ["REVERSE", "DROP_FIRST", "SORT_ASC"],
        "io": [
            {"input": [3, 1, 2], "output": [1, 3]},
            {"input": [4, 2, 5, 1], "output": [2, 4, 5]},
        ],
    },
    {
        "id": "train-002",
        "split": "train",
        "verified": True,
        "program": ["REVERSE", "DROP_FIRST", "ROTATE_LEFT"],
        "io": [([1, 2, 3], [2, 1])],
    },
]


def completion_output(
    sample_index: int,
    text: str,
    *,
    stage1_prompt: int = 10,
    stage2_prompt: int = 0,
    sampled: int = 5,
    injected: int = 0,
    completion: int = 4,
    thinking: int = 0,
    answer: int = 4,
    trimmed: int = 1,
) -> dict:
    return {
        "sample_index": sample_index,
        "text": text,
        "token_ids": list(range(completion)),
        "n_stage1_prompt_tokens": stage1_prompt,
        "n_stage2_prompt_tokens": stage2_prompt,
        "n_sampled_tokens": sampled,
        "n_injected_tokens": injected,
        "n_completion_tokens": completion,
        "n_thinking_tokens": thinking,
        "n_answer_tokens": answer,
        "n_terminal_tokens_trimmed": trimmed,
    }


def summary_for(rows: list[dict]) -> dict:
    accounting = harness.extract_token_accounting(rows)
    return {
        "model": harness.REQUIRED_MODEL_ID,
        "model_revision": harness.MODEL_REVISION,
        "counts": {
            "requests": accounting.requests,
            "completions": accounting.completions,
            "unique_input_prompt_tokens": accounting.unique_input_prompt_tokens,
            "stage1_logical_prompt_tokens": accounting.stage1_logical_prompt_tokens,
            "stage2_logical_prompt_tokens": accounting.stage2_logical_prompt_tokens,
            "logical_model_input_tokens": accounting.logical_model_input_tokens,
            "sampled_tokens": accounting.sampled_tokens,
            "injected_tokens": accounting.injected_tokens,
        },
    }


class PromptBuilderTests(unittest.TestCase):
    def test_macro_prompt_is_train_only_and_has_strict_contract(self) -> None:
        record = harness.build_macro_proposal_record(
            "proposal-0",
            primitives=PRIMITIVES,
            verified_programs=VERIFIED_PROGRAMS,
            max_macros=3,
            meta={"arm": "model_discovered"},
        )
        self.assertEqual(record["id"], "proposal-0")
        self.assertNotIn("prompt", record)
        self.assertEqual(record["meta"]["prompt_kind"], "macro_proposal")
        self.assertEqual(record["meta"]["arm"], "model_discovered")
        system, user = (message["content"] for message in record["messages"])
        self.assertIn("Qwen", harness.REQUIRED_MODEL_ID)
        self.assertIn("MACRO: DESCRIPTIVE_NAME", system)
        self.assertIn("Do not return Python", system)
        self.assertIn("VERIFIED TRAIN-ONLY", user)
        self.assertIn("[train-001] PROGRAM: REVERSE | DROP_FIRST | SORT_ASC", user)
        self.assertIn("PROPOSE EXACTLY 3", user)
        self.assertIn("exactly 3 macro lines", user)
        self.assertNotIn("IO 1:", user)
        self.assertNotIn("IN=", user)

    def test_macro_prompt_fails_closed_on_eval_or_unverified_examples(self) -> None:
        leaked = [dict(VERIFIED_PROGRAMS[0], split="eval")]
        with self.assertRaisesRegex(ValueError, "refusing leakage"):
            harness.build_macro_proposal_messages(
                primitives=PRIMITIVES, verified_programs=leaked, max_macros=2
            )
        unverified = [dict(VERIFIED_PROGRAMS[0], verified=False)]
        with self.assertRaisesRegex(ValueError, "verified=True"):
            harness.build_macro_proposal_messages(
                primitives=PRIMITIVES, verified_programs=unverified, max_macros=2
            )
        malformed_io = [dict(VERIFIED_PROGRAMS[0], io=[{"input": [1, 2, 3]}])]
        with self.assertRaisesRegex(ValueError, "needs input and output"):
            harness.build_macro_proposal_messages(
                primitives=PRIMITIVES, verified_programs=malformed_io, max_macros=2
            )

    def test_solver_prompt_displays_expansions_and_both_depth_limits(self) -> None:
        record = harness.build_solver_record(
            "eval-8",
            primitives=PRIMITIVES,
            macros=[
                harness.MacroDefinition(
                    "M0", ("REVERSE", "DROP_FIRST"), "reverse then trim"
                )
            ],
            solved_demonstrations=[
                {
                    "id": "format-train-1",
                    "split": "train",
                    "verified": True,
                    "program": ["M0", "SORT_ASC"],
                    "io": [([3, 1, 2], [1, 3])],
                }
            ],
            io_examples=[([3, 2, 1], [2, 3]), ([7, 4], [7])],
            max_surface_calls=3,
            max_expanded_primitive_depth=5,
        )
        system, user = (message["content"] for message in record["messages"])
        self.assertIn("PROGRAM: TOKEN | TOKEN", system)
        self.assertIn("Do not return Python", system)
        self.assertIn("M0 := REVERSE | DROP_FIRST (reverse then trim)", user)
        self.assertIn("SOLVED TRAIN-ONLY FORMAT DEMONSTRATIONS", user)
        self.assertIn("[format-train-1]", user)
        self.assertIn("PROGRAM: M0 | SORT_ASC", user)
        self.assertIn("Maximum surface calls in returned PROGRAM: 3", user)
        self.assertIn("Maximum expanded primitive depth", user)
        self.assertIn("5", user)
        self.assertIn("If X := A | B", user)
        self.assertIn("X | C has 2 surface calls and expanded primitive depth 3", user)
        self.assertIn("SURFACE-FIRST PROCEDURE", user)
        self.assertIn("Propose the shortest legal surface programs first", user)
        self.assertIn("Expand every legal macro alias", user)
        self.assertIn("Execute each expanded candidate against every visible example", user)
        self.assertIn("Return the shortest candidate that passes every visible example", user)
        self.assertIn("Prefer a macro alias whenever its exact expansion is present", system)

    def test_solver_demonstrations_fail_closed_on_leakage_and_depth(self) -> None:
        common = dict(
            primitives=PRIMITIVES,
            macros={"M0": ("REVERSE", "DROP_FIRST")},
            io_examples=[([1, 2], [2, 1])],
            max_surface_calls=2,
            max_expanded_primitive_depth=3,
        )
        with self.assertRaisesRegex(ValueError, "refusing leakage"):
            harness.build_solver_messages(
                **common,
                solved_demonstrations=[
                    {
                        "id": "eval-demo",
                        "split": "eval",
                        "verified": True,
                        "program": ["M0"],
                        "io": [([1, 2], [1])],
                    }
                ],
            )
        with self.assertRaisesRegex(ValueError, "expanded depth"):
            harness.build_solver_messages(
                **common,
                solved_demonstrations=[
                    {
                        "id": "too-deep",
                        "split": "train",
                        "verified": True,
                        "program": ["M0", "M0"],
                        "io": [([1, 2], [])],
                    }
                ],
            )

    def test_noncallable_chunks_change_only_permissions_and_keep_expanded_demos(self) -> None:
        kwargs = dict(
            primitives=PRIMITIVES,
            macros=[
                harness.MacroDefinition(
                    "M0", ("REVERSE", "DROP_FIRST"), "reverse then trim"
                )
            ],
            solved_demonstrations=[
                {
                    "id": "format-train-expanded",
                    "split": "train",
                    "verified": True,
                    "program": ["REVERSE", "DROP_FIRST", "SORT_ASC"],
                    "io": [([3, 1, 2], [1, 3])],
                }
            ],
            io_examples=[([3, 2, 1], [2, 3])],
            max_surface_calls=3,
            max_expanded_primitive_depth=3,
        )
        callable_messages = harness.build_solver_messages(**kwargs, macros_callable=True)
        common_messages = harness.build_solver_messages(**kwargs, macros_callable=False)

        callable_system = callable_messages[0]["content"]
        expected_common_system = callable_system.replace(
            "Use only tokens in the supplied BASE and MACRO inventories. Macro tokens are "
            "legal calls and stand for their exact displayed primitive expansion. Prefer a "
            "macro alias whenever its exact expansion is present in a candidate; the alias is "
            "one surface call.",
            "Use only tokens in the supplied BASE inventory. Macro aliases are reference "
            "chunks, are not legal calls, and must not appear in the returned program. Keep any "
            "matching expansion as BASE tokens.",
        )
        self.assertEqual(common_messages[0]["content"], expected_common_system)

        callable_user = callable_messages[1]["content"]
        expected_common_user = callable_user.replace(
            "FROZEN VERIFIED MACROS:",
            "VERIFIED COMMON CHUNKS (not legal output tokens):",
        )
        self.assertEqual(common_messages[1]["content"], expected_common_user)
        for prompt in (callable_user, common_messages[1]["content"]):
            self.assertIn("If X := A | B", prompt)
            self.assertIn("SURFACE-FIRST PROCEDURE", prompt)
            self.assertIn("Reject any candidate whose expanded primitive depth exceeds 3", prompt)
        callable_demo = callable_user.split(
            "SOLVED TRAIN-ONLY FORMAT DEMONSTRATIONS:", 1
        )[1].split("\n\nLIMITS:", 1)[0]
        common_demo = common_messages[1]["content"].split(
            "SOLVED TRAIN-ONLY FORMAT DEMONSTRATIONS:", 1
        )[1].split("\n\nLIMITS:", 1)[0]
        self.assertEqual(callable_demo, common_demo)
        self.assertIn("PROGRAM: REVERSE | DROP_FIRST | SORT_ASC", callable_demo)
        self.assertNotIn("PROGRAM: M0", callable_demo)

        with self.assertRaisesRegex(ValueError, "outside its inventory"):
            harness.build_solver_messages(
                **{
                    **kwargs,
                    "solved_demonstrations": [
                        {
                            "id": "compressed-demo",
                            "split": "train",
                            "verified": True,
                            "program": ["M0", "SORT_ASC"],
                            "io": [([3, 1, 2], [1, 3])],
                        }
                    ],
                },
                macros_callable=False,
            )


class ParserTests(unittest.TestCase):
    def test_macro_parser_tolerates_thinking_fence_and_whitespace_only(self) -> None:
        text = "reasoning here</think>\n```text\n MACRO: TRIM_BACK = REVERSE | DROP_FIRST \n\nMACRO: CANON = SORT_ASC | ROTATE_LEFT | SORT_ASC\n```<|im_end|>"
        proposals = harness.parse_macro_proposals(
            text, allowed_primitives=set(PRIMITIVES), max_macros=2
        )
        self.assertEqual(
            proposals,
            (
                harness.MacroProposal("TRIM_BACK", ("REVERSE", "DROP_FIRST")),
                harness.MacroProposal(
                    "CANON", ("SORT_ASC", "ROTATE_LEFT", "SORT_ASC")
                ),
            ),
        )
        self.assertEqual(
            harness.parse_macro_proposals(
                "  NONE  ", allowed_primitives=set(PRIMITIVES), max_macros=2
            ),
            (),
        )

    def test_macro_parser_rejects_prose_unknowns_and_bad_lengths(self) -> None:
        for text, message in (
            ("Here is one:\nMACRO: X = REVERSE | DROP_FIRST", "invalid macro line"),
            ("MACRO: X = REVERSE | M0", "non-primitive"),
            ("MACRO: X = REVERSE", "invalid macro line"),
            (
                "MACRO: X = REVERSE | DROP_FIRST | SORT_ASC | ROTATE_LEFT",
                "invalid macro line",
            ),
            ("MACRO: lower = REVERSE | DROP_FIRST", "invalid macro line"),
        ):
            with self.subTest(text=text):
                with self.assertRaisesRegex(ValueError, message):
                    harness.parse_macro_proposals(
                        text, allowed_primitives=set(PRIMITIVES), max_macros=3
                    )

    def test_line_local_macro_extractor_recovers_first_eight_from_v1_failure(self) -> None:
        expansions = [
            "REVERSE | DROP_FIRST",
            "DROP_FIRST | REVERSE",
            "REVERSE | SORT_ASC",
            "SORT_ASC | REVERSE",
            "ROTATE_LEFT | DROP_FIRST",
            "DROP_FIRST | ROTATE_LEFT",
            "SORT_ASC | DROP_FIRST",
            "REVERSE | ROTATE_LEFT",
            "ROTATE_LEFT | SORT_ASC",
            "SORT_ASC | ROTATE_LEFT",
        ]
        lines = [
            f"MACRO: CANDIDATE_{index} = {expansion}"
            for index, expansion in enumerate(expansions)
        ]
        raw = (
            "manual corpus scan that exhausted the thinking budget</think>\n"
            + "\n".join(lines)
            + "\nWait, I should check the constraints.<|im_end|>"
        )
        extracted = harness.extract_macro_proposal_lines(
            raw, allowed_primitives=set(PRIMITIVES), max_macros=8
        )
        self.assertEqual(len(extracted.proposals), 8)
        self.assertEqual(extracted.proposals[0].name, "CANDIDATE_0")
        self.assertEqual(extracted.proposals[-1].name, "CANDIDATE_7")
        self.assertEqual(extracted.total_valid_lines, 10)
        self.assertTrue(extracted.extra_valid_lines_capped)
        self.assertEqual(
            [line.text for line in extracted.rejected_nonblank_lines],
            ["Wait, I should check the constraints."],
        )
        with self.assertRaisesRegex(ValueError, "returned 11 macros"):
            harness.parse_macro_proposals(
                raw, allowed_primitives=set(PRIMITIVES), max_macros=8
            )

    def test_line_local_macro_extractor_audits_fences_prose_and_bad_lines(self) -> None:
        raw = """reasoning</think>
```text
Here are the candidates:
MACRO: GOOD = REVERSE | DROP_FIRST
MACRO: BAD = REVERSE | M0
MACRO: REVERSE = REVERSE | DROP_FIRST
MACRO: GOOD = DROP_FIRST | REVERSE
MACRO: ALSO_GOOD = SORT_ASC | ROTATE_LEFT | SORT_ASC
```<|endoftext|>"""
        extracted = harness.extract_macro_proposal_lines(
            raw, allowed_primitives=set(PRIMITIVES), max_macros=3
        )
        self.assertEqual(
            extracted.proposals,
            (
                harness.MacroProposal("GOOD", ("REVERSE", "DROP_FIRST")),
                harness.MacroProposal(
                    "ALSO_GOOD", ("SORT_ASC", "ROTATE_LEFT", "SORT_ASC")
                ),
            ),
        )
        self.assertFalse(extracted.extra_valid_lines_capped)
        self.assertEqual(extracted.total_valid_lines, 2)
        rejected = extracted.rejected_nonblank_lines
        self.assertEqual(len(rejected), 6)
        self.assertEqual(rejected[0].text, "```text")
        self.assertIn("grammar", rejected[0].reason)
        self.assertIn("non-primitive", rejected[2].reason)
        self.assertIn("collides", rejected[3].reason)
        self.assertIn("duplicate", rejected[4].reason)
        self.assertEqual(rejected[5].text, "```")

    def test_program_parser_enforces_one_line_inventory_and_surface_limit(self) -> None:
        self.assertEqual(
            harness.parse_program(
                "</think>\n```\n PROGRAM: M0 | SORT_ASC \n```\n<|im_end|>",
                allowed_tokens={*PRIMITIVES, "M0"},
                max_surface_calls=2,
            ),
            ("M0", "SORT_ASC"),
        )
        with self.assertRaisesRegex(ValueError, "outside its supplied inventory"):
            harness.parse_program(
                "PROGRAM: M9", allowed_tokens={*PRIMITIVES, "M0"}
            )
        with self.assertRaisesRegex(ValueError, "surface calls"):
            harness.parse_program(
                "PROGRAM: M0 | REVERSE | SORT_ASC",
                allowed_tokens={*PRIMITIVES, "M0"},
                max_surface_calls=2,
            )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            harness.parse_program(
                "PROGRAM: M0\nThis works.", allowed_tokens={*PRIMITIVES, "M0"}
            )

    def test_batch_parsers_keep_malformed_samples(self) -> None:
        rows = [
            {
                "id": "t1",
                "n_prompt_tokens": 10,
                "outputs": [
                    completion_output(0, "PROGRAM: M0 | SORT_ASC"),
                    completion_output(1, "I used Python."),
                ],
            }
        ]
        parsed = harness.parse_program_outputs(
            rows, allowed_tokens={*PRIMITIVES, "M0"}, max_surface_calls=2
        )
        self.assertEqual(parsed[0].program, ("M0", "SORT_ASC"))
        self.assertIsNone(parsed[0].parse_error)
        self.assertIsNone(parsed[1].program)
        self.assertIn("invalid program line", parsed[1].parse_error or "")


class AccountingAndInvocationTests(unittest.TestCase):
    def test_exact_accounting_separates_sampled_injected_and_logical_prompt(self) -> None:
        rows = [
            {
                "id": "a",
                "n_prompt_tokens": 10,
                "outputs": [
                    completion_output(0, "PROGRAM: REVERSE", sampled=5),
                    completion_output(
                        1,
                        "PROGRAM: M0",
                        stage2_prompt=14,
                        sampled=9,
                        injected=2,
                        completion=10,
                        thinking=5,
                        answer=3,
                        trimmed=1,
                    ),
                ],
            },
            {
                "id": "b",
                "n_prompt_tokens": 7,
                "outputs": [
                    completion_output(
                        0,
                        "PROGRAM: SORT_ASC",
                        stage1_prompt=7,
                        sampled=6,
                        completion=5,
                    )
                ],
            },
        ]
        accounting = harness.extract_token_accounting(rows)
        self.assertEqual(accounting.requests, 2)
        self.assertEqual(accounting.completions, 3)
        self.assertEqual(accounting.unique_input_prompt_tokens, 17)
        self.assertEqual(accounting.stage1_logical_prompt_tokens, 27)
        self.assertEqual(accounting.stage2_logical_prompt_tokens, 14)
        self.assertEqual(accounting.logical_model_input_tokens, 41)
        self.assertEqual(accounting.sampled_tokens, 20)
        self.assertEqual(accounting.injected_tokens, 2)
        self.assertEqual(accounting.completion_tokens, 19)
        self.assertEqual(len(accounting.per_completion), 3)

        summary = summary_for(rows)
        self.assertEqual(harness.extract_token_accounting(rows, summary), accounting)
        summary["counts"]["sampled_tokens"] += 1
        with self.assertRaisesRegex(ValueError, "sampled_tokens"):
            harness.extract_token_accounting(rows, summary)

    def test_batch_path_uses_injected_vllm_runner_without_model_load(self) -> None:
        rows = [
            {
                "id": "dry",
                "n_prompt_tokens": 10,
                "outputs": [completion_output(0, "PROGRAM: REVERSE")],
            }
        ]
        summary = summary_for(rows)
        events: list[str] = []

        class FakeVLLMRunner:
            def __init__(self, _config: harness.EngineConfig):
                events.append("init")

            def generate(self, records, sampling):
                self.records = records
                self.sampling = sampling
                events.append("generate")
                return rows, summary

            def close(self):
                events.append("close")

        result = harness.run_vllm_batch(
            [{"id": "dry", "messages": [{"role": "user", "content": "dry"}]}],
            harness.SamplingConfig(thinking="off", greedy=True, max_tokens=8),
            runner_factory=FakeVLLMRunner,
        )
        self.assertEqual(events, ["init", "generate", "close"])
        self.assertEqual(result.accounting.sampled_tokens, 5)
        self.assertEqual(result.summary["model"], harness.REQUIRED_MODEL_ID)


if __name__ == "__main__":
    unittest.main()
