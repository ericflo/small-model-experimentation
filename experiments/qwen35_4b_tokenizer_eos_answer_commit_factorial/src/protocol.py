"""Model-free protocol primitives for the tokenizer-EOS answer commit study."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence


MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
HF_MODEL_EOS_ID = 248044
TOKENIZER_EOS_ID = 248046
NEWLINE_ID = 198


@dataclasses.dataclass(frozen=True)
class CommitResult:
    policy: str
    stop_token_id: int
    valid_trace: bool
    content_token_ids: tuple[int, ...]
    strict_exact: bool
    failure: str | None


def evaluate_answer_commit(
    sampled_token_ids: Sequence[int],
    *,
    stop_token_id: int,
    expected_token_ids: Sequence[int],
    policy: str,
) -> CommitResult:
    """Authenticate first-stop geometry and parse every pre-commit token strictly.

    vLLM includes an explicit stop token in sampled token IDs. A real trace may
    therefore contain the registered stop token exactly once and only at the
    final sampled position. The stop token is boundary metadata, not answer
    content; every preceding token must equal the registered expected answer.
    """

    sampled = tuple(int(token_id) for token_id in sampled_token_ids)
    expected = tuple(int(token_id) for token_id in expected_token_ids)
    positions = tuple(
        index for index, token_id in enumerate(sampled) if token_id == stop_token_id
    )
    if not positions:
        return CommitResult(
            policy=policy,
            stop_token_id=stop_token_id,
            valid_trace=False,
            content_token_ids=sampled,
            strict_exact=False,
            failure="missing_registered_stop",
        )
    if positions != (len(sampled) - 1,):
        return CommitResult(
            policy=policy,
            stop_token_id=stop_token_id,
            valid_trace=False,
            content_token_ids=sampled[: positions[0]],
            strict_exact=False,
            failure="tokens_after_first_registered_stop",
        )
    content = sampled[:-1]
    return CommitResult(
        policy=policy,
        stop_token_id=stop_token_id,
        valid_trace=True,
        content_token_ids=content,
        strict_exact=content == expected,
        failure=None if content == expected else "precommit_content_mismatch",
    )


def smoke_cases() -> dict[str, CommitResult]:
    """Return frozen synthetic cases spanning the answer-boundary contract."""

    expected = (78041, 25, 357, 735, 426)  # PROGRAM: A | F
    return {
        "tokenizer_clean": evaluate_answer_commit(
            (*expected, TOKENIZER_EOS_ID),
            stop_token_id=TOKENIZER_EOS_ID,
            expected_token_ids=expected,
            policy="tokenizer_eos_answer_stage",
        ),
        "hf_boundary_control": evaluate_answer_commit(
            (*expected, TOKENIZER_EOS_ID, NEWLINE_ID, HF_MODEL_EOS_ID),
            stop_token_id=HF_MODEL_EOS_ID,
            expected_token_ids=expected,
            policy="hf_model_eos_answer_stage",
        ),
        "early_tokenizer_stop": evaluate_answer_commit(
            (78041, 25, TOKENIZER_EOS_ID),
            stop_token_id=TOKENIZER_EOS_ID,
            expected_token_ids=expected,
            policy="tokenizer_eos_answer_stage",
        ),
        "interior_and_terminal_stop": evaluate_answer_commit(
            (*expected, TOKENIZER_EOS_ID, 999, TOKENIZER_EOS_ID),
            stop_token_id=TOKENIZER_EOS_ID,
            expected_token_ids=expected,
            policy="tokenizer_eos_answer_stage",
        ),
        "missing_stop": evaluate_answer_commit(
            expected,
            stop_token_id=TOKENIZER_EOS_ID,
            expected_token_ids=expected,
            policy="tokenizer_eos_answer_stage",
        ),
        "extra_precommit_byte": evaluate_answer_commit(
            (*expected, NEWLINE_ID, TOKENIZER_EOS_ID),
            stop_token_id=TOKENIZER_EOS_ID,
            expected_token_ids=expected,
            policy="tokenizer_eos_answer_stage",
        ),
    }


def validate_smoke_cases(cases: dict[str, CommitResult]) -> None:
    """Fail unless every positive/control/malformed case has frozen behavior."""

    expected = {
        "tokenizer_clean": (True, True, None),
        "hf_boundary_control": (True, False, "precommit_content_mismatch"),
        "early_tokenizer_stop": (True, False, "precommit_content_mismatch"),
        "interior_and_terminal_stop": (
            False,
            False,
            "tokens_after_first_registered_stop",
        ),
        "missing_stop": (False, False, "missing_registered_stop"),
        "extra_precommit_byte": (True, False, "precommit_content_mismatch"),
    }
    observed = {
        label: (result.valid_trace, result.strict_exact, result.failure)
        for label, result in cases.items()
    }
    if observed != expected:
        raise RuntimeError(
            f"answer-commit smoke contract changed: expected={expected}, observed={observed}"
        )
