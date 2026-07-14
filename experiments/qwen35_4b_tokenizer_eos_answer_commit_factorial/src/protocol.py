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


@dataclasses.dataclass(frozen=True)
class BoundaryPairResult:
    valid_pair: bool
    compared_tokens: int
    tokenizer_event: str
    hf_event: str
    failure: str | None


def _authenticate_terminal_event(
    sampled_token_ids: Sequence[int],
    *,
    registered_stop_token_id: int,
    event: str,
    cap: int,
) -> str | None:
    sampled = tuple(int(token_id) for token_id in sampled_token_ids)
    if event == "stop":
        positions = tuple(
            index
            for index, token_id in enumerate(sampled)
            if token_id == registered_stop_token_id
        )
        if positions != (len(sampled) - 1,):
            return "malformed_registered_stop"
        if len(sampled) > cap:
            return "answer_cap_overflow"
        return None
    if event == "length":
        if len(sampled) != cap:
            return "short_output_relabeled_length"
        if registered_stop_token_id in sampled:
            return "registered_stop_ignored_before_cap"
        return None
    return "unknown_terminal_event"


def is_answer_cap_contact(
    sampled_token_ids: Sequence[int], *, finish_reason: str, cap: int = 24
) -> bool:
    """Apply the frozen cap metric, including a stop emitted on token `cap`."""

    return len(tuple(sampled_token_ids)) >= cap or finish_reason == "length"


def authenticate_boundary_pair(
    tokenizer_sampled_token_ids: Sequence[int],
    hf_sampled_token_ids: Sequence[int],
    *,
    tokenizer_event: str,
    hf_event: str,
    cap: int = 24,
) -> BoundaryPairResult:
    """Fail closed unless a paired trace agrees through its earliest event.

    Metadata identity (prompt, seed, thought source, adjacency, and engine) is
    authenticated by the transaction layer. This primitive authenticates the
    token/event geometry for every pair, including pairs ending at an HF stop
    or a shared length cap before tokenizer EOS is observed.
    """

    tokenizer_ids = tuple(int(token_id) for token_id in tokenizer_sampled_token_ids)
    hf_ids = tuple(int(token_id) for token_id in hf_sampled_token_ids)
    tokenizer_failure = _authenticate_terminal_event(
        tokenizer_ids,
        registered_stop_token_id=TOKENIZER_EOS_ID,
        event=tokenizer_event,
        cap=cap,
    )
    if tokenizer_failure is not None:
        return BoundaryPairResult(
            valid_pair=False,
            compared_tokens=0,
            tokenizer_event=tokenizer_event,
            hf_event=hf_event,
            failure=f"tokenizer_{tokenizer_failure}",
        )
    hf_failure = _authenticate_terminal_event(
        hf_ids,
        registered_stop_token_id=HF_MODEL_EOS_ID,
        event=hf_event,
        cap=cap,
    )
    if hf_failure is not None:
        return BoundaryPairResult(
            valid_pair=False,
            compared_tokens=0,
            tokenizer_event=tokenizer_event,
            hf_event=hf_event,
            failure=f"hf_{hf_failure}",
        )
    compared = min(len(tokenizer_ids), len(hf_ids))
    if tokenizer_ids[:compared] != hf_ids[:compared]:
        return BoundaryPairResult(
            valid_pair=False,
            compared_tokens=compared,
            tokenizer_event=tokenizer_event,
            hf_event=hf_event,
            failure="sampled_prefix_divergence",
        )
    return BoundaryPairResult(
        valid_pair=True,
        compared_tokens=compared,
        tokenizer_event=tokenizer_event,
        hf_event=hf_event,
        failure=None,
    )


def boundary_pair_smoke_cases() -> dict[str, BoundaryPairResult]:
    """Return frozen synthetic cases for the fail-closed all-pair gate."""

    return {
        "tokenizer_stop_first": authenticate_boundary_pair(
            (10, 11, TOKENIZER_EOS_ID),
            (10, 11, TOKENIZER_EOS_ID, NEWLINE_ID, HF_MODEL_EOS_ID),
            tokenizer_event="stop",
            hf_event="stop",
        ),
        "hf_stop_first": authenticate_boundary_pair(
            (10, HF_MODEL_EOS_ID, 11, TOKENIZER_EOS_ID),
            (10, HF_MODEL_EOS_ID),
            tokenizer_event="stop",
            hf_event="stop",
        ),
        "shared_cap": authenticate_boundary_pair(
            (10, 11, 12),
            (10, 11, 12),
            tokenizer_event="length",
            hf_event="length",
            cap=3,
        ),
        "prefix_divergence": authenticate_boundary_pair(
            (10, 11, TOKENIZER_EOS_ID),
            (10, 12, HF_MODEL_EOS_ID),
            tokenizer_event="stop",
            hf_event="stop",
        ),
        "short_length_claim": authenticate_boundary_pair(
            (10, 11),
            (10, 11),
            tokenizer_event="length",
            hf_event="length",
            cap=3,
        ),
    }


def validate_boundary_pair_smoke_cases(
    cases: dict[str, BoundaryPairResult],
) -> None:
    expected = {
        "tokenizer_stop_first": (True, None),
        "hf_stop_first": (True, None),
        "shared_cap": (True, None),
        "prefix_divergence": (False, "sampled_prefix_divergence"),
        "short_length_claim": (
            False,
            "tokenizer_short_output_relabeled_length",
        ),
    }
    observed = {
        label: (result.valid_pair, result.failure) for label, result in cases.items()
    }
    if observed != expected:
        raise RuntimeError(
            f"boundary-pair smoke contract changed: expected={expected}, "
            f"observed={observed}"
        )


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
