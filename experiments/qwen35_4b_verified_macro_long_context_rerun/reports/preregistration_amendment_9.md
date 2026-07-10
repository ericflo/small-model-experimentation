# Preregistration amendment 9: distinguish reasoning-cap contact from answer restart

Date: 2026-07-10. Frozen while complete base-smoke think@32,768 was still inside vLLM and before it
returned rows. This amendment follows an independent code-path audit; no generated output,
termination count, parser result, task score, or oracle result from that arm was available.

## Observation

In budget mode, stage 1 is allowed at most `thinking_budget` sampled tokens. A completion can emit
`</think>` naturally before that boundary and then begin its answer inside the same stage-1 call. If
the combined reasoning-plus-partial-answer reaches the stage-1 length boundary, the runner correctly
discards the partial answer, retains only the reasoning tokens before `</think>`, and samples a fresh
answer with the full independent `answer_max_tokens=512` allowance. Such a row has
`forced_close=false`, `stage1_finish_reason=length`, and usually
`n_thinking_tokens + 1 < thinking_budget`, where the extra token is the sampled `</think>`.

The current gate nevertheless treats every stage-1 length finish as a reasoning-cap contact. That
conflates an exhausted combined stage-1 call with exhausted reasoning. It can falsely reject a row
whose reasoning closed naturally and whose retained answer was regenerated under the registered
answer allowance. At the largest rung it could turn a runner bookkeeping artifact into a false
setup-inconclusive result.

## Frozen repair

1. In budget mode, define the historical **reasoning-cap contact** gate as either
   `forced_close=true` or `n_thinking_tokens + 1 >= thinking_budget`. The `+1` counts the sampled
   `</think>` token that is excluded from `n_thinking_tokens`; a close in the final stage-1 slot has
   demonstrated no boundary headroom. A stage-1 length finish is not by itself a reasoning-cap
   contact when `forced_close=false` and the close occurred before that final slot.
2. Retain and report three raw diagnostics separately: `stage1_finish_reason=length`,
   `forced_close=true`, and the subset with `stage1_finish_reason=length`, `forced_close=false`, and
   `n_thinking_tokens + 1 < thinking_budget`, called **answer restarts after natural reasoning
   close**. `forced_close=true` can also reflect an early terminal token without a sampled
   `</think>`, so name it a forced intervention rather than claiming every such row literally used
   every budget token. It remains an unresolved interface/termination contact because the runner
   had to supply the missing close.
3. Continue to apply the stage-independent answer-limit rule from amendment 7 to the fresh answer:
   its length finish or `n_answer_tokens >= 512` remains an answer-limit contact. Discarded partial
   stage-1 answer tokens never enter parsing, scoring, `n_completion_tokens`, or answer-headroom
   claims. They remain included in `n_sampled_tokens` and matched-compute accounting.
4. Run periodic-loop classification only for forced-intervention or boundary contacts. Compute
   natural-thinking headroom from every row that closed reasoning before the final stage-1 slot,
   including rows whose partial stage-1 answer was restarted.
5. Enforce identical definitions in runtime selection and offline analysis. Add parity tests for
   (a) forced close at the budget, (b) early terminal without a close, (c) close in the final
   stage-1 slot, (d) earlier natural close plus stage-1 length plus successful fresh answer, and
   (e) earlier natural close plus stage-1 length plus answer-limit contact.
6. Recompute termination metadata for all exact-valid cached rows. Generation does not change: the
   model, runner, prompts, seeds, sampled rows, and answer restart behavior remain frozen.
7. After the active pre-amendment process exits, rerun smoke selection under amendments 7 and 9
   before accepting any 32,768 matrix. The temporary 49k sentinel still prevents that old process
   from starting an unamended higher-rung call.

## Companion context-envelope audit

The same code-only audit tokenized every frozen prompt class and verified the largest-rung reserve:
`61,440` thinking tokens, two injected close tokens, and `512` answer tokens. Maximum
prompt-plus-reserve totals are 62,169 for calibration, 62,170 for the heldout interface, 62,944 for
base smoke, 63,014 for designed smoke, 63,017 for current full arms, and 65,432 for train-only macro
proposal, all within `max_model_len=65,536`. The proposal has only 104 tokens of spare context, so
preparation must freeze its current 3,478-token maximum and fail before model loading if that prompt
or the 104-token minimum headroom drifts. Cached preflight validation must recheck the recorded
generation reserve and prompt-plus-reserve arithmetic, not only prompt hashes.

Installed vLLM resolves the registered 65,536-token sequence limit with chunked prefill enabled.
`max_num_batched_tokens=32,768` is a scheduler-per-iteration allowance, not an effective sequence
cap. No hidden tokenizer truncation, uncounted BOS token, stop-token mismatch, or lower engine
ceiling was found.

## Interpretation boundary

This correction removes a false positive from the reasoning-cap gate; it cannot make a truncated
fresh answer adequate, alter decoded text, recover a parsed candidate, or select a rung from task
performance. A row that never emitted `</think>` remains a forced-intervention contact even when it
ended early, a close in the final slot remains a boundary contact, and a fresh answer that reaches
512 tokens still contacts the answer boundary. The smoke and full scientific thresholds are
otherwise unchanged.
