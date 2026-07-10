# Preregistration amendment 7: stage-independent answer-limit contact

Date: 2026-07-10. Frozen while complete base-smoke think@32,768 was still inside vLLM and before it
returned rows. This amendment follows a runner-code audit, not generated output inspection.

## Observation

The two-stage budget runner enforces `answer_max_tokens=512` directly when reasoning reaches its
thinking limit and the runner injects a close before stage 2. When stage 1 naturally emits
`</think>` and later the terminal token, however, `_ordinary_output` accepts the combined completion
under the much larger thinking-stage `max_tokens`. Such a naturally closed answer could contain 512
or more semantic answer tokens while retaining `finish_reason=stop` and `truncated=false`.

This is an accounting/enforcement gap in the anti-censoring gate. It does not affect the rejected
base@16,384 diagnosis: all 144 samples were force-closed, and its 60 answer truncations are literal
stage-2 length finishes. Completed calibration/interface answers are far below 512. The current
32,768 result is not yet available.

## Frozen repair

1. Define **answer-limit contact** stage-independently as either an existing answer-length finish or
   `n_answer_tokens >= answer_max_tokens`.
2. Use answer-limit contact everywhere the protocol previously used answer truncation: calibration,
   interface, workload probes, scientific rung selection, smoke/full adequacy, analyzer candidates,
   summaries, and verdict gates. Keep the historical `answer_truncation` field names for artifact
   compatibility, but document that they now include natural-stage limit contacts.
3. Recompute this metadata classification on every existing row. Generation need not be repeated:
   the model, prompts, sampled tokens, parser, and task metrics do not change, and any row exposed to
   more than the nominal answer allowance can only reject its rung—it can never enter an accepted
   matrix.
4. Count equality as contact. A 512-token semantic answer has not demonstrated termination within a
   512-token sampled allowance and therefore cannot be called nonbinding.
5. After the active pre-amendment process exits, rerun smoke selection and analysis from exact-valid
   cached rows under the amended classifier before accepting any 32,768 matrix. The temporary 49k
   sentinel still prevents the old process from starting a higher-rung K=12 call.

## Interpretation boundary

This repair makes the documented answer allowance conservative and stage-independent. It cannot
improve a task score, recover a candidate, change sample order, or select a favorable rung from
correctness. If natural answers repeatedly reach the limit even at a larger reasoning allowance,
the setup remains inconclusive rather than becoming a macro failure.
