# Preregistration amendment 5: smoke-budget escalation and failed-rung short circuit

Date: 2026-07-10. Frozen after the complete think@16,384 base-smoke arm and interruption of
think@16,384 designed-ceiling before it returned any row, and before the next GPU call.

## Observation

The complete base arm contains 144 samples. All 144 contacted the thinking cap. Thirteen contacts
pass the frozen exact-periodic-loop detector, leaving 131/144 unresolved cap contacts (90.97%). In
addition, 60/144 answers truncated at the unchanged 512-token answer allowance (41.67%). The vLLM
artifact records 2,391,698 sampled tokens in 2,138.606 seconds, or 1,118.34 sampled tokens/second.

These are termination and serving metadata only. No generated text, parser output, candidate
correctness, oracle result, or task-level score was inspected. The automatically started
designed-ceiling arm was interrupted before it returned any row. Its preflight is not a scientific
result and no partial row can be carried forward.

The completed base arm alone makes the 16,384-token rung ineligible: its unresolved-cap and
answer-truncation counts cannot satisfy the registered censoring gates. This is an inference-envelope
failure, not evidence for or against verified macro invention.

## Frozen escalation

1. Extend the registered thinking ladder to `[16384, 32768, 49152, 61440]`.
2. Keep `answer_max_tokens=512` and `max_model_len=65536` unchanged, along with the model,
   revision, vLLM backend, prompts, sampling parameters, seeds, K values, loop detector, parser,
   and scientific decision rules.
3. Permit a completed-arm failed-rung short circuit. If one completed arm's censoring counts already
   make the complete matrix unable to pass its registered termination gates, stop the remaining
   arm before rows are returned and advance to the next ladder rung. This branch uses termination
   counts only; generated content and task metrics remain uninspected.
4. Treat every row from a failed lower rung as diagnostic only. Lower-rung rows are neither pooled
   with, substituted into, nor scored alongside a later rung.
5. At each candidate rung, run base first. If it is adequate, run designed ceiling; if either
   completed arm is inadequate, reject that rung and advance. A rung is selectable only after the
   complete base/designed matrix passes the frozen termination gates, and that complete matrix is
   the sole matrix eligible for smoke scoring. Exhausting 61,440 remains setup-inconclusive rather
   than a scientific negative.

The largest rung fits the existing engine context without changing the answer allowance. The
observed maximum rendered smoke prompt is 990 tokens and the injected close sequence is two tokens,
so `990 + 61440 + 2 + 512 = 62944`, leaving 2,592 tokens below `max_model_len=65536`.

**Arithmetic erratum, recorded before completion of the 32,768 rung:** 990 is the maximum *base*
prompt. The already frozen designed-arm preflight has a 1,060-token maximum, so the matrix-wide
bound is `1060 + 61440 + 2 + 512 = 63014`, leaving 2,522 tokens. The registered largest rung still
fits; no inference setting or branch changes.

The frozen train-only macro-proposal prompt is longer than either smoke arm at 3,478 tokens. Its
largest-rung preflight is `3478 + 61440 + 2 + 512 = 65432`, 104 tokens below the engine limit. It
therefore also fits the registered ladder. Any future rung above 61,440 would require a larger
`max_model_len`; it may not consume that remaining guard band.

## Interpretation boundary

Amendment 4's train-only calibration and interface evidence did not predict the fresh base-smoke
termination distribution. The appropriate correction is therefore to enlarge the frozen inference
envelope and rerun whole matrices, not to call the hypothesis negative, lower K, inspect outputs to
tune prompts, or pool censored rows. Scientific interpretation begins only after a complete rung
passes the registered termination gates.
