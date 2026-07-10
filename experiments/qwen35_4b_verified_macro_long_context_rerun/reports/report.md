# Qwen3.5-4B Verified Macro Invention Long-Context Rerun Report

## Summary

This follow-up has repaired the parent's apparent interface failure but has not yet produced an
eligible scientific comparison. Under the uv-managed vLLM stack, the disjoint plan-given interface
passed 16/16 records at think@16,384. The fresh induction base arm at that same allowance then hit a
different termination regime: 131/144 samples remained unresolved after exact-loop exclusions and
60/144 answer stages truncated. Those rows were rejected before parser or correctness inspection.
At think@32,768, all 144 samples still required forced closure: 81 exact loops, 63 unresolved
contacts, and 37 answer-limit contacts. That rung was also excluded. The max-seqs-64 K=4 probe at
think@49,152 finished with 34/48 loops, 14/48 unresolved contacts, and 13/48 answer-limit contacts,
but pre-result KV-capacity evidence had already made it diagnostic-only. Fresh selection is running
through separate scheduler follow-ups. The capacity-fit 61k attempt stopped before a receipt when
its implicit CUDA-graph list proved not to cover the active width. Fresh exact-capture 49k and 61k
probes then failed termination despite passing both live-KV and exact-graph gates. The terminal
selector is `pass=false` with no selected budget, so no K=12 matrix or semantic analysis was
authorized. No capped row is a negative macro result.

## Research Program Fit

Primary program: `operator_and_skill_inventories`. The experiment also supplies direct evidence to
`test_time_reasoning_budget`: a budget calibrated on supplied-plan transcription did not transfer
to fresh program induction in the same procedural substrate. Structured execution and benchmark
generalization become relevant only after an uncensored mined/base/hint/random comparison exists.

## Method

The only model is `Qwen/Qwen3.5-4B` at revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Every model-facing request uses the experiment-local
single-file vLLM runner; Transformers inference and backend mixing are forbidden. Inputs are copied
byte-for-byte from parent commit `1c8c5bbb81d2a67618891597205ceb2f40f498d8`: 800 train-only
construction programs, 12 fresh v2 smoke tasks, and 120 never-prompted full tasks with exact
execution and behaviorally verified depth five.

The original calibration and independent interface gate used supplied train-only plans. Scientific
smoke uses fresh I/O-only induction tasks and compares base primitives with the generator-known
designed macro ceiling at matched K=12. Amendments 1--12 record scheduler diagnostics, the Ada
batch-invariance boundary, exact periodic-loop handling, concurrency optimization, the extended
scientific ladder `[16384, 32768, 49152, 61440]`, exact cap accounting, external scientific/full
durability, and fail-early full-run audit hardening. If either completed K=12 arm rejects the 32,768
matrix, the two higher rungs first use non-scored K=4 base workload probes to locate a viable
allowance. Lower rungs and probes can be rejected from content-blind finish/count metadata plus the
preregistered token-ID periodicity test, but a rung is scored
only when both arms are complete at K=12 and adequate at the same budget. Amendment 12 additionally
freezes max-seqs 19 at 49k and 15 at 61k in a separate protocol because the measured cache cannot
hold the former max-seqs-64 scheduling wave without recomputation.

## Results

### Corrected setup and interface

| Stage / allowance | Samples | Unresolved cap contacts | Answer-limit contacts | Gate result |
| --- | ---: | ---: | ---: | --- |
| train-only calibration / 16,384 | 64 | 3 (4.69%) | 0 | pass |
| disjoint plan-given interface / 16,384 | 64 | 0 | 0 | pass, 16/16 records covered |
| fresh base induction smoke / 16,384 | 144 | 131 (90.97%) | 60 (41.67%) | censored; excluded |
| fresh base induction smoke / 32,768 | 144 | 63 (43.75%) | 37 (25.69%) | censored; excluded |
| overcommitted base K=4 probe / 49,152 | 48 | 14 (29.17%) | 13 (27.08%) | diagnostic-only; excluded |

The interface produced 63/64 strict valid macro-using samples; every one of its 16 records had at
least one exact optimal rewrite. Its 12 raw cap contacts were all classified by the frozen
exact-token periodic-tail detector, so none remained unresolved.

The fresh base arm sampled 2,391,698 tokens in 2,138.606 seconds (1,118.34 tokens/s). All 144
samples reached 16,384; 13 tails passed the periodic-loop detector and 131 did not. Because base
alone irreversibly rejected the rung, the automatically started designed arm was interrupted before
it returned rows. No decoded output, parser status, visible score, hidden grade, or oracle result
informed the amendment or escalation.

At 32,768, the amendment-9 replay found no earlier-close answer restarts: every sample reached the
reasoning boundary and required intervention. Eighty-one tails were exact periodic loops, while 63
remained unresolved; 37 fresh answers hit 512 tokens. The arm sampled 4,739,527 tokens in 5,971.182
seconds (793.73 tokens/s). Base alone again rejected the rung without a designed-arm call, so the
precommitted branch advanced to the non-scored K=4 base probe at 49,152. This decision also used no
decoded output, parser result, visible score, hidden grade, or oracle result.

Before that probe returned, amendment 12 proved from engine capacity and installed scheduler source
that max-seqs 64 could not hold its long active contexts without eviction/recomputation. The probe
therefore became irrevocably decision-ineligible. It later confirmed 48/48 forced interventions,
34 exact periodic loops, 14 unresolved contacts, and 13 answers at 512 tokens. Sampling 2,366,620
tokens took 4,035.356 seconds (586.47 tokens/s). The receipt-bound diagnostic was classified from
finish/count metadata and token IDs only; no decoded or scored content was inspected. Fresh budget
selection moved to `qwen35_4b_verified_macro_capacity_fit_rerun` at max-seqs 19/15.

That capacity-fit follow-up completed its fresh 49k K=4 probe within live KV capacity, but all 48
samples still contacted the reasoning boundary: 37 were exact loops, 11 remained unresolved, and
9 answers reached their limit. It generated 2,364,643 sampled tokens in 5,012.451 seconds
(471.754 tokens/s). A source/runtime audit then found that the implicit CUDA-graph configuration
resolved only through widths 16 and 8 for requested maxima 19 and 15. The 61k capacity-fit attempt
was therefore stopped before a receipt; it produced no reusable rows and no decoded or scored
content was inspected.

The separate `qwen35_4b_verified_macro_exact_cudagraph_rerun` froze explicit graph lists and fresh
artifacts. At 49k, vLLM resolved `[1, 2, 4, 8, 16, 19]` exactly and the live audit fit 963,072
required cache tokens into 996,864, leaving 33,792. The fresh probe nevertheless failed all three
termination thresholds: 38/48 exact loops, 10/48 unresolved contacts, and 6/48 answer-limit
contacts. It generated 2,363,163 sampled tokens in 4,809.081 seconds (491.396 tokens/s), a
descriptive 4.16% improvement over the closest implicit-capture probe. This is not a causal
throughput benchmark because scheduling can alter sampled trajectories.

The terminal 61k probe also passed its runtime envelope: vLLM resolved FULL decode graphs at
`[1, 2, 4, 8, 15]`, covering max-seqs 15 exactly, and fit 950,400 required cache tokens into
997,888 with 47,488 of headroom. All 48 samples still reached the reasoning boundary: 40 were exact
loops, 8 remained unresolved, and 4 answers reached 512 tokens. It generated 2,951,995 sampled
tokens in 7,422.886 seconds (397.688 tokens/s). The terminal selection therefore records
`pass=false` and `selected_thinking_budget=null`; no K=12 arm, semantic analysis, or macro result is
authorized. Neither probe decision used decoded or scored content.

### Scientific smoke and full comparison

Pending an uncensored complete rung. Full generation remains blocked by the unchanged smoke gate.

## Controls

- Byte-level frozen-data verification and zero forbidden train/evaluation overlap.
- Only one model, revision, backend, prompt family, and shared rung per comparison matrix.
- Content-blind finish/count and preregistered token-ID periodicity decisions; no decoded semantics
  or scores enter budget selection, and lower rungs are never pooled or scored.
- Exact periodic-tail classification reported separately from unresolved cap contact.
- Stage-independent 512-token answer-limit contact; natural answers cannot bypass the stage-2 cap.
- Amendment-9 reasoning-cap accounting separates forced intervention and final-slot close from an
  earlier natural close whose partial answer was merely restarted after the raw stage-1 length cap.
- A model-free proposal-record/hash guard freezes the 3,478-token proposal bound, 65,432-token
  largest-rung total, and at least 104 tokens of context headroom; exact runtime preflight remains
  mandatory.
- Strict one-line parser, visible-only selection, and hidden-only final grading remain unchanged.
- Base/design smoke precedes all mined, hint, random, Qwen-ranked, and full-task generation.

## Oracle Versus Deployable Evidence

Not yet eligible. Oracle and selected correctness are deliberately unavailable until the complete
smoke matrix passes termination. If it does, full results will report hidden oracle coverage
separately from the visible-only selected line and from the matched-token sample-more base prefix.

## Interpretation

The parent's 768-token result was not a durable alias-interface limit: adequate reasoning cleared
the broader K=4 record-level gate. The new roadblock is workload-conditioned provisioning.
Supplied-plan transcription and I/O-only induction share the same syntax and model but have radically
different trace-length distributions. The operational lesson is to calibrate on the actual workload
class, retain exact termination evidence, and treat a generation ceiling as missing evidence rather
than task failure.

This still says nothing about whether mined macros improve induction. That claim requires the
unchanged full conjunction against base sampling, non-callable hints, matched random libraries,
macro mediation, and the no-reuse control.

## Next Experiments

Stop increasing the context allowance: the exact-capture ladder has reached its registered terminal
rung without selecting a budget. The next attempt should be a separate, preregistered symmetric
loop-control experiment that applies the same intervention to every compared arm while preserving
the unresolved-boundary and answer-limit gates. Do not reuse censored rows, decode them to tune the
intervention, or claim a macro effect without a fresh termination-adequate K=12 matrix.

## Artifact Manifest

Raw superseded scheduler/context diagnostics are preserved under the standard external artifact
root with a deterministic tree checksum. Current compact audits, config, preregistration amendments,
metadata, and reasonably sized calibration/interface rows are listed in
[`artifact_manifest.yaml`](artifact_manifest.yaml). Raw scientific smoke and full rows live only
in their canonical external roots and are bound by tracked deterministic catalogs and per-file
receipt hashes; repository-local promotion copies are forbidden.
