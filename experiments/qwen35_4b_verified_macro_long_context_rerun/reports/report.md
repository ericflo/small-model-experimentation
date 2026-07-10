# Qwen3.5-4B Verified Macro Invention Long-Context Rerun Report

## Summary

This follow-up has repaired the parent's apparent interface failure but has not yet produced an
eligible scientific comparison. Under the uv-managed vLLM stack, the disjoint plan-given interface
passed 16/16 records at think@16,384. The fresh induction base arm at that same allowance then hit a
different termination regime: 131/144 samples remained unresolved after exact-loop exclusions and
60/144 answer stages truncated. Those rows were rejected before parser or correctness inspection.
The scientific smoke is escalating on the frozen ladder; no capped row is a negative macro result.

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
designed macro ceiling at matched K=12. Amendments 1--6 record scheduler diagnostics, the Ada
batch-invariance boundary, exact periodic-loop handling, concurrency optimization, and the extended
scientific ladder `[16384, 32768, 49152, 61440]`. If either completed K=12 arm rejects the 32,768
matrix, the two higher rungs first use non-scored K=4 base workload probes to locate a viable
allowance. Lower rungs and probes can be rejected from termination metadata, but a rung is scored
only when both arms are complete at K=12 and adequate at the same budget.

## Results

### Corrected setup and interface

| Stage at think@16,384 | Samples | Unresolved cap contacts | Answer truncations | Gate result |
| --- | ---: | ---: | ---: | --- |
| train-only calibration | 64 | 3 (4.69%) | 0 | pass |
| disjoint plan-given interface | 64 | 0 | 0 | pass, 16/16 records covered |
| fresh base induction smoke | 144 | 131 (90.97%) | 60 (41.67%) | censored; excluded |

The interface produced 63/64 strict valid macro-using samples; every one of its 16 records had at
least one exact optimal rewrite. Its 12 raw cap contacts were all classified by the frozen
exact-token periodic-tail detector, so none remained unresolved.

The fresh base arm sampled 2,391,698 tokens in 2,138.606 seconds (1,118.34 tokens/s). All 144
samples reached 16,384; 13 tails passed the periodic-loop detector and 131 did not. Because base
alone irreversibly rejected the rung, the automatically started designed arm was interrupted before
it returned rows. No decoded output, parser status, visible score, hidden grade, or oracle result
informed the amendment or escalation.

### Scientific smoke and full comparison

Pending an uncensored complete rung. Full generation remains blocked by the unchanged smoke gate.

## Controls

- Byte-level frozen-data verification and zero forbidden train/evaluation overlap.
- Only one model, revision, backend, prompt family, and shared rung per comparison matrix.
- Metadata-only budget decisions; lower rungs never pooled or scored.
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

First finish the registered escalation. If smoke passes, run the full matrix with fixed resumable
shards and external raw-artifact manifests; if it fails semantically under a termination-adequate
rung, branch any slot-conditioned interface redesign into a separate experiment. Do not use the
censored 16k rows to tune prompts or claim a macro effect.

## Artifact Manifest

Raw superseded scheduler/context diagnostics are preserved under the standard external artifact
root with a deterministic tree checksum. Current compact audits, config, preregistration amendments,
metadata, and reasonably sized calibration/interface rows are listed in
[`artifact_manifest.yaml`](artifact_manifest.yaml). Raw scientific smoke and full rows will be
externalized with tracked per-file hashes before landing if they exceed repository precedent.
