# Preregistration amendment 6: higher-rung workload probes

Date: 2026-07-10. Frozen while the complete base-smoke think@32,768 arm was still inside vLLM and
before it returned rows. This branch therefore does not use its termination result, decoded output,
parser status, or correctness.

## Motivation

The complete base@16,384 arm already established that the plan-given calibration did not transfer
to fresh induction: 131/144 contacts remained unresolved and 60/144 answers truncated. A complete
K=12 base arm at 32,768 is now running under amendment 5. If that arm also rejects its rung, blindly
running another 144-sample base arm at both 49,152 and 61,440 would spend several GPU-hours merely
to locate a nonbinding ceiling.

Budget location is a termination problem, not a correctness comparison. It can therefore use a
smaller non-scored sample without weakening the complete matrix that is eventually scored.

## Conditional branch, frozen before the trigger is known

1. If complete base@32,768 is termination-adequate, make no change: continue the registered
   designed@32,768 arm and score only if the complete matrix is adequate.
2. If complete base@32,768 is termination-inadequate, do not begin a K=12 arm at the next rung.
   For each remaining rung in ascending order, run a **workload budget probe** over all 12 frozen
   base-smoke prompts at K=4 (48 completions). The prompt, model, revision, vLLM runner, temperature,
   top-p, top-k, answer allowance, record seed derivation, and task order remain unchanged; only
   `n=4` and the registered thinking rung identify the probe protocol.
3. A probe is adequate only when unresolved cap contact and answer truncation are each below 5%
   (at most 2/48) and exact periodic-loop rate is at most 25% (at most 12/48), using the already
   frozen token-only classifier. Selection reads no decoded text, parser result, visible score,
   hidden grade, or oracle result.
4. At the first adequate probe rung, run the complete base/designed K=12 smoke matrix from scratch.
   Probe rows are never promoted, pooled, substituted, or scored. On Ada, the K=4 probe and K=12
   matrix are explicitly different batch protocols and are not token-paired.
5. If a K=12 arm rejects that rung, continue with a K=4 probe at the next registered rung before
   another complete matrix. If the 61,440 probe or complete matrix is inadequate, report the current
   65,536-context setup as inconclusive. Any later extension requires a larger engine context and a
   new amendment; the 104-token proposal guard band is unavailable as another rung.

## Interpretation boundary

The probe saves budget-search compute only. It cannot pass the semantic smoke gate, estimate a
macro effect, contribute candidates, or reduce K in any scored comparison. The scientific unit,
fresh task set, complete K=12 base/designed matrix, hidden-label boundary, and all semantic gates
remain unchanged.
