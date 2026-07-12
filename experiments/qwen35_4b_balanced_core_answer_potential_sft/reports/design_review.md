# Adversarial Design Review

## Verdict

`sound_with_scope_and_resource_fixes_applied`.

The balanced funnel can answer whether potential-selected full traces beat serious controls on three
families. It cannot complete or rescue the parent's nine-family/pivot claim. The following fixes are part of
the frozen design.

## Findings And Applied Fixes

1. **A post-calibration rewrite would be invalid.** The work is a new experiment, declares every observed
   input, and leaves the parent preregistration untouched.
2. **The family subset could be cherry-picked.** The core is defined by complete leading blocks in the
   parent's pre-existing train order. No family score or correctness selected it. Claims remain three-family.
3. **Inherited data could mutate silently.** Import requires the frozen source-index SHA-256, exact aggregate
   counts, and every shard receipt. The new index records provenance and never rewrites inherited shards.
4. **Finishing on another batch shape could change Ada samples.** Generation remains one task per logical
   call, N=64, identical sampling, engine geometry, and stable task seeds. Existing tasks are never replayed
   or compared token-pairwise across shapes.
5. **Answer potential already lost to length in calibration.** `shortest_natural` is now mandatory and may
   win the experiment. A potential result must beat the strongest of shortest, random, and success-RFT.
6. **Dropping the empty arm weakens answer-only attribution.** The user-approved six-arm budget replaces it
   with the more threatening shortest control. The conclusion is selector-relative, not a complete
   decomposition of answer-seam learning.
7. **Fast scoring could silently change the metric.** New parity covers the entire joint target, empty
   baselines, and gains, not only canonical answer likelihood. Bulk scoring is blocked before model work if
   any registered discrepancy exceeds 0.15 mean nats/token.
8. **Canonical-only scoring could exploit a formatting accident.** The already-observed format stability
   (tau-b 0.841) is disclosed as a design input. All train candidates use one uniform canonical boundary.
9. **Joint score can be dominated by boundary length.** It is normalized by canonical-answer token count as
   preregistered in the parent, reported alongside raw boundary/answer sums, and gets its own treatment rather
   than replacing answer gain.
10. **Shortest has fewer supervised tokens.** Rows and optimizer steps match, but token dose intentionally
    does not. Actual thought, supervised, and forward tokens are mandatory; a shortest win supports an
    efficiency/compression hypothesis, not evidence for answer potential.
11. **Success-RFT may have sparse task coverage.** Report unique tasks/rows before oversampling and the exact
    common-task comparison. Oversampling cannot be described as more independent evidence.
12. **Natural-close survivorship can select easier tasks.** Eligibility is identical across arms and any
    deficient task is excluded symmetrically. Family/level close rates and selected-task coverage are reported.
13. **Task-shuffle may mismatch length.** Use a one-to-one within-family/level derangement minimizing total
    length mismatch; preserve both target and source task IDs in SFT records.
14. **Six long-trace adapters can consume the saved budget.** Hard-stop at 360 tasks, omit branches, use the
    parity-gated vLLM scorer, and make Stage B conditional. No evaluation shortcut changes Stage-A arms.
15. **Staging can become selective reporting.** Stage A always evaluates every arm on all three declared
    subsets. The trigger, optional arms, and terminal negative are frozen before training.
16. **The positive bar may be mathematically impossible on a saturated split.** A machine-readable [0,1]
    reachability check precedes Stage B. An impossible gate is recorded as a stop, never repaired in place.
17. **Matched-compute sample-more could disappear after a training win.** It is mandatory in Stage B and
    required for the mission verdict, using actual prompt plus sampled tokens.
18. **Runtime LoRA can no-op on Qwen3.5.** Merge into the full composite, require base/base determinism, and
    reject every trained arm with zero same-prompt token differences.
19. **Held-family tasks might leak through copied imports.** Training imports only the local train-family
    modules. Held-family registries are loaded exclusively by evaluation scoring after datasets are frozen.
20. **A null could be overgeneralized.** The report must say core-scope negative and preserve the unfinished
    broader parent. Nine-family and pivot conclusions are prohibited.

## Remaining Limits

- One model and one procedural gym.
- Three training families selected by saved ordering, not random family sampling.
- Calibration influenced the control set and compute schedule.
- One mandatory training seed unless Stage B triggers.
- Canonical answers are oracle-side at curation time.
- The physical 16,384-token context still censors some long searches.

## Expensive-Run Authorization

Authorized only after the design commit is pushed, the implementation test suite passes, inherited receipts
validate, the 32-row joint parity gate passes, and the exact long-row training stress receipt remains valid on
the current environment.

## Post-Freeze Instrument Audit — 2026-07-12

The candidate vLLM scorer did not earn authorization: its task-diverse 32-row maximum was 0.692447 against
the frozen 0.15 ceiling. No bulk score existed. The scientifically conservative repair is not a softer gate
or friendlier row set; it is uniform use of the single-context Transformers reference that defined the other
side of the comparison. This removes batching from the measurement and preserves every scientific selector,
control, and decision threshold. Bulk scoring is authorized only under that amended backend.
