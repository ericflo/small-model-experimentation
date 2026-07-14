# Adversarial Design Review

## Verdict

Proceed only after the immutable design receipt, full CPU smoke, pinned-model
preflight, source/soup hash checks, and repeated behavioral canary. Fresh deep
qualification and exact-logit locality are necessary gates; neither can be
waived because deep passed the predecessor.

## Findings And Resolutions

1. **This direction was chosen after seeing deep pass.** That is legitimate
   follow-up selection but not confirmation. Resolution: two entirely new
   qualification blocks repeat the unchanged strict three-policy rule and
   audit contrasts before any target logits or training.
2. **Calling a one-teacher update “composition” could be semantic laundering.**
   Resolution: the initial student is already the joint soup, and the final
   checkpoint must improve both quick and deep strata beyond the better source
   in each block. Preserving quick is an outcome requirement, not an assumption.
3. **Dropping quick from the route comparison would make deep easier to select.**
   Resolution: quick remains the alternate policy at every qualification and
   online state. Deep must strictly beat both quick and current student.
4. **Four-branch argmax produced severe quick winner's curse.** Resolution:
   selection and audit remain disjoint; deep must again pass both block signs
   and pooled bounds. No pooled-only or observed-margin repair is allowed.
5. **Doubling deep units relative to the unrun two-teacher recipe changes dose.**
   Resolution: total updates and 75/25 capability-anchor mass remain fixed;
   five exact updates gate locality, all controls receive the same dose, and
   every round records initial/final loss, overlap, entropy, gradients, and
   token ledgers.
6. **An unconditioned-deep control could receive easier states.** Resolution:
   non-deep-selected controls come from the identical candidate pools and are
   matched one-to-one without replacement, preserving kind and preferring
   exact cell. Match tiers are frozen in the manifest and reported.
7. **The control matcher could silently cross from atoms to episodes.**
   Resolution: `kind` is the loosest legal tier; failure to find 60 matches
   requests another candidate batch or stops at the frozen maximum.
8. **Caching control logits could leak controls into the primary.** Resolution:
   primary and control state IDs are disjoint, roles are explicit, each arm
   selects exactly 60 capability plus the shared 20 anchors, and tests bind the
   inventory before GPU code.
9. **Wrong-teacher quick could differ in raw loss pressure.** Resolution:
   control backward loss is scaled to the primary round's initial probe loss;
   raw loss and scale remain separately visible.
10. **The fixed soup could be reconstructed differently or corrupted.**
    Resolution: reuse the exact predecessor checkpoint by immutable file hash,
    validate its source adapter hashes and explicit merge receipt, and repeat
    the four-arm behavior canary. New outputs use a separate artifact root.
11. **Failure-only training can improve a conditional metric while hurting the
    distribution.** Resolution: qualification is conditional, but final
    confirmation is unconditional over all families/levels, with source,
    transfer, retention, router, and sample-more comparisons.
12. **The soup anchor may merely pull the model back to its start.** Resolution:
    that is its declared role. No-update soup, parameter soups, target overlap,
    and the final soup comparison expose a no-op or interpolation result.
13. **Off-policy SFT may use longer or easier targets.** Resolution: it uses the
    best of the same four deep selection branches, identical state identities,
    the same active-position cap and update count, and pressure matching; its
    generation-token ledger remains separate.
14. **Controls trained only at one seed can be weak.** Resolution: seed 42 is
    primary and all controls are fixed before output; seeds 43/44 replicate the
    proposed method. Controls are mechanism falsifiers, not variance estimates.
15. **A visible router or verifier could be smuggled into deployment.**
    Resolution: the primary artifact is one explicitly merged checkpoint. The
    verifier is training-only; the visible two-checkpoint router is a baseline.
16. **Best-of-8 is compute-favored.** Resolution: intentionally so. A learned
    capability gain must shift the single-checkpoint proposal distribution,
    not merely beat a weak greedy source.
17. **Backend mixing could create false deltas.** Resolution: every procedural
    arm uses the identical pinned vLLM runner and engine geometry; Transformers
    is restricted to target caching, training, and exact-logit locality.
18. **Benchmark success could contaminate later training.** Resolution:
    benchmark access is last, run-only, aggregate-only, and machine-gated by
    procedural confirmation. No benchmark content can enter this experiment.

## Required Pre-Output Checks

- all 14 copied gym families pass oracle/random/degenerate selftests;
- epsilon deep advantage passes while zero, quick-only, and negative-block
  cases fail;
- the assembler produces exactly 60 deep, 20 anchors, and 60 disjoint
  kind-preserving non-advantage controls;
- every MOPD arm consumes exactly its registered 80 units and locality uses
  15 deep/5 soup;
- corrected sparse top-k value and gradient match full reverse KL;
- episode replay reproduces visible observations and action flags exactly;
- all seeds, hashes, engine geometry, source identities, and absolute gates are
  reachable and frozen;
- frozen design files are committed and hash-bound before any model loads.
