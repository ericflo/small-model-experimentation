# Adversarial Design Review

## Verdict

Proceed only after the immutable design receipt, CPU selftests, exact-engine
preflight, soup installation canary, and state-replay smoke. The route gate is
the expensive experiment's most important result; training is conditional.

## Findings and resolutions

1. **Selecting the largest noisy branch mean creates winner's curse.** A
   teacher can look best by chance even when it has no value. Resolution: four
   samples select, four disjoint samples estimate both advantages. Only audit
   outcomes enter inference.
2. **Teacher-versus-teacher is insufficient.** The better endpoint can still
   be worse than the already stronger soup student. Resolution: a route must
   beat both the alternate teacher and current student; otherwise it abstains.
3. **A globally positive router can hide one useless teacher.** That would be
   ordinary single-teacher distillation, not capability composition.
   Resolution: support, block sign, and pooled lower bound must pass separately
   for quick and deep on both audit contrasts.
4. **An arbitrary positive margin would repeat the predecessor's error.** A
   bounded score has no universal practical increment. Resolution: strict
   `delta > 0`, replication, and uncertainty; minimum counts govern
   identifiability only.
5. **Failure-only state acquisition can inflate a conditional claim.** The
   route estimand is intentionally conditional on residual states, but final
   capability is not. Resolution: state selection is declared explicitly and
   all final decisions cover the unconditional full procedural distribution.
6. **Mid-thought atom states may be an artificial seam.** Resolution: prefixes
   are exact autonomous student tokens, continuation scoring includes natural
   close/parse/answer behavior, and final deployment never receives a prefix.
   Episode states additionally test real environment transitions.
7. **Episode replay can silently branch from different hidden states.**
   Resolution: reconstruct from seed, replay every visible action, assert every
   observation and `action_ok` flag, and checksum the state descriptor before
   generation. Any mismatch aborts the block.
8. **A verifier can become a hidden deployment router.** Resolution: it exists
   only at training/state-acquisition time. The primary artifact is one merged
   checkpoint. Visible two-checkpoint routing is an explicit deployment
   baseline, not smuggled into the model arm.
9. **Dynamic rerouting after each round could move the estimand.** Resolution:
   the algorithm, decode, branch count, comparisons, abstention, state sampler,
   and quotas are frozen. Re-estimating value on fresh current-student states is
   the definition of on-policy routing, not post-hoc rule tuning.
10. **Dense loss may punish valid correction forks despite outcome advantage.**
    Resolution: teachers receive no privileged hint, share origin, and operate
    on the identical state; initial KL/top-k overlap, entropy, exact-logit
    locality, and round overlap are recorded. Five unsafe updates stop all
    scaling.
11. **Starting from quick would weaken the baseline.** Resolution: regenerate
    the historically strongest 40/60 soup and require improvement over it, all
    endpoints, and fresh local soups. This makes any pass frontier-extending.
12. **Starting from soup deviates from MOPD's shared-root initialization.**
    This is deliberate: the question is whether local teacher residuals can
    improve the current best checkpoint. The no-update soup and fixed-deep
    controls reveal whether updates merely undo the merge.
13. **Control compute can drift.** Resolution: reuse primary state IDs and
    token masks, bind update/position counts, rescale initial objective pressure
    where losses differ, and publish forward/input/sampled token ledgers. The
    best-of-8 hurdle is intentionally compute-favored against the learned arm.
14. **A visible router is underspecified.** Resolution: freeze the deployable
    public rule—quick only for L1-L2 atoms, deep for L3-L6 atoms and every
    episode. The training-only verifier router is reported as oracle evidence,
    never substituted for this baseline.
15. **Multiple comparisons can make one seed look good.** Resolution: seed 42
    is primary before training, seeds 43/44 are directional replications, two
    final blocks are mandatory, no checkpoint selection is allowed, and every
    arm/item remains in the output.
16. **A procedural proxy can fail to transport to the blackbox instrument.**
    Resolution: no benchmark is opened until the stricter procedural test
    passes. A procedural pass is mechanism evidence; only the sealed run-only
    instrument can support cross-substrate capability language.
17. **Benchmark contamination is irreversible.** Resolution: benchmark code
    and data remain read-forbidden; no imports/references in experiment Python;
    CLI access is receipt-gated and aggregate-only.
18. **Reusing prior checkpoints can hide corruption or a no-op.** Resolution:
    bind source receipts/hashes, independently construct the soup, require all
    deltas nonzero, run same-prefix behavioral canaries, and evaluate only
    explicit merged composites through the pinned vLLM backend.
19. **The upstream medium frontier was initially underpowered.** Before this
    design lock, C54 pooled nine apex medium events and corrected `+0.345` at
    n=3 to `+0.321 ± 0.017 SE`; its tier router reaches rather than decisively
    clears that ceiling. Resolution: no source rank is assumed locally, and
    any terminal blackbox claim requires at least eight paired medium events.
    The visible C54 tier router remains a mandatory baseline.

## Required pre-output checks

- all 14 copied gym families pass oracle/random/degenerate selftests;
- synthetic route analyzer proves positive epsilon can pass while zero,
  winner's-curse-only, one-teacher-only, and block-unstable cases fail;
- episode reconstruction reproduces observations and scores exactly;
- corrected top-k value and gradient match full reverse KL at full vocabulary;
- state IDs, branch seeds, and selection/audit samples are disjoint;
- training units are consume-once and teacher/anchor quotas exact;
- every mandatory absolute gate is mathematically reachable;
- frozen files are hash-bound to an ancestor commit before loading a model.
