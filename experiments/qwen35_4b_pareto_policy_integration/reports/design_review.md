# Adversarial Design Review

## Verdict

Proceed after the implementation smoke and checkpoint-installation gates. This
is a materially different and better-posed successor, but its strongest risk is
proxy mismatch: C54's Pareto split was established on the held-out instrument,
whereas training and teacher audit must stay on firewall-clean procedural data.

## Findings and resolutions

1. **The old absolute teacher bar was invalid.** A bounded metric makes fixed
   absolute gains incomparable across baselines. Resolution: positive paired
   delta plus replication and a bootstrap lower bound; no effect-size floor.
2. **“Any positive sample” would still be too weak.** One lucky procedural item
   is not a teacher. Resolution: paired items, stratified bootstrap, and two
   independently seeded positive blocks. Evaluation size supplies sensitivity.
3. **Saturation is competence, not failure.** `ferrier` cannot improve much but
   still represents tool-use behavior worth preserving. Resolution: explicit
   retention-anchor semantics; saturation never vetoes the experiment.
4. **C54 policies may not be complementary on this proxy.** Resolution: require
   the crossover before MOPD. Do not infer it from training labels or benchmark
   family scores.
5. **Data-union controls are unusually strong.** `apex60` was already dominated,
   but relying on historical results would weaken attribution. Resolution:
   regenerate parameter merges and compute-overmatched union SFT locally.
6. **Runtime LoRA can silently no-op.** Resolution: explicit composite merge,
   nonzero delta receipts, same-prefix behavioral canaries, and vLLM model
   fingerprints before any score is accepted.
7. **MOPD can quietly become off-policy.** Resolution: fresh current-student
   rollouts each round, consume-once rows, recorded policy hashes, and a frozen
   off-policy control.
8. **Top-k loss was previously easy to misstate.** Resolution: retain the
   synthetic full-vocabulary value/gradient equivalence tests for the corrected
   `p_s log(p_s/p_t) - p_s + p_t` equation.
9. **A one-checkpoint claim could hide a router.** Resolution: routed inference
   is reported only as an upper reference; the primary artifact is one merged
   4B checkpoint with no adapter/router choice at deployment.
10. **Matched sampling must remain the baseline to beat.** Resolution: it is a
    terminal requirement, not a teacher-qualification hurdle. This prevents an
    intermediate gate from killing a useful teacher while retaining the repo's
    deployable standard.
11. **Benchmark contamination risk.** Resolution: no suite imports, content,
    items, transcripts, or result details are read. The run-only CLI remains
    closed until every procedural gate passes and stores aggregate results.
12. **Multiple comparisons and optional stopping.** Resolution: fixed two
    blocks and fixed sample counts; no arm/item filtering. One-sided bounds are
    applied only to preregistered directional comparisons and all raw aggregate
    arm results are retained.

## Required pre-expensive checks

- CPU selftests for all copied gym families and held-out-family exclusion.
- Exact MOPD value/gradient tests.
- Training encode audit and two-step nonzero adapter smoke for both data shapes.
- Explicit merge and same-backend behavior-change canary.
- Deterministic analyzer test showing a +epsilon paired effect can pass while a
  zero or unstable effect cannot.
