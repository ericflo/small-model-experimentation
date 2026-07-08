# Adversarial design review — verdict: sound_with_fixes (all must-fixes applied before the run)

Reviewed by a dedicated adversarial agent against the actual scripts before any GPU-scale run.
Key empirical anchors the reviewer verified: C18's real yield (80 pairs, depth mix {1:49, 2:24, 3:7},
57/90 tasks yielding nothing) and C46's P(True) never saturating (0/2163 values ≥ 0.99).

## Confounds caught → design changes

1. **Global top-n confidence selection bakes in depth collapse** (top-80 of a ~1000-candidate pool
   concentrates in easy depth-1 tasks), conflating the noise falsifier with the diversity falsifier.
   → PRIMARY arm is now `conf_strat` (depth-stratified top-P(True), quotas ∝ pool candidate counts —
   generator metadata, verifier-free); naive `conf_global` kept as a reported ablation.
2. **rand floor was uniform over unique candidates**, down-weighting high-frequency (correct-leaning)
   draws and inflating the conf-vs-rand contrast. → rand now samples ∝ draw frequency
   (Efraimidis–Spirakis weighted shuffle).
3. freq tiebreak injects self-consistency into the confidence arms — minor (P(True) doesn't saturate);
   p_true now stored at 6dp; exec∩conf_strat Jaccard + mean freq reported per arm.
4. exec differs from conf in ordering as well as keep-test — accepted (exec must stay C18-identical);
   overlap reporting disambiguates a null.
5. Single LoRA seed / ~30 optimizer steps per arm — pre-registered as a limit in the README.

## Must-fixes applied

1. **Smoke/full artifact collision** (smoke pool/train files silently reused by a full run):
   all smoke outputs now `_smoke`-suffixed; full run asserts train_tasks.jsonl has exactly 90 tasks.
2. **Canary now gates**: run.py hard-stops unless pool AUROC ≥ 0.65, conf_strat−rand purity gap ≥ 0.10,
   matched n ≥ 60 (below gate, "judge doesn't transfer" is the cheap finding; `--force` to override).
3. Conf depth policy pre-registered (stratified primary + global ablation, both trained; conf_global
   evaluated nothink-only to bound eval cost).
4. **Size mismatch is now a hard trim** to min(n) across arms (matched optimizer steps), not a warning.
5. **Calibration protocol extended**: fixed-judge-set pass (judge-change isolated) PLUS a
   self-distribution pass per arm (model judges its OWN think-mode candidates — the actual round-2
   flywheel number). Inflation headline switched to P(True)-on-INCORRECT drift.
6. **Recovery ratio now joint-bootstrapped** (same task resamples in numerator/denominator) with CI;
   decision rule switched to the pre-registered TRICHOTOMY (conf~rand / intermediate / conf~exec) —
   n=25/depth cannot resolve a 0.8-vs-1.0 ratio cutoff.
7. **Smoke now exercises the hour-5 failure modes**: adapter-loaded eval, adapter-loaded calib (fixed +
   self-dist), and analyze.py all run in `run.py --smoke`; auroc() None-guards added for single-class
   judge sets.

## Nice-to-haves adopted

Within-task AUROC in the harvest canary; judge-set K 8→12; eval_ladder default depths pinned to
[1,2,3] (depth-4 comparability cell consciously dropped); configs/default.yaml populated with the
actual constants. Deferred: purity-matched-diversity 4th arm (run only if conf < exec to complete the
noise-vs-diversity autopsy); 2-seed exec/conf training.

## Power (pre-registered expectation)

At n=25/depth with C18 effect sizes: exec-vs-base think-cov depth-2 is detectable; the run can
distinguish conf~rand from conf~exec (trichotomy), NOT recovery 0.8 from 1.0. Statements about the
ratio are magnitude-with-CI, never a pass/fail cutoff.
