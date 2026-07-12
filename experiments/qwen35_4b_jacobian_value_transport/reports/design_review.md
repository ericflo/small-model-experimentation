# Adversarial Design Review

Reviewed before the expensive run. The question is not whether Jacobian features
look interpretable; it is whether they provide a specific causal lever unavailable
to the already-failed ActAdd intervention.

## 1. Oracle answer injection could masquerade as capability

**Threat.** Writing the correct operation directly may simply encode part of the
answer.

**Resolution.** Oracle patches are labeled upper-bound mechanism evidence. G0
requires a downstream prompt-local consequence, not only direct concept report.
G2 requires exact full-task execution and compares tasks sharing operation type but
having different parameters. A deployable claim is deferred to a separate experiment
whose controller cannot access operations or hidden tests.

## 2. Final trace labels create false token credit

**Threat.** Correct traces contain bad steps later repaired; failed traces contain
good steps. Broadcasting final correctness to every thought token would create a
difficulty probe, not causal credit.

**Resolution.** Estimate value from fresh continuations at common-prefix checkpoints
and compare sibling branches within task. Whole-trace token labeling is prohibited.

## 3. Forced thought closure creates an unreachable state

**Threat.** C51 showed that teacher-forced answer potential after injected closure
can score a state the model almost never reaches.

**Resolution.** Value continuations remain in the native thinking channel and must
close naturally. Insufficient natural closure fails the gate. Teacher-forced margins
are oracle diagnostics only and never substitute for rollout value.

## 4. A J direction may only be a relabeled answer-logit direction

**Threat.** Late-layer directions or direct report tasks can trivially control the
next token.

**Resolution.** Require two adjacent earlier/middle layers and a downstream
consequence effect. Compare the ordinary logit lens at the same layers. Report direct
and consequence effects separately.

## 5. Multi-token aliases make coordinate identity arbitrary

**Threat.** Operation names and pseudowords may tokenize differently, invalidating
single-token J directions.

**Resolution.** The positive-control vocabulary is tokenizer-audited and restricted
to one token with its actual leading-space context. Multi-token operation concepts
are treated as exploratory phrase directions and cannot satisfy G0.

## 6. Layer search can manufacture a result

**Threat.** Sweeping layers and coefficients on the evaluation set creates hidden
multiple testing.

**Resolution.** Frozen layer set; calibration is split into selection and confirmation
halves. The selected adjacent band and coefficient are applied unchanged to G1/G2.
All swept cells remain reported.

## 7. Padding and cache behavior can corrupt Qwen's hybrid recurrence

**Threat.** Left/right padding and retroactive cache edits can change linear-attention
state independently of the intervention.

**Resolution.** Confirmatory causal generation uses batch size one, no padding, full
prefix recomputation, and `use_cache=False` in every arm. Batched fitting uses identical
token sequences within each replicated batch. Cache-enabled runs are timing-only.

## 8. Full Jacobian averaging may overweight early positions

**Threat.** The reference implementation sums future target cotangents then averages
source positions, so early source positions receive more target mass.

**Resolution.** Store and test an explicit equal-source-target-pair estimator. If the
paper-compatible summed-target estimator is also run, label it separately and do not
mix matrices or directions between arms.

## 9. Coordinate normalization and nonorthogonality can cause norm artifacts

**Threat.** Pseudoinverse swaps among unequal, correlated vectors can inject much more
energy than controls.

**Resolution.** Normalize dictionary columns, log condition numbers and residual delta
norms, reject ill-conditioned pairs, and norm-match every control at every example.
Report residual norm and parse/termination effects.

## 10. “Non-J” is undefined without a full sparse frame

**Threat.** Calling a vector orthogonal to a few targeted tokens “non-J” would overstate
the control.

**Resolution.** G0 uses an explicitly named targeted-span-orthogonal control. The G2
non-J remainder is permitted only after a full fitted matrix supports top-candidate
sparse pursuit. If full decomposition is unavailable, that mandatory arm fails closed.

## 11. Backend and compute mismatches can explain outcome changes

**Threat.** HF intervention arms and vLLM baselines are not sample-comparable; extra
recomputation or tokens could create an apparent gain.

**Resolution.** All result-bearing arms use the same HF implementation and full-prefix
recomputation. Generated-token ceilings, seeds, and attempted continuations are equal.
Wall time is reported but is not the matched-compute unit; model forward token counts
are the compute accounting basis.

## 12. A positive oracle patch may not generalize or deploy

**Threat.** Task-specific donor content or exact labels can produce a fragile local
effect.

**Resolution.** Hold out task seeds, operation parameters, string/register families,
and hard depth. Even a replicated G2 pass is labeled oracle causal transport, not
deployable capability. It authorizes, but does not merge with, a separate non-oracle
controller or counterfactual-reflection experiment.

## Review verdict

Proceed only in the registered order. The design has strong negative controls and
hard seams against the most likely false positives. The main residual risk is that
the single-token positive-control frame transfers poorly to multi-token operation
semantics; that limitation should produce a scoped negative rather than post-hoc
alias engineering on evaluation data.
