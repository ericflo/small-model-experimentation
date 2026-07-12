# Adversarial Design Review: Prospective Prefix J-Value

Completed after seam replication and before opening `value_fit`, writing a
scientific value row, or making a new model call. The review assumes that an
attractive J-value result can arise from answer-logit tautology, target identity,
whole-trace leakage, endpoint dominance, path pseudo-replication, or adaptive
analysis even when no useful prospective state exists.

## Verdict

Implement only the frozen 120-coordinate, task-held-out, within-task/fraction
ranking study in `prefix_value_preregistration.md`. Half-prefix prospective
performance, direct/activity baselines, task bootstrap, and within-group
feature shuffles are load-bearing. Keep `causal_confirmation` unopened and all
causal commands unavailable.

## 1. The replicated seam is silently relabeled J evidence

The seam read no activation or coordinate. It licenses measurement only.

**Hardening:** preserve separate seam and value decisions. Value starts from a
new config, boundary receipt, preregistration, design review, code hashes, tests,
and outcome-blind smoke.

## 2. The value split was inspected while choosing the design

That would turn it into selection data before the boundary exists.

**Hardening:** implementation and tests use synthetic rows only. Runtime opens
`value_fit` only after the value implementation commit is anchored and pushed.
The causal file is never read by prefix-value code.

## 3. Final correctness is painted onto every thought token

Whole-trace labeling can make adjacent or future tokens appear causal and treats
correlated positions as independent.

**Hardening:** exactly two frozen prefix states per path. The half prefix predicts
one later full-cap terminal value; the endpoint localizes seam-adjacent signal.
Tasks, not tokens/prefixes, are fold and bootstrap units.

## 4. The answer suffix leaks into the J feature

Capturing after `</think>\n\nFirst:` would make the coordinate an answer readout.

**Hardening:** feature forward is prompt plus live thought prefix only; capture
the final included thought token. Runtime asserts absence of close and slot.
Answer-side slot forwards are separate baselines/labels.

## 5. Later thought tokens leak into the half-prefix activation

Capturing position 512 from the full 1,024 sequence can change historical
activations under Qwen hybrid kernels and exposes future context operationally.

**Hardening:** recompute each live prefix as its own sequence. Never slice an
activation from the full sequence. Store exact sequence length and token hash.

## 6. Endpoint readout masquerades as forward value

The full-prefix coordinate sits immediately before the forced answer.

**Hardening:** require half-prefix prospective AUC >=0.58 separately. Overall
AUC and endpoint signal cannot rescue a half-prefix miss.

## 7. The target label is changed at each prefix

Immediate half-prefix commit probability measures stop-now value, not the later
outcome of the sampled continuation.

**Hardening:** both fractions share the path's full-cap correct-alias probability
as label. Prefix slot margin is a baseline only.

## 8. Gold answer identity is embedded in the feature

Selecting only the gold J coordinate would be an oracle direction.

**Hardening:** primary feature always includes all 24 concepts at all five
layers in fixed order. The model receives no gold index. The five gold
coordinates are a separately labeled oracle baseline the primary must beat.

## 9. Alias priors explain the score

The seam audit found severe target heterogeneity.

**Hardening:** split whole tasks with alias stratification; center within each
task/fraction; score only within-task path ordering; include a gold one-hot
identity baseline and alias-stratified diagnostics.

## 10. Task difficulty explains the score

Globally easy tasks have both high labels and stable representations.

**Hardening:** subtract feature and label means within `(task, fraction)` and
evaluate only candidate pairs from the same group. No cross-task row AUC is a
gate.

## 11. Fraction identity explains the score

Endpoint states and values can differ systematically from midpoint states.

**Hardening:** center separately within each task and fraction; report and gate
the half fraction separately.

## 12. Held-out tasks leak through paths

Random row folds would put sibling paths/fractions in train and validation.

**Hardening:** deterministic group folds at the whole-task unit. Tests assert no
task appears in more than one fold.

## 13. Validation statistics leak into standardization

Using all rows for mean/scale can subtly reveal held-out geometry.

**Hardening:** within-group feature centering is an available label-free policy;
all across-group standardization parameters come only from training tasks.
Zero-variance columns use scale one.

## 14. Ridge strength is tuned on the value result

Even a small L2 sweep can select noise.

**Hardening:** exactly one L2=1 ridge estimator, no intercept after centering, no
hyperparameter search, early stopping, sign flip, or refit on all tasks for the
scientific score.

## 15. Pooled paths create impressive precision

There are 288 prefixes but only 48 independent tasks.

**Hardening:** macro-average pairwise agreement per task; bootstrap tasks. Row
count is a completeness gate only.

## 16. Tiny probability differences become ranking wins

Floating-point noise can generate arbitrary order among nearly tied labels.

**Hardening:** compare a pair only when terminal correct-alias probability differs
by at least 0.01. Tied predictions score 0.5.

## 17. Ordinary confidence already explains value

J need not matter if a scalar answer margin ranks the same paths.

**Hardening:** fit/evaluate same-prefix constrained top1-minus-top2 margin under
identical folds and metric; require +0.02 and a positive task-bootstrap lower
difference.

## 18. A direct gold-coordinate readout explains value

The frozen J dictionary is a logit pullback, so gold-coordinate activity is a
serious tautology control.

**Hardening:** fit the five gold-coordinate features under the identical
pipeline; require +0.03 and a positive task-bootstrap lower difference. Scope a
pass as multivariate J information, not metaphysical certainty.

## 19. The label itself is used as a baseline feature

Prefix gold probability would trivially predict itself, especially at 1.0.

**Hardening:** store it as an explicitly excluded direct-tautology diagnostic.
Neither gold probability nor gold logit enters any fitted feature set.

## 19a. Any wide residual readout would work

A 120-dimensional J model can beat scalar baselines even when J geometry is not
special.

**Hardening:** build 24 outcome-blind random coordinates per layer, exactly
orthogonal to the full J span before any outcome, for an equal-width 120-feature
generic-state baseline. Require J +0.02 and a positive paired task-bootstrap
lower difference. This readout control is distinct from later post-bf16 causal
delta controls.

## 20. Shuffling across tasks creates an easy null

Global shuffle destroys alias and difficulty structure.

**Hardening:** shuffle complete J vectors only among three sibling paths within
the same task/fraction, then refit. Repeat 32 times and gate mean null AUC.

## 21. A lucky shuffled null is overinterpreted

A single three-path permutation can deviate materially by chance.

**Hardening:** freeze 32 stable repeats before outcomes. Report mean, range, and
all repeat AUCs; the mean must remain within 0.05 of chance.

## 22. Lens rank or concept order changes

Misordered columns could silently map aliases to the wrong features.

**Hardening:** verify lens hash, five layer keys, rank 24 each, exact concept
order, 120-vector width, and alias-to-concept index before model loading/scoring.

## 23. Natural close changes prefix semantics

A shorter naturally closed thought differs from a forced cap path.

**Hardening:** retain the registered seam policy: prefixes are fractions of
available pre-close thought and store exact length/mode. Malformed early EOS is
unusable. Report mode/length diagnostics; completeness and finite gates apply.

## 24. Cached and cache-free states are mixed

Generation uses cache while feature extraction uses full recompute.

**Hardening:** this matches the established seam: sampled tokens come from the
audited cached generator; all compared feature/slot rows are exact cache-free
replays under one model/backend. Store both contracts and do not compare seed-
matched outputs across backends.

## 25. The value model is secretly a deployable capability result

Training uses gold terminal values and compares sampled paths.

**Hardening:** label a pass measurement/oracle. Do not claim accuracy gain,
causal transport, or installation. A separate label-free controller and
matched-compute endpoint remain required.

## 26. A value pass immediately opens causal confirmation

Control construction and causal semantics are not implemented or audited yet.

**Hardening:** prefix-value pass licenses code work only. `control-calibration`
and `causal-confirmation` continue to raise fatal unavailable errors.

## 27. A failed gate is rescued by another fraction or subgroup

Alias/fraction/task slices can always reveal an attractive cell.

**Hardening:** all registered gates are conjunctive. Per-alias, mode, and
endpoint diagnostics cannot rescue overall or half-prefix failure.

## 28. An interrupted run leaves partial scientific evidence

Reading partial labels could guide a retry.

**Hardening:** keep rows in memory and emit counts-only progress. Write scientific
rows and summary only after exact cardinality, fold, finite, hash, and metric
contracts pass. No resume/retry within this experiment.

## 29. The implementation boundary is self-attested

Editing code after audit would invalidate the freeze.

**Hardening:** the value config records a design commit and hashes its config
payload, preregistration, review, runner, model operations, coordinate module,
value-analysis module, and tests. Runtime verifies all before opening data.

## 30. Causal data leaks through a generic loader

Convenience code could enumerate all manifest splits.

**Hardening:** prefix-value runtime names and hashes only `value_fit`; it must not
call `build_splits`, glob the data directory, or read the causal path. A test
monkeypatches the JSONL reader and asserts the only scientific data path opened
is `value_fit.jsonl`.

## Required pre-run audit

After implementation, run outcome-blind unit tests and model smoke using a
synthetic eight-token prefix only. The smoke may record architecture, lens rank,
feature width, token positions, finite values, and separation of feature/slot
contexts. It may not open `value_fit`, record a task label, chosen alias, correct
alias probability, or trace text. Commit and push the audit/boundary before the
one scientific prefix-value run.
