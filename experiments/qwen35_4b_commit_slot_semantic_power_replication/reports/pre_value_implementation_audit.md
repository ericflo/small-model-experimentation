# Pre-Value Outcome-Blind Implementation Audit

Completed after implementing the prefix-value path and before anchoring its
boundary, opening `value_fit`, or making a new model call. No value/causal task,
trace, correct-alias probability, chosen alias, or J feature was inspected.

## Verdict

Proceed only after committing and pushing this exact implementation, anchoring
all recorded hashes to that commit, rerunning repository checks, and passing one
outcome-blind value-model smoke. The scientific value stage gets one run. Causal
implementation remains unavailable.

## Audited assertions

1. The original design receipt and both complete seam decisions remain required.
2. Selection and confirmation summaries must be passing with exact labels
   `POWERED_COMMIT_SLOT_SEAM_QUALIFIED` and
   `POWERED_COMMIT_SLOT_SEAM_REPLICATED` at cap 1,024.
3. Runtime rehashes all ten selection/confirmation raw files and verifies that
   confirmation still points to the frozen selection summary.
4. Value-model smoke calls the license verifier with
   `verify_value_split=False`; its receipt confirms both value and causal splits
   remained unopened.
5. Prefix-value verifies only the recorded `value_fit` hash, then its dedicated
   loader opens exactly `value_fit.jsonl`. A monkeypatched test observes no
   other scientific data path.
6. `control-calibration` and `causal-confirmation` still call fatal
   `unavailable`; no placeholder causal row can be written.
7. The new boundary fails while `commit` is pending, before model load or data
   access. Runtime later verifies the anchored commit is an ancestor.
8. Boundary hashes cover the value-config payload, preregistration, adversarial
   review, runner, model operations, coordinate module, value analysis, tests,
   and this audit, at both the anchor commit and current worktree.
9. Value config preserves all inherited prefix-value bars from the original
   preregistration; tests reject a changed AUC gate.
10. Fixed model ID/revision, bf16 SDPA backend, cap, sampler, aliases, slot,
    lens hash, layer band, pseudoinverse tolerance, fold count, L2, and
    standardization remain unchanged.
11. One cached generation produces each full path. Feature and slot states are
    separate cache-free replays under the same model/backend.
12. Feature input contains prompt plus live thought prefix only. The capture
    method rejects close/EOS tokens and returns explicit `close_present=false`
    and `slot_present=false` checks.
13. Half-prefix capture recomputes only its exact live sequence; it never reads a
    historical position from the 1,024-token sequence.
14. Capture position is the final included thought token. Prefix length, full
    feature-context hash, prefix-thought hash, and terminal-thought hash are
    preserved per row.
15. Fractions 0.5 and 1.0 share one path's full-cap correct-alias-probability
    label. Tests reject fraction-specific labels.
16. Endpoint slot readout is reused exactly for fraction 1.0; half-prefix slot
    margin comes from its own separate forced slot.
17. Gold prefix probability is stored only as a direct-tautology diagnostic and
    does not appear in any fitted feature key.
18. The primary vector is all 24 frozen concepts at layers 4--8 in layer-major
    lens order: exactly 120 features. There is no coordinate/layer selection.
19. Five pseudoinverses are computed once per run and reused. The earlier draft's
    per-prefix refactorization footgun was removed before any model call without
    changing coordinate arithmetic.
20. Every frozen J dictionary must have effective rank 24. Concept/alias order,
    source layers, width, and all numeric finiteness are runtime contracts.
21. The equal-width non-J baseline uses 24 deterministic random coordinates per
    layer, projected out of the complete J span and orthonormalized. Tests on the
    exact lens verify byte-determinism, shape 2,560x24, and projection <=1e-5.
22. Scientific cardinality is exactly 144 traces and 288 prefix rows. Every
    `(task, fraction)` contains trace indices 0/1/2; missing paths yield
    `INVALID_PREFIX_VALUE`, not a smaller favorable pool.
23. Malformed pre-cap paths cannot be replayed. Natural-close paths use fractions
    of their exact available pre-close thought under the registered seam policy.
24. Rows are sorted stably by task, fraction, and trace before analysis and score
    attachment. OOF score arrays retain this order.
25. All paths and fractions of a task receive one deterministic fold. Alias-
    stratified assignment spreads every four-or-more-task alias across all four
    folds; tests assert sibling isolation.
26. Features and terminal labels are centered only within `(task, fraction)`.
    Validation labels do not enter scoring; across-group feature location/scale
    and ridge coefficients use training tasks only.
27. One L2=1 ridge model is used per feature family. No intercept, sweep, early
    stopping, refit-selected sign, coordinate selection, or probability
    threshold exists.
28. Eligible comparisons require terminal-label gap >=0.01. Prediction ties earn
    0.5. Scores macro-average path pairs within whole tasks.
29. Primary, correct-alias activity, ordinary slot margin, alias identity, and
    equal-width non-J random features all use identical folds, centering,
    standardization, estimator, and task-pair metric.
30. Half-prefix prospective AUC has its own gate; endpoint performance cannot
    rescue it.
31. Task-bootstrap lower bounds use 10,000 stable resamples for primary AUC and
    paired primary-minus-correct, minus-margin, and minus-non-J effects.
32. Thirty-two null refits shuffle complete J vectors only among sibling paths
    within task/fraction. Task, alias, fraction, dimension, and distributions
    stay fixed.
33. Alias-stratified, fold, fraction, per-task, null-repeat, and every individual
    gate diagnostic are preserved; none can rescue a conjunctive miss.
34. The final all-value-task ridge model is stored only for a separately audited
    untouched successor. Prefix-value pass remains measurement/oracle evidence.
35. Raw scientific trace/prefix/model artifacts and their hashes are written only
    after complete contracts. Counts-only progress is the only mid-run output.
36. Sixteen experiment tests pass; Python syntax and whitespace checks pass.

## Outcome-blind checks executed

- Synthetic prospective signal reaches AUC 1.0 while constant direct, margin,
  identity, and non-J baselines remain 0.5.
- Repeated within-group shuffle stays near chance under the frozen null logic.
- Fraction-label mismatch, missing sibling path, changed inherited gate, causal
  loader access, and unanchored boundary each fail for the intended reason.
- The committed seam-license verifier rehashed both stages while returning
  `value_split_opened=false` and `causal_split_opened=false`.

## Remaining accepted limitations

- Ridge ranking is trained with gold terminal values and is not deployable yet.
- Three paths/task estimate only a coarse local ordering; task macro/bootstrap
  prevents pseudo-replication but not low within-task resolution.
- Non-J random readout is one frozen generic subspace family, not an exhaustive
  comparison to every residual representation.
- Forced cap/slot remains a counterfactual deployment policy; natural free-form
  behavior is outside this stage.
- A value pass does not establish causal transport. It licenses a new exact-
  control design review, not causal data access.

None permits adapting this experiment after its scientific output.
