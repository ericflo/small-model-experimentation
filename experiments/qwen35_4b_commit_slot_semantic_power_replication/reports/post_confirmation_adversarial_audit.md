# Post-Confirmation Adversarial Audit

Completed after the automatic confirmation decision and before implementing or
opening value/J stages. This audit cannot change the frozen seam verdict. It
asks which stronger interpretations the result does and does not support, and
which nuisance controls must become load-bearing downstream.

## Verdict

Accept `POWERED_COMMIT_SLOT_SEAM_REPLICATED` exactly at its registered scope:
ordered cap-1,024 thought contains task-general information that improves a
fixed constrained semantic commit over both an exact token-multiset shuffle and
the same syntax with no generated thought. Do not call this autonomous answer
generation, internal certainty, J-space value, or installed capability.

Value/J implementation is now licensed but not yet present. It may proceed only
after a separate outcome-blind design/implementation review commits task-held-
out identity controls, direct-logit baselines, dynamic sequence geometry, and a
sealed causal split.

## 1. Confirmation was rescued by qualification

It was not. Qualification and confirmation independently used 113 disjoint
tasks, 339 traces, identical frozen gates, and separate task bootstraps.
Confirmation alone scored 98/339 ordered versus 47/339 shuffled, with a
one-sided task lower bound of +9.44pp. The terminal decision reads only the
confirmation gate after verifying qualification hashes.

## 2. A pooled statistic hides stage instability

The pooled statistic is diagnostic only. Separate two-sided task intervals are
[7.96pp, 19.17pp] and [8.26pp, 21.83pp]. Confirmation minus qualification was
+1.47pp with interval [-7.37pp, 10.32pp]. Neither stage needs pooling.

## 3. Three traces per task create pseudo-replication

The preregistered lower bound and post-decision uncertainty use the task as the
resampling unit. With exactly three paths per task, task macro and pooled point
means coincide, but path count is never the bootstrap sample size.

## 4. One or two tasks drive the gain

Ordered-minus-shuffled task effects were positive/zero/negative on 35/72/6
qualification tasks and 34/71/8 confirmation tasks. Mixed-outcome counts were
32 and 31, both above the frozen 28-task floor. The effect is sparse but not a
single-task event.

## 5. Alias masking manufactures semantic success

The mask guarantees format, not correctness. Before masking, an alias was
already the full-vocabulary top token on 88.20% and 87.61% of rows, and aliases
held 66.79% and 66.35% mean probability mass. The same suffix and mask are used
for ordered, shuffled, and no-thought comparisons.

## 6. The fixed suffix rather than thought content causes the gain

The suffix repairs answer mode, but it is identical in every slot arm. Ordered
thought beat the syntax-only no-thought arm by +17.40pp and +21.83pp and beat an
identical-length/token-bag thought by +13.57pp and +15.04pp. The causal contrast
is coherent order/content under this deployed suffix, not suffix availability.

## 7. Correct answer tokens are copied from thought

Exact shuffling preserves every mention token. Correct-alias mentions occurred
in only 16.2% and 19.8% of ordered prefixes. Ordered success with versus without
a correct mention was 29.1% versus 26.8% in qualification and 26.9% versus 29.4%
in confirmation. This diagnostic does not support a mention-copy explanation.

## 8. The shuffle is a complete model of incoherent reasoning

It is not. It tests token order/coherence against the identical token bag and
length. It is out of distribution and cannot show that every coherent wrong or
foreign thought would behave the same. The claim is ordered-over-shuffled
semantic contribution, not a complete causal decomposition of reasoning.

## 9. The result establishes natural or free-form capability

Every path was force-committed at cap 1,024. Close-only free-form success was
20/339 in each stage and 91%--93% of those outputs exhausted the answer cap.
The result is a counterfactual deployed commit interface, not autonomous
termination or ordinary generation.

## 10. Alias breadth eliminates identity bias

It does not. Correct successes spanned 11 and 10 targets and choices spanned all
12 aliases, satisfying frozen breadth gates, but confirmation `horse` had 0/30
ordered successes. Shuffle beat ordered for `tiger` and `river`. Raw alias
activity, gold-token logit, or slot margin can therefore look predictive
without representing task-general value.

**Downstream hardening:** task-held-out folds; centered features fit on training
tasks only; explicit correct-alias activity, slot-margin, and alias-identity
baselines; incremental AUC/gain gates; and alias-stratified diagnostics. No
feature may be selected on causal-confirmation tasks.

## 11. A correct-alias Jacobian is automatically a certainty direction

It is not. A Jacobian coordinate derived from the gold answer can be a direct
readout of the label or downstream logit geometry. Prefix-value measurement
must show out-of-task predictive information beyond the correct-alias activity
and ordinary output margin. Causal transport must act at a prefix position from
which later model computation consumes the state, not merely write the scored
answer logit at the commit token.

## 12. Whole-trace labels identify the causal token

They do not. Assigning final correctness to every prior token creates severe
temporal label leakage and within-trace dependence. Registered prefix fractions
must be analyzed as prefix states, grouped by task/path, and compared only to
outcomes downstream of that prefix. A future token-local controller needs its
own temporal/causal test.

## 13. Fixed control geometry can be reused across sequence lengths

Prior native-thought smoke measured up to 0.0625 historical-token activation
drift when suffix length changed. Every post-bf16 scalar/random/identity control
must therefore be constructed and audited at the live prefix length and layer.
Pre-cast orthogonality or a single calibration length is insufficient.

## 14. A positive value decoder would install capability

It would not. Gold-labeled prefix ranking is measurement; donor selection using
gold outcomes is oracle. Installation requires a label-free policy trained only
on fit tasks and a fresh held-out capability gain over frozen inference and a
matched-compute sampling baseline under the same backend.

## 15. The reserved data are now fair game for iterative tuning

Only the 48 value-fit tasks may inform representation/threshold choices. The 48
causal-confirmation tasks remain sealed through implementation, calibration,
and feature selection. A failed confirmation is preserved; no same-experiment
retry, layer sweep, alias remap, or threshold relaxation may rescue it.

## Required next boundary

Before any new model call, commit and push:

1. prefix/outcome definitions and exact row cardinalities;
2. task-held-out folds and identity/direct-logit baselines;
3. J-coordinate extraction with train-only centering;
4. dynamic per-prefix post-bf16 control construction and tolerances;
5. sealed causal-confirmation hash checks;
6. unit tests for leakage, task grouping, prefix alignment, and direct-logit
   tautologies; and
7. automatic stop labels that cannot be rescued by pooled or post-hoc analyses.
