# Adversarial Design Review: Counterfactual Order-Support Selector

Verdict before implementation: **sound only as a retrospective feasibility
gate**. It is not a capability result and cannot bypass a fresh matched-compute
comparison.

## 1. This is just the parent's ordered-over-shuffle result restated

The parent used the gold correct alias to report mean probability 0.239 ordered
versus 0.153 shuffled. That group effect does not say which of 12 aliases gained
for a new task. This experiment removes gold from prediction and commits to the
full vector argmax. Failure would be new and strategically useful.

## 2. Shuffling is an expensive hidden verifier

It is label-free but not free: three extra 1,024-token prefills accompany three
generated thoughts. Therefore K=3 majority/soft ensemble is only a same-pool
signal baseline, not the mission baseline. The report must never say “beats
sample more.” A subsequent experiment must compare against K=6 ordered paths at
matched actual forward tokens, and should additionally report attention FLOPs
because prefill and decode tokens have different cost.

## 3. The counterfactual may cancel alias priors without isolating reasoning

Ordered and shuffled rows share exact token multisets, length, close, slot, and
alias vocabulary, but order can alter generic recency, syntax, or positional
effects. A correct-alias-balanced task-mismatched shuffled distribution retains
alias nuisance while breaking task relevance. The candidate must beat it with
paired uncertainty. This control is oracle-balanced and cannot be deployed.

## 4. Three paths make the probability delta noisy

That is part of the intended deployment dose. Do not tune K, trim paths, select
only confident traces, or pool the two 113-task stages. Both stages must pass
unchanged. The task, not trace or alias, is the bootstrap unit.

## 5. Mean delta can choose an alias no path selected

That is allowed and potentially the mechanism: probability mass may contain a
weak common semantic contribution. It means individual-choice pass@3 is not an
oracle ceiling. Reports must separately show whether the candidate answer was
in the ordered argmax pool without treating absence as invalid.

## 6. Soft ensembling is the obvious simpler explanation

Argmax mean ordered probability, majority, max confidence, minimum entropy, and
first trace are mandatory. The candidate must beat every one by 3pp and have a
positive paired lower bound. Choosing whichever baseline happens to lose most
would be winner's-curse reasoning.

## 7. Probability subtraction is one of many adaptive transforms

Only raw probability difference is primary because it follows directly from the
replicated causal contrast and is bounded. Log ratio, logit reconstruction,
alias residualization, per-trace support, J features, sign flips, and learned
weights cannot rescue a failure. They require another experiment.

## 8. Alias identity can manufacture apparent breadth or accuracy

Require eight predicted aliases and successes across eight correct aliases on
each stage. The task-mismatch control cycles within correct-alias strata.
Per-alias outcomes are reported; stages may not be pooled to fill missing
support.

## 9. Hidden labels can leak through control construction

Deployable prediction functions take only probability matrices and public alias
order. Grading and mismatch construction are separate. Unit tests mutate
`correct_alias`, `correct`, and correct-alias probability fields and require
identical candidate/baseline predictions. The mismatch control is named oracle
in outputs.

## 10. Existing confirmation is not truly untouched

Its aggregate seam outcomes are published, although this derived selector has
never been computed. Treat a locked pass as a prospective secondary analysis,
not independent data collection. Even two passes merely license fresh tasks.
The confirmation files remain absent until qualification passes and its boundary
is pushed.

## 11. A negative could be blamed on the parent slot being artificial

The slot is the exact deployment interface whose semantic power replicated.
The question is scoped to that interface. Failure means the group-level causal
contrast is not a useful label-free selector there; it does not deny coherent
thought or all counterfactual selectors.

## 12. A positive could still be below a no-thought or single-path system

Mandatory first-path and all K=3 selectors prevent this. Candidate accuracy must
also remain within a 15%--70% headroom band. Report absolute accuracy and
win/loss ties, not only relative lift.

## 13. Bootstrap multiplicity can be hidden

All five baseline comparisons plus the mismatch comparison are conjunctive and
each needs a lower bound above zero. No best-comparison p-value is selected.
The frozen point-gain requirement is additionally 3pp for every comparison.

## 14. Confirmation files are easy to access accidentally

The experiment-local confirmation directory is absent. The runner never reads
the parent's paths, checks local expected hashes, requires a passing
qualification and a hash-locked boundary receipt, and fails before loading rows
otherwise. Tests exercise missing-file and failed-qualification paths.

## 15. This drifts away from Jacobian/J-space science

It follows the stronger lesson from the J-value negative: the relevant causal
object may be a vector-valued forward counterfactual, not a learned scalar state
coordinate. Exact order destruction is a finite counterfactual analogue of
Jacobian attribution. A positive would motivate cheaper local Jacobian
approximations; a negative prevents spending GPU time on that approximation.

## Reachability and stop rules

Every absolute gate is mathematically reachable: all baselines are bounded by
one and the candidate's maximum permitted 0.70 leaves a 0.03 gain possible
whenever a measured baseline/control is at most 0.67. The qualification runner
must record actual reachability and fail `GATE_INFEASIBLE` if any comparator
exceeds 0.67. This is a design stop, not permission to weaken a gate.

Terminal routing:

- `GATE_INFEASIBLE`: stop; do not reinterpret.
- `NO_ORDER_SUPPORT_SELECTOR`: retire this exact transform; confirmation sealed.
- `ORDER_SUPPORT_QUALIFIED`: commit/push, then authorize only frozen secondary
  confirmation.
- `ORDER_SUPPORT_CONFIRMATION_FAIL`: retire; no fresh GPU successor.
- `RETROSPECTIVE_ORDER_SUPPORT_REPLICATED`: create a new fresh matched-compute
  experiment; still no capability claim.
