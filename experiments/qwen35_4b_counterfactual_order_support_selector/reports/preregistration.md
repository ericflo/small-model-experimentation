# Preregistration: Counterfactual Order-Support Selector

Frozen before computing any derived qualification selector accuracy.

## Scientific boundary

This is a retrospective signal qualification on already committed procedural
rows. It cannot establish a matched-compute capability gain. A pass only
licenses a fresh experiment comparing three ordered generations plus three
shuffle prefills against six ordered generations at matched actual forward
tokens on new tasks.

Only `Qwen/Qwen3.5-4B` revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` produced the source rows. No model
is loaded here. Benchmark contents are never used.

## Frozen data boundary

- Qualification: 113 tasks, three ordered and paired exact-token-shuffle rows
  per task, copied byte-for-byte from the parent selection stage.
- Confirmation: a disjoint pre-existing 113-task parent stage. Its expected
  hashes are frozen, but the files are absent from this experiment.
- Confirmation may be copied only after qualification passes, its result and a
  confirmation-boundary receipt are committed and pushed, and the boundary
  commit is an ancestor of the current checkout.
- Stages never read parent paths. They read only their experiment-local split.

## Primary deployable rule

For task `t`, trace `i`, and public alias `a`, let `r_tia` and `s_tia` be the
ordered and exact-token-shuffled constrained alias probabilities. Compute

`d_ta = mean_i(r_tia - s_tia)`

and predict the lowest public alias index attaining `max_a d_ta`. The rule uses
no outcome, correct alias, task family, target operation, trace correctness, or
learned parameter. This raw probability delta is the sole primary candidate;
log ratios, residualization, subsets, and sign flips are diagnostic-only and
cannot rescue it.

## Deployable baselines

All use the same three ordered probability vectors:

1. first trace;
2. majority of ordered argmax choices, tied by mean ordered probability and
   then public alias order;
3. argmax mean ordered probability (soft ensemble);
4. chosen alias of the trace with largest maximum ordered probability; and
5. chosen alias of the trace with minimum constrained entropy.

The primary candidate must beat every baseline; selecting the weakest baseline
or pooling them is forbidden.

## Controls

- Reverse delta predicts `argmin_a d_ta`; the candidate must beat it by 10pp.
- Task-mismatched shuffle subtracts the mean shuffled distribution of the next
  task in a deterministic cycle within the same correct-alias stratum. This
  uses hidden labels only to preserve alias nuisance in a harder mechanism
  control. It is not deployable and never influences the candidate.
- Mutation tests replace all outcome fields while holding probability vectors
  fixed; every deployable prediction must remain identical.

## Gates, independently on both stages

- candidate accuracy in [0.15, 0.70];
- candidate minus each deployable baseline and mismatch control at least 0.03;
- one-sided 95% paired-task bootstrap lower bound above zero for every such
  difference (10,000 resamples; frozen seed 20260713 with stable name hashing);
- at least eight distinct predicted aliases and eight distinct correct aliases
  among candidate successes; and
- candidate minus reverse delta at least 0.10.

Every gate is conjunctive. Qualification failure is terminal
`NO_ORDER_SUPPORT_SELECTOR` and confirmation stays absent. Qualification pass
authorizes the frozen confirmation only. Confirmation pass is
`RETROSPECTIVE_ORDER_SUPPORT_REPLICATED`; failure is
`ORDER_SUPPORT_CONFIRMATION_FAIL`. Neither terminal state is claim-grade.

## Artifacts and uncertainty

Store split summary, task-level predictions/outcomes, all baseline/control
accuracies, point differences, paired bootstrap bounds, breadth, exact source
hashes, code/config hashes, and the confirmation-open flag. The task is the
only resampling unit.
