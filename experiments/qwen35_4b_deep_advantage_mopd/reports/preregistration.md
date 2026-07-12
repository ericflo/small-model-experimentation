# Preregistration

Frozen before any task-model output for this experiment.

## Primary Hypothesis

The exact strict same-prefix rule will requalify deep on two new soup-state
blocks. Conditional on that prerequisite, corrected deep-top-50 MOPD on only
deep-selected current-student states can produce one Qwen3.5-4B checkpoint
whose held-out joint capability exceeds both sources, the initial soup,
visible routing, matched controls, and execution-filtered sample-more without
retention or transfer loss.

## Fixed Policies And Provenance

- `quick`: C54 `quick_blend`, explicitly merged from the pinned base.
- `deep`: C54 `deep_apex`, explicitly merged from the same pinned base.
- `soup`: the completed predecessor's immutable explicit 40% quick / 60% deep
  delta merge, file SHA-256
  `04610723f3f46d0a094ae0e5bc1a491bb6ad9e0fb6c8a84417dfe5e527f15b50`.
- `student_r`: the explicitly merged primary output after round `r`; round
  `r+1` states must come from that exact checkpoint.
- No other model may generate, teach, score, judge, or supply capability.

The new run rechecks every source/merge hash and repeats the behavioral canary.
Reusing the exact soup avoids another numerically different reconstruction;
all new adapters/checkpoints live under this experiment's separate external
artifact root.

## Fresh State Construction

Only the copied firewall-clean procedural gym may produce states. The 12
trained families can supply updates; `brinework` and `spindle` remain
transfer-only. Qualification seeds `98100/98200`, online seeds `98300` through
`98600`, and confirmation seeds `98700/98800` are disjoint from the
predecessor and from one another.

Each qualification block contains 144 failed atom states and 48 failed episode
states chosen round-robin over frozen `(family, kind, level)` cells before any
teacher branch exists. Atom states preserve the exact autonomous student prompt
plus the registered mid-thought token prefix. Episode states reconstruct the
exact visible history immediately before the first invalid action, or before
the final action if every action was syntactically valid. Replay mismatch,
insufficient supply, or duplicated identity stops the stage.

## Frozen Deep Route And Estimand

For every state and each policy in `{quick, deep, student}`, draw four
selection continuations and four disjoint audit continuations with the same
vLLM protocol. Deep routes iff its selection mean is strictly above both quick
and student. Quick may be selected diagnostically, but it cannot authorize
training; ties and every other outcome abstain. There is no effect-size margin.

For deep-selected states, the primary audit contrasts are:

- `D_student = V_audit(deep,s) - V_audit(student,s)`;
- `D_quick = V_audit(deep,s) - V_audit(quick,s)`.

Inference uses state-level audit means, resamples states inside frozen
`(block, family, kind, level)` cells, and macro-averages cells. Continuations
are repeated measurements, not independent task units.

MOPD is authorized only if deep routes at least 16 states in each block, both
contrast macros are positive in each block, and both pooled one-sided 95%
stratified-bootstrap lower bounds exceed zero. Quick and combined-router
summaries are diagnostic only. Any failure is a terminal teacher-replication
result and stops before target caching, locality, or training.

## Conditional Deep-Only Update

Four online rounds are fixed. Each round reruns the identical three-policy,
four-branch selection rule on fresh states from the exact current student. It
freezes 60 consume-once deep-selected capability states and 20 consume-once
successful soup anchors. A state never contributes twice within a round.

On the current student's exact completion, the selected target supplies its
full-normalizer top-50 distribution. The primary minimizes corrected sparse
reverse KL,

`p_s log(p_s / p_t) - p_s + p_t`,

at at most the final 256 natural policy positions. Injected close tokens are
masked. The loss never substitutes a teacher-generated continuation. Each
round has 20 optimizer updates, gradient accumulation 4, learning rate `1e-5`,
rank 32, alpha 64, and the frozen soup target on the 20 anchors. Non-finite
values, mean corrected loss above 0.10, decreasing target top-50 overlap,
missing quota, or provenance mismatch stops and preserves the receipt.

## Matched Mechanism Controls

The assembler also freezes 60 failed states that did not select deep. Each is
matched one-to-one to a primary state without replacement using the first
available tier in `exact cell -> family/kind -> kind/level -> kind`; no
cross-kind fallback is legal. These states never enter the primary update.

- `non_advantage_route`: deep MOPD on those 60 matched non-deep-selected
  states plus the same 20 anchors. This tests whether positive routing matters.
- `wrong_teacher`: quick MOPD on the exact 60 primary states plus the same 20
  anchors. This tests whether the selected source identity matters.
- `offpolicy_sft`: SFT on the highest-scoring deep selection continuation for
  each primary state, plus matched student anchors. This tests dense on-policy
  distribution transfer against ordinary best-trajectory imitation.
- `soup25/50/75`: explicit source-delta parameter soups.
- `quick`, `deep`, `soup`: frozen source and no-update baselines.

All trained arms use 80 total updates across four rounds, the same active
position cap, consume-once geometry, and initial-objective pressure matching.
The seed-42 primary data are reused by controls; controls never generate a
friendlier state distribution.

## Locality Gate

Before full training, run five implementation-identical primary updates (15
deep units, 5 soup anchors), explicitly merge, and measure full-vocabulary
batch-of-one logits on those exact units. Require finite values, completed
updates, training safety, mean per-row median centered non-target drift
`<=0.10`, relative entropy drop `<=0.10`, and mean corrected top-k loss
`<=0.10`. Failure stops every full training arm.

## Final Procedural Decision

Primary seed 42 and fixed replications 43/44 are trained without checkpoint
selection. Two fresh full-distribution blocks cover all 14 families, quick and
deep atoms, and deep episodes. The visible router deploys quick for L1-L2 atoms
and deep otherwise. Soup best-of-8 returns the verifier-best of all eight
samples with every sampled token counted.

Seed 42 passes only if:

1. its joint paired delta has positive means in both blocks and a positive
   one-sided 95% lower bound versus quick, deep, soup, visible routing,
   `non_advantage_route`, `wrong_teacher`, `offpolicy_sft`, and all parameter
   soups;
2. its quick and deep stratum means each exceed the better source mean in each
   block;
3. seeds 43 and 44 have positive joint deltas versus quick, deep, and soup in
   both blocks;
4. no registered retention cell or transfer-family macro regresses more than
   0.02 versus soup; and
5. seed-42 greedy joint performance exceeds verifier-best soup best-of-8 in
   both blocks.

No fixed gain magnitude is required. Failure after qualification is a negative
for this deep-routed MOPD recipe, not permission to pick a seed, state subset,
round, or threshold after observation.

## Benchmark Firewall

Only a passing procedural receipt can authorize the run-only Menagerie CLI.
No experiment code may import or read benchmark family code, data, items,
transcripts, or item-level outputs. Conditional events use the frozen namespace
starting at 56201, the same backend/decode for paired arms, three quick events,
and eight medium events. Every event is reported. A blackbox claim additionally
requires the primary to beat the appropriate quick/deep visible source and soup
in paired aggregate on both tiers; procedural evidence alone remains scoped to
the copied gym.

## Interpretation Map

- Deep route fails: the predecessor's deep signal did not replicate; MOPD is
  still untested.
- Route passes, locality fails: useful deep states exist but this dense update
  is non-local at the frozen dose.
- Primary matches non-advantage deep: deep pressure, not advantage routing,
  explains any movement.
- Primary matches wrong-teacher quick: selected teacher identity is not causal.
- Primary beats controls but not sources/router/sample-more: partial
  installation, not a capability breakthrough.
- Every procedural and blackbox gate passes: evidence that a verified local
  residual can be installed into one 4B beyond the measured source frontier.
