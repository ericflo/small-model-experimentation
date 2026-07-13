# Qwen3.5-4B Jacobian Counterfactual Branching

This experiment tests whether a balanced bank of early J-space edits can shift
the proposal distribution of native reasoning, rather than trying to value a
finished thought.

## Research Program

- Primary: `interpretability_and_diagnostics`.
- Secondary: `test_time_reasoning_budget`, `evidence_conditioned_selection`,
  and `structured_execution_and_compilers`.
- Direct parents: `qwen35_4b_jacobian_transport_control_replication`,
  `qwen35_4b_commit_slot_semantic_power_replication`, and
  `qwen35_4b_counterfactual_order_support_selector`.

## Question

Starting from one shared 512-token native thought, can 12 zero-sum semantic J
branches create more useful 512-token continuations than clean or generic
branches and beat fully independent sample-more at matched model compute?

## Hypothesis

The replicated early J directions are causally consumed concept state, while
terminal J/value readouts and probability attribution fail. A balanced branch
bank uses each of the 12 public alias directions once, with deltas centered to
sum exactly to zero. Averaging final alias probabilities cancels direct branch
bias, but systematic hypothesis exploration may uncover a correct continuation
that repeated stochastic decoding misses.

## Setup

- Only `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Frozen lens SHA-256
  `e373b6e93956fdfc5cb446e9bee8249655707c8258a7868f0653d11f1ffd0213`,
  layers 4--8, first 12 public aliases.
- Layer norm anchors are the median replicated donor-clamp norms from the
  independent transport confirmation: 2.09603, 0.09454, 0.13078, 0.14686,
  and 0.06200.
- Fresh exact-depth-two procedural tasks: four label-sealed mechanics, 24
  qualification, and 48 untouched confirmation.
- One shared prefix runs to 512 tokens. Cache is forked before its final token;
  each arm consumes 12 continuations to the fixed 1,024 cap.
- J deltas are the centered 12-alias dictionary, scaled by one common multiplier
  per layer so branch deltas sum exactly to zero.
- Non-J deltas are built by an orthogonal rotation preserving the complete 12 x
  12 J branch Gram matrix at every layer, then repaired/audited after bf16.
- Primary output is argmax mean final constrained alias probability across the
  12 branches. No branch target, score, alpha, or seed uses the correct alias.

## Arms and matched compute

1. `j_balanced`: 12 centered J branches from the shared midpoint.
2. `clean_shared`: 12 stochastic branches from the identical midpoint/cache.
3. `non_j_gram`: 12 Gram/norm-matched J-orthogonal branches.
4. `full_sample_more`: a frozen master pool of 12 independent 1,024-token
   thoughts, reported at sampled-token- and total-forward-token-matched prefixes
   plus the conservative full K=12 overmatch.

All arms use Transformers bf16 SDPA because activation intervention requires
internals. Backend mixing is forbidden. Every prompt/decode/prefill/cache/slot
token and attention-shape proxy is counted.

## Gates

Before any correctness-scored continuation, four mechanics tasks select the
smallest frozen multiplier from `[0.5, 1.0, 2.0]` that reaches at least 60%
immediate target selection, at least +0.15 mean target-probability lift, at least
+35pp target-selection specificity over non-J, exact zero-sum branch geometry,
and 100% post-bf16 numeric controls. No multiplier passing is terminal
`NO_NATIVE_J_BRANCH_CONTROL`.

On qualification, J must beat every clean/non-J/compute-matched deployable
selector by at least 10pp with one-sided paired-task lower bound above zero,
preserve or improve oracle answer coverage by 5pp, span at least eight predicted
and six successful aliases, and pass all resource/backend/numeric contracts.
Only then may identical untouched confirmation open. Stages never pool.

## Run

Pending implementation boundary. All model stages fail closed meanwhile.

## Status

Design and adversarial review in progress. No model or outcome has run.

## Artifacts

- `idea_intake.md`: novelty and routing.
- `reports/preregistration.md`: immutable scientific rules.
- `reports/design_review.md`: adversarial review before implementation/GPU use.
- `assets/context_lens.pt`: byte-identical causal lens anchor.
- `src/`: frozen tasks, branch geometry, cache-fork model operations, and pure
  statistics.
- `reports/artifact_manifest.yaml`: external/omitted artifact policy.
