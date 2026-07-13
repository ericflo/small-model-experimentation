# Idea Intake: State-Formation Capacity Adjudication

## Program fit

- Primary program: `structured_execution_and_compilers`; this is a controlled question about whether
  a recurrent latent execution state can be installed under two adaptation parameterizations.
- Secondary program: `posttraining_and_adaptation`; factorized LoRA versus direct full-shape updates,
  conditional objective removal, and adaptation-disabled diagnostics make the result reusable evidence
  about update mechanisms. The state-formation endpoint remains primary, so this secondary fit does
  not add a transfer, capability, or generic posttraining claim.
- Existing or new program: both are existing programs.
- Closest program scorecard reviewed: Structured Execution And Compilers, whose current next step
  explicitly calls for shared initialization, controlled stochastic streams, state-only diagnosis,
  and three fixed-final seeds.
- Related future queue item: `supervision_causality_ablation`. This experiment is narrower: its joint
  versus state-only arm exists only to make the LoRA/full-rank capacity adjudication interpretable.

## Prior evidence

- Anchor 1: [`qwen35_4b_state_carry_vs_state_bag`](../qwen35_4b_state_carry_vs_state_bag/README.md).
  Its mechanically valid 300-step rank-32 LoRA pilot reached joint state accuracy `0.00459` against
  the `0.40` gate. It raised the capacity concern but was single-seed and jointly optimized state and
  answer.
- Anchor 2:
  [`qwen35_4b_state_carry_vs_state_bag_fullrank_delta`](../qwen35_4b_state_carry_vs_state_bag_fullrank_delta/README.md).
  Its direct-delta pilot reached `0.00277`, but shared state tensors and training dropout were not
  matched across the two experiments, and simultaneous Carry/query failures made its authoritative
  disposition `PILOT_PROMOTION_BLOCKED` rather than a capacity closure.
- Anchor 3: [`qwen_fastweight_hook`](../qwen_fastweight_hook/README.md). Its small 256-dimensional,
  answer-supervised recurrent hook produced favorable 100-item fluctuations that disappeared on
  250-item retests, motivating dense state supervision, fixed-final selection, and multi-seed
  evidence.
- Closest duplicate or near-duplicate: `qwen35_4b_state_carry_vs_state_bag_fullrank_delta`. It is not
  sufficient because its full-rank arm was a separate single-seed pilot with unmatched construction
  RNG/dropout and a mixed verdict ladder.

## Novelty claim

This is the first repository experiment to compare rank-32 and direct-full-rank extra-call adaptation
under bit-identical common loop-state initialization and matched per-microbatch dropout, using three
fixed-final seeds and a sequential joint/state-only control ladder whose terminal axis is state
formation alone.

## Mechanism

Rank-32 updates may be too constrained to install the transition operator needed by a repeated native
Qwen block, even though the carried activation is full width. Direct full-shape deltas remove that
factorization constraint. A LoRA joint pass falsifies low-rank prevention in this design; a robust
direct-recipe rescue after a LoRA miss supports practical relief. Failure of both recipes does not
falsify or prove a rank mechanism because direct optimization may also have failed. The conditional
state-only pattern then gives a descriptive signature consistent with objective interaction where
possible (without causal identification) and otherwise leaves the
registered recipe bottleneck unresolved.

The cross-capacity comparison is deliberately called practical direct-full-shape recipe evidence:
factorization also changes parameter count, adaptation FLOPs, and optimizer geometry. Shared
initialization, dropout, row order, groupwise clipping, targets, and schedule remove avoidable
confounds without pretending the two optimizers are mathematically identical or identifying rank
alone.

## Control plan

- Baseline: the intact rank-32 LoRA joint arm, not Bag and not an answer-only model.
- Mechanism-discriminating control: conditional direct-full-rank joint adaptation on the same targets,
  seeds, initialization bundles, row order, and dropout schedule.
- Objective control: state-only training for LoRA after a LoRA joint miss and for full rank only after
  a full-rank joint miss.
- Adaptation-dependence control: reevaluate every trained checkpoint with adaptation disabled while
  keeping the trained shared state modules and heads intact. If intact misses but disabled passes,
  report `ADAPTATION_DISABLED_REVERSAL` without changing the intact-checkpoint branch; this applies to
  LoRA and full rank on trigger data and is also reported for both joint capacities on sealed data.
- Setup positive controls: a tiny fixed overfit path plus an oracle/readout-path check; neither is a
  result or a tunable pilot.
- Parent-implementation control: on copied deterministic base/input/A/B tensors, the actual custom
  LoRA hook must match a pinned PEFT `lora.layer.Linear` reference in adapted output and A/B gradients
  at alpha/r = 2 in two regimes. FP32/dropout-off uses `atol=1e-6`, `rtol=1e-5`; live-like
  bf16-autocast/dropout-0.05 uses `atol=2e-3`, `rtol=1e-2`. The device RNG is reset to the same seed
  immediately before each forward so PEFT and the custom native-dropout path consume the matched
  realized-mask position.
- Shift checks: unseen semantic depths and the joint held-out transition-family plus surface-template
  split are gated and reported separately by seed and depth. Three fresh splits remain sealed: 768
  trained-depth rows at seed 73307 (depths 2–4, 256/depth) and 1,024 deep rows at each of seeds 73305
  and 73306 (depths 5–12, 128/depth). They open only after a LoRA miss, completion of every Stage-B
  fixed final and trigger evaluation, a direct-full-shape trigger pass, and dedicated no-prior-access
  authorization. Each of the six capacity×seed evaluation jobs scores all three sealed splits, so the
  third domain closes the shallow replication hole without adding selected model runs. A full-rank
  trigger miss leaves all three unopened and mandates state-only directly.
- Selection-safe replication control: trigger failure categories map one-to-one from `trained`,
  `depth`, and `joint` to `contrast_validation`, `contrast_depth`, and `contrast_joint`. Every category
  that failed on trigger must fail again in its counterpart before a full-rank absolute pass can
  support rescue; additional sealed failures are allowed. A missing category replication emits
  `LORA_TRIGGER_FAILURE_CATEGORIES_NOT_REPLICATED_ON_SEALED_CONTRAST` and stops.
- Sealed retry control: an interrupted same-cell evaluation can replay only after exactly one new
  content-addressed `FAILED_ATTEMPT_ARCHIVED` receipt is tracked, revalidated against its archived
  bytes and lineage, and appended to the existing access-ledger event. Completed evaluations cannot
  be archived as failures, initial access rejects an archive predating its event, and one archived
  attempt cannot license multiple retries. Ledger revisions hold a separate stable lock inode and use
  an atomic temporary-file-and-parent-directory-fsynced replacement.
- Hidden-label boundary: procedural state labels supervise training and registered evaluation only.
  No benchmark file is read, no evaluation row selects a checkpoint, and no result changes seeds,
  steps, objectives, thresholds, or branch order.

## Evidence output

- Program evidence update: after a valid terminal result, record whether LoRA formed state, full rank
  rescued it, or neither parameterization did under valid controls in the primary structured-execution
  ledger; mirror only the update-mechanism implications licensed by the result into the secondary
  posttraining-and-adaptation program.
- Claim ledger or synthesis update: only after the complete registered branch; a setup failure or
  partial seed matrix creates no claim.
- Reusable artifact: common adaptation-hook interface, serialized common-state initialization
  bundles, matched-dropout receipts, groupwise optimizer receipts, and adaptation-disabled state
  evaluation.
- Stop or branch condition: LoRA joint pass prohibits full rank; valid LoRA joint miss mandates LoRA
  state-only and full-rank joint; after a full-rank trigger pass, a LoRA intact pass in every required
  cell across all three sealed domains emits
  `LORA_TRIGGER_MISS_NOT_REPLICATED_ON_SEALED_CONTRAST` and stops regardless of full-rank score; a
  valid full-rank intact miss otherwise mandates full-rank state-only. With full rank passing, any
  missing corresponding LoRA failure-category replication emits
  `LORA_TRIGGER_FAILURE_CATEGORIES_NOT_REPLICATED_ON_SEALED_CONTRAST` and stops before rescue. Any
  mechanics/control failure stops for repair.

## Decision

- Run experiment: yes, after preregistration, adversarial design review, implementation review, and
  live setup gates pass.
- Create program: no.
- Write synthesis only: no; the existing evidence cannot identify the capacity effect.
- Defer: result-bearing execution is deferred until the documented gates are machine-enforced.
