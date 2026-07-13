# Idea Intake

## Program Fit

- Programs: `interpretability_and_diagnostics` (primary),
  `test_time_reasoning_budget`, `evidence_conditioned_selection`, and
  `structured_execution_and_compilers`.
- Existing or new program: existing.
- Closest scorecards reviewed: all four named programs.
- Related queue item: `thinking_content_vs_compute_control`; not a duplicate
  because coherent content is already causal and this changes continuation
  proposals rather than measuring content attribution.

## Prior Evidence

- Anchor 1: `qwen35_4b_jacobian_transport_control_replication` proves early J
  edits causally transport a supplied concept and later consequence.
- Anchor 2: `qwen35_4b_commit_slot_semantic_power_replication` proves coherent
  native thought reaches a fixed answer interface, but scalar J value fails.
- Anchor 3: `qwen35_4b_counterfactual_order_support_selector` finds terminal
  vector attribution beats majority but fails strong value/relevance controls.
- Closest near-duplicate: `qwen35_4b_native_thought_jacobian_value_transport`.
  It tried to rank/patch high-value existing paths; this experiment creates a
  balanced set of new continuations without knowing value or the answer.

## Novelty Claim

Can a zero-sum bank of all 12 alias J directions at a 512-token midpoint
systematically diversify the next 512 reasoning tokens, so their final ensemble
beats clean and generic branches plus matched-compute independent sampling?

## Mechanism

The replicated J lens is a causally writable semantic coordinate system. Center
all 12 public alias directions so their branch deltas sum exactly to zero, fork
one shared native midpoint, and let each branch continue freely. Direct forcing
bias should cancel when final probability vectors are averaged, while systematic
hypothesis coverage may escape a mode that ordinary sampling repeats. Failure of
native-prefix target controllability, equality with Gram-matched non-J branches,
or loss to compute-matched full-path sampling falsifies the mechanism.

## Control Plan

- Baseline: 12 clean continuations of the same midpoint plus a master pool of 12
  fully independent 1,024-token traces; compare at sampled-token and total-
  forward-token matched prefixes and report the full K=12 overmatch.
- Mechanism-falsifying control: 12 J-orthogonal branch deltas with exactly the
  J branch Gram matrix/norms, including post-bf16 span and norm checks.
- Shift or robustness check: label-free mechanics, 24 fresh qualification
  tasks, then 48 untouched confirmation tasks with identical gates.
- Hidden-label boundary: branch construction, alpha selection, seeds, and
  ensemble output never consume the correct alias. Gold grades only after all
  arms and resource receipts are complete.

## Evidence Output

- Program evidence update: all four programs at terminal gates.
- Claim/synthesis: no claim before independent confirmation and matched-compute
  capability gain.
- Reusable artifact: cache-forking native-thought generator, balanced J branch
  bank, exact Gram-matched non-J control, and resource matcher.
- Stop/branch: no native target controllability stops before continuations;
  qualification must beat all clean/non-J/full-path baselines by 10pp with task
  uncertainty to open confirmation.

## Decision

- Run experiment: yes, mechanics first.
- Create program: no.
- Write synthesis only: no; proposal-shifting J branches are untested.
- Defer: all correctness-scored continuations until label-free mechanics passes.
