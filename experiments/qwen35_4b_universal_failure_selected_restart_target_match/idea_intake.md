# Idea Intake: Failure-Selected Counterfactual Restart Curriculum

## Program Fit

- Program: `agentic_breadth_installation`.
- Closest scorecard: Agentic Breadth Installation.
- Related-work searches:
  - `make related QUERY="short pre-failure decision boundary on-policy correction exact target token exposure universal execution induction probe replay"`
  - `make related QUERY="first-error localization confidence dip short prefix targeted repair emission seam forced-close context-only"`

## Prior Evidence

- Closest near-duplicate:
  `qwen35_4b_universal_on_policy_prefix_repair_token_match`. It selected 60 real
  parent failures but trained after 47,123 masked parent-prefix tokens. The candidate
  scored 15/26 versus replay's 18/26, lost 0/1 on the six execute/induct/probe rows,
  and had 33,421 fewer supervised target tokens. That is a terminal negative for long
  masked failure-prefix continuation, not for task-level failure selection.
- C58 found first slips locally reachable and obtained partial improvement by removing
  the failed context and recomputing in isolation. This motivates restarting before
  the error rather than carrying the error state forward.
- C50's positive breadth result concentrated supervision at a deployment-relevant
  interface and kept the target canonical and terse. Full truth-audited restarts keep
  both the decisive computation and answer seam loss-bearing.
- C53 warns that same-recipe continued SFT saturates; replay must remain the active
  control, not the explanation.

## Novelty Claim

This is the first universal-curriculum trial to use the stronger replay composite's
fresh failures only as a task selector while deliberately removing the parent's
failed trajectory from the supervised context, under exact forward and target
exposure matching.

## Mechanism

The stronger deployed parent attempts 624 fresh procedural tasks spanning all 13
universal skills. A frozen model-free rule ranks correctness/cap failures before
bounded-compute failures and selects four per skill. Each selected training example
starts from the original prompt and supplies the generator's concise executable
solution, close, and exact answer. The intervention is therefore a counterfactual
restart before the error, not a continuation after it.

## Control Plan

- Incumbent: authenticated merged `replay_after_close` parent.
- Active control: continue that same parent on replay only.
- Shared training geometry: 320 rows, 200 aligned byte-identical replay rows, one
  epoch, 40 optimizer updates, training seed 48.
- Exposure gate: exact equality of forward tokens, nonzero/loss-bearing target
  tokens, and absolute loss mass, with zero tokenizer skips. If the deterministic
  three-axis solver cannot satisfy all axes, the design stops before training.
- Selection gate: four eligible failures in every skill; no quota borrowing or
  threshold tuning after rollout.
- Hidden-label boundary: aggregate seed 78,140 is sealed until strict fresh-local
  promotion at seed 88,010. No benchmark content can enter selection or training.

## Evidence Output

- Fresh substrate, runner-input, overlap, rollout, failure-inventory, selection,
  exact-exposure, training, merge, local, and conditional promotion receipts.
- Negative quota, feasibility, training, or local outcomes are preserved as results.
- A local pass must still beat both parent and replay overall and on the combined
  execute/induct/probe subtotal before one aggregate event may run.

## Decision

Proceed through the model-free design review and one parent rollout only. Training is
not authorized until observed failures are frozen, the replay artifact is copied
self-contained, and a second adversarial review verifies exact three-axis exposure.

Reserved construction/rollout/selection/training/local/aggregate seeds are
`77114/66114/55114/48/88010/78140`. They cannot change after the corresponding model
event.
