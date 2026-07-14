# Counterfactual Plan Reflection Transfer Experiment Log

## 2026-07-14 — discovery and scaffold

- Re-read the workspace paper's methods, intervention figures, counterfactual
  reflection section, method ablations, formal sparse-frame definition, and
  multi-token extensions.
- Discovery rejected a generic within-thought correctness-coordinate experiment as a
  duplicate of the terminal J-value line.
- Selected the paper's distinct training claim: loss on a counterfactual reflection
  branch may shape behavior on an untrained action branch.
- Named `qwen35_4b_bank_the_thoughts` as closest near-duplicate and the concurrently
  active on-policy prefix-repair experiment as a non-duplicate neighboring line.
- Created a fresh experiment and a model-free construction smoke. No model,
  tokenizer, GPU, adapter, benchmark, claim allocation, or hidden result was touched.

## 2026-07-14 — adversarial HOLD and construction repair

- Independent review of commit `3eae868d182f4a02848f6415d8eaafdb87465336`
  returned HOLD: proposed string geometry was impossible, state explosion escaped,
  and 14/30 smoke plans were not uniquely identified by visible examples.
- Kept tokenizer/model/GPU/training/Jacobian work sealed.
- Expanded the string and list primitive libraries, removed a list symmetry that had
  insufficient unique slot support, and replaced rejection allocation with exhaustive
  exact-depth catalog enumeration plus deterministic split allocation.
- Required each target to have one global depth-three spelling and exactly one
  depth-three program (with no depth-zero/two alternative) on its seven visible examples.
- Full configured CPU construction now produces 504/504 tasks with zero collisions
  and complete operation-position support. State-exploding candidates are rejected.
- Shuffled arms now preserve task truth in immutable fields and place the donor only
  in explicit supervision fields; every donor is wrong on the recipient's visible or
  query behavior.
- Added a Python audit-hook firewall that denies benchmark-root opens and directory
  enumeration. Remaining review defects concern mechanism isolation, exact rendering,
  training parity, retention, gates, and result-separated Jacobian work.
- Published construction repair commit `83a55cf3887dd681790aeee1e8d1070cea4b8d15`;
  exact Validate Repository run `29345252095` and Publish Research Site run
  `29345252135` both completed successfully.

## 2026-07-14 — design-contract implementation

- Added a 72-task calibration split and 48 real, visible-identifiable exact-depth-1/2
  retention tasks. Full model-free geometry is now 576 depth-three plus 48 retention.
- Added a correct non-reflective auxiliary-label arm. Its target is identical to the
  correct reflection arm; its only semantic change is the first instruction sentence,
  and exact rendered prompt-token equality is a prerequisite gate.
- Restricted correct/shuffled donor permutations within each family and 18-row
  optimizer group. Each group has six tasks per family and identical correct/shuffled
  target multisets, allowing exact per-step token exposure checks.
- Froze the QLoRA recipe, target-only Qwen thinking-channel masks, two training seeds,
  final-only checkpoint selection, vLLM generation geometry, paired-bootstrap effect
  thresholds, per-family breadth gates, positive-control sanity, retention margins,
  and no-seed-selection staging.
- Implemented immutable record construction, token/mask encoding, parity receipts, a
  guarded trainer, and executable decision-gate analysis. None was run with a real
  tokenizer or model because adversarial HOLD remains in force.
- Removed the conditional J stage from this result-bearing experiment. A behavioral
  replication can only license a separate experiment with fresh J fit, confirmation,
  and causal evidence.
