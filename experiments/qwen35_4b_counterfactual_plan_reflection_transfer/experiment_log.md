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
