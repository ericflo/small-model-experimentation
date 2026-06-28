# Experiment Log

## Setup

- Created a standalone typed-bytecode expert-iteration experiment directory.
- Required artifact layout: `src/`, `runs/`, `analysis/`, `analysis/figures/`,
  `reports/`, `checkpoint_manifest.csv`, and large checkpoints under
  `large_artifacts/qwen_typed_bytecode_expert_iteration/checkpoints/`.

## Iterations

### Smoke: `smoke_typed_bytecode_ei`

- Purpose: validate the end-to-end script, logging, checkpoint layout, and
  typed VM execution.
- Result: unconstrained slot decoding emitted invalid bytecode, so candidate
  search had no useful foothold.
- Decision: add stack-depth-constrained bytecode decoding instead of training
  longer.

### Smoke: `smoke_typed_bytecode_ei_v2`

- Purpose: retest after typed/stack-constrained decoding.
- Result: direct programs became valid often enough to measure, but the model
  was too small and undertrained for answer-verified search to collect useful
  targets.
- Decision: move to a real pilot with more warm-start supervision.

### Pilot: `pilot_typed_bytecode_ei_s128`

- Purpose: test seed supervision, answer-verified expert iteration, and dense
  full supervision on a modest run.
- Result: full supervision reached about 60% fresh paired direct accuracy and
  about 80% search accuracy; expert iteration was positive but weak.
- Decision: isolate the dense supervised ceiling with larger trace coverage.

### Pilot: `pilot_supervised_ceiling_s2048`

- Purpose: determine whether the typed-bytecode target is learnable with enough
  dense traces.
- Result: full supervision reached about 93-95% direct accuracy on fresh splits
  and about 98% with local search.
- Decision: the bytecode ABI is learnable; focus the final run on whether
  answer-verified expert iteration compounds.

### Pilot: `pilot_expert_iteration_r3_s256`

- Purpose: run a larger answer-verified self-training loop with three rounds.
- Result: fresh paired direct accuracy improved from 32.8% to 56.2%, while
  search improved from 66.8% to 79.7%.
- Decision: run a larger main configuration combining four expert rounds and a
  dense supervised ceiling under the same evaluation splits.

### Main: `main_typed_bytecode_ei_s384_u4096`

- Purpose: final standalone measurement.
- Result: fresh paired direct accuracy moved from 61.5% under the seed compiler
  to 73.0% after four expert-iteration rounds; dense full supervision reached
  99.6%. Hard-composition direct accuracy moved from 45.9% to 53.9% under
  expert iteration and reached 80.7% under full supervision.
- Decision: write up as a positive result for typed-bytecode supervision and a
  partial result for answer-only expert iteration. The next bottleneck is
  process verification or prefix-level search, not the bytecode ABI itself.

### Frozen-Qwen Pilot: `qwen_head_pilot_s384_u2048`

- Purpose: attach the typed-bytecode compiler head to frozen `Qwen/Qwen3-4B`
  hidden states rather than only using the compact controlled compiler.
- Result: fresh paired direct accuracy improved from 17.6% under the seed
  Qwen-head compiler to 50.4% after three expert-iteration rounds. Dense
  full-supervised Qwen-head training reached 94.5% fresh paired direct accuracy.
- Decision: include this as a Qwen-attached pilot in the standalone paper. The
  next step should train Qwen adapters or improve process verification; the
  frozen-head result shows signal but still leaves a large gap to dense traces.
