# Qwen Register Trace Refiner

## Abstract

This experiment tests a latent-program repair loop for a Qwen3-4B register compiler. A fixed compiler reads a text prompt and emits a register program for modular arithmetic. A deterministic runtime executes the program. The new component enumerates local register-program edits and trains a verifier to select a better candidate from each local repair set without seeing the target answer or target trace at evaluation time.

On fresh length-24 programs, the learned guarded refiner improves standard-prompt execution accuracy from 23.4% to 26.6%, with an oracle upper bound of 37.1% inside the same candidate set. It does not improve paraphrase accuracy, where the oracle itself reaches only 7.0%. On paired standard/paraphrase evaluation, the refiner improves 12.3% to 12.9%, with an 18.6% oracle ceiling. The result is positive but narrow: local repair helps, but the correct program is often outside the top-3/two-edit neighborhood, and learned candidate selection remains hard.

## Method

Each example is a length-24 chain of operations modulo 97. The text prompt describes an initial value and a sequence of add, subtract, and multiply updates. A fixed Qwen3-4B register compiler outputs:

- one initial-value register,
- one operation register per step,
- one argument register per step.

The runtime executes the argmax register program exactly. The refiner then constructs a candidate set around that program:

- keep the base program,
- edit the initial value,
- edit one operation,
- edit one argument,
- edit one operation and argument at the same step,
- edit two argument slots.

Each slot uses the top-3 compiler alternatives. For length 24 this creates 1,299 candidates per example.

Candidates are featurized by compiler log-probabilities, edit structure, soft-runtime trace likelihoods, operation/argument statistics, and the candidate execution trace. A small transformer scores the candidate trace plus summary features. Training labels are produced offline by exact execution against the target trace. At evaluation time, the verifier sees only candidate features and traces, not the target answer or target states.

A guarded selector is tuned on validation: use the learned candidate only when it beats the base candidate by a score margin; otherwise retain the base program. The selected validation guard threshold was 0.25.

## Results

Primary run: `main_register_trace_refiner_s512`.

| split | base | learned/guarded | oracle |
|---|---:|---:|---:|
| train_len24 | 14.8% | 16.6% | 21.9% |
| val_len24 | 15.6% | 17.2% | 20.3% |
| fresh_standard_len24 | 23.4% | 26.6% | 37.1% |
| fresh_paraphrase_len24 | 4.7% | 4.7% | 7.0% |
| fresh_paired_len24 | 12.3% | 12.9% | 18.6% |

Fresh paired details:

| metric | base | learned/guarded | oracle |
|---|---:|---:|---:|
| executor accuracy | 12.3% | 12.9% | 18.6% |
| program exact | 12.1% | 12.7% | 18.4% |
| state prefix fraction | 79.5% | 79.7% | 80.7% |
| pair both correct | 1.6% | 1.6% | 1.6% |
| pair state consistency | 1.6% | 1.6% | 1.6% |

Candidate-set profile:

| split | candidates/example | positive candidates/example | oracle found |
|---|---:|---:|---:|
| train_len24 | 1299.0 | 0.45 | 21.9% |
| val_len24 | 1299.0 | 0.36 | 20.3% |
| fresh_standard_len24 | 1299.0 | 0.68 | 37.1% |
| fresh_paraphrase_len24 | 1299.0 | 0.15 | 7.0% |
| fresh_paired_len24 | 1299.0 | 0.47 | 18.6% |

Figures:

- `../analysis/figures/executor_accuracy_by_split.png`
- `../analysis/figures/oracle_gap_recovered.png`
- `../analysis/figures/candidate_set_profile.png`

## Discussion

The experiment supports three conclusions.

First, local repair is a real lever. On fresh standard length-24 programs, the learned refiner recovers 22.9% of the available oracle gap and improves exact execution by 3.2 percentage points.

Second, local top-k repair is not enough for robust paraphrase behavior. The oracle ceiling is only 7.0% on paraphrase examples, which means the correct program usually is not present in the searched neighborhood.

Third, verifier selection remains difficult even when the correct repair is present. The standard split has a 37.1% oracle ceiling, but the learned guarded selector reaches 26.6%. Better candidate scoring, broader search, or iterative edit policies are needed before this becomes a large test-time-compute gain.

The most direct next step is to widen the repair distribution without exploding candidates: use a learned proposal policy over suspect slots, add three-edit candidates only around low-confidence prefixes, and train the verifier/editor jointly on base-wrong repairable cases.

