# Experiment Log

## Setup

- Created a standalone diagnostic directory at `/workspace/experiments/qwen35_4b_deployable_information_ceiling_sweep`.
- Packaged 160 evaluation records covering library sizes 64, 128, 256, and 512 across `pair_affine_mod` and `pair_compare_gate`.
- No model training is used.

## Hypothesis

If the low-information regime is information-limited, then more active probes or more initial observations should raise the deployable greedy uniform-posterior policy. If the target-aware oracle remains far above the deployable policy, that residual gap should be treated as target-knowledge headroom rather than directly recoverable deployable performance.

## Planned Sweep

- Policies:
  - `greedy_uniform_split`: choose the full-pool probe with minimum expected survivors under a uniform posterior over verifier-surviving candidates.
  - `target_aware_oracle`: choose the full-pool probe that minimizes survivors for the actual hidden target.
- Visible observations: 4, 8, 12, and 16 total.
- Active probe budgets: 0 through 10.

## Results

- Completed the full sweep over 160 records.
- `pair_affine_mod` is solved by the deployable greedy policy once either budget or visible observations are modestly increased.
- `pair_compare_gate` is not intrinsically stuck:
  - 4 visible, budget 3: greedy 3.8%, oracle 73.8%.
  - 4 visible, budget 10: greedy 86.2%, oracle 98.8%.
  - 8 visible, budget 3: greedy 46.2%, oracle 93.8%.
  - 12 visible, budget 3: greedy 73.8%, oracle 95.0%.
  - 16 visible, budget 3: greedy 91.2%, oracle 97.5%.
  - 16 visible, budget 10: greedy 98.8%, oracle 98.8%.
- Interpretation: the low-information failure at budget 3 is primarily an observation-budget problem. The target-aware oracle gap at four visible cases is not directly trainable selector headroom, but the deployable policy catches up when given enough observations.
