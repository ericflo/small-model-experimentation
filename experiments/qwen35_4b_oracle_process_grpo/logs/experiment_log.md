# Experiment Log

## 2026-06-24

- Created a standalone verifier-MDP experiment for Qwen3.5-4B process control.
- Implemented a typed `list[int] -> int` operator-pair environment with two output regimes:
  - `pair_affine_mod`: higher-information numeric outputs.
  - `pair_compare_gate`: lower-information symbolic outputs.
- Implemented deterministic candidate filtering, same-budget oracle probe selection, max-split heuristic, and deployable model-action evaluation.
- Smoke-built a tiny dataset and found zero-signal post-solve states. Updated dataset generation to omit states where all probe actions have identical reward.
- Built the full dataset: 480 train records, 128 eval records, 1138 informative train states, 299 informative eval states.
- Ran non-model baselines: random, max-split, oracle.
- Ran Qwen base action policy.
- Trained SFT process policy for 360 optimizer steps.
- Evaluated SFT adapter on the full held-out ladder.
- Trained process-DPO for 160 optimizer steps from the SFT adapter.
- Evaluated process-DPO.
- Trained shuffled-reward process-DPO control for 160 optimizer steps from the same SFT adapter.
- Evaluated shuffled-reward control.
- Trained GRPO for 120 optimizer steps from the SFT adapter.
- Evaluated GRPO.
- Evaluated feature-scrambled process-DPO to test dependence on candidate-bucket summaries.

