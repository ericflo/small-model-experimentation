# Experiment Log

## 2026-06-24

- Created a standalone verifier-MDP experiment for Qwen3.5-4B probe synthesis and process control.
- Implemented the verifier-MDP harness as a standalone experiment package.
- Added a 96-case probe bank per task and a deployable `mined8` action source that selects eight probes by target-independent candidate-bucket statistics.
- Added full-pool non-model baselines to measure the value of scanning the whole probe bank.
- Smoke-built a 24-query-pool dataset and confirmed the oracle labels were not position-collapsed.
- The first full 96-query build was too slow due repeated operator-output recomputation. Patched the environment to compute operator output tables once per state.
- Built the full dataset: 300 train records, 160 eval records, 619 informative train states, 334 informative eval states.
- Ran non-model baselines:
  - `max_split_random8`, `oracle_random8`
  - `random_mined8`, `max_split_mined8`, `oracle_mined8`
  - `fullpool_max_split`, `fullpool_oracle`
- Evaluated base Qwen on mined-eight prompts.
- Trained SFT for 260 optimizer steps from Qwen3.5-4B.
- Evaluated SFT on the full mined-eight eval ladder.
- Trained process-DPO for 120 optimizer steps from SFT.
- Evaluated process-DPO on the full mined-eight eval ladder.
- Trained process-GRPO for 80 optimizer steps from SFT.
- Evaluated process-GRPO on the full mined-eight eval ladder.
- Evaluated SFT with scrambled displayed features.

## Main Readout

- Random-eight max-split reached 42.5% hidden-all at budget 3.
- Mined-eight max-split reached 49.4%, showing the deployable mined action source improves the baseline action set.
- Mined-eight oracle reached 61.3%, while full-pool oracle reached 86.9%, showing major remaining probe-generation headroom.
- Base Qwen reached 47.5%; SFT improved to 51.3% and dropped to 47.5% under feature scrambling.
- DPO and GRPO did not improve on SFT in this run.

## Interpretation

The main line-2 bottleneck is now sharper: the verifier can expose much better observations than the current mined-eight action source reliably surfaces, and Qwen can learn a modest selector improvement once those probes are displayed. The next target should be trainable probe generation or ranking over a larger candidate bank, not more optimization over the same eight actions.
