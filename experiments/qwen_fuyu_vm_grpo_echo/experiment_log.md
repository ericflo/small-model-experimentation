# Experiment Log

## 2026-06-24

- Created a fresh standalone experiment directory.
- Selected intervention: Fuyu-style whole-network VM loop with trajectory-level
  GRPO-style training and structured ECHO observation prediction.
- Interface commitment: no natural-language action-token generation. Qwen
  receives prompt tokens plus dense VM-state tokens through `inputs_embeds` and
  emits only structured head predictions.
- Large artifacts will be stored under
  `large_artifacts/qwen_fuyu_vm_grpo_echo/checkpoints/`.
- Smoke runs validated the end-to-end dense-state loop, checkpointing, and CSV
  outputs.
- The first terminal-only GRPO smoke showed the expected sparse-reward failure:
  zero successful sampled rollouts and high false-STOP rate. I stopped the
  larger pilot before training on that weaker objective.
- Updated the experiment to use dense process reward: each sampled VM edit is
  scored by whether it preserves reachable verified completion within the
  remaining rollout budget, with explicit penalties for false STOP and
  destroying reachability.
- Added a DAgger-style teacher anchor during RL rounds, an ECHO ablation knob
  (`--echo_loss_weight 0`), and a shuffled-reward control
  (`--shuffle_rollout_rewards 1`).
- Pilot `pilot_shaped_echo_s32_20260624`: shaped GRPO produced usable process
  signal (9.4% sampled rollout success, 56.3% reachable-after edits), but one
  update reduced mean learned accuracy from 10.0% to 7.1% and collapsed K=8
  measured accuracy to 0.0%.
- Added a process-preference arm (`--algorithm process_dpo`) after the shaped
  GRPO gate failure. It trains on on-policy states labeled by repair/oracle
  actions with sampled bad actions as pairwise negatives.
- Pilot `pilot_process_dpo_s32_20260624`: process preference collected 226
  on-policy states, found repair labels for 97, and reached 80.2% pairwise rank
  accuracy, but deployment still regressed (learned mean 9.2% -> 7.9%, forced
  mean 9.2% -> 5.8%).
- Wrote standalone reports with charts:
  `reports/qwen_fuyu_vm_grpo_echo_report.md` and
  `reports/qwen_fuyu_vm_grpo_echo_report.html`.
