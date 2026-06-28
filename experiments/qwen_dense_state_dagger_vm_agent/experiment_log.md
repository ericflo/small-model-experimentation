# Experiment Log

## 2026-06-24

- Created a fresh standalone experiment directory.
- Selected intervention: dense-state DAgger VM agent.
- Core idea: use Qwen as one recurrent transition. At each turn, structured VM
  state is projected into dense tokens and fed into Qwen with the task prompt.
  Direct action/value heads choose the next edit or `STOP`.
- Main training loop: oracle behavior cloning from blank programs, followed by
  DAgger rounds on states reached by the learned policy.
- Main failure targeted: premature `STOP` and off-oracle rollout drift.
- Large artifacts will be stored under
  `large_artifacts/qwen_dense_state_dagger_vm_agent/checkpoints/`.

### Smoke: `smoke_dense_state_dagger`

- Command scale: 8 train examples, 4 examples per eval split, 1 BC epoch,
  1 DAgger round, K in `{0,1,2}`.
- Outcome: completed end-to-end with Qwen load, LoRA attachment, dense
  `inputs_embeds` state tokens, BC training, DAgger state collection,
  evaluation, CSV/JSON outputs, and LoRA+dense-head checkpoint writes.
- Smoke metric quality is not meaningful at this size; the run is only a
  structural validation.

### Pilot: `pilot_dense_state_dagger_s96_r2`

- Scale: 96 train examples, 16 examples per eval split, 2 BC epochs,
  2 DAgger rounds, K in `{0,2,4,8}`.
- Native Qwen ranged from 25.0% to 62.5% on the small eval splits.
- Oracle teacher reached up to 87.5% on fresh splits and 43.8% on the hard
  split.
- Learned policy stayed weak: best round-1 fresh-standard accuracy was 18.8%,
  best hard accuracy was 6.2%; round 2 regressed after STOP calibration shifted.
- Diagnosis: direct 97-way argument prediction was poor, and rollouts repeated
  no-op structural edits. DAgger collected useful failure states but the action
  interface was too lossy.

### Patch: joint action scoring and prompt constant masks

- Added prompt-derived argument masks, with `0` and calendar constant `7` as
  built-in legal constants.
- Replaced independent kind/slot/value decoding with joint action scoring over
  STOP, OP-slot, and ARG-slot choices.
- Added full action cross-entropy so training optimizes the same complete edit
  chosen at rollout time.
- Typed rollout decoding now blocks no-op edit actions.
- Smoke run `smoke_dense_state_joint_action` completed end-to-end and removed
  the repeated no-op behavior seen in the first pilot.

### Pilot: `pilot_joint_action_s96_r2`

- Same scale and seed as the first pilot.
- BC action accuracy improved from 48.6% to 57.0%; argument accuracy improved
  from 12.6% to 41.4%.
- DAgger round-1 action accuracy improved from 68.7% to 71.6%; DAgger round-2
  action accuracy improved from 56.8% to 65.3%.
- DAgger round-2 rollout success improved from 25.0% to 36.5%.
- Best held-out learned accuracies were still low: 31.2% on fresh paraphrase,
  18.8% on fresh standard, 12.5% on the paired split by blank baseline only,
  and 6.2% on hard composition.
- Main implication: the action interface patch is real, but not enough. The
  main run should keep joint actions and add stronger STOP/value calibration.

### Main: `main_joint_action_calibrated_s256_r2`

- Scale: 256 train tasks, 32 examples per eval split, 3 BC epochs, 2 DAgger
  rounds, K in `{0,2,4,8,12}`.
- Calibration changes from pilot: `stop_loss_weight=8.0`,
  `value_loss_weight=1.0`, `false_stop_weight=4.0`, and
  `dagger_lr=3e-5`.
- BC reached 80.1% action accuracy, 53.9% argument accuracy, and 85.3% STOP
  accuracy.
- DAgger round 2 reached 81.4% action accuracy, 63.4% argument accuracy, and
  87.9% STOP accuracy.
- DAgger collection improved rollout success from 64.5% to 66.0% and reduced
  false STOP states from 56 to 36.
- Final checkpoint best accuracies by split: mixed 40.6%, standard 40.6%,
  paraphrase 53.1%, paired 34.4%, hard 31.2%.
- Native Qwen baselines by split: mixed 21.9%, standard 56.2%, paraphrase
  40.6%, paired 31.2%, hard 25.0%.
- Oracle teacher remained far higher: 81.2% to 96.9% depending on split.
- Reports generated:
  `experiments/qwen_dense_state_dagger_vm_agent/reports/report.md` and
  `experiments/qwen_dense_state_dagger_vm_agent/reports/report.html`.
