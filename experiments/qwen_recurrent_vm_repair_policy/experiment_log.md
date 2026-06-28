# Experiment Log

## 2026-06-24

- Created a fresh standalone experiment directory.
- Selected intervention: recurrent VM repair policy with learned STOP.
- Core idea: train a Qwen-attached policy so that one forward pass proposes one
  edit to the current VM program, then the VM executes the edited program and
  feeds the new execution state back into the same policy.
- Training plan:
  - train a seed compiler from frozen Qwen hidden states;
  - collect oracle edit trajectories from seed programs to correct programs;
  - train a teacher-forced edit/STOP policy;
  - run the learned policy to collect off-trajectory states;
  - retrain with DAgger-style mixed teacher/off-policy states;
  - evaluate K-step curves and learned STOP behavior.
- Large artifacts will be stored in
  `large_artifacts/qwen_recurrent_vm_repair_policy/checkpoints/`.

## Iteration Notes

- Implemented the standalone VM core and recurrent repair experiment script.
- Smoke run `smoke_recurrent_vm_repair` completed end to end.
- Smoke diagnostics exposed two implementation issues before scaling: STOP
  labels were too sparse, and quick validation was reading the base row instead
  of the learned K-step row. Patched both.
- Smoke run `smoke_recurrent_vm_repair_v2` verified STOP-state augmentation,
  but the flat action head was too sparse.
- Replaced the flat edit-action head with factorized `kind`, `slot`, `opcode`,
  and `argument` heads.
- Smoke run `smoke_recurrent_vm_repair_factorized` verified the factorized
  policy. The oracle edit curve improved strongly with K, confirming that the
  recurrent setup has headroom, but the tiny learned policy was not yet useful.
- Pilot run `pilot_recurrent_vm_repair_s96_c256` completed. The oracle edit
  loop reached 95-100% accuracy by K=8-16 on validation and paired/hard
  generalization splits, but the learned policy mainly failed through false
  STOP decisions on incorrect programs.
- Patched the policy rollout path with optional conservative STOP margining,
  typed VM action masking, and component-level policy diagnostics. This keeps
  the recurrent VM hypothesis unchanged while targeting the observed learned
  policy failure mode.
- Pilot run `pilot_recurrent_vm_repair_masked_s96_c256` showed that typed
  masking and STOP margining increased imitation diagnostics but harmed final
  accuracy. Component diagnostics identified argument prediction as the weak
  point: kind, slot, and opcode accuracy were high, while argument accuracy
  remained low.
- Patched the repair policy to cross-attend to frozen Qwen token hidden states,
  rather than relying only on a pooled prompt vector. This directly targets
  numeric/detail recall during edit prediction.
- Pilot run `pilot_recurrent_vm_repair_crossattn_s96_c256` improved learned
  rollout behavior relative to the pooled-policy pilot: DAgger rollout success
  rose from roughly 25% to 30%, and the best validation learned K-curve rose to
  about 20%. The policy remains far below the oracle, so the next patch adds
  multiple DAgger rounds to attack off-policy distribution shift.
- Pilot run `pilot_recurrent_vm_repair_crossattn_dagger3_s96_c256` made the
  strongest learned-policy improvement so far. The third DAgger trajectory
  reached roughly 49% rollout success, component argument accuracy rose to
  roughly 70%, and learned validation accuracy reached roughly 27% at K=8 while
  the oracle remained near 100%. Main run will use cross-attention and three
  DAgger rounds.
- Main run `main_recurrent_vm_repair_crossattn_dagger3_s192_c1024` completed.
  The final learned recurrent policy improved all splits over the seed compiler:
  validation 10.9% -> 34.4%, fresh standard 10.9% -> 35.9%, fresh paraphrase
  15.6% -> 39.8%, fresh paired 9.4% -> 32.0%, and hard composition 14.8% ->
  31.2% with learned STOP. DAgger rollout success rose from 32.4% to 59.3%
  across three rounds, and argument action accuracy rose to 82.3%. The oracle
  repair ceiling remained 95.3-100.0%, so the learned policy is useful but still
  far from the reachable repair ceiling.
- Generated aggregate CSVs, five figures, a standalone Markdown report, and a
  standalone HTML report in `analysis/` and `reports/`.
