# Experiment Log

## 2026-06-24

- Created a fresh standalone experiment directory.
- Selected intervention: Qwen VM-agent ECHO with QLoRA.
- Core idea: use Qwen itself as the recurrent transition. Each turn emits one
  VM edit action, the VM executes it, and the returned observation is appended
  to the next turn's context.
- Main control: action-only QLoRA on the same transcripts, with VM observation
  tokens present as context but masked out of the loss.
- Main treatment: ECHO QLoRA, where VM observation tokens also receive
  cross-entropy loss.
- Large artifacts will be stored in
  `large_artifacts/qwen_vm_agent_echo_qlora/checkpoints/`.

## Iteration Notes

- Implemented the standalone VM-agent script with a seed compiler, textual
  edit-action trajectories, action-only QLoRA, ECHO QLoRA, generation-time VM
  rollouts, and CSV/checkpoint writing.
- Smoke run `smoke_vm_agent_echo` completed end to end with the ECHO arm. It
  validated model loading, LoRA training, weighted token masks, generation,
  parsing, VM execution, run artifacts, and separate large checkpoint storage.
- Smoke diagnostics exposed two issues before scaling: parse-rate accounting
  treated unparsable actions as success when zero actions were parsed, and ECHO
  supervision included the initial observation rather than only action-caused VM
  observations. Patched both and added rollout sample logging.
- Patched smoke run `smoke_vm_agent_echo_v2` completed. Parse-rate accounting
  now correctly reports zero for unparsable generations. The tiny one-epoch
  model mostly copied program-text fragments such as `15:PAD`, so the first
  pilot must test whether a modest amount of action-token exposure is enough to
  establish the edit-action grammar before assessing answer accuracy.
- Pilot run `pilot_vm_agent_echo_s64_a96` trained both action-only and ECHO
  arms successfully, and both learned the action grammar. However, the weak seed
  compiler produced invalid initial programs on every eval split, so the agent
  mostly learned PAD cleanup actions and the oracle K-sweep had no useful
  headroom at the tested K values. Added a controlled `blank` initialization
  mode that starts every prompt from a valid `PUSH 0; END` program, making the
  next pilot a direct test of Qwen as an iterative VM compiler.
- Pilot run `pilot_vm_agent_echo_blank_a128` removed the weak compiler
  confound. Oracle K=8 reached 62.5% to 87.5% across the small eval splits,
  proving the blank VM pathway has real headroom. Action-only learned valid
  action syntax and occasionally compiled correct programs, but overused STOP.
  ECHO improved K=8 accuracy on most splits, including hard composition
  (43.75% vs. 18.75% action-only), while preserving 100% parse rate.
- Patched eval to roll out each example once to the maximum requested K and
  score intermediate snapshots. This keeps K-sweeps from repeating generation
  work.
- Added an explicit STOP rule to the prompt: stop only when the current VM state
  is a valid solution; otherwise edit one slot.
- Pilot run `pilot_vm_agent_echo_blank_exactstop_a128` tested exact-program STOP
  supervision. It made trajectories longer and helped some easier splits, but
  hard composition fell to 18.75% at K=8. Decision: do not use exact-program
  STOP for the main run; scale the blank ECHO setup with answer-based STOP and
  keep action-only as the control.
- Main run `main_vm_agent_echo_blank_a512_stoprule` completed with 512 training
  tasks, 32 examples per eval split, K in {0, 2, 4, 8}, and both action-only and
  ECHO arms. Added a native direct-answer Qwen baseline after the run so the
  report can distinguish VM-loop gains from what the base language model already
  answers without the VM.
