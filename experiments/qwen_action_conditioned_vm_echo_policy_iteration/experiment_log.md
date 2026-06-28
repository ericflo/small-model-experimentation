# Experiment Log

## 2026-06-24

- Created a fresh standalone experiment directory.
- Selected intervention: action-conditioned VM-ECHO policy iteration.
- Core idea: generate candidate bytecode programs from a frozen-Qwen compiler,
  execute those candidates in the VM, train a consequence model on candidate
  observations, then distill consequence-selected candidates back into the
  compiler.
- Large artifacts will be stored in
  `large_artifacts/qwen_action_conditioned_vm_echo_policy_iteration/checkpoints/`.

## Implementation Notes

- Built `src/qwen_action_conditioned_vm_echo_policy_iteration_experiment.py`.
- Built `src/analyze_qwen_action_conditioned_vm_echo_policy_iteration.py`.
- Reused the typed bytecode VM core as local experiment source in
  `src/typed_bytecode_core.py`, with checkpoint paths redirected to the
  experiment-specific large-artifact directory.
- Added checkpoint tracking in `checkpoint_manifest.csv`.
- Generated both Markdown and HTML reports under `reports/`.

## Iteration Record

- `smoke_action_vm_echo`: verified that Qwen feature extraction, candidate
  generation, consequence training, target selection, distillation, metrics,
  and checkpoint writing all run end to end.
- `pilot_action_vm_echo_s96`: exposed two bookkeeping issues: inconsistent
  target-selection CSV schemas and quick validation using the wrong prompt
  feature set.
- Patched the target-selection schema, fixed quick-validation prompt features,
  and restored the best quick-validation consequence checkpoint before target
  selection.
- `pilot_action_vm_echo_s96_v2`: unfiltered learned target selection chose all
  available candidates and had low target precision.
- `pilot_action_vm_echo_s96_thr070`: tested a 0.7 learned-score threshold. It
  selected fewer targets with higher known correctness and improved pilot direct
  accuracy, so the threshold was used for the main run.
- `main_action_vm_echo_s192_thr070`: completed the scaled run with 192 seed
  examples, 1024 candidate-training prompts, 1024 full-supervised examples,
  and 128-example validation/fresh/hard splits.

## Main Result

- Validation selector gap: base top-1 `10.2%`, learned selector `10.9%`,
  candidate oracle `36.7%`.
- Learned-policy distillation improved several downstream splits despite weak
  learned selection: fresh-paired direct `18.0% -> 22.7%`, fresh-paired answer
  search `51.6% -> 64.1%`, hard-composition answer search `35.2% -> 52.3%`.
- Fully supervised training remained the practical ceiling: fresh-paired direct
  `82.0%`, fresh-paired answer search `94.5%`, hard-composition direct `51.6%`.
- Main conclusion: the candidate-conditioned learning signal is real but still
  too weak. The next experiment should target selector precision with pairwise
  preference training, harder negatives, or direct answer-representation access.

## Final Artifacts

- Markdown report:
  `reports/qwen_action_conditioned_vm_echo_policy_iteration_report.md`.
- HTML report:
  `reports/qwen_action_conditioned_vm_echo_policy_iteration_report.html`.
- Analysis summary: `analysis/summary.md`.
- Figures: `analysis/figures/`.
- Checkpoints:
  `large_artifacts/qwen_action_conditioned_vm_echo_policy_iteration/checkpoints/`.
