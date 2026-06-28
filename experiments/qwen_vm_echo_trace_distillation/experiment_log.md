# Experiment Log

## 2026-06-24

- Created a fresh standalone experiment directory for VM-ECHO trace
  distillation.
- Chosen intervention: add VM observation prediction losses to the frozen-Qwen
  typed-bytecode compiler head.
- Initial design: compare `baseline` and `vm_echo` arms from the same randomly
  initialized compiler head, with identical frozen Qwen feature caches,
  decoder architecture, candidate search, and training split sizes.
- Large artifacts will be stored in
  `large_artifacts/qwen_vm_echo_trace_distillation/checkpoints/`.

### Smoke and Pilot Iteration

- Smoke run `smoke_vm_echo` verified end-to-end execution, but showed the
  first trace mask over-weighted padded post-`END` slots.
- Updated VM observation labels to supervise only active slots through `END`.
- Smoke run `smoke_vm_echo_masked` verified the corrected active-slot mask.
- Pilot run `pilot_vm_echo_s96` with `echo_weight=0.35` learned observation
  signals but damaged candidate search.
- Pilot runs `pilot_vm_echo_s96_w010` and `pilot_vm_echo_s96_w003` swept lower
  weights. `0.03` preserved the candidate search surface best and was selected
  for the main comparison.

### Main Run

- Main run: `main_vm_echo_s192_w003`.
- Main setting: `echo_weight=0.03`, two expert rounds, matched baseline and
  VM-ECHO arms from the same compiler initialization.
- Key result: VM-ECHO learned trace observations but did not produce a broad
  direct-accuracy improvement.
- Full-supervised fresh paired: direct accuracy tied at 84.4%; search accuracy
  improved from 93.0% to 96.1%; trace-top observation accuracy improved from
  0.7% to 43.1%.
- Expert-round-2 hard composition: search accuracy improved from 46.9% to
  51.6%; direct accuracy tied at 8.6%.
