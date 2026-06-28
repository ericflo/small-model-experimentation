# Experiment Log

## 2026-06-24

- Created a fresh standalone experiment directory.
- Selected intervention: in-policy VM-ECHO distillation.
- Core idea: a frozen-Qwen compiler proposes typed bytecode programs; the VM
  executes those proposals; integrated observation heads learn validity, final
  value, trace top, trace depth, and answer-correctness for the compiler's own
  candidates while answer-verified repairs are distilled into the compiler.
- Controls: answer-verified repair distillation, matched gold-trace distillation,
  and full-supervised training.
- Large artifacts will be stored in
  `large_artifacts/qwen_inpolicy_vm_echo_distillation/checkpoints/`.

## Iteration Notes

- Implemented the standalone VM core, integrated compiler/ECHO experiment
  script, README, checkpoint manifest, and run log.
- Smoke run `smoke_inpolicy_vm_echo` passed end to end, including Qwen feature
  extraction, seed training, answer-verified control, in-policy VM-ECHO training,
  gold-trace control, full-supervised control, metrics, and checkpoint writing.
- The first smoke exposed a train-log CSV schema issue because ECHO phases write
  additional loss columns. Patched the CSV appender to rewrite with a union
  schema when later rows add columns.
- Smoke run `smoke_inpolicy_vm_echo_v2` verified the fixed schema.
- Pilot `pilot_inpolicy_vm_echo_s96_w010` used 96 seed examples and 256
  candidate prompts. It found answer-verified repair targets for 33.2% of
  candidate prompts. In-policy VM-ECHO improved fresh-paired direct accuracy
  over answer-verified distillation (`21.9%` vs `12.5%`) and hard-composition
  direct accuracy (`17.2%` vs `7.8%`). The ECHO reranker itself was weak, so the
  useful effect appeared through training, not inference-time selection.
- Pilot `pilot_inpolicy_vm_echo_s96_w035` tested a stronger ECHO loss. It was
  worse on fresh-paired direct accuracy (`12.5%`), so the main run will use
  `echo_loss_weight=0.1`.
- Main run `main_inpolicy_vm_echo_s192_w010` completed with 192 seed examples,
  1024 candidate prompts, 1024 full-supervised examples, and 128 examples per
  evaluation split.
- Main candidate surface: 246,784 sampled candidates, 8.3% answer-correct
  candidate rate, 64.3% valid candidate rate, and 44.8% prompt-level oracle
  found rate. Both answer-only and VM-ECHO branches selected 459 verified repair
  targets.
- Main result: VM-ECHO learned VM observations and improved some answer-search
  metrics over answer-verified distillation, including hard-composition search
  `47.7% -> 51.6%`, but it did not produce a broad direct-accuracy gain.
  Fresh-paired direct accuracy stayed at `8.6%`, while the full-supervised
  ceiling reached `76.6%` direct and `95.3%` search.
- Pilot `pilot_inpolicy_vm_echo_s96_w010_r2` tested two in-policy rounds. It did
  not justify a second main run because gains were inconsistent and the
  fresh-paired/hard direct signal weakened.

## Final Artifacts

- Markdown report: `reports/qwen_inpolicy_vm_echo_distillation_report.md`.
- HTML report: `reports/qwen_inpolicy_vm_echo_distillation_report.html`.
- Analysis summary: `analysis/summary.md`.
- Figures: `analysis/figures/`.
- Main run files: `runs/main_inpolicy_vm_echo_s192_w010/`.
- Checkpoints:
  `large_artifacts/qwen_inpolicy_vm_echo_distillation/checkpoints/`.
