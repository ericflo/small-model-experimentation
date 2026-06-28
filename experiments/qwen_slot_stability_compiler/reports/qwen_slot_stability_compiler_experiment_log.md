# Qwen Slot-Stability Compiler Experiment Log

## Objective

Test whether paired paraphrase consistency improves a Qwen numeric-copy compiler's ordered slot binding across wording variants and longer arithmetic chains.

## Experiment Question

Can a QLoRA-adapted model learn a more stable ordered compiler interface when each training batch includes paired renderings of the same underlying program and the copied slot distributions are explicitly regularized to agree?

## Planned Conditions

1. Smoke tests for the paired dataset and stability loss.
2. Main QLoRA numeric-copy compiler with paired slot-stability loss.
3. Matched QLoRA numeric-copy compiler without the stability loss, trained on the same paired data distribution.
4. Optional final-answer controls if the main comparison is ambiguous.

## Primary Metrics

- `executor_accuracy`: exact execution accuracy after copying and running the compiled program.
- `program_exact`: exact full compiled-program correctness.
- `executor_pair_answer_consistency`: whether paired standard/paraphrase renderings execute to the same predicted answer.
- `executor_pair_both_correct`: whether both renderings in a pair are exactly correct.
- `compiler_pair_program_consistency`: whether paired renderings compile to the same init/op/arg program.
- `init_accuracy`, `op_accuracy`, `arg_accuracy`: copied symbol correctness.
- `op_pos_accuracy`, `arg_pos_accuracy`: ordered slot localization.
- Standard and paraphrase splits are evaluated separately at multiple chain lengths.

## Artifact Policy

Lightweight outputs stay in `experiments/qwen_slot_stability_compiler/runs/`.

Large adapters and head checkpoints stay in `large_artifacts/qwen_slot_stability_compiler/checkpoints/`.

## Log

### 2026-06-21

- Created standalone experiment directory and external checkpoint path.
- Implemented paired program rendering, paired batch sampling, and slot-stability losses.
- Found and fixed a NaN gradient in the stability KL. The issue came from `F.kl_div` taking a target `log(0)` path for masked lexical classes. Replaced it with an explicit probability-times-log-ratio symmetric KL.
- Added paired evaluation splits so standard/paraphrase renderings of the same latent program can be scored for answer and program consistency.
- Ran tiny-model smoke tests and a Qwen3-4B QLoRA smoke test. Both completed after the KL fix.
- Ran a 300-step Qwen3-4B QLoRA stability pilot and a matched no-stability pilot. The stability arm showed a large paired length-12 consistency advantage at step 300, but the result needed a longer matched check.
- Added `--train_seed` so the control arm can share the stability arm's compiler initialization and batch sampling seed while keeping the same generated datasets.
- Ran final 600-step arms:
  - `main_qwen3_4b_qlora_slot_stability_mixed_l12_s600`
  - `control_qwen3_4b_qlora_paired_no_stability_mixed_l12_s600`
  - `control_qwen3_4b_qlora_answer_only_mixed_l12_s600`
- Final result: trace-supervised numeric-copy compilation works; final-answer-only QLoRA stays at chance. The paired stability loss is mixed rather than decisive. It improves standard L12, paraphrase L24, and paired L24 exact execution, but it trails or ties the matched no-stability compiler on other splits and does not reliably improve paired consistency.
