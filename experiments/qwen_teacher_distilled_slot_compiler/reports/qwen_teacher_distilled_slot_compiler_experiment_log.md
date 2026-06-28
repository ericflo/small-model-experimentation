# Qwen Teacher-Distilled Slot Compiler Experiment Log

## Objective

Test whether oracle slot-teacher losses improve a Qwen numeric-copy compiler's long-chain exact execution and paired paraphrase consistency.

## Experiment Question

Can a text compiler bind executable slots more reliably when hard trace supervision is augmented with soft local oracle slot targets and slot-representation matching?

## Planned Conditions

1. Tiny-model smoke test for teacher losses and checkpoint snapshots.
2. Qwen smoke test for 4-bit QLoRA teacher training.
3. Matched light-state control with no teacher losses.
4. Teacher-distilled light-state compiler with soft position and representation losses.
5. Low-weight soft-position-only teacher arm.
6. Fresh selected-checkpoint retest on length-24 programs.

## Primary Selection Rule

Select the saved checkpoint with the highest `paired_len24_executor_accuracy`.

## Primary Metrics

- `paired_len24_executor_accuracy`
- `fresh_paired_len24_executor_accuracy`
- `fresh_paired_len24_compiler_pair_state_consistency`
- `arg_accuracy`
- `arg_pos_accuracy`
- `program_exact`

## Artifact Policy

Lightweight outputs stay in:

```text
experiments/qwen_teacher_distilled_slot_compiler/runs/
experiments/qwen_teacher_distilled_slot_compiler/analysis/
experiments/qwen_teacher_distilled_slot_compiler/reports/
```

Large adapters and head checkpoints stay in:

```text
large_artifacts/qwen_teacher_distilled_slot_compiler/checkpoints/
```

## Log

### 2026-06-22

- Created standalone experiment directory.
- Forked the checkpoint-selected Qwen numeric-copy compiler harness into a teacher-distilled slot harness.
- Added soft oracle position teacher losses.
- Added oracle slot-representation matching losses.
- Added standalone README and experiment log.
- Ran `smoke_tiny_teacher` with a tiny frozen causal LM. The run completed, saved validation checkpoints, generated analysis outputs, and confirmed that teacher position and representation losses are present in the train log.
- Ran `smoke_qwen3_4b_teacher` with `Qwen/Qwen3-4B`, 4-bit QLoRA, state loss weight 0.25, teacher position weight 0.1, and teacher representation weight 0.05. The run completed, saved real adapter checkpoints, and showed teacher losses in a reasonable scale relative to the total loss.
- Ran `main_control_light_state_s900`, the matched light-state control with no teacher losses. The selected checkpoint is step 800 with 30.5% paired length-24 exact execution, 67.2% paired state consistency, 37.5% standard length-24 exact execution, and 25.0% paraphrase length-24 exact execution. The final checkpoint fell to 23.4% paired length-24.
- Ran `main_teacher_slot_distill_s900` with teacher position weight 0.1 and teacher representation weight 0.05. The selected checkpoint is step 800 with 28.1% paired length-24 exact execution, 59.4% paired state consistency, 37.5% standard length-24 exact execution, and 21.9% paraphrase length-24 exact execution. The final checkpoint fell to 22.7% paired length-24.
- Ran a fresh selected-checkpoint retest for the matched control and heavier teacher arms at length 24 with `eval_size=256` and `eval_seed=92001`. The control scored 27.1% paired exact execution and 72.7% paired state consistency. The heavier teacher arm scored 27.9% paired exact execution and 55.5% paired state consistency. The answer accuracy difference was tiny, but the teacher arm lost substantial paired compiler consistency.
- Added and ran `main_teacher_softpos_low_s900` after inspecting the heavier teacher result. This arm removes representation matching and uses only a low soft-position teacher weight of 0.03. It briefly improved mid-training length-24 transfer at step 600: 13.3% paired length-24 exact execution versus 1.6% for the matched control at the same step. The selected checkpoint was still step 800 and underperformed the control: 23.4% paired length-24 exact execution, 32.8% paired state consistency, 29.7% standard length-24 exact execution, and 15.6% paraphrase length-24 exact execution.
- Ran the same fresh selected-checkpoint retest for `main_teacher_softpos_low_s900`. It scored 18.4% paired length-24 exact execution and 41.0% paired state consistency. This confirms that low-weight soft slot supervision did not improve the selected checkpoint and damaged paraphrase robustness.

## Result

Oracle slot-teacher losses did not beat the matched no-teacher light-state compiler. The best fresh paired length-24 exact execution was effectively tied between the control and the heavier teacher arm, 27.1% versus 27.9%, while paired state consistency dropped from 72.7% to 55.5%. The low-weight soft-position-only variant was worse at 18.4% paired exact execution and 41.0% paired state consistency.

The most useful observation is negative: the compiler already localizes symbols well enough that extra oracle slot imitation is not the current bottleneck. The next experiment should move the training signal closer to execution correctness or search, rather than adding more slot-position supervision.
