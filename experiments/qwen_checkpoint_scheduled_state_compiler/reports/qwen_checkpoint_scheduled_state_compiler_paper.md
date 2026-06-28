# Qwen Checkpoint-Selected Scheduled-State Compiler

## Abstract

This experiment tests whether intermediate-state supervision improves a QLoRA-adapted `Qwen/Qwen3-4B` numeric-copy compiler when the training process saves real validation checkpoints and reports the best saved checkpoint rather than the final optimizer step.

Each prompt describes an initial value and a sequence of add, subtract, and multiply updates modulo 97. The model reads the prompt, copies the initial value, operation sequence, and operation arguments from token-level maps, then executes the copied program with an invisible modular-arithmetic runtime. The primary target is exact execution on length-24 programs, including paired standard/paraphrased renderings of the same latent program.

The main result is positive but specific. Light intermediate-state supervision improves selected-checkpoint length-24 robustness. On a fresh held-out retest of selected checkpoints, the no-state compiler reaches 25.0% paired length-24 exact execution. Constant state loss weight 0.25 reaches 32.8%, and a full-early/light-long schedule reaches 32.2%. The full-early/light-long schedule also gives the best paired state consistency, 70.7%, versus 40.6% for the no-state compiler. In contrast, full early state loss followed by zero long-stage state loss underperforms at 19.1% paired length-24 exact execution.

Checkpoint selection is essential. Every main arm's best selected checkpoint is at step 800, while the final step 900 checkpoint is worse on paired length-24 exact execution.

## Setup

- Base model: `Qwen/Qwen3-4B`
- Loader: `AutoModelForCausalLM`
- Quantization: 4-bit NF4
- Trainable update: LoRA rank 8, alpha 16, dropout 0.05, target `all-linear`
- Trainable LoRA parameters: 16,515,072
- Compiler head width: 768
- Task: modular arithmetic programs modulo 97
- Curriculum: `short:1:4:200`, `medium:1:8:200`, `train:1:12:200`, `long:8:24:300`
- Training examples: 512 paired programs per curriculum stage, rendered as standard/paraphrase pairs
- Training batch size: 4
- Validation size during training: 64 examples per unpaired split and 64 paired programs per paired split
- Fresh retest size: 256 standard length-24 programs, 256 paraphrase length-24 programs, and 256 paired length-24 programs per selected checkpoint
- Selection metric: `paired_len24_executor_accuracy`
- Large checkpoint root: `large_artifacts/qwen_checkpoint_scheduled_state_compiler/checkpoints/`

The compiler predicts token positions for the initial value, ordered operation slots, and ordered argument slots. Values and operations are copied from deterministic token maps. The copied program is executed exactly modulo 97. For state-supervised arms, the differentiable executor also produces a distribution over the modular state after every active operation, and training includes NLL against the true state trajectory according to the schedule being tested.

## Conditions

| Run | State-loss schedule | Purpose |
|---|---:|---|
| `main_no_state_selected_s900` | all stages 0.0 | Matched compiler with no intermediate-state loss. |
| `main_state_w025_selected_s900` | all stages 0.25 | Constant light state supervision. |
| `main_state_l12_off_long_selected_s900` | 1.0 through length-12 training, 0.0 in long stage | Strong early state scaffold, removed for long programs. |
| `main_state_l12_w025_long_selected_s900` | 1.0 through length-12 training, 0.25 in long stage | Strong early state scaffold, light state objective for long programs. |

All main arms use the same base model, LoRA configuration, curriculum, paired training distribution, explicit training seed, and validation splits. Each arm is launched separately so it starts from the same base model rather than inheriting weights from another condition.

## Selected Checkpoints

Each validation point saves an adapter and compiler-head checkpoint. The selected checkpoint is the saved checkpoint with the highest paired length-24 exact execution on the in-training validation split.

| Run | Selected step | Paired L24 exact | Paired state consistency | Standard L24 exact | Paraphrase L24 exact | Final paired L24 exact |
|---|---:|---:|---:|---:|---:|---:|
| No state | 800 | 27.3% | 43.8% | 35.9% | 12.5% | 15.6% |
| State 0.25 constant | 800 | 30.5% | 60.9% | 34.4% | 26.6% | 12.5% |
| State 1.0 then 0.0 | 800 | 22.7% | 32.8% | 34.4% | 4.7% | 18.0% |
| State 1.0 then 0.25 | 800 | 30.5% | 75.0% | 39.1% | 21.9% | 15.6% |

The selected-checkpoint table shows two things. First, light state loss improves the primary validation metric. Second, the final checkpoint is not a reliable estimator of best model quality: all four arms peak before step 900 on paired length-24 exact execution.

## Fresh Retest

The selected checkpoints were retested on fresh length-24 programs generated with a different seed and larger sample size.

| Run | Standard L24 exact | Paraphrase L24 exact | Paired L24 exact | Paired state consistency | Paired both-correct |
|---|---:|---:|---:|---:|---:|
| No state | 32.0% | 19.5% | 25.0% | 40.6% | 17.2% |
| State 0.25 constant | 34.0% | 31.2% | 32.8% | 55.9% | 29.7% |
| State 1.0 then 0.0 | 31.2% | 7.4% | 19.1% | 29.3% | 6.2% |
| State 1.0 then 0.25 | 34.0% | 28.1% | 32.2% | 70.7% | 30.1% |

The fresh retest confirms that the selected-checkpoint improvement is not just a validation-set artifact. Constant light state loss gives the highest fresh paired exact execution, 32.8%. The full-early/light-long schedule is nearly tied on exact execution at 32.2% and gives the strongest representation-level result, with 70.7% paired state consistency.

The full-early/off-long schedule is worse than the no-state control. That result argues against treating strong state loss as a pure early scaffold that should simply be removed for long programs. Some light long-stage state pressure appears useful.

## Interpretation

The experiment supports three claims.

First, checkpoint selection is a real experimental requirement for this training setup. The best saved checkpoints occur at step 800, not the final step. Reporting only final checkpoints would reverse or obscure the main result.

Second, intermediate-state supervision helps when it is light enough. Constant weight 0.25 is the best exact-execution arm on the fresh paired retest. The full-early/light-long schedule is the best state-consistency arm and is nearly tied on exact execution. Both outperform the no-state compiler on fresh paired length-24 exact execution.

Third, strong state supervision is not automatically better. Full early state loss followed by zero long-stage state loss underperforms on paired exact execution, paraphrase exact execution, and paired state consistency. The state target is useful as a bias, but too much or poorly scheduled pressure can steer the compiler toward brittle slot/state behavior.

The absolute accuracy remains modest. The best fresh paired length-24 exact execution is 32.8%, so the compiler is still limited by compounding slot and argument errors. The improvement is nevertheless meaningful because it appears on the hardest paired/paraphrased target and survives a fresh retest.

## Limitations

- Single main seed.
- Synthetic arithmetic task.
- The selected checkpoint is chosen on a validation split that is evaluated repeatedly during training; the fresh retest mitigates but does not replace multi-seed confirmation.
- The method improves a structured compiler interface, not the base model's unconstrained generation.
- The fresh retest evaluates selected checkpoints only, not every saved checkpoint on fresh data.

## Next Experiment

The next experiment should replicate the two best arms with multiple seeds and one training-policy change: reduce or anneal the long-stage learning rate after step 800, or stop long-stage training at the selected checkpoint. The current result shows that useful models are being produced and then degraded. The most direct next question is whether the degradation is an optimizer schedule problem rather than an architecture or objective problem.

## Artifacts

Small files:

- Source: `experiments/qwen_checkpoint_scheduled_state_compiler/src/`
- Runs: `experiments/qwen_checkpoint_scheduled_state_compiler/runs/`
- Analysis: `experiments/qwen_checkpoint_scheduled_state_compiler/analysis/`
- Report: `experiments/qwen_checkpoint_scheduled_state_compiler/reports/qwen_checkpoint_scheduled_state_compiler_paper.md`
- Experiment log: `experiments/qwen_checkpoint_scheduled_state_compiler/reports/qwen_checkpoint_scheduled_state_compiler_experiment_log.md`
- Manifest: `experiments/qwen_checkpoint_scheduled_state_compiler/checkpoint_manifest.csv`

Large files:

- Checkpoints: `large_artifacts/qwen_checkpoint_scheduled_state_compiler/checkpoints/`

