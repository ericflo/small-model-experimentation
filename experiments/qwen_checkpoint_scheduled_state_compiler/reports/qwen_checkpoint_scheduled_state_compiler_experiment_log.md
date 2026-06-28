# Qwen Checkpoint-Selected Scheduled-State Compiler Experiment Log

## Objective

Test whether checkpoint selection and stage-specific intermediate-state supervision improve a Qwen numeric-copy compiler's long-chain exact execution.

## Experiment Question

Can a QLoRA-adapted compiler reach better length-24 execution when state supervision is treated as a curriculum tool rather than a fixed objective, and when the reported model is the best saved validation checkpoint rather than the final optimizer step?

## Planned Conditions

1. Smoke tests for scheduled state weights and checkpoint snapshots.
2. Matched no-state curriculum compiler with validation checkpoint selection.
3. Constant light scheduled-state compiler with state loss weight 0.25 in every stage.
4. State-through-medium compiler with state loss active before the long stage and disabled in the long stage.
5. Optional state-through-medium-light-long compiler with full state loss before the long stage and weight 0.25 in the long stage.

## Primary Selection Rule

Select the saved checkpoint with the highest `paired_len24_executor_accuracy`.

## Primary Evaluation Metrics

- `paired_len24_executor_accuracy`
- `paired_len24_compiler_pair_state_consistency`
- `standard_len24_executor_accuracy`
- `paraphrase_len24_executor_accuracy`
- `state_prefix_fraction`
- `program_exact`

## Artifact Policy

Lightweight outputs stay in:

```text
experiments/qwen_checkpoint_scheduled_state_compiler/runs/
experiments/qwen_checkpoint_scheduled_state_compiler/analysis/
experiments/qwen_checkpoint_scheduled_state_compiler/reports/
```

Large adapters and head checkpoints stay in:

```text
large_artifacts/qwen_checkpoint_scheduled_state_compiler/checkpoints/
```

## Log

### 2026-06-22

- Created standalone experiment directory.
- Forked the Qwen numeric-copy compiler harness into a checkpoint-selected scheduled-state harness.
- Added stage-specific state-loss schedules.
- Added validation checkpoint snapshots and selected-checkpoint metadata.
- Added standalone README and experiment log.
- Ran `smoke_tiny_scheduled_state` with a tiny frozen causal LM. The run completed, saved validation checkpoints, selected a checkpoint by `paired_len3_executor_accuracy`, generated analysis CSVs, and verified that state loss was active in the short stage and inactive in the long stage.
- Ran `smoke_qwen3_4b_scheduled_state` with `Qwen/Qwen3-4B`, 4-bit QLoRA, and scheduled state loss. The run completed, saved real adapter checkpoints, generated analysis CSVs, and verified that the large artifacts are stored outside the experiment directory.
- Ran `main_no_state_selected_s900`. The selected checkpoint is step 800 with 27.3% paired length-24 exact execution, 43.8% paired state consistency, 35.9% standard length-24 exact execution, and 12.5% paraphrase length-24 exact execution. The final step 900 checkpoint fell to 15.6% paired length-24, confirming that checkpoint selection is necessary for this setup.
- Ran `main_state_w025_selected_s900`. The selected checkpoint is step 800 with 30.5% paired length-24 exact execution, 60.9% paired state consistency, 34.4% standard length-24 exact execution, and 26.6% paraphrase length-24 exact execution. The final step 900 checkpoint fell to 12.5% paired length-24 and 0.0% paraphrase length-24.
- Ran `main_state_l12_off_long_selected_s900`. The schedule used state loss weight 1.0 through the train stage and 0.0 in the long stage. The selected checkpoint is step 800 with 22.7% paired length-24 exact execution, 32.8% paired state consistency, 34.4% standard length-24 exact execution, and 4.7% paraphrase length-24 exact execution. The final checkpoint reached 37.5% standard length-24 but only 18.0% paired length-24.
- Ran `main_state_l12_w025_long_selected_s900`. The schedule used state loss weight 1.0 through the train stage and 0.25 in the long stage. The selected checkpoint is step 800 with 30.5% paired length-24 exact execution, 75.0% paired state consistency, 39.1% standard length-24 exact execution, and 21.9% paraphrase length-24 exact execution. The final checkpoint fell to 15.6% paired length-24.
- Added an independent selected-checkpoint retest on fresh length-24 programs with 256 standard examples, 256 paraphrase examples, and 256 paired programs per arm. Fresh paired length-24 exact execution: no-state 25.0%, constant 0.25 state 32.8%, full-early/off-long 19.1%, full-early/light-long 32.2%. Fresh paired state consistency: no-state 40.6%, constant 0.25 state 55.9%, full-early/off-long 29.3%, full-early/light-long 70.7%.
