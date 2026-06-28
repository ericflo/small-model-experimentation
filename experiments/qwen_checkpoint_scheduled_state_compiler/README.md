# Qwen Checkpoint-Selected Scheduled-State Compiler

Standalone experiment testing whether stage-specific intermediate-state supervision and validation checkpoint selection improve a Qwen numeric-copy compiler on long modular-arithmetic programs.

The compiler reads hidden states from `Qwen/Qwen3-4B`, copies the initial value, operations, and arguments from token-level maps, executes the copied program modulo 97 with an invisible runtime, and reports exact execution metrics. The intervention is not a new parser or task; it is the training policy around state supervision:

- save a real checkpoint at every validation point;
- select the checkpoint by paired length-24 exact execution;
- compare no state loss, constant light state loss, and scheduled state loss that is active before the long-chain stage and reduced or disabled during it.

## Layout

```text
src/qwen_checkpoint_scheduled_state_compiler_experiment.py  training and evaluation harness
src/analyze_qwen_checkpoint_scheduled_state_compiler.py     run aggregation, selected-checkpoint table, and plots
src/evaluate_selected_qwen_checkpoint_scheduled_state_compiler.py
                                                           fresh retest for selected checkpoints
runs/                                                       lightweight JSON and CSV outputs
analysis/                                                   aggregate CSVs and generated figures
reports/                                                    experiment log and standalone write-up
checkpoint_manifest.csv                                     generated list of large checkpoint files
```

Large checkpoints are stored outside the experiment directory:

```text
large_artifacts/qwen_checkpoint_scheduled_state_compiler/checkpoints/
```

## Main Metrics

- `executor_accuracy`: exact final answer after copying and executing the compiled program.
- `program_exact`: exact full compiled-program correctness.
- `state_all_exact`: every intermediate modular state is correct.
- `state_prefix_fraction`: fraction of the program executed before the first state error.
- `compiler_pair_state_consistency`: whether standard and paraphrased renderings compile to the same complete state trajectory.
- `selected_checkpoint`: saved checkpoint chosen by `paired_len24_executor_accuracy`.

## Reading Order

1. `reports/qwen_checkpoint_scheduled_state_compiler_experiment_log.md`
2. `analysis/summary.md`
3. `analysis/selected_checkpoints.csv`
4. `analysis/selected_retest_metrics.csv`
5. `analysis/final_metrics.csv`
6. `reports/qwen_checkpoint_scheduled_state_compiler_paper.md`
