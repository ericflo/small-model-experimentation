# Qwen Numeric-Copy Compiler

**Status:** finished

Standalone numeric-copy compiler experiment for a Qwen causal language model.

The experiment tests whether a model can learn token roles and ordered program slots while numeric and operator symbols are copied from a deterministic token map. The frozen pilot trains only the compiler heads over frozen hidden states. The full condition trains QLoRA adapters plus the same numeric-copy compiler.

## Layout

```text
src/qwen_numeric_copy_compiler_experiment.py    training and evaluation harness
src/analyze_qwen_numeric_copy_compiler.py       run aggregation and plots
runs/                                           lightweight JSON and CSV outputs
reports/                                        experiment log and standalone write-up
analysis/                                       aggregate tables and figures
```

Large checkpoints are stored outside the experiment directory:

```text
large_artifacts/qwen_numeric_copy_compiler/checkpoints/
```

## Reading Order

1. `reports/qwen_numeric_copy_compiler_paper.md`
2. `analysis/summary.md`
3. `analysis/final_metrics.csv`
4. `reports/qwen_numeric_copy_compiler_experiment_log.md`

## Main Artifact

The main trained QLoRA condition is:

```text
large_artifacts/qwen_numeric_copy_compiler/checkpoints/main_qwen3_4b_qlora_numeric_copy_trace_mixed_l12/
```

The checkpoint manifest is:

```text
experiments/qwen_numeric_copy_compiler/checkpoint_manifest.csv
```
