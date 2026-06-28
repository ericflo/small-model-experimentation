# Qwen LoRA Parser Compiler

Standalone QLoRA-attached parser/compiler experiment.

The experiment tests whether live LoRA updates on a small causal language model can make its hidden states readable as a symbolic modular-arithmetic program. The pilot gate is a token-role and symbol tagger trained through the model; the full condition adds a differentiable executor that compiles predicted program symbols into the final answer.

## Layout

```text
src/qwen_lora_parser_compiler_experiment.py     training and evaluation harness
src/analyze_qwen_lora_parser_compiler.py        run aggregation and plots
runs/                                           lightweight JSON and CSV outputs
reports/                                        experiment log and standalone write-up
analysis/                                       aggregate tables and figures
```

Large checkpoints are stored outside the experiment directory:

```text
large_artifacts/qwen_lora_parser_compiler/checkpoints/
```

## Reading Order

1. `reports/qwen_lora_parser_compiler_paper.md`
2. `analysis/summary.md`
3. `analysis/final_metrics.csv`
4. `reports/qwen_lora_parser_compiler_experiment_log.md`

## Large Artifacts

You only need the large artifact directory to load adapters or resume/evaluate a checkpoint. The main trained condition is:

```text
large_artifacts/qwen_lora_parser_compiler/checkpoints/main_qwen3_4b_qlora_trace_argstrong_mixed_l12/
```

The checkpoint manifest is:

```text
experiments/qwen_lora_parser_compiler/checkpoint_manifest.csv
```
