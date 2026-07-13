# Qwen LoRA Typed-Bytecode Trace Compiler

**Status:** finished

Standalone experiment for training a Qwen 4B model, with QLoRA adapters, to compile natural-language tasks into executable typed bytecode.

The experiment keeps lightweight outputs in this directory and stores large checkpoints separately:

```text
experiments/qwen_lora_typed_bytecode_trace_compiler/
  src/        training and analysis code
  runs/       per-run JSON/CSV logs
  analysis/   aggregate tables and figures
  reports/    standalone Markdown and HTML report

large_artifacts/qwen_lora_typed_bytecode_trace_compiler/checkpoints/
  adapter and head checkpoints
```

## Main Question

Can a small posttraining change, implemented as QLoRA adapters plus a typed-bytecode compiler head, make a local Qwen 4B model reliably emit executable programs whose VM result answers the prompt?

## Reading Order

1. `reports/qwen_lora_typed_bytecode_trace_compiler_paper.md`
2. `analysis/summary.md`
3. `analysis/final_metrics.csv`
4. `experiment_log.md`

## Large Artifacts

Large checkpoints are intentionally not stored under this experiment directory. The checkpoint manifest is:

```text
checkpoint_manifest.csv
```
