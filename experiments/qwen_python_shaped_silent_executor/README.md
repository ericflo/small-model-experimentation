# Qwen Python-Shaped Silent Executor

**Status:** finished

Standalone experiment for testing whether a Qwen 4B model can learn to execute
Python-shaped mini-programs using private latent compute positions instead of
emitting an explicit chain-of-thought trace.

Small artifacts live here:

```text
experiments/qwen_python_shaped_silent_executor/
  src/        experiment and reporting code
  runs/       per-run configs, metrics, logs
  reports/    standalone Markdown/HTML report and figures
```

Large artifacts are stored separately:

```text
large_artifacts/qwen_python_shaped_silent_executor/
  checkpoints/  QLoRA adapters and heads
  caches/       optional cached tensors
```

Primary report:

```text
reports/qwen_python_shaped_silent_executor_report.md
reports/qwen_python_shaped_silent_executor_report.html
```

