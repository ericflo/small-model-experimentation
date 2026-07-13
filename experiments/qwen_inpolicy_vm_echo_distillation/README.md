# Qwen In-Policy VM-ECHO Distillation

**Status:** finished

Standalone experiment for testing whether a frozen-Qwen typed-bytecode compiler
improves when it learns the VM consequences of its own proposed programs during
repair distillation.

Small artifacts live in this directory:

```text
experiments/qwen_inpolicy_vm_echo_distillation/
```

Large checkpoints live separately here:

```text
large_artifacts/qwen_inpolicy_vm_echo_distillation/checkpoints/
```

## Layout

- `src/`: experiment and analysis scripts.
- `runs/`: per-run metrics, logs, and dataset manifests.
- `analysis/`: aggregate CSVs and generated figures.
- `reports/`: Markdown and HTML reports.
- `experiment_log.md`: running journal.
- `checkpoint_manifest.csv`: run-to-checkpoint map.

## Question

Can in-policy VM-observation learning turn sampled program failures into useful
posttraining signal, beyond answer-verified repair targets alone?
