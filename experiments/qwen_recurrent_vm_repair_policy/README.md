# Qwen Recurrent VM Repair Policy

Standalone experiment for testing whether a Qwen-attached typed-bytecode
compiler improves when the model is trained as a repeated repair transition
operator instead of a one-shot program generator.

Small artifacts live here:

```text
experiments/qwen_recurrent_vm_repair_policy/
```

Large checkpoints live separately here:

```text
large_artifacts/qwen_recurrent_vm_repair_policy/checkpoints/
```

## Layout

- `src/`: experiment and analysis scripts.
- `runs/`: per-run metrics, logs, and dataset manifests.
- `analysis/`: aggregate CSVs and generated figures.
- `reports/`: Markdown and HTML reports.
- `experiment_log.md`: running journal.
- `checkpoint_manifest.csv`: run-to-checkpoint map.

## Question

Can one Qwen-conditioned forward pass serve as one loop iteration in an
execution-feedback repair process?

The recurrent policy receives:

```text
prompt features + current bytecode + VM final value + VM trace
```

and predicts either one edit action or `STOP`. The edited program is executed,
then fed back to the same policy for another step.
