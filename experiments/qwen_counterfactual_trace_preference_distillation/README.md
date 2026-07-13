# Qwen Counterfactual Trace Preference Distillation

**Status:** finished

Standalone experiment for testing whether a Qwen-attached typed-bytecode
compiler improves when repair candidates are trained with hard counterfactual
trace preferences instead of final-answer labels alone.

Small artifacts live here:

```text
experiments/qwen_counterfactual_trace_preference_distillation/
```

Large checkpoints live separately here:

```text
large_artifacts/qwen_counterfactual_trace_preference_distillation/checkpoints/
```

## Layout

- `src/`: experiment and analysis scripts.
- `runs/`: per-run metrics, logs, and dataset manifests.
- `analysis/`: aggregate CSVs and generated figures.
- `reports/`: Markdown and HTML reports.
- `experiment_log.md`: running journal.
- `checkpoint_manifest.csv`: run-to-checkpoint map.

## Question

Can a candidate preference objective over executable VM traces select useful
repair programs and distill those choices back into a deployable compiler?

The candidate quality order is:

```text
invalid < valid_wrong < answer_correct < trace_consistent < canonical
```

The preference model trains on counterfactual groups where the candidate set
contains a better executable program than the base decode.
