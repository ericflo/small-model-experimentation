# Qwen Search-Augmented Rollout Distillation

**Status:** finished

This standalone experiment trains `Qwen/Qwen3-4B` as a recurrent controller for
a typed bytecode VM.

The model receives a task prompt plus dense projected VM-state tokens. It
predicts one VM edit action or `STOP`, the VM executes the edited program, and
the same model is called again for the next recurrent step.

The central intervention is search-augmented rollout distillation: policy-visited
states are labeled by bounded answer-verified repair search, not only by a
gold-program edit trace. The model also receives pairwise action-ranking
supervision from repair-positive and repair-negative edits.

Large checkpoints are stored outside this directory:

```text
large_artifacts/qwen_search_augmented_rollout_distillation/checkpoints/
```

Read `experiment_log.md` for the iteration record. Final reports are written
under `reports/`.

Main outputs:

- `reports/search_augmented_rollout_distillation_report.md`
- `reports/search_augmented_rollout_distillation_report.html`
- `analysis/main_active_accuracy.png`
- `analysis/main_k_curves_learned.png`
- `runs/main_search_r1_rank00_e2_20260624/`
