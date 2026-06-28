# Experiment Log

## 2026-06-27

- Created standalone compiler-pilot package.
- Chose constrained candidate scoring to isolate operation/composition choice from JSON syntax failures.
- Planned arms: first-visible candidate baseline, frozen Qwen constrained scorer, QLoRA constrained scorer.
- Planned headline split: depth-1 operation selection vs depth-2/3 composition.
- Built 180 train, 40 validation, and 48 eval records. Eval depth split: 27 depth-1, 15 depth-2, 6 depth-3.
- Oracle constrained candidate coverage: 48/48.
- First-visible baseline: 45/48 filtered execution accuracy.
- Random-visible baseline: 87.5%-93.8% filtered accuracy across five seeds, mean 90.0%.
- Frozen Qwen constrained scorer: 44/48 filtered accuracy, 19/21 on depth-2+.
- QLoRA constrained scorer: 48/48 filtered accuracy, 21/21 on depth-2+.
- Generated final report and charts under `reports/`.
