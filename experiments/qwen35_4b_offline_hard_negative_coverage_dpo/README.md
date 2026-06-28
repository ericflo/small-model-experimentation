# qwen35_4b_offline_hard_negative_coverage_dpo

Standalone experiment package for an offline hard-negative preference objective aimed at improving coverage from a fixed Qwen3.5-4B generator.

The experiment mines task-local preference pairs from fresh sampled candidate pools: hidden-correct candidates are preferred over hard negatives that are visible-passing but hidden-wrong, or otherwise high-order parsed failures. A small QLoRA adapter is trained with a DPO-style contrastive loss and evaluated on held-out MBPP tasks against tuned inference-only sampling baselines.

Large model artifacts are stored outside this directory under:

`/workspace/large_artifacts/qwen35_4b_offline_hard_negative_coverage_dpo`

