# Qwen3.5-4B Joint Shortlister Ladder

**Status:** finished

This standalone experiment tests whether Qwen3.5-4B can use an in-context typed operator inventory to emit a joint two-operator shortlist for programs of the form `LEFT(xs), RIGHT(xs)`.

The package keeps source, configs, small datasets, logs, result JSON, CSV summaries, figures, and the written report in this directory. Large model artifacts are stored under `/workspace/large_artifacts/qwen35_4b_joint_shortlister_ladder` so this directory can be downloaded without adapter checkpoints.

## Design

- Model: local Qwen3.5-4B with a QLoRA adapter.
- Output format: exact joint pair `LLL,RRR`.
- Library ladder: `64, 128, 256, 512` operators.
- Templates: numeric `pair_affine_mod` and low-information `pair_compare_gate`.
- Aliases: randomized per record so codes must be read from the inventory.
- Controls: base model, trained model, and trained model with shuffled inventory descriptions.
- Observations: six random visible cases and six max-split designed cases.
- Metrics: joint pair recall@k up to the configured beam width, marginal LEFT/RIGHT recall@k, and observation survivor counts.

## Commands

```bash
python scripts/build_dataset.py
python scripts/train_joint_shortlister.py
python scripts/eval_joint_shortlister.py
python scripts/make_report.py
```

The main report is `reports/qwen35_4b_joint_shortlister_ladder_report.md`.
