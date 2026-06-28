# Qwen3.5-4B Bucket-Belief Probe Ranker

This standalone experiment tests whether Qwen3.5-4B can convert target-aware probe headroom into a deployable probe-ranking policy.

The model does not emit operators. For each verifier state, the system mines the top 8 candidate probes by target-independent split quality. For each probe, the prompt shows the output buckets that surviving candidate programs would produce. Qwen is trained to predict which bucket contains the hidden target program. At rollout time, the model scores each candidate probe by predicted expected survivors and the verifier executes the probe with the smallest score.

Large model artifacts are intentionally outside this directory:

- `/workspace/large_artifacts/qwen35_4b_bucket_belief_probe_ranker/models/bucket_sft_lora`
- `/workspace/large_artifacts/qwen35_4b_bucket_belief_probe_ranker/cache/bucket_train_examples.jsonl`

Main outputs:

- `reports/qwen35_4b_bucket_belief_probe_ranker_report.md`
- `reports/figures/`
- `reports/*.csv`
- `reports/eval/*.json`
- `run_logs/`
- `logs/experiment_log.md`

Reproduction order:

1. `python scripts/build_dataset.py --train-per-cell 50 --eval-per-cell 20 --states-per-record 3 --query-pool-cases 96 --action-source mined8`
2. `python scripts/build_bucket_dataset.py`
3. `python scripts/build_bucket_dataset.py --records data/eval_records.jsonl --states data/process_eval_states.jsonl --out data/bucket_eval_examples.jsonl`
4. `python scripts/eval_bucket_ranker.py --policy split_top1 --name split_top1 --max-budget 3`
5. `python scripts/eval_bucket_ranker.py --policy oracle_topk --name oracle_top8 --max-budget 3`
6. `python scripts/eval_bucket_ranker.py --policy fullpool_oracle --name fullpool_oracle --max-budget 3`
7. `python scripts/eval_bucket_ranker.py --policy base_bucket --name base_bucket_ranker --max-budget 3`
8. `python scripts/train_bucket_sft.py --max-steps 220 --batch-size 2 --grad-accum 2`
9. `python scripts/eval_bucket_ranker.py --policy adapter_bucket --name sft_bucket_ranker --adapter-dir /workspace/large_artifacts/qwen35_4b_bucket_belief_probe_ranker/models/bucket_sft_lora --max-budget 3`
10. `python scripts/make_report.py`
