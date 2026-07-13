# Qwen3.5-4B Oracle Process GRPO

**Status:** finished

This standalone experiment trains Qwen3.5-4B as a process controller inside an executable verifier MDP. The model does not emit operators directly. It chooses among displayed probe actions, and exhaustive search updates the candidate set deterministically.

Large model artifacts are intentionally outside this directory:

- `/workspace/large_artifacts/qwen35_4b_oracle_process_grpo/models/sft_process_lora`
- `/workspace/large_artifacts/qwen35_4b_oracle_process_grpo/models/dpo_process_lora`
- `/workspace/large_artifacts/qwen35_4b_oracle_process_grpo/models/dpo_shuffled_lora`
- `/workspace/large_artifacts/qwen35_4b_oracle_process_grpo/models/grpo_process_lora`

Main outputs:

- `reports/qwen35_4b_oracle_process_grpo_report.md`
- `reports/figures/`
- `reports/*.csv`
- `run_logs/`
- `logs/experiment_log.md`

Reproduction order:

1. `python scripts/build_dataset.py --train-per-cell 80 --eval-per-cell 16 --states-per-record 3`
2. Run non-model baselines with `scripts/eval_policy.py`.
3. `python scripts/train_sft_policy.py --max-steps 360 --batch-size 2 --grad-accum 2`
4. `python scripts/train_dpo_policy.py --max-steps 160 --batch-size 2 --grad-accum 2`
5. `python scripts/train_dpo_policy.py --max-steps 160 --batch-size 2 --grad-accum 2 --shuffle-rewards --output-dir /workspace/large_artifacts/qwen35_4b_oracle_process_grpo/models/dpo_shuffled_lora`
6. `python scripts/train_grpo_policy.py --max-steps 120 --batch-size 2 --grad-accum 2 --group-size 8`
7. `python scripts/make_report.py`

