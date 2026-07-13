# Qwen3.5-4B Oracle Probe Synthesis MDP

**Status:** finished

This standalone line-2 experiment tests whether Qwen3.5-4B can exploit richer verifier probes. The model does not emit operators. It chooses among eight displayed probe inputs, but those probes are mined from a 96-case bank using target-independent candidate-bucket statistics.

Large model artifacts are intentionally outside this directory:

- `/workspace/large_artifacts/qwen35_4b_oracle_probe_synthesis_mdp/models/sft_process_lora`
- `/workspace/large_artifacts/qwen35_4b_oracle_probe_synthesis_mdp/models/dpo_process_lora`
- `/workspace/large_artifacts/qwen35_4b_oracle_probe_synthesis_mdp/models/grpo_process_lora`

Main outputs:

- `reports/qwen35_4b_oracle_probe_synthesis_mdp_report.md`
- `reports/figures/`
- `reports/*.csv`
- `reports/eval/*.json`
- `run_logs/`
- `logs/experiment_log.md`

Reproduction order:

1. `python scripts/build_dataset.py --train-per-cell 50 --eval-per-cell 20 --states-per-record 3 --query-pool-cases 96 --action-source mined8`
2. Run non-model baselines with `scripts/eval_policy.py` for `random`, `max_split`, `oracle`, `fullpool_max_split`, and `fullpool_oracle`.
3. `python scripts/eval_policy.py --policy base --name base_mined8 --action-source mined8 --max-budget 3`
4. `python scripts/train_sft_policy.py --max-steps 260 --batch-size 2 --grad-accum 2`
5. `python scripts/train_dpo_policy.py --max-steps 120 --batch-size 2 --grad-accum 2`
6. `python scripts/train_grpo_policy.py --max-steps 80 --batch-size 2 --grad-accum 2 --group-size 8`
7. `python scripts/eval_policy.py --policy adapter --name sft_scrambled_features --adapter-dir /workspace/large_artifacts/qwen35_4b_oracle_probe_synthesis_mdp/models/sft_process_lora --action-source mined8 --max-budget 3 --scramble-features`
8. `python scripts/make_report.py`
