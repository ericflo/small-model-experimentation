# Qwen3.5-4B Adaptive Evidence Budget Policy

**Status:** finished

This standalone experiment tests whether Qwen3.5-4B can be posttrained as a STOP/MORE controller for an executable verifier.

The verifier chooses the next probe by target-independent expected split. The model does not choose probes and does not name operators. Its only deployable decision is whether to stop and commit the current verifier-selected program or request one more executable observation, up to a maximum budget of ten probes.

Large artifacts are intentionally outside this directory:

- `/workspace/large_artifacts/qwen35_4b_adaptive_evidence_budget_policy/models/budget_sft_lora`

Main outputs:

- `reports/qwen35_4b_adaptive_evidence_budget_policy_report.md`
- `reports/figures/`
- `reports/*.csv`
- `reports/eval/*.json`
- `run_logs/`
- `logs/experiment_log.md`

Reproduction:

```bash
python scripts/build_dataset.py --train-per-cell 40 --eval-per-cell 20 --query-pool-cases 96 --max-budget 10
python scripts/eval_budget_policy.py --policy fixed --name fixed_budget3 --fixed-budget 3 --max-budget 10
python scripts/eval_budget_policy.py --policy fixed --name fixed_budget6 --fixed-budget 6 --max-budget 10
python scripts/eval_budget_policy.py --policy fixed --name fixed_budget10 --fixed-budget 10 --max-budget 10
python scripts/eval_budget_policy.py --policy threshold --name threshold_100 --threshold 100 --max-budget 10
python scripts/eval_budget_policy.py --policy threshold --name threshold_1000 --threshold 1000 --max-budget 10
python scripts/eval_budget_policy.py --policy oracle_stop --name oracle_stop --max-budget 10
python scripts/eval_budget_policy.py --policy base --name base_budget_policy --max-budget 10
python scripts/train_budget_sft.py --max-steps 220 --batch-size 2 --grad-accum 2
python scripts/eval_budget_policy.py --policy adapter --name sft_budget_policy --adapter-dir /workspace/large_artifacts/qwen35_4b_adaptive_evidence_budget_policy/models/budget_sft_lora --max-budget 10
python scripts/make_report.py
```
