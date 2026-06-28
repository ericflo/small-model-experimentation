# Qwen3.5-4B Bucket-Belief Probe Ranker

## Objective

This standalone experiment tests whether a Qwen3.5-4B LoRA can turn verifier state into a deployable probe-ranking policy by predicting which candidate-output bucket contains the hidden target program. The model does not name operators and does not see the answer at inference time; it scores candidate probes by the expected survivors implied by its bucket probabilities.

## Experimental Design

- Substrate: two-operator typed program search with a 96-case probe pool, four visible observations, sixteen hidden checks, and library sizes 64, 128, 256, and held-out 512 at evaluation.
- Candidate probes for the learned ranker: top 8 remaining probes by target-independent split statistics.
- Training target: for each candidate probe, predict the output bucket that contains the true target program.
- Rollout rule: choose the probe with the smallest model-predicted expected survivor count, observe its true output, update the verifier candidate set, and repeat for three probes.
- Controls: target-independent split top-1, target-aware oracle over the same top-8 candidate probes, target-aware oracle over the full 96-case pool, and the untrained base model under the same bucket-scoring rule.

## Data Summary

- Process records: train=300, eval=160.
- Bucket SFT examples: 4952.
- Bucket SFT states: 619.

## Results

- At budget 3, the SFT bucket ranker reached 50.0% hidden-all accuracy versus 48.8% for target-independent split top-1, 48.8% for the base bucket ranker, 61.3% for an oracle over the same top-8 probe set, and 86.9% for the full-pool oracle.
- The recoverable top-8 oracle gap is 12.5 points over split top-1; the SFT ranker captured 1.3 points of that gap.

### Budget-3 Overall

| Arm | Hidden-all accuracy | Candidates left | Hidden-equivalent left |
|---|---:|---:|---:|
| Split top-1 | 48.8% | 1005.0 | 149.1 |
| Oracle over top-8 | 61.3% | 657.7 | 146.8 |
| Oracle over full pool | 86.9% | 267.1 | 145.5 |
| Base bucket ranker | 48.8% | 795.6 | 156.6 |
| SFT bucket ranker | 50.0% | 1020.1 | 149.2 |

### Budget-3 by Template

| Arm | Affine-mod | Compare-gate |
|---|---:|---:|
| Split top-1 | 93.8% | 3.8% |
| Oracle over top-8 | 98.8% | 23.8% |
| Oracle over full pool | 100.0% | 73.8% |
| Base bucket ranker | 93.8% | 3.8% |
| SFT bucket ranker | 95.0% | 5.0% |

## Interpretation

The experiment is intentionally decisive about whether target-aware oracle headroom can be converted into deployable ranking by a learned bucket-belief model. A lift over split top-1 means the model learned a useful non-uniform belief over output buckets. A result near split top-1 means the oracle gap is mostly unavailable without additional state, candidate representation, or truly generative probe construction.

The base-model arm matters because it distinguishes learned bucket inference from prompt priors. The oracle-over-top-8 arm matters because it bounds what any ranker can gain when it is restricted to the same split-mined candidate probes. The full-pool oracle remains a headroom measurement, not a deployable result.

## Figures

- `reports/figures/budget_curve.png`
- `reports/figures/budget3_accuracy.png`
- `reports/figures/budget3_by_template.png`
- `reports/figures/bucket_sft_loss.png`

## Reproduction

Run from this experiment directory:

```bash
python scripts/build_dataset.py --train-per-cell 50 --eval-per-cell 20 --states-per-record 3 --query-pool-cases 96 --action-source mined8
python scripts/build_bucket_dataset.py
python scripts/build_bucket_dataset.py --records data/eval_records.jsonl --states data/process_eval_states.jsonl --out data/bucket_eval_examples.jsonl
python scripts/eval_bucket_ranker.py --policy split_top1 --name split_top1 --max-budget 3
python scripts/eval_bucket_ranker.py --policy oracle_topk --name oracle_top8 --max-budget 3
python scripts/eval_bucket_ranker.py --policy fullpool_oracle --name fullpool_oracle --max-budget 3
python scripts/eval_bucket_ranker.py --policy base_bucket --name base_bucket_ranker --max-budget 3
python scripts/train_bucket_sft.py --max-steps 220 --batch-size 2 --grad-accum 2
python scripts/eval_bucket_ranker.py --policy adapter_bucket --name sft_bucket_ranker --adapter-dir /workspace/large_artifacts/qwen35_4b_bucket_belief_probe_ranker/models/bucket_sft_lora --max-budget 3
python scripts/make_report.py
```
