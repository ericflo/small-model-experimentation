# Qwen Learned Repair Verifier

This experiment trains a small non-oracle verifier to rerank local repairs of
programs compiled by a fixed Qwen-attached numeric compiler.

The compiler reads a modular-arithmetic prompt, copies an initial value, a
sequence of operations, and operation arguments, then executes the copied program
modulo 97. The learned verifier does not see the true answer or true state
trajectory at test time. It receives candidate-edit features and chooses one
candidate from a local top-k/two-edit repair neighborhood.

## Main Result

Primary run: `main_rich_learned_verifier_s512`

Fresh length-24 results:

| Split | Base | Learned verifier | Pair rerank | Oracle ceiling |
|---|---:|---:|---:|---:|
| Standard | 28.5% | 44.1% | n/a | 90.6% |
| Paraphrase | 28.5% | 48.0% | n/a | 86.7% |
| Paired | 30.3% | 47.3% | 51.0% | 88.1% |

The learned verifier recovers 29.4% of the base-to-oracle gap on the fresh
paired split. A paired consistency reranker raises paired executor accuracy to
51.0% and paired both-correct accuracy to 46.5%.

## Layout

```text
experiments/qwen_learned_repair_verifier/
  src/       experiment and analysis scripts
  reports/   standalone paper, HTML report, and experiment log
  analysis/  regenerated CSVs, summary, and figures
  runs/      per-run metrics and train logs, without large checkpoints
  checkpoint_manifest.csv

large_artifacts/qwen_learned_repair_verifier/checkpoints/
  fixed_compiler_step00800/
  main_rich_learned_verifier_s512/learned_verifier.pt
```

## Reproduction

Run the main experiment:

```bash
python experiments/qwen_learned_repair_verifier/src/qwen_learned_repair_verifier_experiment.py \
  --run_name main_rich_learned_verifier_s512 \
  --train_examples 512 \
  --val_examples 128 \
  --eval_examples 256 \
  --eval_pairs 256 \
  --repair_topk 3 \
  --repair_max_edits 2 \
  --verifier_epochs 18 \
  --qwen_batch_size 8 \
  --verifier_width 192
```

Regenerate analysis:

```bash
python experiments/qwen_learned_repair_verifier/src/analyze_qwen_learned_repair_verifier.py
```

## Key Files

- `analysis/summary.md`
- `analysis/final_metrics.csv`
- `analysis/all_final_metrics.csv`
- `analysis/figures/executor_accuracy.png`
- `reports/qwen_learned_repair_verifier_experiment_log.md`
- `reports/qwen_learned_repair_verifier_paper.md`
