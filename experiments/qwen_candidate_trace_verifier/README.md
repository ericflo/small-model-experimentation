# Qwen Candidate-Trace Verifier

This experiment trains a transformer verifier over candidate execution traces for
local repairs of Qwen-compiled modular-arithmetic programs.

The fixed compiler reads a prompt, copies an initial value plus 24 operation and
argument slots, and executes the copied program modulo 97. Candidate repair
search enumerates 1,299 local programs for each length-24 prompt. The trace
verifier chooses among those candidates without access to the true answer or true
state trajectory at test time.

## Main Result

Primary run: `main_trace_verifier_s512`

| Split | Base | Trace verifier | Pair rerank | Oracle ceiling |
|---|---:|---:|---:|---:|
| Fresh standard L24 | 28.5% | 50.4% | n/a | 90.6% |
| Fresh paraphrase L24 | 28.5% | 55.5% | n/a | 86.7% |
| Fresh paired L24 | 30.3% | 53.7% | 56.2% | 88.1% |

The trace verifier recovers 40.5% of the base-to-oracle gap on the fresh paired
split. Pair reranking improves paired exact execution to 56.2% and paired
both-correct accuracy to 52.3%.

## Layout

```text
experiments/qwen_candidate_trace_verifier/
  src/       experiment and analysis scripts
  reports/   standalone paper, HTML report, and experiment log
  analysis/  regenerated CSVs, summary, and figures
  runs/      per-run metrics and train logs, without large checkpoints
  checkpoint_manifest.csv

large_artifacts/qwen_candidate_trace_verifier/checkpoints/
  fixed_compiler_step00800/
  main_trace_verifier_s512/candidate_trace_verifier.pt
```

## Reproduction

Run the main experiment:

```bash
python experiments/qwen_candidate_trace_verifier/src/qwen_candidate_trace_verifier_experiment.py \
  --run_name main_trace_verifier_s512 \
  --train_examples 512 \
  --val_examples 128 \
  --eval_examples 256 \
  --eval_pairs 256 \
  --repair_topk 3 \
  --repair_max_edits 2 \
  --verifier_epochs 18 \
  --qwen_batch_size 8 \
  --trace_d_model 128 \
  --trace_layers 3 \
  --trace_heads 4
```

Regenerate analysis:

```bash
python experiments/qwen_candidate_trace_verifier/src/analyze_qwen_candidate_trace_verifier.py
```

## Key Files

- `analysis/summary.md`
- `analysis/final_metrics.csv`
- `analysis/all_final_metrics.csv`
- `analysis/figures/executor_accuracy.png`
- `reports/qwen_candidate_trace_verifier_experiment_log.md`
- `reports/qwen_candidate_trace_verifier_paper.md`
