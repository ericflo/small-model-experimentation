# Qwen Register-Token Structured Runtime

This standalone experiment tests whether a Qwen-attached model can write a
program into fixed register tokens and have that program executed by a fixed
cyclic runtime. The bridge reads only register-token hidden states. It predicts
an initial residue, primitive operation routes, and operation arguments. A
deterministic modulo-97 runtime executes the predicted program.

The main intervention is training pressure on the runtime trajectory:

- supervised executable slots;
- supervised intermediate cyclic states;
- paired consistency between two prompt renderings of the same latent program.

## Result

The main run solves the trained length range and partially lifts length-24
standard execution, but it does not solve prompt-invariant long-chain execution.

| Split | Executor exact | Program exact | Init | Op | Arg | Prefix | Pair both | Pair state consistency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Standard L4 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | n/a | n/a |
| Standard L8 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | n/a | n/a |
| Standard L12 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | n/a | n/a |
| Standard L24 | 25.0% | 25.0% | 100.0% | 93.8% | 89.7% | 80.5% | n/a | n/a |
| Paraphrase L24 | 5.5% | 4.7% | 100.0% | 88.9% | 83.8% | 81.0% | n/a | n/a |
| Paired L24 | 11.7% | 10.2% | 100.0% | 89.7% | 85.7% | 79.1% | 1.6% | 1.6% |

Matched state-loss control without paired consistency:

| Split | Main | No-pair control |
|---|---:|---:|
| Standard L24 | 25.0% | 3.9% |
| Paraphrase L24 | 5.5% | 1.6% |
| Paired L24 | 11.7% | 1.6% |

## Layout

```text
experiments/qwen_register_structured_runtime/
  src/       experiment and analysis scripts
  reports/   experiment log and final writeup
  analysis/  regenerated CSVs, summary, and figures
  runs/      per-run metrics and training logs, without large checkpoints
  checkpoint_manifest.csv

large_artifacts/qwen_register_structured_runtime/checkpoints/
  saved adapters and bridge heads
```

## Reproduction

Main run:

```bash
python experiments/qwen_register_structured_runtime/src/qwen_register_structured_runtime_experiment.py \
  --output_dir experiments/qwen_register_structured_runtime/runs/main_structured_trace_state_consistency_s600 \
  --checkpoint_dir large_artifacts/qwen_register_structured_runtime/checkpoints/main_structured_trace_state_consistency_s600 \
  --variants structured_trace_state_consistency \
  --register_style bare \
  --curriculum_stages short:1:4:150,medium:1:8:150,train:1:12:150,long:8:24:150 \
  --train_size 512 \
  --answer_train_size 512 \
  --eval_size 128 \
  --eval_lengths 4,8,12,24 \
  --train_batch_size 4 \
  --eval_batch_size 8 \
  --register_width 512 \
  --register_layers 1 \
  --register_heads 4 \
  --head_width 512 \
  --state_loss_weight 1.0 \
  --pair_consistency_loss_weight 0.1 \
  --pair_state_consistency_loss_weight 0.5 \
  --lr 0.0002 \
  --lora_r 8 \
  --lora_alpha 16
```

Regenerate analysis after runs:

```bash
python experiments/qwen_register_structured_runtime/src/analyze_qwen_register_structured_runtime.py
```

## Key Files

- `reports/qwen_register_structured_runtime_experiment_log.md`
- `reports/qwen_register_structured_runtime_paper.md`
- `reports/qwen_register_structured_runtime_paper.html`
- `analysis/summary.md`
- `analysis/all_final_metrics.csv`
- `checkpoint_manifest.csv`
