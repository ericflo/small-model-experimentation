# Qwen Register-Token Latent Compiler

**Status:** finished

This standalone experiment tests whether a Qwen-attached model can write an
executable modular-arithmetic program into a fixed bank of appended register
tokens. A trainable bridge reads only those register hidden states, predicts an
initial value plus per-step operations and arguments, and an invisible executor
runs the predicted program modulo 97.

## Result

The main trace-supervised run learned a real register interface for short and
medium chains, but did not solve robust length-24 generalization.

| Split | Executor exact | Program exact | Init | Op | Arg | Pair both |
|---|---:|---:|---:|---:|---:|---:|
| Standard L4 | 88.3% | 88.3% | 88.3% | 100.0% | 100.0% | n/a |
| Standard L8 | 94.5% | 94.5% | 94.5% | 100.0% | 100.0% | n/a |
| Standard L12 | 94.5% | 94.5% | 94.5% | 100.0% | 100.0% | n/a |
| Standard L24 | 21.9% | 21.1% | 91.4% | 97.3% | 95.2% | n/a |
| Paraphrase L24 | 3.1% | 2.3% | 89.1% | 92.5% | 88.5% | n/a |
| Paired L24 | 12.5% | 12.1% | 92.2% | 94.7% | 92.4% | 1.6% |

Two controls stayed at chance under comparable budgets:

| Run | Standard L24 | Paraphrase L24 | Paired L24 |
|---|---:|---:|---:|
| Direct answer head | 3.1% | 0.0% | 1.6% |
| Register answer-only | 1.6% | 0.0% | 0.0% |

## Layout

```text
experiments/qwen_register_token_latent_compiler/
  src/       experiment and analysis scripts
  reports/   experiment log and final writeup in Markdown/HTML
  analysis/  regenerated CSVs, summary, and figures
  runs/      per-run metrics and training logs, without large checkpoints
  checkpoint_manifest.csv

large_artifacts/qwen_register_token_latent_compiler/checkpoints/
  saved adapters and bridge heads
```

## Reproduction

Main trace-supervised run:

```bash
python experiments/qwen_register_token_latent_compiler/src/qwen_register_token_latent_compiler_experiment.py \
  --output_dir experiments/qwen_register_token_latent_compiler/runs/main_register_trace_s600 \
  --checkpoint_dir large_artifacts/qwen_register_token_latent_compiler/checkpoints/main_register_trace_s600 \
  --variants register_trace \
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
  --init_trace_loss_weight 4 \
  --op_trace_loss_weight 1 \
  --arg_trace_loss_weight 4 \
  --eval_every 150 \
  --stage_eval_every 150 \
  --max_length 768 \
  --lr 0.0002 \
  --lora_r 8 \
  --lora_alpha 16
```

Regenerate analysis after runs:

```bash
python experiments/qwen_register_token_latent_compiler/src/analyze_qwen_register_token_latent_compiler.py
```

## Key Files

- `reports/qwen_register_token_latent_compiler_experiment_log.md`
- `reports/qwen_register_token_latent_compiler_paper.md`
- `reports/qwen_register_token_latent_compiler_paper.html`
- `analysis/summary.md`
- `analysis/all_final_metrics.csv`
- `checkpoint_manifest.csv`
