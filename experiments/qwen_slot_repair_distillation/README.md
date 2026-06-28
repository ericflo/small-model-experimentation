# Qwen Slot Repair Distillation

This standalone experiment tests whether local repair headroom can be distilled
into a deployable gated slot editor for a frozen Qwen-attached numeric compiler.
The compiler emits an executable modular-arithmetic program. Offline local
repair search supplies corrected-program labels when available, and the editor
learns to decide which init/op/arg slots to edit plus what replacement values to
emit from the base compiler trace.

At evaluation time, the editor emits one program. It does not receive target
answers, target states, or a candidate list to rerank.

Small experiment files live here:

```text
experiments/qwen_slot_repair_distillation/
```

Large checkpoints live separately:

```text
large_artifacts/qwen_slot_repair_distillation/checkpoints/
```

## Main Result

The primary run was `main_slot_repair_distill_s512`.

| Split | Base | Editor | Oracle |
|---|---:|---:|---:|
| validation | 32.8% | 35.2% | 85.2% |
| fresh standard | 28.5% | 27.7% | 87.1% |
| fresh paraphrase | 25.0% | 18.8% | 85.2% |
| fresh paired | 25.4% | 24.8% | 86.1% |

The editor learned a validation signal but did not transfer cleanly to fresh
prompt distributions. The local oracle ceiling stayed high, so nearby corrected
programs usually exist; the failure is the one-shot policy's ability to select
the right sparse edits and values from the base trace alone.

## Primary Command

```bash
python experiments/qwen_slot_repair_distillation/src/qwen_slot_repair_distillation_experiment.py \
  --run_name main_slot_repair_distill_s512 \
  --train_examples 512 \
  --val_examples 128 \
  --eval_examples 256 \
  --eval_pairs 256 \
  --repair_topk 3 \
  --repair_max_edits 2 \
  --repair_max_pair_arg_slots 24 \
  --qwen_batch_size 8 \
  --editor_d_model 128 \
  --editor_layers 3 \
  --editor_heads 4 \
  --editor_ff_mult 4 \
  --editor_epochs 18 \
  --editor_lr 0.001 \
  --editor_target_mode oracle_or_gold \
  --changed_slot_weight 4.0 \
  --unchanged_value_loss_weight 0.05 \
  --edit_gate_pos_weight 8.0 \
  --edit_threshold_grid 0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7 \
  --max_length 384 \
  --seed 66 \
  --eval_seed 66001
```

## Analysis

```bash
python experiments/qwen_slot_repair_distillation/src/analyze_qwen_slot_repair_distillation.py
```

Key outputs:

- `analysis/summary.md`
- `analysis/final_metrics.csv`
- `analysis/all_final_metrics.csv`
- `analysis/figures/executor_accuracy.png`
- `analysis/figures/paired_details.png`
- `analysis/figures/training_curve.png`
- `analysis/figures/iteration_summary.png`
- `reports/qwen_slot_repair_distillation_paper.md`
- `reports/qwen_slot_repair_distillation_paper.html`
- `reports/qwen_slot_repair_distillation_experiment_log.md`
- `checkpoint_manifest.csv`

