# Experiment Log

## 2026-06-23

Created a standalone hidden VM curriculum-repair experiment.

Design:

- Use the same typed hidden VM families inside this experiment: arithmetic chains, calendar shifts, unit transforms, list aggregation, boolean thresholding, and lookup/adjust rules.
- Train `Qwen/Qwen3-4B` with QLoRA and a hidden compiler head.
- Use a staged length curriculum so the model first sees shorter programs, then length-6 programs.
- Evaluate fresh standard, paraphrase, paired, length-8 hard, and length-10 harder splits.
- Add a verifier-guided local repair pass that edits predicted hidden programs and keeps candidates whose deterministic execution matches the known answer.
- Distill verified repair targets back into the compiler with trace-style value losses, while avoiding stale token-position losses for repaired slots.

Initial planned runs:

- `smoke_curriculum_repair`: tiny end-to-end check.
- `pilot_curriculum_repair_l6`: small run to verify curriculum and repair target construction.
- `main_curriculum_trace_s512`: trace/curriculum control without repair distillation.
- `main_curriculum_repair_s512`: trace/curriculum plus verifier-guided repair distillation.

Smoke run:

```bash
python experiments/qwen_hidden_vm_curriculum_repair/src/qwen_hidden_vm_curriculum_repair_experiment.py \
  --run_name smoke_curriculum_repair --variant trace \
  --train_examples 12 --val_examples 6 --eval_examples 6 --eval_pairs 4 --domain_eval_examples 2 \
  --train_steps 2 --repair_steps 1 --train_batch_size 1 --eval_batch_size 2 \
  --max_steps 4 --train_max_len 3 --eval_length 3 --hard_length 4 --harder_length 0 --max_length 384 \
  --curriculum_schedule 2:1,3:2 --repair_source_examples 12 --repair_train_topk 2 --repair_eval_topk 2 --repair_max_edits 1 \
  --log_every 1 --seed 101 --eval_seed 101001
```

Smoke result:

- Completed successfully in 16.8 seconds.
- Verified local repair metrics were produced on every eval split.
- The repair-target phase built 4 targets from 12 source examples, with verified repairs found for 50.0% of source examples and changed-program repairs for 33.3%.
- Checkpoint and run CSVs were written under the new standalone artifact layout.

Pilot 1:

```bash
python experiments/qwen_hidden_vm_curriculum_repair/src/qwen_hidden_vm_curriculum_repair_experiment.py \
  --run_name pilot_curriculum_repair_l6_s256 --variant trace \
  --train_examples 256 --val_examples 72 --eval_examples 72 --eval_pairs 48 --domain_eval_examples 16 \
  --train_steps 260 --repair_steps 120 --train_batch_size 2 --eval_batch_size 4 \
  --max_steps 10 --train_max_len 6 --eval_length 6 --hard_length 8 --harder_length 10 --max_length 768 \
  --curriculum_schedule 4:120,6:260 --repair_source_examples 256 --repair_train_topk 3 --repair_eval_topk 3 --repair_max_edits 2 \
  --lr 5e-5 --executor_loss_weight 0.2 --state_loss_weight 0.05 \
  --log_every 40 --seed 102 --eval_seed 102001
```

Pilot 1 result:

- Completed in 301.5 seconds.
- Fresh paired hidden-VM executor accuracy: 45.8%; verified local repair: 100.0%.
- Hard length-8 standard executor accuracy: 23.6%; verified local repair: 100.0%.
- Harder length-10 standard executor accuracy: 19.4%; verified local repair: 98.6%.
- The repair target builder found verified candidates for 99.6% of source examples, but `repair_only` without unchanged programs produced only 86 training targets from 256 source examples.
- Interpretation: local repair headroom is extremely high, but changed-only repair distillation is too narrow and does not by itself solve the argmax compiler.

Pilot 2 change:

- Include unchanged verified programs in the repair target set.
- Add an eval at the transition point before repair distillation, so the repair phase can be measured rather than inferred.

Pilot 2:

```bash
python experiments/qwen_hidden_vm_curriculum_repair/src/qwen_hidden_vm_curriculum_repair_experiment.py \
  --run_name pilot_curriculum_repair_keep_l6_s256 --variant trace \
  --train_examples 256 --val_examples 72 --eval_examples 72 --eval_pairs 48 --domain_eval_examples 16 \
  --train_steps 260 --repair_steps 120 --train_batch_size 2 --eval_batch_size 4 \
  --max_steps 10 --train_max_len 6 --eval_length 6 --hard_length 8 --harder_length 10 --max_length 768 \
  --curriculum_schedule 4:120,6:260 --repair_source_examples 256 --repair_train_topk 3 --repair_eval_topk 3 --repair_max_edits 2 --repair_include_unchanged \
  --lr 5e-5 --executor_loss_weight 0.2 --state_loss_weight 0.05 \
  --log_every 40 --eval_every 260 --seed 103 --eval_seed 103001
```

Pilot 2 result:

- Completed in 375.0 seconds.
- Transition-point fresh paired executor accuracy before repair distillation: 44.8%.
- Final fresh paired executor accuracy after repair distillation: 61.5%.
- Transition-point hard length-8 standard executor accuracy: 36.1%; final: 62.5%.
- Transition-point harder length-10 standard executor accuracy: 11.1%; final: 44.4%.
- Repair target builder produced 253 targets from 256 source examples, with verified repairs found for 98.8% and changed-program repairs for 26.6%.
- Interpretation: retaining unchanged verified programs makes repair distillation dense enough to improve the argmax compiler. Use this construction for the main repair run.

Main run plan:

- `main_curriculum_trace_s512`: trace/curriculum control, no repair distillation.
- `main_curriculum_repair_s512`: same trace/curriculum prefix and seed, then repair distillation with unchanged verified programs included.

Main trace control:

```bash
python experiments/qwen_hidden_vm_curriculum_repair/src/qwen_hidden_vm_curriculum_repair_experiment.py \
  --run_name main_curriculum_trace_s512 --variant trace \
  --train_examples 512 --val_examples 144 --eval_examples 192 --eval_pairs 128 --domain_eval_examples 32 \
  --train_steps 700 --repair_steps 0 --train_batch_size 2 --eval_batch_size 8 \
  --max_steps 10 --train_max_len 6 --eval_length 6 --hard_length 8 --harder_length 10 --max_length 768 \
  --curriculum_schedule 4:240,6:700 --repair_source_examples 512 --repair_train_topk 3 --repair_eval_topk 3 --repair_max_edits 2 --repair_include_unchanged \
  --lr 5e-5 --executor_loss_weight 0.2 --state_loss_weight 0.05 \
  --log_every 70 --eval_every 700 --seed 104 --eval_seed 104001
```

Main trace result:

- Completed in 605.8 seconds.
- Fresh paired executor accuracy: 71.9%; verified local repair: 98.8%.
- Hard length-8 standard executor accuracy: 55.2%; verified local repair: 100.0%.
- Hard length-8 paraphrase executor accuracy: 41.1%; verified local repair: 99.0%.
- Harder length-10 standard executor accuracy: 30.2%; verified local repair: 99.5%.
- Harder length-10 paraphrase executor accuracy: 16.7%; verified local repair: 99.0%.

Main repair-distillation run:

```bash
python experiments/qwen_hidden_vm_curriculum_repair/src/qwen_hidden_vm_curriculum_repair_experiment.py \
  --run_name main_curriculum_repair_s512 --variant trace \
  --train_examples 512 --val_examples 144 --eval_examples 192 --eval_pairs 128 --domain_eval_examples 32 \
  --train_steps 700 --repair_steps 220 --train_batch_size 2 --eval_batch_size 8 \
  --max_steps 10 --train_max_len 6 --eval_length 6 --hard_length 8 --harder_length 10 --max_length 768 \
  --curriculum_schedule 4:240,6:700 --repair_source_examples 512 --repair_train_topk 3 --repair_eval_topk 3 --repair_max_edits 2 --repair_include_unchanged \
  --lr 5e-5 --executor_loss_weight 0.2 --state_loss_weight 0.05 \
  --log_every 70 --eval_every 700 --seed 104 --eval_seed 104001
```

Main repair result:

- Completed in 809.1 seconds.
- The curriculum prefix exactly matched the trace control at step 700: fresh paired 71.9%, hard length-8 standard 55.2%, harder length-10 standard 30.2%.
- Repair target builder produced 512 targets from 512 examples, with verified targets found for 100.0% and changed-program repairs for only 11.3%.
- After 220 repair-distillation steps, fresh paired executor accuracy fell to 35.2%.
- Hard length-8 standard fell to 10.4%; hard length-8 paraphrase fell to 14.6%.
- Harder length-10 standard fell to 9.4%; harder length-10 paraphrase fell to 8.9%.
- Verified local repair remained high at evaluation time, but the learned argmax compiler was damaged.

Main interpretation:

The curriculum itself is a positive result: training on lengths 1-6 produces nontrivial hard length-8/10 transfer in the trace-only control. Target-aware local repair reveals enormous headroom, often near 99-100%. But naive repair distillation from final-answer-verified programs is unsafe at scale: because many verified programs are non-canonical and the repair phase disables token-position selection losses, it can move the compiler away from the stable prompt-to-slot policy learned by trace supervision.

Reports to generate:

- `analysis/summary.md`
- `reports/qwen_hidden_vm_curriculum_repair_paper.md`
- `reports/qwen_hidden_vm_curriculum_repair_paper.html`
- figures under `analysis/figures/`
- `checkpoint_manifest.csv`
