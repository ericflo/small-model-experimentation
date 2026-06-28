# Experiment Log

## 2026-06-23

Created a standalone hidden VM on-policy canonical repair experiment.

Design:

- Use `Qwen/Qwen3-4B` with QLoRA and a hidden typed VM compiler.
- Train on mixed task families: arithmetic chains, calendar shifts, unit transforms, list aggregation, boolean thresholding, and lookup/adjust rules.
- Use a length curriculum before repair training.
- Evaluate standard, paraphrase, paired, hard length, harder length, and per-domain splits.
- Generate local repair candidates from the current compiler policy.
- Accept a repair target only when the candidate's full intermediate state trajectory matches the canonical trajectory, not merely when the final answer matches.
- Train directly on target slot values while retaining gold trace, gold selection, executor, and state losses as stabilizers.
- Keep large model artifacts under `large_artifacts/`.

Initial planned runs:

- `smoke_onpolicy_canonical_repair`: tiny end-to-end check.
- `pilot_onpolicy_canonical_repair_s192`: small run to verify target quality and stability.
- `main_trace_control_s512`: curriculum-only control.
- `main_gold_control_s512`: curriculum plus one extra gold-only on-policy-format epoch.
- `main_repair_only_s512`: curriculum plus canonical repair-only targets.
- `main_repair_or_gold_s512`: curriculum plus canonical repair targets with gold fallback.

Smoke run:

```bash
python experiments/qwen_hidden_vm_onpolicy_canonical_repair/src/qwen_hidden_vm_onpolicy_canonical_repair_experiment.py \
  --run_name smoke_onpolicy_canonical_repair --variant trace \
  --train_examples 12 --val_examples 6 --eval_examples 6 --eval_pairs 4 --domain_eval_examples 2 \
  --train_steps 2 --onpolicy_rounds 1 --epochs_per_round 1 --train_batch_size 1 --eval_batch_size 2 \
  --max_steps 4 --train_max_len 3 --eval_length 3 --hard_length 4 --harder_length 0 --max_length 384 \
  --curriculum_schedule 2:1,3:2 --onpolicy_source_examples 12 --repair_train_topk 2 --repair_eval_topk 2 --repair_max_edits 1 \
  --target_mode repair_or_gold --repair_verifier_mode state \
  --log_every 1 --eval_every 2 --seed 121 --eval_seed 121001
```

Smoke result:

- Completed successfully in 29.2 seconds.
- The on-policy target pass produced active rows for 100.0% of the 12 source examples.
- Canonical state-verified repairs were found for 41.7% of source examples.
- Changed-program repairs were 33.3%; program-exact repaired targets were 25.0%.
- Final smoke fresh paired hidden-VM accuracy was 25.0%; this is only a wiring check.
- Analyzer generated markdown, HTML, CSV summaries, and charts.

Pilot 1:

```bash
python experiments/qwen_hidden_vm_onpolicy_canonical_repair/src/qwen_hidden_vm_onpolicy_canonical_repair_experiment.py \
  --run_name pilot_onpolicy_canonical_repair_s192 --variant trace \
  --train_examples 192 --val_examples 72 --eval_examples 72 --eval_pairs 48 --domain_eval_examples 16 \
  --train_steps 260 --onpolicy_rounds 1 --epochs_per_round 1 --train_batch_size 2 --eval_batch_size 4 \
  --max_steps 10 --train_max_len 6 --eval_length 6 --hard_length 8 --harder_length 10 --max_length 768 \
  --curriculum_schedule 4:120,6:260 --onpolicy_source_examples 192 --repair_train_topk 3 --repair_eval_topk 3 --repair_max_edits 2 \
  --target_mode repair_or_gold --repair_verifier_mode state \
  --lr 5e-5 --executor_loss_weight 0.2 --state_loss_weight 0.05 \
  --gold_trace_loss_weight 0.25 --gold_selection_loss_weight 0.25 \
  --log_every 40 --eval_every 260 --seed 122 --eval_seed 122001
```

Pilot 1 result:

- Completed in 367.6 seconds.
- Curriculum transition before on-policy training was strong: fresh paired 74.0%, hard length-8 standard 63.9%, harder length-10 standard 43.1%.
- State-verified local repair at transition was high: fresh paired 99.0%, hard length-8 standard 95.8%, harder length-10 standard 72.2%.
- Canonical on-policy targets were dense: active rows 100.0%, found 100.0%, changed 17.7%, program-exact 98.4%.
- After one full on-policy epoch, fresh paired fell to 50.0%, hard length-8 standard to 29.2%, and harder length-10 standard to 23.6%.
- Interpretation: target availability is not the problem; the repair phase is too destabilizing at the initial learning rate and regularization strength.

Pilot 2 change:

- Add `--onpolicy_lr_multiplier`.
- Keep the same curriculum prefix seed, but run the on-policy epoch at a lower effective LR.
- Increase gold trace and gold selection regularization during on-policy training.

Pilot 2:

```bash
python experiments/qwen_hidden_vm_onpolicy_canonical_repair/src/qwen_hidden_vm_onpolicy_canonical_repair_experiment.py \
  --run_name pilot_onpolicy_canonical_stable_s192 --variant trace \
  --train_examples 192 --val_examples 72 --eval_examples 72 --eval_pairs 48 --domain_eval_examples 16 \
  --train_steps 260 --onpolicy_rounds 1 --epochs_per_round 1 --onpolicy_lr_multiplier 0.2 \
  --train_batch_size 2 --eval_batch_size 4 \
  --max_steps 10 --train_max_len 6 --eval_length 6 --hard_length 8 --harder_length 10 --max_length 768 \
  --curriculum_schedule 4:120,6:260 --onpolicy_source_examples 192 --repair_train_topk 3 --repair_eval_topk 3 --repair_max_edits 2 \
  --target_mode repair_or_gold --repair_verifier_mode state \
  --lr 5e-5 --executor_loss_weight 0.2 --state_loss_weight 0.05 \
  --repair_trace_loss_weight 0.25 --gold_trace_loss_weight 1.0 --gold_selection_loss_weight 1.0 \
  --log_every 40 --eval_every 260 --seed 122 --eval_seed 122001
```

Pilot 2 result:

- Completed in 361.3 seconds.
- The curriculum prefix matched Pilot 1 on the important splits: fresh paired 74.0%, hard length-8 standard 63.9%, harder length-10 standard 43.1%.
- Target quality also matched: active 100.0%, found 100.0%, changed 17.7%, program-exact 98.4%.
- After the gentler on-policy epoch, fresh paired still fell to 47.9%, hard length-8 standard to 34.7%, and harder length-10 standard to 16.7%.
- Interpretation: lower LR and stronger gold regularization are not enough. The likely failure is representation drift or over-updating during the repair phase.

Pilot 3 change:

- Freeze Qwen/LoRA during on-policy training and update only the compiler head.
- Reset the optimizer before on-policy training.
- Keep the same curriculum prefix seed for direct comparison.

Pilot 3:

```bash
python experiments/qwen_hidden_vm_onpolicy_canonical_repair/src/qwen_hidden_vm_onpolicy_canonical_repair_experiment.py \
  --run_name pilot_onpolicy_canonical_headonly_s192 --variant trace \
  --train_examples 192 --val_examples 72 --eval_examples 72 --eval_pairs 48 --domain_eval_examples 16 \
  --train_steps 260 --onpolicy_rounds 1 --epochs_per_round 1 --onpolicy_lr_multiplier 0.2 --no-onpolicy_train_lora \
  --train_batch_size 2 --eval_batch_size 4 \
  --max_steps 10 --train_max_len 6 --eval_length 6 --hard_length 8 --harder_length 10 --max_length 768 \
  --curriculum_schedule 4:120,6:260 --onpolicy_source_examples 192 --repair_train_topk 3 --repair_eval_topk 3 --repair_max_edits 2 \
  --target_mode repair_or_gold --repair_verifier_mode state \
  --lr 5e-5 --executor_loss_weight 0.2 --state_loss_weight 0.05 \
  --repair_trace_loss_weight 0.25 --gold_trace_loss_weight 1.0 --gold_selection_loss_weight 1.0 \
  --log_every 40 --eval_every 260 --seed 122 --eval_seed 122001
```

Pilot 3 result:

- Completed in 362.9 seconds.
- Same curriculum transition as Pilot 1 and 2: fresh paired 74.0%, hard length-8 standard 63.9%, harder length-10 standard 43.1%.
- Same target quality: active 100.0%, found 100.0%, changed 17.7%, program-exact 98.4%.
- After head-only on-policy training, fresh paired fell to 63.5%, hard length-8 standard to 48.6%, and harder length-10 standard to 36.1%.
- Interpretation: freezing Qwen/LoRA helps, but a full on-policy pass is still too large. The repair phase should be a limited nudge, not a full epoch.

Pilot 4 change:

- Add `--onpolicy_max_batches`.
- Keep head-only on-policy training, but cap the repair phase to 24 batches.

Pilot 4:

```bash
python experiments/qwen_hidden_vm_onpolicy_canonical_repair/src/qwen_hidden_vm_onpolicy_canonical_repair_experiment.py \
  --run_name pilot_onpolicy_canonical_headonly_cap24_s192 --variant trace \
  --train_examples 192 --val_examples 72 --eval_examples 72 --eval_pairs 48 --domain_eval_examples 16 \
  --train_steps 260 --onpolicy_rounds 1 --epochs_per_round 1 --onpolicy_lr_multiplier 0.2 --no-onpolicy_train_lora --onpolicy_max_batches 24 \
  --train_batch_size 2 --eval_batch_size 4 \
  --max_steps 10 --train_max_len 6 --eval_length 6 --hard_length 8 --harder_length 10 --max_length 768 \
  --curriculum_schedule 4:120,6:260 --onpolicy_source_examples 192 --repair_train_topk 3 --repair_eval_topk 3 --repair_max_edits 2 \
  --target_mode repair_or_gold --repair_verifier_mode state \
  --lr 5e-5 --executor_loss_weight 0.2 --state_loss_weight 0.05 \
  --repair_trace_loss_weight 0.25 --gold_trace_loss_weight 1.0 --gold_selection_loss_weight 1.0 \
  --log_every 40 --eval_every 260 --seed 122 --eval_seed 122001
```

Pilot 4 result:

- Completed in 325.7 seconds.
- Same curriculum transition: fresh paired 74.0%, hard length-8 standard 63.9%, harder length-10 standard 43.1%.
- Same target quality: active 100.0%, found 100.0%, changed 17.7%, program-exact 98.4%.
- After 24 head-only on-policy batches, fresh paired fell to 60.4%, hard length-8 standard to 47.2%, and harder length-10 standard to 31.9%.
- Interpretation: limiting repair batches did not solve the problem. The main runs should treat canonical on-policy repair as a hypothesis under test, not as an assumed improvement.

Main run plan:

- Use matched seed `124` and eval seed `124001`.
- Run a trace-only control.
- Run a gold-only head-only on-policy control.
- Run a repair-only head-only on-policy arm.
- Run a repair-or-gold head-only on-policy arm.
- Select the deployable result by fresh paired and hard-length accuracy, not by repair headroom alone.

Main trace control:

```bash
python experiments/qwen_hidden_vm_onpolicy_canonical_repair/src/qwen_hidden_vm_onpolicy_canonical_repair_experiment.py \
  --run_name main_trace_control_s512 --variant trace \
  --train_examples 512 --val_examples 144 --eval_examples 192 --eval_pairs 128 --domain_eval_examples 32 \
  --train_steps 700 --onpolicy_rounds 0 --train_batch_size 2 --eval_batch_size 8 \
  --max_steps 10 --train_max_len 6 --eval_length 6 --hard_length 8 --harder_length 10 --max_length 768 \
  --curriculum_schedule 4:240,6:700 --onpolicy_source_examples 512 --repair_train_topk 3 --repair_eval_topk 3 --repair_max_edits 2 \
  --target_mode repair_or_gold --repair_verifier_mode state \
  --lr 5e-5 --executor_loss_weight 0.2 --state_loss_weight 0.05 \
  --log_every 70 --eval_every 700 --seed 124 --eval_seed 124001
```

Main trace result:

- Completed in 619.6 seconds.
- Fresh paired hidden-VM accuracy: 59.0%; state-verified repair headroom: 89.5%.
- Hard length-8 standard: 54.2%; hard length-8 paraphrase: 36.5%.
- Harder length-10 standard: 35.9%; harder length-10 paraphrase: 20.3%.

Main gold-only control:

```bash
python experiments/qwen_hidden_vm_onpolicy_canonical_repair/src/qwen_hidden_vm_onpolicy_canonical_repair_experiment.py \
  --run_name main_gold_control_s512 --variant trace \
  --train_examples 512 --val_examples 144 --eval_examples 192 --eval_pairs 128 --domain_eval_examples 32 \
  --train_steps 700 --onpolicy_rounds 1 --epochs_per_round 1 --onpolicy_lr_multiplier 0.2 --no-onpolicy_train_lora \
  --train_batch_size 2 --eval_batch_size 8 \
  --max_steps 10 --train_max_len 6 --eval_length 6 --hard_length 8 --harder_length 10 --max_length 768 \
  --curriculum_schedule 4:240,6:700 --onpolicy_source_examples 512 --repair_train_topk 3 --repair_eval_topk 3 --repair_max_edits 2 \
  --target_mode gold_only --repair_verifier_mode state \
  --lr 5e-5 --executor_loss_weight 0.2 --state_loss_weight 0.05 \
  --repair_trace_loss_weight 0.25 --gold_trace_loss_weight 1.0 --gold_selection_loss_weight 1.0 \
  --log_every 70 --eval_every 700 --seed 124 --eval_seed 124001
```

Main gold-only result:

- Completed in 821.4 seconds.
- Target pass sanity check: active 100.0%, changed 0.0%, program-exact 100.0%.
- Fresh paired hidden-VM accuracy: 60.9%.
- Hard length-8 standard: 53.6%; hard length-8 paraphrase: 27.6%.
- Harder length-10 standard: 31.8%; harder length-10 paraphrase: 16.7%.
- Interpretation: a head-only extra gold pass gives a small fresh paired gain but weakens hard-length paraphrase and length-10 robustness.

Main repair-only:

```bash
python experiments/qwen_hidden_vm_onpolicy_canonical_repair/src/qwen_hidden_vm_onpolicy_canonical_repair_experiment.py \
  --run_name main_repair_only_s512 --variant trace \
  --train_examples 512 --val_examples 144 --eval_examples 192 --eval_pairs 128 --domain_eval_examples 32 \
  --train_steps 700 --onpolicy_rounds 1 --epochs_per_round 1 --onpolicy_lr_multiplier 0.2 --no-onpolicy_train_lora \
  --train_batch_size 2 --eval_batch_size 8 \
  --max_steps 10 --train_max_len 6 --eval_length 6 --hard_length 8 --harder_length 10 --max_length 768 \
  --curriculum_schedule 4:240,6:700 --onpolicy_source_examples 512 --repair_train_topk 3 --repair_eval_topk 3 --repair_max_edits 2 \
  --target_mode repair_only --repair_verifier_mode state \
  --lr 5e-5 --executor_loss_weight 0.2 --state_loss_weight 0.05 \
  --repair_trace_loss_weight 0.25 --gold_trace_loss_weight 1.0 --gold_selection_loss_weight 1.0 \
  --log_every 70 --eval_every 700 --seed 124 --eval_seed 124001
```

Main repair-only result:

- Completed in 834.1 seconds.
- Target pass: active 99.6%, found 99.6%, changed 17.4%, program-exact 96.5%.
- Fresh paired hidden-VM accuracy: 60.9%.
- Hard length-8 standard: 53.1%; hard length-8 paraphrase: 27.6%.
- Harder length-10 standard: 31.8%; harder length-10 paraphrase: 16.7%.
- Interpretation: repair-only does not separate from the gold-only control.

Main repair-or-gold:

```bash
python experiments/qwen_hidden_vm_onpolicy_canonical_repair/src/qwen_hidden_vm_onpolicy_canonical_repair_experiment.py \
  --run_name main_repair_or_gold_s512 --variant trace \
  --train_examples 512 --val_examples 144 --eval_examples 192 --eval_pairs 128 --domain_eval_examples 32 \
  --train_steps 700 --onpolicy_rounds 1 --epochs_per_round 1 --onpolicy_lr_multiplier 0.2 --no-onpolicy_train_lora \
  --train_batch_size 2 --eval_batch_size 8 \
  --max_steps 10 --train_max_len 6 --eval_length 6 --hard_length 8 --harder_length 10 --max_length 768 \
  --curriculum_schedule 4:240,6:700 --onpolicy_source_examples 512 --repair_train_topk 3 --repair_eval_topk 3 --repair_max_edits 2 \
  --target_mode repair_or_gold --repair_verifier_mode state \
  --lr 5e-5 --executor_loss_weight 0.2 --state_loss_weight 0.05 \
  --repair_trace_loss_weight 0.25 --gold_trace_loss_weight 1.0 --gold_selection_loss_weight 1.0 \
  --log_every 70 --eval_every 700 --seed 124 --eval_seed 124001
```

Main repair-or-gold result:

- Completed in 836.3 seconds.
- Target pass: active 100.0%, found 99.6%, changed 17.4%, program-exact 96.5%.
- Fresh paired hidden-VM accuracy: 60.9%; state-verified repair headroom: 89.1%.
- Hard length-8 standard: 53.1%; hard length-8 paraphrase: 27.6%.
- Harder length-10 standard: 31.8%; harder length-10 paraphrase: 16.7%.
- Interpretation: canonical repair targets are high quality, but folding them back into the compiler does not outperform gold-only training and reduces hard-length robustness relative to trace-only.

Main conclusion:

Canonical on-policy repair should not be scaled in this form. The state verifier is valuable as a headroom or candidate-selection mechanism, but direct distillation into the compiler behaves like an extra gold pass and weakens the length-generalization stress tests.
