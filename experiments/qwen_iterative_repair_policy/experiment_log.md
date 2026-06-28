# Qwen Iterative Repair Policy Experiment Log

## 2026-06-23

Objective: test whether repeated learned repairs over a hidden executable
program can recover exact execution from a frozen Qwen-attached compiler. The
experiment must be standalone, keep its own logs and artifacts, and produce
Markdown and HTML reports with charts.

Initial setup:

- Created `experiments/qwen_iterative_repair_policy/`.
- Created `large_artifacts/qwen_iterative_repair_policy/checkpoints/`.
- Localized a frozen compiler checkpoint into the new large-artifact namespace:
  `large_artifacts/qwen_iterative_repair_policy/checkpoints/fixed_compiler_step00800`.
- Seeded source from the existing Qwen numeric compiler utilities and began
  replacing one-shot editor evaluation with explicit iterative repair.

Smoke run:

```bash
python experiments/qwen_iterative_repair_policy/src/qwen_iterative_repair_policy_experiment.py \
  --run_name smoke_iterative_repair_policy \
  --train_examples 8 --val_examples 4 --eval_examples 4 --eval_pairs 4 \
  --editor_epochs 1 --qwen_batch_size 4 --repair_topk 2 --repair_max_edits 1 \
  --repair_max_pair_arg_slots 4 --editor_d_model 64 --editor_layers 1 \
  --editor_heads 4 --augment_cases_per_group 1 --augment_top_pool 4 \
  --include_target_noedit_case --stats_candidate_traces 2 \
  --train_select_k_values 1,2 --eval_k_values 0,1,2 \
  --max_edits_per_iteration 1 --max_length 384 --seed 201 --eval_seed 201001
```

Smoke result:

- Completed successfully in 12.9 seconds after model load.
- Wrote run metrics, train log, results JSON, and an iterative repair policy
  checkpoint.
- Verified K-sweep rows are emitted for `K=0,1,2`.
- Tiny split numbers are not interpreted; this was only a wiring check.

Pilot 1:

```bash
python experiments/qwen_iterative_repair_policy/src/qwen_iterative_repair_policy_experiment.py \
  --run_name pilot_iterative_k1_sparse_s96 \
  --train_examples 96 --val_examples 48 --eval_examples 64 --eval_pairs 64 \
  --editor_epochs 6 --qwen_batch_size 8 --repair_topk 3 --repair_max_edits 2 \
  --repair_max_pair_arg_slots 24 --editor_d_model 128 --editor_layers 2 \
  --editor_heads 4 --augment_cases_per_group 2 --augment_top_pool 16 \
  --include_target_noedit_case --stats_candidate_traces 6 \
  --train_select_k_values 1,2,3 --eval_k_values 0,1,2,3 \
  --max_edits_per_iteration 1 --unchanged_value_loss_weight 0.05 \
  --edit_gate_pos_weight 8.0 --changed_slot_weight 4.0 \
  --max_length 384 --seed 202 --eval_seed 202001
```

Pilot 1 result:

- Completed in 580.9 seconds.
- Validation improved from 27.1% base to 33.3% at `K=1`, but fresh transfer
  failed.
- Fresh standard length-24 fell from 31.2% base to 29.7%; fresh paraphrase fell
  from 23.4% to 15.6-17.2%; fresh paired fell from 32.0% to 27.3%.
- Interpretation: the policy learned a validation-specific sparse repair
  trigger. The next iteration should select checkpoints on a paired validation
  split and reduce noisy fallback edits when no local oracle candidate exists.

Iteration after pilot 1:

- Added optional `val_paired_len24` construction with `--val_pairs`.
- Added `--selection_split` so checkpoint selection can use paired validation.
- Changed selection priority to paired both-correct when the selection split is
  paired.

Pilot 2:

```bash
python experiments/qwen_iterative_repair_policy/src/qwen_iterative_repair_policy_experiment.py \
  --run_name pilot_iterative_paired_select_s96 \
  --train_examples 96 --val_examples 48 --val_pairs 48 \
  --selection_split val_paired_len24 --eval_examples 64 --eval_pairs 64 \
  --editor_epochs 8 --qwen_batch_size 8 --repair_topk 3 --repair_max_edits 2 \
  --repair_max_pair_arg_slots 24 --editor_d_model 128 --editor_layers 2 \
  --editor_heads 4 --augment_cases_per_group 3 --augment_top_pool 24 \
  --augment_repairable_only --include_target_noedit_case \
  --stats_candidate_traces 8 --train_select_k_values 1,2,3 \
  --eval_k_values 0,1,2,3 --max_edits_per_iteration 1 \
  --editor_target_mode oracle_or_base --unchanged_value_loss_weight 0.1 \
  --edit_gate_pos_weight 6.0 --changed_slot_weight 5.0 \
  --max_length 384 --seed 203 --eval_seed 203001
```

Pilot 2 result:

- Completed in 960.5 seconds.
- Paired validation selected a very conservative checkpoint: validation paired
  improved from 29.2% base to 31.2% at `K=1`, but fresh standard, paraphrase,
  and paired metrics all copied the base exactly.
- Interpretation: paired selection prevented the damage seen in pilot 1, but
  the direct raw-value edit policy did not learn robust edits.

Iteration after pilot 2:

- Added a candidate-scorer method. It trains a trace verifier over the local
  candidate set, then applies it iteratively by moving only to candidates within
  one edit of the current candidate at each step.
- Kept `K` as an explicit eval axis; `K=2` can reach two-edit candidates through
  one-edit transitions instead of selecting them directly at `K=1`.

Candidate-scorer smoke:

```bash
python experiments/qwen_iterative_repair_policy/src/qwen_iterative_repair_policy_experiment.py \
  --method candidate_scorer --run_name smoke_iterative_candidate_scorer \
  --train_examples 8 --val_examples 4 --eval_examples 4 --eval_pairs 4 \
  --verifier_epochs 1 --qwen_batch_size 4 --repair_topk 2 \
  --repair_max_edits 1 --repair_max_pair_arg_slots 4 --trace_d_model 64 \
  --trace_layers 1 --trace_heads 4 --eval_k_values 0,1,2 \
  --max_edits_per_iteration 1 --max_length 384 --seed 204 --eval_seed 204001
```

Candidate-scorer smoke result:

- Completed successfully in 9.3 seconds after model load.
- Verified verifier training, checkpoint writing, and iterative candidate
  K-sweep metrics.

Candidate-scorer pilot:

```bash
python experiments/qwen_iterative_repair_policy/src/qwen_iterative_repair_policy_experiment.py \
  --method candidate_scorer --run_name pilot_iterative_candidate_scorer_s128 \
  --train_examples 128 --val_examples 64 --val_pairs 64 \
  --selection_split val_paired_len24 --eval_examples 96 --eval_pairs 96 \
  --verifier_epochs 8 --qwen_batch_size 8 --repair_topk 3 --repair_max_edits 2 \
  --repair_max_pair_arg_slots 24 --trace_d_model 128 --trace_layers 2 \
  --trace_heads 4 --base_positive_group_weight 0.25 \
  --repairable_group_weight 8.0 --no_positive_group_weight 1.0 \
  --no_positive_base_weight 0.2 --eval_k_values 0,1,2,3 \
  --max_edits_per_iteration 1 --candidate_accept_margin 0.0 \
  --max_length 384 --seed 205 --eval_seed 205001
```

Candidate-scorer pilot result:

- Completed in 326.9 seconds.
- Fresh standard length-24 improved from 24.0% base to 35.4% at `K=2`.
- Fresh paraphrase length-24 improved from 30.2% base to 33.3% at `K=2`.
- Fresh paired length-24 improved from 29.2% base to 39.6% at `K=1`.
- Full learned scoring without iterative constraints reached 34.9% fresh paired,
  so the constrained iterative path gave an additional +4.7 pp on that split.
- Interpretation: candidate scoring is the first positive repair-policy arm.
  The effect is not monotonic in K on all splits, so the main run should report
  the full K sweep rather than one headline K.

Main run:

```bash
python experiments/qwen_iterative_repair_policy/src/qwen_iterative_repair_policy_experiment.py \
  --method candidate_scorer --run_name main_iterative_candidate_scorer_s384 \
  --train_examples 384 --val_examples 128 --val_pairs 128 \
  --selection_split val_paired_len24 --eval_examples 192 --eval_pairs 128 \
  --verifier_epochs 12 --qwen_batch_size 8 --repair_topk 3 \
  --repair_max_edits 2 --repair_max_pair_arg_slots 24 --trace_d_model 128 \
  --trace_layers 2 --trace_heads 4 --base_positive_group_weight 0.25 \
  --repairable_group_weight 8.0 --no_positive_group_weight 1.0 \
  --no_positive_base_weight 0.2 --eval_k_values 0,1,2,3 \
  --max_edits_per_iteration 1 --candidate_accept_margin 0.0 \
  --max_length 384 --seed 206 --eval_seed 206001
```

Main result:

- Completed in 757.3 seconds.
- Fresh standard length-24 improved from 29.7% base to 44.8% at `K=2`
  versus an 89.6% oracle.
- Fresh paraphrase length-24 improved from 29.7% base to 51.6% at `K=2`
  versus an 88.0% oracle.
- Fresh paired length-24 improved from 30.1% base to 52.7% at `K=2`
  versus an 89.1% oracle.
- On fresh paired prompts, unconstrained learned scoring reached 52.0%, so the
  iterative one-edit path slightly exceeded full learned selection while keeping
  the repair transition sparse.
- `K=3` did not improve over `K=2`, consistent with the candidate set being
  capped at two edits.

Report generation:

```bash
python experiments/qwen_iterative_repair_policy/src/analyze_qwen_iterative_repair_policy.py
```

Generated:

- `analysis/all_final_metrics.csv`
- `analysis/main_final_metrics.csv`
- `analysis/fresh_main_summary.csv`
- `analysis/verifier_train_logs.csv`
- `analysis/direct_policy_train_logs.csv`
- `analysis/figures/accuracy_by_k_main.png`
- `analysis/figures/fresh_accuracy_bars.png`
- `analysis/figures/oracle_gap_recovered.png`
- `analysis/figures/verifier_training_curve.png`
- `analysis/figures/iteration_path_comparison.png`
- `reports/qwen_iterative_repair_policy_paper.md`
- `reports/qwen_iterative_repair_policy_paper.html`
- `checkpoint_manifest.csv`
