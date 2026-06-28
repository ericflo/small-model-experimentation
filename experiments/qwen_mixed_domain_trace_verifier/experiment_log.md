# Qwen Mixed-Domain Trace Verifier Experiment Log

## 2026-06-23

Objective: train and evaluate a learned verifier/reranker for local candidate
traces emitted by a frozen Qwen-attached hidden VM compiler on mixed deterministic
domains. The report must be standalone and must not depend on earlier experiment
narrative.

Initial setup:

- Created standalone experiment directory.
- Heavy artifacts will live under
  `/workspace/large_artifacts/qwen_mixed_domain_trace_verifier/checkpoints`.
- Planned primary run: freeze the existing Qwen-attached mixed-domain compiler,
  enumerate local candidates, train a small trace verifier, evaluate base/prior,
  soft-executor support, learned verifier, paired reranker, and oracle ceiling.
- Copied the frozen mixed-domain trace compiler into this experiment's own
  large-artifact area:
  `/workspace/large_artifacts/qwen_mixed_domain_trace_verifier/checkpoints/fixed_mixed_vm_trace_compiler_s512`.
- Added local standalone source:
  - `src/mixed_vm_core.py` contains the mixed-domain VM generator, compiler, and
    executor code used by this experiment.
  - `src/qwen_mixed_domain_trace_verifier_experiment.py` loads the frozen
    compiler, builds local candidate neighborhoods, trains a candidate-trace
    verifier, evaluates base/prior/soft-trace/learned/pair-rerank/oracle
    selectors, and writes run metrics plus checkpoint manifests.

Smoke run:

- Command: `python src/qwen_mixed_domain_trace_verifier_experiment.py --run_name smoke_mixed_domain_trace_verifier --train_examples 12 --val_examples 6 --eval_examples 6 --eval_pairs 4 --domain_eval_examples 0 --verifier_epochs 1 --qwen_batch_size 4 --repair_topk 2 --repair_max_edits 1 --repair_max_pair_arg_slots 4 --trace_d_model 64 --trace_layers 1 --trace_heads 4`
- Result: completed end to end. The run wrote `runs/smoke_mixed_domain_trace_verifier/metrics.csv`,
  `runs/smoke_mixed_domain_trace_verifier/results.json`, and a verifier
  checkpoint under `large_artifacts/qwen_mixed_domain_trace_verifier/checkpoints/smoke_mixed_domain_trace_verifier/`.
- Takeaway: loader, local candidate construction, feature tensor dimensions,
  verifier training, pair reranking path, and artifact writing are functional.

Pilot 1:

- Command: `python src/qwen_mixed_domain_trace_verifier_experiment.py --run_name pilot_mixed_domain_trace_verifier_s96 --train_examples 96 --val_examples 32 --eval_examples 48 --eval_pairs 32 --domain_eval_examples 12 --verifier_epochs 6 --qwen_batch_size 8 --repair_topk 3 --repair_max_edits 2 --repair_max_pair_arg_slots 10 --trace_d_model 96 --trace_layers 2 --trace_heads 4`
- Result: oracle headroom was large but the learned verifier selected the base
  candidate on every split (`learned_changed_fraction=0.0` on final metrics).
- Diagnosis: verifier training was dominated by base-correct candidate groups;
  the model learned preservation but not repair selection.
- Iteration: added group-level loss weights so base-correct groups retain a
  preservation objective, while wrong-but-repairable groups carry more gradient.

Pilot 2:

- Command: `python src/qwen_mixed_domain_trace_verifier_experiment.py --run_name pilot_weighted_len6_trace_verifier_s128 --train_examples 128 --val_examples 48 --eval_examples 64 --eval_pairs 40 --domain_eval_examples 12 --train_min_len 6 --train_max_len 6 --verifier_epochs 12 --qwen_batch_size 8 --repair_topk 3 --repair_max_edits 2 --repair_max_pair_arg_slots 10 --trace_d_model 128 --trace_layers 2 --trace_heads 4 --base_positive_group_weight 0.25 --repairable_group_weight 8.0 --no_positive_group_weight 1.0`
- Result: learned selection improved length-6 fresh paired from 40.0% to 43.8%;
  pair reranking improved it to 46.25%. Hard length-8/10 did not improve.
- Takeaway: weighted repair training can move off the base candidate, but a
  length-6-only verifier does not learn enough about longer traces.

Pilot 3:

- Command: `python src/qwen_mixed_domain_trace_verifier_experiment.py --run_name pilot_weighted_len6_8_val8_trace_verifier_s160 --train_examples 160 --val_examples 48 --eval_examples 64 --eval_pairs 40 --domain_eval_examples 12 --train_min_len 6 --train_max_len 8 --val_length 8 --verifier_epochs 12 --qwen_batch_size 8 --repair_topk 3 --repair_max_edits 2 --repair_max_pair_arg_slots 10 --trace_d_model 128 --trace_layers 2 --trace_heads 4 --base_positive_group_weight 0.25 --repairable_group_weight 8.0 --no_positive_group_weight 1.0`
- Result: length-8 validation improved from 33.3% to 41.7%, hard standard
  length 8 improved from 40.6% to 43.8%, hard paraphrase length 8 from 25.0%
  to 32.8%, and harder standard length 10 from 14.1% to 17.2%. Easy length-6
  standard degraded, showing an over-editing tradeoff.
- Final plan: run two final arms, one optimized for length-6 repair and one
  optimized for length-8/generalization, then compare both against the oracle
  candidate ceiling.

Final arm A: length-6 verifier

- Command: `python src/qwen_mixed_domain_trace_verifier_experiment.py --run_name main_len6_weighted_trace_verifier_s512 --train_examples 512 --val_examples 128 --eval_examples 192 --eval_pairs 128 --domain_eval_examples 48 --train_min_len 6 --train_max_len 6 --verifier_epochs 18 --qwen_batch_size 8 --repair_topk 3 --repair_max_edits 2 --repair_max_pair_arg_slots 10 --trace_d_model 128 --trace_layers 2 --trace_heads 4 --base_positive_group_weight 0.25 --repairable_group_weight 8.0 --no_positive_group_weight 1.0`
- Result: fresh paired length 6 improved from 47.3% base to 53.1% learned.
  Hard length 8 and harder length 10 improved slightly. Best validation epoch:
  16.

Final arm B: length-6-to-8 verifier selected on length 8

- Command: `python src/qwen_mixed_domain_trace_verifier_experiment.py --run_name main_len6_8_val8_weighted_trace_verifier_s512 --train_examples 512 --val_examples 128 --eval_examples 192 --eval_pairs 128 --domain_eval_examples 48 --train_min_len 6 --train_max_len 8 --val_length 8 --verifier_epochs 18 --qwen_batch_size 8 --repair_topk 3 --repair_max_edits 2 --repair_max_pair_arg_slots 10 --trace_d_model 128 --trace_layers 2 --trace_heads 4 --base_positive_group_weight 0.25 --repairable_group_weight 8.0 --no_positive_group_weight 1.0`
- Result: fresh paired length 6 improved from 46.1% base to 57.4% learned.
  Hard paraphrase length 8 improved from 22.9% to 30.7%, harder standard
  length 10 from 18.8% to 24.5%, and harder paraphrase length 10 from 3.6%
  to 5.7%. Best validation epoch: 12/17 tied at 35.9% length-8 validation.

Analysis and report:

- Added `src/analyze_qwen_mixed_domain_trace_verifier.py`.
- Generated aggregate metrics:
  - `analysis/final_metrics.csv`
  - `analysis/main_final_metrics.csv`
  - `analysis/all_final_metrics.csv`
  - `analysis/main_verifier_train_logs.csv`
- Generated figures:
  - `analysis/accuracy_by_split.png`
  - `analysis/oracle_gap_recovered.png`
  - `analysis/paired_consistency.png`
  - `analysis/domain_breakdown_len68.png`
  - `analysis/validation_curves.png`
- Generated standalone reports:
  - `reports/qwen_mixed_domain_trace_verifier_paper.md`
  - `reports/qwen_mixed_domain_trace_verifier_paper.html`
- Rebuilt `checkpoint_manifest.csv` with fixed compiler and verifier checkpoint
  paths plus byte sizes.
