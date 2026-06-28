# Qwen Verifier-Guided Slot Repair

Standalone experiment testing whether exact-answer verifier search can repair near-miss compiled programs from a QLoRA-adapted `Qwen/Qwen3-4B` numeric-copy compiler.

The compiler reads prompt hidden states, copies an initial value, operation sequence, and operation arguments from token-level maps, and executes the copied program modulo 97. The repair evaluator searches small local edits around the copied program and keeps the highest-prior candidate whose intermediate state trajectory satisfies the verifier.

## Main Question

How much length-24 exact execution is recoverable if the compiler is almost right and a state-trajectory verifier can select locally repaired candidates?

## Main Result

On fresh paired length-24 programs, the selected Qwen compiler scored 27.5% exact execution before repair and 91.0% after top-3/two-edit state-verifier repair. Repaired program exact was 90.0%, so the search usually recovered the true program rather than only an equivalent final answer. A one-edit ablation reached 70.5%.

The result is a headroom result, not a deployable inference recipe: the primary verifier uses the true intermediate state trajectory.

## Layout

```text
src/qwen_verifier_guided_slot_repair_experiment.py  training, evaluation, and repair search
src/analyze_qwen_verifier_guided_slot_repair.py     aggregation, selected checkpoint table, and plots
src/evaluate_selected_qwen_verifier_guided_slot_repair.py
                                                     fresh retest for selected checkpoints
runs/                                                lightweight JSON and CSV outputs
analysis/                                            aggregate CSVs and generated figures
reports/                                             experiment log and standalone write-up
checkpoint_manifest.csv                              generated list of large checkpoint files
```

Large checkpoints are stored outside the experiment directory:

```text
large_artifacts/qwen_verifier_guided_slot_repair/checkpoints/
```

## Main Metrics

- `executor_accuracy`: exact final answer from the unrepaired compiled program.
- `repair_executor_accuracy`: exact final answer after verifier-guided local repair.
- `program_exact`: exact unrepaired compiled program.
- `repair_program_exact`: exact repaired compiled program.
- `repair_found_fraction`: fraction of examples with at least one state-verifier-satisfying local candidate.
- `repair_changed_fraction`: fraction where repair selected a different program.
- `repair_pair_state_consistency`: paired standard/paraphrase consistency after repair.

## Reading Order

1. `reports/qwen_verifier_guided_slot_repair_experiment_log.md`
2. `analysis/summary.md`
3. `analysis/selected_checkpoints.csv`
4. `analysis/selected_retest_metrics.csv`
5. `analysis/selected_retest_metrics_one_edit.csv`
6. `analysis/final_metrics.csv`
7. `reports/qwen_verifier_guided_slot_repair_paper.md`
