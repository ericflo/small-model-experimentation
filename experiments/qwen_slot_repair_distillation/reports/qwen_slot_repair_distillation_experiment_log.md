# Qwen Slot Repair Distillation Experiment Log

## Objective

Train a deployable slot editor that distills local repair headroom into direct
corrected-program predictions for a frozen Qwen-attached numeric compiler. At
evaluation time, the editor emits one program and does not receive target
answers, target states, or a candidate list to rerank.

## File Layout

- Source: `experiments/qwen_slot_repair_distillation/src/`
- Runs: `experiments/qwen_slot_repair_distillation/runs/`
- Analysis CSVs and figures: `experiments/qwen_slot_repair_distillation/analysis/`
- Reports: `experiments/qwen_slot_repair_distillation/reports/`
- Large checkpoints: `large_artifacts/qwen_slot_repair_distillation/checkpoints/`
- Manifest: `experiments/qwen_slot_repair_distillation/checkpoint_manifest.csv`

## Success Criteria

- Keep source, runs, analysis, reports, and manifest in this experiment
  directory.
- Store bulky checkpoints under `large_artifacts/qwen_slot_repair_distillation/`.
- Evaluate base compiler, deployable editor, and local oracle ceiling.
- Run smoke, pilot, and main configurations.
- Iterate when a design fails rather than one-shotting the final run.
- Produce CSVs, figures, a markdown report, and an HTML report.

## Implementation Iterations

### Direct Full-Slot Editor

The first editor predicted every init/op/arg slot directly.

- `smoke_slot_repair_distill`: path check passed, but the model changed many
  slots and destroyed accuracy.
- `smoke_slot_repair_distill_bias`: adding base-copy logit bias made the editor
  copy the base program.
- `pilot_slot_repair_distill_s96_b5`: copied the base program; fresh paired
  stayed 25.8% -> 25.8%.
- `pilot_slot_repair_distill_s96_b3_w8`: edited too aggressively; fresh paired
  fell 28.9% -> 10.9%.

Conclusion: full-slot prediction is the wrong deployable interface. It lacks an
explicit decision about whether a slot should be edited.

### Gated Edit Policy

The second editor predicts edit gates separately from replacement values. It
starts from the base program, then applies only slots whose gate probability
passes a validation-selected threshold.

- `smoke_slot_repair_gated`: path check passed.
- `pilot_slot_repair_gated_s96`: validation improved 25.0% -> 31.2%, but fresh
  paired fell 30.5% -> 28.1%.
- `pilot_slot_repair_gated_s96_oracle_base`: using copy-base fallback for
  no-oracle examples was too conservative and still hurt fresh paired
  24.2% -> 20.3%.
- `pilot_slot_repair_gated_s96_value_stabilized`: adding a small unchanged-slot
  value loss stabilized false-positive edits. Fresh standard improved
  28.1% -> 35.9%, paraphrase 23.4% -> 25.0%, and paired 28.1% -> 32.8%.

Conclusion: gated editing plus weak unchanged-slot value supervision was the
only pilot worth scaling.

## Primary Run

Run: `main_slot_repair_distill_s512`

Configuration:

- 512 train examples
- 128 validation examples
- 256 fresh standard examples
- 256 fresh paraphrase examples
- 256 paired programs, evaluated as 512 paired prompt variants
- top-3/two-edit local teacher
- gated edit policy, 128 hidden width, 3 layers
- `editor_target_mode=oracle_or_gold`
- `unchanged_value_loss_weight=0.05`
- validation threshold grid: `0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7`

Final selected checkpoint:

- best epoch: 18
- best threshold: 0.7

Main metrics:

| Split | Base | Editor | Oracle | Gap recovered |
|---|---:|---:|---:|---:|
| train_len24 | 27.5% | 31.2% | 85.4% | 6.4% |
| val_len24 | 32.8% | 35.2% | 85.2% | 4.5% |
| fresh_standard_len24 | 28.5% | 27.7% | 87.1% | -1.3% |
| fresh_paraphrase_len24 | 25.0% | 18.8% | 85.2% | -10.4% |
| fresh_paired_len24 | 25.4% | 24.8% | 86.1% | -1.0% |

Fresh paired diagnostics:

- editor average edits per program: 1.17
- editor gate precision: 66.9%
- editor gate recall: 59.7%
- editor pair both-correct: 16.8% versus base 23.4%
- editor pair state consistency: 37.1% versus base 69.5%

## Interpretation

The main run is not a successful local-oracle distillation. The editor learned
a validation signal but did not improve fresh distributions. The high local
oracle ceiling means the repair headroom is real; this experiment shows that a
single feed-forward gate/value policy over engineered base traces is not enough
to recover that headroom robustly.

The most useful next direction is likely not a larger version of this exact
editor. A stronger follow-up should keep some form of execution-grounded
selection at inference time, or train the Qwen-attached compiler on on-policy
repair traces so the base compiler itself moves toward repairable/correct
programs instead of relying on a small post-hoc editor.

## Final Artifacts

- Markdown report: `reports/qwen_slot_repair_distillation_paper.md`
- HTML report: `reports/qwen_slot_repair_distillation_paper.html`
- Summary: `analysis/summary.md`
- Main metrics: `runs/main_slot_repair_distill_s512/metrics.csv`
- Main train log: `runs/main_slot_repair_distill_s512/editor_train_log.csv`
- Figures: `analysis/figures/`
- Checkpoint manifest: `checkpoint_manifest.csv`

