# Trace-Keyed Symbol Repair Experiment Log

## 2026-06-20 Setup

Objective: design and run a standalone experiment in execution-conditioned program repair where the execution trace carries task-specific information that is not recoverable from repository context alone.

Directory policy:

- Small, downloadable experiment package: `/workspace/experiments/trace_keyed_symbol_repair/`.
- Large artifacts excluded from the small package: `/workspace/large_artifacts/trace_keyed_symbol_repair/`.
- Model adapters and checkpoints go under `large_artifacts/trace_keyed_symbol_repair/models/`.
- Reports, logs, configs, figures, and compact JSON/JSONL result summaries go under `experiments/trace_keyed_symbol_repair/`.

Initial hypothesis:

A repair model trained on `wrong patched state + failed execution trace -> corrective diff` will outperform no-trace and shuffled-trace controls when the correct fix requires a literal symbol/value revealed only by the failed test output.

Design constraint:

The paper and artifacts must be standalone. They should not depend on or cite earlier experiments.

## 2026-06-20 Dataset Design and First Build

Design:

- Each task has one production file, `src/repair_target.py`.
- The wrong-patched file contains a random wrong `CANONICAL_TOKEN`.
- The correct patch replaces that wrong token with a random expected token.
- The expected token is not present in repository context or the issue text.
- The expected token is present in the failed pytest output, both in the assertion source and in a custom `TRACE_KEY expected_token=... actual_token=...` message.
- A no-trace model cannot infer the token except by guessing.
- A shuffled-trace model sees a plausible trace but for the wrong record, so copying from it should produce an incorrect token.

Splits:

- Train: 240 records.
- IID validation: 60 records using the same token styles as training.
- Format-holdout validation: 60 records using unseen token formats.

Token styles:

- Train styles: `dash_upper`, `underscore_upper`, `mixed_hex`.
- Format-holdout styles: `colon_upper`, `dot_lower`.

Validation performed by the builder:

- The wrong patch fails visible tests.
- The target corrective diff applies to the wrong-patched file.
- The repaired file passes visible and hidden tests.
- The repaired file compiles.
- The expected token is absent from current files.
- The expected token is present in the failing trace.
- The wrong token is present in current files.

Generated files:

- `data/repair_train.jsonl`
- `data/repair_val_iid.jsonl`
- `data/repair_val_format_holdout.jsonl`
- `data/repair_all.jsonl`
- `data/dataset_manifest.json`

First build result:

- All 360 records passed the builder invariants.
- Data directory size: about 3.6 MB.
- Full validation was slow because it runs pytest multiple times per generated record; this is acceptable for the final dataset and worth recording as a reproducibility cost.

## 2026-06-20 Frozen Pilot

Command:

`python experiments/trace_keyed_symbol_repair/scripts/eval_trace_keyed.py --data experiments/trace_keyed_symbol_repair/data/repair_val_iid.jsonl --output experiments/trace_keyed_symbol_repair/reports/frozen_trace_iid_pilot10.json --condition trace --max-records 10 --max-new-tokens 192`

Result:

- Model: `Qwen/Qwen2.5-Coder-3B-Instruct`, revision `488639f1ff808d1d3d0ba301aef8c11461451ec5`.
- Records: 10 IID validation examples.
- Repair@1: 0/10.
- Patch apply rate: 0/10.
- Expected-token copy rate: 1/10.

Observed failure mode:

- The frozen model usually generated a diff from the original placeholder token to the current wrong token.
- That patch does not apply to the wrong-patched tree.
- This confirms the task is not solved by the frozen model and is suitable for a training pilot.

Next step:

- Train a small trace-conditioned pilot adapter on 80 records for one epoch, then evaluate on 20 IID records.

## 2026-06-20 Trace Pilot Training

Command:

`python scripts/train_repair_lora.py --train experiments/trace_keyed_symbol_repair/data/repair_train.jsonl --eval experiments/trace_keyed_symbol_repair/data/repair_val_iid.jsonl --mode trace --model-id Qwen/Qwen2.5-Coder-3B-Instruct --revision 488639f1ff808d1d3d0ba301aef8c11461451ec5 --output-dir large_artifacts/trace_keyed_symbol_repair/models/pilot_trace_lora --max-length 2048 --epochs 1 --lr 2e-4 --rank 16 --alpha 32 --dropout 0.05 --grad-accum 8 --save-steps 20 --eval-steps 20 --max-train-records 80`

Training observations:

- Trainable parameters: 29,933,568, about 0.96% of the model.
- Training took about 50 seconds.
- Training loss dropped from about `0.13` to `0.0006`.
- Eval loss on all 60 IID records after the epoch: `0.0005632`.

Pilot trace evaluation command:

`python experiments/trace_keyed_symbol_repair/scripts/eval_trace_keyed.py --data experiments/trace_keyed_symbol_repair/data/repair_val_iid.jsonl --output experiments/trace_keyed_symbol_repair/reports/pilot_trace_iid20.json --condition trace --adapter large_artifacts/trace_keyed_symbol_repair/models/pilot_trace_lora --max-records 20 --max-new-tokens 192`

Pilot trace evaluation result:

- Records: 20 IID validation examples.
- Repair@1: 20/20.
- Patch apply rate: 20/20.
- Syntax-valid rate: 20/20.
- Expected-token copy rate: 20/20.

Interpretation:

- The trace-conditioned task is learnable with a small LoRA and a small training subset.
- Next control check: evaluate the same pilot adapter with no trace and shuffled traces.

## 2026-06-20 Pilot Trace Controls

No-trace control:

- Command output: `reports/pilot_trace_adapter_no_trace_iid20.json`.
- Records: 20 IID validation examples.
- Repair@1: 0/20.
- Patch apply rate: 20/20.
- Expected-token copy rate: 0/20.
- Interpretation: the adapter can produce syntactically valid patches without the trace, but cannot identify the hidden expected token.

Shuffled-trace control:

- Command output: `reports/pilot_trace_adapter_shuffled_trace_iid20.json`.
- Records: 20 IID validation examples.
- Repair@1: 0/20.
- Patch apply rate: 20/20.
- Expected-token copy rate: 0/20.
- Interpretation: when given another record's trace, the adapter copies or uses a wrong token-shaped value and fails the hidden test.

Token-budget check:

- `max_new_tokens=64` truncates long token strings and causes 0/5 repairs.
- `max_new_tokens=128` preserves 5/5 repairs in a spot check.
- Final evaluation will use `max_new_tokens=128`.

Decision:

- The pilot establishes the intended causal contrast: normal trace succeeds, no trace fails, shuffled trace fails.
- Proceed to full one-epoch training for trace, no-trace, shuffled-trace, and final-patch conditions on all 240 train records.

## 2026-06-20 Full Adapter Training

Shared hyperparameters:

- Base model: `Qwen/Qwen2.5-Coder-3B-Instruct`.
- Revision: `488639f1ff808d1d3d0ba301aef8c11461451ec5`.
- QLoRA rank: 16.
- LoRA alpha: 32.
- Dropout: 0.05.
- Epochs: 1.
- Learning rate: `2e-4`.
- Max length: 2048.
- Gradient accumulation: 8.
- Train records: 240.
- Eval records during training: 60 IID validation records.

Adapters:

- Trace SFT: `large_artifacts/trace_keyed_symbol_repair/models/trace_lora`.
- No-trace SFT: `large_artifacts/trace_keyed_symbol_repair/models/no_trace_lora`.
- Shuffled-trace SFT: `large_artifacts/trace_keyed_symbol_repair/models/shuffled_trace_lora`.
- Final-patch SFT: `large_artifacts/trace_keyed_symbol_repair/models/final_patch_lora`.

Training observations:

- Trace SFT final train loss: about `0.0105`; final eval loss: `4.971e-05`.
- No-trace SFT final train loss: about `0.4329`; final eval loss: `0.4074`.
- Shuffled-trace SFT final train loss: about `0.4332`; final eval loss: `0.3953`.
- Final-patch SFT final train loss: about `0.5052`; final eval loss: `0.4231`.

Interpretation before executable evaluation:

- Only the normal trace condition fits the target distribution well.
- The other conditions remain high-loss because the random expected token is not present in the model input or is present only in an unrelated shuffled trace.
- Next step: executable diff evaluation on IID and format-holdout splits.

## 2026-06-20 Packaging and Reporting Setup

Package structure:

- Small experiment package: `/workspace/experiments/trace_keyed_symbol_repair/`.
- Large artifact package: `/workspace/large_artifacts/trace_keyed_symbol_repair/`.
- Adapter directories are excluded from the small package and listed in `large_artifacts_manifest.md`.
- Added `PACKAGE_README.md`, updated `README.md`, and copied dependency pins into `requirements.txt`.

Report generator:

- Added `scripts/make_report.py`.
- The script reads pilot and final evaluation JSON files, writes CSV summary tables, generates figures in `figures/`, writes `reports/trace_keyed_symbol_repair_summary.md`, writes `reports/trace_keyed_symbol_repair_paper.md`, and refreshes the large-artifact manifest.
- The paper is written as a standalone document and does not depend on prior experiment context.

Current final evaluation status at setup time:

- Core suite is running with `max_new_tokens=128`.
- Completed full frozen trace IID and format-holdout evaluations.
- Completed full final-patch IID evaluation.
- Final-patch format-holdout evaluation is in progress.

## 2026-06-20 Final Core Evaluation

Command:

`python experiments/trace_keyed_symbol_repair/scripts/run_final_evaluations.py --suite core --max-new-tokens 128`

Core results:

- Frozen base + trace, IID: 0/60 repair@1, patch apply 0/60, expected-token copy 11/60.
- Frozen base + trace, format holdout: 0/60 repair@1, patch apply 0/60, expected-token copy 2/60.
- Final-patch SFT + final patch, IID: 0/60 repair@1, patch apply 60/60, expected-token copy 0/60.
- Final-patch SFT + final patch, format holdout: 0/60 repair@1, patch apply 60/60, expected-token copy 0/60.
- No-trace SFT + no trace, IID: 0/60 repair@1, patch apply 60/60, expected-token copy 0/60.
- No-trace SFT + no trace, format holdout: 0/60 repair@1, patch apply 60/60, expected-token copy 0/60.
- Shuffled-trace SFT + real trace, IID: 0/60 repair@1, patch apply 60/60, expected-token copy 0/60.
- Shuffled-trace SFT + real trace, format holdout: 0/60 repair@1, patch apply 60/60, expected-token copy 0/60.
- Trace SFT + real trace, IID: 60/60 repair@1, patch apply 60/60, expected-token copy 60/60.
- Trace SFT + real trace, format holdout: 60/60 repair@1, patch apply 60/60, expected-token copy 60/60.

Interpretation:

- The trace-trained adapter learned to copy the trace-revealed expected token and repair both IID and held-out token formats.
- Controls that lacked the correct trace, or were trained with shuffled trace evidence, emitted syntactically valid diffs but did not recover the expected token.
- The frozen base sometimes copied a token-shaped string, but generated patches against the wrong source state, so none applied.

Next step:

- Run ablations that keep the trace-trained adapter fixed but remove the trace or replace it with shuffled traces at evaluation time.

## 2026-06-20 Trace Adapter Input Ablations

Command:

`python experiments/trace_keyed_symbol_repair/scripts/run_final_evaluations.py --suite ablation --max-new-tokens 128`

Ablation results:

- Trace SFT + no trace, IID: 0/60 repair@1, patch apply 60/60, expected-token copy 0/60, wrong-token removed 57/60.
- Trace SFT + no trace, format holdout: 0/60 repair@1, patch apply 60/60, expected-token copy 0/60, wrong-token removed 54/60.
- Trace SFT + shuffled trace, IID: 0/60 repair@1, patch apply 60/60, expected-token copy 0/60, wrong-token removed 60/60.
- Trace SFT + shuffled trace, format holdout: 0/60 repair@1, patch apply 60/60, expected-token copy 0/60, wrong-token removed 57/60.

Interpretation:

- The trained trace adapter depends on the correct trace at inference time.
- With no trace, it often emits valid patches that remove the wrong token but cannot recover the expected token.
- With shuffled traces, it emits valid patches keyed to incorrect evidence and fails the hidden tests.

Next step:

- Generate final CSV summaries, figures, standalone paper, and package manifest.

## 2026-06-20 Final Report and Package Verification

Report generation command:

`python experiments/trace_keyed_symbol_repair/scripts/make_report.py`

Generated report artifacts:

- `reports/trace_keyed_symbol_repair_paper.md`
- `reports/trace_keyed_symbol_repair_summary.md`
- `reports/final_core_results.csv`
- `reports/final_ablation_results.csv`
- `reports/pilot_results.csv`
- `figures/core_repair_rates.png`
- `figures/expected_token_copy_rates.png`
- `figures/trace_ablation_repair_rates.png`
- `large_artifacts_manifest.md`

Verification:

- All experiment scripts compiled successfully with `python -m py_compile`.
- The paper and summary contain no references to prior experiment package names.
- Required report, figure, README, log, manifest, and CSV artifacts are present.
- Final evaluation JSON count: 14.
- No files larger than 50 MB remain inside `/workspace/experiments/trace_keyed_symbol_repair/`.
- Removed transient `__pycache__` and `.ipynb_checkpoints` directories from the small package.
- No trace-keyed experiment evaluation processes remain running.

Final artifact sizes:

- Small package: `/workspace/experiments/trace_keyed_symbol_repair/`, about 7.6 MB.
- Large artifacts: `/workspace/large_artifacts/trace_keyed_symbol_repair/`, about 1.6 GB.

Previous experiment packaging status:

- Small package: `/workspace/experiments/execution_conditioned_repair/`, about 13 MB.
- Large artifacts: `/workspace/large_artifacts/execution_conditioned_repair/`, about 13 GB.
