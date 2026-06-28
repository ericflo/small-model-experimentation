# Qwen 3.5 4B Verified Edit Closure Log

## Objective

Test whether a bounded verifier-guided symbolic edit closure can improve held-out executable DSL repair over normal candidate sampling for Qwen 3.5 4B.

## Design Commitments

- Use only `Qwen/Qwen3.5-4B`.
- Train a fresh DSL baseline adapter inside this standalone experiment.
- Keep the training budget fixed at 240 records.
- Keep adapter/checkpoint files outside the compact experiment directory.
- Evaluate normal model candidates and edit-closure-expanded candidates on IID, support, and held-out ceiling splits.
- Report both verifier-selected hidden success and hidden-oracle closure coverage.
- Generate a final markdown report and charts.

## Hypotheses

1. Many held-out failures are valid DSL programs that are one or two local semantic edits away from the target.
2. Visible execution can select the corrected local edit without hidden-case access.
3. If hidden-oracle closure coverage is much higher than verifier-selected closure success, the bottleneck is visible-case discrimination rather than candidate support.
4. If hidden-oracle closure coverage is low, this edit family is not enough and a larger program search or different representation is needed.

## Planned Runs

1. Build deterministic datasets from seed `20260630`.
2. Train `static60_lora` on 180 base records plus 60 support bridge records.
3. Evaluate `static60_lora` on IID, support, and ceiling splits with normal visible reranking.
4. Run edit closure on the normal baseline results for IID, support, and ceiling.
5. If edit closure materially improves ceiling hidden success, inspect whether the improvement is selected by visible cases or only present under hidden oracle.
6. Generate charts and final report.
7. Audit compact artifact size and large artifact separation.

## Step Log

- Initialized standalone experiment directory and large artifact directory.
- Copied stable DSL, data generation, prompt, training, and baseline evaluator utilities.
- Added bounded symbolic DSL edit closure and closure evaluation script.
- `python -m compileall src scripts` passed before dataset generation.
- Local closure smoke test passed: the edit closure generated the intended `len text` repair from a `count_eq text needle` failure within two edit rounds.
- Built deterministic datasets with seed `20260630`.
- Dataset counts used for the main experiment: static60 train 240, IID eval 60, support eval 120, ceiling eval 120.
- Training mix for `static60_lora`: 180 base-family records plus 60 support bridge records.
- Confirmed that no held-out ceiling family appears in the static60 training set.
- Confirmed that target DSL programs execute correctly on all visible and hidden cases for static60 train, IID eval, support eval, and ceiling eval.
- Trained `static60_lora` from `Qwen/Qwen3.5-4B` for 2 epochs on the 240-record static60 DSL set.
- Training completed in 909.3 seconds with final `train_loss=0.1042`.
- Final trainer IID eval on the 60-record IID eval file produced `eval_loss=8.394e-05`.
- Large adapter/checkpoint artifacts were written outside the compact experiment directory at `/workspace/large_artifacts/qwen35_4b_verified_edit_closure/models/static60_lora` (`445M` after training).
- Baseline eval completed for all splits.
- IID baseline greedy and reranked hidden all-pass: 60/60.
- Support baseline greedy and reranked hidden all-pass: 120/120.
- Ceiling baseline greedy hidden all-pass: 48/120; ceiling baseline visible-reranked hidden all-pass: 47/120.
- Ran closure smoke check on 5 ceiling records; schema and runtime were valid.
- First closure selector iteration exposed visible-tie regressions on IID because shortest-program tie-breaking chose degenerate visible-perfect programs.
- Replaced closure visible tie-breaking with stable first-seen selection so nearer seed candidates are preserved under equal visible pass counts.
- Added conservative and strict closure acceptance policies:
  - conservative accepts closure only when visible pass count improves over baseline;
  - strict accepts closure only when closure reaches all visible cases and baseline does not.
- Reran closure evals with the revised selector and acceptance metrics.
- IID closure hidden all-pass after selector revision: 60/60.
- Support closure hidden all-pass after selector revision: 120/120.
- Ceiling visible-selected closure hidden all-pass: 62/120.
- Ceiling conservative closure hidden all-pass: 61/120.
- Ceiling strict visible-all closure hidden all-pass: 61/120.
- Ceiling hidden-oracle closure all-pass: 69/120.
- Strict closure accepted 39/120 ceiling edits and had 4 hidden pass-count damage cases.
- Generated final report at `reports/qwen35_4b_verified_edit_closure_report.md`.
- Generated charts under `reports/figures/`: `ceiling_hidden_success.png`, `closure_by_family.png`, and `closure_candidate_counts.png`.
- Final audit:
  - `python -m compileall src scripts` passed.
  - Removed generated `__pycache__` directories after compile audit.
  - No files larger than 50M are present in the compact experiment directory.
  - Compact experiment directory size: `6.8M`.
  - Large adapter/checkpoint directory size: `445M`.
  - No train or eval processes remained running.
  - Stale-reference scan returned no hits for old experiment names or non-Qwen3.5 model names.
  - Final dataset line counts: train 240, IID 60, support 120, ceiling 120.
