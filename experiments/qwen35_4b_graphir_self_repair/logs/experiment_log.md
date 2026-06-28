# Qwen 3.5 4B GraphIR Self Repair Log

## Objective

Test whether fixed-budget posttraining can improve held-out executable repair by moving from freeform DSL generation to typed register-graph configuration plus a verifier-guided repair step.

## Design Commitments

- Use only `Qwen/Qwen3.5-4B`.
- Keep each trained adapter at 240 records.
- Keep the compact experiment directory downloadable by storing adapters and checkpoints outside it.
- Train a DSL baseline, a GraphIR construction adapter, and a GraphIR repair adapter.
- Evaluate support-family generalization, held-out ceiling-family generalization, IID retention, and direct graph repair.
- Keep trace controls optional unless the main pipeline result remains positive or ambiguous.
- Generate a final markdown report and charts.

## Starting Hypotheses

1. GraphIR should reduce syntactic and parenthesization failures by forcing one operation per register assignment.
2. GraphIR should make composition errors more visible to the verifier because each intermediate is executable.
3. A repair adapter should improve over construction-only by editing the selected graph after visible-case execution.
4. Shuffled traces should hurt if the model is using execution evidence rather than only memorizing output shape.

## Planned Runs

1. Build deterministic datasets from seed `20260701`.
2. Train `dsl_static60_lora`.
3. Train `graphir_construct_lora`.
4. Train `graphir_repair_lora`.
5. Evaluate DSL baseline on IID, support, and ceiling splits.
6. Evaluate GraphIR construction and construction+repair on IID, support, and ceiling splits.
7. Optionally run GraphIR trace controls on the ceiling split if the main result needs disambiguation.
8. Run direct corrupted-GraphIR repair diagnostic on the ceiling split.
9. Generate charts and final report.
10. Audit compact artifact size and large artifact separation.

## Step Log

- Initialized standalone experiment directory and large artifact directory.
- Added GraphIR compiler, parser, executor, and visible-case scoring.
- Added task-specific prompts for DSL baseline, GraphIR construction, and GraphIR repair.
- Added dataset builder for aligned DSL, GraphIR construction, and GraphIR repair records.
- Added generic QLoRA trainer and evaluation scripts.
- Added report generator scaffold.
- `python -m compileall src scripts` passed before dataset generation.
- Built deterministic datasets with seed `20260701`.
- Dataset counts: DSL train 240, GraphIR construct train 240, GraphIR repair train 240, IID eval 60, support eval 120, ceiling eval 120, corrupted ceiling repair eval 120.
- Training mix per adapter: 180 base-family records plus 60 support bridge records.
- Confirmed that no held-out ceiling family appears in any training set.
- Confirmed target GraphIR executes correctly on train and ceiling eval records.
- Corrupted GraphIR repair candidates are nontrivial: train candidates pass all visible cases in 9/240 records and all hidden cases in 6/240 records; ceiling diagnostic candidates pass all visible cases in 19/120 records and all hidden cases in 4/120 records.
- Trained `dsl_static60_lora` for 2 epochs / 60 optimizer steps on 240 DSL records.
- `dsl_static60_lora` training summary: runtime 877.9s, train loss 0.1078, eval loss 0.0003974 on the 24-record training-time eval subset.
- Saved `dsl_static60_lora` under `/workspace/large_artifacts/qwen35_4b_graphir_self_repair/models/dsl_static60_lora`.
- Trained `graphir_construct_lora` for 2 epochs / 60 optimizer steps on 240 GraphIR construction records.
- `graphir_construct_lora` training summary: runtime 890.7s, train loss 0.05635, eval loss 0.0003681 on the 24-record training-time eval subset.
- Saved `graphir_construct_lora` under `/workspace/large_artifacts/qwen35_4b_graphir_self_repair/models/graphir_construct_lora`.
- Trained `graphir_repair_lora` for 2 epochs / 60 optimizer steps on 240 corrupted-candidate GraphIR repair records.
- `graphir_repair_lora` training summary: runtime 918.7s, train loss 0.05824, eval loss 0.4679 on the 24-record corrupted-ceiling training-time eval subset.
- Saved `graphir_repair_lora` under `/workspace/large_artifacts/qwen35_4b_graphir_self_repair/models/graphir_repair_lora`.
- Evaluation token-budget adjustment: target GraphIR outputs were measured before full ceiling/support evaluation. Tokenized target lengths were max 69 tokens on support and max 77 tokens on ceiling, so GraphIR support/ceiling and repair evaluations were capped at 96 new tokens. This preserved margin while preventing non-answer tails from dominating runtime.
- Added `scripts/eval_graphir_cached_repair.py` so construction results can be repaired without rerunning construction. This preserves the same selected construct graph while avoiding duplicate GPU work.
- DSL IID evaluation: 60/60 greedy hidden all-pass and 60/60 rerank hidden all-pass, `num_samples=0`, `max_new_tokens=96`.
- GraphIR construct IID evaluation: 60/60 construction hidden all-pass, `num_samples=0`, `max_new_tokens=160`.
- GraphIR pipeline IID evaluation: 60/60 construction hidden all-pass and 60/60 repair hidden all-pass, `num_samples=0`, `max_new_tokens=160`, visible-pass repair skip enabled.
- DSL support evaluation: 120/120 greedy hidden all-pass and 120/120 rerank hidden all-pass, `num_samples=3`, `max_new_tokens=96`.
- DSL ceiling evaluation: 33/120 greedy hidden all-pass and 35/120 rerank hidden all-pass, `num_samples=1`, `max_new_tokens=96`.
- GraphIR support construction evaluation: 118/120 hidden all-pass, `num_samples=0`, `max_new_tokens=96`.
- GraphIR support cached repair evaluation: improved support from 118/120 to 120/120 hidden all-pass, `num_samples=1`, `max_new_tokens=96`, visible-pass repair skip enabled.
- GraphIR ceiling construction evaluation: 26/120 hidden all-pass, `num_samples=1`, `max_new_tokens=96`.
- GraphIR ceiling cached repair evaluation: improved actual pipeline from 26/120 to 29/120 hidden all-pass, `num_samples=1`, `max_new_tokens=96`, visible-pass repair skip enabled.
- Direct corrupted-GraphIR repair diagnostic: corrupted candidate input scored 4/120 hidden all-pass; greedy repair improved it to 32/120 hidden all-pass, `num_samples=0`, `max_new_tokens=96`.
- Main conclusion: GraphIR plus repair did not beat the DSL baseline on held-out ceiling families. The actual pipeline ended at 29/120 hidden all-pass versus DSL rerank at 35/120.
- Interpretation: the repair adapter learned some repair behavior on synthetic corrupted GraphIR candidates, but it did not transfer enough to the actual construction adapter's error distribution.
- Trace controls were not run. The main result was below baseline and the direct repair diagnostic resolved the key ambiguity: repair skill exists, but actual construction errors remain the bottleneck.
- Generated final report at `reports/qwen35_4b_graphir_self_repair_report.md`.
- Generated charts: `figures/ceiling_hidden_success.png` and `figures/ceiling_by_family.png`.
