# Qwen 3.5 4B Balanced Discriminative Bridge Log

## Objective

Test whether a balanced frontier bridge curriculum becomes more useful when the trace cases are deliberately discriminative against hard aliases and seed-adapter mistakes, while preserving the same 240-record training budget and using only `Qwen/Qwen3.5-4B`.

## Design Commitments

- Keep all trained conditions at 240 records.
- Keep bridge-family allocation equal: 6 records for each of 10 frontier families.
- Compare random base training, normal static bridge training, alias-discriminative bridge training, and model-discriminative bridge training.
- Evaluate on both normal frontier records and hard frontier records.
- Store trained adapters and checkpoints under `/workspace/large_artifacts/qwen35_4b_balanced_discriminative_bridge/models`.
- Keep the compact experiment directory downloadable without large model artifacts.

## Starting Hypotheses

1. Equal frontier coverage is likely more important than reallocating bridge records across families.
2. The useful knob is within-family trace selection: hard visible cases should remove shortcut programs that normal random cases leave viable.
3. Seed-adapter wrong programs provide useful selectors only if combined with a broader alias bank, because sampled model errors can be sparse and idiosyncratic.
4. A harder frontier split should reveal whether a near-perfect normal frontier result is genuine compositional robustness or just eval saturation.

## Planned Runs

1. Build deterministic datasets from seed `20260627`.
2. Train `seed_lora`.
3. Train `static_bridge_lora`.
4. Train `alias_discriminative_bridge_lora`.
5. Mine seed-adapter wrong programs on the hard mining pool.
6. Train `model_discriminative_bridge_lora`.
7. Evaluate all adapters on normal frontier, hard frontier, and IID retention.
8. Evaluate trace controls for bridge adapters on hard frontier.
9. Generate a final report and large-artifact manifest.

## Step Log

- Initialized standalone directory and large artifact directory.
- Added hard-case input generation for all frontier families.
- Added alias-selector banks for frontier families.
- Added alias-discriminative bridge dataset generation.
- Added normal and hard frontier eval splits.
- Updated model-discriminative mining to use fixed equal allocation and alias-bank fallback.
- Built datasets with seed `20260627`.
- Dataset counts: seed train 240, static bridge train 240, alias-discriminative bridge train 240, base anchor 180, static bridge records 60, alias-discriminative bridge records 60, IID eval 60, normal frontier eval 120, hard frontier eval 120, mining pool 240.
- Bridge allocation audit: every frontier family has 6 records in both static and alias-discriminative bridge sets.
- Selector audit: static bridge records used normal cases and eliminated 4.10 wrong programs on average; alias-discriminative bridge records used hard cases and eliminated 5.10 wrong programs on average.
- Trained `seed_lora` for 2 epochs / 60 optimizer steps on 240 seed records.
- `seed_lora` training summary: runtime 847.3s, train loss 0.1115, eval loss 0.0001022 on the 24-record training-time eval subset.
- Saved `seed_lora` under `/workspace/large_artifacts/qwen35_4b_balanced_discriminative_bridge/models/seed_lora`.
- Mined seed-adapter generations on the 240-record hard mining pool with greedy plus 2 samples per record.
- Mining runtime: 49m46s.
- Mining found 276 executable wrong candidates across the pool.
- Model-error bridge coverage: 46/60 selected bridge rows had source-row seed-adapter wrong programs; 48/60 rows had model wrong programs after adding family-level wrong selectors; 12/60 rows used alias fallback.
- Model-discriminative bridge audit: 6 records per frontier family, hard selector cases, average 6.90 selector programs per record, average 6.80 eliminated selector programs per record.
- Trained `static_bridge_lora` for 2 epochs / 60 optimizer steps on 240 records.
- `static_bridge_lora` training summary: runtime 905.3s, train loss 0.1112, eval loss 0.0001177 on the 24-record training-time eval subset.
- Saved `static_bridge_lora` under `/workspace/large_artifacts/qwen35_4b_balanced_discriminative_bridge/models/static_bridge_lora`.
- Trained `alias_discriminative_bridge_lora` for 2 epochs / 60 optimizer steps on 240 records.
- `alias_discriminative_bridge_lora` training summary: runtime 930.5s, train loss 0.1040, eval loss 0.0005526 on the 24-record training-time eval subset.
- Saved `alias_discriminative_bridge_lora` under `/workspace/large_artifacts/qwen35_4b_balanced_discriminative_bridge/models/alias_discriminative_bridge_lora`.
- Trained `model_discriminative_bridge_lora` for 2 epochs / 60 optimizer steps on 240 records.
- `model_discriminative_bridge_lora` training summary: runtime 888.3s, train loss 0.1062, eval loss 0.0001196 on the 24-record training-time eval subset.
- Saved `model_discriminative_bridge_lora` under `/workspace/large_artifacts/qwen35_4b_balanced_discriminative_bridge/models/model_discriminative_bridge_lora`.
- Updated evaluation generation to request sampled return sequences in one call per prompt while preserving the same greedy plus sampled candidate semantics.
- Evaluated normal frontier split with greedy plus 3 sampled candidates per record:
  - `seed_lora`: greedy hidden 68/120, rerank hidden 75/120.
  - `static_bridge_lora`: greedy hidden 118/120, rerank hidden 118/120.
  - `alias_discriminative_bridge_lora`: greedy hidden 106/120, rerank hidden 106/120.
  - `model_discriminative_bridge_lora`: greedy hidden 87/120, rerank hidden 94/120.
- Evaluated hard frontier split with greedy plus 3 sampled candidates per record:
  - `seed_lora`: greedy hidden 68/120, rerank hidden 72/120.
  - `static_bridge_lora`: greedy hidden 119/120, rerank hidden 119/120.
  - `alias_discriminative_bridge_lora`: greedy hidden 107/120, rerank hidden 108/120.
  - `model_discriminative_bridge_lora`: greedy hidden 95/120, rerank hidden 99/120.
- Evaluated IID retention split greedily:
  - `seed_lora`: 60/60 hidden all-pass.
  - `static_bridge_lora`: 60/60 hidden all-pass.
  - `alias_discriminative_bridge_lora`: 60/60 hidden all-pass.
  - `model_discriminative_bridge_lora`: 60/60 hidden all-pass.
- Ran hard-frontier trace controls for `static_bridge_lora`:
  - Correct trace prompt: 119/120 hidden all-pass.
  - No trace prompt: 109/120 hidden all-pass.
  - Shuffled trace prompt: 19/120 hidden all-pass and 20/120 visible all-pass.
- Main readout: normal static bridge records were the strongest condition. Hard-case alias-discriminative and model-discriminative trace selection reduced frontier performance despite preserving IID retention.
- Trace-control readout: the static adapter learned substantial parametric structure, but correct visible traces still carried test-time information; semantically wrong traces strongly misled the learned bridge interface.
- Generated final report with next-experiment options at `reports/qwen35_4b_balanced_discriminative_bridge_report.md`.
- Final audit:
  - `python -m compileall src scripts` passed.
  - Removed generated `__pycache__` directories after compile audit.
  - Stale-name scan found no matches.
  - Compact directory large-file scan found no files over 50M.
  - No train/eval/mining processes were running.
  - Compact directory size: 13M.
  - Large adapter directory size: 1.8G.
