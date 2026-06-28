# Qwen 3.5 4B Static Bridge Ceiling Breaker Log

## Objective

Test whether fixed-budget static bridge posttraining transfers from support bridge families to deeper held-out composition families that are absent from bridge training.

## Design Commitments

- Use only `Qwen/Qwen3.5-4B`.
- Keep each trained adapter at 240 records.
- Keep the compact experiment directory downloadable by storing trained adapters and checkpoints outside it.
- Train a seed baseline, a 60-record static bridge adapter, and an 80-record static bridge adapter.
- Evaluate support-family generalization, held-out ceiling-family generalization, IID retention, and trace controls.
- Generate a final markdown report and charts.

## Starting Hypotheses

1. Static bridge examples should strongly improve support-family repair relative to the seed adapter.
2. Held-out ceiling families should be harder than support families and may expose whether bridge training learns a reusable trace interface or mostly family templates.
3. Increasing bridge dose from 60 to 80 may help if the ceiling result is limited by bridge-token exposure.
4. Shuffled traces should be harmful if the adapter is using visible execution semantics rather than ignoring the trace.

## Planned Runs

1. Build deterministic datasets from seed `20260630`.
2. Train `seed_lora`.
3. Train `static60_lora`.
4. Train `static80_lora`.
5. Evaluate all adapters on support, ceiling, and IID splits.
6. Run ceiling trace controls for the stronger static bridge adapter, and for both static adapters if time permits.
7. Generate charts and a final report.
8. Audit compact artifact size and large artifact separation.

## Step Log

- Initialized standalone experiment directory and large artifact directory.
- Added held-out ceiling composition families to the data generator.
- Replaced dataset builder with the seed/static60/static80 ceiling-breaker design.
- Replaced report builder with markdown plus PNG chart generation.
- `python -m compileall src scripts` passed before dataset generation.
- Built deterministic datasets with seed `20260630`.
- Dataset counts: seed train 240, static60 train 240, static80 train 240, support eval 120, ceiling eval 120, IID eval 60.
- Static 60 bridge allocation: 6 records per support bridge family, 60 total.
- Static 80 bridge allocation: 8 records per support bridge family, 80 total.
- Static 60 selector summary: average 4.10 eliminated wrong programs per bridge record, zero remaining static distractors on average.
- Static 80 selector summary: average 4.10 eliminated wrong programs per bridge record, zero remaining static distractors on average.
- Confirmed that no held-out ceiling family appears in either static bridge training set.
- Trained `seed_lora` for 2 epochs / 60 optimizer steps on 240 seed records.
- `seed_lora` training summary: runtime 835.7s, train loss 0.1080, eval loss 0.0001593 on the 24-record training-time eval subset.
- Saved `seed_lora` under `/workspace/large_artifacts/qwen35_4b_static_bridge_ceiling_breaker/models/seed_lora`.
- Trained `static60_lora` for 2 epochs / 60 optimizer steps on 240 records.
- `static60_lora` training summary: runtime 873.7s, train loss 0.1047, eval loss 0.00004824 on the 24-record training-time eval subset.
- Saved `static60_lora` under `/workspace/large_artifacts/qwen35_4b_static_bridge_ceiling_breaker/models/static60_lora`.
- Trained `static80_lora` for 2 epochs / 60 optimizer steps on 240 records.
- `static80_lora` training summary: runtime 868.4s, train loss 0.1058, eval loss 0.0001918 on the 24-record training-time eval subset.
- Saved `static80_lora` under `/workspace/large_artifacts/qwen35_4b_static_bridge_ceiling_breaker/models/static80_lora`.
- Support eval, `seed_lora`, trace prompt, 3 samples: greedy hidden 61/120, reranked hidden 64/120.
- Support eval, `static60_lora`, trace prompt, 3 samples: greedy hidden 119/120, reranked hidden 120/120.
- Support eval, `static80_lora`, trace prompt, 3 samples: greedy hidden 120/120, reranked hidden 120/120.
- Ceiling eval, `seed_lora`, trace prompt, 3 samples: greedy hidden 15/120, reranked hidden 24/120.
- Ceiling eval, `static60_lora`, trace prompt, 3 samples: greedy hidden 46/120, reranked hidden 53/120.
- Ceiling eval, `static80_lora`, trace prompt, 3 samples: greedy hidden 48/120, reranked hidden 49/120.
- IID eval, `seed_lora`, trace prompt, greedy only: hidden 60/60.
- IID eval, `static60_lora`, trace prompt, greedy only: hidden 60/60.
- IID eval, `static80_lora`, trace prompt, greedy only: hidden 60/60.
- Trace control, `static60_lora`, no-trace prompt, greedy only on ceiling: hidden 18/120.
- Trace control, `static60_lora`, shuffled-trace prompt, greedy only on ceiling: hidden 8/120.
- Generated final report at `reports/qwen35_4b_static_bridge_ceiling_breaker_report.md`.
- Generated figures: `figures/support_ceiling_rerank_hidden.png`, `figures/ceiling_trace_controls.png`, and `figures/ceiling_by_family.png`.

## Main Readout

- Static bridge posttraining saturated the support split: 64/120 reranked hidden for the seed adapter versus 120/120 for both static bridge adapters.
- The same training did transfer to held-out ceiling families, but only partially: 24/120 reranked hidden for the seed adapter, 53/120 for `static60_lora`, and 49/120 for `static80_lora`.
- The 80-record bridge dose did not improve the ceiling result over the 60-record bridge dose, despite equal support saturation and perfect IID retention.
- Trace alignment mattered. On `static60_lora`, aligned ceiling greedy hidden was 46/120, no-trace was 18/120, and shuffled-trace was 8/120.
- Strong transfer families were trace-aligned absent/contains/count compositions: `text_absent_mod_code` 12/12, `text_value_gate_label` 12/12, `token_absent_length_code` 11/12, and `token_count_mod_length_code` 10/12 under `static60_lora`.
- Persistent failures concentrated in sort/join and deeper numeric composition: `sorted_index_sum_branch_label` 0/12, `sorted_join_contains_code` 0/12, `sum_length_mod_gate_label` 1/12, and `sum_len_mod_label` 1/12 under `static60_lora`.

## Final Interpretation

The experiment supports a narrower version of the trace-conditioned bridge hypothesis. A small amount of static bridge posttraining can teach an executable repair interface that generalizes beyond the support families, and the trace controls show the gain is not just prompt formatting. However, support-family saturation was not enough to break the deeper ceiling: the adapter learned some reusable symbolic moves but did not robustly compose sort, join, sum, length, modulo, and branch operations in the hardest held-out families. The next experiment should therefore target breadth and compositional coverage of bridge families rather than increasing the dose of the same support families.
