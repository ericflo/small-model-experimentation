# Post-Smoke Prefix-Value Audit

Completed after the one outcome-blind value-model smoke and before opening
`value_fit` or running a scientific prefix row.

## Verdict

Proceed with the single registered prefix-value run after committing and pushing
this receipt. The smoke passed every implementation/firewall check and recorded
no task outcome. `causal_confirmation` remains sealed.

## Receipt checks

- Exact model `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, 32 layers, width 2,560.
- Frozen lens hash exact; layers 4--8 each effective rank 24.
- Live feature context contained prompt plus eight thought tokens only, ended at
  position 383 in a 384-token sequence, and explicitly contained no close or
  slot.
- The separate forced-slot prefill was 388 tokens: exactly the feature context
  plus one close and the three frozen slot tokens `[271, 5170, 25]`.
- J feature width was 120 and finite.
- Layer-matched non-J feature width was 120 and finite. Maximum projection back
  into J-space was `2.6700374178290076e-7`, far below `1e-5`.
- Cached eight-token generation passed its input-length contract.
- `outcomes_recorded`, `correctness_recorded`, `chosen_alias_recorded`, and
  `trace_text_recorded` were all false.
- `value_split_opened` and `causal_split_opened` were both false; the seam
  license recorded only the expected value-split hash from the already-committed
  confirmation receipt.
- Peak GPU allocation was 8,510,865,408 bytes. No recovery or backend change
  occurred.

No threshold, fold, feature, layer, seed, label, or interpretation changed after
this smoke.
