# Post-Model Smoke 001 Audit

Completed after outcome-blind `MODEL_SMOKE_FAIL`. No logits, probabilities,
correct aliases, hidden examples, target pipelines, or outcomes were retained.

## Valid receipts

- All 20 paired non-J rows pass.
- Maximum realized norm error is `9.2639e-6` <= `1e-5`.
- Maximum realized full-J-span projection is `0.0099543` <= `0.01`.
- All 60 full/J/mean/additive/wrong/logit intervention rows pass exact-once,
  finite, and nonzero rules (with preregistered later-layer full-donor zeros).
- Position, within-probe source/donor length, tokenization, lens, model, boundary,
  and donor-immutability contracts pass.
- The exact 512-token prefix is preserved with SHA-256
  `92d2453ef64981746f708e238f4f4560ebf41e4b8c4aa1369d6e3364a9f6fc81`.

## Failure

The cache-free anchor activation differed by `0.078125` between the direct and
consequence contexts, above the frozen `0.001` limit. Direct closed at absolute
position 988 while consequence closed at 1189: their suffixes were 14 versus
216 tokens. The parent short-context clamps had exact zero suffix difference.

This is an implementation/sequence-shape failure, not a scientific outcome. On
Qwen3.5's hybrid recurrent/attention stack, full-sequence scan arithmetic can
change with total sequence shape even at an earlier causal position. Relaxing
the tolerance would erase the clean-donor contract and is forbidden.

## Geometry-only repair

Give both probes the identical public result-label table. Add a frozen 25-token
direct-control instruction before the existing direct query, producing exactly
216 suffix tokens for both probes under the pinned tokenizer. The scientific
endpoints, expected tokens, result table, model, prefix, layer band, coordinate
method, control thresholds, and outcome gates do not change. Add a hard cross-
probe equal-length assertion and retain the `0.001` causal gate unchanged.

This repair is determined solely from token counts and activation geometry. It
does not inspect or authorize outcome logits. A second pushed implementation
boundary is required before one retry.
