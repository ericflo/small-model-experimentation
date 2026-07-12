# Qwen3.5-4B Jacobian Transport Control Replication Report

## Status

`CONTROL_CALIBRATION_PASS`; confirmation is not run.

## Frozen purpose

Replicate the parent context-local semantic-transport effect on fresh mappings
while repairing its post-bf16 random-control failure. See `preregistration.md`
and `design_review.md`.

## Outcome-blind model smoke

The fourth, repaired smoke passes all 20 random layer deltas across direct and
consequence prompts, two independent arms, and layers 4--8:

- maximum relative norm error: `9.0113e-6` (gate `1e-5`);
- maximum realized J-span projection fraction: `0.0098674` (gate `0.01`);
- exact causal-suffix activation difference: `0.0`;
- four layer-8 rows used exact neighboring-bf16 pair repair, with at most two
  pairs; and
- no outcomes or logits are present in the smoke receipt.

Failed attempts 001--003 are preserved beside the passing receipt. They are
engineering evidence about sequential bf16 quantization, not scientific model
outcomes. The post-smoke section of `design_review.md` audits the implementation
repair before calibration.

## Numeric calibration firewall

The frozen 24-item calibration passed all 480 expected rows:

- maximum relative norm error: `9.8216e-6`;
- maximum realized J-span projection fraction: `0.00999293`;
- 37 rows used exact lattice repair, including 34/96 layer-8 rows;
- maximum lattice repair: three neighboring-bf16 coordinate pairs;
- both prompt kinds and both random arms contribute exactly 240 rows each; and
- exact causal-suffix difference: `0.0`.

The row schema contains only item/prompt/arm/layer identity, candidate and
iteration indices, delta norms, the two numeric errors, lattice pair count, and
pass state. The summary records `logits_recorded=false` and
`outcomes_recorded=false`.

## Current inference boundary

No transport conclusion is licensed yet. Calibration unlocks implementation of
the preregistered confirmation runner, but the 48 untouched mappings remain
unopened until this numeric boundary is committed and pushed.
