# Qwen3.5-4B Jacobian Transport Control Replication Report

## Status

`MODEL_SMOKE_PASS`; numeric calibration and confirmation are not run.

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

## Current inference boundary

No transport conclusion is licensed. The frozen 24-item calibration must pass
all 480 numeric rows before the 48-item confirmation can be opened.
