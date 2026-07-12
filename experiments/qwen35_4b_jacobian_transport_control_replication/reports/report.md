# Qwen3.5-4B Jacobian Transport Control Replication Report

## Status

`REPLICATED_J_TRANSPORT` on the single untouched confirmation run.

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

The calibration boundary was committed and pushed before the hash-locked runner
opened confirmation. The runner itself was then committed, pushed, and green in
CI before the one result-bearing run.

## Untouched confirmation

| Intervention | Direct target | Mapped target | Own wrong target |
| --- | ---: | ---: | ---: |
| Baseline | 0/48 | 0/48 | 0/48 |
| Full target donor | 48/48 | 48/48 | 0/48 |
| All-24 J target clamp | 48/48 | 48/48 | 0/48 |
| Source/target pair J | 48/48 | 46/48 | 0/48 |
| Wrong-donor J | 0/48 | 0/48 | 48/48 |
| Concept logit lens | 0/48 | 0/48 | 0/48 |
| Random A | 0/48 | 0/48 | 0/48 |
| Random B | 0/48 | 0/48 | 0/48 |

All arms parsed on 48/48 items. Mean target-minus-source margin moved from
`-11.8079` to `+11.1797` direct and from `-9.4245` to `+8.2604` on the mapped
consequence. Both random arms stayed at baseline-like negative margins. Both
paired 10,000-resample J-minus-random 95% intervals were `[1.0, 1.0]`.

Every one of 960 confirmation random-control layer deltas passed after bf16
application. Maximum norm error was `9.9709e-6` and maximum J-span projection
fraction was `0.0099970`; 47 rows used exact lattice repair and no row required
more than three coordinate pairs. Exact causal-suffix activation difference was
zero.

## Frozen decision

Every clean, donor, direct-shift, consequence-shift, worse-random, two-bootstrap,
wrong-donor specificity, parse, causal, and numeric gate passes. The frozen
terminal label is `REPLICATED_J_TRANSPORT`.

## Interpretation and limits

The late answer-position Jacobian in the grandparent was writable but did not
transport. In contrast, the early selected-token clamp controls both the concept
and a later computation that consumes it, survives a fresh replication, and is
specific to donor identity. This is strong evidence for a compact causally
consumed concept state in this prompt-local task.

It is still an oracle mechanism. The target concept and clean donor coordinates
are supplied; the task is a procedural lookup; there is no learned controller,
native `<think>` prefix, installed capability, or comparison to matched-compute
sampling. The result licenses those experiments but cannot substitute for them.
No repository claim ID is allocated while the ledger re-grade remains open.
