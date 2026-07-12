# Qwen3.5-4B Native-Thought Seam Budget Ladder Report

## Status

Terminal `NO_BUDGET_SELECTED`; confirmation ineligible.

## Purpose

The direct parent stopped because all 48 native-thinking traces were still
inside `<think>` at 160 tokens. This separate experiment selects and confirms a
natural-close cap before any continuation-value or causal J-space measurement.

## Method

See `preregistration.md`. The selection split uses paired right-censoring at
256/512/1024; the confirmation split opens only the smallest selected cap. No
close token is injected and no cap-bound trace receives an answer continuation.

## Results

The non-result-bearing smoke loaded the exact pinned 32-layer, 2560-wide model,
verified all special/alias token IDs, rendered a 472-token prompt, and sampled
eight tokens. Its forward-input lengths were `[472, 1, 1, 1, 1, 1, 1, 1]`, so
the prefill-plus-KV-cache contract passed. Correctness was not computed or
stored.

The complete 16-task, three-trace selection stage then produced:

| cap | close | parse/all | usable | mixed tasks | cap contact | pass |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 256 | 0/48 | 0/48 | 0/48 | 0 | 48/48 | no |
| 512 | 0/48 | 0/48 | 0/48 | 0 | 48/48 | no |
| 1024 | 0/48 | 0/48 | 0/48 | 0 | 48/48 | no |

Every trace stopped as `think_cap_without_close` after exactly 1,024 thought
tokens. The stage sampled 49,152 tokens through 49,152 audited cached forwards
in 1,618.080 seconds. Every scientific row passed the prefill/one-token cache
contract. The row file SHA-256 is
`17e3b107154079ecd857af45544c92c2e11b13cd495edfeb6eb24dcf97f5d39c`.

After the automatic terminal decision, token diagnostics found 0/48 exact
periodic tails over the final 256 tokens for periods 1--32. Maximum trigram
reuse ranged from 10 to 34 (median 17.5). Sampled tails showed continued task
analysis and rechecking, but qualitative content is diagnostic only and cannot
rescue the failed close gate.

## Frozen decision

No rung satisfies even the first natural-close requirement, so the smallest-cap
selector returns no cap and writes terminal `NO_BUDGET_SELECTED`. The untouched
24-task confirmation split is ineligible. It was not generated or scored.

## Interpretation

The natural seam is absent through 1,024 steps for this prompt/workload under the
audited cached backend. This repeats the parent's interface failure at a 6.4x
larger ceiling and on fresh tasks, but it still precedes every J-space question.
No evidence about value decodability, certainty, causal transport, or capability
was produced.

The preregistration forbids appending 2,048 or opening confirmation at a larger
cap. The next scientifically distinct route is an explicit forced-commit
controller: inject close as the deployed policy, calibrate autonomous answer
parse/headroom on new tasks, and define prefix value under that exact policy.
Such a state is counterfactual relative to natural closure (C51), so reports must
say so; its legitimacy comes only from testing the same protocol at deployment.
If that seam passes, a later causal J edit must replay the live prefix and build
exact post-bf16 controls per length.

## Interpretation Boundary

This negative is setup evidence only. It establishes that natural closure is
not the usable interface through 1,024 on this workload; it does not lower the
evidence bar for any forced-commit, value, or capability result.

## Artifact Manifest

See `artifact_manifest.yaml`.
