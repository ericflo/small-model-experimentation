# Qwen3.5-4B Forced-Commit Jacobian Value Transport Report

## Status

Terminal `FORCED_COMMIT_SEAM_FAIL`; all later stages ineligible.

## Purpose

Natural closure failed on 48/48 traces through 1,024 tokens. This successor
treats close injection as the explicit deployed budget-controller action, first
validates that interface on fresh data, and only then asks whether forced-policy
continuation value is readable and causal in J space.

## Method

See `preregistration.md`. Selection tests 256/512/1024 paired prefixes and freezes
the smallest forced-commit policy meeting parse, headroom, mixed-task, and answer
termination gates. Untouched confirmation opens only that cap. Later value and
causal stages are strictly gated and currently refuse placeholders.

## Results

CPU smoke produced 96/96 unique fresh exact-depth-two tasks, zero overlap with
all three scientific parents, no visible depth-one fits, exact replicated lens
hash, and reachable seam gates.

The non-result-bearing model smoke verified the exact pinned model, special and
alias token IDs, and rank 24 at lens layers 4--8. Its native trace forward input
lengths were `[375, 1, 1, 1, 1, 1, 1, 1]`; forced replay appended eight thought
tokens plus close and used `[384, 1]`. Both cache audits passed. Close injection
was explicitly marked counterfactual and no correctness was computed or stored.
The scientific seam selection then completed 48 traces and 144 paired policy
rows. All 48 traces contacted 1,024, so every cap used forced close.

| cap | forced parse | exact success | mixed tasks | answer-cap contact |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 6/48 (0.125) | 1/48 (0.0208) | 1 | 44/48 (0.9167) |
| 512 | 8/48 (0.1667) | 1/48 (0.0208) | 1 | 41/48 (0.8542) |
| 1024 | 9/48 (0.1875) | 1/48 (0.0208) | 1 | 46/48 (0.9583) |

None approached the frozen 90% overall/forced parse gates, 5% success floor,
six mixed tasks, or <=5% answer-cap contact. The run sampled 49,152 thought and
2,225 answer tokens in 1,640.539 seconds. All trace and replay cache audits
passed. Trace SHA-256 is
`af5ed5ef6df892d98760c51c4881a1e29da911fb74549bebfe1b17be0cf6fbd8`;
policy rows SHA-256 is
`432668875b5183d1a32f649e41686951362bf3e88c6c374ee7045903b2c2cfe6`.

## Post-decision parser diagnostic

Some correctly shaped answers placed the special EOS token directly after the
alias, which the frozen whitespace parser rejected. A regex diagnostic accepting
that suffix raised parse counts from 6/8/9 to 7/11/10 and correct counts from
1/1/1 to 1/2/2. Robust parse was still only 14.6%/22.9%/20.8%, answer-cap contact
was unchanged, and every gate remained far out of reach. The registered result
therefore remains valid; parser relaxation cannot rescue it.

Decoded rows inspected only after the automatic decision show the primary
failure: after `</think>`, the model frequently began a new analysis, explained
a free-form operation sequence, or refused to commit rather than entering the
requested `First:` slot.

## Frozen decision

No cap passes, so selection returns no cap and terminal
`FORCED_COMMIT_SEAM_FAIL`. The untouched seam-confirmation, value-fit, and
causal-confirmation task sets remain unopened. No J-space conclusion is licensed.

## Interpretation

Close-token injection is not equivalent to an answer-emission seam for this
model/workload. This independently echoes C51: a counterfactual post-thinking
state can exist yet fail deployment-shaped expression. The failed dimension is
now localized more sharply than “forced close”: the controller supplied a mode
delimiter but not an output slot, and the model often re-entered analysis.

The next experiment must be distinct and fresh. A fixed commit-slot controller
may append `</think>` plus syntax `First:` and then measure the next alias choice,
with the close-only free-form policy retained as a control. Supplying syntax does
not supply answer identity. Constrained/slot behavior must be reported as its own
deployment interface, not natural reasoning or a capability gain. Only after
that interface preserves correctness headroom may J value reopen.

## Interpretation Boundary

Injected close remained counterfactual and unusable. The terminal result is an
interface negative, not evidence about value, J-space causality, or installed
capability.

## Artifact Manifest

See `artifact_manifest.yaml`.
