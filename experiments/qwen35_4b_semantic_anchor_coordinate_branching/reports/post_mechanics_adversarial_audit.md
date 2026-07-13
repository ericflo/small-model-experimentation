# Post-Mechanics Adversarial Audit

Completed after the one authorized mechanics run and before any continuation or
shared scientific interpretation. This audit cannot change frozen gates.

## Formal verdict

Retain `INVALID_MECHANICS_CONTROL` with the qualifier **parse**. The numeric and
intervention firewalls pass; the frozen unrestricted-output contract does not.

- 880/880 unique finite outcome rows exist.
- 880/880 unique numeric rows pass.
- 2,240/2,240 unique intervention rows pass.
- All four exact native prefixes match; boundary ancestry and donor immutability
  pass.
- Mechanics and calibration row objects are identical after the frozen unique
  identity sort. Raw JSONL files are not byte-identical because row order
  differs, correcting the stronger pre-run wording.
- Full-vocabulary parse is 56/880 (`0.063636`) versus `0.95`; direct parse is
  56/440 and consequence parse is 0/440.
- Every consequence full top is a formatting opener: 294 leading-space
  backticks and 146 leading-space opening brackets.

There is no off-by-one or wrong-token-ID evidence. The frozen score is the next
token after `Result label:` and the registered leading-space concept IDs are
correct. Generating past the formatting opener or accepting optional prefixes
would be an unregistered post-hoc repair and is prohibited here.

## Constrained diagnostic outcomes

The 12-way conditional distribution shows a direct write but no computed
transport:

| Arm | Direct target | Consequence target |
| --- | ---: | ---: |
| literal target text | 43/44 | 6/44 |
| full donor | 43/44 | 6/44 |
| donor J | 42/44 | 5/44 |
| additive J | 38/44 | 5/44 |
| worse non-J | 0/44 | 4/44 |

Wrong-donor J chooses its own direct alias on 42/44 and the registered target on
0/44, but its own consequence label on only 5/44. Donor-J consequence lift over
source is `0.0017007` versus `0.15`; rate advantage over the worse non-J is
1/44 versus `0.35`. Consequence successes cover only four aliases and four
labels. Ignoring parse counterfactually would therefore stop at
`ANCHOR_PROBE_UNREACHABLE`, not open J continuation.

## Critical composed-permutation confound

The advertised randomized endpoint is not present. Alias-to-operation and
operation-to-result-label are each cyclically shifted by `+task_index`. Those
shifts cancel under composition, leaving the same alias-to-label map in all four
mechanics tasks (for example, `cat -> north` and `horse -> winter`). The count of
unique component maps is four; the count of unique composed maps is one.

This defect defeats the anti-shortcut rationale. Alias breadth, label breadth,
and task support could not prove prompt-local computation even if they passed.
Future designs must test the composition directly, not merely test that each
component permutation changes or responds to a separate seed.

## What can and cannot be concluded

Defensible:

- continuation must remain sealed;
- the exact late opaque-anchor/one-token consequence interface is invalid and
  unreachable;
- direct conditional alias writing does not imply downstream consequence
  transport; and
- donor J supplies no useful conditional consequence advantage over text,
  source, or matched non-J in this run.

Not defensible:

- a general `NO_NATIVE_ANCHOR_STATE_TRANSPORT` conclusion;
- a general negative about J-space semantic transport;
- a computation claim from the five constrained donor-J successes; or
- any capability, certainty, selection, or posttraining claim.

Literal text and full donor also fail the consequence interface, so this run
cannot isolate J as the bottleneck. The fixed composition further prevents the
intended computation test.

## Decision

Freeze this result-bearing experiment. Do not change the parser, regenerate
labels, add layers/alphas, or open correctness here. The warranted fresh
successor moves concrete text hypotheses before reasoning, generates naturally
parseable full continuations, tests composed-map independence automatically,
and must beat matched-compute sampling with visible-only selection before it can
count as deployable elicitation.
