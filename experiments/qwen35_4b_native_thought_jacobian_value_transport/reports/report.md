# Qwen3.5-4B Native-Thought Jacobian Value Transport Report

## Status

Terminal `NO_NATURAL_SEAM`; value and causal stages canceled.

## Purpose

Transfer the independently replicated context-local J mechanism from a
prompt-local lookup token to a natural native-thinking token, while separating a
task-general scalar continuation-value coordinate from full answer identity.

## Frozen method

- Fresh, first-operation-identifiable procedural list tasks only.
- Frozen replicated 24-token lens and layer band 4--8.
- Natural close, full-prefix batch-one Transformers recomputation without cache.
- Continuation-defined prefix value, held-out-by-task linear J-space readout,
  and a minimum-norm scalar coordinate clamp.
- Exact quantization-aware random controls and explicit identity, shuffled-axis,
  logit, raw, J/non-J, ActAdd, and wrong-donor arms.

## CPU evidence

All 80 generated tasks are fingerprint-unique and disjoint from the direct
Jacobian parent. Exhaustive enumeration certifies one visible-data-consistent
first-operation type. The CPU adversarial pass rejected `negate` as a target
because its two-step compositions are algebraically reorderable; it remains a
prompt distractor/second operation. This prevents alternate valid explanations
from being counted as model failures.

## Current inference boundary

Revision, token, alias, lens rank, generation, capture, and J-coordinate plumbing
pass. Both two-task smoke traces reached the frozen 160-token cap without natural
close. Historical-token activations also changed by 0.0625 when evaluated under
different suffix lengths, above the frozen causal-invariance threshold. The
smoke itself was not a scientific sample and recorded no correctness.

## Frozen seam result

The 16-task, three-trace-per-task scientific seam was opened once with unchanged
sampling and the frozen 160-token thought cap.

| metric | result | gate |
| --- | ---: | ---: |
| Natural close | 0/48 (0.0%) | >=80% |
| Parseable alias | 0/48 (0.0%) | >=90% |
| Exact alias success | 0/48 (0.0%) | 5%--95% |
| Mixed-value tasks | 0/16 | >=6 |
| Thought-cap contact | 48/48 | diagnostic |

Every row stopped as `think_cap_without_close`, with exactly 160 thought tokens.
The full-recompute runner made 7,632 forwards in 389.9 seconds. There is no
natural reachable prefix-to-answer seam at this budget and prompt grammar.

## Frozen decision

Decision: `NO_NATURAL_SEAM`. Per preregistration, G1 value fitting, post-bf16
control calibration, donor selection, and causal confirmation are ineligible.
No outcome from the untouched 32 causal tasks was opened.

## Interpretation

The failure precedes the J-space hypothesis. The model continued reasoning until
the cap on every trace, so assigning value at 0.33/0.67 and demanding a natural
answer continuation would evaluate prefixes whose answer seam is never reached
under the allowed budget. The 0.0625 historical-activation length sensitivity
also warns that later patching needs per-length dynamic controls rather than a
fixed raw-activation invariance assumption.

A separate successor may preregister a natural-close budget ladder on a selection
split, freeze the smallest viable cap, and match control geometry at every live
sequence length. Raising 160 or weakening invariance in this result-bearing
experiment is forbidden.

## Oracle boundary

Continuation labels and high/low donor selection are hidden-label oracle inputs.
Even a terminal causal pass cannot support a capability claim or replace the
required learned non-oracle matched-sampling experiment.
