# Parent Failure Inventory

The frozen model-free miner authenticated the 288-row parent rollout, graded only the
experiment-owned procedural tasks, and satisfied every preregistered class quota. This
is substrate evidence, not evidence that prefix-repair training improves capability.

## Outcome inventory

- 230 of 288 parent rows met at least one frozen failure condition; 58 passed all
  registered conditions.
- All 230 failed rows exposed a clean reachable thinking-channel prefix; zero failed
  rows were excluded as unreachable.
- Failure-reason occurrences were 114 wrong answers, 73 generation-cap contacts, 73
  missing answers, 48 delayed commits, and one declaration-as-operation detection.
  Reasons overlap within a row.
- Inventory SHA-256: `7230af523b4e5036ea3f9191a4711fbe27992bb91f792ef62887c571576ddfe7`.

## Frozen selection

| Failure class | Reachable failures | Selected | Prefix tokens: min | mean | max | sum |
|---|---:|---:|---:|---:|---:|---:|
| declaration / operation | 35 | 10 | 149 | 852.1 | 1,024 | 8,521 |
| state transition | 41 | 10 | 1,024 | 1,024.0 | 1,024 | 10,240 |
| bounded induction | 46 | 10 | 1,024 | 1,024.0 | 1,024 | 10,240 |
| probe scoring | 24 | 10 | 181 | 876.6 | 1,024 | 8,766 |
| repair propagation | 36 | 10 | 303 | 902.6 | 1,024 | 9,026 |
| commit serialization | 48 | 10 | 33 | 33.0 | 33 | 330 |
| **Total** | **230** | **60** | **33** | **785.4** | **1,024** | **47,123** |

The selected source has SHA-256
`301415384c941e158c7d97e0368e5026533648c01d7af8540d9ae791ba4d84b8`.
Selection boundaries are 42 generation-cap boundaries, ten first tokens beyond the
commit budget, and eight answer boundaries. Forty-two rows carry
`generation_cap + missing_answer`, six commit rows additionally carry those two
reasons but cut at the earlier registered commit boundary, four commit rows carry
only delayed commit, and eight rows carry only wrong answer.

## Compute-review implication

The quota risk is closed, but the selected treatment is deliberately severe: 47,123
masked parent-prefix tokens precede the correction targets, and 48 selected rows
contacted the generation cap. The second adversarial review must therefore report
exact encoded sequence lengths, zero skips, masked versus loss-bearing tokens, and
the exact-forward-token replay match. No training stage is authorized before that
receipt is committed and both repository workflows pass.
