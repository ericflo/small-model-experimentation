# Foofah Program Ensemble Consensus

## Question

Can Qwen3.5-4B improve Foofah table-transformation accuracy by generating several independently prompted executable `transform(table)` programs, verifying them on the visible example, and selecting by output consensus on the held-out input?

The benchmark is Foofah (`https://github.com/markjin1990/foofah_benchmarks`). Hidden answers are used only for evaluation.

## Setup

Each of 250 tasks was evaluated with one direct JSON answer and three program variants:

- `verified_structural`
- `structural_python`
- `row_column_rule`

Each program variant received one visible-feedback repair attempt if the initial program failed the visible example. A program candidate was eligible for selection only if it passed the visible example and executed on the held-out input.

## Main Result

| selector | exact held-out | rate | program commits | program precision | direct-miss recoveries | direct-correct losses |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct JSON | 111/250 | 44.4% | 0 | - | 0 | 0 |
| Direct/program agreement only | 111/250 | 44.4% | 0 | - | 0 | 0 |
| Program consensus >= 3 | 114/250 | 45.6% | 19 | 84.2% | 3 | 0 |
| Program consensus >= 2 | 122/250 | 48.8% | 53 | 81.1% | 11 | 0 |
| First visible-passing program | 130/250 | 52.0% | 101 | 73.3% | 23 | 4 |

The best deployed policy was **first visible-passing program**: 130/250 (52.0%), versus direct JSON at 111/250 (44.4%). It recovered 23 direct misses but lost 4 direct-correct cases.

The oracle union of direct JSON or any visible-correct program reached 135/250 (54.0%). That leaves 5 cases of selector headroom after the best deployed policy.

## Consensus

Consensus was safer but too conservative:

- Consensus >= 2 committed 53 times with 81.1% precision, recovering 11 direct misses.
- Consensus >= 3 committed 19 times with 84.2% precision, recovering 3 direct misses.

There were 101 tasks with at least one visible-passing program, 57 with at least two visible-passing programs, and 53 with an output cluster of size at least two. The ensemble had 24 direct-miss tasks where at least one visible program was hidden-correct.

## Variant Diagnostics

| variant | visible pass | visible precision | initial visible pass | repair-added visible |
| --- | ---: | ---: | ---: | ---: |
| row_column_rule | 64/250 (25.6%) | 79.7% | 52 | 12 |
| structural_python | 57/250 (22.8%) | 73.7% | 29 | 28 |
| verified_structural | 59/250 (23.6%) | 71.2% | 53 | 6 |

## Iteration

The first six-case smoke exposed direct-output parsing fragility, so JSON extraction was updated to accept the first valid array prefix and the direct prompt was tightened to forbid prose or markdown.

After that fix, a six-case smoke reached direct 5/6 and first-visible program 5/6. A harder stride-10 smoke with the first prompt set showed no oracle gain and poor visible-program precision, so the weak minimal-code variant was replaced with `verified_structural`.

The full run used the revised three-variant ensemble with one repair round per variant.

## Read

The ensemble generated real additional candidate coverage. Direct JSON solved 111 tasks; direct plus any visible-correct program could solve 135. The simple first-visible selector captured most of that gain, reaching 130.

The specific hypothesis that independent program consensus would be the best selector did not hold. Consensus improved precision over first-visible fallback but under-recovered too many direct misses. On this benchmark, visible-example pass plus fixed prompt order was a better deployed selector than requiring agreement.

## Caveats

- The full run uses one greedy direct answer and one greedy generation per program variant, with one greedy repair attempt per variant.
- The ensemble has three prompt variants; larger or sampled ensembles may change the coverage/precision tradeoff.
- Program execution is sandboxed and limited to safe builtins plus `re`, `math`, `Counter`, and `defaultdict`.
- Exact table matching is strict after converting cells to strings.
