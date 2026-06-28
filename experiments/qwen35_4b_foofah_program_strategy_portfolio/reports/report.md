# Qwen3.5-4B Foofah Program Strategy Portfolio

## Question

Can a small, searched portfolio of executable program-generation strategies improve Foofah table-transformation accuracy over direct JSON generation on held-out task families?

The experiment searches strategy prompts on train/dev families, freezes the selected portfolio, and evaluates it once on held-out families. Hidden answers are used for measurement and for train/dev strategy selection only, never as inputs to generation.

## Selected Portfolio

- Selected variants: `verified_structural, cell_parser, row_column_rule, header_aware, split_fold_unpivot`
- Selected policy: `consensus_2`
- Selection rule: greedy ordered variants on train; choose prefix and selector maximizing dev exact accuracy with loss/precision/token tie-breaks

## Main Held-Out Result

| arm | exact | rate | program commits | program precision | direct-miss recoveries | direct-correct losses | forward tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct JSON | 21/50 | 42.0% | 0 | - | 0 | 0 | 73911 |
| Selected portfolio | 24/50 | 48.0% | 14 | 78.6% | 4 | 1 | 890030 |
| Direct OR selected-program oracle | 29/50 | 58.0% | - | - | - | - | - |

## Held-Out Selector Tradeoff

| selector | exact | rate | program commits | program precision | recoveries | losses |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `direct` | 21/50 | 42.0% | 0 | - | 0 | 0 |
| `first_visible_program` | 28/50 | 56.0% | 23 | 78.3% | 8 | 1 |
| `consensus_2` | 24/50 | 48.0% | 14 | 78.6% | 4 | 1 |
| `consensus_3` | 24/50 | 48.0% | 11 | 90.9% | 3 | 0 |

## Figures

- `reports/figures/accuracy_by_split.png`
- `reports/figures/portfolio_prefix_search.png`
- `reports/figures/variant_quality.png`
- `reports/figures/test_selector_tradeoff.png`

## Read

The primary dev-selected policy is a real but modest positive: `consensus_2` improves held-out accuracy from 21/50 to 24/50, with 4 direct-miss recoveries and 1 direct-correct loss. It is safer than committing every visible-passing program, but it leaves a large share of candidate coverage unused.

The strongest predeclared held-out selector is `first_visible_program`: 28/50, with 8 direct-miss recoveries and 1 direct-correct loss. This nearly reaches the direct-or-program oracle of 29/50. It was not chosen by the dev selector, so it should be read as an important selector-mismatch finding rather than as the primary frozen-policy result.

The gain is highly family-concentrated. `potters_wheel_merge_split` moves from 0/5 direct to 4/5 with first-visible programs, and `synthetic_8` moves from 0/5 direct to 4/5. Other families, especially `crime_data_wrangler`, `potters_wheel_unfold`, and `potters_wheel_unfold2`, mostly consume repair budget without producing deployable correct programs.

The cost is substantial: direct generation used 73,911 forward tokens on test, while the five-strategy portfolio used 890,030. This is not yet an efficient policy. The experiment establishes that strategy portfolios can create real held-out executable coverage, but the next iteration should be an adaptive budget/router that spends program attempts only on families or cases likely to benefit.

The most actionable result is therefore not "use all five strategies everywhere." It is: executable program strategies create complementary candidates, visible-pass first fallback captures most of the candidate oracle on some structural families, and the dev-selected consensus rule is too conservative for singleton-correct strategy hits.

## Family Readout

| family | direct | first-visible | consensus_2 | oracle |
| --- | ---: | ---: | ---: | ---: |
| agriculture | 4/5 | 3/5 | 3/5 | 4/5 |
| crime_data_wrangler | 0/5 | 0/5 | 0/5 | 0/5 |
| potters_wheel_merge_split | 0/5 | 4/5 | 3/5 | 4/5 |
| potters_wheel_unfold | 2/5 | 2/5 | 2/5 | 2/5 |
| potters_wheel_unfold2 | 0/5 | 0/5 | 0/5 | 0/5 |
| proactive_wrangling_fold | 5/5 | 5/5 | 5/5 | 5/5 |
| synthetic_12 | 5/5 | 5/5 | 5/5 | 5/5 |
| synthetic_25 | 4/5 | 4/5 | 4/5 | 4/5 |
| synthetic_48 | 1/5 | 1/5 | 1/5 | 1/5 |
| synthetic_8 | 0/5 | 4/5 | 1/5 | 4/5 |

## Iteration Notes

- Eight strategy prompts were smoke-tested first. The full eight-strategy sweep was too slow, and three variants were repair-heavy or non-incremental in the smoke.
- The train/dev pilot used five pruned strategies on one case per train/dev family. Train showed direct 9/30, consensus_2 10/30, oracle 12/30. Dev showed direct 4/10, consensus_2 5/10, oracle 5/10.
- The frozen held-out run evaluated all 50 cases from the 10 test families with the dev-selected five-strategy portfolio.
- The held-out split revealed a mismatch: dev favored consensus_2, but singleton correct visible-program hits on `synthetic_8` made first-visible much better on test.

## Caveats

- Family-heldout evaluation is stricter than a random case split but still uses one external benchmark.
- Program candidates are verified on the visible example only; hidden answers are used solely for evaluation and train/dev strategy selection.
- The selected strategy portfolio is greedy and small; it is not a global optimum over all possible prompts.
- The dev split is only 10 cases, so selector choice is noisy; this run directly showed that dev selected a conservative policy that underfit singleton-correct held-out families.
- The portfolio is expensive because every test case receives five program attempts and visible-feedback repairs. Accuracy gains should not be read without the token-cost column.
- Exact table matching normalizes all cells to strings and requires exact row/column equality.
