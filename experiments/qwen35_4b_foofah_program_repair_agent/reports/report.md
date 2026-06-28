# Foofah Program-Repair Agent

## Question

Can Qwen3.5-4B improve table transformation accuracy by writing an executable `transform(table)` program, observing visible-example failures, and repairing the program over several rounds before held-out execution?

The benchmark is Foofah (`https://github.com/markjin1990/foofah_benchmarks`), scored by exact equality to held-out `TestAnswer` tables.

## Result

| arm | exact held-out | rate |
| --- | ---: | ---: |
| Direct JSON generation | 138/250 | 55.2% |
| Initial visible-verified program | 40/250 | 16.0% |
| Final repaired visible-verified program | 62/250 | 24.8% |
| Direct with program fallback on direct parse failure | 142/250 | 56.8% |
| Oracle union: direct OR final program | 156/250 | 62.4% |

Repair raised visible-verified program correctness from 40 to 62 cases, adding 22 program-correct cases while losing 0.

The final program arm contributed 18 direct-miss recoveries. The oracle union reached 156/250, a +18 case headroom over direct JSON generation.

## Verification Risk

Final programs passed the visible example on 78/250 cases. Of those, 16 were hidden-wrong, a false-pass rate of 20.5%.

Direct/program agreement occurred on 55 cases, with 44 correct (80.0% precision).

## Repair By Round

| round | attempted | visible pass | visible-pass and hidden-correct |
| ---: | ---: | ---: | ---: |
| 0 | 250 | 50 | 40 |
| 1 | 200 | 20 | 16 |
| 2 | 180 | 5 | 5 |
| 3 | 175 | 3 | 1 |

Mean code-generation rounds per case: 3.22. Including direct JSON generation, the mean model-generation calls per case were about 4.22.

## Iteration

Before the full run, the repair prompt was tested and revised:

| smoke | n | initial program | final program | oracle union | program-only | visible false-pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prefix6 r2 | 6 | 83.3% | 83.3% | 100.0% | 0 | 0.0% |
| standard spread25 r2 | 25 | 8.0% | 8.0% | 44.0% | 0 | 71.4% |
| strict spread25 r2 | 25 | 8.0% | 16.0% | 52.0% | 2 | 42.9% |

The first hard-spread repair prompt increased visible-pass but added no hidden-correct programs. The stricter repair prompt explicitly warned against visible-output hardcoding and improved the same spread from 2 to 4 final program-correct cases, with 2 program-only recoveries.

## Diagnostics

Program-only recoveries: exp0_11_5.txt, exp0_13_3.txt, exp0_13_4.txt, exp0_13_5.txt, exp0_22_1.txt, exp0_45_2.txt, exp0_51_3.txt, exp0_51_5.txt, exp0_5_3.txt, exp0_5_4.txt, exp0_8_1.txt, exp0_8_2.txt, exp0_8_4.txt, exp0_8_5.txt, exp0_potters_wheel_divide_3.txt, exp0_potters_wheel_merge_split_3.txt, exp0_potters_wheel_merge_split_4.txt, exp0_potters_wheel_merge_split_5.txt.

Repair-added program-correct files: exp0_11_2.txt, exp0_11_3.txt, exp0_11_4.txt, exp0_11_5.txt, exp0_22_1.txt, exp0_26_3.txt, exp0_27_3.txt, exp0_27_5.txt, exp0_33_5.txt, exp0_40_2.txt, exp0_40_3.txt, exp0_40_4.txt, exp0_40_5.txt, exp0_47_3.txt, exp0_47_4.txt, exp0_5_4.txt, exp0_5_5.txt, exp0_8_1.txt, exp0_8_2.txt, exp0_8_4.txt, exp0_potters_wheel_divide_3.txt, exp0_proactive_wrangling_fold_4.txt.

Visible-pass hidden-wrong files: exp0_13_1.txt, exp0_22_2.txt, exp0_22_3.txt, exp0_22_4.txt, exp0_24_1.txt, exp0_24_2.txt, exp0_26_1.txt, exp0_27_1.txt, exp0_29_4.txt, exp0_33_1.txt, exp0_34_1.txt, exp0_40_1.txt, exp0_48_3.txt, exp0_5_1.txt, exp0_potters_wheel_merge_split_1.txt, exp0_potters_wheel_unfold2_2.txt.

Agreement-hidden-wrong files: exp0_22_2.txt, exp0_22_3.txt, exp0_22_4.txt, exp0_24_3.txt, exp0_24_5.txt, exp0_26_1.txt, exp0_27_1.txt, exp0_34_1.txt, exp0_40_1.txt, exp0_48_3.txt, exp0_potters_wheel_unfold2_3.txt.

## Read

The repair loop produced a real coverage gain over one-shot program induction: execution feedback converted additional failed programs into held-out-correct programs. It is not a standalone replacement for direct generation, but it is a complementary tool path.

The deployability issue remains selection. Visible-example verification is useful but incomplete; about one fifth of visible-passing final programs were hidden-wrong. The clean positive signal is the oracle union and the direct-miss recoveries, not naive commit-on-visible-pass.

## Caveats

- This is greedy single-sample direct generation and greedy repair generation.
- The loop stops at the first visible-passing program, matching deployment where hidden answers are unavailable.
- Generated code is sandboxed and limited to safe builtins plus `re`, `math`, `Counter`, and `defaultdict`.
- Exact table matching is strict after converting cells to strings.
