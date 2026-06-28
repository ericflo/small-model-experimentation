# Foofah Ephemeral Program Induction

## Question

Can Qwen3.5-4B improve external table transformations by writing a bespoke executable `transform(table)` function, verifying it on the visible example, and executing it on the held-out input?

The benchmark is Foofah (`https://github.com/markjin1990/foofah_benchmarks`), scored by exact equality to held-out `TestAnswer` tables.

## Result

| arm | exact held-out | rate |
| --- | ---: | ---: |
| Direct JSON generation | 138/250 | 55.2% |
| Visible-verified generated program | 38/250 | 15.2% |
| Direct with program fallback on direct parse failure | 139/250 | 55.6% |
| Oracle union: direct OR program | 148/250 | 59.2% |

Program induction found code for 250/250 cases and executed on the visible example for 202/250.
It passed the visible example on 47/250, but 9 of those visible-pass programs were hidden-wrong.

## Overlap

| bucket | count |
| --- | ---: |
| Both direct and program correct | 28 |
| Direct only | 110 |
| Program only | 10 |
| Neither | 102 |

Agreement between direct output and program execution occurred on 34/250 cases, with precision 28/34 (82.4%).

Program-only recoveries: exp0_13_2.txt, exp0_13_4.txt, exp0_13_5.txt, exp0_45_2.txt, exp0_51_5.txt, exp0_8_2.txt, exp0_potters_wheel_merge_split_2.txt, exp0_potters_wheel_merge_split_3.txt, exp0_potters_wheel_merge_split_4.txt, exp0_potters_wheel_merge_split_5.txt.

Agreement-hidden-wrong cases: exp0_22_2.txt, exp0_22_3.txt, exp0_22_4.txt, exp0_27_1.txt, exp0_34_1.txt, exp0_40_1.txt.

## Iteration

Before the full run, four prompt smokes were run:

| prompt smoke | n | direct | program | oracle union | agreement precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| induce prefix8 | 8 | 100.0% | 62.5% | 100.0% | 100.0% |
| context prefix8 | 8 | 100.0% | 87.5% | 100.0% | 100.0% |
| context spread25 | 25 | 44.0% | 12.0% | 48.0% | 33.3% |
| context_v2 spread25 | 25 | 44.0% | 12.0% | 44.0% | 75.0% |

The full run used `context`: it had lower agreement precision than `context_v2` on the hard spread, but it preserved the only direct-failure recovery in that spread and therefore had higher coverage headroom.

## Read

The generated-program route tests a tool-use idea: the model emits a bespoke executable artifact, the artifact is checked on the visible example, and the checked artifact is executed on the held-out input. The result should be read through coverage and selection separately.

The executable-program arm is real but weak on this benchmark. It creates some correct programs outside direct generation (`program_only=10`), but visible-example verification is not enough to make it deployable: false-pass among visible-pass programs is 9/47 (19.1%).

The decisive number is the oracle union. If it is meaningfully above direct generation, there is headroom for a better selector or verifier over direct-vs-program outputs. If it is close to direct generation, ephemeral program induction is not adding much capability on Foofah.

## Caveats

- This is greedy single-sample direct generation and greedy single-sample code generation.
- Generated code is sandboxed and limited to safe builtins plus `re`, `math`, `Counter`, and `defaultdict`.
- The program is verified only on the visible example before held-out execution; visible-pass hidden-wrong is expected and measured.
- Exact table matching is strict after converting cells to strings.
