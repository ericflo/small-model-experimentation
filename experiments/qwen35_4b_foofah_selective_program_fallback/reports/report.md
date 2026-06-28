# Foofah Selective Program Fallback

## Question

Can Qwen3.5-4B improve deployed Foofah table-transformation accuracy by using a visible-verified executable `transform(table)` program as a fallback to direct JSON generation, and do counterexample-style probe inputs make that fallback decision safer?

The benchmark is Foofah (`https://github.com/markjin1990/foofah_benchmarks`). Each task has visible input-output examples and a held-out test input. Hidden answers are used only for evaluation.

## Candidate Pool

The package contains 250 task records. Each record has:

- one direct JSON answer for the held-out input,
- one executable program candidate after visible-example repair,
- visible-example execution status for the program,
- held-out exact-match labels for evaluation.

## Main Result

| selector | exact held-out | rate | program commits | program precision | direct-miss recoveries | direct-correct losses |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct JSON | 138/250 | 55.2% | 0 | - | 0 | 0 |
| Program only if direct parse fails | 142/250 | 56.8% | 5 | 80.0% | 4 | 0 |
| Program on disagreement if probe support >= 0.50 | 140/250 | 56.0% | 5 | 40.0% | 2 | 0 |
| Visible program, veto disagreement if probe support < 0.67 | 146/250 | 58.4% | 62 | 83.9% | 8 | 0 |
| Program on visible disagreement only | 156/250 | 62.4% | 26 | 69.2% | 18 | 0 |
| Program whenever visible example passes | 156/250 | 62.4% | 78 | 79.5% | 18 | 0 |

The strongest deployed policy was also the simplest: **commit the program whenever it passes the visible example**. It reached 156/250 (62.4%), improving direct JSON by +18 cases with 0 direct-correct losses.

The key diagnostic is the visible-disagreement slice. There were 26 cases where the program passed the visible example but disagreed with the direct answer. In that slice, direct JSON was hidden-correct on 0 cases, while the program was hidden-correct on 18 cases. That made visible-program fallback strongly complementary to direct generation in this candidate pool.

## Counterexample Probes

For each visible-disagreement case, the evaluator generated up to three deterministic probe input tables and asked Qwen3.5-4B for direct JSON outputs on those probes. The candidate program was also executed on the same probes. Probe support is the fraction of comparable probes where direct output and program output agreed.

The probe mechanism did **not** improve selection:

- Probe support >= 0.50 recovered 2 direct misses and reached 140/250 (56.0%).
- Visible-program fallback with a probe-support veto recovered 8 direct misses and reached 146/250 (58.4%).
- Mean probe agreement on the decision slice was 28.3%.

Probe support was not a reliable correctness signal. Among visible-disagreement cases with comparable probes, mean support was 25.0% for hidden-correct programs and 33.3% for hidden-wrong programs.

## Iteration

The experiment used three stages:

1. A no-model selector diagnostic over all 250 cases established that visible-program fallback reached 156/250 and parse-failure fallback reached 142/250.
2. A small model-probe smoke on 8 visible-disagreement cases showed probe thresholds rejecting most useful program wins.
3. A full model-probe pass on all 26 visible-disagreement cases confirmed that probe support was weaker than the simple visible-pass rule.

## Read

The useful result is not that counterexample-stressed direct agreement solved selection. It did not. The useful result is that visible-example execution alone was a strong fallback gate for this candidate pool: every direct-correct case survived, and the visible-passing programs recovered 18 direct misses.

The counterexample probes failed for a concrete reason: the independent direct channel often agreed with the same wrong extrapolation, while rejecting many correct programs. Generated probes added cost and reduced accuracy under thresholded policies.

## Caveats

- The candidate pool is fixed and included in `data/candidate_records.jsonl`.
- Program fallback is evaluated only for programs that pass the visible example.
- Probe answers are greedy Qwen3.5-4B direct JSON generations on synthetic probe inputs, not ground truth.
- Hidden answers are used only for evaluation and policy comparison.
- Exact table matching is strict after normalizing cells to strings.
