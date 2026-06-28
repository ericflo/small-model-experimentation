# Foofah Direct Qwen vs Frozen ABI

## Question

Does the frozen Foofah table-transform ABI add value over directly asking Qwen3.5-4B to transform the held-out table?

This package compares exact held-out `TestAnswer` accuracy on the same 250 Foofah cases from `https://github.com/markjin1990/foofah_benchmarks`.

## Result

Direct Qwen is the stronger arm on this external structural-transform benchmark:

| arm | exact held-out | rate |
| --- | ---: | ---: |
| Direct Qwen greedy JSON generation | 138/250 | 55.2% |
| Frozen ABI oracle coverage | 45/250 | 18.0% |
| Frozen ABI first-visible selection | 43/250 | 17.2% |
| Direct Qwen OR ABI first-visible fallback | 147/250 | 58.8% |

Direct parse rate was 236/250 (94.4%) with a 768-token generation cap.

## Overlap

| bucket | count |
| --- | ---: |
| Direct and ABI both correct/covered | 35 |
| Direct only | 103 |
| ABI only | 10 |
| Neither | 102 |

Direct accuracy on ABI-covered cases: 35/45 (77.8%).

Direct accuracy on ABI-uncovered cases: 103/205 (50.2%).

The practical fallback union (`direct exact OR ABI first-visible`) reaches 147/250 (58.8%), a +9 case lift over direct generation alone.

## Read

The remaining compiler niche did not appear as the dominant path on Foofah under this test. The ABI's structural table-transform coverage was only 18.0%, and direct Qwen solved many cases outside the ABI's expressivity (`direct_only=103`). The frozen ABI still has a small complementary slice (`abi_only=10`; first-visible adds 9 deployable cases), but it is a fallback, not the main route.

The important interpretation is not that direct generation is perfect. It is not: exact accuracy is 55.2%, parse failures remain 14, and long-output cases are penalized by the 768-token cap. The point is narrower and decisive for this gate: on an independent Foofah benchmark, the ABI/compiler route does not beat simply asking the base model to emit the transformed table.

## Diagnostics

By `NumSamples`, direct-vs-ABI accuracy is in `reports/comparison_summary.json` and `reports/figures/by_num_samples.png`.

Example direct-only files: exp0_10_1.txt, exp0_10_2.txt, exp0_10_3.txt, exp0_10_4.txt, exp0_10_5.txt, exp0_11_1.txt, exp0_11_2.txt, exp0_11_3.txt, exp0_11_4.txt, exp0_12_1.txt, exp0_12_2.txt, exp0_12_3.txt, exp0_12_4.txt, exp0_12_5.txt, exp0_17_3.txt, exp0_17_4.txt, exp0_17_5.txt, exp0_19_1.txt, exp0_19_2.txt, exp0_19_3.txt.

Example ABI-only files: exp0_13_1.txt, exp0_13_2.txt, exp0_13_3.txt, exp0_13_4.txt, exp0_13_5.txt, exp0_33_1.txt, exp0_45_2.txt, exp0_51_3.txt, exp0_51_4.txt, exp0_51_5.txt.

## Caveats

- This is greedy direct generation with a 768-token cap, not a best-possible direct-generation system.
- The ABI baseline is frozen from the prior Foofah gate and imported into `data/abi_*` for a standalone comparison.
- Exact table matching is strict string-table equality after normalizing cells to strings.
