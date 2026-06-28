# qwen35_4b_passk_coverage_rl

## Purpose

This standalone experiment tests a posttraining objective aimed at coverage rather than single-sample imitation. The adapter is trained with online sampled groups, execution rewards, and a Pass@K-style set utility. The primary readout is whether a small adapter improves held-out coverage at matched K against tuned inference-only sampling.

## Protocol

1. Build MBPP train and held-out records with one public test and hidden tests.
2. Run base default and base hot sampling on a small held-out smoke split.
3. Run a tiny online Pass@K RL smoke.
4. If smoke works, run a pilot with more train and held-out tasks.
5. Compare coverage@K, diversity, pass@1 proxy, and token usage.
6. Write a report with figures.

## Running Notes

- 2026-06-26: Created fresh standalone experiment and large-artifact directories.
- 2026-06-26: Smoke baseline on 4 held-out test tasks: base default K=4 covered 4/4; base hot K=4 covered 3/4.
- 2026-06-26: First online Pass@K RL smoke found zero full-pass rollouts across 4 train groups and hurt held-out smoke coverage (2/4). This exposed sparse reward starvation.
- 2026-06-26: Added positive-gated training that skips zero-positive groups. Smoke v2 took 4 updates after 6 attempts and tied matched-temperature held-out smoke coverage (3/4), but functional diversity dropped.
- 2026-06-26: Added unsaturated-positive gating to skip both zero-positive and all-positive groups. Pilot training took 6 updates after 16 attempts, skipping 7 zero-positive groups and 3 saturated-positive groups.
- 2026-06-26: Pilot held-out comparison on 16 test tasks: base t=0.9 K=4 covered 8/16, tuned-hot K=4 covered 11/16, Pass@K RL adapter K=4 covered 7/16, and base t=0.9 K=8 covered 10/16. The adapter failed the matched-K gate and reduced pass@1 proxy and functional diversity.
