# Experiment Log

## 2026-06-27

- Created standalone experiment package.
- Defined a frozen generic transformation ABI and two curated deterministic transformation domains.
- Planned verifier smoke test on coincidence-style false coverage before reading domain coverage.
- Planned headline metrics: raw coverage, counterexample-filtered coverage, depth split, false-pass rate, and coverage by domain.
- Smoke test was tightened before the first accepted run so broad generic predicates can pass thin raw examples and then be removed by adversarial counterexamples.
- Expanded the evaluation suite to 20 CSV/ETL tasks and 20 date/ID/string tasks while keeping the ABI fixed.
- Final run: raw coverage 39/40 (97.5%), counterexample-filtered coverage 37/40 (92.5%).
- CSV/ETL filtered coverage: 20/20 (100.0%). Date/ID/string filtered coverage: 17/20 (85.0%).
- Smoke test: raw coverage 2/2, filtered coverage 0/2, confirming the counterexample filter catches known coincidence-style false coverage.
