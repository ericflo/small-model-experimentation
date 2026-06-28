# Experiment Log

## 2026-06-27

- Created standalone experiment directory.
- Copied deterministic expression DSL into local src/dsl_core.py for standalone execution.

### Run `smoke_v1`
- Tasks: 6
- Candidate rows: 210
- Candidate-support full-task exact: 16.7%
- Direct greedy full-task exact: 66.7%
- Program oracle full-task exact: 33.3%

### Run `main_v1`
- Tasks: 40
- Candidate rows: 1428
- Candidate-support full-task exact: 22.5%
- Direct greedy full-task exact: 50.0%
- Program oracle full-task exact: 27.5%

### Run `smoke_v2`
- Tasks: 6
- Candidate rows: 210
- Candidate-support full-task exact: 16.7%
- Direct greedy full-task exact: 66.7%
- Program oracle full-task exact: 33.3%

### Iteration note
- `main_v2` with 128 real pseudo tables and 128 shuffled pseudo tables was interrupted because the shuffled control dominated wall-clock before the 10-task progress marker.
- Adjusted the implementation so the shuffled-pseudo control has its own smaller table/program budget while the real pseudo-label crystallizer remains bounded separately.

### Run `main_v2_bounded`
- Tasks: 40
- Candidate rows: 1428
- Candidate-support full-task exact: 22.5%
- Direct greedy full-task exact: 50.0%
- Program oracle full-task exact: 27.5%

### Run `main_final`
- Tasks: 40
- Candidate rows: 1428
- Candidate-support full-task exact: 22.5%
- Pseudo-program table full-task exact: 17.5%
- Pseudo-program with direct fallback full-task exact: 50.0%
- Direct greedy full-task exact: 50.0%
- Program oracle full-task exact: 27.5%

### Run `main_final`
- Tasks: 40
- Candidate rows: 1428
- Candidate-support full-task exact: 22.5%
- Pseudo-program table full-task exact: 17.5%
- Pseudo-program with direct fallback full-task exact: 50.0%
- Direct greedy full-task exact: 50.0%
- Program oracle full-task exact: 27.5%
