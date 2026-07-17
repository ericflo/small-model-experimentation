# Source

This eval-only confirmation cell runs no experiment-local model code: every
model event goes through the shared trusted aggregate gateway
(`scripts/run_benchmark_aggregate.py`, sha-pinned by the runner). All cell
logic lives in `scripts/` (`run_benchmark.py`, `check_benchmark.py`,
`power_analysis.py`, `run.py`).
