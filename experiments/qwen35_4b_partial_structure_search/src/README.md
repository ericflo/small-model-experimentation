# Source components

This experiment is self-contained. It does not import task implementations from another experiment or from
`benchmarks/`.

- `families.py` defines the frozen 16-type list DSL, its 32 concrete parameterized operations, deterministic
  execution, exact uncapped behavioral minimum-depth auditing, and fresh `visible` / `label_probe` / `hidden`
  task generation.
- `vllm_runner.py` is the repository's pinned high-throughput Qwen3.5-4B inference runner copied into this
  experiment. Model-facing scripts use it for every result-bearing arm so backend metadata and sampled-token
  accounting remain comparable.

The CPU protocol and orchestration live in `../scripts/`: `build_data.py` creates frozen calibration,
oracle-development, and primary tasks with semantic oracles; `data_audit.py` independently revalidates their
integrity; `oracle_gate.py` tests live-prefix usefulness only on calibration/development data before GPU work;
and `full_brute.py` measures the exact depth-5 model-free baseline. Calibration, model scoring, search, and
analysis are separate scripts so oracle labels cannot enter deployment prompts by generic task serialization.

Unit tests are in `../tests/`. Generated task/oracle tables belong in `../data/`, run receipts in `../runs/`,
derived summaries in `../analysis/`, and final narrative and artifact declarations in `../reports/`.
