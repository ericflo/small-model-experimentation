# qwen35_4b_code_abi_oracle_coverage_ladder

Standalone no-training experiment for a code-primitive ABI oracle coverage gate.

The experiment measures whether a small verified ABI of deterministic code primitives can express decomposable MBPP-style tasks. It does not train or save model checkpoints. It builds task slices, enumerates ABI candidate programs, verifies them by execution, and reports coverage, false-visible-pass rates, candidate counts, and task-slice diagnostics.

Final report:

- `reports/final_report.md`
- `reports/figures/`

Main result on the first 160 MBPP test records: final reusable ABI oracle coverage reached 134/160 (83.75%) under the available tests. First visible-consistent candidate accuracy was 95/160 (59.4%), so this is a compiler/substrate coverage gate, not a deployable solver by itself.
