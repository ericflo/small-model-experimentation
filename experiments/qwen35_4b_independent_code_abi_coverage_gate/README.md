# qwen35_4b_independent_code_abi_coverage_gate

**Status:** finished

Standalone no-training coverage gate for an independently specified code ABI.

The experiment freezes a general-purpose Python/stdlib-style primitive inventory before evaluation, then measures oracle coverage on MBPP calibration and held-out slices. It does not train a model and does not add kernels after inspecting held-out misses.

Final report will be written to `reports/final_report.md`.

