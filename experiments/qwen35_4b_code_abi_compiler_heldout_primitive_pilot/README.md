# qwen35_4b_code_abi_compiler_heldout_primitive_pilot

**Status:** finished

Standalone experiment for a frozen code-primitive ABI compiler pilot.

The experiment freezes a reusable ABI, measures held-out oracle coverage before training, then trains/evaluates a Qwen3.5-4B QLoRA compiler to emit ABI programs that are executed by a deterministic interpreter. Checkpoints are stored outside this directory under `/workspace/large_artifacts/qwen35_4b_code_abi_compiler_heldout_primitive_pilot/`.

